#!/usr/bin/env python3
"""optimize_model.py — ONNX optimization + int8 quantization pipeline.

Runs inside the lambda-build Docker container, between model download (step 3-5)
and operator config generation (step 8). The output feeds all downstream steps.

Pipeline steps:
  1. Transformer optimization   (onnxruntime.transformers, attention fusion disabled — fallback: skip)
  2a. Static shape specialization  (batch=1, seq_len=max_length — hard fail)
  2b. Symbolic shape inference   (fallback: skip)
  3. ORT graph optimization      (offline ORT_ENABLE_ALL — hard fail)
  4. Weight quantization         (--quant-scheme int8: dynamic QInt8; gptq4: calibrated
                                  4-bit MatMulNBits; or skipped for pre-quantized inputs — hard fail)
"""

import argparse
import shutil
import sys
from pathlib import Path


def _find_batch_and_seq_dims(graph):
    """Return (batch_dim_param, seq_dim_param) by inspecting graph input shapes.

    Scans all inputs for symbolic dimension parameters whose names match
    known batch/sequence patterns. Returns None for each if not found.
    """
    batch_patterns = ("batch",)
    seq_patterns = ("sequence", "seq_len", "seq_length", "seq", "max_seq")

    batch_param = None
    seq_param = None

    for inp in graph.input:
        shape = inp.type.tensor_type.shape
        if shape is None:
            continue
        for dim in shape.dim:
            if not dim.dim_param:
                continue
            p = dim.dim_param.lower()
            if batch_param is None and any(pat in p for pat in batch_patterns):
                batch_param = dim.dim_param
            if seq_param is None and any(pat in p for pat in seq_patterns):
                seq_param = dim.dim_param

    return batch_param, seq_param


def _step0_inline_external_data(input_path: Path, output_path: Path) -> None:
    """Inline any external tensor data into a self-contained ONNX file.

    ONNX models with large weights use external data files (e.g. model.onnx_data).
    References survive file copies but point to a filename relative to the original
    directory. Inlining makes every intermediate file self-contained so no pipeline
    step needs to carry the external data file alongside it.
    """
    import onnx
    from onnx import numpy_helper

    model = onnx.load(str(input_path), load_external_data=False)
    has_external = any(
        t.data_location == onnx.TensorProto.EXTERNAL for t in model.graph.initializer
    )
    if not has_external:
        shutil.copy2(input_path, output_path)
        print("    no external data; copied as-is")
        return

    # Load external tensor data into raw_data fields, then rebuild as inline tensors
    onnx.load_external_data_for_model(model, str(input_path.parent))
    inlined = 0
    for i, init in enumerate(model.graph.initializer):
        if init.data_location == onnx.TensorProto.EXTERNAL:
            arr = numpy_helper.to_array(init, base_dir=str(input_path.parent))
            inline_tensor = numpy_helper.from_array(arr, name=init.name)
            model.graph.initializer[i].CopyFrom(inline_tensor)
            inlined += 1
    onnx.save(model, str(output_path))
    print(f"    inlined {inlined} external tensor(s)")


def _step1_transformer_opt(
    input_path: Path, output_path: Path, model_type: str
) -> None:
    """Transformer graph optimization. Skips (with warning) on any failure.

    Attention fusion is disabled because models like jina-embeddings-v5 use a
    broadcast attention bias of shape [1, heads, 1, seq_len] (ALiBi-style),
    while ORT's MultiHeadAttention kernel requires [1, heads, seq_len, seq_len].
    Fusing produces a graph that is structurally valid but fails at inference.
    """
    try:
        from onnxruntime.transformers import optimizer
        from onnxruntime.transformers.fusion_options import FusionOptions

        options = FusionOptions(model_type)
        options.enable_attention = False
        opt_model = optimizer.optimize_model(
            str(input_path), model_type=model_type, optimization_options=options
        )
        opt_model.save_model_to_file(str(output_path))
        print(f"    transformer opt: OK ({model_type}, attention fusion disabled)")
    except Exception as exc:
        print(
            f"    WARNING: transformer opt failed ({exc}); skipping step",
            file=sys.stderr,
        )
        shutil.copy2(input_path, output_path)


def _step2a_shape_specialization(
    input_path: Path, output_path: Path, max_length: int
) -> None:
    """Fix batch=1 and seq_len=max_length using symbolic dim param names."""
    import onnx
    from onnxruntime.tools import onnx_model_utils

    model = onnx.load(str(input_path))
    batch_param, seq_param = _find_batch_and_seq_dims(model.graph)

    if batch_param:
        onnx_model_utils.make_dim_param_fixed(model.graph, batch_param, 1)
        print(f"    shape: fixed '{batch_param}' = 1")
    else:
        print(
            "    shape: no batch dim param found; skipping batch fix", file=sys.stderr
        )

    if seq_param:
        onnx_model_utils.make_dim_param_fixed(model.graph, seq_param, max_length)
        print(f"    shape: fixed '{seq_param}' = {max_length}")
    else:
        print("    shape: no seq dim param found; skipping seq fix", file=sys.stderr)

    onnx.save(model, str(output_path))


def _step2b_shape_inference(input_path: Path, output_path: Path) -> None:
    """Symbolic shape inference. Skips (with warning) on any failure."""
    try:
        import onnx
        from onnxruntime.tools.symbolic_shape_infer import SymbolicShapeInference

        model = SymbolicShapeInference.infer_shapes(
            onnx.load(str(input_path)),
            auto_merge=True,
            int_max=2**31 - 1,
        )
        onnx.save(model, str(output_path))
        print("    shape inference: OK")
    except Exception as exc:
        print(
            f"    WARNING: shape inference failed ({exc}); skipping step",
            file=sys.stderr,
        )
        shutil.copy2(input_path, output_path)


def _step3_ort_graph_opt(input_path: Path, output_path: Path) -> None:
    """Offline ORT graph optimization at ORT_ENABLE_ALL level. Hard fails."""
    import onnxruntime as ort

    opts = ort.SessionOptions()
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    opts.optimized_model_filepath = str(output_path)
    ort.InferenceSession(
        str(input_path),
        sess_options=opts,
        providers=["CPUExecutionProvider"],
    )
    print("    ORT graph opt (all): OK")


def _step4_int8_quantize(input_path: Path, output_path: Path) -> None:
    """Dynamic int8 quantization. Hard fails."""
    import logging

    from onnxruntime.quantization import QuantType, quantize_dynamic

    # Suppress the "please run pre-processing" warning — steps 2b (symbolic
    # shape inference) and 3 (ORT graph optimization) already cover it.
    prev_level = logging.root.level
    logging.root.setLevel(logging.ERROR)
    try:
        quantize_dynamic(str(input_path), str(output_path), weight_type=QuantType.QInt8)
    finally:
        logging.root.setLevel(prev_level)
    print("    int8 quantization: OK")


class _FixtureCalibrationReader:
    """CalibrationDataReader over fixture texts, for GPTQ weight quantization.

    Reuses gen_reference_vectors' tokenization/feed helpers (sibling module under
    /scripts) so calibration feeds match the correctness harness exactly. Duck-typed
    (get_next/rewind) rather than subclassing onnxruntime.quantization.CalibrationDataReader
    so it needs no onnxruntime import and stays unit-testable with a fake tokenizer.
    GPTQ drives it by calling get_next() until it returns None, then rewind() per pass.
    """

    def __init__(self, tokenizer, fixture_path, input_names, num_samples, max_tokens):
        from gen_reference_vectors import KNOWN_INPUTS, build_feeds, read_texts, truncate_ids

        known = [n for n in input_names if n in KNOWN_INPUTS]
        self._feeds = [
            build_feeds(known, truncate_ids(tokenizer.encode(text).ids, max_tokens))
            for text in read_texts(str(fixture_path), num_samples)
        ]
        self._it = iter(self._feeds)

    def get_next(self):
        return next(self._it, None)

    def rewind(self):
        self._it = iter(self._feeds)


def _step4_gptq_4bit(
    input_path: Path,
    output_path: Path,
    tokenizer_path: Path,
    fixture_path: Path,
    num_samples: int,
    max_tokens: int,
    block_size: int = 32,
) -> None:
    """Calibrated 4-bit weight-only quantization (GPTQ) → MatMulNBits bits=4. Hard fails.

    Operates on the step-3 fp32 graph (same input point as int8), emitting MatMulNBits
    with T1=float — the identical op/attrs jina's `q4` ships, which already builds and
    passes the cosine gate on the neoverse-n1 minimal path, so no new kernel is needed.
    block_size=32 matches jina's granularity for an apples-to-apples "same coarseness,
    but calibrated" comparison. accuracy_level is left at the default to stay on that
    proven kernel.

    NOTE: verify the onnxruntime 1.27.0 signatures via scripts/spike_gptq_4bit.py before
    relying on this; the matmul_4bits_quantizer API has drifted across releases. If the
    single-file save exceeds the 2GB protobuf limit (the fp32 embedding table stays
    unquantized), switch use_external_data_format to True and copy the sidecar into the
    ORT converter's input dir in build_target.sh.
    """
    import importlib
    import logging

    import onnx
    from tokenizers import Tokenizer

    from gen_reference_vectors import KNOWN_INPUTS

    # matmul_4bits_quantizer was renamed to matmul_nbits_quantizer around ORT 1.20;
    # import whichever the installed onnxruntime provides.
    nbits = None
    for name in (
        "onnxruntime.quantization.matmul_nbits_quantizer",
        "onnxruntime.quantization.matmul_4bits_quantizer",
    ):
        try:
            nbits = importlib.import_module(name)
            break
        except ModuleNotFoundError as e:
            # A missing inner dependency (e.g. onnx_ir) must surface, not be
            # mistaken for "this quantizer module name isn't present".
            if e.name != name:
                raise
            continue
    if nbits is None:
        raise ImportError(
            "no MatMulNBits quantizer module in onnxruntime.quantization"
        )

    model = onnx.load(str(input_path))
    input_names = [i.name for i in model.graph.input if i.name in KNOWN_INPUTS]
    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    reader = _FixtureCalibrationReader(
        tokenizer, fixture_path, input_names, num_samples, max_tokens
    )
    algo_config = nbits.GPTQWeightOnlyQuantConfig(
        calibration_data_reader=reader, block_size=block_size
    )
    quantizer = nbits.MatMulNBitsQuantizer(
        model,
        block_size=block_size,
        is_symmetric=True,
        accuracy_level=None,
        algo_config=algo_config,
    )
    prev_level = logging.root.level
    logging.root.setLevel(logging.ERROR)
    try:
        quantizer.process()
    finally:
        logging.root.setLevel(prev_level)
    quantizer.model.save_model_to_file(str(output_path), use_external_data_format=False)
    print("    GPTQ 4-bit quantization: OK")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="ONNX optimization + int8 quantization pipeline"
    )
    parser.add_argument(
        "--input", required=True, type=Path, help="Input ONNX model path"
    )
    parser.add_argument(
        "--output", required=True, type=Path, help="Output ONNX model path"
    )
    parser.add_argument(
        "--model_type",
        required=True,
        help="Model type for transformer optimizer (e.g. bert, gpt2)",
    )
    parser.add_argument(
        "--max_length",
        type=int,
        help="Optional sequence length for shape specialization; omit to keep dynamic shapes",
    )
    parser.add_argument(
        "--target_platform",
        required=True,
        choices=["arm", "amd64"],
        help="Target platform (arm or amd64)",
    )
    parser.add_argument(
        "--work_dir",
        required=True,
        type=Path,
        help="Directory for intermediate ONNX files (inside WORK_DIR, cleaned by build trap)",
    )
    parser.add_argument(
        "--skip-int8-quantize",
        action="store_true",
        help="Run steps 0/1/2b/3 and stop before step 4 (for pre-quantized ONNX inputs)",
    )
    parser.add_argument(
        "--quant-scheme",
        choices=["int8", "gptq4"],
        default="int8",
        help="Step-4 quantization scheme: int8 dynamic (default) or gptq4 "
        "(calibrated 4-bit weight quantization). Ignored when --skip-int8-quantize is set.",
    )
    parser.add_argument(
        "--calibration-tokenizer",
        type=Path,
        help="tokenizer.json for gptq4 calibration (required with --quant-scheme gptq4)",
    )
    parser.add_argument(
        "--calibration-fixture",
        type=Path,
        help="JSONL fixture with 'text' records for gptq4 calibration "
        "(required with --quant-scheme gptq4)",
    )
    parser.add_argument(
        "--calibration-num-samples",
        type=int,
        default=3,
        help="Number of fixture samples to calibrate gptq4 on",
    )
    parser.add_argument(
        "--calibration-max-tokens",
        type=int,
        default=128,
        help="Truncate each gptq4 calibration sample to this many tokens",
    )
    parser.add_argument(
        "--reference-output",
        type=Path,
        help="Also write the pre-int8 (step 3) graph here, for correctness comparison",
    )
    args = parser.parse_args()

    if not args.input.exists():
        print(f"ERROR: input not found: {args.input}", file=sys.stderr)
        sys.exit(1)

    if args.quant_scheme == "gptq4" and not args.skip_int8_quantize:
        missing = [
            flag
            for flag, value in (
                ("--calibration-tokenizer", args.calibration_tokenizer),
                ("--calibration-fixture", args.calibration_fixture),
            )
            if value is None or not value.exists()
        ]
        if missing:
            print(
                "ERROR: --quant-scheme gptq4 requires existing "
                f"{' and '.join(missing)}",
                file=sys.stderr,
            )
            sys.exit(1)

    if args.max_length is not None and args.max_length <= 0:
        print(
            f"ERROR: --max_length must be positive, got {args.max_length}",
            file=sys.stderr,
        )
        sys.exit(1)

    args.work_dir.mkdir(parents=True, exist_ok=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)

    w = args.work_dir
    step0_out = w / "00_inline_input.onnx"
    step1_out = w / "01_transformer_opt.onnx"
    step2a_out = w / "02_fixed_shape.onnx"
    step2b_out = w / "03_shape_inferred.onnx"
    step3_out = w / "04_ort_optimized.onnx"

    print("==> Step 0: inline external data")
    _step0_inline_external_data(args.input, step0_out)

    print("==> Step 1: transformer optimization")
    _step1_transformer_opt(step0_out, step1_out, args.model_type)

    step2_input = step1_out
    if args.max_length is not None:
        print("==> Step 2a: static shape specialization")
        _step2a_shape_specialization(step1_out, step2a_out, args.max_length)
        step2_input = step2a_out
    else:
        print(
            "==> Step 2a: static shape specialization (skipped; keeping dynamic shapes)"
        )

    print("==> Step 2b: symbolic shape inference")
    _step2b_shape_inference(step2_input, step2b_out)

    print("==> Step 3: ORT graph optimization")
    _step3_ort_graph_opt(step2b_out, step3_out)

    if args.reference_output is not None:
        args.reference_output.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(step3_out, args.reference_output)
        print(f"    reference graph written: {args.reference_output}")

    if args.skip_int8_quantize:
        print("==> Step 4: quantization (skipped)")
        shutil.copy2(step3_out, args.output)
    elif args.quant_scheme == "gptq4":
        print("==> Step 4: GPTQ 4-bit weight quantization")
        _step4_gptq_4bit(
            step3_out,
            args.output,
            args.calibration_tokenizer,
            args.calibration_fixture,
            args.calibration_num_samples,
            args.calibration_max_tokens,
        )
    else:
        print("==> Step 4: dynamic int8 quantization")
        _step4_int8_quantize(step3_out, args.output)

    print(f"==> Optimization complete: {args.output}")


if __name__ == "__main__":
    main()
