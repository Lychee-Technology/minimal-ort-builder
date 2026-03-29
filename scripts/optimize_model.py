#!/usr/bin/env python3
"""optimize_model.py — ONNX optimization + int8 quantization pipeline.

Runs inside the lambda-build Docker container, between model download (step 3-5)
and operator config generation (step 8). The output feeds all downstream steps.

Pipeline steps:
  1. Transformer optimization   (onnxruntime.transformers — fallback: skip)
  2a. Static shape specialization  (batch=1, seq_len=max_length — hard fail)
  2b. Symbolic shape inference   (fallback: skip)
  3. ORT graph optimization      (offline ORT_ENABLE_ALL — hard fail)
  4. Dynamic int8 quantization   (weight_type=QInt8 — hard fail)
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


def _step1_transformer_opt(input_path: Path, output_path: Path, model_type: str) -> None:
    """Transformer graph optimization. Skips (with warning) on any failure."""
    try:
        from onnxruntime.transformers import optimizer
        opt_model = optimizer.optimize_model(str(input_path), model_type=model_type)
        opt_model.save_model_to_file(str(output_path))
        print(f"    transformer opt: OK ({model_type})")
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
        print("    shape: no batch dim param found; skipping batch fix", file=sys.stderr)

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
    print("    ORT graph opt: OK")


def _step4_int8_quantize(input_path: Path, output_path: Path) -> None:
    """Dynamic int8 quantization. Hard fails."""
    from onnxruntime.quantization import QuantType, quantize_dynamic

    quantize_dynamic(str(input_path), str(output_path), weight_type=QuantType.QInt8)
    print("    int8 quantization: OK")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="ONNX optimization + int8 quantization pipeline"
    )
    parser.add_argument("--input", required=True, type=Path, help="Input ONNX model path")
    parser.add_argument("--output", required=True, type=Path, help="Output ONNX model path")
    parser.add_argument(
        "--model_type",
        required=True,
        help="Model type for transformer optimizer (e.g. bert, gpt2)",
    )
    parser.add_argument(
        "--max_length", required=True, type=int, help="Sequence length for shape specialization"
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
    args = parser.parse_args()

    if not args.input.exists():
        print(f"ERROR: input not found: {args.input}", file=sys.stderr)
        sys.exit(1)

    if args.max_length <= 0:
        print(f"ERROR: --max_length must be positive, got {args.max_length}", file=sys.stderr)
        sys.exit(1)

    args.work_dir.mkdir(parents=True, exist_ok=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)

    w = args.work_dir
    step1_out = w / "01_transformer_opt.onnx"
    step2a_out = w / "02_fixed_shape.onnx"
    step2b_out = w / "03_shape_inferred.onnx"
    step3_out = w / "04_ort_optimized.onnx"

    print("==> Step 1: transformer optimization")
    _step1_transformer_opt(args.input, step1_out, args.model_type)

    print("==> Step 2a: static shape specialization")
    _step2a_shape_specialization(step1_out, step2a_out, args.max_length)

    print("==> Step 2b: symbolic shape inference")
    _step2b_shape_inference(step2a_out, step2b_out)

    print("==> Step 3: ORT graph optimization")
    _step3_ort_graph_opt(step2b_out, step3_out)

    print("==> Step 4: dynamic int8 quantization")
    _step4_int8_quantize(step3_out, args.output)

    print(f"==> Optimization complete: {args.output}")


if __name__ == "__main__":
    main()
