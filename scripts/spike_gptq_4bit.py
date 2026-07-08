#!/usr/bin/env python3
"""spike_gptq_4bit.py — measure calibrated 4-bit weight quantization fidelity (issue #27).

BLOCKING DECISION GATE. Run this MANUALLY in the lambda-build container before wiring
`q4gptq` into the pipeline. It downloads the fp32 `onnx/model.onnx`, produces GPTQ / HQQ /
RTN 4-bit variants of it with `MatMulNBitsQuantizer`, and reports how close each stays to
full precision — both the pooled last-token (normalized) embedding the retrieval path uses
and the raw `output[0]` tensor — on the same fixture methodology as issue #27's table
(so numbers compare directly to q4f16≈0.616 / model_quantized≈0.994).

Not wired into CI. Not a test (adding it triggers no contract). Standalone tool.

Decision (see the plan / issue #27):
  * RTN should reproduce jina's ~0.62 — a sanity check that the harness is faithful.
  * If GPTQ (pooled, min sample) > ~0.95 → ship q4gptq (consider replacing q4f16).
  * If GPTQ ≈ RTN (~0.62) → GPTQ did not help this nano model; abandon, prefer 8-bit `quantized`.
  * If HQQ lands within ~0.02 of GPTQ → prefer HQQ (data-free) and drop the calibration reader.

Example (inside the container, with HF_TOKEN set if needed):
  python3 /scripts/spike_gptq_4bit.py --work-dir /tmp/spike \\
      --fixture /fixtures/jane-austen_pride-and-prejudice.jsonl

CAVEAT: the `onnxruntime.quantization.matmul_4bits_quantizer` API (config-class kwargs,
`MatMulNBitsQuantizer.__init__` signature, whether `bits` is accepted) has drifted across
ORT releases. The usage below targets onnxruntime 1.27.0; if it errors, inspect the installed
source (`python3 -c "import onnxruntime.quantization.matmul_4bits_quantizer as m; help(m)"`)
and adjust. Pin the exact signatures here once confirmed.
"""

import argparse
import subprocess
import sys
from pathlib import Path

import numpy as np

# Reuse the tokenization/feed helpers the correctness harness already uses so the
# calibration feeds and scoring feeds match the rest of the pipeline exactly.
from gen_reference_vectors import KNOWN_INPUTS, build_feeds, read_texts, truncate_ids

DEFAULT_REPO = "jinaai/jina-embeddings-v5-text-nano-retrieval"


def _run(cmd):
    print(f"    $ {' '.join(str(c) for c in cmd)}")
    subprocess.run(cmd, check=True)


def download(repo_id, revision, path, dest):
    """hf download a single file into dest, return the local path."""
    _run(["hf", "download", repo_id, path, "--revision", revision, "--local-dir", str(dest)])
    local = dest / path
    if not local.exists():
        sys.exit(f"spike: expected {local} after download")
    return local


def build_feeds_for(input_names, ids):
    """build_feeds restricted to the inputs the model actually declares."""
    known = [n for n in input_names if n in KNOWN_INPUTS]
    return build_feeds(known, ids)


class FixtureCalibrationReader:
    """onnxruntime CalibrationDataReader over fixture texts (for GPTQ)."""

    def __init__(self, tokenizer, fixture, input_names, num_samples, max_tokens):
        self._feeds = [
            build_feeds_for(input_names, truncate_ids(tokenizer.encode(t).ids, max_tokens))
            for t in read_texts(fixture, num_samples)
        ]
        self._it = iter(self._feeds)

    def get_next(self):
        return next(self._it, None)

    def __iter__(self):
        # ORT 1.27's GPTQ path iterates the reader (for data in reader) instead of
        # calling get_next(); yield a fresh pass over the stored feeds each time.
        return iter(self._feeds)

    def rewind(self):
        self._it = iter(self._feeds)


def pooled_last_token(output):
    """Last-token (this model's pooling), L2-normalized. output: (1, seq, dim)."""
    vec = np.asarray(output, dtype=np.float32).reshape(output.shape[-2], output.shape[-1])[-1]
    norm = np.linalg.norm(vec)
    return vec / norm if norm else vec


def cosine(a, b):
    a = np.asarray(a, dtype=np.float32).reshape(-1)
    b = np.asarray(b, dtype=np.float32).reshape(-1)
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    return float(np.dot(a, b) / denom) if denom else 0.0


def load_nbits_api():
    """Import the MatMulNBits quantizer module across onnxruntime versions.

    Renamed matmul_4bits_quantizer → matmul_nbits_quantizer around ORT 1.20. Prints
    the module's public symbols so config/class-name drift is diagnosable in one run.
    """
    import importlib

    last_err = None
    for name in (
        "onnxruntime.quantization.matmul_nbits_quantizer",
        "onnxruntime.quantization.matmul_4bits_quantizer",
    ):
        try:
            mod = importlib.import_module(name)
            print(f"    using {name}; symbols: "
                  f"{[s for s in dir(mod) if 'Config' in s or 'Quantizer' in s]}")
            return mod
        except ModuleNotFoundError as e:
            # Only treat "this module name isn't present" as a reason to try the
            # next candidate; a missing INNER dependency (e.g. onnx_ir) must surface.
            if e.name != name:
                raise
            last_err = e
    raise ImportError(
        "no MatMulNBits quantizer module found in onnxruntime.quantization"
    ) from last_err


def quantize(algo, model_path, out_path, block_size, reader_factory):
    """Produce a 4-bit MatMulNBits variant. Returns the output path.

    Verify these signatures against the installed onnxruntime if they raise; the
    symbol list printed by load_nbits_api() shows the exact config class names.
    """
    import onnx

    mod = load_nbits_api()

    # ORT 1.27 placement: bits/block_size/is_symmetric are MatMulNBitsQuantizer kwargs.
    # GPTQ/HQQ configs carry their own block_size; RTN has none, so algo_config=None
    # makes the quantizer build a DefaultWeightOnlyQuantConfig (plain RTN weight-only).
    if algo == "gptq":
        algo_config = mod.GPTQWeightOnlyQuantConfig(
            calibration_data_reader=reader_factory(), block_size=block_size
        )
    elif algo == "hqq":
        algo_config = mod.HQQWeightOnlyQuantConfig(block_size=block_size, bits=4)
    elif algo == "rtn":
        algo_config = None
    else:
        raise ValueError(f"unknown algo {algo!r}")

    model = onnx.load(str(model_path))
    quant = mod.MatMulNBitsQuantizer(
        model, bits=4, block_size=block_size, algo_config=algo_config
    )
    quant.process()
    # Keep single-file if it fits; the pipeline's converter dir-scan prefers that.
    quant.model.save_model_to_file(str(out_path), use_external_data_format=False)
    size = out_path.stat().st_size
    print(f"    {algo}: wrote {out_path.name} ({size / 1e6:.1f} MB, single-file)")
    if size > 2_000_000_000:
        print("    WARNING: >2GB — pipeline must use use_external_data_format=True")
    return out_path


def score(model_path, tokenizer, fixture, num_samples, max_tokens, ref_pooled, ref_raw):
    """Return (pooled_cosines, raw_cosines) vs the fp32 references."""
    import onnxruntime as ort

    sess = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
    input_names = [i.name for i in sess.get_inputs()]
    out_name = sess.get_outputs()[0].name
    pooled, raw = [], []
    for i, text in enumerate(read_texts(fixture, num_samples)):
        ids = truncate_ids(tokenizer.encode(text).ids, max_tokens)
        out = sess.run([out_name], build_feeds_for(input_names, ids))[0]
        pooled.append(cosine(pooled_last_token(out), ref_pooled[i]))
        raw.append(cosine(out, ref_raw[i]))
    return pooled, raw


def main(argv=None):
    parser = argparse.ArgumentParser(description="Measure 4-bit quant fidelity (issue #27)")
    parser.add_argument("--repo-id", default=DEFAULT_REPO)
    parser.add_argument("--revision", default="main")
    parser.add_argument("--primary", default="onnx/model.onnx", help="fp32 golden to quantize")
    parser.add_argument("--companion", default="onnx/model.onnx_data")
    parser.add_argument("--tokenizer-file", default="tokenizer.json")
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--fixture", type=Path, required=True)
    parser.add_argument("--num-samples", type=int, default=3)
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--block-size", type=int, default=32, help="match jina's 32 for apples-to-apples")
    parser.add_argument("--algos", default="rtn,gptq,hqq", help="comma list of rtn,gptq,hqq")
    args = parser.parse_args(argv)

    import onnxruntime as ort
    from tokenizers import Tokenizer

    args.work_dir.mkdir(parents=True, exist_ok=True)
    dl = args.work_dir / "model"

    print("==> Downloading fp32 golden + tokenizer")
    fp32 = download(args.repo_id, args.revision, args.primary, dl)
    if args.companion:
        download(args.repo_id, args.revision, args.companion, dl)
    tok_path = download(args.repo_id, args.revision, args.tokenizer_file, dl)
    tokenizer = Tokenizer.from_file(str(tok_path))

    print("==> Computing fp32 reference embeddings")
    ref_sess = ort.InferenceSession(str(fp32), providers=["CPUExecutionProvider"])
    ref_inputs = [i.name for i in ref_sess.get_inputs()]
    ref_out_name = ref_sess.get_outputs()[0].name
    ref_pooled, ref_raw = [], []
    for text in read_texts(str(args.fixture), args.num_samples):
        ids = truncate_ids(tokenizer.encode(text).ids, args.max_tokens)
        out = ref_sess.run([ref_out_name], build_feeds_for(ref_inputs, ids))[0]
        ref_pooled.append(pooled_last_token(out))
        ref_raw.append(out)

    def reader_factory():
        return FixtureCalibrationReader(
            tokenizer, str(args.fixture), ref_inputs, args.num_samples, args.max_tokens
        )

    rows = []
    for algo in [a.strip() for a in args.algos.split(",") if a.strip()]:
        print(f"==> Quantizing: {algo} (block_size={args.block_size}, bits=4)")
        out_path = args.work_dir / f"model_{algo}4.onnx"
        quantize(algo, fp32, out_path, args.block_size, reader_factory)
        pooled, raw = score(
            out_path, tokenizer, str(args.fixture), args.num_samples,
            args.max_tokens, ref_pooled, ref_raw,
        )
        rows.append((algo, pooled, raw))

    print("\n==> Fidelity vs fp32 (cosine; higher is better)")
    print(f"{'algo':<8} {'pooled(min)':>12} {'pooled(mean)':>13} {'raw(mean)':>11}   per-sample pooled")
    for algo, pooled, raw in rows:
        per = " / ".join(f"{c:.3f}" for c in pooled)
        print(
            f"{algo:<8} {min(pooled):>12.4f} {float(np.mean(pooled)):>13.4f} "
            f"{float(np.mean(raw)):>11.4f}   {per}"
        )
    print("\nReference points (issue #27): q4f16 pooled≈0.616 (min 0.178); model_quantized≈0.994.")


if __name__ == "__main__":
    main()
