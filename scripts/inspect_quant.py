#!/usr/bin/env python3
"""inspect_quant.py — dump weight-quantization op attributes from ONNX models (issue #27).

Answers "why is jina's official q4f16 worse than our own 4-bit RTN?" by showing, per
model, the bits / block_size / symmetry of every quantized op:

  * MatMulNBits          — the compute weights (the 84 attn/MLP matmuls)
  * GatherBlockQuantized — the token embedding table

The decisive comparison: jina's q4/q4f16 quantize BOTH the matmuls and the embedding
table to 4-bit, whereas our pipeline (MatMulNBitsQuantizer) only quantizes the matmuls
and leaves the embedding fp32 — which is why our RTN scored higher (0.79 vs 0.62) and
why our artifact is ~469 MB vs jina's ~124 MB. This script makes that concrete.

Reads only the graph proto (load_external_data=False), so it is fast, needs only `onnx`,
and does NOT download the multi-hundred-MB `.onnx_data` sidecars.

Run in the container (or anywhere onnx is installed):

  python3 /scripts/inspect_quant.py --work-dir /tmp/inspect
  # add local models to compare, e.g. the spike outputs:
  python3 /scripts/inspect_quant.py --work-dir /tmp/inspect \\
      /tmp/spike/model_rtn4.onnx /tmp/spike/model_gptq4.onnx
"""

import argparse
import subprocess
import sys
from collections import Counter
from pathlib import Path

DEFAULT_REPO = "jinaai/jina-embeddings-v5-text-nano-retrieval"
# jina variants worth comparing. Only the .onnx graph is fetched (no _data sidecar).
DEFAULT_FILES = ("onnx/model_q4f16.onnx", "onnx/model_q4.onnx", "onnx/model_quantized.onnx")

QUANT_OPS = ("MatMulNBits", "GatherBlockQuantized")
# Input index at which an optional zero_points input appears for each op. Its presence
# (a non-empty input name) means asymmetric quantization; absence means symmetric.
ZERO_POINT_INPUT_INDEX = {"MatMulNBits": 3, "GatherBlockQuantized": 3}


def _run(cmd):
    print(f"    $ {' '.join(str(c) for c in cmd)}")
    subprocess.run(cmd, check=True)


def download(repo_id, revision, path, dest):
    _run(["hf", "download", repo_id, path, "--revision", revision, "--local-dir", str(dest)])
    local = dest / path
    if not local.exists():
        sys.exit(f"inspect_quant: expected {local} after download")
    return local


def _attrs(node):
    import onnx

    return {a.name: onnx.helper.get_attribute_value(a) for a in node.attribute}


def _has_zero_points(node):
    idx = ZERO_POINT_INPUT_INDEX.get(node.op_type)
    return idx is not None and len(node.input) > idx and node.input[idx] != ""


def inspect(path):
    """Return {op_type: Counter((bits, block_size, sym/asym, accuracy_level) -> count)}
    plus counts of any remaining plain MatMul nodes (weights left unquantized)."""
    import onnx

    model = onnx.load(str(path), load_external_data=False)
    summary = {op: Counter() for op in QUANT_OPS}
    plain_matmul = 0
    for node in model.graph.node:
        if node.op_type in QUANT_OPS:
            a = _attrs(node)
            key = (
                a.get("bits"),
                a.get("block_size"),
                "asym" if _has_zero_points(node) else "sym",
                a.get("accuracy_level"),
            )
            summary[node.op_type][key] += 1
        elif node.op_type == "MatMul":
            plain_matmul += 1
    return summary, plain_matmul


def _report(label, path):
    summary, plain_matmul = inspect(path)
    size_mb = Path(path).stat().st_size / 1e6
    print(f"\n=== {label} ({size_mb:.1f} MB graph proto) ===")
    for op in QUANT_OPS:
        rows = summary[op]
        if not rows:
            print(f"  {op}: (none)")
            continue
        total = sum(rows.values())
        print(f"  {op}: {total} node(s)")
        for (bits, block, sym, acc), n in sorted(rows.items(), key=lambda kv: -kv[1]):
            acc_s = "" if acc in (None, 0) else f", accuracy_level={acc}"
            print(f"      {n:>3}× bits={bits} block_size={block} {sym}{acc_s}")
    print(f"  MatMul (unquantized, fp32/fp16 weights): {plain_matmul}")


def main(argv=None):
    parser = argparse.ArgumentParser(description="Dump quantization-op attributes (issue #27)")
    parser.add_argument("models", nargs="*", type=Path,
                        help="extra local .onnx models to inspect (e.g. spike outputs)")
    parser.add_argument("--repo-id", default=DEFAULT_REPO)
    parser.add_argument("--revision", default="main")
    parser.add_argument("--files", default=",".join(DEFAULT_FILES),
                        help="comma list of jina .onnx graphs to download and inspect")
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--no-download", action="store_true",
                        help="skip the jina downloads; inspect only the local models given")
    args = parser.parse_args(argv)

    args.work_dir.mkdir(parents=True, exist_ok=True)

    if not args.no_download:
        for path in [f.strip() for f in args.files.split(",") if f.strip()]:
            local = download(args.repo_id, args.revision, path, args.work_dir)
            _report(f"jina {path}", local)

    for model in args.models:
        if not model.exists():
            print(f"\n=== {model} === MISSING (skipped)")
            continue
        _report(str(model), model)

    print(
        "\nReading: 'sym' = no zero_points input (symmetric); 'asym' = zero_points present.\n"
        "GatherBlockQuantized present = the token embedding table is quantized; absent =\n"
        "the embedding stays fp32 (our pipeline leaves it fp32 → larger but higher-fidelity)."
    )


if __name__ == "__main__":
    main()
