#!/usr/bin/env python3
"""run_benchmark.py — per-target inference benchmark orchestrator.

Runs inside the lambda-build (AL2023) container, one invocation per matrix
target. Obtains a shipped tarball (either by building it fresh from the manifest
entry, or from a release asset the workflow pre-downloaded on the runner),
compiles `bench.c` against the extracted `libonnxruntime.so`, replays token feeds
through the shipped `model.ort`, and writes a merged metrics JSON:

    bench-<id_safe>_<quant>.json

Configuration is supplied via environment variables (same contract as
build_target.sh), plus:

    SOURCE       "build" (run build_target.sh) or "release" (use pre-downloaded)
    RELEASE_DIR  where the release tarball was mounted (default /release)
    WARMUP       untimed warmup iterations (default 10)
    ITERS        timed iterations (default 100)
    NUM_SAMPLES  fixture texts to cycle (default 3)
    MAX_TOKENS   token truncation length (default 128)
"""

import json
import os
import subprocess
import sys
import tarfile
import tempfile
from pathlib import Path

ORT_REPO = "https://github.com/microsoft/onnxruntime.git"
FIXTURE_NAME = "jane-austen_pride-and-prejudice.jsonl"
SCRIPTS_DIR = Path(__file__).resolve().parent


def _env(name, default=None, required=False):
    val = os.environ.get(name, default)
    if required and not val:
        sys.exit(f"run_benchmark: {name} is required")
    return val


def _run(cmd, **kwargs):
    printable = " ".join(str(c) for c in cmd)
    print(f"==> {printable}", flush=True)
    return subprocess.run(cmd, check=True, **kwargs)


def _id_safe(target_id):
    return target_id.replace("/", "__")


def build_tarball(output_dir):
    """SOURCE=build: run build_target.sh from the inherited env."""
    print("==> Building target from manifest entry", flush=True)
    _run([str(SCRIPTS_DIR / "build_target.sh")])
    return None  # tarball located by caller from OUTPUT_DIR


def sparse_clone_headers(ort_version, dest):
    """Sparse-checkout just the ORT C API header dir at the given tag."""
    _run([
        "git", "clone", "--depth", "1", "--branch", ort_version,
        "--filter=blob:none", "--sparse", ORT_REPO, str(dest),
    ])
    _run([
        "git", "-C", str(dest), "sparse-checkout", "set",
        "include/onnxruntime/core/session",
    ])
    include_dir = dest / "include" / "onnxruntime" / "core" / "session"
    if not (include_dir / "onnxruntime_c_api.h").is_file():
        sys.exit(f"run_benchmark: onnxruntime_c_api.h not found under {include_dir}")
    return include_dir


def main():
    source = _env("SOURCE", "build")
    target_id = _env("TARGET_ID", required=True)
    quant = _env("QUANT", required=True)
    id_safe = _id_safe(target_id)
    output_dir = Path(_env("OUTPUT_DIR", "/output"))
    fixture_dir = Path(_env("FIXTURE_DIR", "/fixtures"))
    warmup = _env("WARMUP", "10")
    iters = _env("ITERS", "100")
    num_samples = _env("NUM_SAMPLES", "3")
    max_tokens = _env("MAX_TOKENS", "128")
    output_dir.mkdir(parents=True, exist_ok=True)

    tarball_name = f"{id_safe}_{quant}_linux-arm64.tar.gz"

    if source == "build":
        build_tarball(output_dir)
        tarball = output_dir / tarball_name
    elif source == "release":
        release_dir = Path(_env("RELEASE_DIR", "/release"))
        tarball = release_dir / tarball_name
    else:
        sys.exit(f"run_benchmark: SOURCE must be 'build' or 'release', got {source!r}")

    if not tarball.is_file():
        sys.exit(f"run_benchmark: tarball not found: {tarball}")
    tarball_bytes = tarball.stat().st_size

    work = Path(tempfile.mkdtemp(prefix="bench-"))
    extract_dir = work / "artifact"
    extract_dir.mkdir()
    print(f"==> Extracting {tarball}", flush=True)
    with tarfile.open(tarball, "r:gz") as tf:
        tf.extractall(extract_dir)

    so_path = extract_dir / "libonnxruntime.so"
    model_path = extract_dir / "model.ort"
    tokenizer_path = extract_dir / "tokenizer.json"
    build_info_path = extract_dir / "build-info.json"
    for required in (so_path, model_path, tokenizer_path, build_info_path):
        if not required.is_file():
            sys.exit(f"run_benchmark: missing {required.name} in tarball")

    build_info = json.loads(build_info_path.read_text(encoding="utf-8"))
    ort_version = build_info.get("ort_version") or _env("ORT_VERSION", required=True)
    ort_git_sha = build_info.get("ort_git_sha", "")

    include_dir = sparse_clone_headers(ort_version, work / "ort-headers")

    # Token feeds only; the harness times Run() and ignores the reference.
    tvbin = work / "bench.tvbin"
    _run([
        sys.executable, str(SCRIPTS_DIR / "gen_reference_vectors.py"),
        "--inputs-only",
        "--model", str(model_path),
        "--tokenizer", str(tokenizer_path),
        "--fixture", str(fixture_dir / FIXTURE_NAME),
        "--output", str(tvbin),
        "--num-samples", num_samples,
        "--max-tokens", max_tokens,
    ])

    bench_bin = work / "bench"
    _run([
        "clang", "-O2", "-o", str(bench_bin), str(SCRIPTS_DIR / "bench.c"),
        "-I", str(include_dir),
        "-L", str(extract_dir),
        "-lonnxruntime",
        "-lm",
        "-Wl,-rpath," + str(extract_dir),
    ])

    proc = _run(
        [str(bench_bin), str(model_path), str(tvbin), warmup, iters],
        capture_output=True, text=True,
    )
    sys.stderr.write(proc.stderr)
    metrics = json.loads(proc.stdout.strip().splitlines()[-1])

    result = {
        "target_id": target_id,
        "quant": quant,
        "ort_version": ort_version,
        "ort_git_sha": ort_git_sha,
        **metrics,
        "so_bytes": so_path.stat().st_size,
        "model_ort_bytes": model_path.stat().st_size,
        "tarball_bytes": tarball_bytes,
    }

    out_file = output_dir / f"bench-{id_safe}_{quant}.json"
    out_file.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"==> wrote {out_file}", flush=True)
    print(json.dumps(result, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
