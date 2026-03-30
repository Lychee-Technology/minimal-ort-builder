# ONNX Optimization Pipeline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Insert a custom ONNX optimization + int8 quantization pipeline (transformer opt → shape fix → shape inference → ORT graph opt → dynamic int8 quant) into the build, running before operator config generation so the minimal ORT library includes quantization operators.

**Architecture:** A new `scripts/optimize_model.py` is called as step 7 in `build_target.sh`, taking the raw `onnx/model.onnx` and producing an optimized+quantized ONNX that all downstream steps consume. `validate_manifest.py` is extended to accept `model_type` in metadata. `builds/release.yaml` is updated to point at the unquantized base model.

**Tech Stack:** Python 3.11, `onnxruntime==1.24.4` (already installed in Docker), `onnx` (already installed), `pytest` for tests.

## Implementation Status Note

This plan is preserved as the original implementation checklist, but the shipped pipeline has intentionally diverged in two places:

- The Docker build image now installs CPU-only `torch` so Step 1 transformer optimization can execute for BERT-family models.
- The default build no longer passes `--max_length`, so Step 2a shape specialization is skipped and the final optimized ONNX / ORT model retains dynamic input lengths.

The current implementation still keeps attention fusion disabled, keeps ORT conversion capped at BASIC, and keeps the smoke test's runtime graph optimization disabled.

---

## File Map

| File | Action | Responsibility |
|------|--------|----------------|
| `scripts/optimize_model.py` | **Create** | 5-step optimization pipeline, CLI entry point |
| `scripts/build_target.sh` | **Modify** | Insert step 7, renumber 7→8 … 13→14/15, update op-config and ORT-convert inputs |
| `builds/release.yaml` | **Modify** | Add `model_type: bert`, change `quant` to `int8`, change `primary` to `onnx/model.onnx`, drop onnx_data companion |
| `scripts/validate_manifest.py` | **Modify** | Add `model_type` to `_STR_META_KEYS` |
| `tests/test_validate_manifest.py` | **Modify** | Add tests for `model_type` acceptance/rejection |
| `tests/test_optimize_model.py` | **Create** | CLI argument validation tests (subprocess-based, no real model needed) |

---

## Task 1: Extend manifest validator to accept `model_type`

**Files:**
- Modify: `scripts/validate_manifest.py:154-165`
- Modify: `tests/test_validate_manifest.py`

- [ ] **Step 1: Write failing tests**

Add to `tests/test_validate_manifest.py`:

```python
def test_metadata_model_type_string_accepted():
    """model_type: bert must be accepted as a string metadata field."""
    manifest = textwrap.dedent("""\
        onnxruntime:
          version: "1.20.1"
        build:
          container_image: public.ecr.aws/lambda/provided:al2023
          target_os: linux
          target_arch: arm64
          cpu_tuning: neoverse-n1
          execution_provider: cpu
          minimal_build: extended
        targets:
          - id: model-a
            quant: int8
            metadata:
              model_type: bert
              max_length: 8192
            model:
              repo_id: org/repo
              revision: main
              primary: onnx/model.onnx
              companions: []
    """)
    result = _run(stdin_text=manifest)
    assert result.returncode == 0, f"stderr: {result.stderr}"


def test_metadata_model_type_non_string_rejected():
    """model_type: 42 (integer) must be rejected."""
    manifest = textwrap.dedent("""\
        onnxruntime:
          version: "1.20.1"
        build:
          container_image: public.ecr.aws/lambda/provided:al2023
          target_os: linux
          target_arch: arm64
          cpu_tuning: neoverse-n1
          execution_provider: cpu
          minimal_build: extended
        targets:
          - id: model-a
            quant: int8
            metadata:
              model_type: 42
              max_length: 8192
            model:
              repo_id: org/repo
              revision: main
              primary: onnx/model.onnx
              companions: []
    """)
    result = _run(stdin_text=manifest)
    assert result.returncode != 0
    assert "model_type" in result.stderr.lower()
```

- [ ] **Step 2: Run tests to confirm they fail**

```bash
.venv/bin/pytest tests/test_validate_manifest.py::test_metadata_model_type_string_accepted tests/test_validate_manifest.py::test_metadata_model_type_non_string_rejected -v
```

Expected: both FAIL (model_type not yet in `_STR_META_KEYS`)

- [ ] **Step 3: Add `model_type` to `_STR_META_KEYS` in `validate_manifest.py`**

In `scripts/validate_manifest.py`, change:

```python
        _STR_META_KEYS = {
            "model_format",
            "pooling",
            "input_kind",
            "query_prefix",
            "document_prefix",
        }
```

to:

```python
        _STR_META_KEYS = {
            "model_format",
            "model_type",
            "pooling",
            "input_kind",
            "query_prefix",
            "document_prefix",
        }
```

- [ ] **Step 4: Run tests to confirm they pass**

```bash
.venv/bin/pytest tests/test_validate_manifest.py -v
```

Expected: all PASS including the two new tests

- [ ] **Step 5: Commit**

```bash
git add scripts/validate_manifest.py tests/test_validate_manifest.py
git commit -m "feat: accept model_type in manifest metadata"
```

---

## Task 2: Update `builds/release.yaml`

**Prerequisite:** Task 1 must be complete. `validate_manifest.py` must already accept `model_type` before updating `release.yaml`, otherwise `test_file_path_argument` (which loads the real manifest) will fail.

**Files:**
- Modify: `builds/release.yaml`

- [ ] **Step 1: Update the jina target**

Replace the `jinaai/jina-embeddings-v5-text-nano-retrieval` target:

```yaml
targets:
  - id: jinaai/jina-embeddings-v5-text-nano-retrieval
    quant: int8
    bundle_extras:
      - tokenizer.json
      - build-info.json
    metadata:
      model_format: ort
      model_type: bert
      pooling: last_token
      input_kind: retrieval
      query_prefix: "Query: "
      document_prefix: "Document: "
      raw_embedding_dimension: 768
      output_embedding_dimension: 768
      max_length: 8192
    model:
      repo_id: jinaai/jina-embeddings-v5-text-nano-retrieval
      revision: main
      primary: onnx/model.onnx
      companions:
        - tokenizer.json
```

Changes from previous:
- `quant`: `q4f16` → `int8`
- `metadata.model_type`: `bert` (new)
- `model.primary`: `onnx/model_q4f16.onnx` → `onnx/model.onnx`
- `model.companions`: removed `onnx/model_q4f16.onnx_data`

- [ ] **Step 2: Validate the manifest**

```bash
.venv/bin/python scripts/validate_manifest.py builds/release.yaml
```

Expected: `OK: manifest is valid`

- [ ] **Step 3: Run full test suite**

```bash
.venv/bin/pytest tests/ -v
```

Expected: all PASS (test_file_path_argument exercises the real release.yaml)

- [ ] **Step 4: Commit**

```bash
git add builds/release.yaml
git commit -m "feat: update jina target to use unquantized base model with int8 quant"
```

---

## Task 3: Create `scripts/optimize_model.py`

**Files:**
- Create: `scripts/optimize_model.py`
- Create: `tests/test_optimize_model.py`

- [ ] **Step 1: Write failing CLI tests**

Create `tests/test_optimize_model.py`:

```python
"""Tests for scripts/optimize_model.py — CLI argument validation."""

import subprocess
import sys
import textwrap
from pathlib import Path

SCRIPT = Path(__file__).parent.parent / "scripts" / "optimize_model.py"


def _run(*extra_args):
    cmd = [sys.executable, str(SCRIPT)] + list(extra_args)
    return subprocess.run(cmd, capture_output=True, text=True)


def test_missing_required_args_exits_nonzero():
    """Running with no args must exit non-zero (argparse error)."""
    result = _run()
    assert result.returncode != 0


def test_missing_input_file_exits_nonzero(tmp_path):
    """--input pointing at a nonexistent file must exit non-zero with a clear error."""
    result = _run(
        "--input", str(tmp_path / "nonexistent.onnx"),
        "--output", str(tmp_path / "out.onnx"),
        "--model_type", "bert",
        "--max_length", "512",
        "--target_platform", "arm",
        "--work_dir", str(tmp_path / "work"),
    )
    assert result.returncode != 0
    assert "not found" in result.stderr.lower() or "error" in result.stderr.lower()


def test_invalid_target_platform_exits_nonzero(tmp_path):
    """--target_platform with an invalid value must exit non-zero.
    argparse rejects the value before any file I/O, so no real input file is needed.
    """
    result = _run(
        "--input", str(tmp_path / "model.onnx"),
        "--output", str(tmp_path / "out.onnx"),
        "--model_type", "bert",
        "--max_length", "512",
        "--target_platform", "sparc",
        "--work_dir", str(tmp_path / "work"),
    )
    assert result.returncode != 0


def test_help_exits_zero():
    """--help must exit 0 and mention key flags."""
    result = _run("--help")
    assert result.returncode == 0
    assert "--input" in result.stdout
    assert "--output" in result.stdout
    assert "--model_type" in result.stdout
    assert "--max_length" in result.stdout
    assert "--target_platform" in result.stdout
```

- [ ] **Step 2: Run tests to confirm they fail**

```bash
.venv/bin/pytest tests/test_optimize_model.py -v
```

Expected: FAIL (script does not exist yet)

- [ ] **Step 3: Create `scripts/optimize_model.py`**

```python
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
```

- [ ] **Step 4: Run tests to confirm they pass**

```bash
.venv/bin/pytest tests/test_optimize_model.py -v
```

Expected: all PASS

- [ ] **Step 5: Commit**

```bash
git add scripts/optimize_model.py tests/test_optimize_model.py
git commit -m "feat: add optimize_model.py — ONNX optimization and int8 quantization pipeline"
```

---

## Task 4: Update `scripts/build_target.sh`

**Files:**
- Modify: `scripts/build_target.sh`

This task has no automated tests (the build runs inside Docker); verify by reading the diff carefully.

Four sub-changes in one edit pass:

1. **Insert new step 7** between step 6 (clone ORT) and old step 7 (op-config)
2. **Renumber** old steps 7–13 to 8–14/15 in their header comments
3. **Update step 8** (op-config): change input from `"${MODEL_DIR}"` to `"${WORK_DIR}/optimized.onnx"`
4. **Update step 11** (ORT convert): change directory arg from `"${MODEL_DIR}/$(dirname "${HF_PRIMARY}")"` to `"${WORK_DIR}/ort_input/"` (a directory containing only `optimized.onnx`)

- [ ] **Step 1: Insert step 7 and create `OPTIMIZED_ONNX` + `ORT_INPUT_DIR` variables**

After the `# 6. Clone ORT source` block (line ~99) and before the current `# 7. Generate reduced operator config` block, insert:

```bash
# ---------------------------------------------------------------------------
# 7. Optimize and quantize ONNX model
# ---------------------------------------------------------------------------
echo "==> Optimizing and quantizing ONNX model"
OPTIMIZED_ONNX="${WORK_DIR}/optimized.onnx"
OPT_WORK_DIR="${WORK_DIR}/optimize"
MODEL_TYPE="$(echo "${MODEL_METADATA}" | jq -r '.model_type // "bert"')"
MAX_LENGTH="$(echo "${MODEL_METADATA}" | jq -r '.max_length // 512')"
case "$(uname -m)" in
    aarch64|arm*) ORT_TARGET_PLATFORM="arm" ;;
    *)            ORT_TARGET_PLATFORM="amd64" ;;
esac
python3 "$(dirname "$0")/optimize_model.py" \
    --input    "${MODEL_DIR}/${HF_PRIMARY}" \
    --output   "${OPTIMIZED_ONNX}" \
    --model_type "${MODEL_TYPE}" \
    --max_length "${MAX_LENGTH}" \
    --target_platform "${ORT_TARGET_PLATFORM}" \
    --work_dir "${OPT_WORK_DIR}"
# NOTE: ORT_TARGET_PLATFORM is set here and reused by step 11 (convert_onnx_models_to_ort.py).
# The variable must remain in scope for the rest of the script (no subshells between steps).

# Place optimized model in its own directory for convert_onnx_models_to_ort.py
# (the converter takes a directory argument and converts all .onnx files inside it)
ORT_INPUT_DIR="${WORK_DIR}/ort_input"
mkdir -p "${ORT_INPUT_DIR}"
cp "${OPTIMIZED_ONNX}" "${ORT_INPUT_DIR}/model.onnx"
```

- [ ] **Step 2: Update step 8 (operator config) to use `OPTIMIZED_ONNX`**

Change the `create_reduced_build_config.py` call from:

```bash
python3 "${ORT_SRC}/tools/python/create_reduced_build_config.py" \
    "${MODEL_DIR}" \
    "${OPERATOR_CONFIG}"
```

to:

```bash
python3 "${ORT_SRC}/tools/python/create_reduced_build_config.py" \
    "${OPTIMIZED_ONNX}" \
    "${OPERATOR_CONFIG}"
```

Note: `create_reduced_build_config.py` accepts either a directory (scans for `.onnx` files) or a single `.onnx` file path. Passing `${OPTIMIZED_ONNX}` directly is valid and more precise — it avoids accidentally picking up other `.onnx` files in `MODEL_DIR`.

Also update the step header comment number from `7` to `8`.

- [ ] **Step 3: Renumber step headers 8–13 → 9–14/15**

Update comment headers only (the actual logic is unchanged):
- `# 8. Build ORT` → `# 9. Build ORT`
- `# 9. Verify libonnxruntime.so exists` → `# 10. Verify libonnxruntime.so exists`
- `# 10. Convert ONNX model to ORT format` → `# 11. Convert ONNX model to ORT format`
- `# 11. Compile and run smoke test` → `# 12. Compile and run smoke test`
- `# 12. Stage artifacts` → `# 13. Stage artifacts`
- `# 12b. Prune stage dir` → `# 14. Prune stage dir`
- `# 13. Create tarball` → `# 15. Create tarball`
- `# 14. Success` → `# 16. Success`

- [ ] **Step 4: Update step 11 (ORT convert) to use `ORT_INPUT_DIR`**

Change the `convert_onnx_models_to_ort.py` call. The current call is:

```bash
ORT_MODEL_DIR="${WORK_DIR}/ort_model"
mkdir -p "${ORT_MODEL_DIR}"
case "$(uname -m)" in
    aarch64|arm*) ORT_TARGET_PLATFORM="arm" ;;
    *)            ORT_TARGET_PLATFORM="amd64" ;;
esac
python3 "${ORT_SRC}/tools/python/convert_onnx_models_to_ort.py" \
    "${MODEL_DIR}/$(dirname "${HF_PRIMARY}")" \
    --optimization_style Fixed \
    --target_platform "${ORT_TARGET_PLATFORM}" \
    --output_dir "${ORT_MODEL_DIR}"
```

Replace with (note: `ORT_TARGET_PLATFORM` is now set in step 7, remove the duplicate `case` block):

```bash
ORT_MODEL_DIR="${WORK_DIR}/ort_model"
mkdir -p "${ORT_MODEL_DIR}"
python3 "${ORT_SRC}/tools/python/convert_onnx_models_to_ort.py" \
    "${ORT_INPUT_DIR}" \
    --optimization_style Fixed \
    --target_platform "${ORT_TARGET_PLATFORM}" \
    --output_dir "${ORT_MODEL_DIR}"
```

- [ ] **Step 5: Review the full diff and run bash syntax check**

```bash
git diff scripts/build_target.sh
bash -n scripts/build_target.sh
```

Verify from the diff:
- Step 7 is new (optimize_model.py call)
- Step 8 op-config uses `${OPTIMIZED_ONNX}` not `${MODEL_DIR}`
- Step 11 convert uses `${ORT_INPUT_DIR}` not `${MODEL_DIR}/...`
- `ORT_TARGET_PLATFORM` is defined once (in step 7), not twice
- Step headers 8–16 are renumbered correctly

Expected from `bash -n`: no output, exit 0

- [ ] **Step 6: Commit**

```bash
git add scripts/build_target.sh
git commit -m "feat: integrate optimize_model.py into build pipeline"
```

---

## Task 5: Final check

- [ ] **Step 1: Run full test suite**

```bash
.venv/bin/pytest tests/ -v
```

Expected: all PASS

- [ ] **Step 2: Validate manifest**

```bash
.venv/bin/python scripts/validate_manifest.py builds/release.yaml
```

Expected: `OK: manifest is valid`

- [ ] **Step 3: Commit and push (if not already pushed)**

```bash
git push
```

---

## Notes for implementer

- **`onnxruntime.tools.onnx_model_utils.make_dim_param_fixed`** signature: `(graph, dim_param: str, value: int)`. It modifies the graph in-place; call `onnx.save()` after.
- **`SymbolicShapeInference.infer_shapes`** returns a new model object; do not mutate the input.
- **`SessionOptions.optimized_model_filepath`** must be set *before* creating the `InferenceSession`; the session creation triggers optimization and writes the file.
- **`quantize_dynamic`** produces a new file at the output path; it does not modify the input file.
- The `_find_batch_and_seq_dims` helper is pure Python (no onnxruntime import) so it can be tested without the Docker environment.
- `MODEL_TYPE` and `MAX_LENGTH` in the bash script use `// "bert"` and `// 512` as jq fallbacks; this means targets without `model_type` in metadata default to `bert` rather than failing the build.
