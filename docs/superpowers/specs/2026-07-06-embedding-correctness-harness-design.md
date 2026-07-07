# Embedding-Correctness Harness — Design

**Date:** 2026-07-06
**Status:** Approved (design); implementation pending
**Branch context:** `codex/eurobert-int8-correctness`

## Context

The build pipeline produces a minimal `libonnxruntime.so` plus a `model.ort`, and
today the only runtime check is `scripts/smoke_test.c`, which loads the model, feeds
**zero-filled** inputs, and passes if inference does not crash. It never inspects
output values, so a build that runs but produces *wrong* embeddings would still ship.

This is especially risky for int8: `optimize_model.py` step 4 applies our own
`QInt8` dynamic quantization, and quantization damage is exactly the kind of failure
that a crash-only smoke test cannot see. (The current jina target is pre-quantized
`q4f16` upstream, so step 4 is skipped; a target we quantize ourselves — e.g. an
EuroBERT export — would exercise it.)

**Goal:** turn the smoke test into a numeric correctness gate that verifies the
shipped artifact's embeddings match a full-precision reference within tolerance.
The mechanism is general (runs for every target); EuroBERT is simply the first
int8 target that motivates it.

Adding an actual EuroBERT manifest target is **out of scope** for this spec (it
needs a named HF repo with an ONNX export) and will be a separate follow-up that
reuses this harness.

## What a failure catches

Because every failure mode manifests as a diverged output tensor, one cosine-
similarity gate covers all of:
- int8 quantization damage (step 4),
- ONNX → ORT conversion regressions (`convert_onnx_models_to_ort.py`),
- reduced-operator minimal-build breakage (missing/incorrect kernels),
- wrong kernel selection at runtime.

## Design decisions (settled)

| Decision | Choice |
|----------|--------|
| Where the check runs | Extend the C smoke test against the shipped `.so` + `model.ort` |
| Reference | Full-precision (pre-int8) ONNX run through the pip `onnxruntime` at build time |
| Pass criterion | Cosine similarity of raw first output tensor ≥ threshold |
| Inputs | Real text tokenized from `tests/data/jane-austen_pride-and-prejudice.jsonl` |
| Defaults | `cosine_threshold = 0.99`, `num_samples = 3`, `max_tokens = 128` |
| Fixture reachability | New read-only bind mount of `tests/data` into the container |
| Failure behavior | Hard-fail the build (same as today's smoke test) |

**Rejected alternative:** a Python-only check against the optimized `.onnx`. Easier
to write, but it validates the `.onnx`, not the minimal `.so`/`.ort` that actually
ships.

## Components & data flow

Wired into `scripts/build_target.sh` between ORT conversion (step 8) and the smoke
test (step 12):

### 1. Reference emit — change to `scripts/optimize_model.py`
Add an optional `--reference-output PATH` argument. When set, the pipeline also
writes the **pre-int8 graph** — the existing step-3 output (`04_ort_optimized.onnx`,
full precision, `ORT_ENABLE_ALL`-optimized) — to `PATH`. This is the correctness
reference.

- For targets we quantize (`step 4` runs), the reference is the fp graph and the
  gate measures quantization + conversion + build fidelity together.
- For pre-quantized targets (`--skip-int8-quantize`), the reference equals the
  shipped graph, so the gate degenerates to a **build-fidelity** check (expected
  cosine ≈ 1.0). Still meaningful — it catches conversion/minimal-build breakage.

Rationale for a stable flag over reaching into `optimize_model.py`'s internal
work-dir filenames: keeps `build_target.sh` decoupled from intermediate naming.

### 2. New `scripts/gen_reference_vectors.py` (runs in-container)
Uses the pip `onnxruntime` and the new `tokenizers` dependency. Steps:

1. Load `tokenizer.json` (already downloaded as a companion) via
   `tokenizers.Tokenizer.from_file(...)`.
2. Read the first `num_samples` text records from the mounted
   `pride-and-prejudice.jsonl`; tokenize each, truncated to `max_tokens`.
3. Inspect the **reference ONNX** input signature and build exactly the tensors it
   declares — `input_ids`, `attention_mask` (all ones over the token length), and
   `token_type_ids` (zeros) only if the model has that input.
4. Run the reference ONNX through pip ORT (`CPUExecutionProvider`); capture the
   **first output tensor** for each sample.
5. Write a compact **binary test-vectors file** consumed by the C smoke test.

**Binary test-vectors file format** (little-endian; exact layout finalized in the
plan):
- Header: magic, version, `num_samples`.
- Per sample:
  - `num_inputs`; per input → name length + name bytes, `ndim`, `dims[]`,
    ONNX element-type enum, and the raw token data.
  - reference output: element count + flattened `float32` values.

A binary format is chosen over JSON so the C reader stays small (no JSON parser).

### 3. Extended `scripts/smoke_test.c`
Replace zero-fill with test-vector-driven comparison:
- Read the test-vectors file path (new argv or env).
- For each sample: create input tensors from the provided token data with the
  provided dtypes/dims, `Run` the session (keeping `ORT_DISABLE_ALL`), take the
  first output tensor, cast fp16 → fp32 if needed, and compute cosine similarity
  against the sample's reference vector.
- Exit non-zero if **any** sample is below `cosine_threshold`; print each sample's
  similarity for diagnostics.

Comparison is over the **raw first output tensor** flattened (e.g. token embeddings
`[1, seq, hidden]`) — no pooling/normalization is replicated in C, because matching
raw outputs implies matching pooled sentence embeddings.

## Configuration

Optional per-target manifest block, read from `MODEL_METADATA`:

```yaml
metadata:
  correctness:
    cosine_threshold: 0.99   # default 0.99
    num_samples: 3           # default 3
    max_tokens: 128          # default 128
```

Absent block → code defaults. `validate_manifest.py` accepts the block as optional
(no hard requirement), so existing targets need no change.

## Dependencies & wiring

- **Build image** (`docker/lambda-build.Dockerfile`): add `tokenizers` to the pip
  install list (lightweight, Rust-backed; no `transformers`/`torch` needed for
  tokenization).
- **Fixture mount** (`.github/workflows/build.yml`): add
  `-v "$(pwd)/tests/data:/fixtures:ro"` to the `docker run` invocation and pass the
  in-container fixture path to `build_target.sh` (via env, e.g. `FIXTURE_DIR`).
  Keeps the committed fixture as the single source of truth.
- **`build_target.sh`**: add `--reference-output <path>` to the **existing** step-7
  `optimize_model.py` invocation (no second call) so the pre-int8 reference graph is
  emitted alongside the optimized model. Then, after ORT conversion, run
  `gen_reference_vectors.py` to produce the test-vectors file, and invoke the
  extended smoke test with that file. Both the reference ONNX and the test-vectors
  file are build intermediates under `WORK_DIR` (cleaned by the existing `trap`) and
  are **not** added to the bundle whitelist.

## Testing (of the harness itself)

- **Unit tests** for `gen_reference_vectors.py` (new `tests/test_gen_reference_vectors.py`),
  following the monkeypatch style in `tests/test_optimize_model.py`:
  - input-tensor selection from an ONNX signature with vs. without `token_type_ids`;
  - tokenization/truncation produces the expected shapes;
  - binary test-vectors round-trip (write then re-read).
- **`optimize_model.py`**: test that `--reference-output` writes the pre-int8 graph
  and that omitting it preserves current behavior.
- **`tests/test_ci_contracts.py`** guards (so future edits can't silently regress):
  - Dockerfile installs `tokenizers`;
  - workflow mounts the fixtures directory;
  - `smoke_test.c` contains the cosine-similarity gate (not crash-only);
  - `onnxruntime==1.27.0` pin unchanged.
- The `plan` job already runs the targeted pytest set; add the new test file to it.

## Out of scope

- Adding a concrete EuroBERT manifest target (separate follow-up; needs a named HF
  repo with an ONNX export).
- Pooling/normalization parity or semantic-quality benchmarking beyond cosine
  similarity of raw outputs.
- Comparing against an independent PyTorch/sentence-transformers ground truth.

## Verification (post-implementation)

- Unit + contract tests:
  `pytest tests/test_validate_manifest.py tests/test_optimize_model.py tests/test_ci_contracts.py tests/test_gen_reference_vectors.py -q`.
- End-to-end: run a container build for the existing jina target; confirm the smoke
  test now prints per-sample cosine similarities and passes (≈ 1.0, since jina is
  pre-quantized so reference == shipped graph).
- Negative check: perturb the reference vectors (or lower a value) and confirm the
  smoke test exits non-zero — proving the gate actually fails on divergence.
