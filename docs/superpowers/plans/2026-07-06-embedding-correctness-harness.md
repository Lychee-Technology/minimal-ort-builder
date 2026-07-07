# Embedding-Correctness Harness Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Turn the crash-only C smoke test into a numeric cosine-similarity gate that verifies the shipped minimal `.so` + `model.ort` reproduces a full-precision ONNX reference on real tokenized inputs.

**Architecture:** `optimize_model.py` emits the pre-int8 graph as a reference. A new `gen_reference_vectors.py` (runs in-container) tokenizes fixture text, runs the reference ONNX through pip `onnxruntime`, and writes a compact binary test-vectors file. `smoke_test.c` gains a comparison mode that feeds those token ids to the minimal artifact and asserts cosine similarity ≥ threshold. `build_target.sh` and the CI workflow wire it together.

**Tech Stack:** Python 3.11+ (`onnxruntime`, `numpy`, `tokenizers`), C (ORT C API + `-lm`), Bash, GitHub Actions, pytest.

## Global Constraints

- ONNX Runtime pin is `onnxruntime==1.27.0` — do not change it. (Dockerfile pip pin and `builds/release.yaml` `v1.27.0`.)
- ORT graph optimization stays at `ORT_ENABLE_ALL`; smoke test keeps `ORT_DISABLE_ALL`. Do not alter these.
- Correctness defaults: `cosine_threshold = 0.99`, `num_samples = 3`, `max_tokens = 128`.
- Binary test-vectors format is little-endian, magic `TVB1`; dtype codes `0=float32, 1=float16, 2=int64, 3=int32`.
- The build container mounts only `/scripts` (repo `scripts/`), `/manifest` (repo `builds/`), `/output`, and (added by this plan) `/fixtures` (repo `tests/data/`). Nothing else from the repo is reachable at build time.
- Adding a concrete EuroBERT manifest target is OUT OF SCOPE (needs a named HF repo with an ONNX export); this plan only builds the general harness.
- Follow existing patterns: pytest tests load scripts by path via `importlib` and monkeypatch `sys.modules` for `onnxruntime` (see `tests/test_optimize_model.py`).

---

### Task 1: Add `tokenizers` to the build image

**Files:**
- Modify: `docker/lambda-build.Dockerfile` (pip install list ~line 34-42; sanity-check `RUN` ~line 51)
- Test: `tests/test_ci_contracts.py`

**Interfaces:**
- Produces: build image with importable `tokenizers`; a contract test guarding it.

- [ ] **Step 1: Write the failing test**

Add to `tests/test_ci_contracts.py`:

```python
def test_dockerfile_installs_tokenizers() -> None:
    """The build image must install the tokenizers library for correctness vectors."""
    text = DOCKERFILE.read_text(encoding="utf-8")
    assert "tokenizers" in text
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_ci_contracts.py::test_dockerfile_installs_tokenizers -q`
Expected: FAIL (`tokenizers` not in Dockerfile).

- [ ] **Step 3: Add the dependency**

In `docker/lambda-build.Dockerfile`, add `tokenizers` to the first pip install block (the one ending with `"onnxruntime==1.27.0"`), on its own line before `"onnxruntime==1.27.0"`:

```dockerfile
        onnx \
        flatbuffers \
        tokenizers \
        "onnxruntime==1.27.0"
```

Then extend the sanity-check `RUN` import line to also import `tokenizers`:

```dockerfile
    && python3 -c "import torch; import onnxruntime; import tokenizers; from onnxruntime.transformers import optimizer; print(torch.__version__)"
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_ci_contracts.py::test_dockerfile_installs_tokenizers -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add docker/lambda-build.Dockerfile tests/test_ci_contracts.py
git commit -m "feat: install tokenizers in build image for correctness vectors"
```

---

### Task 2: `optimize_model.py --reference-output`

**Files:**
- Modify: `scripts/optimize_model.py` (argparse block ~line 191-228; main body ~line 270-278)
- Test: `tests/test_optimize_model.py`

**Interfaces:**
- Produces: `optimize_model.py` accepts `--reference-output PATH`; when set, copies the step-3 (pre-int8) graph to `PATH`. Consumed by `build_target.sh` (Task 7).

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_optimize_model.py`:

```python
def test_help_mentions_reference_output_flag():
    result = _run("--help")
    assert result.returncode == 0
    assert "--reference-output" in result.stdout


def test_main_writes_reference_output_when_requested(monkeypatch, tmp_path):
    """--reference-output must persist the pre-int8 (step 3) graph."""
    module = _load_module()

    monkeypatch.setattr(module, "_step0_inline_external_data",
                        lambda i, o: Path(o).write_bytes(b"s0"))
    monkeypatch.setattr(module, "_step1_transformer_opt",
                        lambda i, o, mt: Path(o).write_bytes(b"s1"))
    monkeypatch.setattr(module, "_step2b_shape_inference",
                        lambda i, o: Path(o).write_bytes(b"s2b"))
    monkeypatch.setattr(module, "_step3_ort_graph_opt",
                        lambda i, o: Path(o).write_bytes(b"REF"))
    monkeypatch.setattr(module, "_step4_int8_quantize",
                        lambda i, o: Path(o).write_bytes(b"q"))

    input_path = tmp_path / "model.onnx"
    input_path.write_bytes(b"x")
    ref_path = tmp_path / "reference.onnx"
    out_path = tmp_path / "out.onnx"

    old_argv = sys.argv
    sys.argv = [
        str(SCRIPT),
        "--input", str(input_path),
        "--output", str(out_path),
        "--model_type", "bert",
        "--target_platform", "arm",
        "--work_dir", str(tmp_path / "work"),
        "--reference-output", str(ref_path),
    ]
    try:
        module.main()
    finally:
        sys.argv = old_argv

    assert ref_path.read_bytes() == b"REF"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_optimize_model.py::test_main_writes_reference_output_when_requested tests/test_optimize_model.py::test_help_mentions_reference_output_flag -q`
Expected: FAIL (unrecognized argument `--reference-output`).

- [ ] **Step 3: Add the argument and emit logic**

In `scripts/optimize_model.py`, add to the argparse block (after `--skip-int8-quantize`):

```python
    parser.add_argument(
        "--reference-output",
        type=Path,
        help="Also write the pre-int8 (step 3) graph here, for correctness comparison",
    )
```

In `main()`, immediately AFTER the `_step3_ort_graph_opt(step2b_out, step3_out)` call (and before the `if args.skip_int8_quantize:` block):

```python
    if args.reference_output is not None:
        args.reference_output.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(step3_out, args.reference_output)
        print(f"    reference graph written: {args.reference_output}")
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_optimize_model.py -q`
Expected: PASS (all, including existing tests).

- [ ] **Step 5: Commit**

```bash
git add scripts/optimize_model.py tests/test_optimize_model.py
git commit -m "feat: add --reference-output to emit pre-int8 graph"
```

---

### Task 3: `gen_reference_vectors.py` pure helpers + binary format

**Files:**
- Create: `scripts/gen_reference_vectors.py`
- Test: `tests/test_gen_reference_vectors.py`

**Interfaces:**
- Produces:
  - `truncate_ids(ids: list[int], max_tokens: int) -> list[int]`
  - `build_feeds(input_names: list[str], ids: list[int]) -> dict[str, np.ndarray]` — returns int64 `[1, len(ids)]` arrays for whichever of `input_ids`/`attention_mask`/`token_type_ids` appear in `input_names`, preserving `input_names` order.
  - `write_vectors(path: str, samples: list[dict]) -> None` and `read_vectors(path: str) -> list[dict]` — round-trip of `{"inputs": dict[str, np.ndarray], "reference": np.ndarray(float32)}`.
- Consumed by: the module's own `main()` (Task 4), the C reader (Task 5), and tests.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_gen_reference_vectors.py`:

```python
"""Tests for scripts/gen_reference_vectors.py."""

import importlib.util
import sys
import types
from pathlib import Path

import numpy as np

SCRIPT = Path(__file__).parent.parent / "scripts" / "gen_reference_vectors.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("gen_reference_vectors", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_truncate_ids():
    module = _load_module()
    assert module.truncate_ids([1, 2, 3, 4, 5], 3) == [1, 2, 3]
    assert module.truncate_ids([1, 2], 5) == [1, 2]


def test_build_feeds_includes_token_type_ids_when_present():
    module = _load_module()
    feeds = module.build_feeds(["input_ids", "attention_mask", "token_type_ids"], [5, 6, 7])
    assert set(feeds) == {"input_ids", "attention_mask", "token_type_ids"}
    assert feeds["input_ids"].tolist() == [[5, 6, 7]]
    assert feeds["attention_mask"].tolist() == [[1, 1, 1]]
    assert feeds["token_type_ids"].tolist() == [[0, 0, 0]]
    assert feeds["input_ids"].dtype == np.int64


def test_build_feeds_omits_token_type_ids_when_absent():
    module = _load_module()
    feeds = module.build_feeds(["input_ids", "attention_mask"], [5, 6])
    assert set(feeds) == {"input_ids", "attention_mask"}


def test_write_read_vectors_roundtrip(tmp_path):
    module = _load_module()
    samples = [{
        "inputs": {
            "input_ids": np.array([[1, 2, 3]], dtype=np.int64),
            "attention_mask": np.array([[1, 1, 1]], dtype=np.int64),
        },
        "reference": np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float32),
    }]
    p = tmp_path / "v.tvbin"
    module.write_vectors(str(p), samples)
    r = module.read_vectors(str(p))
    assert len(r) == 1
    assert list(r[0]["inputs"].keys()) == ["input_ids", "attention_mask"]
    assert r[0]["inputs"]["input_ids"].tolist() == [[1, 2, 3]]
    assert r[0]["inputs"]["input_ids"].dtype == np.int64
    assert np.allclose(r[0]["reference"], [0.1, 0.2, 0.3, 0.4], atol=1e-6)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_gen_reference_vectors.py -q`
Expected: FAIL (module does not exist).

- [ ] **Step 3: Create the module with helpers**

Create `scripts/gen_reference_vectors.py`:

```python
#!/usr/bin/env python3
"""gen_reference_vectors.py — build correctness test vectors for the smoke test.

Runs inside the lambda-build container. Tokenizes fixture text with the model's
tokenizer.json, runs the pre-int8 reference ONNX through the pip onnxruntime, and
writes a compact binary file (magic 'TVB1', little-endian) of token inputs plus the
reference output for each sample. The C smoke test replays these inputs against the
shipped minimal .so/.ort and compares outputs by cosine similarity.
"""

import argparse
import json
import struct
import sys
from pathlib import Path

import numpy as np

MAGIC = b"TVB1"
KNOWN_INPUTS = ("input_ids", "attention_mask", "token_type_ids")

_DTYPE_TO_CODE = {
    np.dtype("float32"): 0,
    np.dtype("float16"): 1,
    np.dtype("int64"): 2,
    np.dtype("int32"): 3,
}
_CODE_TO_DTYPE = {code: dt for dt, code in _DTYPE_TO_CODE.items()}


def truncate_ids(ids, max_tokens):
    return list(ids)[:max_tokens]


def build_feeds(input_names, ids):
    """Build model feeds for whichever known inputs the model declares."""
    length = len(ids)
    feeds = {}
    for name in input_names:
        if name == "input_ids":
            feeds[name] = np.array([list(ids)], dtype=np.int64)
        elif name == "attention_mask":
            feeds[name] = np.ones((1, length), dtype=np.int64)
        elif name == "token_type_ids":
            feeds[name] = np.zeros((1, length), dtype=np.int64)
    return feeds


def write_vectors(path, samples):
    with open(path, "wb") as f:
        f.write(MAGIC)
        f.write(struct.pack("<I", len(samples)))
        for sample in samples:
            feeds = sample["inputs"]
            f.write(struct.pack("<I", len(feeds)))
            for name, arr in feeds.items():
                name_bytes = name.encode("utf-8")
                f.write(struct.pack("<I", len(name_bytes)))
                f.write(name_bytes)
                f.write(struct.pack("<I", _DTYPE_TO_CODE[arr.dtype]))
                f.write(struct.pack("<I", arr.ndim))
                for dim in arr.shape:
                    f.write(struct.pack("<q", int(dim)))
                data = np.ascontiguousarray(arr).tobytes()
                f.write(struct.pack("<Q", len(data)))
                f.write(data)
            ref = np.ascontiguousarray(
                np.asarray(sample["reference"], dtype=np.float32).reshape(-1)
            )
            f.write(struct.pack("<Q", ref.size))
            f.write(ref.tobytes())


def read_vectors(path):
    with open(path, "rb") as f:
        magic = f.read(4)
        if magic != MAGIC:
            raise ValueError(f"bad magic: {magic!r}")
        (num_samples,) = struct.unpack("<I", f.read(4))
        samples = []
        for _ in range(num_samples):
            (num_inputs,) = struct.unpack("<I", f.read(4))
            inputs = {}
            for _ in range(num_inputs):
                (name_len,) = struct.unpack("<I", f.read(4))
                name = f.read(name_len).decode("utf-8")
                (code,) = struct.unpack("<I", f.read(4))
                (ndim,) = struct.unpack("<I", f.read(4))
                dims = [struct.unpack("<q", f.read(8))[0] for _ in range(ndim)]
                (nbytes,) = struct.unpack("<Q", f.read(8))
                raw = f.read(nbytes)
                arr = np.frombuffer(raw, dtype=_CODE_TO_DTYPE[code]).reshape(dims)
                inputs[name] = arr
            (ref_count,) = struct.unpack("<Q", f.read(8))
            ref = np.frombuffer(f.read(ref_count * 4), dtype=np.float32).copy()
            samples.append({"inputs": inputs, "reference": ref})
        return samples
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_gen_reference_vectors.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add scripts/gen_reference_vectors.py tests/test_gen_reference_vectors.py
git commit -m "feat: add reference-vector helpers and binary format"
```

---

### Task 4: `gen_reference_vectors.py` main orchestration

**Files:**
- Modify: `scripts/gen_reference_vectors.py` (append `read_texts` + `main`)
- Test: `tests/test_gen_reference_vectors.py`

**Interfaces:**
- Consumes: `truncate_ids`, `build_feeds`, `write_vectors` (Task 3).
- Produces: `read_texts(fixture_path: str, num_samples: int) -> list[str]`; `main(argv: list[str] | None = None) -> None` with flags `--model --tokenizer --fixture --output --num-samples --max-tokens`. Imports `onnxruntime` and `tokenizers` lazily inside `main`.

- [ ] **Step 1: Write the failing test**

Add to `tests/test_gen_reference_vectors.py`:

```python
def test_main_writes_vectors_file(monkeypatch, tmp_path):
    module = _load_module()

    class FakeEnc:
        def __init__(self, ids):
            self.ids = ids

    class FakeTokenizer:
        @classmethod
        def from_file(cls, path):
            return cls()

        def encode(self, text):
            return FakeEnc([1, 2, 3, 4])

    fake_tok_mod = types.ModuleType("tokenizers")
    fake_tok_mod.Tokenizer = FakeTokenizer
    monkeypatch.setitem(sys.modules, "tokenizers", fake_tok_mod)

    class FakeIO:
        def __init__(self, name):
            self.name = name

    class FakeSession:
        def get_inputs(self):
            return [FakeIO("input_ids"), FakeIO("attention_mask")]

        def get_outputs(self):
            return [FakeIO("last_hidden_state")]

        def run(self, output_names, feeds):
            seq = feeds["input_ids"].shape[1]
            return [np.ones((1, seq, 2), dtype=np.float32)]

    fake_ort = types.ModuleType("onnxruntime")
    fake_ort.InferenceSession = lambda *a, **k: FakeSession()
    monkeypatch.setitem(sys.modules, "onnxruntime", fake_ort)

    fixture = tmp_path / "f.jsonl"
    fixture.write_text(
        '{"text":"hello world"}\n{"text":"second sentence"}\n{"text":"third one"}\n',
        encoding="utf-8",
    )
    tokenizer = tmp_path / "tokenizer.json"
    tokenizer.write_text("{}", encoding="utf-8")
    model = tmp_path / "ref.onnx"
    model.write_bytes(b"x")
    out = tmp_path / "v.tvbin"

    module.main([
        "--model", str(model),
        "--tokenizer", str(tokenizer),
        "--fixture", str(fixture),
        "--output", str(out),
        "--num-samples", "2",
        "--max-tokens", "3",
    ])

    r = module.read_vectors(str(out))
    assert len(r) == 2
    # ids [1,2,3,4] truncated to 3 -> output (1,3,2) -> 6 floats
    assert r[0]["reference"].size == 6
    assert r[0]["inputs"]["input_ids"].tolist() == [[1, 2, 3]]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_gen_reference_vectors.py::test_main_writes_vectors_file -q`
Expected: FAIL (`main` / `read_texts` not defined).

- [ ] **Step 3: Append `read_texts` and `main`**

Append to `scripts/gen_reference_vectors.py`:

```python
def read_texts(fixture_path, num_samples):
    texts = []
    with open(fixture_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            text = record.get("text")
            if not text:
                continue
            texts.append(text)
            if len(texts) >= num_samples:
                break
    if not texts:
        raise ValueError(f"no usable 'text' records in {fixture_path}")
    return texts


def main(argv=None):
    parser = argparse.ArgumentParser(description="Generate correctness test vectors")
    parser.add_argument("--model", required=True, help="Reference (pre-int8) ONNX path")
    parser.add_argument("--tokenizer", required=True, help="tokenizer.json path")
    parser.add_argument("--fixture", required=True, help="JSONL fixture with 'text' field")
    parser.add_argument("--output", required=True, help="Output .tvbin path")
    parser.add_argument("--num-samples", type=int, default=3)
    parser.add_argument("--max-tokens", type=int, default=128)
    args = parser.parse_args(argv)

    import onnxruntime as ort
    from tokenizers import Tokenizer

    tokenizer = Tokenizer.from_file(args.tokenizer)
    session = ort.InferenceSession(args.model, providers=["CPUExecutionProvider"])
    input_names = [i.name for i in session.get_inputs()]
    output_name = session.get_outputs()[0].name

    texts = read_texts(args.fixture, args.num_samples)
    samples = []
    for text in texts:
        ids = truncate_ids(tokenizer.encode(text).ids, args.max_tokens)
        feeds = build_feeds(input_names, ids)
        output = session.run([output_name], feeds)[0]
        reference = np.asarray(output, dtype=np.float32).reshape(-1)
        samples.append({"inputs": feeds, "reference": reference})

    write_vectors(args.output, samples)
    print(f"    wrote {len(samples)} reference sample(s) to {args.output}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_gen_reference_vectors.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add scripts/gen_reference_vectors.py tests/test_gen_reference_vectors.py
git commit -m "feat: orchestrate reference-vector generation"
```

---

### Task 5: Comparison mode in `smoke_test.c`

**Files:**
- Modify: `scripts/smoke_test.c` (full rewrite — adds comparison mode, keeps zero-fill)
- Test: `tests/test_ci_contracts.py`

**Interfaces:**
- Consumes: the `.tvbin` file written by `write_vectors` (Task 3).
- Produces: `./smoke_test <model.ort>` (zero-fill, legacy) and `./smoke_test <model.ort> <vectors.tvbin> <cosine_threshold>` (comparison; exit non-zero if any sample below threshold). Consumed by `build_target.sh` (Task 7). Requires linking `-lm`.

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_ci_contracts.py`:

```python
def test_smoke_test_has_cosine_similarity_gate() -> None:
    """Smoke test must compare outputs by cosine similarity against a threshold."""
    text = SMOKE_TEST.read_text(encoding="utf-8")
    assert "cosine" in text
    assert "threshold" in text
    assert "TVB1" in text


def test_smoke_test_keeps_zero_fill_mode() -> None:
    """The single-arg zero-fill path must still exist for debugging."""
    text = SMOKE_TEST.read_text(encoding="utf-8")
    assert "run_zerofill" in text
    assert "run_comparison" in text
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_ci_contracts.py::test_smoke_test_has_cosine_similarity_gate tests/test_ci_contracts.py::test_smoke_test_keeps_zero_fill_mode -q`
Expected: FAIL.

- [ ] **Step 3: Rewrite `scripts/smoke_test.c`**

Replace the entire file with:

```c
/*
 * smoke_test.c — minimal ORT session loader with two modes.
 *
 *   ./smoke_test <model.ort>
 *       Zero-fill inference; exit 0 if the session loads and Run() succeeds.
 *
 *   ./smoke_test <model.ort> <vectors.tvbin> <cosine_threshold>
 *       Replay tokenized inputs from the test-vectors file and compare the first
 *       output tensor to the stored reference by cosine similarity. Exit 0 only
 *       if every sample is >= threshold.
 *
 * Runtime graph optimization is disabled (ORT_DISABLE_ALL): the model is already
 * fully optimized offline, and re-optimizing at load time introduces fused ops
 * whose shape requirements conflict with broadcast attention bias.
 */
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "onnxruntime_c_api.h"

#define ORT_CHECK(expr, label)                                              \
    do {                                                                    \
        OrtStatus *_s = (expr);                                             \
        if (_s) {                                                           \
            fprintf(stderr, "SMOKE FAIL: %s: %s\n",                         \
                    (label), api->GetErrorMessage(_s));                     \
            api->ReleaseStatus(_s);                                         \
            goto cleanup;                                                   \
        }                                                                   \
    } while (0)

static ONNXTensorElementDataType code_to_ort(uint32_t code) {
    switch (code) {
        case 0: return ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT;
        case 1: return ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16;
        case 2: return ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64;
        case 3: return ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32;
        default: return ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT;
    }
}

static float half_to_float(uint16_t h) {
    uint32_t sign = (uint32_t)(h & 0x8000u) << 16;
    uint32_t exp  = (h >> 10) & 0x1Fu;
    uint32_t mant = h & 0x3FFu;
    uint32_t f;
    if (exp == 0) {
        if (mant == 0) {
            f = sign;
        } else {
            exp = 127 - 15 + 1;
            while ((mant & 0x400u) == 0) { mant <<= 1; exp--; }
            mant &= 0x3FFu;
            f = sign | (exp << 23) | (mant << 13);
        }
    } else if (exp == 0x1Fu) {
        f = sign | 0x7F800000u | (mant << 13);
    } else {
        f = sign | ((exp - 15 + 127) << 23) | (mant << 13);
    }
    float out;
    memcpy(&out, &f, sizeof(out));
    return out;
}

/* Convert the first output tensor to a newly-allocated float array. */
static float *tensor_to_float(const OrtApi *api, OrtValue *val, size_t *out_count) {
    OrtTensorTypeAndShapeInfo *info = NULL;
    if (api->GetTensorTypeAndShape(val, &info)) return NULL;
    size_t count = 0;
    api->GetTensorShapeElementCount(info, &count);
    ONNXTensorElementDataType t;
    api->GetTensorElementType(info, &t);
    api->ReleaseTensorTypeAndShapeInfo(info);

    void *data = NULL;
    if (api->GetTensorMutableData(val, &data)) return NULL;

    float *out = malloc((count ? count : 1) * sizeof(float));
    if (!out) return NULL;
    for (size_t i = 0; i < count; i++) {
        switch (t) {
            case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT:
                out[i] = ((float *)data)[i]; break;
            case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16:
                out[i] = half_to_float(((uint16_t *)data)[i]); break;
            case ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE:
                out[i] = (float)((double *)data)[i]; break;
            default:
                out[i] = 0.0f; break;
        }
    }
    *out_count = count;
    return out;
}

static double cosine(const float *a, const float *b, size_t n) {
    double dot = 0.0, na = 0.0, nb = 0.0;
    for (size_t i = 0; i < n; i++) {
        dot += (double)a[i] * (double)b[i];
        na  += (double)a[i] * (double)a[i];
        nb  += (double)b[i] * (double)b[i];
    }
    if (na == 0.0 || nb == 0.0) return 0.0;
    return dot / (sqrt(na) * sqrt(nb));
}

static int read_exact(FILE *f, void *dst, size_t n) {
    return fread(dst, 1, n, f) == n;
}

/* Comparison mode: replay tokenized inputs, compare output[0] by cosine. */
static int run_comparison(const OrtApi *api, OrtSession *session,
                          OrtMemoryInfo *mem_info, OrtAllocator *allocator,
                          const char *tvpath, double threshold) {
    int rc = 1;
    char *out_name = NULL;
    FILE *f = fopen(tvpath, "rb");
    if (!f) {
        fprintf(stderr, "SMOKE FAIL: cannot open %s\n", tvpath);
        return 1;
    }

    char magic[4];
    uint32_t num_samples = 0;
    if (!read_exact(f, magic, 4) || memcmp(magic, "TVB1", 4) != 0 ||
        !read_exact(f, &num_samples, 4)) {
        fprintf(stderr, "SMOKE FAIL: bad or truncated test-vectors header\n");
        fclose(f);
        return 1;
    }

    if (api->SessionGetOutputName(session, 0, allocator, &out_name)) {
        fprintf(stderr, "SMOKE FAIL: SessionGetOutputName failed\n");
        fclose(f);
        return 1;
    }

    int all_ok = 1;
    for (uint32_t si = 0; si < num_samples; si++) {
        uint32_t num_inputs = 0;
        if (!read_exact(f, &num_inputs, 4)) {
            fprintf(stderr, "SMOKE FAIL: truncated sample %u\n", si);
            all_ok = 0;
            break;
        }

        char **names = calloc(num_inputs, sizeof(char *));
        OrtValue **tensors = calloc(num_inputs, sizeof(OrtValue *));
        void **bufs = calloc(num_inputs, sizeof(void *));
        int sample_bad = (!names || !tensors || !bufs);

        for (uint32_t ii = 0; ii < num_inputs && !sample_bad; ii++) {
            uint32_t name_len = 0, dcode = 0, ndim = 0;
            if (!read_exact(f, &name_len, 4)) { sample_bad = 1; break; }
            names[ii] = malloc(name_len + 1);
            if (!names[ii] || !read_exact(f, names[ii], name_len)) { sample_bad = 1; break; }
            names[ii][name_len] = '\0';
            if (!read_exact(f, &dcode, 4) || !read_exact(f, &ndim, 4)) { sample_bad = 1; break; }
            int64_t *dims = calloc(ndim ? ndim : 1, sizeof(int64_t));
            for (uint32_t d = 0; d < ndim; d++) {
                if (!read_exact(f, &dims[d], 8)) { sample_bad = 1; break; }
            }
            uint64_t nbytes = 0;
            if (sample_bad || !read_exact(f, &nbytes, 8)) { free(dims); sample_bad = 1; break; }
            bufs[ii] = malloc(nbytes ? nbytes : 1);
            if (!bufs[ii] || !read_exact(f, bufs[ii], nbytes)) { free(dims); sample_bad = 1; break; }
            OrtStatus *ts = api->CreateTensorWithDataAsOrtValue(
                mem_info, bufs[ii], nbytes, dims, ndim, code_to_ort(dcode), &tensors[ii]);
            free(dims);
            if (ts) {
                fprintf(stderr, "SMOKE FAIL: CreateTensor: %s\n", api->GetErrorMessage(ts));
                api->ReleaseStatus(ts);
                sample_bad = 1;
                break;
            }
        }

        uint64_t ref_count = 0;
        float *ref = NULL;
        if (!sample_bad && read_exact(f, &ref_count, 8)) {
            ref = malloc((ref_count ? ref_count : 1) * sizeof(float));
            if (!ref || !read_exact(f, ref, ref_count * sizeof(float))) sample_bad = 1;
        } else {
            sample_bad = 1;
        }

        OrtValue *output = NULL;
        if (!sample_bad) {
            OrtStatus *rs = api->Run(session, NULL,
                (const char *const *)names, (const OrtValue *const *)tensors, num_inputs,
                (const char *const *)&out_name, 1, &output);
            if (rs) {
                fprintf(stderr, "SMOKE FAIL: Run: %s\n", api->GetErrorMessage(rs));
                api->ReleaseStatus(rs);
                sample_bad = 1;
            }
        }

        if (!sample_bad && output) {
            size_t got = 0;
            float *ov = tensor_to_float(api, output, &got);
            if (!ov) {
                sample_bad = 1;
            } else if (got != (size_t)ref_count) {
                fprintf(stderr, "SMOKE FAIL: sample %u output count %zu != ref %llu\n",
                        si, got, (unsigned long long)ref_count);
                sample_bad = 1;
            } else {
                double sim = cosine(ov, ref, got);
                printf("  sample %u: cosine=%.6f (threshold %.6f)\n", si, sim, threshold);
                if (sim < threshold) all_ok = 0;
            }
            free(ov);
        }
        if (sample_bad) all_ok = 0;

        if (output) api->ReleaseValue(output);
        for (uint32_t ii = 0; ii < num_inputs; ii++) {
            if (tensors && tensors[ii]) api->ReleaseValue(tensors[ii]);
            if (bufs) free(bufs[ii]);
            if (names) free(names[ii]);
        }
        free(tensors);
        free(bufs);
        free(names);
        free(ref);
        if (sample_bad) break;
    }

    if (all_ok) {
        printf("SMOKE OK: all %u sample(s) within cosine threshold %.6f\n",
               num_samples, threshold);
        rc = 0;
    } else {
        fprintf(stderr, "SMOKE FAIL: one or more samples below cosine threshold\n");
    }

    api->AllocatorFree(allocator, out_name);
    fclose(f);
    return rc;
}

/* Zero-fill mode: verify the session runs on all-zero inputs. */
static int run_zerofill(const OrtApi *api, OrtSession *session,
                        OrtMemoryInfo *mem_info, OrtAllocator *allocator) {
    int exit_code = 1;
    size_t input_count = 0;
    char **input_names = NULL;
    OrtValue **input_tensors = NULL;
    void **input_bufs = NULL;
    int64_t input_ids_seq_len = 1;
    size_t output_count = 0;
    char **output_names = NULL;
    OrtValue **output_tensors = NULL;

    ORT_CHECK(api->SessionGetInputCount(session, &input_count), "SessionGetInputCount");
    input_names = calloc(input_count, sizeof(char *));
    input_tensors = calloc(input_count, sizeof(OrtValue *));
    input_bufs = calloc(input_count, sizeof(void *));
    if (!input_names || !input_tensors || !input_bufs) {
        fprintf(stderr, "SMOKE FAIL: out of memory\n");
        goto cleanup;
    }

    for (size_t i = 0; i < input_count; i++) {
        char *name = NULL;
        OrtTypeInfo *type_info = NULL;
        const OrtTensorTypeAndShapeInfo *shape_info = NULL;
        size_t ndim = 0;
        int64_t *dims = NULL;
        ORT_CHECK(api->SessionGetInputName(session, i, allocator, &name), "SessionGetInputName");
        ORT_CHECK(api->SessionGetInputTypeInfo(session, i, &type_info), "SessionGetInputTypeInfo");
        ORT_CHECK(api->CastTypeInfoToTensorInfo(type_info, &shape_info), "CastTypeInfoToTensorInfo");
        ORT_CHECK(api->GetDimensionsCount(shape_info, &ndim), "GetDimensionsCount");
        dims = calloc(ndim ? ndim : 1, sizeof(int64_t));
        if (!dims) {
            api->AllocatorFree(allocator, name);
            api->ReleaseTypeInfo(type_info);
            fprintf(stderr, "SMOKE FAIL: out of memory\n");
            goto cleanup;
        }
        ORT_CHECK(api->GetDimensions(shape_info, dims, ndim), "GetDimensions");
        if (strcmp(name, "input_ids") == 0 && ndim >= 2 && dims[1] > 0) {
            input_ids_seq_len = dims[1];
        }
        free(dims);
        api->AllocatorFree(allocator, name);
        api->ReleaseTypeInfo(type_info);
    }

    for (size_t i = 0; i < input_count; i++) {
        ORT_CHECK(api->SessionGetInputName(session, i, allocator, &input_names[i]),
                  "SessionGetInputName");
        OrtTypeInfo *type_info = NULL;
        ORT_CHECK(api->SessionGetInputTypeInfo(session, i, &type_info), "SessionGetInputTypeInfo");
        const OrtTensorTypeAndShapeInfo *shape_info = NULL;
        ORT_CHECK(api->CastTypeInfoToTensorInfo(type_info, &shape_info), "CastTypeInfoToTensorInfo");
        ONNXTensorElementDataType elem_type;
        ORT_CHECK(api->GetTensorElementType(shape_info, &elem_type), "GetTensorElementType");
        size_t ndim = 0;
        ORT_CHECK(api->GetDimensionsCount(shape_info, &ndim), "GetDimensionsCount");
        int64_t *dims = calloc(ndim ? ndim : 1, sizeof(int64_t));
        if (!dims) {
            api->ReleaseTypeInfo(type_info);
            fprintf(stderr, "SMOKE FAIL: out of memory\n");
            goto cleanup;
        }
        ORT_CHECK(api->GetDimensions(shape_info, dims, ndim), "GetDimensions");
        api->ReleaseTypeInfo(type_info);

        size_t total_elems = 1;
        for (size_t d = 0; d < ndim; d++) {
            if (dims[d] <= 0) dims[d] = 1;
            if (strcmp(input_names[i], "attention_mask") == 0 && ndim >= 2 && d == 1) {
                dims[1] = input_ids_seq_len;
            }
            total_elems *= (size_t)dims[d];
        }

        size_t elem_size;
        switch (elem_type) {
            case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT:   elem_size = 4; break;
            case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64:   elem_size = 8; break;
            case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32:   elem_size = 4; break;
            case ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE:  elem_size = 8; break;
            case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16: elem_size = 2; break;
            default:                                    elem_size = 4; break;
        }

        input_bufs[i] = calloc(total_elems, elem_size);
        if (!input_bufs[i]) {
            free(dims);
            fprintf(stderr, "SMOKE FAIL: out of memory\n");
            goto cleanup;
        }
        OrtStatus *ts = api->CreateTensorWithDataAsOrtValue(
                mem_info, input_bufs[i], total_elems * elem_size,
                dims, ndim, elem_type, &input_tensors[i]);
        free(dims);
        if (ts) {
            fprintf(stderr, "SMOKE FAIL: CreateTensorWithDataAsOrtValue: %s\n",
                    api->GetErrorMessage(ts));
            api->ReleaseStatus(ts);
            goto cleanup;
        }
        fprintf(stdout, "  input[%zu] = %s  (elem_type=%d, total_elems=%zu)\n",
                i, input_names[i], (int)elem_type, total_elems);
    }

    ORT_CHECK(api->SessionGetOutputCount(session, &output_count), "SessionGetOutputCount");
    output_names = calloc(output_count, sizeof(char *));
    output_tensors = calloc(output_count, sizeof(OrtValue *));
    if (!output_names || !output_tensors) {
        fprintf(stderr, "SMOKE FAIL: out of memory\n");
        goto cleanup;
    }
    for (size_t i = 0; i < output_count; i++) {
        ORT_CHECK(api->SessionGetOutputName(session, i, allocator, &output_names[i]),
                  "SessionGetOutputName");
    }

    ORT_CHECK(api->Run(session, NULL,
                       (const char *const *)input_names, input_tensors, input_count,
                       (const char *const *)output_names, output_count, output_tensors),
              "Run");
    fprintf(stdout, "SMOKE OK: inference succeeded (%zu input(s), %zu output(s))\n",
            input_count, output_count);
    exit_code = 0;

cleanup:
    if (output_tensors) {
        for (size_t i = 0; i < output_count; i++)
            if (output_tensors[i]) api->ReleaseValue(output_tensors[i]);
        free(output_tensors);
    }
    if (output_names) {
        for (size_t i = 0; i < output_count; i++)
            if (output_names[i]) api->AllocatorFree(allocator, output_names[i]);
        free(output_names);
    }
    if (input_tensors) {
        for (size_t i = 0; i < input_count; i++)
            if (input_tensors[i]) api->ReleaseValue(input_tensors[i]);
        free(input_tensors);
    }
    if (input_bufs) {
        for (size_t i = 0; i < input_count; i++) free(input_bufs[i]);
        free(input_bufs);
    }
    if (input_names) {
        for (size_t i = 0; i < input_count; i++)
            if (input_names[i]) api->AllocatorFree(allocator, input_names[i]);
        free(input_names);
    }
    return exit_code;
}

int main(int argc, char *argv[]) {
    if (argc != 2 && argc != 4) {
        fprintf(stderr, "Usage: %s <model.ort> [vectors.tvbin cosine_threshold]\n", argv[0]);
        return 1;
    }
    const char *model_path = argv[1];
    int exit_code = 1;

    const OrtApiBase *base = OrtGetApiBase();
    if (!base) {
        fprintf(stderr, "SMOKE FAIL: OrtGetApiBase() returned NULL\n");
        return 1;
    }
    const OrtApi *api = base->GetApi(ORT_API_VERSION);
    if (!api) {
        fprintf(stderr, "SMOKE FAIL: could not get ORT API\n");
        return 1;
    }

    OrtEnv *env = NULL;
    OrtSessionOptions *opts = NULL;
    OrtSession *session = NULL;
    OrtMemoryInfo *mem_info = NULL;
    OrtAllocator *allocator = NULL;

    ORT_CHECK(api->CreateEnv(ORT_LOGGING_LEVEL_WARNING, "smoke_test", &env), "CreateEnv");
    ORT_CHECK(api->CreateSessionOptions(&opts), "CreateSessionOptions");
    ORT_CHECK(api->SetSessionGraphOptimizationLevel(opts, ORT_DISABLE_ALL),
              "SetSessionGraphOptimizationLevel");
    ORT_CHECK(api->CreateSession(env, model_path, opts, &session), "CreateSession");
    fprintf(stdout, "SMOKE OK: loaded %s\n", model_path);
    fflush(stdout);

    ORT_CHECK(api->CreateCpuMemoryInfo(OrtArenaAllocator, OrtMemTypeDefault, &mem_info),
              "CreateCpuMemoryInfo");
    ORT_CHECK(api->GetAllocatorWithDefaultOptions(&allocator),
              "GetAllocatorWithDefaultOptions");

    if (argc == 4) {
        exit_code = run_comparison(api, session, mem_info, allocator, argv[2], atof(argv[3]));
    } else {
        exit_code = run_zerofill(api, session, mem_info, allocator);
    }

cleanup:
    if (mem_info) api->ReleaseMemoryInfo(mem_info);
    if (session)  api->ReleaseSession(session);
    if (opts)     api->ReleaseSessionOptions(opts);
    if (env)      api->ReleaseEnv(env);
    return exit_code;
}
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_ci_contracts.py -q`
Expected: PASS (new tests plus existing `test_smoke_test_*`).

- [ ] **Step 5: Commit**

```bash
git add scripts/smoke_test.c tests/test_ci_contracts.py
git commit -m "feat: add cosine-similarity comparison mode to smoke test"
```

---

### Task 6: Validate the optional `correctness` metadata block

**Files:**
- Modify: `scripts/validate_manifest.py` (inside the `metadata is not None` block, ~after line 183)
- Test: `tests/test_validate_manifest.py`

**Interfaces:**
- Produces: `validate()` rejects a malformed `metadata.correctness` block; accepts a well-formed one or its absence.

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_validate_manifest.py` (self-contained; uses `yaml` + the module's `validate`):

```python
import importlib.util
from pathlib import Path

import pytest
import yaml

_VM = Path(__file__).parent.parent / "scripts" / "validate_manifest.py"


def _vm():
    spec = importlib.util.spec_from_file_location("validate_manifest", _VM)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_MANIFEST_WITH_CORRECTNESS = """
onnxruntime:
  version: "1.27.0"
build:
  container_image: x
  target_os: linux
  target_arch: arm64
  cpu_tuning: neoverse-n1
  execution_provider: cpu
  minimal_build: extended
targets:
  - id: t
    quant: q4f16
    metadata:
      correctness:
        cosine_threshold: 0.99
        num_samples: 3
        max_tokens: 128
    model:
      repo_id: r
      revision: main
      primary: onnx/model.onnx
      companions: []
"""


def test_valid_correctness_block_passes():
    _vm().validate(yaml.safe_load(_MANIFEST_WITH_CORRECTNESS))


def test_correctness_threshold_must_be_number():
    data = yaml.safe_load(_MANIFEST_WITH_CORRECTNESS)
    data["targets"][0]["metadata"]["correctness"]["cosine_threshold"] = "high"
    with pytest.raises(SystemExit):
        _vm().validate(data)


def test_correctness_num_samples_must_be_positive():
    data = yaml.safe_load(_MANIFEST_WITH_CORRECTNESS)
    data["targets"][0]["metadata"]["correctness"]["num_samples"] = 0
    with pytest.raises(SystemExit):
        _vm().validate(data)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_validate_manifest.py -k correctness -q`
Expected: FAIL (bad blocks currently pass, so the two `raises(SystemExit)` tests fail).

- [ ] **Step 3: Add validation**

In `scripts/validate_manifest.py`, inside `validate()`, at the END of the `if metadata is not None:` block (after the `_INT_META_KEYS` loop):

```python
            correctness = metadata.get("correctness")
            if correctness is not None:
                if not isinstance(correctness, dict):
                    _fail(f"{ctx}.metadata: 'correctness' must be a mapping")
                ct = correctness.get("cosine_threshold")
                if ct is not None:
                    if isinstance(ct, bool) or not isinstance(ct, (int, float)):
                        _fail(
                            f"{ctx}.metadata.correctness: 'cosine_threshold' must be a number"
                        )
                    if not (0.0 < ct <= 1.0):
                        _fail(
                            f"{ctx}.metadata.correctness: 'cosine_threshold' must be in (0, 1]"
                        )
                for key in ("num_samples", "max_tokens"):
                    v = correctness.get(key)
                    if v is not None:
                        if isinstance(v, bool) or not isinstance(v, int) or v <= 0:
                            _fail(
                                f"{ctx}.metadata.correctness: '{key}' must be a positive integer"
                            )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_validate_manifest.py -q`
Expected: PASS (all).

- [ ] **Step 5: Commit**

```bash
git add scripts/validate_manifest.py tests/test_validate_manifest.py
git commit -m "feat: validate optional correctness metadata block"
```

---

### Task 7: Wire the harness into `build_target.sh`

**Files:**
- Modify: `scripts/build_target.sh` (config defaults ~line 22; step 7 ~line 111-129; step 12 ~line 283-302)
- Test: `tests/test_ci_contracts.py`

**Interfaces:**
- Consumes: `optimize_model.py --reference-output` (Task 2), `gen_reference_vectors.py` (Task 4), the two-mode `smoke_test` (Task 5), the `/fixtures` mount (Task 8).
- Produces: build runs the correctness comparison when `tokenizer.json` is present; otherwise falls back to zero-fill.

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_ci_contracts.py`:

```python
def test_build_script_generates_reference_vectors() -> None:
    text = BUILD_SCRIPT.read_text(encoding="utf-8")
    assert "gen_reference_vectors.py" in text
    assert "--reference-output" in text


def test_build_script_passes_correctness_defaults() -> None:
    text = BUILD_SCRIPT.read_text(encoding="utf-8")
    assert ".correctness.cosine_threshold // 0.99" in text
    assert ".correctness.num_samples // 3" in text
    assert ".correctness.max_tokens // 128" in text


def test_build_script_links_libm_for_smoke_test() -> None:
    text = BUILD_SCRIPT.read_text(encoding="utf-8")
    assert "-lm" in text
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_ci_contracts.py -k "reference_vectors or correctness_defaults or libm" -q`
Expected: FAIL.

- [ ] **Step 3: Edit `scripts/build_target.sh`**

(a) After `OUTPUT_DIR="${OUTPUT_DIR:-/output}"` (~line 22), add:

```bash
FIXTURE_DIR="${FIXTURE_DIR:-/fixtures}"
```

(b) In step 7, define the reference path and correctness params. After the `MODEL_TYPE="$(...)"` line (~line 111), add:

```bash
REFERENCE_ONNX="${WORK_DIR}/reference.onnx"
COSINE_THRESHOLD="$(echo "${MODEL_METADATA}" | jq -r '.correctness.cosine_threshold // 0.99')"
NUM_SAMPLES="$(echo "${MODEL_METADATA}" | jq -r '.correctness.num_samples // 3')"
MAX_TOKENS="$(echo "${MODEL_METADATA}" | jq -r '.correctness.max_tokens // 128')"
```

Then add `--reference-output "${REFERENCE_ONNX}"` to the existing `optimize_model.py` invocation (extend the command, keeping `"${OPTIMIZE_EXTRA_ARGS[@]}"` last):

```bash
python3 "$(dirname "$0")/optimize_model.py" \
    --input "${MODEL_DIR}/${HF_PRIMARY}" \
    --output "${OPTIMIZED_ONNX}" \
    --model_type "${MODEL_TYPE}" \
    --target_platform "${ORT_TARGET_PLATFORM}" \
    --work_dir "${OPT_WORK_DIR}" \
    --reference-output "${REFERENCE_ONNX}" \
    "${OPTIMIZE_EXTRA_ARGS[@]}"
```

(c) In step 12, add `-lm` to the smoke-test compile command:

```bash
clang -o "${SMOKE_BIN}" "${SMOKE_SRC}" \
    -I "${ORT_SRC}/include/onnxruntime/core/session" \
    -L "$(dirname "${BUILT_SO}")" \
    -lonnxruntime \
    -lm \
    -Wl,-rpath,"$(dirname "${BUILT_SO}")"
```

Then replace the smoke-test invocation block (the `set +e` … `set -e` region, ~lines 294-297) with:

```bash
TOKENIZER_JSON="${MODEL_DIR}/tokenizer.json"
SMOKE_ARGS=("${ORT_MODEL_PATH}")
if [ -f "${TOKENIZER_JSON}" ]; then
    echo "==> Generating correctness reference vectors"
    TV_FILE="${WORK_DIR}/reference_vectors.tvbin"
    python3 "$(dirname "$0")/gen_reference_vectors.py" \
        --model "${REFERENCE_ONNX}" \
        --tokenizer "${TOKENIZER_JSON}" \
        --fixture "${FIXTURE_DIR}/jane-austen_pride-and-prejudice.jsonl" \
        --output "${TV_FILE}" \
        --num-samples "${NUM_SAMPLES}" \
        --max-tokens "${MAX_TOKENS}"
    SMOKE_ARGS=("${ORT_MODEL_PATH}" "${TV_FILE}" "${COSINE_THRESHOLD}")
else
    echo "    tokenizer.json not present; running zero-fill smoke only"
fi

set +e
"${SMOKE_BIN}" "${SMOKE_ARGS[@]}" 2>&1 | tee "${SMOKE_LOG}"
SMOKE_EXIT=${PIPESTATUS[0]}
set -e
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_ci_contracts.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add scripts/build_target.sh tests/test_ci_contracts.py
git commit -m "feat: run embedding-correctness comparison in build"
```

---

### Task 8: Mount the fixture and wire CI

**Files:**
- Modify: `.github/workflows/build.yml` (install step line 23; pytest line 29; docker run lines 67-87)
- Test: `tests/test_ci_contracts.py` (add mount/deps tests; UPDATE the existing pytest-line test)

**Interfaces:**
- Consumes: `tests/test_gen_reference_vectors.py` (Task 3/4), `FIXTURE_DIR` (Task 7).
- Produces: CI mounts `tests/data` into the container and runs the new test file with numpy available.

- [ ] **Step 1: Write/adjust the failing tests**

In `tests/test_ci_contracts.py`, UPDATE the existing `test_workflow_runs_targeted_pytest_before_matrix_emit` assertion string to include the new test file:

```python
    assert (
        "pytest tests/test_validate_manifest.py tests/test_optimize_model.py "
        "tests/test_ci_contracts.py tests/test_gen_reference_vectors.py -q"
        in text
    )
```

Then ADD:

```python
def test_workflow_mounts_fixtures_dir() -> None:
    """The build container must mount the fixtures dir and expose FIXTURE_DIR."""
    text = WORKFLOW.read_text(encoding="utf-8")
    assert "tests/data:/fixtures:ro" in text
    assert "FIXTURE_DIR=/fixtures" in text


def test_workflow_installs_numpy_for_tests() -> None:
    """The plan job must install numpy so reference-vector tests can import it."""
    text = WORKFLOW.read_text(encoding="utf-8")
    assert "pip install" in text
    line = next(l for l in text.splitlines() if "pip install" in l)
    assert "numpy" in line
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_ci_contracts.py -k "workflow" -q`
Expected: FAIL (mount/numpy/pytest-line not present yet).

- [ ] **Step 3: Edit `.github/workflows/build.yml`**

Install line (23):

```yaml
        run: pip install pyyaml pytest numpy
```

Pytest line (29):

```yaml
        run: pytest tests/test_validate_manifest.py tests/test_optimize_model.py tests/test_ci_contracts.py tests/test_gen_reference_vectors.py -q
```

In the `docker run` block, add the fixtures mount alongside the other `-v` lines:

```yaml
            -v "$(pwd)/tests/data:/fixtures:ro" \
```

and add the env var alongside the other `-e` lines:

```yaml
            -e FIXTURE_DIR=/fixtures \
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_ci_contracts.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add .github/workflows/build.yml tests/test_ci_contracts.py
git commit -m "ci: mount fixtures and run reference-vector tests"
```

---

## Final verification (after all tasks)

- [ ] **Full test suite:**

Run: `pytest tests/test_validate_manifest.py tests/test_optimize_model.py tests/test_ci_contracts.py tests/test_gen_reference_vectors.py -q`
Expected: all PASS.

- [ ] **End-to-end build (heavier; the real proof):** build the Docker image and run the existing jina target locally, e.g.:

```bash
docker build -t ort-build:local -f docker/lambda-build.Dockerfile docker/
docker run --rm \
  -v "$(pwd)/scripts:/scripts:ro" \
  -v "$(pwd)/builds:/manifest:ro" \
  -v "$(pwd)/tests/data:/fixtures:ro" \
  -v "$(pwd)/output:/output" \
  -e TARGET_ID=jinaai/jina-embeddings-v5-text-nano-retrieval \
  -e ORT_VERSION=v1.27.0 -e QUANT=q4f16 \
  -e HF_REPO_ID=jinaai/jina-embeddings-v5-text-nano-retrieval \
  -e HF_REVISION=main -e HF_PRIMARY=onnx/model_q4f16.onnx \
  -e HF_COMPANIONS="onnx/model_q4f16.onnx_data tokenizer.json" \
  -e BUNDLE_EXTRAS="tokenizer.json build-info.json" \
  -e MODEL_METADATA='{"model_type":"bert"}' \
  -e CPU_TUNING=neoverse-n1 -e EXECUTION_PROVIDER=cpu -e MINIMAL_BUILD=extended \
  -e FIXTURE_DIR=/fixtures -e OUTPUT_DIR=/output -e PYTHONUNBUFFERED=1 \
  ort-build:local /scripts/build_target.sh
```

Expected: smoke log prints `sample N: cosine=…` lines and `SMOKE OK: all N sample(s) within cosine threshold`. For jina (pre-quantized, reference == shipped graph) the similarities should be ≈ 1.0.

- [ ] **Negative check:** temporarily lower one reference value (or raise the threshold to `1.0`) and re-run; confirm the smoke test exits non-zero — proving the gate fails on divergence. Revert afterward.

---

## Self-Review Notes (author)

- **Spec coverage:** reference emit → Task 2; gen_reference_vectors (tokenize/run/write) → Tasks 3-4; C cosine gate → Task 5; metadata config → Task 6; deps + build wiring → Tasks 1, 7; fixture mount + CI → Task 8. EuroBERT manifest target intentionally deferred (out of scope in spec).
- **Type consistency:** `build_feeds`, `write_vectors`/`read_vectors`, `truncate_ids`, `read_texts`, `main(argv)` names match across tasks; the `.tvbin` layout in `write_vectors` (Task 3) matches the C reader in `run_comparison` (Task 5) field-for-field (magic, u32 counts, i64 dims, u64 byte lengths, f32 reference).
- **Dtype codes** `0/1/2/3` are identical in Python (`_DTYPE_TO_CODE`) and C (`code_to_ort`).
