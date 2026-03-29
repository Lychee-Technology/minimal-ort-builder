# ONNX Optimization Pipeline Design

**Date:** 2026-03-28
**Status:** Approved

## Overview

Replace the current approach of downloading a pre-quantized HuggingFace model and passing it directly to `convert_onnx_models_to_ort.py` with a custom optimization pipeline that starts from the unquantized `onnx/model.onnx` and produces a fully optimized, int8-quantized ONNX model.

The optimized model feeds into both the existing operator config generation step (so quantization operators are included in the minimal ORT build) and the final ORT format conversion step.

## Pipeline

```
onnx/model.onnx  (fp32/fp16, dynamic shapes, from HuggingFace)
       │
       ▼ scripts/optimize_model.py   ← NEW: runs between download and op-config
       │
       ├─ Step 1: Transformer optimization
       │          onnxruntime.transformers.optimizer.optimize_model()
       │          model_type from manifest metadata.model_type
       │          → work/optimize/01_transformer_opt.onnx
       │          [fallback: skip on failure, log warning, continue from input]
       │
       ├─ Step 2a: Static shape specialization
       │          onnxruntime.tools.onnx_model_utils.make_dim_param_fixed()
       │          batch=1, seq_len=metadata.max_length
       │          dimension param names discovered by inspecting graph inputs
       │          → work/optimize/02_fixed_shape.onnx
       │          [hard fail if max_length missing from metadata]
       │
       ├─ Step 2b: Symbolic shape inference
       │          onnxruntime.tools.symbolic_shape_infer.SymbolicShapeInference.infer_shapes()
       │          → work/optimize/03_shape_inferred.onnx
       │          [fallback: skip on failure, log warning]
       │
       ├─ Step 3: ORT graph optimization (offline)
       │          SessionOptions.graph_optimization_level = ORT_ENABLE_ALL
       │          SessionOptions.optimized_model_filepath = ...
       │          → work/optimize/04_ort_optimized.onnx
       │          [hard fail]
       │
       └─ Step 4: Dynamic int8 quantization
                  onnxruntime.quantization.quantize_dynamic(weight_type=QuantType.QInt8)
                  → work/optimized.onnx   (final output; passed to downstream steps)
                  [hard fail]
       │
       ▼ create_reduced_build_config.py  (step 8, unchanged — now uses optimized model)
       │
       ▼ build ORT  (step 9, unchanged — operator config includes QLinear* ops)
       │
       ▼ convert_onnx_models_to_ort.py  (step 11, updated input directory)
       │
       ▼
  model.ort
```

## Step Numbering in `build_target.sh`

After this change the steps are:

| # | Description |
|---|-------------|
| 1 | Validate env vars |
| 2 | Create temp directories |
| 3 | Download primary model (`onnx/model.onnx`) |
| 4 | Download companion files |
| 5 | Verify downloaded files |
| 6 | Clone ORT source |
| **7** | **NEW: Optimize + quantize model (`optimize_model.py`)** |
| 8 | Generate reduced operator config (from optimized model) |
| 9 | Build ORT |
| 10 | Verify `libonnxruntime.so` |
| 11 | Convert optimized ONNX → ORT format |
| 12 | Smoke test |
| 13 | Stage artifacts |
| 14 | Prune oversized artifacts |
| 15 | Create tarball |

Previous steps 7–13 shift to 8–15.

## Components

### `scripts/optimize_model.py`

New script. Runs inside the existing Docker build container, between model download and operator config generation. CLI:

```
python3 optimize_model.py \
  --input    path/to/onnx/model.onnx \
  --output   path/to/work/optimized.onnx \
  --model_type bert \
  --max_length 8192 \
  --target_platform arm \
  --work_dir path/to/work/optimize/
```

`--work_dir` is a subdirectory of `$WORK_DIR` (e.g. `$WORK_DIR/optimize/`). It is ephemeral — cleaned up automatically by the existing `trap 'rm -rf "${WORK_DIR}"' EXIT` in `build_target.sh`. No special cleanup needed.

Each pipeline step saves an intermediate file under `--work_dir` (numbered `01_` through `04_`) for debugging. The final output is written to `--output`.

**Shape specialization detail:** `make_dim_param_fixed()` requires knowing the symbolic dimension parameter names in the graph (e.g. `"batch_size"`, `"sequence_length"`). The script will inspect the model's graph inputs at runtime to find dimension params that match known patterns (`batch`, `sequence`, `seq_len`, etc.) rather than hardcoding names. This handles non-standard BERT variants like EuroBERT.

### `scripts/build_target.sh`

New step 7 calls `optimize_model.py` on the downloaded `onnx/model.onnx`, outputting `$WORK_DIR/optimized.onnx`.

Step 8 (`create_reduced_build_config.py`) is updated to point at `$WORK_DIR/optimized.onnx` instead of `$MODEL_DIR`, so the operator config includes quantization operators.

Step 11 (`convert_onnx_models_to_ort.py`) is updated: the directory argument changes from `${MODEL_DIR}/$(dirname "${HF_PRIMARY}")` to a new directory containing only `optimized.onnx` (e.g. `$WORK_DIR/ort_input/`). The converter scans a directory and converts all `.onnx` files in it.

`model_type` and `max_length` are extracted from `$MODEL_METADATA` via `jq`.

### `builds/release.yaml`

Changes per target:

1. `metadata.model_type` added (required; passed to transformer optimizer)
2. `metadata.model_format` updated if needed
3. `quant` field updated to `int8` (was `q4f16`; artifact tarball name reflects actual quantization)
4. `model.primary` changed to `onnx/model.onnx` (was `onnx/model_q4f16.onnx`)
5. `model.companions` updated: drop `onnx/model_q4f16.onnx_data`, keep `tokenizer.json`

## Manifest Contract

```yaml
targets:
  - id: jinaai/jina-embeddings-v5-text-nano-retrieval
    quant: int8                     # ← was q4f16; pipeline now produces int8
    metadata:
      model_type: bert              # ← new required field
      max_length: 8192              # existing; used for shape specialization
      ...
    model:
      primary: onnx/model.onnx      # ← was onnx/model_q4f16.onnx
      companions:
        - tokenizer.json            # ← drop onnx/model_q4f16.onnx_data
```

`model_type` must be a value accepted by `onnxruntime.transformers.optimizer.optimize_model()`. For non-standard BERT variants (e.g. EuroBERT), use `bert`; the optimizer falls back gracefully if the graph structure doesn't match.

## Docker Dependencies

No new packages required. `onnx` and `onnxruntime==1.24.4` are already installed. All needed modules (`onnxruntime.transformers`, `onnxruntime.tools`, `onnxruntime.quantization`) are submodules of the `onnxruntime` package.

## Error Handling

| Step | On failure |
|------|-----------|
| Transformer optimization | Warn + skip; continue from previous step's output |
| Shape specialization | Hard fail (missing `max_length` = misconfigured manifest) |
| Symbolic shape inference | Warn + skip |
| ORT graph optimization | Hard fail (broken model) |
| Int8 quantization | Hard fail (output required by all downstream steps) |
