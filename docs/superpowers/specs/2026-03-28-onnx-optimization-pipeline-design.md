# ONNX Optimization Pipeline Design

**Date:** 2026-03-28
**Status:** Approved

## Overview

Replace the current approach of downloading a pre-quantized HuggingFace model and passing it directly to `convert_onnx_models_to_ort.py` with a custom optimization pipeline that starts from the unquantized `onnx/model.onnx` and produces a fully optimized, int8-quantized ONNX model before ORT conversion.

## Pipeline

```
onnx/model.onnx  (fp32/fp16, dynamic shapes, from HuggingFace)
       │
       ▼ scripts/optimize_model.py
       │
       ├─ Step 1: Transformer optimization
       │          onnxruntime.transformers.optimizer.optimize_model()
       │          model_type from manifest metadata.model_type
       │          → work/01_transformer_opt.onnx
       │          [fallback: skip on failure, log warning, continue from input]
       │
       ├─ Step 2a: Static shape specialization
       │          onnxruntime.tools.onnx_model_utils
       │          batch=1, seq_len=metadata.max_length
       │          → work/02_fixed_shape.onnx
       │          [hard fail if max_length missing from metadata]
       │
       ├─ Step 2b: Symbolic shape inference
       │          onnxruntime.tools.symbolic_shape_infer.SymbolicShapeInference.infer_shapes()
       │          → work/03_shape_inferred.onnx
       │          [fallback: skip on failure, log warning]
       │
       ├─ Step 3: ORT graph optimization (offline)
       │          SessionOptions.graph_optimization_level = ORT_ENABLE_ALL
       │          SessionOptions.optimized_model_filepath = ...
       │          → work/04_ort_optimized.onnx
       │          [hard fail]
       │
       └─ Step 4: Dynamic int8 quantization
                  onnxruntime.quantization.quantize_dynamic(weight_type=QuantType.QInt8)
                  → <output>
                  [hard fail]
       │
       ▼ convert_onnx_models_to_ort.py  (existing step, unchanged)
       │
       ▼
  model.ort
```

## Components

### `scripts/optimize_model.py`

New script. Runs inside the existing Docker build container. CLI interface:

```
python3 optimize_model.py \
  --input  path/to/model.onnx \
  --output path/to/optimized_int8.onnx \
  --model_type bert \
  --max_length 8192 \
  --target_platform arm \
  --work_dir path/to/work/
```

Each pipeline step saves an intermediate file to `--work_dir` for debugging. Steps are numbered (`01_`, `02_`, etc.) so failure point is immediately visible.

### `scripts/build_target.sh`

New step 10 calls `optimize_model.py` on the downloaded `onnx/model.onnx`. The existing `convert_onnx_models_to_ort.py` call (now step 11) is updated to point at the script's output directory instead of the raw model directory.

`model_type` and `max_length` are extracted from `$MODEL_METADATA` via `jq`.

### `builds/release.yaml`

Two changes per target:
1. `metadata.model_type` field added (used by optimizer)
2. `model.primary` changed from pre-quantized variant to `onnx/model.onnx`
3. `model.companions` updated accordingly (drop `onnx_data` if not needed for base model)

## Manifest Contract

```yaml
metadata:
  model_type: bert       # new required field; passed to onnxruntime.transformers.optimizer
  max_length: 8192       # existing field; used for shape specialization (seq_len)
```

`model_type` must be one of the values accepted by `onnxruntime.transformers.optimizer.optimize_model()`. For non-standard BERT variants (e.g. EuroBERT), use `bert` as the closest approximation; the optimizer will fall back gracefully if the graph structure doesn't match.

## Docker Dependencies

No new packages required. `onnx` and `onnxruntime==1.24.4` are already installed in `lambda-build.Dockerfile`. All needed modules (`onnxruntime.transformers`, `onnxruntime.tools`, `onnxruntime.quantization`) are part of the `onnxruntime` package.

## Error Handling

| Step | On failure |
|------|-----------|
| Transformer optimization | Warn + skip; continue from previous step's output |
| Shape specialization | Hard fail (missing `max_length` = misconfigured manifest) |
| Symbolic shape inference | Warn + skip |
| ORT graph optimization | Hard fail (broken model) |
| Int8 quantization | Hard fail (output required) |

## Target: jinaai/jina-embeddings-v5-text-nano-retrieval

- Architecture: EuroBERT (BERT-variant encoder-only); `model_type: bert` is the correct approximation
- `max_length: 8192`
- `primary`: change from `onnx/model_q4f16.onnx` → `onnx/model.onnx`
- Companions: `tokenizer.json` (drop `onnx/model_q4f16.onnx_data`)
