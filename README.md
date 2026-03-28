# minimal-ort-builder

A CI/CD pipeline that compiles a minimal, model-specific build of [ONNX Runtime](https://github.com/microsoft/onnxruntime) (ORT) targeting AWS Lambda on ARM64. Given a manifest describing one or more Hugging Face ONNX models, it downloads each model, generates a reduced operator configuration, compiles ORT with only the operators that model requires, runs a smoke test, and publishes a versioned tarball per target to a GitHub Release.

---

## Table of Contents

1. [Overview](#overview)
2. [Manifest authoring (`builds/release.yaml`)](#manifest-authoring)
3. [primary and companions explained](#primary-and-companions)
4. [Validation rules](#validation-rules)
5. [Local Docker build](#local-docker-build)
6. [Tarball contents](#tarball-contents)
7. [GitHub Actions workflow](#github-actions-workflow)
8. [Scope limitations (v1)](#scope-limitations-v1)

---

## Overview

`minimal-ort-builder` produces a trimmed `libonnxruntime.so` that contains only the operators needed by a specific ONNX model. Smaller binaries are important on Lambda where cold-start time and package size matter. The build runs inside the `public.ecr.aws/lambda/provided:al2023` container to match the Lambda execution environment exactly.

---

## Manifest authoring

All build configuration lives in a single file: `builds/release.yaml`. Add or modify targets there; the CI pipeline reads nothing else.

```yaml
onnxruntime:
  version: "1.20.1"   # ORT release tag; applies to every target

build:
  container_image: "public.ecr.aws/lambda/provided:al2023"  # base image for compilation
  target_os: linux
  target_arch: arm64
  cpu_tuning: neoverse-n1   # passed as -mcpu=<value> to gcc/g++
  execution_provider: cpu   # only "cpu" is supported in v1
  minimal_build: extended   # value passed to ORT's --minimal_build flag
  bundle_extras:            # optional: extra files to include in the tarball (plain filenames only)
    - build-info.json

targets:
  - id: phi3-mini-q4f16     # unique slug; used in artifact names
    quant: q4f16            # quantisation identifier; included in the tarball filename
    model:
      repo_id: microsoft/Phi-3-mini-4k-instruct-onnx   # Hugging Face repo in "owner/repo" format
      revision: main                                    # branch, tag, or full commit SHA
      primary: onnx/model_q4f16.onnx                   # path to the .onnx file inside the repo
      companions:
        - onnx/model_q4f16.onnx_data                   # external data files required alongside the model
        - tokenizer.json
```

To add a second target, append another entry under `targets` with a distinct `id`.

---

## primary and companions

Some ONNX models store large weight tensors in separate external data files rather than embedding them inside the `.onnx` file itself. When that is the case, the runtime requires both files to be present in the same directory at inference time.

- **`primary`** — the `.onnx` file. This is the file passed to ORT's operator-extraction tool and to the smoke test.
- **`companions`** — a list of all external data files that must be present alongside `primary`. The build script downloads each companion and stages it at the root of the tarball by basename.

For single-file models that embed all weights internally, set `companions` to an empty list:

```yaml
companions: []
```

The validator rejects manifests where `primary` also appears in `companions`, and rejects any path that is absolute or contains `..` traversal segments.

---

## Validation rules

The following rules are enforced by `scripts/validate_manifest.py`:

| # | Rule |
|---|------|
| 1 | Top-level keys `onnxruntime`, `build`, and `targets` are all required |
| 2 | `onnxruntime` must contain `version` |
| 3 | `build` must contain `container_image`, `target_os`, `target_arch`, `cpu_tuning`, `execution_provider`, and `minimal_build` |
| 4 | `build.bundle_extras`, if present, must be a list of plain filenames (no path separators, no leading dot) |
| 5 | `targets` must be a non-empty list |
| 6 | Each target must have `id`, `quant`, and `model` |
| 6b | `quant` must match `^[a-z0-9][a-z0-9\-]*$` (e.g. `fp32`, `q4f16`, `int8`) |
| 7 | Target `id` values must be unique across all targets |
| 8 | Targets must not contain a per-target `onnxruntime` key (the global one is used) |
| 9 | Each `model` must have `repo_id`, `revision`, `primary`, and `companions` |
| 9b | `companions` must be a list (not a string or other scalar) |
| 10 | `primary` must not also appear in `companions` |
| 11 | All model paths (`primary` and every companion) must be relative — no leading `/` and no `..` segments |

Run the validator locally before pushing:

```bash
pip install pyyaml
python scripts/validate_manifest.py builds/release.yaml
```

On success it prints `OK: manifest is valid` and exits 0. On failure it prints `ERROR: <message>` to stderr and exits non-zero.

---

## Local Docker build

**Prerequisites:** Docker with ARM64 emulation or a native ARM64 host (e.g. Apple Silicon Mac with Docker Desktop, or an AWS Graviton instance).

### Step 1 — build the image

```bash
docker build --platform linux/arm64 \
  -t ort-lambda-build \
  -f docker/lambda-build.Dockerfile .
```

This installs the compiler toolchain, `cmake`, `ninja`, `jq`, and the Python packages (`huggingface_hub`, `onnx`, etc.) needed by the build script.

### Step 2 — run a single target

Replace the env-var values below with the fields from your manifest entry:

```bash
mkdir -p output
docker run --rm \
  --platform linux/arm64 \
  -v "$(pwd)/scripts:/scripts:ro" \
  -v "$(pwd)/builds:/manifest:ro" \
  -v "$(pwd)/output:/output" \
  -e TARGET_ID=phi3-mini-q4f16 \
  -e ORT_VERSION=1.20.1 \
  -e QUANT=q4f16 \
  -e HF_REPO_ID=microsoft/Phi-3-mini-4k-instruct-onnx \
  -e HF_REVISION=main \
  -e HF_PRIMARY=onnx/model_q4f16.onnx \
  -e HF_COMPANIONS="onnx/model_q4f16.onnx_data tokenizer.json" \
  -e BUNDLE_EXTRAS="build-info.json" \
  -e CPU_TUNING=neoverse-n1 \
  -e EXECUTION_PROVIDER=cpu \
  -e MINIMAL_BUILD=extended \
  -e OUTPUT_DIR=/output \
  ort-lambda-build /scripts/build_target.sh
```

The output tarball is written to `output/<TARGET_ID_SAFE>_<QUANT>_linux-arm64.tar.gz` (slashes in `TARGET_ID` are replaced with `__`).

**Multiple companion files:** space-separate them in the `HF_COMPANIONS` value:

```bash
-e HF_COMPANIONS="onnx/weights.onnx_data onnx/extra.bin"
```

**Gated / private Hugging Face models:** export your token before running, or add it as an env var:

```bash
export HF_TOKEN=hf_xxx
docker run ... -e HF_TOKEN ...
```

The build script reads `HF_TOKEN` automatically from the environment when calling `huggingface-cli download`.

---

## Tarball contents

Each target produces one `.tar.gz` named `<target_id_safe>_<quant>_linux-arm64.tar.gz`. The bundle always contains:

| File | Description |
|---|---|
| `libonnxruntime.so` | Minimal ORT shared library compiled for this target's model |
| `model.ort` | The model converted to ORT format (required by a minimal build) |
| `<companion basenames>` | External data files and any other companions listed in the manifest (e.g. `tokenizer.json`) |
| `SHA256SUMS` | SHA-256 checksums of all bundle files |

Additional files can be included via `build.bundle_extras` in the manifest. The default configuration adds:

| File | Description |
|---|---|
| `build-info.json` | ORT version, model source (repo, revision, primary path), ORT git SHA, and build settings |

Files generated during the build but not in the whitelist (e.g. `operators.config`, `smoke-test.log`, `manifest.snapshot.yaml`) are pruned before the tarball is created.

---

## GitHub Actions workflow

The workflow in `.github/workflows/build.yml` has three jobs: `plan`, `build`, and `publish`.

### Tag-triggered release

Push a version tag to start a full release:

```bash
git tag v1.0.0
git push origin v1.0.0
```

1. **`plan`** — validates `builds/release.yaml` and emits a build matrix (one entry per target).
2. **`build`** — one job per target, running in parallel on `ubuntu-24.04-arm` (native ARM64) inside the AL2023 Lambda container. Each job uploads its tarball as a GitHub Actions artifact.
3. **`publish`** — downloads all artifacts and attaches them to the GitHub Release created for the tag. This job only runs on tag pushes, never on `workflow_dispatch`.

### Manual rebuild (`workflow_dispatch`)

Trigger a run from the GitHub Actions UI or via `gh workflow run`. All targets are built exactly as above (`plan` + `build`), but the `publish` job is skipped. Tarballs are available as Actions artifacts for the duration of the retention period and never create or update a Release.

---

## Scope limitations (v1)

The following are intentional constraints in the current version:

- **Linux ARM64 only** — no x86-64, no Windows, no macOS.
- **Hugging Face model sources only** — no S3, no local file paths, no direct HTTP URLs.
- **CPU execution provider only** — no CUDA, CoreML, DirectML, or other EPs.
- **No Lambda layer zip packaging** — the output is a plain `.tar.gz`; wrapping it into a Lambda layer zip is left to the consumer.
- **One global ORT version per manifest** — all targets in a manifest are compiled against the same `onnxruntime.version`; per-target version overrides are not supported.
