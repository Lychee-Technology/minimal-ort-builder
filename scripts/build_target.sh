#!/usr/bin/env bash
# build_target.sh — ORT minimal build entrypoint, runs inside the lambda/provided:al2023 container.
# All configuration is supplied via environment variables.

set -euo pipefail

# ---------------------------------------------------------------------------
# 1. Validate required env vars; set defaults for optional ones
# ---------------------------------------------------------------------------
: "${TARGET_ID:?TARGET_ID is required}"
: "${ORT_VERSION:?ORT_VERSION is required}"
: "${HF_REPO_ID:?HF_REPO_ID is required}"
: "${HF_REVISION:?HF_REVISION is required}"
: "${HF_PRIMARY:?HF_PRIMARY is required}"
: "${HF_COMPANIONS:=""}"         # optional, may be empty
CPU_TUNING="${CPU_TUNING:-neoverse-n1}"
EXECUTION_PROVIDER="${EXECUTION_PROVIDER:-cpu}"
MINIMAL_BUILD="${MINIMAL_BUILD:-extended}"
OUTPUT_DIR="${OUTPUT_DIR:-/output}"

echo "==> Configuration"
echo "    TARGET_ID          = ${TARGET_ID}"
echo "    ORT_VERSION        = ${ORT_VERSION}"
echo "    HF_REPO_ID         = ${HF_REPO_ID}"
echo "    HF_REVISION        = ${HF_REVISION}"
echo "    HF_PRIMARY         = ${HF_PRIMARY}"
echo "    HF_COMPANIONS      = ${HF_COMPANIONS}"
echo "    CPU_TUNING         = ${CPU_TUNING}"
echo "    EXECUTION_PROVIDER = ${EXECUTION_PROVIDER}"
echo "    MINIMAL_BUILD      = ${MINIMAL_BUILD}"
echo "    OUTPUT_DIR         = ${OUTPUT_DIR}"

mkdir -p "${OUTPUT_DIR}"

# ---------------------------------------------------------------------------
# 2. Create temp directories
# ---------------------------------------------------------------------------
echo "==> Creating working directories"
WORK_DIR=$(mktemp -d)
trap 'rm -rf "${WORK_DIR}"' EXIT
MODEL_DIR="${WORK_DIR}/model"
ORT_SRC="${WORK_DIR}/onnxruntime"
BUILD_DIR="${WORK_DIR}/build"
STAGE_DIR="${WORK_DIR}/stage"
mkdir -p "${MODEL_DIR}" "${BUILD_DIR}" "${STAGE_DIR}"

# ---------------------------------------------------------------------------
# 3. Download primary model
# ---------------------------------------------------------------------------
echo "==> Downloading primary model: ${HF_PRIMARY}"
huggingface-cli download "${HF_REPO_ID}" "${HF_PRIMARY}" \
    --revision "${HF_REVISION}" \
    --local-dir "${MODEL_DIR}"

# ---------------------------------------------------------------------------
# 4. Download each companion file
# ---------------------------------------------------------------------------
if [ -n "${HF_COMPANIONS}" ]; then
    echo "==> Downloading companion files"
    for companion in ${HF_COMPANIONS}; do
        echo "    Downloading companion: ${companion}"
        huggingface-cli download "${HF_REPO_ID}" "${companion}" \
            --revision "${HF_REVISION}" \
            --local-dir "${MODEL_DIR}"
    done
fi

# ---------------------------------------------------------------------------
# 5. Verify all files exist
# ---------------------------------------------------------------------------
echo "==> Verifying downloaded files"
PRIMARY_PATH="${MODEL_DIR}/${HF_PRIMARY}"
if [ ! -f "${PRIMARY_PATH}" ]; then
    echo "ERROR: primary model not found at ${PRIMARY_PATH}" >&2
    exit 1
fi

if [ -n "${HF_COMPANIONS}" ]; then
    for companion in ${HF_COMPANIONS}; do
        COMPANION_PATH="${MODEL_DIR}/${companion}"
        if [ ! -f "${COMPANION_PATH}" ]; then
            echo "ERROR: companion file not found at ${COMPANION_PATH}" >&2
            exit 1
        fi
    done
fi

# ---------------------------------------------------------------------------
# 6. Clone ORT source
# ---------------------------------------------------------------------------
echo "==> Cloning ORT ${ORT_VERSION}"
git clone --depth 1 --branch "${ORT_VERSION}" \
    https://github.com/microsoft/onnxruntime.git "${ORT_SRC}"

# ---------------------------------------------------------------------------
# 7. Generate reduced operator config
# ---------------------------------------------------------------------------
echo "==> Generating reduced operator config"
OPERATOR_CONFIG="${WORK_DIR}/operators.config"
python3 "${ORT_SRC}/tools/python/create_reduced_build_config.py" \
    "${MODEL_DIR}" \
    "${OPERATOR_CONFIG}"

# ---------------------------------------------------------------------------
# 8. Build ORT
# ---------------------------------------------------------------------------
echo "==> Building ORT (this will take a while)"

CMAKE_EXTRA_DEFINES=(
    "onnxruntime_BUILD_SHARED_LIB=ON"
    "CMAKE_C_COMPILER=clang"
    "CMAKE_CXX_COMPILER=clang++"
    "CMAKE_EXE_LINKER_FLAGS=-fuse-ld=lld"
    "CMAKE_SHARED_LINKER_FLAGS=-fuse-ld=lld"
    "CMAKE_MODULE_LINKER_FLAGS=-fuse-ld=lld"
    "CMAKE_C_COMPILER_LAUNCHER=ccache"
    "CMAKE_CXX_COMPILER_LAUNCHER=ccache"
)

if [ "${CPU_TUNING}" = "neoverse-n1" ]; then
    CMAKE_EXTRA_DEFINES+=(
        # Use -mtune (not -mcpu) so Clang tunes for neoverse-n1 without
        # auto-enabling SVE. -mcpu=neoverse-n1 with Clang 20 defines
        # __ARM_FEATURE_SVE which conflicts with ORT's SVE kernel gating.
        "CMAKE_CXX_FLAGS=-mtune=neoverse-n1"
        "CMAKE_C_FLAGS=-mtune=neoverse-n1"
    )
fi

ORT_NO_SVE_FLAG=()
if [ "${CPU_TUNING}" = "neoverse-n1" ]; then
    # Prevent ORT's build.py from setting -Donnxruntime_USE_SVE=ON,
    # which would reference SVE kernel symbols that aren't compiled in a
    # minimal build without the full SVE source list.
    ORT_NO_SVE_FLAG=(--no_sve)
fi

python3 "${ORT_SRC}/tools/ci_build/build.py" \
    --build_dir "${BUILD_DIR}" \
    --config Release \
    --skip_tests \
    --allow_running_as_root \
    --minimal_build "${MINIMAL_BUILD}" \
    --disable_ml_ops \
    --disable_rtti \
    --enable_reduced_operator_type_support \
    --include_ops_by_config "${OPERATOR_CONFIG}" \
    --parallel \
    "${ORT_NO_SVE_FLAG[@]}" \
    --cmake_extra_defines "${CMAKE_EXTRA_DEFINES[@]}"

# ---------------------------------------------------------------------------
# 9. Verify libonnxruntime.so exists
# ---------------------------------------------------------------------------
echo "==> Verifying build output"
BUILT_SO="${BUILD_DIR}/Release/libonnxruntime.so"
if [ ! -f "${BUILT_SO}" ]; then
    echo "ERROR: libonnxruntime.so not found at ${BUILT_SO}" >&2
    exit 1
fi

# ---------------------------------------------------------------------------
# 10. Convert ONNX model to ORT format (minimal build requires ORT format)
# ---------------------------------------------------------------------------
echo "==> Converting ONNX model to ORT format"
ORT_MODEL_DIR="${WORK_DIR}/ort_model"
mkdir -p "${ORT_MODEL_DIR}"
python3 "${ORT_SRC}/tools/python/convert_onnx_models_to_ort.py" \
    "${MODEL_DIR}/$(dirname "${HF_PRIMARY}")" \
    --optimization_style Fixed \
    --output_dir "${ORT_MODEL_DIR}"

# The converter mirrors the directory structure; find the .ort file
ORT_MODEL_PATH=$(find "${ORT_MODEL_DIR}" -name "*.ort" | head -1)
if [ -z "${ORT_MODEL_PATH}" ]; then
    echo "ERROR: no .ort model found after conversion in ${ORT_MODEL_DIR}" >&2
    exit 1
fi
echo "    ORT model: ${ORT_MODEL_PATH}"

# ---------------------------------------------------------------------------
# 11. Compile and run smoke test
# ---------------------------------------------------------------------------
echo "==> Running smoke test"
SMOKE_SRC="$(dirname "$0")/smoke_test.c"
SMOKE_BIN="${WORK_DIR}/smoke_test"
SMOKE_LOG="${STAGE_DIR}/smoke-test.log"

clang -o "${SMOKE_BIN}" "${SMOKE_SRC}" \
    -I "${ORT_SRC}/include/onnxruntime/core/session" \
    -L "$(dirname "${BUILT_SO}")" \
    -lonnxruntime \
    -Wl,-rpath,"$(dirname "${BUILT_SO}")"

set +e
"${SMOKE_BIN}" "${ORT_MODEL_PATH}" 2>&1 | tee "${SMOKE_LOG}"
SMOKE_EXIT=${PIPESTATUS[0]}
set -e

if [ "${SMOKE_EXIT}" -ne 0 ]; then
    echo "ERROR: smoke test failed with exit code ${SMOKE_EXIT}" >&2
    exit 1
fi

# ---------------------------------------------------------------------------
# 12. Stage artifacts
# ---------------------------------------------------------------------------
echo "==> Staging artifacts"

cp "${BUILT_SO}" "${STAGE_DIR}/libonnxruntime.so"
cp "${OPERATOR_CONFIG}" "${STAGE_DIR}/operators.config"
cp "${ORT_MODEL_PATH}" "${STAGE_DIR}/model.ort"

jq -n \
  --arg target_id          "${TARGET_ID}" \
  --arg ort_version        "${ORT_VERSION}" \
  --arg ort_git_sha        "$(git -C "${ORT_SRC}" rev-parse HEAD)" \
  --arg hf_repo_id         "${HF_REPO_ID}" \
  --arg hf_revision        "${HF_REVISION}" \
  --arg hf_primary         "${HF_PRIMARY}" \
  --arg cpu_tuning         "${CPU_TUNING}" \
  --arg execution_provider "${EXECUTION_PROVIDER}" \
  --arg minimal_build      "${MINIMAL_BUILD}" \
  '{target_id:$target_id,ort_version:$ort_version,ort_git_sha:$ort_git_sha,hf_repo_id:$hf_repo_id,hf_revision:$hf_revision,hf_primary:$hf_primary,cpu_tuning:$cpu_tuning,execution_provider:$execution_provider,minimal_build:$minimal_build}' \
  > "${STAGE_DIR}/build-info.json"

if [ -f "/manifest/release.yaml" ]; then
    echo "    Copying manifest snapshot"
    cp "/manifest/release.yaml" "${STAGE_DIR}/manifest.snapshot.yaml"
fi

echo "    Computing SHA256SUMS"
(cd "${STAGE_DIR}" && sha256sum libonnxruntime.so operators.config model.ort build-info.json smoke-test.log > SHA256SUMS)

# ---------------------------------------------------------------------------
# 13. Create tarball
# ---------------------------------------------------------------------------
echo "==> Creating tarball"
TARBALL="${OUTPUT_DIR}/ort-${ORT_VERSION}-${TARGET_ID}-linux-arm64.tar.gz"
tar -czf "${TARBALL}" -C "${STAGE_DIR}" .

# ---------------------------------------------------------------------------
# 14. Success
# ---------------------------------------------------------------------------
echo "==> Build complete"
echo "    Tarball: ${TARBALL}"
