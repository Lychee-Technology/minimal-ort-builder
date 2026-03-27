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
MODEL_DIR="${WORK_DIR}/model"
ORT_SRC="${WORK_DIR}/onnxruntime"
BUILD_DIR="${WORK_DIR}/build"
STAGE_DIR="${WORK_DIR}/stage"
mkdir -p "${MODEL_DIR}" "${BUILD_DIR}" "${STAGE_DIR}"

# ---------------------------------------------------------------------------
# 3. Download primary model
# ---------------------------------------------------------------------------
echo "==> Downloading primary model: ${HF_PRIMARY}"
if [ -n "${HF_TOKEN:-}" ]; then
    huggingface-cli download "${HF_REPO_ID}" "${HF_PRIMARY}" \
        --revision "${HF_REVISION}" \
        --local-dir "${MODEL_DIR}" \
        --token "${HF_TOKEN}"
else
    huggingface-cli download "${HF_REPO_ID}" "${HF_PRIMARY}" \
        --revision "${HF_REVISION}" \
        --local-dir "${MODEL_DIR}"
fi

# ---------------------------------------------------------------------------
# 4. Download each companion file
# ---------------------------------------------------------------------------
if [ -n "${HF_COMPANIONS}" ]; then
    echo "==> Downloading companion files"
    for companion in ${HF_COMPANIONS}; do
        echo "    Downloading companion: ${companion}"
        if [ -n "${HF_TOKEN:-}" ]; then
            huggingface-cli download "${HF_REPO_ID}" "${companion}" \
                --revision "${HF_REVISION}" \
                --local-dir "${MODEL_DIR}" \
                --token "${HF_TOKEN}"
        else
            huggingface-cli download "${HF_REPO_ID}" "${companion}" \
                --revision "${HF_REVISION}" \
                --local-dir "${MODEL_DIR}"
        fi
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
echo "==> Cloning ORT v${ORT_VERSION}"
git clone --depth 1 --branch "v${ORT_VERSION}" \
    https://github.com/microsoft/onnxruntime.git "${ORT_SRC}"

# ---------------------------------------------------------------------------
# 7. Generate reduced operator config
# ---------------------------------------------------------------------------
echo "==> Generating reduced operator config"
OPERATOR_CONFIG="${WORK_DIR}/operators.config"
python3 "${ORT_SRC}/tools/python/gen_opkernel_def.py" \
    --models "${PRIMARY_PATH}" \
    --output "${OPERATOR_CONFIG}"

# ---------------------------------------------------------------------------
# 8. Build ORT
# ---------------------------------------------------------------------------
echo "==> Building ORT (this will take a while)"

CMAKE_EXTRA_DEFINES=(
    "onnxruntime_BUILD_SHARED_LIB=ON"
    "onnxruntime_DEX_NUM_THREADS=0"
)

if [ "${CPU_TUNING}" = "neoverse-n1" ]; then
    CMAKE_EXTRA_DEFINES+=(
        "-DCMAKE_CXX_FLAGS=-mcpu=neoverse-n1"
        "-DCMAKE_C_FLAGS=-mcpu=neoverse-n1"
    )
fi

python3 "${ORT_SRC}/tools/ci_build/build.py" \
    --build_dir "${BUILD_DIR}" \
    --config Release \
    --skip_tests \
    --minimal_build "${MINIMAL_BUILD}" \
    --disable_ml_ops \
    --disable_rtti \
    --enable_reduced_operator_type_support \
    --include_ops_by_config "${OPERATOR_CONFIG}" \
    --parallel \
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
# 10. Compile and run smoke test
# ---------------------------------------------------------------------------
echo "==> Running smoke test"
SMOKE_SRC="$(dirname "$0")/smoke_test.c"
SMOKE_BIN="${WORK_DIR}/smoke_test"
SMOKE_LOG="${STAGE_DIR}/smoke-test.log"

gcc -o "${SMOKE_BIN}" "${SMOKE_SRC}" \
    -I "${ORT_SRC}/include/onnxruntime/core/session" \
    -L "$(dirname "${BUILT_SO}")" \
    -lonnxruntime \
    -Wl,-rpath,"$(dirname "${BUILT_SO}")"

set +e
"${SMOKE_BIN}" "${PRIMARY_PATH}" 2>&1 | tee "${SMOKE_LOG}"
SMOKE_EXIT=${PIPESTATUS[0]}
set -e

if [ "${SMOKE_EXIT}" -ne 0 ]; then
    echo "ERROR: smoke test failed with exit code ${SMOKE_EXIT}" >&2
    exit 1
fi

# ---------------------------------------------------------------------------
# 11. Stage artifacts
# ---------------------------------------------------------------------------
echo "==> Staging artifacts"

cp "${BUILT_SO}" "${STAGE_DIR}/libonnxruntime.so"
cp "${OPERATOR_CONFIG}" "${STAGE_DIR}/operators.config"

cat > "${STAGE_DIR}/build-info.json" <<EOF
{
  "target_id": "${TARGET_ID}",
  "ort_version": "${ORT_VERSION}",
  "ort_git_sha": "$(git -C "${ORT_SRC}" rev-parse HEAD)",
  "hf_repo_id": "${HF_REPO_ID}",
  "hf_revision": "${HF_REVISION}",
  "hf_primary": "${HF_PRIMARY}",
  "cpu_tuning": "${CPU_TUNING}",
  "execution_provider": "${EXECUTION_PROVIDER}",
  "minimal_build": "${MINIMAL_BUILD}"
}
EOF

if [ -f "/manifest/release.yaml" ]; then
    echo "    Copying manifest snapshot"
    cp "/manifest/release.yaml" "${STAGE_DIR}/manifest.snapshot.yaml"
fi

echo "    Computing SHA256SUMS"
(cd "${STAGE_DIR}" && sha256sum libonnxruntime.so operators.config build-info.json > SHA256SUMS)

# ---------------------------------------------------------------------------
# 12. Create tarball
# ---------------------------------------------------------------------------
echo "==> Creating tarball"
TARBALL="${OUTPUT_DIR}/ort-${ORT_VERSION}-${TARGET_ID}-linux-arm64.tar.gz"
tar -czf "${TARBALL}" -C "${STAGE_DIR}" .

# ---------------------------------------------------------------------------
# 13. Success
# ---------------------------------------------------------------------------
echo "==> Build complete"
echo "    Tarball: ${TARBALL}"
