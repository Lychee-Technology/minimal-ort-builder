#!/usr/bin/env bash
# build_target.sh — ORT minimal build entrypoint, runs inside the lambda/provided:al2023 container.
# All configuration is supplied via environment variables.

set -euo pipefail

# ---------------------------------------------------------------------------
# 1. Validate required env vars; set defaults for optional ones
# ---------------------------------------------------------------------------
: "${TARGET_ID:?TARGET_ID is required}"
: "${ORT_VERSION:?ORT_VERSION is required}"
: "${QUANT:?QUANT is required}"
: "${HF_REPO_ID:?HF_REPO_ID is required}"
: "${HF_REVISION:?HF_REVISION is required}"
: "${HF_PRIMARY:?HF_PRIMARY is required}"
: "${HF_COMPANIONS:=""}"         # optional, may be empty
: "${BUNDLE_EXTRAS:=""}"         # optional, space-separated list of extra filenames to include
: "${MODEL_METADATA:="{}"}"     # optional, JSON object of model metadata from manifest
CPU_TUNING="${CPU_TUNING:-neoverse-n1}"
EXECUTION_PROVIDER="${EXECUTION_PROVIDER:-cpu}"
MINIMAL_BUILD="${MINIMAL_BUILD:-extended}"  # "basic" breaks CPU EP runtime partitioning; "extended" is the minimum viable level
OUTPUT_DIR="${OUTPUT_DIR:-/output}"
FIXTURE_DIR="${FIXTURE_DIR:-/fixtures}"

echo "==> Configuration"
echo "    TARGET_ID          = ${TARGET_ID}"
echo "    ORT_VERSION        = ${ORT_VERSION}"
echo "    QUANT              = ${QUANT}"
echo "    HF_REPO_ID         = ${HF_REPO_ID}"
echo "    HF_REVISION        = ${HF_REVISION}"
echo "    HF_PRIMARY         = ${HF_PRIMARY}"
echo "    HF_COMPANIONS      = ${HF_COMPANIONS}"
echo "    BUNDLE_EXTRAS      = ${BUNDLE_EXTRAS}"
echo "    MODEL_METADATA     = ${MODEL_METADATA}"
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
hf download "${HF_REPO_ID}" "${HF_PRIMARY}" \
    --revision "${HF_REVISION}" \
    --local-dir "${MODEL_DIR}"

# ---------------------------------------------------------------------------
# 4. Download each companion file
# ---------------------------------------------------------------------------
if [ -n "${HF_COMPANIONS}" ]; then
    echo "==> Downloading companion files"
    for companion in ${HF_COMPANIONS}; do
        echo "    Downloading companion: ${companion}"
        hf download "${HF_REPO_ID}" "${companion}" \
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
# 5b. Decide whether to run int8 dynamic quantization (optimize step 4)
#
# The build pipeline can only ever *produce* int8 dynamic quantization itself
# (optimize_model.py step 4 = QuantType.QInt8). Every other quantization scheme
# (e.g. q4f16) must arrive already-quantized from Hugging Face, so running step 4
# on it would corrupt an already-quantized graph. We therefore run step 4 only
# for the opt-in set of quant labels that name pipeline-produced int8; anything
# else is treated as pre-quantized and skips it. Safe by default: an unrecognised
# (future) pre-quantized scheme skips int8 rather than being re-quantized.
# See issue #21. QUANT is validated lowercase, so no case-folding is needed.
#
# CAVEAT: the quant label names the *output* scheme (it feeds the tarball
# filename), whereas int8|q8 here means "the pipeline should *produce* int8".
# These only collide if a target ships an *already*-quantized q8 model from HF:
# labelling it quant: q8 would re-quantize it. No such target exists today; if
# one is added, label it with a non-q8 scheme (or revisit this case) so it is
# treated as pre-quantized.
# ---------------------------------------------------------------------------
# QUANT_SCHEME selects which step-4 quantization the pipeline runs on the fp32
# primary (only meaningful when PREQUANTIZED_PRIMARY=0):
#   int8  → dynamic QInt8 (issue #23)
#   gptq4 → calibrated 4-bit MatMulNBits (issue #27); replaces jina's uncalibrated
#           RTN q4/q4f16 (~0.62 cosine) with a calibration-fitted 4-bit export.
PREQUANTIZED_PRIMARY=1
QUANT_SCHEME=""
case "${QUANT}" in
    int8|q8) PREQUANTIZED_PRIMARY=0; QUANT_SCHEME=int8  ;;
    q4gptq)  PREQUANTIZED_PRIMARY=0; QUANT_SCHEME=gptq4 ;;
esac

# ---------------------------------------------------------------------------
# 6. Clone ORT source
# ---------------------------------------------------------------------------
echo "==> Cloning ORT ${ORT_VERSION}"
git clone --depth 1 --branch "${ORT_VERSION}" \
    https://github.com/microsoft/onnxruntime.git "${ORT_SRC}"

# ---------------------------------------------------------------------------
# 7. Prepare ONNX model for ORT conversion
# ---------------------------------------------------------------------------
OPTIMIZED_ONNX="${WORK_DIR}/optimized.onnx"
MODEL_TYPE="$(echo "${MODEL_METADATA}" | jq -r '.model_type // "bert"')"
REFERENCE_ONNX="${WORK_DIR}/reference.onnx"
COSINE_THRESHOLD="$(echo "${MODEL_METADATA}" | jq -r '.correctness.cosine_threshold // 0.99')"
NUM_SAMPLES="$(echo "${MODEL_METADATA}" | jq -r '.correctness.num_samples // 3')"
MAX_TOKENS="$(echo "${MODEL_METADATA}" | jq -r '.correctness.max_tokens // 128')"
case "$(uname -m)" in
    aarch64|arm*) ORT_TARGET_PLATFORM="arm" ;;
    *)            ORT_TARGET_PLATFORM="amd64" ;;
esac

echo "==> Optimizing ONNX model"
OPT_WORK_DIR="${WORK_DIR}/optimize"
OPTIMIZE_EXTRA_ARGS=()
if [ "${PREQUANTIZED_PRIMARY}" = "1" ]; then
    OPTIMIZE_EXTRA_ARGS+=(--skip-int8-quantize)
elif [ "${QUANT_SCHEME}" = "gptq4" ]; then
    # Calibrated 4-bit weight quantization: feed the model's own tokenizer (a
    # downloaded companion) and the shared fixture through GPTQ. NUM_SAMPLES /
    # MAX_TOKENS mirror the correctness gate so calibration and scoring align.
    OPTIMIZE_EXTRA_ARGS+=(
        --quant-scheme gptq4
        --calibration-tokenizer "${MODEL_DIR}/tokenizer.json"
        --calibration-fixture "${FIXTURE_DIR}/jane-austen_pride-and-prejudice.jsonl"
        --calibration-num-samples "${NUM_SAMPLES}"
        --calibration-max-tokens "${MAX_TOKENS}"
    )
fi
python3 "$(dirname "$0")/optimize_model.py" \
    --input "${MODEL_DIR}/${HF_PRIMARY}" \
    --output "${OPTIMIZED_ONNX}" \
    --model_type "${MODEL_TYPE}" \
    --target_platform "${ORT_TARGET_PLATFORM}" \
    --work_dir "${OPT_WORK_DIR}" \
    --reference-output "${REFERENCE_ONNX}" \
    "${OPTIMIZE_EXTRA_ARGS[@]}"
# NOTE: ORT_TARGET_PLATFORM is set here and reused by step 11 (convert_onnx_models_to_ort.py).
# The variable must remain in scope for the rest of the script (no subshells between steps).

# Place optimized model in its own directory for convert_onnx_models_to_ort.py
# (the converter takes a directory argument and converts all .onnx files inside it)
ORT_INPUT_DIR="${WORK_DIR}/ort_input"
mkdir -p "${ORT_INPUT_DIR}"
cp "${OPTIMIZED_ONNX}" "${ORT_INPUT_DIR}/model.onnx"

# ---------------------------------------------------------------------------
# 8. Convert ONNX → ORT format (using the installed onnxruntime Python
#    package).  Runs BEFORE the build so the operator config (step 9) is
#    derived from the same model the smoke test will load.
#
#    --optimization_style Fixed: graph optimization is applied once at
#    conversion time and frozen into the .ort file. optimize_model.py already
#    emits an ORT_ENABLE_ALL-optimized ONNX, so the converter is expected to
#    preserve that graph in .ort form rather than downgrade it.
# ---------------------------------------------------------------------------
echo "==> Converting ONNX to ORT format (using installed onnxruntime)"
ORT_MODEL_DIR="${WORK_DIR}/ort_model"
mkdir -p "${ORT_MODEL_DIR}"
python3 "${ORT_SRC}/tools/python/convert_onnx_models_to_ort.py" \
    "${ORT_INPUT_DIR}" \
    --optimization_style Fixed \
    --target_platform "${ORT_TARGET_PLATFORM}" \
    --output_dir "${ORT_MODEL_DIR}"

EXPECTED_ORT_MODEL="${ORT_MODEL_DIR}/model.ort"
if [ ! -f "${EXPECTED_ORT_MODEL}" ]; then
    echo "ERROR: expected ORT model missing at ${EXPECTED_ORT_MODEL}" >&2
    echo "ERROR: ORT output directory contents:" >&2
    find "${ORT_MODEL_DIR}" -maxdepth 2 -type f | sort >&2
    exit 1
fi
ORT_MODEL_PATH="${EXPECTED_ORT_MODEL}"
echo "    ORT model: ${ORT_MODEL_PATH}"

# ---------------------------------------------------------------------------
# 9. Generate reduced operator config (from the CONVERTED .ort, not the .onnx)
#
# The config must list exactly the ops the minimal build will register at
# runtime. convert_onnx_models_to_ort.py (--optimization_style Fixed, ENABLE_ALL)
# can FUSE nodes into ops that are absent from the pre-conversion .onnx — most
# notably, dynamic int8 quantization emits MatMulInteger + Mul, which the
# converter fuses into the MatMulIntegerToFloat contrib op. Deriving the config
# from the .onnx therefore omits that fused op, and the minimal build then fails
# at CreateSession with "Could not find an implementation for MatMulIntegerToFloat".
# Reading the .ort (via --format ORT) makes the config match the shipped graph
# exactly. For pre-quantized targets the .ort op set is identical to what already
# shipped, so this is a no-op for them.
# ---------------------------------------------------------------------------
echo "==> Generating reduced operator config"
OPERATOR_CONFIG="${WORK_DIR}/operators.config"
python3 "${ORT_SRC}/tools/python/create_reduced_build_config.py" \
    --format ORT \
    "${ORT_MODEL_DIR}" \
    "${OPERATOR_CONFIG}"

# ---------------------------------------------------------------------------
# 10. Build ORT
# ---------------------------------------------------------------------------
echo "==> Building ORT (this will take a while)"

CMAKE_EXTRA_DEFINES=(
    "onnxruntime_BUILD_SHARED_LIB=ON"
    "CMAKE_C_COMPILER=clang"
    "CMAKE_CXX_COMPILER=clang++"
    # -O3: upgrades from CMake Release's default -O2; enables additional loop
    # unrolling, auto-vectorization, and inter-procedural optimisations.
    # -flto=thin: ThinLTO — lightweight whole-program analysis at link time
    # (cross-TU inlining, dead-symbol stripping).  Requires matching flags on
    # all three linker variables below.  lld (already required) has native
    # ThinLTO support; no linker plugin needed.
    "CMAKE_C_FLAGS=-O3 -flto=thin"
    "CMAKE_CXX_FLAGS=-O3 -flto=thin"
    "CMAKE_EXE_LINKER_FLAGS=-fuse-ld=lld -flto=thin"
    "CMAKE_SHARED_LINKER_FLAGS=-fuse-ld=lld -flto=thin"
    "CMAKE_MODULE_LINKER_FLAGS=-fuse-ld=lld -flto=thin"
)

if [ "${CPU_TUNING}" = "neoverse-n1" ]; then
    CMAKE_EXTRA_DEFINES+=(
        # Neoverse N1 = ARMv8.2-A baseline + dotprod + fp16 + rcpc (FEAT_LRCPC).
        #
        # Extensions that generate different code automatically (Clang, no intrinsics):
        #
        #   +dotprod  MLAS: QgemmU8X8KernelUdot.S, QgemmS8S8KernelSdot.S, etc.
        #             (4x INT8 GEMM throughput; selected at runtime via HWCAP_ASIMDDP)
        #             Non-MLAS: Clang auto-vectorizes int8 dot-product loops to
        #             sdot/udot. Zero effect on float loops.
        #
        #   +fp16     MLAS: HalfGemmKernelNeon.S + 9 other half-precision kernel
        #             files compiled per-file with -march=armv8.2-a+fp16 in cmake.
        #             Non-MLAS: enables vectorized _Float16 arithmetic. No effect
        #             on float (fp32) loops — Clang never auto-downcasts fp32→fp16.
        #
        #   +rcpc     NOT in armv8.2-a baseline; must be listed explicitly.
        #             Clang lowers memory_order_acquire loads to ldapr instead of
        #             ldar. ldapr is RCpc (cheaper: only orders with prior stores,
        #             not prior loads) and avoids pipeline stalls on N1. ORT's
        #             thread pool uses acquire/release heavily for barrier counters
        #             and task queue synchronisation → measurable latency reduction.
        #
        # armv8.2-a already implies: lse (ldadd/cas auto-generated from
        #   std::atomic RMW — eliminates LL/SC retry loops in the thread pool),
        #   crc, rdm. Listing them explicitly would be redundant.
        #
        # Extensions with zero codegen effect (Clang never auto-generates them):
        #   crypto (AES/SHA require explicit intrinsics)
        #   ras    (microarchitectural error-reporting, no user-space insns)
        #   ssbs   (speculative store bypass: OS/hypervisor PSTATE bit, not code)
        #
        # MLAS kernels override their own -march per-file via
        # set_source_files_properties, so this global flag only influences the
        # non-MLAS ORT C++ code (op dispatch, thread pool, allocators, etc.).
        #
        # +sve is intentionally absent — --no_sve is passed below — because
        # Clang 20 auto-enables SVE for neoverse-n1 via -mcpu, defining
        # __ARM_FEATURE_SVE and causing linker errors against SVE symbols that
        # are absent in a minimal build.
        "CMAKE_CXX_FLAGS=-O3 -flto=thin -march=armv8.2-a+dotprod+fp16+rcpc -mtune=neoverse-n1"
        "CMAKE_C_FLAGS=-O3 -flto=thin -march=armv8.2-a+dotprod+fp16+rcpc -mtune=neoverse-n1"
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
# 11. Verify libonnxruntime.so exists
# ---------------------------------------------------------------------------
echo "==> Verifying build output"
BUILT_SO="${BUILD_DIR}/Release/libonnxruntime.so"
if [ ! -f "${BUILT_SO}" ]; then
    echo "ERROR: libonnxruntime.so not found at ${BUILT_SO}" >&2
    exit 1
fi

# ---------------------------------------------------------------------------
# 12. Compile and run smoke test
# ---------------------------------------------------------------------------
echo "==> Running smoke test"
SMOKE_SRC="$(dirname "$0")/smoke_test.c"
SMOKE_BIN="${WORK_DIR}/smoke_test"
SMOKE_LOG="${STAGE_DIR}/smoke-test.log"

clang -o "${SMOKE_BIN}" "${SMOKE_SRC}" \
    -I "${ORT_SRC}/include/onnxruntime/core/session" \
    -L "$(dirname "${BUILT_SO}")" \
    -lonnxruntime \
    -lm \
    -Wl,-rpath,"$(dirname "${BUILT_SO}")"

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

if [ "${SMOKE_EXIT}" -ne 0 ]; then
    echo "ERROR: smoke test failed with exit code ${SMOKE_EXIT}" >&2
    exit 1
fi

# ---------------------------------------------------------------------------
# 13. Stage artifacts
# ---------------------------------------------------------------------------
echo "==> Staging artifacts"

cp "${BUILT_SO}" "${STAGE_DIR}/libonnxruntime.so"
cp "${OPERATOR_CONFIG}" "${STAGE_DIR}/operators.config"
cp "${ORT_MODEL_PATH}" "${STAGE_DIR}/model.ort"

# Copy companion files (e.g. tokenizer.json) into the stage dir by basename.
# The tarball is a flat bundle, so all files land at the root level.
if [ -n "${HF_COMPANIONS}" ]; then
    for companion in ${HF_COMPANIONS}; do
        COMPANION_PATH="${MODEL_DIR}/${companion}"
        COMPANION_BASENAME="$(basename "${companion}")"
        echo "    Staging companion: ${COMPANION_BASENAME}"
        cp "${COMPANION_PATH}" "${STAGE_DIR}/${COMPANION_BASENAME}"
    done
fi

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
  --argjson model_metadata "${MODEL_METADATA}" \
  '{target_id:$target_id,ort_version:$ort_version,ort_git_sha:$ort_git_sha,hf_repo_id:$hf_repo_id,hf_revision:$hf_revision,hf_primary:$hf_primary,cpu_tuning:$cpu_tuning,execution_provider:$execution_provider,minimal_build:$minimal_build,model_metadata:$model_metadata}' \
  > "${STAGE_DIR}/build-info.json"

if [ -f "/manifest/release.yaml" ]; then
    echo "    Copying manifest snapshot"
    cp "/manifest/release.yaml" "${STAGE_DIR}/manifest.snapshot.yaml"
fi

# ---------------------------------------------------------------------------
# 14. Prune stage dir to the bundle whitelist
#
# Core files always kept: libonnxruntime.so, model.ort.
# All other files included only if listed in BUNDLE_EXTRAS (space-separated).
# companions are download-only — they are NOT auto-included; list any runtime
# companions (e.g. tokenizer.json) explicitly in bundle_extras in the manifest.
# Everything else (operators.config, manifest.snapshot.yaml, smoke-test.log,
# and any other temp file that lands here) is removed.
# ---------------------------------------------------------------------------
echo "    Pruning bundle to whitelist"
declare -A KEEP
KEEP["libonnxruntime.so"]=1
KEEP["model.ort"]=1

# Explicit extras from manifest (e.g. tokenizer.json, build-info.json)
if [ -n "${BUNDLE_EXTRAS}" ]; then
    for extra in ${BUNDLE_EXTRAS}; do
        KEEP["${extra}"]=1
    done
fi

for f in "${STAGE_DIR}"/*; do
    fname="$(basename "${f}")"
    if [ "${KEEP[$fname]+_}" != "_" ]; then
        echo "      Removing from bundle: ${fname}"
        rm -f "${f}"
    fi
done

echo "    Computing SHA256SUMS"
(cd "${STAGE_DIR}" && find . -type f ! -name SHA256SUMS | sort | xargs sha256sum > SHA256SUMS)

# ---------------------------------------------------------------------------
# 15. Create tarball
# ---------------------------------------------------------------------------
echo "==> Creating tarball"
# TARGET_ID may contain '/' (e.g. "jinaai/jina-embeddings-v5-text-nano-retrieval");
# replace with '__' so the tarball is a flat file under OUTPUT_DIR.
TARGET_ID_SAFE="${TARGET_ID//\//__}"
TARBALL="${OUTPUT_DIR}/${TARGET_ID_SAFE}_${QUANT}_linux-arm64.tar.gz"
mkdir -p "${OUTPUT_DIR}"
tar -czf "${TARBALL}" -C "${STAGE_DIR}" .

# ---------------------------------------------------------------------------
# 16. Success
# ---------------------------------------------------------------------------
echo "==> Build complete"
echo "    Tarball: ${TARBALL}"
echo "    Size:    $(du -sh "${TARBALL}" | cut -f1)"
