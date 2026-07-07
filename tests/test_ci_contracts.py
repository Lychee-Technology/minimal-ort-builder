"""Regression guards for CI workflow and build script contracts."""

from pathlib import Path


ROOT = Path(__file__).parent.parent
BUILD_SCRIPT = ROOT / "scripts" / "build_target.sh"
WORKFLOW = ROOT / ".github" / "workflows" / "build.yml"
SMOKE_TEST = ROOT / "scripts" / "smoke_test.c"
DOCKERFILE = ROOT / "docker" / "lambda-build.Dockerfile"

RELEASE_MANIFEST = ROOT / "builds" / "release.yaml"


def test_build_script_uses_fixed_ort_conversion() -> None:
    """CI must convert optimized ONNX to a single fixed ORT artifact."""
    text = BUILD_SCRIPT.read_text(encoding="utf-8")
    assert "--optimization_style Fixed" in text
    assert "--optimization_style Runtime" not in text


def test_build_script_selects_named_model_ort() -> None:
    """Artifact selection must target the stable model.ort path explicitly."""
    text = BUILD_SCRIPT.read_text(encoding="utf-8")
    assert 'EXPECTED_ORT_MODEL="${ORT_MODEL_DIR}/model.ort"' in text
    assert 'find "${ORT_MODEL_DIR}" -name "*.ort"' not in text


def test_build_script_dumps_ort_output_dir_on_missing_artifact() -> None:
    """Failure diagnostics must show ORT output contents before exiting."""
    text = BUILD_SCRIPT.read_text(encoding="utf-8")
    assert 'find "${ORT_MODEL_DIR}" -maxdepth 2 -type f | sort >&2' in text


def test_workflow_runs_targeted_pytest_before_matrix_emit() -> None:
    """The plan job must exercise regression tests before build fan-out."""
    text = WORKFLOW.read_text(encoding="utf-8")
    assert "- name: Run regression tests" in text
    assert (
        "pytest tests/test_validate_manifest.py tests/test_optimize_model.py "
        "tests/test_ci_contracts.py tests/test_gen_reference_vectors.py -q"
        in text
    )


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


def test_dockerfile_installs_cpu_torch() -> None:
    """The build image must install CPU-only torch for transformer optimization."""
    text = DOCKERFILE.read_text(encoding="utf-8")
    assert "download.pytorch.org/whl/cpu" in text
    assert "torch==" in text


def test_dockerfile_keeps_onnxruntime_pinned() -> None:
    """Torch support must not loosen the onnxruntime version pin."""
    text = DOCKERFILE.read_text(encoding="utf-8")
    assert '"onnxruntime==1.27.0"' in text


def test_build_script_keeps_converter_at_enable_all() -> None:
    """The build script should not downgrade ORT conversion away from ENABLE_ALL."""
    text = BUILD_SCRIPT.read_text(encoding="utf-8")
    assert "ORT_ENABLE_ALL" in text
    assert (
        "sed -i 's/return ort.GraphOptimizationLevel.ORT_ENABLE_ALL/"
        "return ort.GraphOptimizationLevel.ORT_ENABLE_BASIC/'"
    ) not in text


def test_build_script_does_not_force_max_length_shape_specialization() -> None:
    """The build should keep dynamic shapes unless max_length is explicitly opted in."""
    text = BUILD_SCRIPT.read_text(encoding="utf-8")
    assert '--max_length "${MAX_LENGTH}"' not in text
    assert ".max_length // 512" not in text


def test_workflow_uses_node24_ready_major_action_versions() -> None:
    """Workflow should pin actions to Node 24-ready major versions."""
    text = WORKFLOW.read_text(encoding="utf-8")
    assert "actions/checkout@v6" in text
    assert "actions/setup-python@v6" in text
    assert "actions/upload-artifact@v7" in text
    assert "actions/download-artifact@v7" in text
    assert "actions/checkout@v6.0.2" not in text
    assert "actions/setup-python@v6.2.0" not in text
    assert "actions/upload-artifact@v7.0.0" not in text
    assert "actions/download-artifact@v7.0.0" not in text


def test_smoke_test_aligns_attention_mask_with_input_ids_sequence_length() -> None:
    """Smoke inputs must keep attention_mask sequence length aligned with input_ids."""
    text = SMOKE_TEST.read_text(encoding="utf-8")
    assert "input_ids_seq_len" in text
    assert (
        'strcmp(name, "input_ids") == 0' in text
        or 'strcmp(input_names[i], "input_ids") == 0' in text
    )
    assert 'strcmp(input_names[i], "attention_mask") == 0' in text
    assert "dims[1] = input_ids_seq_len" in text


def test_smoke_test_still_disables_runtime_graph_optimization() -> None:
    """The smoke test must keep runtime graph optimization disabled."""
    text = SMOKE_TEST.read_text(encoding="utf-8")
    assert "ORT_DISABLE_ALL" in text


def test_release_manifest_uses_published_quantized_onnx_for_jina_nano() -> None:
    """The Jina nano target should consume the published quantized ONNX artifact."""
    text = RELEASE_MANIFEST.read_text(encoding="utf-8")
    assert "primary: onnx/model_q4f16.onnx" in text
    assert "- onnx/model_q4f16.onnx_data" in text


def test_build_script_skips_only_step4_for_prequantized_primary() -> None:
    """Pre-quantized ONNX inputs should still run optimize_model.py and only skip step 4."""
    text = BUILD_SCRIPT.read_text(encoding="utf-8")
    assert 'PREQUANTIZED_PRIMARY=0' in text
    assert 'basename "${HF_PRIMARY}"' in text
    assert 'Using pre-quantized ONNX model directly' not in text
    assert '--skip-int8-quantize' in text
    assert 'python3 "$(dirname "$0")/optimize_model.py"' in text


def test_dockerfile_installs_tokenizers() -> None:
    """The build image must install the tokenizers library for correctness vectors."""
    text = DOCKERFILE.read_text(encoding="utf-8")
    assert "tokenizers" in text


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
