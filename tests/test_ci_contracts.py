"""Regression guards for CI workflow and build script contracts."""

from pathlib import Path


ROOT = Path(__file__).parent.parent
BUILD_SCRIPT = ROOT / "scripts" / "build_target.sh"
WORKFLOW = ROOT / ".github" / "workflows" / "build.yml"
SMOKE_TEST = ROOT / "scripts" / "smoke_test.c"


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
        "pytest tests/test_validate_manifest.py tests/test_optimize_model.py tests/test_ci_contracts.py -q"
        in text
    )


def test_workflow_cache_key_tracks_build_inputs() -> None:
    """The build cache key must include workflow-relevant scripts."""
    text = WORKFLOW.read_text(encoding="utf-8")
    expected = "hashFiles('builds/release.yaml', 'scripts/build_target.sh', 'scripts/optimize_model.py', 'docker/lambda-build.Dockerfile')"
    assert expected in text


def test_workflow_uses_node24_ready_major_action_versions() -> None:
    """Workflow should pin actions to Node 24-ready major versions."""
    text = WORKFLOW.read_text(encoding="utf-8")
    assert "actions/checkout@v6" in text
    assert "actions/setup-python@v6" in text
    assert "actions/cache@v5" in text
    assert "actions/upload-artifact@v7" in text
    assert "actions/download-artifact@v7" in text
    assert "actions/checkout@v6.0.2" not in text
    assert "actions/setup-python@v6.2.0" not in text
    assert "actions/cache@v5.0.4" not in text
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
