"""Regression guards for CI workflow and build script contracts."""

from pathlib import Path


ROOT = Path(__file__).parent.parent
BUILD_SCRIPT = ROOT / "scripts" / "build_target.sh"
WORKFLOW = ROOT / ".github" / "workflows" / "build.yml"


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
