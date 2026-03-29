"""Tests for scripts/optimize_model.py — CLI argument validation."""

import subprocess
import sys
from pathlib import Path

SCRIPT = Path(__file__).parent.parent / "scripts" / "optimize_model.py"


def _run(*extra_args):
    cmd = [sys.executable, str(SCRIPT)] + list(extra_args)
    return subprocess.run(cmd, capture_output=True, text=True)


def test_missing_required_args_exits_nonzero():
    """Running with no args must exit non-zero (argparse error)."""
    result = _run()
    assert result.returncode != 0


def test_missing_input_file_exits_nonzero(tmp_path):
    """--input pointing at a nonexistent file must exit non-zero with a clear error."""
    result = _run(
        "--input", str(tmp_path / "nonexistent.onnx"),
        "--output", str(tmp_path / "out.onnx"),
        "--model_type", "bert",
        "--max_length", "512",
        "--target_platform", "arm",
        "--work_dir", str(tmp_path / "work"),
    )
    assert result.returncode != 0
    assert "not found" in result.stderr.lower() or "error" in result.stderr.lower()


def test_invalid_target_platform_exits_nonzero(tmp_path):
    """--target_platform with an invalid value must exit non-zero.
    argparse rejects the value before any file I/O, so no real input file is needed.
    """
    result = _run(
        "--input", str(tmp_path / "model.onnx"),
        "--output", str(tmp_path / "out.onnx"),
        "--model_type", "bert",
        "--max_length", "512",
        "--target_platform", "sparc",
        "--work_dir", str(tmp_path / "work"),
    )
    assert result.returncode != 0


def test_help_exits_zero():
    """--help must exit 0 and mention key flags."""
    result = _run("--help")
    assert result.returncode == 0
    assert "--input" in result.stdout
    assert "--output" in result.stdout
    assert "--model_type" in result.stdout
    assert "--max_length" in result.stdout
    assert "--target_platform" in result.stdout
