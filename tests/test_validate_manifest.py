"""Tests for scripts/validate_manifest.py — written before implementation (TDD)."""

import subprocess
import sys
import textwrap
from pathlib import Path

SCRIPT = Path(__file__).parent.parent / "scripts" / "validate_manifest.py"

VALID_MANIFEST = textwrap.dedent("""\
    release:
      name: test-release
      notes: ""
    onnxruntime:
      version: "1.20.1"
    build:
      container_image: public.ecr.aws/lambda/provided:al2023
      target_os: linux
      target_arch: arm64
      cpu_tuning: neoverse-n1
      execution_provider: cpu
      minimal_build: extended
    targets:
      - id: model-a
        model:
          repo_id: org/repo
          revision: main
          primary: onnx/model.onnx
          companions: []
""")


def _run(stdin_text: str | None = None, file_arg: str | None = None):
    """Invoke the validator and return CompletedProcess."""
    cmd = [sys.executable, str(SCRIPT)]
    if file_arg is not None:
        cmd.append(file_arg)
    else:
        cmd.append("-")
    return subprocess.run(
        cmd,
        input=stdin_text,
        capture_output=True,
        text=True,
    )


def test_valid_manifest_exits_zero():
    result = _run(stdin_text=VALID_MANIFEST)
    assert result.returncode == 0, f"stderr: {result.stderr}"
    assert "OK" in result.stdout


def test_duplicate_ids_fail():
    manifest = textwrap.dedent("""\
        release:
          name: test-release
          notes: ""
        onnxruntime:
          version: "1.20.1"
        build:
          container_image: public.ecr.aws/lambda/provided:al2023
          target_os: linux
          target_arch: arm64
          cpu_tuning: neoverse-n1
          execution_provider: cpu
          minimal_build: extended
        targets:
          - id: model-a
            model:
              repo_id: org/repo
              revision: main
              primary: onnx/model.onnx
              companions: []
          - id: model-a
            model:
              repo_id: org/repo
              revision: main
              primary: onnx/other.onnx
              companions: []
    """)
    result = _run(stdin_text=manifest)
    assert result.returncode != 0
    assert "duplicate" in result.stderr.lower()


def test_primary_in_companions_fails():
    manifest = textwrap.dedent("""\
        release:
          name: test-release
          notes: ""
        onnxruntime:
          version: "1.20.1"
        build:
          container_image: public.ecr.aws/lambda/provided:al2023
          target_os: linux
          target_arch: arm64
          cpu_tuning: neoverse-n1
          execution_provider: cpu
          minimal_build: extended
        targets:
          - id: model-a
            model:
              repo_id: org/repo
              revision: main
              primary: onnx/model.onnx
              companions:
                - onnx/model.onnx
    """)
    result = _run(stdin_text=manifest)
    assert result.returncode != 0
    assert "primary" in result.stderr.lower()


def test_absolute_path_fails():
    manifest = textwrap.dedent("""\
        release:
          name: test-release
          notes: ""
        onnxruntime:
          version: "1.20.1"
        build:
          container_image: public.ecr.aws/lambda/provided:al2023
          target_os: linux
          target_arch: arm64
          cpu_tuning: neoverse-n1
          execution_provider: cpu
          minimal_build: extended
        targets:
          - id: model-a
            model:
              repo_id: org/repo
              revision: main
              primary: /absolute/model.onnx
              companions: []
    """)
    result = _run(stdin_text=manifest)
    assert result.returncode != 0
    assert "relative" in result.stderr.lower() or "absolute" in result.stderr.lower()


def test_dotdot_path_fails():
    manifest = textwrap.dedent("""\
        release:
          name: test-release
          notes: ""
        onnxruntime:
          version: "1.20.1"
        build:
          container_image: public.ecr.aws/lambda/provided:al2023
          target_os: linux
          target_arch: arm64
          cpu_tuning: neoverse-n1
          execution_provider: cpu
          minimal_build: extended
        targets:
          - id: model-a
            model:
              repo_id: org/repo
              revision: main
              primary: ../model.onnx
              companions: []
    """)
    result = _run(stdin_text=manifest)
    assert result.returncode != 0


def test_missing_required_field_fails():
    manifest = textwrap.dedent("""\
        release:
          name: test-release
          notes: ""
        build:
          container_image: public.ecr.aws/lambda/provided:al2023
          target_os: linux
          target_arch: arm64
          cpu_tuning: neoverse-n1
          execution_provider: cpu
          minimal_build: extended
        targets:
          - id: model-a
            model:
              repo_id: org/repo
              revision: main
              primary: onnx/model.onnx
              companions: []
    """)
    result = _run(stdin_text=manifest)
    assert result.returncode != 0
