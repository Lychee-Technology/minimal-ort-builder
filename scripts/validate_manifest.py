#!/usr/bin/env python3
"""Validate a build manifest YAML file.

Usage:
    python scripts/validate_manifest.py builds/release.yaml
    python scripts/validate_manifest.py -   # reads from stdin

Exits 0 and prints 'OK: manifest is valid' on success.
Exits non-zero and prints 'ERROR: <message>' to stderr on failure.
"""

import sys
import yaml


def _fail(message: str) -> None:
    print(f"ERROR: {message}", file=sys.stderr)
    sys.exit(1)


def _require_keys(mapping: dict, keys: list[str], context: str) -> None:
    for key in keys:
        if key not in mapping:
            _fail(f"missing required key '{key}' in {context}")


def _is_safe_path(path) -> bool:
    """Return True if the path is relative and contains no '..' segments."""
    if not isinstance(path, str):
        _fail(f"path value must be a string, got {type(path).__name__}")
    if path.startswith("/"):
        return False
    parts = path.replace("\\", "/").split("/")
    return ".." not in parts


def validate(data: dict) -> None:
    # Rule 1: required top-level keys
    _require_keys(data, ["release", "onnxruntime", "build", "targets"], "top level")

    # Rule 2: release sub-keys
    _require_keys(data["release"], ["name", "notes"], "release")

    # Rule 3: onnxruntime sub-keys
    _require_keys(data["onnxruntime"], ["version"], "onnxruntime")

    # Rule 4: build sub-keys
    _require_keys(
        data["build"],
        [
            "container_image",
            "target_os",
            "target_arch",
            "cpu_tuning",
            "execution_provider",
            "minimal_build",
        ],
        "build",
    )

    # Rule 5: targets must be a non-empty list
    targets = data["targets"]
    if not isinstance(targets, list) or len(targets) == 0:
        _fail("'targets' must be a non-empty list")

    seen_ids: set[str] = set()

    for i, target in enumerate(targets):
        ctx = f"targets[{i}]"

        # Rule 6: each target must have 'id' and 'model'
        _require_keys(target, ["id", "model"], ctx)

        # Rule 7: target ids must be unique
        tid = target["id"]
        if tid in seen_ids:
            _fail(f"duplicate target id '{tid}'")
        seen_ids.add(tid)

        # Rule 8: no per-target 'onnxruntime' key
        if "onnxruntime" in target:
            _fail(f"{ctx} must not contain an 'onnxruntime' key")

        # Rule 9: model must have required sub-keys
        model = target["model"]
        _require_keys(
            model, ["repo_id", "revision", "primary", "companions"], f"{ctx}.model"
        )

        primary = model["primary"]
        companions = model["companions"] or []

        # Rule 9b: companions must be a list
        if not isinstance(companions, list):
            _fail(
                f"{ctx}.model: 'companions' must be a list, got {type(companions).__name__}"
            )

        # Rule 10: primary must NOT appear in companions
        if primary in companions:
            _fail(
                f"{ctx}.model: primary path '{primary}' must not appear in companions"
            )

        # Rule 11: all model paths must be relative (no leading '/', no '..')
        for path in [primary] + list(companions):
            if not _is_safe_path(path):
                _fail(
                    f"{ctx}.model: path '{path}' must be relative "
                    f"(no leading '/' and no '..' segments)"
                )


def main() -> None:
    if len(sys.argv) != 2:
        print(
            "Usage: validate_manifest.py <path> | -",
            file=sys.stderr,
        )
        sys.exit(1)

    arg = sys.argv[1]
    if arg == "-":
        raw = sys.stdin.read()
    else:
        try:
            with open(arg, "r", encoding="utf-8") as fh:
                raw = fh.read()
        except OSError as exc:
            _fail(str(exc))

    try:
        data = yaml.safe_load(raw)
    except yaml.YAMLError as exc:
        _fail(f"YAML parse error: {exc}")

    if not isinstance(data, dict):
        _fail("manifest must be a YAML mapping at the top level")

    validate(data)
    print("OK: manifest is valid")


if __name__ == "__main__":
    main()
