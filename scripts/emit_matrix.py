#!/usr/bin/env python3
"""Emit a GitHub Actions matrix JSON from a validated build manifest.

Usage:
    python scripts/emit_matrix.py builds/release.yaml
    python scripts/emit_matrix.py -   # reads from stdin

Prints a one-line JSON array to stdout.  No validation is performed; the
manifest is assumed to be already validated.
"""

import json
import sys

import yaml


def emit(data: dict) -> list[dict]:
    build = data["build"]
    ort_version = data["onnxruntime"]["version"]

    entries = []
    for target in data["targets"]:
        model = target["model"]
        companions = model.get("companions") or []
        bundle_extras = target.get("bundle_extras") or []
        metadata = target.get("metadata") or {}
        entries.append(
            {
                "target_id": target["id"],
                "target_id_safe": target["id"].replace("/", "__"),
                "quant": target["quant"],
                "ort_version": ort_version,
                "container_image": build["container_image"],
                "cpu_tuning": build["cpu_tuning"],
                "execution_provider": build["execution_provider"],
                "minimal_build": build["minimal_build"],
                "hf_repo_id": model["repo_id"],
                "hf_revision": model["revision"],
                "hf_primary": model["primary"],
                "hf_companions": " ".join(companions),
                "bundle_extras": " ".join(bundle_extras),
                "model_metadata": json.dumps(metadata, sort_keys=True),
            }
        )
    return entries


def main() -> None:
    if len(sys.argv) != 2:
        print("Usage: emit_matrix.py <path> | -", file=sys.stderr)
        sys.exit(1)

    arg = sys.argv[1]
    if arg == "-":
        raw = sys.stdin.read()
    else:
        try:
            with open(arg, "r", encoding="utf-8") as fh:
                raw = fh.read()
        except OSError as exc:
            print(f"ERROR: {exc}", file=sys.stderr)
            sys.exit(1)

    try:
        data = yaml.safe_load(raw)
    except yaml.YAMLError as exc:
        print(f"ERROR: YAML parse error: {exc}", file=sys.stderr)
        sys.exit(1)

    try:
        matrix = emit(data)
    except (KeyError, TypeError) as exc:
        print(f"ERROR: malformed manifest: {exc}", file=sys.stderr)
        sys.exit(1)
    print(json.dumps(matrix, sort_keys=True))


if __name__ == "__main__":
    main()
