#!/usr/bin/env python3
"""render_benchmark_report.py — aggregate per-target bench JSON into a report.

    render_benchmark_report.py <input_dir> [output_dir]

Globs `bench-*.json` from <input_dir>, renders a side-by-side Markdown comparison
table (one row per target/quant), and writes:

    <output_dir>/report.md      the Markdown table
    <output_dir>/results.json   the combined list of all target results

If $GITHUB_STEP_SUMMARY is set, the table is also appended there so it renders in
the Actions run UI. <output_dir> defaults to the current directory.
"""

import glob
import json
import os
import sys
from pathlib import Path

# (json_key, column header) in issue-specified order.
COLUMNS = [
    ("target_id", "target_id"),
    ("quant", "quant"),
    ("load_ms", "load_ms"),
    ("mean_ms", "mean_ms"),
    ("p50_ms", "p50_ms"),
    ("p90_ms", "p90_ms"),
    ("p99_ms", "p99_ms"),
    ("throughput_ips", "throughput_ips"),
    ("peak_rss_kb", "peak_rss_kb"),
    ("so_bytes", "so_bytes"),
    ("model_ort_bytes", "model_ort_bytes"),
    ("tarball_bytes", "tarball_bytes"),
    ("ort_version", "ort_version"),
]


def _fmt(key, value):
    if value is None:
        return ""
    if key.endswith("_ms"):
        return f"{float(value):.3f}"
    if key == "throughput_ips":
        return f"{float(value):.2f}"
    if key.endswith("_bytes") or key.endswith("_kb"):
        return f"{int(value):,}"
    return str(value)


def render_table(results):
    headers = [header for _, header in COLUMNS]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in results:
        cells = [_fmt(key, row.get(key)) for key, _ in COLUMNS]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


def main(argv=None):
    argv = argv if argv is not None else sys.argv[1:]
    if not argv:
        sys.exit("Usage: render_benchmark_report.py <input_dir> [output_dir]")
    input_dir = Path(argv[0])
    output_dir = Path(argv[1]) if len(argv) > 1 else Path(".")
    output_dir.mkdir(parents=True, exist_ok=True)

    results = []
    for path in sorted(glob.glob(str(input_dir / "bench-*.json"))):
        with open(path, "r", encoding="utf-8") as fh:
            results.append(json.load(fh))
    results.sort(key=lambda r: (r.get("target_id", ""), r.get("quant", "")))

    table = render_table(results)
    report = "# Benchmark comparison\n\n" + table

    (output_dir / "report.md").write_text(report, encoding="utf-8")
    (output_dir / "results.json").write_text(
        json.dumps(results, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    summary_path = os.environ.get("GITHUB_STEP_SUMMARY")
    if summary_path:
        with open(summary_path, "a", encoding="utf-8") as fh:
            fh.write(report)

    print(f"rendered {len(results)} result(s) to {output_dir / 'report.md'}")
    sys.stdout.write(report)


if __name__ == "__main__":
    main()
