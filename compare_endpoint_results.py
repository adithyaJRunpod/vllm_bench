#!/usr/bin/env python3
"""
Compare GuideLLM benchmark results from two endpoints side-by-side.

Usage:
    python3 compare_endpoint_results.py --dir <log_directory> --label-a <name> --label-b <name>

Expects files named like: <label>_run<N>.json
"""

import argparse
import json
import os
import sys
from pathlib import Path


def extract_metrics(json_path: str) -> dict:
    with open(json_path) as f:
        data = json.load(f)

    b = data["benchmarks"][0]
    m = b["metrics"]
    s = "successful"

    def safe_get(metric, stat):
        try:
            return m[metric][s][stat]
        except (KeyError, TypeError, IndexError):
            return None

    def safe_percentile(metric, pct):
        try:
            percs = m[metric][s]["percentiles"]
            key = f"p{pct}"
            if key not in percs:
                key = f"p{pct:03d}"
            return percs[key]
        except (KeyError, TypeError, IndexError):
            return None

    totals = m.get("request_totals", {})

    return {
        "req_per_sec": safe_get("requests_per_second", "mean"),
        "concurrency_median": safe_get("request_concurrency", "median"),
        "input_tokens_median": safe_get("prompt_token_count", "median"),
        "output_tokens_median": safe_get("output_token_count", "median"),
        "output_tok_per_sec_mean": safe_get("output_tokens_per_second", "mean"),
        "total_tok_per_sec_mean": safe_get("tokens_per_second", "mean"),
        "ttft_median_ms": safe_get("time_to_first_token_ms", "median"),
        "ttft_p95_ms": safe_percentile("time_to_first_token_ms", 95),
        "itl_median_ms": safe_get("inter_token_latency_ms", "median"),
        "itl_p95_ms": safe_percentile("inter_token_latency_ms", 95),
        "tpot_median_ms": safe_get("time_per_output_token_ms", "median"),
        "tpot_p95_ms": safe_percentile("time_per_output_token_ms", 95),
        "latency_median_sec": safe_get("request_latency", "median"),
        "latency_p95_sec": safe_percentile("request_latency", 95),
        "total_completed": totals.get("successful") or totals.get("total"),
        "total_errored": totals.get("errored", 0),
    }


def median_of(values):
    vals = sorted(v for v in values if v is not None)
    if not vals:
        return None
    mid = len(vals) // 2
    if len(vals) % 2 == 1:
        return vals[mid]
    return (vals[mid - 1] + vals[mid]) / 2


def aggregate_runs(run_metrics: list[dict]) -> dict:
    if len(run_metrics) == 1:
        return run_metrics[0]
    keys = run_metrics[0].keys()
    agg = {}
    for k in keys:
        agg[k] = median_of([r[k] for r in run_metrics])
    return agg


def fmt(val, decimals=1, suffix=""):
    if val is None:
        return "N/A"
    if isinstance(val, float):
        return f"{val:,.{decimals}f}{suffix}"
    return f"{val:,}{suffix}"


def multiplier(a_val, b_val, direction="higher"):
    """Return a multiplier string showing how overdrive compares to standard.
    >1.0x means overdrive is better, <1.0x means standard is better.
    For 'higher' metrics: ratio = b/a.  For 'lower' metrics: ratio = a/b."""
    if a_val is None or b_val is None or a_val == 0 or b_val == 0:
        return "—"
    if direction == "higher":
        ratio = b_val / a_val
    else:
        ratio = a_val / b_val
    return f"{ratio:.1f}x"


def print_comparison(agg_a: dict, agg_b: dict, label_a: str, label_b: str,
                     runs_a: int, runs_b: int):
    col_metric = 24
    col_val = 14
    col_diff = 12

    header = (
        f"{'Metric':<{col_metric}}"
        f"{label_a:>{col_val}}"
        f"{label_b:>{col_val}}"
        f"{'Overdrive':>{col_diff}}"
    )
    sep = "=" * len(header)
    thin_sep = "-" * len(header)

    print(sep)
    print(header)
    print(sep)

    # higher-is-better metrics get normal diff, lower-is-better get inverted display
    rows = [
        ("Requests/sec",      "req_per_sec",           1, "",    "higher"),
        ("Output tok/s",      "output_tok_per_sec_mean",0, "",   "higher"),
        ("Total tok/s",       "total_tok_per_sec_mean", 0, "",   "higher"),
        (None, None, None, None, None),  # separator
        ("E2E Latency p50",   "latency_median_sec",     2, "s",  "lower"),
        ("E2E Latency p95",   "latency_p95_sec",        2, "s",  "lower"),
        (None, None, None, None, None),
        ("TTFT p50",          "ttft_median_ms",          0, "ms", "lower"),
        ("TTFT p95",          "ttft_p95_ms",             0, "ms", "lower"),
        (None, None, None, None, None),
        ("ITL p50",           "itl_median_ms",           1, "ms", "lower"),
        ("ITL p95",           "itl_p95_ms",              1, "ms", "lower"),
        (None, None, None, None, None),
        ("TPOT p50",          "tpot_median_ms",          1, "ms", "lower"),
        ("TPOT p95",          "tpot_p95_ms",             1, "ms", "lower"),
        (None, None, None, None, None),
        ("Completed reqs",    "total_completed",         0, "",   "higher"),
        ("Errors",            "total_errored",           0, "",   "lower"),
    ]

    lines = []
    lines.append(sep)
    lines.append(header)
    lines.append(sep)

    for row in rows:
        if row[0] is None:
            lines.append(thin_sep)
            continue

        name, key, dec, suf, direction = row
        a_val = agg_a.get(key)
        b_val = agg_b.get(key)

        diff_str = multiplier(a_val, b_val, direction)

        lines.append(
            f"{name:<{col_metric}}"
            f"{fmt(a_val, dec, suf):>{col_val}}"
            f"{fmt(b_val, dec, suf):>{col_val}}"
            f"{diff_str:>{col_diff}}"
        )

    lines.append(sep)
    lines.append("")
    lines.append(f"  {label_a}: {runs_a} run(s)  |  {label_b}: {runs_b} run(s)")
    if runs_a > 1 or runs_b > 1:
        lines.append("  Values are medians across runs.")
    lines.append("  Overdrive column: >1.0x = overdrive wins, <1.0x = standard wins")

    output = os.linesep.join(lines)
    sys.stdout.write(output + os.linesep)
    sys.stdout.flush()


def main():
    parser = argparse.ArgumentParser(description="Compare two endpoint benchmark results")
    parser.add_argument("--dir", required=True, help="Directory with benchmark JSON files")
    parser.add_argument("--label-a", required=True, help="Label for endpoint A")
    parser.add_argument("--label-b", required=True, help="Label for endpoint B")
    args = parser.parse_args()

    log_dir = Path(args.dir)
    if not log_dir.is_dir():
        print(f"Error: {log_dir} is not a directory", file=sys.stderr)
        sys.exit(1)

    label_a = args.label_a
    label_b = args.label_b

    runs_a = []
    runs_b = []

    for json_file in sorted(log_dir.glob("*.json")):
        name = json_file.stem
        if name == "environment":
            continue

        try:
            metrics = extract_metrics(str(json_file))
        except Exception as e:
            print(f"Warning: Failed to parse {json_file}: {e}", file=sys.stderr)
            continue

        if name.startswith(f"{label_a}_run"):
            runs_a.append(metrics)
        elif name.startswith(f"{label_b}_run"):
            runs_b.append(metrics)

    if not runs_a:
        print(f"Error: No results found for '{label_a}'", file=sys.stderr)
        sys.exit(1)
    if not runs_b:
        print(f"Error: No results found for '{label_b}'", file=sys.stderr)
        sys.exit(1)

    agg_a = aggregate_runs(runs_a)
    agg_b = aggregate_runs(runs_b)

    print_comparison(agg_a, agg_b, label_a, label_b, len(runs_a), len(runs_b))

    # Save comparison as CSV
    csv_path = log_dir / "comparison.csv"
    with open(csv_path, "w") as f:
        f.write(f"metric,{label_a},{label_b},overdrive_multiplier\n")
        metric_keys = [
            ("req_per_sec", "Requests/sec", "higher"),
            ("output_tok_per_sec_mean", "Output tok/s", "higher"),
            ("total_tok_per_sec_mean", "Total tok/s", "higher"),
            ("latency_median_sec", "E2E Latency p50 (s)", "lower"),
            ("latency_p95_sec", "E2E Latency p95 (s)", "lower"),
            ("ttft_median_ms", "TTFT p50 (ms)", "lower"),
            ("ttft_p95_ms", "TTFT p95 (ms)", "lower"),
            ("itl_median_ms", "ITL p50 (ms)", "lower"),
            ("itl_p95_ms", "ITL p95 (ms)", "lower"),
            ("tpot_median_ms", "TPOT p50 (ms)", "lower"),
            ("tpot_p95_ms", "TPOT p95 (ms)", "lower"),
            ("total_completed", "Completed reqs", "higher"),
            ("total_errored", "Errors", "lower"),
        ]
        for key, name, direction in metric_keys:
            a_val = agg_a.get(key)
            b_val = agg_b.get(key)
            diff = multiplier(a_val, b_val, direction)
            a_str = f"{a_val}" if a_val is not None else ""
            b_str = f"{b_val}" if b_val is not None else ""
            f.write(f"{name},{a_str},{b_str},{diff}\n")

    sys.stdout.write(f"{os.linesep}  CSV saved: {csv_path}{os.linesep}")
    sys.stdout.flush()


if __name__ == "__main__":
    main()
