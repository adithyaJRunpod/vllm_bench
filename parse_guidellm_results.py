#!/usr/bin/env python3
"""
Parse GuideLLM benchmark JSON files and produce a summary CSV + console table.

Usage:
    python3 parse_guidellm_results.py <log_directory>

Expects files named like: <config>_run<N>.json
Example: baseline_run1.json, fp8_run2.json, fp8-eagle3_run1.json
"""

import json
import csv
import sys
import re
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

    return {
        "req_per_sec": safe_get("requests_per_second", "mean"),
        "concurrency_median": safe_get("request_concurrency", "median"),
        "input_tokens_median": safe_get("prompt_token_count", "median"),
        "output_tokens_median": safe_get("output_token_count", "median"),
        "output_tok_per_sec_median": safe_get("output_tokens_per_second", "median"),
        "output_tok_per_sec_mean": safe_get("output_tokens_per_second", "mean"),
        "total_tok_per_sec_median": safe_get("tokens_per_second", "median"),
        "ttft_median_ms": safe_get("time_to_first_token_ms", "median"),
        "ttft_mean_ms": safe_get("time_to_first_token_ms", "mean"),
        "ttft_p95_ms": safe_percentile("time_to_first_token_ms", 95),
        "ttft_p99_ms": safe_percentile("time_to_first_token_ms", 99),
        "itl_median_ms": safe_get("inter_token_latency_ms", "median"),
        "itl_mean_ms": safe_get("inter_token_latency_ms", "mean"),
        "itl_p95_ms": safe_percentile("inter_token_latency_ms", 95),
        "itl_p99_ms": safe_percentile("inter_token_latency_ms", 99),
        "tpot_median_ms": safe_get("time_per_output_token_ms", "median"),
        "tpot_mean_ms": safe_get("time_per_output_token_ms", "mean"),
        "tpot_p95_ms": safe_percentile("time_per_output_token_ms", 95),
        "latency_median_sec": safe_get("request_latency", "median"),
        "latency_mean_sec": safe_get("request_latency", "mean"),
        "latency_p95_sec": safe_percentile("request_latency", 95),
        "total_requests": m.get("request_totals", {}).get("successful")
              or m.get("request_totals", {}).get("total"),
        "duration_sec": b.get("duration"),
    }


def fmt_num(val, decimals=0):
    """Format a number with comma separators. Returns 'N/A' for None."""
    if val is None:
        return "N/A"
    if isinstance(val, float):
        if decimals == 0:
            return f"{val:,.0f}"
        return f"{val:,.{decimals}f}"
    return f"{val:,}"


def main():
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} <log_directory>", file=sys.stderr)
        sys.exit(1)

    log_dir = Path(sys.argv[1])
    if not log_dir.is_dir():
        print(f"Error: {log_dir} is not a directory", file=sys.stderr)
        sys.exit(1)

    pattern = re.compile(r"^(.+)_run(\d+)\.json$")
    results = []

    def natural_sort_key(path):
        return [int(t) if t.isdigit() else t.lower()
                for t in re.split(r'(\d+)', path.name)]

    for json_file in sorted(log_dir.glob("*_run*.json"), key=natural_sort_key):
        match = pattern.match(json_file.name)
        if not match:
            continue
        config_name = match.group(1)
        run_num = int(match.group(2))

        try:
            metrics = extract_metrics(str(json_file))
        except Exception as e:
            print(f"Warning: Failed to parse {json_file}: {e}", file=sys.stderr)
            continue

        results.append({"config": config_name, "run": run_num, **metrics})

    if not results:
        print("No results found.", file=sys.stderr)
        sys.exit(1)

    # --- CSV with all detailed fields ---
    csv_path = log_dir / "summary.csv"
    fieldnames = list(results[0].keys())
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)
    print(f"CSV saved to: {csv_path}")
    print()

    # --- Clean console table ---
    # Throughput: mean (aggregate server rate)
    # Latency: median (typical per-request experience)
    col_config = 24
    col = 12

    header = (
        f"{'Config':<{col_config}}"
        f"{'Req/s':>{col}}"
        f"{'Out tok/s':>{col}}"
        f"{'TTFT (ms)':>{col}}"
        f"{'ITL (ms)':>{col}}"
        f"{'E2EL (ms)':>{col}}"
    )
    separator = "-" * len(header)
    print(separator)
    print(header)
    print(separator)

    for r in results:
        e2el_ms = r["latency_median_sec"] * 1000 if r["latency_median_sec"] else None
        print(
            f"{r['config']:<{col_config}}"
            f"{fmt_num(r['req_per_sec'], 1):>{col}}"
            f"{fmt_num(r['output_tok_per_sec_mean']):>{col}}"
            f"{fmt_num(r['ttft_median_ms']):>{col}}"
            f"{fmt_num(r['itl_median_ms']):>{col}}"
            f"{fmt_num(e2el_ms):>{col}}"
        )

    print(separator)
    print()

    # --- Per-config averages (when multiple runs per config) ---
    configs_seen = []
    for r in results:
        if r["config"] not in configs_seen:
            configs_seen.append(r["config"])

    if len(configs_seen) > 1:
        print("--- Per-config averages (across runs) ---")
        print()

        print(separator)
        print(header)
        print(separator)

        for config in configs_seen:
            runs = [r for r in results if r["config"] == config]

            def avg(key):
                vals = [r[key] for r in runs if r[key] is not None]
                return sum(vals) / len(vals) if vals else None

            e2el_ms = avg("latency_median_sec")
            if e2el_ms is not None:
                e2el_ms *= 1000

            print(
                f"{config:<{col_config}}"
                f"{fmt_num(avg('req_per_sec'), 1):>{col}}"
                f"{fmt_num(avg('output_tok_per_sec_mean')):>{col}}"
                f"{fmt_num(avg('ttft_median_ms')):>{col}}"
                f"{fmt_num(avg('itl_median_ms')):>{col}}"
                f"{fmt_num(e2el_ms):>{col}}"
            )

        print(separator)


if __name__ == "__main__":
    main()
