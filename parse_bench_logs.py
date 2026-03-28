import csv
import re
import sys
from pathlib import Path

log_root = Path("logs")
sweep_type = sys.argv[1] if len(sys.argv) > 1 else None

if sweep_type not in ("rps", "concurrency"):
    for candidate in ("rps_sweep", "concurrency_sweep"):
        candidate_dir = log_root / candidate
        if candidate_dir.is_dir() and any(candidate_dir.iterdir()):
            sweep_type = "rps" if candidate == "rps_sweep" else "concurrency"
    if sweep_type is None:
        print("Usage: python3 parse_bench_logs.py [rps|concurrency]")
        print("No sweep results found in logs/")
        sys.exit(1)

sweep_dir = log_root / ("rps_sweep" if sweep_type == "rps" else "concurrency_sweep")
runs = sorted([p for p in sweep_dir.iterdir() if p.is_dir()])
if not runs:
    print(f"No runs found in {sweep_dir}/")
    sys.exit(1)

latest = runs[-1]

patterns = {
    "failed_requests": r"Failed requests:\s+([0-9]+)",
    "request_throughput": r"Request throughput \(req/s\):\s+([0-9.]+)",
    "total_token_throughput": r"Total token throughput \(tok/s\):\s+([0-9.]+)",
    "mean_ttft_ms": r"Mean TTFT \(ms\):\s+([0-9.]+)",
    "p50_ttft_ms": r"P50 TTFT \(ms\):\s+([0-9.]+)",
    "p90_ttft_ms": r"P90 TTFT \(ms\):\s+([0-9.]+)",
    "mean_e2el_ms": r"Mean E2EL \(ms\):\s+([0-9.]+)",
    "p50_e2el_ms": r"P50 E2EL \(ms\):\s+([0-9.]+)",
    "p90_e2el_ms": r"P90 E2EL \(ms\):\s+([0-9.]+)",
}

rows = []
for bench_file in sorted(latest.glob("*_bench.log")):
    text = bench_file.read_text(errors="ignore")
    experiment = bench_file.name.replace("_bench.log", "")
    row = {"experiment": experiment}

    if sweep_type == "rps":
        row["request_rate"] = int(experiment.replace("rps", ""))
        sort_key = "request_rate"
    else:
        row["max_concurrency"] = int(experiment.replace("c", ""))
        sort_key = "max_concurrency"

    for key, pattern in patterns.items():
        m = re.search(pattern, text)
        row[key] = float(m.group(1)) if m and "." in m.group(1) else (int(m.group(1)) if m else None)

    rows.append(row)

rows = sorted(rows, key=lambda x: x[sort_key])

results_dir = Path("results") / sweep_dir.name
results_dir.mkdir(parents=True, exist_ok=True)
out_file = results_dir / f"{latest.name}.csv"

with out_file.open("w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    writer.writeheader()
    writer.writerows(rows)

print(f"Wrote {out_file}")