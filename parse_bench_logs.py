import csv
import re
from pathlib import Path

log_root = Path("logs")
latest = sorted([p for p in log_root.iterdir() if p.is_dir() and p.name.startswith("concurrency_sweep_")])[-1]

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
    row["max_concurrency"] = int(experiment.replace("c", ""))

    for key, pattern in patterns.items():
        m = re.search(pattern, text)
        row[key] = float(m.group(1)) if m and "." in m.group(1) else (int(m.group(1)) if m else None)

    rows.append(row)

rows = sorted(rows, key=lambda x: x["max_concurrency"])

results_dir = Path("results")
results_dir.mkdir(exist_ok=True)
out_file = results_dir / f"{latest.name}.csv"

with out_file.open("w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    writer.writeheader()
    writer.writerows(rows)

print(f"Wrote {out_file}")