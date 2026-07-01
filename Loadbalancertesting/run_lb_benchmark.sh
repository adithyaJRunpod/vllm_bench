#!/usr/bin/env bash
set -euo pipefail

############################################
# Load Balancer Streaming Benchmark
# Single workload: 256 input tokens, 1024 output tokens (decode-heavy)
#
# This script mimics run_guidellm_workload_sweep.sh but uses a custom
# streaming benchmark against the RunPod Load Balancer /v1/completions
# endpoint, providing TTFT, ITL, and throughput metrics.
#
# Required env vars:
#   RUNPOD_API_KEY    RunPod API key (e.g. rpa_xxx)
#   RUNPOD_ENDPOINT   RunPod LB endpoint ID (e.g. jcja1rjzitd515)
#
# Optional overrides:
#   NUM_REQUESTS=50       Total requests to send
#   CONCURRENCY=8         Max parallel requests
#   INPUT_TOKENS=256      Approx input token count
#   OUTPUT_TOKENS=1024    Max output tokens
#   OUTPUT_DIR=results    Directory for JSON results
#
# Usage:
#   RUNPOD_API_KEY=rpa_xxx \
#   RUNPOD_ENDPOINT=jcja1rjzitd515 \
#   bash Loadbalancertesting/run_lb_benchmark.sh
############################################

RUNPOD_API_KEY="${RUNPOD_API_KEY:?Set RUNPOD_API_KEY}"
RUNPOD_ENDPOINT="${RUNPOD_ENDPOINT:?Set RUNPOD_ENDPOINT}"

NUM_REQUESTS="${NUM_REQUESTS:-50}"
CONCURRENCY="${CONCURRENCY:-8}"
INPUT_TOKENS="${INPUT_TOKENS:-256}"
OUTPUT_TOKENS="${OUTPUT_TOKENS:-1024}"
OUTPUT_DIR="${OUTPUT_DIR:-results}"

BASE_URL="https://${RUNPOD_ENDPOINT}.api.runpod.ai"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

echo "============================================================"
echo "  RunPod Load Balancer Benchmark Runner"
echo "============================================================"
echo "  Endpoint:      ${BASE_URL}"
echo "  Workload:      decode-heavy (${INPUT_TOKENS}in / ${OUTPUT_TOKENS}out)"
echo "  Requests:      ${NUM_REQUESTS}"
echo "  Concurrency:   ${CONCURRENCY}"
echo "  Output dir:    ${OUTPUT_DIR}"
echo "============================================================"
echo ""

# Ensure output directory exists
mkdir -p "${SCRIPT_DIR}/${OUTPUT_DIR}"

OUTPUT_FILE="${SCRIPT_DIR}/${OUTPUT_DIR}/lb_bench_${TIMESTAMP}.json"

# Check for httpx dependency
if ! python3 -c "import httpx" 2>/dev/null; then
  echo "[*] Installing httpx..."
  pip install httpx --quiet
fi

echo "[*] Starting benchmark: decode-heavy (${INPUT_TOKENS}in / ${OUTPUT_TOKENS}out)"
echo ""

python3 "${SCRIPT_DIR}/bench_lb.py" \
  --base-url "${BASE_URL}" \
  --api-key "${RUNPOD_API_KEY}" \
  --num-requests "${NUM_REQUESTS}" \
  --concurrency "${CONCURRENCY}" \
  --input-tokens "${INPUT_TOKENS}" \
  --output-tokens "${OUTPUT_TOKENS}" \
  --output-file "${OUTPUT_FILE}"

echo ""
echo "[*] Benchmark complete."
echo "[*] Results saved to: ${OUTPUT_FILE}"
echo ""
