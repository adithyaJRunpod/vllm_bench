#!/usr/bin/env bash
set -euo pipefail

############################################
# RPS sweep for vLLM
# Finds throughput ceiling and latency knee
# Assumes the vLLM server is already running
############################################

MODEL="${VLLM_MODEL:?Set VLLM_MODEL before running (e.g. export VLLM_MODEL=Qwen/Qwen3-8B)}"
BASE_URL="http://127.0.0.1:8000"
API_KEY="${VLLM_API_KEY:-}"

MAX_CONCURRENCY="${VLLM_MAX_CONCURRENCY:?Set VLLM_MAX_CONCURRENCY (e.g. export VLLM_MAX_CONCURRENCY=256)}"
INPUT_LEN="${VLLM_INPUT_LEN:?Set VLLM_INPUT_LEN (e.g. export VLLM_INPUT_LEN=128)}"
OUTPUT_LEN="${VLLM_OUTPUT_LEN:?Set VLLM_OUTPUT_LEN (e.g. export VLLM_OUTPUT_LEN=128)}"
NUM_PROMPTS=1024

RPS_VALUES=(5 10 15 20 25 30 35 40 50 60 80 100 120 150 200 250 300 400 500)

OUTDIR="logs/rps_sweep/$(date +%F_%H%M%S)"
mkdir -p "$OUTDIR"

echo "--- Server check (logging what is currently running) ---"
{
  echo "timestamp: $(date -Iseconds)"
  echo "server_cmd: $(ps -eo args= | grep '[v]llm serve' || echo 'NO vLLM SERVER RUNNING')"
  echo "input_len: $INPUT_LEN"
  echo "output_len: $OUTPUT_LEN"
  echo "max_concurrency: $MAX_CONCURRENCY"
} | tee "$OUTDIR/environment.txt"
echo ""

TOTAL=${#RPS_VALUES[@]}
echo "Output directory: $OUTDIR"
echo "Running $TOTAL RPS levels: ${RPS_VALUES[*]}"
echo "Fixed max_concurrency=$MAX_CONCURRENCY"

RUN=0
for rps in "${RPS_VALUES[@]}"; do
  RUN=$((RUN + 1))
  echo ""
  echo "=========================================="
  echo "[$RUN/$TOTAL] Running benchmark: request_rate=$rps"
  echo "=========================================="

  CMD=(vllm bench serve
    --backend openai-chat
    --base-url "$BASE_URL"
    --endpoint /v1/chat/completions
    --model "$MODEL"
    --dataset-name random
    --random-input-len "$INPUT_LEN"
    --random-output-len "$OUTPUT_LEN"
    --num-prompts "$NUM_PROMPTS"
    --max-concurrency "$MAX_CONCURRENCY"
    --request-rate "$rps"
    --percentile-metrics ttft,itl,e2el
    --metric-percentiles 50,90,99
  )

  [[ -n "$API_KEY" ]] && CMD+=(--header "Authorization=Bearer $API_KEY")

  "${CMD[@]}" > "$OUTDIR/rps${rps}_bench.log" 2>&1
  echo "Done. Log: $OUTDIR/rps${rps}_bench.log"
  sleep 3
done

echo ""
echo "All done. Logs in: $OUTDIR"
echo "Run 'python3 parse_bench_logs.py' to extract the summary CSV."
