#!/usr/bin/env bash
set -euo pipefail

############################################
# Concurrency sweep for vLLM
# Assumes the vLLM server is already running
############################################

MODEL="${VLLM_MODEL:?Set VLLM_MODEL before running (e.g. export VLLM_MODEL=Qwen/Qwen3-8B)}"
BASE_URL="http://127.0.0.1:8000"
API_KEY="${VLLM_API_KEY:-}"

REQUEST_RATE="${VLLM_REQUEST_RATE:?Set VLLM_REQUEST_RATE (e.g. export VLLM_REQUEST_RATE=30)}"
INPUT_LEN="${VLLM_INPUT_LEN:?Set VLLM_INPUT_LEN (e.g. export VLLM_INPUT_LEN=128)}"
OUTPUT_LEN="${VLLM_OUTPUT_LEN:?Set VLLM_OUTPUT_LEN (e.g. export VLLM_OUTPUT_LEN=128)}"
NUM_PROMPTS=1024

CONCURRENCY_VALUES=(4 8 16 24 32 48 64 96 128 150 180 200 256)

OUTDIR="logs/concurrency_sweep/$(date +%F_%H%M%S)"
mkdir -p "$OUTDIR"

echo "--- Server check (logging what is currently running) ---"
{
  echo "timestamp: $(date -Iseconds)"
  echo "server_cmd: $(ps -eo args= | grep '[v]llm serve' || echo 'NO vLLM SERVER RUNNING')"
  echo "input_len: $INPUT_LEN"
  echo "output_len: $OUTPUT_LEN"
  echo "request_rate: $REQUEST_RATE"
} | tee "$OUTDIR/environment.txt"
echo ""

TOTAL=${#CONCURRENCY_VALUES[@]}
echo "Output directory: $OUTDIR"
echo "Running $TOTAL concurrency levels: ${CONCURRENCY_VALUES[*]}"

RUN=0
for c in "${CONCURRENCY_VALUES[@]}"; do
  RUN=$((RUN + 1))
  echo ""
  echo "=========================================="
  echo "[$RUN/$TOTAL] Running benchmark: max_concurrency=$c"
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
    --max-concurrency "$c"
    --request-rate "$REQUEST_RATE"
    --percentile-metrics ttft,itl,e2el
    --metric-percentiles 50,90,99
  )

  [[ -n "$API_KEY" ]] && CMD+=(--header "Authorization=Bearer $API_KEY")

  "${CMD[@]}" > "$OUTDIR/c${c}_bench.log" 2>&1
  echo "Done. Log: $OUTDIR/c${c}_bench.log"
  sleep 3
done

echo ""
echo "All done. Logs in: $OUTDIR"
echo "Run 'python parse_bench_logs.py' to extract the summary CSV."