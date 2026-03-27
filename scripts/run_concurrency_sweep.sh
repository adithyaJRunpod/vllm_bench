#!/usr/bin/env bash
set -euo pipefail

############################################
# Concurrency sweep for vLLM
# Assumes the vLLM server is already running
############################################

MODEL="Qwen/Qwen3-8B"
BASE_URL="http://127.0.0.1:8000"
API_KEY="${VLLM_API_KEY:-}"

REQUEST_RATE=30
NUM_PROMPTS=1024
INPUT_LEN=128
OUTPUT_LEN=128
TEMPERATURE=0

CONCURRENCY_VALUES=(8 16 24 32 48 64)

OUTDIR="logs/concurrency_sweep_$(date +%F_%H%M%S)"
mkdir -p "$OUTDIR"

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
    --temperature "$TEMPERATURE"
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