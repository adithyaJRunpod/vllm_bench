#!/usr/bin/env bash
set -euo pipefail

############################################
# Config Comparison (ShareGPT)
# Head-to-head comparison of baseline, FP8,
# and EAGLE3+FP8 at production load.
#
# Usage:
#   VLLM_MODEL=Qwen/Qwen3-8B \
#   VLLM_REQUEST_RATE=80 \
#   VLLM_MAX_CONCURRENCY=96 \
#   bash scripts/run_config_compare.sh
############################################

MODEL="${VLLM_MODEL:?Set VLLM_MODEL (e.g. export VLLM_MODEL=Qwen/Qwen3-8B)}"
EAGLE_MODEL="${EAGLE_MODEL:-RedHatAI/Qwen3-8B-speculator.eagle3}"
EAGLE_METHOD="${EAGLE_METHOD:-eagle3}"
EAGLE_K="${EAGLE_K:-3}"

NUM_RUNS="${NUM_RUNS:-2}"
NUM_PROMPTS="${VLLM_NUM_PROMPTS:-50}"
NUM_WARMUPS="${VLLM_NUM_WARMUPS:-5}"
MAX_CONCURRENCY="${VLLM_MAX_CONCURRENCY:-96}"
REQUEST_RATE="${VLLM_REQUEST_RATE:-80}"
TEMPERATURE="${VLLM_TEMPERATURE:-0}"

COMMON_SERVER_FLAGS="--host 0.0.0.0 --port 8000"
BASE_URL="http://127.0.0.1:8000"
API_KEY="${VLLM_API_KEY:-}"

SHAREGPT_PATH="${SHAREGPT_PATH:-/tmp/sharegpt.json}"
SHAREGPT_URL="https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered/resolve/main/ShareGPT_V3_unfiltered_cleaned_split.json"

OUTDIR="logs/config_compare/$(date +%F_%H%M%S)"
mkdir -p "$OUTDIR"

declare -a CONFIG_ORDER
declare -A CONFIG_FLAGS

CONFIG_ORDER=(baseline fp8 fp8-eagle3)
CONFIG_FLAGS=(
  [baseline]=""
  [fp8]="--quantization fp8 --kv-cache-dtype fp8"
  [fp8-eagle3]="--quantization fp8 --kv-cache-dtype fp8 --speculative-config {\"model\":\"$EAGLE_MODEL\",\"method\":\"$EAGLE_METHOD\",\"num_speculative_tokens\":$EAGLE_K}"
)

# ── helpers ──────────────────────────────────────────────

stop_server() {
  echo "Stopping vLLM server..."
  pkill -f "vllm serve" 2>/dev/null || true
  sleep 5
  if pgrep -f "vllm serve" >/dev/null 2>&1; then
    echo "Force-killing vLLM..."
    pkill -9 -f "vllm serve" 2>/dev/null || true
    sleep 3
  fi
}

start_server() {
  local label="$1"
  local flags="$2"
  echo "Starting vLLM server [$label]: vllm serve $MODEL $COMMON_SERVER_FLAGS $flags"
  set +B
  # shellcheck disable=SC2086
  nohup vllm serve "$MODEL" $COMMON_SERVER_FLAGS $flags \
    > "$OUTDIR/${label}_server.log" 2>&1 &
  set -B
  echo "Server PID: $!"
}

wait_for_server() {
  local max_wait=360 elapsed=0
  echo "Waiting for server to be ready (max ${max_wait}s)..."
  while [ $elapsed -lt $max_wait ]; do
    if curl -s --max-time 2 "$BASE_URL/v1/models" >/dev/null 2>&1; then
      echo "Server ready after ${elapsed}s."
      return 0
    fi
    sleep 5
    elapsed=$((elapsed + 5))
    printf "  %ds...\r" "$elapsed"
  done
  echo "ERROR: Server did not start within ${max_wait}s."
  return 1
}

run_bench() {
  local label="$1" run_num="$2"
  local logfile="$OUTDIR/${label}_run${run_num}.log"

  echo "  [$label] run $run_num/$NUM_RUNS  (prompts=$NUM_PROMPTS, conc=$MAX_CONCURRENCY, rps=$REQUEST_RATE)"

  local CMD=(vllm bench serve
    --backend openai-chat
    --base-url "$BASE_URL"
    --endpoint /v1/chat/completions
    --model "$MODEL"
    --dataset-name sharegpt
    --dataset-path "$SHAREGPT_PATH"
    --num-prompts "$NUM_PROMPTS"
    --num-warmups "$NUM_WARMUPS"
    --request-rate "$REQUEST_RATE"
    --max-concurrency "$MAX_CONCURRENCY"
    --temperature "$TEMPERATURE"
    --percentile-metrics ttft,itl,e2el
    --metric-percentiles 50,90,99
  )

  [[ -n "$API_KEY" ]] && CMD+=(--header "Authorization=Bearer $API_KEY")

  "${CMD[@]}" > "$logfile" 2>&1
  echo "  Done → $logfile"
}

extract_metric() {
  local file="$1" pattern="$2"
  grep -oP "${pattern}\s*\K[0-9.]+" "$file" 2>/dev/null | head -1 || echo "N/A"
}

# ── pre-flight ───────────────────────────────────────────

if [ ! -f "$SHAREGPT_PATH" ]; then
  echo "Downloading ShareGPT dataset..."
  wget -q -O "$SHAREGPT_PATH" "$SHAREGPT_URL"
fi

TOTAL=${#CONFIG_ORDER[@]}
echo "=== Config Comparison (ShareGPT) ==="
{
  echo "timestamp: $(date -Iseconds)"
  echo "model: $MODEL"
  echo "eagle_model: $EAGLE_MODEL"
  echo "eagle_method: $EAGLE_METHOD"
  echo "eagle_k: $EAGLE_K"
  echo "num_runs: $NUM_RUNS"
  echo "num_prompts: $NUM_PROMPTS"
  echo "max_concurrency: $MAX_CONCURRENCY"
  echo "request_rate: $REQUEST_RATE"
  echo "temperature: $TEMPERATURE"
  echo "vllm_version: $(python3 -c 'import vllm; print(vllm.__version__)' 2>/dev/null || echo unknown)"
  echo "gpu: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo unknown)"
  echo "gpu_memory: $(nvidia-smi --query-gpu=memory.total --format=csv,noheader 2>/dev/null || echo unknown)"
  echo ""
  for name in "${CONFIG_ORDER[@]}"; do
    echo "config [$name]: vllm serve $MODEL $COMMON_SERVER_FLAGS ${CONFIG_FLAGS[$name]}"
  done
} | tee "$OUTDIR/environment.txt"
echo ""

# ── Run each config ──────────────────────────────────────

RUN=0
for name in "${CONFIG_ORDER[@]}"; do
  RUN=$((RUN + 1))
  flags="${CONFIG_FLAGS[$name]}"

  echo ""
  echo "============================================"
  echo "  [$RUN/$TOTAL] $name"
  echo "  Flags: ${flags:-(none)}"
  echo "============================================"

  stop_server
  start_server "$name" "$flags"

  if ! wait_for_server; then
    echo "SKIPPING $name — server failed to start. Check $OUTDIR/${name}_server.log"
    continue
  fi

  for ((r=1; r<=NUM_RUNS; r++)); do
    run_bench "$name" "$r"
    sleep 2
  done
done

stop_server

# ── Summary ──────────────────────────────────────────────

echo ""
echo "============================================"
echo "  RESULTS SUMMARY"
echo "============================================"
echo ""

CSV_HEADER="config,run,req_s,output_tok_s,mean_ttft_ms,median_ttft_ms,p99_ttft_ms,mean_tpot_ms,median_tpot_ms,p99_tpot_ms,mean_itl_ms,p99_itl_ms,mean_e2el_ms,p99_e2el_ms"
echo "$CSV_HEADER" | tee "$OUTDIR/summary.csv"

for name in "${CONFIG_ORDER[@]}"; do
  for ((r=1; r<=NUM_RUNS; r++)); do
    logfile="$OUTDIR/${name}_run${r}.log"
    if [ ! -f "$logfile" ]; then continue; fi

    req_s=$(extract_metric "$logfile" "Request throughput \(req/s\):")
    otps=$(extract_metric "$logfile" "Output token throughput \(tok/s\):")
    ttft_mean=$(extract_metric "$logfile" "Mean TTFT \(ms\):")
    ttft_med=$(extract_metric "$logfile" "Median TTFT \(ms\):")
    ttft_p99=$(extract_metric "$logfile" "P99 TTFT \(ms\):")
    tpot_mean=$(extract_metric "$logfile" "Mean TPOT \(ms\):")
    tpot_med=$(extract_metric "$logfile" "Median TPOT \(ms\):")
    tpot_p99=$(extract_metric "$logfile" "P99 TPOT \(ms\):")
    itl_mean=$(extract_metric "$logfile" "Mean ITL \(ms\):")
    itl_p99=$(extract_metric "$logfile" "P99 ITL \(ms\):")
    e2el_mean=$(extract_metric "$logfile" "Mean E2EL \(ms\):")
    e2el_p99=$(extract_metric "$logfile" "P99 E2EL \(ms\):")

    echo "$name,$r,$req_s,$otps,$ttft_mean,$ttft_med,$ttft_p99,$tpot_mean,$tpot_med,$tpot_p99,$itl_mean,$itl_p99,$e2el_mean,$e2el_p99" \
      | tee -a "$OUTDIR/summary.csv"
  done
done

echo ""
echo "--- Detailed output per config (warm run) ---"
for name in "${CONFIG_ORDER[@]}"; do
  echo ""
  echo "=== $name (run $NUM_RUNS) ==="
  logfile="$OUTDIR/${name}_run${NUM_RUNS}.log"
  if [ -f "$logfile" ]; then
    grep -E "(Successful|Benchmark duration|Request throughput|Output token throughput|Total Token throughput|Mean TTFT|Median TTFT|P99 TTFT|Mean TPOT|Median TPOT|P99 TPOT|Mean ITL|Median ITL|P99 ITL|Mean E2EL|P99 E2EL)" "$logfile" | sed 's/^/  /'
  else
    echo "  (no log found)"
  fi
done

echo ""
echo "Full logs: $OUTDIR/"
echo "Done."
