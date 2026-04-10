#!/usr/bin/env bash
set -euo pipefail

############################################
# Speculative Decoding A/B Test (ShareGPT)
# Compares baseline vs spec decode using real
# conversation data. Supports vanilla draft
# models and EAGLE heads.
#
# Usage:
#   VLLM_MODEL=Qwen/Qwen3-8B \
#   SPEC_MODEL=Qwen/Qwen3-0.6B \
#   bash scripts/run_spec_decode_sharegpt.sh
#
#   # With EAGLE:
#   VLLM_MODEL=Qwen/Qwen3-8B \
#   SPEC_MODEL=RedHatAI/Qwen3-8B-speculator.eagle3 \
#   SPEC_METHOD=eagle \
#   bash scripts/run_spec_decode_sharegpt.sh
############################################

MODEL="${VLLM_MODEL:?Set VLLM_MODEL (e.g. export VLLM_MODEL=Qwen/Qwen3-8B)}"
SPEC_MODEL="${SPEC_MODEL:?Set SPEC_MODEL (e.g. export SPEC_MODEL=Qwen/Qwen3-0.6B)}"
SPEC_METHOD="${SPEC_METHOD:-draft_model}"
SPEC_TOKENS="${SPEC_TOKENS:-3,5}"
NUM_RUNS="${NUM_RUNS:-2}"
NUM_PROMPTS="${VLLM_NUM_PROMPTS:-20}"
NUM_WARMUPS="${VLLM_NUM_WARMUPS:-3}"
MAX_CONCURRENCY="${VLLM_MAX_CONCURRENCY:-1}"
REQUEST_RATE="${VLLM_REQUEST_RATE:-1}"
TEMPERATURE="${VLLM_TEMPERATURE:-0}"

EXTRA_SERVER_FLAGS="${EXTRA_SERVER_FLAGS:-}"
COMMON_SERVER_FLAGS="--host 0.0.0.0 --port 8000"
BASE_URL="http://127.0.0.1:8000"
API_KEY="${VLLM_API_KEY:-}"

SHAREGPT_PATH="${SHAREGPT_PATH:-/tmp/sharegpt.json}"
SHAREGPT_URL="https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered/resolve/main/ShareGPT_V3_unfiltered_cleaned_split.json"

OUTDIR="logs/spec_decode_sharegpt/$(date +%F_%H%M%S)"
mkdir -p "$OUTDIR"

IFS=',' read -ra K_VALUES <<< "$SPEC_TOKENS"

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
  local label="$1"; shift
  echo "Starting vLLM server [$label]: vllm serve $MODEL $COMMON_SERVER_FLAGS $EXTRA_SERVER_FLAGS $*"
  set +B
  # shellcheck disable=SC2086
  nohup vllm serve "$MODEL" $COMMON_SERVER_FLAGS $EXTRA_SERVER_FLAGS "$@" \
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

  echo "  [$label] run $run_num/$NUM_RUNS  (prompts=$NUM_PROMPTS, conc=$MAX_CONCURRENCY, rps=$REQUEST_RATE, temp=$TEMPERATURE)"

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
  grep -oP "${pattern}\K[0-9.]+" "$file" 2>/dev/null | head -1 || echo "N/A"
}

# ── pre-flight ───────────────────────────────────────────

if [ ! -f "$SHAREGPT_PATH" ]; then
  echo "Downloading ShareGPT dataset..."
  wget -q -O "$SHAREGPT_PATH" "$SHAREGPT_URL"
fi

echo "=== Speculative Decoding A/B Test (ShareGPT) ==="
{
  echo "timestamp: $(date -Iseconds)"
  echo "model: $MODEL"
  echo "spec_model: $SPEC_MODEL"
  echo "spec_method: $SPEC_METHOD"
  echo "spec_tokens: $SPEC_TOKENS"
  echo "num_runs: $NUM_RUNS"
  echo "num_prompts: $NUM_PROMPTS"
  echo "max_concurrency: $MAX_CONCURRENCY"
  echo "request_rate: $REQUEST_RATE"
  echo "temperature: $TEMPERATURE"
  echo "extra_server_flags: $EXTRA_SERVER_FLAGS"
  echo "vllm_version: $(python3 -c 'import vllm; print(vllm.__version__)' 2>/dev/null || echo unknown)"
  echo "gpu: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo unknown)"
  echo "gpu_memory: $(nvidia-smi --query-gpu=memory.total --format=csv,noheader 2>/dev/null || echo unknown)"
} | tee "$OUTDIR/environment.txt"
echo ""

# ── Phase 1: Baseline ───────────────────────────────────

echo "============================================"
echo "  PHASE 1 — BASELINE (no speculation)"
echo "============================================"

stop_server
start_server "baseline"

if ! wait_for_server; then
  echo "FATAL: baseline server failed to start."
  exit 1
fi

for ((r=1; r<=NUM_RUNS; r++)); do
  run_bench "baseline" "$r"
  sleep 2
done

# ── Phase 2: Spec decode for each k ─────────────────────

for k in "${K_VALUES[@]}"; do
  label="${SPEC_METHOD}-k${k}"
  echo ""
  echo "============================================"
  echo "  PHASE 2 — ${SPEC_METHOD^^} k=$k"
  echo "============================================"

  stop_server
  start_server "$label" \
    --speculative-config "{\"model\":\"$SPEC_MODEL\",\"method\":\"$SPEC_METHOD\",\"num_speculative_tokens\":$k}"

  if ! wait_for_server; then
    echo "SKIPPING $label — server failed to start. Check $OUTDIR/${label}_server.log"
    continue
  fi

  for ((r=1; r<=NUM_RUNS; r++)); do
    run_bench "$label" "$r"
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
echo "config,run,output_tok_s,mean_tpot_ms,median_tpot_ms,p99_tpot_ms,mean_ttft_ms,median_ttft_ms,p99_ttft_ms,mean_itl_ms,p99_itl_ms" \
  | tee "$OUTDIR/summary.csv"

ALL_LABELS=("baseline")
for k in "${K_VALUES[@]}"; do
  ALL_LABELS+=("${SPEC_METHOD}-k${k}")
done

for label in "${ALL_LABELS[@]}"; do
  for ((r=1; r<=NUM_RUNS; r++)); do
    logfile="$OUTDIR/${label}_run${r}.log"
    if [ ! -f "$logfile" ]; then continue; fi

    otps=$(extract_metric "$logfile" "Output token throughput:\s+")
    tpot_mean=$(extract_metric "$logfile" "Mean TPOT \(ms\):\s+")
    tpot_med=$(extract_metric "$logfile" "Median TPOT \(ms\):\s+")
    tpot_p99=$(extract_metric "$logfile" "P99 TPOT \(ms\):\s+")
    ttft_mean=$(extract_metric "$logfile" "Mean TTFT \(ms\):\s+")
    ttft_med=$(extract_metric "$logfile" "Median TTFT \(ms\):\s+")
    ttft_p99=$(extract_metric "$logfile" "P99 TTFT \(ms\):\s+")
    itl_mean=$(extract_metric "$logfile" "Mean ITL \(ms\):\s+")
    itl_p99=$(extract_metric "$logfile" "P99 ITL \(ms\):\s+")

    echo "$label,$r,$otps,$tpot_mean,$tpot_med,$tpot_p99,$ttft_mean,$ttft_med,$ttft_p99,$itl_mean,$itl_p99" \
      | tee -a "$OUTDIR/summary.csv"
  done
done

echo ""
echo "Full logs: $OUTDIR/"

echo ""
echo "--- Detailed output per config ---"
for label in "${ALL_LABELS[@]}"; do
  echo ""
  echo "=== $label (run $NUM_RUNS — warm) ==="
  logfile="$OUTDIR/${label}_run${NUM_RUNS}.log"
  if [ -f "$logfile" ]; then
    grep -E "(Successful|Benchmark duration|Request throughput|Output token throughput|Mean TTFT|Median TTFT|P99 TTFT|Mean TPOT|Median TPOT|P99 TPOT|Mean ITL|P99 ITL)" "$logfile" | sed 's/^/  /'
  else
    echo "  (no log found)"
  fi
done

echo ""
echo "Done."
