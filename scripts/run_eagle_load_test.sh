#!/usr/bin/env bash
set -euo pipefail

############################################
# Eagle Speculative Decoding Load Test
# Compares baseline vs Eagle across three
# load levels: low, medium, high concurrency.
# Uses ShareGPT for realistic workloads.
############################################

MODEL="${VLLM_MODEL:-meta-llama/Llama-3.1-8B-Instruct}"
EAGLE_HEAD="${VLLM_EAGLE_HEAD:-yuhuili/EAGLE-LLaMA3-Instruct-8B}"
SPEC_METHOD="${VLLM_SPEC_METHOD:-eagle3}"
NUM_SPEC_TOKENS="${VLLM_SPEC_TOKENS:-5}"
BASE_URL="http://127.0.0.1:8000"
API_KEY="${VLLM_API_KEY:-}"
SHAREGPT_PATH="${SHAREGPT_PATH:-/tmp/sharegpt.json}"

COMMON_SERVER_FLAGS="--host 0.0.0.0 --port 8000"

declare -A LOAD_RPS LOAD_CONC LOAD_PROMPTS LOAD_WARMUPS
LOAD_RPS=([low]=1   [medium]=5   [high]=10)
LOAD_CONC=([low]=1  [medium]=8   [high]=32)
LOAD_PROMPTS=([low]=30 [medium]=50 [high]=80)
LOAD_WARMUPS=([low]=3  [medium]=5  [high]=5)
LOAD_ORDER=(low medium high)

OUTDIR="logs/eagle_load_test/$(date +%F_%H%M%S)"
mkdir -p "$OUTDIR"

# ── helpers (reused from tuning sweep) ──────────────────

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
  echo "Starting vLLM server [$label]: vllm serve $MODEL $COMMON_SERVER_FLAGS $*"
  set +B
  # shellcheck disable=SC2086
  nohup vllm serve "$MODEL" $COMMON_SERVER_FLAGS "$@" \
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
  local label="$1" load="$2"
  local rps="${LOAD_RPS[$load]}"
  local conc="${LOAD_CONC[$load]}"
  local prompts="${LOAD_PROMPTS[$load]}"
  local warmups="${LOAD_WARMUPS[$load]}"
  local logfile="$OUTDIR/${label}_${load}_bench.log"

  echo "  [$label @ $load] RPS=$rps  concurrency=$conc  prompts=$prompts"

  local CMD=(vllm bench serve
    --backend openai-chat
    --base-url "$BASE_URL"
    --endpoint /v1/chat/completions
    --model "$MODEL"
    --dataset-name sharegpt
    --dataset-path "$SHAREGPT_PATH"
    --num-prompts "$prompts"
    --num-warmups "$warmups"
    --request-rate "$rps"
    --max-concurrency "$conc"
    --temperature 0
    --percentile-metrics ttft,itl,e2el
    --metric-percentiles 50,90,99
  )

  [[ -n "$API_KEY" ]] && CMD+=(--header "Authorization=Bearer $API_KEY")

  "${CMD[@]}" > "$logfile" 2>&1

  echo "  Done → $logfile"
}

# ── pre-flight checks ──────────────────────────────────

if [ ! -f "$SHAREGPT_PATH" ]; then
  echo "Downloading ShareGPT dataset..."
  wget -q -O "$SHAREGPT_PATH" \
    https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered/resolve/main/ShareGPT_V3_unfiltered_cleaned_split.json
fi

echo "=== Eagle Load Test ==="
{
  echo "timestamp: $(date -Iseconds)"
  echo "model: $MODEL"
  echo "eagle_head: $EAGLE_HEAD"
  echo "spec_method: $SPEC_METHOD"
  echo "num_speculative_tokens: $NUM_SPEC_TOKENS"
  echo "vllm_version: $(python3 -c 'import vllm; print(vllm.__version__)' 2>/dev/null || echo unknown)"
  echo "gpu: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo unknown)"
  echo "gpu_memory: $(nvidia-smi --query-gpu=memory.total --format=csv,noheader 2>/dev/null || echo unknown)"
  echo ""
  for load in "${LOAD_ORDER[@]}"; do
    echo "load_$load: RPS=${LOAD_RPS[$load]} conc=${LOAD_CONC[$load]} prompts=${LOAD_PROMPTS[$load]}"
  done
} | tee "$OUTDIR/environment.txt"
echo ""

# ── Phase 1: Baseline ──────────────────────────────────

echo "============================================"
echo "  PHASE 1 — BASELINE (no speculation)"
echo "============================================"

stop_server
start_server "baseline"

if ! wait_for_server; then
  echo "FATAL: baseline server failed to start."
  exit 1
fi

for load in "${LOAD_ORDER[@]}"; do
  run_bench "baseline" "$load"
  sleep 3
done

# ── Phase 2: Eagle ─────────────────────────────────────

echo ""
echo "============================================"
echo "  PHASE 2 — EAGLE (speculative decoding)"
echo "============================================"

stop_server
start_server "eagle" \
  --speculative-config "{\"model\":\"$EAGLE_HEAD\",\"method\":\"$SPEC_METHOD\",\"num_speculative_tokens\":$NUM_SPEC_TOKENS}"

if ! wait_for_server; then
  echo "FATAL: eagle server failed to start."
  exit 1
fi

for load in "${LOAD_ORDER[@]}"; do
  run_bench "eagle" "$load"
  sleep 3
done

stop_server

# ── Summary ─────────────────────────────────────────────

echo ""
echo "============================================"
echo "  RESULTS SUMMARY"
echo "============================================"
echo ""

for load in "${LOAD_ORDER[@]}"; do
  echo "--- $load load (RPS=${LOAD_RPS[$load]}, conc=${LOAD_CONC[$load]}) ---"
  for variant in baseline eagle; do
    logfile="$OUTDIR/${variant}_${load}_bench.log"
    if [ -f "$logfile" ]; then
      echo "  [$variant]"
      grep -E "(Successful|Benchmark duration|Total input|Total generated|Request throughput|Output token throughput|Total Token throughput|Mean TTFT|Median TTFT|P99 TTFT|Mean TPOT|Median TPOT|Mean ITL|Median ITL|P99 ITL)" "$logfile" | sed 's/^/    /'
    fi
  done
  echo ""
done

echo "Full logs: $OUTDIR/"
echo "Done."
