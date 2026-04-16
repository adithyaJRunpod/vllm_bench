#!/usr/bin/env bash
set -euo pipefail

############################################
# Concurrency Sweep using GuideLLM
# Runs a single baseline vLLM server and
# benchmarks at multiple concurrency levels
# to find the optimal operating point.
#
# Usage:
#   VLLM_MODEL=Qwen/Qwen3-8B \
#   bash scripts/run_guidellm_c_sweep.sh
############################################

MODEL="${VLLM_MODEL:?Set VLLM_MODEL (e.g. export VLLM_MODEL=Qwen/Qwen3-8B)}"

CONCURRENCY_LEVELS="${GUIDELLM_C_LEVELS:-16,32,64,96,128}"
INPUT_TOKENS="${GUIDELLM_INPUT_TOKENS:-550}"
INPUT_STDEV="${GUIDELLM_INPUT_STDEV:-55}"
OUTPUT_TOKENS="${GUIDELLM_OUTPUT_TOKENS:-150}"
OUTPUT_STDEV="${GUIDELLM_OUTPUT_STDEV:-15}"
TEXT_SOURCE="${GUIDELLM_TEXT_SOURCE:-data:prideandprejudice.txt.gz}"
MAX_REQUESTS="${GUIDELLM_MAX_REQUESTS:-200}"
RANDOM_SEED="${GUIDELLM_SEED:-42}"

DTYPE="${VLLM_DTYPE:-auto}"
GPU_UTIL="${VLLM_GPU_UTIL:-0.95}"
MAX_MODEL_LEN="${VLLM_MAX_MODEL_LEN:-8192}"
EXTRA_SERVER_FLAGS="${VLLM_EXTRA_FLAGS:---no-enable-prefix-caching}"

COMMON_SERVER_FLAGS="--host 0.0.0.0 --port 8000 --dtype $DTYPE --gpu-memory-utilization $GPU_UTIL --max-model-len $MAX_MODEL_LEN $EXTRA_SERVER_FLAGS"
BASE_URL="http://localhost:8000"

DATA_CFG="prompt_tokens=$INPUT_TOKENS,output_tokens=$OUTPUT_TOKENS,source=$TEXT_SOURCE"
[[ "$INPUT_STDEV" -gt 0 ]] 2>/dev/null && DATA_CFG="prompt_tokens=$INPUT_TOKENS,prompt_tokens_stdev=$INPUT_STDEV,output_tokens=$OUTPUT_TOKENS,output_tokens_stdev=$OUTPUT_STDEV,source=$TEXT_SOURCE"

OUTDIR="logs/guidellm_c_sweep/$(date +%F_%H%M%S)"
mkdir -p "$OUTDIR"

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
  local extra_flags="${2:-}"
  echo "Starting vLLM server [$label]: vllm serve $MODEL $COMMON_SERVER_FLAGS $extra_flags"
  set +B
  # shellcheck disable=SC2086
  nohup vllm serve "$MODEL" $COMMON_SERVER_FLAGS $extra_flags \
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

warmup_server() {
  echo "Warming up server (200 requests at concurrency 64)..."
  guidellm benchmark run \
    --target "$BASE_URL" \
    --model "$MODEL" \
    --data "$DATA_CFG" \
    --profile concurrent \
    --rate 64 \
    --max-requests 200 \
    --random-seed "$RANDOM_SEED" \
    --disable-console \
    --outputs /dev/null 2>/dev/null || true
  echo "Warmup done."
}

# ── pre-flight ───────────────────────────────────────────

IFS=',' read -ra C_LEVELS <<< "$CONCURRENCY_LEVELS"
TOTAL=${#C_LEVELS[@]}

echo "=== GuideLLM Concurrency Sweep ==="
{
  echo "timestamp: $(date -Iseconds)"
  echo "model: $MODEL"
  echo "config: baseline (no quantization, no speculation)"
  echo "server_cmd: vllm serve $MODEL $COMMON_SERVER_FLAGS"
  echo "concurrency_levels: $CONCURRENCY_LEVELS"
  echo "input_tokens: $INPUT_TOKENS (stdev=$INPUT_STDEV)"
  echo "output_tokens: $OUTPUT_TOKENS (stdev=$OUTPUT_STDEV)"
  echo "text_source: $TEXT_SOURCE"
  echo "data_config: $DATA_CFG"
  echo "max_requests: $MAX_REQUESTS"
  echo "random_seed: $RANDOM_SEED"
  echo "guidellm_version: $(python3 -c 'import guidellm; print(guidellm.__version__)' 2>/dev/null || echo unknown)"
  echo "vllm_version: $(python3 -c 'import vllm; print(vllm.__version__)' 2>/dev/null || echo unknown)"
  echo "gpu: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo unknown)"
  echo "gpu_memory: $(nvidia-smi --query-gpu=memory.total --format=csv,noheader 2>/dev/null || echo unknown)"
} | tee "$OUTDIR/environment.txt"
echo ""

# ── stop anything running, start baseline server ────────

echo "Killing any existing vLLM server..."
stop_server

echo "Starting baseline vLLM server (no quantization, no speculation)..."
start_server "baseline" ""

if ! wait_for_server; then
  echo "FATAL: Baseline server failed to start. Check $OUTDIR/baseline_server.log"
  exit 1
fi

warmup_server

# ── sweep concurrency levels ─────────────────────────────

RUN=0
for C in "${C_LEVELS[@]}"; do
  RUN=$((RUN + 1))
  echo ""
  echo "============================================"
  echo "  [$RUN/$TOTAL] Concurrency = $C"
  echo "============================================"

  json_out="$OUTDIR/c${C}_run1.json"
  console_out="$OUTDIR/c${C}_run1.log"

  echo "  Running: input=$INPUT_TOKENS, output=$OUTPUT_TOKENS, conc=$C, requests=$MAX_REQUESTS"

  guidellm benchmark run \
    --target "$BASE_URL" \
    --model "$MODEL" \
    --data "$DATA_CFG" \
    --profile concurrent \
    --rate "$C" \
    --max-requests "$MAX_REQUESTS" \
    --random-seed "$RANDOM_SEED" \
    --outputs "$json_out" \
    > "$console_out" 2>&1

  echo "  Done → $json_out"
  sleep 3
done

stop_server

# ── Summary ──────────────────────────────────────────────

echo ""
echo "============================================"
echo "  C-SWEEP RESULTS"
echo "============================================"
echo ""

python3 "$(dirname "$0")/../parse_guidellm_results.py" "$OUTDIR"

echo ""
echo "Full logs: $OUTDIR/"
echo "Done. Pick the concurrency level with the best throughput + acceptable latency."
