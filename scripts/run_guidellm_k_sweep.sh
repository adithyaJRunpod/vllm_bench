#!/usr/bin/env bash
set -euo pipefail

############################################
# K-Sweep: Baseline vs FP8 vs FP8+EAGLE3
# at different speculative token counts (K).
#
# Uses realistic token length variation
# (stdev enabled by default).
#
# Usage:
#   VLLM_MODEL=Qwen/Qwen3-8B \
#   GUIDELLM_CONCURRENCY=128 \
#   bash scripts/run_guidellm_k_sweep.sh
############################################

MODEL="${VLLM_MODEL:?Set VLLM_MODEL (e.g. export VLLM_MODEL=Qwen/Qwen3-8B)}"

EAGLE_MODEL="${EAGLE_MODEL:-RedHatAI/Qwen3-8B-speculator.eagle3}"
EAGLE_METHOD="${EAGLE_METHOD:-eagle3}"
K_VALUES="${K_VALUES:-1,2,3,5,7}"

CONCURRENCY="${GUIDELLM_CONCURRENCY:?Set GUIDELLM_CONCURRENCY from C-sweep results}"
INPUT_TOKENS="${GUIDELLM_INPUT_TOKENS:-550}"
INPUT_STDEV="${GUIDELLM_INPUT_STDEV:-55}"
OUTPUT_TOKENS="${GUIDELLM_OUTPUT_TOKENS:-150}"
OUTPUT_STDEV="${GUIDELLM_OUTPUT_STDEV:-15}"
TEXT_SOURCE="${GUIDELLM_TEXT_SOURCE:-data:prideandprejudice.txt.gz}"
MAX_REQUESTS="${GUIDELLM_MAX_REQUESTS:-500}"
RANDOM_SEED="${GUIDELLM_SEED:-42}"
NUM_RUNS="${NUM_RUNS:-2}"

DTYPE="${VLLM_DTYPE:-auto}"
GPU_UTIL="${VLLM_GPU_UTIL:-0.95}"
MAX_MODEL_LEN="${VLLM_MAX_MODEL_LEN:-8192}"

COMMON_SERVER_FLAGS="--host 0.0.0.0 --port 8000 --dtype $DTYPE --gpu-memory-utilization $GPU_UTIL --max-model-len $MAX_MODEL_LEN"
BASE_URL="http://localhost:8000"

DATA_CFG="prompt_tokens=$INPUT_TOKENS,output_tokens=$OUTPUT_TOKENS,source=$TEXT_SOURCE"
[[ "$INPUT_STDEV" -gt 0 ]] 2>/dev/null && DATA_CFG="prompt_tokens=$INPUT_TOKENS,prompt_tokens_stdev=$INPUT_STDEV,output_tokens=$OUTPUT_TOKENS,output_tokens_stdev=$OUTPUT_STDEV,source=$TEXT_SOURCE"

OUTDIR="logs/guidellm_k_sweep/$(date +%F_%H%M%S)"
mkdir -p "$OUTDIR"

# ── build config matrix ──────────────────────────────────

declare -a CONFIG_ORDER
declare -A CONFIG_FLAGS

CONFIG_ORDER=(baseline fp8-full)
CONFIG_FLAGS=(
  [baseline]=""
  [fp8-full]="--quantization fp8 --kv-cache-dtype fp8"
)

IFS=',' read -ra K_LEVELS <<< "$K_VALUES"
for K in "${K_LEVELS[@]}"; do
  name="fp8-eagle3-k${K}"
  CONFIG_ORDER+=("$name")
  CONFIG_FLAGS[$name]="--quantization fp8 --kv-cache-dtype fp8 --speculative-config {\"model\":\"$EAGLE_MODEL\",\"method\":\"$EAGLE_METHOD\",\"num_speculative_tokens\":$K}"
done

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
  echo "Warming up server (10 requests at concurrency 2)..."
  guidellm benchmark run \
    --target "$BASE_URL" \
    --model "$MODEL" \
    --data "$DATA_CFG" \
    --profile concurrent \
    --rate 2 \
    --max-requests 10 \
    --random-seed "$RANDOM_SEED" \
    --disable-console \
    --outputs /dev/null 2>/dev/null || true
  echo "Warmup done."
}

run_bench() {
  local label="$1" run_num="$2"
  local json_out="$OUTDIR/${label}_run${run_num}.json"
  local console_out="$OUTDIR/${label}_run${run_num}.log"

  echo "  [$label] run $run_num/$NUM_RUNS  (input=$INPUT_TOKENS±$INPUT_STDEV, output=$OUTPUT_TOKENS±$OUTPUT_STDEV, conc=$CONCURRENCY, requests=$MAX_REQUESTS)"

  guidellm benchmark run \
    --target "$BASE_URL" \
    --model "$MODEL" \
    --data "$DATA_CFG" \
    --profile concurrent \
    --rate "$CONCURRENCY" \
    --max-requests "$MAX_REQUESTS" \
    --random-seed "$RANDOM_SEED" \
    --outputs "$json_out" \
    > "$console_out" 2>&1

  echo "  Done → $json_out"
}

# ── pre-flight ───────────────────────────────────────────

TOTAL=${#CONFIG_ORDER[@]}
echo "=== GuideLLM K-Sweep ($TOTAL configs) ==="
{
  echo "timestamp: $(date -Iseconds)"
  echo "model: $MODEL"
  echo "eagle_model: $EAGLE_MODEL"
  echo "eagle_method: $EAGLE_METHOD"
  echo "k_values: $K_VALUES"
  echo "num_runs: $NUM_RUNS"
  echo "concurrency: $CONCURRENCY"
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

  warmup_server

  for ((r=1; r<=NUM_RUNS; r++)); do
    run_bench "$name" "$r"
    sleep 2
  done
done

stop_server

# ── Summary ──────────────────────────────────────────────

echo ""
echo "============================================"
echo "  K-SWEEP RESULTS"
echo "============================================"
echo ""

python3 "$(dirname "$0")/../parse_guidellm_results.py" "$OUTDIR"

echo ""
echo "Full logs: $OUTDIR/"
echo "Done. Compare K values to find the optimal speculative token count."
