#!/usr/bin/env bash
set -euo pipefail

############################################
# C × Config Sweep: Baseline vs FP8 vs
# FP8+EAGLE3 across concurrency levels.
#
# Outer loop = configs (server restart each)
# Inner loop = C levels (no restart needed)
#
# Usage:
#   VLLM_MODEL=Qwen/Qwen3-8B \
#   bash scripts/run_guidellm_c_config_sweep.sh
############################################

MODEL="${VLLM_MODEL:?Set VLLM_MODEL (e.g. export VLLM_MODEL=Qwen/Qwen3-8B)}"

EAGLE_MODEL="${EAGLE_MODEL:-RedHatAI/Qwen3-8B-speculator.eagle3}"
EAGLE_METHOD="${EAGLE_METHOD:-eagle3}"
SPEC_K="${SPEC_K:-3}"

C_LEVELS="${GUIDELLM_C_LEVELS:-4,16,64,128,256}"
INPUT_TOKENS="${GUIDELLM_INPUT_TOKENS:-128}"
INPUT_STDEV="${GUIDELLM_INPUT_STDEV:-10}"
OUTPUT_TOKENS="${GUIDELLM_OUTPUT_TOKENS:-128}"
OUTPUT_STDEV="${GUIDELLM_OUTPUT_STDEV:-10}"
TEXT_SOURCE="${GUIDELLM_TEXT_SOURCE:-data:prideandprejudice.txt.gz}"
MAX_REQUESTS="${GUIDELLM_MAX_REQUESTS:-500}"
RANDOM_SEED="${GUIDELLM_SEED:-42}"
NUM_RUNS="${NUM_RUNS:-3}"

DTYPE="${VLLM_DTYPE:-auto}"
GPU_UTIL="${VLLM_GPU_UTIL:-0.95}"
MAX_MODEL_LEN="${VLLM_MAX_MODEL_LEN:-8192}"

COMMON_SERVER_FLAGS="--host 0.0.0.0 --port 8000 --dtype $DTYPE --gpu-memory-utilization $GPU_UTIL --max-model-len $MAX_MODEL_LEN"
BASE_URL="http://localhost:8000"

DATA_CFG="prompt_tokens=$INPUT_TOKENS,output_tokens=$OUTPUT_TOKENS,source=$TEXT_SOURCE"
if [[ "$INPUT_STDEV" -gt 0 ]] 2>/dev/null; then
  DATA_CFG="prompt_tokens=$INPUT_TOKENS,prompt_tokens_stdev=$INPUT_STDEV,output_tokens=$OUTPUT_TOKENS,output_tokens_stdev=$OUTPUT_STDEV,source=$TEXT_SOURCE"
fi

OUTDIR="logs/guidellm_c_config_sweep/$(date +%F_%H%M%S)"
mkdir -p "$OUTDIR"

# ── config matrix ────────────────────────────────────────

declare -a CONFIG_ORDER
declare -A CONFIG_FLAGS

CONFIG_ORDER=(
  baseline
  fp8-full
  fp8-eagle3-k${SPEC_K}
)

CONFIG_FLAGS=(
  [baseline]=""
  [fp8-full]="--quantization fp8 --kv-cache-dtype fp8"
  [fp8-eagle3-k${SPEC_K}]="--quantization fp8 --kv-cache-dtype fp8 --speculative-config {\"model\":\"$EAGLE_MODEL\",\"method\":\"$EAGLE_METHOD\",\"num_speculative_tokens\":$SPEC_K}"
)

IFS=',' read -ra C_ARRAY <<< "$C_LEVELS"

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

run_bench() {
  local label="$1" conc="$2" run_num="$3"
  local json_out="$OUTDIR/${label}-c${conc}_run${run_num}.json"
  local console_out="$OUTDIR/${label}-c${conc}_run${run_num}.log"

  echo "  [$label] C=$conc  run $run_num/$NUM_RUNS  (input=$INPUT_TOKENS±$INPUT_STDEV, output=$OUTPUT_TOKENS±$OUTPUT_STDEV, requests=$MAX_REQUESTS)"

  guidellm benchmark run \
    --target "$BASE_URL" \
    --model "$MODEL" \
    --data "$DATA_CFG" \
    --profile concurrent \
    --rate "$conc" \
    --max-requests "$MAX_REQUESTS" \
    --random-seed "$RANDOM_SEED" \
    --outputs "$json_out" \
    > "$console_out" 2>&1

  echo "  Done → $json_out"
}

# ── pre-flight ───────────────────────────────────────────

NUM_CONFIGS=${#CONFIG_ORDER[@]}
NUM_C=${#C_ARRAY[@]}
TOTAL_BENCHMARKS=$((NUM_CONFIGS * NUM_C * NUM_RUNS))

echo "=== GuideLLM C × Config Sweep ==="
echo "=== $NUM_CONFIGS configs × $NUM_C C-levels × $NUM_RUNS runs = $TOTAL_BENCHMARKS benchmarks ==="
{
  echo "timestamp: $(date -Iseconds)"
  echo "model: $MODEL"
  echo "eagle_model: $EAGLE_MODEL"
  echo "eagle_method: $EAGLE_METHOD"
  echo "spec_k: $SPEC_K"
  echo "c_levels: $C_LEVELS"
  echo "num_runs: $NUM_RUNS"
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

# ── Run each config across all C levels ──────────────────

CFG_NUM=0
for name in "${CONFIG_ORDER[@]}"; do
  CFG_NUM=$((CFG_NUM + 1))
  flags="${CONFIG_FLAGS[$name]}"

  echo ""
  echo "============================================"
  echo "  Config [$CFG_NUM/$NUM_CONFIGS]: $name"
  echo "  Flags: ${flags:-(none)}"
  echo "  C levels: ${C_LEVELS}"
  echo "============================================"

  stop_server
  start_server "$name" "$flags"

  if ! wait_for_server; then
    echo "SKIPPING $name — server failed to start. Check $OUTDIR/${name}_server.log"
    continue
  fi

  warmup_server

  for C in "${C_ARRAY[@]}"; do
    echo ""
    echo "  --- $name @ C=$C ---"
    for ((r=1; r<=NUM_RUNS; r++)); do
      run_bench "$name" "$C" "$r"
      sleep 2
    done
  done
done

stop_server

# ── Summary ──────────────────────────────────────────────

echo ""
echo "============================================"
echo "  C × CONFIG SWEEP RESULTS"
echo "============================================"
echo ""

python3 "$(dirname "$0")/../parse_guidellm_results.py" "$OUTDIR"

echo ""
echo "Full logs: $OUTDIR/"
echo "Done. Compare configs across concurrency levels."
