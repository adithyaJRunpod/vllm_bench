#!/usr/bin/env bash
set -euo pipefail

############################################
# MMLU Eval: BF16 vs FP8 vs FP8+EAGLE3
# Verify that FP8 quantization and
# speculative decoding don't degrade
# model quality.
#
# Requires: pip install lm-eval
############################################

MODEL="${VLLM_MODEL:?Set VLLM_MODEL (e.g. export VLLM_MODEL=Qwen/Qwen3-8B)}"
EAGLE_MODEL="${EAGLE_MODEL:-RedHatAI/Qwen3-8B-speculator.eagle3}"
EAGLE_METHOD="${EAGLE_METHOD:-eagle3}"
SPEC_K="${SPEC_K:-3}"

DTYPE="bfloat16"
GPU_UTIL="${VLLM_GPU_UTIL:-0.95}"
MAX_MODEL_LEN="${VLLM_MAX_MODEL_LEN:-8192}"
NUM_CONCURRENT="${EVAL_NUM_CONCURRENT:-32}"
TASKS="${EVAL_TASKS:-mmlu}"

COMMON_SERVER_FLAGS="--host 0.0.0.0 --port 8000 --dtype $DTYPE --gpu-memory-utilization $GPU_UTIL --max-model-len $MAX_MODEL_LEN"
BASE_URL="http://localhost:8000"

OUTDIR="logs/eval/$(date +%F_%H%M%S)"
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

# ── helpers ──────────────────────────────────────────────

stop_server() {
  echo "Stopping vLLM server..."
  pkill -f "vllm serve" 2>/dev/null || true
  sleep 5
  if pgrep -f "vllm serve" >/dev/null 2>&1; then
    pkill -9 -f "vllm serve" 2>/dev/null || true
    sleep 3
  fi
}

start_server() {
  local label="$1"
  local extra_flags="${2:-}"
  echo "Starting vLLM server [$label]: vllm serve $MODEL $COMMON_SERVER_FLAGS $extra_flags"
  # shellcheck disable=SC2086
  nohup vllm serve "$MODEL" $COMMON_SERVER_FLAGS $extra_flags \
    > "$OUTDIR/${label}_server.log" 2>&1 &
  echo "Server PID: $!"
}

wait_for_server() {
  local max_wait=240 elapsed=0
  echo "Waiting for server to be ready (max ${max_wait}s)..."
  while [ $elapsed -lt $max_wait ]; do
    if curl -s --max-time 2 "$BASE_URL/v1/models" >/dev/null 2>&1; then
      echo "Server ready after ${elapsed}s."
      return 0
    fi
    sleep 5
    elapsed=$((elapsed + 5))
  done
  echo "ERROR: Server did not start within ${max_wait}s."
  return 1
}

run_eval() {
  local label="$1"
  echo "Running $TASKS eval for [$label]..."
  lm_eval \
    --model local-completions \
    --model_args "model=$MODEL,base_url=$BASE_URL/v1,num_concurrent=$NUM_CONCURRENT,tokenized_requests=False" \
    --tasks "$TASKS" \
    --apply_chat_template \
    --output_path "$OUTDIR/$label" \
    2>&1 | tee "$OUTDIR/${label}_eval.log"
  echo "Done → $OUTDIR/$label"
}

# ── pre-flight ───────────────────────────────────────────

NUM_CONFIGS=${#CONFIG_ORDER[@]}

echo "=== MMLU Eval: $NUM_CONFIGS configs ==="
{
  echo "timestamp: $(date -Iseconds)"
  echo "model: $MODEL"
  echo "dtype: $DTYPE"
  echo "tasks: $TASKS"
  echo "num_concurrent: $NUM_CONCURRENT"
  echo "configs: ${CONFIG_ORDER[*]}"
  echo "vllm_version: $(python3 -c 'import vllm; print(vllm.__version__)' 2>/dev/null || echo unknown)"
  echo "lm_eval_version: $(python3 -c 'import lm_eval; print(lm_eval.__version__)' 2>/dev/null || echo unknown)"
  echo "gpu: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo unknown)"
  echo ""
  for name in "${CONFIG_ORDER[@]}"; do
    echo "config [$name]: vllm serve $MODEL $COMMON_SERVER_FLAGS ${CONFIG_FLAGS[$name]}"
  done
} | tee "$OUTDIR/environment.txt"
echo ""

# ── Run each config ──────────────────────────────────────

CFG_NUM=0
for name in "${CONFIG_ORDER[@]}"; do
  CFG_NUM=$((CFG_NUM + 1))
  flags="${CONFIG_FLAGS[$name]}"

  echo ""
  echo "============================================"
  echo "  [$CFG_NUM/$NUM_CONFIGS] $name"
  echo "  Flags: ${flags:-(none)}"
  echo "============================================"

  stop_server
  start_server "$name" "$flags"

  if ! wait_for_server; then
    echo "SKIPPING $name — server failed to start. Check $OUTDIR/${name}_server.log"
    continue
  fi

  run_eval "$name"
done

stop_server

# ── Summary ──────────────────────────────────────────────

echo ""
echo "============================================"
echo "  EVAL RESULTS"
echo "============================================"

for name in "${CONFIG_ORDER[@]}"; do
  echo ""
  echo "--- $name ---"
  if [ -f "$OUTDIR/${name}_eval.log" ]; then
    tail -30 "$OUTDIR/${name}_eval.log" | grep -iE "mmlu|acc|Groups" || echo "(no results found)"
  else
    echo "(skipped)"
  fi
done

echo ""
echo "Full logs: $OUTDIR/"
echo "Done. Compare accuracy scores above."
