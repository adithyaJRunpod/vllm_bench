#!/usr/bin/env bash
set -euo pipefail

############################################
# max-num-batched-tokens sweep for vLLM
# Restarts the server for each value.
# Best run with long prompts (e.g. 2048/512)
# where chunked prefill actually kicks in.
############################################

MODEL="${VLLM_MODEL:?Set VLLM_MODEL before running (e.g. export VLLM_MODEL=Qwen/Qwen2.5-32B-Instruct-FP8)}"
BASE_URL="http://127.0.0.1:8000"
API_KEY="${VLLM_API_KEY:-}"

REQUEST_RATE="${VLLM_REQUEST_RATE:?Set VLLM_REQUEST_RATE (e.g. export VLLM_REQUEST_RATE=120)}"
MAX_CONCURRENCY="${VLLM_MAX_CONCURRENCY:?Set VLLM_MAX_CONCURRENCY (e.g. export VLLM_MAX_CONCURRENCY=128)}"
INPUT_LEN="${VLLM_INPUT_LEN:-2048}"
OUTPUT_LEN="${VLLM_OUTPUT_LEN:-512}"
NUM_PROMPTS="${VLLM_NUM_PROMPTS:-512}"

DTYPE="${VLLM_DTYPE:-auto}"
GPU_UTIL="${VLLM_GPU_UTIL:-0.95}"
MAX_MODEL_LEN="${VLLM_MAX_MODEL_LEN:-8192}"

COMMON_FLAGS="--host 0.0.0.0 --port 8000 --dtype $DTYPE --gpu-memory-utilization $GPU_UTIL --max-model-len $MAX_MODEL_LEN"

BATCHED_TOKEN_VALUES=(512 1024 2048 4096 8192 16384)

OUTDIR="logs/batched_tokens_sweep/$(date +%F_%H%M%S)"
mkdir -p "$OUTDIR"

TOTAL=${#BATCHED_TOKEN_VALUES[@]}
echo "Output directory: $OUTDIR"
echo "Testing $TOTAL max-num-batched-tokens values: ${BATCHED_TOKEN_VALUES[*]}"
echo "Workload: input=$INPUT_LEN output=$OUTPUT_LEN prompts=$NUM_PROMPTS"
echo ""

echo "--- Capturing environment ---"
{
  echo "timestamp: $(date -Iseconds)"
  echo "model: $MODEL"
  echo "vllm_version: $(python3 -c 'import vllm; print(vllm.__version__)' 2>/dev/null || echo unknown)"
  echo "gpu: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo unknown)"
  echo "gpu_memory: $(nvidia-smi --query-gpu=memory.total --format=csv,noheader 2>/dev/null || echo unknown)"
  echo "driver: $(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null || echo unknown)"
  echo "request_rate: $REQUEST_RATE"
  echo "max_concurrency: $MAX_CONCURRENCY"
  echo "input_len: $INPUT_LEN"
  echo "output_len: $OUTPUT_LEN"
  echo "num_prompts: $NUM_PROMPTS"
  echo "max_model_len: $MAX_MODEL_LEN"
  echo "common_flags: $COMMON_FLAGS"
  echo ""
  for v in "${BATCHED_TOKEN_VALUES[@]}"; do
    echo "config [bt-$v]: vllm serve $MODEL $COMMON_FLAGS --max-num-batched-tokens $v"
  done
} | tee "$OUTDIR/environment.txt"
echo ""

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
  local bt_val="$2"
  echo "Starting vLLM server [$label]: vllm serve $MODEL $COMMON_FLAGS --max-num-batched-tokens $bt_val"
  set +B
  # shellcheck disable=SC2086
  nohup vllm serve "$MODEL" $COMMON_FLAGS --max-num-batched-tokens "$bt_val" > "$OUTDIR/${label}_server.log" 2>&1 &
  set -B
  echo "Server PID: $!"
}

wait_for_server() {
  local max_wait=300
  local elapsed=0
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
  echo "Warming up server (16 prompts)..."
  local warmup_cmd=(vllm bench serve
    --backend openai-chat
    --base-url "$BASE_URL"
    --endpoint /v1/chat/completions
    --model "$MODEL"
    --dataset-name random
    --random-input-len "$INPUT_LEN"
    --random-output-len "$OUTPUT_LEN"
    --num-prompts 16
    --max-concurrency 4
    --request-rate 10
  )
  [[ -n "$API_KEY" ]] && warmup_cmd+=(--header "Authorization=Bearer $API_KEY")
  "${warmup_cmd[@]}" >/dev/null 2>&1 || true
  echo "Warmup done."
}

RUN=0
for bt in "${BATCHED_TOKEN_VALUES[@]}"; do
  RUN=$((RUN + 1))
  label="bt-${bt}"

  echo ""
  echo "=========================================="
  echo "[$RUN/$TOTAL] max-num-batched-tokens=$bt"
  echo "=========================================="

  stop_server
  start_server "$label" "$bt"

  if ! wait_for_server; then
    echo "SKIPPING $label — server failed to start. Check $OUTDIR/${label}_server.log"
    continue
  fi

  warmup_server

  echo "Running benchmark..."
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
    --request-rate "$REQUEST_RATE"
    --percentile-metrics ttft,itl,e2el
    --metric-percentiles 50,90,99
  )

  [[ -n "$API_KEY" ]] && CMD+=(--header "Authorization=Bearer $API_KEY")

  "${CMD[@]}" > "$OUTDIR/${label}_bench.log" 2>&1
  echo "Done. Log: $OUTDIR/${label}_bench.log"
done

stop_server
echo ""
echo "All done. Logs in: $OUTDIR"
echo "Server stopped to avoid stale config leaking into subsequent sweeps."
echo "Run 'python3 parse_bench_logs.py batched_tokens' to extract the summary CSV."
