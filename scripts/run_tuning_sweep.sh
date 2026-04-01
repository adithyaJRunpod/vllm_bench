#!/usr/bin/env bash
set -euo pipefail

############################################
# Tuning sweep for vLLM
# Restarts the vLLM server with different
# flag combos and benchmarks each one.
# Each config changes ONE variable from baseline.
############################################

MODEL="${VLLM_MODEL:?Set VLLM_MODEL before running (e.g. export VLLM_MODEL=Qwen/Qwen3-8B)}"
BASE_URL="http://127.0.0.1:8000"
API_KEY="${VLLM_API_KEY:-}"

REQUEST_RATE="${VLLM_REQUEST_RATE:?Set VLLM_REQUEST_RATE (e.g. export VLLM_REQUEST_RATE=120)}"
MAX_CONCURRENCY="${VLLM_MAX_CONCURRENCY:?Set VLLM_MAX_CONCURRENCY (e.g. export VLLM_MAX_CONCURRENCY=128)}"
INPUT_LEN="${VLLM_INPUT_LEN:?Set VLLM_INPUT_LEN (e.g. export VLLM_INPUT_LEN=128)}"
OUTPUT_LEN="${VLLM_OUTPUT_LEN:?Set VLLM_OUTPUT_LEN (e.g. export VLLM_OUTPUT_LEN=128)}"
NUM_PROMPTS=1024

DTYPE="${VLLM_DTYPE:-auto}"
GPU_UTIL="${VLLM_GPU_UTIL:-0.95}"
MAX_MODEL_LEN="${VLLM_MAX_MODEL_LEN:-8128}"

COMMON_FLAGS="--host 0.0.0.0 --port 8000 --dtype $DTYPE --gpu-memory-utilization $GPU_UTIL --max-model-len $MAX_MODEL_LEN"

declare -A CONFIGS
CONFIGS=(
  [baseline]="--enforce-eager"
  [prefix-caching]="--enable-prefix-caching"
  [max-seqs-64]="--max-num-seqs 64"
  [max-seqs-256]="--max-num-seqs 256"
  [max-seqs-512]="--max-num-seqs 512"
  [kv-cache-fp8]="--kv-cache-dtype fp8"
  [batched-tokens-4096]="--max-num-batched-tokens 4096"
  [batched-tokens-8192]="--max-num-batched-tokens 8192"
)

CONFIG_ORDER=(baseline prefix-caching max-seqs-64 max-seqs-256 max-seqs-512 kv-cache-fp8 batched-tokens-4096 batched-tokens-8192)

OUTDIR="logs/tuning_sweep/$(date +%F_%H%M%S)"
mkdir -p "$OUTDIR"

TOTAL=${#CONFIG_ORDER[@]}
echo "Output directory: $OUTDIR"
echo "Testing $TOTAL configs: ${CONFIG_ORDER[*]}"
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
  echo ""
  for name in "${CONFIG_ORDER[@]}"; do
    echo "config [$name]: vllm serve $MODEL $COMMON_FLAGS ${CONFIGS[$name]}"
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
  local name="$1"
  local extra_flags="$2"
  echo "Starting vLLM server [$name]: vllm serve $MODEL $COMMON_FLAGS $extra_flags"
  # shellcheck disable=SC2086
  nohup vllm serve "$MODEL" $COMMON_FLAGS $extra_flags > "$OUTDIR/${name}_server.log" 2>&1 &
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
for name in "${CONFIG_ORDER[@]}"; do
  RUN=$((RUN + 1))
  extra_flags="${CONFIGS[$name]}"

  echo ""
  echo "=========================================="
  echo "[$RUN/$TOTAL] Config: $name"
  echo "  Flags: $extra_flags"
  echo "=========================================="

  stop_server
  start_server "$name" "$extra_flags"

  if ! wait_for_server; then
    echo "SKIPPING $name — server failed to start. Check $OUTDIR/${name}_server.log"
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

  "${CMD[@]}" > "$OUTDIR/${name}_bench.log" 2>&1
  echo "Done. Log: $OUTDIR/${name}_bench.log"
done

echo ""
echo "All done. Logs in: $OUTDIR"
echo "Run 'python3 parse_bench_logs.py tuning' to extract the summary CSV."
