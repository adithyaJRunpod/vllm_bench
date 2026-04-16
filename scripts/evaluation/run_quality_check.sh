#!/usr/bin/env bash
set -euo pipefail

############################################
# Quality Check: Compare outputs from
# Baseline (BF16) vs FP8 vs FP8+EAGLE3
# to verify FP8 doesn't degrade quality.
############################################

MODEL="${VLLM_MODEL:?Set VLLM_MODEL (e.g. export VLLM_MODEL=Qwen/Qwen3-8B)}"
EAGLE_MODEL="${EAGLE_MODEL:-RedHatAI/Qwen3-8B-speculator.eagle3}"
PROMPT="Explain how photosynthesis works in 3 paragraphs."
MAX_TOKENS=256
OUTDIR="logs/quality_check/$(date +%F_%H%M%S)"
mkdir -p "$OUTDIR"

stop_server() {
  pkill -f "vllm serve" 2>/dev/null || true
  sleep 5
  if pgrep -f "vllm serve" >/dev/null 2>&1; then
    pkill -9 -f "vllm serve" 2>/dev/null || true
    sleep 3
  fi
}

wait_for_server() {
  local max_wait=180 elapsed=0
  echo "Waiting for server to be ready (max ${max_wait}s)..."
  while [ $elapsed -lt $max_wait ]; do
    if curl -s --max-time 2 http://localhost:8000/v1/models >/dev/null 2>&1; then
      echo "Server ready after ${elapsed}s."
      return 0
    fi
    sleep 5
    elapsed=$((elapsed + 5))
  done
  echo "ERROR: Server did not start within ${max_wait}s."
  return 1
}

query_model() {
  local label="$1"
  echo "Querying [$label]..."
  curl -s http://localhost:8000/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d "{\"model\":\"$MODEL\",\"messages\":[{\"role\":\"user\",\"content\":\"$PROMPT\"}],\"max_tokens\":$MAX_TOKENS}" \
    > "$OUTDIR/${label}_raw.json"

  python3 -c "
import json
d = json.load(open('$OUTDIR/${label}_raw.json'))
if 'choices' in d:
    print(d['choices'][0]['message']['content'])
else:
    print('ERROR:', json.dumps(d, indent=2)[:500])
" | tee "$OUTDIR/${label}.txt"
  echo ""
}

# === Baseline (BF16) ===
echo ""
echo "============================================"
echo "  [1/3] Baseline (BF16)"
echo "============================================"
stop_server
nohup vllm serve "$MODEL" --host 0.0.0.0 --port 8000 \
  > "$OUTDIR/baseline_server.log" 2>&1 &
wait_for_server
query_model "baseline"

# === FP8 ===
echo ""
echo "============================================"
echo "  [2/3] FP8 (weights + KV cache)"
echo "============================================"
stop_server
nohup vllm serve "$MODEL" --host 0.0.0.0 --port 8000 \
  --quantization fp8 --kv-cache-dtype fp8 \
  > "$OUTDIR/fp8_server.log" 2>&1 &
wait_for_server
query_model "fp8"

# === FP8 + EAGLE3 ===
echo ""
echo "============================================"
echo "  [3/3] FP8 + EAGLE3"
echo "============================================"
stop_server
nohup vllm serve "$MODEL" --host 0.0.0.0 --port 8000 \
  --quantization fp8 --kv-cache-dtype fp8 \
  --speculative-config "{\"model\":\"$EAGLE_MODEL\",\"method\":\"eagle3\",\"num_speculative_tokens\":3}" \
  > "$OUTDIR/eagle3_server.log" 2>&1 &
wait_for_server
query_model "fp8_eagle3"

stop_server

# === Side-by-side comparison ===
echo ""
echo "============================================"
echo "  QUALITY COMPARISON"
echo "============================================"
echo ""
echo "--- BASELINE (BF16) ---"
cat "$OUTDIR/baseline.txt"
echo ""
echo "--- FP8 ---"
cat "$OUTDIR/fp8.txt"
echo ""
echo "--- FP8 + EAGLE3 ---"
cat "$OUTDIR/fp8_eagle3.txt"
echo ""
echo "============================================"
echo "Full logs: $OUTDIR/"
echo "Done."
