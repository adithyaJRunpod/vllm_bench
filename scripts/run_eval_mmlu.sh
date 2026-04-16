#!/usr/bin/env bash
set -euo pipefail

############################################
# MMLU Eval: BF16 vs FP8 vs FP8+EAGLE3
# Uses lm-eval's built-in vLLM backend
# (no server needed).
#
# Requires: pip install "lm-eval[api]"
############################################

MODEL="${VLLM_MODEL:?Set VLLM_MODEL (e.g. export VLLM_MODEL=Qwen/Qwen3-8B)}"
GPU_UTIL="${VLLM_GPU_UTIL:-0.95}"
MAX_MODEL_LEN="${VLLM_MAX_MODEL_LEN:-8192}"
TASKS="${EVAL_TASKS:-mmlu}"
BATCH_SIZE="${EVAL_BATCH_SIZE:-auto}"

OUTDIR="logs/eval/$(date +%F_%H%M%S)"
mkdir -p "$OUTDIR"

COMMON_ARGS="pretrained=$MODEL,gpu_memory_utilization=$GPU_UTIL,max_model_len=$MAX_MODEL_LEN"

echo "=== MMLU Eval: BF16 vs FP8 vs FP8+EAGLE3 ==="
{
  echo "timestamp: $(date -Iseconds)"
  echo "model: $MODEL"
  echo "tasks: $TASKS"
  echo "batch_size: $BATCH_SIZE"
  echo "gpu_memory_utilization: $GPU_UTIL"
  echo "max_model_len: $MAX_MODEL_LEN"
  echo "vllm_version: $(python3 -c 'import vllm; print(vllm.__version__)' 2>/dev/null || echo unknown)"
  echo "lm_eval_version: $(python3 -c 'import lm_eval; print(lm_eval.__version__)' 2>/dev/null || echo unknown)"
  echo "gpu: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo unknown)"
} | tee "$OUTDIR/environment.txt"
echo ""

# ── [1/3] Baseline (BF16) ───────────────────────────────

echo ""
echo "============================================"
echo "  [1/3] Baseline (BF16)"
echo "============================================"

lm_eval \
  --model vllm \
  --model_args "$COMMON_ARGS,dtype=bfloat16" \
  --tasks "$TASKS" \
  --batch_size "$BATCH_SIZE" \
  --output_path "$OUTDIR/baseline" \
  2>&1 | tee "$OUTDIR/baseline_eval.log"

echo "Done → $OUTDIR/baseline"

# ── [2/3] FP8 (weights + KV cache) ──────────────────────

echo ""
echo "============================================"
echo "  [2/3] FP8 (weights + KV cache)"
echo "============================================"

lm_eval \
  --model vllm \
  --model_args "$COMMON_ARGS,dtype=bfloat16,quantization=fp8,kv_cache_dtype=fp8" \
  --tasks "$TASKS" \
  --batch_size "$BATCH_SIZE" \
  --output_path "$OUTDIR/fp8" \
  2>&1 | tee "$OUTDIR/fp8_eval.log"

echo "Done → $OUTDIR/fp8"

# ── [3/3] FP8 + EAGLE3 speculative decoding ──────────────
# lm-eval's --model_args can't pass speculative_config (dict with commas
# breaks the comma-separated parser), so we call the Python API directly.

EAGLE3_MODEL="${EAGLE3_MODEL:-RedHatAI/Qwen3-8B-speculator.eagle3}"
EAGLE3_K="${EAGLE3_K:-3}"

echo ""
echo "============================================"
echo "  [3/3] FP8 + EAGLE3 (k=$EAGLE3_K)"
echo "============================================"

python3 - "$OUTDIR" "$MODEL" "$GPU_UTIL" "$MAX_MODEL_LEN" "$TASKS" "$BATCH_SIZE" "$EAGLE3_MODEL" "$EAGLE3_K" <<'PYEOF' 2>&1 | tee "$OUTDIR/fp8_eagle3_eval.log"
import sys, json, os
import lm_eval

outdir, model, gpu_util, max_len, tasks, batch, eagle_model, eagle_k = sys.argv[1:9]

results = lm_eval.simple_evaluate(
    model="vllm",
    model_args={
        "pretrained": model,
        "gpu_memory_utilization": float(gpu_util),
        "max_model_len": int(max_len),
        "dtype": "bfloat16",
        "quantization": "fp8",
        "kv_cache_dtype": "fp8",
        "speculative_config": {
            "model": eagle_model,
            "method": "eagle3",
            "num_speculative_tokens": int(eagle_k),
        },
    },
    tasks=tasks.split(","),
    batch_size=batch if batch == "auto" else int(batch),
)

out_path = os.path.join(outdir, "fp8_eagle3")
os.makedirs(out_path, exist_ok=True)
with open(os.path.join(out_path, "results.json"), "w") as f:
    json.dump(results["results"], f, indent=2, default=str)

from lm_eval.utils import make_table
print(make_table(results))
PYEOF

echo "Done → $OUTDIR/fp8_eagle3"

# ── Summary ──────────────────────────────────────────────

echo ""
echo "============================================"
echo "  EVAL RESULTS"
echo "============================================"
echo ""
echo "--- Baseline (BF16) ---"
tail -30 "$OUTDIR/baseline_eval.log" | grep -iE "mmlu|acc|Groups" || echo "(no results found)"
echo ""
echo "--- FP8 ---"
tail -30 "$OUTDIR/fp8_eval.log" | grep -iE "mmlu|acc|Groups" || echo "(no results found)"
echo ""
echo "--- FP8 + EAGLE3 ---"
tail -30 "$OUTDIR/fp8_eagle3_eval.log" | grep -iE "mmlu|acc|Groups" || echo "(no results found)"
echo ""
echo "Full logs: $OUTDIR/"
echo "Done. Compare accuracy scores above."
