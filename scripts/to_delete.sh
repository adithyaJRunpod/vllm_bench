#!/usr/bin/env bash
set -euo pipefail

############################################
# Workload Sweep using GuideLLM
# Tests multiple input/output token profiles
# against a remote RunPod serverless endpoint.
# No local server management needed.
#
# Required env vars:
#   RUNPOD_API_KEY    RunPod API key (e.g. rpa_xxx)
#   RUNPOD_ENDPOINT   RunPod endpoint ID (e.g. nygvk2hd95hual)
#   MODEL             Model ID as reported by /v1/models (e.g. qwen/qwen3-8b)
#
# Optional overrides (defaults shown):
#   ENDPOINT_TYPE=queue                          "queue" or "lb" (load balancer)
#   TEXT_SOURCE=data:prideandprejudice.txt.gz   GuideLLM text corpus
#   REQUEST_FORMAT=/v1/chat/completions         API endpoint format
#   RANDOM_SEED=42                              Reproducibility seed
#   NUM_RUNS=2                                  Repeated runs per workload
#   STDEV_PCT=10                                Stdev as % of token count
#   WORKLOAD_FILTER=                              Run only this workload name
#
# Workloads (hardcoded, edit matrix below to customize):
#   short-short      128in/128out     C=32  300 reqs
#   prefill-heavy    1024in/256out    C=32  300 reqs
#   decode-heavy     256in/1024out    C=32  300 reqs
#   balanced-heavy   2000in/2000out   C=16  300 reqs
#
# Usage:
#   # Queue-based (default):
#   RUNPOD_API_KEY=rpa_xxx \
#   RUNPOD_ENDPOINT=nygvk2hd95hual \
#   MODEL=qwen/qwen3-8b \
#   bash scripts/run_guidellm_workload_sweep.sh
#
#   # Load balancer:
#   RUNPOD_API_KEY=rpa_xxx \
#   RUNPOD_ENDPOINT=jcja1rjzitd515 \
#   MODEL=Qwen/Qwen3-8B \
#   ENDPOINT_TYPE=lb \
#   bash scripts/run_guidellm_workload_sweep.sh
############################################

RUNPOD_API_KEY="${RUNPOD_API_KEY:?Set RUNPOD_API_KEY}"
RUNPOD_ENDPOINT="${RUNPOD_ENDPOINT:?Set RUNPOD_ENDPOINT}"
MODEL="${MODEL:?Set MODEL (e.g. qwen/qwen3-8b)}"

TEXT_SOURCE="${TEXT_SOURCE:-data:prideandprejudice.txt.gz}"
REQUEST_FORMAT="${REQUEST_FORMAT:-/v1/chat/completions}"
RANDOM_SEED="${RANDOM_SEED:-42}"
NUM_RUNS="${NUM_RUNS:-2}"
STDEV_PCT="${STDEV_PCT:-10}"
WORKLOAD_FILTER="${WORKLOAD_FILTER:-}"
ENDPOINT_TYPE="${ENDPOINT_TYPE:-queue}"

if [[ "$ENDPOINT_TYPE" == "lb" ]]; then
  BASE_URL="https://${RUNPOD_ENDPOINT}.api.runpod.ai"
  BACKEND_ARGS="{\"api_key\": \"${RUNPOD_API_KEY}\", \"validate_backend\": false}"
else
  BASE_URL="https://api.runpod.ai/v2/${RUNPOD_ENDPOINT}/openai"
  BACKEND_ARGS="{\"api_key\": \"${RUNPOD_API_KEY}\", \"validate_backend\": false}"
fi

OUTDIR="logs/guidellm_workload_sweep/$(date +%F_%H%M%S)"
mkdir -p "$OUTDIR"

# ── workload matrix ───────────────────────────────────
#   name          input  output  concurrency  max_requests

WORKLOADS=(
  "short-short       128    128   32   300"
  "prefill-heavy    1024    256   32   300"
  "decode-heavy      256   1024   32   300"
  "balanced-heavy   2000   2000   16   300"
)

# ── helpers ──────────────────────────────────────────

warmup_endpoint() {
  echo "Warming up endpoint (30 requests at C=16, 128/64 tokens)..."
  guidellm benchmark run \
    --target "$BASE_URL" \
    --model "$MODEL" \
    --request-format "$REQUEST_FORMAT" \
    --backend-args "$BACKEND_ARGS" \
    --data "prompt_tokens=128,output_tokens=64" \
    --profile concurrent \
    --rate 16 \
    --max-requests 30 \
    --random-seed "$RANDOM_SEED" \
    --disable-console \
    --outputs /dev/null 2>/dev/null || true
  echo "Warmup done."
}

run_bench() {
  local name="$1" input="$2" output="$3" conc="$4" max_req="$5" run_num="$6"

  local in_stdev=$(( input * STDEV_PCT / 100 ))
  local out_stdev=$(( output * STDEV_PCT / 100 ))
  local data_cfg="prompt_tokens=${input},prompt_tokens_stdev=${in_stdev},output_tokens=${output},output_tokens_stdev=${out_stdev},source=${TEXT_SOURCE}"

  local json_out="$OUTDIR/${name}_run${run_num}.json"
  local console_out="$OUTDIR/${name}_run${run_num}.log"

  echo "  [$name] run $run_num/$NUM_RUNS  (in=${input}±${in_stdev}, out=${output}±${out_stdev}, C=${conc}, reqs=${max_req})"

  guidellm benchmark run \
    --target "$BASE_URL" \
    --model "$MODEL" \
    --request-format "$REQUEST_FORMAT" \
    --backend-args "$BACKEND_ARGS" \
    --data "$data_cfg" \
    --profile concurrent \
    --rate "$conc" \
    --max-requests "$max_req" \
    --random-seed "$RANDOM_SEED" \
    --outputs "$json_out" \
    > "$console_out" 2>&1

  echo "  Done → $json_out"
}

# ── pre-flight ───────────────────────────────────────

TOTAL=${#WORKLOADS[@]}
echo "=== GuideLLM Workload Sweep ($TOTAL workloads × $NUM_RUNS runs) ==="
{
  echo "timestamp: $(date -Iseconds)"
  echo "model: $MODEL"
  echo "endpoint: $RUNPOD_ENDPOINT"
  echo "base_url: $BASE_URL"
  echo "request_format: $REQUEST_FORMAT"
  echo "num_runs: $NUM_RUNS"
  echo "stdev_pct: ${STDEV_PCT}%"
  echo "text_source: $TEXT_SOURCE"
  echo "random_seed: $RANDOM_SEED"
  echo "guidellm_version: $(python3 -c 'import guidellm; print(guidellm.__version__)' 2>/dev/null || echo unknown)"
  echo ""
  echo "workloads:"
  for entry in "${WORKLOADS[@]}"; do
    read -r wname winput woutput wconc wreqs <<< "$entry"
    local_in_sd=$(( winput * STDEV_PCT / 100 ))
    local_out_sd=$(( woutput * STDEV_PCT / 100 ))
    echo "  $wname: in=${winput}±${local_in_sd} out=${woutput}±${local_out_sd} C=${wconc} reqs=${wreqs}"
  done
} | tee "$OUTDIR/environment.txt"
echo ""

# ── warmup ───────────────────────────────────────────

warmup_endpoint

# ── run each workload ────────────────────────────────

RUN=0
for entry in "${WORKLOADS[@]}"; do
  read -r wname winput woutput wconc wreqs <<< "$entry"
  [[ -n "$WORKLOAD_FILTER" && "$wname" != "$WORKLOAD_FILTER" ]] && continue
  RUN=$((RUN + 1))

  echo ""
  echo "============================================"
  echo "  [$RUN/$TOTAL] $wname"
  echo "  Input: $winput  Output: $woutput  C: $wconc  Requests: $wreqs"
  echo "============================================"

  for ((r=1; r<=NUM_RUNS; r++)); do
    run_bench "$wname" "$winput" "$woutput" "$wconc" "$wreqs" "$r"
    sleep 3
  done
done

# ── summary ──────────────────────────────────────────

echo ""
echo "============================================"
echo "  WORKLOAD SWEEP RESULTS"
echo "============================================"
echo ""

python3 "$(dirname "$0")/../parse_guidellm_results.py" "$OUTDIR"

echo ""
echo "Full logs: $OUTDIR/"
echo "Done."
