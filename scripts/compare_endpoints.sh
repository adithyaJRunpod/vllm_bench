#!/usr/bin/env bash
set -euo pipefail

############################################
# Endpoint Comparison using GuideLLM
# Benchmarks two RunPod serverless endpoints
# with identical workload and produces a
# side-by-side comparison table.
#
# Required env vars:
#   RUNPOD_API_KEY    RunPod API key (e.g. rpa_xxx)
#   STANDARD_ENDPOINT RunPod endpoint ID for standard endpoint
#   OVERDRIVE_ENDPOINT RunPod endpoint ID for overdrive endpoint
#   MODEL             Model ID as reported by /v1/models
#
# Optional overrides (defaults shown):
#   INPUT_TOKENS=2000                   Prompt token target
#   OUTPUT_TOKENS=2000                  Output token target
#   STDEV_PCT=10                        Stdev as % of token count
#   CONCURRENCY=16                      Concurrent streams
#   MAX_REQUESTS=300                    Requests per benchmark
#   NUM_RUNS=2                          Repeated runs per endpoint
#   TEXT_SOURCE=data:prideandprejudice.txt.gz   GuideLLM text corpus
#   REQUEST_FORMAT=/v1/chat/completions         API endpoint format
#   RANDOM_SEED=42                              Reproducibility seed
#
# Usage:
#   STANDARD_ENDPOINT=abc123 \
#   OVERDRIVE_ENDPOINT=xyz789 \
#   MODEL=meta-llama/llama-3.1-8b-instruct \
#   bash scripts/compare_endpoints.sh
############################################

RUNPOD_API_KEY="${RUNPOD_API_KEY:?Set RUNPOD_API_KEY}"
STANDARD_ENDPOINT="${STANDARD_ENDPOINT:?Set STANDARD_ENDPOINT (RunPod endpoint ID)}"
OVERDRIVE_ENDPOINT="${OVERDRIVE_ENDPOINT:?Set OVERDRIVE_ENDPOINT (RunPod endpoint ID)}"
MODEL="${MODEL:?Set MODEL (e.g. meta-llama/llama-3.1-8b-instruct)}"

LABEL_A="standard"
LABEL_B="overdrive"
INPUT_TOKENS="${INPUT_TOKENS:-2000}"
OUTPUT_TOKENS="${OUTPUT_TOKENS:-2000}"
STDEV_PCT="${STDEV_PCT:-10}"
CONCURRENCY="${CONCURRENCY:-16}"
MAX_REQUESTS="${MAX_REQUESTS:-300}"
NUM_RUNS="${NUM_RUNS:-2}"
TEXT_SOURCE="${TEXT_SOURCE:-data:prideandprejudice.txt.gz}"
REQUEST_FORMAT="${REQUEST_FORMAT:-/v1/chat/completions}"
RANDOM_SEED="${RANDOM_SEED:-42}"

IN_STDEV=$(( INPUT_TOKENS * STDEV_PCT / 100 ))
OUT_STDEV=$(( OUTPUT_TOKENS * STDEV_PCT / 100 ))
DATA_CFG="prompt_tokens=${INPUT_TOKENS},prompt_tokens_stdev=${IN_STDEV},output_tokens=${OUTPUT_TOKENS},output_tokens_stdev=${OUT_STDEV},source=${TEXT_SOURCE}"

BACKEND_ARGS="{\"api_key\": \"${RUNPOD_API_KEY}\", \"validate_backend\": false}"

OUTDIR="logs/endpoint_compare/$(date +%F_%H%M%S)"
mkdir -p "$OUTDIR"

# ── helpers ──────────────────────────────────────────

warmup_endpoint() {
  local label="$1" base_url="$2"
  echo "Warming up $label (30 requests at C=16, 128/64 tokens)..."
  guidellm benchmark run \
    --target "$base_url" \
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
  echo "Warmup done for $label."
}

run_bench() {
  local label="$1" base_url="$2" run_num="$3"
  local json_out="$OUTDIR/${label}_run${run_num}.json"
  local console_out="$OUTDIR/${label}_run${run_num}.log"

  echo "  [$label] run $run_num/$NUM_RUNS  (in=${INPUT_TOKENS}±${IN_STDEV}, out=${OUTPUT_TOKENS}±${OUT_STDEV}, C=${CONCURRENCY}, reqs=${MAX_REQUESTS})"

  guidellm benchmark run \
    --target "$base_url" \
    --model "$MODEL" \
    --request-format "$REQUEST_FORMAT" \
    --backend-args "$BACKEND_ARGS" \
    --data "$DATA_CFG" \
    --profile concurrent \
    --rate "$CONCURRENCY" \
    --max-requests "$MAX_REQUESTS" \
    --random-seed "$RANDOM_SEED" \
    --outputs "$json_out" \
    > "$console_out" 2>&1

  echo "  Done → $json_out"
}

# ── pre-flight ───────────────────────────────────────

URL_A="https://api.runpod.ai/v2/${STANDARD_ENDPOINT}/openai"
URL_B="https://api.runpod.ai/v2/${OVERDRIVE_ENDPOINT}/openai"

echo "=== Endpoint Comparison ==="
{
  echo "timestamp: $(date -Iseconds)"
  echo "model: $MODEL"
  echo ""
  echo "standard_endpoint: $STANDARD_ENDPOINT"
  echo "  url: $URL_A"
  echo "overdrive_endpoint: $OVERDRIVE_ENDPOINT"
  echo "  url: $URL_B"
  echo ""
  echo "workload:"
  echo "  input_tokens: ${INPUT_TOKENS}±${IN_STDEV}"
  echo "  output_tokens: ${OUTPUT_TOKENS}±${OUT_STDEV}"
  echo "  concurrency: $CONCURRENCY"
  echo "  max_requests: $MAX_REQUESTS"
  echo "  num_runs: $NUM_RUNS"
  echo "  text_source: $TEXT_SOURCE"
  echo "  random_seed: $RANDOM_SEED"
  echo ""
  echo "guidellm_version: $(python3 -c 'import guidellm; print(guidellm.__version__)' 2>/dev/null || echo unknown)"
} | tee "$OUTDIR/environment.txt"
echo ""

# ── warmup both endpoints ────────────────────────────

warmup_endpoint "$LABEL_A" "$URL_A"
echo ""
warmup_endpoint "$LABEL_B" "$URL_B"
echo ""

# ── benchmark endpoint A ─────────────────────────────

echo "============================================"
echo "  Benchmarking: standard ($STANDARD_ENDPOINT)"
echo "============================================"

for ((r=1; r<=NUM_RUNS; r++)); do
  run_bench "$LABEL_A" "$URL_A" "$r"
  sleep 3
done

echo ""

# ── benchmark endpoint B ─────────────────────────────

echo "============================================"
echo "  Benchmarking: overdrive ($OVERDRIVE_ENDPOINT)"
echo "============================================"

for ((r=1; r<=NUM_RUNS; r++)); do
  run_bench "$LABEL_B" "$URL_B" "$r"
  sleep 3
done

# ── comparison summary ───────────────────────────────

echo ""
echo "============================================"
echo "  ENDPOINT COMPARISON RESULTS"
echo "============================================"
echo ""

python3 "$(dirname "$0")/../compare_endpoint_results.py" \
  --dir "$OUTDIR" \
  --label-a "$LABEL_A" \
  --label-b "$LABEL_B"

echo ""
echo "Full logs: $OUTDIR/"
echo "Done."
