# vllm_bench

Benchmark scripts for vLLM on RunPod H100 GPUs.

## Pod Setup (fresh pod)

```bash
# Install vLLM, GuideLLM, and lm-eval
pip install vllm==0.18.0
pip install guidellm==0.6.0
pip install "lm-eval[api]"

# Clone repo & fix line endings
cd /workspace
git clone https://github.com/adithyaJRunpod/vllm_bench.git
cd vllm_bench
sed -i 's/\r$//' scripts/*.sh

# Set model
export VLLM_MODEL=Qwen/Qwen3-8B

# 1. Concurrency sweep (find saturation point)
export GUIDELLM_C_LEVELS=16,32,64,96,128,192,256
export GUIDELLM_INPUT_TOKENS=128
export GUIDELLM_OUTPUT_TOKENS=128
export GUIDELLM_INPUT_STDEV=0
export GUIDELLM_OUTPUT_STDEV=0
export GUIDELLM_MAX_REQUESTS=200
export GUIDELLM_SEED=42
bash scripts/run_guidellm_c_sweep.sh

# 2. Config comparison (all 8 configs at chosen C)
export GUIDELLM_CONCURRENCY=90
export GUIDELLM_INPUT_TOKENS=128
export GUIDELLM_OUTPUT_TOKENS=128
export GUIDELLM_INPUT_STDEV=0
export GUIDELLM_OUTPUT_STDEV=0
export GUIDELLM_MAX_REQUESTS=500
export GUIDELLM_SEED=42
export NUM_RUNS=3
bash scripts/run_guidellm_config_compare.sh

# 3. C × Config sweep (baseline vs fp8-full vs fp8-eagle3-k3)
export GUIDELLM_C_LEVELS=4,16,64,96,128,256
export GUIDELLM_INPUT_TOKENS=128
export GUIDELLM_OUTPUT_TOKENS=128
export GUIDELLM_INPUT_STDEV=10
export GUIDELLM_OUTPUT_STDEV=10
export GUIDELLM_MAX_REQUESTS=500
export GUIDELLM_SEED=42
export NUM_RUNS=3
bash scripts/run_guidellm_c_config_sweep.sh

# Parse results
python3 parse_guidellm_results.py logs/<results_dir>
```
