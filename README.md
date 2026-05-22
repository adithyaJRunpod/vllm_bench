# vllm_bench

Benchmark scripts for vLLM on RunPod H100 GPUs.

## Pod Setup (fresh pod)

```bash
# Clone repo & fix line endings
cd /workspace
git clone https://github.com/adithyaJRunpod/vllm_bench.git
cd vllm_bench
find scripts -name '*.sh' -exec sed -i 's/\r$//' {} +

# Install pinned deps (vLLM 0.21.0 — required for MiniMax M2.5 + DFlash)
pip install -r requirements.txt

# RunPod sets HF_HUB_ENABLE_HF_TRANSFER=1; hf_transfer in requirements.txt
# satisfies that. To disable fast download instead: export HF_HUB_ENABLE_HF_TRANSFER=0

# Pre-download large models (recommended for 200GB+ weights)
# huggingface-cli login   # required for gated models (e.g. DFlash head)
# huggingface-cli download MiniMaxAI/MiniMax-M2.5
# huggingface-cli download z-lab/MiniMax-M2.5-DFlash

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
