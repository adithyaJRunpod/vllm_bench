# vllm_bench

Benchmark scripts for vLLM on RunPod H100 GPUs.

## Pod Setup (fresh pod)

```bash
# Install vLLM
pip install vllm

# Clone repo & fix line endings
cd /workspace
git clone https://github.com/adithyaJRunpod/vllm_bench.git
cd vllm_bench
sed -i 's/\r$//' scripts/*.sh

# Set env vars (adjust for your workload)
export VLLM_MODEL=Qwen/Qwen3-8B
export VLLM_REQUEST_RATE=30
export VLLM_MAX_CONCURRENCY=32
export VLLM_INPUT_LEN=2048
export VLLM_OUTPUT_LEN=512

# Run sweep
bash scripts/run_tuning_sweep.sh

# Parse results
python3 parse_bench_logs.py tuning
```
