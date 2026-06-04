# Kimi K2.5 Speculative Decoding Benchmark Results

**Date:** 2026-05-22  
**Hardware:** 8x NVIDIA H200 SXM (RunPod)  
**Software:** vLLM 0.21.0, CUDA 13, torch 2.11.0+cu130  
**Model:** moonshotai/Kimi-K2.5 (compressed-tensors WNA16 Marlin MoE, ~595 GB)  
**API:** `/v1/chat/completions`  
**Concurrency:** 16  
**Requests:** 200 per run  

## Speculative Decoding Heads

| Head | Model | Size | Method |
|------|-------|------|--------|
| Eagle3 | lightseekorg/kimi-k2.5-eagle3-mla | ~6 GB | eagle3 (MTP with MLA) |
| DFlash | z-lab/Kimi-K2.5-DFlash | ~6 GB | dflash (block diffusion) |

## Results: 1024 input / 512 output

| Config | Out tok/s | Req/s | E2EL (ms) | TPOT (ms) | vs Baseline |
|--------|-----------|-------|-----------|-----------|-------------|
| baseline | 1,157 | 2.1 | 7,000 | 13.6 | — |
| eagle3-k3 | 1,421 | 2.6 | 5,900 | 11.6 | +23% tok/s |
| dflash-k3 | 1,339 | 2.4 | 6,200 | 12.1 | +16% tok/s |
| dflash-k8 | 1,127 | 2.0 | 7,400 | 14.4 | -3% tok/s |

## Results: 1024 input / 2048 output

| Config | Out tok/s | Req/s | E2EL (ms) | TPOT (ms) | TTFT (ms) | ITL (ms) | vs Baseline |
|--------|-----------|-------|-----------|-----------|-----------|----------|-------------|
| baseline | 1,067 | 0.5 | 29,300 | 14.3 | 8,441 | 9.9 | — |
| eagle3-k3 | 1,521 | 0.7 | 20,500 | 10.0 | 6,276 | 6.6 | +43% tok/s |
| dflash-k3 | 1,447 | 0.7 | 21,300 | 10.3 | 6,516 | 6.8 | +36% tok/s |

## Key Findings

1. **Eagle3 k=3 is the best spec decode method on vLLM** for Kimi K2.5 — +23% at 512 output, +43% at 2048 output.
2. **DFlash k=3 is close behind** — +16% at 512, +36% at 2048. Gap narrows with longer output.
3. **DFlash k=8 hurt performance** due to vLLM scheduler throttling (`max_num_scheduled_tokens` capped to 1024).
4. **Longer output amplifies spec decode benefit** — gains nearly doubled from 512 to 2048 tokens.
5. **TTFT/ITL are unreliable at 512 output** (reported 0.0 due to Kimi reasoning parser chunking). Reliable at 2048.

## Notes

- DFlash was designed for SGLang; vLLM support is via PR39930 and may not be fully optimized.
- Eagle3's MLA architecture matches Kimi K2.5's native MLA, reducing KV cache overhead.
- Kimi K2.5 uses compressed-tensors WNA16 Marlin MoE — do NOT pass `--quantization fp8`.
- `/v1/completions` (raw text) gives very low acceptance for Eagle3; use `/v1/chat/completions`.

## Serve Commands

### Baseline
```bash
vllm serve moonshotai/Kimi-K2.5 \
  --tensor-parallel-size 8 --mm-encoder-tp-mode data --trust-remote-code \
  --tool-call-parser kimi_k2 --reasoning-parser kimi_k2 \
  --gpu-memory-utilization 0.95 --max-model-len 8192 \
  --host 0.0.0.0 --port 8000
```

### Eagle3 k=3
```bash
vllm serve moonshotai/Kimi-K2.5 \
  --tensor-parallel-size 8 --mm-encoder-tp-mode data --trust-remote-code \
  --tool-call-parser kimi_k2 --reasoning-parser kimi_k2 \
  --gpu-memory-utilization 0.95 --max-model-len 8192 \
  --host 0.0.0.0 --port 8000 \
  --speculative-config '{"model":"lightseekorg/kimi-k2.5-eagle3-mla","method":"eagle3","num_speculative_tokens":3}'
```

### DFlash k=3
```bash
vllm serve moonshotai/Kimi-K2.5 \
  --tensor-parallel-size 8 --mm-encoder-tp-mode data --trust-remote-code \
  --tool-call-parser kimi_k2 --reasoning-parser kimi_k2 \
  --gpu-memory-utilization 0.95 --max-model-len 8192 \
  --host 0.0.0.0 --port 8000 \
  --speculative-config '{"method":"dflash","model":"z-lab/Kimi-K2.5-DFlash","num_speculative_tokens":3}'
```

### GuideLLM Benchmark
```bash
guidellm benchmark run \
  --target http://localhost:8000 \
  --model moonshotai/Kimi-K2.5 \
  --request-format "/v1/chat/completions" \
  --data "prompt_tokens=1024,output_tokens=2048,source=data:prideandprejudice.txt.gz" \
  --profile concurrent \
  --rate 16 \
  --max-requests 200 \
  --random-seed 42 \
  --outputs /workspace/bench_logs/<config>.json
```
