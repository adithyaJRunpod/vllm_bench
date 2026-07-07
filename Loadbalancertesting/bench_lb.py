#!/usr/bin/env python3
"""
RunPod Streaming Benchmark
============================
Benchmarks both Load Balancer and Serverless RunPod endpoints
using SSE streaming to measure TTFT, ITL, E2E latency, and throughput.

Supports two modes:
  - lb:         Load Balancer endpoint (subdomain URL, custom SSE format)
  - serverless: Serverless endpoint (path-based URL, OpenAI SSE format)

Single workload: 256 input tokens, 1024 max output tokens.

Usage:
    # Load Balancer mode (default)
    python bench_lb.py \
        --mode lb \
        --endpoint-id <ENDPOINT_ID> \
        --api-key <RUNPOD_API_KEY>

    # Serverless mode
    python bench_lb.py \
        --mode serverless \
        --endpoint-id <ENDPOINT_ID> \
        --api-key <RUNPOD_API_KEY>

    # Auto-detect from base-url
    python bench_lb.py \
        --base-url https://<ENDPOINT_ID>.api.runpod.ai \
        --api-key <RUNPOD_API_KEY>

Or via environment variables:
    RUNPOD_API_KEY=rpa_xxx \
    RUNPOD_ENDPOINT=jcja1rjzitd515 \
    RUNPOD_MODE=serverless \
    python bench_lb.py
"""

import asyncio
import time
import json
import os
import sys
import statistics
import argparse
from datetime import datetime

try:
    import httpx
except ImportError:
    print("ERROR: httpx is required. Install with: pip install httpx")
    sys.exit(1)


WORKLOAD = {
    "name": "decode-heavy",
    "input_tokens": 256,
    "output_tokens": 1024,
    "concurrency": 8,
    "num_requests": 50,
}

PROMPT_FILLER = (
    "Explain in detail the architecture of modern distributed computing systems, "
    "including concepts like consensus algorithms, fault tolerance, data replication, "
    "sharding strategies, load balancing mechanisms, and the trade-offs between "
    "consistency and availability in large-scale deployments. "
)


def build_prompt(target_tokens: int) -> str:
    chars_per_token = 4
    target_len = target_tokens * chars_per_token
    repeats = (target_len // len(PROMPT_FILLER)) + 1
    return (PROMPT_FILLER * repeats)[:target_len]


async def send_streaming_request(client, url, headers, payload, results, semaphore, req_id, mode="lb"):
    async with semaphore:
        start = time.perf_counter()
        ttft = None
        chunks = []
        total_text = ""
        finish_reason = None
        usage_tokens = None

        try:
            async with client.stream("POST", url, headers=headers, json=payload) as resp:
                if resp.status_code != 200:
                    body = await resp.aread()
                    results.append({
                        "id": req_id,
                        "status": "error",
                        "code": resp.status_code,
                        "body": body.decode()[:200],
                    })
                    return

                async for line in resp.aiter_lines():
                    if not line or not line.startswith("data: "):
                        continue

                    data_part = line[6:]
                    if data_part == "[DONE]":
                        break

                    now = time.perf_counter()

                    try:
                        chunk = json.loads(data_part)
                    except json.JSONDecodeError:
                        continue

                    if mode == "serverless":
                        choices = chunk.get("choices", [])
                        if not choices:
                            continue
                        delta = choices[0].get("delta", {})
                        content = delta.get("content")
                        if content:
                            if ttft is None:
                                ttft = now - start
                            chunks.append(now)
                            total_text += content
                        fr = choices[0].get("finish_reason")
                        if fr:
                            finish_reason = fr
                        if chunk.get("usage"):
                            usage_tokens = chunk["usage"].get("completion_tokens")
                    else:
                        if ttft is None:
                            ttft = now - start
                        chunks.append(now)
                        total_text = chunk.get("text", total_text)
                        finish_reason = chunk.get("finish_reason", finish_reason)

            end = time.perf_counter()
            itl_values = []
            for i in range(1, len(chunks)):
                itl_values.append((chunks[i] - chunks[i - 1]) * 1000)

            if usage_tokens:
                num_tokens = usage_tokens
            else:
                num_tokens = max(1, len(total_text) // 4)

            results.append({
                "id": req_id,
                "status": "success",
                "latency": end - start,
                "ttft_ms": ttft * 1000 if ttft else 0,
                "itl_ms": itl_values,
                "num_chunks": len(chunks),
                "num_tokens": num_tokens,
                "output_len": len(total_text),
                "finish_reason": finish_reason,
            })

        except Exception as e:
            results.append({
                "id": req_id,
                "status": "error",
                "code": str(type(e).__name__),
                "body": str(e)[:200],
            })


async def run_benchmark(args):
    mode = args.mode

    if mode == "lb":
        url = f"{args.base_url.rstrip('/')}/v1/completions"
    else:
        url = f"{args.base_url.rstrip('/')}/v1/completions"

    headers = {
        "Authorization": f"Bearer {args.api_key}",
        "Content-Type": "application/json",
    }

    prompt_text = build_prompt(args.input_tokens)
    payload = {
        "prompt": prompt_text,
        "max_tokens": args.output_tokens,
        "temperature": 0.7,
        "top_p": 0.9,
        "stream": True,
    }

    if mode == "serverless":
        payload["stream_options"] = {"include_usage": True}

    separator = "=" * 64
    print(f"\n{separator}")
    print(f"  RunPod Benchmark ({mode.upper()} mode)")
    print(f"  Workload: {WORKLOAD['name']}")
    print(f"{separator}")
    print(f"  Endpoint:     {url}")
    print(f"  Mode:         {mode}")
    print(f"  Requests:     {args.num_requests}")
    print(f"  Concurrency:  {args.concurrency}")
    print(f"  Input tokens: ~{args.input_tokens}")
    print(f"  Output tokens: max {args.output_tokens}")
    print(f"  Timestamp:    {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{separator}\n")

    print("  Sending requests...", end="", flush=True)

    results = []
    semaphore = asyncio.Semaphore(args.concurrency)

    start_time = time.perf_counter()
    async with httpx.AsyncClient(timeout=300.0) as client:
        tasks = [
            send_streaming_request(client, url, headers, payload, results, semaphore, i, mode)
            for i in range(args.num_requests)
        ]
        await asyncio.gather(*tasks)
    total_time = time.perf_counter() - start_time

    print(" done.\n")

    successes = [r for r in results if r["status"] == "success"]
    errors = [r for r in results if r["status"] == "error"]

    print(f"{separator}")
    print(f"  RESULTS")
    print(f"{separator}")
    print(f"  Total requests:  {len(results)}")
    print(f"  Successful:      {len(successes)}")
    print(f"  Errors:          {len(errors)}")
    print(f"  Total duration:  {total_time:.2f}s")

    if errors:
        print(f"\n  --- Errors (first 3) ---")
        for e in errors[:3]:
            print(f"    [{e.get('code')}] {e.get('body', '')[:100]}")

    if successes:
        latencies = [r["latency"] for r in successes]
        ttfts = [r["ttft_ms"] for r in successes]
        all_itls = [itl for r in successes for itl in r["itl_ms"]]
        num_chunks = [r["num_chunks"] for r in successes]

        def percentile(data, pct):
            idx = min(int(len(data) * pct / 100), len(data) - 1)
            return sorted(data)[idx]

        print(f"\n  --- Latency (end-to-end) ---")
        print(f"    Min:     {min(latencies):.3f}s")
        print(f"    Median:  {statistics.median(latencies):.3f}s")
        print(f"    Mean:    {statistics.mean(latencies):.3f}s")
        print(f"    P90:     {percentile(latencies, 90):.3f}s")
        print(f"    P95:     {percentile(latencies, 95):.3f}s")
        print(f"    P99:     {percentile(latencies, 99):.3f}s")
        print(f"    Max:     {max(latencies):.3f}s")

        print(f"\n  --- TTFT (Time to First Token) ---")
        print(f"    Min:     {min(ttfts):.1f}ms")
        print(f"    Median:  {statistics.median(ttfts):.1f}ms")
        print(f"    Mean:    {statistics.mean(ttfts):.1f}ms")
        print(f"    P90:     {percentile(ttfts, 90):.1f}ms")
        print(f"    P95:     {percentile(ttfts, 95):.1f}ms")
        print(f"    P99:     {percentile(ttfts, 99):.1f}ms")
        print(f"    Max:     {max(ttfts):.1f}ms")

        if all_itls:
            print(f"\n  --- ITL (Inter-Token Latency) ---")
            print(f"    Min:     {min(all_itls):.1f}ms")
            print(f"    Median:  {statistics.median(all_itls):.1f}ms")
            print(f"    Mean:    {statistics.mean(all_itls):.1f}ms")
            print(f"    P90:     {percentile(all_itls, 90):.1f}ms")
            print(f"    P95:     {percentile(all_itls, 95):.1f}ms")
            print(f"    P99:     {percentile(all_itls, 99):.1f}ms")
            print(f"    Max:     {max(all_itls):.1f}ms")

        num_tokens = [r["num_tokens"] for r in successes]
        total_output_tokens = sum(num_tokens)
        token_source = "actual" if mode == "serverless" else "estimated (len/4)"
        print(f"\n  --- Throughput ---")
        print(f"    Requests/s:         {len(successes) / total_time:.2f}")
        print(f"    Output tokens/s:    {total_output_tokens / total_time:.1f}")
        print(f"    Avg tokens/request: {statistics.mean(num_tokens):.1f} ({token_source})")
        print(f"    Total output tokens:{total_output_tokens}")
        print(f"    Avg chunks/request: {statistics.mean(num_chunks):.1f} (decode steps)")

    print(f"\n{separator}\n")

    if args.output_file:
        report = {
            "timestamp": datetime.now().isoformat(),
            "config": {
                "base_url": args.base_url,
                "mode": mode,
                "workload": WORKLOAD["name"],
                "input_tokens": args.input_tokens,
                "output_tokens": args.output_tokens,
                "num_requests": args.num_requests,
                "concurrency": args.concurrency,
            },
            "summary": {
                "total_time_s": total_time,
                "success_count": len(successes),
                "error_count": len(errors),
            },
            "results": results,
        }
        if successes:
            report["summary"]["latency_median_s"] = statistics.median(latencies)
            report["summary"]["latency_p95_s"] = percentile(latencies, 95)
            report["summary"]["ttft_median_ms"] = statistics.median(ttfts)
            report["summary"]["ttft_p95_ms"] = percentile(ttfts, 95)
            if all_itls:
                report["summary"]["itl_median_ms"] = statistics.median(all_itls)
                report["summary"]["itl_p95_ms"] = percentile(all_itls, 95)
            report["summary"]["throughput_req_per_s"] = len(successes) / total_time
            report["summary"]["throughput_tok_per_s"] = total_output_tokens / total_time

        with open(args.output_file, "w") as f:
            json.dump(report, f, indent=2)
        print(f"  Results saved to: {args.output_file}")


def main():
    parser = argparse.ArgumentParser(
        description="RunPod Streaming Benchmark (LB & Serverless) - 256in / 1024out"
    )
    parser.add_argument(
        "--mode",
        choices=["lb", "serverless"],
        default=os.environ.get("RUNPOD_MODE", "lb"),
        help="Endpoint mode: 'lb' for Load Balancer, 'serverless' for queue-based (default: lb)",
    )
    parser.add_argument(
        "--endpoint-id",
        default=os.environ.get("RUNPOD_ENDPOINT"),
        help="RunPod endpoint ID (or set RUNPOD_ENDPOINT env var)",
    )
    parser.add_argument(
        "--base-url",
        default=os.environ.get("RUNPOD_BASE_URL"),
        help="Full base URL (overrides --endpoint-id and --mode auto-detection)",
    )
    parser.add_argument(
        "--api-key",
        default=os.environ.get("RUNPOD_API_KEY"),
        help="RunPod API key (or set RUNPOD_API_KEY env var)",
    )
    parser.add_argument(
        "--num-requests",
        type=int,
        default=WORKLOAD["num_requests"],
        help=f"Total number of requests (default: {WORKLOAD['num_requests']})",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=WORKLOAD["concurrency"],
        help=f"Max concurrent requests (default: {WORKLOAD['concurrency']})",
    )
    parser.add_argument(
        "--input-tokens",
        type=int,
        default=WORKLOAD["input_tokens"],
        help=f"Approx input token count (default: {WORKLOAD['input_tokens']})",
    )
    parser.add_argument(
        "--output-tokens",
        type=int,
        default=WORKLOAD["output_tokens"],
        help=f"Max output tokens (default: {WORKLOAD['output_tokens']})",
    )
    parser.add_argument(
        "--output-file",
        default=None,
        help="Path to save JSON results (optional)",
    )

    args = parser.parse_args()

    if not args.api_key:
        parser.error("--api-key is required (or set RUNPOD_API_KEY env var)")

    if not args.base_url:
        if not args.endpoint_id:
            parser.error("--endpoint-id or --base-url is required (or set RUNPOD_ENDPOINT env var)")
        if args.mode == "lb":
            args.base_url = f"https://{args.endpoint_id}.api.runpod.ai"
        else:
            args.base_url = f"https://api.runpod.ai/v2/{args.endpoint_id}/openai"
    else:
        if "api.runpod.ai/v2/" in args.base_url:
            args.mode = "serverless"
        elif ".api.runpod.ai" in args.base_url:
            args.mode = "lb"

    asyncio.run(run_benchmark(args))


if __name__ == "__main__":
    main()
