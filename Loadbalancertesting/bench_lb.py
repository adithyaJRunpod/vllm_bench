#!/usr/bin/env python3
"""
Load Balancer Streaming Benchmark
==================================
Mimics run_guidellm_workload_sweep.sh but targets the RunPod Load Balancer
endpoint directly using /v1/completions with SSE streaming.

Single workload: 256 input tokens, 1024 max output tokens.

Usage:
    python bench_lb.py \
        --base-url https://<ENDPOINT_ID>.api.runpod.ai \
        --api-key <RUNPOD_API_KEY> \
        --num-requests 50 \
        --concurrency 8

Or via environment variables:
    RUNPOD_API_KEY=rpa_xxx \
    RUNPOD_ENDPOINT=jcja1rjzitd515 \
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


async def send_streaming_request(client, url, headers, payload, results, semaphore, req_id):
    async with semaphore:
        start = time.perf_counter()
        ttft = None
        chunks = []
        total_text = ""
        finish_reason = None

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
                    if ttft is None:
                        ttft = now - start

                    try:
                        chunk = json.loads(data_part)
                        chunks.append(now)
                        total_text = chunk.get("text", total_text)
                        finish_reason = chunk.get("finish_reason", finish_reason)
                    except json.JSONDecodeError:
                        continue

            end = time.perf_counter()
            itl_values = []
            for i in range(1, len(chunks)):
                itl_values.append((chunks[i] - chunks[i - 1]) * 1000)

            results.append({
                "id": req_id,
                "status": "success",
                "latency": end - start,
                "ttft_ms": ttft * 1000 if ttft else 0,
                "itl_ms": itl_values,
                "num_chunks": len(chunks),
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

    separator = "=" * 64
    print(f"\n{separator}")
    print(f"  RunPod Load Balancer Benchmark")
    print(f"  Workload: {WORKLOAD['name']}")
    print(f"{separator}")
    print(f"  Endpoint:     {url}")
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
            send_streaming_request(client, url, headers, payload, results, semaphore, i)
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

        total_output_tokens = sum(num_chunks)
        print(f"\n  --- Throughput ---")
        print(f"    Requests/s:         {len(successes) / total_time:.2f}")
        print(f"    Output tokens/s:    {total_output_tokens / total_time:.1f}")
        print(f"    Avg tokens/request: {statistics.mean(num_chunks):.1f}")
        print(f"    Total output tokens:{total_output_tokens}")

    print(f"\n{separator}\n")

    if args.output_file:
        report = {
            "timestamp": datetime.now().isoformat(),
            "config": {
                "base_url": args.base_url,
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
        description="RunPod Load Balancer Streaming Benchmark (256in / 1024out)"
    )
    parser.add_argument(
        "--base-url",
        default=os.environ.get("RUNPOD_BASE_URL")
        or (
            f"https://{os.environ['RUNPOD_ENDPOINT']}.api.runpod.ai"
            if os.environ.get("RUNPOD_ENDPOINT")
            else None
        ),
        help="Base URL of the LB endpoint (or set RUNPOD_ENDPOINT env var)",
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

    if not args.base_url:
        parser.error("--base-url is required (or set RUNPOD_ENDPOINT env var)")
    if not args.api_key:
        parser.error("--api-key is required (or set RUNPOD_API_KEY env var)")

    asyncio.run(run_benchmark(args))


if __name__ == "__main__":
    main()
