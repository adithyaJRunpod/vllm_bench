#!/usr/bin/env python3
"""
Deploy a vLLM serverless endpoint on RunPod.

Creates a template + endpoint with properly serialized env vars,
avoiding JSON escaping issues from the RunPod UI.

Usage:
    export RUNPOD_API_KEY=rpa_xxx
    python scripts/deploy_runpod.py configs/meta/llama3.1_8b_instruct.yaml

    # Or with explicit options:
    python scripts/deploy_runpod.py configs/meta/llama3.1_8b_instruct.yaml \
        --name "llama-8b-fp8-eagle3" \
        --gpu AMPERE_80 \
        --hf-token hf_xxx
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import requests
import yaml


RUNPOD_GRAPHQL_URL = "https://api.runpod.io/graphql"
VLLM_WORKER_IMAGE = "runpod/worker-v1-vllm:v0.20.2"

GPU_CHOICES = ["AMPERE_80", "ADA_80_PRO"]


def graphql(api_key: str, query: str, variables: dict | None = None) -> dict:
    resp = requests.post(
        RUNPOD_GRAPHQL_URL,
        headers={"Content-Type": "application/json"},
        params={"api_key": api_key},
        json={"query": query, "variables": variables or {}},
        timeout=30,
    )
    resp.raise_for_status()
    data = resp.json()
    if "errors" in data:
        raise RuntimeError(f"GraphQL errors: {json.dumps(data['errors'], indent=2)}")
    return data["data"]


def load_yaml_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def yaml_to_env_vars(config: dict, hf_token: str | None = None) -> list[dict]:
    """Convert vLLM CLI-style YAML config to RunPod env var list."""
    key_map = {
        "model": "MODEL_NAME",
        "gpu-memory-utilization": "GPU_MEMORY_UTILIZATION",
        "max-model-len": "MAX_MODEL_LEN",
        "dtype": "DTYPE",
        "trust-remote-code": "TRUST_REMOTE_CODE",
        "quantization": "QUANTIZATION",
        "kv-cache-dtype": "KV_CACHE_DTYPE",
        "speculative-config": "SPECULATIVE_CONFIG",
        "tensor-parallel-size": "TENSOR_PARALLEL_SIZE",
        "enable-prefix-caching": "ENABLE_PREFIX_CACHING",
        "enforce-eager": "ENFORCE_EAGER",
        "max-num-seqs": "MAX_NUM_SEQS",
        "enable-chunked-prefill": "ENABLE_CHUNKED_PREFILL",
        "mm-encoder-tp-mode": "MM_ENCODER_TP_MODE",
        "tool-call-parser": "TOOL_CALL_PARSER",
        "reasoning-parser": "REASONING_PARSER",
    }

    env = {}
    for key, value in config.items():
        env_key = key_map.get(key, key.upper().replace("-", "_"))

        if env_key == "SPECULATIVE_CONFIG" and isinstance(value, str):
            # Parse YAML string value then re-serialize as clean JSON
            try:
                parsed = json.loads(value)
                env[env_key] = json.dumps(parsed)
            except json.JSONDecodeError:
                env[env_key] = value
        else:
            env[env_key] = str(value)

    # Set SERVED_MODEL_NAME to match MODEL_NAME
    if "MODEL_NAME" in env:
        env.setdefault("SERVED_MODEL_NAME", env["MODEL_NAME"])

    # Sensible defaults for production
    env.setdefault("ENFORCE_EAGER", "false")
    env.setdefault("ENABLE_PREFIX_CACHING", "true")

    if hf_token:
        env["HF_TOKEN"] = hf_token

    return [{"key": k, "value": v} for k, v in env.items()]


def create_template(api_key: str, name: str, env_vars: list[dict]) -> str:
    """Create a RunPod serverless template and return its ID."""
    env_gql = ", ".join(
        f'{{ key: {json.dumps(e["key"])}, value: {json.dumps(e["value"])} }}'
        for e in env_vars
    )

    query = f"""
    mutation {{
      saveTemplate(input: {{
        containerDiskInGb: 20,
        dockerArgs: "",
        env: [{env_gql}],
        imageName: "{VLLM_WORKER_IMAGE}",
        isServerless: true,
        name: "{name}",
        volumeInGb: 0,
        volumeMountPath: "/runpod-volume"
      }}) {{
        id
        name
      }}
    }}
    """
    data = graphql(api_key, query)
    template = data["saveTemplate"]
    print(f"  Template created: id={template['id']} name={template['name']}")
    return template["id"]


def create_endpoint(
    api_key: str,
    name: str,
    template_id: str,
    gpu_ids: str = "AMPERE_80",
    workers_min: int = 0,
    workers_max: int = 2,
    idle_timeout: int = 60,
) -> dict:
    """Create a RunPod serverless endpoint and return its details."""
    query = f"""
    mutation {{
      saveEndpoint(input: {{
        gpuIds: "{gpu_ids}",
        idleTimeout: {idle_timeout},
        name: "{name}",
        templateId: "{template_id}",
        workersMax: {workers_max},
        workersMin: {workers_min},
        scalerType: "QUEUE_DELAY",
        scalerValue: 4
      }}) {{
        id
        name
        gpuIds
        templateId
        workersMax
        workersMin
        idleTimeout
      }}
    }}
    """
    data = graphql(api_key, query)
    endpoint = data["saveEndpoint"]
    print(f"  Endpoint created: id={endpoint['id']} name={endpoint['name']}")
    return endpoint


def test_endpoint(api_key: str, endpoint_id: str, model_name: str) -> bool:
    """Send a test request to verify the endpoint is working."""
    url = f"https://api.runpod.ai/v2/{endpoint_id}/openai/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": model_name,
        "messages": [{"role": "user", "content": "Say hello in one word."}],
        "max_tokens": 5,
    }

    print(f"\n  Sending test request (this may take 60-120s for cold start)...")
    try:
        resp = requests.post(url, headers=headers, json=payload, timeout=180)
        if resp.status_code == 200:
            data = resp.json()
            content = data["choices"][0]["message"]["content"]
            print(f"  Response: {content}")
            return True
        else:
            print(f"  HTTP {resp.status_code}: {resp.text[:200]}")
            return False
    except requests.Timeout:
        print("  Timed out (worker may still be starting)")
        return False


def main():
    parser = argparse.ArgumentParser(description="Deploy vLLM endpoint on RunPod")
    parser.add_argument("config", help="Path to vLLM YAML config file")
    parser.add_argument("--name", help="Endpoint name (default: derived from config)")
    parser.add_argument(
        "--gpu",
        default="AMPERE_80",
        choices=GPU_CHOICES,
        help="GPU tier (default: AMPERE_80 = H100/A100 80GB)",
    )
    parser.add_argument("--workers-min", type=int, default=0)
    parser.add_argument("--workers-max", type=int, default=2)
    parser.add_argument("--idle-timeout", type=int, default=60)
    parser.add_argument("--hf-token", help="HuggingFace token (or set HF_TOKEN env var)")
    parser.add_argument(
        "--test", action="store_true", help="Send a test request after deployment"
    )
    args = parser.parse_args()

    api_key = os.environ.get("RUNPOD_API_KEY")
    if not api_key:
        print("Error: RUNPOD_API_KEY environment variable not set")
        sys.exit(1)

    hf_token = args.hf_token or os.environ.get("HF_TOKEN")

    config_path = Path(args.config)
    if not config_path.exists():
        print(f"Error: Config file not found: {config_path}")
        sys.exit(1)

    config = load_yaml_config(config_path)
    model_name = config.get("model", "unknown")
    endpoint_name = args.name or f"vllm-{config_path.stem}"

    print(f"Deploying: {model_name}")
    print(f"Config: {config_path}")
    print(f"GPU: {args.gpu}")
    print()

    # Build env vars
    env_vars = yaml_to_env_vars(config, hf_token)
    print("Environment variables:")
    for e in env_vars:
        val = e["value"]
        if e["key"] == "HF_TOKEN":
            val = val[:8] + "..."
        print(f"  {e['key']}={val}")
    print()

    # Create template
    print("Creating template...")
    template_id = create_template(api_key, f"tpl-{endpoint_name}", env_vars)

    # Create endpoint
    print("Creating endpoint...")
    endpoint = create_endpoint(
        api_key,
        endpoint_name,
        template_id,
        gpu_ids=args.gpu,
        workers_min=args.workers_min,
        workers_max=args.workers_max,
        idle_timeout=args.idle_timeout,
    )

    endpoint_id = endpoint["id"]
    base_url = f"https://api.runpod.ai/v2/{endpoint_id}/openai"

    print()
    print("=" * 60)
    print(f"  Endpoint ID:   {endpoint_id}")
    print(f"  Base URL:      {base_url}")
    print(f"  Model:         {model_name}")
    print(f"  GPU:           {args.gpu}")
    print(f"  Workers:       {args.workers_min}-{args.workers_max}")
    print("=" * 60)
    print()
    print("To use with your benchmark script:")
    print(f"  export RUNPOD_ENDPOINT={endpoint_id}")
    print(f"  export MODEL={model_name}")
    print(f"  bash scripts/run_guidellm_workload_sweep.sh")

    if args.test:
        print()
        test_endpoint(api_key, endpoint_id, model_name)


if __name__ == "__main__":
    main()
