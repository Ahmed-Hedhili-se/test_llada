"""Quick server health check and smoke test.

Usage:
    python3 -m eval.check_server --base-url http://localhost:8000
"""

import argparse
import json
import sys
import urllib.request


def check(base_url: str):
    base_url = base_url.rstrip("/")

    # Health
    try:
        r = urllib.request.urlopen(f"{base_url}/health", timeout=5)
        print(f"  /health: {r.status} {json.loads(r.read())}")
    except Exception as e:
        print(f"  /health FAILED: {e}")
        sys.exit(1)

    # Smoke generation
    payload = json.dumps({
        "model": "inclusionAI/LLaDA-MoE-7B-A1B-Instruct",
        "messages": [{"role": "user", "content": "What is 2 + 2?"}],
        "max_tokens": 64,
        "steps": 32,
        "block_length": 32,
        "temperature": 0.0,
    }).encode()

    req = urllib.request.Request(
        f"{base_url}/v1/chat/completions",
        data=payload,
        headers={"Content-Type": "application/json"},
    )
    try:
        r = urllib.request.urlopen(req, timeout=120)
        data = json.loads(r.read())
        content = data["choices"][0]["message"]["content"]
        tokens = data["usage"]["completion_tokens"]
        timing = data.get("timing", {})
        print(f"  Generation OK: {repr(content[:120])}")
        print(f"  Tokens: {tokens}  |  {timing}")
    except Exception as e:
        print(f"  Generation FAILED: {e}")
        sys.exit(1)

    print("\nServer is healthy and generating correctly.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default="http://localhost:8000")
    args = ap.parse_args()
    print(f"Checking server at {args.base_url}...")
    check(args.base_url)


if __name__ == "__main__":
    main()
