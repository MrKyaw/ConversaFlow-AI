"""Optional latency probe against a running ConversaFlow server."""

from __future__ import annotations

import argparse
import statistics
import time

import httpx


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--api-key", default="dev-key-123")
    parser.add_argument("--requests", type=int, default=50)
    args = parser.parse_args()

    headers = {"X-API-Key": args.api_key}
    with httpx.Client(base_url=args.base_url, headers=headers, timeout=10.0) as client:
        session = client.post("/v1/sessions", json={}).json()["session_id"]
        latencies = []
        for _ in range(args.requests):
            start = time.perf_counter()
            client.post("/v1/turns", json={"session_id": session, "text": "hello"})
            latencies.append((time.perf_counter() - start) * 1000.0)
    latencies.sort()
    print(f"requests={len(latencies)}")
    print(f"p50={statistics.median(latencies):.2f}ms")
    print(f"p95={latencies[int(0.95 * len(latencies)) - 1]:.2f}ms")
    print(f"mean={statistics.mean(latencies):.2f}ms")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
