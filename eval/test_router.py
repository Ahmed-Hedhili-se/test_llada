"""Routing behaviour of src/router.py, against mock replicas.

Deliberately does not need a GPU or the 14 GiB checkpoint. What is being tested
is dispatch policy, health tracking and failure handling -- none of which
involve the model, and all of which are the parts that decide whether an
8-GPU deployment actually spreads load or quietly funnels it onto one card.

The mock replicas hold a request open for a configurable delay, which is what
makes the least-outstanding policy observable: with instant responses every
policy looks identical, because nothing is ever concurrently in flight.
"""

from __future__ import annotations

import asyncio
import os
import sys
import threading
import time

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class MockReplica:
    """A stand-in server: /health, plus a /v1/chat/completions that sleeps."""

    def __init__(self, port: int, delay: float = 0.30, healthy: bool = True):
        self.port, self.delay, self.healthy = port, delay, healthy
        self.received = 0
        self.concurrent = 0
        self.max_concurrent = 0
        self._thread = None
        self._server = None

    def _build(self):
        from fastapi import FastAPI, Response
        app = FastAPI()

        @app.get("/health")
        def health():
            if not self.healthy:
                return Response(status_code=503)
            return {"status": "ok"}

        @app.get("/v1/quantization")
        def quant():
            return {"quantized": False, "port": self.port}

        @app.post("/v1/chat/completions")
        async def chat(body: dict):
            self.received += 1
            self.concurrent += 1
            self.max_concurrent = max(self.max_concurrent, self.concurrent)
            try:
                await asyncio.sleep(self.delay)
                return {"served_by": self.port, "echo": body.get("marker")}
            finally:
                self.concurrent -= 1

        return app

    def start(self):
        import uvicorn
        cfg = uvicorn.Config(self._build(), host="127.0.0.1", port=self.port,
                             log_level="error")
        self._server = uvicorn.Server(cfg)
        self._thread = threading.Thread(target=self._server.run, daemon=True)
        self._thread.start()

    def stop(self):
        if self._server:
            self._server.should_exit = True


def _wait_until(pred, timeout=15.0, what="condition"):
    end = time.time() + timeout
    while time.time() < end:
        if pred():
            return
        time.sleep(0.1)
    raise AssertionError(f"timed out waiting for {what}")


def _run_router(pool_urls, timeout=30.0):
    """Start the router in-process against the given backends."""
    import uvicorn
    import src.router as router

    router.POOL = router.Pool(pool_urls, timeout=timeout)
    cfg = uvicorn.Config(router.app, host="127.0.0.1", port=8399, log_level="error")
    server = uvicorn.Server(cfg)
    t = threading.Thread(target=server.run, daemon=True)
    t.start()
    return router, server


def main() -> int:
    import requests

    replicas = [MockReplica(8391, delay=0.30), MockReplica(8392, delay=0.30)]
    for r in replicas:
        r.start()
    _wait_until(lambda: all(
        _ok(f"http://127.0.0.1:{r.port}/health") for r in replicas
    ), what="mock replicas")

    router, server = _run_router([f"http://127.0.0.1:{r.port}" for r in replicas])
    base = "http://127.0.0.1:8399"
    _wait_until(lambda: _ok(f"{base}/health"), what="router")
    _wait_until(lambda: requests.get(f"{base}/health", timeout=5).json()
                ["healthy_replicas"] == 2, what="both replicas marked healthy")
    print("  [ok] router starts and discovers both replicas")

    # --- load is actually spread -------------------------------------------
    import concurrent.futures as cf
    N = 16
    with cf.ThreadPoolExecutor(max_workers=N) as ex:
        futs = [ex.submit(requests.post, f"{base}/v1/chat/completions",
                          json={"marker": i}, timeout=60) for i in range(N)]
        results = [f.result() for f in futs]
    assert all(r.status_code == 200 for r in results), "some requests failed"
    counts = [r.received for r in replicas]
    assert sum(counts) == N, f"replicas saw {sum(counts)} of {N} requests"
    # Least-outstanding should split an even burst near-evenly. Allow slack for
    # scheduling jitter, but a 16/0 split would mean the policy is not working.
    assert min(counts) >= N // 4, f"load not spread: {counts}"
    print(f"  [ok] {N} concurrent requests spread across replicas: {counts}")
    assert max(r.max_concurrent for r in replicas) > 1, (
        "no replica ever had >1 in flight -- the mock delay is too short to "
        "make the policy observable"
    )
    print(f"  [ok] concurrency reached replicas "
          f"(max in flight: {[r.max_concurrent for r in replicas]})")

    # --- a dead replica leaves rotation instead of failing requests ---------
    replicas[0].healthy = False
    _wait_until(lambda: requests.get(f"{base}/health", timeout=5).json()
                ["healthy_replicas"] == 1, timeout=30,
                what="unhealthy replica to be detected")
    before = replicas[0].received
    ok = sum(requests.post(f"{base}/v1/chat/completions", json={"marker": "x"},
                           timeout=60).status_code == 200 for _ in range(6))
    assert ok == 6, f"only {ok}/6 succeeded with one replica down"
    assert replicas[0].received == before, "requests still routed to a down replica"
    print("  [ok] unhealthy replica removed from rotation, requests still served")

    # --- and comes back ----------------------------------------------------
    replicas[0].healthy = True
    _wait_until(lambda: requests.get(f"{base}/health", timeout=5).json()
                ["healthy_replicas"] == 2, timeout=30, what="replica recovery")
    print("  [ok] replica returns to rotation when it recovers")

    # --- introspection -----------------------------------------------------
    snap = requests.get(f"{base}/v1/replicas", timeout=5).json()
    assert len(snap["replicas"]) == 2
    assert all("inflight" in r and "completed" in r for r in snap["replicas"])
    print("  [ok] /v1/replicas reports per-replica load")

    # --- all replicas down -> 503, not a hang or a 500 ---------------------
    for r in replicas:
        r.healthy = False
    _wait_until(lambda: requests.get(f"{base}/health", timeout=5).json()
                ["healthy_replicas"] == 0, timeout=30, what="all replicas down")
    resp = requests.post(f"{base}/v1/chat/completions", json={"marker": "y"}, timeout=30)
    assert resp.status_code == 503, f"expected 503 with no healthy replica, got {resp.status_code}"
    print("  [ok] no healthy replica -> 503")

    server.should_exit = True
    for r in replicas:
        r.stop()
    print("\nPASS - routing, health tracking, failover and recovery all behave.")
    return 0


def _ok(url) -> bool:
    import requests
    try:
        return requests.get(url, timeout=2).status_code == 200
    except Exception:
        return False


if __name__ == "__main__":
    raise SystemExit(main())
