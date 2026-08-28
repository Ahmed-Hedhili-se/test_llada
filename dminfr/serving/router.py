"""Data-parallel router: N single-GPU replicas behind one endpoint.

Why this exists rather than tensor parallelism
----------------------------------------------
``dminfr/serving/server.py`` disables request batching whenever ``tp_size > 1`` (four
guards, see its ``_batch_collector`` docstring): above one rank every request
serialises through ``request_lock``, because a batch would have to survive
``dist.broadcast_object_list`` to ``worker_loop`` on every step. So a TP=8
deployment gets excellent single-request latency and roughly an eighth of the
throughput the hardware can do.

For throughput the model does not need TP at all -- it is ~14 GiB against an
80 GiB H100. Running one independent replica per GPU keeps the validated
single-GPU batched path and scales close to linearly. What was missing was a
single endpoint in front of them; ``benchmarks/throughput/run_throughput.py
--base-urls`` round-robins, but that is a benchmark client, not a router.

Routing policy
--------------
Least-outstanding, not round-robin. Requests here are not uniform: a replica
runs a whole batch to completion, so one that just accepted 32 requests is busy
for ~35s while an idle one answers immediately. Round-robin would keep feeding
the busy replica on its turn; dispatching to whichever replica has the fewest
in-flight requests tracks actual load instead. Ties break randomly so that
several routers (or a burst) do not synchronise onto the same replica.

Failure handling
----------------
A replica that fails its health probe is taken out of rotation and retried in
the background, so one dead GPU degrades capacity instead of failing requests.
Requests already in flight to it still fail -- they cannot be retried safely,
because generation is not idempotent from the client's point of view and a
retry would double GPU work during an incident.

Usage::

    python -m dminfr.serving.router --backends http://localhost:8001,http://localhost:8002 \\
        --port 8000

    # or let start_dp.sh launch the replicas and the router together
"""

from __future__ import annotations

import argparse
import asyncio
import os
import random
import time
from typing import Dict, List, Optional

import aiohttp
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, Response

app = FastAPI(title="LLaDA-MoE DP Router")

#: Per-request dispatch logging. Default OFF.
#:
#: This was unconditional, and it is a blocking write to stdout executed from
#: inside the asyncio event loop on the completion path of every request. At
#: the concurrencies this router exists to serve it stalls the loop that is
#: also proxying every other in-flight request. Measured on 2xH100: two
#: replicas driven directly, without the router, reach 1190 tok/s; through the
#: router the same pair reached 896 -- a 24.7% tax that is not GPU contention
#: (running both replicas concurrently but unrouted costs only 7%).
#:
#: Set LLADA_ROUTER_LOG=1 to get it back for debugging.
LOG_EACH = os.environ.get("LLADA_ROUTER_LOG", "0") != "0"

#: Endpoints proxied verbatim to a chosen replica.
PROXIED = ("/v1/chat/completions", "/v1/completions")

#: Endpoints answered from any healthy replica -- they describe the model, not
#: a request, so every replica returns the same thing.
BROADCAST = ("/v1/models", "/v1/config", "/v1/quantization")


class Replica:
    def __init__(self, url: str):
        self.url = url.rstrip("/")
        self.inflight = 0
        self.healthy = False
        self.total = 0
        self.failures = 0
        self.last_error: Optional[str] = None

    def snapshot(self) -> Dict:
        return {
            "url": self.url, "healthy": self.healthy, "inflight": self.inflight,
            "completed": self.total, "failures": self.failures,
            "last_error": self.last_error,
        }


class Pool:
    def __init__(self, urls: List[str], timeout: float):
        self.replicas = [Replica(u) for u in urls]
        self.timeout = timeout
        self._session: Optional[aiohttp.ClientSession] = None

    async def session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            # No total timeout: a batched generation legitimately takes
            # minutes, and aiohttp's 5-minute default would abort healthy
            # requests. sock_connect still bounds the "replica is gone" case.
            self._session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=self.timeout, sock_connect=10)
            )
        return self._session

    def pick(self) -> Replica:
        live = [r for r in self.replicas if r.healthy]
        if not live:
            raise HTTPException(503, "no healthy replica")
        fewest = min(r.inflight for r in live)
        return random.choice([r for r in live if r.inflight == fewest])

    async def probe_once(self) -> None:
        s = await self.session()
        for r in self.replicas:
            try:
                async with s.get(f"{r.url}/health",
                                 timeout=aiohttp.ClientTimeout(total=5)) as resp:
                    was = r.healthy
                    r.healthy = resp.status == 200
                    if r.healthy and not was:
                        print(f"[router] {r.url} is up", flush=True)
            except Exception as exc:
                if r.healthy:
                    print(f"[router] {r.url} went down: {exc}", flush=True)
                r.healthy = False
                r.last_error = str(exc)

    async def probe_forever(self, every: float) -> None:
        while True:
            await self.probe_once()
            await asyncio.sleep(every)


POOL: Optional[Pool] = None


@app.get("/health")
async def health():
    live = [r for r in POOL.replicas if r.healthy]
    # Degraded (some replicas down) still reports ok: the endpoint is usable,
    # and returning 503 would take the whole deployment out of an upstream load
    # balancer over a single dead GPU.
    return {
        "status": "ok" if live else "unavailable",
        "healthy_replicas": len(live),
        "total_replicas": len(POOL.replicas),
    }


@app.get("/v1/replicas")
async def replicas():
    """Per-replica load and health -- the view that shows whether requests are
    actually spread, rather than piling onto one GPU."""
    return {
        "replicas": [r.snapshot() for r in POOL.replicas],
        "total_inflight": sum(r.inflight for r in POOL.replicas),
    }


async def _forward(path: str, payload: dict) -> JSONResponse:
    replica = POOL.pick()
    replica.inflight += 1
    started = time.monotonic()
    try:
        s = await POOL.session()
        async with s.post(f"{replica.url}{path}", json=payload) as resp:
            # Pass the replica's bytes straight through. The previous version
            # did `await resp.json()` then handed the dict to JSONResponse,
            # which re-serialised it -- a full parse and a full re-encode of
            # every completion body, on the event loop, per request, to
            # reproduce bytes the replica had already produced. The router
            # does not inspect the body, so there is nothing to parse it for.
            raw = await resp.read()
            if resp.status != 200:
                replica.failures += 1
            return Response(
                content=raw,
                status_code=resp.status,
                media_type=resp.headers.get("content-type", "application/json"),
            )
    except Exception as exc:
        replica.failures += 1
        replica.last_error = str(exc)
        # Not retried on another replica: generation is expensive and not
        # idempotent in cost, so a retry storm during an incident would double
        # GPU load exactly when it is least available.
        raise HTTPException(502, f"replica {replica.url} failed: {exc}") from exc
    finally:
        replica.inflight -= 1
        replica.total += 1
        if LOG_EACH:
            print(f"[router] {path} -> {replica.url} "
                  f"({time.monotonic() - started:.2f}s, inflight now {replica.inflight})",
                  flush=True)


@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    return await _forward("/v1/chat/completions", await request.json())


@app.post("/v1/completions")
async def completions(request: Request):
    return await _forward("/v1/completions", await request.json())


@app.get("/v1/models")
async def models():
    return await _broadcast_get("/v1/models")


@app.get("/v1/quantization")
async def quantization():
    return await _broadcast_get("/v1/quantization")


@app.get("/v1/config")
async def get_config():
    return await _broadcast_get("/v1/config")


async def _broadcast_get(path: str):
    """Answer from the first healthy replica. These describe the model, so all
    replicas agree -- unless they were launched with different flags, which
    /v1/replicas plus each replica's own /v1/quantization will expose."""
    s = await POOL.session()
    for r in POOL.replicas:
        if not r.healthy:
            continue
        try:
            async with s.get(f"{r.url}{path}",
                             timeout=aiohttp.ClientTimeout(total=10)) as resp:
                return JSONResponse(status_code=resp.status, content=await resp.json())
        except Exception:
            continue
    raise HTTPException(503, "no healthy replica")


@app.post("/v1/config")
async def set_config(request: Request):
    """Applied to EVERY replica -- a config change that landed on one only
    would make the deployment answer inconsistently depending on routing."""
    payload = await request.json()
    s = await POOL.session()
    results = {}
    for r in POOL.replicas:
        if not r.healthy:
            results[r.url] = "unhealthy, skipped"
            continue
        try:
            async with s.post(f"{r.url}/v1/config", json=payload,
                              timeout=aiohttp.ClientTimeout(total=15)) as resp:
                results[r.url] = await resp.json()
        except Exception as exc:
            results[r.url] = f"failed: {exc}"
    return {"applied": results}


@app.on_event("startup")
async def _startup():
    await POOL.probe_once()
    asyncio.create_task(POOL.probe_forever(every=10.0))


def main() -> int:
    global POOL
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--backends", required=True,
                    help="comma-separated replica URLs, e.g. "
                         "http://localhost:8001,http://localhost:8002")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--timeout", type=float, default=3600.0,
                    help="per-request ceiling in seconds. Deliberately large: a "
                         "batched 1024-token generation legitimately runs for "
                         "minutes, and aiohttp's 5-minute default would abort "
                         "healthy requests mid-flight.")
    args = ap.parse_args()

    urls = [u.strip() for u in args.backends.split(",") if u.strip()]
    if not urls:
        raise SystemExit("--backends listed no URLs")
    POOL = Pool(urls, timeout=args.timeout)

    import uvicorn
    print("=" * 64)
    print(f"  DP router on {args.host}:{args.port} -> {len(urls)} replica(s)")
    for u in urls:
        print(f"    {u}")
    print("  policy: least-outstanding; GET /v1/replicas for live load")
    print("=" * 64)
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
