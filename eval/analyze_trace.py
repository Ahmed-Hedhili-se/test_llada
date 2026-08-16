"""
Aggregate a torch.profiler Chrome trace into a kernel-time ranking.

This project has repeatedly needed the same question answered -- "where is
GPU time actually going now?" -- and has answered it by hand each time (see
README's trace-analysis sections). The ranking goes stale every time a kernel
lands, so it needs re-deriving, not re-reading.

Reports, per trace:
  1. GPU busy time, launch count, and idle-gap distribution. The gap
     histogram matters more than the total: many small gaps means
     launch-overhead-bound (the ~8us signature this project found before),
     a few large ones means a real stall.
  2. Top kernels by total GPU time.
  3. Semantic buckets (MoE / attention / RMSNorm / RoPE / dense GEMM / ...),
     which is what actually drives "what do we optimize next".

Traces are parsed line-wise by regex rather than json.load -- PyTorch writes
one event per line, and a 128-step trace is ~360MB, which json.load would
expand to several GB of Python objects. Falls back to json.load if the
line-wise pass finds nothing (older/compacted traces).

Usage:
    python -m eval.analyze_trace                    # newest complete trace in traces/
    python -m eval.analyze_trace path/to/trace.json
    python -m eval.analyze_trace --top 40
"""

import argparse
import collections
import glob
import json
import os
import re
import sys

# Order matters: first match wins, so put specific patterns before generic ones.
BUCKETS = [
    ("MoE grouped GEMM",   [r"fused_moe_kernel"]),
    ("Attention",          [r"flash_fwd", r"scaled_dot_product", r"attention"]),
    ("MoE routing/align",  [r"radixSort", r"bitonicSort", r"gatherTopK", r"Histogram",
                            r"DeviceScan", r"DeviceRadixSort", r"cumsum"]),
    ("RMSNorm",            [r"MeanOps", r"rsqrt_kernel", r"pow_tensor_scalar"]),
    ("RoPE",               [r"CatArrayBatchedCopy", r"neg_kernel"]),
    ("Softmax / argmax",   [r"SoftMax", r"ArgMaxOps", r"softmax"]),
    ("Dense GEMM",         [r"gemm", r"cutlass", r"ampere_", r"sm\d+_xmma", r"volta_", r"turing_"]),
    ("Reductions",         [r"reduce_kernel", r"ReduceOp"]),
    ("Copies / casts",     [r"direct_copy", r"CopyKernel", r"Memcpy", r"Memset", r"copy_"]),
    ("Indexing / scatter", [r"index_elementwise", r"scatter", r"gather", r"IndexKernel"]),
    ("Elementwise",        [r"elementwise_kernel", r"vectorized_elementwise", r"BinaryFunctor",
                            r"CUDAFunctor", r"unrolled_elementwise"]),
    ("Triton (other)",     [r"^triton_"]),
]

GPU_CATS = {"kernel", "gpu_memcpy", "gpu_memset"}

_EVENT_RE = re.compile(
    r'"name"\s*:\s*"((?:[^"\\]|\\.)*)".*?"cat"\s*:\s*"([^"]*)".*?"dur"\s*:\s*([0-9.]+).*?"ts"\s*:\s*([0-9.]+)'
)
_EVENT_RE_ALT = re.compile(
    r'"cat"\s*:\s*"([^"]*)".*?"name"\s*:\s*"((?:[^"\\]|\\.)*)".*?"ts"\s*:\s*([0-9.]+).*?"dur"\s*:\s*([0-9.]+)'
)


def _iter_events_linewise(path):
    """Yield (name, cat, ts, dur). PyTorch writes one JSON object per line."""
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            if '"dur"' not in line:
                continue
            m = _EVENT_RE.search(line)
            if m:
                yield m.group(1), m.group(2), float(m.group(4)), float(m.group(3))
                continue
            m = _EVENT_RE_ALT.search(line)
            if m:
                yield m.group(2), m.group(1), float(m.group(3)), float(m.group(4))


def _iter_events_json(path):
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        doc = json.load(f)
    events = doc["traceEvents"] if isinstance(doc, dict) else doc
    for e in events:
        if e.get("ph") == "X" and "dur" in e:
            yield e.get("name", ""), e.get("cat", ""), float(e["ts"]), float(e["dur"])


def load_events(path):
    events = [e for e in _iter_events_linewise(path)]
    if not events:
        print("  (line-wise parse found nothing; falling back to json.load)", file=sys.stderr)
        events = list(_iter_events_json(path))
    return events


def bucket_for(name):
    for label, patterns in BUCKETS:
        for p in patterns:
            if re.search(p, name, re.IGNORECASE):
                return label
    return "Other"


def analyze(path, top_n):
    size_mb = os.path.getsize(path) / 1e6
    print("=" * 78)
    print(f"{os.path.basename(path)}  ({size_mb:.0f} MB)")
    print("=" * 78)

    events = load_events(path)
    gpu = [(n, ts, d) for (n, c, ts, d) in events if c in GPU_CATS]
    launches = sum(1 for (_, c, _, _) in events if c == "cuda_runtime")

    if not gpu:
        print("No GPU kernel events found — was the trace exported with CUDA activity?")
        return

    by_time = collections.Counter()
    by_count = collections.Counter()
    for name, _, dur in gpu:
        by_time[name] += dur
        by_count[name] += 1
    total = sum(by_time.values())

    # Wall-clock span and idle gaps: sort by start, walk the timeline.
    gpu.sort(key=lambda x: x[1])
    span_start = gpu[0][1]
    span_end = max(ts + d for (_, ts, d) in gpu)
    span = span_end - span_start
    gaps = []
    cursor = span_start
    for _, ts, d in gpu:
        if ts > cursor:
            gaps.append(ts - cursor)
        cursor = max(cursor, ts + d)

    print(f"\n  GPU busy        : {total / 1000:.1f} ms")
    print(f"  Wall-clock span : {span / 1000:.1f} ms")
    print(f"  Busy %          : {total / span * 100:.1f}%")
    print(f"  Kernel events   : {len(gpu):,}")
    print(f"  CPU launch calls: {launches:,}")
    if gaps:
        gaps.sort()
        idle = sum(gaps)
        print(f"  Idle gaps       : {len(gaps):,}  totalling {idle / 1000:.1f} ms "
              f"({idle / span * 100:.1f}%)")
        print(f"  Gap size        : mean {idle / len(gaps):.2f}us  "
              f"median {gaps[len(gaps) // 2]:.2f}us  max {gaps[-1]:.0f}us")
        small = sum(1 for g in gaps if g < 20)
        print(f"  Gaps < 20us     : {small:,} ({small / len(gaps) * 100:.1f}%) "
              f"— high share means launch-overhead-bound, not one fixable stall")

    print(f"\n  ── Top {top_n} kernels by GPU time " + "─" * 34)
    print(f"  {'%':>7}  {'ms':>9}  {'count':>7}  kernel")
    for name, t in by_time.most_common(top_n):
        print(f"  {t / total * 100:6.2f}%  {t / 1000:8.2f}  {by_count[name]:7d}  {name[:80]}")

    buckets_t = collections.Counter()
    buckets_n = collections.Counter()
    for name, t in by_time.items():
        b = bucket_for(name)
        buckets_t[b] += t
        buckets_n[b] += by_count[name]

    print(f"\n  ── Semantic buckets " + "─" * 47)
    print(f"  {'%':>7}  {'ms':>9}  {'launches':>9}  bucket")
    for b, t in buckets_t.most_common():
        print(f"  {t / total * 100:6.2f}%  {t / 1000:8.2f}  {buckets_n[b]:9d}  {b}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("trace", nargs="?", default=None,
                    help="Trace file. Default: newest complete .json in traces/.")
    ap.add_argument("--top", type=int, default=30)
    args = ap.parse_args()

    path = args.trace
    if path is None:
        # .tmp files are exports still in flight — skip them.
        candidates = [p for p in glob.glob("traces/*.json") if os.path.getsize(p) > 0]
        if not candidates:
            print("No complete traces in traces/ (only .tmp?). Run with PROFILE_BATCHES=1.")
            sys.exit(1)
        path = max(candidates, key=os.path.getmtime)

    analyze(path, args.top)


if __name__ == "__main__":
    main()
