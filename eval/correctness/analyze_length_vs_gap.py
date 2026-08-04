"""
Corroborating check for the block-commit-staleness hypothesis (see
INVESTIGATION_LOG.md Part 2, residual gap after commit a5f6ebe).

Takes matched --save-transcripts JSON from a model_update run and an HF run
(same --seed/--limit/--task, so item order/idx lines up) and buckets
per-item correctness by response length -- if cached's accuracy gap to HF
grows with response length, that supports the theory that longer CoT
responses cross more block-boundary commits and accumulate more staleness.

Pure Python + stdlib only -- no torch/GPU needed, safe to run anywhere,
including off the GPU box once you've copied the transcript JSONs down.

Usage:
    python eval/correctness/analyze_length_vs_gap.py \\
        --cached results/correctness/optimized_mmlu_pro_8block_transcripts.json \\
        --hf     results/correctness/hf_mmlu_pro_transcripts.json
"""

import argparse
import json


def load(path):
    with open(path) as f:
        return json.load(f)


def bucket_label(n_chars, edges):
    for lo, hi in edges:
        if lo <= n_chars < hi:
            return f"{lo}-{hi if hi < 10**9 else '+'}"
    return "?"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cached", required=True, help="model_update transcript JSON")
    ap.add_argument("--hf", required=True, help="HF transcript JSON")
    args = ap.parse_args()

    cached = {r["idx"]: r for r in load(args.cached)}
    hf = {r["idx"]: r for r in load(args.hf)}
    common = sorted(set(cached) & set(hf))
    if not common:
        print("No overlapping idx between the two transcript files -- "
              "did you use the same --seed/--limit/--task for both runs?")
        return

    edges = [(0, 100), (100, 300), (300, 600), (600, 1200), (1200, 10**9)]

    stats = {bucket_label(0, edges): None}  # placeholder to keep order stable
    buckets = {bucket_label(lo, edges): {"cached_correct": 0, "hf_correct": 0, "n": 0}
               for lo, hi in edges}

    for idx in common:
        c, h = cached[idx], hf[idx]
        length = len(c.get("raw_response") or "")
        b = bucket_label(length, edges)
        buckets[b]["n"] += 1
        buckets[b]["cached_correct"] += bool(c.get("correct"))
        buckets[b]["hf_correct"] += bool(h.get("correct"))

    print(f"{'response length (chars)':<28} {'n':>5} {'model_update acc':>18} {'HF acc':>10} {'gap':>8}")
    print("-" * 75)
    for lo, hi in edges:
        b = bucket_label(lo, edges)
        d = buckets[b]
        if d["n"] == 0:
            continue
        c_acc = d["cached_correct"] / d["n"]
        h_acc = d["hf_correct"] / d["n"]
        print(f"{b:<28} {d['n']:>5} {c_acc*100:>16.1f}% {h_acc*100:>9.1f}% {(h_acc-c_acc)*100:>+7.1f}pt")

    total_n = len(common)
    total_c = sum(bool(cached[i].get("correct")) for i in common) / total_n
    total_h = sum(bool(hf[i].get("correct")) for i in common) / total_n
    print("-" * 75)
    print(f"{'overall':<28} {total_n:>5} {total_c*100:>16.1f}% {total_h*100:>9.1f}% {(total_h-total_c)*100:>+7.1f}pt")
    print()
    print("If the gap column grows monotonically (or roughly so) with response")
    print("length, that supports the block-commit-staleness hypothesis: longer")
    print("CoT responses cross more block boundaries -> more accumulated drift")
    print("from HF's fully-recomputed-every-step reference.")


if __name__ == "__main__":
    main()
