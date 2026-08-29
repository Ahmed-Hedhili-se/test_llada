"""Aggregate the milestone runs into one comparison table (and plots).

    python profiling/summarize.py
    python profiling/summarize.py --plots        # also write PNGs

Reads every ``profiling/results/<label>/`` directory produced by
``profile_milestone.sh`` and pulls out the numbers a before/after report needs:

  from stdout.txt          baseline and optimized latency, tok/s, speedup
  from run_info.txt        commit, date, wall time, GPU, which tuned config
  from *_cuda_gpu_kern_sum kernel count, total kernel time, top kernels
  from *_cuda_api_sum      launch count, sync count, memcpy count

Writes ``SUMMARY.md`` and ``summary.csv`` next to the results, and with
``--plots`` a PNG per metric.

Standard library only, by design -- matplotlib is imported lazily and only
when ``--plots`` is passed, so this adds no dependency to the project. If it
is missing you still get the table, which is what the report actually needs.

The nsys CSV column names drift between versions, so columns are located by
substring rather than by index.
"""

from __future__ import annotations

import argparse
import csv
import io
import os
import re
import sys

RESULTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")

#: Presentation order. Anything not listed sorts after these, alphabetically.
PREFERRED = ["baseline", "m1_fused_moe_kv", "m2_host_sync", "m3_mem_traffic",
             "m4_launch_count", "m5_rope_final", "m6_head"]


def _read(path):
    try:
        return io.open(path, encoding="utf-8", errors="replace").read()
    except OSError:
        return ""


def parse_run_info(d):
    txt = _read(os.path.join(d, "run_info.txt"))
    out = {}
    for line in txt.splitlines():
        if ":" in line:
            k, _, v = line.partition(":")
            k, v = k.strip(), v.strip()
            if k and v and k not in out:
                out[k] = v
    return out


def parse_stdout(d):
    """Latency / throughput / speedup from the harness's own output."""
    txt = _read(os.path.join(d, "stdout.txt"))
    out = {}

    # Baseline line has plain "N tok/s"; the optimized line says "tok/s/seq".
    for m in re.finditer(r"Mean:\s*([\d.]+)s.*?([\d.]+)\s*tok/s(/seq)?", txt):
        mean, tps, is_opt = float(m.group(1)), float(m.group(2)), bool(m.group(3))
        key = "opt" if is_opt else "base"
        out.setdefault(key + "_mean_s", mean)
        out.setdefault(key + "_tok_s", tps)

    m = re.search(r"Speedup\s*:\s*([\d.]+)x", txt)
    if m:
        out["speedup"] = float(m.group(1))
    m = re.search(r"Token Divergence\s*:\s*(\d+)/(\d+)", txt)
    if m:
        out["divergence"] = f"{m.group(1)}/{m.group(2)}"
    return out


def _csv_rows(path):
    txt = _read(path)
    if not txt.strip():
        return []
    return list(csv.DictReader(io.StringIO(txt)))


def _col(row, *needles):
    """Find a value by fuzzy column name -- nsys renames these between versions."""
    for key in row:
        if key is None:
            continue
        k = key.lower()
        if all(n.lower() in k for n in needles):
            return row[key]
    return None


def _num(v):
    if v is None:
        return None
    v = str(v).replace(",", "").replace("%", "").strip()
    try:
        return float(v)
    except ValueError:
        return None


def parse_kernels(d, label):
    rows = _csv_rows(os.path.join(d, f"{label}_cuda_gpu_kern_sum.csv"))
    if not rows:
        return {}, []
    total_ns, total_inst, ranked = 0.0, 0.0, []
    for r in rows:
        ns = _num(_col(r, "total", "time")) or 0.0
        inst = _num(_col(r, "instances")) or _num(_col(r, "count")) or 0.0
        name = _col(r, "name") or "?"
        total_ns += ns
        total_inst += inst
        ranked.append((name, ns, inst))
    ranked.sort(key=lambda x: -x[1])
    return {"kernel_time_ms": total_ns / 1e6,
            "kernel_launches": int(total_inst),
            "distinct_kernels": len(rows)}, ranked[:8]


def parse_api(d, label):
    rows = _csv_rows(os.path.join(d, f"{label}_cuda_api_sum.csv"))
    if not rows:
        return {}
    out = {"api_launches": 0, "api_syncs": 0, "api_memcpy": 0}
    for r in rows:
        name = (_col(r, "name") or "").strip()
        n = _num(_col(r, "num", "calls")) or _num(_col(r, "count")) or 0.0
        low = name.lower()
        if "launchkernel" in low:
            out["api_launches"] += int(n)
        elif "synchronize" in low:
            out["api_syncs"] += int(n)
        elif "memcpy" in low:
            out["api_memcpy"] += int(n)
    return out


def collect():
    if not os.path.isdir(RESULTS):
        return []
    labels = [x for x in os.listdir(RESULTS)
              if os.path.isdir(os.path.join(RESULTS, x))]
    def order(l):
        return (PREFERRED.index(l), "") if l in PREFERRED else (len(PREFERRED), l)
    labels.sort(key=order)

    out = []
    for label in labels:
        d = os.path.join(RESULTS, label)
        rec = {"label": label}
        rec.update(parse_run_info(d))
        rec.update(parse_stdout(d))
        k, top = parse_kernels(d, label)
        rec.update(k)
        rec.update(parse_api(d, label))
        rec["_top"] = top
        out.append(rec)
    return out


COLUMNS = [
    ("label",           "Milestone",        "{}"),
    ("commit",          "Commit",           "{}"),
    ("commit_date",     "Date",             "{}"),
    ("opt_mean_s",      "Latency (s)",      "{:.2f}"),
    ("opt_tok_s",       "Tok/s",            "{:.2f}"),
    ("speedup",         "Speedup",          "{:.2f}x"),
    ("kernel_launches", "Kernel launches",  "{:,}"),
    ("kernel_time_ms",  "Kernel time (ms)", "{:.1f}"),
    ("api_syncs",       "Syncs",            "{:,}"),
    ("api_memcpy",      "Memcpy",           "{:,}"),
]


def fmt(rec, key, spec):
    v = rec.get(key)
    if v is None or v == "":
        return "—"
    try:
        return spec.format(v)
    except (ValueError, TypeError):
        return str(v)


def write_markdown(recs, path):
    L = []
    L.append("# Milestone comparison\n")
    L.append("Generated by `profiling/summarize.py`. One row per "
             "`profiling/results/<label>/`.\n")
    if recs:
        gpus = {r.get("gpu", "?") for r in recs}
        cfgs = {r.get("tuned_config", "?") for r in recs}
        if len(gpus) > 1:
            L.append(f"> **Warning — mixed hardware across runs: {sorted(gpus)}.** "
                     "These rows are not comparable.\n")
        if len(cfgs) > 1:
            L.append("> **Warning — the tuned kernel config differed between runs.** "
                     "Older milestones may have fallen back to hardcoded tile shapes "
                     "while others used tuned ones; see `profiling/README.md`.\n")
    L.append("| " + " | ".join(c[1] for c in COLUMNS) + " |")
    L.append("|" + "|".join("---" for _ in COLUMNS) + "|")
    for r in recs:
        L.append("| " + " | ".join(fmt(r, k, s) for k, _, s in COLUMNS) + " |")

    base = [r.get("base_tok_s") for r in recs if r.get("base_tok_s")]
    if len(base) > 1:
        spread = 100 * (max(base) - min(base)) / min(base)
        L.append(f"\n## Control arm\n\nThe frozen baseline measured "
                 f"{min(base):.2f}–{max(base):.2f} tok/s across runs "
                 f"(**{spread:.1f}% spread**). The baseline code is identical at "
                 f"every milestone, so this is the environment's own noise — "
                 f"treat any optimized-path difference smaller than it as "
                 f"unresolved.\n")

    L.append("\n## Top kernels per milestone\n")
    for r in recs:
        top = r.get("_top") or []
        if not top:
            continue
        L.append(f"\n**{r['label']}**\n")
        L.append("| Kernel | Time (ms) | Launches |")
        L.append("|---|---:|---:|")
        for name, ns, inst in top:
            short = name if len(name) <= 58 else name[:55] + "..."
            L.append(f"| `{short}` | {ns/1e6:.2f} | {int(inst):,} |")
    io.open(path, "w", encoding="utf-8", newline="\n").write("\n".join(L) + "\n")


def write_csv(recs, path):
    keys = [c[0] for c in COLUMNS] + ["base_tok_s", "base_mean_s",
                                      "wall_seconds", "gpu", "divergence"]
    with io.open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        w.writeheader()
        for r in recs:
            w.writerow(r)


def write_plots(recs, outdir):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("  matplotlib not installed -- skipping plots "
              "(pip install matplotlib). The table is written regardless.")
        return 0

    panels = [("opt_tok_s", "Throughput (tok/s)", "higher is better"),
              ("speedup", "Speedup vs baseline", "higher is better"),
              ("kernel_launches", "Kernel launches", "lower is better"),
              ("api_syncs", "CPU-GPU synchronisations", "lower is better")]
    made = 0
    for key, title, hint in panels:
        pts = [(r["label"], r.get(key)) for r in recs if r.get(key) is not None]
        if len(pts) < 2:
            continue
        labels = [p[0] for p in pts]
        vals = [p[1] for p in pts]
        fig, ax = plt.subplots(figsize=(max(6, 1.5 * len(pts)), 4))
        bars = ax.bar(range(len(vals)), vals, color="#4C78A8")
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=8)
        ax.set_title(f"{title}  ({hint})", fontsize=11)
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", alpha=0.3, linewidth=0.5)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, b.get_height(),
                    f"{v:,.0f}" if v >= 100 else f"{v:.2f}",
                    ha="center", va="bottom", fontsize=8)
        fig.tight_layout()
        p = os.path.join(outdir, f"plot_{key}.png")
        fig.savefig(p, dpi=150)
        plt.close(fig)
        print(f"  wrote {os.path.basename(p)}")
        made += 1
    return made


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--plots", action="store_true",
                    help="also write PNG charts (needs matplotlib)")
    args = ap.parse_args()

    recs = collect()
    if not recs:
        print(f"no runs found in {RESULTS}")
        print("run profiling/profile_milestone.sh first -- see profiling/README.md")
        return 1

    md = os.path.join(RESULTS, "SUMMARY.md")
    cs = os.path.join(RESULTS, "summary.csv")
    write_markdown(recs, md)
    write_csv(recs, cs)
    print(f"{len(recs)} run(s) summarised")
    print(f"  wrote {os.path.relpath(md)}")
    print(f"  wrote {os.path.relpath(cs)}")
    if args.plots:
        write_plots(recs, RESULTS)

    missing = [r["label"] for r in recs if r.get("kernel_launches") is None]
    if missing:
        print(f"\nnote: no nsys CSVs for {', '.join(missing)} -- "
              "kernel/sync columns are blank for those "
              "(ran with --no-nsys, or nsys was unavailable).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
