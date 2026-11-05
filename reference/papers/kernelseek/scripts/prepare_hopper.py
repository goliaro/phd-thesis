#!/usr/bin/env python3
"""Reduce the H200 benchmark batch (runs_h200/hopper-20261030T0535Z, 3 repeats) to one CSV per scenario
set with the same schema as the harness output, taking the per-scenario median across repeats for every
numeric column. Output: paper/data/hopper/results_hopper_{claim,attntest,edge,real}.csv"""
import csv, glob, os, statistics, math, sys
HERE = os.path.dirname(os.path.abspath(__file__))
BATCH = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "..", "..", "runs_h200", "hopper-20261030T0535Z")
OUT = os.path.join(HERE, "..", "data", "hopper")
os.makedirs(OUT, exist_ok=True)
NUM = ["ours_x", "cudnn_x", "fi_x", "base_us", "cand_us", "cudnn_us", "fi_us", "err_ratio"]
for st in ("claim", "attntest", "edge", "real"):
    reps = sorted(glob.glob(os.path.join(BATCH, "rep-*", "docs", f"results_hopper_{st}.csv")))
    rows = {}
    for rp in reps:
        for r in csv.DictReader(open(rp)):
            rows.setdefault(r["scenario"], []).append(r)
    fields = list(csv.DictReader(open(reps[0])).fieldnames)
    with open(os.path.join(OUT, f"results_hopper_{st}.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields); w.writeheader()
        for name, rs in rows.items():
            out = dict(rs[0])
            out["status"] = "OK" if all(r["status"] == "OK" for r in rs) else "/".join(sorted({r["status"] for r in rs}))
            for k in NUM:
                vals = []
                for r in rs:
                    try:
                        v = float(r[k]);
                        if math.isfinite(v): vals.append(v)
                    except (ValueError, TypeError): pass
                out[k] = f"{statistics.median(vals):.4f}" if vals else ""
            w.writerow(out)
    print(st, len(rows), "scenarios from", len(reps), "repeats")
