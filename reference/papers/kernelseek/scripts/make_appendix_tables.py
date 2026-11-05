#!/usr/bin/env python3
"""Per-scenario appendix tables (longtable) from paper/data/{hopper,blackwell}/results_*.csv.
Writes sections/tables/per_scenario_hopper.tex and per_scenario_blackwell.tex."""
import csv, math, os
HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "..", "sections", "tables")
os.makedirs(OUT, exist_ok=True)
SETS = ["attntest", "edge", "claim", "real"]


def f(x):
    try:
        v = float(x); return v if math.isfinite(v) else None
    except (TypeError, ValueError):
        return None


def tex(s):
    return s.replace("_", "\\_").replace("%", "\\%").replace("&", "\\&")


def fmt(v, d=2):
    return "---" if v is None else f"{v:.{d}f}"


for arch, up, dev in (("hopper", "FA3", "H200"), ("blackwell", "FA4", "B200")):
    lines = []
    lines.append(f"\\begin{{longtable}}{{@{{}}llrrrrr@{{}}}}")
    lines.append(f"\\caption{{Per-scenario results on {dev}. Speedups are upstream ({up}) time over the implementation's time; "
                 f"``best'' is our speedup over the fastest available baseline (upstream, cuDNN, or FlashInfer). "
                 f"``---'': the implementation does not support the scenario.}}\\label{{tab:per-scenario-{arch}}}\\\\")
    lines.append("\\toprule Set & Scenario & " + up + " ($\\mu$s) & ours & cuDNN & FlashInfer & best \\\\ \\midrule \\endfirsthead")
    lines.append("\\toprule Set & Scenario & " + up + " ($\\mu$s) & ours & cuDNN & FlashInfer & best \\\\ \\midrule \\endhead")
    lines.append("\\bottomrule \\endfoot")
    for s in SETS:
        path = os.path.join(HERE, "..", "data", arch, f"results_{arch}_{s}.csv")
        for r in csv.DictReader(open(path)):
            o, cu, fi, base = f(r["ours_x"]), f(r["cudnn_x"]), f(r["fi_x"]), f(r["base_us"])
            if r["status"] != "OK":
                lines.append(f"{s} & {tex(r['scenario'])} & --- & \\multicolumn{{4}}{{l}}{{{tex(r['status'])}}} \\\\")
                continue
            best = (o / max([1.0] + ([fi] if fi else []) + ([cu] if cu else []))) if o else None
            lines.append(f"{s} & {tex(r['scenario'])} & {fmt(base, 1)} & {fmt(o)} & {fmt(cu)} & {fmt(fi)} & {fmt(best)} \\\\")
    lines.append("\\end{longtable}")
    open(os.path.join(OUT, f"per_scenario_{arch}.tex"), "w").write("\n".join(lines) + "\n")
    print(arch, len(lines) - 6, "rows")
