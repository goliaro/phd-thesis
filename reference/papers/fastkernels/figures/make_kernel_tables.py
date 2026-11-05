#!/usr/bin/env python3
"""Kernel-level agent results on the default FastKernels capture (B200).

Reads each agent's three ``fastkernels bench`` runs and writes the paper tables:

* independent -- ``ind/bench_results/bench.json``: every raw candidate, benched
  ``--standalone`` (imported lower-level modules resolve to the baseline);
* composed    -- ``comp/bench_results/bench.json``: the independent winners, all
  active at once (imports resolve to the agent's own lower-level winners);
* sequential  -- ``seq/bench_results/bench.json``: winners of the sequential
  campaign (L2/L3 regenerated on top of frozen lower-level winners), all active.

A winner is a stem whose non-skipped scenarios all PASSED with geomean speedup
>= 1.0 (the agents' own gate, ``select_fk_winners.py``). A stem is *correct* in a
bench if all its non-skipped scenarios PASSED; its speedup is the geomean over
those scenarios.

Outputs: tables/kernel_levels.tex, tables/kernel_composition.tex and
figures/data/kernel_results.json (per-stem results, for later analysis).
"""

from __future__ import annotations

import json
import math
import re
from collections import defaultdict
from pathlib import Path

PAPER_DIR = Path(__file__).resolve().parent.parent
FK_ROOT = PAPER_DIR.parent  # sibling checkouts of the agent run directories
TARGETS_TXT = PAPER_DIR / "figures" / "data" / "kernel_targets_default_b200.txt"
OUT_JSON = PAPER_DIR / "figures" / "data" / "kernel_results.json"
OUT_LEVELS = PAPER_DIR / "tables" / "kernel_levels.tex"
OUT_COMP = PAPER_DIR / "tables" / "kernel_composition.tex"

# (display name, run directory relative to FK_ROOT)
AGENTS = [
    ("Dr.~Kernel", "drkernel-b200-20260919/campaign"),
    ("Claude Code", "claude-fk-runs"),
    ("KDA", "kda-fk-runs"),
    ("AKO", "ako4x-fk-runs"),
    ("Codex", "codex-fk-runs"),
]
MODES = [
    ("ind", "Independent", "ind/bench_results/bench.json"),
    ("comp", "Composed", "comp/bench_results/bench.json"),
    ("seq", "Sequential", "seq/bench_results/bench.json"),
]
LEVELS = (1, 2, 3)
FAST_P = (1.0, 2.0)  # fast_p columns: correct and speedup > p

# Harness artifacts, excluded for every agent:
#  * oasis_rollout: the harness always SKIPs it (whole-model rollout driver, not a kernel).
#  * SampleDiffusion: multi-step stochastic sampling loop; it amplifies benign rounding
#    differences, so every agent whose single-step DiffusionModule passes fails it at
#    ~60% matched elements. The stem is scored on DiffusionModule alone.
EXCLUDED_STEMS = {(3, "oasis_rollout")}
EXCLUDED_CLASSES = {"alphafold3_diffusion_module:SampleDiffusion"}

QUAL_RE = re.compile(r"\.L([1-4])\.([^:]+):(\w+)")


def geomean(values: list[float]) -> float | None:
    return math.exp(sum(math.log(v) for v in values) / len(values)) if values else None


def load_targets() -> list[tuple[int, str]]:
    targets = []
    for line in TARGETS_TXT.read_text().splitlines():
        if line.strip() and not line.startswith("#"):
            level, stem = line.split()[:2]
            if (int(level), stem) not in EXCLUDED_STEMS:
                targets.append((int(level), stem))
    return targets


def per_stem(bench: Path) -> dict[tuple[int, str], dict]:
    """Per-stem verdict: correct iff every non-skipped scenario PASSED."""
    rows = defaultdict(list)
    for row in json.loads(bench.read_text()).get("scenarios") or []:
        m = QUAL_RE.search(row.get("op") or "")
        if not m:
            continue
        level, stem, cls = int(m.group(1)), m.group(2), m.group(3)
        if f"{stem}:{cls}" in EXCLUDED_CLASSES or (level, stem) in EXCLUDED_STEMS:
            continue
        rows[(level, stem)].append(row)
    out = {}
    for key, rs in rows.items():
        timed = [r for r in rs if r.get("status") != "SKIPPED"]
        ok = bool(timed) and all(
            r.get("status") == "PASSED" and (r.get("speedup") or 0) > 0 for r in timed)
        out[key] = {
            "correct": ok,
            "speedup": geomean([r["speedup"] for r in timed]) if ok else None,
            "statuses": sorted({str(r.get("status")) for r in rs}),
        }
    return out


def fmt_x(v: float | None) -> str:
    return "---" if v is None else f"{v:.2f}$\\times$"


def main() -> None:
    targets = load_targets()
    n_level = {lvl: sum(1 for t in targets if t[0] == lvl) for lvl in LEVELS}
    results: dict[str, dict[str, dict]] = {}
    for name, root in AGENTS:
        results[name] = {}
        for mode, _, rel in MODES:
            stems = per_stem(FK_ROOT / root / rel)
            results[name][mode] = {k: v for k, v in stems.items() if k in set(targets)}

    # ---- Table 1: correctness / speedup / fast_1 per level and mode ----------
    level_rows = []
    summary = {}
    for name, _ in AGENTS:
        summary[name] = {}
        for mode, label, _ in MODES:
            cells = []
            summary[name][mode] = {}
            for lvl in LEVELS:
                recs = [v for k, v in results[name][mode].items() if k[0] == lvl]
                sp = [v["speedup"] for v in recs if v["correct"]]
                fast = {p: sum(1 for s in sp if s > p) for p in FAST_P}
                g = geomean(sp)
                summary[name][mode][f"L{lvl}"] = {
                    "benched": len(recs), "correct": len(sp), "geomean": g,
                    **{f"fast_{p:g}": n for p, n in fast.items()}}
                cells += [str(len(sp)), fmt_x(g), *(str(fast[p]) for p in FAST_P)]
            level_rows.append((name, label, cells))

    ncol = 2 + len(FAST_P)  # Corr., Geo., fast_p...
    head = " & ".join(
        f"\\multicolumn{{{ncol}}}{{c}}{{\\textbf{{L{lvl}}} ({n_level[lvl]} targets)}}" for lvl in LEVELS)
    rules = "".join(f"\\cmidrule(lr){{{3 + ncol * i}-{2 + ncol * (i + 1)}}}" for i in range(len(LEVELS)))
    sub = " & ".join(["Corr. & Geo. & " + " & ".join(f"fast$_{{{p:g}}}$" for p in FAST_P)] * len(LEVELS))
    lines = [
        "\\begin{tabular}{@{}ll" + "r" * ncol * len(LEVELS) + "@{}}",
        "\\toprule",
        f" & & {head} \\\\",
        rules,
        f"Agent & Mode & {sub} \\\\",
        "\\midrule",
    ]
    for i, (name, label, cells) in enumerate(level_rows):
        first = i % len(MODES) == 0
        agent_cell = f"\\multirow{{{len(MODES)}}}{{*}}{{{name}}}" if first else ""
        if first and i:
            lines.append("\\midrule")
        lines.append(f"{agent_cell} & {label} & " + " & ".join(cells) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}", ""]
    OUT_LEVELS.parent.mkdir(parents=True, exist_ok=True)
    OUT_LEVELS.write_text("\n".join(lines))

    # ---- Table 2: paired composition effect and sequential recovery ----------
    def paired(name: str, a: str, b: str, lvl: int) -> dict:
        A, B = results[name][a], results[name][b]
        both = [k for k in A if k[0] == lvl and A[k]["correct"] and k in B and B[k]["correct"]]
        broke = [k for k in A if k[0] == lvl and A[k]["correct"] and k in B and not B[k]["correct"]]
        ga = geomean([A[k]["speedup"] for k in both])
        gb = geomean([B[k]["speedup"] for k in both])
        return {"n": len(both), "from": ga, "to": gb,
                "ratio": (gb / ga) if both else None, "broke": len(broke)}

    comp_rows = []
    pairs = {}
    for name, _ in AGENTS:
        pairs[name] = {}
        cells = []
        for a, b in (("ind", "comp"), ("comp", "seq")):
            for lvl in (2, 3):
                p = paired(name, a, b, lvl)
                pairs[name][f"{a}->{b}:L{lvl}"] = p
                cells += [str(p["n"]), fmt_x(p["ratio"])]
                if a == "ind":
                    cells.append(str(p["broke"]))
        comp_rows.append((name, cells))
    lines = [
        "\\begin{tabular}{@{}l rrr rrr rr rr@{}}",
        "\\toprule",
        " & \\multicolumn{6}{c}{\\makecell{\\textbf{Composition effect}\\\\(Independent $\\to$ Composed)}}"
        " & \\multicolumn{4}{c}{\\makecell{\\textbf{Sequential}\\\\(Composed $\\to$ Seq.)}} \\\\",
        "\\cmidrule(lr){2-7}\\cmidrule(lr){8-11}",
        " & \\multicolumn{3}{c}{L2} & \\multicolumn{3}{c}{L3}"
        " & \\multicolumn{2}{c}{L2} & \\multicolumn{2}{c}{L3} \\\\",
        "\\cmidrule(lr){2-4}\\cmidrule(lr){5-7}\\cmidrule(lr){8-9}\\cmidrule(lr){10-11}",
        "Agent & $n$ & Ratio & Broke & $n$ & Ratio & Broke & $n$ & Ratio & $n$ & Ratio \\\\",
        "\\midrule",
    ]
    lines += [f"{name} & " + " & ".join(cells) + " \\\\" for name, cells in comp_rows]
    lines += ["\\bottomrule", "\\end{tabular}", ""]
    OUT_COMP.write_text("\n".join(lines))

    # ---- JSON dump + console summary -----------------------------------------
    OUT_JSON.write_text(json.dumps({
        "targets_per_level": n_level,
        "excluded_stems": sorted(f"L{l}:{s}" for l, s in EXCLUDED_STEMS),
        "excluded_classes": sorted(EXCLUDED_CLASSES),
        "summary": summary,
        "paired": pairs,
        "per_stem": {name: {mode: {f"L{k[0]}:{k[1]}": v for k, v in sorted(d.items())}
                            for mode, d in modes.items()}
                     for name, modes in results.items()},
    }, indent=1) + "\n")

    print(f"targets per level: {n_level}")
    for name, label, cells in level_rows:
        print(f"  {name:12} {label:12} " + "  |  ".join(
            " ".join(cells[ncol * i:ncol * (i + 1)]).replace("$\\times$", "x") for i in range(len(LEVELS))))
    print("paired (n, ratio[, broke]):  ind->comp L2 | L3   ||   comp->seq L2 | L3")
    for name, cells in comp_rows:
        print(f"  {name:12} " + " ".join(cells).replace("$\\times$", "x"))
    print(f"wrote {OUT_LEVELS.relative_to(PAPER_DIR)}, {OUT_COMP.relative_to(PAPER_DIR)}, "
          f"{OUT_JSON.relative_to(PAPER_DIR)}")


if __name__ == "__main__":
    main()
