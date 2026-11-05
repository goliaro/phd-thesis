#!/usr/bin/env python3
"""fast_p curves for the kernel agents (independent mode, default capture, B200).

fast_p = fraction of targets whose kernel is correct and faster than p times the
production baseline. Reads figures/data/kernel_results.json (written by
make_kernel_tables.py) and writes figures/kernel_fastp.pdf.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

PAPER_DIR = Path(__file__).resolve().parent.parent
IN_JSON = PAPER_DIR / "figures" / "data" / "kernel_results.json"
OUT_PDF = PAPER_DIR / "figures" / "kernel_fastp.pdf"

MODE = "ind"
# Okabe-Ito subset, validated for CVD separation; markers and dashes carry identity
# for the green/purple pair and for grayscale print.
STYLE = {
    "Dr.~Kernel": ("Dr. Kernel", "#0072B2", "o", (0, (1, 1))),
    "Claude Code": ("Claude Code", "#009E73", "s", (0, (4, 1.5))),
    "KDA": ("KDA", "#D55E00", "^", (0, (6, 1.5, 1, 1.5))),
    "AKO": ("AKO", "#CC79A7", "D", "-"),
    "Codex": ("Codex", "#E69F00", "v", (0, (3, 1, 1, 1, 1, 1))),
}
P_MIN, P_MAX = 0.5, 64.0
MARK_AT = (1.0, 2.0, 4.0, 8.0, 16.0)


def fast_p(speedups: list[float], n_targets: int, p: np.ndarray) -> np.ndarray:
    s = np.asarray(speedups)
    return np.array([(s > x).sum() for x in p]) / n_targets * 100.0


def main() -> None:
    data = json.loads(IN_JSON.read_text())
    n_targets = {int(k): v for k, v in data["targets_per_level"].items()}
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 8, "axes.linewidth": 0.6,
        "pdf.fonttype": 42, "ps.fonttype": 42,
    })
    fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.2), sharey=True)
    p = np.geomspace(P_MIN, P_MAX, 400)
    for ax, lvl in zip(axes, (1, 2, 3)):
        for agent, (label, color, marker, dash) in STYLE.items():
            recs = data["per_stem"][agent][MODE]
            sp = [v["speedup"] for k, v in recs.items() if k.startswith(f"L{lvl}:") and v["correct"]]
            ax.plot(p, fast_p(sp, n_targets[lvl], p), color=color, lw=1.4, ls=dash, label=label)
            marks = np.array(MARK_AT)
            ax.plot(marks, fast_p(sp, n_targets[lvl], marks), ls="none", marker=marker,
                    ms=4, color=color, mec="white", mew=0.6)
        for x in (1.0, 2.0):
            ax.axvline(x, color="#9a9a9a", lw=0.6, ls="--", zorder=0)
        ax.set_xscale("log", base=2)
        ax.set_xlim(P_MIN, P_MAX)
        ax.set_xticks([0.5, 1, 2, 4, 8, 16, 32, 64])
        ax.set_xticklabels(["0.5", "1", "2", "4", "8", "16", "32", "64"])
        ax.set_ylim(0, 100)
        ax.set_title(f"L{lvl} ({n_targets[lvl]} targets)", fontsize=8.5)
        ax.set_xlabel("speedup threshold $p$")
        ax.grid(axis="y", color="#e6e6e6", lw=0.5)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel(r"fast$_p$ (% of targets)")
    handles = [plt.Line2D([], [], color=c, ls=d, lw=1.4, marker=m, ms=4, mec="white", mew=0.6)
               for _, c, m, d in STYLE.values()]
    fig.legend(handles, [s[0] for s in STYLE.values()], loc="upper center", ncol=4,
               frameon=False, bbox_to_anchor=(0.5, 1.04))
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(OUT_PDF, bbox_inches="tight")
    print(f"wrote {OUT_PDF.relative_to(PAPER_DIR)}")


if __name__ == "__main__":
    main()
