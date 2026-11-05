#!/usr/bin/env python3
"""Category-level end-to-end speedup bars.

Each bar is the geometric mean of the per-architecture geomeans in
appendix Tables tab:bench_detail and tab:bench_detail_b (B200), or
tab:bench_detail_h200 and tab:bench_detail_h200_b (H200).

Usage: make_eval_e2e_figure.py [b200|h200]   (default: b200)
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

GPU = sys.argv[1] if len(sys.argv) > 1 else "b200"
OUT_PDF = Path(__file__).resolve().parent / {"b200": "eval_e2e.pdf", "h200": "eval_e2e_h200.pdf"}[GPU]

THR = "#0072B2"
LAT = "#D55E00"

# (label, throughput, latency)
CATEGORIES_B200 = [
    ("Dense & MoE LLMs", [0.95, 1.06, 1.41, 1.10, 0.95, 0.93, 1.00], [1.02, 1.00, 0.94, 1.20, 0.95, 1.04, 1.09]),
    ("Linear attention", [2.10, 0.96, 1.22, 1.44, 3.40, 1.21, 1.21, 1.95, 0.84], [2.02, 0.95, 1.26, 1.75, 1.73, 0.99, 1.04, 1.95, 1.10]),
    ("Vision / audio", [1.31, 1.01, 1.03, 1.04, 1.07, 2.30], [1.10, 1.11, 1.02, 1.06, 1.03, 2.34]),
    ("Multimodal", [1.24, 1.26, 1.59, 1.38, 1.01, 0.94, 0.94], [1.01, 1.05, 0.95, 1.07, 1.06, 0.94, 0.91]),
    ("Edge & detection", [1.03, 1.00, 1.04, 1.14, 0.97], [1.04, 1.04, 1.09, 0.94, 0.90]),
    ("3D / robotics", [1.00, 0.98, 0.98, 1.01, 3.09, 1.13], [0.99, 0.98, 0.99, 1.02, 4.17, 1.13]),
    ("Recommendation", [1.10, 1.02, 2.35, 3.13, 0.96], [1.37, 1.04, 0.97, 1.33, 0.96]),
    ("World models", [1.17, 0.97], [1.16, 0.98]),
]

# GLM-5.2 (NVFP4) is Blackwell-only, so Dense & MoE has one fewer model on H200.
CATEGORIES_H200 = [
    ("Dense & MoE LLMs", [0.97, 1.02, 1.01, 1.04, 0.96, 0.92], [1.00, 1.01, 1.10, 0.92, 1.04, 1.03]),
    ("Linear attention", [1.08, 0.99, 1.34, 1.34, 3.05, 1.02, 1.07, 1.75, 0.96], [1.03, 0.96, 1.29, 1.67, 1.75, 0.99, 0.89, 1.75, 1.06]),
    ("Vision / audio", [1.11, 1.00, 1.01, 1.13, 1.04, 2.85], [1.02, 1.10, 1.02, 1.10, 0.95, 2.05]),
    ("Multimodal", [1.13, 1.21, 1.56, 1.43, 1.00, 0.94, 1.00], [0.99, 1.02, 0.98, 1.01, 1.06, 0.89, 1.04]),
    ("Edge & detection", [1.04, 1.00, 1.06, 1.04, 1.00], [1.11, 0.99, 0.99, 2.61, 0.85]),
    ("3D / robotics", [1.00, 0.99, 1.00, 1.00, 2.98, 1.14], [1.00, 0.99, 0.99, 0.95, 3.71, 1.17]),
    ("Recommendation", [1.07, 1.02, 1.37, 3.77, 1.04], [1.34, 1.04, 1.04, 2.59, 1.05]),
    ("World models", [1.17, 0.99], [1.13, 1.00]),
]

CATEGORIES = {"b200": CATEGORIES_B200, "h200": CATEGORIES_H200}[GPU]


def geomean(xs: list[float]) -> float:
    return math.exp(sum(math.log(x) for x in xs) / len(xs))


def main() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "axes.linewidth": 0.7,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    labels = [f"{c[0]} ({len(c[1])})" for c in CATEGORIES]
    thr = [geomean(c[1]) for c in CATEGORIES]
    lat = [geomean(c[2]) for c in CATEGORIES]

    fig, ax = plt.subplots(figsize=(5.5, 3.15))
    y = np.arange(len(labels))[::-1]
    height = 0.36
    ax.barh(y + height / 2, thr, height, color=THR, zorder=3)
    ax.barh(y - height / 2, lat, height, color=LAT, zorder=3)
    ax.axvline(1.0, color="0.15", ls="--", lw=1.0, zorder=4)

    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=10)
    ax.set_xlabel("Speedup vs. reference", fontsize=10)
    ax.set_xlim(0, 2.15)
    ax.set_xticks([0, 1, 2])
    ax.set_xticklabels(["0", "1×", "2×"], fontsize=10)
    ax.set_ylim(-0.6, len(labels) - 0.4)
    ax.xaxis.grid(True, ls=":", lw=0.6, color="0.75", zorder=0)
    ax.set_axisbelow(True)
    ax.tick_params(axis="y", length=0, pad=3)
    ax.tick_params(axis="x", length=3, pad=2, labelsize=10)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(
        handles=[
            Patch(facecolor=THR, label="Throughput"),
            Patch(facecolor=LAT, label="Latency"),
            Line2D([0], [0], color="0.15", ls="--", lw=1.0, label="Parity"),
        ],
        loc="lower center",
        bbox_to_anchor=(0.55, 1.02),
        frameon=False,
        fontsize=10,
        ncol=3,
        borderaxespad=0.0,
        handlelength=1.4,
        columnspacing=1.1,
    )
    fig.subplots_adjust(left=0.30, right=0.97, top=0.88, bottom=0.14)
    fig.savefig(OUT_PDF, bbox_inches="tight", pad_inches=0.08)
    fig.savefig(OUT_PDF.with_suffix(".png"), dpi=160, bbox_inches="tight", pad_inches=0.08)
    print(f"wrote {OUT_PDF}")
    for lab, t, l in zip(labels, thr, lat):
        print(f"{lab:22s}  thr={t:.2f}  lat={l:.2f}")


if __name__ == "__main__":
    main()
