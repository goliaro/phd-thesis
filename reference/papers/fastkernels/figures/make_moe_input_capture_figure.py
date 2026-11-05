#!/usr/bin/env python3
"""Render the MoE input-capture comparison figure.

The default input is the sibling ``moe-expert-distribution-main`` checkout,
which contains the Qwen3-VL unified three-way routing run. Pass a different
results directory as the first argument to regenerate from another run.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import PercentFormatter

PAPER_DIR = Path(__file__).resolve().parent.parent
DEFAULT_RESULTS = (
    PAPER_DIR.parent / "moe-expert-distribution-main" / "results" / "moe_expert_unified"
)
OUTPUT_PDF = PAPER_DIR / "figures" / "moe_input_capture.pdf"


def load_series(results_dir: Path):
    items = [
        ("Real requests", "expert_stats_real.json", "#3d5a80", "-"),
        ("Random IDs", "expert_stats_random_tokens.json", "#e07a5f", "--"),
        ("Random tensors", "expert_stats_random_gate.json", "#2a9d8f", "-."),
    ]
    series = []
    for label, filename, color, linestyle in items:
        with (results_dir / filename).open() as fh:
            stats = json.load(fh)
        counts = np.asarray(stats["per_expert_counts"], dtype=float)
        freq = counts / counts.sum()
        series.append((label, stats, freq, np.sort(freq)[::-1], color, linestyle))
    return series


def main() -> None:
    results_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_RESULTS
    series = load_series(results_dir)

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8.5,
            "axes.titlesize": 9.5,
            "axes.labelsize": 8.5,
            "xtick.labelsize": 7.5,
            "ytick.labelsize": 7.5,
            "legend.fontsize": 7.2,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )

    fig, axes = plt.subplots(
        1,
        3,
        figsize=(7.4, 2.55),
        constrained_layout=True,
        gridspec_kw={"width_ratios": [1.35, 0.82, 0.82]},
    )
    ranks = np.arange(1, 129)

    ax = axes[0]
    for label, _stats, _freq, sorted_freq, color, linestyle in series:
        ax.plot(ranks[:64], sorted_freq[:64] * 100, color=color, lw=1.8, ls=linestyle)
    ax.axhline(100 / 128, color="0.65", ls=":", lw=1.0)
    ax.text(38, 4.65, "Real requests", color=series[0][4], fontsize=7.1)
    ax.text(38, 4.28, "Random IDs", color=series[1][4], fontsize=7.1)
    ax.text(38, 3.91, "Random tensors", color=series[2][4], fontsize=7.1)
    ax.set_title("Hot-expert skew", pad=6)
    ax.set_xlabel("Expert rank")
    ax.set_ylabel("Share of assignments")
    ax.yaxis.set_major_formatter(PercentFormatter(xmax=100, decimals=0))
    ax.set_xlim(1, 64)
    ax.set_ylim(0, 5.8)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", color="0.9", lw=0.6)

    ax = axes[1]
    labels = [row[0] for row in series]
    colors = [row[4] for row in series]
    gini = [row[1]["gini"] for row in series]
    bars = ax.bar(range(len(series)), gini, color=colors, width=0.68)
    ax.set_title("Load imbalance", pad=6)
    ax.set_ylabel("Gini coefficient")
    ax.set_ylim(0, 0.56)
    ax.set_xticks(range(len(series)))
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", color="0.9", lw=0.6)
    for bar, value in zip(bars, gini):
        ax.text(bar.get_x() + bar.get_width() / 2, value + 0.015, f"{value:.2f}",
                ha="center", va="bottom", fontsize=7.2)

    ax = axes[2]
    real_top16 = set(np.argsort(series[0][2])[::-1][:16])
    top16_overlap = [
        len(real_top16 & set(np.argsort(freq)[::-1][:16]))
        for _label, _stats, freq, _sorted_freq, _color, _linestyle in series
    ]
    bars = ax.bar(range(len(series)), top16_overlap, color=colors, width=0.68)
    ax.set_title("Hot-expert identity", pad=6)
    ax.set_ylabel("Top-16 overlap with real")
    ax.set_ylim(0, 16.8)
    ax.set_yticks([0, 4, 8, 12, 16])
    ax.set_xticks(range(len(series)))
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", color="0.9", lw=0.6)
    for bar, value in zip(bars, top16_overlap):
        ax.text(bar.get_x() + bar.get_width() / 2, value + 0.45, str(value),
                ha="center", va="bottom", fontsize=7.2)

    fig.savefig(OUTPUT_PDF, bbox_inches="tight", pad_inches=0.03)
    print(f"wrote {OUTPUT_PDF}")


if __name__ == "__main__":
    main()
