#!/usr/bin/env python3
"""Case-study figure (both architectures) and the numbers behind Table 1.

Reads paper/data/{hopper,blackwell}/results_<arch>_{attntest,edge,claim,real}.csv (harness schema; each row has
ours_x, cudnn_x, fi_x = speedup of that implementation over the upstream kernel on one scenario) and draws a
full-width 2x2 figure:
  (a,c) geomean speedup per scenario set vs. upstream, vs. the fastest open-source baseline (upstream or
        FlashInfer, per scenario), vs. the fastest available baseline (also cuDNN), for Hopper and Blackwell;
  (b,d) every paired scenario sorted by speedup vs. the fastest available baseline.

Usage: python3 scripts/plot_casestudy.py [--out figures/casestudy.pdf]
"""
import argparse, csv, math, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
SETS = ["attntest", "edge", "claim", "real"]
ARCHES = [("hopper", "Hopper (H200), upstream = FA3"), ("blackwell", "Blackwell (B200), upstream = FA4")]
C_UP, C_OSS, C_ALL = "#2a78d6", "#eb6834", "#1baf7a"     # validated categorical palette (dataviz skill)
INK, INK2, GRID = "#0b0b0b", "#52514e", "#d9d8d3"


def fnum(x):
    try:
        v = float(x)
        return v if math.isfinite(v) else None
    except (TypeError, ValueError):
        return None


def load(arch):
    rows = []
    for s in SETS:
        path = os.path.join(HERE, "..", "data", arch, f"results_{arch}_{s}.csv")
        for r in csv.DictReader(open(path)):
            o = fnum(r["ours_x"])
            runnable = r["status"] == "OK"
            fi, cu = fnum(r["fi_x"]), fnum(r["cudnn_x"])
            best_oss = max([1.0] + ([fi] if fi else []))
            best_all = max([1.0] + ([fi] if fi else []) + ([cu] if cu else []))
            rows.append(dict(set=s, name=r["scenario"], runnable=runnable, paired=runnable and o is not None,
                             up=o, oss=(o / best_oss) if o else None, all=(o / best_all) if o else None,
                             cudnn=cu, fi=fi))
    return rows


def gm(v):
    return math.exp(sum(map(math.log, v)) / len(v))


def stats(rows):
    out = {}
    for g in SETS + ["all"]:
        rs = [r for r in rows if (g == "all" or r["set"] == g)]
        pr = [r for r in rs if r["paired"]]
        out[g] = dict(n=len(rs), runnable=sum(r["runnable"] for r in rs), paired=len(pr),
                      up=gm([r["up"] for r in pr]), oss=gm([r["oss"] for r in pr]), all=gm([r["all"] for r in pr]),
                      up_ge1=sum(r["up"] >= 1 for r in pr), oss_ge1=sum(r["oss"] >= 1 for r in pr), all_ge1=sum(r["all"] >= 1 for r in pr),
                      up_min=min(r["up"] for r in pr), all_min=min(r["all"] for r in pr),
                      cudnn_n=sum(1 for r in pr if r["cudnn"]), cudnn_beat=sum(1 for r in pr if r["cudnn"] and r["up"] > r["cudnn"]),
                      fi_n=sum(1 for r in pr if r["fi"]), fi_beat=sum(1 for r in pr if r["fi"] and r["up"] > r["fi"]),
                      cudnn_fastest=sum(1 for r in pr if (r["cudnn"] or 0) > max(1.0, r["fi"] or 0)))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(HERE, "..", "figures", "casestudy.pdf"))
    a = ap.parse_args()
    plt.rcParams.update({
        "font.size": 7, "axes.labelsize": 7, "xtick.labelsize": 6.5, "ytick.labelsize": 6.5,
        "legend.fontsize": 6.5, "axes.edgecolor": GRID, "axes.linewidth": 0.6,
        "xtick.color": INK2, "ytick.color": INK2, "axes.labelcolor": INK, "text.color": INK, "pdf.fonttype": 42,
    })
    fig, axes = plt.subplots(2, 2, figsize=(6.9, 3.6), gridspec_kw=dict(width_ratios=[1.0, 1.15], hspace=0.75, wspace=0.28))
    letters = "abcd"
    series = [("vs. upstream", "up", C_UP), ("vs. fastest open-source", "oss", C_OSS), ("vs. fastest available", "all", C_ALL)]
    for row, (arch, title) in enumerate(ARCHES):
        rows = load(arch); st = stats(rows)
        ax1, ax2 = axes[row]
        groups = SETS + ["all"]; x = np.arange(len(groups)); w = 0.26
        for i, (lab, k, col) in enumerate(series):
            ax1.bar(x + (i - 1) * w, [st[g][k] for g in groups], width=w - 0.03, color=col, linewidth=0, zorder=3, label=lab)
        ax1.axhline(1.0, color=INK2, lw=0.8, ls=(0, (3, 2)), zorder=2)
        ax1.set_xticks(x); ax1.set_xticklabels([f"{g}\n(n={st[g]['paired']})" for g in groups])
        ax1.set_ylabel("geomean speedup"); ax1.set_ylim(0, 2.2); ax1.set_yticks([0, 0.5, 1.0, 1.5, 2.0])
        ax1.yaxis.grid(True, color=GRID, lw=0.5, zorder=0); ax1.set_axisbelow(True)
        for sp in ("top", "right"): ax1.spines[sp].set_visible(False)
        ax1.set_title(f"({letters[2*row]}) {title.split(',')[0]}: geomean by set ({title.split('= ')[1]} = 1)", loc="left", fontsize=7, pad=4)

        v = np.array(sorted(r["all"] for r in rows if r["paired"]))
        n_ge1 = int((v >= 1.0).sum()); cross = int((v < 1.0).sum())
        ax2.bar(np.arange(len(v)), v, width=0.85, color=C_ALL, linewidth=0, zorder=3)
        ax2.set_yscale("log"); ax2.set_ylim(0.3, 3.2)
        ax2.set_yticks([0.5, 1, 2, 3]); ax2.set_yticklabels(["0.5", "1", "2", "3"]); ax2.yaxis.set_minor_locator(matplotlib.ticker.NullLocator())
        ax2.axhline(1.0, color=INK2, lw=0.8, ls=(0, (3, 2)), zorder=2)
        ax2.axvline(cross - 0.5, color=INK2, lw=0.6, ls=(0, (1, 2)), zorder=2)
        ax2.text(cross + 3, 2.9, f"{n_ge1} scenarios at or above 1.0", fontsize=6, color=INK2, va="top", ha="left")
        ax2.text(cross - 3, 2.9, f"{cross} below", fontsize=6, color=INK2, va="top", ha="right")
        ax2.yaxis.grid(True, color=GRID, lw=0.5, zorder=0); ax2.set_axisbelow(True)
        ax2.set_xlim(-1, len(v)); ax2.set_xticks([])
        ax2.set_ylabel("speedup vs. fastest\navailable baseline"); ax2.set_xlabel(f"{len(v)} paired scenarios, sorted")
        for sp in ("top", "right"): ax2.spines[sp].set_visible(False)
        ax2.set_title(f"({letters[2*row+1]}) {title.split(',')[0]}: per scenario, vs. fastest available", loc="left", fontsize=7, pad=4)

        print(f"== {arch} ==")
        for g in groups:
            s = st[g]
            print(f"{g:9s} n={s['n']:3d} runnable={s['runnable']:3d} paired={s['paired']:3d} up={s['up']:.3f} oss={s['oss']:.3f} all={s['all']:.3f} "
                  f">=1: {s['up_ge1']}/{s['oss_ge1']}/{s['all_ge1']} min up={s['up_min']:.3f} all={s['all_min']:.3f} "
                  f"cudnn {s['cudnn_beat']}/{s['cudnn_n']} fi {s['fi_beat']}/{s['fi_n']} cudnn_fastest={s['cudnn_fastest']}")
    axes[0][0].legend(loc="lower left", bbox_to_anchor=(0.0, 1.18), ncol=3, frameon=False, handlelength=0.9, columnspacing=1.0, handletextpad=0.4, borderaxespad=0.0)
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    fig.savefig(a.out, bbox_inches="tight", pad_inches=0.02)
    fig.savefig(os.path.splitext(a.out)[0] + ".png", dpi=200, bbox_inches="tight", pad_inches=0.02)


if __name__ == "__main__":
    main()
