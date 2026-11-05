#!/usr/bin/env python3
"""Evaluation figures.

  figures/h2h.pdf      head-to-head on the baseline's own 26-config evaluator: equipped agent vs KernelSeek
  figures/evo.pdf      AVO-style evolutionary search on the Blackwell dense-prefill kernel, then one question
  figures/messages.pdf primary act of the human messages and the question categories, by campaign

Inputs (paths below): the baseline checkpoint's candidates.jsonl, the Cursor head-to-head session export,
and paper/analysis/traces/human_turns.jsonl. Prints the numbers used in the text.
"""
import json, os, re, sys, collections, math
from datetime import datetime, timedelta, timezone
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
FIG = os.path.join(HERE, "..", "figures")
SCRATCH = "/private/tmp/claude-501/-Users-goliaro-Desktop-fk-agent/d52db17f-2668-4b0a-8bd4-8ff6dfae9afd/scratchpad"
CAND = os.path.join(SCRATCH, "ckpt/cuda-agent-checkpoints/run_20260616_011948/candidates.jsonl")
CURSOR = ("/Users/goliaro/Desktop/fk-agent/flash-cuda/reference/flash-cuda-chats/fa3/cursor/"
          "69fff5d7-d194-4a1c-9ec8-de9efe54b617/69fff5d7-d194-4a1c-9ec8-de9efe54b617.jsonl")
TURNS = os.path.join(HERE, "..", "analysis", "traces", "human_turns.jsonl")

C1, C2, C3 = "#2a78d6", "#eb6834", "#1baf7a"     # validated categorical slots (dataviz skill)
INK, INK2, GRID = "#0b0b0b", "#52514e", "#d9d8d3"
plt.rcParams.update({"font.size": 7, "axes.labelsize": 7, "xtick.labelsize": 6.5, "ytick.labelsize": 6.5,
                     "legend.fontsize": 6.5, "axes.edgecolor": GRID, "axes.linewidth": 0.6,
                     "xtick.color": INK2, "ytick.color": INK2, "axes.labelcolor": INK, "text.color": INK,
                     "pdf.fonttype": 42})


def style(ax):
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.yaxis.grid(True, color=GRID, lw=0.5)
    ax.set_axisbelow(True)


def cursor_ts(s):
    m = re.match(r"\w+, (\w+ \d+, \d{4}), (\d+:\d+ [AP]M) \(UTC([+-]\d+)\)", s)
    dt = datetime.strptime(m.group(1) + " " + m.group(2), "%b %d, %Y %I:%M %p")
    return (dt - timedelta(hours=int(m.group(3)))).replace(tzinfo=timezone.utc)


# ---------------------------------------------------------------- head-to-head
def baseline_series():
    rows = [json.loads(l) for l in open(CAND)][:133]
    t0 = rows[0]["timestamp"]
    xs, gm, mn, best_gm, best_mn = [], [], [], [], []
    bg = bm = 0.0
    for r in rows:
        e = r.get("eval") or {}
        g = e.get("geomean_ratio") or 0.0
        m = e.get("min_ratio") or 0.0
        if r.get("is_best"):
            bg, bm = g, m
        xs.append((r["timestamp"] - t0) / 3600)
        gm.append(g); mn.append(m); best_gm.append(bg); best_mn.append(bm)
    return np.array(xs), np.array(gm), np.array(mn), np.array(best_gm), np.array(best_mn), rows


def kernelseek_series():
    """(ts, geomean, min) for every full 26-config result reported in the head-to-head session."""
    cur = None; pts = []
    for line in open(CURSOR, errors="ignore"):
        o = json.loads(line)
        if o.get("role") == "user":
            m = re.search(r"<timestamp>(.*?)</timestamp>", json.dumps(o))
            if m:
                cur = cursor_ts(m.group(1))
            continue
        s = json.dumps(o).replace("\\n", " ")
        for m in re.finditer(r"geomean[^0-9]{0,14}([0-9]\.[0-9]{2,3})", s):
            ctx = s[max(0, m.start() - 160): m.end() + 60]
            if re.search(r"26/26|all 26|26 config|FULL|[Ff]ull (?:suite|sweep|trial)|Full 26", ctx) and "weighted" not in s[m.start()-12:m.start()]:
                mm = re.search(r"min(?:_ratio)?[^0-9]{0,12}([0-9]\.[0-9]{2,3})", ctx)
                pts.append((cur, float(m.group(1)), float(mm.group(1)) if mm else None))
    # keep the running maximum reported per user-turn (reports are of the current best kernel)
    byts = collections.OrderedDict()
    for t, g, m in pts:
        if t is None: continue
        cur_g, cur_m = byts.get(t, (0, None))
        if g > cur_g:
            byts[t] = (g, m if m is not None else cur_m)
    out = [(t, g, m) for t, (g, m) in byts.items()]
    out.sort(key=lambda x: x[0])
    return out


def h2h_figure():
    xs, gm, mn, bgm, bmn, rows = baseline_series()
    ks = kernelseek_series()
    t0k = datetime(2026, 6, 20, 6, 33, tzinfo=timezone.utc)   # goal message: optimize for all 26 configs
    kx = np.array([(t - t0k).total_seconds() / 3600 for t, _, _ in ks])
    kg = np.array([g for _, g, _ in ks])
    km = [(x, m) for x, (_, _, m) in zip(kx, ks) if m is not None]
    # running best for KernelSeek too (what the lineage would keep)
    kbest = np.maximum.accumulate(kg)

    # human messages in the head-to-head window, by act
    turns = [json.loads(l) for l in open(TURNS)]
    turns = [t for t in turns if t["session"].startswith("69fff5d7") and t["ts"]]
    tq = [(datetime.fromisoformat(t["ts"]) - t0k).total_seconds() / 3600 for t in turns if t["label"] in ("Q", "C")]
    td = [(datetime.fromisoformat(t["ts"]) - t0k).total_seconds() / 3600 for t in turns if t["label"] not in ("Q", "C")]
    tq = [x for x in tq if 0 <= x <= 75]; td = [x for x in td if 0 <= x <= 75]

    fig, ax = plt.subplots(figsize=(3.3, 2.5))
    ax.axhline(1.0, color=INK2, lw=0.8, ls=(0, (3, 2)), zorder=1)
    # idle gap in the KernelSeek session (no human or agent activity)
    gap0 = (datetime(2026, 6, 21, 14, 45, tzinfo=timezone.utc) - t0k).total_seconds() / 3600
    gap1 = (datetime(2026, 6, 22, 17, 26, tzinfo=timezone.utc) - t0k).total_seconds() / 3600
    ax.axvspan(gap0, gap1, color="#f1f0ec", zorder=0)
    ax.text((gap0 + gap1) / 2, 0.62, "no activity\n(KernelSeek)", ha="center", va="center", fontsize=5.5, color=INK2)

    ax.plot(xs, gm, ".", color=C1, ms=2.2, alpha=0.55, zorder=2)
    ax.plot(xs, bgm, color=C1, lw=1.6, zorder=3, label="equipped agent, geomean")
    ax.plot(xs, bmn, color=C1, lw=1.0, ls=(0, (2, 1.5)), zorder=3, label="equipped agent, worst scenario")
    ax.plot(kx, kbest, color=C2, lw=1.6, zorder=4, label="KernelSeek, geomean")
    ax.plot(kx, kg, "o", color=C2, ms=2.6, zorder=5, mec="white", mew=0.4)
    if km:
        ax.plot([x for x, _ in km], [m for _, m in km], "s", color=C2, ms=2.6, zorder=5, mec="white", mew=0.4)
        ax.plot([x for x, _ in km], [m for _, m in km], color=C2, lw=1.0, ls=(0, (2, 1.5)), zorder=4, label="KernelSeek, worst scenario")
    # question ticks
    ax.plot(tq, [0.02] * len(tq), "|", color=C2, ms=5, mew=0.9, zorder=6)
    ax.plot(td, [0.02] * len(td), "|", color="#b8b7b0", ms=3.5, mew=0.6, zorder=5)
    ax.text(74.5, 0.07, "ticks: human turns (orange = question or correction)", fontsize=5, color=INK2, va="bottom", ha="right")

    ax.set_xlim(0, 75); ax.set_ylim(0, 1.25)
    ax.set_xlabel("wall-clock hours since the start of each campaign")
    ax.set_ylabel("speedup over FA3 (26 configurations)")
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0, 1.25])
    style(ax)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.22), ncol=2, frameon=False, handlelength=1.6, columnspacing=1.2, borderaxespad=0.0, labelspacing=0.3)
    fig.savefig(os.path.join(FIG, "h2h.pdf"), bbox_inches="tight", pad_inches=0.02)
    fig.savefig(os.path.join(FIG, "h2h.png"), dpi=200, bbox_inches="tight", pad_inches=0.02)

    print("== head-to-head ==")
    print(f"baseline: {len(rows)} steps over {xs[-1]:.1f} h; best gm {bgm[-1]:.3f} (step {int(np.argmax(bgm))}), "
          f"best min {bmn[-1]:.3f}; max per-step gm {gm.max():.3f} at step {int(np.argmax(gm))}; "
          f"correct {sum(1 for r in rows if r['status']=='correct')}/{len(rows)}")
    print(f"kernelseek: {len(ks)} full-suite reports; first {ks[0][0]} {ks[0][1]}; "
          f"first >=0.95 at {[ (t.strftime('%m-%d %H:%M'), g) for t,g,_ in ks if g>=0.95][0]}; "
          f"first >=1.0 at {[ (t.strftime('%m-%d %H:%M'), g) for t,g,_ in ks if g>=1.0][0]}; "
          f"best {max(g for _,g,_ in ks):.3f} at {max(ks, key=lambda x:x[1])[0]}")
    print(f"kernelseek human turns in window: {len(tq)} questions/corrections, {len(td)} other; hours to best "
          f"{(max(ks, key=lambda x:x[1])[0]-t0k).total_seconds()/3600:.1f}, idle gap {gap1-gap0:.1f} h")


# ---------------------------------------------------------------- evolution
def evo_figure():
    gens = ["seed", "gen 1", "gen 2", "gen 3", "gen 4", "question"]
    vals = [0.287, 0.675, 0.743, 0.765, 0.777, 1.00]
    fig, ax = plt.subplots(figsize=(3.3, 1.7))
    ax.axhline(1.0, color=INK2, lw=0.8, ls=(0, (3, 2)), zorder=1)
    x = np.arange(len(gens))
    cols = [C1] * 5 + [C2]
    ax.bar(x, vals, width=0.62, color=cols, linewidth=0, zorder=3)
    for xi, v in zip(x, vals):
        ax.text(xi, v + 0.025, f"{v:.2f}", ha="center", va="bottom", fontsize=6, color=INK2)
    ax.set_xticks(x); ax.set_xticklabels(gens)
    ax.set_ylim(0, 1.18); ax.set_yticks([0, 0.5, 1.0])
    ax.set_ylabel("speedup over FA4\n(hd128 dense prefill)")
    style(ax)
    fig.savefig(os.path.join(FIG, "evo.pdf"), bbox_inches="tight", pad_inches=0.02)
    fig.savefig(os.path.join(FIG, "evo.png"), dpi=200, bbox_inches="tight", pad_inches=0.02)


# ---------------------------------------------------------------- messages
ACT = {"D": "directive", "Q": "question", "A": "approval", "C": "correction", "G": "goal"}
CAT = {"progress_measurement": "measurement check", "reference_diff": "bound differencing",
       "benchmark_coverage_fairness": "benchmark validity", "gap_triage": "gap triage",
       "agent_process": "agent process", "limit_challenge": "challenge a limit",
       "strategy_options": "strategy options", "hypothesis_check": "hypothesis check", "repo_meta": "repo / housekeeping"}


def messages_figure():
    turns = [json.loads(l) for l in open(TURNS)]
    # dedupe exact repeats of (session, text)
    seen = set(); dd = []
    for t in turns:
        k = (t["session"], t.get("ts"), t["text"][:200])
        if k in seen: continue
        seen.add(k); dd.append(t)
    print(f"== messages == {len(turns)} rows, {len(dd)} after dedupe")
    acts = collections.Counter(t["label"] for t in dd)
    acts_t = {tg: collections.Counter(t["label"] for t in dd if t["target"] == tg) for tg in ("FA3", "FA4")}
    print("acts:", dict(acts), {k: dict(v) for k, v in acts_t.items()})
    interrog = sum(1 for t in dd if t.get("interrogative"))
    print(f"phrased as a question: {interrog}/{len(dd)} = {100*interrog/len(dd):.1f}%")
    cats = collections.Counter(c for t in dd for c in (t.get("question_categories") or []))
    cats_t = {tg: collections.Counter(c for t in dd if t["target"] == tg for c in (t.get("question_categories") or [])) for tg in ("FA3", "FA4")}
    print("categories:", dict(cats)); print({k: dict(v) for k, v in cats_t.items()})
    shared = set(cats_t["FA3"]) & set(cats_t["FA4"]); print("categories present in both campaigns:", sorted(shared), "of", len(cats))
    # operator-specificity of question-like messages
    ql = [t for t in dd if t["label"] in ("Q", "C")]
    up = re.compile(r"\b(FA3|FA4|flash[- ]?attention|cutlass|cute|cudnn|flashinfer|trt)", re.I)
    attn = re.compile(r"\b(attention|softmax|kv|k/v|decode|prefill|head|causal|varlen|paged|gqa|mqa|split)", re.I)
    n_up = sum(1 for t in ql if up.search(t["text"])); n_attn = sum(1 for t in ql if attn.search(t["text"]))
    print(f"question-like messages: {len(ql)}; mention an upstream/reference: {n_up}; mention attention-specific terms: {n_attn}")

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(3.3, 3.3), gridspec_kw=dict(height_ratios=[5, 8], hspace=0.45))
    order = ["D", "Q", "A", "C", "G"]
    y = np.arange(len(order))
    fa3 = [acts_t["FA3"][k] for k in order]; fa4 = [acts_t["FA4"][k] for k in order]
    ax1.barh(y, fa3, color=C1, height=0.62, linewidth=0, label="Hopper campaign", zorder=3)
    ax1.barh(y, fa4, left=fa3, color=C2, height=0.62, linewidth=0, label="Blackwell campaign", zorder=3)
    for yi, (a, b) in enumerate(zip(fa3, fa4)):
        ax1.text(a + b + 1, yi, str(a + b), va="center", fontsize=6, color=INK2)
    ax1.set_yticks(y); ax1.set_yticklabels([ACT[k] for k in order]); ax1.invert_yaxis()
    ax1.set_xlim(0, max(np.array(fa3) + np.array(fa4)) * 1.25)
    ax1.set_title(f"(a) primary act of {len(dd)} messages", loc="left", fontsize=7, pad=3)
    ax1.xaxis.grid(True, color=GRID, lw=0.5); ax1.yaxis.grid(False); ax1.set_axisbelow(True)
    for sp in ("top", "right"): ax1.spines[sp].set_visible(False)
    ax1.legend(loc="center right", frameon=False, handlelength=0.9, fontsize=6, borderaxespad=0.3)

    corder = [k for k, _ in cats.most_common() if k != "repo_meta"]
    y2 = np.arange(len(corder))
    c3 = [cats_t["FA3"][k] for k in corder]; c4 = [cats_t["FA4"][k] for k in corder]
    ax2.barh(y2, c3, color=C1, height=0.62, linewidth=0, zorder=3)
    ax2.barh(y2, c4, left=c3, color=C2, height=0.62, linewidth=0, zorder=3)
    for yi, (a, b) in enumerate(zip(c3, c4)):
        ax2.text(a + b + 0.3, yi, str(a + b), va="center", fontsize=6, color=INK2)
    ax2.set_yticks(y2); ax2.set_yticklabels([CAT[k] for k in corder]); ax2.invert_yaxis()
    ax2.set_xlim(0, max(np.array(c3) + np.array(c4)) * 1.25)
    ax2.set_title("(b) question categories", loc="left", fontsize=7, pad=3)
    ax2.xaxis.grid(True, color=GRID, lw=0.5); ax2.yaxis.grid(False); ax2.set_axisbelow(True)
    for sp in ("top", "right"): ax2.spines[sp].set_visible(False)
    fig.savefig(os.path.join(FIG, "messages.pdf"), bbox_inches="tight", pad_inches=0.02)
    fig.savefig(os.path.join(FIG, "messages.png"), dpi=200, bbox_inches="tight", pad_inches=0.02)


if __name__ == "__main__":
    os.makedirs(FIG, exist_ok=True)
    h2h_figure(); evo_figure(); messages_figure()
