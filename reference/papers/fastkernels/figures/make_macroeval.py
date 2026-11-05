#!/usr/bin/env python3
"""MacroEval (sections/metrics.tex) on ``fastkernels e2e`` results.

    python figures/make_macroeval.py --results OUT/results [--out-json figures/data/macroeval.json]
        [--tables-dir tables] [--universe all|present|complete]

Input (written by ``fastkernels e2e``, see fastkernels/e2e/core.py): one directory per
scenario, ``OUT/results/<NN>_<model_slug>/``, holding ``baseline.json``, ``noise.json``
(the baseline re-run in a fresh process and scored against the baseline: the run-to-run
nondeterminism floor) and one ``<set>.json`` per candidate set.

Scoring (every knob lives in ``CONFIG`` below):

* items i = scenarios (models), grouped into the paper's architecture families f;
* per-sample calibrated correctness C = 1 if d <= g_i, (f_i - d) / (f_i - g_i) if
  g_i < d < f_i, 0 if d >= f_i, where g_i = max(g_floor, q-quantile of the noise run's
  per-sample d), capped at g_cap * f_i, and f_i is set per family;
* C_i = mean over samples; valid_i = (C_i >= tau) for runs that finished with speedups;
  crashed / empty / incomplete / baseline-failed / missing items are invalid with C_i = 0;
* s_i = s_thr^lam * s_lat^(1 - lam), with s_thr / s_lat the geomeans of the throughput /
  latency workload speedups (kind taken from the baseline timing of that workload);
  S_f = geomean of s_i over the valid items of f; S_macro = geomean of S_f over families
  with a valid item; C_macro and Coverage_macro = means over families of the per-family
  means; Score = S_macro * C_macro * Coverage_macro.

Two views per candidate set: *deployable* (the subset left after drop-and-retry, set
status ``ok``) and *all winners* (the chain's first attempt, every winner kernel active;
valid only if that run finished). The orchestrator stores per-sample d only for the final
run of a chain, so an all-winners run that is not itself the final run is scored on its
mean d as a single-sample proxy. That happens only for runs the orchestrator judged
broken (mean d > broken_threshold = 0.5): they cannot reach tau = 0.9 for any f_i <= 1
unless g_i > 0.44 (mean C <= 0.5 / (1 - g_i)), so the proxy never changes validity; it can
only understate their C_i (0 under the default f = 0.5).

Per agent, the two candidate sets are also pooled into one leaderboard row: each
(model, set) pair is a separate item of its family.

Outputs: <out-json> (all numbers) and, in <tables-dir>, e2e_macroeval.tex, e2e_drops.tex,
e2e_per_model.tex and e2e_sensitivity.tex (booktabs tabulars, no table wrapper). When
--results points inside figures/data/e2e_fixture/, outputs default to that directory so
synthetic numbers never land in tables/.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from collections import Counter
from pathlib import Path

PAPER_DIR = Path(__file__).resolve().parent.parent
FIXTURE_DIR = PAPER_DIR / "figures" / "data" / "e2e_fixture"
OUT_JSON = PAPER_DIR / "figures" / "data" / "macroeval.json"
TABLES_DIR = PAPER_DIR / "tables"

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
FAMILIES = [
    "Dense & MoE LLMs",
    "Linear Attention & New Archs",
    "Vision / Video / Audio",
    "Multimodal & Encoders",
    "Edge & Detection",
    "3D / Robotics / Science",
    "Recommendation & Specialized",
    "World Models",
]
# default.yaml order (the <NN> prefix of the result directories): (HF model, label, family)
SCENARIOS = [
    ("meta-llama/Llama-3.1-8B-Instruct", "Llama-3.1-8B", FAMILIES[0]),
    ("Qwen/Qwen3-Next-80B-A3B-Instruct", "Qwen3-Next-80B", FAMILIES[1]),
    ("moonshotai/Kimi-Linear-48B-A3B-Instruct", "Kimi-Linear-48B", FAMILIES[1]),
    ("openai/gpt-oss-120b", "GPT-OSS-120B", FAMILIES[0]),
    ("fla-hub/gla-2.7B-100B", "GLA-2.7B", FAMILIES[1]),
    ("black-forest-labs/FLUX.1-dev", "FLUX.1-dev", FAMILIES[2]),
    ("Qwen/Qwen3-VL-235B-A22B-Instruct-FP8", "Qwen3-VL-235B", FAMILIES[3]),
    ("jameslahm/yolov10n", "YOLOv10n", FAMILIES[4]),
    ("OpenFold/OpenFold3", "OpenFold3", FAMILIES[5]),
    ("BAAI/bge-m3", "BGE-M3", FAMILIES[6]),
    ("Etched/oasis-500m", "Oasis-500M", FAMILIES[7]),
]
# Candidate sets are named <agent key>-<mode key> (e.g. claude-ind).
AGENTS = [("drkernel", "Dr.~Kernel"), ("claude", "Claude Code"), ("kda", "KDA"), ("ako", "AKO")]
MODES = [("ind", "Independent"), ("seq", "Sequential")]
POOLED_LABEL = "Ind.\\,+\\,Seq."

CONFIG = {
    # validity: valid_i = C_i >= tau
    "tau": 0.9,
    # g_i = max(g_floor, g_quantile of the noise run's per-sample d), then capped at g_cap * f_i;
    # a perturb.json (fastkernels.e2e.perturb: benign-numerics run, for models whose noise run
    # is bit-identical) takes precedence over the noise run
    "g_quantile": 0.95,
    "g_floor": 1e-3,
    # g_i when the noise run is missing, failed or has no correctness data
    "g_fallback": 0.02,
    "g_cap": 0.5,
    # f_i: discrepancy at which the output counts as unusable, per family
    "f": {fam: 0.5 for fam in FAMILIES},
    # per-model overrides, {hf_name: value}; applied before the sensitivity scaling
    "g_override": {},
    "f_override": {},
    # s_i = s_thr^lam * s_lat^(1 - lam); "blend" is the default leaderboard
    "lambdas": {"blend": 0.5, "thr": 1.0, "lat": 0.0},
    # mirrors `fastkernels e2e --broken-threshold` (only used to label all-winners runs "broken")
    "broken_threshold": 0.5,
    # a model with an empty deployable subset (every kernel dropped, or none requested):
    # "invalid" -> no coverage, C_i = 0; "baseline" -> valid with C_i = 1 and s_i = 1
    "empty_policy": "invalid",
    # threshold-sensitivity sweep: g_i and f_i are scaled by these factors
    "sensitivity_scales": [0.5, 1.0, 2.0],
}

RAN = ("ok", "broken")  # item outcomes for which a run finished and is scored


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------
def slug(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", text).strip("_")


def finite(x) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x)


def geomean(values) -> float | None:
    vals = [v for v in values if finite(v) and v > 0]
    return math.exp(sum(math.log(v) for v in vals) / len(vals)) if vals else None


def mean(values) -> float | None:
    vals = [v for v in values if finite(v)]
    return sum(vals) / len(vals) if vals else None


def quantile(values: list[float], q: float) -> float:
    """Linear-interpolation quantile (numpy's default)."""
    xs = sorted(values)
    pos = q * (len(xs) - 1)
    lo = math.floor(pos)
    hi = min(lo + 1, len(xs) - 1)
    return xs[lo] + (xs[hi] - xs[lo]) * (pos - lo)


def clean_d(values) -> list[float]:
    """Per-sample discrepancies clipped to [0, 1]; NaN / missing count as maximal (1)."""
    return [min(1.0, max(0.0, float(v))) if finite(v) else 1.0 for v in values or []]


def calibrated(d: float, g: float, f: float) -> float:
    if d <= g:
        return 1.0
    if d >= f:
        return 0.0
    return (f - d) / (f - g)


def set_name(agent: str, mode: str) -> str:
    return f"{agent}-{mode}"


def all_sets() -> list[str]:
    return [set_name(a, m) for a, _ in AGENTS for m, _ in MODES]


def read_json(path: Path, warnings: list[str]) -> dict | None:
    try:
        return json.loads(path.read_text())
    except FileNotFoundError:
        return None
    except (OSError, json.JSONDecodeError) as exc:  # e.g. caught mid-write
        warnings.append(f"{path.parent.name}/{path.name}: unreadable ({exc}); treated as missing")
        return None


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------
def load_results(results: Path, warnings: list[str]) -> dict[int, dict]:
    """{scenario index: {"dir", "baseline", "noise", "sets": {set: json|None}, "extra"}}."""
    by_model = {m: i for i, (m, _, _) in enumerate(SCENARIOS)}
    by_slug = {slug(m): i for i, (m, _, _) in enumerate(SCENARIOS)}
    raw: dict[int, dict] = {}
    if not results.is_dir():
        warnings.append(f"results directory {results} does not exist")
        return raw
    for d in sorted(p for p in results.iterdir() if p.is_dir()):
        base = read_json(d / "baseline.json", warnings)
        idx = by_model.get((base or {}).get("model"))
        if idx is None:
            m = re.match(r"\d+_(.*)$", d.name)
            idx = by_slug.get(m.group(1) if m else d.name)
        if idx is None:
            warnings.append(f"{d.name}: not a known scenario (see SCENARIOS); ignored")
            continue
        if idx in raw:
            warnings.append(f"{d.name}: duplicate of {raw[idx]['dir']}; ignored")
            continue
        known = {"baseline", "noise", "perturb", *all_sets()}
        raw[idx] = {
            "dir": d.name,
            "baseline": base,
            "noise": read_json(d / "noise.json", warnings),
            "perturb": read_json(d / "perturb.json", warnings) if (d / "perturb.json").is_file() else None,
            "sets": {s: read_json(d / f"{s}.json", warnings) for s in all_sets()},
            "extra": sorted(p.stem for p in d.glob("*.json") if p.stem not in known),
        }
    return raw


def workload_kinds(raw_i: dict, run: dict | None = None) -> dict[str, str]:
    kinds = {}
    for src in (run or {}).get("timings") or {}, (raw_i.get("baseline") or {}).get("timings") or {}:
        for wl, t in src.items():  # baseline last: it wins
            if isinstance(t, dict) and t.get("kind") in ("throughput", "latency"):
                kinds[wl] = t["kind"]
    return kinds


def split_speedups(speedups: dict | None, kinds: dict[str, str]) -> dict:
    thr, lat, unknown = [], [], []
    for wl, x in (speedups or {}).items():
        if not finite(x) or x <= 0:
            continue
        {"throughput": thr, "latency": lat}.get(kinds.get(wl), unknown).append(x)
    return {"s_thr": geomean(thr), "s_lat": geomean(lat), "n_thr": len(thr), "n_lat": len(lat),
            "unknown_workloads": unknown}


def blend(s_thr: float | None, s_lat: float | None, lam: float) -> float | None:
    """s_thr^lam * s_lat^(1-lam); an item lacking one kind uses the other for every lambda."""
    if s_thr is None or s_lat is None:
        return s_thr if s_lat is None else s_lat
    return s_thr ** lam * s_lat ** (1.0 - lam)


def per_sample(run: dict | None) -> list[float]:
    return clean_d(((run or {}).get("correctness") or {}).get("per_sample"))


# ---------------------------------------------------------------------------
# Calibration
# ---------------------------------------------------------------------------
def calibrate(raw: dict[int, dict], cfg: dict, g_scale: float = 1.0, f_scale: float = 1.0) -> dict[int, dict]:
    out = {}
    for i, (model, _, fam) in enumerate(SCENARIOS):
        raw_i = raw.get(i) or {}
        noise = raw_i.get("noise")
        ok = bool(noise) and noise.get("status") == "ok"
        ds = per_sample(noise) if ok else []
        if ok and not ds and finite(noise.get("mean_d")):
            ds, d_src = clean_d([noise["mean_d"]]), "mean_d"
        else:
            d_src = "per_sample"
        f0 = cfg["f_override"].get(model, cfg["f"][fam])
        pert = raw_i.get("perturb")
        pds = per_sample(pert) if pert and pert.get("status") == "ok" else []
        if model in cfg["g_override"]:
            g0, src = cfg["g_override"][model], "override"
        elif pds:
            g0 = max(cfg["g_floor"], quantile(pds, cfg["g_quantile"]))
            src = f"perturb p{100 * cfg['g_quantile']:g} (per_sample)"
        elif ds:
            g0 = max(cfg["g_floor"], quantile(ds, cfg["g_quantile"]))
            src = f"noise p{100 * cfg['g_quantile']:g} ({d_src})"
        else:
            g0, src = cfg["g_fallback"], "fallback (no usable noise run)"
        f = f0 * f_scale
        g = g0 * g_scale
        rec = {"g": min(g, cfg["g_cap"] * f), "f": f, "g_source": src, "g_capped": g > cfg["g_cap"] * f,
               "noise_status": None if noise is None else noise.get("status"),
               "noise_n": len(ds), "noise_mean_d": mean(ds),
               "noise_quantile_d": quantile(ds, cfg["g_quantile"]) if ds else None}
        if ok:
            sp = split_speedups(noise.get("speedups"), workload_kinds(raw_i, noise))
            rec.update(noise_s_thr=sp["s_thr"], noise_s_lat=sp["s_lat"],
                       noise_s=blend(sp["s_thr"], sp["s_lat"], cfg["lambdas"]["blend"]))
        out[i] = rec
    return out


# ---------------------------------------------------------------------------
# Items
# ---------------------------------------------------------------------------
def evaluate_item(i: int, raw_i: dict | None, sname: str, view: str, cal: dict, cfg: dict) -> dict:
    """One (model, set, view) item: outcome, C_i, validity and speedups."""
    model, _, fam = SCENARIOS[i]
    rec = {"model": model, "family": fam, "outcome": None, "detail": None, "valid": False, "C": 0.0,
           "n_samples": 0, "d_source": None, "mean_d": None, "s_thr": None, "s_lat": None,
           "s": {k: None for k in cfg["lambdas"]}, "n_dropped": None}
    if raw_i is None or raw_i.get("baseline") is None:
        rec["outcome"] = "missing"
        return rec
    if raw_i["baseline"].get("status") != "ok":
        rec["outcome"] = "baseline_failed"
        return rec
    r = raw_i["sets"].get(sname)
    if r is None:
        rec["outcome"] = "missing"
        return rec
    st = r.get("status")
    rec["n_dropped"] = len(r.get("dropped") or [])
    if st in ("baseline_failed", "orchestrator_error"):
        rec["outcome"] = st
        return rec
    attempts = r.get("attempts") or []
    run, ds = None, []
    if view == "deployable":
        if st == "ok":
            rec["outcome"], run, ds = "ok", r, per_sample(r)
        elif st == "empty":
            rec["outcome"] = "empty"
        else:
            rec["outcome"] = st or "unknown"
    else:
        aw = r.get("all_winners") or (attempts[0] if attempts else None)
        if aw is None:
            rec["outcome"] = "empty" if st == "empty" else "no_attempt"
        elif aw.get("status") != "ok":
            rec["outcome"], rec["detail"] = "crash", aw.get("category") or aw.get("status")
        else:
            run = aw
            broken = finite(aw.get("mean_d")) and aw["mean_d"] > cfg["broken_threshold"]
            rec["outcome"] = "broken" if broken else "ok"
            # the all-winners attempt is the chain's final run -> its per-sample d are stored
            if st == "ok" and attempts and attempts[-1].get("n") == aw.get("n"):
                ds = per_sample(r)

    if rec["outcome"] == "empty" and cfg["empty_policy"] == "baseline":
        rec.update(valid=True, C=1.0, s_thr=1.0, s_lat=1.0, s={k: 1.0 for k in cfg["lambdas"]},
                   detail="empty (scored as baseline)")
        return rec
    if rec["outcome"] not in RAN:
        return rec

    if ds:
        rec["d_source"] = "per_sample"
    elif finite(run.get("mean_d")):
        ds, rec["d_source"] = clean_d([run["mean_d"]]), "mean_d_proxy"
    sp = split_speedups(run.get("speedups"), workload_kinds(raw_i, run if view == "deployable" else None))
    rec.update(s_thr=sp["s_thr"], s_lat=sp["s_lat"], n_thr=sp["n_thr"], n_lat=sp["n_lat"])
    if sp["unknown_workloads"]:
        rec["unknown_workloads"] = sp["unknown_workloads"]
    rec["s"] = {k: blend(sp["s_thr"], sp["s_lat"], lam) for k, lam in cfg["lambdas"].items()}
    if not ds:
        rec["detail"] = "no correctness data"
        return rec
    g, f = cal[i]["g"], cal[i]["f"]
    rec["n_samples"] = len(ds)
    rec["mean_d"] = mean(ds)
    rec["C"] = mean(calibrated(d, g, f) for d in ds)
    if rec["s"]["blend"] is None:
        rec["detail"] = "no speedups"
        return rec
    rec["valid"] = rec["C"] >= cfg["tau"] - 1e-12
    return rec


def aggregate(items: list[dict], cfg: dict) -> dict:
    """MacroEval over a list of items (families = those with at least one item)."""
    fams = [f for f in FAMILIES if any(it["family"] == f for it in items)]
    per_fam = {}
    for fam in fams:
        its = [it for it in items if it["family"] == fam]
        valid = [it for it in its if it["valid"]]
        per_fam[fam] = {"n": len(its), "n_valid": len(valid),
                        "C_f": mean(it["C"] for it in its),
                        "coverage_f": len(valid) / len(its),
                        "S_f": {k: geomean(it["s"][k] for it in valid) for k in cfg["lambdas"]}}
    n_valid = sum(it["valid"] for it in items)
    c_macro = mean(per_fam[f]["C_f"] for f in fams)
    cov_macro = mean(per_fam[f]["coverage_f"] for f in fams)
    s_macro = {k: geomean(per_fam[f]["S_f"][k] for f in fams) for k in cfg["lambdas"]}
    return {
        "n_items": len(items), "n_valid": n_valid,
        "coverage": n_valid / len(items) if items else None,
        "coverage_macro": cov_macro, "C_macro": c_macro,
        "S_macro": s_macro,
        "score": {k: (s * c_macro * cov_macro if s is not None else 0.0) if items else None
                  for k, s in s_macro.items()},
        "n_families": len(fams), "n_families_valid": sum(per_fam[f]["n_valid"] > 0 for f in fams),
        "families": per_fam,
    }


def drop_stats(raw: dict[int, dict], sname: str, universe: list[int], cfg: dict) -> dict:
    status = Counter()
    cats = Counter()
    aw_crash_cats = Counter()
    out = {"models": len(universe), "requested": 0, "kept": 0, "unresolved": 0,
           "aw_crash": 0, "aw_broken": 0, "aw_none": 0, "empty": 0, "empty_none_requested": 0,
           "dropped_unique_kernels": 0}
    dropped_kernels = set()
    for i in universe:
        raw_i = raw.get(i)
        r = None if raw_i is None or raw_i.get("baseline") is None else raw_i["sets"].get(sname)
        if r is None:
            status["missing"] += 1
            continue
        st = r.get("status") or "unknown"
        status[st] += 1
        if st in ("baseline_failed", "orchestrator_error"):
            continue
        requested = r.get("kernels_requested") or []
        dropped = r.get("dropped") or []
        out["requested"] += len(requested)
        if st == "ok":  # kernels_final is the final attempt's `only`: it still lists import drops
            out["kept"] += len(set(r.get("kernels_final") or []) - {d.get("kernel") for d in dropped})
        for d in dropped:
            cats[d.get("category") or "unknown"] += 1
            dropped_kernels.add(d.get("kernel"))
        if st not in ("ok", "empty"):
            out["unresolved"] += len(requested) - len(dropped)
        aw = r.get("all_winners") or ((r.get("attempts") or [None])[0])
        if aw is None:
            out["aw_none"] += 1
        elif aw.get("status") != "ok":
            out["aw_crash"] += 1
            aw_crash_cats[aw.get("category") or "unknown"] += 1
        elif finite(aw.get("mean_d")) and aw["mean_d"] > cfg["broken_threshold"]:
            out["aw_broken"] += 1
        if st == "empty":
            out["empty"] += 1
            out["empty_none_requested"] += not requested
    out["dropped"] = sum(cats.values())
    out["dropped_by_category"] = dict(cats)
    out["dropped_unique_kernels"] = len(dropped_kernels)
    out["aw_crash_by_category"] = dict(aw_crash_cats)
    out["status"] = dict(status)
    return out


# ---------------------------------------------------------------------------
# Scoring passes
# ---------------------------------------------------------------------------
def score_all(raw: dict[int, dict], universe: list[int], cal: dict, cfg: dict) -> dict:
    items = {s: {view: [evaluate_item(i, raw.get(i), s, view, cal, cfg) for i in universe]
                 for view in ("deployable", "all_winners")} for s in all_sets()}
    sets = {s: {view: aggregate(items[s][view], cfg) for view in items[s]} for s in all_sets()}
    agents = {a: {view: aggregate([it for m, _ in MODES for it in items[set_name(a, m)][view]], cfg)
                  for view in ("deployable", "all_winners")} for a, _ in AGENTS}
    return {"items": items, "sets": sets, "agents": agents}


def ranking(scores: dict[str, float | None]) -> dict[str, int]:
    """Competition ranking by score (1 = best); None ranks last."""
    val = {k: (v if v is not None else -math.inf) for k, v in scores.items()}
    return {k: 1 + sum(w > val[k] + 1e-12 for w in val.values()) for k in val}


def discordant(r1: dict[str, int], r2: dict[str, int]) -> int:
    keys = list(r1)
    return sum((r1[a] - r1[b]) * (r2[a] - r2[b]) < 0
               for x, a in enumerate(keys) for b in keys[x + 1:])


def sensitivity(raw: dict[int, dict], universe: list[int], cfg: dict) -> dict:
    scales = cfg["sensitivity_scales"]

    def one(gs: float, fs: float) -> dict:
        res = score_all(raw, universe, calibrate(raw, cfg, gs, fs), cfg)
        s_scores = {s: res["sets"][s]["deployable"]["score"]["blend"] for s in all_sets()}
        a_scores = {a: res["agents"][a]["deployable"]["score"]["blend"] for a, _ in AGENTS}
        return {"g_scale": gs, "f_scale": fs,
                "set_scores": s_scores, "set_ranks": ranking(s_scores),
                "agent_scores": a_scores, "agent_ranks": ranking(a_scores),
                "agent_coverage_macro": {a: res["agents"][a]["deployable"]["coverage_macro"] for a, _ in AGENTS},
                "agent_C_macro": {a: res["agents"][a]["deployable"]["C_macro"] for a, _ in AGENTS}}

    joint = {f"{k:g}x": one(k, k) for k in scales}
    grid = {f"g{gs:g}x_f{fs:g}x": one(gs, fs) for gs in scales for fs in scales}
    ref = joint.get("1x") or one(1.0, 1.0)
    for d in list(joint.values()) + list(grid.values()):
        d["agent_discordant_pairs_vs_1x"] = discordant(ref["agent_ranks"], d["agent_ranks"])
        d["set_discordant_pairs_vs_1x"] = discordant(ref["set_ranks"], d["set_ranks"])
        d["agent_top_changed"] = (min(d["agent_ranks"], key=d["agent_ranks"].get)
                                  != min(ref["agent_ranks"], key=ref["agent_ranks"].get))
    return {"joint": joint, "grid": grid}


# ---------------------------------------------------------------------------
# LaTeX
# ---------------------------------------------------------------------------
def fx(v: float | None, nd: int = 2) -> str:
    return "---" if v is None else f"{v:.{nd}f}$\\times$"


def fn(v: float | None, nd: int = 2) -> str:
    return "---" if v is None else f"{v:.{nd}f}"


def header_comment(meta: dict) -> str:
    return (f"% Generated by figures/make_macroeval.py ({meta['n_with_baseline']}/{len(SCENARIOS)} scenarios "
            f"with a baseline, universe={meta['universe']}: {meta['n_items']} items, "
            f"{meta['n_families']} families). Do not edit by hand.")


def agent_rows(render, pooled: bool = True) -> list[str]:
    """Rows grouped by agent: one per mode (plus the pooled row), like tables/kernel_levels.tex."""
    lines = []
    labels = [(m, lab) for m, lab in MODES] + ([("pooled", POOLED_LABEL)] if pooled else [])
    for x, (a, name) in enumerate(AGENTS):
        if x:
            lines.append("\\midrule")
        for y, (m, lab) in enumerate(labels):
            agent_cell = f"\\multirow{{{len(labels)}}}{{*}}{{{name}}}" if y == 0 else ""
            lines.append(f"{agent_cell} & {lab} & " + " & ".join(render(a, m)) + " \\\\")
    return lines


def table_macroeval(res: dict, meta: dict) -> str:
    def cells(a: str, m: str) -> list[str]:
        src = res["agents"][a] if m == "pooled" else res["sets"][set_name(a, m)]
        d, w = src["deployable"], src["all_winners"]
        return [f"{d['n_valid']}/{d['n_items']}", fn(d["coverage_macro"]), fn(d["C_macro"]),
                fx(d["S_macro"]["blend"]), fx(d["S_macro"]["thr"]), fx(d["S_macro"]["lat"]),
                fn(d["score"]["blend"]),
                f"{w['n_valid']}/{w['n_items']}", fn(w["coverage_macro"])]

    lines = [
        header_comment(meta),
        "\\begin{tabular}{@{}ll rrr rrr r rr@{}}",
        "\\toprule",
        " & & \\multicolumn{7}{c}{\\textbf{Deployable subset}} & \\multicolumn{2}{c}{\\textbf{All winners}} \\\\",
        "\\cmidrule(lr){3-9}\\cmidrule(lr){10-11}",
        " & & & & & \\multicolumn{3}{c}{$S_{\\mathrm{macro}}$} & & & \\\\",
        "\\cmidrule(lr){6-8}",
        "Agent & Mode & Valid & Cov$_{\\mathrm{macro}}$ & $C_{\\mathrm{macro}}$ & $\\lambda{=}0.5$ & Thr. & Lat."
        " & Score & Valid & Cov$_{\\mathrm{macro}}$ \\\\",
        "\\midrule",
        *agent_rows(cells, pooled=False),  # main-text table: per-mode rows only (space)
        "\\bottomrule",
        "\\end{tabular}",
        "",
    ]
    return "\n".join(lines)


DROP_CATS = [("compile", "Comp."), ("runtime", "Run."), ("timeout", "T/O"), ("import", "Imp."),
             ("incorrect", "Inc.")]


def table_drops(drops: dict, meta: dict) -> str:
    def cells(a: str, m: str) -> list[str]:
        if m == "pooled":
            ds = [drops[set_name(a, mm)] for mm, _ in MODES]
        else:
            ds = [drops[set_name(a, m)]]

        def tot(key: str) -> int:
            return sum(d[key] for d in ds)

        def cat(c: str) -> int:
            return sum(d["dropped_by_category"].get(c, 0) for d in ds)

        def st(*keys: str) -> int:
            return sum(d["status"].get(k, 0) for d in ds for k in keys)

        return [str(tot("requested")), str(tot("kept")), *(str(cat(c)) for c, _ in DROP_CATS),
                str(tot("aw_crash")), str(tot("aw_broken")), str(tot("empty")),
                str(st("incomplete", "orchestrator_error", "unknown")), str(st("baseline_failed", "missing"))]

    lines = [
        header_comment(meta),
        "\\begin{tabular}{@{}ll rr rrrrr rrrrr@{}}",
        "\\toprule",
        " & & \\multicolumn{2}{c}{\\textbf{Kernels}} & \\multicolumn{5}{c}{\\textbf{Dropped, by cause}}"
        " & \\multicolumn{5}{c}{\\textbf{Models}} \\\\",
        "\\cmidrule(lr){3-4}\\cmidrule(lr){5-9}\\cmidrule(lr){10-14}",
        "Agent & Mode & Req. & Kept & " + " & ".join(lab for _, lab in DROP_CATS)
        + " & \\makecell{AW\\\\crash} & \\makecell{AW\\\\broken} & Empty & \\makecell{Not\\\\conv.} & N/A \\\\",
        "\\midrule",
        *agent_rows(cells),
        "\\bottomrule",
        "\\end{tabular}",
        "",
    ]
    return "\n".join(lines)


AW_CODE = {"ok": "ok", "broken": "broken", "crash": "crash", "empty": "none", "no_attempt": "none"}
DEP_CODE = {"ok": "ok", "empty": "empty", "incomplete": "n.c."}
WHOLE_ROW = {"missing": "---", "baseline_failed": "baseline failed", "orchestrator_error": "orchestrator error"}


def table_per_model(res: dict, cal: dict, universe: list[int], cfg: dict, meta: dict) -> str:
    order = sorted(universe, key=lambda i: (FAMILIES.index(SCENARIOS[i][2]), i))
    width = 5

    def block(i: int, a: str, m: str) -> list[str]:
        s = set_name(a, m)
        pos = universe.index(i)
        dep = res["items"][s]["deployable"][pos]
        aw = res["items"][s]["all_winners"][pos]
        if dep["outcome"] in WHOLE_ROW:
            return [f"\\multicolumn{{{width}}}{{c}}{{{WHOLE_ROW[dep['outcome']]}}}"]
        status = f"{AW_CODE.get(aw['outcome'], aw['outcome'])}\\,/\\,{DEP_CODE.get(dep['outcome'], dep['outcome'])}"
        d = fn(dep["mean_d"], 3)
        if dep["outcome"] in RAN and not dep["valid"] and dep["mean_d"] is not None:
            d += "$^\\dagger$"
        return [status, "---" if dep["n_dropped"] is None else str(dep["n_dropped"]),
                fn(dep["s_thr"]), fn(dep["s_lat"]), d]

    sub = "AW\\,/\\,Dep. & Drop & $s_{\\mathrm{thr}}$ & $s_{\\mathrm{lat}}$ & $\\bar d$"
    lines = [
        header_comment(meta),
        "\\begin{tabular}{@{}lrrl" + "".join(["@{\\hspace{12pt}}lrrrr"] * len(MODES)) + "@{}}",
        "\\toprule",
        " & \\multicolumn{2}{c}{\\textbf{Noise}} & & "
        + " & ".join(f"\\multicolumn{{{width}}}{{c}}{{\\textbf{{{lab}}}}}" for _, lab in MODES) + " \\\\",
        "\\cmidrule(lr){2-3}" + "".join(f"\\cmidrule(lr){{{5 + width * x}-{4 + width * (x + 1)}}}"
                                        for x in range(len(MODES))),
        "Model & $\\bar d$ & $g_i$ & Agent & " + " & ".join([sub] * len(MODES)) + " \\\\",
        "\\midrule",
    ]
    for x, i in enumerate(order):
        if x:
            lines.append("\\midrule")
        c = cal[i]
        noise_d = fn(c["noise_mean_d"], 3) if c["noise_status"] == "ok" else "---"
        g = fn(c["g"], 3) + ("" if c["g_source"].startswith("noise") else "$^\\ddagger$")
        n = len(AGENTS)
        for y, (a, name) in enumerate(AGENTS):
            head = ([f"\\multirow{{{n}}}{{*}}{{{SCENARIOS[i][1]}}}", f"\\multirow{{{n}}}{{*}}{{{noise_d}}}",
                     f"\\multirow{{{n}}}{{*}}{{{g}}}"] if y == 0 else ["", "", ""])
            row = head + [name] + [cell for m, _ in MODES for cell in block(i, a, m)]
            lines.append(" & ".join(row) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}", ""]
    return "\n".join(lines)


def table_sensitivity(sens: dict, meta: dict) -> str:
    keys = list(sens["joint"])

    def cells(a: str, m: str) -> list[str]:
        out = []
        for k in keys:
            d = sens["joint"][k]
            if m == "pooled":  # ranked among agents
                sc, rk, n = d["agent_scores"][a], d["agent_ranks"][a], len(d["agent_ranks"])
            else:  # ranked among candidate sets
                sc, rk, n = d["set_scores"][set_name(a, m)], d["set_ranks"][set_name(a, m)], len(d["set_ranks"])
            out += [fn(sc), f"{rk}/{n}"]
        return out

    lines = [
        header_comment(meta),
        "\\begin{tabular}{@{}ll " + " ".join(["rr"] * len(keys)) + "@{}}",
        "\\toprule",
        " & & " + " & ".join(f"\\multicolumn{{2}}{{c}}{{$({k[:-1]}{{\\times}})\\,g_i, f_i$}}" for k in keys)
        + " \\\\",
        "".join(f"\\cmidrule(lr){{{3 + 2 * x}-{4 + 2 * x}}}" for x in range(len(keys))),
        "Agent & Mode & " + " & ".join(["Score & Rank"] * len(keys)) + " \\\\",
        "\\midrule",
        *agent_rows(cells),
        "\\bottomrule",
        "\\end{tabular}",
        "",
    ]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--results", type=Path, required=True, help="fastkernels e2e OUT/results directory")
    ap.add_argument("--out-json", type=Path, default=None, help=f"default: {OUT_JSON.relative_to(PAPER_DIR)}")
    ap.add_argument("--tables-dir", type=Path, default=None, help=f"default: {TABLES_DIR.relative_to(PAPER_DIR)}")
    ap.add_argument("--with-codex", action="store_true",
                    help="also score the Codex sets (codex-ind, codex-seq)")
    ap.add_argument("--universe", choices=("all", "present", "complete"), default="all",
                    help="all: every scenario is an item, missing results count as invalid (paper numbers); "
                         "present: scenarios with a baseline.json; complete: scenarios whose baseline and "
                         "every set result exist (the last two are for runs in progress)")
    args = ap.parse_args(argv)
    if args.with_codex and ("codex", "Codex") not in AGENTS:
        AGENTS.append(("codex", "Codex"))

    results = args.results.resolve()
    in_fixture = FIXTURE_DIR.resolve() in results.parents
    out_json = args.out_json or (FIXTURE_DIR / "macroeval.json" if in_fixture else OUT_JSON)
    tables_dir = args.tables_dir or (FIXTURE_DIR / "tables" if in_fixture else TABLES_DIR)
    cfg = CONFIG

    warnings: list[str] = []
    raw = load_results(results, warnings)
    if args.universe == "all":
        universe = list(range(len(SCENARIOS)))
    elif args.universe == "present":
        universe = [i for i in range(len(SCENARIOS)) if raw.get(i) and raw[i]["baseline"] is not None]
    else:
        universe = [i for i in range(len(SCENARIOS)) if raw.get(i) and raw[i]["baseline"] is not None
                    and all(raw[i]["sets"][s] is not None for s in all_sets())]
    if not universe:
        warnings.append("no scenario in the universe; nothing to score")

    cal = calibrate(raw, cfg)
    for i in universe:
        if cal[i]["g_capped"]:
            warnings.append(f"{SCENARIOS[i][0]}: noise g exceeds g_cap * f; capped to {cal[i]['g']:.3f}")
        if not cal[i]["g_source"].startswith("noise") and raw.get(i) and (raw[i].get("baseline") or {}).get("status") == "ok":
            warnings.append(f"{SCENARIOS[i][0]}: {cal[i]['g_source']} -> g = {cal[i]['g']:.3f}")
    res = score_all(raw, universe, cal, cfg)
    drops = {s: drop_stats(raw, s, universe, cfg) for s in all_sets()}
    sens = sensitivity(raw, universe, cfg)
    for s in all_sets():
        for it in res["items"][s]["deployable"] + res["items"][s]["all_winners"]:
            if it.get("unknown_workloads"):
                warnings.append(f"{it['model']} / {s}: workloads without a known kind ignored: "
                                f"{it['unknown_workloads']}")
    extra = sorted({e for r in raw.values() for e in r["extra"]})
    if extra:
        warnings.append(f"result files for sets outside AGENTS x MODES ignored: {extra}")

    fams = [f for f in FAMILIES if any(SCENARIOS[i][2] == f for i in universe)]
    meta = {"universe": args.universe, "n_items": len(universe), "n_families": len(fams),
            "n_with_baseline": sum(1 for r in raw.values() if r["baseline"] is not None)}

    inventory = {}
    for i, (model, _, fam) in enumerate(SCENARIOS):
        r = raw.get(i)
        inventory[model] = {
            "family": fam, "in_universe": i in universe, "dir": r and r["dir"],
            "baseline": r and r["baseline"] and r["baseline"].get("status"),
            "noise": r and r["noise"] and r["noise"].get("status"),
            "sets": {s: (r["sets"][s] or {}).get("status") if r else None for s in all_sets()}}

    def pos_map(item_list: list[dict]) -> dict[str, dict]:
        return {it["model"]: it for it in item_list}

    out = {
        "generated_by": "figures/make_macroeval.py",
        "universe": {"mode": args.universe, "models": [SCENARIOS[i][0] for i in universe], "families": fams},
        "config": cfg,
        "inventory": inventory,
        "calibration": {SCENARIOS[i][0]: cal[i] for i in range(len(SCENARIOS))},
        "sets": {s: {"agent": s.rsplit("-", 1)[0], "mode": s.rsplit("-", 1)[1],
                     "deployable": res["sets"][s]["deployable"],
                     "all_winners": res["sets"][s]["all_winners"],
                     "drops": drops[s],
                     "items": {"deployable": pos_map(res["items"][s]["deployable"]),
                               "all_winners": pos_map(res["items"][s]["all_winners"])}}
                 for s in all_sets()},
        "agents_pooled": res["agents"],
        "sensitivity": sens,
        "warnings": warnings,
    }
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(out, indent=1) + "\n")

    tables_dir.mkdir(parents=True, exist_ok=True)
    written = {
        "e2e_macroeval.tex": table_macroeval(res, meta),
        "e2e_drops.tex": table_drops(drops, meta),
        "e2e_per_model.tex": table_per_model(res, cal, universe, cfg, meta),
        "e2e_sensitivity.tex": table_sensitivity(sens, meta),
    }
    for name, text in written.items():
        (tables_dir / name).write_text(text)

    # ---- console summary ----------------------------------------------------
    print(f"universe={args.universe}: {len(universe)} items in {len(fams)} families; "
          f"{meta['n_with_baseline']}/{len(SCENARIOS)} scenarios have a baseline")
    print(f"{'set':14} {'valid':>6} {'cov_m':>6} {'C_m':>6} {'S_m':>6} {'S_thr':>6} {'S_lat':>6} "
          f"{'score':>6} | {'AW val':>6} {'AW cov':>6}")

    def line(name: str, src: dict) -> str:
        d, w = src["deployable"], src["all_winners"]
        f2 = lambda v: "   -  " if v is None else f"{v:6.3f}"  # noqa: E731
        return (f"{name:14} {d['n_valid']:>3}/{d['n_items']:<2} {f2(d['coverage_macro'])} {f2(d['C_macro'])} "
                f"{f2(d['S_macro']['blend'])} {f2(d['S_macro']['thr'])} {f2(d['S_macro']['lat'])} "
                f"{f2(d['score']['blend'])} | {w['n_valid']:>3}/{w['n_items']:<2} {f2(w['coverage_macro'])}")

    for a, _ in AGENTS:
        for m, _ in MODES:
            print(line(set_name(a, m), res["sets"][set_name(a, m)]))
        print(line(f"{a} (pooled)", res["agents"][a]))
    print("sensitivity (g_i, f_i scaled jointly), agent ranks by pooled Score:")
    for k, d in sens["joint"].items():
        order = sorted(d["agent_ranks"], key=lambda a: (d["agent_ranks"][a], a))
        print(f"  {k:>5}: " + "  ".join(f"{d['agent_ranks'][a]}.{a}({fn(d['agent_scores'][a], 3)})" for a in order)
              + f"   discordant pairs vs 1x: {d['agent_discordant_pairs_vs_1x']}")
    for w in warnings:
        print(f"warning: {w}")
    rel = lambda p: p.relative_to(PAPER_DIR) if PAPER_DIR in p.resolve().parents else p  # noqa: E731
    print(f"wrote {rel(out_json.resolve())} and {', '.join(written)} in {rel(tables_dir.resolve())}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
