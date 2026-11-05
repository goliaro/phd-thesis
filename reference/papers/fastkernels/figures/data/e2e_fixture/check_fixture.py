#!/usr/bin/env python3
"""Hand check of figures/make_macroeval.py on the synthetic fixture.

    python figures/data/e2e_fixture/make_fixture.py
    python figures/make_macroeval.py --results figures/data/e2e_fixture/results
    python figures/data/e2e_fixture/check_fixture.py

Family "Dense & MoE LLMs" = {Llama-3.1-8B, GPT-OSS-120B}, set claude-ind, worked by hand
(defaults: g = max(1e-3, p95 of noise d), f = 0.5, tau = 0.9, lambda = 0.5):

Llama   noise d = [0, 0, .002, .004]  -> p95 at rank 0.95*3 = 2.85: .002 + .85*.002 = .0037 = g
        cand  d = [.002, .003, .1, 0] -> C = [1, 1, (.5-.1)/(.5-.0037) = .805964, 1]
        C_i = 3.805964/4 = .951491 >= .9 -> valid
        speedups thr {1.2, 1.1}, lat {1.05, .95}: s_thr = 1.32^.5 = 1.148913,
        s_lat = .9975^.5 = .998749, s = (1.32*.9975)^.25 = 1.3167^.25 = 1.071203
GPT-OSS noise d = [.001]*4 -> g = .001; crash (L1:mxfp4_moe dropped), then deployable:
        cand  d = [0, .001, .0005, .05] -> C = [1, 1, 1, (.5-.05)/(.5-.001) = .901804]
        C_i = 3.901804/4 = .975451 -> valid
        thr {1.5, 1.3}, lat {1.1, 1.3}: s_thr = 1.95^.5 = 1.396424, s_lat = 1.43^.5 = 1.195826,
        s = (1.95*1.43)^.25 = 2.7885^.25 = 1.292238
Family  C_f = (.951491 + .975451)/2 = .963471, Coverage_f = 1,
        S_f = (1.071203 * 1.292238)^.5 = 1.176541 (lambda=.5),
        S_f^thr = (1.32*1.95)^.25 = 1.266637, S_f^lat = (.9975*1.43)^.25 = 1.092854
All winners: GPT-OSS crashed -> invalid, C = 0; Llama's first attempt is its final run, so it
        keeps its per-sample d: C_f = .951491/2 = .475746, Coverage_f = .5, S_f = 1.071203.

Also checked: every set's Score = S_macro * C_macro * Coverage_macro recomputed from its
per-family numbers, FLUX's noise-calibrated g, and the expected outcome of every planted case.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

HERE = Path(__file__).resolve().parent
FAM = "Dense & MoE LLMs"
TOL = 2e-6


def close(a, b, what):
    assert a is not None and abs(a - b) <= TOL * max(1.0, abs(b)), f"{what}: got {a}, expected {b}"
    print(f"  ok  {what:44} {a:.6f}")


def main() -> None:
    m = json.loads((HERE / "macroeval.json").read_text())
    cal = m["calibration"]
    ci = m["sets"]["claude-ind"]
    llama, gpt = "meta-llama/Llama-3.1-8B-Instruct", "openai/gpt-oss-120b"
    print("hand check: Dense & MoE LLMs / claude-ind")
    close(cal[llama]["g"], 0.0037, "g Llama")
    close(cal[gpt]["g"], 0.001, "g GPT-OSS")
    dep = ci["items"]["deployable"]
    close(dep[llama]["C"], 0.951491, "C_i Llama")
    close(dep[gpt]["C"], 0.975451, "C_i GPT-OSS")
    assert dep[llama]["valid"] and dep[gpt]["valid"]
    close(dep[llama]["s_thr"], 1.148913, "s_thr Llama")
    close(dep[llama]["s_lat"], 0.998749, "s_lat Llama")
    close(dep[llama]["s"]["blend"], 1.071203, "s Llama")
    close(dep[gpt]["s_thr"], 1.396424, "s_thr GPT-OSS")
    close(dep[gpt]["s_lat"], 1.195826, "s_lat GPT-OSS")
    close(dep[gpt]["s"]["blend"], 1.292238, "s GPT-OSS")
    fam = ci["deployable"]["families"][FAM]
    close(fam["C_f"], 0.963471, "C_f")
    close(fam["coverage_f"], 1.0, "Coverage_f")
    close(fam["S_f"]["blend"], 1.176541, "S_f (lambda=0.5)")
    close(fam["S_f"]["thr"], 1.266637, "S_f (lambda=1)")
    close(fam["S_f"]["lat"], 1.092854, "S_f (lambda=0)")
    aw = ci["all_winners"]["families"][FAM]
    assert ci["items"]["all_winners"][gpt]["outcome"] == "crash"
    assert ci["items"]["all_winners"][llama]["d_source"] == "per_sample"
    close(aw["C_f"], 0.475746, "all winners C_f")
    close(aw["coverage_f"], 0.5, "all winners Coverage_f")
    close(aw["S_f"]["blend"], 1.071203, "all winners S_f")

    print("macro consistency: Score = S_macro * C_macro * Coverage_macro, per set/view/lambda")
    for s, rec in list(m["sets"].items()) + [(f"{a} pooled", r) for a, r in m["agents_pooled"].items()]:
        for view in ("deployable", "all_winners"):
            agg = rec[view]
            fams = agg["families"].values()
            c = sum(f["C_f"] for f in fams) / len(fams)
            cov = sum(f["coverage_f"] for f in fams) / len(fams)
            assert abs(c - agg["C_macro"]) < 1e-12 and abs(cov - agg["coverage_macro"]) < 1e-12
            for lam in ("blend", "thr", "lat"):
                sf = [f["S_f"][lam] for f in fams if f["n_valid"]]
                smac = math.exp(sum(map(math.log, sf)) / len(sf)) if sf else None
                assert (smac is None) == (agg["S_macro"][lam] is None), (s, view, lam)
                if smac is not None:
                    assert abs(smac - agg["S_macro"][lam]) < 1e-12
                    assert abs(smac * c * cov - agg["score"][lam]) < 1e-12
    print("  ok  all sets and pooled agents")

    print("noisy noise run (FLUX): g = p95 of its noise d")
    flux = "black-forest-labs/FLUX.1-dev"
    nd = sorted(json.loads(next((HERE / "results").glob("05_*/noise.json")).read_text())
                ["correctness"]["per_sample"])
    pos = 0.95 * (len(nd) - 1)
    lo = int(pos)
    close(cal[flux]["g"], nd[lo] + (nd[lo + 1] - nd[lo]) * (pos - lo), "g FLUX")
    assert m["sets"]["claude-ind"]["items"]["deployable"][flux]["valid"]

    print("planted outcomes")
    expect = {  # (set, model, view): (outcome, valid)
        ("drkernel-ind", llama, "all_winners"): ("crash", False),
        ("drkernel-ind", llama, "deployable"): ("ok", True),
        ("ako-ind", "Qwen/Qwen3-Next-80B-A3B-Instruct", "all_winners"): ("broken", False),
        ("ako-ind", "Qwen/Qwen3-Next-80B-A3B-Instruct", "deployable"): ("ok", True),
        ("kda-seq", "Qwen/Qwen3-Next-80B-A3B-Instruct", "all_winners"): ("crash", False),
        ("drkernel-ind", "moonshotai/Kimi-Linear-48B-A3B-Instruct", "deployable"): ("empty", False),
        ("claude-seq", "moonshotai/Kimi-Linear-48B-A3B-Instruct", "deployable"): ("ok", True),
        ("kda-ind", "moonshotai/Kimi-Linear-48B-A3B-Instruct", "deployable"): ("incomplete", False),
        ("kda-ind", gpt, "deployable"): ("ok", False),
        ("ako-seq", "fla-hub/gla-2.7B-100B", "deployable"): ("missing", False),
        ("claude-ind", "Qwen/Qwen3-VL-235B-A22B-Instruct-FP8", "deployable"): ("baseline_failed", False),
        ("ako-ind", "jameslahm/yolov10n", "deployable"): ("orchestrator_error", False),
        ("claude-ind", "OpenFold/OpenFold3", "deployable"): ("missing", False),
        ("drkernel-seq", "BAAI/bge-m3", "all_winners"): ("empty", False),
    }
    for (s, model, view), (outcome, valid) in expect.items():
        it = m["sets"][s]["items"][view][model]
        assert (it["outcome"], it["valid"]) == (outcome, valid), (s, model, view, it["outcome"], it["valid"])
        if not it["valid"]:
            assert it["C"] < 0.9
    assert m["sets"]["ako-ind"]["items"]["all_winners"]["Qwen/Qwen3-Next-80B-A3B-Instruct"]["d_source"] \
        == "mean_d_proxy"
    assert cal["jameslahm/yolov10n"]["g_source"].startswith("fallback")
    d = m["sets"]["claude-seq"]["drops"]
    assert d["dropped_by_category"] == {"import": 1} and d["requested"] - d["kept"] == 1
    print(f"  ok  {len(expect)} planted cases")
    print("ALL CHECKS PASSED")


if __name__ == "__main__":
    main()
