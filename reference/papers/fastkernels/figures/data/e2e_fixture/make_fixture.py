#!/usr/bin/env python3
"""Synthetic ``fastkernels e2e`` results (``results/``) for testing figures/make_macroeval.py.

    python figures/data/e2e_fixture/make_fixture.py
    python figures/make_macroeval.py --results figures/data/e2e_fixture/results
    python figures/data/e2e_fixture/check_fixture.py

Same file layout and JSON schema as fastkernels/e2e/core.py (baseline.json, noise.json,
<set>.json per scenario directory). Deterministic. Covered cases:

* 00 Llama / claude-ind, 03 GPT-OSS / claude-ind: hand-specified numbers (the "Dense & MoE
  LLMs" hand check in check_fixture.py); 03 is a crash-then-deployable chain;
* 00 / drkernel-ind: all-winners compile crash, then deployable;
* 01 / ako-ind: all-winners broken (mean d > 0.5), bisection, "incorrect" drop;
* 01 / kda-seq: timeout without a named culprit, found by bisection;
* 02 / drkernel-ind: every kernel dropped -> empty deployable subset;
* 02 / claude-seq: a kernel fails to import (``not_swapped``) on an otherwise ok run;
* 02 / kda-ind: crash without culprit, bisection inconclusive -> incomplete;
* 03 / kda-ind: runs (status ok) but mean d ~ 0.2 -> invalid under MacroEval;
* 04 / ako-seq: result file missing (run in progress);
* 05: noisy noise run (d up to 0.2) -> lenient g; claude-ind would be invalid without it;
* 06: baseline failed (every set baseline_failed, no noise.json);
* 07: noise.json missing (g falls back); ako-ind orchestrator_error;
* 08: scenario directory missing;
* 09 / drkernel-seq: no kernels requested for this model -> empty, no attempts.
"""

from __future__ import annotations

import json
import math
import random
import shutil
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT = HERE / "results"
SEED = 7

WL = {  # (workload, kind, unit, baseline value)
    "llm": [("mixed", "throughput", "tok/s", 5200.0), ("long-context", "throughput", "tok/s", 2100.0),
            ("single-request", "latency", "s", 3.1), ("fixed-batch-32", "latency", "s", 6.4)],
    "diffusion": [("1024x1024", "throughput", "img/s", 0.42), ("512x512", "throughput", "img/s", 1.6),
                  ("single-1024x1024", "latency", "s", 2.9), ("single-512x512", "latency", "s", 0.8)],
    "vlm": [("text-only", "throughput", "tok/s", 3100.0), ("image", "throughput", "tok/s", 1900.0),
            ("video", "throughput", "tok/s", 900.0), ("single-image", "latency", "s", 1.7),
            ("single-video", "latency", "s", 4.2)],
    "detection": [("coco-val", "throughput", "img/s", 850.0), ("single-image", "latency", "s", 0.0041),
                  ("batch-4", "latency", "s", 0.0093)],
    "structure": [("short", "throughput", "struct/s", 0.9), ("medium", "throughput", "struct/s", 0.31),
                  ("single-short", "latency", "s", 1.2), ("single-medium", "latency", "s", 3.4)],
    "embedding": [("bge-m3-mldr-docs", "throughput", "doc/s", 410.0), ("single-request", "latency", "s", 0.012),
                  ("fixed-batch-32", "latency", "s", 0.19)],
    "world": [("short-bs4-16f-4ddim", "throughput", "frame/s", 55.0),
              ("medium-bs8-24f-4ddim", "throughput", "frame/s", 71.0),
              ("latency-bs1-8f-4ddim", "latency", "s", 0.36)],
}
SCEN = [  # (model, tp, workload group, kernels this model uses)
    ("meta-llama/Llama-3.1-8B-Instruct", 1, "llm",
     ["L1:embedding", "L1:linear", "L1:rms_norm", "L1:rotary_emb", "L1:silu_and_mul", "L2:attention"]),
    ("Qwen/Qwen3-Next-80B-A3B-Instruct", 2, "llm",
     ["L1:gated_delta_rule", "L1:linear", "L1:rms_norm", "L1:silu_and_mul", "L2:moe_router", "L3:decoder_layer"]),
    ("moonshotai/Kimi-Linear-48B-A3B-Instruct", 2, "llm",
     ["L1:kda_chunk", "L1:rms_norm", "L1:short_conv", "L2:mla_attention", "L2:moe_router"]),
    ("openai/gpt-oss-120b", 2, "llm",
     ["L1:mxfp4_moe", "L1:rms_norm", "L1:rotary_emb", "L2:attention_sinks", "L2:moe_router"]),
    ("fla-hub/gla-2.7B-100B", 1, "llm", ["L1:gla_chunk", "L1:rms_norm", "L1:swiglu", "L2:gla_attention"]),
    ("black-forest-labs/FLUX.1-dev", 1, "diffusion",
     ["L1:gelu", "L1:layer_norm", "L1:linear", "L1:rotary_emb", "L2:adaln_modulation", "L2:flux_attention"]),
    ("Qwen/Qwen3-VL-235B-A22B-Instruct-FP8", 4, "vlm",
     ["L1:fp8_linear", "L1:rms_norm", "L2:moe_router", "L2:vision_attention"]),
    ("jameslahm/yolov10n", 1, "detection", ["L1:conv2d", "L1:max_pool2d", "L1:sigmoid", "L1:silu",
                                            "L2:yolov10_attention"]),
    ("OpenFold/OpenFold3", 1, "structure", ["L1:layer_norm", "L1:triangle_multiplication",
                                            "L2:triangle_attention"]),
    ("BAAI/bge-m3", 1, "embedding", ["L1:dense_attention", "L1:gelu", "L1:layer_norm", "L2:encoder_mlp"]),
    ("Etched/oasis-500m", 1, "world", ["L1:conv3d", "L1:group_norm", "L1:silu", "L2:spatial_attention",
                                       "L2:temporal_attention"]),
]
SETS = [f"{a}-{m}" for a in ("drkernel", "claude", "kda", "ako") for m in ("ind", "seq")]
# random-run knobs per agent: (mean log speedup, P(bad sample), kernels requested fraction)
AGENT = {"drkernel": (0.00, 0.00, 0.5), "claude": (0.05, 0.02, 1.0), "kda": (0.03, 0.04, 0.9),
         "ako": (0.07, 0.03, 1.0)}

rng = random.Random(SEED)


def slug(text: str) -> str:
    import re
    return re.sub(r"[^A-Za-z0-9._-]+", "_", text).strip("_")


def write(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=1))


def timings(group: str, speedups: dict[str, float]) -> dict:
    out = {}
    for wl, kind, unit, v in WL[group]:
        s = speedups[wl]
        out[wl] = {"kind": kind, "unit": unit, "value": v * s if kind == "throughput" else v / s}
    return out


def rand_speedups(group: str, mu: float, sd: float = 0.04) -> dict[str, float]:
    return {wl: round(math.exp(rng.gauss(mu, sd)), 4) for wl, *_ in WL[group]}


def rand_d(n: int, lo: float, hi: float, p_bad: float = 0.0) -> list[float]:
    return [round(rng.uniform(0.05, 0.3) if rng.random() < p_bad else rng.uniform(lo, hi), 5) for _ in range(n)]


def correctness(ps: list[float]) -> dict:
    return {"per_sample": ps, "summary": {"n": len(ps), "top1_agreement": 1 - sum(ps) / len(ps)}}


def attempt(n: int, purpose: str, only: list[str], exclude: list[str], status: str = "ok", *,
            speedups: dict | None = None, mean_d: float | None = None, error: str = "",
            category: str | None = None, culprits: list[str] | None = None,
            not_swapped: list[str] | None = None) -> dict:
    a = {"n": n, "purpose": purpose, "only": only, "exclude": exclude, "status": status,
         "error": error, "swapped": [k for k in only if k not in (not_swapped or [])],
         "not_swapped": not_swapped or [], "t_import_s": 41.3, "t_total_s": 812.0}
    if status == "ok":
        a.update(speedups=speedups, mean_d=mean_d)
    else:
        a.update(category=category, culprits=culprits or [])
    return a


def chain(model: str, sname: str, group: str, requested: list[str], attempts: list[dict],
          dropped: list[dict], final: tuple[dict, list[float]] | None = None, status: str | None = None) -> dict:
    """Assemble a <set>.json the way E2E._drop_and_retry does."""
    s = {"kernels_requested": requested, "dropped": dropped, "attempts": attempts,
         "all_winners": attempts[0] if attempts else None}
    if final is not None:
        info, ps = final
        s.update(status="ok", kernels_final=info["only"], timings=timings(group, info["speedups"]),
                 speedups=info["speedups"], correctness=correctness(ps), mean_d=sum(ps) / len(ps))
    elif status == "empty":
        s.update(status="empty", kernels_final=[], note="no deployable kernels: result equals the baseline")
    else:
        s.update(status="incomplete", kernels_final=None,
                 note=f"stopped after {len(attempts)} attempts without a deployable subset")
    s.update(model=model, set=sname)
    return s


def clean_ok(model, sname, group, requested, speedups, ps):
    a1 = attempt(1, "full", requested, [], speedups=speedups, mean_d=sum(ps) / len(ps))
    return chain(model, sname, group, requested, [a1], [], (a1, ps))


def random_ok(idx: int, sname: str) -> dict:
    model, _, group, kernels = SCEN[idx]
    agent, mode = sname.split("-")
    mu, p_bad, frac = AGENT[agent]
    mu += 0.01 if mode == "seq" else 0.0
    mu -= 0.06 if idx == 10 else 0.0  # oasis: agents mostly slow it down
    k = max(1, round(frac * len(kernels)))
    requested = sorted(rng.sample(kernels, k))
    lo, hi = (0.0, 0.006) if group in ("llm", "vlm") else (0.0, 0.004)
    return clean_ok(model, sname, group, requested, rand_speedups(group, mu), rand_d(16, lo, hi, p_bad))


def main() -> None:
    if OUT.exists():
        shutil.rmtree(OUT)
    for idx, (model, tp, group, kernels) in enumerate(SCEN):
        if idx == 8:  # OpenFold3: never started
            continue
        d = OUT / f"{idx:02d}_{slug(model)}"
        base_t = timings(group, {wl: 1.0 for wl, *_ in WL[group]})
        if idx == 6:  # baseline crashed: sets are baseline_failed, no noise run
            err = "OutOfMemoryError: CUDA out of memory. Tried to allocate 2.00 GiB"
            write(d / "baseline.json", {"model": model, "tp": tp, "status": "crash", "candidates": False,
                                        "candidate_dir": None, "adapter": "vlm", "error": err,
                                        "t_total_s": 311.0, "_dir": f"/out/runs/{d.name}/baseline"})
            for s in SETS:
                write(d / f"{s}.json", {"model": model, "set": s, "status": "baseline_failed", "error": err})
            continue
        write(d / "baseline.json", {"model": model, "tp": tp, "status": "ok", "candidates": False,
                                    "candidate_dir": None, "adapter": group, "timings": base_t,
                                    "t_run_s": 900.0, "t_total_s": 1010.0, "_dir": f"/out/runs/{d.name}/baseline"})

        # ---- noise run -------------------------------------------------------
        if idx == 0:
            noise_ps = [0.0, 0.0, 0.002, 0.004]
        elif idx == 3:
            noise_ps = [0.001, 0.001, 0.001, 0.001]
        elif idx == 5:
            noise_ps = rand_d(16, 0.02, 0.2)  # noisy: fresh-compile nondeterminism is large here
        else:
            noise_ps = rand_d(16, 0.0, 0.003)
        if idx != 7:  # 07: noise run never finished
            nsp = rand_speedups(group, 0.0, 0.01)
            write(d / "noise.json", {"model": model, "status": "ok", "error": None,
                                     "timings": timings(group, nsp), "speedups": nsp,
                                     "correctness": correctness(noise_ps),
                                     "mean_d": sum(noise_ps) / len(noise_ps)})

        # ---- candidate sets --------------------------------------------------
        for s in SETS:
            key = (idx, s)
            if key == (4, "ako-seq"):
                continue  # still running
            if key == (0, "claude-ind"):  # hand check
                res = clean_ok(model, s, group, kernels,
                               {"mixed": 1.2, "long-context": 1.1, "single-request": 1.05, "fixed-batch-32": 0.95},
                               [0.002, 0.003, 0.1, 0.0])
            elif key == (3, "claude-ind"):  # hand check: crash -> drop culprit -> deployable
                ps = [0.0, 0.001, 0.0005, 0.05]
                sp = {"mixed": 1.5, "long-context": 1.3, "single-request": 1.1, "fixed-batch-32": 1.3}
                a1 = attempt(1, "full", kernels, [], "crash", category="runtime", culprits=["L1:mxfp4_moe"],
                             error="RuntimeError: CUDA error: an illegal memory access was encountered")
                rest = [k for k in kernels if k != "L1:mxfp4_moe"]
                a2 = attempt(2, "full", rest, ["L1:mxfp4_moe"], speedups=sp, mean_d=sum(ps) / len(ps))
                dropped = [{"kernel": "L1:mxfp4_moe", "category": "runtime", "attempt": 1,
                            "reason": a1["error"]}]
                res = chain(model, s, group, kernels, [a1, a2], dropped, (a2, ps))
            elif key == (0, "drkernel-ind"):  # all-winners compile crash
                req = ["L1:rms_norm", "L1:rotary_emb", "L1:silu_and_mul"]
                ps = rand_d(16, 0.0, 0.004)
                a1 = attempt(1, "full", req, [], "crash", category="compile", culprits=["L1:rotary_emb"],
                             error="InternalTorchDynamoError: AttributeError: 'NoneType' has no attribute 'shape'")
                a2 = attempt(2, "full", ["L1:rms_norm", "L1:silu_and_mul"], ["L1:rotary_emb"],
                             speedups=rand_speedups(group, 0.01), mean_d=sum(ps) / len(ps))
                res = chain(model, s, group, req, [a1, a2],
                            [{"kernel": "L1:rotary_emb", "category": "compile", "attempt": 1,
                              "reason": a1["error"]}], (a2, ps))
            elif key == (1, "ako-ind"):  # broken all-winners -> bisection -> incorrect drop
                req = ["L1:gated_delta_rule", "L1:linear", "L1:rms_norm", "L2:moe_router"]
                ps = rand_d(16, 0.0, 0.005)
                bad = lambda n, only: attempt(n, "bisect", only, [], speedups=rand_speedups(group, 0.05),  # noqa: E731
                                              mean_d=0.81)
                good = lambda n, only: attempt(n, "bisect", only, [], speedups=rand_speedups(group, 0.03),  # noqa: E731
                                               mean_d=0.002)
                atts = [attempt(1, "full", req, [], speedups=rand_speedups(group, 0.08), mean_d=0.83),
                        bad(2, req[:2]), good(3, req[:1]), bad(4, req[1:2])]
                dropped = [{"kernel": "L1:linear", "category": "incorrect", "attempt": 4,
                            "reason": "found by bisection (incorrect)"}]
                rest = [k for k in req if k != "L1:linear"]
                final = attempt(5, "full", rest, ["L1:linear"], speedups=rand_speedups(group, 0.06),
                                mean_d=sum(ps) / len(ps))
                res = chain(model, s, group, req, atts + [final], dropped, (final, ps))
            elif key == (1, "kda-seq"):  # timeout, no culprit named -> bisection
                req = ["L1:gated_delta_rule", "L1:rms_norm", "L2:moe_router", "L3:decoder_layer"]
                ps = rand_d(16, 0.0, 0.005)
                to = dict(status="crash", category="timeout", error="TIMEOUT")
                atts = [attempt(1, "full", req, [], **to), attempt(2, "bisect", req[:2], [], **to),
                        attempt(3, "bisect", req[:1], [], speedups=rand_speedups(group, 0.02), mean_d=0.001),
                        attempt(4, "bisect", req[1:2], [], **to)]
                rest = [k for k in req if k != "L1:rms_norm"]
                final = attempt(5, "full", rest, ["L1:rms_norm"], speedups=rand_speedups(group, 0.04),
                                mean_d=sum(ps) / len(ps))
                res = chain(model, s, group, req, atts + [final],
                            [{"kernel": "L1:rms_norm", "category": "timeout", "attempt": 4,
                              "reason": "found by bisection (timeout)"}], (final, ps))
            elif key == (2, "drkernel-ind"):  # every kernel dropped -> empty
                req = ["L1:rms_norm", "L1:short_conv"]
                a1 = attempt(1, "full", req, [], "crash", category="compile", culprits=["L1:short_conv"],
                             error="torch._dynamo.exc.Unsupported: call_function in skip_files")
                a2 = attempt(2, "full", ["L1:rms_norm"], ["L1:short_conv"], "crash", category="runtime",
                             culprits=["L1:rms_norm"], error="RuntimeError: shape '[2, -1, 2048]' is invalid")
                res = chain(model, s, group, req, [a1, a2],
                            [{"kernel": "L1:short_conv", "category": "compile", "attempt": 1, "reason": a1["error"]},
                             {"kernel": "L1:rms_norm", "category": "runtime", "attempt": 2, "reason": a2["error"]}],
                            status="empty")
            elif key == (2, "claude-seq"):  # import failure on an ok run
                req = kernels
                ps = rand_d(16, 0.0, 0.005)
                a1 = attempt(1, "full", req, [], speedups=rand_speedups(group, 0.05), mean_d=sum(ps) / len(ps),
                             not_swapped=["L2:mla_attention"])
                res = chain(model, s, group, req, [a1],
                            [{"kernel": "L2:mla_attention", "category": "import", "attempt": 1,
                              "reason": "candidate failed to import / define its class"}], (a1, ps))
            elif key == (2, "kda-ind"):  # crash, no culprit, halves both run -> incomplete
                req = ["L1:kda_chunk", "L1:rms_norm", "L2:mla_attention", "L2:moe_router"]
                atts = [attempt(1, "full", req, [], "crash", category="runtime",
                                error="RuntimeError: NCCL error: unhandled system error"),
                        attempt(2, "bisect", req[:2], [], speedups=rand_speedups(group, 0.03), mean_d=0.002),
                        attempt(3, "bisect", req[2:], [], speedups=rand_speedups(group, 0.03), mean_d=0.003)]
                res = chain(model, s, group, req, atts, [], status="incomplete")
            elif key == (3, "kda-ind"):  # runs, but wrong enough to fail tau
                req = kernels[:4]
                ps = [0.2, 0.19, 0.22, 0.21, 0.0, 0.2, 0.18, 0.2]
                res = clean_ok(model, s, group, req, rand_speedups(group, 0.04), ps)
            elif key == (5, "claude-ind"):  # d within the (noisy) noise band
                res = clean_ok(model, s, group, kernels, rand_speedups(group, 0.05), rand_d(16, 0.05, 0.15))
            elif key == (5, "drkernel-seq"):  # compile crash on a named culprit, then deployable
                req = ["L1:gelu", "L1:layer_norm", "L2:flux_attention"]
                ps = rand_d(16, 0.02, 0.15)
                a1 = attempt(1, "full", req, [], "crash", category="compile", culprits=["L2:flux_attention"],
                             error="torch._inductor.exc.LoweringException: NotImplementedError")
                a2 = attempt(2, "full", req[:2], ["L2:flux_attention"], speedups=rand_speedups(group, 0.0),
                             mean_d=sum(ps) / len(ps))
                res = chain(model, s, group, req, [a1, a2],
                            [{"kernel": "L2:flux_attention", "category": "compile", "attempt": 1,
                              "reason": a1["error"]}], (a2, ps))
            elif key == (7, "ako-ind"):
                res = {"status": "orchestrator_error",
                       "traceback": "Traceback (most recent call last):\n  ...\nKeyError: '_dir'\n",
                       "model": model, "set": s}
            elif key == (9, "drkernel-seq"):  # the set has no kernel this model uses
                res = chain(model, s, group, [], [], [], status="empty")
            else:
                res = random_ok(idx, s)
            write(d / f"{s}.json", res)
    print(f"wrote {sum(1 for _ in OUT.rglob('*.json'))} files under {OUT.relative_to(HERE.parent.parent.parent)}")


if __name__ == "__main__":
    main()
