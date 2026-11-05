#!/usr/bin/env python3
"""Render L4 throughput and latency geomean-speedup breakdowns.

Usage: make_eval_speedup_breakdown.py [b200|h200]   (default: b200)
"""

from __future__ import annotations

import math
import re
import statistics
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

PAPER_DIR = Path(__file__).resolve().parent.parent
RESULTS_DIR = Path("/Users/goliaro/Desktop/fk/fastkernels-results")
LAYOUTS = {
    # 2x4 panels for a full-width figure* float.
    "wide": {
        "nrows": 2,
        "ncols": 4,
        "figsize": (12.2, 4.35),
        "font": 1.0,
        "adjust": dict(left=0.045, right=0.995, top=0.90, bottom=0.28, wspace=0.28, hspace=0.85),
        "ylabel_xy": (0.012, 0.58),
    },
    # 4x2 panels that stay legible at \linewidth in the single-column appendix.
    # Mild rotation keeps x-tick labels short enough that a modest hspace
    # clears the next title without large empty bands.
    "tall": {
        "nrows": 4,
        "ncols": 2,
        "figsize": (7.0, 8.8),
        "font": 1.3,
        "adjust": dict(
            left=0.09, right=0.995, top=0.97, bottom=0.07, wspace=0.16, hspace=0.90
        ),
        "ylabel_xy": (0.015, 0.53),
    },
}
# gpu -> (throughput pdf, latency pdf, layout)
# Both GPUs use the tall (4x2) layout so the panels stay legible at
# \linewidth in the single-column appendix.
OUTPUTS = {
    "b200": ("eval_speedup_breakdown_b200.pdf", "eval_latency_breakdown_b200.pdf", "tall"),
    "h200": ("eval_speedup_breakdown_h200.pdf", "eval_latency_breakdown_h200.pdf", "tall"),
}
GPU = sys.argv[1] if len(sys.argv) > 1 else "b200"
RESULTS_MD = RESULTS_DIR / f"validate-results-{GPU}.md"
THR_PDF = PAPER_DIR / "figures" / OUTPUTS[GPU][0]
LAT_PDF = PAPER_DIR / "figures" / OUTPUTS[GPU][1]
LAYOUT = OUTPUTS[GPU][2]

YMAX = 2.20
CLIP = 2.12

# (display name, family title)
META: dict[tuple[str, str], tuple[str, str]] = {
    ("meta-llama/Llama-3.1-8B-Instruct", "vLLM"): ("Llama-3.1", "Dense & MoE LLMs"),
    ("mistralai/Mixtral-8x7B-Instruct-v0.1", "vLLM"): ("Mixtral", "Dense & MoE LLMs"),
    ("nvidia/GLM-5.2-NVFP4", "vLLM"): ("GLM-5.2", "Dense & MoE LLMs"),
    ("microsoft/bitnet-b1.58-2B-4T", "microsoft-bitnet-gpu"): ("BitNet 1.58b", "Dense & MoE LLMs"),
    ("openai/gpt-oss-120b", "vLLM"): ("GPT-OSS", "Dense & MoE LLMs"),
    ("meta-llama/Llama-3.1-8B-Instruct", "SGLang"): ("EAGLE-3", "Dense & MoE LLMs"),
    ("google/gemma-4-26B-A4B-it", "vLLM"): ("Gemma-4", "Dense & MoE LLMs"),
    ("state-spaces/mamba-2.8b-hf", "vLLM"): ("Mamba", "Linear Attention\n& New Archs"),
    ("mistralai/Mamba-Codestral-7B-v0.1", "vLLM"): ("Mamba2", "Linear Attention\n& New Archs"),
    ("fla-hub/rwkv7-2.9B-g1", "FLA"): ("RWKV-7", "Linear Attention\n& New Archs"),
    ("fla-hub/gla-2.7B-100B", "FLA"): ("GLA", "Linear Attention\n& New Archs"),
    ("fla-hub/retnet-2.7B-100B", "FLA"): ("RetNet", "Linear Attention\n& New Archs"),
    ("Qwen/Qwen3-Next-80B-A3B-Instruct", "vLLM"): ("Qwen-3-Next", "Linear Attention\n& New Archs"),
    ("moonshotai/Kimi-Linear-48B-A3B-Instruct", "vLLM"): ("Kimi-Linear", "Linear Attention\n& New Archs"),
    ("ttt_e2e", "jax"): ("TTT-E2E", "Linear Attention\n& New Archs"),
    ("ai21labs/AI21-Jamba-Mini-1.7", "vLLM"): ("Jamba", "Linear Attention\n& New Archs"),
    ("black-forest-labs/FLUX.1-dev", "vllm-omni"): ("FLUX.1-Dev", "Vision / Video /\nAudio"),
    ("hunyuanvideo-community/HunyuanVideo-1.5-Diffusers-480p_t2v", "vllm-omni"): (
        "HunyuanVideo",
        "Vision / Video /\nAudio",
    ),
    ("stabilityai/stable-diffusion-xl-base-1.0", "diffusers"): ("SDXL", "Vision / Video /\nAudio"),
    ("facebook/sam3.1", "sam3"): ("SAM3.1", "Vision / Video /\nAudio"),
    ("openai/whisper-large-v3", "vLLM"): ("Whisper", "Vision / Video /\nAudio"),
    ("FunAudioLLM/Fun-CosyVoice3-0.5B-2512", "vllm-omni"): ("CosyVoice3", "Vision / Video /\nAudio"),
    ("Qwen/Qwen2-VL-7B-Instruct", "vLLM"): ("Qwen2-VL", "Multimodal\n& Encoders"),
    ("Qwen/Qwen3-VL-8B-Instruct", "vLLM"): ("Qwen3-VL-8B", "Multimodal\n& Encoders"),
    ("Qwen/Qwen3-VL-235B-A22B-Instruct-FP8", "vLLM"): ("Qwen3-VL-235B", "Multimodal\n& Encoders"),
    ("Qwen/Qwen2.5-Omni-7B", "vLLM"): ("Qwen-2.5-Omni", "Multimodal\n& Encoders"),
    ("google/siglip2-so400m-patch16-naflex", "timm"): ("SigLIP-2", "Multimodal\n& Encoders"),
    ("facebook/dinov3-vit7b16-pretrain-lvd1689m", "timm"): ("DINOv3", "Multimodal\n& Encoders"),
    ("timm/swinv2_large_window12_192.ms_in22k", "timm"): ("SwinV2", "Multimodal\n& Encoders"),
    ("timm/mobilenetv4_conv_medium.e500_r256_in1k", "timm"): ("MobileNetV4", "Edge &\nDetection"),
    ("facebook/convnextv2-base-22k-384", "transformers"): ("ConvNeXtV2", "Edge &\nDetection"),
    ("timm/efficientnetv2_rw_m.agc_in1k", "timm"): ("EfficientNetV2", "Edge &\nDetection"),
    ("jameslahm/yolov10n", "reference"): ("YOLOv10", "Edge &\nDetection"),
    ("PekingU/rtdetr_v2_r101vd", "reference"): ("RTDetrV2", "Edge &\nDetection"),
    ("Voxel51/gaussian_splatting", "gsplat"): ("3DGS", "3D / Robotics /\nScience"),
    ("NVlabs/instant-ngp", "pyngp"): ("InstantNGP", "3D / Robotics /\nScience"),
    ("Pointcept/PointTransformerV3", "official-detached"): ("PTv3", "3D / Robotics /\nScience"),
    ("OpenFold/OpenFold3", "reference"): ("OpenFold3", "3D / Robotics /\nScience"),
    ("/home/yak/.fastkernels/third_party/openpi-assets/pi0_aloha_pen_uncap_pytorch", "openpi"): (
        "Pi0",
        "3D / Robotics /\nScience",
    ),
    ("dp3", "3D-Diffusion-Policy"): ("DP3", "3D / Robotics /\nScience"),
    ("dlrmv2", "torchrec.models.dlrm.DLRM"): ("DLRMv2", "Recommendation\n& Specialized"),
    ("lightgcn", "torch_geometric.nn.models.LightGCN"): ("LightGCN", "Recommendation\n& Specialized"),
    ("BAAI/bge-m3", "vLLM"): ("BGE-M3", "Recommendation\n& Specialized"),
    ("colbert-ir/colbertv2.0", "vLLM"): ("ColBERTv2", "Recommendation\n& Specialized"),
    ("GSAI-ML/LLaDA-8B-Instruct", "fastdllm-dual"): ("LLaDA", "Recommendation\n& Specialized"),
    ("Etched/oasis-500m", "open-oasis"): ("Oasis", "World Models"),
    ("facebook/vjepa2-vitl-fpc64-256", "transformers"): ("V-JEPA 2", "World Models"),
}

FAMILY_ORDER = [
    "Dense & MoE LLMs",
    "Linear Attention\n& New Archs",
    "Vision / Video /\nAudio",
    "Multimodal\n& Encoders",
    "Edge &\nDetection",
    "3D / Robotics /\nScience",
    "Recommendation\n& Specialized",
    "World Models",
]


def geomean(xs: list[float]) -> float:
    return math.exp(sum(math.log(x) for x in xs) / len(xs))


def parse_speedup(s: str) -> float:
    return float(s.strip().rstrip("x"))


def parse_table(lines: list[str], section_title: str) -> list[dict]:
    rows: list[dict] = []
    in_sec = False
    header_done = False
    for line in lines:
        if line.strip() == section_title:
            in_sec = True
            header_done = False
            continue
        if in_sec and line.startswith("MODEL") and "WORKLOAD" in line:
            continue
        if in_sec and line.startswith("---"):
            header_done = True
            continue
        if in_sec and header_done:
            if not line.strip():
                if rows:
                    break
                continue
            if line.startswith("LATENCY") or line.startswith("THROUGHPUT"):
                break
            parts = re.split(r"\s{2,}", line.strip())
            if len(parts) < 4:
                continue
            model, workload, ref, speedup = parts[:4]
            rows.append(
                {
                    "model": model,
                    "workload": workload,
                    "ref": ref,
                    "speedup": parse_speedup(speedup),
                }
            )
    return rows


def load_models() -> list[dict]:
    lines = RESULTS_MD.read_text().splitlines()
    thr = parse_table(lines, "THROUGHPUT / E2E WORKLOADS")
    lat = parse_table(lines, "LATENCY WORKLOADS")
    thr_by: dict[tuple[str, str], list[float]] = defaultdict(list)
    lat_by: dict[tuple[str, str], list[float]] = defaultdict(list)
    for r in thr:
        thr_by[(r["model"], r["ref"])].append(r["speedup"])
    for r in lat:
        lat_by[(r["model"], r["ref"])].append(r["speedup"])
    out = []
    for key, tvals in thr_by.items():
        name, family = META[key]
        lvals = lat_by.get(key, [])
        out.append(
            {
                "key": key,
                "name": name,
                "family": family,
                "thr": geomean(tvals),
                "lat": geomean(lvals) if lvals else None,
                "nT": len(tvals),
                "nL": len(lvals),
            }
        )
    return out


def family_rows(models: list[dict], kind: str) -> list[tuple[str, list[tuple[str, float]]]]:
    families = []
    for title in FAMILY_ORDER:
        rows = []
        for m in models:
            if m["family"] != title:
                continue
            val = m[kind]
            if val is None:
                continue
            rows.append((m["name"], val))
        rows.sort(key=lambda r: r[1], reverse=True)
        families.append((title, rows))
    return families


def render(
    families: list[tuple[str, list[tuple[str, float]]]], ylabel: str, out: Path, layout: str
) -> None:
    cfg = LAYOUTS[layout]
    fs = cfg["font"]
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 7.2 * fs,
            "axes.linewidth": 0.55,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    fig, axes = plt.subplots(cfg["nrows"], cfg["ncols"], figsize=cfg["figsize"])
    cmap = plt.get_cmap("tab10")
    fig.subplots_adjust(**cfg["adjust"])
    nmax = max(len(rows) for _, rows in families)
    bar_width = 0.78

    for ax, (title, rows), color_idx in zip(axes.flat, families, range(8)):
        labels = [r[0] for r in rows]
        vals = np.array([r[1] for r in rows], dtype=float)
        x = np.arange(len(vals))
        color = cmap(color_idx)
        ax.bar(x, np.minimum(vals, CLIP), color=color, width=bar_width, zorder=3, edgecolor="none")
        ax.axhline(1.0, color="0.2", ls="--", lw=0.8, zorder=4)
        ax.set_ylim(0.0, YMAX)
        ax.set_yticks([0.0, 0.5, 1.0, 1.5, 2.0])
        ax.set_xlim(-0.65, nmax - 0.35)
        ax.set_xticks(x)
        ax.tick_params(axis="x", length=2.2, pad=1.5, labelsize=5.6 * fs)
        ax.tick_params(axis="y", length=2.2, pad=1.2, labelsize=6.2 * fs)
        ax.set_xticklabels(
            labels, rotation=22, ha="right", rotation_mode="anchor", fontsize=5.6 * fs
        )
        ax.set_title(title, fontsize=8.0 * fs, pad=4)
        ax.yaxis.grid(True, ls=":", lw=0.35, color="0.85", zorder=0)
        ax.set_axisbelow(True)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
        for xi, v in zip(x, vals):
            ax.text(
                xi,
                min(v, CLIP) + 0.06,
                f"{v:.2f}x",
                ha="center",
                va="bottom",
                fontsize=5.4 * fs,
            )

    fig.text(*cfg["ylabel_xy"], ylabel, rotation=90, va="center", ha="center", fontsize=7.6 * fs)
    fig.savefig(out, bbox_inches="tight", pad_inches=0.03)
    print(f"wrote {out}")


def r2(x: float) -> str:
    return f"{x:.2f}"


def main() -> None:
    models = load_models()
    render(
        family_rows(models, "thr"),
        "Throughput speedup vs. production reference",
        THR_PDF,
        LAYOUT,
    )
    render(
        family_rows(models, "lat"),
        "Latency speedup vs. production reference",
        LAT_PDF,
        LAYOUT,
    )

    thrs = [m["thr"] for m in models]
    lats = [m["lat"] for m in models if m["lat"] is not None]
    print("N", len(models), "lat N", len(lats))
    print(
        "thr geomean",
        r2(geomean(thrs)),
        "median",
        r2(statistics.median(thrs)),
        "arith",
        r2(statistics.mean(thrs)),
        ">1",
        sum(1 for x in thrs if x > 1),
    )
    print(
        "lat geomean",
        r2(geomean(lats)),
        "median",
        r2(statistics.median(lats)),
        "arith",
        r2(statistics.mean(lats)),
        ">1",
        sum(1 for x in lats if x > 1),
    )
    print("\n=== per model ===")
    for m in sorted(models, key=lambda z: z["family"] + z["name"]):
        lat = r2(m["lat"]) if m["lat"] is not None else "---"
        print(f"{m['name']:20s}  thr {r2(m['thr'])}  lat {lat}")


if __name__ == "__main__":
    main()
