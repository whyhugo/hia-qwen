#!/usr/bin/env python3
"""Generate poster figures from evaluation metrics.

Reads PCC values straight from each run's metrics.json (with fallback to the
3-epoch preliminary numbers if a file is missing), so re-running after the
6-epoch training finishes auto-updates the figures. Outputs PNG (300 dpi) +
PDF (vector) + SVG into figures/.

Usage:
    python scripts/make_poster_figures.py
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

ROOT = Path(__file__).resolve().parents[1]
FIG_DIR = ROOT / "figures"

HIA_METRICS = ROOT / "outputs/joint_lora_stage2/eval_test/metrics.json"
ABL_METRICS = ROOT / "outputs/joint_lora_stage2_ablation/eval_test/metrics.json"
BASE_METRICS = Path("/datas/store163/whyhugo/ms-qwen2-reproduce/outputs/multi_all/ft/metrics.json")

# Metrics shown in the comparison chart (all three conditions have a value).
CHART_KEYS = [
    ("phone.accuracy", "phone\naccuracy"),
    ("sentence.accuracy", "sentence\naccuracy"),
    ("sentence.fluency", "sentence\nfluency"),
    ("sentence.prosody", "sentence\nprosody"),
    ("sentence.total", "sentence\ntotal"),
]

# 3-epoch preliminary fallbacks (used only if a metrics.json is unavailable).
FALLBACK = {
    "ablation": {"phone.accuracy": 0.057, "sentence.accuracy": 0.068,
                 "sentence.fluency": 0.100, "sentence.prosody": 0.092, "sentence.total": 0.068},
    "hia": {"phone.accuracy": 0.315, "sentence.accuracy": 0.632,
            "sentence.fluency": 0.669, "sentence.prosody": 0.670, "sentence.total": 0.656},
    "baseline": {"phone.accuracy": 0.544, "sentence.accuracy": 0.744,
                 "sentence.fluency": 0.670, "sentence.prosody": 0.680, "sentence.total": 0.768},
}

# Colorblind-safe, print-friendly palette.
C_ABL = "#9A9A93"   # gray  — no acoustic input
C_HIA = "#2E6FB5"   # blue  — HIA soft prompt
C_BASE = "#1D9E75"  # teal  — raw audio baseline


def load_pcc(path: Path, fallback: dict) -> dict:
    if not path.exists():
        print(f"[warn] {path} missing -> using preliminary fallback values")
        return dict(fallback)
    data = json.loads(path.read_text())
    metrics = data.get("metrics", {})
    out = {}
    for key, _ in CHART_KEYS:
        v = metrics.get(key, {})
        pcc = v.get("pcc")
        out[key] = float(pcc) if pcc is not None else 0.0
    return out


def save_all(fig, stem: str) -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf", "svg"):
        out = FIG_DIR / f"{stem}.{ext}"
        fig.savefig(out, dpi=300, bbox_inches="tight", transparent=False)
        print(f"  wrote {out}")


def make_comparison_chart() -> None:
    abl = load_pcc(ABL_METRICS, FALLBACK["ablation"])
    hia = load_pcc(HIA_METRICS, FALLBACK["hia"])
    base = load_pcc(BASE_METRICS, FALLBACK["baseline"])

    labels = [lab for _, lab in CHART_KEYS]
    keys = [k for k, _ in CHART_KEYS]
    abl_v = [abl[k] for k in keys]
    hia_v = [hia[k] for k in keys]
    base_v = [base[k] for k in keys]

    x = range(len(keys))
    w = 0.27

    fig, ax = plt.subplots(figsize=(9, 5))
    b1 = ax.bar([i - w for i in x], abl_v, w, label="Ablation (no HIA, no acoustic)", color=C_ABL)
    b2 = ax.bar(list(x), hia_v, w, label="HIA-Qwen (GOP soft prompt)", color=C_HIA)
    b3 = ax.bar([i + w for i in x], base_v, w, label="Baseline (raw audio)", color=C_BASE)

    for bars in (b1, b2, b3):
        for rect in bars:
            h = rect.get_height()
            ax.annotate(f"{h:.3f}", (rect.get_x() + rect.get_width() / 2, h),
                        ha="center", va="bottom", fontsize=8,
                        xytext=(0, 2), textcoords="offset points")

    ax.set_ylabel("PCC (Pearson correlation)", fontsize=11)
    ax.set_ylim(0, 0.88)
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels, fontsize=10)
    ax.legend(frameon=False, fontsize=9, loc="upper left")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.yaxis.grid(True, linestyle="--", alpha=0.4)
    ax.set_axisbelow(True)
    fig.tight_layout()
    save_all(fig, "fig1_ablation_pcc_comparison")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Paper-style architecture diagram helpers
# ---------------------------------------------------------------------------

# Light fills with dark edges — the academic-paper look.
PALETTE = {
    "frozen":  ("#ECEBE6", "#8A887F"),   # gray — frozen module
    "hia":     ("#E6F1FB", "#2E6FB5"),   # blue — HIA internals
    "tensor":  ("#FFFFFF", "#9A988F"),   # white — tensor / hidden state
    "proj":    ("#E1F5EE", "#1D9E75"),   # teal — trainable projector
    "inject":  ("#FAEEDA", "#BA7517"),   # amber — injection / prompt
    "llm":     ("#E6F1FB", "#2E6FB5"),   # blue — Qwen decoder
    "lora":    ("#F1E7F7", "#8E44AD"),   # purple — LoRA adapter (trainable)
    "out":     ("#E8F5E9", "#3C8C4A"),   # green — output
}


def pbox(ax, cx, cy, w, h, title, kind, sublines=None, dashed=False,
         title_size=11, sub_size=8.5, mono=False):
    """Centered paper-style box: light fill, dark edge, dark text."""
    fill, edge = PALETTE[kind]
    x, y = cx - w / 2, cy - h / 2
    rect = FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.015,rounding_size=0.04",
        linewidth=1.3, facecolor=fill, edgecolor=edge,
        linestyle=(0, (5, 3)) if dashed else "solid",
    )
    ax.add_patch(rect)
    sublines = sublines or []
    if sublines:
        ax.text(cx, y + h - 0.30, title, ha="center", va="center",
                fontsize=title_size, color="#1a1a1a", weight="bold")
        for i, s in enumerate(sublines):
            ax.text(cx, y + h - 0.62 - i * 0.34, s, ha="center", va="center",
                    fontsize=sub_size, color="#333333",
                    family="monospace" if mono else None)
    else:
        ax.text(cx, cy, title, ha="center", va="center",
                fontsize=title_size, color="#1a1a1a", weight="bold")


def container(ax, x0, y0, x1, y1, label, edge="#8A887F", dashed=True):
    """Group container with a top-left label tag."""
    rect = FancyBboxPatch(
        (x0, y0), x1 - x0, y1 - y0,
        boxstyle="round,pad=0.01,rounding_size=0.05",
        linewidth=1.4, facecolor="none", edgecolor=edge,
        linestyle=(0, (6, 3)) if dashed else "solid",
    )
    ax.add_patch(rect)
    ax.text(x0 + 0.18, y1 - 0.02, label, ha="left", va="center",
            fontsize=9.5, color=edge, weight="bold",
            bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none"))


def arrow(ax, x1, y1, x2, y2, label=None, lw=1.3):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>",
                                 mutation_scale=13, linewidth=lw,
                                 color="#555555", shrinkA=2, shrinkB=2))
    if label:
        ax.text((x1 + x2) / 2 + 0.18, (y1 + y2) / 2, label, ha="left", va="center",
                fontsize=7.6, color="#666666", family="monospace")


def make_architecture_diagram() -> None:
    fig, ax = plt.subplots(figsize=(11, 14.8))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 19.9)
    ax.axis("off")

    xL, xM, xR = 2.7, 7.0, 11.3

    # ---- Input -----------------------------------------------------------
    pbox(ax, xM, 18.5, 4.2, 0.8, "Speech waveform", "frozen")
    arrow(ax, xM, 18.1, xM, 17.65)
    pbox(ax, xM, 17.2, 5.0, 0.8, "Kaldi forced alignment + GOP", "frozen")
    arrow(ax, xM, 16.8, xM, 16.05, label="GOP [B,T,2]")

    # ---- HIA encoder container ------------------------------------------
    container(ax, 0.6, 12.5, 13.4, 16.0, "HIA encoder  (frozen)", edge="#2E6FB5", dashed=True)
    pbox(ax, xM, 15.45, 6.4, 0.66, "Phone / word embedding + positional encoding", "hia",
         title_size=9.5)
    arrow(ax, xM, 15.12, xM, 14.78)
    pbox(ax, xM, 14.35, 6.8, 0.78, "Interactive Attention Module (IAM)", "hia",
         sublines=["depth = 3,  heads = 1,  d = 48"], title_size=10, sub_size=8)
    arrow(ax, xM, 13.85, xM, 13.5)
    pbox(ax, xM, 13.05, 7.6, 0.72, "Hierarchical readout  (phone / word / utterance)", "hia",
         title_size=9.5)

    # ---- Three hidden states (fan-out) ----------------------------------
    arrow(ax, xM - 1.4, 12.69, xL, 11.95)
    arrow(ax, xM, 12.69, xM, 11.95)
    arrow(ax, xM + 1.4, 12.69, xR, 11.95)
    pbox(ax, xL, 11.45, 3.1, 0.86, "H_phn", "tensor", sublines=["[B, Tp, 48]"],
         title_size=10.5, sub_size=8.5, mono=True)
    pbox(ax, xM, 11.45, 3.1, 0.86, "H_wrd", "tensor", sublines=["[B, Tw, 48]"],
         title_size=10.5, sub_size=8.5, mono=True)
    pbox(ax, xR, 11.45, 3.1, 0.86, "H_utt", "tensor", sublines=["[B, 1, 48]"],
         title_size=10.5, sub_size=8.5, mono=True)

    # ---- Projector container (trainable) --------------------------------
    container(ax, 0.6, 7.7, 13.4, 10.85, "Modal-alignment projectors  (trainable)",
              edge="#1D9E75", dashed=False)
    for cx, name in ((xL, "Phone projector"), (xM, "Word projector"), (xR, "Utt projector")):
        arrow(ax, cx, 11.02, cx, 10.55)
        pbox(ax, cx, 9.75, 3.4, 1.4, name, "proj",
             sublines=["Linear 48 → 4096", "GELU", "Linear 4096 → 4096"],
             title_size=10, sub_size=8)

    # ---- Projected soft tokens -> converge ------------------------------
    arrow(ax, xL, 9.05, xL, 8.0, label="P_phn")
    arrow(ax, xM, 9.05, xM, 8.0, label="P_wrd")
    arrow(ax, xR, 9.05, xR, 8.0, label="P_utt")

    # ---- Prompt construction + injection --------------------------------
    container(ax, 0.6, 4.0, 13.4, 7.4, "Prompt construction & soft-token injection",
              edge="#BA7517", dashed=False)
    pbox(ax, 3.7, 6.05, 5.4, 1.7, "Text prompt template", "inject",
         sublines=["system + reference text", "reference phone sequence",
                   "<|hia_utt|> <|hia_word|> <|hia_phone|>", "+ JSON schema instruction"],
         title_size=9.5, sub_size=7.6, mono=False)
    pbox(ax, 10.2, 6.45, 4.6, 0.66, "Token embedding layer", "inject", title_size=9.5)
    arrow(ax, 6.4, 6.45, 7.9, 6.45)
    pbox(ax, 10.2, 5.0, 4.6, 0.92, "Embedding-level merge", "inject",
         sublines=["replace placeholder embeddings with P_phn / P_wrd / P_utt"],
         title_size=9.5, sub_size=7.4)
    arrow(ax, 10.2, 6.12, 10.2, 5.46)
    # soft tokens feeding the merge
    arrow(ax, xL, 7.7, 8.6, 5.2, lw=1.0)
    arrow(ax, xR, 7.7, 11.8, 5.2, lw=1.0)

    arrow(ax, xM, 3.98, xM, 3.4, label="inputs_embeds")

    # ---- Qwen decoder + LoRA --------------------------------------------
    container(ax, 1.6, 1.45, 12.4, 3.25, "Qwen2-Audio-7B decoder  (frozen, 4-bit)",
              edge="#2E6FB5", dashed=True)
    pbox(ax, 5.2, 2.3, 5.6, 1.05, "Self-attention × N", "llm",
         sublines=["q_proj  k_proj  v_proj  o_proj"], title_size=9.5, sub_size=8)
    pbox(ax, 10.0, 2.3, 3.4, 1.05, "LoRA adapter", "lora",
         sublines=["r = 8,  α = 16"], title_size=9.5, sub_size=8)
    arrow(ax, 8.0, 2.3, 8.3, 2.3, lw=1.0)

    arrow(ax, xM, 1.43, xM, 0.95)

    # ---- Output ----------------------------------------------------------
    pbox(ax, xM, 0.5, 6.6, 0.78, "Autoregressive JSON  (sentence / word / phone)", "out",
         title_size=9.5)

    # ---- Legend ----------------------------------------------------------
    legend = [("Frozen (dashed)", "frozen"), ("HIA / LLM", "hia"),
              ("Trainable projector", "proj"), ("Injection", "inject"),
              ("LoRA", "lora"), ("Output", "out")]
    lx = 0.7
    for name, kind in legend:
        fill, edge = PALETTE[kind]
        ax.add_patch(FancyBboxPatch((lx, 19.45), 0.34, 0.34,
                     boxstyle="round,pad=0.01", linewidth=1.1,
                     facecolor=fill, edgecolor=edge))
        ax.text(lx + 0.46, 19.62, name, ha="left", va="center", fontsize=8, color="#333333")
        lx += 2.15

    fig.tight_layout()
    save_all(fig, "fig2_architecture")
    plt.close(fig)


def main() -> None:
    print("Generating figure 1: ablation PCC comparison")
    make_comparison_chart()
    print("Generating figure 2: architecture diagram")
    make_architecture_diagram()
    print(f"\nDone. Figures in {FIG_DIR}/")


if __name__ == "__main__":
    main()
