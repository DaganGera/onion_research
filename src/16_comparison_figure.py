"""The comparison figure: one dot chart per K, every method in the same row order (best average at the top).

Dot = mean macro-F1 over the 3 seeds, line = lowest to highest seed. Blue = PRGA (filled: with DINOv2 cache, hollow:
CLIP only), orange = the base paper, green = PlantCaFo-style cache, grey = everything else.
Output: figures/comparison_dots.png and comparison_dots_portrait.png
"""
import importlib

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from common import FIGURES

ev = importlib.import_module("15_tables_and_figures")

METHODS = {  # run name -> label shown on the chart
    "PRGA2Bft-clip": "PRGA + DINOv2 cache (2 backbones)",
    "PRGA-clip": "PRGA (1 backbone)",
    "PlantCaFoLite-clip": "PlantCaFo-style cache (2 backbones)",
    "CLIPAdapter-clip": "CLIP-Adapter (IJCV'24)",
    "TaskRes-clip": "TaskRes (CVPR'23)",
    "CoOp-clip": "CoOp (IJCV'22)",
    "CLAP-clip": "CLAP (CVPR'24)",
    "TipAdapterF-clip": "Tip-Adapter-F (ECCV'22)",
    "GraphAdapterLite-clip": "GraphAdapter-lite (NeurIPS'23)",
    "LinearProbe-clip": "Linear probe",
    "BasePaperExact": "Base paper, exact (Ahmad et al. 2025)",
}
BLUE, ORANGE, AQUA, GREY = "#2a78d6", "#eb6834", "#1baf7a", "#a3a29c"
STYLE = {"PRGA2Bft-clip": (BLUE, True), "PRGA-clip": (BLUE, False), "BasePaperExact": (ORANGE, True),
         "PlantCaFoLite-clip": (AQUA, True)}
INK, INK2, GRID, SURF = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
K_ORDER = ["1", "2", "4", "8", "16", "full"]
K_TITLE = {"1": "K = 1 photo per class", "2": "K = 2", "4": "K = 4", "8": "K = 8", "16": "K = 16",
           "full": "all training photos"}


def main(portrait=False):
    df = ev.load_runs()
    df = df[df.method.isin(METHODS)]
    g = df.groupby(["method", "K"]).macro_f1
    stat = g.agg(["mean", "min", "max"]).reset_index()
    order = stat.groupby("method")["mean"].mean().sort_values(ascending=True).index.tolist()   # best at the top
    y = {m: i for i, m in enumerate(order)}

    rows, cols = (3, 2) if portrait else (2, 3)
    fig, axes = plt.subplots(rows, cols, figsize=(10.5, 12.5) if portrait else (13.5, 8.6), dpi=200, sharey=True,
                             facecolor=SURF)
    for ax, K in zip(axes.flat, K_ORDER):
        ax.set_facecolor(SURF)
        s = stat[stat.K == K].set_index("method")
        left = max(0.0, np.floor((s["min"].min() - 0.02) * 20) / 20)
        pad = 0.012 * (1 - left)
        best = round(s["mean"].max(), 3)
        for m in order:
            if m not in s.index:
                continue
            col, filled = STYLE.get(m, (GREY, True))
            r = s.loc[m]
            ax.plot([r["min"], r["max"]], [y[m]] * 2, color=col, lw=1.6, solid_capstyle="round", zorder=2,
                    alpha=0.9 if m in STYLE else 0.7)
            ax.scatter([r["mean"]], [y[m]], s=58, zorder=3, color=col if filled else SURF, edgecolors=col
                       if not filled else SURF, linewidths=2.0 if not filled else 1.5)
            top = round(r["mean"], 3) == best                                       # ties are all bold
            if m in STYLE or top:                                                     # selective direct labels
                ax.text(r["max"] + pad, y[m], f"{r['mean']:.3f}", va="center", ha="left", fontsize=8,
                        color=INK if m in STYLE else INK2, fontweight="bold" if top else "normal")
        ax.set_xlim(left, 1 + 0.13 * (1 - left))
        ax.set_title(K_TITLE[K], loc="left", color=INK, fontsize=11, pad=8)
        ax.grid(axis="x", color=GRID, lw=0.8)
        ax.set_axisbelow(True)
        ax.tick_params(colors=INK2, labelsize=8, length=0)
        for side in ("top", "right", "left"):
            ax.spines[side].set_visible(False)
        ax.spines["bottom"].set_color(GRID)
        ax.set_ylim(-0.7, len(order) - 0.3)
    for ax in axes[:, 0]:
        ax.set_yticks(range(len(order)), [METHODS[m] for m in order])
        ax.tick_params(axis="y", labelsize=9, labelcolor=INK)
    for ax in axes[-1]:
        ax.set_xlabel("macro-F1 on the cleaned test set (higher is better)", color=INK2, fontsize=9)
    fig.suptitle("PRGA vs the base paper and the methods from its comparison table",
                 x=0.012, ha="left", fontsize=14, color=INK, y=0.995)
    sub = ("Dot = mean of 3 seeds; line = lowest to highest seed. Same row order in every panel (best average at the "
           "top). Bold number = best mean in that panel. Onion bulbs, 4 classes, 7,612 test photos.")
    if portrait:
        sub = sub.replace(" (best average at the top).", "\n(best average at the top).")
    fig.text(0.012, 0.972 if portrait else 0.962, sub, fontsize=9, color=INK2, ha="left", va="top")
    fig.tight_layout(rect=(0, 0, 1, 0.94 if portrait else 0.94), h_pad=2.0)
    name = "comparison_dots_portrait.png" if portrait else "comparison_dots.png"
    fig.savefig(FIGURES / name, facecolor=SURF)
    print("wrote", FIGURES / name)


if __name__ == "__main__":
    main()
    main(portrait=True)
