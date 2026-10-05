"""Collect every finished run into the results table and draw the figures.

results/main_table.csv / .md    macro-F1 mean +- std per method and K (3 seeds)
figures/f1_vs_shots.png         macro-F1 vs K for the main methods
figures/f1_vs_shots_sota.png    the same for the base paper's comparison methods
figures/confusion_prga.png      PRGA confusion matrix
figures/per_class_f1.png        per-class F1, PRGA vs Tip-Adapter-F
"""
import glob
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, f1_score

from common import CLASSES, FIGURES, RESULTS, SEEDS, SPLITS

K_ORDER = ["1", "2", "4", "8", "16", "full"]
# one fixed colour per method, so a method keeps its colour in every figure
SERIES = {"PRGA2Bft-clip": ("PRGA + DINOv2 cache (2 backbones)", "#0d366b"),
          "PRGA-clip": ("PRGA (1 backbone)", "#2a78d6"),
          "TipAdapterF-clip": ("Tip-Adapter-F", "#eb6834"),
          "BasePaperExact": ("Base paper, exact replication", "#4a3aa7"),
          "BasePaperExact-noGraph": ("Same recipe, no graph (control)", "#8f8a80"),
          "PlantCaFoLite-clip": ("PlantCaFo-style cache", "#eda100"),
          "EfficientNetB0-finetune": ("EfficientNet-B0", "#e87ba4"),
          "LinearProbe-clip": ("Linear probe", "#008300")}
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
LS = {"BasePaperExact-noGraph": (0, (5, 2))}              # dashed: control
# the methods from the base paper's own comparison table (Table 1), re-implemented in adapters.py
COMPARED = {"BasePaperExact": ("Base paper (Ahmad et al. 2025), exact", "#4a3aa7"),
        "TipAdapterF-clip": ("Tip-Adapter-F (ECCV'22)", "#eb6834"),
        "TaskRes-clip": ("TaskRes (CVPR'23)", "#008300"),
        "GraphAdapterLite-clip": ("GraphAdapter-lite (NeurIPS'23)", "#e87ba4"),
        "CLIPAdapter-clip": ("CLIP-Adapter (IJCV'24)", "#eda100"),
        "CLAP-clip": ("CLAP (CVPR'24)", "#1baab0"),
        "CoOp-clip": ("CoOp (IJCV'22)", "#8f8a80"),
        "PlantCaFoLite-clip": ("PlantCaFo-style cache (2 backbones)", "#8a5a00"),
        "PRGA-clip": ("PRGA (1 backbone)", "#2a78d6"),
        "PRGA2Bft-clip": ("PRGA + DINOv2 cache (2 backbones)", "#0d366b")}
EXCLUDE = {"PRGA-clip-samefeats"}      # a re-run of PRGA used only inside 13_prga_dinov2.py as a sanity reference


def load_runs():
    """All finished runs: <name>.csv when a sweep completed, otherwise its resumable <name>.jsonl log."""
    frames = []
    for f in glob.glob(str(RESULTS / "runs" / "*")):
        if "_abl" in f or not f.endswith((".csv", ".jsonl")):
            continue
        if f.endswith(".jsonl"):
            if Path(f[:-6] + ".csv").exists():
                continue
            frames.append(pd.read_json(f, lines=True))
        else:
            frames.append(pd.read_csv(f))
    df = pd.concat(frames)
    df["K"] = df["K"].astype(str)
    return df[df.K.isin(K_ORDER) & ~df.method.isin(EXCLUDE)].drop_duplicates(["method", "K", "seed"])


def main_table(df):
    g = df.groupby(["method", "K"])
    tab = pd.DataFrame({"f1_mean": g.macro_f1.mean(), "f1_std": g.macro_f1.std().fillna(0),
                        "acc_mean": g.acc.mean(), "acc_std": g.acc.std().fillna(0)}).reset_index()
    tab.to_csv(RESULTS / "main_table_long.csv", index=False)
    cell = tab.assign(v=lambda t: t.f1_mean.map("{:.3f}".format) + " ± " + t.f1_std.map("{:.3f}".format))
    wide = cell.pivot(index="method", columns="K", values="v").reindex(columns=[k for k in K_ORDER if k in cell.K.values])
    order = tab[tab.K == "4"].sort_values("f1_mean").method.tolist()
    wide = wide.reindex([m for m in order if m in wide.index] + [m for m in wide.index if m not in order])
    wide.to_csv(RESULTS / "main_table.csv")
    with open(RESULTS / "main_table.md", "w") as f:
        f.write("Macro-F1 on the cleaned, object-disjoint test set (7,612 photos; mean ± std over 3 seeds). Zero-shot rows need no shots.\n\n")
        f.write(wide.fillna("").to_markdown() + "\n")
    print(wide.to_string())
    return tab


def fig_lines(tab, series=None, fname="f1_vs_shots.png", title="Few-shot onion bulb classification (variety x health)"):
    series = SERIES if series is None else series
    fig, ax = plt.subplots(figsize=(9.6, 4.6), dpi=200)
    x = np.arange(len(K_ORDER))
    zs = tab[tab.method == "ZeroShot-clip"]
    if len(zs):
        ax.axhline(zs.f1_mean.iloc[0], color=INK2, lw=1.2, ls=(0, (4, 3)))
        ax.text(x[0] - .12, zs.f1_mean.iloc[0] - 0.006, "zero-shot CLIP (descriptors)", color=INK2, va="top", fontsize=8)
    for m, (label, col) in series.items():
        t = tab[tab.method == m].set_index("K").reindex(K_ORDER)
        if t.f1_mean.isna().all():
            continue
        ax.plot(x, t.f1_mean, color=col, lw=2, ls=LS.get(m, "-"), marker="o", ms=5, label=label, zorder=3,
                markeredgecolor="white", markeredgewidth=1.2)
        ax.fill_between(x, t.f1_mean - t.f1_std, t.f1_mean + t.f1_std, color=col, alpha=0.12, lw=0)
    ax.set_xticks(x, ["1", "2", "4", "8", "16", "all pool"])
    ax.set_xlabel("training images per class (K)", color=INK2)
    ax.set_ylabel("macro-F1 (cleaned test set, 7,612 photos)", color=INK2)
    ax.grid(axis="y", color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=INK2, labelsize=8)
    ax.set_xlim(-0.2, len(K_ORDER) - 1 + 0.9)
    ax.legend(frameon=False, fontsize=8, loc="upper left", bbox_to_anchor=(1.01, 1.0), labelcolor=INK)
    ax.set_title(title, loc="left", color=INK, fontsize=11)
    fig.tight_layout()
    fig.savefig(FIGURES / fname)
    plt.close(fig)


def fig_confusion(method="PRGA-clip", K="16", seed=1):
    p = RESULTS / "preds" / f"{method}_K{K}_s{seed}.npy"
    if not p.exists():
        return
    y = pd.read_csv(SPLITS / "test80.csv").label.values
    pred = np.load(p).astype(np.float32).argmax(1)
    cm = confusion_matrix(y, pred, labels=range(len(CLASSES)), normalize="true")
    fig, ax = plt.subplots(figsize=(5.2, 4.4), dpi=200)
    ax.imshow(cm, cmap=matplotlib.colors.LinearSegmentedColormap.from_list(
        "blue", ["#fcfcfb", "#cde2fb", "#6da7ec", "#256abf", "#0d366b"]), vmin=0, vmax=1)
    for i in range(len(CLASSES)):
        for j in range(len(CLASSES)):
            if cm[i, j] >= 0.05:
                ax.text(j, i, f"{cm[i, j]:.2f}", ha="center", va="center", fontsize=9,
                        color="white" if cm[i, j] > 0.5 else INK)
    ax.set_xticks(range(len(CLASSES)), CLASSES, rotation=30, ha="right", fontsize=8)
    ax.set_yticks(range(len(CLASSES)), CLASSES, fontsize=8)
    ax.set_xlabel("predicted", color=INK2)
    ax.set_ylabel("true", color=INK2)
    ax.set_title(f"PRGA confusion matrix (K={K}, row-normalised)", loc="left", color=INK, fontsize=10)
    fig.tight_layout()
    fig.savefig(FIGURES / "confusion_prga.png")
    plt.close(fig)


def fig_per_class(K="16"):
    y = pd.read_csv(SPLITS / "test80.csv").label.values
    res = {}
    for m in ("PRGA-clip", "TipAdapterF-clip"):
        fs = [f1_score(y, np.load(RESULTS / "preds" / f"{m}_K{K}_s{s}.npy").astype(np.float32).argmax(1),
                       average=None, labels=range(len(CLASSES)), zero_division=0)
              for s in SEEDS if (RESULTS / "preds" / f"{m}_K{K}_s{s}.npy").exists()]
        if fs:
            res[m] = np.mean(fs, 0)
    if len(res) < 2:
        return
    order = np.argsort(res["PRGA-clip"] - res["TipAdapterF-clip"])
    fig, ax = plt.subplots(figsize=(7.2, 3), dpi=200)
    yy = np.arange(len(CLASSES))
    for k, (m, off) in enumerate((("TipAdapterF-clip", 0.2), ("PRGA-clip", -0.2))):
        label, col = SERIES[m]
        ax.barh(yy + off, res[m][order], height=0.36, color=col, label=label)
    ax.set_yticks(yy, [CLASSES[i] for i in order], fontsize=7)
    ax.set_xlabel(f"per-class F1 (K={K}, mean of 3 seeds)", color=INK2)
    ax.grid(axis="x", color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    fig.tight_layout()
    fig.savefig(FIGURES / "per_class_f1.png")
    plt.close(fig)


def main():
    df = load_runs()
    tab = main_table(df)
    fig_lines(tab)
    fig_lines(tab, COMPARED, "f1_vs_shots_sota.png", "Base paper and the methods it compares against, on onion bulbs")
    fig_confusion()
    fig_per_class()


if __name__ == "__main__":
    main()
