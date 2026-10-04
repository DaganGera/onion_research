"""Step 0b — visual check of the cleaning decisions (00_clean_dataset.py): montages of pairs around the RANSAC threshold,
of removed near-duplicates, of label conflicts, and of the most similar remaining train/test pairs after the split.

  python 00b_clean_inspect.py --bands          pairs by inlier count, to calibrate MIN_INLIERS by eye
  python 00b_clean_inspect.py --leakcheck      after 02_split.py: highest train/test similarities that remain
Images: figures/clean_*.png
"""
import argparse

import pandas as pd
import torch
from PIL import Image, ImageDraw

from common import FIGURES, RAW, ROOT, SPLITS, load_feats

CLEAN = ROOT / "data/clean"


def montage(pairs, fn, W=220):
    """pairs: list of (left_path, right_path, label)."""
    rows = (len(pairs) + 1) // 2
    h = W * 3 // 4
    im = Image.new("RGB", (4 * W + 30, rows * (h + 18)), "white")
    d = ImageDraw.Draw(im)
    for k, (a, b, txt) in enumerate(pairs):
        y, x0 = (k // 2) * (h + 18), (k % 2) * (2 * W + 20)
        for x, q in enumerate((a, b)):
            t = Image.open(RAW / q).convert("RGB")
            t.thumbnail((W, h))
            im.paste(t, (x0 + x * (W + 2), y + 15))
        d.text((x0, y + 1), f"#{k} {txt}", fill="red")
    im.save(FIGURES / fn)
    print("wrote", FIGURES / fn)


def bands():
    pc = pd.read_csv(CLEAN / "pairs_checked.csv")
    pc = pc[pc.same_class]
    print(pc.inliers.describe())
    for lo, hi in ((4, 7), (7, 10), (10, 13), (13, 20)):
        s = pc[(pc.inliers >= lo) & (pc.inliers < hi)]
        print(f"inliers [{lo},{hi}): {len(s)} pairs")
        s = s.sample(min(8, len(s)), random_state=0)
        montage([(r.a, r.b, f"inl {r.inliers} cos {r.cos:.3f}") for r in s.itertuples()], f"clean_inliers_{lo}_{hi}.png")
    rem = pd.read_csv(CLEAN / "removed.csv")
    for reason, fn in (("near-identical", "clean_removed_neardup.png"), ("label conflict", "clean_removed_conflict.png")):
        s = rem[rem.reason.str.startswith(reason)]
        s = s.sample(min(8, len(s)), random_state=0)
        if len(s):
            montage([(r.path, r.kept_twin, "removed | kept") for r in s.itertuples()], fn)


def leakcheck():
    m = pd.read_csv(CLEAN / "meta_clean.csv")
    f = load_feats("clip_global")
    row = {p: i for i, p in enumerate(f["paths"])}
    X = f["feats"].float()
    pool, test = pd.read_csv(SPLITS / "pool20.csv"), pd.read_csv(SPLITS / "test80.csv")
    assert not set(pool.group) & set(test.group)
    P, T = X[[row[p] for p in pool.path]], X[[row[p] for p in test.path]]
    s, j = (T @ P.T).max(1)
    o = s.argsort(descending=True)[:16]
    print("test->pool max CLIP cosine, quantiles 50/90/99/100 %:",
          [round(float(q), 4) for q in torch.quantile(s, torch.tensor([.5, .9, .99, 1.]))])
    montage([(test.path.iloc[int(i)], pool.path.iloc[int(j[i])], f"test|pool cos {float(s[i]):.4f}") for i in o],
            "clean_leakcheck_top16.png")
    print(f"photos: pool {len(pool)}, test {len(test)}, total {len(m)}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--bands", action="store_true")
    ap.add_argument("--leakcheck", action="store_true")
    a = ap.parse_args()
    if a.bands:
        bands()
    if a.leakcheck:
        leakcheck()
