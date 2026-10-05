"""Live demo: train the final model from K photos per class, then classify photos it has never seen.

  python demo.py                  K = 4 photos per class, support set 1, 12 example photos
  python demo.py --k 1 --n 16     train on a single photo per class
  python demo.py --model tipf     the same with Tip-Adapter-F, for comparison
  python demo.py --mistakes       show only photos the model got wrong

What happens:
  1. load the cached CLIP / DINOv2 features (made once by 06_extract_features.py)
  2. train PRGA on the K support photos per class, then add the DINOv2 cache (as in 13_prga_dinov2.py)
  3. score the whole held-out test set (7,612 photos) and print accuracy and macro-F1
  4. draw random test photos with the true class and the prediction -> figures/demo_predictions.png
Training takes seconds, because the big networks are frozen and their features are already on disk.
"""
import argparse
import importlib
import time

import matplotlib
import numpy as np
import pandas as pd
from PIL import Image, ImageOps

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from common import CLASSES, FIGURES, RAW, SPLITS, metrics, seed_all
from fewshot import PRGA, Store, TipAdapterF, onehot
from harness import load_text

dino = importlib.import_module("13_prga_dinov2")       # reuse the exact final-model code, no copy


def train_final(T, tr, va):
    """PRGA (validation-chosen configuration) + DINOv2 cache with fine-tuned keys. Returns a predict function."""
    m = PRGA(T, **dino.CHOSEN[1]).fit(tr, va)
    k2 = dino.dino_keys(tr, finetune=True)
    L = onehot(tr.y)
    a2, b2 = dino.tune(dino.prga_logits(m, va), va.g2, k2, L, va.y)
    return lambda b: dino.fused(dino.prga_logits(m, b), b.g2, k2, L, a2, b2).softmax(1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", default="4", choices=["1", "2", "4", "8", "16", "full"])
    ap.add_argument("--seed", type=int, default=1, choices=[1, 2, 3])
    ap.add_argument("--n", type=int, default=12, help="how many test photos to show")
    ap.add_argument("--model", default="final", choices=["final", "tipf"])
    ap.add_argument("--mistakes", action="store_true", help="show only wrongly classified photos")
    a = ap.parse_args()

    seed_all(a.seed)
    T = load_text("clip", "desc")
    store = Store("clip", second="dinov2", need=("global", "aug", "regions"))
    sup = pd.read_csv(SPLITS / f"support_K{a.k}_s{a.seed}.csv")
    tr = store.batch(sup, with_aug=True)
    va = store.batch(pd.read_csv(SPLITS / f"val_K{a.k}_s{a.seed}.csv"))
    test_df = pd.read_csv(SPLITS / "test80.csv")
    te = store.batch(test_df)
    print(f"support set: {len(sup)} photos ({a.k} per class), validation: {len(va)}, test: {len(te)}")

    t0 = time.time()
    if a.model == "final":
        predict = train_final(T, tr, va)
        name = "PRGA + DINOv2 cache"
    else:
        m = TipAdapterF(T).fit(tr, va)
        predict, name = m.predict, "Tip-Adapter-F"
    print(f"trained {name} in {time.time() - t0:.1f} s")

    probs = predict(te).cpu()
    pred = probs.argmax(1)
    r = metrics(te.y.numpy(), pred.numpy())
    print(f"test set: accuracy {r['acc']:.3f}, macro-F1 {r['macro_f1']:.3f}")

    rng = np.random.default_rng(0)
    pool = np.where(pred != te.y)[0] if a.mistakes else np.arange(len(test_df))
    pick = rng.choice(pool, size=min(a.n, len(pool)), replace=False)
    cols = 4
    rows = int(np.ceil(len(pick) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(3.2 * cols, 3.5 * rows), dpi=120)
    for ax, i in zip(np.atleast_1d(axes).flat, pick):
        true, guess, conf = CLASSES[te.y[i]], CLASSES[pred[i]], probs[i].max().item()
        with Image.open(RAW / test_df.path[i]) as im:
            ax.imshow(ImageOps.fit(im.convert("RGB"), (256, 256)))
        ok = true == guess
        ax.set_title(f"true: {true}\npred: {guess} ({conf:.0%})", fontsize=8, color="#1a7f37" if ok else "#c0392b")
        print(f"{'OK   ' if ok else 'WRONG'} {test_df.path[i]:60s} true={true:22s} pred={guess:22s} {conf:.0%}")
    for ax in np.atleast_1d(axes).flat:
        ax.axis("off")
    fig.suptitle(f"{name}, trained on {a.k} photo(s) per class - test macro-F1 {r['macro_f1']:.3f}", fontsize=11)
    fig.tight_layout()
    out = FIGURES / ("demo_mistakes.png" if a.mistakes else "demo_predictions.png")
    fig.savefig(out)
    print("saved", out)


if __name__ == "__main__":
    main()
