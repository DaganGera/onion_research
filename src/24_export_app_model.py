"""Train the final model (PRGA + DINOv2 cache) once and save it for the web app (app.py). Run on Kaggle.

  DATASET=onion python 24_export_app_model.py --backbone clip
  DATASET=bees  python 24_export_app_model.py --backbone clip_l14

For K = 4 and "full" (seed 1) it trains exactly as 18_equal_training.py does, scores the test set once (to print
next to the checkpoint), and saves everything prediction needs: the graph network's weights, the refined cache keys,
alpha/beta/gamma, the DINOv2 keys and a2/b2, the class texts and the OWLv2 prompts.
Output: checkpoints/app_<domain>_K<K>.pt and app_examples/ (a few test photos, right and wrong, for the app).
"""
import argparse
import importlib

import pandas as pd
import torch
from PIL import Image

from common import CKPT, CLASSES, DATASET, DROOT, PAD_SQUARE, RAW, SPLITS, metrics, seed_all
from fewshot import PRGA, Store, onehot
from harness import load_text

dino = importlib.import_module("13_prga_dinov2")
owl = importlib.import_module("05_detect_regions")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backbone", default="clip")
    ap.add_argument("--shots", nargs="+", default=["4", "full"])
    ap.add_argument("--seed", type=int, default=1)
    a = ap.parse_args()
    T = load_text(a.backbone, "desc")
    store = Store(a.backbone, second="dinov2", need=("global", "aug", "regions"))
    test_df = pd.read_csv(SPLITS / "test80.csv").drop_duplicates("path").reset_index(drop=True)
    te = store.batch(test_df)
    CKPT.mkdir(parents=True, exist_ok=True)
    for K in a.shots:
        s = a.seed
        tr = store.batch(pd.read_csv(SPLITS / f"support_K{K}_s{s}.csv"), with_aug=True)
        va = store.batch(pd.read_csv(SPLITS / f"val_K{K}_s{s}.csv"))
        seed_all(s)
        m = PRGA(T, **dino.CHOSEN[1]).fit(tr, va)
        k2, L = dino.dino_keys(tr, finetune=True), onehot(tr.y)
        a2, b2 = dino.tune(dino.prga_logits(m, va), va.g2, k2, L, va.y)
        probs = dino.fused(dino.prga_logits(m, te), te.g2, k2, L, a2, b2).softmax(1)
        score = metrics(te.y.numpy(), probs.argmax(1).numpy())
        ck = dict(domain=DATASET, backbone=a.backbone, pad=PAD_SQUARE, classes=CLASSES, K=K, seed=s, n_support=len(tr),
                  T=T, cfg=m.cfg, test_graph=m.test_graph, net=m.net.state_dict(),
                  s={k: float(v) for k, v in m.s.items()}, keys=m.keys.cpu(), L=m.L.cpu(),
                  k2=k2.cpu(), a2=float(a2), b2=float(b2),
                  owl=dict(object=owl.OBJECT_PROMPTS, instance=owl.INSTANCE_PROMPTS, spot=owl.SPOT_PROMPTS),
                  test=dict(n=len(test_df), **{k: round(v, 4) for k, v in score.items()}))
        torch.save(ck, CKPT / f"app_{DATASET}_K{K}.pt")
        print(f"{DATASET} K={K}: {len(tr)} support photos, test macro-F1 {score['macro_f1']:.4f}", flush=True)

        if K == "full":       # a few test photos for the app: 3 right and 1 wrong per class
            out = DROOT / "app_examples"
            out.mkdir(exist_ok=True)
            pred = probs.argmax(1).numpy()
            for c, name in enumerate(CLASSES):
                rows = test_df.index[test_df.label == c]
                right = [i for i in rows if pred[i] == c][:3]
                wrong = [i for i in rows if pred[i] != c][:1]
                for tag, ix in (("right", right), ("wrong", wrong)):
                    for j, i in enumerate(ix):
                        im = Image.open(RAW / test_df.path[i]).convert("RGB")
                        im.thumbnail((800, 800))                       # keep the example files small
                        im.save(out / f"{name.replace(' ', '_')}_{tag}{j}.jpg", quality=90)


if __name__ == "__main__":
    main()
