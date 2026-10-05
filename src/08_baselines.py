"""Simple baselines on the cached features: zero-shot CLIP, linear probe, Tip-Adapter, Tip-Adapter-F and the
PlantCaFo-style two-backbone cache. Also the leakage check (random split vs our split).

  python 08_baselines.py --methods zs lp tip tipf
  python 08_baselines.py --methods cafo                       (needs DINOv2 features)
  python 08_baselines.py --backbone bioclip --methods zs tipf
  python 08_baselines.py --methods zs tipf --text template    (generic templates instead of the descriptions)
  python 08_baselines.py --methods lp knn --leakage           (same model, random split vs scene-disjoint split)
"""
import argparse

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier

from common import RESULTS, SPLITS, metrics
from fewshot import LinearProbe, PlantCaFoLite, Store, TipAdapter, TipAdapterF, ZeroShot
from harness import ALL_K, load_text, run

NAMES = {"zs": "ZeroShot", "lp": "LinearProbe", "tip": "TipAdapter", "tipf": "TipAdapterF",
         "cafo": "PlantCaFoLite", "knn": "1NN"}


def leakage_check(name, method, store, make):
    """Train on the 20 % pool and test on the 80 %, once with a naive random split and once with our scene-disjoint
    split. A large gap means the random split lets near-copies of training photos into the test set."""
    rows = []
    for split, prefix in (("grouped", ""), ("random", "random_")):
        tr = store.batch(pd.read_csv(SPLITS / f"{prefix}pool20.csv"))
        te = store.batch(pd.read_csv(SPLITS / f"{prefix}test80.csv"))
        if method == "knn":      # 1-nearest-neighbour: pure memorisation, so the most sensitive to copies
            pred = KNeighborsClassifier(1, metric="cosine").fit(tr.g.numpy(), tr.y.numpy()).predict(te.g.numpy())
        elif method == "lp":
            pred = LogisticRegression(C=10.0, max_iter=3000).fit(tr.g.numpy(), tr.y.numpy()).predict(te.g.numpy())
        else:
            # augmented views exist only for our pool, so both splits use the plain feature as their single "view"
            tr.aug = tr.g[:, None]
            no_val = te.__class__(y=tr.y[:0], g=tr.g[:0])
            pred = make().fit(tr, no_val).predict(te).argmax(1).numpy()
        rows.append(dict(method=name, split=split, **metrics(te.y.numpy(), pred)))
        print(rows[-1], flush=True)
    pd.DataFrame(rows).to_csv(RESULTS / f"leakage_{name}.csv", index=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backbone", default="clip")
    ap.add_argument("--methods", nargs="+", default=["zs", "lp", "tip", "tipf"])
    ap.add_argument("--text", default="desc", choices=["desc", "template"])
    ap.add_argument("--shots", nargs="+", default=ALL_K)
    ap.add_argument("--leakage", action="store_true")
    a = ap.parse_args()

    bb = a.backbone
    T = load_text(bb, a.text)
    store = Store(bb, second="dinov2" if "cafo" in a.methods else None, need=("global", "aug"))
    makers = {
        "zs": lambda: ZeroShot(T),
        "lp": lambda: LinearProbe(),
        "tip": lambda: TipAdapter(T),
        "tipf": lambda: TipAdapterF(T),
        "cafo": lambda: PlantCaFoLite(T),
    }
    for m in a.methods:
        name = f"{NAMES[m]}-{bb}" + ("" if a.text == "desc" else "-template")
        if a.leakage:
            leakage_check(name, m, store, makers.get(m))
            continue
        if m == "zs":            # zero-shot uses no training photos: one run is enough
            run(name, makers[m], store, shots=a.shots[:1], seeds=[1])
        else:
            run(name, makers[m], store, shots=a.shots)


if __name__ == "__main__":
    main()
