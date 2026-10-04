"""Step 1.6 / 2.4 / 3.3: baselines on cached features.

  python 06_baselines.py --methods zs lp tip tipf            (Day 1)
  python 06_baselines.py --methods grid                      (base paper reproduction, needs clip_grid)
  python 06_baselines.py --methods cafo                      (PlantCaFo-lite, needs dinov2 features)
  python 06_baselines.py --backbone bioclip --methods zs tipf
  python 06_baselines.py --methods lp knn --leakage          (random vs grouped split)
"""
import argparse

from fewshot import PRGA, LinearProbe, PlantCaFoLite, Store, TipAdapter, TipAdapterF, ZeroShot
from common import RESULTS
from harness import ALL_K, load_text, run


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backbone", default="clip")
    ap.add_argument("--methods", nargs="+", default=["zs", "lp", "tip", "tipf"])
    ap.add_argument("--text", default="desc", choices=["desc", "template"])
    ap.add_argument("--shots", nargs="+", default=ALL_K)
    ap.add_argument("--leakage", action="store_true")
    a = ap.parse_args()

    bb = a.backbone
    T = load_text(bb, a.text) if bb != "dinov2" else None
    need = ("global", "aug", "grid") if "grid" in a.methods else ("global", "aug")
    store = Store(bb, second="dinov2" if "cafo" in a.methods else None, need=need)
    makers = {
        "zs": lambda: ZeroShot(T),
        "lp": lambda: LinearProbe(),
        "tip": lambda: TipAdapter(T),
        "tipf": lambda: TipAdapterF(T),
        "grid": lambda: PRGA(T, nodes="grid", text_nodes=False, test_graph=False),
        "cafo": lambda: PlantCaFoLite(T),
    }
    names = {"zs": "ZeroShot", "lp": "LinearProbe", "tip": "TipAdapter", "tipf": "TipAdapterF",
             "grid": "GridGraph_basepaper", "cafo": "PlantCaFoLite", "knn": "1NN"}
    for m in a.methods:
        name = f"{names[m]}-{bb}" + ("" if a.text == "desc" else "-template")
        if a.leakage:
            # same model and pool size; only the split differs. Augmented views exist only for the grouped pool,
            # so both splits use the plain global features (one "view" per image).
            import pandas as pd
            from common import SPLITS, metrics
            rows = []
            for split in ("grouped", "random"):
                pre = "" if split == "grouped" else "random_"
                tr = store.batch(pd.read_csv(SPLITS / f"{pre}pool20.csv"))
                te = store.batch(pd.read_csv(SPLITS / f"{pre}test80.csv"))
                tr.aug = tr.g[:, None]
                va = te.__class__(y=tr.y[:0], g=tr.g[:0])
                if m == "knn":                      # 1-nearest-neighbour: pure memorisation, most sensitive to copies
                    from sklearn.neighbors import KNeighborsClassifier
                    pred = KNeighborsClassifier(1, metric="cosine").fit(tr.g.numpy(), tr.y.numpy()).predict(te.g.numpy())
                elif m == "lp":
                    from sklearn.linear_model import LogisticRegression
                    lr = LogisticRegression(C=10.0, max_iter=3000).fit(tr.g.numpy(), tr.y.numpy())
                    pred = lr.predict(te.g.numpy())
                else:
                    pred = makers[m]().fit(tr, va).predict(te).argmax(1).numpy()
                rows.append(dict(method=name, split=split, **metrics(te.y.numpy(), pred)))
                print(rows[-1], flush=True)
            pd.DataFrame(rows).to_csv(RESULTS / f"leakage_{name}.csv", index=False)
            continue
        shots = [k for k in a.shots if not (m == "zs" and k != a.shots[0])]  # zero-shot needs one run
        run(name, makers[m], store, shots=shots, seeds=[1] if m == "zs" else [1, 2, 3])


if __name__ == "__main__":
    main()
