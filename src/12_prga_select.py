"""Choose the PRGA configuration using validation data only (the test set is not touched here).

Each candidate is trained for K in {1, 4, 16} x 3 seeds; the winner is the one with the highest mean validation
macro-F1. 11_prga.py reads the result.

  python 12_prga_select.py          single-backbone candidates -> results/prga_selection_val.csv
  python 12_prga_select.py --dino   with a DINOv2 cache inside -> results/prga_selection_val_dino.csv
"""
import sys

import pandas as pd

sys.path.insert(0, __import__("os").path.dirname(__import__("os").path.abspath(__file__)))
from common import RESULTS, SPLITS, seed_all  # noqa: E402
from fewshot import PRGA, Store  # noqa: E402
from harness import load_text  # noqa: E402

T = load_text("clip", "desc")
DINO = "--dino" in sys.argv
store = Store("clip", second="dinov2" if DINO else None, need=("global", "aug", "regions"))
CANDS = {
    "train-only-graph+dinov2-cache": dict(test_graph=False, second=True),
    "full+dinov2-cache":             dict(second=True),
} if DINO else {
    "full":                    dict(),
    "train-only-graph":        dict(test_graph=False),
    "no-geometry":             dict(use_geom=False),
    "spots-only":              dict(keep_kinds=(3,)),
    "train-only+no-geometry":  dict(test_graph=False, use_geom=False),
    "full-H128":               dict(H=128),
    "train-only-graph-H128":   dict(test_graph=False, H=128),
}
rows = []
for name, cfg in CANDS.items():
    r = {"candidate": name, "cfg": str(cfg)}
    for K in ("1", "4", "16"):
        v = []
        for s in (1, 2, 3):
            seed_all(s)
            tr = store.batch(pd.read_csv(SPLITS / f"support_K{K}_s{s}.csv"), with_aug=True)
            va = store.batch(pd.read_csv(SPLITS / f"val_K{K}_s{s}.csv"))
            v.append(PRGA(T, **cfg).fit(tr, va).val_f1)
        r[f"val_K{K}"] = round(sum(v) / 3, 4)
    r["val_mean"] = round((r["val_K1"] + r["val_K4"] + r["val_K16"]) / 3, 4)
    rows.append(r)
    print(r, flush=True)
df = pd.DataFrame(rows).sort_values("val_mean", ascending=False)
df.to_csv(RESULTS / ("prga_selection_val_dino.csv" if DINO else "prga_selection_val.csv"), index=False)
print("SELECTED:", df.iloc[0]["candidate"])
