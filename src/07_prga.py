"""Step 3.2 / 4.1 / 4.2: PRGA (ours) — main sweep and ablations.

  python 07_prga.py                         final PRGA (config chosen on VALIDATION by results/prga_select.py)
  python 07_prga.py --v0                    the first, un-selected configuration (kept for transparency)
  python 07_prga.py --ablations --shots 1 4 ablation table
"""
import ast
import argparse

import pandas as pd

from common import RESULTS
from fewshot import PRGA, PlantCaFoLite, Store
from harness import ALL_K, load_text, run

ABLATIONS = {
    # name                     kwargs for PRGA                         text features
    "PRGA-full":               (dict(), "desc"),
    "abl-grid-nodes":          (dict(nodes="grid"), "desc"),
    "abl-no-text-nodes":       (dict(text_nodes=False), "desc"),
    "abl-template-text":       (dict(), "template"),
    "abl-no-gate":             (dict(gate=False), "desc"),
    "abl-no-geometry":         (dict(use_geom=False), "desc"),
    "abl-M0-object-only":      (dict(M=0), "desc"),
    "abl-instances-only":      (dict(keep_kinds=(2,)), "desc"),
    "abl-spots-only":          (dict(keep_kinds=(3,)), "desc"),
    "abl-train-only-graph":    (dict(test_graph=False), "desc"),
}


def selected_config():
    """Best candidate by mean validation macro-F1 over K in {1, 4, 16} (never the test set)."""
    frames = [pd.read_csv(p) for p in (RESULTS / "prga_selection_val.csv", RESULTS / "prga_selection_val_dino.csv")
              if p.exists()]
    if not frames:
        return "full", {}
    best = pd.concat(frames).sort_values("val_mean", ascending=False).iloc[0]
    return best["candidate"], ast.literal_eval(best["cfg"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ablations", action="store_true")
    ap.add_argument("--shots", nargs="+", default=ALL_K)
    ap.add_argument("--only", nargs="*", default=None)
    ap.add_argument("--v0", action="store_true")
    a = ap.parse_args()
    if not a.ablations:
        T = load_text("clip", "desc")
        if a.v0:
            store = Store("clip", need=("global", "aug", "regions"))
            run("PRGA-v0-clip", lambda: PRGA(T), store, shots=a.shots)
            return
        name, cfg = selected_config()
        print("final PRGA configuration (selected on validation):", name, cfg)
        store = Store("clip", second="dinov2" if cfg.get("second") else None, need=("global", "aug", "regions"))
        run("PRGA-clip", lambda: PRGA(T, **cfg), store, shots=a.shots)
        return
    store = Store("clip", need=("global", "aug", "grid", "regions"))
    rows = []
    for name, (kw, text) in ABLATIONS.items():
        if a.only and name not in a.only:
            continue
        T = load_text("clip", text)
        df = run(name, lambda: PRGA(T, **kw), store, shots=a.shots, save_preds=False, tag="_abl")
        rows.append(df)
    out = pd.concat(rows)
    summ = out.groupby(["method", "K"])[["acc", "macro_f1"]].agg(["mean", "std"]).round(4)
    summ.to_csv(RESULTS / f"ablations_K{'-'.join(a.shots)}.csv")
    print(summ.to_string())


if __name__ == "__main__":
    main()
