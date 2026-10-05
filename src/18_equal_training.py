"""Is the comparison fair when every method trains for a different number of epochs?

PRGA keeps its best checkpoint on validation (checked every 5 of 60 epochs), and the base paper's model its best epoch.
The other trained methods used their last epoch. This script gives every trained comparison method the same
treatment and re-runs everything on one GPU, so hardware differences cannot explain a gap:

  last-epoch   the method exactly as in the main table (its usual number of epochs, last epoch)
  best-epoch   twice its usual number of epochs; validation macro-F1 is measured at 10 evenly spaced epochs and the
               best checkpoint is kept (on top of each method's usual hyper-parameter search). Test set used once.
  PRGA and PRGA + DINOv2 are re-run unchanged (they already select their checkpoint on validation).

  python 18_equal_training.py [--methods tipf clipadapter taskres clap graphadapter cafo prga coop] [--shots ...]
                              [--seeds ...] [--backbone clip|bioclip]
Output: results/equal_training/<method>.jsonl (resumable), results/equal_training.csv / .md
"""
import argparse
import importlib
import json
import time

import pandas as pd

from adapters import CLAP, CLIPAdapter, CoOp, GraphAdapter, TaskRes
from common import RESULTS, SEEDS, SPLITS, metrics, seed_all
from fewshot import PRGA, PlantCaFoLite, Store, TipAdapterF
from harness import ALL_K, load_text

dino = importlib.import_module("13_prga_dinov2")
OUT = RESULTS / "equal_training"
OUT.mkdir(parents=True, exist_ok=True)
LABEL = {"tipf": "Tip-Adapter-F", "clipadapter": "CLIP-Adapter", "taskres": "TaskRes", "clap": "CLAP",
         "graphadapter": "GraphAdapter", "cafo": "PlantCaFo-style cache", "coop": "CoOp",
         "prga": "PRGA", "prga_dino": "PRGA + DINOv2 cache"}


def make(method, K, T):
    return {"tipf": lambda: TipAdapterF(T), "clipadapter": lambda: CLIPAdapter(T), "taskres": lambda: TaskRes(T),
            "clap": lambda: CLAP(T), "graphadapter": lambda: GraphAdapter(T), "cafo": lambda: PlantCaFoLite(T),
            "coop": lambda: CoOp(K)}[method]()


def val_f1(va, scores):
    """validation macro-F1 (used to choose the backbone, never the test set). The PRGA + DINOv2 a2/b2 are tuned on
    this same validation set, so its value is optimistic; compare backbones within one method."""
    return metrics(va.y.numpy(), scores.argmax(1).numpy())["macro_f1"] if len(va) else float("nan")


def done_runs(log):
    if not log.exists():
        return set()
    return {(r["method"], r["variant"], str(r["K"]), r["seed"]) for r in map(json.loads, log.read_text().splitlines())}


def write(log, **r):
    with open(log, "a") as f:
        f.write(json.dumps(r) + "\n")
    print(f"{r['method']:22s} {r['variant']:10s} K={r['K']:>4s} s={r['seed']} epochs={r['epochs']:>4} "
          f"best={r['best_epoch']:>4}  macroF1={r['macro_f1']:.4f}  ({r['fit_s']:.0f}s)", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--methods", nargs="+",
                    default=["tipf", "clipadapter", "taskres", "clap", "graphadapter", "cafo", "prga", "coop"])
    ap.add_argument("--shots", nargs="+", default=ALL_K)
    ap.add_argument("--seeds", nargs="+", type=int, default=SEEDS)
    ap.add_argument("--backbone", default="clip", choices=["clip", "bioclip"])
    a = ap.parse_args()
    T = load_text(a.backbone, "desc")
    store = Store(a.backbone, second="dinov2", need=("global", "aug", "regions"))
    tag = "" if a.backbone == "clip" else f" [{a.backbone}]"
    te = store.batch(pd.read_csv(SPLITS / "test80.csv"))
    y_te = te.y.numpy()

    for method in a.methods:
        log = OUT / f"{method}{'' if a.backbone == 'clip' else '_' + a.backbone}.jsonl"
        done = done_runs(log)
        for K in a.shots:
            for s in a.seeds:
                tr = store.batch(pd.read_csv(SPLITS / f"support_K{K}_s{s}.csv"), with_aug=True)
                va = store.batch(pd.read_csv(SPLITS / f"val_K{K}_s{s}.csv"))
                if method == "prga":
                    if ("PRGA" + tag, "as-is", K, s) in done:
                        continue
                    seed_all(s)
                    t0 = time.time()
                    m = PRGA(T, **dino.CHOSEN[1]).fit(tr, va)
                    base = dino.prga_logits(m, te)
                    k2 = dino.dino_keys(tr, finetune=True)
                    L = dino.onehot(tr.y)
                    base_v = dino.prga_logits(m, va)
                    a2, b2 = dino.tune(base_v, va.g2, k2, L, va.y)
                    fit_s = time.time() - t0
                    for name, logits, lv in (("PRGA" + tag, base, base_v),
                                             ("PRGA + DINOv2 cache" + tag, dino.fused(base, te.g2, k2, L, a2, b2),
                                              dino.fused(base_v, va.g2, k2, L, a2, b2))):
                        write(log, method=name, variant="as-is", K=K, seed=s, epochs=60, best_epoch=-1,
                              fit_s=fit_s, val_macro_f1=val_f1(va, lv), **metrics(y_te, logits.argmax(1).numpy()))
                    continue
                for variant in ("last-epoch", "best-epoch"):
                    if (LABEL[method] + tag, variant, K, s) in done:
                        continue
                    seed_all(s)
                    m = make(method, K, T)
                    if variant == "best-epoch":
                        m.epochs *= 2
                        m.pick_epoch = True
                    t0 = time.time()
                    m.fit(tr, va)
                    fit_s = time.time() - t0
                    pred = m.predict(te).argmax(1).numpy()
                    vf1 = val_f1(va, m.predict(va))
                    write(log, method=LABEL[method] + tag, variant=variant, K=K, seed=s, epochs=m.epochs,
                          best_epoch=getattr(m, "best_epoch", m.epochs) if variant == "best-epoch" else m.epochs,
                          fit_s=fit_s, val_macro_f1=vf1, **metrics(y_te, pred))
    summarise()


def summarise():
    rows = [json.loads(line) for f in sorted(OUT.glob("*.jsonl")) for line in f.read_text().splitlines()]
    df = pd.DataFrame(rows)
    df["K"] = pd.Categorical(df.K.astype(str), ALL_K, ordered=True)
    tab = df.groupby(["method", "variant", "K"], observed=True).macro_f1.agg(["mean", "std", "count"]).reset_index()
    tab.to_csv(RESULTS / "equal_training.csv", index=False)
    cell = tab.assign(v=tab["mean"].map("{:.3f}".format) + " ± " + tab["std"].map("{:.3f}".format))
    wide = cell.pivot_table(index=["method", "variant"], columns="K", values="v", aggfunc="first", observed=True)
    with open(RESULTS / "equal_training.md", "w") as f:
        f.write("Macro-F1 on the test set (mean ± std over 3 seeds), every method re-run on the same GPU.\n"
                "last-epoch = usual training; best-epoch = 2x epochs, best of 10 validation checkpoints.\n\n")
        f.write(wide.to_markdown() + "\n")
    print(wide.to_string())


if __name__ == "__main__":
    main()
