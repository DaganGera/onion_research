"""Step 4.3 — can PRGA be improved, judged on VALIDATION only? Candidate: a complementary DINOv2 cache.

Motivation (general, not onion-specific): CLIP is trained on image-text pairs and DINOv2 by self-supervision on images
only; their errors are partly independent, which is why multi-backbone caches (CaFo, CVPR 2023; PlantCaFo, Plant
Phenomics 2025) help. In our table the DINOv2 cache is exactly what separates PlantCaFo-lite from Tip-Adapter-F.

  A  PRGA (validation-selected configuration, unchanged)
  B  A + training-free DINOv2 cache:   logits_A + a2 * exp(-b2 (1 - q2 K2^T)) L_bal
  C  A + DINOv2 cache with keys fine-tuned Tip-Adapter-F style (20 epochs on the 10 augmented views)

Protocol (no test-set information used for any decision):
  - each validation set is split BY ONION COMMUNITY into half T (tune a2, b2) and half S (score);
  - the variant is chosen ONCE, globally: highest mean macro-F1 on the S halves over all K x seeds;
  - only then is the chosen variant run on the test set (a2, b2 re-tuned on the whole validation set, as for every
    other method). Outputs: results/prga_improve_val.csv, results/runs/PRGA2B-clip.jsonl (if chosen)
"""
import itertools
import json

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from common import DEVICE, RESULTS, SEEDS, SPLITS, l2n, metrics, seed_all
from fewshot import PRGA, Store, balance, macro_f1, onehot
from harness import ALL_K, load_text

A2 = (0, 0.5, 1, 2, 4, 8, 16, 32)
B2 = (1, 3, 5, 7, 9)
CHOSEN = ("train-only+no-geometry", dict(test_graph=False, use_geom=False))     # 07b_prga_select.py, validation


@torch.no_grad()
def prga_logits(m, b):
    n, mk, gm = [x.to(DEVICE) for x in m._nodes(b)]
    g = b.g.to(DEVICE)
    fq, Tq = m._embed(g, n, mk, gm)
    if not m.test_graph:
        fq = g
    return m._logits(g, fq, Tq, m.keys, m.L, m.s).cpu()


def dino_keys(tr, finetune, epochs=20):
    k = l2n(tr.aug2.mean(1))
    if not finetune:
        return k
    keys = nn.Parameter(k.clone().to(DEVICE))
    X, Y, L = tr.aug2.to(DEVICE), tr.y.to(DEVICE), balance(onehot(tr.y)).to(DEVICE)
    opt = torch.optim.AdamW([keys], lr=1e-3, eps=1e-4)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs * X.shape[1])
    for _ in range(epochs):
        for v in torch.randperm(X.shape[1]):
            loss = F.cross_entropy(8.0 * torch.exp(-5.0 * (1 - X[:, v] @ keys.T)) @ L, Y)
            opt.zero_grad(); loss.backward(); opt.step(); sch.step()
    return keys.detach().cpu()


def fused(base, q2, k2, L, a2, b2):
    return base + a2 * torch.exp(-b2 * (1 - q2 @ k2.T)) @ balance(L)


def tune(base, q2, k2, L, y):
    return max(itertools.product(A2, B2), key=lambda ab: macro_f1(fused(base, q2, k2, L, *ab), y))


def main():
    store = Store("clip", second="dinov2", need=("global", "aug", "regions"))
    T = load_text("clip", "desc")
    rows = []
    for K in ALL_K:
        for s in SEEDS:
            seed_all(s)
            tr = store.batch(pd.read_csv(SPLITS / f"support_K{K}_s{s}.csv"), with_aug=True)
            vdf = pd.read_csv(SPLITS / f"val_K{K}_s{s}.csv")
            va = store.batch(vdf)
            m = PRGA(T, **CHOSEN[1]).fit(tr, va)
            base_v = prga_logits(m, va)
            L = onehot(tr.y)
            rng = np.random.default_rng(100 + s)
            comm = vdf.group.unique()
            half = set(rng.choice(comm, size=len(comm) // 2, replace=False))
            iT = torch.tensor(vdf.group.isin(half).values)
            iS = ~iT
            r = dict(K=K, seed=s, n_tune=int(iT.sum()), n_score=int(iS.sum()),
                     A=macro_f1(base_v[iS], va.y[iS]))
            for name, ft in (("B", False), ("C", True)):
                k2 = dino_keys(tr, ft)
                a2, b2 = tune(base_v[iT], va.g2[iT], k2, L, va.y[iT])
                r[name] = macro_f1(fused(base_v[iS], va.g2[iS], k2, L, a2, b2), va.y[iS])
                r[f"{name}_a2"], r[f"{name}_b2"] = a2, b2
            rows.append(r)
            print({k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()}, flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(RESULTS / "prga_improve_val.csv", index=False)
    means = df[["A", "B", "C"]].mean()
    print("\nmean macro-F1 on the held-out validation halves (S):\n", means.round(4).to_string())
    print(df.groupby("K")[["A", "B", "C"]].mean().round(4).to_string())
    pick = means.idxmax()
    print("CHOSEN on validation:", pick, flush=True)
    if pick == "A":
        print("No improvement on validation: PRGA stays unchanged; the test set is not touched.")
        return

    # ---- test, once, for the chosen variant (a2, b2 re-tuned on the whole validation set) ----
    test = store.batch(pd.read_csv(SPLITS / "test80.csv"))
    name = "PRGA2B-clip" if pick == "B" else "PRGA2Bft-clip"
    ref = "PRGA-clip-samefeats"            # plain PRGA on the SAME local features: the before/after comparison
    logs = {n: RESULTS / "runs" / f"{n}_improve.jsonl" for n in (name, ref)}
    for f in logs.values():
        f.unlink(missing_ok=True)
    for K in ALL_K:
        for s in SEEDS:
            seed_all(s)
            tr = store.batch(pd.read_csv(SPLITS / f"support_K{K}_s{s}.csv"), with_aug=True)
            va = store.batch(pd.read_csv(SPLITS / f"val_K{K}_s{s}.csv"))
            m = PRGA(T, **CHOSEN[1]).fit(tr, va)
            k2 = dino_keys(tr, pick == "C")
            L = onehot(tr.y)
            a2, b2 = tune(prga_logits(m, va), va.g2, k2, L, va.y)
            base_t = prga_logits(m, test)
            for n_, probs, extra in ((ref, base_t.softmax(1), {}),
                                     (name, fused(base_t, test.g2, k2, L, a2, b2).softmax(1), dict(a2=a2, b2=b2))):
                r = dict(method=n_, K=K, seed=s, n_support=len(tr), **extra,
                         **metrics(test.y.numpy(), probs.argmax(1).numpy()))
                np.save(RESULTS / "preds" / f"{n_}_K{K}_s{s}.npy", probs.numpy().astype(np.float16))
                with open(logs[n_], "a") as f:
                    f.write(json.dumps(r) + "\n")
                print(f"{n_:22s} K={K} s={s} {extra} test macroF1={r['macro_f1']:.4f}", flush=True)


if __name__ == "__main__":
    main()
