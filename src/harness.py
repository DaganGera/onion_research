"""Runs a few-shot method over K x seeds on the fixed test set and stores per-run metrics + predictions.
Resumable: every finished (K, seed) run is appended to results/runs/<name>.jsonl and skipped on a rerun."""
import json
import time

import numpy as np
import pandas as pd

from common import RESULTS, SEEDS, SPLITS, load_feats, metrics, seed_all

ALL_K = ["1", "2", "4", "8", "16", "full"]


def load_text(backbone="clip", kind="desc"):
    return load_feats(f"{backbone}_text_{kind}")["T"].float()


def run(name, make, store, shots=ALL_K, seeds=SEEDS, test_csv="test80.csv", save_preds=True, tag="", pass_k=False):
    test_df = pd.read_csv(SPLITS / test_csv)
    te = store.batch(test_df)
    log = RESULTS / "runs" / f"{name}{tag}.jsonl"
    done = {}
    if log.exists():
        for line in log.read_text().splitlines():
            if line.strip():
                r = json.loads(line)
                done[(str(r["K"]), int(r["seed"]))] = r
    rows = []
    for K in shots:
        for s in seeds:
            if (str(K), s) in done:
                rows.append(done[(str(K), s)])
                continue
            seed_all(s)
            tr = store.batch(pd.read_csv(SPLITS / f"support_K{K}_s{s}.csv"), with_aug=True)
            va = store.batch(pd.read_csv(SPLITS / f"val_K{K}_s{s}.csv"))
            t0 = time.time()
            m = (make(K) if pass_k else make()).fit(tr, va)
            fit_s = time.time() - t0
            probs = m.predict(te)
            r = dict(method=name, K=K, seed=s, n_support=len(tr), fit_s=round(fit_s, 2),
                     **metrics(te.y.numpy(), probs.argmax(1).numpy()))
            if hasattr(m, "n_params"):
                r["params"] = m.n_params()
            rows.append(r)
            print(f"{name:28s} K={K:>4s} seed={s}  acc={r['acc']:.4f}  macroF1={r['macro_f1']:.4f}  ({fit_s:.1f}s)",
                  flush=True)
            if save_preds:
                np.save(RESULTS / "preds" / f"{name}{tag}_K{K}_s{s}.npy", probs.numpy().astype(np.float16))
            with open(log, "a") as f:
                f.write(json.dumps(r) + "\n")
    df = pd.DataFrame(rows)
    df.to_csv(RESULTS / "runs" / f"{name}{tag}.csv", index=False)
    return df


(RESULTS / "preds").mkdir(exist_ok=True, parents=True)
(RESULTS / "runs").mkdir(exist_ok=True, parents=True)
