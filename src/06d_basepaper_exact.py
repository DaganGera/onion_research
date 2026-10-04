"""Step 6d (GPU) — train and evaluate the base paper's model exactly as described (basepaper.py), on the leak-free split.

Unlike the earlier re-implementations, the graph sees FRESH augmentations every step, as in the paper: every epoch each
support photo is randomly resized-cropped + flipped, resized to 336x336, cut into the 26 Fig.-2 windows, every window is
resized to 224x224 and encoded by the frozen CLIP. (Earlier versions used a few precomputed views, an approximation.)

The CLIP encodings of one step are shared by several heads (= model configurations), so ablations cost almost nothing:
  BasePaperExact               the paper: Fig.-2 windows, Attention1*Attention2, psi={mean,max,std}, one-hot cache,
                               W_c from "a photo of a [CLASS]"                                      <- main result
  BasePaperExact-bal           same, class-balanced cache values (needed for the unequal full pool)
  BasePaperExact-desc          same, W_c from our hand-written visual descriptors (what our other methods use)
  BasePaperExact-A1 / -A2      Table 3 of the paper: only Attention 1 / only Attention 2
  BasePaperExact-noEdges       Table 5 of the paper: no inter-patch edges
  BasePaperExact-aggFig1       psi = {min, max, softmax} as drawn in Fig. 1
  BasePaperExact-legacyWin     the earlier implementation's 26-window reading (4x4 grid only)
  BasePaperExact-rhoLastId     no ReLU after the LAST graph layer (GAT's output-layer convention): tests whether the
                               literal Eq.-1 ReLU pushes f_hat out of CLIP space (train query != test query)
  BasePaperExact-noGraph       CONTROL: identical recipe, but the training query is the plain whole-image window
                               feature = Tip-Adapter-F with fresh augmentations. Graph helps <=> BasePaperExact > this.
Epoch 0 (untrained keys) is a candidate in the epoch selection: best_epoch = 0 means training did not help on val.
Every head is trained with L in {1, 2} and (alpha, beta)_train in TRAIN_AB; L, (alpha, beta)_train, the epoch and the
test-time alpha/beta are chosen on validation only (the paper: alpha, beta 'tuned empirically'; L not given).

  python 06d_basepaper_exact.py [--shots 1 2 4 8 16 full] [--seeds 1 2 3] [--device cuda:0] [--tag gpu0]
Resumable: finished (K, seed) runs are read back from results/runs/<name>@<tag>.jsonl.
Outputs: results/runs/*.jsonl, results/preds/*.npy, checkpoints/BasePaperExact_Kfull_s1.pt
"""
import argparse
import itertools
import json
import time

import numpy as np
import open_clip
import pandas as pd
import torch
import torch.nn.functional as F
import torchvision.transforms as T
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from basepaper import ALL_WINDOWS, SIZE, Head
from common import CKPT, CLASSES, CLIP_NAME, NUM_WORKERS, RAW, RESULTS, SPLITS, l2n, load_feats, metrics, seed_all

MEAN = (0.48145466, 0.4578275, 0.40821073)
STD = (0.26862954, 0.26130258, 0.27577711)
AUG = T.Compose([T.RandomResizedCrop(SIZE, scale=(0.5, 1.0), interpolation=T.InterpolationMode.BICUBIC),
                 T.RandomHorizontalFlip(), T.PILToTensor()])
TRAIN_AB = [(1.0, 1.0), (10.0, 1.0), (10.0, 5.0)]     # alpha, beta used in training: UNSPECIFIED -> chosen on validation
ALPHAS = (0.1, 0.25, 0.5, 1, 2, 3, 5, 8, 12, 16, 24, 32, 48, 64, 96)             # validation grid (Fig. 5 searches alpha, beta)
BETAS = (0.5, 1, 2, 3, 4, 5, 7, 9)
CONFIGS = [  # name, cache, text, head kwargs
    ("BasePaperExact", "onehot", "paper", {}),
    ("BasePaperExact-bal", "balanced", "paper", {}),
    ("BasePaperExact-desc", "onehot", "desc", {}),
    ("BasePaperExact-A1", "onehot", "paper", {"mode": "a1"}),
    ("BasePaperExact-A2", "onehot", "paper", {"mode": "a2"}),
    ("BasePaperExact-noEdges", "onehot", "paper", {"mode": "noedge"}),
    ("BasePaperExact-aggFig1", "onehot", "paper", {"psi": ("min", "max", "softmax")}),
    ("BasePaperExact-legacyWin", "onehot", "paper", {"windows": "legacy"}),
    ("BasePaperExact-rhoLastId", "onehot", "paper", {"act_last": "identity"}),   # no ReLU after the last layer (GAT)
    ("BasePaperExact-noGraph", "onehot", "paper", {"query": "plain"}),           # control: same recipe, no graph
]


class Support(Dataset):
    def __init__(self, paths, labels):
        self.paths, self.labels = paths, labels

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        return AUG(Image.open(RAW / self.paths[i]).convert("RGB")), self.labels[i]


class WindowEncoder:
    """uint8 [B,3,336,336] -> L2-normalised CLIP features of every window [B, len(ALL_WINDOWS), 512]."""

    def __init__(self, device, chunk=int(__import__("os").environ.get("ONION_CHUNK", "24"))):
        self.dev, self.chunk = device, chunk
        self.model = open_clip.create_model_and_transforms(CLIP_NAME, pretrained="openai")[0].to(device).eval().half()
        self.tok = open_clip.get_tokenizer(CLIP_NAME)
        self.mean = torch.tensor(MEAN, device=device).view(1, 3, 1, 1)
        self.std = torch.tensor(STD, device=device).view(1, 3, 1, 1)

    @torch.no_grad()
    def __call__(self, x):
        out = []
        for xb in x.split(self.chunk):
            xb = (xb.to(self.dev, non_blocking=True).float() / 255 - self.mean) / self.std
            crops = torch.stack([F.interpolate(xb[:, :, t:b, l:r], size=224, mode="bicubic", align_corners=False,
                                               antialias=True) for t, l, b, r in ALL_WINDOWS], 1)
            f = self.model.encode_image(crops.flatten(0, 1).half()).float()
            out.append(l2n(f).view(len(xb), len(ALL_WINDOWS), -1))
        return torch.cat(out)

    @torch.no_grad()
    def text(self, template="a photo of a {}."):
        return l2n(self.model.encode_text(self.tok([template.format(c) for c in CLASSES]).to(self.dev)).float())


class Feats:
    def __init__(self):
        g, a = load_feats("clip_global"), load_feats("clip_aug")
        self.g, self.a = g["feats"].float(), a["feats"].float()
        self.g_row = {p: i for i, p in enumerate(g["paths"])}
        self.a_row = {p: i for i, p in enumerate(a["paths"])}

    def glob(self, df):
        return self.g[[self.g_row[p] for p in df.path]]

    def keys0(self, df):                       # Tip-Adapter cache: mean of 10 augmented views, L2-normalised
        return l2n(self.a[[self.a_row[p] for p in df.path]].mean(1))


def mf1(logits, y):
    return metrics(y.cpu().numpy(), logits.argmax(1).cpu().numpy())["macro_f1"]


def fit(K, seed, fs, enc, W_text, dev, epochs, bs):
    seed_all(seed)
    sup = pd.read_csv(SPLITS / f"support_K{K}_s{seed}.csv")
    val = pd.read_csv(SPLITS / f"val_K{K}_s{seed}.csv")
    y = torch.tensor(sup.label.values, device=dev)
    vg, vy = fs.glob(val).to(dev), torch.tensor(val.label.values, device=dev)
    k0 = fs.keys0(sup).to(dev)
    L1 = F.one_hot(y, len(CLASSES)).float()
    Ls = {"onehot": L1, "balanced": L1 / L1.sum(0, keepdim=True).clamp_min(1)}

    heads, opts = [], []
    for name, cache, text, kw in CONFIGS:
        for layers in ((1,) if kw.get("query") == "plain" else (1, 2)):
            for a0, b0 in TRAIN_AB:
                torch.manual_seed(seed)
                h = Head(name, k0, Ls[cache], W_text[text], layers, **kw).to(dev)
                h.layers_n, h.ab_train, h.keys0 = layers, (a0, b0), k0
                heads.append(h)
    loader = DataLoader(Support(sup.path.tolist(), sup.label.tolist()), batch_size=bs, shuffle=True,
                        num_workers=NUM_WORKERS, drop_last=False, persistent_workers=NUM_WORKERS > 0,
                        generator=torch.Generator().manual_seed(seed))
    steps = epochs * len(loader)
    for h in heads:
        opt = torch.optim.AdamW(h.parameters(), lr=1e-3, eps=1e-4)
        opts.append((opt, torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)))
        with torch.no_grad():                                       # epoch 0 = untrained keys (training-free cache)
            h.best, h.best_keys, h.best_ep = mf1(h.test_logits(vg, *h.ab_train), vy), h.keys.detach().clone(), 0

    for ep in range(epochs):
        for x, yb in loader:
            feats = enc(x)                                          # shared by all heads
            yb = yb.to(dev)
            for h, (opt, sch) in zip(heads, opts):
                h.train()
                loss = F.cross_entropy(h.train_logits(feats, *h.ab_train), yb)
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
                sch.step()
        with torch.no_grad():                                       # test-time path on validation: no graph
            for h in heads:
                s = mf1(h.test_logits(vg, *h.ab_train), vy)
                if s > h.best:
                    h.best, h.best_keys, h.best_ep = s, h.keys.detach().clone(), ep + 1

    with torch.no_grad():                                           # alpha, beta grid on validation (Fig. 5)
        for h in heads:
            h.a, h.b = max(itertools.product(ALPHAS, BETAS),
                           key=lambda ab: mf1(h.test_logits(vg, *ab, keys=h.best_keys), vy))
            h.val = mf1(h.test_logits(vg, h.a, h.b, keys=h.best_keys), vy)
    chosen = {}
    for name, *_ in CONFIGS:                                        # layers + training alpha/beta chosen on validation
        chosen[name] = max((h for h in heads if h.name == name), key=lambda h: h.val)
    return chosen, len(sup)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shots", nargs="+", default=["1", "2", "4", "8", "16", "full"])
    ap.add_argument("--seeds", nargs="+", type=int, default=[1, 2, 3])
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--bs", type=int, default=256)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--tag", default="gpu0")
    a = ap.parse_args()
    dev = a.device
    (RESULTS / "runs").mkdir(parents=True, exist_ok=True)
    (RESULTS / "preds").mkdir(parents=True, exist_ok=True)
    CKPT.mkdir(parents=True, exist_ok=True)
    fs = Feats()
    enc = WindowEncoder(dev)
    desc = load_feats("clip_text_desc")["T"].float().to(dev)
    W_text = {"paper": enc.text(), "desc": desc}
    test = pd.read_csv(SPLITS / "test80.csv")
    te_g, te_y = fs.glob(test).to(dev), test.label.values

    zs = metrics(te_y, (100 * te_g @ W_text["paper"].T).argmax(1).cpu().numpy())
    pd.DataFrame([dict(method="ZeroShot-clip-paperprompt", K="1", seed=1, **zs)]).to_csv(
        RESULTS / "runs" / "ZeroShot-clip-paperprompt.csv", index=False)
    print(f"zero-shot CLIP, 'a photo of a [CLASS]': macroF1={zs['macro_f1']:.4f}", flush=True)

    logs = {name: RESULTS / "runs" / f"{name}@{a.tag}.jsonl" for name, *_ in CONFIGS}
    done = set()
    for f in logs.values():
        if f.exists():
            done |= {(str(r["K"]), int(r["seed"]), r["method"]) for r in map(json.loads, f.read_text().splitlines())}
    for K in a.shots:
        for s in a.seeds:
            if all((K, s, n) in done for n, *_ in CONFIGS):
                continue
            t0 = time.time()
            chosen, n_sup = fit(K, s, fs, enc, W_text, dev, a.epochs, a.bs)
            fit_s = time.time() - t0
            for name, h in chosen.items():
                with torch.no_grad():
                    probs = h.test_logits(te_g, h.a, h.b, keys=h.best_keys).softmax(1).cpu()
                kd = float((F.normalize(h.best_keys, dim=1) * F.normalize(h.keys0, dim=1)).sum(1).mean())
                r = dict(method=name, K=K, seed=s, n_support=n_sup, fit_s=round(fit_s, 1), params=h.n_params(),
                         layers=h.layers_n, ab_train=list(h.ab_train), key_cos_to_start=round(kd, 5), best_epoch=h.best_ep, alpha=h.a, beta=h.b, val_macro_f1=round(h.val, 4),
                         **metrics(te_y, probs.argmax(1).numpy()))
                np.save(RESULTS / "preds" / f"{name}_K{K}_s{s}.npy", probs.numpy().astype(np.float16))
                with open(logs[name], "a") as f:
                    f.write(json.dumps(r) + "\n")
                print(f"{name:26s} K={K:>4s} s={s} L={h.layers_n} ep={h.best_ep:2d} a={h.a} b={h.b} "
                      f"val={h.val:.3f} test macroF1={r['macro_f1']:.4f}", flush=True)
                if name == "BasePaperExact" and K == "full" and s == 1:
                    torch.save(dict(state=h.state_dict(), layers=h.layers_n, alpha=h.a, beta=h.b,
                                    keys=h.best_keys.cpu()), CKPT / "BasePaperExact_Kfull_s1.pt")
            print(f"--- K={K} seed={s} done in {fit_s:.0f}s", flush=True)


if __name__ == "__main__":
    main()
