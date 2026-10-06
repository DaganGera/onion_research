"""Bee domain pilot: try many cheap fixes on a small slice before any full run (DATASET=bees, Kaggle).

Why: on bees every frozen-feature method stays near 0.6 macro-F1, and validation (same videos as the support photos)
is ~15 points above test (other videos). Two suspected causes, each with fixes that leave PRGA and the caches as they
are:
  1. the mite is ~20 px in a 224 px input, one ViT patch:   -> mite-scale tiles (18 tiles of 80x80 px per bee) as
                                                               PRGA's region nodes; a mite prototype from support boxes
  2. features are dominated by the video (glass, light):     -> per-video centring: subtract each video's mean
                                                               feature, computed from unlabelled frames
  3. maybe the backbone:                                      -> CLIP B/16, CLIP L/14, SigLIP B/16, BioCLIP; DINOv2
                                                               small or base as the second cache

Slice: K = 4 and 16, seed 1. Support photos come from 3 pool videos and validation from the 3 other pool videos, so
validation measures the same cross-video transfer the test set needs. Test = 2,000 random test photos (for reference
only; decisions use validation). Output: results/pilot.csv
"""
import ast
import importlib
import itertools
import json
import os
import time

import numpy as np
import open_clip
import pandas as pd
import torch
import torchvision.transforms as T
from PIL import Image

from common import CLASSES, DATASET, DEVICE, PROMPTS, RAW, RESULTS, SPLITS, l2n, load_meta, metrics, pad_square, seed_all
from fewshot import PRGA, Batch, PlantCaFoLite, TipAdapterF, onehot

assert DATASET == "bees"
dino = importlib.import_module("13_prga_dinov2")
SUPPORT_VIDEOS = ["2017-09-25_16-03-38", "2017-09-29_15-31-49", "2017-08-28_16-31-55"]
TILE, STRIDE, N_AUG = 80, 40, 10
CLIPS = {"clip_b16": ("ViT-B-16-quickgelu", "openai"), "clip_l14": ("ViT-L-14-quickgelu", "openai"),
         "siglip_b16": ("ViT-B-16-SigLIP", "webli"), "bioclip": ("hf-hub:imageomics/bioclip", None)}
DINOS = {"dinov2_s": "facebook/dinov2-small", "dinov2_b": "facebook/dinov2-base"}
ONLY = os.environ.get("PILOT_BACKBONES")       # optional: comma-separated subset


def tiles(im):
    w, h = im.size
    return [im.crop((x, y, x + TILE, y + TILE)) for y in range(0, h - TILE + 1, STRIDE)
            for x in range(0, w - TILE + 1, STRIDE)]


def mite_crop(im, b):
    x0, y0, x1, y1 = b
    m = max(x1 - x0, y1 - y0) * 0.5
    return im.crop((max(0, x0 - m), max(0, y0 - m), min(im.width, x1 + m), min(im.height, y1 + m)))


class Encoder:
    def __init__(self, name):
        self.name, self.text = name, name in CLIPS
        if self.text:
            arch, pre = CLIPS[name]
            self.model, _, prep = open_clip.create_model_and_transforms(arch, pretrained=pre)
            self.tok = open_clip.get_tokenizer(arch)
            size = prep.transforms[0].size
            size = size if isinstance(size, int) else size[0]
            norm = prep.transforms[-1]
        else:
            from transformers import AutoModel
            self.model, size = AutoModel.from_pretrained(DINOS[name]), 224
            norm = T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        self.model = self.model.to(DEVICE).eval().half()
        self.tf = T.Compose([T.Resize((size, size), interpolation=T.InterpolationMode.BICUBIC), T.ToTensor(), norm])
        self.aug = T.Compose([T.RandomResizedCrop(size, scale=(0.5, 1.0), interpolation=T.InterpolationMode.BICUBIC),
                              T.RandomHorizontalFlip(), T.ToTensor(), norm])

    @torch.no_grad()
    def enc(self, ims, tf=None, bs=256):
        out = []
        for i in range(0, len(ims), bs):
            x = torch.stack([(tf or self.tf)(im) for im in ims[i:i + bs]]).to(DEVICE).half()
            f = self.model.encode_image(x) if self.text else self.model(pixel_values=x).pooler_output
            out.append(l2n(f.float()).cpu())
        return torch.cat(out)

    @torch.no_grad()
    def class_text(self, classes):
        desc, temps = json.load(open(PROMPTS / "descriptors.json")), json.load(open(PROMPTS / "templates.json"))
        e = lambda t: l2n(self.model.encode_text(self.tok(t).to(DEVICE)).float()).cpu()  # noqa: E731
        tm = torch.stack([l2n(e([t.format("a " + c) for t in temps]).mean(0)) for c in classes])
        return l2n(torch.stack([l2n(e(desc[c]).mean(0)) for c in classes]) + 0.5 * tm)


def main():
    meta = load_meta().set_index("path")
    pool, test = pd.read_csv(SPLITS / "pool20.csv"), pd.read_csv(SPLITS / "test80.csv")
    frac = min(1.0, 2000 / len(test))
    test = pd.concat([test[test.label == c].sample(frac=frac, random_state=0) for c in (0, 1)])
    val = pool[~pool.video.isin(SUPPORT_VIDEOS)]
    sup_pool = pool[pool.video.isin(SUPPORT_VIDEOS)]
    supports = {}
    for K in (4, 16):
        rng = np.random.default_rng(1000 * K + 1)
        supports[K] = pd.concat([sup_pool[sup_pool.label == c].iloc[rng.choice((sup_pool.label == c).sum(), K, replace=False)]
                                 for c in (0, 1)])
    sup_all = pd.concat(supports.values()).drop_duplicates("path")
    # unlabelled frames per video for the centring means (labels never used)
    bg = pd.concat([d.sample(min(len(d), 150), random_state=0) for _, d in meta.reset_index().groupby("video")])
    paths = sorted(set(val.path) | set(test.path) | set(sup_all.path) | set(bg.path))
    print(f"support videos {SUPPORT_VIDEOS}; val {len(val)} photos from {val.video.nunique()} other pool videos; "
          f"test {len(test)}; images to encode {len(paths)}", flush=True)
    ims = {p: Image.open(RAW / p).convert("RGB") for p in paths}
    video = meta.video.to_dict()

    F = {}
    names = [n for n in list(CLIPS) + list(DINOS) if not ONLY or n in ONLY.split(",")]
    for name in names:
        t0 = time.time()
        e = Encoder(name)
        sq = [pad_square(ims[p]) for p in paths]
        g = e.enc(sq)
        tl = e.enc([t for p in paths for t in tiles(ims[p])]).view(len(paths), -1, g.shape[1])
        seed_all(0)
        sp = sorted(sup_all.path)
        aug = e.enc([pad_square(ims[p]) for p in sp for _ in range(N_AUG)], tf=e.aug).view(len(sp), N_AUG, -1)
        mites = {}
        for p in sp:
            b = ast.literal_eval(meta.loc[p, "mite_boxes"])
            if meta.loc[p, "label"] == 1 and b:
                mites[p] = e.enc([mite_crop(ims[p], x) for x in b])
        F[name] = dict(row={p: i for i, p in enumerate(paths)}, g=g, tiles=tl, aug_row={p: i for i, p in enumerate(sp)},
                       aug=aug, mites=mites, T=e.class_text(CLASSES) if e.text else None)
        print(f"{name}: dim {g.shape[1]}, {tl.shape[1]} tiles, {time.time() - t0:.0f}s", flush=True)
        del e
        torch.cuda.empty_cache()

    cache = {}

    def centred(name, centre):
        """features, optionally minus the per-video mean (from the unlabelled frames of that video).
        Returns global, tiles, support views, mite crops."""
        if (name, centre) in cache:
            return cache[name, centre]
        f = F[name]
        if not centre:
            cache[name, centre] = f["g"], f["tiles"], f["aug"], f["mites"]
            return cache[name, centre]
        vids = np.array([video[p] for p in f["row"]])
        g, tl = f["g"].clone(), f["tiles"].clone()
        mg, mt = {}, {}
        for v, d in bg.groupby("video"):
            ix = [f["row"][p] for p in d.path]
            mg[v], mt[v] = f["g"][ix].mean(0), f["tiles"][ix].mean((0, 1))
            sel = torch.tensor(np.where(vids == v)[0])
            g[sel] = l2n(g[sel] - mg[v])
            tl[sel] = l2n(tl[sel] - mt[v])
        aug = torch.stack([l2n(f["aug"][i] - mg[video[p]]) for p, i in f["aug_row"].items()])
        mites = {p: l2n(m - mt[video[p]]) for p, m in f["mites"].items()}
        cache[name, centre] = g, tl, aug, mites
        return cache[name, centre]

    def batch(name, centre, df, with_aug=False, second=None):
        g, tl, aug, _ = centred(name, centre)
        ix = torch.tensor([F[name]["row"][p] for p in df.path])
        b = Batch(y=torch.tensor(df.label.values), g=g[ix], grid=tl[ix])
        if with_aug:
            b.aug = aug[torch.tensor([F[name]["aug_row"][p] for p in df.path])]
        if second:
            g2, _, aug2, _ = centred(second, centre)
            b.g2 = g2[torch.tensor([F[second]["row"][p] for p in df.path])]
            if with_aug:
                b.aug2 = aug2[torch.tensor([F[second]["aug_row"][p] for p in df.path])]
        return b

    def mite_proto(name, centre, tr_df):
        """box-supervised: score = max over tiles of (sim to a support mite crop - sim to healthy support tiles)."""
        f = F[name]
        _, tl, _, mites = centred(name, centre)
        M = torch.cat([mites[p] for p in tr_df.path if p in mites])
        H = tl[torch.tensor([f["row"][p] for p in tr_df[tr_df.label == 0].path])].flatten(0, 1)

        def score(df):
            t = tl[torch.tensor([f["row"][p] for p in df.path])]
            return ((t @ M.T).max(-1).values - (t @ H.T).max(-1).values).max(-1).values

        sv, st = score(va_df), score(te_df)
        thr = max(np.quantile(sv.numpy(), np.linspace(0.02, 0.98, 97)),
                  key=lambda q: metrics(va_df.label.values, (sv > q).long().numpy())["macro_f1"])
        return (metrics(va_df.label.values, (sv > thr).long().numpy())["macro_f1"],
                metrics(te_df.label.values, (st > thr).long().numpy())["macro_f1"])

    rows = []
    text_bbs = [n for n in names if n in CLIPS]
    dinos = [n for n in names if n in DINOS]
    va_df, te_df = val, test
    for name, centre, K in itertools.product(text_bbs, (False, True), (4, 16)):
        tr_df = supports[K]
        Tn = F[name]["T"]
        tr, va, te = (batch(name, centre, tr_df, True), batch(name, centre, va_df), batch(name, centre, te_df))

        def log(method, pv, pt, **kw):
            r = dict(backbone=name, centre=centre, K=K, method=method,
                     val_f1=round(metrics(va.y.numpy(), pv.argmax(1).numpy())["macro_f1"] if pv is not None else kw.pop("vf"), 4),
                     test_f1=round(metrics(te.y.numpy(), pt.argmax(1).numpy())["macro_f1"] if pt is not None else kw.pop("tf"), 4), **kw)
            rows.append(r)
            print(r, flush=True)

        seed_all(1)
        m = TipAdapterF(Tn).fit(tr, va)
        log("Tip-Adapter-F", m.predict(va), m.predict(te))
        seed_all(1)
        m = PRGA(Tn, nodes="grid", **dino.CHOSEN[1]).fit(tr, va)
        bv, bt = dino.prga_logits(m, va), dino.prga_logits(m, te)
        log("PRGA (tile nodes)", bv, bt)
        vf, tf_ = mite_proto(name, centre, tr_df)
        log("Mite prototype (support boxes)", None, None, vf=vf, tf=tf_)
        for d in dinos:
            tr2, va2, te2 = (batch(name, centre, tr_df, True, d), batch(name, centre, va_df, second=d),
                             batch(name, centre, te_df, second=d))
            seed_all(1)
            m2 = PlantCaFoLite(Tn).fit(tr2, va2)
            log(f"PlantCaFo-style + {d}", m2.predict(va2), m2.predict(te2))
            k2, L = dino.dino_keys(tr2, finetune=True), onehot(tr2.y)
            a2, b2 = dino.tune(bv, va2.g2, k2, L, va2.y)
            log(f"PRGA (tile nodes) + {d} cache", dino.fused(bv, va2.g2, k2, L, a2, b2),
                dino.fused(bt, te2.g2, k2, L, a2, b2))
        pd.DataFrame(rows).to_csv(RESULTS / "pilot.csv", index=False)
    df = pd.DataFrame(rows)
    print("\n=== mean over K=4 and K=16 (validation decides; test for reference) ===")
    print(df.groupby(["backbone", "centre", "method"])[["val_f1", "test_f1"]].mean().sort_values("val_f1", ascending=False)
          .round(3).to_string())


if __name__ == "__main__":
    main()
