"""Bee domain, pictures: correctly detected test bees, Grad-CAM for every model, regions and ablations (DATASET=bees).

One setting is shown: CLIP ViT-L/14, K = 16, seed 1 (the full run's support and validation sets), on 400 random test
bees. Every model is trained here exactly as in 18_equal_training.py (usual epochs), on features computed here for
those photos only.

Grad-CAM for any model: every model's decision is a function of the test photo's CLIP vector g (PRGA's region nodes
and the DINOv2 cache are held fixed). We take w = d(log p_varroa - log p_healthy)/dg by central differences through
the trained model, then back-propagate g.w through the frozen CLIP image encoder to the input of its last block and form the usual
Grad-CAM map (ReLU of gradient x activation, summed over channels) on the 16 x 16 patch grid. "On mite" = share of
Varroa bees whose hottest patch lies inside an annotated mite box (pointing game).

Output: figures/bees_*.png, results/visual_summary.csv
"""
import ast
import importlib

import matplotlib
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw

from adapters import CLAP, CLIPAdapter, GraphAdapter, TaskRes
from backbones import Backbone
from common import (CLASSES, DATASET, DEVICE, FIGURES, PROMPTS, REGIONS, RESULTS, SPLITS, l2n, load_meta, metrics,
                    read_json, seed_all)
from fewshot import PRGA, Batch, PlantCaFoLite, TipAdapterF, ZeroShot, onehot, with_kind

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

assert DATASET == "bees"
feat = importlib.import_module("06_extract_features")
dino = importlib.import_module("13_prga_dinov2")
TEXT = importlib.import_module("07_text_embeddings")
K, SEED, N_TEST, N_AUG = "16", 1, 800, 10


def class_text(bb):
    desc = read_json(PROMPTS / "descriptors.json")
    T = torch.stack([l2n(bb.encode_text([t.format(TEXT.class_phrase(c)) for t in TEXT.TEMPLATES]).mean(0)) for c in CLASSES])
    D = torch.stack([l2n(bb.encode_text(desc[c]).mean(0)) for c in CLASSES])
    return l2n(D + 0.5 * T)


def encode(bb, paths, regions, with_aug):
    ims = [feat.open_rgb(p) for p in paths]
    g = torch.cat([bb.encode_image(torch.stack([bb.transform(im) for im in ims[i:i + 64]])) for i in range(0, len(ims), 64)])
    out = dict(g=g)
    if regions is not None:
        reg, mask, geom, kind = [], [], [], []
        for p, im in zip(paths, ims):
            crops, mk, gm, kd = feat.region_nodes(im, regions[p])
            reg.append(bb.encode_image(torch.stack([bb.transform(c) for c in crops])))
            mask.append(mk), geom.append(gm), kind.append(kd)
        out.update(reg=torch.stack(reg), mask=torch.stack(mask), geom=with_kind(torch.stack(geom), torch.stack(kind)))
    if with_aug:
        seed_all(0)
        out["aug"] = torch.stack([bb.encode_image(torch.stack([bb.aug_transform(im) for _ in range(N_AUG)])) for im in ims])
    return out


def make_batch(df, c, d):
    b = Batch(y=torch.tensor(df.label.values), g=c["g"], reg=c.get("reg"), mask=c.get("mask"), geom=c.get("geom"),
              g2=d["g"])
    if "aug" in c:
        b.aug, b.aug2 = c["aug"], d["aug"]
    return b


def sub(b, ix):
    return Batch(**{k: (v[ix] if torch.is_tensor(v) else v) for k, v in b.__dict__.items()})


def main():
    meta = load_meta().drop_duplicates("path").set_index("path")    # 2 files are listed twice in gt.csv
    regions = read_json(REGIONS / "regions.json")
    sup = pd.read_csv(SPLITS / f"support_K{K}_s{SEED}.csv")
    val = pd.read_csv(SPLITS / f"val_K{K}_s{SEED}.csv").sample(600, random_state=0)
    test = pd.read_csv(SPLITS / "test80.csv")
    # natural class ratio (75 % healthy), so macro-F1 is comparable with the results table
    test = test.drop_duplicates("path").sample(N_TEST, random_state=0).reset_index(drop=True)

    clip = Backbone("clip_l14")
    T = class_text(clip)
    C = {n: encode(clip, df.path.tolist(), regions, n == "sup") for n, df in (("sup", sup), ("val", val), ("test", test))}
    dv = Backbone("dinov2")
    Dn = {n: encode(dv, df.path.tolist(), None, n == "sup") for n, df in (("sup", sup), ("val", val), ("test", test))}
    del dv
    tr, va, te = (make_batch(sup, C["sup"], Dn["sup"]), make_batch(val, C["val"], Dn["val"]),
                  make_batch(test, C["test"], Dn["test"]))

    models = {}
    for name, make in (("Zero-shot CLIP", lambda: ZeroShot(T)), ("Tip-Adapter-F", lambda: TipAdapterF(T)),
                       ("PlantCaFo-style", lambda: PlantCaFoLite(T)), ("CLIP-Adapter", lambda: CLIPAdapter(T)),
                       ("TaskRes", lambda: TaskRes(T)), ("CLAP", lambda: CLAP(T)), ("GraphAdapter", lambda: GraphAdapter(T)),
                       ("PRGA", lambda: PRGA(T, **dino.CHOSEN[1]))):
        seed_all(SEED)
        m = make().fit(tr, va)
        models[name] = lambda b, m=m: m.predict(b).float().cpu()
        if name == "PRGA":
            k2, L = dino.dino_keys(tr, finetune=True), onehot(tr.y)
            a2, b2 = dino.tune(dino.prga_logits(m, va), va.g2, k2, L, va.y)
            models["PRGA + DINOv2"] = lambda b, m=m, k2=k2, L=L, a2=a2, b2=b2: dino.fused(
                dino.prga_logits(m, b), b.g2, k2, L, a2, b2).softmax(1).float().cpu()

    probs = {n: f(te) for n, f in models.items()}
    pred = pd.DataFrame({n: p.argmax(1).numpy() for n, p in probs.items()})
    summary = [dict(model=n, test_macro_f1=round(metrics(test.label.values, pred[n].values)["macro_f1"], 4),
                    says_varroa=round(float(pred[n].mean()), 3)) for n in models]
    print(f"test bees {len(test)}, of which Varroa {test.label.mean():.1%}", flush=True)
    print(pd.DataFrame(summary).to_string(), flush=True)
    FIGURES.mkdir(parents=True, exist_ok=True)

    def bee(p, box=True):
        im = feat.open_rgb(p).copy()
        if box:
            d = ImageDraw.Draw(im)
            for b in ast.literal_eval(meta.loc[p, "mite_boxes"]):
                d.rectangle(b, outline=(255, 255, 0), width=2)
        return im

    # 1. correctly detected test bees (by the final model), with every model's verdict
    final = "PRGA + DINOv2"
    ok = pred[final].values == test.label.values
    pick = ([i for i in np.where(ok & (test.label.values == 1))[0][:6]] +
            [i for i in np.where(ok & (test.label.values == 0))[0][:6]])
    fig, axes = plt.subplots(2, 6, figsize=(15, 8.5))
    for ax, i in zip(axes.flat, pick):
        ax.imshow(bee(test.path[i]))
        right = [n for n in models if pred[n][i] == test.label[i]]
        ax.set_title(f"{'VARROA' if test.label[i] else 'healthy'}  p={probs[final][i, 1]:.2f}\n"
                     f"{len(right)}/{len(models)} models right", fontsize=9)
        ax.axis("off")
    fig.suptitle(f"Test bees detected correctly by {final} (CLIP L/14, 16 photos per class). Yellow = annotated mite",
                 fontsize=11)
    fig.tight_layout()
    fig.savefig(FIGURES / "bees_correct_examples.png", dpi=110)
    plt.close(fig)

    # 2. which model gets which bee right (40 test bees)
    sel = np.r_[np.where(test.label.values == 1)[0][:20], np.where(test.label.values == 0)[0][:20]]
    grid = np.array([[pred[n][i] == test.label[i] for i in sel] for n in models], dtype=float)
    fig, ax = plt.subplots(figsize=(13, 4.2))
    ax.imshow(grid, cmap="RdYlGn", vmin=0, vmax=1, aspect="auto")
    ax.set_yticks(range(len(models)), [f"{n} ({s['test_macro_f1']:.2f})" for n, s in zip(models, summary)], fontsize=8)
    ax.set_xticks([9.5, 29.5], ["20 Varroa bees", "20 healthy bees"])
    ax.axvline(19.5, color="k")
    ax.set_title("Right (green) / wrong (red) per model and test bee; macro-F1 on 400 test bees in brackets", fontsize=10)
    fig.tight_layout()
    fig.savefig(FIGURES / "bees_model_agreement.png", dpi=110)
    plt.close(fig)

    # 3. Grad-CAM for every model
    vis = clip.model.visual.float()
    clip.model.float()
    acts = {}
    blk = vis.transformer.resblocks[-2]      # input of the last block: its patch tokens still feed the class token
    blk.register_forward_hook(lambda mod, i, o: acts.__setitem__("a", o))

    def direction(name, i, eps=1e-3):
        b = sub(te, [i])
        D = b.g.shape[1]
        rows = torch.cat([b.g + eps * torch.eye(D), b.g - eps * torch.eye(D)])
        bb_ = sub(te, [i] * (2 * D))
        bb_.g = rows
        p = models[name](bb_).clamp_min(1e-8).log()
        s = p[:, 1] - p[:, 0]
        return (s[:D] - s[D:]) / (2 * eps)

    def gradcam(i, w):
        x = clip.transform(feat.open_rgb(test.path[i])).unsqueeze(0).to(DEVICE).float().requires_grad_(True)
        g = l2n(clip.model.encode_image(x))
        a = acts["a"]
        a.retain_grad()
        (g[0] @ w.to(DEVICE)).backward()
        A, G = a, a.grad
        if A.shape[0] != 1:                     # (L, N, C) layout
            A, G = A.transpose(0, 1), G.transpose(0, 1)
        cam = F.relu((A[0, 1:] * G[0, 1:]).sum(-1))
        n = int(cam.numel() ** 0.5)
        return cam.view(n, n).detach().cpu().numpy()

    def to_bee(cam, p):
        """map the cam (on the padded square) back onto the 160 x 280 bee image."""
        im = feat.open_rgb(p)
        W, H = im.size
        S = max(W, H)
        big = np.array(Image.fromarray(cam / (cam.max() + 1e-8)).resize((S, S), Image.BILINEAR))
        x0, y0 = (S - W) // 2, (S - H) // 2
        return big[y0:y0 + H, x0:x0 + W]

    varroa = [i for i in range(len(test)) if test.label[i] == 1 and ast.literal_eval(meta.loc[test.path[i], "mite_boxes"])]
    show = [i for i in pick if test.label[i] == 1][:4]
    names = list(models)
    hits = {n: [] for n in names}
    cams = {}
    for i in varroa:
        boxes = ast.literal_eval(meta.loc[test.path[i], "mite_boxes"])
        for n in names:
            m = to_bee(gradcam(i, direction(n, i)), test.path[i])
            y, x = np.unravel_index(m.argmax(), m.shape)
            hits[n].append(any(b[0] <= x <= b[2] and b[1] <= y <= b[3] for b in boxes))
            if i in show:
                cams[i, n] = m
    for s in summary:
        s["gradcam_on_mite"] = round(float(np.mean(hits[s["model"]])), 3)
        s["n_varroa_with_box"] = len(varroa)
    pd.DataFrame(summary).to_csv(RESULTS / "visual_summary.csv", index=False)
    print(pd.DataFrame(summary).to_string(), flush=True)

    fig, axes = plt.subplots(len(show), len(names) + 1, figsize=(2.0 * (len(names) + 1), 3.6 * len(show)))
    for r, i in enumerate(show):
        axes[r, 0].imshow(bee(test.path[i]))
        axes[r, 0].set_title("bee + mite box", fontsize=8)
        for c, n in enumerate(names, 1):
            axes[r, c].imshow(bee(test.path[i]))
            axes[r, c].imshow(cams[i, n], cmap="jet", alpha=0.45)
            if r == 0:
                axes[r, c].set_title(f"{n}\non mite {np.mean(hits[n]):.0%}", fontsize=8)
    for ax in axes.flat:
        ax.axis("off")
    fig.suptitle("Grad-CAM of each model's Varroa decision through the frozen CLIP L/14 encoder "
                 f"('on mite' = hottest patch inside the mite box, {len(varroa)} test bees)", fontsize=10)
    fig.tight_layout()
    fig.savefig(FIGURES / "bees_gradcam_models.png", dpi=100)
    plt.close(fig)

    # 4. what PRGA's region nodes see: OWLv2 boxes vs the annotated mite
    fig, axes = plt.subplots(1, 8, figsize=(16, 4.8))
    colors = {"object": (0, 200, 0), "instance": (0, 120, 255), "spot": (255, 0, 0)}
    for ax, i in zip(axes, [i for i in varroa][:8]):
        im = bee(test.path[i])
        d = ImageDraw.Draw(im)
        r = regions[test.path[i]]
        if r.get("object"):
            d.rectangle(r["object"]["box"], outline=colors["object"], width=1)
        for x in r["regions"]:
            d.rectangle(x["box"], outline=colors[x["kind"]], width=2)
        ax.imshow(im)
        ax.axis("off")
    fig.suptitle("PRGA's region nodes from OWLv2: green = bee, blue = body part, red = 'spot'; yellow = annotated mite",
                 fontsize=10)
    fig.tight_layout()
    fig.savefig(FIGURES / "bees_regions.png", dpi=110)
    plt.close(fig)


if __name__ == "__main__":
    main()
