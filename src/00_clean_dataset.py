"""Step 0 — clean the bulb dataset BEFORE any split or training: integrity, duplicates, same-scene re-shots.

Why: the photographers shot each onion / pile many times, on the same cloth, sometimes in different sessions (files far
apart, e.g. Onion05365 and Onion05883 are the same onion). The earlier CLIP capture-session clustering missed such
re-shots, and some ended up on both sides of the split (visually confirmed). Global CLIP similarity cannot tell "same
object" from "similar scene" (single linkage on it chains 20-80 % of a class into one group), so re-shots are found with
local-feature GEOMETRIC VERIFICATION, which is specific to the same physical scene:

  1. integrity    every file decodes; size recorded
  2. exact dups   identical bytes (MD5): keep one file
  3. candidates   same-class pairs with CLIP cosine >= CAND (cheap recall filter; cross-class pairs >= CAND_X are also
                  checked, to find label conflicts)
  4. pre-filter   SIFT (RootSIFT, N_KP keypoints at 640 px) matched on the GPU for every candidate pair: Lowe ratio
                  test + mutual nearest neighbour; pairs with >= MIN_MATCH matches go on
  5. verify       RANSAC homography on the matched keypoints (OpenCV); a pair is the SAME SCENE if >= MIN_INLIERS inliers
  6. groups       union-find over verified pairs + exact duplicates + the previous capture-session groups
                  -> every re-shot of an object is in one group, so a group-wise split cannot leak it
  7. near-dups    inside a group, a photo that is a verified match with CLIP >= NEAR_DUP to an already kept photo is
                  REMOVED (burst shots: same view, adds no information, double-counts in the test set)
  8. conflicts    verified cross-class pairs are reported (and both photos removed) - the same object cannot be both classes

Outputs: data/clean/meta_clean.csv (kept photos + new `group`), data/clean/removed.csv (photo, reason, kept twin),
         data/clean/pairs_verified.csv, data/clean/report.md, figures/clean_examples_*.png
  python 00_clean_dataset.py              (GPU recommended for step 4; CPU fallback works, slower)
"""
import hashlib
import os
from concurrent.futures import ProcessPoolExecutor

import cv2
import numpy as np
import pandas as pd
import torch
from PIL import Image

from common import FEATS, RAW, ROOT, load_feats

OUT = ROOT / "data/clean"
CAND, CAND_X = 0.92, 0.94
N_KP, SIDE = 400, 640
RATIO, MIN_MATCH, MIN_INLIERS, RANSAC_PX = 0.8, 8, 10, 8.0
NEAR_DUP = 0.99
WORKERS = int(os.environ.get("ONION_CLEAN_WORKERS", "6"))
DEV = "cuda" if torch.cuda.is_available() else "cpu"
cv2.setNumThreads(1)                                    # one OpenCV thread per worker process (no oversubscription)


def _init_worker():
    cv2.setNumThreads(1)
    torch.set_num_threads(1)


def _md5_size(p):
    try:
        b = open(RAW / p, "rb").read()
        with Image.open(RAW / p) as im:
            im.verify()
        with Image.open(RAW / p) as im:
            w, h = im.size
        return hashlib.md5(b).hexdigest(), w, h, ""
    except Exception as e:                                   # noqa: BLE001
        return "", 0, 0, repr(e)


def _sift(p):
    im = cv2.imread(str(RAW / p), cv2.IMREAD_GRAYSCALE)
    s = SIDE / max(im.shape)
    im = cv2.resize(im, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
    kp, d = cv2.SIFT_create(nfeatures=N_KP).detectAndCompute(im, None)
    xy = np.zeros((N_KP, 2), np.float32)
    D = np.zeros((N_KP, 128), np.float32)
    n = 0 if d is None else min(len(kp), N_KP)
    if n:
        d = d[:n] / (np.abs(d[:n]).sum(1, keepdims=True) + 1e-7)              # RootSIFT: L1-normalise, sqrt
        D[:n] = np.sqrt(d)
        xy[:n] = np.array([k.pt for k in kp[:n]], np.float32)
    return xy, D.astype(np.float16), n


def _ransac(args):
    pa, pb = args
    if len(pa) < MIN_MATCH:
        return 0
    H, mask = cv2.findHomography(pa, pb, cv2.RANSAC, RANSAC_PX)
    return 0 if mask is None else int(mask.sum())


class UF:
    def __init__(self, n):
        self.p = np.arange(n)

    def find(self, i):
        while self.p[i] != i:
            self.p[i] = self.p[self.p[i]]
            i = self.p[i]
        return i

    def union(self, a, b):
        a, b = self.find(a), self.find(b)
        if a != b:
            self.p[max(a, b)] = min(a, b)


@torch.no_grad()
def gpu_matches(D, nkp, pairs, bs=1024):
    """Ratio-test + mutual-NN matches for every pair; returns counts and the matched index pairs."""
    Dt = torch.from_numpy(D).to(DEV)                                    # [N, K, 128] fp16
    valid = torch.arange(N_KP, device=DEV)[None] < torch.from_numpy(nkp).to(DEV)[:, None]
    counts = np.zeros(len(pairs), np.int32)
    idx_out = {}
    for s in range(0, len(pairs), bs):
        pr = torch.from_numpy(pairs[s:s + bs]).to(DEV)
        A, B = Dt[pr[:, 0]].float(), Dt[pr[:, 1]].float()
        va, vb = valid[pr[:, 0]], valid[pr[:, 1]]
        dist = (2 - 2 * A @ B.transpose(1, 2)).clamp_min(0).sqrt()     # RootSIFT vectors have unit L2 norm
        dist = dist.masked_fill(~va[:, :, None] | ~vb[:, None, :], 9.0)
        d2, j = dist.topk(2, dim=2, largest=False)                      # a -> b
        ok = d2[..., 0] < RATIO * d2[..., 1]
        back = dist.argmin(1)                                           # b -> a
        mutual = back.gather(1, j[..., 0]) == torch.arange(N_KP, device=DEV)[None]
        good = ok & mutual & va
        counts[s:s + len(pr)] = good.sum(1).cpu().numpy()
        for r in (good.sum(1) >= MIN_MATCH).nonzero().squeeze(1).tolist():
            ia = good[r].nonzero().squeeze(1)
            idx_out[s + r] = (ia.cpu().numpy(), j[r, ia, 0].cpu().numpy())
    return counts, idx_out


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    meta = pd.read_csv(ROOT / "data/meta.csv")
    n = len(meta)
    print(f"[1] integrity + MD5 for {n} photos", flush=True)
    with ProcessPoolExecutor(WORKERS, initializer=_init_worker) as ex:
        res = list(ex.map(_md5_size, meta.path, chunksize=64))
    meta["md5"] = [r[0] for r in res]
    meta["w"], meta["h"] = [r[1] for r in res], [r[2] for r in res]
    bad = [(p, r[3]) for p, r in zip(meta.path, res) if r[3]]
    print(f"    unreadable: {len(bad)}", flush=True)

    f = load_feats("clip_global")
    row = {p: i for i, p in enumerate(f["paths"])}
    X = f["feats"].float()[[row[p] for p in meta.path]].to(DEV)
    lab = torch.tensor(meta.label.values, device=DEV)

    print("[3] candidate pairs", flush=True)
    cand = []
    for i0 in range(0, n, 2048):
        S = X[i0:i0 + 2048] @ X.T
        I = torch.arange(i0, min(i0 + 2048, n), device=DEV)[:, None]
        J = torch.arange(n, device=DEV)[None]
        same = lab[i0:i0 + 2048, None] == lab[None]
        keep = (J > I) & (((S >= CAND) & same) | ((S >= CAND_X) & ~same))
        r, c = keep.nonzero(as_tuple=True)
        cand.append(torch.stack([r + i0, c, (S[r, c] * 1e4).round().int()], 1).cpu())
    cand = torch.cat(cand).numpy()
    pairs, cos = cand[:, :2].astype(np.int64), cand[:, 2] / 1e4
    print(f"    {len(pairs)} candidate pairs", flush=True)

    cache = FEATS / f"sift_rootsift_{N_KP}.npz"
    if cache.exists():
        z = np.load(cache)
        XY, D, NK = z["xy"], z["d"], z["n"]
    else:
        print(f"[4a] SIFT for {n} photos ({WORKERS} workers)", flush=True)
        with ProcessPoolExecutor(WORKERS, initializer=_init_worker) as ex:
            out = list(ex.map(_sift, meta.path, chunksize=32))
        XY = np.stack([o[0] for o in out])
        D = np.stack([o[1] for o in out])
        NK = np.array([o[2] for o in out], np.int32)
        np.savez(cache, xy=XY, d=D, n=NK)
    print("[4b] GPU matching", flush=True)
    counts, idx = gpu_matches(D, NK, pairs)
    print(f"    pairs with >= {MIN_MATCH} mutual ratio-test matches: {len(idx)}", flush=True)

    print("[5] RANSAC verification", flush=True)
    keys = sorted(idx)
    jobs = [(XY[pairs[k, 0]][idx[k][0]], XY[pairs[k, 1]][idx[k][1]]) for k in keys]
    with ProcessPoolExecutor(WORKERS, initializer=_init_worker) as ex:
        inl = list(ex.map(_ransac, jobs, chunksize=256))
    inliers = np.zeros(len(pairs), np.int32)
    inliers[keys] = inl
    ver = inliers >= MIN_INLIERS
    pv = pd.DataFrame({"a": meta.path.values[pairs[:, 0]], "b": meta.path.values[pairs[:, 1]], "cos": cos,
                       "matches": counts, "inliers": inliers,
                       "same_class": meta.label.values[pairs[:, 0]] == meta.label.values[pairs[:, 1]]})
    pv[pv.matches >= MIN_MATCH].to_csv(OUT / "pairs_checked.csv", index=False)
    pv = pv[ver]
    pv.to_csv(OUT / "pairs_verified.csv", index=False)
    print(f"    verified same-scene pairs: {ver.sum()} (cross-class: {int((~pv.same_class).sum())})", flush=True)

    print("[6] groups", flush=True)
    uf = UF(n)
    for g, ix in meta.groupby("group").indices.items():                 # previous capture sessions
        for i in ix[1:]:
            uf.union(ix[0], i)
    for _, ix in meta.groupby("md5").indices.items():                   # exact duplicates
        for i in ix[1:]:
            uf.union(ix[0], i)
    vp = pairs[ver]
    vsame = meta.label.values[vp[:, 0]] == meta.label.values[vp[:, 1]]
    for a, b in vp[vsame]:
        uf.union(a, b)
    root = np.array([uf.find(i) for i in range(n)])

    print("[7] removals", flush=True)
    reason, twin = {}, {}
    for p, e in bad:
        reason[p] = f"unreadable: {e}"
    for _, ix in meta.groupby("md5").indices.items():
        for i in ix[1:]:
            reason.setdefault(meta.path[i], "exact duplicate (identical bytes)")
            twin.setdefault(meta.path[i], meta.path[ix[0]])
    vcos = cos[ver]
    near = [(a, b) for (a, b), s, sc in zip(vp, vcos, vsame) if sc and s >= NEAR_DUP]
    near_adj = {}
    for a, b in near:
        near_adj.setdefault(a, []).append(b)
        near_adj.setdefault(b, []).append(a)
    kept = set()
    for i in range(n):                                                  # greedy, in file order: keep first, drop its twins
        p = meta.path[i]
        if p in reason:
            continue
        hit = [j for j in near_adj.get(i, []) if j in kept]
        if hit:
            reason[p] = "near-identical re-shot (verified, CLIP >= %.2f)" % NEAR_DUP
            twin[p] = meta.path[hit[0]]
        else:
            kept.add(i)
    for a, b in vp[~vsame]:
        for x, y in ((a, b), (b, a)):
            reason.setdefault(meta.path[x], "label conflict: same scene as a photo of another class")
            twin.setdefault(meta.path[x], meta.path[y])

    clean = meta[~meta.path.isin(reason)].copy()
    # renumber groups 0..G-1 on the kept photos
    r = root[clean.index.values]
    clean["group"] = pd.Series(r).map({v: k for k, v in enumerate(pd.unique(r))}).values
    clean["group_old"] = meta.loc[clean.index, "group"].values
    clean.to_csv(OUT / "meta_clean.csv", index=False)
    rem = pd.DataFrame({"path": list(reason), "reason": list(reason.values()),
                        "kept_twin": [twin.get(p, "") for p in reason]})
    rem.to_csv(OUT / "removed.csv", index=False)

    gs = clean.groupby("class_name").group.nunique()
    lines = ["# Dataset cleaning report (src/00_clean_dataset.py)", "",
             f"- photos before: **{n}**; removed: **{len(rem)}**; kept: **{len(clean)}**",
             f"- capture-session groups before: {meta.group.nunique()}; after merging verified re-shots: "
             f"**{clean.group.nunique()}**",
             f"- candidate pairs (CLIP >= {CAND} same class, >= {CAND_X} cross class): {len(pairs)}; "
             f"with >= {MIN_MATCH} matches: {len(idx)}; verified (>= {MIN_INLIERS} RANSAC inliers): {int(ver.sum())}",
             f"- verified pairs that were in DIFFERENT old groups (the re-shots the old grouping missed): "
             f"**{int(sum(meta.group.values[a] != meta.group.values[b] for a, b in vp[vsame]))}**", "",
             "## Removed, by reason", "", rem.reason.value_counts().to_frame("photos").to_markdown(), "",
             "## Kept photos per class", "",
             pd.DataFrame({"photos": clean.groupby("class_name").size(), "groups": gs,
                           "largest group": clean.groupby(["class_name", "group"]).size().groupby(level=0).max()}
                          ).to_markdown(), ""]
    (OUT / "report.md").write_text("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
