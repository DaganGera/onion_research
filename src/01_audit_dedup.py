"""Step 1.2 / 2.2: audit the onion-bulb dataset and group near-identical photos.

Folder layout: RAW/<health>/<variety>/<single|multiple>/OnionNNNNN.jpg  ->  4 classes (health x variety).

Two photos share a GROUP (and therefore always land on the same side of the split) if they
  (a) have identical bytes (MD5),
  (b) have perceptual hashes within HAMMING_MAX bits (same class), or
  (c) --refine: fall in the same CAPTURE-SESSION cluster: average-linkage agglomerative clustering of CLIP
      embeddings within a class, cut at cosine CLIP_DUP. The dataset photographs each onion (or pile) many times
      on the same cloth; those re-shots are near-identical for CLIP. Average linkage (not single linkage) is used
      because single linkage chains everything into one giant group. Needs features/clip_global.pt.
Images whose exact bytes appear under two classes are dropped.
Output: data/meta.csv, results/audit.csv
"""
import argparse
import hashlib

import imagehash
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

from common import CLASS_MAP, CLASSES, META_RAW, RAW, RESULTS, load_feats

HAMMING_MAX = 6
CLIP_DUP = 0.93


class UnionFind:
    def __init__(self, n):
        self.p = list(range(n))

    def find(self, i):
        while self.p[i] != i:
            self.p[i] = self.p[self.p[i]]
            i = self.p[i]
        return i

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.p[rb] = ra


def scan():
    rows = []
    for (health, variety), cname in CLASS_MAP.items():
        for qdir in sorted((RAW / health / variety).iterdir()):
            files = sorted(f for f in qdir.iterdir() if f.is_file())
            for f in tqdm(files, desc=f"{cname}/{qdir.name}", mininterval=10):
                b = f.read_bytes()
                with Image.open(f) as im:
                    im = im.convert("RGB")
                    ph = str(imagehash.phash(im))
                    w, h = im.size
                rows.append(dict(path=str(f.relative_to(RAW)), label=CLASSES.index(cname), class_name=cname,
                                 health=health.split(". ")[1].lower(), variety=variety.split(". ")[1].lower(),
                                 quantity=qdir.name.split(". ")[1].lower(), md5=hashlib.md5(b).hexdigest(),
                                 phash=ph, w=w, h=h))
    return pd.DataFrame(rows)


def pairs_within_class(df, sim_fn, thr, uf, desc):
    n_pairs = 0
    for lab in tqdm(range(len(CLASSES)), desc=desc):
        idx = np.where(df.label.values == lab)[0]
        ii, jj = np.where(np.triu(sim_fn(idx) >= thr, k=1))
        for a, b in zip(ii, jj):
            uf.union(idx[a], idx[b])
        n_pairs += len(ii)
    return n_pairs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--refine", action="store_true", help="also merge CLIP near-duplicates (needs clip_global)")
    a = ap.parse_args()

    df = pd.read_csv(META_RAW) if (a.refine and META_RAW.exists()) else scan()
    md5_labels = df.groupby("md5")["label"].nunique()
    conflict = set(md5_labels[md5_labels > 1].index)
    n_conf = int(df.md5.isin(conflict).sum())
    df = df[~df.md5.isin(conflict)].reset_index(drop=True)

    uf = UnionFind(len(df))
    for _, idx in df.groupby("md5").groups.items():
        idx = list(idx)
        for j in idx[1:]:
            uf.union(idx[0], j)

    def phash_sim(idx):   # similarity = -Hamming distance, so ">= -HAMMING_MAX" means "distance <= HAMMING_MAX"
        bits = np.array([[int(c) for c in bin(int(h, 16))[2:].zfill(64)] for h in df.phash.values[idx]],
                        dtype=np.float32)
        return -(bits @ (1 - bits).T + (1 - bits) @ bits.T)
    # pHash pairs are single-linkage and chain whole sessions together; in --refine mode the CLIP session
    # clusters already contain every pHash near-duplicate, so pHash is used only for the first (pre-CLIP) pass.
    n_ph = 0 if a.refine else pairs_within_class(df, phash_sim, -HAMMING_MAX, uf, "phash")

    n_clip = 0
    if a.refine:
        f = load_feats("clip_global")
        row = {p: i for i, p in enumerate(f["paths"])}
        X = f["feats"].float()[[row[p] for p in df.path]].numpy()
        from sklearn.cluster import AgglomerativeClustering
        for lab in range(len(CLASSES)):
            idx = np.where(df.label.values == lab)[0]
            D = np.clip(1 - X[idx] @ X[idx].T, 0, None)
            cl = AgglomerativeClustering(n_clusters=None, metric="precomputed", linkage="average",
                                         distance_threshold=1 - CLIP_DUP).fit(D).labels_
            for c in np.unique(cl):
                members = idx[cl == c]
                for j in members[1:]:
                    uf.union(members[0], j)
                n_clip += len(members) - 1

    roots = [uf.find(i) for i in range(len(df))]
    remap = {r: k for k, r in enumerate(dict.fromkeys(roots))}
    df["group"] = [remap[r] for r in roots]
    df.to_csv(META_RAW, index=False)

    audit = df.groupby("class_name").agg(files=("path", "size"), unique_md5=("md5", "nunique"),
                                         groups=("group", "nunique"),
                                         single=("quantity", lambda q: (q == "single").sum()),
                                         multiple=("quantity", lambda q: (q == "multiple").sum()))
    audit.loc["TOTAL"] = audit.sum()
    audit.to_csv(RESULTS / "audit.csv")
    print(audit.to_string())
    print(f"dropped {n_conf} label-conflict images; phash pairs {n_ph}; clip pairs {n_clip}; "
          f"{df.group.nunique()} groups in {len(df)} images")


if __name__ == "__main__":
    main()
