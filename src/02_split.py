"""Step 1.3: OBJECT-DISJOINT 20/80 split of the CLEANED dataset (data/clean/meta_clean.csv), plus K-shot support sets.

The cleaning step (00_clean_dataset.py) verified, with SIFT + RANSAC, which photos show the same SCENE (same objects and/or the same
textured background in the same place; stricter than 'same onion')
(data/clean/pairs_verified.csv). The photographers re-used the same onions on many cloths, so the "same-scene" graph
is highly connected: its connected components swallow up to 85 % of a class (all pile photos of a class become one
component), and a component-wise split would leave e.g. no pile photos in the training pool. Therefore:

  1. nodes   = capture sessions (00/01 audit groups: never split), edges = verified same-scene pairs, weight = inliers
  2. Louvain communities on that graph, per class (seed 0)
  3. communities are assigned to the pool in random order until the pool holds 20 % of the class's SINGLE-onion and
     20 % of its PILE photos (stratified by class x quantity)
  4. PURGE: every test photo with a same-scene link to any pool photo is removed from the test set
     (data/clean/purged_test.csv). Link = SIFT+RANSAC-verified pair OR CLIP cosine >= 0.95. Geometric verification is
     precise but misses re-arranged piles and texture-less single onions; the CLIP threshold was set by eye on random
     train/test pairs just below it (figures/clean_residual_*.png).
  5. support/validation inside the pool: K communities per class for the support set; a validation photo with a
     verified link to a support photo is dropped from that validation set (so validation is object-disjoint too).

splits/pool20.csv, splits/test80.csv, splits/support_K{K}_s{S}.csv, splits/val_K{K}_s{S}.csv,
splits/random_pool20.csv / random_test80.csv (naive random split, leakage experiment only)
"""
import numpy as np
import pandas as pd
from networkx import Graph
from networkx.algorithms.community import louvain_communities
from sklearn.model_selection import train_test_split

from common import C, ROOT, SEEDS, SHOTS, SPLITS, load_feats, load_meta

POOL_FRAC = 0.20
CLIP_PURGE = 0.95          # visual check (figures/clean_residual_*.png): same onion + same cloth still occurs at 0.95-0.955
CLEAN = ROOT / "data/clean"


def communities(df, pairs):
    """community id per photo: Louvain on the session graph of each class."""
    sess = dict(zip(df.path, df.group_old))
    comm = {}
    next_id = 0
    for lab in range(C):
        d = df[df.label == lab]
        G = Graph()
        G.add_nodes_from(d.group_old.unique())
        p = pairs[pairs.a.isin(d.path) & pairs.b.isin(d.path)]
        for a, b, w in zip(p.a.map(sess), p.b.map(sess), p.inliers):
            if a != b:
                G.add_edge(a, b, weight=G.get_edge_data(a, b, {"weight": 0})["weight"] + w)
        for c in louvain_communities(G, weight="weight", seed=0):
            for s in c:
                comm[s] = next_id
            next_id += 1
    return df.group_old.map(comm).values


_X = {}


def _clip():
    if not _X:
        f = load_feats("clip_global")
        _X["row"], _X["X"] = {p: i for i, p in enumerate(f["paths"])}, f["feats"].float()
    return _X["row"], _X["X"]


def linked(paths_a, paths_b, pairs):
    """paths in paths_a with a same-scene link to any path in paths_b: SIFT+RANSAC-verified pair (precise, misses
    re-arranged piles and texture-less single onions) OR CLIP cosine >= CLIP_PURGE (catches those)."""
    A, B = set(paths_a), set(paths_b)
    x = pairs[(pairs.a.isin(A) & pairs.b.isin(B))].a
    y = pairs[(pairs.b.isin(A) & pairs.a.isin(B))].b
    row, X = _clip()
    la, lb = list(paths_a), list(paths_b)
    s = (X[[row[p] for p in la]] @ X[[row[p] for p in lb]].T).max(1).values
    return set(x) | set(y) | {la[i] for i in (s >= CLIP_PURGE).nonzero().squeeze(1).tolist()}


def main():
    df = load_meta()
    pairs = pd.read_csv(CLEAN / "pairs_verified.csv")
    pairs = pairs[pairs.same_class & pairs.a.isin(df.path) & pairs.b.isin(df.path)]
    df["group"] = communities(df, pairs)

    rng = np.random.default_rng(0)
    pool_groups = []
    for lab in range(C):
        d = df[df.label == lab]
        target = d.groupby("quantity").size() * POOL_FRAC
        have = pd.Series(0, index=target.index)
        comp = d.groupby(["group", "quantity"]).size().unstack(fill_value=0).reindex(columns=target.index, fill_value=0)
        for g in rng.permutation(comp.index.values):
            if ((have + comp.loc[g]) <= target * 1.05).all():
                pool_groups.append(g)
                have += comp.loc[g]
    pool, test = df[df.group.isin(pool_groups)], df[~df.group.isin(pool_groups)]
    purge = linked(test.path, pool.path, pairs)
    pd.DataFrame({"path": sorted(purge), "reason": f"same-scene link to a pool photo (SIFT-verified or CLIP >= {CLIP_PURGE})"}).to_csv(
        CLEAN / "purged_test.csv", index=False)
    test = test[~test.path.isin(purge)]
    assert not set(pool.group) & set(test.group), "group leakage between pool and test"
    assert not linked(test.path, pool.path, pairs), "verified same-scene pair across pool/test"
    assert pool.label.nunique() == C and test.label.nunique() == C
    pool.to_csv(SPLITS / "pool20.csv", index=False)
    test.to_csv(SPLITS / "test80.csv", index=False)

    def val_without_links(val, sup):
        return val[~val.path.isin(linked(val.path, sup.path, pairs))]

    for K in SHOTS:
        for s in SEEDS:
            rng = np.random.default_rng(1000 * K + s)
            picks = []
            for lab in range(C):
                groups = pool[pool.label == lab].group.unique()
                chosen = rng.choice(groups, size=min(K, len(groups)), replace=False)
                for g in chosen:
                    rows = pool[pool.group == g]
                    picks.append(rows.iloc[rng.integers(len(rows))])
            sup = pd.DataFrame(picks)
            val = val_without_links(pool[~pool.group.isin(sup.group)], sup)
            sup.to_csv(SPLITS / f"support_K{K}_s{s}.csv", index=False)
            val.to_csv(SPLITS / f"val_K{K}_s{s}.csv", index=False)

    # "full" setting: the whole pool, with 20 % of its communities held out as validation
    for s in SEEDS:
        rng = np.random.default_rng(9000 + s)
        groups = pool.group.unique()
        val_groups = set(rng.choice(groups, size=len(groups) // 5, replace=False))
        sup = pool[~pool.group.isin(val_groups)]
        sup.to_csv(SPLITS / f"support_Kfull_s{s}.csv", index=False)
        val_without_links(pool[pool.group.isin(val_groups)], sup).to_csv(SPLITS / f"val_Kfull_s{s}.csv", index=False)

    # naive random split, same size, ignoring groups (leakage experiment only)
    rp, rt = train_test_split(df, train_size=len(pool), stratify=df.label, random_state=0)
    rp.to_csv(SPLITS / "random_pool20.csv", index=False)
    rt.to_csv(SPLITS / "random_test80.csv", index=False)

    tab = pd.DataFrame({"pool_imgs": pool.groupby("class_name").size(),
                        "pool_communities": pool.groupby("class_name").group.nunique(),
                        "pool_pile_share": pool.groupby("class_name").quantity.apply(lambda q: (q == "multiple").mean()),
                        "test_imgs": test.groupby("class_name").size(),
                        "test_pile_share": test.groupby("class_name").quantity.apply(lambda q: (q == "multiple").mean()),
                        "purged_from_test": df[df.path.isin(purge)].groupby("class_name").size()}).fillna(0)
    print(tab.round(3).to_string())
    print(f"pool {len(pool)} / test {len(test)} / purged {len(purge)} / communities {df.group.nunique()}")
    tab.round(3).to_csv(CLEAN / "split_summary.csv")


if __name__ == "__main__":
    main()
