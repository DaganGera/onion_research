"""Second domain: honeybees with and without the Varroa mite (VarroaDataset, Zenodo 4085044). Run with DATASET=bees.

The images are crops of tracked bees cut from laboratory videos: <video>.mp4-bee_id_<track>-<frame>-1.png. Frames of
the same bee are near-copies, and all bees in one video share the same glass, light and background. A random split
would put the same bee, or at least the same recording, in training and test (the same leakage problem as the
re-photographed onions). CLIP cannot be used to find the copies here: on these crops every bee is close to every other
bee (nearest-neighbour cosine 0.97-0.99 for almost all photos, audit on Kaggle). The file names give the structure
directly, so:

  1. labels from gt.csv: 0 = healthy, 1 = Varroa. gt.csv also has 864 photos with label 3, which the dataset's README
     does not define; they are left out. The mite boxes are kept for checking the region detector.
  2. group = video (one recording session). Groups are never split.
  3. about 20 % of each class goes to the training pool, chosen as whole videos; the other videos are the test set
  4. K-shot support sets: K different bee tracks per class from the pool, one frame each; validation = pool frames of
     the other tracks. "full" = the whole pool with 20 % of its tracks held out for validation.
     (inside the pool, tracks rather than videos are held out: the pool has only a few videos)
  5. a naive random split of the same size, only to measure how much leakage would inflate the scores

Output: data/meta.csv, data/clean/meta_clean.csv, splits/*.csv
"""
import os
import re

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from common import CLASSES, DATASET, DROOT, RAW, SEEDS, SHOTS, SPLITS

assert DATASET == "bees", "run with DATASET=bees"
POOL_FRAC = 0.20
CLEAN = DROOT / "data/clean"
NAME = re.compile(r"(?P<video>.+?)\.mp4-bee_id_(?P<track>\d+)-(?P<frame>\d+)-\d+\.png$")


def read_gt():
    gt = RAW / "gt.csv" if (RAW / "gt.csv").exists() else DROOT / "data/gt.csv"
    rows = []
    for line in gt.read_text().splitlines():
        parts = line.split()
        if len(parts) < 2:
            continue
        nums = [float(x) for x in parts[2:]]
        m = NAME.search(os.path.basename(parts[0]))
        rows.append(dict(path=parts[0], label=int(parts[1]), video=m["video"], track=f'{m["video"]}#{m["track"]}',
                         frame=int(m["frame"]), orig_split=parts[0].split("/")[0],
                         mite_boxes=str([nums[i:i + 4] for i in range(0, len(nums) - 3, 4)])))
    return pd.DataFrame(rows)


def main():
    df = read_gt()
    df.to_csv(DROOT / "data/meta.csv", index=False)
    print("labels in gt.csv:", df.label.value_counts().to_dict())
    print("label-3 photos with mite boxes:", int((df[df.label == 3].mite_boxes != "[]").sum()), "of", int((df.label == 3).sum()))
    df = df[df.label.isin([0, 1])].reset_index(drop=True)
    df["class_name"] = df.label.map(dict(enumerate(CLASSES)))
    df["group"] = df.video
    assert (RAW / df.path.iloc[0]).exists(), RAW / df.path.iloc[0]
    per_video = df.groupby(["video", "label"]).size().unstack(fill_value=0)
    print(f"{len(df)} photos, {df.video.nunique()} videos, {df.track.nunique()} bee tracks")
    print(per_video.assign(orig_split=df.groupby("video").orig_split.first()).to_string())
    CLEAN.mkdir(parents=True, exist_ok=True)
    df.to_csv(CLEAN / "meta_clean.csv", index=False)

    # pool = whole videos, until each class has about 20 % of its photos (search over random orders, keep the closest)
    target = df.label.value_counts().sort_index() * POOL_FRAC
    counts, tgt, videos = per_video[[0, 1]].values, target.values, per_video.index.values
    best, best_err = None, np.inf
    rng = np.random.default_rng(0)
    for _ in range(20000):
        have, chosen = np.zeros(2), []
        for v in rng.permutation(len(videos)):
            if ((have + counts[v]) <= tgt * 1.25).all():
                chosen.append(videos[v])
                have += counts[v]
        err = float((abs(have - tgt) / tgt).max())
        if err < best_err and len(chosen) < len(videos):
            best, best_err = chosen, err
    pool, test = df[df.video.isin(best)], df[~df.video.isin(best)]
    print(f"pool videos {best}: class counts {pool.label.value_counts().sort_index().to_dict()} "
          f"(target {target.round().astype(int).to_dict()}, worst relative error {best_err:.2f})")
    assert not set(pool.video) & set(test.video) and pool.label.nunique() == 2 and test.label.nunique() == 2
    SPLITS.mkdir(parents=True, exist_ok=True)
    pool.to_csv(SPLITS / "pool20.csv", index=False)
    test.to_csv(SPLITS / "test80.csv", index=False)

    for K in SHOTS:
        for s in SEEDS:
            rng = np.random.default_rng(1000 * K + s)
            picks = []
            for lab in range(len(CLASSES)):
                tracks = pool[pool.label == lab].track.unique()
                for t in rng.choice(tracks, size=min(K, len(tracks)), replace=False):
                    rows = pool[(pool.track == t) & (pool.label == lab)]
                    picks.append(rows.iloc[rng.integers(len(rows))])
            sup = pd.DataFrame(picks)
            sup.to_csv(SPLITS / f"support_K{K}_s{s}.csv", index=False)
            pool[~pool.track.isin(sup.track)].to_csv(SPLITS / f"val_K{K}_s{s}.csv", index=False)
    for s in SEEDS:
        rng = np.random.default_rng(9000 + s)
        tracks = pool.track.unique()
        val_tracks = set(rng.choice(tracks, size=len(tracks) // 5, replace=False))
        pool[~pool.track.isin(val_tracks)].to_csv(SPLITS / f"support_Kfull_s{s}.csv", index=False)
        pool[pool.track.isin(val_tracks)].to_csv(SPLITS / f"val_Kfull_s{s}.csv", index=False)

    rp, rt = train_test_split(df, train_size=len(pool), stratify=df.label, random_state=0)
    rp.to_csv(SPLITS / "random_pool20.csv", index=False)
    rt.to_csv(SPLITS / "random_test80.csv", index=False)
    tab = pd.DataFrame({"pool": pool.groupby("class_name").size(), "pool_tracks": pool.groupby("class_name").track.nunique(),
                        "pool_videos": pool.groupby("class_name").video.nunique(),
                        "test": test.groupby("class_name").size(), "test_videos": test.groupby("class_name").video.nunique()})
    print(tab.to_string())
    tab.to_csv(CLEAN / "split_summary.csv")


if __name__ == "__main__":
    main()
