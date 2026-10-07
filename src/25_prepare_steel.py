"""Third domain: steel surface defects (NEU surface defect database). Run with DATASET=steel.

NEU has 6 defect classes of hot-rolled steel strips (crazing, inclusion, patches, pitted surface, rolled-in scale,
scratches), 300 grayscale 200 x 200 photos each. Each photo is a separate patch, so there are no scenes or videos to
group; exact duplicates (same bytes) are removed and the split is stratified by class:

  1. labels from the file name (crazing_12.jpg -> crazing), or the folder name if the file name has no class
  2. exact duplicates (MD5) are dropped; near-copies are reported (CLIP cosine is checked later by the leakage test)
  3. 20 % of each class -> training pool, 80 % -> test
  4. K-shot support sets from the pool (seeds 1-3); validation = the rest of the pool; "full" = pool minus 20 %

Output: data/meta.csv, data/clean/meta_clean.csv, splits/*.csv
"""
import hashlib
import os

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from common import CLASSES, DATASET, DROOT, RAW, SEEDS, SHOTS, SPLITS

assert DATASET == "steel", "run with DATASET=steel"
KEYS = {"crazing": 0, "inclusion": 1, "patches": 2, "pitted": 3, "rolled": 4, "scratches": 5}


def label_of(path):
    for part in (os.path.basename(path).lower(), *reversed(path.lower().split("/")[:-1])):
        for k, v in KEYS.items():
            if part.startswith(k):
                return v
    return None


def main():
    files = sorted(str(p.relative_to(RAW)) for p in RAW.rglob("*")
                   if p.suffix.lower() in (".jpg", ".jpeg", ".png", ".bmp") and p.is_file())
    rows = []
    for f in files:
        lab = label_of(f)
        if lab is None:
            continue
        md5 = hashlib.md5((RAW / f).read_bytes()).hexdigest()
        rows.append(dict(path=f, label=lab, class_name=CLASSES[lab], md5=md5))
    df = pd.DataFrame(rows)
    print(f"{len(files)} image files, {len(df)} with a class; per class:", df.class_name.value_counts().to_dict())
    (DROOT / "data").mkdir(parents=True, exist_ok=True)
    df.to_csv(DROOT / "data/meta.csv", index=False)
    df = df.drop_duplicates("md5").reset_index(drop=True)
    df["group"] = df.path
    print(f"after removing exact duplicates: {len(df)}")
    (DROOT / "data/clean").mkdir(parents=True, exist_ok=True)
    df.to_csv(DROOT / "data/clean/meta_clean.csv", index=False)

    pool, test = train_test_split(df, train_size=0.2, stratify=df.label, random_state=0)
    SPLITS.mkdir(parents=True, exist_ok=True)
    pool.to_csv(SPLITS / "pool20.csv", index=False)
    test.to_csv(SPLITS / "test80.csv", index=False)
    for K in SHOTS:
        for s in SEEDS:
            rng = np.random.default_rng(1000 * K + s)
            sup = pd.concat([pool[pool.label == c].iloc[rng.choice((pool.label == c).sum(), K, replace=False)]
                             for c in range(len(CLASSES))])
            sup.to_csv(SPLITS / f"support_K{K}_s{s}.csv", index=False)
            pool[~pool.path.isin(sup.path)].to_csv(SPLITS / f"val_K{K}_s{s}.csv", index=False)
    for s in SEEDS:
        sup, val = train_test_split(pool, test_size=0.2, stratify=pool.label, random_state=9000 + s)
        sup.to_csv(SPLITS / f"support_Kfull_s{s}.csv", index=False)
        val.to_csv(SPLITS / f"val_Kfull_s{s}.csv", index=False)
    print(f"pool {len(pool)} / test {len(test)}")


if __name__ == "__main__":
    main()
