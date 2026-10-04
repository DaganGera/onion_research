# Dataset cleaning and the leak-free split

Scripts: `src/00_clean_dataset.py` (cleaning), `src/00b_clean_inspect.py` (visual checks), `src/02_split.py` (split).
Outputs: `data/clean/` (`meta_clean.csv`, `removed.csv`, `purged_test.csv`, `pairs_verified.csv`, `report.md`,
`split_summary.csv`) and the montages `figures/clean_*.png` referred to below.

## Why the earlier split was not clean enough

The dataset (Kulkarni et al., Mendeley Data 2025, 12,260 bulb photos) was shot with one phone. Each onion, or pile of
onions, was photographed many times, often on the same cloth, and **the same onions were re-used across sessions and
backgrounds**. The earlier pipeline grouped photos by exact bytes (MD5), perceptual hash and a CLIP "capture-session"
clustering, then split by group. Looking at the most similar train/test pairs of that split showed **the same onion on
the same cloth on both sides** (e.g. `Onion05365` in test and `Onion05883` in train, CLIP cosine 0.986). The session
clustering missed them because those re-shots were taken in different sessions. Duplicates were also only *grouped*,
never *removed*: 27 exact duplicate files were still in the data, 20 of them in the test set.

## What the cleaning does

| step | method | result |
|---|---|---|
| integrity | every file decoded (`PIL.verify` + load) | 0 unreadable files |
| exact duplicates | MD5 of the file bytes; keep the first | **27 removed** |
| same-scene detection | candidate pairs = same class and CLIP cosine ≥ 0.92 (1,406,308 pairs) → RootSIFT keypoints (400 per photo, 640 px) matched on the GPU (Lowe ratio 0.8 + mutual nearest neighbour; 433,150 pairs with ≥ 8 matches) → **RANSAC homography, ≥ 10 inliers** | 79,194 verified same-scene pairs; **0 across classes** (a check of specificity) |
| near-identical frames | verified pair **and** CLIP ≥ 0.99: keep the first, remove the rest | **557 removed** |
| **cleaned dataset** | | **11,676 photos** (`data/clean/meta_clean.csv`) |

**What the geometric check really detects: the same *scene*, not strictly the same onion.** Keypoints are found
wherever there is texture, and a shaggy rug or a patterned cloth has much more texture than a smooth onion skin. In
`figures/clean_sift_example.png` almost all 159 verified matches are on the rug: the two photos share the same
rug patch and camera position, while the onions themselves may differ. For a leak-free split this errs on the safe side
(it groups more photos together than strictly necessary, and also stops a model from scoring by recognising a
background). Where this document says a pair is the *same onion*, that was confirmed by eye, not by the algorithm.

Calibration was done by eye: `figures/clean_inliers_*.png` shows random verified pairs per inlier band (≥ 10 inliers:
almost all the same scene; 7–9: mixed).

## Finding: the dataset has far fewer physical onions than photos

Verified pairs that the old session grouping had put in **different** groups: **29,465**. Many link the same onions
photographed on *different* backgrounds (e.g. the same rotten red onions on a white cloth and on a wooden floor). As a
result, the "same-scene" graph is highly connected: its connected components contain up to 85 % of a class. A
component-wise split would put nearly all pile photos in the test set (in a trial run: 0 % pile photos in the
healthy-red training pool vs 29 % in test). That is a distribution shift we would be creating ourselves, so we did not use it.

## The split (`src/02_split.py`)

1. Nodes = capture sessions (never split); edges = verified same-scene pairs, weighted by inliers.
2. Louvain communities per class (seed 0): 960 communities.
3. Communities go to the training pool in random order until it holds 20 % of each class's single-onion **and** 20 % of
   its pile photos (stratified by class × quantity).
4. **Purge:** a test photo is removed if it is linked to *any* pool photo, either by a SIFT-verified pair or by CLIP
   cosine ≥ 0.95. The second rule is needed because geometric verification is precise but misses re-arranged piles
   and texture-less single onions: `figures/clean_leakcheck_top16.png` (before the CLIP rule) shows, for example, the
   same onion on the same red cloth at cosine 0.981 that SIFT did not verify. The 0.95 threshold was set by looking at
   random train/test pairs just below it (`figures/clean_residual_0.95.png`, `clean_residual_0.955.png`).
5. Support and validation sets inside the pool follow the same rule (no validation photo linked to a support photo).

| class | pool photos | pool pile share | test photos | test pile share | purged from test |
|---|---|---|---|---|---|
| healthy red onion | 744 | 0.169 | 2532 | 0.190 | 533 |
| healthy white onion | 833 | 0.247 | 2696 | 0.217 | 453 |
| unhealthy red onion | 401 | 0.474 | 1100 | 0.301 | 414 |
| unhealthy white onion | 410 | 0.493 | 1284 | 0.403 | 276 |
| **total** | **2,388** | | **7,612** | | **1,676** |

## Checks after the split

- verified same-scene pairs between train pool and test: **0** (asserted in the code)
- highest train↔test CLIP cosine: **0.950** (by construction)
- support ∩ validation communities: 0, for every K and seed

## Remaining risk (stated, not hidden)

Random train/test pairs in the band just below the threshold (`figures/clean_residual_0.945_after.png`, 908 test
photos with a pool neighbour at cosine 0.945–0.95) are mostly different onions and scenes. A few (e.g. a plain white
onion on the same black cloth) *could* be the same onion; such pairs cannot be verified even by eye. Our estimate is that
**at most ~1–2 % of test photos** may still share an onion with a training photo. The purge also removed pile photos more
often than single-onion photos (piles look alike), so test has slightly fewer piles than the pool (table above).

## Consequence for the numbers

All results in `results/` are on this cleaned, object-disjoint split. They are **not comparable** with the numbers in
`onion_research/README.md` or `archive_v1/` (earlier, leakier splits) and are expected to be lower: a model can no
longer score points by recognising an onion it has already seen.
