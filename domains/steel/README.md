# Third domain: steel surface defects (industrial generalisation test)

Does PRGA only work on the domains it was designed on (onions, bees), or does the same code carry over to a
different industry unchanged? To test this we ran it, with no changes to the model, on **NEU-CLS**, the standard
hot-rolled steel strip defect dataset.

Everything ran on Kaggle (2 × T4) with `kaggle/kernel_steel_full/` (a seed-1 pilot first: `kaggle/kernel_steel_pilot`,
results in `results_pilot/`). `DATASET=steel` switches paths, classes and prompts (`src/common.py`).

## Data and split

- **NEU-CLS** (Song & Yan 2013): 1,800 grey-scale 200 × 200 images, 300 per class, 6 classes: crazing, inclusion,
  patches, pitted surface, rolled-in scale, scratches. Labels come from the file names.
- `src/25_prepare_steel.py`: one exact duplicate (same bytes) removed → 1,799 images; stratified split into a training
  pool of 359 images (20 %) and a **test set of 1,440 images**. NEU has no video or scene structure to group by.
- Support sets: K = 1, 2, 4, 8, 16 photos per class and "full" (the pool minus 20 % kept for validation), 3 seeds.
- Class descriptions (`prompts/descriptors.json`) were written from the standard NEU defect definitions before any
  image or result was looked at (`prompts/notes.md`). OWLv2 prompts: `prompts/owl_prompts.json`.
- Backbone: the pilot's validation chose **CLIP ViT-L/14** for PRGA + DINOv2; both backbones were run for every method.

## Results

Macro-F1 on the 1,440 test images, mean ± std over 3 seeds (`results/equal_training.md`, `results/runs/`).
Every adapter is trained the same way (`src/18_equal_training.py`).

| method | backbone | K=1 | K=2 | K=4 | K=8 | K=16 | all |
|---|---|---|---|---|---|---|---|
| **PRGA + DINOv2 cache** (chosen) | L/14 + DINOv2 | **0.834** ± .055 | **0.945** ± .011 | 0.962 ± .004 | 0.982 ± .012 | 0.991 ± .003 | 0.993 ± .001 |
| PRGA + DINOv2 cache | B/16 + DINOv2 | 0.830 ± .050 | 0.941 ± .008 | **0.969** ± .005 | **0.988** ± .008 | **0.995** ± .001 | **0.996** ± .002 |
| PRGA | L/14 | 0.697 ± .020 | 0.879 ± .030 | 0.927 ± .001 | 0.955 ± .005 | 0.985 ± .003 | 0.992 ± .001 |
| PlantCaFo-style cache | L/14 + DINOv2 | 0.798 ± .018 | 0.938 ± .013 | 0.955 ± .006 | 0.975 ± .008 | 0.985 ± .004 | 0.983 ± .003 |
| PlantCaFo-style cache | B/16 + DINOv2 | 0.782 ± .055 | 0.938 ± .017 | 0.959 ± .005 | 0.967 ± .018 | 0.986 ± .004 | 0.980 ± .016 |
| Linear probe | L/14 | 0.687 ± .024 | 0.897 ± .045 | 0.939 ± .009 | 0.963 ± .011 | 0.978 ± .003 | 0.988 ± .004 |
| CLIP-Adapter | L/14 | 0.679 ± .048 | 0.892 ± .041 | 0.919 ± .008 | 0.955 ± .027 | 0.981 ± .002 | 0.993 ± .002 |
| TaskRes | L/14 | 0.701 ± .041 | 0.885 ± .035 | 0.922 ± .007 | 0.954 ± .022 | 0.979 ± .006 | 0.994 ± .001 |
| CLAP | L/14 | 0.709 ± .058 | 0.856 ± .071 | 0.919 ± .016 | 0.959 ± .014 | 0.978 ± .003 | 0.992 ± .001 |
| Tip-Adapter-F | L/14 | 0.682 ± .054 | 0.881 ± .048 | 0.927 ± .015 | 0.953 ± .019 | 0.974 ± .004 | 0.983 ± .003 |
| GraphAdapter | L/14 | 0.682 ± .100 | 0.757 ± .039 | 0.805 ± .039 | 0.910 ± .053 | 0.955 ± .008 | 0.975 ± .007 |
| Base paper (Ahmad et al. 2025) | B/16 | 0.694 ± .049 | 0.820 ± .024 | 0.775 ± .062 | 0.837 ± .017 | 0.868 ± .034 | 0.880 ± .011 |
| EfficientNet-B0, fine-tuned | – | 0.522 ± .071 | 0.619 ± .096 | 0.748 ± .084 | 0.819 ± .019 | 0.892 ± .016 | 0.944 ± .007 |
| Zero-shot CLIP | L/14 | 0.148 | | | | | |

B/16 results for the other adapters are in `results/equal_training.md`; they are within about 0.01 of L/14.

- **PRGA + DINOv2 is best or within the seed spread of the best at every K.** The clearest lead is at 1 shot (+3.6
  points over the next method). From 16 shots almost every adapter is above 0.97, so the room for any method to stand
  out is small.
- **The validation choice of L/14 was not the test optimum:** B/16 is 0.4-0.7 points better on test at 4-16 shots.
  The L/14 row is the honest headline because it was chosen on validation; the B/16 row is shown for completeness.
- **Zero-shot CLIP is near chance (0.15 for 6 classes):** CLIP does not know these defect names, so the few labelled
  images do all the work, which is the setting PRGA is built for.
- The base paper's method and a fine-tuned CNN fall far behind at few shots, as on onions.

### Ablations (PRGA alone, L/14, `results/ablations_K1-2-4-8-16-full_clip_l14.csv`)

Starting from the base-paper-style full configuration (graph also at test time, box geometry on edges):

| variant | K=1 | K=4 | K=16 | all |
|---|---|---|---|---|
| full configuration | 0.457 | 0.812 | 0.952 | 0.976 |
| graph only in training (= chosen PRGA setting) | **0.697** | **0.928** | 0.984 | **0.993** |
| grid of patches instead of OWLv2 regions | 0.636 | 0.914 | **0.985** | 0.985 |
| no text nodes | 0.465 | 0.776 | 0.945 | 0.962 |
| generic template text instead of descriptions | 0.445 | 0.780 | 0.937 | 0.968 |
| no gate | 0.462 | 0.785 | 0.951 | 0.969 |

- **Graph only in training** is again the biggest gain (+11.6 points at 4 shots), as on onions: the same train/test
  query fix matters in a third domain.
- **Text nodes and written class descriptions help** (−3.6 / −3.2 points at 4 shots without them).
- **A grid beats OWLv2 regions here.** Steel defects are textures spread over the whole patch, not objects, so
  object-detector boxes add little. This matches what we found on onions (large signs) and differs from bees (tiny mite).

## Web-app model

`checkpoints/app_steel_K4.pt` and `app_steel_Kfull.pt` (seed 1, L/14 + DINOv2, from `src/24_export_app_model.py`):

| model | test macro-F1 | calibration error (ECE) raw → calibrated |
|---|---|---|
| 4 photos per class | 0.964 → 0.981 after calibration | 0.034 → 0.012 |
| all training photos | 0.992 | 0.105 (unchanged) |

For the all-photos model the validation photos had no errors, so calibration had nothing to fit and the guard in
`src/calibration.py` left the model unchanged; its probabilities are therefore too low on average (under-confident).

## Limitations

- NEU images are clean, centred lab patches; real production-line images (lighting, oil, motion blur) are untested.
- No grouping by steel strip is possible with NEU, so near-identical patches from the same strip may be on both
  sides of the split. This may inflate all methods equally.
- Three seeds; differences below about two standard deviations are not meaningful.

## Data

K. Song and Y. Yan, "A noise robust method based on completed local binary patterns for hot-rolled steel strip
surface defects," Applied Surface Science, vol. 285, 2013 (NEU surface defect database).
