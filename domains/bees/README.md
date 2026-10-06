# Second domain: Varroa mites on honeybees

Can the same few-shot methods tell a **healthy honeybee** from a **bee carrying a Varroa mite** after seeing 1 to 16
photos per class? The mite is a 1-2 mm reddish-brown spot on the bee, a much smaller sign than onion rot. That is why
we chose this domain: it tests whether PRGA's region nodes help when the sign is tiny.

Everything here was run on Kaggle (2 × T4) with `kaggle/kernel_bees_run/bees_run.py`. All code is the same as for
the onions; `DATASET=bees` switches the paths, classes and prompts (`src/common.py`).

## Data and split

**VarroaDataset** (Schurischuster & Kampel 2020, [Zenodo 4085044](https://zenodo.org/record/4085044)): crops of
single bees (160 × 280 px) cut from 12 laboratory videos recorded in 2017, with mite boxes for infested bees.

- Labels 0 (healthy, 9,562) and 1 (Varroa, 3,083) are used. `gt.csv` also has 864 photos with **label 3**, which the
  dataset's README does not define (all 864 have mite boxes). They are left out rather than guessed.
- **Split by whole video** (`src/20_prepare_bees.py`): bees in one video share the same glass, light and
  background, and consecutive frames are near-copies. The training pool is 6 videos (2,672 photos, about 20 % of each
  class); the test set is the other 6 videos (9,973 photos). No video is in both.
- CLIP could not be used to find near-copies here: on these crops every bee is close to every other bee (nearest-
  neighbour cosine 0.97-0.99 for almost all photos), so the file names (video, bee id, frame) define the groups.
- Images are padded to a square instead of centre-cropped, for every method, so the head and abdomen are not cut off.

**Leakage check** (train on the pool, test on the rest; `results/leakage_*.csv`):

| model | random split | video-disjoint split |
|---|---|---|
| 1-nearest-neighbour (CLIP) | 0.624 | 0.556 |
| linear probe (CLIP) | 0.675 | 0.577 |

A random split would overstate macro-F1 by about 7-10 points.

## Results

Macro-F1 on the 9,973 test photos, mean ± std over 3 support sets. Comparison methods use the fair protocol of
`src/18_equal_training.py` (2 × epochs, best of 10 validation checkpoints). PRGA uses the configuration chosen on
the **onion** validation data, unchanged. For two classes, always answering "healthy" gives macro-F1 0.43, and random
guessing gives about 0.50.

| method | K=1 | K=2 | K=4 | K=8 | K=16 | all (2,138) |
|---|---|---|---|---|---|---|
| PRGA + DINOv2 cache | 0.412 ± .157 | 0.499 ± .122 | 0.587 ± .017 | 0.574 ± .041 | **0.601** ± .011 | 0.696 ± .005 |
| PRGA | 0.454 ± .140 | 0.514 ± .101 | **0.589** ± .025 | 0.576 ± .040 | 0.597 ± .026 | 0.699 ± .002 |
| Tip-Adapter-F | **0.567** ± .003 | 0.562 ± .054 | 0.584 ± .037 | **0.592** ± .025 | 0.596 ± .018 | 0.607 ± .009 |
| PlantCaFo-style cache | 0.534 ± .037 | 0.559 ± .046 | 0.563 ± .032 | 0.574 ± .024 | 0.580 ± .027 | 0.626 ± .016 |
| CLIP-Adapter | 0.466 ± .051 | 0.474 ± .073 | 0.575 ± .037 | 0.565 ± .037 | 0.590 ± .014 | 0.693 ± .011 |
| GraphAdapter | 0.510 ± .094 | 0.508 ± .067 | 0.580 ± .001 | 0.573 ± .038 | 0.573 ± .011 | 0.664 ± .017 |
| TaskRes | 0.380 ± .148 | 0.483 ± .076 | 0.585 ± .023 | 0.523 ± .093 | 0.588 ± .016 | 0.677 ± .014 |
| CLAP | 0.510 ± .049 | 0.513 ± .047 | 0.563 ± .015 | 0.560 ± .030 | 0.594 ± .009 | 0.635 ± .036 |
| Ahmad et al. 2025 (replication) | 0.473 ± .001 | 0.428 ± .127 | 0.551 ± .025 | 0.530 ± .057 | 0.536 ± .013 | 0.427 ± .000 |
| same recipe, no graph (control) | 0.535 ± .023 | 0.549 ± .038 | 0.567 ± .037 | 0.553 ± .018 | 0.566 ± .004 | 0.615 ± .031 |
| Linear probe | 0.306 ± .075 | 0.436 ± .110 | 0.552 ± .034 | 0.509 ± .085 | 0.575 ± .017 | 0.659 ± .011 |
| EfficientNet-B0, fine-tuned | 0.396 ± .099 | 0.413 ± .042 | 0.512 ± .052 | 0.526 ± .054 | 0.478 ± .008 | **0.770** ± .012 |

Zero-shot CLIP (no training photos): 0.561 with our class descriptions, 0.551 with generic templates, 0.474 with
"a photo of a [CLASS]".

**BioCLIP instead of CLIP.** BioCLIP (trained on the Tree of Life, insects included) was run for PRGA,
PRGA + DINOv2, Tip-Adapter-F and the PlantCaFo-style cache. Its validation macro-F1 was lower than CLIP's for all four
(e.g. PRGA 0.626 vs 0.652), so **CLIP is kept**. The test set agrees (mean over all K: PRGA 0.569 vs 0.571).
BioCLIP zero-shot: 0.509. Rows: `results/equal_training.md`.

### Do region nodes help when the sign is tiny?

PRGA ablations on bees (`results/ablations_K1-2-4-8-16-full.csv`). This uses PRGA's default configuration (graph used
at test time, geometry on), which is why the "regions" row differs from the table above:

| PRGA nodes | K=1 | K=4 | K=16 | all |
|---|---|---|---|---|
| OWLv2 regions (object + 2 body parts + 2 spots) | 0.585 | 0.565 | 0.595 | **0.725** |
| 3 × 3 grid instead of regions | 0.566 | 0.542 | 0.592 | 0.676 |
| object box only | 0.577 | 0.584 | 0.584 | 0.673 |
| regions, no text nodes | 0.580 | 0.581 | 0.589 | 0.720 |
| regions, generic text instead of descriptions | 0.569 | 0.556 | 0.564 | 0.720 |

OWLv2 put a "spot" region on the annotated mite in **38 %** of Varroa photos (`results/region_check.csv`), and found
spots on healthy bees too (0.45 per photo vs 0.80 on Varroa photos).

## Second round: a stronger backbone (CLIP ViT-L/14)

**Pilot first** (`src/22_bees_pilot.py`, `results/pilot.csv`): K = 4 and 16, seed 1. Support photos came from 3 pool
videos and validation from the other 3 pool videos, so validation measured cross-video transfer. Test was 2,000
random test photos. It tried 4 text backbones (CLIP B/16, CLIP L/14, SigLIP B/16, BioCLIP) × 2 DINOv2 sizes ×
mite-scale tiles × per-video centring × a mite prototype built from support boxes. The rule, fixed beforehand: go
only if a variant beats the baseline by ≥ 5 validation points and test moves the same way. Only **CLIP L/14**
passed, for every method (test +6 to +11 points at K = 4 and 16). Tiles, centring and the mite prototype gave no
consistent gain, so the full run changes **only the backbone**: same split, same OWLv2 boxes, same methods and fair
protocol (`kaggle/kernel_bees_l14`).

**Full run, CLIP ViT-L/14** (macro-F1, mean ± std over 3 seeds; comparison methods best-epoch):

| method | K=1 | K=2 | K=4 | K=8 | K=16 | all |
|---|---|---|---|---|---|---|
| PRGA + DINOv2 cache | 0.665 ± .017 | 0.688 ± .018 | 0.663 ± .053 | 0.675 ± .019 | 0.679 ± .029 | 0.730 ± .013 |
| PRGA | 0.654 ± .013 | 0.683 ± .010 | 0.664 ± .051 | 0.669 ± .012 | 0.678 ± .029 | **0.737** ± .002 |
| CLIP-Adapter | **0.670** ± .044 | 0.692 ± .044 | 0.699 ± .006 | **0.730** ± .015 | 0.698 ± .026 | 0.730 ± .007 |
| TaskRes | 0.626 ± .077 | **0.694** ± .028 | **0.701** ± .004 | 0.704 ± .023 | **0.710** ± .012 | 0.735 ± .006 |
| GraphAdapter | 0.656 ± .056 | 0.675 ± .002 | 0.694 ± .036 | 0.714 ± .005 | 0.699 ± .020 | 0.729 ± .010 |
| CLAP | 0.571 ± .068 | 0.656 ± .042 | 0.679 ± .030 | 0.645 ± .112 | 0.691 ± .035 | 0.712 ± .017 |
| Tip-Adapter-F | 0.556 ± .123 | 0.681 ± .026 | 0.677 ± .013 | 0.692 ± .047 | 0.677 ± .023 | 0.651 ± .020 |
| PlantCaFo-style cache | 0.666 ± .019 | 0.666 ± .030 | 0.632 ± .025 | 0.694 ± .029 | 0.667 ± .015 | 0.611 ± .003 |
| Tip-Adapter (no training) | 0.610 ± .028 | 0.685 ± .073 | 0.687 ± .003 | 0.713 ± .023 | 0.650 ± .023 | 0.674 ± .003 |
| Linear probe | 0.453 ± .181 | 0.612 ± .100 | 0.647 ± .038 | 0.692 ± .043 | 0.692 ± .035 | 0.731 ± .007 |

Zero-shot CLIP L/14 with our descriptions: 0.605. Leakage check with L/14: random split vs video split, 1-NN 0.728
vs 0.634, linear probe 0.798 vs 0.687.

**Reading it honestly:**
- The bigger backbone lifts **every** method by about 8-15 points (e.g. PRGA + DINOv2 at K = 16: 0.601 → 0.679).
- **PRGA does not win on bees.** It is tied at K = 1-2 (and the most stable there, std ≈ .01-.02), best or tied with
  all photos (0.737), but **3-6 points behind CLIP-Adapter and TaskRes at K = 4-16**.
- Most methods now sit at 0.68-0.73: the backbone, not the adapter, sets the level on this task. A fine-tuned CNN
  with all photos (0.770, earlier run) is still higher.
- As expected from the pilot, the gain from 1 seed (PRGA + DINOv2 0.66 / 0.69 at K = 4 / 16) held at K = 16
  (0.679) but shrank at K = 4 (0.663).

## What this shows (first round, CLIP B/16)

1. **This task is hard for every frozen-feature method.** With 1-16 photos per class nothing gets far above 0.60
   macro-F1 (chance is about 0.50). CLIP's features hardly separate infested from healthy bees: zero-shot is 0.56 and
   a linear probe on all 2,138 training photos reaches only 0.66.
2. **PRGA does not win at 1-2 shots on bees.** Tip-Adapter-F is best at K = 1 (0.567 vs 0.454) and PRGA's variance is
   high there. From K = 4 up, PRGA, PRGA + DINOv2 and Tip-Adapter-F are within the seed spread of each other.
3. **With all training photos, the fine-tuned CNN is best (0.770).** It can learn the mite's pixels directly; the
   frozen CLIP features cannot. Among frozen-feature methods, PRGA is best with all data (0.699; 0.725 in its default
   configuration).
4. **Region nodes help here, unlike on onions**, but only with enough data: with all photos, OWLv2 regions beat a grid
   by 4.9 points and the object box alone by 5.2 points. At K ≤ 16 the differences are within noise. This supports the
   reason for choosing this domain, with a caveat: it shows up only at the "all" setting.
5. **The base paper's graph again hurts:** its own no-graph control is better at every K.
6. **Leakage matters here too:** a random split would overstate scores by 7-10 points.

Honest bottom line: the bee domain is a **negative result for few-shot CLIP methods, PRGA included**. It points to
the next step: features that see the mite (higher resolution crops around OWLv2 spots, or a backbone fine-tuned on
bee images) rather than a better adapter.

## Files

- `classes.json`, `prompts/` (class descriptions written before any bee result existed, see `prompts/notes.md`;
  OWLv2 prompts; text templates)
- `data/clean/meta_clean.csv` (labels, video, mite boxes), `data/clean/split_summary.csv`, `splits/`
- `regions/regions.json` (OWLv2 boxes), `results/` (all runs, tables and Kaggle logs)
- Re-run: push `kaggle/kernel_bees_run` (it downloads the data from Zenodo if needed).
