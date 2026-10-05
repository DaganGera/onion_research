# Few-shot onion bulb grading with CLIP

Can a model tell **healthy from unhealthy onions** (red and white, so 4 classes) after seeing only **1 to 16 photos
per class**? This repository answers that on the public onion bulb dataset. It contains:

- **a clean evaluation**: duplicate photos and re-shots of the same scene are removed, and no scene appears in both
  training and test data;
- **a replication of the base paper**, Ahmad et al. 2025, *Advancing Cache-Based Few-Shot Classification via
  Patch-Driven Relational Gated Graph Attention* ([arXiv:2512.12498](https://arxiv.org/abs/2512.12498)), with its
  ablations;
- **the methods that paper compares against** (Tip-Adapter-F, TaskRes, GraphAdapter, CLIP-Adapter, CLAP), plus CoOp,
  a PlantCaFo-style two-backbone cache, BioCLIP, SCOLD and a fine-tuned CNN. Links and short summaries:
  [docs/COMPARED_METHODS.md](docs/COMPARED_METHODS.md);
- **PRGA** (Prompt-grounded Region Graph Adapter), our model, and **PRGA + DINOv2 cache**, its two-backbone version.

All numbers are macro-F1 on 7,612 held-out test photos, mean ± std over 3 random support sets. Every design choice was
made on validation data; the test set was used once per configuration.

## Results

| method | backbones | K=1 | K=2 | K=4 | K=8 | K=16 | all (≈1.8k) |
|---|---|---|---|---|---|---|---|
| **PRGA + DINOv2 cache** | CLIP + DINOv2 | **0.813** ± .059 | 0.858 ± .029 | **0.905** ± .016 | **0.930** ± .040 | **0.962** ± .016 | **0.985** ± .001 |
| **PRGA** | CLIP | 0.799 ± .047 | 0.820 ± .044 | 0.890 ± .017 | 0.886 ± .065 | 0.937 ± .013 | 0.983 ± .002 |
| PlantCaFo-style cache | CLIP + DINOv2 | 0.797 ± .053 | **0.867** ± .017 | 0.891 ± .026 | **0.930** ± .025 | 0.943 ± .026 | 0.968 ± .005 |
| CoOp | CLIP | 0.687 ± .042 | 0.628 ± .033 | 0.828 ± .042 | 0.869 ± .044 | 0.950 ± .002 | 0.970 ± .009 |
| CLIP-Adapter | CLIP | 0.780 ± .041 | 0.783 ± .055 | 0.833 ± .019 | 0.891 ± .039 | 0.944 ± .006 | 0.980 ± .004 |
| TaskRes | CLIP | 0.694 ± .086 | 0.761 ± .035 | 0.843 ± .020 | 0.884 ± .045 | 0.940 ± .007 | 0.975 ± .010 |
| CLAP | CLIP | 0.689 ± .108 | 0.737 ± .032 | 0.823 ± .039 | 0.883 ± .040 | 0.937 ± .010 | 0.967 ± .004 |
| Tip-Adapter-F | CLIP | 0.764 ± .033 | 0.804 ± .058 | 0.849 ± .016 | 0.888 ± .009 | 0.925 ± .003 | 0.888 ± .019 |
| GraphAdapter (single-layer variant) | CLIP | 0.745 ± .048 | 0.767 ± .047 | 0.785 ± .038 | 0.826 ± .011 | 0.884 ± .023 | 0.945 ± .014 |
| Ahmad et al. 2025 (replication) | CLIP | 0.609 ± .057 | 0.678 ± .039 | 0.709 ± .024 | 0.792 ± .079 | 0.820 ± .030 | 0.662 ± .126 |
| Linear probe | CLIP | 0.423 ± .095 | 0.568 ± .089 | 0.736 ± .077 | 0.862 ± .052 | 0.929 ± .012 | 0.967 ± .011 |
| EfficientNet-B0, fine-tuned | — | 0.270 ± .014 | 0.326 ± .045 | 0.422 ± .052 | 0.488 ± .046 | 0.550 ± .020 | 0.892 ± .018 |
| Tip-Adapter-F on BioCLIP | BioCLIP | 0.397 ± .039 | 0.470 ± .006 | 0.511 ± .041 | 0.577 ± .048 | 0.708 ± .050 | 0.591 ± .034 |
| Tip-Adapter-F on SCOLD | SCOLD | 0.216 ± .032 | 0.237 ± .022 | 0.325 ± .044 | 0.474 ± .011 | 0.630 ± .035 | 0.434 ± .046 |

Zero-shot CLIP (no training photos at all): 0.751 with our class descriptions, 0.553 with generic templates, 0.548
with "a photo of a [CLASS]". With only 3 seeds, differences below about 2 points are not significant.

![Every method at every K](figures/comparison_dots.png)

**In short.** PRGA + DINOv2 cache is first or tied for first at every K except K = 2, where the PlantCaFo-style cache
is 0.9 points ahead (within the seed spread). Among the CLIP-only methods, PRGA is best at K = 1, 2, 4 and with all
data. Every method from the base paper's comparison table does better than the base paper's own model on this
dataset.

### Fairness check: same GPU, every method picks its best epoch

In the table above PRGA and the base paper keep their best checkpoint on validation, while the other trained methods
use their last epoch, and the runs were on Kaggle T4 GPUs. `src/18_equal_training.py` removes both differences: every
method is re-run on one GPU (RTX 3050 laptop), the comparison methods train for twice their usual number of epochs, and
each keeps the best of 10 validation checkpoints. Macro-F1, mean ± std over 3 seeds (`results/equal_training.md`):

| method | K=1 | K=2 | K=4 | K=8 | K=16 | all |
|---|---|---|---|---|---|---|
| **PRGA + DINOv2 cache** | 0.793 ± .063 | 0.862 ± .010 | **0.890** ± .024 | **0.930** ± .030 | **0.955** ± .017 | **0.985** ± .001 |
| **PRGA** | 0.782 ± .055 | 0.827 ± .049 | 0.871 ± .022 | 0.888 ± .059 | 0.932 ± .018 | **0.985** ± .001 |
| PlantCaFo-style cache | **0.796** ± .052 | **0.869** ± .012 | 0.881 ± .035 | 0.924 ± .027 | 0.943 ± .021 | 0.974 ± .014 |
| GraphAdapter | 0.795 ± .026 | 0.805 ± .034 | 0.808 ± .014 | 0.853 ± .016 | 0.896 ± .016 | 0.954 ± .015 |
| CLIP-Adapter | 0.787 ± .035 | 0.803 ± .023 | 0.811 ± .018 | 0.896 ± .047 | 0.947 ± .009 | 0.978 ± .003 |
| Tip-Adapter-F | 0.759 ± .033 | 0.808 ± .054 | 0.827 ± .026 | 0.869 ± .054 | 0.923 ± .006 | 0.935 ± .012 |
| TaskRes | 0.741 ± .097 | 0.779 ± .043 | 0.837 ± .018 | 0.880 ± .047 | 0.942 ± .010 | 0.972 ± .014 |
| CLAP | 0.704 ± .099 | 0.759 ± .026 | 0.805 ± .033 | 0.885 ± .038 | 0.936 ± .011 | 0.973 ± .008 |
| CoOp | 0.666 ± .069 | 0.706 ± .056 | 0.813 ± .048 | 0.886 ± .031 | 0.943 ± .006 | 0.976 (1 seed) |

Best-epoch selection helps some methods (TaskRes +5 points and GraphAdapter +5 points at K = 1, Tip-Adapter-F +4
points with all data). After it, **PRGA + DINOv2 cache is still best at K = 4, 8, 16 and with all data. At K = 1 and
2 it is tied with the PlantCaFo-style cache** (and with GraphAdapter at K = 1): the gaps of 0.3-0.7 points are far
inside the seed spread. Laptop and Kaggle numbers for the same code differ by up to about 2 points (GPU arithmetic),
so compare within one table, not across them. CoOp with all photos was stopped early to free the GPU (2 of 3 seeds at
the usual length, 1 at double length); `python 18_equal_training.py --methods coop` resumes it.

## How it works

Diagrams of every part, with the code explained step by step and the parameter count of every model:
[docs/MODELS_EXPLAINED.md](docs/MODELS_EXPLAINED.md).

![The whole pipeline](figures/arch_pipeline.png)

The same pipeline as a graph of nodes, following one test photo:

![Pipeline as nodes](figures/arch_pipeline_nodes.png)

| base paper | PRGA | PRGA + DINOv2 cache |
|---|---|---|
| ![](figures/arch_base_paper.png) | ![](figures/arch_prga.png) | ![](figures/arch_prga_dinov2.png) |

The same in text form:

### The whole pipeline

```
 ┌──────────────────┐    ┌──────────────────────┐    ┌──────────────────────────┐
 │  12,260 photos   │───►│  1. Clean the data   │───►│  2. Split 20 % / 80 %    │
 │  4 classes       │    │  remove duplicates   │    │  no scene in both parts  │
 └──────────────────┘    │  and re-shots        │    └────────────┬─────────────┘
                         └──────────────────────┘                 │
                                                                  ▼
 ┌────────────────────────────────────────────────────────────────────────────────┐
 │  3. Frozen feature extractors (run once, results saved to disk)                │
 │     CLIP ViT-B/16 (image + text)    DINOv2-S (image)    OWLv2 (finds regions)  │
 └──────────────────────────────────────┬─────────────────────────────────────────┘
                                        ▼
 ┌────────────────────────────────────────────────────────────────────────────────┐
 │  4. PRGA graph: the only part that is trained (1.1 M parameters)               │
 │     links the photo, its regions and the 4 class descriptions                  │
 └──────────────────────────────────────┬─────────────────────────────────────────┘
                                        ▼
 ┌────────────────────────────────────────────────────────────────────────────────┐
 │  5. Score each class: zero-shot + CLIP cache + prototypes + DINOv2 cache       │
 └──────────────────────────────────────┬─────────────────────────────────────────┘
                                        ▼
                healthy red / unhealthy red / healthy white / unhealthy white
```

Steps 1-2 happen once for the dataset. Step 3 runs the big pretrained networks once per photo; they are never trained.
Steps 4-5 are the few-shot model: it trains in seconds because it only sees the saved feature vectors.

### Box 1: cleaning (`src/02_clean_dataset.py`)

```
 photos ──► same bytes? (MD5) ──► CLIP finds look-alike pairs ──► RootSIFT keypoints ──► RANSAC geometry check
            drop copies           (cosine ≥ 0.92, same class)     matched on the GPU      ≥ 10 inliers = same scene
                                                                                                │
 11,676 photos ◄── drop burst shots (CLIP ≥ 0.99) ◄── group all same-scene photos ◄────────────┘
```

Of 1.4 M candidate pairs, 79,194 are confirmed as the same scene (none across classes). 27 byte-identical copies and
557 near-identical burst frames are removed. Details: [docs/DATA_CLEANING.md](docs/DATA_CLEANING.md).

### Box 2: the split (`src/04_split.py`)

```
 same-scene graph ──► Louvain communities ──► 20 % of communities ──► remove every test photo that is
 (photos + pairs)     (clusters of scenes)    become the training     linked to a training photo
                                              pool (2,388 photos)     → test set: 7,612 photos
                                                     │
                                                     ▼
                    support set: K photos per class, each from a different community (3 random draws)
                                 "all" = the pool minus 20 % of its communities
                    validation:  the rest of the pool, minus photos linked to the support set
```

Why it matters: on a naive random split, a 1-nearest-neighbour classifier scores 0.985 macro-F1 because near-copies of
training photos sit in the test set. On this split it scores 0.883.

### Box 3: features (`src/05_detect_regions.py`, `src/06_extract_features.py`, `src/07_text_embeddings.py`)

```
 photo ──► CLIP image encoder ─────────────────────────────────────────► g    whole photo, 512 numbers
 photo ──► 10 random crops + flips ──► CLIP ───────────────────────────► 10 × 512   (training photos only)
 photo ──► OWLv2 + text prompts ──► boxes: object, ≤ 2 onions, ≤ 2 spots ──► crop ──► CLIP ──► 5 × 512
 photo ──► DINOv2-S ───────────────────────────────────────────────────► d    384 numbers
 8 descriptions per class ──► CLIP text encoder ──► average (+ ½ template) ─► T  4 × 512 (one per class)
```

OWLv2 is an open-vocabulary detector: you give it text ("an onion", "black mould", "a rotten spot") and it returns
boxes. The class descriptions are in `prompts/descriptors.json`, for example *"a red onion covered with black powdery
mould"*.

### Box 4: the PRGA graph (`src/fewshot.py`: `PRGANet`, `GatedGAT`, `Readout`)

```
 nodes (up to 10) ─┬─ [whole photo] [object] [onion 1] [onion 2] [spot 1] [spot 2]      image nodes
                   └─ [healthy red] [unhealthy red] [healthy white] [unhealthy white]   text nodes
        │  every node: Linear 512 → 256  +  a learned "node type" vector
        │  every pair of nodes gets an edge feature: their cosine similarity
        ▼
 ┌─ gated graph-attention layer, applied twice ───────────────────────────────────┐
 │  attention   score_ij = LeakyReLU(a·Wh_i + a'·Wh_j + u·e_ij), softmax over j   │
 │  message     m_i = Σ_j attention_ij · W h_j        (4 heads)                   │
 │  gate        z_i = sigmoid(W_g [h_i, m_i])         how much of the message     │
 │  update      h_i = LayerNorm(z_i · m_i + (1 − z_i) · h_i)                      │
 └────────────────────────────────────────────────────────────────────────────────┘
        │                                            │
        ▼                                            ▼
 image nodes → mean + max + attention pool    text nodes → Linear 256 → 512, added to T
 → Linear → added to g → f̂                    → T̂ : the 4 class descriptions adapted to this photo
```

### Box 5: the score (`PRGA._logits` in `src/fewshot.py`, `fused` in `src/13_prga_dinov2.py`)

```
 score_c =  100 · g·T_c                              zero-shot: does the photo match class c's description?
         +  α Σ_j exp(−β (1 − g·K_j)) L̃_jc           CLIP cache: how close is it to the support photos of class c?
         +  γ · 100 · g·T̂_c                          prototypes: the description, adapted to this photo
         +  a₂ Σ_j exp(−b₂ (1 − d·D_j)) L̃_jc         DINOv2 cache: the same closeness, with DINOv2 features
```

`K_j` are the support photos run through the graph (f̂), `D_j` their DINOv2 vectors, `L̃` the class labels with each
class's column divided by its number of photos. The predicted class is the highest score. Validation chose to query
the cache with the plain CLIP vector `g` of the test photo; the graph acts through the cache keys `K_j` and the
prototypes `T̂`.

### Training

```
 for epoch in 1 … 60:
     for each of the 10 augmented views, in random order:
         keys  = graph(support photos)               query = this view of every support photo
         loss  = cross-entropy(score(query), true class), label smoothing 0.1
         AdamW step (lr 0.001, weight decay 1e-4, one-cycle schedule)
     every 5 epochs: macro-F1 on validation → keep the best weights
 then: pick α, β by grid search on validation
 then: DINOv2 cache keys fine-tuned for 20 epochs, a₂, b₂ picked by grid search on validation
```

Training time for one support set is seconds to a minute on a laptop GPU. Epochs used by the other methods:

| method | epochs |
|---|---|
| PRGA | 60, best of every 5th epoch on validation |
| DINOv2 cache keys (in PRGA + DINOv2) | 20 |
| base paper replication | up to 20, best epoch on validation (epoch 0 = untrained is allowed) |
| Tip-Adapter-F, PlantCaFo-style cache | 20 |
| CLIP-Adapter, TaskRes, GraphAdapter | 50 |
| CLAP | 300 |
| CoOp | 50 / 50 / 100 / 100 / 200 for K = 1 / 2 / 4 / 8 / 16, 20 with all data |
| EfficientNet-B0 | 40, or 15 with all data |

An epoch is one pass over the augmented views of every support photo.

## Try it

```
uv sync
cd src
python demo.py              # trains PRGA + DINOv2 cache on 4 photos per class, tests on 7,612 photos
python demo.py --mistakes   # show photos it gets wrong
python demo.py --k 1        # one photo per class
```

It needs the cached features in `features/` and the photos in `data/raw/` (see Reproduce). It prints accuracy and
macro-F1 and saves `figures/demo_predictions.png`. Scores can differ by up to about one point from the table because
the table was computed on Kaggle T4 GPUs and GPU arithmetic is not bit-identical across cards.

## What we found

**The base paper's graph does not help on onions** ([docs/BASE_PAPER_REPLICATION.md](docs/BASE_PAPER_REPLICATION.md)).
Implemented from the paper's equations and figures, with every unspecified detail documented. The same training recipe
without the graph scores higher at every K (14 of 18 runs), and the paper's attention ablations make no measurable
difference. Likely reason: training refines the query with the graph, but testing uses the plain CLIP vector; the two
have a cosine of only about 0.78, so training moves the cache keys away from where test photos lie.

**What drives PRGA** (`results/ablations_K1-4.csv`). Replacing the class descriptions with generic templates lowers
K = 1 macro-F1 from 0.796 to 0.622; removing the text nodes lowers it to 0.769. Replacing the OWLv2 regions with the
object box alone or with a 3 × 3 grid makes no significant difference (0.789 / 0.799).

**A second backbone helps.** Adding the DINOv2 cache was chosen on held-out halves of the validation sets (0.890 →
0.909) and improves PRGA in 16 of 18 test comparisons (2 ties).

**Domain-specific backbones transfer poorly.** BioCLIP and SCOLD do much worse than plain CLIP on bulbs; SCOLD's
zero-shot score is below chance.

## Inference cost

Median time per photo, one photo at a time (`src/17_cost_benchmark.py`, `results/cost_benchmark.csv`):

| component | GPU (RTX 3050 Laptop, fp16) | CPU (8 threads) | parameters |
|---|---|---|---|
| CLIP ViT-B/16, whole photo | 8.9 ms | 170 ms | 86 M |
| CLIP ViT-B/16, 5 region crops | 21.9 ms | 761 ms | (shared) |
| DINOv2-S | 6.1 ms | 64 ms | 22 M |
| PRGA head | 1.3 ms | 0.7 ms | 1.1 M |
| OWLv2-B/16 detection | 126.9 ms | not measured | 155 M |

The full pipeline takes about 165 ms per photo on this GPU, and OWLv2 is about 77 % of that. Since the regions did not
measurably improve accuracy in the ablations, a version without OWLv2 (about 16 ms per photo) is the obvious next
step; its accuracy has not been tested yet.

## Repository

```
src/
  01_audit_photos.py        list every photo, group identical ones
  02_clean_dataset.py       remove duplicates and same-scene re-shots      (box 1)
  03_check_cleaning.py      pictures for checking the cleaning by eye
  04_split.py               scene-disjoint split, support and validation sets   (box 2)
  05_detect_regions.py      OWLv2 boxes                                     (box 3)
  06_extract_features.py    CLIP / DINOv2 / BioCLIP / SCOLD features        (box 3)
  07_text_embeddings.py     class description vectors                       (box 3)
  08_baselines.py           zero-shot, linear probe, Tip-Adapter(-F), PlantCaFo-style, leakage check
  09_base_paper.py          the base paper's model, trained and tested
  10_comparison_methods.py  CLIP-Adapter, TaskRes, CLAP, GraphAdapter, CoOp
  11_prga.py                PRGA and its ablations                          (boxes 4-5)
  12_prga_select.py         choose PRGA's configuration on validation
  13_prga_dinov2.py         PRGA + DINOv2 cache, the final model
  14_cnn_baseline.py        EfficientNet-B0
  15_tables_and_figures.py  results table and figures
  16_comparison_figure.py   the dot-chart comparison
  17_cost_benchmark.py      time per photo
  18_equal_training.py      fairness check: every method picks its best epoch on validation
  19_architecture_figures.py  the architecture diagrams
  demo.py                   train and test the final model in one go

  common.py      paths, classes, seeds, metrics          fewshot.py   feature store, baselines, PRGA
  backbones.py   loads the frozen networks               adapters.py  comparison methods
  basepaper.py   the base paper's model                  harness.py   runs a method over all K and seeds
kaggle/          scripts that ran everything on Kaggle (2 × T4 GPUs)
docs/            models explained, code guide, data cleaning, base-paper replication, compared methods
prompts/  regions/  splits/  data/clean/  results/  figures/
```

A file-by-file walk through the code, with the shapes of every tensor: [docs/CODE_GUIDE.md](docs/CODE_GUIDE.md).

## Reproduce

```
uv sync
# photos: data/raw/onion_bulbs/Onion Image Dataset/2. Bulb   (or set ONION_RAW to their folder)
cd src
python 01_audit_photos.py
python 06_extract_features.py --backbone clip --mode global
python 01_audit_photos.py --refine
python 02_clean_dataset.py && python 03_check_cleaning.py --bands
python 04_split.py && python 03_check_cleaning.py --leakcheck
python 05_detect_regions.py
python 06_extract_features.py --backbone clip --mode aug        # also: --mode grid, --mode regions
python 06_extract_features.py --backbone dinov2 --mode global   # also --mode aug; the same for bioclip and scold
python 07_text_embeddings.py                                   # also --backbone bioclip / scold
bash ../kaggle/run_pipeline.sh                                 # every method, all K and seeds, tables, figures
bash ../kaggle/run_extra.sh                                    # PRGA + DINOv2 cache
```

Every training script can be stopped and restarted: finished (method, K, seed) runs are skipped.

## Limitations

- One dataset (one phone, one site, four classes); other crops and cameras are untested.
- Three seeds per setting, so small differences are within noise.
- The base paper has no public code; the replication fills documented gaps.
- The comparison methods are our re-implementations of each paper's main equation on frozen features, not the
  authors' code. Ta-Adapter and CAA were not re-implemented.
- Same-scene detection also groups photos that only share a textured background, which makes the split conservative.
  A few test photos may still show an onion from the training pool without a detectable geometric match.
- Labels are per photo, not per onion.

## Data

Kulkarni et al., onion bulb image dataset, Mendeley Data 2025, DOI
[10.17632/42bcyncfhy.1](https://doi.org/10.17632/42bcyncfhy.1), CC BY 4.0.
