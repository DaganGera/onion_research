# Code guide

A walk through the code, in the order the data flows. It goes from "what does this file do" down to "what shape is
this tensor and why". Read it next to the code. The pictures of each stage are in the README ("How it works").

## 0. Vocabulary

| word | meaning here |
|---|---|
| **K, shots** | number of training photos per class (1, 2, 4, 8, 16 or "full" ≈ 1.8k in total) |
| **support set** | the K photos per class the model is trained on |
| **pool** | the 20 % of the data that support and validation photos are drawn from; the other 80 % is the test set |
| **seed** | which random support set (1, 2 or 3). Every result is the mean over 3 seeds |
| **feature / embedding** | the vector a frozen network outputs for a photo or a sentence. CLIP: 512 numbers, DINOv2-S: 384 |
| **L2-normalised** | scaled to length 1, so a dot product `a·b` is the cosine similarity (1 = same direction) |
| **cache** | Tip-Adapter's memory: support-photo vectors (keys) + their labels (values) |
| **macro-F1** | F1 per class, averaged over the 4 classes, so every class counts equally |
| **validation** | photos from the pool, not in the support set, used for every choice. The test set is never used for choices |

## 1. Shared files (read these first)

### `src/common.py`: paths and helpers
- `ROOT`, `RAW` (photo folder, `ONION_RAW` overrides it), `SPLITS`, `FEATS`, `RESULTS`, ...: every path in one place.
- `CLASS_MAP` / `CLASSES`: folder names → the 4 class names. Single-onion and pile photos share a class.
- `SEEDS = [1, 2, 3]`, `SHOTS = [1, 2, 4, 8, 16]`.
- `CLIP_NAME = "ViT-B-16-quickgelu"`: OpenAI's CLIP weights need the QuickGELU variant in open_clip.
- `save_feats` / `load_feats`: features are saved as `.pt` dictionaries. `save_feats` writes to a temporary file and
  renames it, so a crash never leaves a half-written file.
- `l2n(x)`: divide by the vector length.
- `metrics(y_true, y_pred)`: accuracy and macro-F1.

### `src/backbones.py`: the frozen networks
`Backbone("clip" | "dinov2" | "bioclip" | "scold")` loads a network in half precision with one interface:
`.transform` (resize 224 + centre crop), `.aug_transform` (random resized crop, scale 0.5-1, + horizontal flip),
`.encode_image(x)`, `.encode_text(list_of_strings)`. Both encoders return L2-normalised vectors. SCOLD needed a
workaround: its code downloads weights that its checkpoint overwrites anyway, so the architecture is built offline and
`scold.pth` is loaded on top.

### `src/harness.py`: runs any method over all K and seeds
`run(name, make, store)` does, for every K and seed:
1. load `support_K{K}_s{seed}.csv` (with augmented views) and `val_K{K}_s{seed}.csv`;
2. `model = make().fit(support, val)`;
3. `probs = model.predict(test)` on all 7,612 test photos;
4. append accuracy and macro-F1 to `results/runs/<name>.jsonl`, save the probabilities to `results/preds/`.
If a (K, seed) line is already in the `.jsonl`, it is skipped, so a run can be stopped and restarted.

## 2. Data preparation

| script | input | output | what to know |
|---|---|---|---|
| `01_audit_photos.py` | photo folders | `data/meta.csv` | one row per photo: path, class, MD5, perceptual hash, size, `group`. Groups merge identical bytes, near-identical hashes, and (with `--refine`) CLIP clusters inside a class. `UnionFind` merges "a and b belong together" pairs into groups |
| `02_clean_dataset.py` | `meta.csv`, `clip_global.pt` | `data/clean/` | see below |
| `03_check_cleaning.py` | `data/clean/` | `figures/clean_*.png` | side-by-side pictures of pairs, to check thresholds by eye |
| `04_split.py` | `meta_clean.csv`, verified pairs | `splits/*.csv` | see below |

**`02_clean_dataset.py`, step by step** (constants at the top of the file):
1. `_md5_size`: open every file; identical MD5 = identical bytes.
2. Candidate pairs: for every photo, CLIP cosine against all others, in chunks of 2048 on the GPU. Same-class pairs
   ≥ `CAND = 0.92` are candidates, cross-class ≥ `CAND_X = 0.94` too (to find label conflicts).
3. `_sift`: 400 RootSIFT keypoints per photo at 640 px. RootSIFT = SIFT with each descriptor L1-normalised and
   square-rooted, which matches better. Cached in `features/sift_rootsift_400.npz`.
4. `gpu_matches`: for each candidate pair, match keypoints with Lowe's ratio test (best match clearly better than the
   second best, `RATIO = 0.8`) and keep only mutual nearest neighbours.
5. `_ransac`: fit a homography with RANSAC (OpenCV). ≥ `MIN_INLIERS = 10` geometrically consistent matches = same
   scene. A photo of a *similar* onion does not pass this; the same onion on the same cloth does.
6. `UF` (union-find) merges verified pairs into groups; inside a group, photos with CLIP ≥ 0.99 to a kept photo are
   burst duplicates and are removed.

**`04_split.py`**:
- `communities`: a graph per class (nodes = audit groups, edges = verified pairs, weight = inliers), split into
  Louvain communities. Communities, not single photos, are the unit of the split.
- Communities go to the pool in random order until it has 20 % of the class's single-onion photos and 20 % of its pile
  photos.
- `linked`: a test photo is removed if it has a verified pair or CLIP cosine ≥ 0.95 with any pool photo.
- Support sets: for each class, K different communities, one random photo from each. Validation: the rest of the
  pool, minus photos linked to the support set (`val_without_links`).
- Also writes a naive random split (`random_*.csv`) used only by the leakage check in `08_baselines.py`.

## 3. Features (run once)

**`05_detect_regions.py`**: OWLv2 receives each photo and text prompts. `OBJECT_PROMPTS` ("an onion", "a pile of
onions") give the object box; `INSTANCE_PROMPTS` give single onions (up to 2); `SPOT_PROMPTS` ("black mould", "a rotten
spot", ...) give small spots (up to 2). Score thresholds and maximum box sizes are constants at the top. OWLv2 pads
photos to a square, so boxes are converted using `max(w, h)`. Output: `regions/regions.json`.

**`06_extract_features.py`**, `--mode`:
- `global`: every photo, centre crop → `[N, 512]`.
- `aug`: pool photos only, 10 random crops/flips each → `[N_pool, 10, 512]`. These are the training views.
- `grid`: 3 × 3 crops → `[N, 9, 512]` (only for the grid ablation).
- `regions`: object box + 4 region crops, each with 15 % margin → `feats [N, 5, 512]`, `mask [N, 5]` (1 = real box,
  0 = empty slot), `geom [N, 5, 5]` (centre x, centre y, width, height, score), `kind` (1 object, 2 onion, 3 spot).

**`07_text_embeddings.py`**: two text vectors per class.
- `template`: average of 5 sentences like "a photo of a healthy red onion."
- `desc` (used by almost everything): average of the 8 hand-written descriptions in `prompts/descriptors.json`, plus
  half of the template vector, re-normalised. This is the `T` matrix `[4, 512]`.

## 4. `src/fewshot.py`: the core

### Data containers
- `Batch`: one set of photos as tensors: `y` labels `[N]`, `g` global CLIP `[N, 512]`, `aug` views `[N, 10, 512]`,
  `reg/mask/geom` region nodes, `g2/aug2` DINOv2. `len(batch)` = number of photos.
- `Store`: loads the feature files once and builds a `Batch` for any split CSV, looking rows up **by path** (never by
  row number, which changed when the dataset was cleaned).

### Small helpers
- `onehot(y)` → `[N, 4]`.
- `balance(L)`: divides each class column by its count. With K photos per class this changes nothing, but in "full" the
  classes have different sizes, and without it the largest class would win the cache vote.
- `tip_logits(q, keys, L, T, α, β)` = `100·q·Tᵀ + α·exp(−β(1 − q·keysᵀ))·balance(L)`. This is Tip-Adapter's
  formula. `exp(−β(1 − cos))` turns a cosine into an affinity between 0 and 1 that falls off quickly for photos that
  are not close; β sets how quickly.
- `search_ab`: try every α in `ALPHAS` and β in `BETAS`, keep the best validation macro-F1.

### Baselines
- `ZeroShot`: `softmax(100·g·Tᵀ)`. No training.
- `LinearProbe`: scikit-learn logistic regression on all augmented views; C ∈ {1, 10, 100} chosen on validation.
- `TipAdapter`: keys = mean of the 10 views per support photo; only α, β are searched.
- `TipAdapterF`: the keys are an `nn.Parameter`, trained 20 epochs with AdamW (lr 1e-3, cosine schedule) on the
  views, α = β = 5 during training, then α, β searched on validation.
- `PlantCaFoLite`: the same, with a CLIP cache and a DINOv2 cache whose scores are added.

### PRGA's network: `PRGANet.forward(g, nodes, mask, geom)`
Shapes for a batch of B photos, with the defaults H = 256, 2 layers, 4 heads:

1. `img = [g, nodes]` → `[B, 6, 512]`: the whole photo + object + 4 region slots. `imask [B, 6]` marks real nodes.
2. During training, each region node is hidden with probability 0.2 (`node_drop`); the whole-photo node never is.
   This keeps the model from relying on one region.
3. `h = inp(img) + type_emb(kind)` → `[B, 6, 256]`. `type_emb` tells the network what each node is (0 whole
   photo, 1 object, 2 onion, 3 spot, 4 text). With `use_geom=True` a small MLP of the box position is added; the
   selected configuration has `use_geom=False`.
4. Edge features `e [B, N, N, 4]`: IoU, centre distance, score product, cosine. With `use_geom=False` the first three
   are zeroed, so the edges are just the cosine similarity between the two nodes' CLIP vectors.
5. Text nodes: the 4 class vectors `T` go through the same `inp` + type 4 and are appended → `h [B, 10, 256]`.
   Image–text edges are cosines `img·T`, text–text edges `T·T`.
6. Two `GatedGAT` layers (below) update all 10 nodes.
7. `readout`: over the image nodes only, concatenate the masked mean, the masked max and an attention-weighted sum →
   `[B, 768]` → Linear → `[B, 512]`. Then `f̂ = l2n(g + readout)`: a correction added to the original vector.
8. Text nodes → `text_out` Linear → `[B, 4, 512]`, `T̂ = l2n(T + that)`: the class descriptions adjusted for this
   photo. Again a correction on top of the original.

Steps 7-8 add corrections to the frozen vectors instead of replacing them, so an untrained network starts close to
plain CLIP.

### One graph layer: `GatedGAT.forward(h, e, node_mask)`
```
Wh            = W h, split into 4 heads of 64                       [B, N, 4, 64]
score_ij      = LeakyReLU(a_src·Wh_i + a_dst·Wh_j + U e_ij)          [B, 4, N, N]   (U: edge features → one number per head)
score_ij      = −∞ where node j is not real                           so empty slots get zero attention
attention     = softmax over j, with dropout                          each node decides whom to listen to
m_i           = Σ_j attention_ij Wh_j                                 [B, N, 256]    the message
z_i           = sigmoid(W_g [h_i ; m_i])                              per-feature gate between 0 and 1
h_i (new)     = LayerNorm(z_i · m_i + (1 − z_i) · h_i)                keep part of the old state, take part of the message
```
This is graph attention (GAT) plus edge features and a gate. The gate lets a node ignore messages that do not help.

### PRGA's score: `PRGA._logits`
```
zs     = 100 · g·Tᵀ                                       zero-shot term (always the plain CLIP vector)
cache  = α · exp(−β (1 − fq·keysᵀ)) · balance(L)           keys = support photos through the graph (f̂)
proto  = γ · 100 · fq·T̂ᵀ                                   image-conditioned class descriptions
```
`fq` is the query. With `test_graph=False` (the configuration chosen on validation) `fq = g`: the test photo is not
passed through the graph for the query, but its regions still produce `T̂`.

### PRGA training: `PRGA.fit(tr, va)`
1. New network; α = 5, β = 5, γ = 0.5 are learnable too.
2. `g_clean` = mean of each support photo's 10 views.
3. 60 epochs. Each epoch goes through the 10 views in random order, one step per view, with **all support photos in
   one batch**:
   - `keys = net(g_clean, regions)`: support photos through the graph;
   - query = view v of every support photo, through the graph for `T̂`;
   - loss = cross-entropy of `_logits` against the true labels, label smoothing 0.1 (targets of 0.925 instead of
     1, which reduces overconfidence);
   - AdamW (lr 1e-3, weight decay 1e-4) with a one-cycle schedule (warm up for 10 %, then decay).
4. Every 5th epoch: compute the keys in eval mode, macro-F1 on validation, keep a copy of the best weights.
5. Load the best weights, then grid-search α and β on validation (γ stays as learned).

`predict` computes `T̂` for the test photos and applies `_logits`.

## 5. The final model: `src/13_prga_dinov2.py`

- `CHOSEN = dict(test_graph=False, use_geom=False)`: the winner of `12_prga_select.py`
  (`results/prga_selection_val.csv`).
- `prga_logits(m, batch)`: PRGA's scores before softmax.
- `dino_keys(tr, finetune)`: DINOv2 cache keys = mean of the 10 DINOv2 views per support photo; with `finetune=True`
  trained 20 epochs (AdamW, cosine schedule) with the loss `CE(8·exp(−5(1 − x·k))·L̃, y)`.
- `fused(base, q2, k2, L, a2, b2) = base + a2·exp(−b2(1 − q2·k2ᵀ))·L̃`.
- `tune`: grid over `A2 × B2` on validation.
- `main()`: first the decision. Each validation set is split by community into halves T and S; a2, b2 are tuned on T
  and scored on S, for variants A (PRGA alone), B (fixed DINOv2 cache) and C (fine-tuned cache). C won (0.890 → 0.909),
  so only C was run on the test set, with a2, b2 re-tuned on the whole validation set.

`src/demo.py` calls these same functions, so the demo is exactly the final model.

## 6. The base paper: `src/basepaper.py` and `src/09_base_paper.py`

`basepaper.py`'s docstring is a table of what the paper specifies and what we had to choose. The pieces:
- `FIG2_WINDOWS`: the 26 patches. The photo is resized to 336 × 336; windows of several shapes on a 3 × 3 and a
  4 × 4 grid, plus the whole photo.
- `RelationalGatedLayer` = Eq. 1-2: attention score = (GAT score `aᵀ[Wh_p ‖ Wh_q]`) × sigmoid(`Wh_p·Wh_q`), softmax,
  weighted sum, ReLU. W starts as the identity so the output starts in CLIP space.
- `MultiAggregation` = Eq. 3: learned mix of mean, max and std over the patches → one vector f̂.
- `Head`: one configuration (graph + its own cache keys). In training the query is f̂ from the graph; at test time
  it is the plain CLIP vector, as the paper says.

`09_base_paper.py`:
- `WindowEncoder`: every training step, cut the freshly augmented photos into all windows, resize each to 224, CLIP.
- `fit`: trains every variant in `CONFIGS` × {1, 2 layers} × 3 training (α, β) values **side by side on the same CLIP
  encodings**, which makes the ablations cheap. After every epoch, validation macro-F1 decides the best epoch (epoch 0
  = untrained cache is a candidate). Then α, β grid on validation, and per variant the best layers/(α, β) on
  validation.
- The `noGraph` variant is the control: identical training, but the query is the whole-photo window without the
  graph. If the graph helped, `BasePaperExact` would beat it; it does not.

## 7. Comparison methods: `src/adapters.py`

All except CoOp extend `_Selectable`: `fit` trains once per value in `grid` (a key hyper-parameter) and keeps the best
on validation. `_train` is a shared loop: mini-batches of 256 views, AdamW or SGD, cosine schedule.

| class | what is learned | formula of the score |
|---|---|---|
| `CLIPAdapter` | MLP 512→128→512 | `100 · l2n(r·MLP(f) + (1−r)·f)·Tᵀ` |
| `TaskRes` | residual R `[4, 512]`, starts at 0 | `100 · f·l2n(T + a·R)ᵀ` |
| `CLAP` | linear layer W, starts at T | `100 · f·l2n(W)ᵀ`, loss + Σ_c λ_c‖W_c − T_c‖² |
| `GraphAdapter` | one linear GCN layer over [T; class prototypes] | `100 · f·l2n(b·l2n(GCN)_text + (1−b)·T)ᵀ` |
| `CoOp` | 4 prompt word vectors | text encoder("[V1][V2][V3][V4] class.") back-propagated |

Paper links and summaries: [COMPARED_METHODS.md](COMPARED_METHODS.md).

## 8. Results and figures

- `15_tables_and_figures.py`: `load_runs` reads every `results/runs/*` file, `main_table` averages over seeds →
  `results/main_table.csv/.md`, then line plots, confusion matrix and per-class F1.
- `16_comparison_figure.py`: the dot chart in the README.
- `17_cost_benchmark.py`: time per photo of each part (CUDA-synchronised, median of 30 runs after warm-up).

## 9. Questions a mentor may ask

**Why not fine-tune a normal CNN?** With 1-16 photos per class it overfits: EfficientNet-B0 reaches 0.27-0.55
macro-F1 at K ≤ 16, against 0.81-0.96 for PRGA + DINOv2. Frozen CLIP already knows what onions, mould and rot look
like; we only learn a small head (1.1 M parameters).

**What is a cache model?** Store the vectors of the training photos. For a new photo, measure how close it is to each
stored photo and let their labels vote, weighted by closeness (`exp(−β(1 − cos))`). Add that vote to zero-shot CLIP.

**What does the graph add?** It lets the photo, its regions and the 4 class descriptions exchange information. Two
outputs are used: refined support vectors for the cache, and class descriptions adapted to the photo. Honest answer
from the ablations: most of the gain over Tip-Adapter-F comes from the text side (class descriptions + text nodes).
Replacing OWLv2 regions with a grid or the object box alone did not change the score significantly.

**Why `test_graph=False`?** It won on validation (`results/prga_selection_val.csv`: 0.875 mean vs 0.825 for the full
graph). Querying the cache with the plain CLIP vector keeps test photos where the keys were trained; the base paper's
problem (train query ≠ test query) is avoided because PRGA trains with the same plain query.

**How do you know there is no leakage?** Duplicates and same-scene re-shots were found with SIFT + RANSAC and removed
or kept on one side; communities, not photos, were split; test photos linked to the pool were purged. Check: 1-NN
drops from 0.985 (random split) to 0.883 (our split). Remaining risk: a re-arranged pile or a texture-less onion may
not be detected.

**Why macro-F1?** The test classes are unbalanced. Accuracy would reward predicting the big class; macro-F1 weights
the 4 classes equally.

**Was the test set used to choose anything?** No. Configurations, epochs, α/β, a₂/b₂ and the DINOv2 variant were all
chosen on validation. Each configuration was run on the test set once.

**Why 3 seeds?** Different support photos give very different results at K = 1 (± 0.06). Three seeds show that spread;
differences below about 2 points are within it.

**Why are your numbers on my laptop slightly different?** The table was computed on Kaggle T4 GPUs. GPU arithmetic is
not bit-identical across cards, and small differences during training change which epoch is best. Same code, same
laptop → same numbers.
