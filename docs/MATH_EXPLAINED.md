# The maths of the pipeline, step by step

This guide follows a photo through the whole pipeline, in order. Each step has the formula, what every symbol means,
and a tiny worked example with small numbers. You only need school algebra plus three ideas: **vectors**, **dot
products** and **exponentials**. Code locations are given so you can check each formula against the code.

---

## 0. Three tools used everywhere

**Vector length and normalisation.** A vector is a list of numbers, e.g. a = (3, 4).
Its length is ‖a‖ = √(3² + 4²) = 5. *Normalising* divides by the length: â = a / ‖a‖ = (0.6, 0.8), length 1.
Code: `l2n()` in `src/common.py`.

**Dot product and cosine similarity.** a · b = a₁b₁ + a₂b₂ + … For two *normalised* vectors the dot product is the
cosine of the angle between them: 1 = same direction, 0 = unrelated, −1 = opposite.
*Example:* â = (0.6, 0.8), b̂ = (0.8, 0.6) → â · b̂ = 0.48 + 0.48 = **0.96**, very similar.
Every "similarity" in this project is this number.

**Softmax: turning scores into probabilities.**
p_c = e^{z_c} / Σ_j e^{z_j}. The results are positive and add up to 1; bigger scores get more probability.
*Example:* scores (2, 1, 0) → e² = 7.39, e¹ = 2.72, e⁰ = 1, total 11.11 → probabilities **(0.665, 0.245, 0.090)**.

---

## 1. Cleaning the data (before any learning)

**1a. Exact duplicates: MD5 hash.** Each file's bytes are turned into a 32-character fingerprint. Identical files
give identical fingerprints, so one copy is kept. (`src/02_clean_dataset.py`)

**1b. Candidate pairs: CLIP cosine.** Two photos of the same class with cosine ≥ 0.92 *might* be the same scene.

**1c. Keypoint matching: RootSIFT + Lowe's ratio test.** SIFT finds up to 400 distinctive points per photo, each
described by a 128-number vector. RootSIFT: divide the vector by the sum of its entries, then take the square root of
each entry (this makes matching more reliable). A point in photo A matches a point in photo B only if
  d₁ < 0.8 · d₂
where d₁ and d₂ are the distances to the nearest and second-nearest point in B. *Example:* d₁ = 0.3, d₂ = 0.5 →
0.3 < 0.4, so it's a clear match. With d₁ = 0.45 it would be rejected as ambiguous.

**1d. Geometry check: RANSAC homography.** If A and B show the same flat scene, one 3×3 matrix H (a homography)
maps every matched point of A onto B. RANSAC repeatedly picks 4 random matches, fits H, and counts **inliers**:
matches that land within 8 pixels of where H predicts. ≥ 10 inliers means "same scene".

**1e. Groups and split.** Verified pairs are joined into groups (union-find). For the split, groups are clustered
with **Louvain communities**, which maximise *modularity*: Q = (edges inside communities) − (edges expected there by
chance). Whole communities go to training or test, so no scene appears in both. For bees, the group is simply the
video.

**Why it matters (leakage), in numbers:** a 1-nearest-neighbour classifier scores 0.985 on a random onion split
but 0.883 on our scene-disjoint split. The 10-point gap is copying, not learning.

---

## 2. Preparing an image

**Resize and crop or pad.** Onions: resize the short side to 224 px, then crop the centre 224 × 224. Bees (tall
160 × 280 crops): pad with black to 280 × 280, then resize to 224, so the mite is never cut off.

**Normalise each colour channel:** x' = (x − μ) / σ, with CLIP's μ, σ per channel.
*Example (red channel):* pixel 0.6, μ = 0.481, σ = 0.269 → (0.6 − 0.481) / 0.269 = **0.442**.

**Augmentation (training only).** Random crops covering 50-100 % of the photo, plus horizontal flips. Each support
photo gets 10 such views, so the model sees small variations.

---

## 3. The frozen encoders: CLIP, DINOv2, OWLv2

**3a. Vision Transformer (ViT): patches.** The 224 × 224 image is cut into squares: 16 px (ViT-B/16) → 14 × 14 = 196
patches; 14 px (ViT-L/14) → 16 × 16 = 256 patches. Each patch becomes a vector, plus one extra "class token".

**3b. Self-attention: how patches share information.**
  Attention(Q, K, V) = softmax(Q Kᵀ / √d) V
Each patch makes a query q, a key k and a value v (three learned linear maps). Patch i looks at patch j with weight
softmax_j(q_i · k_j / √d) and collects the weighted sum of the values. √d keeps the scores from getting too large.
*Example:* scores for 3 patches (2, 1, 0) → weights (0.665, 0.245, 0.090) → the output is mostly patch 1's value.

**3c. CLIP's training (done by OpenAI; we only use the result).** Images and captions are encoded into the same
space. In a batch of N pairs, the loss pushes each image towards its own caption and away from the other N − 1:
  loss_i = −log( e^{s·cos(img_i, txt_i)} / Σ_j e^{s·cos(img_i, txt_j)} )     (InfoNCE, s ≈ 100)
Result: a photo and a sentence describing it end up close together.

**3d. Zero-shot classification.** Score per class = 100 · cos(photo, class text); then softmax.
*Example:* cos(photo, "a bee with a varroa mite") = 0.30, cos(photo, "a healthy bee") = 0.25 → scores 30 and 25 →
p(varroa) = e³⁰ / (e³⁰ + e²⁵) = 1 / (1 + e⁻⁵) = **0.993**. The ×100 makes small cosine differences decisive.

**3e. Class text vectors from descriptions** (`src/07_text_embeddings.py`):
  t_c = normalise( mean of normalised description vectors  +  0.5 × template vector )
Several expert sentences per class ("a small reddish-brown oval mite on its back", …) are averaged, so no single
wording dominates.

**3f. DINOv2.** A ViT trained without labels or text, by *self-distillation*: a student network must output the same
thing as a slowly-updated copy of itself (the teacher) on different crops of the same image. Its features capture
texture and shape. It has no text side, so it can't do zero-shot.

**3g. OWLv2: finding regions from text.** For every candidate box, OWLv2 gives a score σ(box · text), where σ is the
sigmoid 1 / (1 + e⁻ˣ) (between 0 and 1). We keep boxes above a threshold, then remove overlapping duplicates with
**non-maximum suppression**, which uses IoU (intersection over union):
  IoU(A, B) = area(A ∩ B) / area(A ∪ B)
*Example:* A = (0,0)–(4,4), B = (2,2)–(6,6): both have area 16, overlap (2,2)–(4,4) = 4 → IoU = 4 / (16 + 16 − 4) =
**0.143**. NMS drops the lower-scoring box if IoU > 0.3. The same IoU is one of PRGA's edge features (step 5).

---

## 4. The cache: Tip-Adapter (the base of everything)

Keep the support photos as a **cache**: keys K = their CLIP vectors, values L = their one-hot labels.
For a test photo with vector f:
  affinity_j = exp(−β (1 − f · k_j))
  logits = 100 f · Tᵀ  +  α · affinity · L̃
The first part is zero-shot CLIP; the second is a vote of similar training photos. β sets how sharply the vote
favours close photos; α sets how much the vote counts. Both are chosen on validation (grid α ∈ {0.5…12},
β ∈ {1…9}). Code: `tip_logits()` in `src/fewshot.py`.

*Example:* α = 2, β = 5. The test bee has cosine 0.9 with a Varroa support photo and 0.7 with a healthy one:
  Varroa: e^{−5 × 0.1} = e^{−0.5} = 0.607 → vote 2 × 0.607 = **1.21**
  healthy: e^{−5 × 0.3} = e^{−1.5} = 0.223 → vote 2 × 0.223 = **0.45**
The cache adds 1.21 − 0.45 = 0.76 in favour of Varroa.

**Balanced values L̃.** When classes have different numbers of support photos (all-photos setting), each class's
vote is averaged instead of summed: L̃ = L / (number of support photos of that class). Otherwise the bigger class
wins just by having more photos. *Example:* 3 healthy photos with affinities 0.6, 0.5, 0.4 → sum 1.5 but average
0.5, comparable to 1 Varroa photo with 0.6.

**Tip-Adapter-F** additionally *trains* the keys K for a few epochs (step 7).

---

## 5. PRGA: the graph adapter (our model)

Code: `GatedGAT`, `Readout`, `PRGANet`, `PRGA` in `src/fewshot.py`.

**5a. Nodes.** One graph per photo:
- image nodes: the whole photo g, the object box, 2 instances (single onions / bee body parts) and 2 suspicious spots
  from OWLv2, each encoded by CLIP;
- text nodes: one per class, the class text vector t_c.
Each node vector x (512-d, or 768-d for L/14) is mapped to 256-d and tagged with its type:
  h_i = W_in x_i + E[type_i]      (type = photo, object, instance, spot or text)

**5b. Edges.** Every pair of nodes is connected. Edge features e_ij = (IoU, centre distance, product of detector
scores, cosine of the CLIP vectors). Text-image and text-text edges use only the cosine. (The selected configuration
switches the box geometry off, so only the cosine is used.)

**5c. One gated graph-attention layer** (4 heads, each on 64 of the 256 numbers):
  1. score: s_ij = LeakyReLU( a_srcᵀ W h_i + a_dstᵀ W h_j + uᵀ e_ij )      (LeakyReLU(x) = x if x > 0, else 0.2x)
  2. attention: a_ij = softmax over j of s_ij      (only real nodes; padded slots get weight 0)
  3. message: m_i = Σ_j a_ij W h_j
  4. gate: z_i = σ( W_g [h_i ; m_i] )      (a number between 0 and 1 per feature)
  5. update: h_i' = LayerNorm( z_i ⊙ m_i + (1 − z_i) ⊙ h_i )
The gate decides how much of the neighbours' message to accept. *Example:* z = σ(0) = 0.5 → keep half the old node,
take half the message. z near 0 → ignore the neighbours, keep yourself. LayerNorm rescales each vector to mean 0 and
standard deviation 1 (then a learned scale), which keeps training stable. Two such layers are stacked.

**5d. Readouts: what the graph produces.**
- refined photo vector: f̂ = normalise( g + W_r [ mean(h), max(h), Σ_i w_i h_i ] ), with w = softmax(qᵀ h_i) over the
  image nodes (attention pooling);
- photo-specific class vectors: t̂_c = normalise( t_c + W_t h_{text,c} ).

**5e. PRGA's score for class c:**
  z_c = 100 g · t_c  +  α Σ_j exp(−β (1 − g · k̂_j)) L̃_jc  +  γ · 100 g · t̂_c
- the cache keys k̂_j are the support photos *refined by the graph*;
- the test photo queries the cache with its **plain** CLIP vector g, the same at training and test time. This is the
  fix for the base paper's mismatch (step 9).
α, β, γ start at 5, 5, 0.5 and are learned; α, β are re-chosen on validation at the end.

**5f. PRGA + DINOv2 cache (final model).** A second cache built from DINOv2 vectors d:
  z_c ← z_c + a₂ Σ_j exp(−b₂ (1 − d · k₂ⱼ)) L̃_jc
k₂ are the DINOv2 support vectors, fine-tuned for 20 epochs; a₂, b₂ come from a grid search on validation. The idea
(from CaFo / PlantCaFo): a model trained with text and a model trained on images alone make different mistakes.

---

## 6. Training (only the small parts; CLIP, DINOv2 and OWLv2 stay frozen)

**Cross-entropy loss with label smoothing.** loss = −Σ_c y_c log p_c, with a softened target
y = 0.9 × one-hot + 0.1 / C (C = number of classes).
*Example (2 classes):* target (0.95, 0.05); prediction p = (0.8, 0.2) → loss = −(0.95 ln 0.8 + 0.05 ln 0.2)
= −(0.95 × −0.223 + 0.05 × −1.609) = 0.212 + 0.080 = **0.292**. Smoothing stops the model from becoming
over-confident on very few photos.

**AdamW optimiser.** Each parameter moves against its gradient, scaled by running averages of the gradient (momentum)
and its square (a per-parameter step size), plus weight decay: w ← w − lr · (m̂ / (√v̂ + ε) + λ w). PRGA: learning
rate 1e-3, weight decay 1e-4, with a one-cycle schedule (warm up for 10 %, then decay).

**Node dropout.** During training each region node is hidden with probability 0.2, so the model can't rely on one
box. The photo node is never hidden.

**Checkpoint selection.** Every 5 epochs (60 in total) the model is scored on validation; the best checkpoint is kept.
The test set is never used for any choice.

---

## 7. The comparison methods in one line each

| method | what is learned | formula idea |
|---|---|---|
| Zero-shot CLIP | nothing | 100 f · t_c |
| Linear probe | a linear classifier | softmax(W f + b) |
| Tip-Adapter-F | the cache keys | step 4, keys trained |
| CLIP-Adapter | small MLP on the image vector | f' = r · MLP(f) + (1 − r) f |
| TaskRes | a residual on the class texts | t_c' = t_c + a · R_c |
| CLAP | a linear probe pulled towards the texts | loss + λ_c ‖w_c − t_c‖² |
| GraphAdapter | GCN over class texts and image prototypes | t' = GCN(A, [T; P]) |
| CoOp | the prompt words | t_c = TextEncoder([v₁ … v_M, class]) |
| PlantCaFo-style | two caches (CLIP + DINOv2) | step 4 twice, added |
| EfficientNet-B0 | the whole CNN | ordinary fine-tuning |

---

## 8. The base paper's graph (for comparison)

Ahmad et al. 2025 cut each training image into 26 patches and connect them. Their attention multiplies two terms:
  att1 = aᵀ [W h_p ‖ W h_q],   att2 = σ( (W h_p)ᵀ W h_q ),   α_pq = softmax_q( LeakyReLU(att1 · att2) )
  h_p' = ReLU( Σ_q α_pq W h_q ),   readout = Σ_m γ_m W_m φ_m(h)   with φ ∈ {mean, max, std}
The refined vector is used to train the cache, but at **test** time the plain CLIP vector queries it. In our logs
these two kinds of query have a cosine of only ~0.78, so training tunes the cache for a different query than the one
used at test. PRGA trains with the same plain query it uses at test.

---

## 9. Measuring results

**Confusion-matrix counts** for one class (Varroa as "positive"):

| | predicted Varroa | predicted healthy |
|---|---|---|
| truly Varroa | TP = 80 | FN = 20 |
| truly healthy | FP = 10 | TN = 890 |

- accuracy = (TP + TN) / all = 970 / 1000 = **0.970**
- precision (of the "Varroa" calls, how many are right) = TP / (TP + FP) = 80 / 90 = **0.889**
- recall (of the real Varroa bees, how many are found) = TP / (TP + FN) = 80 / 100 = **0.800**
- F1 = 2 · P · R / (P + R) = 2 × 0.889 × 0.8 / 1.689 = **0.842**
- for the healthy class: P = 890 / 910 = 0.978, R = 890 / 900 = 0.989, F1 = **0.983**
- **macro-F1** = average over classes = (0.842 + 0.983) / 2 = **0.913**

Accuracy (0.97) looks excellent because 90 % of the bees are healthy. Macro-F1 (0.91) shows the weaker Varroa class,
which is why it's our main metric.

**AUROC.** Rank all test photos by their Varroa probability. AUROC = the probability that a random infested bee is
ranked above a random healthy bee (0.5 = random, 1 = perfect). It doesn't depend on any threshold.

**Mean ± standard deviation over 3 seeds.** *Example:* 0.80, 0.85, 0.83 → mean 0.827; deviations −0.027, 0.023,
0.003 → std = √((0.000711 + 0.000544 + 0.000011) / 2) = **0.025**. Differences smaller than about 2 std are not
convincing.

---

## 10. Making the confidence honest: calibration

A model is **calibrated** when "80 % sure" is right 80 % of the time. Code: `src/calibration.py`.

**Expected calibration error (ECE).** Put predictions into confidence bins and compare each bin's accuracy with its
average confidence:
  ECE = Σ_bins (share of photos in bin) × | accuracy_bin − confidence_bin |
*Example:* 40 % of photos have confidence ≈ 0.9 but are right only 70 % of the time; 60 % have confidence ≈ 0.6 and
are right 60 % of the time → ECE = 0.4 × |0.7 − 0.9| + 0.6 × 0 = **0.08**.

**Temperature and class bias**, fitted on validation:
  p = softmax( (z + b) / τ )
τ > 1 makes the model less sure, τ < 1 more sure; b shifts classes (e.g. corrects a lean towards "healthy").
τ and b minimise the cross-entropy on the validation photos.
*Example:* scores (2, 0): τ = 1 → p = 0.881; τ = 2 → softmax(1, 0) = **0.731**.

**"Not sure" threshold.** The lowest confidence t at which validation photos above t are ≥ 90 % accurate; the app
says "check by hand" below it.

---

## 11. Explaining a prediction (the app)

**Score breakdown (exact).** PRGA + DINOv2's score is a sum (steps 5e-5f):
  z = text match + CLIP cache + photo-specific prototypes + DINOv2 cache
so each part's contribution can be shown exactly. Each part is centred (its average over the classes subtracted),
because only differences between classes change the decision.

**Region importance (occlusion of graph nodes).** Remove box i from the graph, re-score, and measure how much the
predicted class's lead shrinks:
  importance_i = margin(all boxes) − margin(without box i),   margin = z_pred − max_{other} z
A positive value means the box supported the decision.

**Grad-CAM (figures).** For the last-but-one ViT block with patch activations A and gradients G of the decision
score: heat_p = ReLU( Σ_channels A_p · G_p ). It shows where in the photo the decision is sensitive. On bees, the
hottest patch lies on the annotated mite for about 70 % of infested bees.

---

## 12. The whole pipeline as one chain

photo → (crop or pad, normalise) → CLIP g, DINOv2 d, OWLv2 boxes → CLIP vectors of the boxes
→ graph (5a-5c) → refined keys k̂ (training) and photo-specific class vectors t̂
→ z = 100 g·T + α·cache_CLIP + γ·100 g·T̂ + a₂·cache_DINOv2 → calibrate → softmax → class + confidence
