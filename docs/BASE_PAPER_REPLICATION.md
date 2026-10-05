# Replicating the base paper: what it says, what the code does, what is unknown

**Paper:** T. Ahmad, A. Sikdar, S. Pradhan, A. Behera, *Advancing Cache-Based Few-Shot Classification via Patch-Driven
Relational Gated Graph Attention*, arXiv:2512.12498v1, 13 Dec 2025.
**Code by the authors:** none. Their GitHub repository (`tasveerahmad/Patch-Relational-Graph-Attention`) contains only
a README: *"Our code and dataset will be made available once paper is accepted."* (checked 3 Oct 2026).
**Our code:** `src/basepaper.py` (model), `src/09_base_paper.py` (training/evaluation, Kaggle T4 GPUs).

Because there is no code, an "exact" replication can only mean: **everything the paper states is implemented as stated,
and every gap is filled with a documented default and, where cheap, tested both ways.** Each item below cites where in
the paper it comes from.

## 1. Paper → code, line by line

| # | the paper says | where | code | status |
|---|---|---|---|---|
| 1 | frozen CLIP ViT-B/16 for image and text | Sec. 4 Impl. | `ViT-B-16-quickgelu`, OpenAI weights, frozen | ✔ as stated |
| 2 | training images: random resize, crop, horizontal flip, then 336×336 | Sec. 4 Impl. | `RandomResizedCrop(336, scale 0.5–1) + flip`, **new random draw every step** | ✔ as stated |
| 3 | (3×3) and (4×4) grid tiling; adjacent tiles merged into multi-scale patches; 26 patches best | Sec. 4, Fig. 2, Table 4 | the window **shapes drawn in Fig. 2** at all positions: 3×3 → 1×1 (9), 2w×3h (2), 3w×2h (2); 4×4 → 2w×4h (3), 4w×2h (3), 3w×2h (6); + whole = 26 | ⚠ list not published; our reading |
| 4 | each patch resized to 224×224, encoded by frozen CLIP → node | Sec. 4 Impl. | bicubic + antialias resize on GPU, L2-normalised | ✔ |
| 5 | fully connected undirected graph | Sec. 3 | dense attention over all 26 nodes incl. self | ✔ |
| 6 | Eq. 1: `h_p = ρ(Σ_q α_pq W h_q)`, ρ non-linear "e.g. ReLU" | Sec. 3, Eq. 1 | ReLU inside every layer | ✔ literal (also tested without ReLU on the last layer) |
| 7 | Eq. 2: `e = aᵀ[Wh_p‖Wh_q] · σ((Wh_p)ᵀWh_q)`, α = softmax(LeakyReLU(e)) | Eq. 2 | exactly this; LeakyReLU slope 0.2 (GAT) | ✔ |
| 8 | Eq. 3: `f̂ = Σ_m γ_m W_m φ_m`, ψ = {mean, max, std}, γ learnable, W_m linear | Eq. 3 | exactly this, then L2-normalised | ✔ (Fig. 1's {min, max, softmax} tested as ablation) |
| 9 | Eq. 4: `A = exp(−β(1 − f_q Θᵀ))`, Θ learnable, Θ₀ = F_train | Eq. 4 | exactly this; Θ₀ = Tip-Adapter's 10-view average | ✔ |
| 10 | Eq. 5: `logits = α A L + f̂ W_cᵀ`, L fixed one-hot | Eq. 5 | exactly this, with CLIP's ×100 on the second term | ✔ |
| 11 | W_c from the CLIP text encoder, prompt "a photo of a [CLASS]" | Fig. 1 | single template "a photo of a {class}." | ✔ |
| 12 | graph only in training; test: `α·exp(−β(1 − f_test Θᵀ))L + f_test W_cᵀ` | Sec. 3, last ¶ | no graph at test | ✔ |
| 13 | AdamW, lr 0.001, cosine annealing | Sec. 4 Setup | `AdamW(lr=1e-3, eps=1e-4)`, `CosineAnnealingLR` | ✔ |
| 14 | α, β "tuned empirically"; grid search heat-maps | Sec. 4, Fig. 5 | grid search on **validation** | ✔ |
| 15 | K = 1, 2, 4, 8, 16; mean of 3 runs | Sec. 4 Setup | same, + "all 20 %" | ✔ |

## 2. What the paper does not specify, and what we chose

| unknown | choice | why |
|---|---|---|
| exact 26-window list | Fig. 2 shapes at every position (above) | uses both grids, merged adjacent tiles, multiple scales, 26 total, 13 per grid (Table 4 has a 13 column). The earlier 4×4-only reading is kept as ablation `legacyWin` |
| number of graph layers | 1 or 2, chosen on validation | Fig. 1 draws two graph stages; not stated |
| epochs, batch size, AdamW ε, weight decay | 20 epochs, batch 256, ε = 1e-4, PyTorch default decay | Tip-Adapter-F's defaults; the paper's stated optimiser settings are exactly Tip-Adapter-F's |
| α, β during training | (1,1), (10,1), (10,5), chosen on validation | not given; (1,1) = Tip-Adapter's initial values. With (1,1) alone the cache term is < 1 next to a zero-shot term of ~30 and training barely touches the keys |
| initialisation | W, W_m = identity; a = Xavier (GAT); γ = 1/3 | identity keeps f̂ in CLIP space at the start; Attention 1 "is nothing more than GAT" |
| loss | cross-entropy | as Tip-Adapter-F |
| epoch selection | best validation macro-F1, **including epoch 0** (untrained) | never the test set; epoch 0 makes "training did not help" visible |
| ×100 logit scale | on the zero-shot term | CLIP / Tip-Adapter convention; Eq. 5 omits it |

## 3. What changed compared with our earlier re-implementations

Two earlier versions of this replication (not kept in this repository) were close, but differed from the paper in six
places:

| | earlier code | paper / now |
|---|---|---|
| ρ in Eq. 1 | none after the last layer; none at all in the 1-layer variant | ReLU in every layer |
| learning-rate schedule | OneCycle (warm-up + cosine) | cosine annealing |
| augmentation | 5 fixed views computed once | fresh augmentation every step |
| patch windows | 4×4 grid only | 3×3 **and** 4×4 grids (Fig. 2) |
| class text | average of 5 custom templates | "a photo of a [CLASS]" (Fig. 1) |
| attention vector init | `randn` (large, ‖a‖ ≈ 22) | Xavier (GAT) |

## 4. Paper ablations reproduced on onions (same run, shared encodings)

| head | paper reference | question |
|---|---|---|
| `BasePaperExact-A1` / `-A2` | Table 3 | only Attention 1 / only Attention 2 |
| `BasePaperExact-noEdges` | Table 5 | no inter-patch edges |
| `BasePaperExact-aggFig1` | Fig. 1 | ψ = {min, max, softmax} |
| `BasePaperExact-legacyWin` | — | the other window reading |
| `BasePaperExact-rhoLastId` | — | does the literal last-layer ReLU hurt? |
| `BasePaperExact-noGraph` | — | **control**: identical recipe, training query = plain whole-image window (= Tip-Adapter-F with fresh augmentations). The graph helps only if `BasePaperExact` beats this. |
| `BasePaperExact-bal` | — | class-balanced cache (for the unequal full pool) |
| `BasePaperExact-desc` | — | our visual descriptors instead of "a photo of a [CLASS]" |

## 5. Results (cleaned object-disjoint split, 7,612 test photos, macro-F1 mean ± std over 3 seeds)

| head | K=1 | K=2 | K=4 | K=8 | K=16 | full |
|---|---|---|---|---|---|---|
| **BasePaperExact** (the paper) | 0.609 ± .057 | 0.678 ± .039 | 0.709 ± .024 | 0.792 ± .079 | 0.820 ± .030 | 0.662 ± .126 |
| -A1 (Attention 1 only) | 0.609 | 0.677 | 0.708 | 0.799 | 0.821 | 0.661 |
| -A2 (Attention 2 only) | 0.615 | 0.679 | 0.708 | 0.798 | 0.817 | 0.663 |
| -noEdges | 0.612 | 0.684 | 0.709 | 0.779 | 0.793 | 0.560 |
| -aggFig1 (ψ = min, max, softmax) | 0.609 | 0.679 | 0.708 | 0.800 | 0.821 | 0.661 |
| -legacyWin (4×4-only windows) | 0.616 | 0.679 | 0.722 | 0.795 | 0.812 | 0.673 |
| -rhoLastId (no ReLU on last layer) | 0.616 | 0.682 | 0.725 | 0.823 | 0.811 | 0.812 |
| -bal (class-balanced cache) | 0.609 | 0.681 | 0.693 | 0.784 | 0.787 | 0.843 |
| -desc (our descriptors as W_c) | 0.774 | 0.797 | 0.794 | 0.830 | 0.835 | 0.701 |
| **-noGraph (control, = Tip-Adapter-F, fresh augmentations)** | **0.651** | **0.695** | **0.789** | **0.861** | **0.909** | **0.946** |
| zero-shot CLIP, "a photo of a [CLASS]" | 0.548 | | | | | |

### What this shows

1. **On this data the graph does not help, it hurts.** The control with the identical recipe but no graph is better at
   every K and in 14 of 18 (K, seed) runs, including all 9 runs at K = 4, 16 and full.
2. **The paper's ablations are not reproduced.** Attention 1 only, Attention 2 only, combined, no edges and the other
   aggregators all score within noise of each other (paper, Tables 3 and 5: combined attention clearly best, removing edges
   costs up to 20 points). On onions the graph's internal design makes no measurable difference.
3. **Why (diagnosis, supported by the logs):**
   - The model's best validation epoch is early (median 5.5 of 20; epoch 0–2 in 8 of 18 runs) while the no-graph control
     trains to late epochs (median 14). Training with the graph-refined query makes the test-time cache *worse*.
   - The keys barely move (cosine between trained and initial keys ≥ 0.999 in a traced K = 1 run). The only way the graph can affect a
     test prediction is through the keys, so the graph's attention design cannot matter, which is what point 2 shows.
   - Train/test query mismatch: the training query f̂ is a ReLU-ed, pooled vector whose cosine with the plain CLIP
     feature is only ≈ 0.78 (traced K = 1 run); the test query is the plain CLIP feature. Removing the last ReLU
     (`rhoLastId`) recovers part of the gap (e.g. full: 0.662 → 0.812), consistent with this diagnosis.
4. **The one-hot cache is unstable on unequal classes** (full pool: 0.54–0.79 across seeds). Validation then turns the cache
   almost off (α = 0.1–0.25). A class-balanced cache fixes it (0.843).
5. **The class text matters far more than the graph:** our descriptors add +17 points at K = 1 (0.609 → 0.774).

### Scope of these conclusions

This is evidence about the method **as written in the paper** on **one** dataset (onion bulbs, 4 classes). It does not
refute the paper's results on its benchmarks (ImageNet, Caltech101, Aircraft, Food101, UCF101, EuroSAT, … and its
soldier dataset): general-object and scene datasets with 2 to 1000 classes, unlike our fine-grained 4-class onion task. Details the paper leaves open (§2) are
our choices; a different reading could behave differently. α reached the top of the search grid (96) in 2 of 18 runs, so
those two runs may be slightly under-tuned. That does not change the comparison.
