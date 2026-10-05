# The methods we compare against

Every method below starts from the same frozen network (CLIP ViT-B/16 unless stated otherwise) and only learns a small
part on top of it. They differ in *what* they learn: the image side, the text side, a memory of the training photos
("cache"), or the prompt words. Our versions are re-implementations of each paper's main equation on our cached
features (`src/fewshot.py`, `src/adapters.py`), not the authors' code.

A useful mental model for all of them: CLIP turns a photo into a vector `f` and a sentence into a vector `t`. Zero-shot
CLIP picks the class whose description vector is closest to the photo vector (`score_c = 100 · f·t_c`). Every
few-shot method adds or changes something in that formula.

## The base paper

**Ahmad, Sikdar, Pradhan, Behera — Advancing Cache-Based Few-Shot Classification via Patch-Driven Relational Gated
Graph Attention** (2025). [arXiv:2512.12498](https://arxiv.org/abs/2512.12498)

Builds on Tip-Adapter (below). During training each support photo is cut into 26 overlapping patches, the patches
become nodes of a graph, and a gated graph-attention network mixes them into one refined vector. That refined vector
is used to train the cache. At test time the graph is not used, so prediction costs the same as Tip-Adapter. The paper
reports gains on 11 standard benchmarks. On onions our replication found that the graph does not help (see
[BASE_PAPER_REPLICATION.md](BASE_PAPER_REPLICATION.md)). Code: `src/basepaper.py`, `src/09_base_paper.py`.

## Methods from the base paper's comparison table

### Tip-Adapter / Tip-Adapter-F (ECCV 2022)
**Zhang et al. — Tip-Adapter: Training-free Adaption of CLIP for Few-shot Classification.**
[arXiv:2111.03930](https://arxiv.org/abs/2111.03930)

Keeps a **cache**: the CLIP vectors of all support photos (keys) and their labels (values). A new photo is compared
with every key; the closer it is, the more that photo's label counts:
`score_c = 100 · f·t_c + α Σ_j exp(−β(1 − f·k_j)) · label_jc`. Tip-Adapter needs no training at all. Tip-Adapter-F
("fine-tuned") also learns the keys for 20 epochs. This is the starting point of both the base paper and PRGA.
Code: `TipAdapter`, `TipAdapterF` in `src/fewshot.py`.

### CLIP-Adapter (IJCV 2024)
**Gao et al. — CLIP-Adapter: Better Vision-Language Models with Feature Adapters.**
[arXiv:2110.04544](https://arxiv.org/abs/2110.04544)

Adds a small two-layer network (512 → 128 → 512) after the image encoder and blends its output with the original
vector: `f' = r · MLP(f) + (1 − r) · f`. The blend ratio `r` keeps most of CLIP's knowledge, so few photos do not
overfit it. Code: `CLIPAdapter` in `src/adapters.py`.

### TaskRes (CVPR 2023)
**Yu et al. — Task Residual for Tuning Vision-Language Models.** [arXiv:2211.10277](https://arxiv.org/abs/2211.10277)

Leaves the image side alone and changes the **text side**: each class vector gets a learned correction ("residual")
that starts at zero, `t'_c = t_c + a · R_c`. The original text knowledge stays fixed; only the task-specific part is
learned. Code: `TaskRes` in `src/adapters.py`.

### GraphAdapter (NeurIPS 2023)
**Li et al. — GraphAdapter: Tuning Vision-Language Models With Dual Knowledge Graph.**
[arXiv:2309.13625](https://arxiv.org/abs/2309.13625)

Builds a graph whose nodes are the class text vectors and the class image prototypes (the mean photo of each class),
connected by similarity. A graph network refines the text vectors using that graph, so related classes inform each
other. Our version uses a single graph layer. Code: `GraphAdapter` in `src/adapters.py`.

### CLAP (CVPR 2024)
**Silva-Rodríguez, Hajimiri, Ben Ayed, Dolz — A Closer Look at the Few-Shot Adaptation of Large Vision-Language
Models.** [arXiv:2312.12730](https://arxiv.org/abs/2312.12730)

Argues that many adapters only win when their settings are tuned on large labelled sets. Proposes a **linear probe**
(one linear layer) that starts from the text vectors and is pulled back towards them. The pull is stronger for classes
where zero-shot CLIP is already confident. Code: `CLAP` in `src/adapters.py`.

## Other methods we added

### CoOp (IJCV 2022)
**Zhou, Yang, Loy, Liu — Learning to Prompt for Vision-Language Models.**
[arXiv:2109.01134](https://arxiv.org/abs/2109.01134)

Instead of writing the prompt "a photo of a [class]" by hand, it learns the prompt words as vectors, passed through the
frozen text encoder. Starts from "a photo of a". Code: `CoOp` in `src/adapters.py`.

### CaFo and PlantCaFo (two backbones)
**Zhang et al. — Prompt, Generate, then Cache: Cascade of Foundation Models makes Strong Few-shot Learners** (CVPR
2023). [arXiv:2303.02151](https://arxiv.org/abs/2303.02151)
**PlantCaFo** (Plant Phenomics 2025), the same idea applied to crop disease.
[doi:10.1016/j.plaphe.2025.100024](https://doi.org/10.1016/j.plaphe.2025.100024)

Uses **several pretrained networks**, each with its own Tip-Adapter-style cache, and adds their scores. CLIP learned
from image–text pairs and DINO from images only, so their mistakes are partly different and the combination is more
accurate. Our "PlantCaFo-style cache" is CLIP + DINOv2 caches with fine-tuned keys (`PlantCaFoLite` in
`src/fewshot.py`). This idea is also what we added to PRGA to get PRGA + DINOv2 cache.

### Linear probe
Logistic regression on the CLIP image vectors, the standard baseline from the CLIP paper (below). `LinearProbe` in
`src/fewshot.py`.

### EfficientNet-B0, fine-tuned
**Tan, Le — EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks** (ICML 2019).
[arXiv:1905.11946](https://arxiv.org/abs/1905.11946)
An ordinary ImageNet CNN with all layers trained on the support photos. Shows why frozen foundation models matter with
very few photos. `src/14_cnn_baseline.py`.

### Domain-specific backbones
**BioCLIP** — Stevens et al., *BioCLIP: A Vision Foundation Model for the Tree of Life* (CVPR 2024).
[arXiv:2311.18803](https://arxiv.org/abs/2311.18803). CLIP trained on 10 M images of plants, animals and fungi.

**SCOLD** — *A Vision-Language Foundation Model for Leaf Disease Identification* (2025).
[arXiv:2505.07019](https://arxiv.org/abs/2505.07019). Trained on 186k leaf-disease image–caption pairs.

We ran Tip-Adapter-F on each. Both did far worse than plain CLIP on onion bulbs: they were trained on leaves and
organisms, not on bulbs photographed on cloth.

## Building blocks used by PRGA

| model | what it does here | paper |
|---|---|---|
| CLIP ViT-B/16 | image and text vectors (frozen) | Radford et al. 2021, [arXiv:2103.00020](https://arxiv.org/abs/2103.00020) |
| DINOv2-S | second image vector (frozen) | Oquab et al. 2023, [arXiv:2304.07193](https://arxiv.org/abs/2304.07193) |
| OWLv2 | finds the onion and lesion-like spots from text prompts (frozen) | Minderer et al. 2023, [arXiv:2306.09683](https://arxiv.org/abs/2306.09683) |
| graph attention (GAT) | the idea behind PRGA's graph layers | Veličković et al. 2018, [arXiv:1710.10903](https://arxiv.org/abs/1710.10903) |

## Not included

Ta-Adapter (Pattern Recognition 2024) trains prompts inside both CLIP encoders, so it would need back-propagation
through the image encoder, which our cached-feature setup avoids. CAA (ICCV 2025) needs a separate disentanglement
module and has no released code.
