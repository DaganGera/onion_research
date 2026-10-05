"""Draw the architecture diagrams (boxes, nodes and arrows) used in the README and docs/MODELS_EXPLAINED.md.

  python 19_architecture_figures.py
Output: figures/arch_pipeline.png, arch_base_paper.png, arch_prga.png, arch_prga_dinov2.png,
        arch_pipeline_nodes.png
Colours: green = data, grey = frozen pretrained network, orange = trained part, blue = fixed computation.
"""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch

from common import FIGURES

DATA, FROZEN, TRAINED, CALC = "#d9f2e3", "#e6e6e3", "#fde3cf", "#dbe8fb"
EDGE = {DATA: "#2e8b57", FROZEN: "#7a7a75", TRAINED: "#d9822b", CALC: "#2a78d6"}
INK, INK2 = "#1b1b1b", "#55554f"


def canvas(w, h, title):
    fig, ax = plt.subplots(figsize=(w, h), dpi=150)
    ax.set_xlim(0, w * 10)
    ax.set_ylim(0, h * 10)
    ax.axis("off")
    ax.text(2, h * 10 - 2, title, fontsize=15, fontweight="bold", color=INK, va="top")
    return fig, ax


def box(ax, x, y, w, h, text, color=CALC, size=9.5, bold_first=True):
    """Rounded box centred at (x, y). The first line of text is bold."""
    ax.add_patch(FancyBboxPatch((x - w / 2, y - h / 2), w, h, boxstyle="round,pad=0.3,rounding_size=1.2",
                                fc=color, ec=EDGE[color], lw=1.4, zorder=2))
    lines = text.split("\n")
    if bold_first and len(lines) > 1:
        ax.text(x, y + h / 2 - 1.2, lines[0], ha="center", va="top", fontsize=size, fontweight="bold", color=INK,
                zorder=3)
        ax.text(x, y + h / 2 - 1.2 - size * 0.32, "\n".join(lines[1:]), ha="center", va="top", fontsize=size - 1.2,
                color=INK2, zorder=3, linespacing=1.35)
    else:
        ax.text(x, y, text, ha="center", va="center", fontsize=size, fontweight="bold" if bold_first else "normal",
                color=INK, zorder=3)


def arrow(ax, p, q, text=None, color=INK2, style="-|>", rad=0.0, ls="-", size=8.5):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle=style, mutation_scale=13, lw=1.3, color=color, ls=ls,
                                 connectionstyle=f"arc3,rad={rad}", zorder=1, shrinkA=2, shrinkB=2))
    if text:
        ax.text((p[0] + q[0]) / 2 + 0.8, (p[1] + q[1]) / 2, text, fontsize=size, color=color, va="center",
                ha="left", zorder=3, bbox=dict(fc="white", ec="none", pad=0.6))


def legend(ax, x, y):
    for i, (c, t) in enumerate(((DATA, "data"), (FROZEN, "frozen network (not trained)"),
                                (TRAINED, "trained part"), (CALC, "fixed computation"))):
        ax.add_patch(FancyBboxPatch((x + i * 30, y), 3, 2, boxstyle="round,pad=0.2", fc=c, ec=EDGE[c], lw=1.2))
        ax.text(x + i * 30 + 4.5, y + 1, t, fontsize=9, va="center", color=INK2)


def save(fig, name):
    fig.tight_layout(pad=0.3)
    fig.savefig(FIGURES / name, facecolor="white")
    plt.close(fig)
    print("wrote", FIGURES / name)


# ----------------------------------------------------------------------------------------------- 1. whole pipeline
def pipeline():
    fig, ax = canvas(13, 9.2, "The whole pipeline")
    box(ax, 15, 78, 22, 9, "12,260 onion photos\n4 classes: healthy / unhealthy\nx red / white", DATA)
    box(ax, 47, 78, 26, 9, "1. Clean\nremove copies and re-shots\n(MD5, SIFT + RANSAC) -> 11,676", CALC)
    box(ax, 81, 78, 30, 9, "2. Split by scene\n20 % pool (support + validation)\n80 % test = 7,612 photos", CALC)
    arrow(ax, (26.5, 78), (33.5, 78))
    arrow(ax, (60.5, 78), (65.5, 78))
    box(ax, 81, 62, 30, 7, "K photos per class\nK = 1, 2, 4, 8, 16 or all; 3 random draws", DATA)
    arrow(ax, (81, 73), (81, 66))
    box(ax, 15, 62, 24, 7, "8 descriptions per class\n\"a red onion covered with mould\"", DATA)

    # frozen networks; the support photos go to the three image networks through one "bus" line
    box(ax, 15, 43, 24, 9, "CLIP text encoder\n63 M\n4 class descriptions -> T", FROZEN)
    box(ax, 43, 43, 24, 9, "CLIP image encoder\nViT-B/16, 86 M\nphoto -> g (512)", FROZEN)
    box(ax, 72, 43, 26, 9, "OWLv2 detector + CLIP\ntext prompts -> boxes -> crops\n-> 5 region vectors", FROZEN)
    box(ax, 108, 43, 24, 9, "DINOv2-S\n22 M, self-supervised\nphoto -> d (384)", FROZEN)
    arrow(ax, (15, 58.3), (15, 47.7))
    ax.plot([81, 81], [58.3, 55], color=INK2, lw=1.3)
    ax.plot([43, 108], [55, 55], color=INK2, lw=1.3)
    for x in (43, 72, 108):
        arrow(ax, (x, 55), (x, 47.7))
    ax.text(90, 51.3, "3. frozen networks: run once,\nvectors saved to disk", ha="center", va="center", fontsize=9,
            color=INK2, style="italic")

    box(ax, 43, 24, 58, 9, "4. PRGA graph (trained, 1.1 M parameters)\nnodes: whole photo, object, 2 onions, 2 spots, 4 class texts\n"
        "-> refined support vectors (cache keys) and photo-specific class descriptions", TRAINED)
    box(ax, 108, 24, 30, 9, "DINOv2 cache (trained keys)\nkeys = support photos' DINOv2\nvectors, fine-tuned 20 epochs", TRAINED)
    arrow(ax, (15, 38.5), (22, 28.7))
    arrow(ax, (43, 38.5), (43, 28.7))
    arrow(ax, (72, 38.5), (64, 28.7))
    arrow(ax, (108, 38.5), (108, 28.7))

    box(ax, 72, 8, 66, 8, "5. Score each class = zero-shot + CLIP cache + prototypes + DINOv2 cache\n"
        "highest score -> healthy red / unhealthy red / healthy white / unhealthy white", CALC)
    arrow(ax, (50, 19.5), (62, 12.2))
    arrow(ax, (104, 19.5), (92, 12.2))
    legend(ax, 2, 0.5)
    save(fig, "arch_pipeline.png")


# ----------------------------------------------------------------------------------------------- 2. base paper
def base_paper():
    fig, ax = canvas(13, 8.4, "Base paper (Ahmad et al. 2025): graph over image patches, used only in training")
    ax.text(2, 74, "TRAINING", fontsize=11, fontweight="bold", color=EDGE[TRAINED])
    box(ax, 12, 63, 18, 8, "support photo\nfresh random crop\n+ flip every step", DATA)
    box(ax, 34, 63, 18, 8, "26 windows\nresize to 336 px,\n3x3 and 4x4 grid tiles", CALC)
    box(ax, 56, 63, 18, 8, "CLIP image encoder\nevery window -> 512\n= 26 patch nodes", FROZEN)
    arrow(ax, (21.5, 63), (24.5, 63))
    arrow(ax, (43.5, 63), (46.5, 63))
    # small patch graph
    import math
    cx, cy, r = 84, 63, 8
    pts = [(cx + r * math.cos(2 * math.pi * i / 8), cy + r * math.sin(2 * math.pi * i / 8)) for i in range(8)]
    for i in range(8):
        for j in range(i + 1, 8):
            ax.plot([pts[i][0], pts[j][0]], [pts[i][1], pts[j][1]], color="#f0b27a", lw=0.7, zorder=1)
    for p in pts:
        ax.add_patch(Circle(p, 1.5, fc=TRAINED, ec=EDGE[TRAINED], lw=1.2, zorder=2))
    ax.text(cx, cy - r - 4, "fully connected patch graph\n(8 of 26 nodes drawn)", ha="center", fontsize=8.5, color=INK2)
    arrow(ax, (65.5, 63), (74, 63))
    box(ax, 112, 63, 22, 13, "Graph attention x L\ne_pq = (a.[Wh_p || Wh_q])\n  x sigmoid(Wh_p . Wh_q)\nsoftmax over q, ReLU",
        TRAINED, size=9)
    arrow(ax, (93.5, 63), (100.5, 63))
    box(ax, 112, 41, 22, 9, "Aggregate patches\nmean, max, std\n-> one vector f_hat", TRAINED)
    arrow(ax, (112, 56), (112, 46))
    box(ax, 76, 41, 34, 9, "Cache lookup with LEARNED keys\nA = exp(-beta (1 - f_hat . Theta))\nTheta starts = support photos", TRAINED)
    arrow(ax, (100.5, 41), (93.5, 41))
    box(ax, 36, 41, 34, 9, "Score\nalpha . A . labels + 100 f_hat . W_c\nW_c = \"a photo of a [class]\"", CALC)
    arrow(ax, (58.5, 41), (53.5, 41))
    box(ax, 9, 41, 14, 7, "loss\ncross-entropy", CALC)
    arrow(ax, (18.5, 41), (16.5, 41))

    ax.plot([2, 128], [30, 30], color="#cccccc", lw=1, ls="--")
    ax.text(2, 26, "TEST", fontsize=11, fontweight="bold", color=EDGE[CALC])
    box(ax, 14, 15, 20, 8, "test photo\nwhole photo,\nno patches", DATA)
    box(ax, 40, 15, 20, 8, "CLIP image encoder\n-> f (512)\nno graph", FROZEN)
    box(ax, 72, 15, 30, 8, "same cache (trained keys)\nalpha . exp(-beta (1 - f . Theta)) . labels\n+ 100 f . W_c", CALC)
    box(ax, 105, 15, 18, 7, "prediction\nhighest score", DATA)
    arrow(ax, (24.5, 15), (29.5, 15))
    arrow(ax, (50.5, 15), (56.5, 15))
    arrow(ax, (87.5, 15), (95.5, 15))
    ax.text(65, 4.5, "Problem we found: training queries the cache with f_hat (graph output), testing with f (plain CLIP). "
            "Their cosine is only ~0.78,\nso training moves the keys away from where test photos lie. "
            "Same recipe without the graph scores higher at every K.", ha="center", fontsize=9, color="#b03a2e")
    save(fig, "arch_base_paper.png")


# ----------------------------------------------------------------------------------------------- 3. PRGA
def prga():
    fig, ax = canvas(14, 9.4, "PRGA: one graph per photo, linking the photo, its regions and the 4 class descriptions")
    box(ax, 13, 72, 23, 9, "CLIP image encoder\nwhole photo -> g (512)", FROZEN)
    box(ax, 13, 52, 23, 9, "OWLv2 + CLIP\nobject box, 2 onions, 2 spots\n-> 5 vectors (512 each)", FROZEN)
    box(ax, 13, 30, 23, 9, "CLIP text encoder\n4 class descriptions\n-> T (4 x 512)", FROZEN)

    img = ["whole\nphoto", "object", "onion 1", "onion 2", "spot 1", "spot 2"]
    txt = ["healthy\nred", "unhealthy\nred", "healthy\nwhite", "unhealthy\nwhite"]
    ip = [(47, 80 - i * 9) for i in range(6)]
    tp = [(71, 72 - i * 11) for i in range(4)]
    allp = ip + tp
    for a in range(len(allp)):
        for b in range(a + 1, len(allp)):
            ax.plot([allp[a][0], allp[b][0]], [allp[a][1], allp[b][1]], color="#f3c79b", lw=0.6, zorder=1)
    for pt, t in zip(ip, img):
        ax.add_patch(Circle(pt, 3.6, fc=TRAINED, ec=EDGE[TRAINED], lw=1.3, zorder=2))
        ax.text(*pt, t, ha="center", va="center", fontsize=7.2, zorder=3, color=INK)
    for pt, t in zip(tp, txt):
        ax.add_patch(Circle(pt, 3.9, fc="#f8d0e0", ec="#b0466e", lw=1.3, zorder=2))
        ax.text(*pt, t, ha="center", va="center", fontsize=7, zorder=3, color=INK)
    ax.text(47, 86.5, "6 image nodes", ha="center", fontsize=9.5, fontweight="bold", color=INK)
    ax.text(71, 79, "4 text nodes", ha="center", fontsize=9.5, fontweight="bold", color="#b0466e")
    ax.text(59, 19, "every node is linked to every other;\nedge value = cosine similarity of the two vectors",
            ha="center", fontsize=8.5, color=INK2)
    arrow(ax, (25, 74), (43, 80))                                       # CLIP image -> whole-photo node
    ax.plot([40.5, 40.5], [73, 33], color=INK2, lw=1.3)                 # bracket over the 5 region nodes
    arrow(ax, (25, 52), (40.5, 53))
    arrow(ax, (25, 27.5), (68.5, 34), rad=0.3)                          # text encoder -> text nodes, below spot 2

    box(ax, 109, 66, 38, 19, "Gated graph-attention layer (applied 2 times)\n"
        "1. h = Linear(512 -> 256) + node-type vector\n"
        "2. score_ij = LeakyReLU(a.Wh_i + a'.Wh_j + u.e_ij)\n"
        "3. attention = softmax over j   (4 heads)\n"
        "4. message m_i = sum_j attention_ij . Wh_j\n"
        "5. gate z_i = sigmoid(W_g [h_i, m_i])\n"
        "6. h_i = LayerNorm(z_i.m_i + (1 - z_i).h_i)\n"
        "training only: each region node hidden w.p. 0.2", TRAINED, size=9.2)
    arrow(ax, (76, 60), (89.5, 64))

    box(ax, 99, 33, 27, 12, "Readout (image nodes)\nmean + max + attention pool\n-> Linear -> add g\n= f_hat (refined photo vector)",
        TRAINED, size=9)
    box(ax, 127, 33, 23, 12, "Text output\nLinear 256 -> 512, add T\n= T_hat: descriptions\nadapted to this photo",
        TRAINED, size=9)
    arrow(ax, (103, 56.2), (99, 39.3))
    arrow(ax, (118, 56.2), (127, 39.3))
    box(ax, 99, 12, 27, 8, "used as cache keys\nf_hat of the support photos = K_j", CALC)
    box(ax, 127, 12, 23, 8, "used in the score\nprototype term gamma.100.g.T_hat", CALC, size=8.8)
    arrow(ax, (99, 26.7), (99, 16.3))
    arrow(ax, (127, 26.7), (127, 16.3))
    legend(ax, 2, 1)
    save(fig, "arch_prga.png")


# ----------------------------------------------------------------------------------------------- 4. PRGA + DINOv2
def prga_dino():
    fig, ax = canvas(13, 9.6, "PRGA + DINOv2 cache: how a test photo gets its class (final model)")
    box(ax, 11, 60, 16, 7, "test photo", DATA, bold_first=True)
    box(ax, 38, 74, 24, 8, "CLIP image encoder\n-> g (512)", FROZEN)
    box(ax, 38, 56, 24, 8, "OWLv2 + CLIP + PRGA graph\n-> T_hat (4 x 512)", TRAINED)
    box(ax, 38, 38, 24, 8, "DINOv2-S\n-> d (384)", FROZEN)
    for y in (74, 56, 38):
        arrow(ax, (19.5, 60), (25.5, y))

    rows = [
        (82, "zero-shot", "100 . g . T_c", "does the photo match class c's description?"),
        (66, "CLIP cache  (weight alpha)", "sum_j exp(-beta (1 - g . K_j)) . L_jc",
         "how close to the support photos of class c\n(K_j = support photos through the graph)"),
        (50, "prototypes  (weight gamma)", "100 . g . T_hat_c", "match with the description adapted to this photo"),
        (34, "DINOv2 cache  (weight a2)", "sum_j exp(-b2 (1 - d . D_j)) . L_jc",
         "closeness to the support photos in DINOv2 space\n(D_j fine-tuned 20 epochs)"),
    ]
    for y, name, f, why in rows:
        box(ax, 82, y, 42, 12, f"{name}\n{f}\n{why}", CALC if "zero" in name else TRAINED, size=9)
    arrow(ax, (50.5, 75), (60.5, 82))
    arrow(ax, (50.5, 73), (60.5, 66))
    arrow(ax, (50.5, 72), (60.5, 51), rad=0.12)
    arrow(ax, (50.5, 56), (60.5, 49))
    arrow(ax, (50.5, 38), (60.5, 34))
    box(ax, 117, 58, 12, 8, "sum\nall four terms", CALC)
    for y in (82, 66, 50, 34):
        arrow(ax, (103.5, y), (110.5, 58))
    box(ax, 117, 40, 17, 9, "prediction\nthe class with\nthe highest sum", DATA)
    arrow(ax, (117, 53.5), (117, 45))
    ax.text(65, 15, "L_jc = 1 / (number of support photos of class c) if support photo j is of class c, else 0  "
            "(each class gets an equal vote)\n"
            "alpha, beta: grid search on validation;  gamma: learned;  a2, b2: grid search on validation.\n"
            "The cache is queried with the plain CLIP vector g (chosen on validation); "
            "the graph acts through the keys K_j and the prototypes T_hat.", ha="center", fontsize=9, color=INK2)
    legend(ax, 2, 1)
    save(fig, "arch_prga_dinov2.png")


# ----------------------------------------------------------------------------------------------- 5. pipeline as nodes
def pipeline_nodes():
    import math
    fig, ax = canvas(14.5, 10.6, "The pipeline as a graph of nodes: how one test photo becomes a class")
    R = 5.4
    nodes = {   # name: (x, y, label, colour)
        "desc":    (9, 84, "class\ndescriptions\n(8 per class)", DATA),
        "photo":   (9, 48, "test\nphoto", DATA),
        "cliptxt": (29, 84, "CLIP\ntext", FROZEN),
        "clipimg": (29, 66, "CLIP\nimage", FROZEN),
        "owl":     (29, 48, "OWLv2\ndetector", FROZEN),
        "dino":    (29, 30, "DINOv2", FROZEN),
        "T":       (48, 84, "T\nclass\nvectors", CALC),
        "g":       (48, 66, "g\nphoto\nvector", CALC),
        "crops":   (48, 48, "5 crops\n-> CLIP\nregions", FROZEN),
        "d":       (48, 30, "d\nDINOv2\nvector", CALC),
        "graph":   (68, 57, "PRGA\ngraph\n(10 nodes)", TRAINED),
        "support": (68, 13, "support\nphotos\n(K per class)", DATA),
        "K":       (86, 77, "K\ncache\nkeys", TRAINED),
        "That":    (86, 52, "T_hat\nadapted\nclasses", TRAINED),
        "D":       (86, 18, "D\nDINOv2\nkeys", TRAINED),
        "zs":      (110, 88, "zero-\nshot", CALC),
        "cache":   (110, 70, "CLIP\ncache", CALC),
        "proto":   (110, 50, "proto-\ntypes", CALC),
        "dcache":  (110, 30, "DINOv2\ncache", CALC),
        "sum":     (125, 59, "sum\nof 4\nscores", CALC),
        "cls":     (138.5, 59, "class", DATA),
    }

    def link(p, q, dashed=False, trim_p=True):
        dx, dy = q[0] - p[0], q[1] - p[1]
        n = math.hypot(dx, dy)
        p0 = (p[0] + dx / n * R, p[1] + dy / n * R) if trim_p else p
        p1 = (q[0] - dx / n * (R + 0.4), q[1] - dy / n * (R + 0.4))
        ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle="-|>", mutation_scale=11, lw=1.1, zorder=1,
                                     color="#d9822b" if dashed else INK2, ls="--" if dashed else "-"))

    xy = {k: v[:2] for k, v in nodes.items()}
    for a, b in [("desc", "cliptxt"), ("cliptxt", "T"), ("photo", "clipimg"), ("photo", "owl"), ("photo", "dino"),
                 ("clipimg", "g"), ("owl", "crops"), ("dino", "d"),
                 ("T", "graph"), ("g", "graph"), ("crops", "graph"), ("graph", "That"),
                 ("T", "zs"), ("K", "cache"), ("That", "proto"), ("d", "dcache"), ("D", "dcache"),
                 ("zs", "sum"), ("cache", "sum"), ("proto", "sum"), ("dcache", "sum"), ("sum", "cls")]:
        link(xy[a], xy[b])
    for a, b in [("graph", "K"), ("support", "graph"), ("support", "D")]:
        link(xy[a], xy[b], dashed=True)
    # g goes to three score terms: draw it once along a "bus" line
    BX = 98
    ax.plot([48 + R, BX], [66, 66], color=INK2, lw=1.1, zorder=1)
    ax.plot([BX, BX], [46, 83], color=INK2, lw=1.1, zorder=1)
    for y0, tgt in ((83, "zs"), (66, "cache"), (46, "proto")):
        link((BX, y0), xy[tgt], trim_p=False)
    ax.text(BX - 1, 63.5, "g", fontsize=9, color=INK2, ha="right", style="italic")

    for x, y, label, col in nodes.values():
        ax.add_patch(Circle((x, y), R, fc=col, ec=EDGE[col], lw=1.5, zorder=2))
        ax.text(x, y, label, ha="center", va="center", fontsize=7.6, color=INK, zorder=3, linespacing=1.15)
    for x, t in ((9, "input"), (29, "frozen\nnetworks"), (48, "vectors"), (68, "trained\ngraph"),
                 (86, "graph\noutputs"), (110, "4 score\nterms"), (133, "decision")):
        ax.text(x, 99, t, ha="center", va="top", fontsize=9, fontweight="bold", color=INK2)
    ax.text(100, 6, "dashed orange = done once per training run: the support photos go through the same\n"
            "networks and graph to build the cache keys K and the DINOv2 keys D", ha="center", fontsize=8.6,
            color="#b5651d")
    legend(ax, 2, 0.3)
    save(fig, "arch_pipeline_nodes.png")


if __name__ == "__main__":
    pipeline()
    base_paper()
    prga()
    prga_dino()
    pipeline_nodes()
