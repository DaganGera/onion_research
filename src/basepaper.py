r"""The base paper's model, replicated from its text, equations and figures:

  Ahmad, Sikdar, Pradhan, Behera, "Advancing Cache-Based Few-Shot Classification via Patch-Driven Relational Gated
  Graph Attention", arXiv:2512.12498v1 (13 Dec 2025). No official code exists (the authors' GitHub repo only says
  "code will be made available once paper is accepted"), so every line below is traced to the paper; where the paper
  is silent the choice is marked UNSPECIFIED and the default of Tip-Adapter-F (which the paper says it follows) is used.

What the paper fixes                                          where in the paper
--------------------------------------------------------------------------------------------------------------------
backbone: frozen CLIP ViT-B/16 (image + text)                 Sec. 4 "Implementation Details"
train aug: random resize + crop + horizontal flip, then the   Sec. 4 "Implementation Details"
   image is resized to 336x336
patches: (3x3) and (4x4) grid tiling, adjacent tiles merged   Sec. 4 + Fig. 2; 26 patches = best (Table 4)
   into multi-scale windows, each resized to 224x224 and
   encoded by the frozen CLIP -> one graph node per window
graph: fully connected, undirected, P nodes                   Sec. 3 "Graph-driven Image Patch Embedding"
Eq. 1  h_p^(l+1) = rho( sum_q alpha_pq W^(l+1) h_q^(l) )      rho = non-linear activation, "e.g., ReLU"
       alpha_pq = softmax_q( LeakyReLU(e_pq) )
Eq. 2  e_pq = (a^T [W h_p || W h_q]) * sigmoid((W h_p)^T (W h_q))
              \_______ Attention 1 _____/   \______ Attention 2 ______/
Eq. 3  f_hat = sum_{m in psi} gamma_m * W_m * phi_m({f_hat_p})  gamma_m learnable, W_m linear projection
       psi = {mean, max, std} (text under Eq. 3)
Eq. 4  A = exp(-beta (1 - f_q Theta^T)),  Theta_0 = F_train    learnable cache keys, initialised with the support
                                                              features; f_hat is L2-normalised before the lookup
Eq. 5  logits = alpha * A L_train + f_hat W_c^T               L_train = fixed one-hot labels, W_c = CLIP text
                                                              weights of "a photo of a [CLASS]" (Fig. 1)
test:  logits = alpha * A(f_test, Theta) L_train + f_test W_c^T  -- NO graph at test time (Sec. 3, last paragraph)
optimiser: AdamW, lr 0.001, cosine annealing                  Sec. 4 "Dataset and Experimental Setup"
alpha, beta: "tuned empirically"                              Sec. 4 + Fig. 5 (grid search)

UNSPECIFIED in the paper -> what is used here (and why)
--------------------------------------------------------------------------------------------------------------------
exact 26-window list      every window SHAPE drawn in Fig. 2, at every position: 3x3 grid -> 1x1 (9), 2w x 3h (2),
                          3w x 2h (2) = 13; 4x4 grid -> 2w x 4h (3), 4w x 2h (3), 3w x 2h (6) = 12; + whole image
                          = 26. (13 per grid also matches the "13" column of Table 4.)
number of layers L        1 or 2, chosen on validation (Fig. 1 draws two graph stages)
rho                       ReLU (the paper's own example), applied in every layer as Eq. 1 writes it
loss                      cross-entropy on Eq. 5 logits (as Tip-Adapter-F)
epochs / batch / eps      20 epochs, batch 256, AdamW eps 1e-4, weight decay = PyTorch default: Tip-Adapter-F
                          defaults, the recipe whose "AdamW + lr 0.001 + cosine annealing" the paper repeats
Theta_0 = F_train         mean of 10 augmented 224-px views per support photo, L2-normalised (Tip-Adapter's cache)
logit scale               100 * f W_c^T, CLIP's logit scale, as in Tip-Adapter's zero-shot term
alpha, beta in training   UNSPECIFIED: (1,1) [Tip-Adapter's initial values], (10,1), (10,5) trained in parallel, chosen on
                          validation; the test-time alpha, beta are then grid-searched on validation (Fig. 5)
initialisation            W, W_m = identity (so the graph starts in CLIP space); a ~ Xavier-uniform as in GAT [39],
                          which Attention 1 "is nothing more than"; gamma_m = 1/|psi|
bias in W, W_m            none (Eq. 1-3 have no bias term)
epoch selection           best validation macro-F1 (never the test set)
psi alternatives          Fig. 1 draws aggregators [min, max, softmax] and PNA scalers [pass, boost, suppress]. On a
                          fully connected graph every node has degree P-1, so all PNA scalers equal 1; the
                          {min, max, softmax} set is run as an ablation ("aggFig1").
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

SIZE = 336


def _grid_windows(n, shapes, size=SIZE):
    """All placements of (w_tiles, h_tiles) windows on an n x n grid of a size x size image -> (top, left, bottom, right)."""
    t = size / n
    out = []
    for w, h in shapes:
        for r in range(n - h + 1):
            for c in range(n - w + 1):
                out.append((round(r * t), round(c * t), round((r + h) * t), round((c + w) * t)))
    return out


# Fig. 2 reading (main): shapes are (width, height) in tiles
FIG2_WINDOWS = (_grid_windows(3, [(1, 1), (2, 3), (3, 2)])
                + _grid_windows(4, [(2, 4), (4, 2), (3, 2)])
                + [(0, 0, SIZE, SIZE)])
# The earlier re-implementation's reading (4x4 grid only, windows of 2-3 tiles per side + whole image): ablation
LEGACY_WINDOWS = _grid_windows(4, [(2, 2), (2, 3), (3, 2), (3, 3)]) + [(0, 0, SIZE, SIZE)]
assert len(FIG2_WINDOWS) == 26 and len(LEGACY_WINDOWS) == 26
ALL_WINDOWS = list(dict.fromkeys(FIG2_WINDOWS + LEGACY_WINDOWS))     # union, encoded once per image per step
IDX = {"fig2": [ALL_WINDOWS.index(w) for w in FIG2_WINDOWS],
       "legacy": [ALL_WINDOWS.index(w) for w in LEGACY_WINDOWS]}


class RelationalGatedLayer(nn.Module):
    """Eq. 1 + Eq. 2. mode: 'a1a2' (paper), 'a1' / 'a2' (Table 3 ablations), 'noedge' (Table 5: no inter-patch edges,
    every node only sees itself)."""

    def __init__(self, d=512, mode="a1a2", act="relu"):
        super().__init__()
        self.mode = mode
        self.W = nn.Linear(d, d, bias=False)
        nn.init.eye_(self.W.weight)
        a = torch.empty(2 * d, 1)
        nn.init.xavier_uniform_(a, gain=1.414)                    # GAT's initialisation of the attention vector
        self.a = nn.Parameter(a.squeeze(1))                       # a = [a_src ; a_dst]
        self.rho = {"relu": nn.ReLU(), "identity": nn.Identity(), "gelu": nn.GELU()}[act]
        self.last_alpha = None

    def forward(self, h):                                          # h [B, P, d]
        Wh = self.W(h)
        d = Wh.shape[-1]
        if self.mode == "noedge":
            return self.rho(Wh)                                    # alpha = identity matrix
        att1 = (Wh @ self.a[:d])[:, :, None] + (Wh @ self.a[d:])[:, None, :]    # a^T [W h_p || W h_q]
        att2 = torch.sigmoid(Wh @ Wh.transpose(1, 2))                           # sigma((W h_p)^T (W h_q))
        e = {"a1a2": att1 * att2, "a1": att1, "a2": att2}[self.mode]
        alpha = torch.softmax(F.leaky_relu(e, 0.2), dim=-1)                     # softmax over q
        self.last_alpha = alpha.detach()
        return self.rho(alpha @ Wh)                                             # Eq. 1


class SoftmaxAgg(nn.Module):
    """Softmax aggregation (Fig. 1's 'softmax' aggregator): per feature, sum_p softmax_p(t x_p) x_p, t learnable."""

    def __init__(self):
        super().__init__()
        self.t = nn.Parameter(torch.ones(1))

    def forward(self, h):
        return (torch.softmax(self.t * h, dim=1) * h).sum(1)


class MultiAggregation(nn.Module):
    """Eq. 3: f_hat = sum_m gamma_m W_m phi_m({h_p}), then L2-normalised (Sec. 3 'Cache Refinement')."""

    def __init__(self, d=512, psi=("mean", "max", "std")):
        super().__init__()
        self.psi = psi
        self.W = nn.ModuleList(nn.Linear(d, d, bias=False) for _ in psi)
        for lin in self.W:
            nn.init.eye_(lin.weight)
        self.gamma = nn.Parameter(torch.full((len(psi),), 1.0 / len(psi)))
        self.soft = SoftmaxAgg() if "softmax" in psi else None

    def phi(self, m, h):
        if m == "mean":
            return h.mean(1)
        if m == "max":
            return h.max(1).values
        if m == "min":
            return h.min(1).values
        if m == "std":
            return h.std(1)
        if m == "softmax":
            return self.soft(h)
        raise ValueError(m)

    def forward(self, h):
        return F.normalize(sum(g * lin(self.phi(m, h)) for g, lin, m in zip(self.gamma, self.W, self.psi)), dim=-1)


class PatchGraph(nn.Module):
    """L relational layers + multi-aggregation: P patch features of one image -> one refined vector f_hat."""

    def __init__(self, layers=2, mode="a1a2", psi=("mean", "max", "std"), act="relu", act_last=None):
        super().__init__()
        acts = [act] * (layers - 1) + [act if act_last is None else act_last]
        self.layers = nn.ModuleList(RelationalGatedLayer(mode=mode, act=a) for a in acts)
        self.pool = MultiAggregation(psi=psi)

    def forward(self, patches):                                    # [B, P, 512] -> [B, 512]
        h = patches
        for layer in self.layers:
            h = layer(h)
        return self.pool(h)


def paper_logits(q, keys, L, W_c, alpha, beta):
    """Eq. 4 + Eq. 5: alpha * exp(-beta (1 - q Theta^T)) L + 100 q W_c^T  (q, keys, W_c L2-normalised)."""
    return alpha * torch.exp(-beta * (1 - q @ keys.T)) @ L + 100 * q @ W_c.T


class Head(nn.Module):
    """One trainable configuration: graph + its own cache keys. Many heads share the same CLIP encodings."""

    def __init__(self, name, keys0, L, W_c, layers, windows="fig2", mode="a1a2", psi=("mean", "max", "std"),
                 act="relu", act_last=None, query="graph"):
        super().__init__()
        self.name, self.windows, self.query = name, windows, query
        self.graph = PatchGraph(layers, mode, psi, act, act_last)
        self.keys = nn.Parameter(keys0.clone())                     # Theta_0 = F_train
        self.register_buffer("L", L)
        self.register_buffer("W_c", W_c)

    def train_logits(self, patches_all, alpha, beta):              # training: graph-refined query (Eq. 3-5)
        if self.query == "plain":                                  # control: Tip-Adapter-F, query = whole-image window
            q = patches_all[:, ALL_WINDOWS.index((0, 0, SIZE, SIZE))]
            return paper_logits(q, self.keys, self.L, self.W_c, alpha, beta)
        q = self.graph(patches_all[:, IDX[self.windows]])
        return paper_logits(q, self.keys, self.L, self.W_c, alpha, beta)

    def test_logits(self, f, alpha, beta, keys=None):              # inference: plain CLIP feature, no graph
        return paper_logits(f, self.keys if keys is None else keys, self.L, self.W_c, alpha, beta)

    def n_params(self):
        if self.query == "plain":
            return self.keys.numel()
        return sum(p.numel() for p in self.parameters())
