"""Few-shot methods that run on cached features: zero-shot CLIP, linear probe, Tip-Adapter(-F), the PlantCaFo-style
two-backbone cache and PRGA. They all work the same way:

    model = Method(...); model.fit(support_batch, val_batch); probs = model.predict(test_batch)
"""
import itertools
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from common import C, DEVICE, l2n, load_feats, load_meta, metrics


# ----------------------------------------------------------------------------- data
@dataclass
class Batch:
    y: torch.Tensor                 # [N]
    g: torch.Tensor                 # [N, D]      global CLIP feature
    aug: torch.Tensor | None = None  # [N, V, D]  augmented global views (support only)
    grid: torch.Tensor | None = None  # [N, 9, D]
    reg: torch.Tensor | None = None  # [N, 1+M, D] object + regions
    mask: torch.Tensor | None = None  # [N, 1+M]
    geom: torch.Tensor | None = None  # [N, 1+M, 6]  (cx, cy, w, h, score, kind) kind: 1 obj 2 instance 3 spot
    g2: torch.Tensor | None = None   # [N, D2] second backbone (DINOv2)
    aug2: torch.Tensor | None = None  # [N, V, D2] augmented views, second backbone

    def __len__(self):
        return len(self.y)

    def to(self, dev):
        return Batch(**{k: (v.to(dev) if torch.is_tensor(v) else v) for k, v in self.__dict__.items()})


class Store:
    """Loads cached feature files once and builds a Batch for any split CSV."""

    def __init__(self, backbone="clip", second=None, need=("global", "aug")):
        meta = load_meta()
        self.label_of = dict(zip(meta.path, meta.label))
        self.f = {}
        for kind in need:
            try:
                self.f[kind] = load_feats(f"{backbone}_{kind}")
            except FileNotFoundError:
                pass
        if second:
            self.f["g2"] = load_feats(f"{second}_global")
            self.f["aug2"] = load_feats(f"{second}_aug")
        # every feature file is indexed by ITS OWN stored paths (never by row position in meta.csv, which changes when
        # the dataset is cleaned)
        self.rows = {k: {p: i for i, p in enumerate(v["paths"])} for k, v in self.f.items() if "paths" in v}
        self.aug_row = {p: i for i, p in enumerate(self.f["aug"]["paths"])} if "aug" in self.f else {}
        self.aug2_row = {p: i for i, p in enumerate(self.f["aug2"]["paths"])} if "aug2" in self.f else {}

    def batch(self, df, with_aug=False):
        ix = lambda k: torch.tensor([self.rows[k][p] for p in df.path], dtype=torch.long)   # noqa: E731
        b = Batch(y=torch.tensor([int(self.label_of[p]) for p in df.path], dtype=torch.long),
                  g=self.f["global"]["feats"][ix("global")].float())
        if "grid" in self.f:
            b.grid = self.f["grid"]["feats"][ix("grid")].float()
        if "regions" in self.f:
            r, idx = self.f["regions"], ix("regions")
            b.reg, b.mask = r["feats"][idx].float(), r["mask"][idx].float()
            b.geom = with_kind(r["geom"][idx].float(), r["kind"][idx] if "kind" in r else None)
        if "g2" in self.f:
            b.g2 = self.f["g2"]["feats"][ix("g2")].float()
        if with_aug:
            b.aug = self.f["aug"]["feats"][torch.tensor([self.aug_row[p] for p in df.path])].float()
            if "aug2" in self.f:
                b.aug2 = self.f["aug2"]["feats"][torch.tensor([self.aug2_row[p] for p in df.path])].float()
        return b


def with_kind(geom5, kind=None):
    """Append the node-kind id as a 6th geometry column (default: object first, then instances)."""
    if kind is None:
        kind = torch.full(geom5.shape[:-1], 2)
        kind[..., 0] = 1
    return torch.cat([geom5, kind.float().unsqueeze(-1)], -1)


def onehot(y):
    return F.one_hot(y, C).float()


def balance(L):
    """Class-balanced cache values: average (not sum) the affinities of each class's cache entries.
    Identical to Tip-Adapter up to a constant when every class has K shots; needed for the imbalanced full pool."""
    return L / L.sum(0, keepdim=True).clamp_min(1)


def macro_f1(probs, y):
    return metrics(y.cpu().numpy(), probs.argmax(1).cpu().numpy())["macro_f1"]


# ----------------------------------------------------------------------------- baselines
class ZeroShot:
    def __init__(self, T, **_):
        self.T = T

    def fit(self, tr, va):
        return self

    def predict(self, te):
        return (100 * te.g @ self.T.T).softmax(1)


class LinearProbe:
    def __init__(self, T=None, **_):
        pass

    def fit(self, tr, va):
        from sklearn.linear_model import LogisticRegression
        X = tr.aug.flatten(0, 1).numpy()
        y = tr.y.repeat_interleave(tr.aug.shape[1]).numpy()
        best = (-1, None)
        for c in (1.0, 10.0, 100.0):
            m = LogisticRegression(C=c, max_iter=3000).fit(X, y)
            s = macro_f1(self._p(m, va.g), va.y) if len(va) else 0
            best = max(best, (s, m), key=lambda t: t[0])
        self.m = best[1]
        return self

    def _p(self, m, X):
        p = np.zeros((len(X), C), dtype=np.float32)
        p[:, m.classes_] = m.predict_proba(X.numpy())
        return torch.from_numpy(p)

    def predict(self, te):
        return self._p(self.m, te.g)


def tip_logits(q, keys, L, T, alpha, beta, zs_q=None):
    """Tip-Adapter: zero-shot prior + alpha * exp(-beta (1 - cos)) cache term."""
    zs = 100 * (q if zs_q is None else zs_q) @ T.T
    A = torch.exp(-beta * (1 - q @ keys.T))
    return zs + alpha * A @ balance(L)


ALPHAS, BETAS = (0.5, 1, 2, 3, 5, 8, 12), (1, 3, 5, 7, 9)


def eval_epochs(epochs):
    """The 10 evenly spaced epochs at which validation is checked when pick_epoch is on."""
    every = max(1, epochs // 10)
    return {e for e in range(every, epochs + 1, every)} | {epochs}


def search_ab(score_fn, va):
    if len(va) == 0:
        return 1.0, 5.0
    return max(itertools.product(ALPHAS, BETAS), key=lambda ab: macro_f1(score_fn(*ab), va.y))


class TipAdapter:
    """Training-free Tip-Adapter: the cache is the (augmentation-averaged) support set."""

    def __init__(self, T, **_):
        self.T = T

    def fit(self, tr, va):
        self.keys, self.L = l2n(tr.aug.mean(1)), onehot(tr.y)
        self.a, self.b = search_ab(lambda a, b: tip_logits(va.g, self.keys, self.L, self.T, a, b), va)
        return self

    def predict(self, te):
        return tip_logits(te.g, self.keys, self.L, self.T, self.a, self.b).softmax(1)


class TipAdapterF(TipAdapter):
    """Tip-Adapter-F: cache keys become learnable and are fine-tuned on augmented support views."""

    pick_epoch = False          # True: keep the best checkpoint on validation (18_equal_training.py)

    def __init__(self, T, epochs=20, lr=1e-3, **_):
        super().__init__(T)
        self.epochs, self.lr = epochs, lr

    def _val_score(self, keys, L, va):
        k = keys.detach().cpu()
        a, b = search_ab(lambda a, b: tip_logits(va.g, k, L.cpu(), self.T, a, b), va)
        return macro_f1(tip_logits(va.g, k, L.cpu(), self.T, a, b), va.y), k

    def fit(self, tr, va):
        T = self.T.to(DEVICE)
        keys = nn.Parameter(l2n(tr.aug.mean(1)).to(DEVICE))
        L = onehot(tr.y).to(DEVICE)
        X, Y = tr.aug.to(DEVICE), tr.y.to(DEVICE)
        opt = torch.optim.AdamW([keys], lr=self.lr, eps=1e-4)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, self.epochs * X.shape[1])
        check = eval_epochs(self.epochs) if self.pick_epoch and len(va) else set()
        best, self.best_epoch = None, self.epochs
        for ep in range(self.epochs):
            for v in torch.randperm(X.shape[1]):
                loss = F.cross_entropy(tip_logits(X[:, v], keys, L, T, 5.0, 5.0), Y)   # alpha_train=5 chosen on val
                opt.zero_grad(); loss.backward(); opt.step(); sched.step()
            if ep + 1 in check:
                sc, k = self._val_score(keys, L, va)
                if best is None or sc > best[0]:
                    best, self.best_epoch = (sc, k), ep + 1
        self.keys, self.L = (best[1] if best else keys.detach().cpu()), L.cpu()
        self.a, self.b = search_ab(lambda a, b: tip_logits(va.g, self.keys, self.L, self.T, a, b), va)
        return self


class PlantCaFoLite:
    """Re-implementation of PlantCaFo's idea: several foundation-model caches (CLIP + DINOv2) fused with the
    CLIP zero-shot prior, each cache fine-tuned Tip-Adapter-F style. Not the authors' code."""

    pick_epoch = False          # True: keep the best checkpoint on validation (18_equal_training.py)

    def __init__(self, T, epochs=20, lr=1e-3, **_):
        self.T, self.epochs, self.lr = T, epochs, lr

    @staticmethod
    def _logits(q, q2, k1, k2, L, T, a1, a2, b):
        L = balance(L)
        return 100 * q @ T.T + a1 * torch.exp(-b * (1 - q @ k1.T)) @ L + a2 * torch.exp(-b * (1 - q2 @ k2.T)) @ L

    def fit(self, tr, va):
        T = self.T.to(DEVICE)
        k1 = nn.Parameter(l2n(tr.aug.mean(1)).to(DEVICE))
        k2 = nn.Parameter(l2n(tr.aug2.mean(1)).to(DEVICE))
        L = onehot(tr.y).to(DEVICE)
        X, X2, Y = tr.aug.to(DEVICE), tr.aug2.to(DEVICE), tr.y.to(DEVICE)
        opt = torch.optim.AdamW([k1, k2], lr=self.lr, eps=1e-4)
        check = eval_epochs(self.epochs) if self.pick_epoch and len(va) else set()
        best, self.best_epoch = None, self.epochs
        for ep in range(self.epochs):
            for v in torch.randperm(X.shape[1]):
                loss = F.cross_entropy(self._logits(X[:, v], X2[:, v], k1, k2, L, T, 2.5, 2.5, 5), Y)
                opt.zero_grad(); loss.backward(); opt.step()
            if ep + 1 in check:
                self.k1, self.k2, self.L = k1.detach().cpu(), k2.detach().cpu(), L.cpu()
                hp, sc = self._search(va)
                if best is None or sc > best[0]:
                    best, self.best_epoch = (sc, self.k1, self.k2, hp), ep + 1
        if best:
            _, self.k1, self.k2, self.hp = best
            self.L = L.cpu()
            return self
        self.k1, self.k2, self.L = k1.detach().cpu(), k2.detach().cpu(), L.cpu()
        self.hp = self._search(va)[0] if len(va) else (1, 1, 5)
        return self

    def _search(self, va):
        """Grid over the two cache weights and the shared sharpness, on validation."""
        grid = list(itertools.product((0.5, 1, 2, 4, 8), (0.5, 1, 2, 4, 8), (1, 3, 5, 7)))
        score = lambda h: macro_f1(self._logits(va.g, va.g2, self.k1, self.k2, self.L, self.T, *h), va.y)  # noqa: E731
        hp = max(grid, key=score)
        return hp, score(hp)

    def predict(self, te):
        return self._logits(te.g, te.g2, self.k1, self.k2, self.L, self.T, *self.hp).softmax(1)


# ----------------------------------------------------------------------------- graph layers
class GatedGAT(nn.Module):
    """Dense multi-head graph attention with edge features and a per-node gate.

    score_ij = LeakyReLU(a_src.Wh_i + a_dst.Wh_j + u.e_ij);  a = softmax_j(score) over real nodes
    m_i = sum_j a_ij Wh_j;  z_i = sigmoid(W_g [h_i || m_i]);  h_i' = LN(z_i * m_i + (1 - z_i) * h_i)
    """

    def __init__(self, dim, heads=4, edge_dim=4, dropout=0.1, gate=True):
        super().__init__()
        self.h, self.dh, self.gate = heads, dim // heads, gate
        self.W = nn.Linear(dim, dim, bias=False)
        self.a_src = nn.Parameter(torch.randn(heads, self.dh) * 0.1)
        self.a_dst = nn.Parameter(torch.randn(heads, self.dh) * 0.1)
        self.U = nn.Linear(edge_dim, heads, bias=False)
        self.Wg = nn.Linear(2 * dim, dim)
        self.ln = nn.LayerNorm(dim)
        self.drop = nn.Dropout(dropout)

    def forward(self, h, e, node_mask):
        B, N, D = h.shape
        Wh = self.W(h).view(B, N, self.h, self.dh)                              # [B,N,H,dh]
        s = (Wh * self.a_src).sum(-1).transpose(1, 2).unsqueeze(-1) \
            + (Wh * self.a_dst).sum(-1).transpose(1, 2).unsqueeze(-2) \
            + self.U(e).permute(0, 3, 1, 2)                                     # [B,H,N,N]
        s = F.leaky_relu(s, 0.2).masked_fill(node_mask[:, None, None, :] == 0, float("-inf"))
        a = self.drop(torch.softmax(s, -1))
        m = torch.einsum("bhij,bjhd->bihd", a, Wh).reshape(B, N, D)
        if self.gate:
            z = torch.sigmoid(self.Wg(torch.cat([h, m], -1)))
            return self.ln(z * m + (1 - z) * h)
        return self.ln(m + h)


class Readout(nn.Module):
    """Multi-aggregation pooling (masked mean + max + attention) over image nodes."""

    def __init__(self, dim, out):
        super().__init__()
        self.q = nn.Linear(dim, 1)
        self.proj = nn.Linear(3 * dim, out)

    def forward(self, h, mask):
        m = mask.unsqueeze(-1)
        mean = (h * m).sum(1) / m.sum(1).clamp_min(1)
        mx = h.masked_fill(m == 0, -1e4).max(1).values
        w = self.q(h).squeeze(-1).masked_fill(mask == 0, float("-inf")).softmax(-1)
        att = (w.unsqueeze(-1) * h).sum(1)
        return self.proj(torch.cat([mean, mx, att], -1))


def box_iou_dist(geom):
    """geom [B,N,>=5] (cx,cy,w,h,score,...) -> iou [B,N,N], centre distance [B,N,N], score product [B,N,N]."""
    cx, cy, w, h, sc = geom[..., :5].unbind(-1)
    x0, y0, x1, y1 = cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2
    ix = (torch.minimum(x1[:, :, None], x1[:, None]) - torch.maximum(x0[:, :, None], x0[:, None])).clamp_min(0)
    iy = (torch.minimum(y1[:, :, None], y1[:, None]) - torch.maximum(y0[:, :, None], y0[:, None])).clamp_min(0)
    inter = ix * iy
    area = w * h
    iou = inter / (area[:, :, None] + area[:, None] - inter).clamp_min(1e-6)
    dist = ((cx[:, :, None] - cx[:, None]) ** 2 + (cy[:, :, None] - cy[:, None]) ** 2).sqrt()
    return iou, dist, sc[:, :, None] * sc[:, None]


# ----------------------------------------------------------------------------- PRGA
class PRGANet(nn.Module):
    """Graph over image nodes (global, object, M regions — or 9 grid patches) and C class-text nodes."""

    def __init__(self, T, D=512, H=256, layers=2, heads=4, dropout=0.2, text_nodes=True, gate=True, use_geom=True):
        super().__init__()
        self.register_buffer("T", T)
        self.text_nodes, self.use_geom = text_nodes, use_geom
        self.inp = nn.Linear(D, H)
        self.type_emb = nn.Embedding(5, H)          # 0 global, 1 object, 2 onion instance / grid patch, 3 spot, 4 text
        self.geo = nn.Sequential(nn.Linear(5, H), nn.GELU(), nn.Linear(H, H))
        self.gnn = nn.ModuleList(GatedGAT(H, heads, 4, dropout, gate) for _ in range(layers))
        self.readout = Readout(H, D)
        self.text_out = nn.Linear(H, D)
        self.node_drop = dropout

    def forward(self, g, nodes, mask, geom):
        """g [B,D] global; nodes [B,K,D] other image nodes; mask [B,K]; geom [B,K,5].
        Returns refined image embedding [B,D] and refined, image-conditioned class prototypes [B,C,D]."""
        B, K, _ = nodes.shape
        img = torch.cat([g[:, None], nodes], 1)                                     # [B,1+K,D]
        imask = torch.cat([torch.ones(B, 1, device=g.device), mask], 1)
        igeom = torch.cat([torch.tensor([.5, .5, 1., 1., 1., 0.], device=g.device).expand(B, 1, 6), geom], 1)
        types = igeom[..., 5].long().clamp(0, 3)                                    # [B,1+K]; global row = 0
        if self.training and self.node_drop > 0:                                    # randomly hide region nodes
            keep = (torch.rand_like(imask) > self.node_drop).float()
            keep[:, 0] = 1
            imask = imask * keep
        h = self.inp(img) + self.type_emb(types)
        if self.use_geom:
            h = h + self.geo(igeom[..., :5])
        N_img = h.shape[1]
        iou, dist, sp = box_iou_dist(igeom)
        cos_ii = img @ img.transpose(1, 2)
        e_ii = torch.stack([iou, dist, sp, cos_ii], -1)
        if not self.use_geom:
            e_ii = e_ii * torch.tensor([0., 0., 0., 1.], device=g.device)
        if self.text_nodes:
            Tn = self.T.expand(B, -1, -1)
            h = torch.cat([h, self.inp(Tn) + self.type_emb.weight[4]], 1)
            cos_it = img @ self.T.T                                                  # [B,Ni,C]
            cos_tt = (self.T @ self.T.T).expand(B, -1, -1)
            N = h.shape[1]
            e = torch.zeros(B, N, N, 4, device=g.device)
            e[:, :N_img, :N_img] = e_ii
            e[:, :N_img, N_img:, 3] = cos_it
            e[:, N_img:, :N_img, 3] = cos_it.transpose(1, 2)
            e[:, N_img:, N_img:, 3] = cos_tt
            nmask = torch.cat([imask, torch.ones(B, self.T.shape[0], device=g.device)], 1)
        else:
            e, nmask = e_ii, imask
        for layer in self.gnn:
            h = layer(h, e, nmask)
        f_hat = l2n(g + self.readout(h[:, :N_img], imask))
        T_hat = l2n(self.T[None] + self.text_out(h[:, N_img:])) if self.text_nodes else None
        return f_hat, T_hat


class PRGA:
    """Prompt-grounded Region Graph Adapter.

    logits = 100 f.T  +  alpha * exp(-beta (1 - f_hat . F_hat)) L  +  gamma * 100 f_hat . T_hat
    nodes='regions' uses the OWLv2 boxes, nodes='grid' a 3x3 grid (ablation). With test_graph=False the graph refines
    the support photos (cache keys) and the class prototypes, but the cache is queried with the plain CLIP feature of
    the test photo. Validation chose test_graph=False.
    """

    def __init__(self, T, nodes="regions", text_nodes=True, gate=True, use_geom=True, test_graph=True,
                 M=None, keep_kinds=None, H=256, layers=2, epochs=60, lr=1e-3, wd=1e-4, dropout=0.2, second=False, **_):
        self.T, self.nodes, self.test_graph, self.M, self.keep_kinds = T, nodes, test_graph, M, keep_kinds
        self.second = second          # add a second-backbone (DINOv2) cache, PlantCaFo-style (needs Store(second=...))
        self.cfg = dict(H=H, layers=layers, dropout=dropout, text_nodes=text_nodes, gate=gate,
                        use_geom=use_geom and nodes == "regions")
        self.epochs, self.lr, self.wd = epochs, lr, wd

    def _nodes(self, b):
        if self.nodes == "grid":
            n = b.grid
            gm = torch.zeros(*n.shape[:2], 6)
            gm[..., 5] = 2
            return n, torch.ones(n.shape[:2]), gm
        n, m, gm = b.reg, b.mask, b.geom
        if self.keep_kinds is not None:                 # ablation: keep only some region kinds (object always kept)
            keep = torch.zeros_like(m, dtype=torch.bool)
            for k in (1, *self.keep_kinds):
                keep |= gm[..., 5] == k
            m = m * keep
        if self.M is not None:
            n, m, gm = n[:, : 1 + self.M], m[:, : 1 + self.M], gm[:, : 1 + self.M]
        return n, m, gm

    def _logits(self, g, fq, Tq, keys, L, s, q2=None):
        zs = 100 * g @ self.net.T.T
        cache = s["alpha"] * torch.exp(-s["beta"] * (1 - fq @ keys.T)) @ balance(L)
        out = zs + cache
        if self.second and q2 is not None:
            out = out + s["alpha2"] * torch.exp(-s["beta"] * (1 - q2 @ self.k2.T)) @ balance(L)
        if Tq is not None:
            out = out + s["gamma"] * 100 * torch.einsum("bd,bcd->bc", fq, Tq)
        return out

    def _embed(self, g, n, m, gm, bs=2048):
        outs_f, outs_t = [], []
        for i in range(0, len(g), bs):
            f, t = self.net(g[i:i + bs], n[i:i + bs], m[i:i + bs], gm[i:i + bs])
            outs_f.append(f)
            outs_t.append(t)
        return torch.cat(outs_f), (torch.cat(outs_t) if outs_t[0] is not None else None)

    def fit(self, tr, va):
        torch.manual_seed(0)
        self.net = PRGANet(self.T, D=self.T.shape[1], **self.cfg).to(DEVICE)
        self.s = {"alpha": nn.Parameter(torch.tensor(5.0, device=DEVICE)),   # same start as Tip-Adapter-F training
                  "beta": nn.Parameter(torch.tensor(5.0, device=DEVICE)),
                  "gamma": nn.Parameter(torch.tensor(0.5, device=DEVICE))}
        params2 = []
        if self.second:
            self.k2 = nn.Parameter(l2n(tr.aug2.mean(1)).to(DEVICE))
            self.s["alpha2"] = nn.Parameter(torch.tensor(2.5, device=DEVICE))
            X2 = tr.aug2.to(DEVICE)
            params2 = [self.k2]
        n, m, gm = [x.to(DEVICE) for x in self._nodes(tr)]
        X, Y, L = tr.aug.to(DEVICE), tr.y.to(DEVICE), onehot(tr.y).to(DEVICE)
        g_clean = l2n(tr.aug.mean(1)).to(DEVICE)
        vn, vm, vgm = [x.to(DEVICE) for x in self._nodes(va)] if len(va) else (None,) * 3
        opt = torch.optim.AdamW(list(self.net.parameters()) + list(self.s.values()) + params2, lr=self.lr,
                                weight_decay=self.wd)
        g2v = va.g2.to(DEVICE) if (self.second and len(va)) else None
        steps = self.epochs * X.shape[1]
        sched = torch.optim.lr_scheduler.OneCycleLR(opt, self.lr, total_steps=steps, pct_start=0.1)
        best, best_state, step = -1, None, 0
        for ep in range(self.epochs):
            self.net.train()
            for v in torch.randperm(X.shape[1]):
                keys, _ = self.net(g_clean, n, m, gm)                        # refined support keys
                gq = X[:, v]
                fq, Tq = self.net(gq, n, m, gm)                              # refined augmented queries
                if not self.test_graph:
                    fq = gq
                q2 = X2[:, v % X2.shape[1]] if self.second else None
                loss = F.cross_entropy(self._logits(gq, fq, Tq, keys, L, self.s, q2), Y, label_smoothing=0.1)
                opt.zero_grad(); loss.backward(); opt.step(); sched.step(); step += 1
            if vn is not None and (ep % 5 == 4 or ep == self.epochs - 1):
                self._freeze(g_clean, n, m, gm, L)
                sc = macro_f1(self._predict(va.g.to(DEVICE), vn, vm, vgm, g2v), va.y)
                if sc > best:
                    best = sc
                    best_state = ({k: v.clone() for k, v in self.net.state_dict().items()},
                                  {k: v.detach().clone() for k, v in self.s.items()},
                                  self.k2.detach().clone() if self.second else None)
        if best_state is not None:
            self.net.load_state_dict(best_state[0])
            self.s = best_state[1]
            if self.second:
                self.k2 = best_state[2]
        self._freeze(g_clean, n, m, gm, L)
        if vn is not None:                       # same post-hoc alpha/beta search on validation as Tip-Adapter
            with torch.no_grad():
                g_v = va.g.to(DEVICE)
                fq, Tq = self._embed(g_v, vn, vm, vgm)
                if not self.test_graph:
                    fq = g_v
                gam = self.s["gamma"]
                extra = {"alpha2": self.s["alpha2"]} if self.second else {}

                def val_score(a, b):
                    s = {"alpha": torch.tensor(float(a)), "beta": torch.tensor(float(b)), "gamma": gam, **extra}
                    return macro_f1(self._logits(g_v, fq, Tq, self.keys, self.L, s, g2v).cpu(), va.y)
                a, b = max(itertools.product(ALPHAS, BETAS), key=lambda ab: val_score(*ab))
                self.s = {"alpha": torch.tensor(float(a), device=DEVICE), "beta": torch.tensor(float(b), device=DEVICE),
                          "gamma": gam, **extra}
        self.val_f1 = best
        return self

    @torch.no_grad()
    def _freeze(self, g, n, m, gm, L):
        self.net.eval()
        self.keys, _ = self._embed(g, n, m, gm)
        self.L = L

    @torch.no_grad()
    def _predict(self, g, n, m, gm, g2=None):
        self.net.eval()
        fq, Tq = self._embed(g, n, m, gm)
        if not self.test_graph:
            fq = g
        return self._logits(g, fq, Tq, self.keys, self.L, self.s, g2).softmax(1).cpu()

    def predict(self, te):
        n, m, gm = [x.to(DEVICE) for x in self._nodes(te)]
        return self._predict(te.g.to(DEVICE), n, m, gm, te.g2.to(DEVICE) if self.second else None)

    def n_params(self):
        return sum(p.numel() for p in self.net.parameters() if p.requires_grad) + 3
