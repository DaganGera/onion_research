"""CONTROLLED-STUDY version of the base paper's graph on PRECOMPUTED views (5 per photo), so that only the graph nodes
change between grid windows and detector regions. Layer definitions follow basepaper.py (rho = ReLU in every layer, GAT
init, cosine schedule, Tip-Adapter-F training defaults). The main, exact replication with fresh augmentations is
06d_basepaper_exact.py.

  nodes    : 26 multi-scale patch features per training image (04b_paper_patches.py); fully connected undirected graph
  Eq. 1-2  : h_p' = rho( sum_q alpha_pq W h_q ),  alpha_pq = softmax_q LeakyReLU(e_pq),
             e_pq = (a^T [W h_p || W h_q]) * sigmoid((W h_p)^T (W h_q))      (Attention 1 x sigmoid(Attention 2))
  Eq. 3    : f_hat = sum_m gamma_m * W_m phi_m({h_p}),  phi in {mean, max, std}, gamma_m learnable, then L2-normalised
  Eq. 4    : A = exp(-beta (1 - f . Theta^T)); Theta = learnable cache keys initialised from the support features
  Eq. 5    : logits = alpha * A L  +  f_hat W_c^T      (W_c = fixed class vectors; x100 as CLIP's logit scale)
  training : the graph-refined f_hat is the QUERY; it retrieves labels from the cache and provides the auxiliary logit;
             the resulting loss trains the graph, the pooling and the cache keys.
  test time: NO graph. f_test is the plain CLIP feature: logits = alpha * A(f_test, Theta) L + f_test W_c^T.

Not specified in the paper (our choices, documented): number of layers and activation (selected on validation between
1 layer/no activation and 2 layers/GELU), initialisation (W, W_m = identity so f_hat starts in CLIP space; gamma =
[1, .05, .05]), training alpha/beta (5/5, then alpha/beta grid-searched on validation as in Tip-Adapter), optimiser
(AdamW, lr 1e-3, 60 epochs, best epoch on validation), one-hot label matrix as in the paper (balanced=True is an option).

PaperRegions (bottom) is the CONTROLLED variant: the very same layer, pooling, cache and training, but the nodes are
[whole photo, object, up to 4 OWLv2 regions] (+ optionally 4 class-description nodes) instead of grid patches. The only
addition to the paper's layer is an optional node mask (a photo may have fewer than 4 regions); with no mask the code
path is the paper's.
"""
import itertools

import torch
import torch.nn as nn
import torch.nn.functional as F

from common import DEVICE, l2n
from fewshot import balance, macro_f1, onehot

ALPHAS_P = (0.1, 0.25, 0.5, 1, 2, 3, 5, 8, 12, 16, 24, 32)        # same validation grid as 06d_basepaper_exact.py
BETAS_P = (0.5, 1, 2, 3, 4, 5, 7, 9)


class RelationalGatedLayer(nn.Module):
    def __init__(self, d=512):
        super().__init__()
        self.W = nn.Linear(d, d, bias=False)
        nn.init.eye_(self.W.weight)
        a = torch.empty(2, d)
        nn.init.xavier_uniform_(a.view(2 * d, 1), gain=1.414)          # GAT initialisation (basepaper.py)
        self.a_src = nn.Parameter(a[0].clone())
        self.a_dst = nn.Parameter(a[1].clone())

    def forward(self, h, mask=None):                                         # h [B, P, d], mask [B, P] (1 = real node)
        Wh = self.W(h)
        att1 = (Wh @ self.a_src)[:, :, None] + (Wh @ self.a_dst)[:, None, :]   # a^T [W h_p || W h_q]
        att2 = torch.sigmoid(Wh @ Wh.transpose(1, 2))                          # sigmoid((W h_p)^T W h_q)
        e = F.leaky_relu(att1 * att2, 0.2)
        if mask is not None:
            e = e.masked_fill(mask[:, None, :] == 0, float("-inf"))            # nobody listens to padding
        return torch.relu(torch.softmax(e, dim=-1) @ Wh)                    # Eq. 1: rho = ReLU in every layer


class MultiAggregation(nn.Module):
    def __init__(self, d=512):
        super().__init__()
        self.W = nn.ModuleList(nn.Linear(d, d, bias=False) for _ in range(3))
        for lin in self.W:
            nn.init.eye_(lin.weight)
        self.gamma = nn.Parameter(torch.full((3,), 1 / 3))

    def forward(self, h, mask=None):
        if mask is None:
            stats = [h.mean(1), h.max(1).values, h.std(1)]
        else:
            m = mask.unsqueeze(-1).to(h.dtype)
            n = m.sum(1).clamp_min(1)
            mean = (h * m).sum(1) / n
            mx = h.masked_fill(m == 0, float("-inf")).max(1).values
            std = ((((h - mean[:, None]) ** 2) * m).sum(1) / (n - 1).clamp_min(1) + 1e-8).sqrt()
            stats = [mean, mx, std]
        return l2n(sum(g * lin(s) for g, lin, s in zip(self.gamma, self.W, stats)))


class PaperGraph(nn.Module):
    def __init__(self, layers=2, act="relu"):
        super().__init__()
        self.layers = nn.ModuleList(RelationalGatedLayer() for _ in range(layers))   # rho is inside each layer
        self.pool = MultiAggregation()

    def forward(self, nodes, mask=None, text=None):                          # nodes [B, P, 512] -> [B, 512]
        h, P = nodes, nodes.shape[1]
        attn_mask = mask
        if text is not None:                                                 # class-description nodes join the graph
            B = h.shape[0]
            h = torch.cat([h, text.unsqueeze(0).expand(B, -1, -1)], 1)
            attn_mask = torch.ones(B, h.shape[1], device=h.device)
            if mask is not None:
                attn_mask[:, :P] = mask
        for layer in self.layers:
            h = layer(h, attn_mask)
        return self.pool(h[:, :P], mask)                                     # pooled over the image nodes only


class PaperBase:
    def __init__(self, T, balanced=False, epochs=20, lr=1e-3, wd=1e-2, variants=((1, "relu"), (2, "relu")),
                 alpha_train=1.0, beta_train=1.0):
        self.T = T.to(DEVICE)
        self.balanced, self.epochs, self.lr, self.wd = balanced, epochs, lr, wd
        self.variants, self.a0, self.b0 = variants, alpha_train, beta_train

    # --- what feeds the graph during training (overridden by PaperRegions) ---
    def _train_inputs(self, tr):
        return tr.pp.to(DEVICE)                                              # [N, V, 26, 512]

    def _n_views(self, inp):
        return inp.shape[1]

    def _query(self, net, inp, v):
        return net(inp[:, v])

    def _logits(self, q, keys, L, a, b):
        return a * torch.exp(-b * (1 - q @ keys.T)) @ L + 100 * q @ self.T.T

    def _fit_one(self, layers, act, tr, va):
        torch.manual_seed(0)
        net = PaperGraph(layers, act).to(DEVICE)
        P, Y = self._train_inputs(tr), tr.y.to(DEVICE)
        V = self._n_views(P)
        L = onehot(Y)
        L = balance(L) if self.balanced else L
        keys = nn.Parameter(l2n(tr.aug.mean(1)).to(DEVICE))                  # Theta_0 = support features
        vg, vy = va.g.to(DEVICE), va.y.to(DEVICE)
        opt = torch.optim.AdamW(list(net.parameters()) + [keys], lr=self.lr, weight_decay=self.wd, eps=1e-4)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, self.epochs * V)      # the paper's schedule
        best, best_keys = -1.0, keys.detach().clone()
        for ep in range(self.epochs):
            net.train()
            for v in torch.randperm(V):
                q = self._query(net, P, v)                                   # graph-refined query (training only)
                loss = F.cross_entropy(self._logits(q, keys, L, self.a0, self.b0), Y)
                opt.zero_grad()
                loss.backward()
                opt.step()
                sched.step()
            if len(vy):
                with torch.no_grad():                                        # test-time path: plain feature, no graph
                    sc = macro_f1(self._logits(vg, keys, L, self.a0, self.b0), vy)
                if sc > best:
                    best, best_keys = sc, keys.detach().clone()
        with torch.no_grad():
            a, b = max(itertools.product(ALPHAS_P, BETAS_P),
                       key=lambda ab: macro_f1(self._logits(vg, best_keys, L, *ab), vy)) if len(vy) else (self.a0, self.b0)
            best = macro_f1(self._logits(vg, best_keys, L, a, b), vy) if len(vy) else best
        return dict(net=net, keys=best_keys, L=L, a=a, b=b, val=best)

    def fit(self, tr, va):
        runs = [self._fit_one(layers, act, tr, va) for layers, act in self.variants]
        m = max(runs, key=lambda r: r["val"])                                # architecture chosen on validation
        self.net, self.keys, self.L, self.a, self.b, self.val_f1 = m["net"], m["keys"], m["L"], m["a"], m["b"], m["val"]
        self.layers = len(m["net"].layers)
        return self

    @torch.no_grad()
    def predict(self, te):
        return self._logits(te.g.to(DEVICE), self.keys, self.L, self.a, self.b).softmax(1).cpu()

    def n_params(self):
        return sum(p.numel() for p in self.net.parameters()) + self.keys.numel()


class PaperRegions(PaperBase):
    """Controlled variant: the paper's graph on detector regions (+ optional class-description nodes)."""

    def __init__(self, T, with_text=False, **kw):
        super().__init__(T, **kw)
        self.with_text = with_text

    def _train_inputs(self, tr):
        return tr.gv.to(DEVICE), tr.rv.to(DEVICE), tr.rm.to(DEVICE)         # [N,V,512], [N,V,5,512], [N,5]

    def _n_views(self, inp):
        return inp[1].shape[1]

    def _query(self, net, inp, v):
        G, R, M = inp
        nodes = torch.cat([G[:, v, None], R[:, v]], 1)                       # whole photo + object + 4 region slots
        mask = torch.cat([torch.ones(len(G), 1, device=DEVICE), M], 1)
        return net(nodes, mask, self.T if self.with_text else None)
