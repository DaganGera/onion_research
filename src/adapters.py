"""CLIP adaptation methods from the base paper's comparison table, re-implemented on our cached CLIP features.

  CLIPAdapter   Gao et al., IJCV 2024     small residual MLP on the image feature:
                                          f' = r * MLP(f) + (1 - r) * f,   logits = 100 f' . T
  TaskRes       Yu et al., CVPR 2023      a learned residual added to the class text vectors:
                                          t'_c = normalise(t_c + a * R_c),  R starts at 0
  CLAP          Silva-Rodriguez et al., CVPR 2024   linear probe that starts from the text vectors and is pulled back
                                          towards them: loss + sum_c lambda_c ||w_c - t_c||^2
  GraphAdapter  Li et al., NeurIPS 2023   class text vectors refined by a one-layer GCN over text and image prototypes
  CoOp          Zhou et al., IJCV 2022    learned prompt words ("[V]1 .. [V]M classname.") through the frozen text encoder,
                                          starting from "a photo of a"
Not included: Ta-Adapter (needs training inside the CLIP image encoder) and CAA (no code released).

These follow each paper's main equation, not the authors' code. All use the same cached views, the same class text
(our descriptions; CoOp learns its own prompt) and validation-chosen hyper-parameters, so the comparison is fair.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from common import CLASSES, CLIP_NAME, DEVICE, l2n
from fewshot import macro_f1


def _views(tr):
    X = tr.aug.to(DEVICE)                                   # [N, V, D]
    return X.flatten(0, 1), tr.y.to(DEVICE).repeat_interleave(X.shape[1])


def _train(params, loss_fn, X, Y, epochs, lr, bs=256, opt="adamw"):
    o = (torch.optim.AdamW(params, lr=lr, eps=1e-4) if opt == "adamw"
         else torch.optim.SGD(params, lr=lr, momentum=0.9))
    steps = epochs * max(1, -(-len(X) // bs))
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(o, steps)
    for _ in range(epochs):
        for idx in torch.randperm(len(X), device=X.device).split(bs):
            loss = loss_fn(X[idx], Y[idx])
            o.zero_grad()
            loss.backward()
            o.step()
            sch.step()


class _Selectable:
    """Fit once per candidate hyper-parameter, keep the best on validation (falls back to the first if no val)."""
    grid = [None]

    def fit(self, tr, va):
        best = None
        for hp in self.grid:
            torch.manual_seed(0)
            state = self._fit(tr, hp)
            s = macro_f1(self._logits(state, va.g.to(DEVICE)), va.y) if len(va) else 0.0
            if best is None or s > best[0]:
                best = (s, state)
        self.state = best[1]
        return self

    @torch.no_grad()
    def predict(self, te):
        return self._logits(self.state, te.g.to(DEVICE)).softmax(1).cpu()


class CLIPAdapter(_Selectable):
    grid = [0.2, 0.6]                                        # residual ratio r (0.2 = the paper's default)

    def __init__(self, T, epochs=50, lr=1e-3, **_):
        self.T, self.epochs, self.lr = T.to(DEVICE), epochs, lr

    def _fit(self, tr, r):
        mlp = nn.Sequential(nn.Linear(512, 128, bias=False), nn.ReLU(), nn.Linear(128, 512, bias=False),
                            nn.ReLU()).to(DEVICE)
        st = dict(mlp=mlp, r=r)
        X, Y = _views(tr)
        _train(mlp.parameters(), lambda x, y: F.cross_entropy(self._logits(st, x), y), X, Y, self.epochs, self.lr)
        return st

    def _logits(self, st, f):
        return 100 * l2n(st["r"] * st["mlp"](f) + (1 - st["r"]) * f) @ self.T.T

    def n_params(self):
        return sum(p.numel() for p in self.state["mlp"].parameters())


class TaskRes(_Selectable):
    grid = [0.5, 1.0]                                        # residual scale a (paper: 0.5 ImageNet, 1 elsewhere)

    def __init__(self, T, epochs=50, lr=1e-3, **_):
        self.T, self.epochs, self.lr = T.to(DEVICE), epochs, lr

    def _fit(self, tr, a):
        R = nn.Parameter(torch.zeros_like(self.T))
        st = dict(R=R, a=a)
        X, Y = _views(tr)
        _train([R], lambda x, y: F.cross_entropy(self._logits(st, x), y), X, Y, self.epochs, self.lr)
        return st

    def _logits(self, st, f):
        return 100 * f @ l2n(self.T + st["a"] * st["R"]).T

    def n_params(self):
        return self.T.numel()


class CLAP(_Selectable):
    grid = [0.1, 0.01]                                       # SGD learning rate (0.1 = the paper's), chosen on validation

    def __init__(self, T, epochs=300, **_):
        self.T, self.epochs = T.to(DEVICE), epochs

    def _fit(self, tr, lr):
        X, Y = _views(tr)
        with torch.no_grad():                                # class-adaptive penalty weights from zero-shot confidence
            p = (100 * X @ self.T.T).softmax(1)
            lam = torch.stack([p[Y == c, c].mean() if (Y == c).any() else p.new_tensor(0.) for c in range(len(self.T))])
        W = nn.Parameter(self.T.clone())
        st = dict(W=W)

        def loss(x, y):
            return F.cross_entropy(self._logits(st, x), y) + (lam[:, None] * (W - self.T) ** 2).sum(1).mean()

        _train([W], loss, X, Y, self.epochs, lr, opt="sgd")
        return st

    def _logits(self, st, f):
        return 100 * f @ l2n(st["W"]).T                     # cosine logits, as in CLAP

    def n_params(self):
        return self.T.numel()


class GraphAdapter(_Selectable):
    grid = [0.3, 0.6]                                        # mixing weight b of the graph-refined text feature

    def __init__(self, T, epochs=50, lr=1e-3, **_):
        self.T, self.epochs, self.lr = T.to(DEVICE), epochs, lr

    @staticmethod
    def _norm_adj(H):
        A = (H @ H.T).clamp_min(0)                           # cosine-similarity edges, self-loops included (cos = 1)
        d = A.sum(1).rsqrt()
        return d[:, None] * A * d[None]

    def _fit(self, tr, b):
        P = l2n(torch.stack([tr.aug[tr.y == c].mean((0, 1)) for c in range(len(self.T))]).to(DEVICE))
        H = torch.cat([self.T, P])                           # [2C, D] textual + visual knowledge nodes
        lin = nn.Linear(512, 512, bias=False).to(DEVICE)
        nn.init.eye_(lin.weight)
        st = dict(lin=lin, H=H, A=self._norm_adj(H), b=b)
        X, Y = _views(tr)
        _train(lin.parameters(), lambda x, y: F.cross_entropy(self._logits(st, x), y), X, Y, self.epochs, self.lr)
        return st

    def _logits(self, st, f):
        G = st["lin"](st["A"] @ st["H"])[: len(self.T)]
        return 100 * f @ l2n(st["b"] * l2n(G) + (1 - st["b"]) * self.T).T

    def n_params(self):
        return 512 * 512


class CoOp:
    """Context optimisation: M learnable context vectors replace the template words; the CLIP text encoder is frozen
    but back-propagated through. Epochs per K follow the CoOp paper (50 for 1-2 shots, 100 for 4-8, 200 for 16)."""

    EPOCHS = {"1": 50, "2": 50, "4": 100, "8": 100, "16": 200, "full": 20}

    def __init__(self, K="1", init="a photo of a", lr=2e-3, bs=32, **_):
        import open_clip
        self.model = open_clip.create_model_and_transforms(CLIP_NAME, pretrained="openai")[0].to(DEVICE).eval()
        for p in self.model.parameters():
            p.requires_grad_(False)
        tok = open_clip.get_tokenizer(CLIP_NAME)
        self.n_ctx = len(init.split())
        self.tokens = tok([f"{init} {c}." for c in CLASSES]).to(DEVICE)     # [C, 77]
        with torch.no_grad():
            emb = self.model.token_embedding(self.tokens)                  # [C, 77, w]
        self.prefix, self.suffix = emb[:, :1], emb[:, 1 + self.n_ctx:]
        self.ctx = nn.Parameter(emb[0, 1:1 + self.n_ctx].clone())          # unified context, init from the words
        self.epochs, self.lr, self.bs = self.EPOCHS[str(K)], lr, bs

    def text_features(self):
        m = self.model
        C = len(CLASSES)
        x = torch.cat([self.prefix, self.ctx.unsqueeze(0).expand(C, -1, -1), self.suffix], 1)
        x = x + m.positional_embedding
        x = m.transformer(x, attn_mask=m.attn_mask)
        x = m.ln_final(x)
        x = x[torch.arange(C), self.tokens.argmax(-1)] @ m.text_projection
        return l2n(x)

    def fit(self, tr, va):
        X, Y = _views(tr)
        _train([self.ctx], lambda x, y: F.cross_entropy(100 * x @ self.text_features().T, y), X, Y,
               self.epochs, self.lr, bs=self.bs, opt="sgd")
        with torch.no_grad():
            self.T = self.text_features()
        return self

    @torch.no_grad()
    def predict(self, te):
        return (100 * te.g.to(DEVICE) @ self.T.T).softmax(1).cpu()

    def n_params(self):
        return self.ctx.numel()
