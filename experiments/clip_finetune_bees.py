"""Side experiment, NOT part of the proposed method: what if CLIP itself is fine-tuned on bee images?

The project keeps every backbone frozen. This script only answers "how much would partial fine-tuning help?" so the
choice can be justified. CLIP ViT-L/14 image encoder: the last N_BLOCKS transformer blocks, the final norm and
projection are trained; everything else stays frozen. The classifier starts as zero-shot CLIP (the class-description
vectors) and is trained with it (FLYP / LP-FT style: logits = 100 * cos(image, class vector)).

Same splits as the main bee runs (DATASET=bees), seed 1, K = 16 and all pool photos. The best epoch is chosen on
validation; the test set is used once. Run on Kaggle: experiments/kaggle_clip_finetune/
Output: results/clip_finetune.csv
"""
import json
import sys
import time
from pathlib import Path

import open_clip
import pandas as pd
import torch
import torch.nn.functional as F
import torchvision.transforms as T
from PIL import Image
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from common import CLASSES, PROMPTS, RAW, RESULTS, SPLITS, metrics, pad_square, seed_all  # noqa: E402

N_BLOCKS, SEED, DEV = 2, 1, "cuda"
EPOCHS = {"16": 30, "full": 6}
MEAN, STD = (0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)
TRAIN_TF = T.Compose([T.Lambda(pad_square), T.RandomResizedCrop(224, scale=(0.5, 1.0)), T.RandomHorizontalFlip(),
                      T.ToTensor(), T.Normalize(MEAN, STD)])
TEST_TF = T.Compose([T.Lambda(pad_square), T.Resize((224, 224)), T.ToTensor(), T.Normalize(MEAN, STD)])


class DS(Dataset):
    def __init__(self, df, tf):
        self.p, self.y, self.tf = df.path.tolist(), df.label.tolist(), tf

    def __len__(self):
        return len(self.p)

    def __getitem__(self, i):
        return self.tf(Image.open(RAW / self.p[i]).convert("RGB")), self.y[i]


@torch.no_grad()
def class_vectors(model, tok):
    desc = json.loads((PROMPTS / "descriptors.json").read_text())
    temps = json.loads((PROMPTS / "templates.json").read_text())
    enc = lambda t: F.normalize(model.encode_text(tok(t).to(DEV)).float(), dim=-1)  # noqa: E731
    tm = torch.stack([F.normalize(enc([t.format("a " + c) for t in temps]).mean(0), dim=-1) for c in CLASSES])
    return F.normalize(torch.stack([F.normalize(enc(desc[c]).mean(0), dim=-1) for c in CLASSES]) + 0.5 * tm, dim=-1)


@torch.no_grad()
def predict(vis, W, df):
    vis.eval()
    out = []
    for x, _ in DataLoader(DS(df, TEST_TF), batch_size=128, num_workers=4):
        with torch.autocast("cuda", dtype=torch.float16):
            f = vis(x.to(DEV))
        out.append((100 * F.normalize(f.float(), dim=-1) @ F.normalize(W, dim=-1).T).softmax(1).cpu())
    return torch.cat(out)


def run(K):
    seed_all(SEED)
    model, _, _ = open_clip.create_model_and_transforms("ViT-L-14-quickgelu", pretrained="openai")
    model = model.to(DEV)
    tok = open_clip.get_tokenizer("ViT-L-14-quickgelu")
    W0 = class_vectors(model, tok)
    vis = model.visual
    for p in vis.parameters():
        p.requires_grad = False
    train_parts = list(vis.transformer.resblocks[-N_BLOCKS:]) + [vis.ln_post]
    for m in train_parts:
        for p in m.parameters():
            p.requires_grad = True
    vis.proj.requires_grad = True
    W = torch.nn.Parameter(W0.clone())
    sup = pd.read_csv(SPLITS / f"support_K{K}_s{SEED}.csv")
    val = pd.read_csv(SPLITS / f"val_K{K}_s{SEED}.csv").sample(frac=1, random_state=0).head(800)
    test = pd.read_csv(SPLITS / "test80.csv").drop_duplicates("path")
    params = [p for p in vis.parameters() if p.requires_grad]
    opt = torch.optim.AdamW([{"params": params, "lr": 1e-5}, {"params": [W], "lr": 1e-3}], weight_decay=0.05)
    epochs = EPOCHS[K]
    dl = DataLoader(DS(sup, TRAIN_TF), batch_size=32, shuffle=True, num_workers=4, drop_last=False)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs * len(dl))
    scaler = torch.amp.GradScaler()
    # class weights: the pool is 80 % healthy
    cw = torch.tensor([1 / max(1, int((sup.label == c).sum())) for c in range(len(CLASSES))], dtype=torch.float32, device=DEV)
    cw = cw / cw.sum() * len(CLASSES)
    val0 = metrics(val.label.values, predict(vis, W.detach(), val).argmax(1).numpy())["macro_f1"]
    best, best_state, best_ep = val0, None, 0
    for ep in range(1, epochs + 1):
        vis.train()
        for x, y in dl:
            with torch.autocast("cuda", dtype=torch.float16):
                f = vis(x.to(DEV))
            logits = 100 * F.normalize(f.float(), dim=-1) @ F.normalize(W, dim=-1).T
            loss = F.cross_entropy(logits, y.to(DEV), weight=cw, label_smoothing=0.1)
            opt.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            sched.step()
        v = metrics(val.label.values, predict(vis, W.detach(), val).argmax(1).numpy())["macro_f1"]
        print(f"K={K} epoch {ep}: val macro-F1 {v:.4f}", flush=True)
        if v > best:
            best, best_ep = v, ep
            best_state = ({k: t.detach().clone() for k, t in vis.state_dict().items()}, W.detach().clone())
    if best_state is not None:
        vis.load_state_dict(best_state[0])
        W = best_state[1]
    p = predict(vis, W.detach(), test)
    m = metrics(test.label.values, p.argmax(1).numpy())
    return dict(K=K, seed=SEED, trained_blocks=N_BLOCKS, epochs=epochs, best_epoch=best_ep, val_macro_f1_zero_shot=round(val0, 4),
                val_macro_f1_best=round(best, 4), test_macro_f1=round(m["macro_f1"], 4), test_acc=round(m["acc"], 4),
                says_varroa=round(float(p.argmax(1).float().mean()), 4), n_test=len(test))


def main():
    rows = []
    for K in ("16", "full"):
        t0 = time.time()
        r = run(K)
        r["minutes"] = round((time.time() - t0) / 60, 1)
        rows.append(r)
        print(r, flush=True)
        pd.DataFrame(rows).to_csv(RESULTS / "clip_finetune.csv", index=False)


if __name__ == "__main__":
    main()
