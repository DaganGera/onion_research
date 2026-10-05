"""A normal CNN for comparison: EfficientNet-B0 (ImageNet weights) fine-tuned on the support photos themselves.

Shows what plain fine-tuning does with 1-16 photos per class. Same splits and test set as everything else.
The test photos are decoded once into memory (uint8, 224x224) so the runs do not re-read thousands of JPEGs.
"""
import numpy as np
import pandas as pd
import timm
import torch
import torch.nn.functional as F
import torchvision.transforms as T
from PIL import Image

from common import C, DEVICE, PAD_SQUARE, RAW, RESULTS, SEEDS, SPLITS, metrics, pad_square, seed_all

MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
TRAIN_TF = T.Compose([T.RandomResizedCrop(224, scale=(0.5, 1.0)), T.RandomHorizontalFlip(), T.ColorJitter(0.2, 0.2, 0.2),
                      T.PILToTensor()])
TEST_TF = T.Compose([T.Resize(224), T.CenterCrop(224), T.PILToTensor()])


def open_rgb(p):
    im = Image.open(RAW / p).convert("RGB")
    return pad_square(im) if PAD_SQUARE else im


def load_u8(paths, tf):
    return torch.stack([tf(open_rgb(p)) for p in paths])


def norm(x):
    return (x.float() / 255 - MEAN.to(x.device)) / STD.to(x.device)


@torch.no_grad()
def predict(model, X, bs=256):
    model.eval()
    out = []
    for i in range(0, len(X), bs):
        with torch.autocast("cuda", dtype=torch.float16):
            out.append(model(norm(X[i:i + bs].to(DEVICE))).float().softmax(1).cpu())
    return torch.cat(out)


def train_one(sup, epochs):
    model = timm.create_model("efficientnet_b0", pretrained=True, num_classes=C).to(DEVICE)
    imgs = [open_rgb(p) for p in sup.path]
    y = torch.tensor(sup.label.values)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-2)
    steps_per_epoch = max(1, int(np.ceil(len(imgs) / 32)))
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, 3e-4, total_steps=epochs * steps_per_epoch)
    scaler = torch.amp.GradScaler()
    # class-balanced sampling so rare classes are seen as often as "healthy"
    w = (1.0 / np.bincount(y.numpy(), minlength=C))[y.numpy()]
    for _ in range(epochs):
        model.train()
        idx = torch.multinomial(torch.tensor(w), len(imgs), replacement=True)
        for i in range(0, len(idx), 32):
            b = idx[i:i + 32]
            x = norm(torch.stack([TRAIN_TF(imgs[j]) for j in b]).to(DEVICE))
            with torch.autocast("cuda", dtype=torch.float16):
                loss = F.cross_entropy(model(x), y[b].to(DEVICE), label_smoothing=0.1)
            opt.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            sched.step()
    return model


def main():
    import json
    test = pd.read_csv(SPLITS / "test80.csv")
    Xte = load_u8(test.path, TEST_TF)
    log = RESULTS / "runs" / "EfficientNetB0-finetune.jsonl"            # resumable: finished runs are skipped
    done = {(str(r["K"]), int(r["seed"])): r for r in
            (json.loads(l) for l in (log.read_text().splitlines() if log.exists() else []) if l.strip())}
    rows = []
    for K in ["1", "2", "4", "8", "16", "full"]:
        for s in SEEDS:
            if (K, s) in done:
                rows.append(done[(K, s)])
                continue
            seed_all(s)
            sup = pd.read_csv(SPLITS / f"support_K{K}_s{s}.csv")
            epochs = 15 if K == "full" else 40
            model = train_one(sup, epochs)
            probs = predict(model, Xte)
            r = dict(method="EfficientNetB0-finetune", K=K, seed=s, n_support=len(sup),
                     **metrics(test.label.values, probs.argmax(1).numpy()))
            rows.append(r)
            print(r, flush=True)
            np.save(RESULTS / "preds" / f"EfficientNetB0-finetune_K{K}_s{s}.npy", probs.numpy().astype(np.float16))
            with open(log, "a") as f:
                f.write(json.dumps(r) + "\n")
            del model
            torch.cuda.empty_cache()
    pd.DataFrame(rows).to_csv(RESULTS / "runs" / "EfficientNetB0-finetune.csv", index=False)


if __name__ == "__main__":
    main()
