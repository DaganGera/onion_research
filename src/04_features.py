"""GPU jobs G1/G2a/G3/G4: encode images with a frozen backbone and cache the features.

  --mode global   every image, centre crop          -> features/{bb}_global.pt   feats [N, D]
  --mode aug      pool images x N_AUG random views   -> features/{bb}_aug.pt      feats [Np, N_AUG, D]
  --mode grid     every image, 3x3 grid crops        -> features/{bb}_grid.pt     feats [N, 9, D]
  --mode regions  object + M OWLv2 region crops      -> features/{bb}_regions.pt  feats [N, 1+M, D], mask, geom, kind
Rows are aligned with data/meta.csv (the file stores the paths so this can always be checked).
"""
import argparse

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from backbones import Backbone
from common import NUM_WORKERS, RAW, REGIONS, SPLITS, load_meta, read_json, save_feats, seed_all

N_AUG = 10
M = 4            # max lesion regions per image
MARGIN = 0.15    # context added around each box


def open_rgb(p):
    return Image.open(RAW / p).convert("RGB")


def grid_crops(im):
    w, h = im.size
    return [im.crop((int(i * w / 3), int(j * h / 3), int((i + 1) * w / 3), int((j + 1) * h / 3)))
            for j in range(3) for i in range(3)]


def crop_box(im, box, margin=MARGIN):
    """box = (x0, y0, x1, y1) in pixels of the original image."""
    w, h = im.size
    x0, y0, x1, y1 = box
    mx, my = (x1 - x0) * margin, (y1 - y0) * margin
    x0, y0 = max(0, x0 - mx), max(0, y0 - my)
    x1, y1 = min(w, x1 + mx), min(h, y1 + my)
    if x1 - x0 < 8 or y1 - y0 < 8:
        return im
    return im.crop((int(x0), int(y0), int(x1), int(y1)))


def region_nodes(im, r):
    """Returns crops [object, region_1..region_M], mask [1+M], geom [1+M, 5] = (cx, cy, w, h, score),
    kind [1+M] = node type (1 object, 2 onion instance, 3 lesion spot, 0 padding)."""
    W, H = im.size
    crops, mask, geom, kind = [], [], [], [1]
    obj = r.get("object")
    if obj:
        crops.append(crop_box(im, obj["box"], margin=0.05))
        b = obj["box"]
        geom.append([(b[0] + b[2]) / 2 / W, (b[1] + b[3]) / 2 / H, (b[2] - b[0]) / W, (b[3] - b[1]) / H, obj["score"]])
    else:
        crops.append(im)
        geom.append([0.5, 0.5, 1.0, 1.0, 0.0])
    mask.append(1.0)
    for reg in r.get("regions", [])[:M]:
        b = reg["box"]
        crops.append(crop_box(im, b))
        geom.append([(b[0] + b[2]) / 2 / W, (b[1] + b[3]) / 2 / H, (b[2] - b[0]) / W, (b[3] - b[1]) / H, reg["score"]])
        mask.append(1.0)
        kind.append(3 if reg.get("kind") == "spot" else 2)
    while len(crops) < 1 + M:
        crops.append(im)
        geom.append([0.0] * 5)
        mask.append(0.0)
        kind.append(0)
    return crops, torch.tensor(mask), torch.tensor(geom, dtype=torch.float32), torch.tensor(kind)


class DS(Dataset):
    def __init__(self, paths, tf, mode, regions=None):
        self.paths, self.tf, self.mode, self.regions = paths, tf, mode, regions

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        im = open_rgb(self.paths[i])
        if self.mode == "global":
            return self.tf(im)
        if self.mode == "aug":
            return torch.stack([self.tf(im) for _ in range(N_AUG)])
        if self.mode == "grid":
            return torch.stack([self.tf(c) for c in grid_crops(im)])
        crops, mask, geom, kind = region_nodes(im, self.regions[self.paths[i]])
        return torch.stack([self.tf(c) for c in crops]), mask, geom, kind


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backbone", default="clip", choices=["clip", "bioclip", "dinov2", "scold"])
    ap.add_argument("--mode", default="global", choices=["global", "aug", "grid", "regions"])
    ap.add_argument("--bs", type=int, default=64)
    a = ap.parse_args()
    seed_all(0)
    bb = Backbone(a.backbone)

    meta = load_meta()
    paths = (pd.read_csv(SPLITS / "pool20.csv") if a.mode == "aug" else meta).path.tolist()
    regions = read_json(REGIONS / "regions.json") if a.mode == "regions" else None
    tf = bb.aug_transform if a.mode == "aug" else bb.transform
    bs = a.bs if a.mode == "global" else max(4, a.bs // 8)
    dl = DataLoader(DS(paths, tf, a.mode, regions), batch_size=bs, num_workers=NUM_WORKERS)

    feats, masks, geoms, kinds = [], [], [], []
    for batch in tqdm(dl, desc=f"{a.backbone}/{a.mode}"):
        if a.mode == "regions":
            batch, mk, gm, kd = batch
            masks.append(mk)
            geoms.append(gm)
            kinds.append(kd)
        if batch.dim() == 5:                                   # [B, V, 3, 224, 224]
            B, V = batch.shape[:2]
            feats.append(bb.encode_image(batch.flatten(0, 1)).view(B, V, -1))
        else:
            feats.append(bb.encode_image(batch))
    out = dict(paths=paths, feats=torch.cat(feats).half())
    if a.mode == "regions":
        out.update(mask=torch.cat(masks), geom=torch.cat(geoms), kind=torch.cat(kinds))
    save_feats(f"{a.backbone}_{a.mode}", **out)
    print("saved", f"{a.backbone}_{a.mode}", tuple(out["feats"].shape))


if __name__ == "__main__":
    main()
