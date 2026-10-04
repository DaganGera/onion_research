"""GPU job: patch features exactly as in the base paper (arXiv 2512.12498, Sec. 4 'Implementation Details' + Fig. 2),
for the training pool.

Paper recipe: the training image is augmented (random resize/crop + horizontal flip) and resized to 336x336; it is
tiled by grids, ADJACENT TILES ARE MERGED into rectangular windows, every window is resized to 224x224 and encoded by
the frozen CLIP; each window feature is one graph node.

Here: a 4x4 grid (84-px tiles); windows of h x w adjacent tiles with h, w in {2, 3}  (9 + 6 + 6 + 4 = 25)
plus the whole image = 26 patches, the paper's best setting. The paper does not publish its exact window list; this is
our reading of Fig. 2 and of its ablation (all 3x3-grid windows = 9 patches, all 4x4-grid windows = 36 patches).

V views per photo: view 0 = plain resize, views 1..V-1 = random augmentation (the paper augments every training step).
Graph inputs are needed only for training photos (the graph is not used at test time), so only the pool is encoded.
Output: features/clip_paperpatches.pt  feats [Npool, V, 26, 512] fp16, L2-normalised; paths match clip_aug.pt
"""
import argparse

import open_clip
import pandas as pd
import torch
import torch.nn.functional as F
import torchvision.transforms as T
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from common import CLIP_NAME, DEVICE, NUM_WORKERS, RAW, SPLITS, l2n, save_feats, seed_all

V, SIZE, GRID = 5, 336, 4
TILE = SIZE // GRID
WINDOWS = [(r * TILE, c * TILE, (r + h) * TILE, (c + w) * TILE)               # (top, left, bottom, right) in pixels
           for h in (2, 3) for w in (2, 3)
           for r in range(GRID - h + 1) for c in range(GRID - w + 1)] + [(0, 0, SIZE, SIZE)]
assert len(WINDOWS) == 26
MEAN = torch.tensor([0.48145466, 0.4578275, 0.40821073], device=DEVICE).view(1, 3, 1, 1)
STD = torch.tensor([0.26862954, 0.26130258, 0.27577711], device=DEVICE).view(1, 3, 1, 1)
PLAIN = T.Compose([T.Resize((SIZE, SIZE), interpolation=T.InterpolationMode.BICUBIC), T.PILToTensor()])
AUG = T.Compose([T.RandomResizedCrop(SIZE, scale=(0.5, 1.0), interpolation=T.InterpolationMode.BICUBIC),
                 T.RandomHorizontalFlip(), T.PILToTensor()])


class Views(Dataset):
    def __init__(self, paths):
        self.paths = paths

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        im = Image.open(RAW / self.paths[i]).convert("RGB")
        return torch.stack([PLAIN(im)] + [AUG(im) for _ in range(V - 1)])          # [V, 3, 336, 336] uint8


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    seed_all(0)
    model = open_clip.create_model_and_transforms(CLIP_NAME, pretrained="openai")[0].to(DEVICE).eval().half()
    paths = pd.read_csv(SPLITS / "pool20.csv").path.tolist()
    if a.limit:
        paths = paths[: a.limit]
    out = []
    for x in tqdm(DataLoader(Views(paths), batch_size=2, num_workers=NUM_WORKERS), mininterval=30):
        B = x.shape[0]
        x = (x.to(DEVICE).flatten(0, 1).float() / 255 - MEAN) / STD                # [B*V, 3, 336, 336]
        crops = torch.stack([F.interpolate(x[:, :, t:b, l:r], size=224, mode="bicubic", align_corners=False,
                                           antialias=True) for t, l, b, r in WINDOWS], 1)   # [B*V, 26, 3, 224, 224]
        f = l2n(model.encode_image(crops.flatten(0, 1).half()).float())
        out.append(f.view(B, V, len(WINDOWS), -1).half().cpu())
    feats = torch.cat(out)
    if not a.limit:
        save_feats("clip_paperpatches", paths=paths, feats=feats, windows=WINDOWS)
    print("patch features", tuple(feats.shape))


if __name__ == "__main__":
    main()
