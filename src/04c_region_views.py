"""GPU job: augmented views of the detector regions for the training pool (controlled comparison, Stage 5b).

Graph node slots per training photo: [whole photo] + [object, region 1..4] (OWLv2 boxes, cropped with margin exactly as in
04_features.py). V = 5 views, like 04b_paper_patches.py:
  view 0     : the plain features already cached (clip_regions.pt for the crops, clip_global.pt for the whole photo)
  views 1..4 : every crop independently randomly resized-cropped + flipped and re-encoded; the whole-photo node takes
               views 0..3 of clip_aug.pt
(The baseline augments the full image once and tiles it; here each region crop is augmented on its own — the closest
cheap equivalent, since the boxes are found on the un-augmented photo.)
Output: features/clip_regionviews.pt   feats [Npool, 5, 5, 512]   gviews [Npool, 5, 512]   mask [Npool, 5]   paths
"""
import importlib.util
from pathlib import Path

import open_clip
import torch
import torchvision.transforms as T
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

import pandas as pd

from common import CLIP_NAME, DEVICE, NUM_WORKERS, RAW, REGIONS, SPLITS, l2n, load_feats, read_json, save_feats, seed_all

spec = importlib.util.spec_from_file_location("feat", Path(__file__).with_name("04_features.py"))
feat = importlib.util.module_from_spec(spec)
spec.loader.exec_module(feat)

V = 5
MEAN, STD = (0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)
AUG = T.Compose([T.RandomResizedCrop(224, scale=(0.5, 1.0), interpolation=T.InterpolationMode.BICUBIC),
                 T.RandomHorizontalFlip(), T.ToTensor(), T.Normalize(MEAN, STD)])


class RegionViews(Dataset):
    def __init__(self, paths, regions):
        self.paths, self.regions = paths, regions

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        p = self.paths[i]
        im = Image.open(RAW / p).convert("RGB")
        crops, _, _, _ = feat.region_nodes(im, self.regions[p])                    # [object, region 1..4] as PIL crops
        return torch.stack([torch.stack([AUG(c) for c in crops]) for _ in range(V - 1)])   # [4, 5, 3, 224, 224]


@torch.no_grad()
def main():
    seed_all(0)
    paths = pd.read_csv(SPLITS / "pool20.csv").path.tolist()
    ga, gg, rf = load_feats("clip_aug"), load_feats("clip_global"), load_feats("clip_regions")
    assert ga["paths"] == paths, "clip_aug rows must follow pool20.csv"
    grow, rrow = {p: i for i, p in enumerate(gg["paths"])}, {p: i for i, p in enumerate(rf["paths"])}
    plain = rf["feats"][[rrow[p] for p in paths]].float()                          # [N, 5, 512] view 0
    mask = rf["mask"][[rrow[p] for p in paths]].float()                            # [N, 5] (slot 0 = object, always 1)
    gviews = torch.cat([gg["feats"][[grow[p] for p in paths]].float()[:, None], ga["feats"][:, : V - 1].float()], 1)

    model = open_clip.create_model_and_transforms(CLIP_NAME, pretrained="openai")[0].to(DEVICE).eval().half()
    regions = read_json(REGIONS / "regions.json")
    out = []
    for x in tqdm(DataLoader(RegionViews(paths, regions), batch_size=4, num_workers=NUM_WORKERS), mininterval=30):
        B = x.shape[0]
        out.append(l2n(model.encode_image(x.to(DEVICE).flatten(0, 2).half()).float()).view(B, V - 1, 5, -1).cpu())
    aug = torch.cat(out)                                                           # [N, 4, 5, 512]
    feats = torch.cat([plain[:, None], aug], 1)                                    # [N, 5, 5, 512]
    save_feats("clip_regionviews", paths=paths, feats=feats.half(), gviews=gviews.half(), mask=mask)
    print("region views", tuple(feats.shape), "global views", tuple(gviews.shape), "mask", tuple(mask.shape))


if __name__ == "__main__":
    main()
