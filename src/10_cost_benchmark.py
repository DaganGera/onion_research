"""Step 6 — inference cost per photo of every pipeline component (latency and parameter count).

Measures, on the available device (GPU if present, else CPU), the median time per photo of:
  CLIP ViT-B/16 on the whole photo, CLIP on the 5 region crops, OWLv2 detection, DINOv2-S, and the PRGA head,
plus a CPU run of the cheap components. Batch size 1 (a single photo arriving), fp16 on GPU, after warm-up.
Output: results/cost_benchmark.csv
"""
import time

import numpy as np
import open_clip
import pandas as pd
import torch
from PIL import Image
from transformers import AutoModel, Owlv2ForObjectDetection, Owlv2Processor

from common import CLIP_NAME, RAW, RESULTS, ROOT, SPLITS, load_meta
from fewshot import PRGANet
from harness import load_text

REPS = 30


def timeit(fn, reps=REPS, dev="cuda"):
    for _ in range(5):
        fn()
    ts = []
    for _ in range(reps):
        if dev == "cuda":
            torch.cuda.synchronize()
        t = time.perf_counter()
        fn()
        if dev == "cuda":
            torch.cuda.synchronize()
        ts.append(time.perf_counter() - t)
    return 1000 * float(np.median(ts))


def nparams(m):
    return sum(p.numel() for p in m.parameters())


@torch.no_grad()
def bench(dev):
    half = dev == "cuda"
    dt = torch.float16 if half else torch.float32
    rows = []
    clip = open_clip.create_model_and_transforms(CLIP_NAME, pretrained="openai")[0].to(dev, dt).eval()
    x1 = torch.randn(1, 3, 224, 224, device=dev, dtype=dt)
    x5 = torch.randn(5, 3, 224, 224, device=dev, dtype=dt)
    rows.append(("CLIP ViT-B/16, whole photo", timeit(lambda: clip.encode_image(x1), dev=dev),
                 nparams(clip.visual) / 1e6))
    rows.append(("CLIP ViT-B/16, 5 region crops", timeit(lambda: clip.encode_image(x5), dev=dev), 0))
    local = ROOT / "third_party/models/dinov2-small"
    dino = AutoModel.from_pretrained(local if (local / ".complete").exists() else "facebook/dinov2-small").to(dev, dt).eval()
    rows.append(("DINOv2-S, whole photo", timeit(lambda: dino(pixel_values=x1), dev=dev), nparams(dino) / 1e6))
    T = load_text("clip", "desc").to(dev)
    net = PRGANet(T, use_geom=False).to(dev).eval()
    g, n = torch.randn(1, 512, device=dev), torch.randn(1, 5, 512, device=dev)
    m, gm = torch.ones(1, 5, device=dev), torch.zeros(1, 5, 6, device=dev)
    rows.append(("PRGA head (graph + caches)", timeit(lambda: net(g, n, m, gm), dev=dev), nparams(net) / 1e6))
    if dev == "cuda":                                   # OWLv2 is only practical on a GPU
        ck = "google/owlv2-base-patch16-ensemble"
        proc = Owlv2Processor.from_pretrained(ck, backend="torchvision")
        owl = Owlv2ForObjectDetection.from_pretrained(ck, torch_dtype=dt).to(dev).eval()
        p = pd.read_csv(SPLITS / "test80.csv").path.iloc[0]
        im = Image.open(RAW / p).convert("RGB")
        prompts = ["an onion", "an onion bulb", "a pile of onions", "a mouldy onion", "a rotten onion", "black mould",
                   "a black spot", "a rotten spot", "a brown bruise", "a sprout", "peeling skin"]
        inp = proc(text=[prompts], images=[im], return_tensors="pt").to(dev)
        inp["pixel_values"] = inp["pixel_values"].to(dt)
        rows.append(("OWLv2-B/16 detection, 11 prompts", timeit(lambda: owl(**inp), reps=10, dev=dev), nparams(owl) / 1e6))
    return [dict(device=dev, component=c, ms_per_photo=round(t, 2), params_millions=round(pm, 2)) for c, t, pm in rows]


def main():
    load_meta()
    rows = []
    if torch.cuda.is_available():
        rows += bench("cuda")
        rows.append(dict(device="cuda", component="GPU name", ms_per_photo=None, params_millions=None,
                         note=torch.cuda.get_device_name(0)))
    torch.set_num_threads(8)
    rows += bench("cpu")
    df = pd.DataFrame(rows)
    df.to_csv(RESULTS / "cost_benchmark.csv", index=False)
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
