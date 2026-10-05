"""Find the onion and small suspicious regions in every photo with OWLv2 (open-vocabulary detector, text prompts).

For every photo: the best object box, plus up to M region boxes of two kinds:
  - "instance": single onions (in pile photos, the separate bulbs), up to N_INST
  - "spot":     small lesion-like areas (mould, rotten spot, bruise, sprout, peeling skin), up to N_SPOT
If one kind finds fewer boxes, the other kind fills the free slots. Already processed photos are skipped on a rerun.

Observation: OWLv2 does not separate rotten from healthy onions ("a rotten onion" also fires on healthy ones), so the
instance prompts work as an onion detector. Classification is left to CLIP and the graph.

OWLv2 pads each photo to a square (bottom/right) before detection, so its normalised boxes are relative to a square
of side max(w, h); boxes are converted back with that side length.
Output: regions/regions.json
"""
import argparse

import torch
from PIL import Image
from torchvision.ops import nms
from tqdm import tqdm
from transformers import Owlv2ForObjectDetection, Owlv2Processor

from common import DEVICE, PROMPTS, RAW, REGIONS, load_meta, read_json, write_json

OBJECT_PROMPTS = ["an onion", "an onion bulb", "a pile of onions"]
INSTANCE_PROMPTS = ["a mouldy onion", "a rotten onion", "an onion"]
SPOT_PROMPTS = ["black mould", "a black spot", "a rotten spot", "a brown bruise", "a sprout", "peeling skin"]
if (PROMPTS / "owl_prompts.json").exists():      # other domains bring their own prompts
    _p = read_json(PROMPTS / "owl_prompts.json")
    OBJECT_PROMPTS, INSTANCE_PROMPTS, SPOT_PROMPTS = _p["object"], _p["instance"], _p["spot"]
SYMPTOM_PROMPTS = INSTANCE_PROMPTS + SPOT_PROMPTS
M, N_INST, N_SPOT = 4, 2, 2
INST_THR, SPOT_THR, OBJ_THR, MAX_AREA, SPOT_MAX_AREA = 0.2, 0.06, 0.05, 0.85, 0.12
CKPT = "google/owlv2-base-patch16-ensemble"


def load():
    proc = Owlv2Processor.from_pretrained(CKPT, backend="torchvision")   # ~9x faster than the default PIL resize
    model = Owlv2ForObjectDetection.from_pretrained(CKPT, torch_dtype=torch.float16).to(DEVICE).eval()
    return proc, model


@torch.no_grad()
def detect(proc, model, images):
    prompts = OBJECT_PROMPTS + SYMPTOM_PROMPTS
    inp = proc(text=[prompts] * len(images), images=images, return_tensors="pt").to(DEVICE)
    inp["pixel_values"] = inp["pixel_values"].half()
    out = model(**inp)
    probs = out.logits.float().sigmoid()                 # [B, P, Q]
    boxes = out.pred_boxes.float()                       # [B, P, 4] cx, cy, w, h in padded-square units
    n_obj = len(OBJECT_PROMPTS)
    res = []
    for b, im in enumerate(images):
        W, H = im.size
        S = max(W, H)
        cx, cy, w, h = boxes[b].unbind(-1)
        xyxy = torch.stack([(cx - w / 2) * S, (cy - h / 2) * S, (cx + w / 2) * S, (cy + h / 2) * S], -1)
        xyxy[:, 0::2] = xyxy[:, 0::2].clamp(0, W)
        xyxy[:, 1::2] = xyxy[:, 1::2].clamp(0, H)
        area = ((xyxy[:, 2] - xyxy[:, 0]) * (xyxy[:, 3] - xyxy[:, 1])) / (W * H)

        os_, oq = probs[b, :, :n_obj].max(-1)
        i = int(os_.argmax())
        obj = ({"box": xyxy[i].tolist(), "score": float(os_[i]), "prompt": OBJECT_PROMPTS[int(oq[i])]}
               if os_[i] >= OBJ_THR else None)

        n_inst = len(INSTANCE_PROMPTS)
        picked = {}
        for kind, lo, hi, thr, amax in (("instance", n_obj, n_obj + n_inst, INST_THR, MAX_AREA),
                                        ("spot", n_obj + n_inst, n_obj + len(SYMPTOM_PROMPTS), SPOT_THR, SPOT_MAX_AREA)):
            ss, sq = probs[b, :, lo:hi].max(-1)
            idx = ((ss >= thr) & (area < amax) & (area > 0.002)).nonzero().squeeze(1)
            cand = []
            if len(idx):
                for j in idx[nms(xyxy[idx], ss[idx], 0.3)].tolist():
                    cand.append({"box": xyxy[j].tolist(), "score": float(ss[j]), "kind": kind,
                                 "prompt": SYMPTOM_PROMPTS[lo - n_obj + int(sq[j])]})
            picked[kind] = cand
        inst, spot = picked["instance"], picked["spot"]
        take_i = min(len(inst), max(N_INST, M - len(spot)))
        take_s = min(len(spot), M - take_i)
        regions = inst[:take_i] + spot[:take_s]
        res.append({"object": obj, "regions": regions})
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bs", type=int, default=8)
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    out_p = REGIONS / "regions.json"
    done = read_json(out_p) if out_p.exists() else {}
    paths = [p for p in load_meta().path if p not in done]
    if a.limit:
        paths = paths[: a.limit]
    proc, model = load()
    for i in tqdm(range(0, len(paths), a.bs), mininterval=30):
        chunk = paths[i: i + a.bs]
        ims = [Image.open(RAW / p).convert("RGB") for p in chunk]
        for p, r in zip(chunk, detect(proc, model, ims)):
            done[p] = r
        if (i // a.bs) % 25 == 0:          # checkpoint every 200 images (resumable after an interruption)
            write_json(done, out_p)
    write_json(done, out_p)
    n_fb = sum(1 for r in done.values() if not r["regions"])
    print(f"{len(done)} images, {n_fb} with no symptom region ({n_fb / len(done):.1%})")


if __name__ == "__main__":
    main()


@torch.no_grad()
def all_instances(proc, model, im, thr=INST_THR, max_n=40):
    """Every individual onion OWLv2 finds (for per-bulb grading of pile photos): list of (box, score)."""
    inp = proc(text=[INSTANCE_PROMPTS], images=[im], return_tensors="pt").to(DEVICE)
    inp["pixel_values"] = inp["pixel_values"].half()
    out = model(**inp)
    W, H = im.size
    S = max(W, H)
    s = out.logits.float().sigmoid()[0].max(-1).values
    cx, cy, w, h = out.pred_boxes.float()[0].unbind(-1)
    xyxy = torch.stack([(cx - w / 2) * S, (cy - h / 2) * S, (cx + w / 2) * S, (cy + h / 2) * S], -1)
    xyxy[:, 0::2] = xyxy[:, 0::2].clamp(0, W)
    xyxy[:, 1::2] = xyxy[:, 1::2].clamp(0, H)
    area = ((xyxy[:, 2] - xyxy[:, 0]) * (xyxy[:, 3] - xyxy[:, 1])) / (W * H)
    idx = ((s >= thr) & (area > 0.003) & (area < MAX_AREA)).nonzero().squeeze(1)
    if not len(idx):
        return []
    keep = idx[nms(xyxy[idx], s[idx], 0.3)][:max_n]
    return [(xyxy[j].tolist(), float(s[j])) for j in keep.tolist()]
