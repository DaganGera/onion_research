"""Encode the class texts with the backbone's text encoder.

features/{bb}_text_template.pt : one vector per class, averaged over simple templates ("a photo of a {class}." ...)
features/{bb}_text_desc.pt     : one vector per class from the hand-written visual descriptions in
                                 prompts/descriptors.json: mean of the description vectors + 0.5 x template vector
"""
import argparse

import torch

from backbones import Backbone
from common import CLASSES, PROMPTS, l2n, read_json, save_feats

TEMPLATES = [
    "a photo of {}.",
    "a close-up photo of {}.",
    "a photo of {} after harvest.",
    "a photo of {} on a cloth.",
    "a cropped photo of {}.",
]
NAMES = {}  # classes already read naturally: "a photo of a healthy red onion." etc.
if (PROMPTS / "templates.json").exists():         # other domains bring their own templates
    TEMPLATES = read_json(PROMPTS / "templates.json")


def class_phrase(c):
    return NAMES.get(c, f"a {c}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backbone", default="clip", choices=["clip", "bioclip", "scold"])
    a = ap.parse_args()
    bb = Backbone(a.backbone)

    T = torch.stack([l2n(bb.encode_text([t.format(class_phrase(c)) for t in TEMPLATES]).mean(0)) for c in CLASSES])
    save_feats(f"{a.backbone}_text_template", T=T, texts=[class_phrase(c) for c in CLASSES])

    desc = read_json(PROMPTS / "descriptors.json")
    dmax = max(len(v) for v in desc.values())
    D = torch.zeros(len(CLASSES), dmax, T.shape[1])
    Dm = torch.zeros(len(CLASSES), dmax)
    for i, c in enumerate(CLASSES):
        e = bb.encode_text(desc[c])
        D[i, : len(e)], Dm[i, : len(e)] = e, 1
    Td = l2n(torch.stack([l2n(D[i, Dm[i] > 0].mean(0)) for i in range(len(CLASSES))]) + 0.5 * T)
    save_feats(f"{a.backbone}_text_desc", T=Td, desc=D, desc_mask=Dm, desc_texts=[desc[c] for c in CLASSES])
    print("saved", T.shape, D.shape)


if __name__ == "__main__":
    main()
