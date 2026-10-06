"""Prediction for the web app: load a checkpoint from 24_export_app_model.py and classify one photo.

The steps are the same as in training, for a single photo:
  OWLv2 finds the object and up to 4 regions (05_detect_regions.detect, with the domain's prompts)
  -> CLIP encodes the whole photo and the region crops; DINOv2 encodes the whole photo
  -> PRGA scores the photo against its refined cache and class prototypes, plus the DINOv2 cache
"""
import importlib

import torch

from backbones import Backbone
from common import DEVICE
from fewshot import PRGA, Batch, PRGANet, with_kind

owl = importlib.import_module("05_detect_regions")
feat = importlib.import_module("06_extract_features")
dino = importlib.import_module("13_prga_dinov2")

_shared = {}


def _owl():
    if "owl" not in _shared:
        _shared["owl"] = owl.load()
    return _shared["owl"]


def _backbone(name, pad):
    if (name, pad) not in _shared:
        _shared[name, pad] = Backbone(name, pad=pad)
    return _shared[name, pad]


class Predictor:
    def __init__(self, path):
        ck = torch.load(path, map_location="cpu", weights_only=False)
        self.ck, self.classes = ck, ck["classes"]
        T = ck["T"].float()
        m = PRGA.__new__(PRGA)                 # rebuild the trained model without re-training
        m.T, m.nodes, m.test_graph, m.M, m.keep_kinds, m.second, m.cfg = T, "regions", ck["test_graph"], None, None, False, ck["cfg"]
        m.net = PRGANet(T, D=T.shape[1], **ck["cfg"]).to(DEVICE)
        m.net.load_state_dict(ck["net"])
        m.net.eval()
        m.keys, m.L = ck["keys"].to(DEVICE), ck["L"].to(DEVICE)
        m.s = {k: torch.tensor(v, device=DEVICE) for k, v in ck["s"].items()}
        self.m = m
        self.clip = _backbone(ck["backbone"], ck["pad"])
        self.dino = _backbone("dinov2", ck["pad"])

    def regions(self, im):
        p = self.ck["owl"]                     # the domain's prompts (module globals read by detect())
        owl.OBJECT_PROMPTS, owl.INSTANCE_PROMPTS, owl.SPOT_PROMPTS = p["object"], p["instance"], p["spot"]
        owl.SYMPTOM_PROMPTS = p["instance"] + p["spot"]
        proc, model = _owl()
        return owl.detect(proc, model, [im])[0]

    @torch.no_grad()
    def __call__(self, im):
        im = im.convert("RGB")
        r = self.regions(im)
        crops, mask, geom, kind = feat.region_nodes(im, r)
        g = self.clip.encode_image(self.clip.transform(im)[None])
        reg = self.clip.encode_image(torch.stack([self.clip.transform(c) for c in crops]))
        g2 = self.dino.encode_image(self.dino.transform(im)[None])
        b = Batch(y=torch.zeros(1, dtype=torch.long), g=g, reg=reg[None], mask=mask[None],
                  geom=with_kind(geom[None], kind[None]), g2=g2)
        ck = self.ck
        logits = dino.fused(dino.prga_logits(self.m, b), g2, ck["k2"], ck["L"], ck["a2"], ck["b2"])
        return dict(zip(self.classes, logits.softmax(1)[0].tolist())), r
