"""Frozen feature extractors with one interface: .transform, .aug_transform, .encode_image, .encode_text."""
import sys

import torch
import torchvision.transforms as T

from common import DEVICE, ROOT, l2n

CLIP_MEAN, CLIP_STD = (0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)
IMNET_MEAN, IMNET_STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)


def _tf(mean, std, aug=False, normalize=True):
    ops = ([T.RandomResizedCrop(224, scale=(0.5, 1.0), interpolation=T.InterpolationMode.BICUBIC),
            T.RandomHorizontalFlip()] if aug else
           [T.Resize(224, interpolation=T.InterpolationMode.BICUBIC), T.CenterCrop(224)])
    ops += [T.ToTensor()] + ([T.Normalize(mean, std)] if normalize else [])
    return T.Compose(ops)


class Backbone:
    has_text = True

    def __init__(self, name):
        self.name = name
        if name in ("clip", "bioclip"):
            import open_clip
            if name == "clip":
                self.model, _, _ = open_clip.create_model_and_transforms("ViT-B-16-quickgelu", pretrained="openai")
                self.tok = open_clip.get_tokenizer("ViT-B-16-quickgelu")
            else:   # BioCLIP = a ViT-B/16 CLIP trained on the Tree of Life; weights fetched by third_party/fetch_bioclip.sh
                local = ROOT / "third_party/models/bioclip"
                if (local / ".complete").exists():
                    self.model, _, _ = open_clip.create_model_and_transforms(
                        "ViT-B-16", pretrained=str(local / "open_clip_pytorch_model.bin"))
                    self.tok = open_clip.get_tokenizer("ViT-B-16")
                else:
                    self.model, _, _ = open_clip.create_model_and_transforms("hf-hub:imageomics/bioclip")
                    self.tok = open_clip.get_tokenizer("hf-hub:imageomics/bioclip")
            self.mean, self.std, norm = CLIP_MEAN, CLIP_STD, True
        elif name == "dinov2":
            from transformers import AutoModel
            local = ROOT / "third_party/models/dinov2-small"
            self.model = AutoModel.from_pretrained(local if (local / ".complete").exists() else "facebook/dinov2-small")
            self.mean, self.std, norm = IMNET_MEAN, IMNET_STD, True
            self.has_text = False
        elif name == "scold":
            import timm
            sys.path.insert(0, str(ROOT / "third_party/scold"))
            import model as scold_model
            # SCOLD's code downloads ImageNet-22k Swin and RoBERTa weights that scold.pth immediately overwrites:
            # build both architectures from config only (no weight download), then load scold.pth.
            from transformers import RobertaConfig, RobertaModel, RobertaTokenizer
            rb = ROOT / "third_party/models/roberta-base"
            rb = str(rb) if (rb / ".complete").exists() else "roberta-base"
            scold_model.create_model = lambda *a, **k: timm.create_model(*a, **{**k, "pretrained": False})
            scold_model.RobertaModel = type("R", (), {"from_pretrained": staticmethod(
                lambda _p: RobertaModel(RobertaConfig.from_pretrained(rb)))})
            self.model = scold_model.LVL()
            sd = torch.load(ROOT / "third_party/scold/scold.pth", map_location="cpu")
            self.model.load_state_dict(sd)
            self.rtok = RobertaTokenizer.from_pretrained(rb)
            self.mean, self.std, norm = IMNET_MEAN, IMNET_STD, False   # SCOLD's inference.py: Resize + ToTensor only
        else:
            raise ValueError(name)
        self.model = self.model.to(DEVICE).eval().half()
        self.transform = _tf(self.mean, self.std, normalize=norm)
        self.aug_transform = _tf(self.mean, self.std, aug=True, normalize=norm)
        self.normalize = norm

    @torch.no_grad()
    def encode_image(self, x):
        x = x.to(DEVICE).half()
        if self.name in ("clip", "bioclip"):
            f = self.model.encode_image(x)
        elif self.name == "dinov2":
            f = self.model(pixel_values=x).pooler_output
        else:
            f = self.model.get_images_features(x)
        return l2n(f.float()).cpu()

    @torch.no_grad()
    def encode_text(self, texts):
        if self.name in ("clip", "bioclip"):
            f = self.model.encode_text(self.tok(texts).to(DEVICE))
        elif self.name == "scold":
            t = self.rtok(texts, return_tensors="pt", padding=True, truncation=True).to(DEVICE)
            f = self.model.get_texts_feature(t["input_ids"], t["attention_mask"])
        else:
            raise ValueError(f"{self.name} has no text encoder")
        return l2n(f.float()).cpu()
