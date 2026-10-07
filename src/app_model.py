"""Prediction and explanations for the web app: load a checkpoint from 24_export_app_model.py and classify one photo.

The steps are the same as in training, for a single photo:
  OWLv2 finds the object and up to 4 regions (05_detect_regions.detect, with the domain's prompts)
  -> CLIP encodes the whole photo and the region crops; DINOv2 encodes the whole photo
  -> PRGA scores the photo against its refined cache and class prototypes, plus the DINOv2 cache
  -> the score is calibrated on validation data (calibration.py): p = softmax((z + b) / tau)

Explanations (all computed from the model itself, not approximated):
  breakdown   the class score z is a sum of four parts, so each part's contribution is shown exactly:
              z = 100 g.T  +  alpha exp(-beta (1 - g.K)) L~  +  gamma 100 g.T^(g)  +  a2 exp(-b2 (1 - d.K2)) L~
              (text match, CLIP cache, photo-specific class prototypes, DINOv2 cache)
  regions     each region node is removed in turn and the photo is re-scored; the drop in the predicted class's
              margin (its calibrated score minus the runner-up's) is that region's importance
  unsure      max probability below the validation threshold -> "not sure, check by hand"
  familiar    how close the photo is to the training photos (DINOv2 cosine to the nearest support photo); below the
              5th percentile of the validation photos -> "unlike the training photos, result unreliable"
  each object for photos with many bees / bulbs: OWLv2 finds every one, each crop is classified on its own
  Grad-CAM    gradient of the calibrated decision (predicted class minus the runner-up) back through PRGA's graph
              and caches into CLIP's image encoder; heat = ReLU(sum over channels of activation x gradient) on the
              patch tokens entering CLIP's last block. Region crops and the DINOv2 cache are held fixed.
  search      OWLv2 looks for any text you type; CLIP scores each found box against the same text
"""
import copy
import importlib

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

import calibration
from backbones import Backbone
from common import DEVICE, l2n
from fewshot import PRGA, Batch, PRGANet, balance, with_kind

owl = importlib.import_module("05_detect_regions")
feat = importlib.import_module("06_extract_features")
dino = importlib.import_module("13_prga_dinov2")
PARTS = ["text match (zero-shot CLIP)", "CLIP cache (similar training photos)",
         "photo-specific class prototypes (graph)", "DINOv2 cache (similar training photos)"]
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
    def __init__(self, path, calibration_mode="bias"):
        """calibration_mode: "bias" (temperature + class bias) or "temperature" (temperature only, when stored)."""
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
        cal = ck.get("calib") or {}
        self.tau, self.bias = cal.get("tau", 1.0), torch.tensor(cal.get("bias", [0.0] * len(self.classes)))
        self.unsure_below = cal.get("unsure_below")
        self.calibration_mode = "bias"
        if calibration_mode == "temperature" and "tau_only" in cal:
            self.tau, self.bias = cal["tau_only"]["tau"], torch.zeros(len(self.classes))
            self.unsure_below = cal["tau_only"]["unsure_below"]
            self.calibration_mode = "temperature"
        self.familiar_below = cal.get("familiar_below")
        self.clip = _backbone(ck["backbone"], ck["pad"])
        self.dino = _backbone("dinov2", ck["pad"])

    def regions(self, im):
        p = self.ck["owl"]                     # the domain's prompts (module globals read by detect())
        owl.OBJECT_PROMPTS, owl.INSTANCE_PROMPTS, owl.SPOT_PROMPTS = p["object"], p["instance"], p["spot"]
        owl.SYMPTOM_PROMPTS = p["instance"] + p["spot"]
        proc, model = _owl()
        return owl.detect(proc, model, [im])[0]

    def _parts(self, b):
        """the four score parts [4, C] for one photo (they sum to the model's logits)."""
        m, ck = self.m, self.ck
        n, mk, gm = [x.to(DEVICE) for x in m._nodes(b)]
        g = b.g.to(DEVICE)
        _, Tq = m._embed(g, n, mk, gm)
        L = balance(m.L)
        zs = 100 * g @ m.net.T.T
        cache = m.s["alpha"] * torch.exp(-m.s["beta"] * (1 - g @ m.keys.T)) @ L
        proto = m.s["gamma"] * 100 * torch.einsum("bd,bcd->bc", g, Tq)
        d2 = ck["a2"] * torch.exp(-ck["b2"] * (1 - b.g2 @ ck["k2"].T)) @ balance(ck["L"])
        return torch.cat([zs.cpu(), cache.cpu(), proto.cpu(), d2]).float()

    def _vis(self):
        """a float32 copy of CLIP's image encoder, for gradients (the main copy runs in float16)."""
        key = ("vis", self.ck["backbone"])
        if key not in _shared:
            v = copy.deepcopy(self.clip.model.visual).float().eval()
            for q in v.parameters():
                q.requires_grad_(False)
            _shared[key] = v
        return _shared[key]

    def _to_photo(self, cam, im):
        """map a patch-grid heat map back onto the photo (undoing the pad-to-square or the centre crop)."""
        W, H = im.size
        cam = cam / (cam.max() + 1e-8)
        if self.ck["pad"]:
            S = max(W, H)
            big = np.array(Image.fromarray(cam.astype(np.float32)).resize((S, S), Image.BILINEAR))
            x0, y0 = (S - W) // 2, (S - H) // 2
            return big[y0:y0 + H, x0:x0 + W]
        s = min(W, H)
        heat = np.zeros((H, W), np.float32)
        x0, y0 = (W - s) // 2, (H - s) // 2
        heat[y0:y0 + s, x0:x0 + s] = np.array(Image.fromarray(cam.astype(np.float32)).resize((s, s), Image.BILINEAR))
        return heat

    def gradcam(self, im, b, c):
        """Grad-CAM of this model's decision for class c (see the module docstring); returns an H x W map in [0, 1]."""
        vis, acts = self._vis(), {}
        hook = vis.transformer.resblocks[-2].register_forward_hook(lambda m, i, o: acts.__setitem__("a", o))
        try:
            with torch.enable_grad():
                x = self.clip.transform(im)[None].to(DEVICE).float().requires_grad_(True)
                g = l2n(vis(x))
                acts["a"].retain_grad()
                bg = Batch(y=b.y, g=g, reg=b.reg, mask=b.mask, geom=b.geom, g2=b.g2)
                z = (self._parts(bg).sum(0) + self.bias) / self.tau
                (z[c] - torch.cat([z[:c], z[c + 1:]]).max()).backward()
        finally:
            hook.remove()
        A, G = acts["a"], acts["a"].grad
        if A.shape[0] != 1:                          # (tokens, batch, channels) layout
            A, G = A.transpose(0, 1), G.transpose(0, 1)
        cam = F.relu((A[0, 1:] * G[0, 1:]).sum(-1))
        n = int(cam.numel() ** 0.5)
        return self._to_photo(cam[: n * n].view(n, n).detach().cpu().numpy(), im)

    def _probs(self, parts):
        return calibration.apply(parts.sum(0, keepdim=True), self.tau, self.bias)[0]

    @torch.no_grad()
    def __call__(self, im, explain=True, cam=True):
        im = im.convert("RGB")
        r = self.regions(im)
        crops, mask, geom, kind = feat.region_nodes(im, r)
        g = self.clip.encode_image(self.clip.transform(im)[None])
        reg = self.clip.encode_image(torch.stack([self.clip.transform(c) for c in crops]))
        g2 = self.dino.encode_image(self.dino.transform(im)[None])
        b = Batch(y=torch.zeros(1, dtype=torch.long), g=g, reg=reg[None], mask=mask[None],
                  geom=with_kind(geom[None], kind[None]), g2=g2)
        parts = self._parts(b)
        probs = self._probs(parts)
        c = int(probs.argmax())
        familiar = float((l2n(g2) @ l2n(self.ck["k2"].float()).T).max())
        def margin(pt):                            # calibrated score of the predicted class minus the runner-up
            z = (pt.sum(0) + self.bias) / self.tau
            return float(z[c] - torch.cat([z[:c], z[c + 1:]]).max())

        base, importance = margin(parts), []       # node 0 = object box, then the regions
        for i in range(int(mask.sum()) if explain else 0):
            mk = mask.clone()
            mk[i] = 0
            b2 = Batch(y=b.y, g=g, reg=reg[None], mask=mk[None], geom=b.geom, g2=g2)
            importance.append(base - margin(self._parts(b2)))
        sure = self.unsure_below is None or float(probs[c]) >= self.unsure_below
        heat = self.gradcam(im, b, c) if cam else None
        return dict(probs=dict(zip(self.classes, probs.tolist())), pred=self.classes[c], sure=sure,
                    parts=parts, importance=importance, regions=r, familiar=familiar, cam=heat,
                    is_familiar=self.familiar_below is None or familiar >= self.familiar_below)

    @torch.no_grad()
    def objects(self, im, prompts=None, thr=0.15, max_n=60):
        """every object OWLv2 finds (each bee / bulb): list of boxes, by the domain's object prompts."""
        im = im.convert("RGB")
        prompts = prompts or self.ck["owl"]["object"]
        proc, model = _owl()
        inp = proc(text=[prompts], images=[im], return_tensors="pt").to(DEVICE)
        inp["pixel_values"] = inp["pixel_values"].half()
        out = model(**inp)
        W, H = im.size
        S = max(W, H)
        sc = out.logits.float().sigmoid()[0].max(-1).values
        cx, cy, w, h = out.pred_boxes.float()[0].unbind(-1)
        boxes = torch.stack([(cx - w / 2) * S, (cy - h / 2) * S, (cx + w / 2) * S, (cy + h / 2) * S], -1)
        boxes[:, 0::2] = boxes[:, 0::2].clamp(0, W)
        boxes[:, 1::2] = boxes[:, 1::2].clamp(0, H)
        area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1]) / (W * H)
        keep = ((sc >= thr) & (area < 0.5) & (area > 0.002)).nonzero().squeeze(1)
        from torchvision.ops import nms
        keep = keep[nms(boxes[keep], sc[keep], 0.3)][:max_n] if len(keep) else keep
        return [dict(box=boxes[j].tolist(), score=float(sc[j])) for j in keep.tolist()]

    def each_object(self, im, **kw):
        """classify every object in a crowded photo on its own (no per-region explanations, for speed)."""
        im = im.convert("RGB")
        found = []
        heat = np.zeros((im.height, im.width), np.float32)
        for o in self.objects(im, **kw):
            x0, y0, x1, y1 = o["box"]
            m = 0.1 * max(x1 - x0, y1 - y0)
            crop = im.crop((max(0, x0 - m), max(0, y0 - m), min(im.width, x1 + m), min(im.height, y1 + m)))
            r = self(crop, explain=False)
            cx0, cy0 = int(max(0, x0 - m)), int(max(0, y0 - m))
            h, w = r["cam"].shape
            heat[cy0:cy0 + h, cx0:cx0 + w] = np.maximum(heat[cy0:cy0 + h, cx0:cx0 + w], r["cam"][: im.height - cy0, : im.width - cx0])
            found.append(dict(box=o["box"], pred=r["pred"], conf=max(r["probs"].values()), sure=r["sure"],
                              familiar=r["is_familiar"]))
        self.last_each_cam = heat
        return found

    @torch.no_grad()
    def search(self, im, text, thr=0.1, top=5):
        """boxes where OWLv2 finds `text` (comma-separated prompts allowed), each with its CLIP match to the text."""
        im = im.convert("RGB")
        prompts = [t.strip() for t in text.split(",") if t.strip()]
        proc, model = _owl()
        inp = proc(text=[prompts], images=[im], return_tensors="pt").to(DEVICE)
        inp["pixel_values"] = inp["pixel_values"].half()
        out = model(**inp)
        W, H = im.size
        S = max(W, H)
        sc, q = out.logits.float().sigmoid()[0].max(-1)
        cx, cy, w, h = out.pred_boxes.float()[0].unbind(-1)
        boxes = torch.stack([(cx - w / 2) * S, (cy - h / 2) * S, (cx + w / 2) * S, (cy + h / 2) * S], -1)
        boxes[:, 0::2] = boxes[:, 0::2].clamp(0, W)
        boxes[:, 1::2] = boxes[:, 1::2].clamp(0, H)
        keep = (sc >= thr).nonzero().squeeze(1)
        from torchvision.ops import nms
        keep = keep[nms(boxes[keep], sc[keep], 0.3)][:top] if len(keep) else keep
        found = []
        if len(keep):
            crops = [feat.crop_box(im, boxes[j].tolist()) for j in keep.tolist()]
            f = self.clip.encode_image(torch.stack([self.clip.transform(c) for c in crops]))
            t = l2n(self.clip.encode_text(prompts))
            sim = (f @ t.T).max(1).values
            for j, s in zip(keep.tolist(), sim.tolist()):
                found.append(dict(box=boxes[j].tolist(), prompt=prompts[int(q[j])], owl=float(sc[j]), clip=s))
        return found
