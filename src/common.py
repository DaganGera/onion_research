"""Shared paths, class list, seeding, feature I/O and metrics used by every script."""
import json
import os
import random
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
RAW = Path(os.environ.get("ONION_RAW", ROOT / "data/raw/onion_bulbs/Onion Image Dataset/2. Bulb"))   # Kaggle: env
META_RAW = ROOT / "data/meta.csv"                    # audit of every photo (01_audit_photos.py)
META_CLEAN = ROOT / "data/clean/meta_clean.csv"      # after 02_clean_dataset.py: what every later step uses
META = META_CLEAN if META_CLEAN.exists() else META_RAW
SPLITS = ROOT / "splits"
FEATS = ROOT / "features"
REGIONS = ROOT / "regions"
PROMPTS = ROOT / "prompts"
RESULTS = ROOT / "results"
FIGURES = ROOT / "figures"
CKPT = ROOT / "checkpoints"
for d in (SPLITS, FEATS, REGIONS, PROMPTS, RESULTS, FIGURES, CKPT):
    d.mkdir(parents=True, exist_ok=True)

# (health folder, variety folder) -> class name. Single- and multiple-bulb photos are merged into the same class.
CLASS_MAP = {
    ("1. Healthy", "1. Red Onion"): "healthy red onion",
    ("2. Unhealthy", "1. Red Onion"): "unhealthy red onion",
    ("1. Healthy", "2. White Onion"): "healthy white onion",
    ("2. Unhealthy", "2. White Onion"): "unhealthy white onion",
}
CLASSES = list(CLASS_MAP.values())
C = len(CLASSES)
SHOTS = [1, 2, 4, 8, 16]
SEEDS = [1, 2, 3]
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CLIP_NAME = "ViT-B-16-quickgelu"          # OpenAI CLIP weights need the QuickGELU variant
# Data-loading workers and CPU threads are capped (configurable) to keep memory and load bounded on small machines.
NUM_WORKERS = int(os.environ.get("ONION_WORKERS", "3"))
# DataLoader workers pass tensors via file descriptors; systemd services get a low open-file limit (1024).
torch.multiprocessing.set_sharing_strategy("file_system")
torch.set_num_threads(int(os.environ.get("ONION_THREADS", "4")))


def seed_all(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_meta():
    import pandas as pd
    return pd.read_csv(META)


def save_feats(name, **tensors):
    tmp = FEATS / f".{name}.pt.tmp"            # atomic: a crash mid-write never leaves a half file
    torch.save(tensors, tmp)
    os.replace(tmp, FEATS / f"{name}.pt")


def load_feats(name):
    return torch.load(FEATS / f"{name}.pt", weights_only=False)


def l2n(x, dim=-1):
    return x / x.norm(dim=dim, keepdim=True).clamp_min(1e-8)


def metrics(y_true, y_pred):
    from sklearn.metrics import accuracy_score, f1_score
    return {
        "acc": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", labels=list(range(C)), zero_division=0)),
    }


def read_json(p):
    with open(p) as f:
        return json.load(f)


def write_json(obj, p):
    os.makedirs(os.path.dirname(p), exist_ok=True)
    tmp = f"{p}.tmp"                           # atomic write-then-rename
    with open(tmp, "w") as f:
        json.dump(obj, f)
    os.replace(tmp, p)
