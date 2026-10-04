"""Step 3.4 — the base paper's model, re-implemented from its method section (fewshot_paper.py), on the leak-free split.

  python 06b_basepaper.py      three variants:
    BasePaperFaithful-template   generic class-name templates (what the paper uses), one-hot cache
    BasePaperFaithful-desc       our visual descriptors as class vectors (same text as PRGA -> controlled comparison)
    BasePaperFaithful-desc-bal   as above with a class-balanced cache (as in our other methods)
Needs features/clip_paperpatches.pt (04b_paper_patches.py). Resumable via harness.
"""
import argparse

from fewshot import Store
from fewshot_paper import PaperBase
from harness import ALL_K, load_text, run

VARIANTS = [("PaperGraphCtrl-grid-desc", "desc", False),          # controlled study: 26 legacy grid windows
            ("PaperGraphCtrl-grid-desc-bal", "desc", True)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shots", nargs="+", default=ALL_K)
    a = ap.parse_args()
    store = Store("clip", need=("global", "aug", "paperpatches"))
    for name, text, bal in VARIANTS:
        T = load_text("clip", text)
        run(name, lambda T=T, bal=bal: PaperBase(T, balanced=bal), store, shots=a.shots)


if __name__ == "__main__":
    main()
