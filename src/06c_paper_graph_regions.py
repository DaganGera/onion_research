"""Step 5b — CONTROLLED comparison: the base paper's graph layer on different nodes (same layer, pooling, cache, training).

  BasePaperFaithful-desc*        26 grid patches                                   (06b_basepaper.py)
  PaperGraph-regions-desc*       whole photo + object + up to 4 OWLv2 regions      <- only the nodes change
  PaperGraph-regions+text-desc*  the same + 4 class-description nodes              <- and class nodes are added
  (*-bal = class-balanced cache; the others use the paper's one-hot cache)
All use our visual descriptors as class vectors. Needs features/clip_regionviews.pt (04c_region_views.py).
"""
import argparse

from fewshot import Store
from fewshot_paper import PaperRegions
from harness import ALL_K, load_text, run

VARIANTS = [("PaperGraphCtrl-regions-desc", False, False),
            ("PaperGraphCtrl-regions-desc-bal", False, True),
            ("PaperGraphCtrl-regions+text-desc", True, False),
            ("PaperGraphCtrl-regions+text-desc-bal", True, True)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shots", nargs="+", default=ALL_K)
    a = ap.parse_args()
    store = Store("clip", need=("global", "aug", "regionviews"))
    T = load_text("clip", "desc")
    for name, with_text, bal in VARIANTS:
        run(name, lambda wt=with_text, b=bal: PaperRegions(T, with_text=wt, balanced=b), store, shots=a.shots)


if __name__ == "__main__":
    main()
