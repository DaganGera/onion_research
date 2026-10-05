"""Run the methods from the base paper's comparison table (implemented in adapters.py) on the same features,
split and seeds as everything else.

  python 10_comparison_methods.py [--methods clipadapter taskres clap graphadapter coop] [--shots ...]
CoOp back-propagates through the CLIP text encoder, so a GPU helps.
"""
import argparse

from fewshot import Store
from harness import ALL_K, load_text, run
from adapters import CLAP, CLIPAdapter, CoOp, GraphAdapter, TaskRes


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--methods", nargs="+", default=["clipadapter", "taskres", "clap", "graphadapter", "coop"])
    ap.add_argument("--shots", nargs="+", default=ALL_K)
    a = ap.parse_args()
    store = Store("clip", need=("global", "aug"))
    T = load_text("clip", "desc")
    makers = {"clipadapter": ("CLIPAdapter-clip", lambda: CLIPAdapter(T)),
              "taskres": ("TaskRes-clip", lambda: TaskRes(T)),
              "clap": ("CLAP-clip", lambda: CLAP(T)),
              "graphadapter": ("GraphAdapterLite-clip", lambda: GraphAdapter(T))}
    for m in a.methods:
        if m == "coop":
            run("CoOp-clip", lambda K: CoOp(K), store, shots=a.shots, pass_k=True)
        else:
            name, make = makers[m]
            run(name, make, store, shots=a.shots)


if __name__ == "__main__":
    main()
