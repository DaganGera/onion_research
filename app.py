"""Web app to try the final model (PRGA + DINOv2 cache) on your own photos: onion bulbs and honeybees.

  .venv/bin/python app.py            then open http://127.0.0.1:7860
  .venv/bin/python app.py --share    also prints a temporary public link (e.g. for a mentor)

The app only predicts; the models were trained on Kaggle by src/24_export_app_model.py and saved in checkpoints/.
Research prototype: see README.md and domains/bees/README.md for what the scores mean.
"""
import argparse
import sys
from pathlib import Path

import gradio as gr
from PIL import ImageDraw

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
from app_model import Predictor  # noqa: E402

CKPT = ROOT / "checkpoints"
EXAMPLES = ROOT / "app_examples"
DOMAINS = {
    "onion": dict(title="Onion bulbs", what="a photo of one or more onion bulbs",
                  note="4 classes: healthy / unhealthy × red / white. Trained on 1 photo set from one site."),
    "bees": dict(title="Honeybees (Varroa mite)", what="a close-up of a single bee, seen from above",
                 note="2 classes: healthy bee / bee with a Varroa mite. Trained on lab video crops; "
                      "scores on new videos are modest (macro-F1 about 0.7)."),
}
TRAINED_ON = {"4 photos per class": "4", "all training photos": "full"}
COLORS = {"object": (0, 170, 0), "instance": (30, 110, 255), "spot": (230, 30, 30)}
_models = {}


def predictor(domain, K):
    if (domain, K) not in _models:
        _models[domain, K] = Predictor(CKPT / f"app_{domain}_K{K}.pt")
    return _models[domain, K]


def draw(im, r):
    im = im.convert("RGB").copy()
    d = ImageDraw.Draw(im)
    w = max(2, im.width // 250)
    if r.get("object"):
        d.rectangle(r["object"]["box"], outline=COLORS["object"], width=w)
    for x in r["regions"]:
        d.rectangle(x["box"], outline=COLORS[x["kind"]], width=w)
    return im


def make_tab(domain):
    info = DOMAINS[domain]

    def run(image, trained_on):
        if image is None:
            return None, None, "Upload a photo first."
        K = TRAINED_ON[trained_on]
        p = predictor(domain, K)
        probs, r = p(image)
        t = p.ck["test"]
        notes = (f"**Model:** PRGA + DINOv2 cache on {p.ck['backbone']}, trained on {p.ck['n_support']} photos "
                 f"({trained_on}). Test macro-F1 of this exact model: **{t['macro_f1']:.3f}** on {t['n']} held-out "
                 f"photos.\n\n**Boxes** (what PRGA's region nodes look at): green = whole object, blue = part / single "
                 f"item, red = suspicious spot. {len(r['regions'])} regions found.\n\n_{info['note']} "
                 f"Research prototype, not a diagnostic tool._")
        return probs, draw(image, r), notes

    with gr.Tab(info["title"]):
        gr.Markdown(f"Upload {info['what']}.")
        with gr.Row():
            with gr.Column():
                image = gr.Image(type="pil", label="Photo")
                trained_on = gr.Radio(list(TRAINED_ON), value="all training photos", label="Model trained on")
                go = gr.Button("Classify", variant="primary")
                ex = sorted((EXAMPLES / domain).glob("*.jpg"))
                if ex:
                    gr.Examples([[str(p)] for p in ex], inputs=[image], label="Test photos (name = true class)")
            with gr.Column():
                label = gr.Label(label="Prediction", num_top_classes=4)
                boxes = gr.Image(label="Regions used by the model")
                notes = gr.Markdown()
        go.click(run, [image, trained_on], [label, boxes, notes])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--share", action="store_true")
    ap.add_argument("--port", type=int, default=7860)
    a = ap.parse_args()
    missing = [f"app_{d}_K{k}.pt" for d in DOMAINS for k in TRAINED_ON.values() if not (CKPT / f"app_{d}_K{k}.pt").exists()]
    if missing:
        sys.exit(f"Missing checkpoints in {CKPT}: {missing}. Build them on Kaggle with src/24_export_app_model.py.")
    with gr.Blocks(title="Few-shot inspection: onions and bees") as demo:
        gr.Markdown("# Few-shot inspection: onion bulbs and honeybees\n"
                    "PRGA + DINOv2 cache: frozen CLIP and DINOv2, a small graph adapter trained on a few labelled "
                    "photos. The first prediction in each tab loads the models (about 10-30 s).")
        for d in DOMAINS:
            make_tab(d)
    demo.launch(server_name="127.0.0.1", server_port=a.port, share=a.share)


if __name__ == "__main__":
    main()
