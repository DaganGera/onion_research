"""Web app to try the final model (PRGA + DINOv2 cache) on your own photos: onion bulbs and honeybees.

  .venv/bin/python app.py            then open http://127.0.0.1:7860
  .venv/bin/python app.py --share    also prints a temporary public link (e.g. for a mentor)

Per photo: calibrated class probabilities (with a "not sure" flag), the regions PRGA's graph uses and how much each
one matters, a breakdown of the score into its four parts, and a text search ("a varroa mite") inside the photo.
The app only predicts; the models were trained on Kaggle by src/24_export_app_model.py and saved in checkpoints/.
Research prototype: see README.md and domains/bees/README.md for what the scores mean.
"""
import argparse
import sys
from pathlib import Path

import gradio as gr
import matplotlib
import numpy as np
from PIL import ImageDraw

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
from app_model import PARTS, Predictor  # noqa: E402

CKPT = ROOT / "checkpoints"
EXAMPLES = ROOT / "app_examples"
DOMAINS = {
    "onion": dict(title="Onion bulbs", what="a photo of one or more onion bulbs", search="black mould, a rotten spot",
                  each="bulb", calibration="bias",
                  note="4 classes: healthy / unhealthy × red / white. Trained on photos from one site."),
    "bees": dict(title="Honeybees (Varroa mite)", what="a close-up of a single bee, seen from above",
                 search="a varroa mite", each="bee", calibration="temperature",
                 note="2 classes: healthy bee / bee with a Varroa mite. Trained on lab video crops; "
                      "on new videos macro-F1 is about 0.7."),
}
DOMAINS["steel"] = dict(title="Steel surface defects", what="a close-up photo of a steel surface (NEU-style patch)",
                       search="a scratch, a crack", each="defect", calibration="bias",
                       note="6 classes: crazing, inclusion, patches, pitted surface, rolled-in scale, scratches "
                            "(NEU hot-rolled steel strips). Industrial generalisation test of the same model.")
TRAINED_ON = {"4 photos per class": "4", "all training photos": "full"}
COLORS = {"object": (0, 170, 0), "instance": (30, 110, 255), "spot": (230, 30, 30)}
SEARCH_COLOR = (160, 32, 240)
_models = {}


def predictor(domain, K):
    if (domain, K) not in _models:
        _models[domain, K] = Predictor(CKPT / f"app_{domain}_K{K}.pt", DOMAINS[domain]["calibration"])
    return _models[domain, K]


def draw_each(im, found):
    """one box per object: red = defect / pest class, green = healthy, orange = not sure."""
    im = im.convert("RGB").copy()
    d = ImageDraw.Draw(im)
    w = max(2, im.width // 300)
    for f in found:
        col = (255, 150, 0) if not f["sure"] else (0, 170, 0) if f["pred"].startswith("healthy") else (230, 30, 30)
        d.rectangle(f["box"], outline=col, width=w)
    return im


def draw(im, r, found):
    im = im.convert("RGB").copy()
    d = ImageDraw.Draw(im)
    w = max(2, im.width // 250)
    boxes = ([("object", r["object"]["box"])] if r.get("object") else []) + [(x["kind"], x["box"]) for x in r["regions"]]
    for i, (kind, b) in enumerate(boxes):
        d.rectangle(b, outline=COLORS[kind], width=w)
        d.text((b[0] + 3, b[1] + 2), str(i), fill=COLORS[kind])
    for f in found:
        d.rectangle(f["box"], outline=SEARCH_COLOR, width=w + 1)
        d.text((f["box"][0] + 3, f["box"][3] - 12), f"{f['owl']:.2f}", fill=SEARCH_COLOR)
    return im


def breakdown_plot(parts, classes):
    """each part's push towards each class, relative to that part's average over the classes."""
    v = (parts - parts.mean(1, keepdims=True)).numpy()
    fig, ax = plt.subplots(figsize=(6.4, 0.8 + 0.55 * len(classes) * len(PARTS) / 2))
    y = np.arange(len(PARTS))
    h = 0.8 / len(classes)
    cols = ["#2a6fdb", "#e07b39", "#3b9c5a", "#9b59b6"]
    for c, name in enumerate(classes):
        ax.barh(y + c * h, v[:, c], height=h, color=cols[c % 4], label=name)
    ax.axvline(0, color="#666", lw=0.8)
    ax.set_yticks(y + h * (len(classes) - 1) / 2, [p.split(" (")[0] for p in PARTS], fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel("push towards the class (score units, centred per part)", fontsize=8)
    ax.legend(fontsize=8, loc="lower right")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.tight_layout()
    return fig


def make_tab(domain):
    info = DOMAINS[domain]

    def run(image, trained_on, query, each):
        if image is None:
            return None, None, None, "Upload a photo first."
        p = predictor(domain, TRAINED_ON[trained_on])
        if each:
            found = p.each_object(image)
            if not found:
                return None, image, None, f"No {info['each']} found in the photo."
            counts = {c: sum(f["pred"] == c for f in found) for c in p.classes}
            unsure = sum(not f["sure"] for f in found)
            unfamiliar = sum(not f["familiar"] for f in found)
            share = {c: n / len(found) for c, n in counts.items()}
            lines = [f"### {len(found)} {info['each']}s found and classified one by one",
                     *[f"- **{c}**: {n} ({share[c]:.0%})" for c, n in counts.items()],
                     f"- not sure: {unsure}" + (f"; unlike the training photos: {unfamiliar}" if unfamiliar else ""),
                     "", "_Boxes: red = defect / pest class, green = healthy, orange = not sure. Each crop is classified "
                     "on its own; small or blurred crops are less reliable._"]
            if unfamiliar > len(found) / 2:
                lines.insert(1, f"**Warning:** most {info['each']}s look unlike the training photos "
                                f"({info['note'].split('.')[0].lower()}), so these counts are unreliable.")
            return share, draw_each(image, found), None, "\n".join(lines)
        out = p(image)
        found = p.search(image, query) if query and query.strip() else []
        t, cal = p.ck["test"], p.ck.get("calib") or {}
        conf = max(out["probs"].values())
        lines = []
        if not out["is_familiar"]:
            lines.append(f"**Warning: this photo looks unlike the training photos** (similarity {out['familiar']:.2f}, "
                         f"95 % of validation photos are above {p.familiar_below:.2f}). The result below is unreliable.")
        if out["sure"]:
            lines.append(f"### {out['pred']} ({conf:.0%})")
        else:
            lines.append(f"### Not sure: leaning to {out['pred']} ({conf:.0%}). Check this one by hand.")
            lines.append(f"_Below the confidence threshold {p.unsure_below:.2f}; on validation, photos above it were "
                         f"right at least 90 % of the time._")
        names = (["object box"] if out["regions"].get("object") else []) + [f"{x['kind']} ({x['prompt']})" for x in out["regions"]["regions"]]
        if out["importance"]:
            lines.append("**How much each region matters**: how much the lead of the predicted class over the "
                         "runner-up shrinks (+) or grows (−) when that box is removed from the graph:")
            lines += [f"- box {i}: {n}: {v:+.2f}" for i, (n, v) in enumerate(zip(names, out["importance"]))]
        if query and query.strip():
            if found:
                lines.append(f"**Search \"{query}\"** (purple boxes): " + "; ".join(
                    f"{f['prompt']}: detector {f['owl']:.2f}, CLIP match {f['clip']:.2f}" for f in found))
            else:
                lines.append(f"**Search \"{query}\":** nothing found above the detector threshold (0.10).")
        lines.append(f"\n**Model:** PRGA + DINOv2 cache on {p.ck['backbone']}, trained on {p.ck['n_support']} photos "
                     f"({trained_on}). Test macro-F1 {t['macro_f1']:.3f} on {t['n']} held-out photos"
                     + (f"; calibration error {cal['test_ece_raw']:.3f} → {cal['test_ece_calibrated']:.3f} after "
                        f"calibration on validation." if cal else "."))
        lines.append(f"\n_{info['note']} Research prototype, not a diagnostic tool._")
        return out["probs"], draw(image, out["regions"], found), breakdown_plot(out["parts"], p.classes), "\n".join(lines)

    with gr.Tab(info["title"]):
        gr.Markdown(f"Upload {info['what']}.")
        with gr.Row():
            with gr.Column():
                image = gr.Image(type="pil", label="Photo")
                trained_on = gr.Radio(list(TRAINED_ON), value="all training photos", label="Model trained on")
                query = gr.Textbox(label="Search inside the photo (optional, comma-separated)",
                                   placeholder=f"e.g. {info['search']}")
                each = gr.Checkbox(label=f"Many {info['each']}s in the photo: find and classify each {info['each']}")
                go = gr.Button("Classify", variant="primary")
                ex = sorted((EXAMPLES / domain).glob("*.jpg"))
                if ex:
                    gr.Examples([[str(p)] for p in ex], inputs=[image], label="Test photos (name = true class)")
            with gr.Column():
                label = gr.Label(label="Prediction (calibrated)", num_top_classes=4)
                boxes = gr.Image(label="Regions (green object, blue part, red spot; numbered) and search hits (purple)")
                parts = gr.Plot(label="Why: the score's four parts")
                notes = gr.Markdown()
        go.click(run, [image, trained_on, query, each], [label, boxes, parts, notes])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--share", action="store_true")
    ap.add_argument("--port", type=int, default=7860)
    a = ap.parse_args()
    ready = [d for d in DOMAINS if all((CKPT / f"app_{d}_K{k}.pt").exists() for k in TRAINED_ON.values())]
    if not ready:
        sys.exit(f"No checkpoints in {CKPT}. Build them on Kaggle with src/24_export_app_model.py.")
    with gr.Blocks(title="Few-shot inspection: onions and bees") as demo:
        gr.Markdown("# Few-shot inspection: onion bulbs, honeybees and steel\n"
                    "PRGA + DINOv2 cache: frozen CLIP and DINOv2, a small graph adapter trained on a few labelled "
                    "photos. The first prediction in each tab loads the models (about 10-30 s).")
        for d in ready:
            make_tab(d)
    demo.launch(server_name="127.0.0.1", server_port=a.port, share=a.share)


if __name__ == "__main__":
    main()
