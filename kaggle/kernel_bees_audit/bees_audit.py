# Kaggle step 1 for the bee domain: download VarroaDataset (Zenodo 4085044), look at how it is organised,
# measure near-duplicate frames with CLIP, and check whether OWLv2 boxes land on the annotated mites.
# Output (/kaggle/working): bees/raw/{train,val,test}/..., bees/gt.csv, bees/clip_global.pt, audit.txt
import os, subprocess, sys, zipfile, glob, collections, json, random, urllib.request

W = "/kaggle/working/bees"
os.makedirs(W + "/raw", exist_ok=True)
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "open_clip_torch"], check=True)
for name in ("gt.csv", "val.zip", "test.zip", "train.zip"):
    dst = f"{W}/{name}"
    if not os.path.exists(dst):
        for attempt in range(5):
            try:
                urllib.request.urlretrieve(f"https://zenodo.org/api/records/4085044/files/{name}/content", dst)
                break
            except Exception as e:
                print("retry", name, e, flush=True)
    if name.endswith(".zip"):
        with zipfile.ZipFile(dst) as z:
            z.extractall(W + "/raw")
        os.remove(dst)
    print("got", name, flush=True)

out = open("/kaggle/working/audit.txt", "w")
def log(*a):
    s = " ".join(str(x) for x in a)
    print(s, flush=True); out.write(s + "\n"); out.flush()

files = sorted(p for p in glob.glob(W + "/raw/**/*", recursive=True) if os.path.isfile(p))
log("files:", len(files))
log("top dirs:", collections.Counter(os.path.relpath(p, W + "/raw").split("/")[0] for p in files))
log("extensions:", collections.Counter(os.path.splitext(p)[1] for p in files))
for p in files[:15] + files[-5:]:
    log("  ", os.path.relpath(p, W + "/raw"))
gt = open(W + "/gt.csv").read().splitlines()
log("gt lines:", len(gt))
for line in gt[:8]:
    log("  gt:", line)

# parse gt: "<path> <label> <x0 y0> <x1 y1> ..." (format from the GitHub README; checked by printing above)
import re
rows = []
for line in gt:
    parts = line.replace(",", " ").split()
    if len(parts) < 2:
        continue
    nums = [float(x) for x in re.findall(r"-?\d+(?:\.\d+)?", " ".join(parts[2:]))]
    rows.append((parts[0], int(float(parts[1])), [nums[i:i + 4] for i in range(0, len(nums) - 3, 4)]))
log("parsed:", len(rows), "labels:", collections.Counter(r[1] for r in rows))
log("boxes per infected image:", collections.Counter(len(r[2]) for r in rows if r[1] == 1).most_common(6))
by_name = {os.path.basename(p): p for p in files}
missing = [r[0] for r in rows if os.path.basename(r[0]) not in by_name]
log("gt paths not found on disk:", len(missing), missing[:3])

# filename structure (video / frame ids?)
stems = [os.path.splitext(os.path.basename(r[0]))[0] for r in rows]
log("name examples:", random.Random(0).sample(stems, 12))
log("name token patterns:", collections.Counter(re.sub(r"\d+", "#", s) for s in stems).most_common(8))

from PIL import Image
sizes = collections.Counter(Image.open(by_name[os.path.basename(r[0])]).size for r in rows[:500])
log("sizes (first 500):", sizes.most_common(5))

# CLIP global features for every image -> near-duplicate structure
import torch, open_clip
dev = "cuda"
model, _, pre = open_clip.create_model_and_transforms("ViT-B-16-quickgelu", pretrained="openai")
model = model.to(dev).eval().half()
paths = [by_name[os.path.basename(r[0])] for r in rows if os.path.basename(r[0]) in by_name]
feats = []
with torch.no_grad():
    for i in range(0, len(paths), 256):
        x = torch.stack([pre(Image.open(p).convert("RGB")) for p in paths[i:i + 256]]).to(dev).half()
        f = model.encode_image(x).float()
        feats.append((f / f.norm(dim=-1, keepdim=True)).cpu())
X = torch.cat(feats)
rel = [os.path.relpath(p, W + "/raw") for p in paths]
torch.save({"paths": rel, "feats": X.half()}, W + "/clip_global.pt")
split = [r.split("/")[0] for r in rel]
lab = {os.path.basename(r[0]): r[1] for r in rows}
y = torch.tensor([lab[os.path.basename(p)] for p in paths])

Xc = X.to(dev)
nn1 = torch.empty(len(X)); cross = {}
for i in range(0, len(X), 2048):
    s = Xc[i:i + 2048] @ Xc.T
    s[torch.arange(s.shape[0]), torch.arange(i, i + s.shape[0])] = -1
    nn1[i:i + 2048] = s.max(1).values.cpu()
log("nearest-neighbour cosine quantiles (any image):",
    [round(float(q), 4) for q in torch.quantile(nn1, torch.tensor([.05, .25, .5, .75, .95]))])
sp = torch.tensor([{"train": 0, "val": 1, "test": 2}.get(s, 3) for s in split])
for a, b in (("test", "train"), ("val", "train")):
    ia, ib = (sp == {"train": 0, "val": 1, "test": 2}[a]).nonzero().squeeze(1), (sp == 0).nonzero().squeeze(1)
    if len(ia) and len(ib):
        m = torch.cat([(Xc[ia[j:j + 2048]] @ Xc[ib].T).max(1).values.cpu() for j in range(0, len(ia), 2048)])
        log(f"{a} -> nearest {b} cosine quantiles:", [round(float(q), 4) for q in torch.quantile(m, torch.tensor([.05, .25, .5, .75, .95]))],
            "| share >= 0.95:", round(float((m >= .95).float().mean()), 3), ">= 0.97:", round(float((m >= .97).float().mean()), 3))

import networkx as nx
for thr in (0.93, 0.95, 0.97):
    G = nx.Graph(); G.add_nodes_from(range(len(X)))
    for i in range(0, len(X), 2048):
        s = Xc[i:i + 2048] @ Xc.T
        r_, c_ = (s >= thr).nonzero(as_tuple=True)
        G.add_edges_from((int(a) + i, int(b)) for a, b in zip(r_.cpu(), c_.cpu()) if int(a) + i < int(b))
    comps = sorted(nx.connected_components(G), key=len, reverse=True)
    mixed = sum(1 for c in comps if len({int(y[k]) for k in c}) > 1)
    log(f"thr {thr}: components {len(comps)}, largest {[len(c) for c in comps[:5]]}, singletons {sum(len(c)==1 for c in comps)}, label-mixed {mixed}")

# OWLv2: do the boxes land on the annotated mites?
from transformers import Owlv2ForObjectDetection, Owlv2Processor
proc = Owlv2Processor.from_pretrained("google/owlv2-base-patch16-ensemble")
owl = Owlv2ForObjectDetection.from_pretrained("google/owlv2-base-patch16-ensemble", torch_dtype=torch.float16).to(dev).eval()
prompts = ["a varroa mite", "a mite", "a small red-brown oval mite", "a tiny brown parasite", "a reddish-brown dot",
           "a honeybee", "a bee"]
inf = [r for r in rows if r[1] == 1 and r[2] and os.path.basename(r[0]) in by_name]
random.Random(0).shuffle(inf)
hits = collections.Counter(); scores = collections.defaultdict(list)
with torch.no_grad():
    for r in inf[:300]:
        im = Image.open(by_name[os.path.basename(r[0])]).convert("RGB")
        Wd, Hd = im.size; S = max(Wd, Hd)
        inp = proc(text=[prompts], images=[im], return_tensors="pt").to(dev)
        inp["pixel_values"] = inp["pixel_values"].half()
        o = owl(**inp)
        pr = o.logits.float().sigmoid()[0]; bx = o.pred_boxes.float()[0]
        for k, p in enumerate(prompts):
            j = int(pr[:, k].argmax()); cx, cy = float(bx[j, 0]) * S, float(bx[j, 1]) * S
            scores[p].append(float(pr[j, k]))
            # gt box may be (x0,y0,x1,y1) or (x,y,w,h): accept centre inside either reading
            ok = any((b[0] <= cx <= b[2] and b[1] <= cy <= b[3]) or (b[0] <= cx <= b[0] + b[2] and b[1] <= cy <= b[1] + b[3]) for b in r[2])
            hits[p] += ok
for p in prompts:
    log(f"OWLv2 top box inside a mite box: {p!r:32s} {hits[p] / min(300, len(inf)):.2f}   median score {sorted(scores[p])[len(scores[p]) // 2]:.3f}")
log("example gt boxes:", [r[2] for r in inf[:3]], "image size", Image.open(by_name[os.path.basename(inf[0][0])]).size)
out.close()
