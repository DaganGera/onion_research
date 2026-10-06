# Kaggle: build the web-app checkpoints (src/24_export_app_model.py) for both domains, one GPU each.
#   onions: features already cached in the onion-fewshot-work dataset (CLIP B/16, DINOv2, OWLv2 regions)
#   bees:   features computed here with CLIP L/14 (same split and OWLv2 boxes as the full bee run)
import glob, os, shutil, subprocess, sys, urllib.request, zipfile

W = "/kaggle/working/ONION"
code = glob.glob("/kaggle/input/**/24_export_app_model.py", recursive=True)[0]
shutil.copytree(os.path.dirname(os.path.dirname(code)), W, dirs_exist_ok=True)
work = os.path.dirname(glob.glob("/kaggle/input/**/run_pipeline.sh", recursive=True)[0])
for d in ("splits", "data", "prompts", "regions"):
    shutil.copytree(f"{work}/{d}", f"{W}/{d}", dirs_exist_ok=True)
os.symlink(f"{work}/features", f"{W}/features")
onion_raw = os.path.dirname(sorted(glob.glob("/kaggle/input/**/1. Healthy", recursive=True), key=len)[0])

raw = W + "/domains/bees/data/raw"
os.makedirs(raw, exist_ok=True)
for name in ("gt.csv", "val.zip", "test.zip", "train.zip"):
    dst = f"{raw}/../{name}"
    urllib.request.urlretrieve(f"https://zenodo.org/api/records/4085044/files/{name}/content", dst)
    if name.endswith(".zip"):
        zipfile.ZipFile(dst).extractall(raw)
        os.remove(dst)
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "open_clip_torch"], check=True)


def run(cmds, gpu, name, extra):
    script = " && ".join(f"python {c}" for c in cmds)
    env = {**os.environ, "PYTHONUNBUFFERED": "1", "CUDA_VISIBLE_DEVICES": str(gpu), **extra}
    return subprocess.Popen(f"({script}) > /kaggle/working/{name}.log 2>&1", shell=True, cwd=W + "/src", env=env)


p_onion = run(["24_export_app_model.py --backbone clip"], 0, "onion", {"DATASET": "onion", "ONION_RAW": onion_raw})
p_bees = run(["20_prepare_bees.py",
              "06_extract_features.py --backbone clip_l14 --mode global", "06_extract_features.py --backbone clip_l14 --mode aug",
              "06_extract_features.py --backbone clip_l14 --mode regions", "07_text_embeddings.py --backbone clip_l14",
              "06_extract_features.py --backbone dinov2 --mode global", "06_extract_features.py --backbone dinov2 --mode aug",
              "24_export_app_model.py --backbone clip_l14"], 1, "bees", {"DATASET": "bees"})
for n, p in (("onion", p_onion), ("bees", p_bees)):
    print(n, "exit", p.wait(), flush=True)
    print(open(f"/kaggle/working/{n}.log").read()[-2500:], flush=True)
out = "/kaggle/working/out"
os.makedirs(out + "/checkpoints", exist_ok=True)
for f in glob.glob(W + "/checkpoints/app_*.pt") + glob.glob(W + "/domains/bees/checkpoints/app_*.pt"):
    shutil.copy(f, out + "/checkpoints/")
for dom, d in (("onion", W + "/app_examples"), ("bees", W + "/domains/bees/app_examples")):
    if os.path.isdir(d):
        shutil.copytree(d, f"{out}/app_examples/{dom}", dirs_exist_ok=True)
subprocess.run("tar czf /kaggle/working/app_export.tar.gz -C /kaggle/working out", shell=True)
shutil.rmtree(W, ignore_errors=True)
