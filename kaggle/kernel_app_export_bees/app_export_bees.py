# Kaggle: rebuild the bee web-app checkpoints (src/24_export_app_model.py) with both calibrations
# (temperature + class bias, and temperature only). Same split, OWLv2 boxes and CLIP L/14 features as before.
import glob, os, shutil, subprocess, sys, urllib.request, zipfile

W = "/kaggle/working/ONION"
code = glob.glob("/kaggle/input/**/24_export_app_model.py", recursive=True)[0]
shutil.copytree(os.path.dirname(os.path.dirname(code)), W, dirs_exist_ok=True)
raw = W + "/domains/bees/data/raw"
os.makedirs(raw, exist_ok=True)
for name in ("gt.csv", "val.zip", "test.zip", "train.zip"):
    dst = f"{raw}/../{name}"
    urllib.request.urlretrieve(f"https://zenodo.org/api/records/4085044/files/{name}/content", dst)
    if name.endswith(".zip"):
        zipfile.ZipFile(dst).extractall(raw)
        os.remove(dst)
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "open_clip_torch"], check=True)
env = {**os.environ, "DATASET": "bees", "PYTHONUNBUFFERED": "1"}
cmds = ["20_prepare_bees.py", "06_extract_features.py --backbone clip_l14 --mode global",
        "06_extract_features.py --backbone clip_l14 --mode aug", "06_extract_features.py --backbone clip_l14 --mode regions",
        "07_text_embeddings.py --backbone clip_l14", "06_extract_features.py --backbone dinov2 --mode global",
        "06_extract_features.py --backbone dinov2 --mode aug", "24_export_app_model.py --backbone clip_l14"]
for c in cmds:
    r = subprocess.run(f"python {c}", shell=True, cwd=W + "/src", env=env, capture_output=True, text=True)
    print(c, "exit", r.returncode, "\n", r.stdout[-1500:], r.stderr[-1500:] if r.returncode else "", flush=True)
os.makedirs("/kaggle/working/out", exist_ok=True)
for f in glob.glob(W + "/domains/bees/checkpoints/app_*.pt"):
    shutil.copy(f, "/kaggle/working/out/")
shutil.rmtree(W, ignore_errors=True)
