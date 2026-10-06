# Kaggle: side experiment experiments/clip_finetune_bees.py (partial CLIP fine-tuning on bees). Not part of the method.
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
subprocess.run("python 20_prepare_bees.py", shell=True, cwd=W + "/src", env=env)
r = subprocess.run("python ../experiments/clip_finetune_bees.py", shell=True, cwd=W + "/src", env=env)
print("exit", r.returncode, flush=True)
f = W + "/domains/bees/results/clip_finetune.csv"
if os.path.exists(f):
    shutil.copy(f, "/kaggle/working/clip_finetune.csv")
    print(open(f).read(), flush=True)
shutil.rmtree(W, ignore_errors=True)
