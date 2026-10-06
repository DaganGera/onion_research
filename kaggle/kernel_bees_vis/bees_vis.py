# Kaggle: bee-domain figures (src/23_bees_visualise.py): correct examples, Grad-CAM per model, OWLv2 regions.
import glob, os, shutil, subprocess, sys, urllib.request, zipfile

W = "/kaggle/working/ONION"
src = glob.glob("/kaggle/input/**/23_bees_visualise.py", recursive=True)[0]
shutil.copytree(os.path.dirname(os.path.dirname(src)), W, dirs_exist_ok=True)
raw = W + "/domains/bees/data/raw"
os.makedirs(raw, exist_ok=True)
for name in ("gt.csv", "val.zip", "test.zip", "train.zip"):
    dst = f"{raw}/../{name}"
    urllib.request.urlretrieve(f"https://zenodo.org/api/records/4085044/files/{name}/content", dst)
    if name.endswith(".zip"):
        zipfile.ZipFile(dst).extractall(raw)
        os.remove(dst)
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "open_clip_torch"], check=True)
r = subprocess.run("python 23_bees_visualise.py", shell=True, cwd=W + "/src",
                   env={**os.environ, "DATASET": "bees", "PYTHONUNBUFFERED": "1"})
print("exit", r.returncode, flush=True)
os.makedirs("/kaggle/working/out", exist_ok=True)
for f in glob.glob(W + "/domains/bees/figures/bees_*.png") + glob.glob(W + "/domains/bees/results/visual_summary.csv"):
    shutil.copy(f, "/kaggle/working/out/")
shutil.rmtree(W, ignore_errors=True)
