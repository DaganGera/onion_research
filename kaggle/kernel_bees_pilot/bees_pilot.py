# Kaggle: bee-domain pilot (src/22_bees_pilot.py) on a small slice. Images come from the bees-audit kernel output.
import glob, os, shutil, subprocess, sys

W = "/kaggle/working/ONION"
src = glob.glob("/kaggle/input/**/22_bees_pilot.py", recursive=True)[0]
shutil.copytree(os.path.dirname(os.path.dirname(src)), W, dirs_exist_ok=True)
found = glob.glob("/kaggle/input/**/bees/gt.csv", recursive=True)
if found and os.path.isdir(os.path.dirname(found[0]) + "/raw/train"):
    os.symlink(os.path.dirname(found[0]) + "/raw", W + "/domains/bees/data/raw")
else:                                                    # download VarroaDataset from Zenodo (about 1.2 GB)
    import urllib.request, zipfile
    raw = W + "/domains/bees/data/raw"
    os.makedirs(raw, exist_ok=True)
    for name in ("gt.csv", "val.zip", "test.zip", "train.zip"):
        dst = f"{raw}/../{name}"
        urllib.request.urlretrieve(f"https://zenodo.org/api/records/4085044/files/{name}/content", dst)
        if name.endswith(".zip"):
            zipfile.ZipFile(dst).extractall(raw)
            os.remove(dst)
print("images:", len(glob.glob(W + "/domains/bees/data/raw/*/videos/*/*.png")), flush=True)
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "open_clip_torch"], check=True)
env = {**os.environ, "DATASET": "bees", "PYTHONUNBUFFERED": "1"}
r = subprocess.run("python 22_bees_pilot.py", shell=True, cwd=W + "/src", env=env)
print("exit", r.returncode, flush=True)
if os.path.exists(W + "/domains/bees/results/pilot.csv"):
    shutil.copy(W + "/domains/bees/results/pilot.csv", "/kaggle/working/pilot.csv")
shutil.rmtree(W, ignore_errors=True)
