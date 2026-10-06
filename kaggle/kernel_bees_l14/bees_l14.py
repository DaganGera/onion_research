# Kaggle: bee domain, full run with CLIP ViT-L/14 as the backbone (chosen by the pilot, src/22_bees_pilot.py).
# Same split, OWLv2 boxes (from the first bee run), methods and fair protocol as the CLIP B/16 run; only the backbone
# changes. Output: /kaggle/working/bees_l14_results.tar.gz
import glob, os, shutil, subprocess, sys, time

W = "/kaggle/working/ONION"
src = glob.glob("/kaggle/input/**/22_bees_pilot.py", recursive=True)[0]
shutil.copytree(os.path.dirname(os.path.dirname(src)), W, dirs_exist_ok=True)
raw = W + "/domains/bees/data/raw"
os.makedirs(raw, exist_ok=True)
import urllib.request, zipfile
for name in ("gt.csv", "val.zip", "test.zip", "train.zip"):
    dst = f"{raw}/../{name}"
    urllib.request.urlretrieve(f"https://zenodo.org/api/records/4085044/files/{name}/content", dst)
    if name.endswith(".zip"):
        zipfile.ZipFile(dst).extractall(raw)
        os.remove(dst)
print("images:", len(glob.glob(raw + "/*/videos/*/*.png")), flush=True)
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "open_clip_torch"], check=True)
ENV = {**os.environ, "DATASET": "bees", "PYTHONUNBUFFERED": "1", "ONION_WORKERS": "4"}
LOG = "/kaggle/working/logs"
os.makedirs(LOG, exist_ok=True)


def run(cmds, gpu, name):
    """run commands one after another on one GPU; returns the process."""
    script = " && ".join(f"python {c}" for c in cmds)
    return subprocess.Popen(f"({script}) > {LOG}/{name}.log 2>&1", shell=True, cwd=W + "/src",
                            env={**ENV, "CUDA_VISIBLE_DEVICES": str(gpu)})


def wait(*ps):
    for p in ps:
        print("exit", p.wait(), flush=True)


t0 = time.time()
wait(run(["20_prepare_bees.py"], 0, "prepare"))
assert os.path.exists(W + "/domains/bees/regions/regions.json")
wait(run(["06_extract_features.py --backbone clip_l14 --mode global", "06_extract_features.py --backbone clip_l14 --mode aug",
          "06_extract_features.py --backbone clip_l14 --mode regions", "07_text_embeddings.py --backbone clip_l14"], 0, "feat_l14"),
     run(["06_extract_features.py --backbone dinov2 --mode global", "06_extract_features.py --backbone dinov2 --mode aug"], 1,
         "feat_dinov2"))
print(f"features done {time.time() - t0:.0f}s", flush=True)
wait(run(["18_equal_training.py --backbone clip_l14 --methods prga tipf cafo",
          "08_baselines.py --backbone clip_l14 --methods zs lp tip"], 0, "gpu0_methods"),
     run(["18_equal_training.py --backbone clip_l14 --methods clipadapter taskres clap graphadapter",
          "08_baselines.py --backbone clip_l14 --methods knn lp --leakage"], 1, "gpu1_methods"))
print(f"all done {time.time() - t0:.0f}s", flush=True)
B = W + "/domains/bees"
for d in ("results",):
    shutil.copytree(f"{B}/{d}", f"/kaggle/working/out/{d}", dirs_exist_ok=True, ignore=shutil.ignore_patterns("*.npy"))
shutil.copytree(LOG, "/kaggle/working/out/logs", dirs_exist_ok=True)
subprocess.run("tar czf /kaggle/working/bees_l14_results.tar.gz -C /kaggle/working out", shell=True)
for f in sorted(glob.glob(LOG + "/*.log")):
    print("=" * 20, f, flush=True)
    print(open(f).read()[-3000:], flush=True)
shutil.rmtree(W, ignore_errors=True)
