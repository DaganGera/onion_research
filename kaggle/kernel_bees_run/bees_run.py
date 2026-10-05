# Kaggle step 2 for the bee domain: leak-free split, regions, features, then the selected methods on 2 x T4.
# Inputs: the code dataset (src/ + domains/bees/ prompts) and the output of the bees-audit kernel (images, gt.csv,
# CLIP features). Output: /kaggle/working/bees_results.tar.gz (results, splits, region boxes, logs).
import glob, os, shutil, subprocess, sys, time

W = "/kaggle/working/ONION"
src = glob.glob("/kaggle/input/**/20_prepare_bees.py", recursive=True)[0]
shutil.copytree(os.path.dirname(os.path.dirname(src)), W, dirs_exist_ok=True)
B = W + "/domains/bees"
os.makedirs(B + "/data", exist_ok=True)
found = glob.glob("/kaggle/input/**/bees/gt.csv", recursive=True)
if found and os.path.isdir(os.path.dirname(found[0]) + "/raw/train"):       # images saved by the bees-audit kernel
    os.symlink(os.path.dirname(found[0]) + "/raw", B + "/data/raw")
    shutil.copy(found[0], B + "/data/gt.csv")
else:                                                                         # otherwise download from Zenodo
    import urllib.request, zipfile
    os.makedirs(B + "/data/raw", exist_ok=True)
    for name in ("gt.csv", "val.zip", "test.zip", "train.zip"):
        dst = f"{B}/data/{name}"
        urllib.request.urlretrieve(f"https://zenodo.org/api/records/4085044/files/{name}/content", dst)
        if name.endswith(".zip"):
            zipfile.ZipFile(dst).extractall(B + "/data/raw")
            os.remove(dst)
print("images:", len(glob.glob(B + "/data/raw/*/videos/*/*.png")), flush=True)
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "open_clip_torch", "timm"], check=True)
ENV = {**os.environ, "DATASET": "bees", "PYTHONUNBUFFERED": "1", "ONION_WORKERS": "4"}
LOG = "/kaggle/working/logs"
os.makedirs(LOG, exist_ok=True)


def sh(cmd, gpu=0, log=None):
    t0 = time.time()
    with open(f"{LOG}/{log or cmd.split()[1]}.log", "a") as f:
        r = subprocess.run(cmd, shell=True, cwd=W + "/src", env={**ENV, "CUDA_VISIBLE_DEVICES": str(gpu)},
                           stdout=f, stderr=subprocess.STDOUT)
    print(f"[{time.time() - t0:6.0f}s] exit {r.returncode}: {cmd}", flush=True)
    return r.returncode


def bg(cmds, gpu, name):
    """run a list of commands one after another on one GPU, in the background."""
    script = " && ".join(f"python {c}" for c in cmds)
    return subprocess.Popen(f"({script}) > {LOG}/{name}.log 2>&1", shell=True, cwd=W + "/src",
                            env={**ENV, "CUDA_VISIBLE_DEVICES": str(gpu)})


# 1. split (needs only the CLIP features from the audit)
assert sh("python 20_prepare_bees.py") == 0
# 2. regions on GPU 1 while GPU 0 extracts the whole-image features
p_reg = bg(["05_detect_regions.py --bs 32"], 1, "regions")
assert sh("python 06_extract_features.py --backbone clip --mode global", log="feat_clip_global") == 0
for m in ("aug", "grid"):
    assert sh(f"python 06_extract_features.py --backbone clip --mode {m}", log=f"feat_clip_{m}") == 0
for m in ("global", "aug"):
    assert sh(f"python 06_extract_features.py --backbone dinov2 --mode {m}", log=f"feat_dinov2_{m}") == 0
assert sh("python 07_text_embeddings.py --backbone clip", log="text") == 0
assert p_reg.wait() == 0
sh("python 21_check_regions_bees.py", log="region_check")
assert sh("python 06_extract_features.py --backbone clip --mode regions", log="feat_clip_regions") == 0

# 3. methods. GPU 1: base paper (trains on images) then the fine-tuned CNN. GPU 0: everything on cached features.
p1 = bg(["09_base_paper.py --tag gpu1", "14_cnn_baseline.py"], 1, "gpu1_basepaper_cnn")
sh("python 08_baselines.py --methods zs lp tip tipf", log="baselines")
sh("python 08_baselines.py --methods zs tipf --text template", log="baselines_template")
sh("python 08_baselines.py --methods knn lp --leakage", log="leakage")
sh("python 18_equal_training.py --methods prga tipf cafo clipadapter taskres clap graphadapter", log="equal_training")
sh("python 11_prga.py --ablations --only PRGA-full abl-grid-nodes abl-no-text-nodes abl-template-text abl-M0-object-only",
   log="ablations")
print("gpu1 exit", p1.wait(), flush=True)

# 3b. BioCLIP (CLIP trained on the Tree of Life, insects included) as a second backbone for the main methods.
#     Which backbone to use is decided by validation macro-F1 (logged per run), never by the test set.
for m in ("global", "aug", "regions"):
    sh(f"python 06_extract_features.py --backbone bioclip --mode {m}", log=f"feat_bioclip_{m}")
sh("python 07_text_embeddings.py --backbone bioclip", log="text_bioclip")
sh("python 08_baselines.py --backbone bioclip --methods zs", log="baselines_bioclip")
sh("python 18_equal_training.py --backbone bioclip --methods prga tipf cafo", log="equal_training_bioclip")

# 4. pack the small outputs
for d in ("results", "splits", "regions", "data/clean"):
    if os.path.exists(f"{B}/{d}"):
        shutil.copytree(f"{B}/{d}", f"/kaggle/working/out/{d}", dirs_exist_ok=True,
                        ignore=shutil.ignore_patterns("*.npy"))
shutil.copytree(LOG, "/kaggle/working/out/logs", dirs_exist_ok=True)
subprocess.run("tar czf /kaggle/working/bees_results.tar.gz -C /kaggle/working out", shell=True)
for f in sorted(glob.glob(LOG + "/*.log")):
    print("=" * 20, f, flush=True)
    print(open(f).read()[-2500:], flush=True)
shutil.rmtree(W, ignore_errors=True)
