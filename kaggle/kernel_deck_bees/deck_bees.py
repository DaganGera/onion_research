# Kaggle: bee-domain runs for the evaluation deck, all with CLIP ViT-L/14.
#   GPU 0: PRGA ablations (11_prga.py --ablations --backbone clip_l14), every K, 3 seeds
#   GPU 1: every method again (fair protocol, best-epoch) saving test probabilities, then the calibrated app model
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
LOG = "/kaggle/working/logs"
os.makedirs(LOG, exist_ok=True)


def run(cmds, gpu, name):
    script = " && ".join(f"python {c}" for c in cmds)
    env = {**os.environ, "DATASET": "bees", "PYTHONUNBUFFERED": "1", "CUDA_VISIBLE_DEVICES": str(gpu)}
    return subprocess.Popen(f"({script}) > {LOG}/{name}.log 2>&1", shell=True, cwd=W + "/src", env=env)


print("prepare", run(["20_prepare_bees.py"], 0, "prepare").wait(), flush=True)
p0 = run(["06_extract_features.py --backbone clip_l14 --mode global", "06_extract_features.py --backbone clip_l14 --mode aug",
          "06_extract_features.py --backbone clip_l14 --mode regions", "07_text_embeddings.py --backbone clip_l14"], 0, "feat_l14")
p1 = run(["06_extract_features.py --backbone dinov2 --mode global", "06_extract_features.py --backbone dinov2 --mode aug",
          "06_extract_features.py --backbone clip_l14 --mode grid"], 1, "feat_dino_grid")
print("features", p0.wait(), p1.wait(), flush=True)
p0 = run(["11_prga.py --ablations --backbone clip_l14"], 0, "ablations_l14")
p1 = run(["18_equal_training.py --backbone clip_l14 --variants best-epoch "
          "--methods prga tipf cafo clipadapter taskres clap graphadapter",
          "24_export_app_model.py --backbone clip_l14"], 1, "methods_export")
print("runs", p0.wait(), p1.wait(), flush=True)
B, out = W + "/domains/bees", "/kaggle/working/out"
for d in ("results", "splits", "checkpoints", "app_examples"):
    if os.path.isdir(f"{B}/{d}"):
        shutil.copytree(f"{B}/{d}", f"{out}/{d}", dirs_exist_ok=True)
shutil.copytree(LOG, out + "/logs", dirs_exist_ok=True)
subprocess.run("tar czf /kaggle/working/deck_bees.tar.gz -C /kaggle/working out", shell=True)
for f in sorted(glob.glob(LOG + "/*.log")):
    print("=" * 20, f, flush=True)
    print(open(f).read()[-2000:], flush=True)
shutil.rmtree(W, ignore_errors=True)
