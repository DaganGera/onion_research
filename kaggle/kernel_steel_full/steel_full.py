# Kaggle: full run for the steel domain (NEU), after the pilot. Every K, 3 seeds, fair protocol, both CLIP backbones
# (CLIP L/14 chosen on pilot validation), ablations, base paper, CNN, and the calibrated web-app model.
import glob, os, shutil, subprocess, sys

W = "/kaggle/working/ONION"
code = glob.glob("/kaggle/input/**/25_prepare_steel.py", recursive=True)[0]
shutil.copytree(os.path.dirname(os.path.dirname(code)), W, dirs_exist_ok=True)
imgs = [p for p in glob.glob("/kaggle/input/**/*", recursive=True) if p.lower().endswith((".jpg", ".bmp", ".png")) and "neu" in p.lower()]
raw = "/kaggle/input/" + sorted({p.split("/")[3] for p in imgs})[0]
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "open_clip_torch", "timm"], check=True)
env = {**os.environ, "DATASET": "steel", "ONION_RAW": raw, "PYTHONUNBUFFERED": "1"}
LOG = "/kaggle/working/logs"
os.makedirs(LOG, exist_ok=True)


def run(cmds, gpu, name):
    script = " && ".join(f"python {c}" for c in cmds)
    return subprocess.Popen(f"({script}) > {LOG}/{name}.log 2>&1", shell=True, cwd=W + "/src",
                            env={**env, "CUDA_VISIBLE_DEVICES": str(gpu)})


assert run(["25_prepare_steel.py"], 0, "prepare").wait() == 0
p0 = run(["05_detect_regions.py --bs 32", "06_extract_features.py --backbone clip --mode global",
          "06_extract_features.py --backbone clip --mode aug", "06_extract_features.py --backbone clip --mode regions",
          "06_extract_features.py --backbone clip --mode grid", "07_text_embeddings.py --backbone clip"], 0, "feat_b16")
p1 = run(["06_extract_features.py --backbone dinov2 --mode global", "06_extract_features.py --backbone dinov2 --mode aug",
          "06_extract_features.py --backbone clip_l14 --mode global", "06_extract_features.py --backbone clip_l14 --mode aug",
          "06_extract_features.py --backbone clip_l14 --mode grid", "07_text_embeddings.py --backbone clip_l14"], 1, "feat_l14")
print("features", p0.wait(), p1.wait(), flush=True)
print("regions l14", run(["06_extract_features.py --backbone clip_l14 --mode regions"], 1, "feat_l14_regions").wait(), flush=True)
M = "--methods prga tipf cafo clipadapter taskres clap graphadapter --variants best-epoch"
p0 = run([f"18_equal_training.py --backbone clip_l14 {M}", "08_baselines.py --backbone clip_l14 --methods zs lp",
          "11_prga.py --ablations --backbone clip_l14", "24_export_app_model.py --backbone clip_l14"], 0, "gpu0")
p1 = run([f"18_equal_training.py --backbone clip {M}", "08_baselines.py --backbone clip --methods zs lp",
          "09_base_paper.py --tag gpu1", "14_cnn_baseline.py"], 1, "gpu1")
print("runs", p0.wait(), p1.wait(), flush=True)
B, out = W + "/domains/steel", "/kaggle/working/out"
for d in ("results", "splits", "regions", "data/clean", "checkpoints", "app_examples"):
    if os.path.isdir(f"{B}/{d}"):
        shutil.copytree(f"{B}/{d}", f"{out}/{d}", dirs_exist_ok=True)
shutil.copytree(LOG, out + "/logs", dirs_exist_ok=True)
subprocess.run("tar czf /kaggle/working/steel_full.tar.gz -C /kaggle/working out", shell=True)
for f in sorted(glob.glob(LOG + "/*.log")):
    print("=" * 20, f, flush=True)
    print(open(f).read()[-2000:], flush=True)
shutil.rmtree(W, ignore_errors=True)
