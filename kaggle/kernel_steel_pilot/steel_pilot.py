# Kaggle: pilot for the third domain, NEU steel surface defects (DATASET=steel). K = 4 and 16, seed 1, the main
# methods with the fair protocol, on CLIP B/16 and CLIP L/14 (backbone to be chosen on validation).
import glob, os, shutil, subprocess, sys

W = "/kaggle/working/ONION"
code = glob.glob("/kaggle/input/**/25_prepare_steel.py", recursive=True)[0]
shutil.copytree(os.path.dirname(os.path.dirname(code)), W, dirs_exist_ok=True)
imgs = [p for p in glob.glob("/kaggle/input/**/*", recursive=True) if p.lower().endswith((".jpg", ".bmp", ".png"))
        and "neu" in p.lower()]
roots = sorted({p.split("/")[3] for p in imgs})
raw = "/kaggle/input/" + roots[0]
print("NEU images:", len(imgs), "root", raw, "examples", imgs[:5], flush=True)
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "open_clip_torch"], check=True)
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
          "07_text_embeddings.py --backbone clip"], 0, "feat_b16")
p1 = run(["06_extract_features.py --backbone dinov2 --mode global", "06_extract_features.py --backbone dinov2 --mode aug",
          "06_extract_features.py --backbone clip_l14 --mode global", "06_extract_features.py --backbone clip_l14 --mode aug",
          "07_text_embeddings.py --backbone clip_l14"], 1, "feat_l14")
print("features", p0.wait(), p1.wait(), flush=True)
print("regions l14", run(["06_extract_features.py --backbone clip_l14 --mode regions"], 1, "feat_l14_regions").wait(), flush=True)
M = "--methods prga tipf cafo clipadapter taskres graphadapter --shots 4 16 --seeds 1 --variants best-epoch"
p0 = run([f"18_equal_training.py --backbone clip {M}", "08_baselines.py --backbone clip --methods zs --shots 4"], 0, "pilot_b16")
p1 = run([f"18_equal_training.py --backbone clip_l14 {M}", "08_baselines.py --backbone clip_l14 --methods zs --shots 4"], 1, "pilot_l14")
print("pilot", p0.wait(), p1.wait(), flush=True)
B = W + "/domains/steel"
out = "/kaggle/working/out"
for d in ("results", "splits", "regions", "data/clean"):
    if os.path.isdir(f"{B}/{d}"):
        shutil.copytree(f"{B}/{d}", f"{out}/{d}", dirs_exist_ok=True, ignore=shutil.ignore_patterns("*.npy"))
shutil.copytree(LOG, out + "/logs", dirs_exist_ok=True)
subprocess.run("tar czf /kaggle/working/steel_pilot.tar.gz -C /kaggle/working out", shell=True)
for f in sorted(glob.glob(LOG + "/*.log")):
    print("=" * 20, f, flush=True)
    print(open(f).read()[-2500:], flush=True)
shutil.rmtree(W, ignore_errors=True)
