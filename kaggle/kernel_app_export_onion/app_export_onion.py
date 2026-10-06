# Kaggle: build the onion web-app checkpoints (src/24_export_app_model.py) from the current onion features
# (onion-app-features; the copy inside onion-fewshot-work is from an older training pool).
import glob, os, shutil, subprocess

W = "/kaggle/working/ONION"
code = glob.glob("/kaggle/input/**/24_export_app_model.py", recursive=True)[0]
shutil.copytree(os.path.dirname(os.path.dirname(code)), W, dirs_exist_ok=True)
work = os.path.dirname(glob.glob("/kaggle/input/**/run_pipeline.sh", recursive=True)[0])
for d in ("splits", "data", "prompts", "regions"):
    shutil.copytree(f"{work}/{d}", f"{W}/{d}", dirs_exist_ok=True)
feats = [os.path.dirname(f) for f in glob.glob("/kaggle/input/**/clip_aug.pt", recursive=True) if "onion-app-features" in f][0]
os.symlink(feats, f"{W}/features")
raw = os.path.dirname(sorted(glob.glob("/kaggle/input/**/1. Healthy", recursive=True), key=len)[0])
r = subprocess.run("python 24_export_app_model.py --backbone clip", shell=True, cwd=W + "/src",
                   env={**os.environ, "DATASET": "onion", "ONION_RAW": raw, "PYTHONUNBUFFERED": "1"})
print("exit", r.returncode, flush=True)
out = "/kaggle/working/out"
os.makedirs(out + "/checkpoints", exist_ok=True)
for f in glob.glob(W + "/checkpoints/app_*.pt"):
    shutil.copy(f, out + "/checkpoints/")
if os.path.isdir(W + "/app_examples"):
    shutil.copytree(W + "/app_examples", out + "/app_examples/onion")
subprocess.run("tar czf /kaggle/working/app_export_onion.tar.gz -C /kaggle/working out", shell=True)
shutil.rmtree(W, ignore_errors=True)
