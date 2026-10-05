# Kaggle entry point: copy the project out of the input datasets, install open_clip, run run_pipeline.sh.
import glob, os, shutil, subprocess, sys

W = "/kaggle/working/ONION"
src = glob.glob("/kaggle/input/**/run_pipeline.sh", recursive=True)[0]
if not os.path.exists(W):
    shutil.copytree(os.path.dirname(src), W)
tp = sorted(glob.glob("/kaggle/input/**/third_party/scold", recursive=True), key=len)
if tp and not os.path.exists(W + "/third_party"):
    shutil.copytree(os.path.dirname(tp[0]), W + "/third_party")      # SCOLD weights, RoBERTa config, DINOv2
assert os.path.exists(W + "/data/clean/meta_clean.csv"), "cleaned dataset missing"
healthy = sorted(glob.glob("/kaggle/input/**/1. Healthy", recursive=True), key=len)[0]
os.environ["ONION_RAW"] = os.path.dirname(healthy)                   # where the photos are on Kaggle
print("RAW =", os.environ["ONION_RAW"], flush=True)
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "open_clip_torch", "tabulate"], check=True)
subprocess.run("nvidia-smi", shell=True)

r = subprocess.run(["bash", W + "/run_pipeline.sh"])
print("pipeline exit", r.returncode)
shutil.rmtree(W + "/features", ignore_errors=True)       # features are inputs, not outputs: keep the output small
