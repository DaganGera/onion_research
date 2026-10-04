# Kaggle entry point: copies the project from the input datasets, installs open_clip, runs the pipeline.
# MODE = "probe" (10-min end-to-end check of every new script) or "full" (whole project except XAI/VLM).
import glob, os, shutil, subprocess, sys
MODE = "full"
W = "/kaggle/working/ONION"
src = [p for p in glob.glob("/kaggle/input/**/run_pipeline.sh", recursive=True)][0]
if not os.path.exists(W):
    shutil.copytree(os.path.dirname(src), W)
tp = sorted(glob.glob("/kaggle/input/**/third_party/scold", recursive=True), key=len)
if tp and not os.path.exists(W + "/third_party"):
    shutil.copytree(os.path.dirname(tp[0]), W + "/third_party")      # SCOLD weights, RoBERTa config, DINOv2
# guard: refuse to run stale code (a Kaggle dataset version can still be processing when the kernel starts)
_code = open(W + "/src/06d_basepaper_exact.py").read()
assert "BasePaperExact-noGraph" in _code and "TRAIN_AB" in _code, "STALE CODE MOUNTED - aborting"
assert os.path.exists(W + "/data/clean/meta_clean.csv"), "CLEAN DATASET MISSING - aborting"
print("code version check OK", flush=True)
healthy = sorted(glob.glob("/kaggle/input/**/1. Healthy", recursive=True), key=len)[0]
os.environ["ONION_RAW"] = os.path.dirname(healthy)
print("RAW =", os.environ["ONION_RAW"], flush=True)
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "open_clip_torch", "tabulate"], check=True)
subprocess.run("nvidia-smi; python -c 'import torch,open_clip,timm,transformers,pandas,sklearn;"
               "print(torch.__version__,open_clip.__version__,timm.__version__,transformers.__version__,pandas.__version__,sklearn.__version__)'",
               shell=True)

def sh(cmd):
    print(f"\n$ {cmd}", flush=True)
    r = subprocess.run(cmd, shell=True, cwd=W + "/src")
    print("exit", r.returncode, flush=True)

if MODE == "probe":
    os.environ["TQDM_DISABLE"] = "1"
    sh("CUDA_VISIBLE_DEVICES=0 python 06d_basepaper_exact.py --shots 1 --seeds 1 --epochs 2 --tag probe0 & "
       "CUDA_VISIBLE_DEVICES=1 python 06d_basepaper_exact.py --shots 16 --seeds 1 --epochs 2 --tag probe1; wait")
    sh("python - <<'P'\nimport time,torch,pandas as pd,sys\nsys.argv=['x']\nimport importlib\nm=importlib.import_module('06d_basepaper_exact')\n"
       "enc=m.WindowEncoder('cuda');x=torch.randint(0,255,(48,3,336,336),dtype=torch.uint8)\nenc(x[:8]);torch.cuda.synchronize();t=time.time();enc(x);torch.cuda.synchronize()\n"
       "dt=time.time()-t;print(f'encode 48 images x {len(m.ALL_WINDOWS)} windows: {dt:.2f}s -> {48/dt:.1f} img/s, {48*len(m.ALL_WINDOWS)/dt:.0f} crops/s')\nP")
    sh("python 06e_sota.py --methods coop clap --shots 1")
    sh("python 06_baselines.py --methods tipf --shots 1")
    sh("python 07_prga.py --v0 --shots 1")
    sh("python - <<'P'\nimport timm;m=timm.create_model('efficientnet_b0',pretrained=True);print('timm weights ok')\nP")
    sh("ls ../results/runs; cat ../results/runs/*probe*.jsonl | head -20")
else:
    r = subprocess.run(["bash", W + "/run_pipeline.sh"])
    print("pipeline exit", r.returncode)
shutil.rmtree(W + "/features", ignore_errors=True)       # inputs, not outputs: keep the output small
