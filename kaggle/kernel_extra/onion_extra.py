# Kaggle entry point for the extra job (see run_extra.sh).
import glob, os, shutil, subprocess, sys
W = "/kaggle/working/ONION"
src = glob.glob("/kaggle/input/**/run_extra.sh", recursive=True)[0]
if not os.path.exists(W):
    shutil.copytree(os.path.dirname(src), W)
os.environ["ONION_RAW"] = os.path.dirname(sorted(glob.glob("/kaggle/input/**/1. Healthy", recursive=True), key=len)[0])
# third-party weights: Kaggle unpacks the dataset WITHOUT its top 'third_party/' folder -> locate by file name
scold = glob.glob("/kaggle/input/**/scold/scold.pth", recursive=True)
assert scold, "SCOLD weights not found"
shutil.copytree(os.path.dirname(scold[0]), W + "/third_party/scold", dirs_exist_ok=True)
for name in ("roberta-base", "dinov2-small"):
    d = glob.glob(f"/kaggle/input/**/models/{name}", recursive=True)
    if d:
        shutil.copytree(d[0], f"{W}/third_party/models/{name}", dirs_exist_ok=True)
        open(f"{W}/third_party/models/{name}/.complete", "w").close()
print("scold files:", os.listdir(W + "/third_party/scold"), flush=True)
assert os.path.exists(W + "/data/clean/meta_clean.csv"), "cleaned dataset missing"
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "open_clip_torch", "tabulate"], check=True)
r = subprocess.run(["bash", W + "/run_extra.sh"])
print("extra exit", r.returncode)
for f in glob.glob(W + "/features/*.pt"):                 # keep only the small rebuilt pool features as outputs
    if not f.endswith(("clip_aug.pt", "dinov2_aug.pt", "scold_aug.pt")):
        os.remove(f)
