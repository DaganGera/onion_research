# Kaggle entry point: finish the CoOp "all photos" runs of 18_equal_training.py on the two T4 GPUs.
import glob, os, shutil, subprocess, sys

W = "/kaggle/working/ONION"
src = glob.glob("/kaggle/input/**/18_equal_training.py", recursive=True)[0]
root = os.path.dirname(os.path.dirname(src))                        # folder that holds src/, features/, splits/ ...
shutil.copytree(root, W, dirs_exist_ok=True)
print("copied from", root, os.listdir(W), flush=True)
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "open_clip_torch"], check=True)
subprocess.run("nvidia-smi", shell=True)

# one seed per GPU; each process appends finished runs to results/equal_training/coop.jsonl (already-done runs skipped)
cmd = "python 18_equal_training.py --methods coop --shots full --seeds {s}"
procs = [subprocess.Popen(cmd.format(s=s), shell=True, cwd=W + "/src",
                          env={**os.environ, "CUDA_VISIBLE_DEVICES": str(gpu), "PYTHONUNBUFFERED": "1"},
                          stdout=open(f"/kaggle/working/coop_seed{s}.log", "w"), stderr=subprocess.STDOUT)
         for gpu, s in ((0, 2), (1, 3))]
for p in procs:
    p.wait()
    print("exit", p.returncode, flush=True)
for s in (2, 3):
    print(open(f"/kaggle/working/coop_seed{s}.log").read()[-1500:], flush=True)
shutil.copy(W + "/results/equal_training/coop.jsonl", "/kaggle/working/coop.jsonl")
shutil.rmtree(W, ignore_errors=True)                                  # keep only the results as output
