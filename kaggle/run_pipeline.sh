#!/usr/bin/env bash
# Whole project except XAI and VLM, on a Kaggle "GPU T4 x2" machine. Two lanes run in parallel, one per GPU:
#   lane 0 (GPU 0): exact base paper model (06d), seeds 1 and 2   -- the expensive part (fresh 336px windows every step)
#   lane 1 (GPU 1): exact base paper model seed 3, then every other method in dependency order
# Every step is resumable (finished K x seed runs are skipped), so a re-run after a timeout continues where it stopped.
set -uo pipefail
cd "$(dirname "$0")/src"
export TQDM_DISABLE=1 PYTHONUNBUFFERED=1 ONION_WORKERS=${ONION_WORKERS:-4} ONION_THREADS=4
LOG=../results/logs; mkdir -p $LOG ../results/runs ../results/preds ../checkpoints ../figures

step() {  # name, command...   (a failed step is logged and the lane continues: one broken baseline must not stop the rest)
  local n=$1; shift
  echo "[$(date +%T)] START $n"; local t=$SECONDS
  if "$@" > $LOG/$n.log 2>&1; then echo "[$(date +%T)] OK    $n ($((SECONDS - t)) s)"
  else echo "[$(date +%T)] FAIL  $n ($((SECONDS - t)) s) -- tail:"; tail -5 $LOG/$n.log; fi
}

lane0() {
  export CUDA_VISIBLE_DEVICES=0
  step basepaper_exact_s12 python 06d_basepaper_exact.py --seeds 1 2 --tag gpu0
}

lane1() {
  export CUDA_VISIBLE_DEVICES=1
  step basepaper_exact_s3  python 06d_basepaper_exact.py --seeds 3 --tag gpu1
  step baselines_clip      python 06_baselines.py --methods zs lp tip tipf
  step baselines_template  python 06_baselines.py --methods zs tipf --text template
  step leakage             python 06_baselines.py --methods lp knn --leakage
  step sota                python 06e_sota.py
  step paper_ctrl_grid     python 06b_basepaper.py
  step paper_ctrl_regions  python 06c_paper_graph_regions.py
  step cafo                python 06_baselines.py --methods cafo
  step bioclip             python 06_baselines.py --backbone bioclip --methods zs tipf
  step scold               python 06_baselines.py --backbone scold --methods zs tipf
  step prga_v0             python 07_prga.py --v0
  step prga_ablations      python 07_prga.py --ablations --shots 1 4
  step prga_select         python 07b_prga_select.py
  step prga_select_dino    python 07b_prga_select.py --dino
  step prga_final          python 07_prga.py
  step cnn                 python 08_cnn_baseline.py
}

features() {  # the cleaned split has a new training pool: rebuild every feature that exists only for pool photos
  export CUDA_VISIBLE_DEVICES=$1; shift
  for s in "$@"; do step feat_$(echo $s | tr ' -' '__') python $s; done
}

nvidia-smi --query-gpu=index,name,memory.total --format=csv
# stage A: features of the new pool (two GPUs). clip aug must exist before anything else uses the cache keys.
features 0 "04_features.py --backbone clip --mode aug" "04b_paper_patches.py" > $LOG/featA0.txt 2>&1 &
F0=$!
features 1 "04_features.py --backbone dinov2 --mode aug" "04_features.py --backbone bioclip --mode aug" \
           "04_features.py --backbone scold --mode aug" > $LOG/featA1.txt 2>&1 &
F1=$!
wait $F0
features 0 "04c_region_views.py" >> $LOG/featA0.txt 2>&1
wait $F1
cat $LOG/featA0.txt $LOG/featA1.txt
lane0 > $LOG/lane0.txt 2>&1 &
P0=$!
lane1 > $LOG/lane1.txt 2>&1 &
P1=$!
while kill -0 $P0 2>/dev/null || kill -0 $P1 2>/dev/null; do sleep 300; echo "--- $(date +%T)"; tail -n 3 $LOG/lane0.txt $LOG/lane1.txt; done
wait
cat $LOG/lane0.txt $LOG/lane1.txt
step eval python 09_eval.py
echo PIPELINE_DONE
