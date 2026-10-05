#!/usr/bin/env bash
# Second Kaggle job (T4 x2): PRGA + DINOv2 cache (13_prga_dinov2.py), the PlantCaFo-style cache on the same
# features, and the SCOLD baseline (it failed in the first job because of a wrong weights path).
set -uo pipefail
cd "$(dirname "$0")/src"
export TQDM_DISABLE=1 PYTHONUNBUFFERED=1 ONION_WORKERS=${ONION_WORKERS:-4} ONION_THREADS=4
LOG=../results/logs; mkdir -p $LOG ../results/runs ../results/preds
step() { local n=$1; shift; echo "[$(date +%T)] START $n"; local t=$SECONDS
  if "$@" > $LOG/$n.log 2>&1; then echo "[$(date +%T)] OK    $n ($((SECONDS - t)) s)"
  else echo "[$(date +%T)] FAIL  $n ($((SECONDS - t)) s) -- tail:"; tail -8 $LOG/$n.log; fi; }
(
  export CUDA_VISIBLE_DEVICES=0
  step feat_clip_aug   python 06_extract_features.py --backbone clip --mode aug
  step feat_dinov2_aug python 06_extract_features.py --backbone dinov2 --mode aug
  step prga_improve    python 13_prga_dinov2.py
  step cafo_samefeats  python 08_baselines.py --methods cafo
) > $LOG/extra_lane0.txt 2>&1 &
P0=$!
(
  export CUDA_VISIBLE_DEVICES=1
  step feat_scold_aug  python 06_extract_features.py --backbone scold --mode aug
  step scold           python 08_baselines.py --backbone scold --methods zs tipf
) > $LOG/extra_lane1.txt 2>&1 &
P1=$!
wait $P0 $P1
cat $LOG/extra_lane0.txt $LOG/extra_lane1.txt
grep -h -A30 "mean macro-F1 on the held-out" $LOG/prga_improve.log
echo EXTRA_DONE
