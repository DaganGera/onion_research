#!/usr/bin/env bash
# Extra Kaggle job (T4 x2): SCOLD baseline (failed in the main run: weights path), DINOv2 features of the new pool,
# and the validation-only PRGA improvement test (07c) + PlantCaFo-lite re-run on the same features.
set -uo pipefail
cd "$(dirname "$0")/src"
export TQDM_DISABLE=1 PYTHONUNBUFFERED=1 ONION_WORKERS=${ONION_WORKERS:-4} ONION_THREADS=4
LOG=../results/logs; mkdir -p $LOG ../results/runs ../results/preds
step() { local n=$1; shift; echo "[$(date +%T)] START $n"; local t=$SECONDS
  if "$@" > $LOG/$n.log 2>&1; then echo "[$(date +%T)] OK    $n ($((SECONDS - t)) s)"
  else echo "[$(date +%T)] FAIL  $n ($((SECONDS - t)) s) -- tail:"; tail -8 $LOG/$n.log; fi; }
(
  export CUDA_VISIBLE_DEVICES=0
  step feat_clip_aug   python 04_features.py --backbone clip --mode aug
  step feat_dinov2_aug python 04_features.py --backbone dinov2 --mode aug
  step prga_improve    python 07c_prga_improve.py
  step cafo_samefeats  python 06_baselines.py --methods cafo
) > $LOG/extra_lane0.txt 2>&1 &
P0=$!
(
  export CUDA_VISIBLE_DEVICES=1
  step feat_scold_aug  python 04_features.py --backbone scold --mode aug
  step scold           python 06_baselines.py --backbone scold --methods zs tipf
) > $LOG/extra_lane1.txt 2>&1 &
P1=$!
wait $P0 $P1
cat $LOG/extra_lane0.txt $LOG/extra_lane1.txt
grep -h -A30 "mean macro-F1 on the held-out" $LOG/prga_improve.log
echo EXTRA_DONE
