#!/usr/bin/env bash
# Run the whole ST1 8-class baseline: image preprocessing, both training runs,
# then the summary. Safe to interrupt and rerun -- finished steps are skipped
# and an interrupted training run resumes from its last checkpoint.
#
# Usage (from baselines/st1/):
#   ./run_all.sh                      # text_only then multimodal, one GPU
#   ./run_all.sh text_only            # a single mode -- launch one per GPU to
#   ./run_all.sh multimodal           # run both at once, then ./run_all.sh summary
#   ./run_all.sh image_only           # optional extra run, not needed for the README
set -euo pipefail

RESULTS="results"
EPOCHS="${EPOCHS:-8}"
PATIENCE="${PATIENCE:-2}"
mkdir -p "$RESULTS"

train() {
  local mode="$1"
  if [ -f "$RESULTS/predictions_${mode}.csv" ]; then
    echo "== $mode: already done, skipping =="
    return
  fi
  if [ "$mode" != text_only ] && [ ! -f "$RESULTS/image_pixel_values.pt" ]; then
    echo "===== [$(date '+%Y-%m-%d %H:%M:%S')] preprocessing images ====="
    python3 precompute_image_pixels.py
  fi
  echo "===== [$(date '+%Y-%m-%d %H:%M:%S')] training $mode ====="
  time python3 finetune_st1.py --mode "$mode" \
    --num_train_epochs "$EPOCHS" --early_stopping_patience "$PATIENCE" \
    2>&1 | tee "$RESULTS/training_${mode}.log"
}

case "${1:-all}" in
  all)        train text_only; train multimodal; python3 summarize.py ;;
  summary)    python3 summarize.py ;;
  text_only|multimodal|image_only) train "$1" ;;
  *)          echo "unknown argument: $1"; exit 1 ;;
esac
