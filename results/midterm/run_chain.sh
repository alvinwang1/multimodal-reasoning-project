#!/bin/bash
# Midterm overnight chain: Idea 1 (7 base configs x 3 orders on all 300 items; the 90 already done are
# skipped), then Idea 3 (budget grid x 3 orders on the 120 test items). Resumable: re-run after a stop.
cd "/Users/ameet/Desktop/Fall 2026/MMML/Project"
export PYTORCH_ENABLE_MPS_FALLBACK=1 HF_HUB_OFFLINE=1
FILTER='UNEXPECTED|^token2wav|^talker|Model config|^Key |^-+\+|^Notes:|can be ignored|LOAD REPORT|^\s*$'

echo "=== Idea 1 start $(date)"
.venv/bin/python -m src.analysis_a.run --split all --perm-configs all 2>&1 \
  | grep --line-buffered -vE "$FILTER" | tee results/midterm/run_idea1.log
echo "=== Idea 1 done $(date)"

BUD=""
for i in 0 0.1 0.25 0.5 1; do for a in 0 0.25 0.5 1; do
  case "$i/$a" in 0/0|1/1|1/0|0/1) continue;; esac   # = q_only, full, img_only, aud_only (already run)
  BUD="$BUD bud_i${i}_a${a}"
done; done
echo "=== Idea 3 start $(date):$BUD"
.venv/bin/python -m src.analysis_a.run --split test --configs $BUD --perm-configs all \
  --out results/midterm/raw_budget.jsonl 2>&1 \
  | grep --line-buffered -vE "$FILTER" | tee results/midterm/run_idea3.log
echo "=== Idea 3 done $(date)"
