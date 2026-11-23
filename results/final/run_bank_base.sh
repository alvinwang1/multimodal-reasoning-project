#!/bin/bash
# Final-report score bank, base part: the 7 base configs x 3 option orders on every item in
# data/final_folds.json, on the M2 Max (same machine and dtype as the midterm Idea 1 run).
# The 300 midterm items are already in results/analysis_a/raw.jsonl and are skipped; ~17.7k passes, ~8 h.
# Resumable: if it stops, run this script again.
cd "/Users/ameet/Desktop/Fall 2026/MMML/Project"
export PYTORCH_ENABLE_MPS_FALLBACK=1 HF_HUB_OFFLINE=1
FILTER='UNEXPECTED|^token2wav|^talker|Model config|^Key |^-+\+|^Notes:|can be ignored|LOAD REPORT|^\s*$'

cp results/analysis_a/raw.jsonl "results/analysis_a/raw.backup_$(date +%Y%m%d_%H%M).jsonl"  # gitignored; keep a copy
echo "=== base bank start $(date)"
.venv/bin/python -m src.analysis_a.run --split final --perm-configs all 2>&1 \
  | grep --line-buffered -vE "$FILTER" | tee -a results/final/run_bank_base.log
echo "=== base bank done $(date)"
.venv/bin/python -m src.final.bank
