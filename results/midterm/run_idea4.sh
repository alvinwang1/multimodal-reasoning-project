#!/bin/bash
cd "$(dirname "$0")/../.."
export PYTORCH_ENABLE_MPS_FALLBACK=1
FILTER='UNEXPECTED|^token2wav|^talker|Model config|^Key |^-+\+|^Notes:|can be ignored|LOAD REPORT|^\s*$'
PY=.venv/bin/python
GEN="gen_cap_desc gen_cap_asr img_gendesc img_asr gencap_aud"

echo "=== B generate surrogates start $(date)"
$PY -m src.midterm.surrogates 2>&1 | grep --line-buffered -vE "$FILTER" | tee -a results/midterm/run_idea4_gen.log

echo "=== C score generated surrogates start $(date)"
$PY -m src.analysis_a.run --split test --configs $GEN --perm-configs all \
  --surrogates results/midterm/generated_surrogates.json --out results/midterm/raw_surrogate.jsonl 2>&1 \
  | grep --line-buffered -vE "$FILTER" | tee -a results/midterm/run_idea4_score.log

echo "=== C2 raw + gold baselines on test, same machine, start $(date)"
$PY -m src.analysis_a.run --split test --perm-configs all --out results/midterm/raw_base_test.jsonl 2>&1 \
  | grep --line-buffered -vE "$FILTER" | tee -a results/midterm/run_idea4_base.log

echo "=== Idea 4 done $(date)"
