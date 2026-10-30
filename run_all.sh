#!/bin/bash
# Runs all four main extraction settings in sequence (resumable).
# run inside the Python environment of README.md
cd "$(dirname "$0")"
for ds in mose sav; do
  for pr in mask click; do
    python extract_signals.py --dataset $ds --prompt $pr --out results/${ds}_${pr}.csv >> results/extract_log_${ds}_${pr}.txt 2>&1
    echo "$(date) finished ${ds}_${pr} (exit $?)" >> results/run_all_status.txt
  done
done
