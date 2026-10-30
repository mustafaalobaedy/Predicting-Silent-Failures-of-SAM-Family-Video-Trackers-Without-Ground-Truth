#!/bin/bash
# Re-extracts all settings with the memory-attention signals (resumable).
# Outputs results/<dataset>_<prompt>_attn.csv; the original columns are identical
# to the earlier CSVs. DAVIS runs first so early results are available quickly.
# run inside the Python environment of README.md
cd "$(dirname "$0")"
for ds in davis mose sav; do
  for pr in mask click; do
    python extract_signals.py --dataset $ds --prompt $pr --out results/${ds}_${pr}_attn.csv >> results/extract_log_${ds}_${pr}_attn.txt 2>&1
    echo "$(date) finished ${ds}_${pr}_attn (exit $?)" >> results/run_attn_status.txt
  done
done
