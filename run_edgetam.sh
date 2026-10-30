#!/bin/bash
# EdgeTAM transfer (journal extension): same MOSE / SA-V videos and prompts as the
# main SAM 2.1 runs, without memory-attention columns. Resumable.
# run inside the Python environment of README.md
cd "$(dirname "$0")"
s=results/run_edgetam_status.txt
echo "start $(date)" >> $s
for ds in mose sav; do
  for pr in mask click; do
    python extract_signals.py --model edgetam --dataset $ds --prompt $pr --lazy --no-attn \
        --out results/edgetam_${ds}_${pr}.csv > results/extract_log_edgetam_${ds}_${pr}.txt 2>&1
    echo "edgetam_${ds}_${pr} exit $? $(date)" >> $s
  done
done
