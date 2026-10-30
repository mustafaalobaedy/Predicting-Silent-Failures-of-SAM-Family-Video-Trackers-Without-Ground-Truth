#!/bin/bash
# VLM-verifier gate (journal extension): waits for the EdgeTAM run to free the GPU,
# renders 25 random videos per setting, scores them with Qwen2.5-VL-3B. Resumable.
# run inside the Python environment of README.md
cd "$(dirname "$0")"
s=results/run_vlm_gate_status.txt
echo "start $(date)" >> $s
while pgrep -f run_edgetam.sh > /dev/null; do sleep 60; done
for ds in mose sav; do
  for pr in mask click; do
    python vlm_gate.py masks --dataset $ds --prompt $pr > results/vlm_gate_log_masks_${ds}_${pr}.txt 2>&1
    echo "masks ${ds}_${pr} exit $? $(date)" >> $s
  done
done
for ds in mose sav; do
  for pr in mask click; do
    python vlm_gate.py vlm --dataset $ds --prompt $pr > results/vlm_gate_log_vlm_${ds}_${pr}.txt 2>&1
    echo "vlm ${ds}_${pr} exit $? $(date)" >> $s
  done
done
