#!/bin/bash
# SAM 3 transfer (journal extension). Waits until the gated facebook/sam3 checkpoint
# can be downloaded (checks every 30 min), runs a DAVIS sanity check, then extracts
# MOSE / SA-V (mask, click) with the SAM 2-style tracker of SAM 3 and runs the same
# analyses as for EdgeTAM. Resumable; status in results/run_sam3_status.txt.
# run inside the Python environment of README.md
cd "$(dirname "$0")"
s=results/run_sam3_status.txt
CK=${SAMFAIL_CKPT:-checkpoints}
echo "start $(date)" >> $s
until [ -f $CK/sam3.pt ]; do
  python -c "from huggingface_hub import hf_hub_download as d; d('facebook/sam3', 'sam3.pt', local_dir='$CK')" \
      > /dev/null 2>> results/run_sam3_download_err.txt && break
  echo "no access yet $(date)" >> $s
  sleep 1800
done
echo "checkpoint ok $(date)" >> $s
# sanity check: with the ground-truth mask prompt the tracker must follow DAVIS objects
python extract_signals.py --model sam3 --dataset davis --prompt mask --limit 2 --lazy --no-attn \
    --out results/sam3_davis_smoke.csv > results/extract_log_sam3_davis_smoke.txt 2>&1
J=$(python -c "import pandas as pd; d=pd.read_csv('results/sam3_davis_smoke.csv'); print(round(d.gt_iou[d.gt_present==1].mean(), 3))")
echo "davis smoke mean J $J $(date)" >> $s
python -c "import sys; sys.exit(0 if $J > 0.6 else 1)" || { echo "smoke check failed, stopping" >> $s; exit 1; }
for ds in mose sav; do
  for pr in mask click; do
    python extract_signals.py --model sam3 --dataset $ds --prompt $pr --lazy --no-attn \
        --out results/sam3_${ds}_${pr}.csv > results/extract_log_sam3_${ds}_${pr}.txt 2>&1
    echo "sam3_${ds}_${pr} exit $? $(date)" >> $s
  done
done
S=(results/sam3_mose_mask.csv results/sam3_mose_click.csv results/sam3_sav_mask.csv results/sam3_sav_click.csv)
python analyze.py "${S[@]}" > results/sam3_analyze.txt 2>&1; echo "analyze $? $(date)" >> $s
python event_eval.py "${S[@]}" --tag sam3 > results/event_eval_sam3_log.txt 2>&1; echo "event_eval $? $(date)" >> $s
python transfer_model.py results/mose_mask.csv:results/sam3_mose_mask.csv results/mose_click.csv:results/sam3_mose_click.csv \
    results/sav_mask.csv:results/sam3_sav_mask.csv results/sav_click.csv:results/sam3_sav_click.csv --tag sam3 \
    > results/transfer_sam3_log.txt 2>&1; echo "transfer $? $(date)" >> $s
python bootstrap_events.py "${S[@]}" --tag sam3 > results/bootstrap_events_sam3_log.txt 2>&1; echo "bootstrap $? $(date)" >> $s
echo "done $(date)" >> $s
