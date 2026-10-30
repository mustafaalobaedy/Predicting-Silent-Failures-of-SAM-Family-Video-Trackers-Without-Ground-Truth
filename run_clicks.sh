#!/bin/bash
# Click-scheduling simulation on 30 videos of each dataset (click prompt, K = 3).
# The GBDT scheduler is trained on the other dataset. Resumable.
# run inside the Python environment of README.md
cd "$(dirname "$0")"
s=results/click_schedule_status.txt
echo "start $(date)" > $s
python click_schedule.py --dataset sav --train results/mose_click.csv \
    --out results/click_schedule_sav.csv --videos 30 --K 3 2>&1 \
    | grep --line-buffered -v "propagate in video\|post-processing\|fill_holes\|UserWarning\|^$" > results/click_schedule_log_sav.txt
echo "sav exit ${PIPESTATUS[0]} $(date)" >> $s
python click_schedule.py --dataset mose --train results/sav_click.csv \
    --out results/click_schedule_mose.csv --videos 30 --K 3 2>&1 \
    | grep --line-buffered -v "propagate in video\|post-processing\|fill_holes\|UserWarning\|^$" > results/click_schedule_log_mose.txt
echo "mose exit ${PIPESTATUS[0]} $(date)" >> $s
