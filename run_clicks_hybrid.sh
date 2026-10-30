#!/bin/bash
# Hybrid click scheduler on the same 30 + 30 videos as run_clicks.sh (K = 3). Resumable.
# run inside the Python environment of README.md
cd "$(dirname "$0")"
s=results/click_schedule_hybrid_status.txt
echo "start $(date)" > $s
for pair in "sav mose" "mose sav"; do
    set -- $pair
    python click_schedule.py --dataset $1 --train results/$2_click.csv \
        --out results/click_schedule_hybrid_$1.csv --videos 30 --K 3 --schedulers hybrid 2>&1 \
        | grep --line-buffered -v "propagate in video\|post-processing\|fill_holes\|UserWarning\|^$" \
        > results/click_schedule_hybrid_log_$1.txt
    echo "$1 exit ${PIPESTATUS[0]} $(date)" >> $s
done
