#!/bin/bash
# CPU analyses of the EdgeTAM runs: frame-level, event-level, bootstrap, transfer from SAM 2.1.
# run inside the Python environment of README.md
cd "$(dirname "$0")"
s=results/run_edgetam_analysis_status.txt
E=(results/edgetam_mose_mask.csv results/edgetam_mose_click.csv results/edgetam_sav_mask.csv results/edgetam_sav_click.csv)
echo "start $(date)" >> $s
python analyze.py results/edgetam_sav_mask.csv results/edgetam_sav_click.csv > results/edgetam_sav_analyze.txt 2>&1; echo "analyze $? $(date)" >> $s
python event_eval.py "${E[@]}" --tag edgetam > results/event_eval_edgetam_log.txt 2>&1; echo "event_eval $? $(date)" >> $s
python transfer_model.py results/mose_mask.csv:results/edgetam_mose_mask.csv results/mose_click.csv:results/edgetam_mose_click.csv \
    results/sav_mask.csv:results/edgetam_sav_mask.csv results/sav_click.csv:results/edgetam_sav_click.csv --tag edgetam \
    > results/transfer_edgetam_log.txt 2>&1; echo "transfer $? $(date)" >> $s
python bootstrap_events.py "${E[@]}" --tag edgetam > results/bootstrap_events_edgetam_log.txt 2>&1; echo "bootstrap $? $(date)" >> $s
