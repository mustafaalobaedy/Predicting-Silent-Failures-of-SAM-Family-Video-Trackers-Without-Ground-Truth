#!/bin/bash
# LVOS v2 val: wait for the official download (dl/retry_val.sh), unzip, check,
# then extract mask and click signals without attention columns. Resumable.
# run inside the Python environment of README.md
cd "$(dirname "$0")"
D=${SAMFAIL_DATA:-data}/LVOS2
s=results/run_lvos_status.txt
echo "start $(date)" > $s
until grep -q "^done\|^gave up" $D/dl/retry_status.txt 2>/dev/null; do sleep 600; done
grep -q "^done" $D/dl/retry_status.txt || { echo "download gave up $(date)" >> $s; exit 1; }
if [ ! -f $D/valid/valid.txt ]; then
    z=$(ls $D/dl/*.zip | head -1)
    mkdir -p $D/unz && unzip -q -n "$z" -d $D/unz || { echo "unzip failed $(date)" >> $s; exit 1; }
    src=$(dirname "$(find $D/unz -maxdepth 4 -type d -name JPEGImages | head -1)")
    ln -sfn "$src" $D/valid
    echo "unzipped $(date) -> $src" >> $s
    python prep_lvos.py >> $s 2>&1 || { echo "prep failed $(date)" >> $s; exit 1; }
fi
for pr in mask click; do
    python extract_signals.py --dataset lvos --prompt $pr --no-attn \
        --out results/lvos_${pr}.csv > results/extract_log_lvos_${pr}.txt 2>&1
    echo "lvos_${pr} exit $? $(date)" >> $s
done
