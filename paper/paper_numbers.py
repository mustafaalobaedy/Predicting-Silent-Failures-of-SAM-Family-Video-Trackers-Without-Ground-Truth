"""Collects the numbers of the DAVIS, MOSE and SA-V settings into paper/paper_numbers.json.

Recomputes the frame-level tables with the current analyze.py (regularized
GBDT), the dataset statistics (EDA) and the failure-type split, and reads the
event-level bootstrap results. Run from the repository root: python paper/paper_numbers.py <scratch dir>
"""
import json
import os
import shutil
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

EXP = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, EXP)
import analyze  # noqa: E402
from event_eval import episodes  # noqa: E402

SCRATCH = sys.argv[1]  # pilot CSVs are copied here so 01 Pilot stays untouched
SETTINGS = {"DAVIS mask": "results/davis_mask.csv",
            "DAVIS click": "results/davis_click.csv",
            "MOSE mask": "results/mose_mask.csv", "MOSE click": "results/mose_click.csv",
            "SA-V mask": "results/sav_mask.csv", "SA-V click": "results/sav_click.csv"}
FRAME_ROWS = ["pred_iou", "obj_score", "boundary_unc", "area_frac", "eprocess",
              "LR (all signals)", "GBDT (all signals)", "GBDT (+temporal)"]

out = {"eda": {}, "frame": {}, "frame_within": {}, "vis_nonempty_prediou": {}}
for name, rel in SETTINGS.items():
    src = os.path.join(EXP, rel)
    if rel.startswith("results/davis"):
        dst = os.path.join(SCRATCH, os.path.basename(src))
        shutil.copy(src, dst)
        src = dst
    d = pd.read_csv(src, dtype={"video": str}).dropna(subset=["pred_iou"])
    e = {"videos": d.video.nunique(), "objects": len(d[["video", "obj"]].drop_duplicates()),
         "frames": len(d), "mean_iou_visible": d[d.gt_present == 1].gt_iou.mean(),
         "absent_share": 1 - d.gt_present.mean()}
    for tau in (0.5, 0.7):
        f = analyze.add_labels(d, tau)
        e[f"fail_rate_{tau}"] = f.mean()
        g = d.assign(fail=f).sort_values(["video", "obj", "frame"]).reset_index(drop=True)
        g["track"] = pd.factorize(g.video + "_" + g.obj.astype(str))[0]
        e[f"episodes_{tau}"] = len(episodes(g)[1])
        if tau == 0.5:
            vis, emp = d.gt_present == 1, d.pred_present == 0
            n = f.sum()
            e["missed"] = (f & vis & emp).sum() / n
            e["bad_mask"] = (f & vis & ~emp).sum() / n
            e["ghost"] = (f & ~vis).sum() / n
            k = vis & ~emp
            out["vis_nonempty_prediou"][name] = roc_auc_score(f[k], -d.pred_iou[k])
    out["eda"][name] = {k: float(v) for k, v in e.items()}
    for tau in (0.5, 0.7):
        t = analyze.run(src, tau)
        out["frame"].setdefault(str(tau), {})[name] = {r: float(t.loc[r, "AUROC"]) for r in FRAME_ROWS}
        out["frame_within"].setdefault(str(tau), {})[name] = {
            r: float(t.loc[r, "AUROC_within"]) for r in FRAME_ROWS}

b = pd.read_csv(os.path.join(EXP, "results/bootstrap_events.csv"))
dfs = pd.read_csv(os.path.join(EXP, "results/bootstrap_events_diffs.csv"))
inv = {v: k for k, v in SETTINGS.items()}
inv.update({"../01 Pilot/results/signals_mask.csv": "DAVIS mask",  # file names used when the bootstrap was run
            "../01 Pilot/results/signals_click.csv": "DAVIS click"})
for x in (b, dfs):
    x["setting"] = x.file.map(lambda f: inv[f])
out["event"] = b.to_dict("records")
out["event_diffs"] = dfs.to_dict("records")
json.dump(out, open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "paper_numbers.json"), "w"), indent=1, default=float)
print(json.dumps({k: out[k] for k in ("eda", "vis_nonempty_prediou")}, indent=1, default=float))
print(pd.DataFrame(out["frame"]["0.5"]).round(3).to_string())
