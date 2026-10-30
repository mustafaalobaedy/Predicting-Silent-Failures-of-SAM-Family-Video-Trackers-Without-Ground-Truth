"""Collects every number used in the journal paper into journal_numbers.json.

The six SAM 2.1 settings (DAVIS, MOSE, SA-V x mask, click) are taken from
paper/paper_numbers.json (paper_numbers.py). Added
here: LVOS-v2, EdgeTAM and SAM 3 (statistics, frame-level AUROC tables written
by analyze.py, event-level bootstrap), zero-shot transfer, conformal risk
control, click scheduling (paired video bootstrap of the gains) and the VLM
gate. Run from the repository root after paper_numbers.py.
"""
import json
import os
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

EXP = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, EXP)
os.chdir(EXP)
import analyze  # noqa: E402
from event_eval import episodes  # noqa: E402

BUILD = os.path.dirname(os.path.abspath(__file__))
out = json.load(open(os.path.join(BUILD, "paper_numbers.json")))
FRAME_ROWS = ["pred_iou", "obj_score", "boundary_unc", "area_frac", "eprocess",
              "LR (all signals)", "GBDT (all signals)", "GBDT (+temporal)"]
DS = {"MOSE mask": "mose_mask", "MOSE click": "mose_click", "SA-V mask": "sav_mask", "SA-V click": "sav_click"}


def stats(path):
    d = pd.read_csv(path, dtype={"video": str}).dropna(subset=["pred_iou"])
    e = {"videos": d.video.nunique(), "objects": len(d[["video", "obj"]].drop_duplicates()),
         "frames": len(d), "mean_iou_visible": d[d.gt_present == 1].gt_iou.mean(),
         "absent_share": 1 - d.gt_present.mean(), "max_len": int(d.groupby("video").frame.max().max()) + 1}
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
            e["vis_nonempty_prediou"] = roc_auc_score(f[k], -d.pred_iou[k])
    return {k: float(v) for k, v in e.items()}


def frame_table(prefix):
    t = pd.read_csv(f"results/{prefix}_auroc_tau0.5.csv", index_col=0)
    return {r: float(t.loc[r, "AUROC"]) for r in FRAME_ROWS if r in t.index}


def events(tag, names):
    b = pd.read_csv(f"results/bootstrap_events_{tag}.csv")
    dfs = pd.read_csv(f"results/bootstrap_events_{tag}_diffs.csv")
    for x in (b, dfs):
        x["setting"] = x.file.map(lambda f: names[f])
    return b.to_dict("records"), dfs.to_dict("records")


def natural(tag):
    e = pd.read_csv(f"results/event_eval{tag}.csv")
    e = e[(e.detector == "eprocess (alpha=0.05)") & (e.tau == 0.5) & e.file.str.startswith("results/")]  # no DAVIS pilot
    return {"fa_min": float(e.fa_rate.min()), "fa_max": float(e.fa_rate.max()),
            "det_min": float(e.det_rate.min()), "det_max": float(e.det_rate.max())}


# ---- LVOS-v2 (SAM 2.1)
lv = {"LVOS mask": "lvos_mask", "LVOS click": "lvos_click"}
out["lvos"] = {"eda": {s: stats(f"results/{p}.csv") for s, p in lv.items()},
               "frame": {s: frame_table(p) for s, p in lv.items()},
               "natural": natural("_lvos")}
out["lvos"]["event"], out["lvos"]["event_diffs"] = events("lvos", {f"results/{p}.csv": s for s, p in lv.items()})
out["natural_sam2"] = natural("")

# ---- other trackers on the same MOSE / SA-V tracks
out["trackers"] = {}
for tr in ("edgetam", "sam3"):
    names = {f"results/{tr}_{p}.csv": s for s, p in DS.items()}
    ev, dif = events(tr, names)
    tf = pd.read_csv(f"results/transfer_{tr}.csv")
    tf["setting"] = tf.target.map(lambda f: names[f])
    out["trackers"][tr] = {"eda": {s: stats(f"results/{tr}_{p}.csv") for s, p in DS.items()},
                           "frame": {s: frame_table(f"{tr}_{p}") for s, p in DS.items()},
                           "event": ev, "event_diffs": dif, "transfer": tf.to_dict("records"),
                           "natural": natural(f"_{tr}")}
out["trackers"]["sam2"] = {"eda": {s: stats(f"results/{p}.csv") for s, p in DS.items()}}

# ---- conformal risk control
out["crc"] = pd.read_csv("results/conformal_alarm_summary.csv").to_dict("records")

# ---- click scheduling: mean J per round, paired video bootstrap of the gain at K = 3
clicks = {}
for ds in ("sav", "mose"):
    c = pd.concat([pd.read_csv(f"results/click_schedule_{ds}.csv"),
                   pd.read_csv(f"results/click_schedule_hybrid_{ds}.csv")])
    c = c.assign(video=c.video.astype(str))
    means = c.groupby(["scheduler", "round"]).mean_J.mean().unstack()
    base = c[c["round"] == 0].groupby(["video", "obj"]).mean_J.first()
    k3 = c[c["round"] == 3].set_index(["video", "obj", "scheduler"]).mean_J.unstack()
    gain = k3.sub(base, axis=0).groupby(level="video").mean()  # per-video mean gain per scheduler
    rng = np.random.default_rng(0)
    idx = rng.integers(0, len(gain), size=(2000, len(gain)))
    diffs = {}
    for a, b in [("gbdt", "pred_iou"), ("gbdt", "uniform"), ("oracle", "gbdt"), ("hybrid", "gbdt"),
                 ("hybrid", "uniform"), ("hybrid", "pred_iou"), ("hybrid", "oracle")]:
        dv = (gain[a] - gain[b]).values
        bs = dv[idx].mean(axis=1)
        diffs[f"{a}-{b}"] = {"diff": float(dv.mean()), "lo": float(np.quantile(bs, 0.025)),
                             "hi": float(np.quantile(bs, 0.975))}
    clicks[ds] = {"means": {s: {int(r): float(v) for r, v in row.items()} for s, row in means.iterrows()},
                  "tracks": int(len(base)), "videos": int(len(gain)), "diffs": diffs,
                  "gain_mean": {s: float(v) for s, v in gain.mean().items()}}
out["clicks"] = clicks

# ---- VLM gate
vs = pd.read_csv("results/vlm_gate_summary.csv")
vs = vs[vs.variant == "full"].drop_duplicates("setting")
out["vlm"] = {"summary": vs[["setting", "items", "videos", "fail_rate", "sec_per_call", "AUROC_vlm",
                             "AUROC_gbdt", "AUROC_pred_iou"]].to_dict("records"),
              "matched": pd.read_csv("results/vlm_gate_matched.csv").to_dict("records")}

json.dump(out, open(os.path.join(BUILD, "journal_numbers.json"), "w"), indent=1, default=float)
print(json.dumps({k: out[k] for k in ("clicks",)}, indent=1)[:3000])
for tr in ("sam2", "edgetam", "sam3"):
    print(tr, {s: round(v["mean_iou_visible"], 3) for s, v in out["trackers"][tr]["eda"].items()})
print(pd.DataFrame(out["vlm"]["summary"]).round(3).to_string())
