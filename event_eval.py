"""Event-level comparison: failure-onset detection delay vs false alarms.

The e-process baseline (arXiv:2602.12983) is built to flag failure *onset*, so
frame-level AUROC undersells it. Here every object track is a stream:

- Failure episode: a run of >= MIN_LEN consecutive failing frames. Its onset
  is the first frame of the run. Failing runs shorter than MIN_LEN are
  "don't care" frames (no detection credit, no false alarm).
- Alarm: score >= threshold on a frame.
- Detected: the score is >= threshold somewhere in [onset - TOL, episode end].
  Delay = frames from onset to the first such frame (0 if already raised).
- False-alarm rate: share of eligible good frames (outside every
  [onset - TOL, end] window and not failing) on which the alarm is on.
  Counting alarm onsets instead would let an always-on detector pay only one
  false alarm per track while detecting every episode.

Thresholds are swept. Each detector is reported at matched false-alarm
budgets (2%, 5%, 10% of good frames), and the e-process also at its natural level
alpha = 0.05 (log wealth >= log 20).

The GBDT score is out-of-fold (GroupKFold by video) and causal: it uses only
current and past frames, so the non-causal frame_frac feature is dropped.

Two onset-aware GBDT variants (same causal features and config) change only
the training target, since the metric ignores frames deep inside an episode:
- "GBDT onset": positives are frames in [onset - TOL, onset + ONSET_K] of each
  episode, negatives are the frames that can raise false alarms; all other
  frames are left out of training.
- "GBDT episode-weighted": the frame-level failure label, with each failing
  frame weighted by 1 / (length of its failing run), so long failures do not
  dominate.

With memory-attention CSVs (*_attn.csv) a "GBDT causal +attn" detector is
added. --tag writes results/event_eval_<tag>{,_curves}.csv instead of
overwriting the main files.

Usage:
    python event_eval.py results/mose_mask.csv [more.csv ...] [--tag attn]
"""
import argparse

import numpy as np
import pandas as pd

from sklearn.model_selection import GroupKFold

from analyze import ATTN, BASE, SINGLE, add_labels, add_temporal, cv_scores, make_hgb

MIN_LEN, TOL = 3, 2
ONSET_K = 2  # frames after onset that still count as onset for "GBDT onset"
FA_BUDGETS = (0.02, 0.05, 0.10)  # share of eligible good frames with the alarm on
DETECTORS = ["eprocess", "pred_iou", "obj_score", "boundary_unc", "GBDT causal"]


def episodes(d):
    """Return (start, onset, end) row indices of failure episodes and a
    per-row mask of frames that may raise false alarms."""
    fail = d.fail.values
    track = d.track.values
    n = len(d)
    new_run = np.r_[True, (fail[1:] != fail[:-1]) | (track[1:] != track[:-1])]
    run_id = np.cumsum(new_run) - 1
    run_start = np.flatnonzero(new_run)
    run_end = np.r_[run_start[1:], n] - 1
    run_fail = fail[run_start] == 1
    run_len = run_end - run_start + 1
    ep = run_fail & (run_len >= MIN_LEN)
    onset, end = run_start[ep], run_end[ep]
    # window start: TOL frames before onset, but not before the track start
    track_first = pd.Series(np.arange(n)).groupby(track).transform("min").values
    start = np.maximum(onset - TOL, track_first[onset])
    can_fa = np.ones(n, bool)
    can_fa[np.isin(run_id, np.flatnonzero(run_fail))] = False  # any failing frame
    for s, e in zip(start, end):
        can_fa[s:e + 1] = False
    return start, onset, end, can_fa


def cv_scores_weighted(d, feats, y, w, n_splits=5):
    """Like analyze.cv_scores, but with a custom target and sample weights.

    Rows with weight 0 are left out of training; every test row is scored.
    Weights are rescaled to mean 1 within each class so the balanced class
    weights of make_hgb() still apply.
    """
    X, groups = d[feats].fillna(0).values, d.video.values
    w = w.astype(float).copy()
    for c in (0, 1):
        k = (y == c) & (w > 0)
        w[k] /= w[k].mean()
    oof = np.zeros(len(d))
    for tr, te in GroupKFold(n_splits).split(X, y, groups):
        tr = tr[w[tr] > 0]
        m = make_hgb()
        m.fit(X[tr], y[tr], sample_weight=w[tr])
        oof[te] = m.predict_proba(X[te])[:, 1]
    return oof


def onset_targets(d, ev):
    """Targets and weights of the two onset-aware variants (see module doc)."""
    start, onset, end, can_fa = ev
    n = len(d)
    y_on = np.zeros(n, int)
    for s, o, e in zip(start, onset, end):
        y_on[s:min(o + ONSET_K, e) + 1] = 1
    w_on = (y_on == 1) | can_fa
    fail, track = d.fail.values, d.track.values
    new_run = np.r_[True, (fail[1:] != fail[:-1]) | (track[1:] != track[:-1])]
    run_len = pd.Series(new_run.cumsum()).map(pd.Series(new_run.cumsum()).value_counts()).values
    w_ep = np.where(fail == 1, 1.0 / run_len, 1.0)
    return y_on, w_on, w_ep


def at_threshold(score, d, ev, thr):
    """Detection rate, false alarms per 100 frames and delays at one threshold."""
    start, onset, end, can_fa = ev
    above = score >= thr
    idx = np.flatnonzero(above)
    k = np.searchsorted(idx, start)
    hit_at = np.where(k < len(idx), idx[np.minimum(k, len(idx) - 1)], -1)
    det = (hit_at >= 0) & (hit_at <= end)
    delay = np.maximum(hit_at[det] - onset[det], 0)
    return {"thr": thr, "det_rate": det.mean(),
            "fa_rate": above[can_fa].mean(),
            "median_delay": np.median(delay) if det.any() else np.nan,
            "mean_delay": delay.mean() if det.any() else np.nan}


def sweep(score, d, ev, n_thr=300):
    thrs = np.unique(np.quantile(score, np.linspace(0.0, 0.9995, n_thr)))
    return pd.DataFrame([at_threshold(score, d, ev, t) for t in thrs])


def at_budget(curve, budget):
    ok = curve[curve.fa_rate <= budget]
    if ok.empty:
        return pd.Series({"det_rate": 0.0, "median_delay": np.nan, "mean_delay": np.nan})
    return ok.sort_values(["det_rate", "fa_rate"], ascending=[False, True]).iloc[0]


def build_scores(path, tau):
    """Load a CSV, label it and compute every detector's out-of-fold score.

    Returns the frame table (sorted by video, obj, frame), the episodes and
    a {detector: score array} dict. Shared with bootstrap_events.py.
    """
    d = pd.read_csv(path, dtype={"video": str}).dropna(subset=["pred_iou"])
    d["fail"] = add_labels(d, tau)
    d = add_temporal(d).reset_index(drop=True)  # sorted by video, obj, frame
    d["track"] = pd.factorize(d.video + "_" + d.obj.astype(str))[0]
    causal = [c for c in d.columns if c.endswith(("_mean5", "_d1"))] + ["pred_iou_min_sofar"]
    scores = {n: SINGLE[n](d).fillna(0).values for n in DETECTORS[:-1]}
    scores["GBDT causal"] = cv_scores(d, BASE + causal, make_hgb)
    if "attn0_ent" in d:  # the attention deltas are causal too
        scores["GBDT causal +attn"] = cv_scores(d, BASE + causal + ATTN, make_hgb)
    ev = episodes(d)
    y_on, w_on, w_ep = onset_targets(d, ev)
    feats = BASE + causal
    scores["GBDT onset"] = cv_scores_weighted(d, feats, y_on, w_on)
    scores["GBDT episode-weighted"] = cv_scores_weighted(d, feats, d.fail.values, w_ep)
    return d, ev, scores


def run(path, tau):
    d, ev, scores = build_scores(path, tau)
    n_ep = len(ev[1])
    print(f"\n=== {path} (tau={tau}) ===  tracks {d.track.nunique()}, frames {len(d)}, "
          f"episodes {n_ep} (min length {MIN_LEN})", flush=True)
    out, curves = [], []
    for name, s in scores.items():
        c = sweep(s, d, ev)
        curves.append(c.assign(detector=name, file=path, tau=tau))
        row = {"file": path, "tau": tau, "detector": name, "episodes": n_ep}
        for b in FA_BUDGETS:
            r = at_budget(c, b)
            row[f"det@FA{b}"] = r.det_rate
            row[f"delay@FA{b}"] = r.median_delay
        out.append(row)
    # e-process at its natural alpha = 0.05 level
    nat = at_threshold(scores["eprocess"], d, ev, np.log(20))
    out.append({"file": path, "tau": tau, "detector": "eprocess (alpha=0.05)",
                "episodes": n_ep, "det_rate": nat["det_rate"],
                "fa_rate": nat["fa_rate"], "median_delay": nat["median_delay"]})
    t = pd.DataFrame(out)
    print(t.drop(columns=["file", "tau", "episodes"]).round(3).to_string(index=False))
    return t, pd.concat(curves)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("csv", nargs="+")
    ap.add_argument("--tag", default="")
    args = ap.parse_args()
    base = "results/event_eval" + (f"_{args.tag}" if args.tag else "")
    res, cur = zip(*[run(p, tau) for p in args.csv for tau in (0.5, 0.7)])
    pd.concat(res).to_csv(f"{base}.csv", index=False)
    pd.concat(cur).to_csv(f"{base}_curves.csv", index=False)
