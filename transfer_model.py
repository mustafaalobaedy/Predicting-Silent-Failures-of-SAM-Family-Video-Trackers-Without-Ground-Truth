"""Zero-shot transfer of the failure predictor from SAM 2.1 to another tracker.

The GBDTs are trained on SAM 2.1 signals (source CSV) and scored on the
signals of another tracker (target CSV, e.g. EdgeTAM) on the same dataset and
prompt. Folds are grouped by video over the union of both CSVs, so a video is
never used for training (with SAM 2.1) and testing (with the target) at once.
The target model's own labels are never used for training.

Reported per (pair, tau): frame-level AUROC of the causal GBDT, and event-level
detection at 2/5/10% false-alarm budgets for the causal GBDT and the
episode-weighted GBDT, next to the target model's pred_iou and the
within-target cross-validated GBDTs (from event_eval.build_scores).

Usage:
    python transfer_model.py results/mose_mask.csv:results/edgetam_mose_mask.csv [...] --tag edgetam
    -> results/transfer_<tag>.csv
"""
import argparse

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold

from analyze import BASE, add_labels, add_temporal, make_hgb
from event_eval import FA_BUDGETS, at_budget, build_scores, episodes, onset_targets, sweep


def prep(path, tau):
    d = pd.read_csv(path, dtype={"video": str}).dropna(subset=["pred_iou"])
    d["fail"] = add_labels(d, tau)
    d = add_temporal(d).reset_index(drop=True)
    d["track"] = pd.factorize(d.video + "_" + d.obj.astype(str))[0]
    return d


def fit(X, y, w=None):
    if w is not None:
        w = w.astype(float).copy()
        for c in (0, 1):  # mean 1 per class, as in event_eval.cv_scores_weighted
            w[y == c] /= w[y == c].mean()
    m = make_hgb()
    m.fit(X, y, sample_weight=w)
    return m


def transfer_scores(src, tgt, feats, n_splits=5):
    """Out-of-fold target scores from models trained on source videos only."""
    _, _, w_src = onset_targets(src, episodes(src))
    videos = np.array(sorted(set(src.video) | set(tgt.video)))
    out = {"GBDT causal (transfer)": np.zeros(len(tgt)),
           "GBDT episode-weighted (transfer)": np.zeros(len(tgt))}
    Xs, ys = src[feats].fillna(0).values, src.fail.values
    Xt = tgt[feats].fillna(0).values
    for _, te_v in GroupKFold(n_splits).split(videos, groups=videos):
        test = set(videos[te_v])
        tr = ~src.video.isin(test).values
        te = tgt.video.isin(test).values
        if not te.any():
            continue
        out["GBDT causal (transfer)"][te] = fit(Xs[tr], ys[tr]).predict_proba(Xt[te])[:, 1]
        out["GBDT episode-weighted (transfer)"][te] = \
            fit(Xs[tr], ys[tr], w_src[tr]).predict_proba(Xt[te])[:, 1]
    return out


def run(src_path, tgt_path, tau):
    src, tgt = prep(src_path, tau), prep(tgt_path, tau)
    feats = BASE + [c for c in tgt.columns if c.endswith(("_mean5", "_d1"))] + ["pred_iou_min_sofar"]
    d, ev, own = build_scores(tgt_path, tau)  # same row order as prep(tgt_path)
    assert (d.video.values == tgt.video.values).all() and (d.frame.values == tgt.frame.values).all()
    scores = {"pred_iou": own["pred_iou"], "eprocess": own["eprocess"],
              "GBDT causal (within)": own["GBDT causal"],
              "GBDT episode-weighted (within)": own["GBDT episode-weighted"],
              **transfer_scores(src, tgt, feats)}
    rows = []
    for name, s in scores.items():
        c = sweep(s, d, ev)
        r = {"source": src_path, "target": tgt_path, "tau": tau, "detector": name,
             "episodes": len(ev[1]), "AUROC": roc_auc_score(d.fail, s)}
        for b in FA_BUDGETS:
            r[f"det@FA{b}"] = at_budget(c, b).det_rate
        rows.append(r)
    t = pd.DataFrame(rows)
    print(f"\n=== {src_path} -> {tgt_path} (tau={tau}) ===", flush=True)
    print(t.drop(columns=["source", "target", "tau"]).round(3).to_string(index=False), flush=True)
    return t


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("pairs", nargs="+", help="source.csv:target.csv")
    ap.add_argument("--tag", default="edgetam")
    args = ap.parse_args()
    res = [run(*p.split(":"), tau) for p in args.pairs for tau in (0.5, 0.7)]
    pd.concat(res).to_csv(f"results/transfer_{args.tag}.csv", index=False)
