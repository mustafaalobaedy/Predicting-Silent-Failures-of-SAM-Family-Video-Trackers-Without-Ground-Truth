"""How well do SAM 2 internal signals predict frame failure?

Failure label (presence-aware):
    target visible   -> failure if IoU(pred, GT) < tau
    target invisible -> failure if SAM 2 still outputs a non-empty mask

Usage:
    python analyze.py results/mose_mask.csv results/mose_click.csv
"""
import sys

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

# raw signal -> score oriented so that a higher value means "more likely failing"
SINGLE = {
    "pred_iou": lambda d: -d.pred_iou,
    "obj_score": lambda d: -d.obj_score,
    "ptr_dist_prev": lambda d: d.ptr_dist_prev,
    "ptr_dist_first": lambda d: d.ptr_dist_first,
    "temporal_iou": lambda d: -d.temporal_iou,
    "boundary_unc": lambda d: d.boundary_unc,
    "logit_margin": lambda d: -d.logit_margin,
    "area_change_prev": lambda d: d.log_area_ratio_prev.abs(),
    "area_change_first": lambda d: d.log_area_ratio_first.abs(),
    "centroid_shift": lambda d: d.centroid_shift,
    "n_components": lambda d: d.n_components,
    "pred_iou_std": lambda d: d.pred_iou_std,
    # trivial baseline: small or empty masks fail (orientation fixed on DAVIS)
    "area_frac": lambda d: -d.area_frac,
    "eprocess": lambda d: d.eprocess,
}
BASE = ["pred_iou", "pred_iou_gap", "pred_iou_std", "obj_score", "ptr_dist_prev",
        "ptr_dist_first", "temporal_iou", "boundary_unc", "logit_margin",
        "log_area_ratio_prev", "log_area_ratio_first", "centroid_shift",
        "n_components", "area_frac"]

# memory-attention signals (present only in the *_attn.csv extractions), per layer
ATTN_RAW = [f"attn{l}_{n}" for l in range(4)
            for n in ["ent", "ent_fg", "ptr_fg", "cond_fg", "recent_fg"]]
ATTN = ATTN_RAW + [f"attn{l}_{n}_delta" for l in range(4) for n in ["ent_fg", "cond_fg"]]
# a-priori orientation: diffuse attention, or attention that drifts away from the
# prompted frame, is taken as a sign of failure
SINGLE_ATTN = {**{f"attn{l}_ent_fg": (lambda d, l=l: d[f"attn{l}_ent_fg"]) for l in range(4)},
               **{f"attn{l}_cond_fg": (lambda d, l=l: -d[f"attn{l}_cond_fg"])
                  for l in range(4)}}


def make_hgb():
    # shallow, strongly regularized trees: the deeper default overfit across videos
    # (train AUROC 0.98 vs test 0.63 on SA-V click; see gbdt_regularization.py)
    return HistGradientBoostingClassifier(max_iter=150, learning_rate=0.05, max_depth=3,
                                          min_samples_leaf=300, l2_regularization=5.0,
                                          class_weight="balanced", random_state=0)


def add_labels(d, tau):
    vis = d.gt_present == 1
    return ((vis & (d.gt_iou < tau)) | (~vis & (d.pred_present == 1))).astype(int)


def eprocess(m, eps=0.75, w=10, ema=0.25):
    """Log e-process of Monroy Munoz et al. (arXiv:2602.12983), unsupervised.

    m is a bounded quality score in [0, 1] (here SAM 2's predicted IoU). The
    null is E[m_t | past] >= eps. Bets use aGRAPA on the last w values and the
    returned score is the log wealth accumulated over the last w frames, so a
    higher value means stronger evidence of failure.
    """
    m = pd.Series(np.clip(m, 0, 1)).ewm(alpha=ema).mean().values
    logx = np.zeros(len(m))
    for t in range(len(m)):
        past = m[max(0, t - w):t]
        mu, var = (past.mean(), past.var()) if len(past) > 1 else (1.0, 0.0)
        lam = (eps - mu) / (var + (eps - mu) ** 2 + 1e-12)
        lam = np.clip(lam, 0, 1 / (2 * eps))
        logx[t] = np.log1p(lam * (eps - m[t]))
    return pd.Series(logx).rolling(w, 1).sum().values


def add_temporal(d, w=5):
    """Causal features: only current and past frames of the same object."""
    d = d.sort_values(["video", "obj", "frame"]).copy()
    g = d.groupby(["video", "obj"])
    for c in ["pred_iou", "obj_score", "temporal_iou", "ptr_dist_prev", "boundary_unc"]:
        d[f"{c}_mean{w}"] = g[c].transform(lambda s: s.rolling(w, 1).mean())
        d[f"{c}_d1"] = g[c].diff().fillna(0)
    d["pred_iou_min_sofar"] = g["pred_iou"].cummin()
    d["frame_frac"] = d.frame / d.n_frames
    if "attn0_ent" in d:
        # named *_delta so they stay out of the "+temporal" feature set
        for l in range(4):
            for n in ["ent_fg", "cond_fg"]:
                d[f"attn{l}_{n}_delta"] = g[f"attn{l}_{n}"].diff().fillna(0)
    d["eprocess"] = np.concatenate(
        [eprocess(g.pred_iou.values) for _, g in g]) if len(d) else []
    return d


def recall_at_fpr(y, s, fpr_target=0.1):
    fpr, tpr, _ = roc_curve(y, s)
    return float(np.interp(fpr_target, fpr, tpr))


def evaluate(y, s):
    return {"AUROC": roc_auc_score(y, s), "AUPRC": average_precision_score(y, s),
            "R@FPR10": recall_at_fpr(y, s)}


def cv_scores(d, feats, model, n_splits=5):
    """Out-of-fold scores, grouped by video so no video is in train and test."""
    X, y, groups = d[feats].fillna(0).values, d.fail.values, d.video.values
    oof = np.zeros(len(d))
    for tr, te in GroupKFold(n_splits).split(X, y, groups):
        m = model()
        m.fit(X[tr], y[tr])
        oof[te] = m.predict_proba(X[te])[:, 1]
    return oof


def run(path, tau=0.5):
    d = pd.read_csv(path, dtype={"video": str}).dropna(subset=["pred_iou"])
    d["fail"] = add_labels(d, tau)
    d = add_temporal(d)
    print(f"\n=== {path}  (tau={tau}) ===")
    print(f"rows {len(d)}, videos {d.video.nunique()}, objects "
          f"{len(d[['video', 'obj']].drop_duplicates())}, mean GT IoU {d.gt_iou.mean():.3f}, "
          f"failure rate {d.fail.mean():.3%}, failing objects "
          f"{(d.groupby(['video','obj']).fail.max()).sum()}, mixed objects "
          f"{(d.groupby(['video','obj']).fail.nunique() == 2).sum()}")
    if d.fail.nunique() < 2:
        print("only one class; skipping")
        return None

    has_attn = "attn0_ent" in d
    single = {**SINGLE, **SINGLE_ATTN} if has_attn else SINGLE
    res = {}
    for name, f in single.items():
        res[name] = evaluate(d.fail, f(d).fillna(0))
    temporal = [c for c in d.columns if c.endswith(("_mean5", "_d1"))] + \
        ["pred_iou_min_sofar", "frame_frac"]
    lr = lambda: make_pipeline(StandardScaler(),
                               LogisticRegression(max_iter=2000, class_weight="balanced"))
    hgb = make_hgb
    learned = {}
    configs = [("LR (all signals)", BASE, lr),
               ("LR (+temporal)", BASE + temporal, lr),
               ("GBDT (all signals)", BASE, hgb),
               ("GBDT (+temporal)", BASE + temporal, hgb)]
    if has_attn:
        configs += [("GBDT (attn only)", ATTN, hgb),
                    ("GBDT (+attn)", BASE + ATTN, hgb),
                    ("GBDT (+temporal+attn)", BASE + temporal + ATTN, hgb)]
    for name, feats, model in configs:
        col = f"oof_{len(learned)}"
        d[col] = cv_scores(d, feats, model)
        res[name] = evaluate(d.fail, d[col])
        learned[name] = lambda g, c=col: g[c]
    t = pd.DataFrame(res).T.sort_values("AUROC", ascending=False)
    t["AUROC_within"] = [within_object_auroc(d, n, single.get(n, learned.get(n)))
                         for n in t.index]
    print(t.round(3).to_string())
    return t


def within_object_auroc(d, name, f):
    """Mean AUROC inside objects that have both failing and good frames.

    This tests whether a signal finds *when* an object fails, not only *which*
    object is hard.
    """
    if f is None:
        return np.nan
    aucs = []
    for _, g in d.groupby(["video", "obj"]):
        if g.fail.nunique() == 2:
            aucs.append(roc_auc_score(g.fail, f(g).fillna(0)))
    return float(np.mean(aucs)) if aucs else np.nan


if __name__ == "__main__":
    for p in sys.argv[1:]:
        for tau in (0.5, 0.7):
            t = run(p, tau)
            if t is not None:
                t.to_csv(p.replace(".csv", f"_auroc_tau{tau}.csv"))
