"""Is the GBDT overfitting across videos? Regularization grid + nested CV.

For each config: out-of-fold AUROC / AUROC_within and the mean train AUROC
(the gap shows overfitting). "nested" picks the config inside each outer
training fold with an inner 3-fold GroupKFold, so its score is not tuned on
test videos. Features are BASE + temporal (the "GBDT (+temporal)" row).

Usage:
    python gbdt_regularization.py results/mose_mask.csv [more.csv ...]
"""
import sys

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold

from analyze import BASE, add_labels, add_temporal, within_object_auroc

CONFIGS = {
    "A default": dict(max_iter=300, learning_rate=0.05),
    "B leaves15": dict(max_iter=200, learning_rate=0.05, max_leaf_nodes=15,
                       min_samples_leaf=100, l2_regularization=1.0),
    "C depth3": dict(max_iter=150, learning_rate=0.05, max_depth=3,
                     min_samples_leaf=300, l2_regularization=5.0),
    "D depth2": dict(max_iter=100, learning_rate=0.05, max_depth=2,
                     min_samples_leaf=500, l2_regularization=10.0),
}


def model(name):
    return HistGradientBoostingClassifier(class_weight="balanced", random_state=0,
                                          **CONFIGS[name])


def inner_pick(X, y, groups):
    """Config with the best mean inner-CV AUROC on the training videos only."""
    score = {}
    for name in CONFIGS:
        aucs = []
        for tr, va in GroupKFold(3).split(X, y, groups):
            m = model(name).fit(X[tr], y[tr])
            aucs.append(roc_auc_score(y[va], m.predict_proba(X[va])[:, 1]))
        score[name] = np.mean(aucs)
    return max(score, key=score.get)


def run(path, tau):
    d = pd.read_csv(path, dtype={"video": str}).dropna(subset=["pred_iou"])
    d["fail"] = add_labels(d, tau)
    d = add_temporal(d)
    temporal = [c for c in d.columns if c.endswith(("_mean5", "_d1"))] + \
        ["pred_iou_min_sofar", "frame_frac"]
    X, y, groups = d[BASE + temporal].fillna(0).values, d.fail.values, d.video.values
    oof = {n: np.zeros(len(d)) for n in list(CONFIGS) + ["nested"]}
    train_auc = {n: [] for n in CONFIGS}
    picks = []
    for tr, te in GroupKFold(5).split(X, y, groups):
        for name in CONFIGS:
            m = model(name).fit(X[tr], y[tr])
            oof[name][te] = m.predict_proba(X[te])[:, 1]
            train_auc[name].append(roc_auc_score(y[tr], m.predict_proba(X[tr])[:, 1]))
        best = inner_pick(X[tr], y[tr], groups[tr])
        picks.append(best)
        oof["nested"][te] = oof[best][te]
    rows = []
    for name, s in oof.items():
        d["_s"] = s
        rows.append({"file": path, "tau": tau, "config": name,
                     "AUROC": roc_auc_score(y, s),
                     "AUROC_within": within_object_auroc(d, name, lambda g: g["_s"]),
                     "train_AUROC": np.mean(train_auc[name]) if name in train_auc else np.nan,
                     "picks": ",".join(p.split()[0] for p in picks) if name == "nested" else ""})
    t = pd.DataFrame(rows)
    t["gap"] = t.train_AUROC - t.AUROC
    print(f"\n=== {path} (tau={tau}) ===\n" + t.drop(columns="file").round(3).to_string(index=False),
          flush=True)
    return t


if __name__ == "__main__":
    out = pd.concat([run(p, tau) for p in sys.argv[1:] for tau in (0.5, 0.7)])
    out.to_csv("results/gbdt_regularization.csv", index=False)
