"""Calibrated alarm thresholds: conformal risk control (CRC) of the false-alarm rate.

event_eval.py picks the threshold on the test data itself (oracle budget
matching). A deployed detector has to fix the threshold beforehand. Here the
threshold of the episode-weighted causal GBDT is set on held-out calibration
videos and checked on unseen test videos.

Loss of video i at threshold t: L_i(t) = share of its eligible good frames
(event_eval.episodes, can_fa) with score >= t. It lies in [0, 1] and does not
increase with t. CRC (Angelopoulos et al., ICLR 2024) picks the smallest t with

    (n * mean_i L_i(t) + 1) / (n + 1) <= alpha,

which gives E[L_test(t)] <= alpha for a new video exchangeable with the n
calibration videos. Videos without eligible good frames are left out of the
calibration. "naive" uses the same rule without the correction (pooled
calibration frames, as event_eval does on test data).

Within-dataset protocol (R repeats): videos are split at random into train
(50%), calibration (25%) and test (25%); the GBDT is refit on train each time.
Cross-dataset protocol (MOSE <-> SA-V, same prompt): the model is trained on
75% of the source videos and calibrated either on the other 25% of the source
("source-cal") or on k target videos ("target-cal k"); the remaining target
videos are the test set.

Reported per protocol: mean test false-alarm rate (video-averaged, the
quantity CRC controls, and frame-pooled), its 90th percentile over repeats,
the share of repeats where the video-averaged rate exceeds alpha, and the
episode detection rate.

Usage:
    python conformal_alarm.py results/mose_mask.csv results/mose_click.csv \
        results/sav_mask.csv results/sav_click.csv [--R 100]
Writes results/conformal_alarm.csv (one row per repeat) and
results/conformal_alarm_summary.csv.
"""
import argparse
import os

import numpy as np
import pandas as pd

from analyze import BASE, add_labels, add_temporal, make_hgb
from event_eval import episodes, onset_targets

ALPHAS = (0.02, 0.05, 0.10)
TAUS = (0.5, 0.7)
TARGET_K = (10, 20)


def load(path, tau):
    """Frame table, episodes, causal feature list and episode weights."""
    d = pd.read_csv(path, dtype={"video": str}).dropna(subset=["pred_iou"])
    d["fail"] = add_labels(d, tau)
    d = add_temporal(d).reset_index(drop=True)
    d["track"] = pd.factorize(d.video + "_" + d.obj.astype(str))[0]
    causal = [c for c in d.columns if c.endswith(("_mean5", "_d1"))] + ["pred_iou_min_sofar"]
    ev = episodes(d)
    _, _, w_ep = onset_targets(d, ev)
    d["w_ep"] = w_ep
    d["vid"] = pd.factorize(d.video)[0]
    return d, ev, BASE + causal


def fit(d, feats, rows):
    """Episode-weighted GBDT on the given rows (weights as in cv_scores_weighted)."""
    X, y, w = d.loc[rows, feats].fillna(0).values, d.fail.values[rows], d.w_ep.values[rows].copy()
    for c in (0, 1):
        w[y == c] /= w[y == c].mean()
    return make_hgb().fit(X, y, sample_weight=w)


def video_counts(score, d, ev, thrs):
    """Per-video false-alarm frames F, eligible frames C, detected episodes D, episodes E."""
    start, onset, end, can_fa = ev
    vid, nv = d.vid.values, d.vid.max() + 1
    F = np.zeros((nv, len(thrs)))
    np.add.at(F, vid[can_fa], score[can_fa][:, None] >= thrs[None, :])
    C = np.bincount(vid[can_fa], minlength=nv).astype(float)
    ep_max = np.array([score[s:e + 1].max() for s, e in zip(start, end)])
    D = np.zeros((nv, len(thrs)))
    np.add.at(D, vid[onset], ep_max[:, None] >= thrs[None, :])
    E = np.bincount(vid[onset], minlength=nv).astype(float)
    return F, C, D, E


def thresholds(F, C, cal, alpha):
    """Index of the CRC and naive thresholds on the calibration videos."""
    cal = cal[C[cal] > 0]
    n = len(cal)
    L = (F[cal] / C[cal, None]).mean(0)          # video-averaged loss, decreasing in t
    pooled = F[cal].sum(0) / C[cal].sum()
    last = F.shape[1] - 1
    crc = np.flatnonzero((n * L + 1) / (n + 1) <= alpha)
    naive = np.flatnonzero(pooled <= alpha)
    return (crc[0] if len(crc) else last), (naive[0] if len(naive) else last)


def evaluate(counts, cal, test, alpha):
    F, C, D, E = counts
    test_fa = test[C[test] > 0]
    out = {}
    for name, j in zip(("crc", "naive"), thresholds(F, C, cal, alpha)):
        out[name] = {"fa_video": (F[test_fa, j] / C[test_fa]).mean(),
                     "fa_pooled": F[test, j].sum() / max(C[test].sum(), 1),
                     "det": D[test, j].sum() / max(E[test].sum(), 1)}
    return out


def grid(score):
    return np.unique(np.r_[np.quantile(score, np.linspace(0, 1, 2001)), np.inf])


def within(path, tau, R, rng):
    d, ev, feats = load(path, tau)
    nv, rows = d.vid.max() + 1, []
    for r in range(R):
        p = rng.permutation(nv)
        tr, cal, test = p[:nv // 2], p[nv // 2: 3 * nv // 4], p[3 * nv // 4:]
        m = fit(d, feats, np.isin(d.vid.values, tr))
        score = m.predict_proba(d[feats].fillna(0).values)[:, 1]
        counts = video_counts(score, d, ev, grid(score[np.isin(d.vid.values, cal)]))
        for a in ALPHAS:
            for meth, v in evaluate(counts, cal, test, a).items():
                rows.append({"protocol": "within", "source": path, "target": path, "tau": tau,
                             "alpha": a, "method": meth, "repeat": r, **v})
    return rows


def cross(src_path, tgt_path, tau, R, rng):
    s, _, fs = load(src_path, tau)
    t, ev_t, ft = load(tgt_path, tau)
    feats = [f for f in fs if f in ft]
    ns, nt, rows = s.vid.max() + 1, t.vid.max() + 1, []
    for r in range(R):
        p = rng.permutation(ns)
        tr, cal_s = p[:3 * ns // 4], p[3 * ns // 4:]
        m = fit(s, feats, np.isin(s.vid.values, tr))
        score_s = m.predict_proba(s[feats].fillna(0).values)[:, 1]
        score_t = m.predict_proba(t[feats].fillna(0).values)[:, 1]
        q = rng.permutation(nt)
        k_max = max(TARGET_K)
        test = q[k_max:]  # same test videos for every calibration source
        # source calibration: thresholds come from the source videos
        thr = grid(score_s[np.isin(s.vid.values, cal_s)])
        cs = video_counts(score_s, s, episodes(s), thr)
        ct = video_counts(score_t, t, ev_t, thr)
        for a in ALPHAS:
            for meth, j in zip(("crc", "naive"), thresholds(cs[0], cs[1], cal_s, a)):
                F, C, D, E = ct
                tf = test[C[test] > 0]
                rows.append({"protocol": "source-cal", "source": src_path, "target": tgt_path,
                             "tau": tau, "alpha": a, "method": meth, "repeat": r,
                             "fa_video": (F[tf, j] / C[tf]).mean(),
                             "fa_pooled": F[test, j].sum() / max(C[test].sum(), 1),
                             "det": D[test, j].sum() / max(E[test].sum(), 1)})
        for k in TARGET_K:
            cal_t = q[:k]
            counts = video_counts(score_t, t, ev_t, grid(score_t[np.isin(t.vid.values, cal_t)]))
            for a in ALPHAS:
                for meth, v in evaluate(counts, cal_t, test, a).items():
                    rows.append({"protocol": f"target-cal {k}", "source": src_path,
                                 "target": tgt_path, "tau": tau, "alpha": a, "method": meth,
                                 "repeat": r, **v})
    return rows


def summarize(df):
    g = df.groupby(["protocol", "source", "target", "tau", "alpha", "method"], sort=False)
    return g.apply(lambda x: pd.Series({
        "fa_video_mean": x.fa_video.mean(), "fa_video_p90": x.fa_video.quantile(0.9),
        "violate": (x.fa_video > x.name[4]).mean(),
        "fa_pooled_mean": x.fa_pooled.mean(), "det_mean": x.det.mean(),
        "det_sd": x.det.std()}), include_groups=False).reset_index()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("csv", nargs="+")
    ap.add_argument("--R", type=int, default=100)
    args = ap.parse_args()
    rng = np.random.default_rng(0)
    rows = []
    for tau in TAUS:
        for p in args.csv:
            rows += within(p, tau, args.R, rng)
            print(f"within {p} tau={tau} done", flush=True)
        for p in args.csv:
            name = os.path.basename(p)
            for q in args.csv:
                qn = os.path.basename(q)
                if p != q and name.split("_")[1] == qn.split("_")[1]:  # same prompt type
                    rows += cross(p, q, tau, args.R, rng)
                    print(f"cross {p} -> {q} tau={tau} done", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv("results/conformal_alarm.csv", index=False)
    s = summarize(df)
    s.to_csv("results/conformal_alarm_summary.csv", index=False)
    pd.set_option("display.width", 200)
    print(s.round(3).to_string(index=False))
