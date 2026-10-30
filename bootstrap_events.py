"""Bootstrap confidence intervals over videos for the event-level results.

Videos are resampled with replacement (B times); all episodes and frames of a
drawn video come along, so within-video correlation is respected. For each
resample and detector, the threshold is re-chosen to give the best detection
rate at a false-alarm rate <= budget, exactly as in event_eval.at_budget.
Paired differences use the same resample for both detectors. The out-of-fold
scores are computed once on the full data and not refit per resample.

Detection at threshold t is equivalent to: the maximum score inside the
episode window [onset - TOL, end] is >= t, so per-video counts can be
tabulated once per threshold and resamples become matrix products.

Usage:
    python bootstrap_events.py results/mose_mask.csv [more.csv ...] [--B 2000]
Writes results/bootstrap_events.csv and results/bootstrap_events_diffs.csv.
"""
import argparse

import numpy as np
import pandas as pd

from event_eval import FA_BUDGETS, build_scores

PAIRS = [("GBDT episode-weighted", "GBDT causal"),
         ("GBDT onset", "GBDT causal"),
         ("GBDT episode-weighted", "boundary_unc"),
         ("GBDT episode-weighted", "pred_iou"),
         ("GBDT episode-weighted", "eprocess"),
         ("GBDT causal", "pred_iou")]


def video_counts(score, d, ev, n_thr=300):
    """Per-video counts on the same threshold grid as event_eval.sweep.

    Returns thresholds, detected episodes D (videos x thr), episodes E,
    false-alarm frames F (videos x thr) and eligible good frames C.
    """
    start, onset, end, can_fa = ev
    thrs = np.unique(np.quantile(score, np.linspace(0.0, 0.9995, n_thr)))
    vid = pd.factorize(d.video)[0]
    n_vid = vid.max() + 1
    ep_max = np.array([score[s:e + 1].max() for s, e in zip(start, end)])
    ep_vid = vid[onset]
    D = np.zeros((n_vid, len(thrs)))
    np.add.at(D, ep_vid, ep_max[:, None] >= thrs[None, :])
    E = np.bincount(ep_vid, minlength=n_vid).astype(float)
    F = np.zeros((n_vid, len(thrs)))
    np.add.at(F, vid[can_fa], score[can_fa][:, None] >= thrs[None, :])
    C = np.bincount(vid[can_fa], minlength=n_vid).astype(float)
    return D, E, F, C


def det_at_budget(W, counts, budget):
    """Best detection rate with false-alarm rate <= budget, per resample (rows of W)."""
    D, E, F, C = counts
    det = (W @ D) / np.maximum(W @ E, 1)[:, None]
    fa = (W @ F) / np.maximum(W @ C, 1)[:, None]
    return np.where(fa <= budget, det, 0.0).max(1)


def run(path, tau, B, rng):
    d, ev, scores = build_scores(path, tau)
    counts = {n: video_counts(s, d, ev) for n, s in scores.items()}
    n_vid = d.video.nunique()
    W = rng.multinomial(n_vid, np.full(n_vid, 1 / n_vid), size=B).astype(float)
    full = np.ones((1, n_vid))
    rows, diffs = [], []
    for b in FA_BUDGETS:
        boot = {n: det_at_budget(W, c, b) for n, c in counts.items()}
        point = {n: det_at_budget(full, c, b)[0] for n, c in counts.items()}
        for n in scores:
            lo, hi = np.percentile(boot[n], [2.5, 97.5])
            rows.append({"file": path, "tau": tau, "budget": b, "detector": n,
                         "det": point[n], "ci_lo": lo, "ci_hi": hi})
        for a, c in PAIRS:
            if a in boot and c in boot:
                dd = boot[a] - boot[c]
                lo, hi = np.percentile(dd, [2.5, 97.5])
                diffs.append({"file": path, "tau": tau, "budget": b, "a": a, "b": c,
                              "diff": point[a] - point[c], "ci_lo": lo, "ci_hi": hi,
                              "p_le_0": (dd <= 0).mean()})
    print(f"=== {path} (tau={tau}) videos {n_vid}, episodes {len(ev[1])}", flush=True)
    print(pd.DataFrame(diffs).query("budget == 0.05")[["a", "b", "diff", "ci_lo", "ci_hi", "p_le_0"]]
          .round(3).to_string(index=False), flush=True)
    return pd.DataFrame(rows), pd.DataFrame(diffs)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("csv", nargs="+")
    ap.add_argument("--B", type=int, default=2000)
    ap.add_argument("--tag", default="")
    args = ap.parse_args()
    base = "results/bootstrap_events" + (f"_{args.tag}" if args.tag else "")
    rng = np.random.default_rng(0)
    res, dif = zip(*[run(p, tau, args.B, rng) for p in args.csv for tau in (0.5, 0.7)])
    pd.concat(res).to_csv(f"{base}.csv", index=False)
    pd.concat(dif).to_csv(f"{base}_diffs.csv", index=False)
