"""Click scheduling: which frame should the user correct next?

Interactive-VOS simulation on top of SAM 2.1 (one object track at a time):
round 0 is the usual one-click prompt on frame 0 and full propagation. In each
of K rounds, a scheduler picks one frame of the current result, a simulated
user corrects it with up to N_CLICKS clicks, and the video is re-propagated.

Simulated user (standard corrective-click protocol): the click goes to the
interior point (distance-transform maximum) of the largest error region,
positive on a missed region and negative on a false-positive region; the next
click uses the updated mask at that frame. A corrected frame becomes a
conditioning frame (add_all_frames_to_correct_as_cond = True; with SAM 2's
default False, propagation overwrites the correction).

Schedulers (a frame is never picked twice; frame 0 is never picked):
- oracle: lowest ground-truth J (upper bound, needs ground truth)
- gbdt: highest failure probability of the episode-weighted causal GBDT,
  trained on the *other* dataset (MOSE <-> SA-V, click prompt), so no video is
  seen in training
- pred_iou: lowest SAM 2 predicted IoU (the model's own confidence)
- uniform: fixed interval, round k of K takes frame round(k * n / (K + 1))
  (all K frames are fixed in advance, as in fixed-interval annotation)
- hybrid: a correction also repairs the frames after it, up to the next
  corrected frame. The GBDT failure probabilities p are turned into the
  failure mass a correction at t could reach, weighted by p(t) because a
  correction needs an error at t to act on,
      gain(t) = p(t) * (1 - GAMMA^(t - prev anchor))
                * sum_{t <= s < next anchor} p(s) * GAMMA^(s - t),
  and the frame with the largest gain is picked (anchors: frame 0 and the
  frames corrected so far). The middle factor discounts frames that an
  earlier anchor still covers, so corrections do not pile up next to each
  other. GAMMA = 0.95 (half-life ~14 frames) was fixed a priori, not tuned.

Reported per round: mean presence-aware J over frames 1..n-1 (absent target:
J = 1 for an empty mask, else 0) and the share of failing frames (tau = 0.5).

Usage:
    python click_schedule.py --dataset sav --train results/mose_click.csv \
        --out results/click_schedule_sav.csv [--videos 40] [--K 3] \
        [--schedulers oracle gbdt pred_iou uniform hybrid]
"""
import argparse
import copy
import os
import time

import cv2
import numpy as np
import pandas as pd
import torch

import extract_signals as es
from analyze import add_labels, add_temporal
from conformal_alarm import fit, load

SCHEDULERS = ["oracle", "gbdt", "pred_iou", "uniform"]
GAMMA = 0.95
N_CLICKS = 3
MUTABLE = ["point_inputs_per_obj", "mask_inputs_per_obj", "output_dict_per_obj",
           "temp_output_dict_per_obj", "frames_tracked_per_obj"]


def jaccard(m, gt):
    if not gt.any():
        return float(not m.any())
    return es.iou(m, gt)


def correction_click(pred, gt):
    """One simulated click on the largest error region, or None if no error."""
    best = None
    for region, label in ((gt & ~pred, 1), (pred & ~gt, 0)):
        n, cc = cv2.connectedComponents(region.astype(np.uint8))
        for i in range(1, n):
            comp = cc == i
            a = comp.sum()
            if best is None or a > best[0]:
                best = (a, comp, label)
    if best is None:
        return None
    dist = cv2.distanceTransform(np.pad(best[1], 1).astype(np.uint8), cv2.DIST_L2, 5)[1:-1, 1:-1]
    y, x = np.unravel_index(np.argmax(dist), dist.shape)
    return np.array([[x, y]], np.float32), np.array([best[2]], np.int32)


def snapshot(state):
    return {k: copy.deepcopy(state[k]) for k in MUTABLE}


def restore(state, snap):
    for k in MUTABLE:
        state[k] = copy.deepcopy(snap[k])


def propagate(predictor, rec, state, gts, oid, video, cond_ious):
    """Propagate the whole video; one feature row per frame >= 1 (as extract_signals)."""
    rows, prev_mask, prev_ptr, first_ptr, first_area = [], None, None, None, None
    rec.buf.clear()
    for fidx, _, logits in predictor.propagate_in_video(state):
        ious = list(rec.buf)
        rec.buf.clear()
        lg = logits[0, 0].float().cpu().numpy()
        od = state["output_dict_per_obj"][0]
        is_cond = fidx in od["cond_frame_outputs"]
        cur = od["cond_frame_outputs" if is_cond else "non_cond_frame_outputs"][fidx]
        ptr = cur["obj_ptr"].float().cpu().numpy().reshape(-1)
        if fidx == 0:
            first_area, prev_mask, prev_ptr = (lg > 0).sum(), lg > 0, ptr
            continue
        if first_ptr is None:
            first_ptr = ptr
        f, m = es.mask_feats(lg, prev_mask, first_area)
        r = {"video": video, "obj": oid, "frame": fidx, "n_frames": len(gts),
             "corrected": int(is_cond), **f,
             "obj_score": float(cur["object_score_logits"].float().cpu().reshape(-1)[0]),
             "ptr_dist_prev": es.cos_dist(ptr, prev_ptr),
             "ptr_dist_first": es.cos_dist(ptr, first_ptr)}
        # cond frames are not re-decoded during propagation: use the IoU
        # predictions recorded when the frame was corrected
        iou = ious[0] if len(ious) == 1 else cond_ious.get(fidx)
        if iou is not None:
            s = np.sort(iou)[::-1]
            r.update(pred_iou=s[0], pred_iou_gap=s[0] - s[1] if len(s) > 1 else 0.0,
                     pred_iou_std=s.std())
        gt = gts[fidx][oid]
        r.update(gt_present=int(gt.any()), pred_present=int(m.any()),
                 gt_iou=es.iou(m, gt), J=jaccard(m, gt))
        rows.append(r)
        prev_mask, prev_ptr = m, ptr
    d = pd.DataFrame(rows)
    d["pred_iou"] = d.pred_iou.ffill().bfill() if "pred_iou" in d else np.nan
    return d


def correct(predictor, rec, state, gts, oid, t, d):
    """Simulated user: up to N_CLICKS corrective clicks on frame t."""
    gt = gts[t][oid]
    pred = None
    ious = None
    for _ in range(N_CLICKS):
        if pred is None:  # current mask at t from the last propagation
            od = state["output_dict_per_obj"][0]
            store = od["cond_frame_outputs"] if t in od["cond_frame_outputs"] \
                else od["non_cond_frame_outputs"]
            pred = predictor._get_orig_video_res_output(state, store[t]["pred_masks"])[1]
            pred = pred[0, 0].float().cpu().numpy() > 0
        click = correction_click(pred, gt)
        if click is None:
            break
        rec.buf.clear()
        _, _, lg = predictor.add_new_points_or_box(
            state, frame_idx=t, obj_id=oid, points=click[0], labels=click[1],
            clear_old_points=False)
        if rec.buf:
            ious = rec.buf[-1]
        pred = lg[0, 0].float().cpu().numpy() > 0
    return ious


def schedule(name, d, picked, k, K, model, feats):
    cand = d[~d.frame.isin(picked) & (d.corrected == 0)]
    if cand.empty:
        return None
    if name == "oracle":
        return int(cand.frame.values[np.argmin(cand.J.values)])
    if name == "pred_iou":
        return int(cand.frame.values[np.argmin(cand.pred_iou.values)])
    if name == "uniform":
        n = d.n_frames.iloc[0]
        t = int(round(k * n / (K + 1)))
        return min(max(t, 1), n - 1)
    x = add_temporal(d.assign(fail=0)).sort_values("frame")
    p = model.predict_proba(x[feats].fillna(0).values)[:, 1]
    ok = x.frame.isin(cand.frame).values
    if name == "gbdt":
        return int(x.frame.values[ok][np.argmax(p[ok])])
    # hybrid: discounted failure mass up to the next anchor, computed backwards
    frames = x.frame.values
    anchor = ~ok | (frames == 0)
    gain = np.zeros(len(p))
    acc = 0.0
    for i in range(len(p) - 1, -1, -1):
        acc = 0.0 if anchor[i] else p[i] + GAMMA * acc
        gain[i] = acc
    prev = np.maximum.accumulate(np.where(anchor, np.arange(len(p)), -1))
    since = frames - np.where(prev >= 0, frames[np.maximum(prev, 0)], 0)
    gain *= p * (1 - GAMMA ** since)
    return int(frames[ok][np.argmax(gain[ok])])


def summary(d, sched, rnd, picked):
    return {"scheduler": sched, "round": rnd, "frames": len(d), "mean_J": d.J.mean(),
            "fail_share": add_labels(d, 0.5).mean(),
            "picked": picked[-1] if picked else -1}


def run_track(predictor, rec, state, gts, oid, video, K, model, feats):
    """Round 0 once, then K rounds per scheduler from a snapshot of round 0."""
    predictor.reset_state(state)
    predictor.add_new_points_or_box(state, frame_idx=0, obj_id=oid,
                                    points=es.center_click(gts[0][oid]),
                                    labels=np.array([1], np.int32))
    d0 = propagate(predictor, rec, state, gts, oid, video, {})
    snap = snapshot(state)
    out = [summary(d0, s, 0, []) for s in SCHEDULERS]
    for s in SCHEDULERS:
        restore(state, snap)
        d, picked, cond_ious = d0, [], {}
        for k in range(1, K + 1):
            t = schedule(s, d, picked, k, K, model, feats)
            if t is None:
                break
            picked.append(t)
            ious = correct(predictor, rec, state, gts, oid, t, d)
            if ious is not None:
                cond_ious[t] = ious
            d = propagate(predictor, rec, state, gts, oid, video, cond_ious)
            out.append(summary(d, s, k, picked))
    return [dict(r, video=video, obj=oid) for r in out]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["mose", "sav"], required=True)
    ap.add_argument("--train", required=True, help="click CSV of the other dataset")
    ap.add_argument("--out", required=True)
    ap.add_argument("--videos", type=int, default=40)
    ap.add_argument("--K", type=int, default=3)
    ap.add_argument("--schedulers", nargs="+", default=SCHEDULERS)
    args = ap.parse_args()
    SCHEDULERS[:] = args.schedulers

    es.DS = es.DATASETS[args.dataset]
    src, _, feats = load(args.train, 0.5)
    model = fit(src, feats, np.ones(len(src), bool))

    torch.backends.cuda.matmul.allow_tf32 = True
    predictor = es.build_sam2_video_predictor(es.CFG, es.CKPT, device="cuda")
    predictor.add_all_frames_to_correct_as_cond = True
    rec = es.HeadRecorder(predictor)
    videos = open(es.DS["list"]).read().split()
    videos = list(np.random.default_rng(0).permutation(videos)[:args.videos])

    rows, done = [], set()
    if os.path.exists(args.out):
        prev = pd.read_csv(args.out, dtype={"video": str})
        rows, done = prev.to_dict("records"), set(prev.video)
    t0 = time.time()
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        for i, v in enumerate(videos):
            if v in done:
                continue
            ts = time.time()
            gts = es.load_gt(v)
            state = predictor.init_state(video_path=os.path.join(es.DS["img"], v),
                                         offload_video_to_cpu=True)
            new = []
            for oid in gts[0]:
                new += run_track(predictor, rec, state, gts, oid, v, args.K, model, feats)
            rows += new
            t = pd.DataFrame(new)
            gain = t[t["round"] == args.K].groupby("scheduler").mean_J.mean() \
                - t[t["round"] == 0].mean_J.mean()
            print(f"[{i+1}/{len(videos)}] {v}: {len(gts[0])} objs, {len(gts)} frames, "
                  f"gain@K " + " ".join(f"{s}={gain.get(s, np.nan):+.3f}" for s in SCHEDULERS)
                  + f", {time.time()-ts:.0f}s", flush=True)
            pd.DataFrame(rows).to_csv(args.out, index=False)
            del state
            torch.cuda.empty_cache()
    print(f"done in {time.time()-t0:.0f}s -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
