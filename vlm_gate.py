"""VLM-verifier gate: call a vision-language model only where the failure
predictor is suspicious, and measure what the verifier adds per call.

A VLM verifier (Qwen2.5-VL-3B-Instruct by default, or 7B; local) checks one object-frame from
two images: frame 0 with the prompted object outlined (from the ground-truth
prompt mask) and frame t with SAM 2.1's predicted mask outlined. With a
non-empty mask it is asked whether the outline still encloses the same object
and only that object (p_fail = P(No)); with an empty mask it is asked whether
the object is visible (p_fail = P(Yes), a missed target). P(.) is read from
the next-token logits of "Yes" and "No", so no text is parsed.

Stages (each resumable):
  masks   re-run SAM 2.1 (deterministic, same rows as the signal CSV) on
          N_VIDEOS random videos and render the overlays of every STRIDE-th
          tracked frame of each object -> <cache>/<setting>/*.jpg + items.csv
  vlm     score every rendered item with the VLM -> results/vlm_gate_<setting>.csv
  analyze join with the out-of-fold scores of the signal CSV (episode-weighted
          GBDT, pred_iou) and compare gates at call rates 5-100%
          -> results/vlm_gate_summary.csv

Usage:
    python vlm_gate.py masks --dataset sav --prompt mask
    python vlm_gate.py vlm --dataset sav --prompt mask
    python vlm_gate.py analyze
"""
from paths import CACHE, EXT  # noqa: E402
import argparse
import glob
import os
import time
from types import SimpleNamespace

import cv2
import numpy as np
import pandas as pd

CACHE = f"{CACHE}"
VLMS = {k: f"{EXT}/models/Qwen2.5-VL-{k.upper()}-Instruct" for k in ("3b", "7b")}
N_VIDEOS, SEED = 25, 0
STRIDE = {"mose": 2, "sav": 4}  # MOSE tracks are about half as long (median 78 vs 158 rows/video)
SIDE = 448  # long side of the rendered images (pixels)
SIDE_CROP, CROP_PAD = 336, 0.5  # object crops: long side, margin around the mask box
RATES = (0.05, 0.10, 0.20, 0.30, 0.50, 1.0)

Q_MASK = ("Image 1 shows a target object outlined in green. Image 2 is a later frame of the "
          "same video, where a tracker outlined in green what it believes is the same object. "
          "Does the green outline in Image 2 enclose the same object as in Image 1, covering "
          "most of it and little else? Answer Yes or No.")
Q_CROP = ("Image 1 shows a target object outlined in green. Image 2 shows a region of a later "
          "frame of the same video, where a tracker outlined in green what it believes is the same "
          "object. Is the object outlined in Image 2 the same object as in Image 1, with the "
          "outline covering the whole object and little else? Answer Yes or No.")
Q_EMPTY = ("Image 1 shows a target object outlined in green. Image 2 is a later frame of the "
           "same video. Is the target object from Image 1 visible in Image 2? Answer Yes or No.")


def render(img_path, mask, out, out_crop):
    """Frame with the mask outlined (long side SIDE), and a crop around the mask
    (box enlarged by CROP_PAD on each side, long side SIDE_CROP)."""
    img = cv2.imread(img_path)
    h, w = img.shape[:2]
    s = SIDE / max(h, w)
    small = cv2.resize(img, (round(w * s), round(h * s)), interpolation=cv2.INTER_AREA)
    crop = small
    if mask is not None and mask.any():
        m = mask.astype(np.uint8)
        cs, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(img, cs, -1, (0, 255, 0), max(2, round(2 / s)))
        small = cv2.resize(img, small.shape[1::-1], interpolation=cv2.INTER_AREA)
        ys, xs = np.nonzero(m)
        x0, x1, y0, y1 = xs.min(), xs.max(), ys.min(), ys.max()
        side = max(x1 - x0, y1 - y0, 64 / s) * (1 + 2 * CROP_PAD)  # square crop
        cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
        xa, ya = int(max(0, cx - side / 2)), int(max(0, cy - side / 2))
        xb, yb = int(min(w, cx + side / 2)), int(min(h, cy + side / 2))
        crop = img[ya:yb, xa:xb]
        c = SIDE_CROP / max(crop.shape[:2])
        crop = cv2.resize(crop, (max(28, round(crop.shape[1] * c)), max(28, round(crop.shape[0] * c))),
                          interpolation=cv2.INTER_AREA if c < 1 else cv2.INTER_LINEAR)
    cv2.imwrite(out, small, [cv2.IMWRITE_JPEG_QUALITY, 92])
    cv2.imwrite(out_crop, crop, [cv2.IMWRITE_JPEG_QUALITY, 92])


def stage_masks(dataset, prompt, n_videos=N_VIDEOS, cache=CACHE, video_ids=None):
    import torch
    import extract_signals as es
    es.DS = dict(es.DATASETS[dataset])
    setting = f"{dataset}_{prompt}"
    out_dir = f"{cache}/{setting}"
    os.makedirs(out_dir, exist_ok=True)
    videos = open(es.DS["list"]).read().split()
    videos = list(np.random.default_rng(SEED).choice(videos, N_VIDEOS, replace=False))[:n_videos]
    if video_ids:  # prompt-development videos (outside the evaluation sample)
        videos = video_ids.split(",")
    torch.backends.cuda.matmul.allow_tf32 = True
    predictor = es.build_sam2_video_predictor(es.CFG, es.CKPT, device="cuda")
    rec, arec = es.HeadRecorder(predictor), SimpleNamespace(buf=[])
    masks = {}
    orig = predictor.propagate_in_video

    def capture(state, *a, **k):  # keep the binary masks run_video computes from
        for fidx, ids, logits in orig(state, *a, **k):
            for j, oid in enumerate(ids if fidx % STRIDE[dataset] == 0 else []):
                masks[(fidx, oid)] = (logits[j, 0] > 0).cpu().numpy()
            yield fidx, ids, logits
    predictor.propagate_in_video = capture
    items_path = f"{out_dir}/items.csv"
    items = pd.read_csv(items_path, dtype={"video": str}) if os.path.exists(items_path) else None
    done = set(items.video) if items is not None else set()
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        for v in videos:
            if v in done:
                continue
            masks.clear()
            rows = es.run_video(predictor, rec, arec, v, prompt)
            gts = es.load_gt(v)
            frames = sorted(glob.glob(os.path.join(es.DS["img"], v, "*.jp*g")))
            new = []
            for oid in gts[0]:
                ref = f"{out_dir}/{v}_{oid}_ref"
                render(frames[0], gts[0][oid], ref + ".jpg", ref + "_crop.jpg")
            for r in rows:
                if r["frame"] % STRIDE[dataset]:
                    continue
                img = f"{out_dir}/{v}_{r['obj']}_{r['frame']:05d}"
                render(frames[r["frame"]], masks[(r["frame"], r["obj"])], img + ".jpg", img + "_crop.jpg")
                new.append({**{k: r[k] for k in ("video", "obj", "frame", "pred_iou",
                                                 "gt_iou", "gt_present", "pred_present")},
                            "ref": f"{v}_{r['obj']}_ref", "img": os.path.basename(img)})
            items = pd.concat([items, pd.DataFrame(new)]) if items is not None else pd.DataFrame(new)
            items.to_csv(items_path, index=False)
            print(f"{v}: {len(new)} items", flush=True)


def stage_vlm(dataset, prompt, cache=CACHE, out_dir="results", variant="full", vlm="3b"):
    import torch
    from PIL import Image
    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
    setting = f"{dataset}_{prompt}"
    items = pd.read_csv(f"{cache}/{setting}/items.csv", dtype={"video": str})
    out = f"{out_dir}/vlm_gate_{setting}" + (f"_{variant}" if variant != "full" else "") \
        + (f"_{vlm}" if vlm != "3b" else "") + ".csv"
    res = pd.read_csv(out, dtype={"video": str}) if os.path.exists(out) else pd.DataFrame()
    done = set(zip(res.video, res.obj, res.frame)) if len(res) else set()
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        VLMS[vlm], dtype=torch.bfloat16, device_map="cuda").eval()
    proc = AutoProcessor.from_pretrained(VLMS[vlm], min_pixels=128 * 28 * 28, max_pixels=SIDE * SIDE)
    tok = proc.tokenizer
    yes, no = tok.encode("Yes")[0], tok.encode("No")[0]
    # the Yes-No logit margin in fp32 from the last hidden state: the bf16 logits
    # (magnitude ~25) are quantized in steps of 0.125, which ties most scores
    w_diff = (model.lm_head.weight[yes] - model.lm_head.weight[no]).float()
    suffix = "_crop" if variant == "crop" else ""
    rows, t0, n = res.to_dict("records"), time.time(), 0
    for it in items.itertuples():
        if (it.video, it.obj, it.frame) in done:
            continue
        q = (Q_CROP if variant == "crop" else Q_MASK) if it.pred_present else Q_EMPTY
        files = (it.ref + suffix, it.img + (suffix if it.pred_present else ""))
        msgs = [{"role": "user", "content": [
            {"type": "text", "text": "Image 1:"}, {"type": "image"},
            {"type": "text", "text": "Image 2:"}, {"type": "image"},
            {"type": "text", "text": q}]}]
        text = proc.apply_chat_template(msgs, add_generation_prompt=True)
        ims = [Image.open(f"{cache}/{setting}/{f}").convert("RGB") for f in (files[0] + ".jpg", files[1] + ".jpg")]
        inp = proc(text=[text], images=ims, return_tensors="pt").to("cuda")
        ts = time.time()
        with torch.inference_mode():
            h = model(**inp, output_hidden_states=True, logits_to_keep=1).hidden_states[-1][0, -1]
        p_yes = torch.sigmoid(h.float() @ w_diff).item()
        rows.append({"video": it.video, "obj": it.obj, "frame": it.frame,
                     "p_yes": p_yes, "p_fail": 1 - p_yes if it.pred_present else p_yes,
                     "sec": time.time() - ts})
        n += 1
        if n % 50 == 0:
            pd.DataFrame(rows).to_csv(out, index=False)
            print(f"{setting}: {len(rows)}/{len(items)}  {(time.time() - t0) / n:.2f} s/item",
                  flush=True)
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"done {setting}: {len(rows)} items", flush=True)


def stage_analyze():
    from sklearn.metrics import roc_auc_score
    from event_eval import build_scores
    out = []
    for f in sorted(glob.glob("results/vlm_gate_*_*.csv")):
        name = os.path.basename(f)[9:-4].split("_")
        if name[0] == "summary":
            continue
        setting, variant = "_".join(name[:2]), "_".join(name[2:]) or "full"
        v = pd.read_csv(f, dtype={"video": str})
        d, _, sc = build_scores(f"results/{setting}.csv", 0.5)
        d = d.assign(gbdt=sc["GBDT episode-weighted"], piou=sc["pred_iou"])
        j = v.merge(d[["video", "obj", "frame", "fail", "gbdt", "piou"]],
                    on=["video", "obj", "frame"], how="inner")
        assert len(j) == len(v), (setting, len(j), len(v))
        y, n = j.fail.values, len(j)
        vlm_flag = j.p_fail.values > 0.5
        base = {"setting": setting, "variant": variant, "items": n, "videos": j.video.nunique(),
                "fail_rate": y.mean(), "sec_per_call": j.sec.median(),
                "AUROC_vlm": roc_auc_score(y, j.p_fail), "AUROC_gbdt": roc_auc_score(y, j.gbdt),
                "AUROC_pred_iou": roc_auc_score(y, j.piou)}
        rng = np.random.default_rng(SEED)
        gates = {"GBDT": j.gbdt.values, "pred_iou": j.piou.values,
                 "random": rng.permutation(n).astype(float)}
        for r in RATES:
            k = max(1, round(r * n))
            for g, s in gates.items():
                called = np.zeros(n, bool)
                called[np.argsort(-s, kind="stable")[:k]] = True
                for rule, flag in (("gate only", called), ("gate + VLM", called & vlm_flag)):
                    tp = (flag & (y == 1)).sum()
                    out.append({**base, "call_rate": r, "gate": g, "rule": rule,
                                "flagged": flag.mean(),
                                "precision": tp / max(flag.sum(), 1),
                                "recall": tp / max(y.sum(), 1),
                                "fpr": (flag & (y == 0)).sum() / max((y == 0).sum(), 1)})
    t = pd.DataFrame(out)
    t.to_csv("results/vlm_gate_summary.csv", index=False)
    with pd.option_context("display.width", 200):
        print(t.groupby(["setting", "variant"])[["items", "fail_rate", "sec_per_call", "AUROC_vlm",
                                    "AUROC_gbdt", "AUROC_pred_iou"]].first().round(3))
        print(t[t.gate != "random"].pivot_table(index=["setting", "variant", "call_rate"], columns=["gate", "rule"],
                                                values=["precision", "recall"]).round(3))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["masks", "vlm", "analyze"])
    ap.add_argument("--dataset", choices=["mose", "sav"])
    ap.add_argument("--prompt", choices=["mask", "click"])
    ap.add_argument("--videos", type=int, default=N_VIDEOS, help="first N of the sampled videos (smoke test)")
    ap.add_argument("--cache", default=CACHE)
    ap.add_argument("--out-dir", default="results")
    ap.add_argument("--vlm", choices=["3b", "7b"], default="3b")
    ap.add_argument("--variant", choices=["full", "crop"], default="full")
    ap.add_argument("--video-ids", default=None, help="comma-separated videos (prompt development)")
    a = ap.parse_args()
    {"masks": lambda: stage_masks(a.dataset, a.prompt, a.videos, a.cache, a.video_ids), "vlm": lambda: stage_vlm(a.dataset, a.prompt, a.cache, a.out_dir, a.variant, a.vlm),
     "analyze": stage_analyze}[a.stage]()
