"""Extract per-frame internal signals from SAM 2.1 (DAVIS-2017 val, MOSE train, SA-V val, LVOS v2 val).

For every (video, object, frame) the script records signals that SAM 2 exposes
without ground truth, plus the ground-truth IoU used to build failure labels.

Usage:
    python extract_signals.py --dataset mose --prompt mask  --out results/mose_mask.csv
    python extract_signals.py --dataset mose --prompt click --out results/mose_click.csv
"""
from paths import CKPT, DATA, EXT  # noqa: E402
import argparse
import os
import sys
import time
from types import SimpleNamespace

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image

# EdgeTAM ships its own `sam2` package with the same API, so the model is chosen
# before anything from sam2 is imported
MODEL = sys.argv[sys.argv.index("--model") + 1] if "--model" in sys.argv else "sam2"
if MODEL == "edgetam":
    sys.path.insert(0, f"{EXT}/EdgeTAM")
elif MODEL == "sam3":  # package `sam3`; SAM 2.1's `sam2` stays importable
    sys.path.insert(0, f"{EXT}/sam3")

import sam2.sam2_video_predictor
from sam2.build_sam import build_sam2_video_predictor
from sam2.utils.misc import _load_img_as_tensor
from sam2.modeling.sam.transformer import apply_rotary_enc

DATASETS = {
    "davis": {"ann": f"{DATA}/DAVIS2017/DAVIS/Annotations/480p",
              "img": f"{DATA}/DAVIS2017/DAVIS/JPEGImages/480p",
              "list": f"{DATA}/DAVIS2017/DAVIS/ImageSets/2017/val.txt"},
    "mose": {"ann": f"{DATA}/MOSE/train/Annotations",
             "img": f"{DATA}/MOSE/train/JPEGImages",
             "list": f"{DATA}/MOSE/subset_n200_seed0.txt"},
    # SA-V val is labelled at 6 fps; tracking runs on those labelled frames
    # (symlinks made by make_sav_6fps.py), one PNG folder per object
    "sav": {"ann": f"{DATA}/SA-V/sav_val/Annotations_6fps",
            "img": f"{DATA}/SA-V/sav_val/JPEGImages_6fps",
            "list": f"{DATA}/SA-V/sav_val/sav_val.txt",
            "per_obj": True},
    # LVOS v2 val is labelled at 6 fps (every 5th frame of 30 fps); videos run
    # up to 2,140 frames, so frames and ground truth are loaded lazily
    "lvos": {"ann": f"{DATA}/LVOS2/valid/Annotations",
             "img": f"{DATA}/LVOS2/valid/JPEGImages",
             "list": f"{DATA}/LVOS2/valid/valid.txt",
             "lazy": True},
}
DS = DATASETS["davis"]
MODELS = {"sam2": ("configs/sam2.1/sam2.1_hiera_b+.yaml",
                    f"{CKPT}/sam2.1_hiera_base_plus.pt"),
          "edgetam": ("configs/edgetam.yaml",
                      f"{EXT}/EdgeTAM/checkpoints/edgetam.pt"),
          "sam3": (None, f"{CKPT}/sam3.pt")}
CFG, CKPT = MODELS[MODEL]


class HeadRecorder:
    """Wraps _forward_sam_heads to collect the IoU predictions of each call.

    SAM 2.1 tracks objects one by one, so a tracked frame produces one call per
    object, in object order; EdgeTAM batches all objects in one call.
    """

    def __init__(self, predictor):
        self.buf = []
        orig = predictor._forward_sam_heads

        def wrapped(*args, **kwargs):
            out = orig(*args, **kwargs)
            # out[2] holds the IoU predictions of the 3 candidate masks; SAM 2.1
            # calls once per object, EdgeTAM (SAM 2.0 predictor) once per frame
            # with all objects batched in object order
            self.buf.extend(out[2].detach().float().cpu().numpy())
            return out

        predictor._forward_sam_heads = wrapped


class AttnRecorder:
    """Records memory-attention statistics of every memory-attention layer.

    The memory holds, in order: the frame-0 conditioning memory, up to 6 past
    frames (oldest first), each MEM_TOK spatial tokens, then the object-pointer
    tokens. For every query position the wrappers recompute the attention
    probabilities (the layer outputs themselves are untouched) and store 64x64
    maps of normalized entropy and of the mass on pointers, on frame 0 and on the
    newest past frame. The layers differ: on DAVIS, layers 0 and 3 attend mostly
    to frame 0 and layer 1 mostly to the newest frame. Like HeadRecorder, it
    relies on one memory-attention pass per object, in object order.
    """
    MEM_TOK = 64 * 64

    def __init__(self, predictor, chunk=1024):
        self.buf = []
        layers = predictor.memory_attention.layers
        for li, layer in enumerate(layers):
            self._wrap(layer.cross_attn_image, li, len(layers), chunk)

    def _wrap(self, attn, li, n_layers, chunk):
        orig = attn.forward

        def wrapped(q, k, v, num_k_exclude_rope=0):
            out = orig(q, k, v, num_k_exclude_rope=num_k_exclude_rope)
            stats = self._stats(attn, q, k, num_k_exclude_rope, chunk)
            # one entry per object: the stats of all layers stacked
            if li == 0:
                self.buf.append([])
            self.buf[-1].append(stats)
            if li == n_layers - 1:
                self.buf[-1] = np.stack(self.buf[-1])
            return out

        attn.forward = wrapped

    def _stats(self, attn, q, k, n_ptr, chunk):
        # same projections and rotary encoding as RoPEAttention.forward
        q = attn._separate_heads(attn.q_proj(q), attn.num_heads)
        k = attn._separate_heads(attn.k_proj(k), attn.num_heads)
        n_rope = k.size(-2) - n_ptr
        q, k[:, :, :n_rope] = apply_rotary_enc(
            q, k[:, :, :n_rope], freqs_cis=attn.freqs_cis,
            repeat_freqs_k=attn.rope_k_repeat)
        q, k = q[0].float(), k[0].float()  # heads x tokens x dim (batch of 1)
        n_k, n_mem = k.size(-2), n_rope // self.MEM_TOK
        maps = []
        for s in range(0, q.size(-2), chunk):
            p = F.softmax(q[:, s:s + chunk] @ k.transpose(-1, -2) / q.size(-1) ** 0.5, -1)
            ent = -(p * torch.log(p.clamp_min(1e-12))).sum(-1) / np.log(n_k)
            ptr = p[..., n_rope:].sum(-1)
            cond = p[..., :self.MEM_TOK].sum(-1)
            recent = p[..., (n_mem - 1) * self.MEM_TOK:n_rope].sum(-1) if n_mem > 1 \
                else torch.zeros_like(ent)
            maps.append(torch.stack([ent, ptr, cond, recent]).mean(1))  # mean over heads
        side = int(np.sqrt(q.size(-2)))
        return torch.cat(maps, -1).reshape(4, side, side).cpu().numpy()


def attn_feats(maps, m, prev_mask):
    """Memory-attention features per layer: global entropy, and statistics over
    the query positions covered by the predicted mask (previous mask if empty).

    maps has shape layers x 4 x 64 x 64 (see AttnRecorder).
    """
    side = maps.shape[-1]
    region = None
    for cand in (m, prev_mask):
        if cand is not None and cand.any():
            region = cv2.resize(cand.astype(np.float32), (side, side),
                                interpolation=cv2.INTER_AREA) > 0
            break
    f = {}
    for li, lm in enumerate(maps):
        f[f"attn{li}_ent"] = float(lm[0].mean())
        for i, name in enumerate(["ent_fg", "ptr_fg", "cond_fg", "recent_fg"]):
            f[f"attn{li}_{name}"] = float(lm[i][region].mean()) \
                if region is not None else np.nan
    return f


class LazyFrames:
    """Drop-in for the frame tensor of SAM 2's init_state, decoded on access.

    SAM 2 keeps every frame as float32 at 1024 x 1024 (12.6 MB each), and on
    the Jetson the "CPU offload" sits in the same memory as the GPU, so long
    videos do not fit. The arithmetic matches load_video_frames_from_jpg_images,
    so the outputs are identical.
    """

    def __init__(self, paths, image_size, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
        self.paths, self.image_size = paths, image_size
        self.mean = torch.tensor(mean, dtype=torch.float32)[:, None, None]
        self.std = torch.tensor(std, dtype=torch.float32)[:, None, None]

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img, _, _ = _load_img_as_tensor(self.paths[idx], self.image_size)
        img = img.float()
        img -= self.mean
        img /= self.std
        return img


def load_frames_lazy(video_path, image_size, offload_video_to_cpu, img_mean=(0.485, 0.456, 0.406),
                     img_std=(0.229, 0.224, 0.225), **kwargs):
    names = sorted((p for p in os.listdir(video_path)
                    if os.path.splitext(p)[-1].lower() in (".jpg", ".jpeg")),
                   key=lambda p: int(os.path.splitext(p)[0]))
    frames = LazyFrames([os.path.join(video_path, n) for n in names], image_size, img_mean, img_std)
    w, h = Image.open(frames.paths[0]).size
    return frames, h, w


class LazyGT:
    """Per-frame ground truth of one palette-PNG video, decoded on access."""

    def __init__(self, ann_dir, ids):
        self.paths = [os.path.join(ann_dir, f) for f in sorted(os.listdir(ann_dir))]
        self.ids = ids

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        p = np.array(Image.open(self.paths[idx]))
        return {i: p == i for i in self.ids}


def load_gt(video):
    """Ground truth as one {object id: boolean mask} dict per frame.

    Only objects visible on frame 0 are tracked, as in the SAM 2 VOS protocol.
    """
    ann_dir = os.path.join(DS["ann"], video)
    if DS.get("lazy") and not DS.get("per_obj"):  # SA-V (per-object PNGs) is short: load eagerly
        first = sorted(os.listdir(ann_dir))[0]
        p0 = np.array(Image.open(os.path.join(ann_dir, first)))
        return LazyGT(ann_dir, [int(i) for i in np.unique(p0) if i != 0])
    if DS.get("per_obj"):
        objs = sorted(os.listdir(ann_dir))
        frames = sorted(os.listdir(os.path.join(ann_dir, objs[0])))
        gts = [{int(o): np.array(Image.open(os.path.join(ann_dir, o, f))) > 0
                for o in objs} for f in frames]
    else:
        pals = [np.array(Image.open(os.path.join(ann_dir, f)))
                for f in sorted(os.listdir(ann_dir))]
        ids = [int(i) for i in np.unique(pals[0]) if i != 0]
        gts = [{i: p == i for i in ids} for p in pals]
    keep = [i for i, m in gts[0].items() if m.any()]
    return [{i: g[i] for i in keep} for g in gts]


def build_predictor(lazy=False):
    """SAM 2.1 / EdgeTAM video predictor, or the SAM 2-style tracker of SAM 3.

    SAM 3's tracker (`Sam3TrackerPredictor`) keeps SAM 2's state layout and
    `_forward_sam_heads`, but batches all objects (like EdgeTAM), works at
    1008 px with mean/std 0.5, takes point prompts in relative coordinates and
    a torch mask, and yields 5 values per frame. Its methods are overridden on
    the instance so run_video and HeadRecorder work unchanged. SAM3_RANDOM=1
    builds it without weights (plumbing test only).
    """
    if MODEL != "sam3":
        if lazy:
            sam2.sam2_video_predictor.load_video_frames = load_frames_lazy
        return build_sam2_video_predictor(CFG, CKPT, device="cuda")
    import sam3.model.sam3_tracking_predictor as s3p
    from sam3.model_builder import build_sam3_video_model
    if lazy:  # sam3 passes its own mean/std (0.5) to the loader
        s3p.load_video_frames = load_frames_lazy
    random = os.environ.get("SAM3_RANDOM") == "1"
    model = build_sam3_video_model(checkpoint_path=None if random else CKPT, load_from_HF=False)
    p = model.tracker
    p.backbone = model.detector.backbone  # the detector itself is not used
    prop, add_mask, add_pts = p.propagate_in_video, p.add_new_mask, p.add_new_points_or_box

    def propagate_in_video(state):
        for fidx, ids, _, video_res, _ in prop(state, start_frame_idx=0, max_frame_num_to_track=None,
                                               reverse=False, propagate_preflight=True):
            yield fidx, ids, video_res

    def add_new_mask(state, frame_idx, obj_id, mask):
        return add_mask(state, frame_idx, obj_id, torch.as_tensor(mask))

    def add_new_points_or_box(state, frame_idx, obj_id, points, labels):
        wh = np.array([state["video_width"], state["video_height"]], np.float32)
        return add_pts(state, frame_idx, obj_id, points=points / wh, labels=labels)

    p.propagate_in_video, p.add_new_mask = propagate_in_video, add_new_mask
    p.add_new_points_or_box = add_new_points_or_box
    p.reset_state = lambda state: None  # the state is dropped after each video
    return p


def center_click(mask):
    """Point inside the object that is farthest from its boundary."""
    dist = cv2.distanceTransform(mask.astype(np.uint8), cv2.DIST_L2, 5)
    y, x = np.unravel_index(np.argmax(dist), dist.shape)
    return np.array([[x, y]], dtype=np.float32)


def iou(a, b):
    union = np.logical_or(a, b).sum()
    return 1.0 if union == 0 else np.logical_and(a, b).sum() / union


def mask_feats(logits, prev_mask, first_area):
    """Ground-truth-free features computed from the predicted mask logits."""
    m = logits > 0
    area = m.sum()
    h, w = m.shape
    f = {"area_frac": area / (h * w)}
    # pixels near the decision boundary, relative to object size
    unc = (np.abs(logits) < 2.0).sum()
    f["boundary_unc"] = unc / max(area, 1)
    f["logit_margin"] = float(logits[m].mean()) if area > 0 else 0.0
    f["log_area_ratio_first"] = np.log((area + 1) / (first_area + 1))
    if prev_mask is not None:
        pa = prev_mask.sum()
        f["log_area_ratio_prev"] = np.log((area + 1) / (pa + 1))
        f["temporal_iou"] = iou(m, prev_mask)
        if area > 0 and pa > 0:
            c1 = np.argwhere(m).mean(0)
            c0 = np.argwhere(prev_mask).mean(0)
            f["centroid_shift"] = np.linalg.norm(c1 - c0) / np.sqrt(pa)
        else:
            f["centroid_shift"] = 0.0
    n_cc = cv2.connectedComponents(m.astype(np.uint8))[0] - 1 if area > 0 else 0
    f["n_components"] = n_cc
    return f, m


def cos_dist(a, b):
    return 1.0 - float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8))


def run_video(predictor, rec, arec, video, prompt):
    gts = load_gt(video)
    obj_ids = list(gts[0])
    if not obj_ids:  # no object on frame 0 (LVOS d83wYdy0): nothing to track
        return []
    state = predictor.init_state(
        video_path=os.path.join(DS["img"], video),
        offload_video_to_cpu=True,
    )
    for oid in obj_ids:
        gt0 = gts[0][oid]
        if prompt == "mask":
            predictor.add_new_mask(state, frame_idx=0, obj_id=oid, mask=gt0)
        else:
            predictor.add_new_points_or_box(
                state, frame_idx=0, obj_id=oid,
                points=center_click(gt0), labels=np.array([1], np.int32))

    rows = []
    prev_masks, prev_ptr, first_ptr, first_area = {}, {}, {}, {}
    rec.buf.clear()
    arec.buf.clear()
    for fidx, out_ids, logits in predictor.propagate_in_video(state):
        ious_per_obj = list(rec.buf)
        attn_per_obj = list(arec.buf)
        rec.buf.clear()
        arec.buf.clear()
        for k, oid in enumerate(out_ids):
            lg = logits[k, 0].float().cpu().numpy()
            gt = gts[fidx][oid]
            od = state["output_dict_per_obj"][k]
            key = "cond_frame_outputs" if fidx in od["cond_frame_outputs"] \
                else "non_cond_frame_outputs"
            cur = od[key][fidx]
            ptr = cur["obj_ptr"].float().cpu().numpy().reshape(-1)
            if fidx == 0:
                first_area[oid] = (lg > 0).sum()
                prev_masks[oid] = lg > 0
                prev_ptr[oid] = ptr
                continue
            # the pointer of a prompted frame differs from tracked ones, so the
            # first tracked frame is used as the reference
            first_ptr.setdefault(oid, ptr)
            f, m = mask_feats(lg, prev_masks.get(oid), first_area[oid])
            r = {"video": video, "obj": oid, "frame": fidx,
                 "n_frames": len(gts), "n_objs": len(obj_ids)}
            r.update(f)
            r["obj_score"] = float(cur["object_score_logits"].float().cpu().reshape(-1)[0])
            r["ptr_dist_prev"] = cos_dist(ptr, prev_ptr[oid])
            r["ptr_dist_first"] = cos_dist(ptr, first_ptr[oid])
            prev_ptr[oid] = ptr
            if len(ious_per_obj) == len(out_ids):
                ious = np.sort(ious_per_obj[k])[::-1]
                r["pred_iou"] = ious[0]
                r["pred_iou_gap"] = ious[0] - ious[1] if len(ious) > 1 else 0.0
                r["pred_iou_std"] = ious.std()
            if len(attn_per_obj) == len(out_ids):
                r.update(attn_feats(attn_per_obj[k], m, prev_masks.get(oid)))
            # ground truth (used only for labels)
            r["gt_present"] = int(gt.sum() > 0)
            r["pred_present"] = int(m.sum() > 0)
            r["gt_iou"] = iou(m, gt)
            rows.append(r)
            prev_masks[oid] = m
    predictor.reset_state(state)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=list(DATASETS), default="davis")
    ap.add_argument("--prompt", choices=["mask", "click"], default="mask")
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--lazy", action="store_true",
                    help="load frames and ground truth lazily (default for lvos)")
    ap.add_argument("--model", choices=list(MODELS), default="sam2",
                    help="edgetam: EdgeTAM (needs --no-attn; its memory is compressed by a perceiver); "
                         "sam3: the SAM 2-style tracker of SAM 3 (needs --no-attn)")
    ap.add_argument("--no-attn", action="store_true",
                    help="skip the memory-attention columns (halves run time)")
    args = ap.parse_args()

    global DS
    DS = dict(DATASETS[args.dataset])
    if args.lazy:
        DS["lazy"] = True
    if MODEL == "sam3" and not args.no_attn:
        ap.error("--model sam3 needs --no-attn (AttnRecorder hooks SAM 2's memory_attention)")
    torch.backends.cuda.matmul.allow_tf32 = True
    predictor = build_predictor(lazy=DS.get("lazy"))
    rec = HeadRecorder(predictor)
    # an empty buffer never matches the object count, so no attn columns
    arec = SimpleNamespace(buf=[]) if args.no_attn else AttnRecorder(predictor)
    videos = open(DS["list"]).read().split()
    if args.limit:
        videos = videos[: args.limit]

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    # resume: keep rows of videos already in the output file
    all_rows, done = [], set()
    if os.path.exists(args.out):
        prev = pd.read_csv(args.out, dtype={"video": str})
        all_rows, done = prev.to_dict("records"), set(prev.video.astype(str))
    t0 = time.time()
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        for i, v in enumerate(videos):
            ts = time.time()
            if v in done:
                continue
            rows = run_video(predictor, rec, arec, v, args.prompt)
            if not rows:
                print(f"[{i+1}/{len(videos)}] {v}: no object on frame 0, skipped", flush=True)
                continue
            all_rows += rows
            df = pd.DataFrame(rows)
            print(f"[{i+1}/{len(videos)}] {v}: {len(rows)} rows, "
                  f"mean IoU {df.gt_iou.mean():.3f}, {time.time()-ts:.1f}s", flush=True)
            pd.DataFrame(all_rows).to_csv(args.out, index=False)
    print(f"done in {time.time()-t0:.0f}s -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
