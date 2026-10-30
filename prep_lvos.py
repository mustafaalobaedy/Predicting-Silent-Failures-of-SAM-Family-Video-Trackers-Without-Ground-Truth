"""Check the unzipped LVOS v2 val set and write its video list.

Expects $SAMFAIL_DATA/LVOS2/valid/{JPEGImages,Annotations}/<video>/ with
one palette PNG per JPEG frame (6 fps), and the official meta JSON. Exits with
an error if the layout differs, so extraction does not start on bad data.

Usage:
    python prep_lvos.py
"""
from paths import DATA  # noqa: E402
import json
import os
import sys

from PIL import Image

ROOT = f"{DATA}/LVOS2/valid"
META = f"{DATA}/LVOS2/dl/meta/valid_meta.json"


def main():
    meta = json.load(open(META))["videos"]
    img, ann = os.path.join(ROOT, "JPEGImages"), os.path.join(ROOT, "Annotations")
    videos = sorted(os.listdir(img))
    errs = []
    if set(videos) != set(meta):
        errs.append(f"video set differs from meta: {len(videos)} vs {len(meta)}")
    for v in videos:
        jpg = sorted(os.path.splitext(f)[0] for f in os.listdir(os.path.join(img, v)))
        png = sorted(os.path.splitext(f)[0] for f in os.listdir(os.path.join(ann, v)))
        if jpg != png:
            errs.append(f"{v}: {len(jpg)} frames vs {len(png)} annotations")
            continue
        p0 = Image.open(os.path.join(ann, v, png[0] + ".png"))
        if p0.mode not in ("P", "L"):
            errs.append(f"{v}: annotation mode {p0.mode}")
    for e in errs[:20]:
        print(e)
    if errs:
        sys.exit(f"{len(errs)} problems; list not written")
    with open(os.path.join(ROOT, "valid.txt"), "w") as f:
        f.write("\n".join(videos) + "\n")
    print(f"ok: {len(videos)} videos -> {ROOT}/valid.txt")


if __name__ == "__main__":
    main()
