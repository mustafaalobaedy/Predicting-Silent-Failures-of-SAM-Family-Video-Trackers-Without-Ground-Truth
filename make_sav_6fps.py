"""Link the labelled (6 fps) SA-V val frames into JPEGImages_6fps.

SA-V val frames are extracted at 24 fps but labelled every 4th frame. Tracking
on the labelled frames only gives a label for every tracked frame and cuts
compute by 4x; frame names are kept, so SAM 2 sorts them correctly.
"""
from paths import DATA  # noqa: E402
import os

ROOT = f"{DATA}/SA-V/sav_val"

for v in open(os.path.join(ROOT, "sav_val.txt")).read().split():
    ann = os.path.join(ROOT, "Annotations_6fps", v)
    obj = sorted(os.listdir(ann))[0]
    out = os.path.join(ROOT, "JPEGImages_6fps", v)
    os.makedirs(out, exist_ok=True)
    for f in sorted(os.listdir(os.path.join(ann, obj))):
        jpg = f.replace(".png", ".jpg")
        dst = os.path.join(out, jpg)
        if not os.path.exists(dst):
            os.symlink(os.path.join(ROOT, "JPEGImages_24fps", v, jpg), dst)
print("done")
