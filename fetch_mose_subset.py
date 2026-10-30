"""Stream MOSE-v1 train from Hugging Face and keep a random subset of videos.

The 22 GB train.tar.gz is read once over HTTP; only the chosen videos are
written to disk.

Usage:
    python fetch_mose_subset.py --n 200 --seed 0 --root $SAMFAIL_DATA/MOSE
"""
from paths import DATA  # noqa: E402
import argparse
import json
import os
import random
import tarfile

from remotezip import RemoteZip

URL = "https://huggingface.co/datasets/FudanCVL/MOSE/resolve/main/MOSE_release.zip"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--root", default=f"{DATA}/MOSE")
    args = ap.parse_args()

    meta = json.load(open(os.path.join(args.root, "MOSE_release/meta_train.json")))
    vids = sorted(meta["videos"])
    keep = set(random.Random(args.seed).sample(vids, args.n))
    with open(os.path.join(args.root, f"subset_n{args.n}_seed{args.seed}.txt"), "w") as f:
        f.write("\n".join(sorted(keep)) + "\n")

    z = RemoteZip(URL)
    tar = tarfile.open(fileobj=z.open("MOSE_release/train.tar.gz"), mode="r|gz")
    n_files = 0
    for k, m in enumerate(tar):
        parts = m.name.split("/")
        if len(parts) == 4 and m.isfile() and parts[2] in keep \
                and not parts[3].startswith("._"):
            tar.extract(m, args.root, filter="data")
            n_files += 1
        if k % 20000 == 0:
            print(f"{k} members read, {n_files} kept, at {m.name}", flush=True)
    print(f"done: {n_files} files", flush=True)


if __name__ == "__main__":
    main()
