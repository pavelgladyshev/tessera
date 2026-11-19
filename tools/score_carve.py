#!/usr/bin/env python3
"""Score carved files against a test image's ground truth.

    python3 tools/score_carve.py IMAGE.truth.json CARVED_DIR [--min-bytes N]

For every file type: how many files the image holds, how many of them are at
least --min-bytes long (DECA's policy guarantees to find those; 512 KiB by
default), how many were carved exactly (same SHA-1), and how many carved
files match nothing. Standard library only.
"""
import argparse, collections, hashlib, json, os

EXT = {"jpg": "jpeg", "png": "png", "pdf": "pdf", "zip": "zip", "mp4": "mp4", "bmp": "bmp"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("truth")
    ap.add_argument("carved")
    ap.add_argument("--min-bytes", type=int, default=512 * 1024)
    a = ap.parse_args()
    truth = json.load(open(a.truth))["files"]
    want = {f["sha1"]: f for f in truth}
    n = collections.Counter(f["type"] for f in truth)
    big = collections.Counter(f["type"] for f in truth if f["size"] >= a.min_bytes)
    exact, exact_big, other = collections.Counter(), collections.Counter(), collections.Counter()
    carved = collections.Counter()
    for name in sorted(os.listdir(a.carved)):
        kind = EXT.get(name.rsplit(".", 1)[-1], "?")
        carved[kind] += 1
        h = hashlib.sha1(open(os.path.join(a.carved, name), "rb").read()).hexdigest()
        f = want.get(h)
        if f is not None and f["type"] == kind:
            exact[kind] += 1
            if f["size"] >= a.min_bytes:
                exact_big[kind] += 1
        else:
            other[kind] += 1
    print(f"{'type':6} {'in image':>8} {'>= min':>7} {'carved':>7} {'exact':>6} {'exact >= min':>13} {'not exact':>10}")
    for kind in ["jpeg", "png", "pdf", "zip", "mp4", "bmp"]:
        print(f"{kind:6} {n[kind]:8} {big[kind]:7} {carved[kind]:7} {exact[kind]:6} "
              f"{exact_big[kind]:6} of {big[kind]:<5} {other[kind]:10}")
    print(f"{'all':6} {sum(n.values()):8} {sum(big.values()):7} {sum(carved.values()):7} {sum(exact.values()):6} "
          f"{sum(exact_big.values()):6} of {sum(big.values()):<5} {sum(other.values()):10}")


if __name__ == "__main__":
    main()
