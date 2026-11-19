"""Export an ndeca PatchMLP checkpoint (.pt) to Tessera's detector format.

    python export_ndeca.py MODEL.pt OUT.tsw [--check-blocks IMAGE N]

The .tsw file holds the network exactly as trained, in float32:

    "TSW1"
    u32 block_size, n_classes, emb_dim, c1, c2, c3, c4, prefix_dim, hidden
    n_classes x (u32 length, bytes)                 class names
    15 x (u32 count, count x f32)                   arrays, [in][out] order:
        emb[256][D]  l1.w[8D][c1] l1.b  l2.w[2c1][c2] l2.b  l3.w[2c2][c3] l3.b
        l4.w[2c3][c4] l4.b  lp.w[32D][P] lp.b  h1.w[pooled+P+256][H] h1.b
        h2.w[H][n_classes] h2.b

With --check-blocks it also writes OUT.check: N blocks read at random 4 KiB
offsets of IMAGE (non-zero ones), each followed by PyTorch's logits, so that
the Go and card implementations can be compared with the original.
"""
import argparse
import os
import struct
import sys

import numpy as np
import torch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("out")
    ap.add_argument("--check-blocks", nargs=2, metavar=("IMAGE", "N"))
    ap.add_argument("--ndeca", default=os.path.expanduser("~/ndeca"), help="the ndeca checkout (for its model code)")
    a = ap.parse_args()

    sys.path.insert(0, a.ndeca)
    from ndeca.model import load_model, featurize

    model, classes, _ = load_model(a.model)
    cfg = model.cfg
    if model.pooling != "multi" or model.front_end != "byte":
        sys.exit("only the byte front end with multi-scale pooling is supported")

    def lin(name):
        layer = getattr(model, name)
        return layer.weight.detach().float().numpy().T.copy(), layer.bias.detach().float().numpy().copy()

    arrays = [model.emb.weight.detach().float().numpy()]
    for name in ("l1", "l2", "l3", "l4", "lp", "h1", "h2"):
        w, b = lin(name)
        arrays += [w, b]

    with open(a.out, "wb") as f:
        f.write(b"TSW1")
        hdr = [cfg["block_size"], len(classes), cfg["emb_dim"], cfg["c1"], cfg["c2"], cfg["c3"], cfg["c4"],
               cfg["prefix_dim"], cfg["hidden"]]
        f.write(struct.pack("<9I", *hdr))
        for c in classes:
            f.write(struct.pack("<I", len(c)) + c.encode())
        for arr in arrays:
            flat = np.ascontiguousarray(arr, dtype="<f4").ravel()
            f.write(struct.pack("<I", flat.size))
            f.write(flat.tobytes())
    print(f"wrote {a.out}: block {cfg['block_size']}, {len(classes)} classes, "
          f"{sum(x.size for x in arrays)} parameters")

    if a.check_blocks:
        image, n = a.check_blocks[0], int(a.check_blocks[1])
        L = cfg["block_size"]
        size = os.path.getsize(image)
        rng = np.random.default_rng(7)
        blocks = []
        with open(image, "rb") as img:
            while len(blocks) < n:
                off = int(rng.integers(0, size // L)) * L
                img.seek(off)
                b = img.read(L)
                if len(b) == L and any(b):
                    blocks.append(np.frombuffer(b, dtype=np.uint8))
        blocks = np.stack(blocks)
        with torch.no_grad():
            idx, hist = featurize(blocks)
            logits = model(torch.from_numpy(idx).long(), torch.from_numpy(hist)).float().numpy()
        with open(a.out + ".check", "wb") as f:
            f.write(struct.pack("<3I", n, L, len(classes)))
            for b, lg in zip(blocks, logits):
                f.write(b.tobytes())
                f.write(np.ascontiguousarray(lg, dtype="<f4").tobytes())
        labels = logits.argmax(1)
        print(f"wrote {a.out}.check: {n} blocks; labels:",
              {classes[k]: int(v) for k, v in zip(*np.unique(labels, return_counts=True))})


if __name__ == "__main__":
    main()
