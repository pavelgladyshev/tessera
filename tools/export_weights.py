#!/usr/bin/env python3
"""Turns a PyTorch checkpoint into the plain file of tensors that
examples/upscale reads.

    python3 tools/export_weights.py realesr-general-x4v3.pth upscale.tsnn

The checkpoint is a state_dict, or a dict with one under "params_ema" or
"params" (as the Real-ESRGAN project's are). The output is the line
"TSNN1", then for every tensor: its name's length and the name, the number
of dimensions and the dimensions (all uint32, little endian), and its
values as float32.
"""
import struct
import sys

import torch


def main():
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    state = torch.load(sys.argv[1], map_location="cpu", weights_only=True)
    for key in ("params_ema", "params", "state_dict"):
        if isinstance(state, dict) and key in state:
            state = state[key]
            break
    with open(sys.argv[2], "wb") as out:
        out.write(b"TSNN1\n")
        for name, tensor in state.items():
            t = tensor.detach().to(torch.float32).contiguous()
            raw = name.encode()
            out.write(struct.pack("<I", len(raw)) + raw)
            out.write(struct.pack("<I", t.dim()) + struct.pack("<%dI" % t.dim(), *t.shape))
            out.write(t.numpy().astype("<f4").tobytes())
            print(f"{name:24s} {tuple(t.shape)}")


if __name__ == "__main__":
    main()
