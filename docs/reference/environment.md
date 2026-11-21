---
title: Environment
parent: Reference
nav_order: 5
---

# Environment

The installer's `~/.config/tessera/env.sh` sets the first three.

| Variable | Meaning |
|---|---|
| `TT_METAL_HOME` | tt-metal's checkout, for building projects |
| `TT_METAL_RUNTIME_ROOT` | the same, for running them |
| `TT_METAL_CACHE` | where tt-metal keeps compiled kernels |
| `TESSERA_TIMEOUT=seconds` | stop a program whose workers run longer than this. Without it the helper still says so after 10 s, 30 s, and then every minute |
| `TESSERA_CORES=N` | what `tt.Cores()` returns under `go run` (128 by default, as on an n300) |
| `TESSERA_CHIPS=1` | use only the card's first chip |
| `TESSERA_DEVICE=N` | use chip N alone |
| `TESSERA_FABRIC=0` | have the host relay channels between chips instead of the fabric |
| `TESSERA_L1_SMALL=bytes` | each core's memory set aside for ttnn's convolutions and pools (32768 in programs that call them, 0 in others) |
| `TESSERA_OPT=s` | build kernels for size from the start (Tessera falls back to that when a kernel is too big) |
| `TESSERA_COMPILE_ONLY=1` | build a worker program's kernels and stop |
| `TESSERA_CARD`, `TESSERA_KERNELS` | another card helper, another kernel directory |
