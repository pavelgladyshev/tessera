---
title: Design notes
nav_order: 6
has_children: true
---

# Design notes

How Tessera works inside, and what was measured while it was built. These
are working notes, in the order things happened, kept because they record
why things are as they are.

| Page | What is in it |
|---|---|
| [The ttnn bridge](ttnn-bridge.md) | how `main` calls ttnn through the card helper, the decisions behind it, and its measurements |
| [Development notes](notes.md) | everything else: the kernels, the stack, channels, the fabric, each with what was measured on the card |

What Go features cost on the card is in
[the benchmarks](https://github.com/pavelgladyshev/tessera/tree/main/bench).
