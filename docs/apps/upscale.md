---
title: Upscale
parent: Sample applications
nav_order: 3
---

# Upscale: bigger pictures with a neural network

A web server: drop a picture on its page and it comes back four times
bigger, enlarged by a neural network, beside a plain enlargement to compare
it with. The source is
[`examples/upscale`](https://github.com/pavelgladyshev/tessera/tree/main/examples/upscale).

| Bilinear, by the workers | The network, through ttnn |
|---|---|
| ![Bilinear enlargement](../images/upscale-bilinear.png) | ![The network's enlargement](../images/upscale-network.png) |

Both are the same 96 × 96 picture, enlarged to 384 × 384 on the card.

## Run it

The network's weights are not in the repository. Download
`realesr-general-x4v3.pth` from the
[Real-ESRGAN project's releases](https://github.com/xinntao/Real-ESRGAN/releases)
and convert it (this needs PyTorch):

```bash
python3 tools/export_weights.py realesr-general-x4v3.pth upscale.tsnn
```

Then:

```bash
go run ./examples/upscale -model upscale.tsnn                 # open http://localhost:8080
go run ./examples/upscale -model upscale.tsnn -demo out.png   # no server: one sample picture
```

On the card: `build/upscale/build/upscale -model upscale.tsnn`.

## How it works

Both halves of Tessera do a part.

* **The network runs through `tt/nn`.** It is Real-ESRGAN's compact model:
  a 3 × 3 convolution to 64 channels, 32 more of them, each followed by a
  PReLU, and a last convolution to 48 channels, which are the 4 × 4 pixels
  of each place in three colours. `main` is a loop over the layers:

  ```go
  for _, l := range net.layers {
  	if l.conv {
  		y = nn.Conv2d(x, l.w, l.b, nn.Conv{Padding: [2]int{1, 1}})
  	} else {
  		y = nn.PReLU(x, l.w)
  	}
  	x.Free()
  	x = y
  }
  ```

* **The bilinear enlargement is done by the workers,** in fixed point.
  Each takes a share of the rows of the result and reports eight pixels at
  a time, packed into three integers.
* **main is the web server,** with Go's `net/http`. Handlers put pictures
  in a queue, and `main` enlarges them one at a time, because
  `tt.Parallel` is `main`'s to call and the card does one thing at a time.

## What it measured

| | |
|---|---|
| The network, 96 × 96 to 384 × 384 | 45 ms |
| The network, 240 × 180 to 960 × 720 | 73 ms |
| The first picture of a size | about 2 s (ttnn builds its programs for the size) |
| The workers' bilinear | about 65 ms |
| Card against CPU | the network's pictures differ by about 2 levels of 255 on average (bf16 through 34 layers) |
| Under `go run` | about 3 s for a 48 × 48 picture; minutes for a large one |

## Limits

Pictures longer than 256 pixels on a side are reduced first: there is no
tiling. Video is not attempted.
