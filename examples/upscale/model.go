package main

// The network: Real-ESRGAN's compact model (SRVGGNetCompact). A 3×3
// convolution to 64 channels and a PReLU, 32 more of each, a convolution to
// 48 channels, which are the 4×4 pixels of each place in three colours,
// and the original enlarged by repeating pixels added on.

import (
	"encoding/binary"
	"fmt"
	"image"
	"image/draw"
	"io"
	"os"

	"github.com/pavelgladyshev/tessera/tt/nn"
)

// layer is a convolution (w has four dimensions, b is its biases) or a
// PReLU (w has one: the slopes).
type layer struct {
	w, b nn.Tensor
	conv bool
}

type network struct {
	layers []layer
}

// loadNet reads the file tools/export_weights.py writes: "TSNN1", then
// tensors, each a name, a shape and float32 values.
func loadNet(path string) (*network, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, fmt.Errorf("%v (make it with tools/export_weights.py, see the top of examples/upscale/main.go)", err)
	}
	defer f.Close()
	head := make([]byte, 6)
	if _, err := io.ReadFull(f, head); err != nil || string(head) != "TSNN1\n" {
		return nil, fmt.Errorf("%s is not a file of tensors", path)
	}
	u32 := func() (uint32, error) {
		var v uint32
		err := binary.Read(f, binary.LittleEndian, &v)
		return v, err
	}
	tensors := map[string]nn.Tensor{}
	var names []string
	for {
		n, err := u32()
		if err == io.EOF {
			break
		}
		if err != nil || n > 256 {
			return nil, fmt.Errorf("%s is damaged", path)
		}
		name := make([]byte, n)
		io.ReadFull(f, name)
		dims, _ := u32()
		if dims < 1 || dims > 4 {
			return nil, fmt.Errorf("%s: %s has %d dimensions", path, name, dims)
		}
		shape, count := make([]int, dims), 1
		for i := range shape {
			d, _ := u32()
			shape[i] = int(d)
			count *= int(d)
		}
		v := make([]float32, count)
		if err := binary.Read(f, binary.LittleEndian, v); err != nil {
			return nil, fmt.Errorf("%s ends in the middle of %s", path, name)
		}
		tensors[string(name)] = nn.FromFloats(shape, v)
		names = append(names, string(name))
	}
	net := &network{}
	for i := 0; ; i++ {
		w, ok := tensors[fmt.Sprintf("body.%d.weight", i)]
		if !ok {
			break
		}
		l := layer{w: w, conv: len(w.Shape()) == 4}
		if l.conv {
			if l.b, ok = tensors[fmt.Sprintf("body.%d.bias", i)]; !ok {
				return nil, fmt.Errorf("%s: convolution %d has no biases", path, i)
			}
		}
		net.layers = append(net.layers, l)
	}
	if len(net.layers) == 0 || !net.layers[len(net.layers)-1].conv ||
		net.layers[len(net.layers)-1].w.Shape()[0] != 3*scale*scale || net.layers[0].w.Shape()[1] != 3 {
		return nil, fmt.Errorf("%s is not the compact x%d model (%d tensors, the first %v)", path, scale, len(names), names[:min(3, len(names))])
	}
	return net, nil
}

func toRGBA(img image.Image) *image.RGBA {
	out := image.NewRGBA(image.Rect(0, 0, img.Bounds().Dx(), img.Bounds().Dy()))
	draw.Draw(out, out.Rect, img, img.Bounds().Min, draw.Src)
	return out
}

// enlarge runs the network on a picture.
func (net *network) enlarge(src *image.RGBA) *image.RGBA {
	w, h := src.Rect.Dx(), src.Rect.Dy()
	in := make([]float32, w*h*3)
	for i := 0; i < w*h; i++ {
		for c := 0; c < 3; c++ {
			in[i*3+c] = float32(src.Pix[i*4+c]) / 255
		}
	}
	x := nn.FromFloats([]int{1, h, w, 3}, in)
	for _, l := range net.layers {
		var y nn.Tensor
		if l.conv {
			y = nn.Conv2d(x, l.w, l.b, nn.Conv{Padding: [2]int{1, 1}})
		} else {
			y = nn.PReLU(x, l.w)
		}
		x.Free()
		x = y
	}
	out := x.Floats() // [h w 48]: channel c*16 + i*4 + j is colour c of pixel (i, j) of the place
	x.Free()
	img := image.NewRGBA(image.Rect(0, 0, w*scale, h*scale))
	for y := 0; y < h*scale; y++ {
		for xx := 0; xx < w*scale; xx++ {
			at := (y*w*scale + xx) * 4
			from := (y/scale*w + xx/scale)
			for c := 0; c < 3; c++ {
				v := out[from*3*scale*scale+c*scale*scale+y%scale*scale+xx%scale] + in[from*3+c]
				img.Pix[at+c] = uint8(max(0, min(255, v*255+0.5)))
			}
			img.Pix[at+3] = 255
		}
	}
	return img
}
