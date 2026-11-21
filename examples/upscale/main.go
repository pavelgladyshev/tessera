// upscale is a web server that makes pictures four times bigger with a
// neural network, Real-ESRGAN's compact model (32 convolutions of 64
// channels), and shows the result beside a plain enlargement.
//
// Both halves of Tessera are at work. The network runs through package
// tt/nn: on the card ttnn's convolutions, under go run plain float32 (slow:
// minutes for a big picture). The plain enlargement, bilinear, is done by
// workers in fixed point, on the card one worker per Tensix core, each with
// its share of the rows.
//
//	go run ./examples/upscale -model upscale.tsnn     # then open http://localhost:8080
//	go run ./examples/upscale -model upscale.tsnn -demo out.png
//
// The model file comes from the project's weights (realesr-general-x4v3.pth,
// from github.com/xinntao/Real-ESRGAN's releases) through
// tools/export_weights.py.
package main

import (
	"flag"
	"fmt"
	"image"
	"image/png"
	"math"
	"os"
	"path/filepath"
	"time"

	"github.com/pavelgladyshev/tessera/tt"
)

const (
	scale   = 4
	maxSide = 256 // of a picture going in; bigger ones are reduced first
)

// Pixels is eight pixels of the workers' enlargement, three bytes each,
// packed into three integers. At is the first one's place: row × width +
// column.
type Pixels struct {
	At, A, B, C int
}

// lerp mixes a and b: t is 0 to 256.
func lerp(a, b byte, t int) int {
	return (int(a)*(256-t) + int(b)*t + 128) >> 8
}

// job is one picture to enlarge. The web server's handlers and -demo make
// jobs; main does them, one at a time, because tt.Parallel is main's to
// call and the card does one thing at a time.
type job struct {
	src            *image.RGBA
	plain, network *image.RGBA
	workers, net   time.Duration
	done           chan struct{}
}

// do hands a job to main and waits for it.
func do(jobs chan<- *job, src *image.RGBA) *job {
	j := &job{src: src, done: make(chan struct{})}
	jobs <- j
	<-j.done
	return j
}

// rawFile writes a picture's pixels, three bytes each, to a file for the
// workers to read.
func rawFile(dir string, src *image.RGBA) string {
	w, h := src.Rect.Dx(), src.Rect.Dy()
	raw := make([]byte, w*h*3)
	for i := 0; i < w*h; i++ {
		copy(raw[i*3:], src.Pix[i*4:i*4+3])
	}
	path := filepath.Join(dir, "picture.rgb")
	if err := os.WriteFile(path, raw, 0o600); err != nil {
		fmt.Fprintln(os.Stderr, "upscale:", err)
		os.Exit(1)
	}
	return path
}

// unpack puts the workers' pixels together.
func unpack(parts []Pixels, ow, oh int) *image.RGBA {
	img := image.NewRGBA(image.Rect(0, 0, ow, oh))
	for _, p := range parts {
		words := [3]int{p.A, p.B, p.C}
		for i := 0; i < 8 && p.At%ow+i < ow; i++ {
			at := (p.At + i) * 4
			for c := 0; c < 3; c++ {
				b := i*3 + c
				img.Pix[at+c] = byte(words[b/8] >> (8 * (b % 8)))
			}
			img.Pix[at+3] = 255
		}
	}
	return img
}

// runDemo enlarges the sample picture and says what came of it.
func runDemo(jobs chan<- *job, layers, size int, path string) {
	j := do(jobs, sample(size))
	fmt.Fprintf(os.Stderr, "upscale: the network took %.2f s, the workers %.2f s\n", j.net.Seconds(), j.workers.Seconds())
	// How much the two enlargements differ says the network did
	// something; the checksum of the workers' one is exact.
	sum, diff := 0, 0
	for i, p := range j.plain.Pix {
		sum = (sum*31 + int(p)) % 1000000007
		diff += int(math.Abs(float64(p) - float64(j.network.Pix[i])))
	}
	if err := writePNG(path, j.network); err != nil {
		fmt.Fprintln(os.Stderr, "upscale:", err)
		os.Exit(1)
	}
	fmt.Printf("%d layers; %dx%d to %dx%d\n", layers, size, size, j.network.Rect.Dx(), j.network.Rect.Dy())
	fmt.Printf("bilinear, by the workers: checksum %d\n", sum)
	fmt.Printf("the network's picture differs from it: %v\n", diff > len(j.plain.Pix))
	close(jobs)
}

// sample draws a small picture to try the network on: shapes with edges
// and fine lines, which a plain enlargement blurs.
func sample(size int) *image.RGBA {
	img := image.NewRGBA(image.Rect(0, 0, size, size))
	for y := 0; y < size; y++ {
		for x := 0; x < size; x++ {
			u, v := float64(x)/float64(size), float64(y)/float64(size)
			r, g, b := 0.10+0.5*v, 0.16+0.25*u, 0.32+0.3*(1-v)
			if d := math.Hypot(u-0.32, v-0.36); d < 0.2 {
				r, g, b = 0.96, 0.72-0.5*d, 0.2
			}
			if d := math.Hypot(u-0.68, v-0.62); d < 0.22 && d > 0.15 {
				r, g, b = 0.3, 0.85, 0.9
			}
			if s := u + v; s > 1.18 && s < 1.9 && int((u-v)*float64(size)/3+64)%2 == 0 {
				r, g, b = 0.95, 0.95, 0.9
			}
			if u > 0.08 && u < 0.5 && v > 0.74 && v < 0.9 && (int(u*float64(size))/2+int(v*float64(size))/2)%2 == 0 {
				r, g, b = 0.9, 0.3, 0.45
			}
			at := (y*size + x) * 4
			img.Pix[at], img.Pix[at+1], img.Pix[at+2], img.Pix[at+3] = uint8(255*r), uint8(255*g), uint8(255*b), 255
		}
	}
	return img
}

func writePNG(path string, img image.Image) error {
	f, err := os.Create(path)
	if err != nil {
		return err
	}
	defer f.Close()
	return png.Encode(f, img)
}

func readPNG(path string) (*image.RGBA, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()
	img, err := png.Decode(f)
	if err != nil {
		return nil, err
	}
	return toRGBA(img), nil
}

func main() {
	modelPath := flag.String("model", "upscale.tsnn", "the network's weights, from tools/export_weights.py")
	addr := flag.String("addr", "localhost:8080", "where the server listens")
	demo := flag.String("demo", "", "instead of serving: enlarge the sample picture, write it to this PNG file and stop")
	size := flag.Int("size", 48, "with -demo: the sample picture's width and height")
	compare := flag.Bool("compare", false, "instead of serving: say how far apart the two PNG files named after the flags are")
	flag.Parse()

	if *compare {
		a, errA := readPNG(flag.Arg(0))
		b, errB := readPNG(flag.Arg(1))
		if errA != nil || errB != nil || a.Rect != b.Rect {
			fmt.Println("the pictures cannot be compared")
			os.Exit(1)
		}
		sum := 0
		for i := range a.Pix {
			sum += int(math.Abs(float64(a.Pix[i]) - float64(b.Pix[i])))
		}
		mean := float64(sum) / float64(len(a.Pix))
		if mean > 4 {
			fmt.Printf("the pictures differ by %.2f levels of 255 on average\n", mean)
			os.Exit(1)
		}
		fmt.Println("the pictures agree within 4 levels of 255 on average")
		return
	}

	net, err := loadNet(*modelPath)
	if err != nil {
		fmt.Fprintln(os.Stderr, "upscale:", err)
		os.Exit(1)
	}
	dir, err := os.MkdirTemp("", "upscale")
	if err != nil {
		fmt.Fprintln(os.Stderr, "upscale:", err)
		os.Exit(1)
	}
	defer os.RemoveAll(dir)
	jobs := make(chan *job)
	if *demo != "" {
		start(func() { runDemo(jobs, len(net.layers), *size, *demo) })
	} else {
		start(func() { serve(*addr, len(net.layers), jobs) })
	}

	for j := range jobs {
		w, h := j.src.Rect.Dx(), j.src.Rect.Dy()
		ow, oh := w*scale, h*scale
		began := time.Now()
		pic := tt.LoadBytes(rawFile(dir, j.src))
		// A worker can report 4096 times eight pixels.
		n := max(min(tt.Cores(), oh), (oh*((ow+7)/8)+4095)/4096)

		// The plain enlargement: every pixel of the result is a mix of
		// the four around its place in the original. Each worker does its
		// share of the rows.
		parts := tt.Parallel(n, func(k int, out *tt.List[Pixels]) {
			var above, below [maxSide * 3]byte
			var px [24]byte
			for y := oh * k / n; y < oh*(k+1)/n; y++ {
				// Where this row falls in the original, in 256ths of a row.
				sy := max(0, (2*y+1)*256/(2*scale)-128)
				y0 := min(sy>>8, h-1)
				y1 := min(y0+1, h-1)
				pic.Read(above[:w*3], y0*w*3)
				pic.Read(below[:w*3], y1*w*3)
				for x := 0; x < ow; x += 8 {
					for i := 0; i < 8; i++ {
						sx := max(0, (2*min(x+i, ow-1)+1)*256/(2*scale)-128)
						x0 := min(sx>>8, w-1)
						x1 := min(x0+1, w-1)
						for c := 0; c < 3; c++ {
							top := lerp(above[x0*3+c], above[x1*3+c], sx&255)
							bottom := lerp(below[x0*3+c], below[x1*3+c], sx&255)
							px[i*3+c] = byte((top*(256-sy&255) + bottom*(sy&255) + 128) >> 8)
						}
					}
					var p [3]int
					for i := 0; i < 24; i++ {
						p[i/8] |= int(px[i]) << (8 * (i % 8))
					}
					out.Append(Pixels{y*ow + x, p[0], p[1], p[2]})
				}
			}
		})

		j.plain = unpack(parts, ow, oh)
		j.workers = time.Since(began)
		began = time.Now()
		j.network = net.enlarge(j.src)
		j.net = time.Since(began)
		close(j.done)
	}
}

// start runs f beside main.
func start(f func()) { go f() }
