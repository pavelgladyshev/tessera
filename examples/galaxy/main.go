// galaxy simulates two disk galaxies colliding under their own gravity and
// draws the collision as an animated GIF.
//
// Every body attracts every other (a direct sum: n² pairs each step). The
// workers share the bodies out between them and compute the accelerations,
// on the card one worker per Tensix core; main, which stays ordinary Go,
// moves the bodies, draws the frames and writes the files.
//
// A worker has no floating point, so the physics is in fixed point: whole
// numbers of 1/32 of a pixel for positions, an integer square root, and a
// 64-bit division for 1/r³. That also makes go run and the card agree to
// the last bit: both print the same checksum and write the same pictures.
//
//	go run ./examples/galaxy                     # galaxy.gif and galaxy.png
//	go run ./examples/galaxy -n 2048 -steps 300  # smaller and shorter
package main

import (
	"encoding/binary"
	"flag"
	"fmt"
	"image"
	"image/color"
	"image/gif"
	"image/png"
	"math"
	"os"
	"path/filepath"
	"sort"

	"github.com/pavelgladyshev/tessera/tt"
)

const (
	maxBodies = 4096
	bound     = 16000     // bodies further out than this no longer count
	soft2     = 250 * 250 // softening: gravity levels off closer than 250 units
	chunk     = 128       // bodies read from the state at a time
	record    = 12        // bytes of a body in the state: x, y, mass
)

// Acc is the acceleration of one body, in 1/1024 of a unit per step².
type Acc struct {
	Body, X, Y int
}

// isqrt returns the whole part of the square root of v.
func isqrt(v uint32) uint32 {
	var r, bit uint32 = 0, 1 << 30
	for bit > v {
		bit >>= 2
	}
	for bit != 0 {
		if v >= r+bit {
			v -= r + bit
			r = r>>1 + bit
		} else {
			r >>= 1
		}
		bit >>= 2
	}
	return r
}

// body is a body as main keeps it: position and velocity in 1/1024 of a
// unit, mass in stars.
type body struct {
	x, y, vx, vy int64
	mass         int32
	galaxy       int
}

type rnd struct{ s uint64 }

func (r *rnd) float() float64 {
	r.s = r.s*6364136223846793005 + 1442695040888963407
	return float64(r.s>>11) / (1 << 53)
}

// disk makes a galaxy: a heavy core and stars on circular orbits around
// it, turning one way or the other (spin), all moving with (vx, vy).
func disk(r *rnd, galaxy, stars int, radius, cx, cy, vx, vy, spin, g float64) []body {
	core := float64(stars) * 3 // most of the mass, as a galaxy's bulge and halo
	radii := make([]float64, stars)
	for i := range radii {
		// Not too close to the core: an orbit must take many steps.
		radii[i] = radius * (0.2 + 0.8*math.Pow(r.float(), 0.7))
	}
	sort.Float64s(radii)
	bodies := []body{{
		x: int64(cx * 1024), y: int64(cy * 1024), vx: int64(vx * 1024), vy: int64(vy * 1024),
		mass: int32(core), galaxy: galaxy,
	}}
	for i, d := range radii {
		// The speed of a circular orbit around what lies inside it.
		v := math.Sqrt(g * (core + float64(i)) * d * d / math.Pow(d*d+soft2, 1.5))
		a := 2 * math.Pi * r.float()
		bodies = append(bodies, body{
			x:      int64((cx + d*math.Cos(a)) * 1024),
			y:      int64((cy + d*math.Sin(a)) * 1024),
			vx:     int64((vx - spin*v*math.Sin(a)) * 1024),
			vy:     int64((vy + spin*v*math.Cos(a)) * 1024),
			mass:   1,
			galaxy: galaxy,
		})
	}
	return bodies
}

// canvas draws the bodies with a glow and a fading trail.
type canvas struct {
	size    int
	trail   [2][]float32 // light from each galaxy
	palette color.Palette
}

func newCanvas(size int) *canvas {
	c := &canvas{size: size}
	for g := range c.trail {
		c.trail[g] = make([]float32, size*size)
	}
	// 16 levels of each galaxy's colour, mixed.
	for a := 0; a < 16; a++ {
		for b := 0; b < 16; b++ {
			fa, fb := float64(a)/15, float64(b)/15
			mix := func(ca, cb float64) uint8 {
				return uint8(min(255, 255*(fa*ca+fb*cb+0.55*fa*fa*fa+0.55*fb*fb*fb)))
			}
			c.palette = append(c.palette, color.RGBA{mix(0.35, 1.0), mix(0.6, 0.55), mix(1.0, 0.25), 255})
		}
	}
	return c
}

func (c *canvas) frame(bodies []body) *image.Paletted {
	n := c.size
	for g := range c.trail {
		for i := range c.trail[g] {
			c.trail[g][i] *= 0.72
		}
	}
	scale := float64(n) / (2 * 8192 * 1024)
	for _, b := range bodies {
		fx, fy := float64(b.x)*scale+float64(n)/2, float64(b.y)*scale+float64(n)/2
		x, y := int(math.Floor(fx)), int(math.Floor(fy))
		if x < 1 || y < 1 || x >= n-2 || y >= n-2 {
			continue
		}
		light := float32(0.5)
		if b.mass > 1 {
			light = 6
		}
		// Spread over the four pixels around the body.
		dx, dy := float32(fx-float64(x)), float32(fy-float64(y))
		t := c.trail[b.galaxy]
		t[y*n+x] += light * (1 - dx) * (1 - dy)
		t[y*n+x+1] += light * dx * (1 - dy)
		t[(y+1)*n+x] += light * (1 - dx) * dy
		t[(y+1)*n+x+1] += light * dx * dy
	}
	img := image.NewPaletted(image.Rect(0, 0, n, n), c.palette)
	level := func(t []float32, i int) int {
		// A little of the neighbours' light, as a glow.
		v := t[i] + 0.3*(t[i-1]+t[i+1]+t[i-n]+t[i+n])
		return int(15.99 * (1 - math.Exp(-0.9*float64(v))))
	}
	for y := 1; y < n-1; y++ {
		for x := 1; x < n-1; x++ {
			i := y*n + x
			img.Pix[i] = uint8(level(c.trail[0], i)*16 + level(c.trail[1], i))
		}
	}
	return img
}

func main() {
	n := flag.Int("n", 4096, "bodies, at most 4096")
	steps := flag.Int("steps", 600, "steps to simulate")
	every := flag.Int("every", 3, "steps between frames")
	size := flag.Int("size", 400, "width and height of the pictures, in pixels")
	out := flag.String("o", "galaxy", "the files to write: <name>.gif and <name>.png")
	workers := flag.Int("workers", 0, "workers (default: one per core)")
	flag.Parse()
	if *n < 16 || *n > maxBodies || *steps < 1 || *every < 1 || *size < 64 {
		fmt.Fprintln(os.Stderr, "galaxy: -n is 16 to 4096, -steps and -every at least 1, -size at least 64")
		os.Exit(2)
	}
	if *workers <= 0 {
		*workers = min(tt.Cores(), *n/8)
	}

	// A big galaxy and a smaller one, falling
	// past each other. g is the pull of one star, set so that the big
	// galaxy's edge takes about 350 steps to go round.
	big := *n * 2 / 3
	g := 36.0 * 36.0 * 2000 / (float64(big) * 4)
	r := &rnd{2026}
	bodies := disk(r, 0, big-1, 2000, -2200, -2100, 9, 0, 1, g)
	bodies = append(bodies, disk(r, 1, *n-big-1, 1300, 4200, 2700, -18, 0, 1, g)...)
	count := len(bodies)
	// The numerator of 1/r³, scaled so that the workers' sums keep 32 bits
	// below the point.
	pull := int(g * (1 << 42))

	dir, err := os.MkdirTemp("", "galaxy")
	if err != nil {
		fmt.Fprintln(os.Stderr, "galaxy:", err)
		os.Exit(1)
	}
	defer os.RemoveAll(dir)
	statePath := filepath.Join(dir, "state.bin")
	raw := make([]byte, count*record)

	draw := newCanvas(*size)
	anim := &gif.GIF{}
	var sheet []*image.Paletted // six frames for the still picture
	frames := (*steps + *every - 1) / *every
	for step := 0; step < *steps; step++ {
		// The state for the workers: where each body is, and its mass (0
		// for one that has left).
		for i, b := range bodies {
			x, y, m := b.x>>10, b.y>>10, b.mass
			if x < -bound || x > bound || y < -bound || y > bound {
				x, y, m = max(-bound, min(bound, x)), max(-bound, min(bound, y)), 0
			}
			binary.LittleEndian.PutUint32(raw[i*record:], uint32(int32(x)))
			binary.LittleEndian.PutUint32(raw[i*record+4:], uint32(int32(y)))
			binary.LittleEndian.PutUint32(raw[i*record+8:], uint32(m))
		}
		if err := os.WriteFile(statePath, raw, 0o600); err != nil {
			fmt.Fprintln(os.Stderr, "galaxy:", err)
			os.Exit(1)
		}
		state := tt.LoadBytes(statePath)
		share := *workers

		accs := tt.Parallel(share, func(w int, res *tt.List[Acc]) {
			var xs, ys, ms [maxBodies]int32
			var buf [chunk * record]byte
			for first := 0; first < count; first += chunk {
				got := state.Read(buf[:], first*record) / record
				for i := 0; i < got; i++ {
					b := buf[i*record : (i+1)*record]
					xs[first+i] = int32(uint32(b[0]) | uint32(b[1])<<8 | uint32(b[2])<<16 | uint32(b[3])<<24)
					ys[first+i] = int32(uint32(b[4]) | uint32(b[5])<<8 | uint32(b[6])<<16 | uint32(b[7])<<24)
					ms[first+i] = int32(uint32(b[8]) | uint32(b[9])<<8 | uint32(b[10])<<16 | uint32(b[11])<<24)
				}
			}
			// This worker's share of the bodies, each against all.
			for i := w * count / share; i < (w+1)*count/share; i++ {
				xi, yi := xs[i], ys[i]
				var ax, ay int64
				for j := 0; j < count; j++ {
					if ms[j] == 0 {
						continue
					}
					dx, dy := xs[j]-xi, ys[j]-yi
					r2 := uint32(dx*dx+dy*dy) + soft2
					q := int64(uint64(pull)/(uint64(r2)*uint64(isqrt(r2)))) * int64(ms[j])
					ax += int64(dx) * q
					ay += int64(dy) * q
				}
				res.Append(Acc{i, int(ax >> 32), int(ay >> 32)})
			}
		})

		for _, a := range accs {
			b := &bodies[a.Body]
			b.vx += int64(a.X)
			b.vy += int64(a.Y)
			b.x += b.vx
			b.y += b.vy
		}
		if step%*every == 0 {
			img := draw.frame(bodies)
			anim.Image = append(anim.Image, img)
			anim.Delay = append(anim.Delay, 4)
			if f := len(anim.Image) - 1; f*5%max(1, frames-1) < 5 && len(sheet) < 6 {
				sheet = append(sheet, img)
			}
		}
	}

	sum := uint64(0)
	for _, b := range bodies {
		sum = sum*1099511628211 + uint64(b.x) ^ uint64(b.y)<<1
	}
	fmt.Printf("%d bodies, %d steps, %d workers: checksum %016x\n", count, *steps, *workers, sum)

	f, err := os.Create(*out + ".gif")
	if err == nil {
		err = gif.EncodeAll(f, anim)
		f.Close()
	}
	if err == nil {
		// A still: six moments side by side.
		still := image.NewPaletted(image.Rect(0, 0, 3**size, 2**size), draw.palette)
		for k, img := range sheet {
			for y := 0; y < *size; y++ {
				copy(still.Pix[(k/3**size+y)*still.Stride+k%3**size:], img.Pix[y**size:(y+1)**size])
			}
		}
		if f, err = os.Create(*out + ".png"); err == nil {
			err = png.Encode(f, still)
			f.Close()
		}
	}
	if err != nil {
		fmt.Fprintln(os.Stderr, "galaxy:", err)
		os.Exit(1)
	}
	fmt.Printf("wrote %s.gif (%d frames) and %s.png\n", *out, len(anim.Image), *out)
}
