package main

// A disk image to carve, for when none is given: pictures and documents
// drawn here, put at sector boundaries among zeros, noise and text, with a
// few decoys that start like a file and are not one. All of it is main's
// business, in ordinary Go.

import (
	"archive/zip"
	"bytes"
	"encoding/binary"
	"fmt"
	"image"
	"image/color"
	"image/gif"
	"image/jpeg"
	"image/png"
	"math"
	"os"
)

type lcg struct{ s uint64 }

func (r *lcg) next(n int) int {
	r.s = r.s*6364136223846793005 + 1442695040888963407
	return int(r.s>>33) % n
}

// picture draws one of a few patterns, in colours that depend on seed.
func picture(w, h, style, seed int) *image.RGBA {
	img := image.NewRGBA(image.Rect(0, 0, w, h))
	ph := float64(seed) * 0.7
	shade := func(t float64) color.RGBA {
		return color.RGBA{
			uint8(127.5 + 127.5*math.Cos(6.283*(t+ph*0.11))),
			uint8(127.5 + 127.5*math.Cos(6.283*(t+0.33+ph*0.07))),
			uint8(127.5 + 127.5*math.Cos(6.283*(t+0.67+ph*0.05))),
			255,
		}
	}
	for y := 0; y < h; y++ {
		for x := 0; x < w; x++ {
			u, v := (float64(x)/float64(w)-0.5)*2, (float64(y)/float64(h)-0.5)*2*float64(h)/float64(w)
			t := 0.0
			switch style % 5 {
			case 0: // plasma
				t = (math.Sin(u*5+ph) + math.Sin(v*7-ph) + math.Sin((u+v)*4+ph) + math.Sin(math.Hypot(u, v)*9)) / 8
			case 1: // the Mandelbrot set
				a, b := u*1.4-0.6, v*1.4
				zr, zi, i := 0.0, 0.0, 0
				for ; i < 80 && zr*zr+zi*zi < 4; i++ {
					zr, zi = zr*zr-zi*zi+a, 2*zr*zi+b
				}
				t = float64(i) / 80
			case 2: // a Julia set
				zr, zi, i := u*1.5, v*1.5, 0
				cr, ci := -0.8+0.05*math.Sin(ph), 0.156+0.05*math.Cos(ph)
				for ; i < 90 && zr*zr+zi*zi < 4; i++ {
					zr, zi = zr*zr-zi*zi+cr, 2*zr*zi+ci
				}
				t = math.Sqrt(float64(i) / 90)
			case 3: // rings
				t = math.Hypot(u+0.3*math.Sin(ph), v) * 3
			case 4: // a spiral
				t = math.Atan2(v, u)/6.283 + math.Hypot(u, v)*1.5
			}
			img.SetRGBA(x, y, shade(t))
		}
	}
	return img
}

func encodeBMP(img *image.RGBA) []byte {
	w, h := img.Rect.Dx(), img.Rect.Dy()
	row := (w*3 + 3) &^ 3
	b := make([]byte, 54+row*h)
	copy(b, "BM")
	binary.LittleEndian.PutUint32(b[2:], uint32(len(b)))
	binary.LittleEndian.PutUint32(b[10:], 54)
	binary.LittleEndian.PutUint32(b[14:], 40)
	binary.LittleEndian.PutUint32(b[18:], uint32(w))
	binary.LittleEndian.PutUint32(b[22:], uint32(h))
	binary.LittleEndian.PutUint16(b[26:], 1)
	binary.LittleEndian.PutUint16(b[28:], 24)
	binary.LittleEndian.PutUint32(b[34:], uint32(row*h))
	for y := 0; y < h; y++ {
		for x := 0; x < w; x++ {
			c := img.RGBAAt(x, h-1-y)
			copy(b[54+y*row+x*3:], []byte{c.B, c.G, c.R})
		}
	}
	return b
}

func encodeGIF(w, h, style, seed int) []byte {
	// A few frames, each a recoloured picture, on a palette of 64.
	pal := make(color.Palette, 64)
	for i := range pal {
		t := float64(i) / 64
		pal[i] = color.RGBA{uint8(255 * t), uint8(255 * (1 - math.Abs(2*t-1))), uint8(255 * (1 - t)), 255}
	}
	g := &gif.GIF{}
	for f := 0; f < 4; f++ {
		src := picture(w, h, style, seed+f)
		p := image.NewPaletted(src.Rect, pal)
		for i := 0; i < w*h; i++ {
			p.Pix[i] = src.Pix[i*4] >> 2
		}
		g.Image = append(g.Image, p)
		g.Delay = append(g.Delay, 25)
	}
	var b bytes.Buffer
	gif.EncodeAll(&b, g)
	return b.Bytes()
}

// encodePDF writes a one-page PDF with a few lines of text.
func encodePDF(title string, lines int) []byte {
	var b bytes.Buffer
	var offsets []int
	obj := func(body string) {
		offsets = append(offsets, b.Len())
		fmt.Fprintf(&b, "%d 0 obj\n%s\nendobj\n", len(offsets), body)
	}
	b.WriteString("%PDF-1.4\n")
	obj("<< /Type /Catalog /Pages 2 0 R >>")
	obj("<< /Type /Pages /Kids [3 0 R] /Count 1 >>")
	obj("<< /Type /Page /Parent 2 0 R /MediaBox [0 0 595 842] /Contents 4 0 R /Resources << /Font << /F1 5 0 R >> >> >>")
	text := "BT /F1 20 Tf 60 780 Td (" + title + ") Tj /F1 11 Tf\n"
	for i := 0; i < lines; i++ {
		text += fmt.Sprintf("0 -16 Td (Line %d of a document that was on this disk.) Tj\n", i+1)
	}
	text += "ET"
	obj(fmt.Sprintf("<< /Length %d >>\nstream\n%s\nendstream", len(text), text))
	obj("<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>")
	xref := b.Len()
	fmt.Fprintf(&b, "xref\n0 %d\n0000000000 65535 f \n", len(offsets)+1)
	for _, o := range offsets {
		fmt.Fprintf(&b, "%010d 00000 n \n", o)
	}
	fmt.Fprintf(&b, "trailer\n<< /Size %d /Root 1 0 R >>\nstartxref\n%d\n%%%%EOF\n", len(offsets)+1, xref)
	return b.Bytes()
}

// encodeZIP makes an archive of text files and a picture; with store, the
// picture is kept as it is, so its signature is there to be found.
func encodeZIP(r *lcg, store bool, picture []byte) []byte {
	var b bytes.Buffer
	z := zip.NewWriter(&b)
	for i := 0; i < 3; i++ {
		f, _ := z.Create(fmt.Sprintf("notes/%d.txt", i))
		for j := 0; j < 200+r.next(400); j++ {
			fmt.Fprintf(f, "entry %d: %d\n", j, r.next(1000000))
		}
	}
	method := zip.Deflate
	if store {
		method = zip.Store
	}
	f, _ := z.CreateHeader(&zip.FileHeader{Name: "picture.png", Method: method})
	f.Write(picture)
	z.Close()
	return b.Bytes()
}

// makeDisk writes the image and returns the files in it.
func makeDisk(path string) []File {
	r := &lcg{7}
	type item struct {
		kind int
		data []byte
	}
	var items []item
	add := func(kind int, data []byte) { items = append(items, item{kind, data}) }
	enc := func(f func(*bytes.Buffer) error) []byte {
		var b bytes.Buffer
		if err := f(&b); err != nil {
			panic(err)
		}
		return b.Bytes()
	}
	for i := 0; i < 12; i++ {
		img := picture(320+40*(i%4), 240+30*(i%3), i, i)
		add(PNG, enc(func(b *bytes.Buffer) error { return png.Encode(b, img) }))
	}
	for i := 0; i < 12; i++ {
		img := picture(480+64*(i%3), 360, i+2, i+20)
		add(JPEG, enc(func(b *bytes.Buffer) error { return jpeg.Encode(b, img, &jpeg.Options{Quality: 88}) }))
	}
	for i := 0; i < 4; i++ {
		add(GIF, encodeGIF(200, 150, i, i+40))
	}
	for i := 0; i < 4; i++ {
		add(BMP, encodeBMP(picture(160+16*i, 120, i+1, i+60)))
	}
	for i := 0; i < 4; i++ {
		add(PDF, encodePDF(fmt.Sprintf("Document %d", i+1), 12+10*i))
	}
	small := enc(func(b *bytes.Buffer) error { return png.Encode(b, picture(96, 72, 1, 80)) })
	for i := 0; i < 4; i++ {
		add(ZIP, encodeZIP(r, i%2 == 0, small))
	}
	// Shuffle.
	for i := len(items) - 1; i > 0; i-- {
		j := r.next(i + 1)
		items[i], items[j] = items[j], items[i]
	}

	var disk []byte
	fill := func(sectors int) {
		start := len(disk)
		disk = append(disk, make([]byte, sectors*sector)...)
		switch r.next(4) {
		case 0: // zeros
		case 1, 2: // noise
			for i := start; i < len(disk); i++ {
				disk[i] = byte(r.next(256))
			}
		case 3: // text
			for i := start; i < len(disk); {
				i += copy(disk[i:], fmt.Sprintf("log line %d: nothing to see here\n", r.next(100000)))
			}
		}
	}
	decoys := [][]byte{
		[]byte("\x89PNG\r\n\x1a\n\x00\x00\x00\x0dIHDRbroken after the header"),
		{0xff, 0xd8, 0xff, 0xe0, 0x00, 0x10, 'J', 'F', 'I', 'F', 0, 1, 1, 0, 0, 1, 0, 1, 0, 0, 0x12, 0x34},
		[]byte("GIF89a\x10\x00\x10\x00\x00\x00\x00\xfenot a block"),
		[]byte("PK\x03\x04 a local header and no directory"),
		[]byte("BM is how many sentences start"),
		[]byte("%PDF-1.7 with no end"),
	}
	var planted []File
	fill(64 + r.next(200))
	for i, it := range items {
		planted = append(planted, File{Offset: len(disk), Length: len(it.data), Kind: it.kind})
		disk = append(disk, it.data...)
		disk = append(disk, make([]byte, (sector-len(disk)%sector)%sector)...)
		fill(40 + r.next(700))
		if i%6 == 2 {
			d := decoys[i/6%len(decoys)]
			disk = append(disk, d...)
			disk = append(disk, make([]byte, sector-len(d))...)
			fill(20 + r.next(100))
		}
	}
	if err := os.WriteFile(path, disk, 0o644); err != nil {
		fmt.Fprintln(os.Stderr, "carver:", err)
		os.Exit(1)
	}
	return planted
}
