package main

// The report: a map of the image with a cell for every stretch of sectors,
// coloured by the kind of file found there, and the files themselves.

import (
	"fmt"
	"html"
	"image"
	"image/color"
	"image/png"
	"os"
	"path/filepath"
	"strings"
)

var kindColor = [kinds]color.RGBA{
	{0xf2, 0xa6, 0x3a, 255}, // jpeg
	{0x4c, 0xb5, 0xf5, 255}, // png
	{0xb3, 0x7f, 0xeb, 255}, // gif
	{0xef, 0x5b, 0x5b, 255}, // pdf
	{0x5c, 0xd6, 0x8f, 255}, // zip
	{0xf5, 0xe0, 0x50, 255}, // bmp
}

// diskMap draws the image as cells, 256 across: file kinds in colour, other
// data grey, zeros dark.
func diskMap(path string, size int, files []File) (*image.RGBA, int) {
	const across, px = 256, 4
	cell := sector // bytes a cell stands for
	for (size+cell-1)/cell > across*192 {
		cell *= 2
	}
	cells := (size + cell - 1) / cell
	rows := (cells + across - 1) / across
	img := image.NewRGBA(image.Rect(0, 0, across*px, rows*px))
	for i := 0; i < len(img.Pix); i += 4 {
		copy(img.Pix[i:], []byte{0x0a, 0x0c, 0x10, 0xff})
	}
	shade := make([]color.RGBA, cells)
	// What is not a file: read a little of each cell.
	if f, err := os.Open(path); err == nil {
		buf := make([]byte, 64)
		for c := range shade {
			n, _ := f.ReadAt(buf, int64(c)*int64(cell))
			shade[c] = color.RGBA{0x16, 0x1a, 0x22, 255}
			for _, b := range buf[:n] {
				if b != 0 {
					shade[c] = color.RGBA{0x3a, 0x42, 0x52, 255}
					break
				}
			}
		}
		f.Close()
	}
	for _, f := range files {
		for c := f.Offset / cell; c <= (f.Offset+f.Length-1)/cell && c < cells; c++ {
			shade[c] = kindColor[f.Kind]
		}
		// The first cell of a file a little brighter, to tell neighbours apart.
		k := kindColor[f.Kind]
		shade[f.Offset/cell] = color.RGBA{uint8(min(255, int(k.R)+50)), uint8(min(255, int(k.G)+50)), uint8(min(255, int(k.B)+50)), 255}
	}
	for c, col := range shade {
		for dy := 0; dy < px-1; dy++ {
			for dx := 0; dx < px-1; dx++ {
				img.SetRGBA(c%across*px+dx, c/across*px+dy, col)
			}
		}
	}
	return img, cell
}

func hex(c color.RGBA) string { return fmt.Sprintf("#%02x%02x%02x", c.R, c.G, c.B) }

func human(n int) string {
	switch {
	case n >= 1<<20:
		return fmt.Sprintf("%.1f MB", float64(n)/(1<<20))
	case n >= 1<<10:
		return fmt.Sprintf("%.1f KB", float64(n)/(1<<10))
	}
	return fmt.Sprintf("%d B", n)
}

func report(dir, path string, size int, files []File) error {
	m, cell := diskMap(path, size, files)
	f, err := os.Create(filepath.Join(dir, "map.png"))
	if err != nil {
		return err
	}
	if err := png.Encode(f, m); err != nil {
		return err
	}
	f.Close()

	var count, bytes [kinds]int
	total := 0
	for _, f := range files {
		count[f.Kind]++
		bytes[f.Kind] += f.Length
		total += f.Length
	}
	var b strings.Builder
	fmt.Fprintf(&b, `<!doctype html>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Carved files</title>
<style>
  :root { color-scheme: dark; }
  body { margin: 0; padding: 32px 16px 64px; background: #0e1116; color: #d7dde8;
         font: 15px/1.5 system-ui, -apple-system, "Segoe UI", sans-serif; }
  main { max-width: 1060px; margin: 0 auto; }
  h1 { font-size: 26px; margin: 0 0 4px; letter-spacing: -0.01em; }
  h2 { font-size: 15px; text-transform: uppercase; letter-spacing: 0.08em; color: #8b95a7; margin: 40px 0 12px; }
  p.sub { color: #8b95a7; margin: 0 0 24px; }
  .stats { display: flex; flex-wrap: wrap; gap: 10px; }
  .stat { background: #171c25; border-radius: 10px; padding: 10px 16px; min-width: 96px; border-top: 3px solid var(--c); }
  .stat b { display: block; font-size: 22px; font-variant-numeric: tabular-nums; }
  .stat span { color: #8b95a7; font-size: 13px; }
  .map { background: #0a0c10; border-radius: 10px; padding: 12px; overflow-x: auto; }
  .map img { display: block; width: 100%%; min-width: 640px; image-rendering: pixelated; }
  .legend { display: flex; flex-wrap: wrap; gap: 16px; margin-top: 10px; color: #8b95a7; font-size: 13px; }
  .legend i { display: inline-block; width: 11px; height: 11px; border-radius: 2px; margin-right: 6px; background: var(--c); }
  .grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(190px, 1fr)); gap: 14px; }
  .card { background: #171c25; border-radius: 10px; overflow: hidden; text-decoration: none; color: inherit;
          border-bottom: 3px solid var(--c); transition: transform .12s; }
  .card:hover { transform: translateY(-3px); }
  .thumb { height: 140px; background: #0a0c10; display: flex; align-items: center; justify-content: center; }
  .thumb img { max-width: 100%%; max-height: 100%%; }
  .thumb span { font: 700 30px ui-monospace, monospace; color: var(--c); }
  .meta { padding: 9px 12px 10px; font-size: 13px; }
  .meta b { display: block; font: 600 13px ui-monospace, monospace; }
  .meta span { color: #8b95a7; }
</style>
<main>
<h1>%d files carved</h1>
<p class="sub">%s &middot; %s &middot; %s recovered</p>
<div class="stats">
`, len(files), html.EscapeString(filepath.Base(path)), human(size), human(total))
	for k := 0; k < kinds; k++ {
		fmt.Fprintf(&b, `<div class="stat" style="--c:%s"><b>%d</b><span>%s &middot; %s</span></div>
`, hex(kindColor[k]), count[k], strings.ToUpper(kindName[k]), human(bytes[k]))
	}
	fmt.Fprintf(&b, `</div>
<h2>Where they were</h2>
<div class="map"><img src="map.png" alt="Map of the disk image"></div>
<div class="legend">
`)
	for k := 0; k < kinds; k++ {
		fmt.Fprintf(&b, `<span><i style="--c:%s"></i>%s</span>`, hex(kindColor[k]), strings.ToUpper(kindName[k]))
	}
	fmt.Fprintf(&b, `<span><i style="--c:#3a4252"></i>other data</span><span><i style="--c:#16191f"></i>zeros</span>
<span>one cell = %s, left to right, top to bottom</span>
</div>
<h2>The files</h2>
<div class="grid">
`, human(cell))
	for _, f := range files {
		n := name(f)
		thumb := fmt.Sprintf(`<span>%s</span>`, strings.ToUpper(kindName[f.Kind]))
		if f.Kind != PDF && f.Kind != ZIP {
			thumb = fmt.Sprintf(`<img loading="lazy" src="%s" alt="">`, n)
		}
		fmt.Fprintf(&b, `<a class="card" style="--c:%s" href="%s"><div class="thumb">%s</div><div class="meta"><b>%s</b><span>sector %d &middot; %s</span></div></a>
`, hex(kindColor[f.Kind]), n, thumb, n, f.Offset/sector, human(f.Length))
	}
	b.WriteString("</div>\n</main>\n")
	return os.WriteFile(filepath.Join(dir, "index.html"), []byte(b.String()), 0o644)
}
