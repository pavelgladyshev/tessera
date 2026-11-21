package main

// The web server: a page to drop a picture on, and the two enlargements
// side by side under a slider.

import (
	"bytes"
	"encoding/json"
	"fmt"
	"image"
	_ "image/gif"
	_ "image/jpeg"
	"image/png"
	"net/http"
	"os"
	"sync"

	xdraw "image/draw"
)

type result struct {
	original, plain, network []byte // PNG
}

type server struct {
	jobs chan<- *job
	mu   sync.Mutex
	res  map[string]*result
	n    int
}

// fit reduces a picture so that neither side is longer than maxSide,
// averaging the pixels that fall together.
func fit(src *image.RGBA) *image.RGBA {
	w, h := src.Rect.Dx(), src.Rect.Dy()
	if w <= maxSide && h <= maxSide {
		return src
	}
	f := (max(w, h) + maxSide - 1) / maxSide
	out := image.NewRGBA(image.Rect(0, 0, w/f, h/f))
	for y := 0; y < h/f; y++ {
		for x := 0; x < w/f; x++ {
			var sum [3]int
			for dy := 0; dy < f; dy++ {
				for dx := 0; dx < f; dx++ {
					at := ((y*f+dy)*w + x*f + dx) * 4
					for c := range sum {
						sum[c] += int(src.Pix[at+c])
					}
				}
			}
			at := (y*out.Rect.Dx() + x) * 4
			for c := range sum {
				out.Pix[at+c] = uint8(sum[c] / (f * f))
			}
			out.Pix[at+3] = 255
		}
	}
	return out
}

func encode(img image.Image) []byte {
	var b bytes.Buffer
	png.Encode(&b, img)
	return b.Bytes()
}

func (s *server) upscale(w http.ResponseWriter, r *http.Request) {
	var src *image.RGBA
	if r.URL.Query().Get("sample") != "" {
		src = sample(96)
	} else {
		r.Body = http.MaxBytesReader(w, r.Body, 32<<20)
		img, _, err := image.Decode(r.Body)
		if err != nil {
			http.Error(w, "that is not a PNG, JPEG or GIF picture", http.StatusBadRequest)
			return
		}
		// On a white ground, for pictures with transparent parts.
		flat := image.NewRGBA(image.Rect(0, 0, img.Bounds().Dx(), img.Bounds().Dy()))
		xdraw.Draw(flat, flat.Rect, image.White, image.Point{}, xdraw.Src)
		xdraw.Draw(flat, flat.Rect, img, img.Bounds().Min, xdraw.Over)
		src = flat
	}
	was := src.Rect
	src = fit(src)
	if src.Rect.Dx() < 8 || src.Rect.Dy() < 8 {
		http.Error(w, "the picture is too small", http.StatusBadRequest)
		return
	}

	j := do(s.jobs, src)
	s.mu.Lock()
	s.n++
	id := fmt.Sprint(s.n)
	if len(s.res) > 16 { // a demonstration keeps little
		s.res = map[string]*result{}
	}
	s.res[id] = &result{encode(src), encode(j.plain), encode(j.network)}
	s.mu.Unlock()

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(map[string]any{
		"id": id, "width": src.Rect.Dx(), "height": src.Rect.Dy(),
		"reduced":   was != src.Rect,
		"workersMs": j.workers.Milliseconds(), "networkMs": j.net.Milliseconds(),
	})
}

func (s *server) picture(w http.ResponseWriter, r *http.Request) {
	s.mu.Lock()
	res := s.res[r.PathValue("id")]
	s.mu.Unlock()
	if res == nil {
		http.NotFound(w, r)
		return
	}
	data := map[string][]byte{"original": res.original, "plain": res.plain, "network": res.network}[r.PathValue("kind")]
	if data == nil {
		http.NotFound(w, r)
		return
	}
	w.Header().Set("Content-Type", "image/png")
	w.Write(data)
}

func serve(addr string, layers int, jobs chan<- *job) {
	s := &server{jobs: jobs, res: map[string]*result{}}
	mux := http.NewServeMux()
	mux.HandleFunc("GET /{$}", func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/html; charset=utf-8")
		w.Write([]byte(page))
	})
	mux.HandleFunc("POST /upscale", s.upscale)
	mux.HandleFunc("GET /picture/{id}/{kind}", s.picture)
	fmt.Fprintf(os.Stderr, "upscale: %d layers loaded; open http://%s\n", layers, addr)
	if err := http.ListenAndServe(addr, mux); err != nil {
		fmt.Fprintln(os.Stderr, "upscale:", err)
		os.Exit(1)
	}
}

const page = `<!doctype html>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Upscale</title>
<style>
  :root { color-scheme: dark; --ink: #e6e9ef; --dim: #8a93a6; --panel: #161b24; --line: #262d3a; --hot: #ffb454; }
  * { box-sizing: border-box; }
  body { margin: 0; background: #0d1016; color: var(--ink);
         font: 15px/1.5 system-ui, -apple-system, "Segoe UI", sans-serif; }
  main { max-width: 980px; margin: 0 auto; padding: 40px 16px 72px; }
  h1 { font-size: 30px; margin: 0; letter-spacing: -0.02em; }
  h1 span { color: var(--hot); }
  p.sub { color: var(--dim); margin: 6px 0 28px; max-width: 62ch; }
  #drop { border: 1.5px dashed #39425a; border-radius: 14px; padding: 34px 16px; text-align: center;
          background: var(--panel); cursor: pointer; transition: border-color .15s, background .15s; }
  #drop.over { border-color: var(--hot); background: #1c2230; }
  #drop b { display: block; font-size: 17px; margin-bottom: 4px; }
  #drop small { color: var(--dim); }
  button { font: inherit; color: var(--ink); background: #232b3a; border: 1px solid var(--line);
           border-radius: 8px; padding: 7px 14px; cursor: pointer; margin-top: 14px; }
  button:hover { border-color: var(--hot); }
  #status { color: var(--dim); min-height: 24px; margin: 18px 0 10px; }
  #status.error { color: #ff7b72; }
  #view { display: none; }
  .compare { position: relative; border-radius: 12px; overflow: hidden; background: #000; user-select: none;
             touch-action: none; cursor: ew-resize; line-height: 0; }
  .compare img { width: 100%; display: block; image-rendering: auto; }
  .compare .top { position: absolute; inset: 0; clip-path: inset(0 0 0 var(--at, 50%)); }
  .compare .bar { position: absolute; top: 0; bottom: 0; left: var(--at, 50%); width: 2px; background: var(--hot);
                  transform: translateX(-1px); }
  .compare .bar::after { content: ""; position: absolute; top: 50%; left: 50%; width: 30px; height: 30px;
                  border-radius: 50%; background: var(--hot); transform: translate(-50%, -50%);
                  box-shadow: 0 2px 10px #0008; }
  .tag { position: absolute; top: 10px; padding: 3px 9px; border-radius: 6px; background: #000a; font-size: 12px;
         line-height: 1.5; letter-spacing: 0.04em; text-transform: uppercase; }
  .tag.l { left: 10px; } .tag.r { right: 10px; color: var(--hot); }
  .facts { display: flex; flex-wrap: wrap; gap: 10px; margin-top: 14px; }
  .fact { background: var(--panel); border: 1px solid var(--line); border-radius: 10px; padding: 9px 14px; }
  .fact b { display: block; font-size: 19px; font-variant-numeric: tabular-nums; }
  .fact span { color: var(--dim); font-size: 13px; }
  a { color: var(--hot); }
</style>
<main>
  <h1>Four times bigger, <span>on the card</span></h1>
  <p class="sub">Drop a picture. A neural network (Real-ESRGAN's compact model) enlarges it through ttnn,
  and the card's worker cores make the plain, bilinear enlargement to compare it with.</p>
  <div id="drop" tabindex="0" role="button" aria-label="Choose a picture">
    <b>Drop a picture here, or click to choose one</b>
    <small>PNG, JPEG or GIF. Pictures longer than 256 pixels on a side are reduced first.</small><br>
    <button id="sample" type="button">Try the sample picture</button>
    <input id="file" type="file" accept="image/png,image/jpeg,image/gif" hidden>
  </div>
  <div id="status" role="status"></div>
  <div id="view">
    <div class="compare" id="compare">
      <img id="plain" alt="Bilinear enlargement">
      <div class="top"><img id="network" alt="The network's enlargement"></div>
      <div class="bar"></div>
      <span class="tag l">Bilinear, by the workers</span><span class="tag r">Neural network</span>
    </div>
    <div class="facts" id="facts"></div>
  </div>
</main>
<script>
const drop = document.getElementById('drop'), file = document.getElementById('file');
const status = document.getElementById('status'), view = document.getElementById('view');
const compare = document.getElementById('compare');
function say(text, bad) { status.textContent = text; status.className = bad ? 'error' : ''; }
async function send(body, query) {
  say('Enlarging…');
  const started = performance.now();
  try {
    const r = await fetch('/upscale' + (query || ''), { method: 'POST', body });
    if (!r.ok) { say(await r.text(), true); return; }
    const j = await r.json();
    document.getElementById('plain').src = '/picture/' + j.id + '/plain';
    document.getElementById('network').src = '/picture/' + j.id + '/network';
    const facts = [
      [j.width + ' × ' + j.height, 'pixels in' + (j.reduced ? ' (reduced)' : '')],
      [4 * j.width + ' × ' + 4 * j.height, 'pixels out'],
      [(j.networkMs / 1000).toFixed(2) + ' s', 'the network'],
      [j.workersMs + ' ms', 'the workers'],
    ];
    document.getElementById('facts').innerHTML = facts.map(f =>
      '<div class="fact"><b>' + f[0] + '</b><span>' + f[1] + '</span></div>').join('') +
      '<div class="fact"><b><a download="upscaled.png" href="/picture/' + j.id + '/network">Download</a></b><span>the result, PNG</span></div>';
    view.style.display = 'block';
    say('Done in ' + ((performance.now() - started) / 1000).toFixed(1) + ' s. Drag the handle to compare.');
  } catch (e) { say('The server did not answer.', true); }
}
drop.addEventListener('click', e => { if (e.target.id !== 'sample') file.click(); });
drop.addEventListener('keydown', e => { if (e.key === 'Enter' || e.key === ' ') file.click(); });
file.addEventListener('change', () => { if (file.files[0]) send(file.files[0]); });
document.getElementById('sample').addEventListener('click', () => send(null, '?sample=1'));
['dragenter', 'dragover'].forEach(n => drop.addEventListener(n, e => { e.preventDefault(); drop.classList.add('over'); }));
['dragleave', 'drop'].forEach(n => drop.addEventListener(n, e => { e.preventDefault(); drop.classList.remove('over'); }));
drop.addEventListener('drop', e => { if (e.dataTransfer.files[0]) send(e.dataTransfer.files[0]); });
function move(e) {
  const box = compare.getBoundingClientRect();
  compare.style.setProperty('--at', Math.max(0, Math.min(100, (e.clientX - box.left) / box.width * 100)) + '%');
}
compare.addEventListener('pointerdown', e => { compare.setPointerCapture(e.pointerId); move(e); });
compare.addEventListener('pointermove', e => { if (e.buttons) move(e); });
</script>
`
