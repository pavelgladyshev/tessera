package main

// The live page (-serve): the simulation runs for as long as someone
// watches, and the page draws every step as it comes from the workers.

import (
	"encoding/binary"
	"fmt"
	"net/http"
	"os"
	"strconv"
	"sync"
	"time"
)

// request is what the page asks of the simulation.
type request struct {
	restart bool
	bodies  int // with restart: how many, or 0 for as many as before
}

// hub stands between main, which simulates, and the page's viewers.
type hub struct {
	mu      sync.Mutex
	frame   []byte // the latest step, as the page reads it
	seq     uint32
	seen    time.Time // when a viewer last asked for a frame
	paused  bool
	pending *request
}

func newHub() *hub { return &hub{} }

// wait returns when the simulation should take its next step: someone is
// watching and it is not paused, or the page asked to start over.
func (h *hub) wait() request {
	for {
		h.mu.Lock()
		if h.pending != nil {
			r := *h.pending
			h.pending = nil
			h.mu.Unlock()
			return r
		}
		watched := time.Since(h.seen) < 3*time.Second
		paused := h.paused
		h.mu.Unlock()
		if watched && !paused {
			return request{}
		}
		time.Sleep(40 * time.Millisecond)
	}
}

// publish gives the viewers a step: six numbers (the frame's number, the
// bodies, how many are the first galaxy's, the step, the time tt.Parallel
// took in microseconds, the workers), then every body's place as two 16-bit
// numbers.
func (h *hub) publish(step int, bodies []body, workers int, took time.Duration) {
	first := 0
	for _, b := range bodies {
		if b.galaxy == 0 {
			first++
		}
	}
	h.mu.Lock()
	h.seq++
	f := make([]byte, 0, 24+4*len(bodies))
	for _, v := range []int{int(h.seq), len(bodies), first, step, int(took.Microseconds()), workers} {
		f = binary.LittleEndian.AppendUint32(f, uint32(v))
	}
	for _, b := range bodies {
		x, y := max(-32767, min(32767, b.x>>10)), max(-32767, min(32767, b.y>>10))
		f = binary.LittleEndian.AppendUint16(f, uint16(int16(x)))
		f = binary.LittleEndian.AppendUint16(f, uint16(int16(y)))
	}
	h.frame = f
	h.mu.Unlock()
}

func serve(addr string, h *hub) {
	mux := http.NewServeMux()
	mux.HandleFunc("GET /{$}", func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/html; charset=utf-8")
		w.Write([]byte(page))
	})
	// The first frame after the one the page has, when there is one.
	mux.HandleFunc("GET /frame", func(w http.ResponseWriter, r *http.Request) {
		after, _ := strconv.Atoi(r.URL.Query().Get("after"))
		for waited := 0; ; waited++ {
			h.mu.Lock()
			h.seen = time.Now()
			f, seq := h.frame, h.seq
			h.mu.Unlock()
			if f != nil && int(seq) != after || waited > 400 {
				w.Header().Set("Content-Type", "application/octet-stream")
				w.Header().Set("Cache-Control", "no-store")
				w.Write(f)
				return
			}
			select {
			case <-r.Context().Done():
				return
			case <-time.After(5 * time.Millisecond):
			}
		}
	})
	mux.HandleFunc("POST /control", func(w http.ResponseWriter, r *http.Request) {
		q := r.URL.Query()
		h.mu.Lock()
		if q.Has("pause") {
			h.paused = q.Get("pause") == "1"
		}
		if q.Has("restart") {
			n, _ := strconv.Atoi(q.Get("bodies"))
			h.pending = &request{restart: true, bodies: n}
			h.paused = false
		}
		h.mu.Unlock()
	})
	fmt.Fprintf(os.Stderr, "galaxy: open http://%s\n", addr)
	if err := http.ListenAndServe(addr, mux); err != nil {
		fmt.Fprintln(os.Stderr, "galaxy:", err)
		os.Exit(1)
	}
}

const page = `<!doctype html>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Galaxy</title>
<style>
  :root { color-scheme: dark; --ink: #e7eaf0; --dim: #8b94a7; --panel: #121722; --line: #252c3a; --a: #6eaaff; --b: #ffaa50; }
  * { box-sizing: border-box; }
  body { margin: 0; background: #05070c; color: var(--ink);
         font: 15px/1.5 system-ui, -apple-system, "Segoe UI", sans-serif; }
  main { max-width: 1120px; margin: 0 auto; padding: 24px 16px 48px; display: grid; gap: 20px;
         grid-template-columns: minmax(0, 1fr) 280px; align-items: start; }
  @media (max-width: 860px) { main { grid-template-columns: 1fr; } }
  canvas { width: 100%; aspect-ratio: 1; display: block; background: #05070c; border-radius: 12px;
           border: 1px solid var(--line); }
  h1 { font-size: 24px; margin: 0 0 4px; letter-spacing: -0.02em; }
  p { color: var(--dim); margin: 0 0 16px; font-size: 14px; }
  .stat { background: var(--panel); border: 1px solid var(--line); border-radius: 10px; padding: 9px 14px; margin-bottom: 8px; }
  .stat b { display: block; font-size: 21px; font-variant-numeric: tabular-nums; }
  .stat span { color: var(--dim); font-size: 13px; }
  .row { display: flex; gap: 8px; margin: 14px 0 8px; }
  button, select { font: inherit; color: var(--ink); background: #1b2230; border: 1px solid var(--line);
                   border-radius: 8px; padding: 7px 12px; cursor: pointer; flex: 1; }
  button:hover, select:hover { border-color: var(--a); }
  label { color: var(--dim); font-size: 13px; display: block; margin-top: 12px; }
  .key { font-size: 13px; color: var(--dim); margin-top: 14px; }
  .key i { display: inline-block; width: 10px; height: 10px; border-radius: 50%; margin-right: 6px; }
</style>
<main>
  <canvas id="sky" width="900" height="900" aria-label="The two galaxies"></canvas>
  <div>
    <h1>Two galaxies</h1>
    <p>Every body pulls on every other. The card's worker cores compute the pulls, in whole numbers, one step at a
    time; this page draws each step as it arrives.</p>
    <div class="stat"><b id="rate">&ndash;</b><span>steps a second</span></div>
    <div class="stat"><b id="took">&ndash;</b><span>a step on the card: starting the workers, and their work</span></div>
    <div class="stat"><b id="pairs">&ndash;</b><span>pairs of bodies in a step</span></div>
    <div class="stat"><b id="step">&ndash;</b><span id="who">step</span></div>
    <label for="bodies">Bodies</label>
    <div class="row" style="margin-top:4px"><select id="bodies">
      <option>512</option><option>1024</option><option>2048</option><option selected>4096</option></select></div>
    <div class="row"><button id="pause">Pause</button><button id="again">Start over</button></div>
    <div class="key"><i style="background:var(--a)"></i>the larger galaxy<br><i style="background:var(--b)"></i>the smaller one</div>
  </div>
</main>
<script>
const $ = id => document.getElementById(id);
const sky = $('sky'), ctx = sky.getContext('2d');
const size = sky.width, scale = size / (2 * 8192);
ctx.fillStyle = '#05070c'; ctx.fillRect(0, 0, size, size);
let seq = 0, paused = false, times = [];
function draw(buf) {
  const head = new Uint32Array(buf, 0, 6), at = new Int16Array(buf, 24);
  const n = head[1], first = head[2];
  seq = head[0];
  // Fade what was there, for trails, then add the bodies' light.
  ctx.globalCompositeOperation = 'source-over';
  ctx.fillStyle = 'rgba(5, 7, 12, 0.3)';
  ctx.fillRect(0, 0, size, size);
  ctx.globalCompositeOperation = 'lighter';
  for (let g = 0; g < 2; g++) {
    ctx.fillStyle = g ? 'rgba(255, 160, 70, 0.8)' : 'rgba(100, 160, 255, 0.8)';
    for (let i = g ? first : 0; i < (g ? n : first); i++) {
      const x = at[2 * i] * scale + size / 2, y = at[2 * i + 1] * scale + size / 2;
      if (i === 0 || i === first) {   // a galaxy's core
        const glow = ctx.createRadialGradient(x, y, 0, x, y, 14);
        glow.addColorStop(0, 'rgba(255,255,255,0.9)'); glow.addColorStop(1, 'rgba(255,255,255,0)');
        const keep = ctx.fillStyle; ctx.fillStyle = glow; ctx.fillRect(x - 14, y - 14, 28, 28); ctx.fillStyle = keep;
      } else {
        ctx.fillRect(x, y, 1.7, 1.7);
      }
    }
  }
  const now = performance.now();
  times.push(now); while (times.length > 30) times.shift();
  if (times.length > 5) $('rate').textContent = ((times.length - 1) * 1000 / (now - times[0])).toFixed(1);
  $('took').textContent = (head[4] / 1000).toFixed(1) + ' ms';
  $('pairs').textContent = (n * n).toLocaleString();
  $('step').textContent = head[3];
  $('who').textContent = 'step, of ' + n + ' bodies on ' + head[5] + ' workers';
}
async function run() {
  for (;;) {
    try {
      const r = await fetch('/frame?after=' + seq);
      const buf = await r.arrayBuffer();
      if (buf.byteLength >= 24 && new Uint32Array(buf, 0, 1)[0] !== seq) draw(buf);
      else await new Promise(done => setTimeout(done, 50));
    } catch (e) { await new Promise(done => setTimeout(done, 1000)); }
  }
}
$('pause').addEventListener('click', () => {
  paused = !paused; $('pause').textContent = paused ? 'Resume' : 'Pause'; times = [];
  fetch('/control?pause=' + (paused ? 1 : 0), { method: 'POST' });
});
function again() {
  paused = false; $('pause').textContent = 'Pause'; times = [];
  ctx.globalCompositeOperation = 'source-over'; ctx.fillStyle = '#05070c'; ctx.fillRect(0, 0, size, size);
  fetch('/control?restart=1&bodies=' + $('bodies').value, { method: 'POST' });
}
$('again').addEventListener('click', again);
$('bodies').addEventListener('change', again);
run();
</script>
`
