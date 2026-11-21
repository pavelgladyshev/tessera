package main

// The web page: a question, and the three steps of its answer as they
// happen.

import (
	"encoding/json"
	"fmt"
	"net/http"
	"os"
	"strings"
)

func serve(addr string, jobs chan<- *job) {
	mux := http.NewServeMux()
	mux.HandleFunc("GET /{$}", func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/html; charset=utf-8")
		w.Write([]byte(page))
	})
	// The steps of an answer, as server-sent events.
	mux.HandleFunc("GET /ask", func(w http.ResponseWriter, r *http.Request) {
		question := strings.TrimSpace(r.URL.Query().Get("q"))
		if question == "" || len(question) > 300 {
			http.Error(w, "a question of up to 300 characters, please", http.StatusBadRequest)
			return
		}
		w.Header().Set("Content-Type", "text/event-stream")
		w.Header().Set("Cache-Control", "no-cache")
		j := &job{question: question, events: make(chan event, 64)}
		jobs <- j // waits its turn: the card answers one question at a time
		for e := range j.events {
			data, _ := json.Marshal(e)
			fmt.Fprintf(w, "data: %s\n\n", data)
			if f, ok := w.(http.Flusher); ok {
				f.Flush()
			}
		}
	})
	fmt.Fprintf(os.Stderr, "ask: open http://%s\n", addr)
	if err := http.ListenAndServe(addr, mux); err != nil {
		fatal("%v", err)
	}
}

const page = `<!doctype html>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Ask the documents</title>
<style>
  :root { color-scheme: dark; --ink: #e7eaf0; --dim: #8b94a7; --panel: #151a23; --line: #262d3b;
          --a: #5cc8ff; --b: #ffb454; --c: #7ee787; }
  * { box-sizing: border-box; }
  body { margin: 0; background: #0c0f15; color: var(--ink);
         font: 15px/1.55 system-ui, -apple-system, "Segoe UI", sans-serif; }
  main { max-width: 860px; margin: 0 auto; padding: 40px 16px 80px; }
  h1 { font-size: 28px; margin: 0; letter-spacing: -0.02em; }
  p.sub { color: var(--dim); margin: 6px 0 24px; max-width: 64ch; }
  form { display: flex; gap: 8px; }
  input { flex: 1; min-width: 0; font: inherit; color: var(--ink); background: var(--panel);
          border: 1px solid var(--line); border-radius: 10px; padding: 11px 14px; }
  input:focus { outline: 2px solid var(--a); outline-offset: 1px; }
  button { font: inherit; font-weight: 600; color: #0c0f15; background: var(--a); border: 0;
           border-radius: 10px; padding: 0 18px; cursor: pointer; }
  button:disabled { opacity: .5; cursor: default; }
  .tries { display: flex; flex-wrap: wrap; gap: 8px; margin: 12px 0 28px; }
  .tries button { font-weight: 400; font-size: 13px; color: var(--dim); background: none;
                  border: 1px solid var(--line); border-radius: 99px; padding: 5px 12px; }
  .tries button:hover { color: var(--ink); border-color: var(--dim); }
  section { border-left: 3px solid var(--k); padding: 2px 0 2px 16px; margin: 0 0 26px; display: none; }
  section h2 { font-size: 13px; text-transform: uppercase; letter-spacing: .08em; color: var(--k); margin: 0 0 2px; }
  section .how { color: var(--dim); font-size: 13px; margin: 0 0 12px; }
  .item { background: var(--panel); border-radius: 10px; padding: 10px 14px; margin-bottom: 8px; font-size: 14px; }
  .item.dropped { opacity: .45; }
  .meter { display: flex; align-items: center; gap: 10px; color: var(--dim); font-size: 12px; margin-bottom: 4px;
           font-variant-numeric: tabular-nums; }
  .meter i { flex: 0 0 120px; height: 5px; border-radius: 3px; background: var(--line); overflow: hidden; }
  .meter i b { display: block; height: 100%; background: var(--k); }
  #answer { font-size: 19px; line-height: 1.5; white-space: pre-wrap; min-height: 28px; }
  #answer.writing::after { content: "▍"; color: var(--c); animation: blink 1s steps(2) infinite; }
  @keyframes blink { 50% { opacity: 0; } }
  @media (prefers-reduced-motion: reduce) { #answer.writing::after { animation: none; } }
  details { margin-top: 12px; color: var(--dim); font-size: 13px; }
  pre { white-space: pre-wrap; background: var(--panel); border-radius: 10px; padding: 12px 14px;
        font: 12px/1.5 ui-monospace, monospace; color: #b7c0d0; overflow-x: auto; }
  .note { color: var(--dim); font-size: 13px; }
</style>
<main>
  <h1>Ask the documents</h1>
  <p class="sub">A program around a small language model: the card's worker cores search the documents,
  and the model, run through ttnn, picks among what they find and writes the answer.</p>
  <form id="form"><input id="q" placeholder="Ask about Tessera" maxlength="300" aria-label="Your question" autofocus>
    <button id="go">Ask</button></form>
  <div class="tries" id="tries"></div>

  <section id="s1" style="--k: var(--a)"><h2>1 &middot; The workers search</h2><p class="how" id="how1"></p><div id="found"></div></section>
  <section id="s2" style="--k: var(--b)"><h2>2 &middot; The model chooses</h2>
    <p class="how">Asked which passage answers the question, the model may reply only A, B or C: the program reads
    the three letters' scores, less what they score with no question at all (a small model leans towards one
    letter). The search's first passage is kept whatever it says.</p><div id="checks"></div></section>
  <section id="s3" style="--k: var(--c)"><h2>3 &middot; The model answers</h2><p class="how" id="how3"></p>
    <div id="answer"></div>
    <details><summary>The prompt the program wrote</summary><pre id="prompt"></pre></details></section>
  <p class="note">The model is SmolLM2-135M-Instruct, a very small one: it is quick, and often wrong.</p>
</main>
<script>
const $ = id => document.getElementById(id);
const tries = ['How many workers can run on one card?', 'Why is int32 faster than int?',
  'What happens when workers run for a long time?', 'How big is a worker’s heap?', 'What does the carver example find?'];
$('tries').innerHTML = tries.map(t => '<button type="button">' + t + '</button>').join('');
$('tries').addEventListener('click', e => { if (e.target.tagName === 'BUTTON') { $('q').value = e.target.textContent; ask(); } });
$('form').addEventListener('submit', e => { e.preventDefault(); ask(); });
function item(text, label, share, dropped) {
  const d = document.createElement('div');
  d.className = 'item' + (dropped ? ' dropped' : '');
  const m = document.createElement('div'); m.className = 'meter';
  m.innerHTML = '<i><b style="width:' + Math.round(100 * share) + '%"></b></i>';
  m.append(label);
  const t = document.createElement('div'); t.textContent = text;
  d.append(m, t);
  return d;
}
let source;
function ask() {
  const q = $('q').value.trim();
  if (!q) return;
  if (source) source.close();
  for (const s of ['s1', 's2', 's3']) $(s).style.display = 'none';
  for (const id of ['found', 'checks', 'answer', 'prompt', 'how1', 'how3']) $(id).textContent = '';
  $('go').disabled = true;
  let top = 0, n = 0;
  const letters = 'ABC';
  source = new EventSource('/ask?q=' + encodeURIComponent(q));
  source.onmessage = m => {
    const e = JSON.parse(m.data);
    if (e.kind === 'searched') {
      $('s1').style.display = 'block';
      $('how1').textContent = e.workers + ' workers scored ' + e.passages + ' passages by the question’s words in ' + e.ms + ' ms.';
    } else if (e.kind === 'found') {
      top = Math.max(top, e.score);
      $('found').append(item(e.text, (e.words ? 'by its words #' + e.words : '') + (e.meaning ? ' by meaning #' + e.meaning : '') + ' · ' + e.source, e.score / top));
    } else if (e.kind === 'check') {
      $('s2').style.display = 'block';
      $('checks').append(item(e.text, letters[n++] + ' · ' + Math.round(100 * e.yes) + '%' + (e.kept ? ' · kept' : ' · dropped'), e.yes, !e.kept));
    } else if (e.kind === 'prompt') {
      $('s3').style.display = 'block'; $('prompt').textContent = e.text; $('answer').className = 'writing';
    } else if (e.kind === 'token') {
      $('s3').style.display = 'block'; $('answer').textContent += e.text;
    } else if (e.kind === 'done') {
      $('answer').className = '';
      if (e.tokens) $('how3').textContent = e.tokens + ' tokens in ' + (e.ms / 1000).toFixed(1) + ' s: ' + (e.tokens * 1000 / Math.max(1, e.ms)).toFixed(1) + ' a second.';
      source.close(); $('go').disabled = false;
    }
  };
  source.onerror = () => { source.close(); $('go').disabled = false; if (!$('answer').textContent) { $('s3').style.display = 'block'; $('answer').textContent = 'The server did not answer.'; } };
}
</script>
`
