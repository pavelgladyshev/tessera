package main

// The language model: a Llama-style transformer (SmolLM2), run through
// package tt/nn. Each of its layers is
//
//	x = x + attention(norm(x))
//	x = x + down(silu(gate(norm(x))) * up(norm(x)))
//
// with rotary position encoding on the attention's queries and keys.
//
// Two things are arranged when the weights are loaded, so that a layer is
// a short list of whole-tensor operations:
//
//   - The rotation of position p turns each pair (a, b) of a query into
//     (a cos - b sin, b cos + a sin). The swap of a and b is a fixed change
//     of columns, so it is folded into a second copy of the weights:
//     q = (x·Wq)·cos + (x·Wq')·sin.
//   - The model has 9 heads of queries and 3 of keys and values, each
//     shared by three query heads. The keys' and values' weights are
//     repeated, so that all are 9 heads wide.
//
// The context has a fixed length: the tokens so far, then padding that the
// causal mask keeps out of sight. ttnn builds its programs for a tensor
// size the first time it sees it, so a context that grew by a token at a
// time would pay for that at every token.

import (
	"encoding/json"
	"math"
	"os"
	"path/filepath"
	"sort"
	"strings"

	"github.com/pavelgladyshev/tessera/tt/nn"
)

type block struct {
	norm1, norm2     nn.Tensor
	wq, wqr, wk, wkr nn.Tensor // queries and keys, and their rotated twins
	wv, wo           nn.Tensor
	gate, up, down   nn.Tensor
}

type model struct {
	context, dim, heads, headDim int
	eps                          float32
	embed, unembed, norm         nn.Tensor
	cos, sin                     nn.Tensor // [context dim]
	blocks                       []block
	pad                          int

	// For the search by meaning (see index).
	table    []float32 // the word vectors, on the host
	centre   []float32
	rarity   map[int]float32
	passages nn.Tensor // [passages dim], each row of length 1
	count    int
	words    []string  // the passages' words, and
	wordVecs nn.Tensor // their vectors: [words dim]
}

// transposed returns a [rows cols] matrix as [cols rows]: PyTorch keeps a
// layer's weights as [out in], and nn.Linear takes [in out].
func transposed(v []float32, rows, cols int) []float32 {
	t := make([]float32, len(v))
	for r := 0; r < rows; r++ {
		for c := 0; c < cols; c++ {
			t[c*rows+r] = v[r*cols+c]
		}
	}
	return t
}

// options say how a model's weights are kept on the card.
type options struct {
	mlp, attention nn.Format // of the MLPs' three matrices, and of the attention's four
	arrange        bool      // prepare the MLPs' weights for block floating point (see arrange)
}

// arrange changes an MLP's weights, without changing what the MLP
// computes, so that block floating point loses less of them. The format
// keeps sixteen neighbours of a row with one exponent, and a neuron of the
// MLP is a column of gate and up and a row of down:
//
//   - each neuron's up weights are scaled to the same size, and its down
//     weights by the inverse (up is linear, so the product is the same);
//   - the neurons are put in order of the size of their gate weights, so
//     that neighbours are alike.
//
// gate and up are [inner d] and down is [d inner], as PyTorch keeps them.
//
// Measured with SmolLM2-1.7B on the card (-eval with -record and -against),
// it made no difference beyond noise at 8 or 4 bits; it is kept as an
// experiment, off by default.
func arrange(gate, up, down []float32, inner, d int) {
	norm := func(v []float32) float64 {
		s := 0.0
		for _, x := range v {
			s += float64(x) * float64(x)
		}
		return math.Sqrt(s)
	}
	mean := 0.0
	for j := 0; j < inner; j++ {
		mean += norm(up[j*d:(j+1)*d]) / float64(inner)
	}
	for j := 0; j < inner; j++ {
		n := norm(up[j*d : (j+1)*d])
		if n == 0 {
			continue
		}
		c := float32(mean / n)
		for i := 0; i < d; i++ {
			up[j*d+i] *= c
			down[i*inner+j] /= c
		}
	}
	order := make([]int, inner)
	size := make([]float64, inner)
	for j := range order {
		order[j], size[j] = j, norm(gate[j*d:(j+1)*d])
	}
	sort.Slice(order, func(a, b int) bool { return size[order[a]] > size[order[b]] })
	g, u, dn := append([]float32(nil), gate...), append([]float32(nil), up...), append([]float32(nil), down...)
	for to, from := range order {
		copy(gate[to*d:(to+1)*d], g[from*d:(from+1)*d])
		copy(up[to*d:(to+1)*d], u[from*d:(from+1)*d])
		for i := 0; i < d; i++ {
			down[i*inner+to] = dn[i*inner+from]
		}
	}
}

func loadModel(dir string, context int, opt options) *model {
	var cfg struct {
		Hidden  int     `json:"hidden_size"`
		Layers  int     `json:"num_hidden_layers"`
		Heads   int     `json:"num_attention_heads"`
		KVHeads int     `json:"num_key_value_heads"`
		Eps     float32 `json:"rms_norm_eps"`
		Theta   float64 `json:"rope_theta"`
		Pad     int     `json:"pad_token_id"`
		Type    string  `json:"model_type"`
	}
	raw, err := os.ReadFile(filepath.Join(dir, "config.json"))
	if err != nil {
		fatal("%v", err)
	}
	if err := json.Unmarshal(raw, &cfg); err != nil || cfg.Type != "llama" || cfg.Heads == 0 || cfg.Hidden%cfg.Heads != 0 || cfg.Heads%cfg.KVHeads != 0 {
		fatal("%s does not describe a Llama-style model", filepath.Join(dir, "config.json"))
	}
	w, err := openWeights(filepath.Join(dir, "model.safetensors"))
	if err != nil {
		fatal("%v", err)
	}
	m := &model{context: context, dim: cfg.Hidden, heads: cfg.Heads, headDim: cfg.Hidden / cfg.Heads, eps: cfg.Eps, pad: cfg.Pad}
	d, hd, half := m.dim, m.headDim, m.headDim/2

	shape, emb := w.read("model.embed_tokens.weight")
	m.embed = nn.FromFloats(shape, emb)
	m.table = emb
	m.unembed = nn.FromFloats([]int{d, shape[0]}, transposed(emb, shape[0], d)) // the same numbers give the next token's scores
	vector := func(name string) nn.Tensor {
		_, v := w.read(name)
		return nn.FromFloats([]int{len(v)}, v)
	}
	m.norm = vector("model.norm.weight")

	// The rotation's angles: position p turns pair i of a head by
	// p / theta^(2i/headDim). The same for every head.
	cos, sin := make([]float32, context*d), make([]float32, context*d)
	for p := 0; p < context; p++ {
		for j := 0; j < d; j++ {
			angle := float64(p) * math.Pow(cfg.Theta, -2*float64(j%hd%half)/float64(hd))
			cos[p*d+j], sin[p*d+j] = float32(math.Cos(angle)), float32(math.Sin(angle))
		}
	}
	m.cos, m.sin = nn.FromFloats([]int{context, d}, cos), nn.FromFloats([]int{context, d}, sin)

	// linear reads a layer's weights as [in out], with its out/headDim
	// heads repeated to make all of them.
	linear := func(name string) []float32 {
		s, v := w.read(name)
		t := transposed(v, s[0], s[1]) // [in, s[0]]
		repeat := d / s[0]
		if repeat == 1 || s[0]%hd != 0 {
			return t
		}
		wide := make([]float32, s[1]*d)
		for i := 0; i < s[1]; i++ {
			for h := 0; h < m.heads; h++ {
				copy(wide[i*d+h*hd:i*d+(h+1)*hd], t[i*s[0]+h/repeat*hd:])
			}
		}
		return wide
	}
	// rotated is the twin of a query's or key's weights: within each head,
	// column j of the first half takes minus column j+half, and column j
	// of the second half takes column j-half.
	rotated := func(v []float32) []float32 {
		r := make([]float32, len(v))
		for i := 0; i < len(v); i += hd {
			for j := 0; j < half; j++ {
				r[i+j] = -v[i+j+half]
				r[i+j+half] = v[i+j]
			}
		}
		return r
	}
	matrix := func(v []float32) nn.Tensor { return nn.FromFloatsAs(opt.attention, []int{len(v) / d, d}, v) }
	for l := 0; l < cfg.Layers; l++ {
		at := func(name string) string { return "model.layers." + itoa(l) + "." + name }
		q, k := linear(at("self_attn.q_proj.weight")), linear(at("self_attn.k_proj.weight"))
		b := block{
			norm1: vector(at("input_layernorm.weight")), norm2: vector(at("post_attention_layernorm.weight")),
			wq: matrix(q), wqr: matrix(rotated(q)), wk: matrix(k), wkr: matrix(rotated(k)),
			wv: matrix(linear(at("self_attn.v_proj.weight"))), wo: matrix(linear(at("self_attn.o_proj.weight"))),
		}
		_, gate := w.read(at("mlp.gate_proj.weight"))
		_, up := w.read(at("mlp.up_proj.weight"))
		_, down := w.read(at("mlp.down_proj.weight"))
		inner := len(gate) / d
		if opt.arrange {
			arrange(gate, up, down, inner, d)
		}
		b.gate = nn.FromFloatsAs(opt.mlp, []int{d, inner}, transposed(gate, inner, d))
		b.up = nn.FromFloatsAs(opt.mlp, []int{d, inner}, transposed(up, inner, d))
		b.down = nn.FromFloatsAs(opt.mlp, []int{inner, d}, transposed(down, d, inner))
		m.blocks = append(m.blocks, b)
	}
	return m
}

func itoa(n int) string {
	if n < 10 {
		return string(rune('0' + n))
	}
	return itoa(n/10) + string(rune('0'+n%10))
}

// tape frees the tensors of a computation when it is done.
type tape []nn.Tensor

func (t *tape) add(x nn.Tensor) nn.Tensor {
	*t = append(*t, x)
	return x
}

func (t *tape) free() {
	for _, x := range *t {
		x.Free()
	}
	*t = (*t)[:0]
}

// next returns the model's score for every token of the vocabulary as the
// one after tokens (at most m.context of them).
func (m *model) next(tokens []int) []float32 {
	var none nn.Tensor
	var t tape
	x := m.run(tokens)
	last := len(tokens) - 1
	row := t.add(nn.Slice(x, []int{last, 0}, []int{last + 1, m.dim}))
	scores := t.add(nn.Linear(row, m.unembed, none)).Floats()
	x.Free()
	t.free()
	return scores
}

// run puts tokens through the layers and returns the last one's states,
// normalized: [context dim].
func (m *model) run(tokens []int) nn.Tensor {
	var none nn.Tensor
	ids := make([]int, m.context)
	for i := range ids {
		ids[i] = m.pad
	}
	copy(ids, tokens)
	n, d := m.context, m.dim
	x := nn.Embedding(ids, m.embed)
	var t tape
	heads := func(a nn.Tensor) nn.Tensor { // [n dim] to [heads n headDim]
		return t.add(nn.Transpose(t.add(nn.Reshape(a, []int{n, m.heads, m.headDim})), 0, 1))
	}
	turned := func(a, w, wr nn.Tensor) nn.Tensor { // a·w with the positions' rotation
		return t.add(nn.Add(
			t.add(nn.Mul(t.add(nn.Linear(a, w, none)), m.cos)),
			t.add(nn.Mul(t.add(nn.Linear(a, wr, none)), m.sin))))
	}
	for _, b := range m.blocks {
		a := t.add(nn.RMSNorm(x, b.norm1, m.eps))
		att := t.add(nn.Attention(heads(turned(a, b.wq, b.wqr)), heads(turned(a, b.wk, b.wkr)), heads(t.add(nn.Linear(a, b.wv, none))), true))
		att = t.add(nn.Reshape(t.add(nn.Transpose(att, 0, 1)), []int{n, d}))
		h := nn.Add(x, t.add(nn.Linear(att, b.wo, none)))
		x.Free()
		a = t.add(nn.RMSNorm(h, b.norm2, m.eps))
		inner := t.add(nn.Mul(t.add(nn.SiLU(t.add(nn.Linear(a, b.gate, none)))), t.add(nn.Linear(a, b.up, none))))
		x = nn.Add(h, t.add(nn.Linear(inner, b.down, none)))
		h.Free()
		t.free()
	}
	out := nn.RMSNorm(x, m.norm, m.eps)
	x.Free()
	t.free()
	return out
}

// best is the token with the highest score.
func best(scores []float32) int {
	at := 0
	for i, s := range scores {
		if s > scores[at] {
			at = i
		}
	}
	return at
}

// index prepares the search by meaning. A text's vector is the sum of its
// tokens' word vectors (the model's first layer, which puts words of like
// meaning near each other), each less the average vector, and weighted by
// how rare the token is among the passages; then scaled to length 1. Two
// texts are near when their vectors' product is large.
func (m *model) index(tok *tokenizer, texts []string) {
	d := m.dim
	in := map[int]int{} // passages a token is in
	tokens := make([][]int, len(texts))
	m.centre = make([]float32, d)
	total := 0
	for i, t := range texts {
		tokens[i] = tok.encode(" " + t)
		seen := map[int]bool{}
		for _, id := range tokens[i] {
			if !seen[id] {
				seen[id] = true
				in[id]++
			}
			for k := 0; k < d; k++ {
				m.centre[k] += m.table[id*d+k]
			}
			total++
		}
	}
	for k := range m.centre {
		m.centre[k] /= float32(max(1, total))
	}
	m.count = len(texts)
	m.rarity = map[int]float32{}
	for id, n := range in {
		m.rarity[id] = float32(math.Log(1 + float64(len(texts))/float64(n)))
	}
	all := make([]float32, 0, len(texts)*d)
	for i := range texts {
		all = append(all, m.vector(tokens[i])...)
	}
	m.passages = nn.FromFloats([]int{len(texts), d}, all)

	// The passages' words, each with its vector, for finding words that
	// mean much the same as a question's (see alike).
	seen := map[string]bool{}
	var vecs []float32
	for _, t := range texts {
		for _, w := range words(t) {
			w = strings.ToLower(w)
			if len(w) < 3 || seen[w] || stop[w] {
				continue
			}
			seen[w] = true
			m.words = append(m.words, w)
			vecs = append(vecs, m.vector(tok.encode(" "+w))...)
		}
	}
	m.wordVecs = nn.FromFloats([]int{len(m.words), d}, vecs)
}

// like is a word of the passages and how near it is to another, 0 to 1.
type like struct {
	word string
	near float32
}

// alike returns the passages' words nearest in meaning to w, at most n of
// them and none further than least: "written" for "wrote", say.
func (m *model) alike(tok *tokenizer, w string, n int, least float32) []like {
	q := nn.FromFloats([]int{m.dim, 1}, m.vector(tok.encode(" "+strings.ToLower(w))))
	product := nn.MatMul(m.wordVecs, q)
	scores := product.Floats()
	q.Free()
	product.Free()
	order := make([]int, len(scores))
	for i := range order {
		order[i] = i
	}
	sort.Slice(order, func(a, b int) bool {
		return scores[order[a]] > scores[order[b]] || scores[order[a]] == scores[order[b]] && order[a] < order[b]
	})
	var out []like
	for _, i := range order[:min(n, len(order))] {
		if scores[i] >= least {
			out = append(out, like{m.words[i], scores[i]})
		}
	}
	return out
}

func (m *model) vector(tokens []int) []float32 {
	d := m.dim
	v := make([]float32, d)
	for _, id := range tokens {
		w, ok := m.rarity[id]
		if !ok { // a token no passage has: as rare as can be
			w = float32(math.Log(1 + float64(m.count)))
		}
		for k := 0; k < d; k++ {
			v[k] += w * (m.table[id*d+k] - m.centre[k])
		}
	}
	norm := 0.0
	for _, x := range v {
		norm += float64(x) * float64(x)
	}
	if norm > 0 {
		for k := range v {
			v[k] /= float32(math.Sqrt(norm))
		}
	}
	return v
}

// nearest returns the n passages nearest in meaning to a text, the
// nearest first: one matrix product, through ttnn on the card.
func (m *model) nearest(tok *tokenizer, text string, n int) []int {
	q := nn.FromFloats([]int{m.dim, 1}, m.vector(tok.encode(" "+text)))
	product := nn.MatMul(m.passages, q)
	scores := product.Floats()
	q.Free()
	product.Free()
	order := make([]int, len(scores))
	for i := range order {
		order[i] = i
	}
	sort.Slice(order, func(a, b int) bool {
		return scores[order[a]] > scores[order[b]] || scores[order[a]] == scores[order[b]] && order[a] < order[b]
	})
	return order[:min(n, len(order))]
}
