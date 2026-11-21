package main

// The model's tokenizer, read from its tokenizer.json: byte-level BPE, as
// GPT-2's, with every digit a piece of its own. Text is cut into pieces (a
// word with the space before it, a run of punctuation, ...), each piece's
// bytes become symbols, and pairs of symbols are merged in the order the
// file ranks them until none of its pairs is left.

import (
	"encoding/json"
	"fmt"
	"os"
	"strings"
	"unicode"
	"unicode/utf8"
)

type tokenizer struct {
	vocab   map[string]int
	words   []string          // by id, in the byte-level alphabet
	rank    map[[2]string]int // of a merge
	special []string          // tokens written as themselves, such as <|im_end|>
	toByte  map[rune]byte
	toRune  [256]rune
	cache   map[string][]int
}

func loadTokenizer(path string) (*tokenizer, error) {
	raw, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}
	var file struct {
		Added []struct {
			ID      int    `json:"id"`
			Content string `json:"content"`
		} `json:"added_tokens"`
		Model struct {
			Vocab  map[string]int    `json:"vocab"`
			Merges []json.RawMessage `json:"merges"`
		} `json:"model"`
	}
	if err := json.Unmarshal(raw, &file); err != nil {
		return nil, fmt.Errorf("%s: %v", path, err)
	}
	t := &tokenizer{vocab: file.Model.Vocab, rank: map[[2]string]int{}, toByte: map[rune]byte{}, cache: map[string][]int{}}
	t.words = make([]string, len(t.vocab))
	for w, id := range t.vocab {
		if id < len(t.words) {
			t.words[id] = w
		}
	}
	for i, m := range file.Model.Merges {
		var pair [2]string
		var joined string
		if json.Unmarshal(m, &joined) == nil { // "a b"
			a, b, _ := strings.Cut(joined, " ")
			pair = [2]string{a, b}
		} else if json.Unmarshal(m, &pair) != nil {
			return nil, fmt.Errorf("%s: merge %d is neither a string nor a pair", path, i)
		}
		t.rank[pair] = i
	}
	for _, a := range file.Added {
		t.special = append(t.special, a.Content)
		t.vocab[a.Content] = a.ID
	}
	// GPT-2's alphabet: a byte that prints stands for itself, the others
	// for characters from 256 up.
	next := rune(256)
	for b := 0; b < 256; b++ {
		r := rune(b)
		if !(b >= 33 && b <= 126 || b >= 161 && b <= 172 || b >= 174) {
			r = next
			next++
		}
		t.toRune[b] = r
		t.toByte[r] = byte(b)
	}
	return t, nil
}

func isLetter(r rune) bool { return unicode.IsLetter(r) }
func isNumber(r rune) bool { return unicode.IsNumber(r) }
func isOther(r rune) bool  { return !unicode.IsSpace(r) && !isLetter(r) && !isNumber(r) }

// pieces cuts text as GPT-2's pattern does:
//
//	's|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+
func pieces(text string) []string {
	rs := []rune(text)
	var out []string
	run := func(i int, is func(rune) bool) int {
		for i < len(rs) && is(rs[i]) {
			i++
		}
		return i
	}
	for i := 0; i < len(rs); {
		end := 0
		for _, c := range []string{"'s", "'t", "'re", "'ve", "'m", "'ll", "'d"} {
			if strings.HasPrefix(string(rs[i:min(len(rs), i+3)]), c) {
				end = i + len(c)
				break
			}
		}
		if end == 0 {
			j := i
			if rs[j] == ' ' && j+1 < len(rs) {
				j++
			}
			switch {
			case isLetter(rs[j]):
				end = run(j, isLetter)
			case isNumber(rs[j]):
				end = run(j, isNumber)
			case isOther(rs[j]):
				end = run(j, isOther)
			default: // white space: all of it, but for the one before a word
				end = run(i, unicode.IsSpace)
				if end < len(rs) && end-i > 1 {
					end--
				}
			}
		}
		out = append(out, string(rs[i:end]))
		i = end
	}
	return out
}

// bpe merges a piece's symbols.
func (t *tokenizer) bpe(piece string) []int {
	if ids, ok := t.cache[piece]; ok {
		return ids
	}
	var syms []string
	for _, b := range []byte(piece) {
		syms = append(syms, string(t.toRune[b]))
	}
	for len(syms) > 1 {
		best, at := -1, -1
		for i := 0; i+1 < len(syms); i++ {
			if r, ok := t.rank[[2]string{syms[i], syms[i+1]}]; ok && (best < 0 || r < best) {
				best, at = r, i
			}
		}
		if at < 0 {
			break
		}
		a, b := syms[at], syms[at+1]
		var merged []string
		for i := 0; i < len(syms); i++ {
			if i+1 < len(syms) && syms[i] == a && syms[i+1] == b {
				merged = append(merged, a+b)
				i++
			} else {
				merged = append(merged, syms[i])
			}
		}
		syms = merged
	}
	ids := make([]int, 0, len(syms))
	for _, s := range syms {
		if id, ok := t.vocab[s]; ok {
			ids = append(ids, id)
		}
	}
	t.cache[piece] = ids
	return ids
}

// encode turns text into tokens.
func (t *tokenizer) encode(text string) []int {
	var ids []int
	for len(text) > 0 {
		// Up to the next special token, and then that token.
		at, which := len(text), ""
		for _, s := range t.special {
			if i := strings.Index(text, s); i >= 0 && i < at {
				at, which = i, s
			}
		}
		// Every digit is a piece of its own, before the pattern is applied.
		var chunk []rune
		flush := func() {
			for _, p := range pieces(string(chunk)) {
				ids = append(ids, t.bpe(p)...)
			}
			chunk = chunk[:0]
		}
		for _, r := range text[:at] {
			if unicode.IsDigit(r) {
				flush()
				ids = append(ids, t.bpe(string(r))...)
			} else {
				chunk = append(chunk, r)
			}
		}
		flush()
		if which != "" {
			ids = append(ids, t.vocab[which])
		}
		text = text[at+len(which):]
	}
	return ids
}

// bytesOf returns the bytes a token stands for.
func (t *tokenizer) bytesOf(id int) []byte {
	if id < 0 || id >= len(t.words) {
		return nil
	}
	for _, s := range t.special {
		if t.vocab[s] == id {
			return []byte(s)
		}
	}
	var b []byte
	for _, r := range t.words[id] {
		if c, ok := t.toByte[r]; ok {
			b = append(b, c)
		}
	}
	return b
}

// decode turns tokens into text.
func (t *tokenizer) decode(ids []int) string {
	var b []byte
	for _, id := range ids {
		b = append(b, t.bytesOf(id)...)
	}
	return strings.ToValidUTF8(string(b), "�")
}

// stream gives text as tokens arrive, holding back the bytes of a
// character that is not complete yet.
type stream struct {
	t       *tokenizer
	pending []byte
}

func (s *stream) add(id int) string {
	s.pending = append(s.pending, s.t.bytesOf(id)...)
	n := len(s.pending)
	for n > 0 && n > len(s.pending)-4 && !utf8.FullRune(s.pending[n-1:]) && !utf8.RuneStart(s.pending[n-1]) {
		n--
	}
	if n > 0 && utf8.RuneStart(s.pending[n-1]) && !utf8.FullRune(s.pending[n-1:]) {
		n--
	}
	out := string(s.pending[:n])
	s.pending = append(s.pending[:0], s.pending[n:]...)
	return out
}
