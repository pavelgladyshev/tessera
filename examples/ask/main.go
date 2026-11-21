// ask answers questions from a set of documents with a small language
// model, and shows how a program is built around one.
//
// A question goes through three steps, and both halves of Tessera do
// their part:
//
//  1. The search. The documents are cut into passages. The workers, on the
//     card one per Tensix core, each score their share of the passages by
//     the question's words (stems, and BM25, in integer arithmetic).
//
//     Two more ways to search are there to try, both from the model's own
//     word vectors through ttnn: -search meaning (or both) scores every
//     passage by how near its vector is to the question's, and -expand
//     adds the passages' words that mean much the same as the question's.
//     On the 22 questions this was measured with (-eval), neither found
//     more than the words alone, which put the right passage among the
//     first three for 21; they are not on by default.
//
//  2. A choice. The program labels the best passages A, B and C, asks the
//     model which one answers the question, and reads the reply off the
//     scores of those three tokens instead of letting the model write: a
//     model's choice among answers the program names is far more
//     dependable than its prose.
//
//  3. The answer. The program puts the passages that passed and the
//     question into a prompt, and the model writes, a token at a time.
//
// The model is SmolLM2-135M-Instruct (30 layers), run through package
// tt/nn: on the card with ttnn, under go run in float32 (slow: use a short
// -context). It is small, so it is often wrong; the point is the program.
//
//	go run ./examples/ask -model smollm2 -context 128 -ask "How many workers can a card run?"
//	./build/ask -model smollm2          # on the card; then open http://localhost:8090
//
// -model names a directory with the model's config.json, tokenizer.json
// and model.safetensors, from huggingface.co/HuggingFaceTB/SmolLM2-135M-Instruct.
// SmolLM2-360M-Instruct works the same way (32 layers, 960 wide): on an
// n300 it writes 12 tokens a second where the 135M model writes 18, and
// its choice among the passages is better.
package main

import (
	"encoding/binary"
	"flag"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"time"

	"github.com/pavelgladyshev/tessera/tt"
)

const (
	maxTerms = 32 // words that count: a question's, and words like them
	maxFound = 3  // passages that go on to the check
	maxWord  = 32 // letters of a word that count
)

// Hit is a passage that has some of the question's words, and its score
// (BM25, times 1000).
type Hit struct {
	Passage, Score int
}

// event is a step of an answer, for whoever asked to follow.
type event struct {
	Kind     string  `json:"kind"` // found, check, prompt, token, done
	Text     string  `json:"text,omitempty"`
	Source   string  `json:"source,omitempty"`
	Score    int     `json:"score,omitempty"`
	Words    int     `json:"words,omitempty"`   // a passage's place by its words, 1 the best; 0 if it has none of them
	Meaning  int     `json:"meaning,omitempty"` // its place by meaning
	Yes      float64 `json:"yes,omitempty"`     // the model's share of its choice for a passage, 0 to 1
	Kept     bool    `json:"kept,omitempty"`
	Millis   int64   `json:"ms,omitempty"`
	Tokens   int     `json:"tokens,omitempty"`
	Passages int     `json:"passages,omitempty"`
	Workers  int     `json:"workers,omitempty"`
}

// job is a question (or, with raw, a text to continue as it is). main
// does the jobs one at a time and sends what happens to events.
type job struct {
	question, raw string
	searchOnly    bool // stop after the search
	events        chan event
}

// stemmed hashes a word, the n letters and digits in w, in lower case and
// without its ending, so that "workers", "worked" and "working" are the
// word "work". Both main and the workers use it: a question's words and a
// passage's must come out the same.
func stemmed(w []byte, n int) uint32 {
	if n > 5 && w[n-3] == 'i' && w[n-2] == 'n' && w[n-1] == 'g' {
		n -= 3
	} else if n > 4 && w[n-2] == 'e' && w[n-1] == 'd' {
		n -= 2
	} else if n > 3 && w[n-1] == 's' && w[n-2] != 's' {
		n--
	}
	if n > 3 && w[n-1] == w[n-2] { // "runn" of running
		n--
	}
	if n > 3 && w[n-1] == 'e' { // "use" of uses and "us" of used
		n--
	}
	h := uint32(2166136261)
	for i := 0; i < n; i++ {
		h = (h ^ uint32(w[i])) * 16777619
	}
	return h
}

// word is stemmed for a word of main's.
func word(w string) uint32 {
	var b [maxWord]byte
	n := copy(b[:], strings.ToLower(w))
	return stemmed(b[:], n)
}

func words(text string) []string {
	return strings.FieldsFunc(text, func(r rune) bool {
		return !(r >= 'a' && r <= 'z' || r >= 'A' && r <= 'Z' || r >= '0' && r <= '9')
	})
}

var stop = map[string]bool{}

func init() {
	for _, w := range strings.Fields("a an the is are was were be been do does did can could will would should of in on at to for from by with and or not it its this that these those what which who how why when where many much there has have had i you we they as than then so if about into over") {
		stop[w] = true
	}
}

func main() {
	modelDir := flag.String("model", "smollm2", "the directory with the model's config.json, tokenizer.json and model.safetensors")
	context := flag.Int("context", 384, "tokens the model sees at once")
	limit := flag.Int("tokens", 80, "the longest answer, in tokens")
	docs := flag.String("docs", "", "a directory whose .md and .txt files are added to the documents")
	addr := flag.String("addr", "localhost:8090", "where the web server listens")
	question := flag.String("ask", "", "instead of serving: answer this question and stop")
	raw := flag.String("complete", "", "instead of serving: continue this text and stop")
	checks := flag.Bool("check", true, "have the model choose among the passages before answering")
	workers := flag.Int("workers", 0, "workers (default: one per core)")
	search := flag.String("search", "words", "how passages are found: words, meaning or both")
	expand := flag.Float64("expand", 0, "how near in meaning (0 to 1) a passage's word must be to a question's to count as well; 0 for the question's own words only")
	eval := flag.String("eval", "", "instead of serving: a file of questions, each with a tab and some text of the passage that answers it; says how often the search finds that passage")
	flag.Parse()

	passages := loadPassages(*docs)
	// How many passages have each word: a rare word says more.
	have := map[uint32]int{}
	total := 0 // words in all the passages
	for _, p := range passages {
		seen := map[uint32]bool{}
		for _, w := range words(p.text) {
			total++
			if h := word(w); !seen[h] {
				seen[h] = true
				have[h]++
			}
		}
	}
	average := max(1, total/max(1, len(passages)))
	// The passages as one file for the workers: how many, where each
	// begins, and their text.
	dir, err := os.MkdirTemp("", "ask")
	if err != nil {
		fatal("%v", err)
	}
	defer os.RemoveAll(dir)
	count := len(passages)
	index := binary.LittleEndian.AppendUint32(nil, uint32(count))
	at := 4 + 4*(count+1)
	var text []byte
	for _, p := range passages {
		index = binary.LittleEndian.AppendUint32(index, uint32(at+len(text)))
		text = append(text, p.text...)
	}
	index = binary.LittleEndian.AppendUint32(index, uint32(at+len(text)))
	if err := os.WriteFile(filepath.Join(dir, "passages.bin"), append(index, text...), 0o600); err != nil {
		fatal("%v", err)
	}
	corpus := tt.LoadBytes(filepath.Join(dir, "passages.bin"))
	n := *workers
	if n <= 0 {
		n = max(1, min(tt.Cores(), count/2))
	}

	tok, err := loadTokenizer(filepath.Join(*modelDir, "tokenizer.json"))
	if err != nil {
		fatal("%v", err)
	}
	began := time.Now()
	m := loadModel(*modelDir, *context)
	texts := make([]string, count)
	for i, p := range passages {
		texts[i] = p.text
	}
	m.index(tok, texts)
	fmt.Fprintf(os.Stderr, "ask: %d passages; the model's %d layers loaded in %.1f s\n", count, len(m.blocks), time.Since(began).Seconds())
	end := tok.vocab["<|im_end|>"]
	letters := []int{tok.encode("A")[0], tok.encode("B")[0], tok.encode("C")[0]}

	jobs := make(chan *job)
	switch {
	case *eval != "":
		start(func() { evaluate(jobs, *eval) })
	case *question != "" || *raw != "":
		start(func() { printed(jobs, *question, *raw) })
	default:
		start(func() { serve(*addr, jobs) })
	}

	for j := range jobs {
		prompt := j.raw
		if j.raw == "" {
			// Step 1: the workers score the passages. The question's
			// words go to them as a file of hashes and weights.
			terms := binary.LittleEndian.AppendUint32(nil, 0)
			used := map[uint32]bool{}
			add := func(h uint32, scale float64) {
				if used[h] || have[h] == 0 || len(used) == maxTerms {
					return
				}
				used[h] = true
				terms = binary.LittleEndian.AppendUint32(terms, h)
				terms = binary.LittleEndian.AppendUint32(terms, uint32(1000*scale*math.Log(1+float64(count)/float64(have[h]))))
			}
			asked := words(j.question)
			for _, w := range asked {
				if !stop[strings.ToLower(w)] && *search != "meaning" {
					add(word(w), 1)
				}
			}
			// And the passages' words that mean much the same as the
			// question's, for less: someone who asks who "wrote" it
			// should find what was "written".
			for _, w := range asked {
				if stop[strings.ToLower(w)] || len(w) < 3 || *search == "meaning" || *expand == 0 {
					continue
				}
				for _, l := range m.alike(tok, w, 3, float32(*expand)) {
					add(word(l.word), 0.5*float64(l.near))
				}
			}
			binary.LittleEndian.PutUint32(terms, uint32(len(used)))
			if err := os.WriteFile(filepath.Join(dir, "question.bin"), terms, 0o600); err != nil {
				fatal("%v", err)
			}
			query := tt.LoadBytes(filepath.Join(dir, "question.bin"))
			began := time.Now()

			hits := tt.Parallel(n, func(k int, out *tt.List[Hit]) {
				var hash, weight [maxTerms]uint32
				var buf [1024]byte
				query.Read(buf[:4+8*maxTerms], 0)
				nt := int(buf[0])
				for i := 0; i < nt; i++ {
					b := buf[4+8*i : 12+8*i]
					hash[i] = uint32(b[0]) | uint32(b[1])<<8 | uint32(b[2])<<16 | uint32(b[3])<<24
					weight[i] = uint32(b[4]) | uint32(b[5])<<8 | uint32(b[6])<<16 | uint32(b[7])<<24
				}
				var letters [maxWord]byte
				var times [maxTerms]int32
				for p := count * k / n; p < count*(k+1)/n && nt > 0; p++ {
					corpus.Read(buf[:8], 4+4*p)
					from := int(uint32(buf[0]) | uint32(buf[1])<<8 | uint32(buf[2])<<16 | uint32(buf[3])<<24)
					to := int(uint32(buf[4]) | uint32(buf[5])<<8 | uint32(buf[6])<<16 | uint32(buf[7])<<24)
					// Count the question's words in the passage, and the
					// passage's words.
					for t := 0; t < nt; t++ {
						times[t] = 0
					}
					length, wordLen := 0, 0
					for off := from; off <= to; off += len(buf) {
						got := 0
						if off < to {
							got = corpus.Read(buf[:min(len(buf), to-off)], off)
						}
						for i := 0; i <= got; i++ {
							c := byte(' ') // one past the end closes the last word
							if i < got {
								c = buf[i]
							} else if off+got < to {
								break
							}
							if c >= 'A' && c <= 'Z' {
								c += 32
							}
							if c >= 'a' && c <= 'z' || c >= '0' && c <= '9' {
								if wordLen < maxWord {
									letters[wordLen] = c
									wordLen++
								}
								continue
							}
							if wordLen > 0 {
								h := stemmed(letters[:], wordLen)
								for t := 0; t < nt; t++ {
									if hash[t] == h {
										times[t]++
									}
								}
								length++
								wordLen = 0
							}
						}
					}
					// BM25: a word counts for more the rarer it is, for
					// little more the second time, and for less in a long
					// passage (k1 = 1.2, b = 0.75).
					score := 0
					for t := 0; t < nt; t++ {
						if f := int(times[t]); f > 0 {
							score += int(weight[t]) * f * 22 * 100 * average / (f*10*100*average + 12*(25*average+75*length))
						}
					}
					if score > 0 {
						out.Append(Hit{p, score})
					}
				}
			})

			sort.Slice(hits, func(a, b int) bool {
				return hits[a].Score > hits[b].Score || hits[a].Score == hits[b].Score && hits[a].Passage < hits[b].Passage
			})
			// By meaning: the passages nearest the question, from ttnn.
			// The two orders are put together by place: a passage scores
			// 1/(60+place) in each (reciprocal rank fusion).
			placeW, placeM := map[int]int{}, map[int]int{}
			fused := map[int]float64{}
			for i, h := range hits {
				placeW[h.Passage] = i + 1
				fused[h.Passage] += 1 / float64(60+i+1)
			}
			if *search != "words" {
				for i, p := range m.nearest(tok, j.question, 20) {
					placeM[p] = i + 1
					fused[p] += 1 / float64(60+i+1)
				}
			}
			order := make([]int, 0, len(fused))
			for p := range fused {
				order = append(order, p)
			}
			sort.Slice(order, func(a, b int) bool {
				return fused[order[a]] > fused[order[b]] || fused[order[a]] == fused[order[b]] && order[a] < order[b]
			})
			order = order[:min(len(order), maxFound)]
			hits = hits[:0]
			for _, p := range order {
				hits = append(hits, Hit{p, int(1e5 * fused[p])})
			}
			j.events <- event{Kind: "searched", Millis: time.Since(began).Milliseconds(), Passages: count, Workers: n}
			for _, h := range hits {
				j.events <- event{Kind: "found", Text: passages[h.Passage].text, Source: passages[h.Passage].source, Score: h.Score,
					Words: placeW[h.Passage], Meaning: placeM[h.Passage]}
			}
			if j.searchOnly {
				j.events <- event{Kind: "done"}
				close(j.events)
				continue
			}

			// Step 2: the model chooses among the passages. The program
			// labels them A, B, C and reads the choice off the scores of
			// those three tokens: the model cannot say anything else.
			// What the search ranked first is kept in any case.
			var kept []string
			for _, h := range hits {
				kept = append(kept, passages[h.Passage].text)
			}
			if *checks && len(hits) > 1 {
				began := time.Now()
				list := ""
				for i, h := range hits {
					list += string(rune('A'+i)) + ". " + passages[h.Passage].text + "\n"
				}
				choose := func(question string) []int {
					return tok.encode(chat("You pick the note that answers a question. Reply with the letter of the note only.",
						list+"\nQuestion: "+question+"\n\nWhich note answers the question?"))
				}
				if ids := choose(j.question); len(ids) < m.context {
					// A small model leans towards one letter whatever it
					// is asked. Asking the same with no question shows
					// the lean, and it is taken off.
					scores, lean := m.next(ids), m.next(choose("N/A"))
					share, total := make([]float64, len(hits)), 0.0
					for i := range hits {
						share[i] = math.Exp(float64(scores[letters[i]] - lean[letters[i]]))
						total += share[i]
					}
					kept = kept[:0]
					for i, h := range hits {
						share[i] /= total
						// The search's best passage stays whatever the model
						// says: a small model's choice is a second opinion.
						keep := i == 0 || share[i] >= 0.45
						if keep {
							kept = append(kept, passages[h.Passage].text)
						}
						j.events <- event{Kind: "check", Text: passages[h.Passage].text, Yes: share[i], Kept: keep, Millis: time.Since(began).Milliseconds()}
					}
				}
			}

			// Step 3: the prompt, of as many passages as fit.
			for len(kept) > 0 {
				prompt = chat("You answer questions about Tessera. Use only the notes you are given. Answer in one or two short sentences.",
					"Notes:\n- "+strings.Join(kept, "\n- ")+"\n\nQuestion: "+j.question)
				if len(tok.encode(prompt))+16 <= m.context {
					break
				}
				kept = kept[:len(kept)-1]
			}
			if len(kept) == 0 {
				j.events <- event{Kind: "token", Text: "The documents have nothing on that."}
				j.events <- event{Kind: "done"}
				close(j.events)
				continue
			}
			j.events <- event{Kind: "prompt", Text: prompt}
		}

		// The model writes, a token at a time, until it ends its turn.
		ids := tok.encode(prompt)
		began := time.Now()
		made := 0
		out := stream{t: tok}
		for made < *limit && len(ids) < m.context {
			id := best(m.next(ids))
			if id == end || id == 0 {
				break
			}
			ids = append(ids, id)
			made++
			if s := out.add(id); s != "" {
				j.events <- event{Kind: "token", Text: s}
			}
		}
		j.events <- event{Kind: "done", Tokens: made, Millis: time.Since(began).Milliseconds()}
		close(j.events)
	}
}

// chat lays out a system message and a user's as the model was trained
// to read them, up to where its own reply begins.
func chat(system, user string) string {
	return "<|im_start|>system\n" + system + "<|im_end|>\n<|im_start|>user\n" + user + "<|im_end|>\n<|im_start|>assistant\n"
}

// start runs f beside main.
func start(f func()) { go f() }

// printed asks one question (or continues one text) and prints what
// happens.
func printed(jobs chan<- *job, question, raw string) {
	j := &job{question: question, raw: raw, events: make(chan event, 16)}
	jobs <- j
	for e := range j.events {
		switch e.Kind {
		case "searched":
			fmt.Printf("%d workers searched %d passages\n", e.Workers, e.Passages)
		case "found":
			fmt.Printf("found (words #%d, meaning #%d): %s\n", e.Words, e.Meaning, e.Text)
		case "check":
			fmt.Printf("choice: %.0f%%, %s: %.40s...\n", 100*e.Yes, map[bool]string{true: "kept", false: "dropped"}[e.Kept], e.Text)
		case "prompt":
			fmt.Println("answer:")
		case "token":
			fmt.Print(e.Text)
		case "done":
			fmt.Println()
			if e.Tokens > 0 {
				fmt.Fprintf(os.Stderr, "ask: %d tokens in %.1f s (%.2f a second)\n", e.Tokens, float64(e.Millis)/1000, float64(e.Tokens)*1000/float64(max(1, e.Millis)))
			}
		}
	}
	close(jobs)
}

// evaluate runs the search for each question of a file and says how often
// the passage that answers it comes first, and among the first three.
func evaluate(jobs chan<- *job, path string) {
	raw, err := os.ReadFile(path)
	if err != nil {
		fatal("%v", err)
	}
	first, among, all := 0, 0, 0
	for _, line := range strings.Split(string(raw), "\n") {
		question, want, ok := strings.Cut(line, "\t")
		if !ok {
			continue
		}
		j := &job{question: question, searchOnly: true, events: make(chan event, 16)}
		jobs <- j
		place, at := 0, 0
		for e := range j.events {
			if e.Kind == "found" {
				at++
				if place == 0 && strings.Contains(e.Text, strings.TrimSpace(want)) {
					place = at
				}
			}
		}
		all++
		if place == 1 {
			first++
		}
		if place > 0 {
			among++
		} else {
			fmt.Printf("missed: %s\n", question)
		}
	}
	fmt.Printf("%d questions: the right passage came first for %d, among the first %d for %d\n", all, first, maxFound, among)
	close(jobs)
}
