// langcheck tests the Go features that worker code can use: multiple
// results, methods, labelled loops, goto, switch with break and
// fallthrough, arrays of arrays, arrays in structs, copying arrays, copy,
// == on arrays and structs, function literals, recursion, append and make,
// strings, maps, the strings, bytes and unicode/utf8 functions, integer
// overflow, which wraps around in Go, math/bits, and Go's order of
// evaluation where C++'s differs (cases that Go's own test suite found), and
// the order of calls.
// Each
// worker runs every test and reports one number per test; go run and the
// card must print the same lines. (Maps are ranged over only to sum or
// count: Go's order changes from run to run.)
package main

import (
	"bytes"
	"fmt"
	"math/bits"
	"strings"
	"unicode/utf8"

	"github.com/pavelgladyshev/tessera/tt"
)

const workers = 4

// Result is one test's number from one worker.
type Result struct {
	Worker, Test, Value int
}

type Point struct{ X, Y int }

func (p Point) Sum() int { return p.X + p.Y }

func (p *Point) Scale(k int) {
	p.X *= k
	p.Y *= k
}

type Counter struct {
	N    int
	Hist [4]int16
}

func (c *Counter) Add(v int) {
	c.N++
	c.Hist[v&3]++
}

// Named embeds Point: X, Y, Sum and Scale are promoted.
type Named struct {
	Point
	Tag uint8
}

type Kind int

func (k Kind) Double() Kind { return 2 * k }

type Grid struct {
	Cells [3][3]int8
}

func divmod(a, b int) (int, int) { return a / b, a % b }

func minmax(v []int) (lo, hi int) {
	lo, hi = v[0], v[0]
	for _, x := range v {
		lo = min(lo, x)
		hi = max(hi, x)
	}
	return
}

func forward(a, b int) (int, int) { return divmod(a, b) }

func results(w int) int {
	q, r := divmod(100+w, 7)
	var v [5]int
	for i := range v {
		v[i] = (i*7+w)%5 - 2
	}
	lo, hi := minmax(v[:])
	a, _ := forward(50, w+3)
	var x, y = divmod(w+20, 3)
	return q*1000000 + r*100000 + (lo+5)*10000 + (hi+5)*1000 + a*10 + x + y
}

func methods(w int) int {
	p := Point{X: w, Y: 2}
	p.Scale(3)
	var c Counter
	for i := range 10 {
		c.Add(i + w)
	}
	var n Named
	n.X = 5
	n.Scale(2)
	k := Kind(w + 1).Double()
	var pts [3]Point
	pts[1] = Point{1, 2}
	pts[1].Scale(5)
	return p.Sum()*1000000 + c.N*10000 + int(c.Hist[1])*1000 + n.Sum()*10 + int(k) + pts[1].Sum()
}

func findPair(v []int, target int) int {
	found := -1
outer:
	for i := 0; i < len(v); i++ {
		for j := i + 1; j < len(v); j++ {
			if v[i]+v[j] == target {
				found = i*100 + j
				break outer
			}
			if v[j] < 0 {
				continue outer
			}
		}
	}
	return found
}

func countDown(n int) int {
	s := 0
loop:
	if n > 0 {
		s += n
		n--
		goto loop
	}
	return s
}

func classify(n int) int {
	r := 0
	switch {
	case n < 0:
		r = -1
	case n == 0:
		r = 100
		fallthrough
	case n < 10:
		r++
	default:
		r = 10
	}
	return r
}

func switchBreak(v []int) int {
	s := 0
	for _, x := range v {
		switch x % 3 {
		case 0:
			if x > 6 {
				break // leaves the switch, not the loop
			}
			s += x
		case 1:
			s += 10 * x
		default:
			continue
		}
		s++
	}
	return s
}

func control(w int) int {
	var v [8]int
	for i := range v {
		v[i] = (i*5+w)%9 - 1
	}
	return findPair(v[:], 7)*100000 + countDown(w+4)*1000 + (classify(w-1)+1)*100 + switchBreak(v[:])
}

func matmul(w int) int {
	var a, b, c [4][4]int
	for i := range 4 {
		for j := range 4 {
			a[i][j] = i + j + w
			b[i][j] = i*j - w
		}
	}
	for i := range 4 {
		for j := range 4 {
			s := 0
			for k := range 4 {
				s += a[i][k] * b[k][j]
			}
			c[i][j] = s
		}
	}
	sum := 0
	for i := range c {
		for _, x := range c[i] {
			sum += x * (i + 1)
		}
	}
	return sum
}

func grids(w int) int {
	var g Grid
	for i := range 3 {
		for j := range 3 {
			g.Cells[i][j] = int8(i*3 + j + w)
		}
	}
	h := g
	h.Cells[1][1] = 0
	same := 0
	if g == h {
		same = 1
	}
	h.Cells[1][1] = g.Cells[1][1]
	if g == h {
		same += 2
	}
	var row [3]int8
	row = g.Cells[2]
	row2 := row
	row2[0] = 99
	eq := 0
	if row == g.Cells[2] {
		eq = 1
	}
	if row2 != row {
		eq += 2
	}
	return same*1000 + eq*100 + int(row[1]) + int(row2[0])
}

func shifts(w int) int {
	var buf [10]int
	for i := range buf {
		buf[i] = i + w
	}
	n := copy(buf[1:], buf[:9]) // overlapping: shifts right by one
	m := copy(buf[:3], buf[5:])
	s := n*100 + m
	for _, x := range buf {
		s = s*3 + x
	}
	return s
}

func closures(w int) int {
	total := 0
	add := func(v int) { total += v }
	for i := range 5 {
		add(i * w)
	}
	sq := func(v int) int { return v * v }
	both := func(a, b int) (int, int) { return a + b, a - b }
	s, d := both(sq(w), 3)
	r := func(k int) int {
		var tmp [4]int
		for i := range tmp {
			tmp[i] = k + i
		}
		return tmp[3] - tmp[0]
	}(w)
	return total*10000 + s*100 + d + r
}

// ---- recursion

func fib(n int) int {
	if n < 2 {
		return n
	}
	return fib(n-1) + fib(n-2)
}

func isEven(n int) bool {
	if n == 0 {
		return true
	}
	return isOdd(n - 1)
}

func isOdd(n int) bool {
	if n == 0 {
		return false
	}
	return isEven(n - 1)
}

// quicksort sorts v in place.
func quicksort(v []int) {
	if len(v) < 2 {
		return
	}
	p := v[len(v)/2]
	i, j := 0, len(v)-1
	for i <= j {
		for v[i] < p {
			i++
		}
		for v[j] > p {
			j--
		}
		if i <= j {
			v[i], v[j] = v[j], v[i]
			i++
			j--
		}
	}
	quicksort(v[:j+1])
	quicksort(v[i:])
}

// layers gives every call an array of its own.
func layers(n, k int) int {
	var mine [6]int
	for i := range mine {
		mine[i] = n*10 + i
	}
	if n == 0 {
		return mine[k%6]
	}
	below := layers(n-1, k+1)
	return below + mine[(k+n)%6]
}

// node is a binary tree kept in a slice: children are indices, -1 is none.
type node struct{ Left, Right, Val int }

func treeSum(t []node, i int) int {
	if i < 0 {
		return 0
	}
	return t[i].Val + treeSum(t, t[i].Left) + treeSum(t, t[i].Right)
}

func depth(n int) int {
	if n == 0 {
		return 0
	}
	return 1 + depth(n-1)
}

func recursion(w int) int {
	var v [16]int
	for i := range v {
		v[i] = (i*7 + w*5) % 11
	}
	quicksort(v[:])
	h := 0
	for _, x := range v {
		h = mix(h, x)
	}
	var t [7]node
	for i := range t {
		t[i] = node{Left: -1, Right: -1, Val: i + w}
		if 2*i+1 < len(t) {
			t[i].Left = 2*i + 1
		}
		if 2*i+2 < len(t) {
			t[i].Right = 2*i + 2
		}
	}
	even := 0
	if isEven(w + 30) {
		even = 1
	}
	return mix(mix(mix(mix(h, fib(15+w)), even), layers(5, w)), treeSum(t[:], 0)*1000+depth(400+w))
}

// ---- append and make

func mix(h, x int) int { return (h*31 + x) % 1000000007 }

func appends(w int) int {
	var s []int
	for i := range 100 + w {
		s = append(s, i*i%97)
	}
	t := append([]int{}, s[10:20]...)
	t = append(t, 1, 2, 3)
	u := make([]int, 5, 10)
	u = append(u, w)
	n := copy(u, t)
	var grid [][]int
	for i := range 4 {
		row := make([]int, i+1)
		for j := range row {
			row[j] = i*j + w
		}
		grid = append(grid, row)
	}
	// A full three-index slice makes append copy: base keeps its values.
	base := []int{1, 2, 3, 4}
	a := append(base[:2:2], 9)
	a[0] = 7
	// Within capacity, append writes in place.
	b := append(base[:1], 8)
	var none []int
	isNil := 0
	if none == nil && append(none, none...) == nil {
		isNil = 1
	}
	pts := []Point{{1, 2}, {3, 4}}
	pts = append(pts, Point{w, w})
	keyed := []int{3: 5, 1: 2}
	h := 0
	for _, x := range s {
		h = mix(h, x)
	}
	for _, x := range t {
		h = mix(h, x)
	}
	for _, x := range u {
		h = mix(h, x)
	}
	for _, r := range grid {
		for _, x := range r {
			h = mix(h, x)
		}
	}
	for _, p := range pts {
		h = mix(h, p.Sum())
	}
	for _, x := range keyed {
		h = mix(h, x)
	}
	return mix(mix(mix(mix(h, len(s)*100+len(t)*10+n), base[0]*1000+base[1]*100+a[0]*10+b[1]), isNil), len(keyed))
}

// ---- strings

type Label struct {
	Name string
	N    int
}

func hashStr(s string) int {
	h := 0
	for i := 0; i < len(s); i++ {
		h = mix(h, int(s[i]))
	}
	return h
}

func describe(n int) string {
	switch {
	case n < 0:
		return "negative"
	case n == 0:
		return "zero"
	}
	return "positive"
}

func strs(w int) int {
	s := "tessera"
	t := s + "-" + "go"
	b := []byte(t)
	b[0] = 'T'
	u := string(b)
	order := 0
	if u < t {
		order = 1
	}
	if u != t {
		order += 2
	}
	if t[:7] == s {
		order += 4
	}
	h := hashStr(u)
	runes, last := 0, 0
	for i, c := range "héllo, 世界\xff!" {
		runes++
		last = i*100000 + int(c)
		h = mix(h, last)
	}
	rs := []rune("héllo")
	back := string(rs[1:3])
	h = mix(h, hashStr(back))
	kind := 0
	switch string(b[1:4]) {
	case "abc":
		kind = 1
	case "ess":
		kind = 2
	}
	acc := ""
	for i := range 5 {
		acc += string(rune('a' + i + w))
	}
	h = mix(h, hashStr(acc))
	labels := []Label{{"one", 1}, {describe(w - 1), w}}
	if labels[1] == (Label{"zero", 1}) {
		kind += 10
	}
	for _, l := range labels {
		h = mix(h, hashStr(l.Name)+l.N)
	}
	var names [3]string
	names[w%3] = "set"
	buf := make([]byte, 0, 4)
	buf = append(buf, "ab"...)
	buf = append(buf, 'c')
	n := copy(buf, "xy")
	h = mix(h, hashStr(string(buf))+n+len(names[0]))
	return mix(mix(mix(h, order), runes*1000+kind), len(t))
}

// Bag holds a string and a slice; Add appends through a pointer receiver.
type Bag struct {
	Name  string
	Items []int
}

func (b *Bag) Add(v int) { b.Items = append(b.Items, v) }

func (b Bag) Total() int {
	t := 0
	for _, v := range b.Items {
		t += v
	}
	return t + len(b.Name)
}

func bags(w int) int {
	var bs []Bag
	for i := range 3 {
		b := Bag{Name: describe(i - 1)}
		for k := range 10 + w {
			b.Add(k * i)
		}
		bs = append(bs, b)
	}
	h := 0
	for _, b := range bs {
		h = mix(h, b.Total())
	}
	var words []string
	for i := range 20 {
		words = append(words, string(rune('a'+i))+describe(i%3-1))
	}
	for _, s := range words {
		h = mix(h, hashStr(s))
	}
	return h
}

// ---- the heap given back

// sample allocates 1 KiB every call; it gives it back when it returns.
func sample(i int) int {
	b := make([]byte, 1024)
	for k := range b {
		b[k] = byte(i + k)
	}
	return int(b[i%1024])
}

func churn(w int, tag string) int {
	total := 0
	for i := range 3000 {
		total += sample(i + w)
	}
	for i := range 50000 {
		name := tag + string(rune('a'+i%26)) // each turn's string is gone by the next
		total += len(name) + int(name[len(name)-1])
	}
	var kept []string
	for i := range 50 {
		kept = append(kept, tag+string(rune('A'+i%26)))
	}
	return mix(mix(total, len(kept)), hashStr(kept[49]+tag))
}

// ---- maps

type Pos struct{ X, Y int }

func maps(w int) int {
	words := strings.Fields("the quick brown fox jumps over the lazy dog the end")
	counts := map[string]int{}
	for _, word := range words {
		counts[word]++
	}
	h := mix(0, len(counts))
	h = mix(h, counts["the"]*100+counts["fox"]*10+counts["cat"])
	if _, ok := counts["cat"]; !ok {
		h = mix(h, 1)
	}
	delete(counts, "the")
	v, ok := counts["the"]
	h = mix(h, v)
	if !ok {
		h = mix(h, 2)
	}
	sum := 0
	for k, n := range counts {
		sum += hashStr(k) * n
	}
	h = mix(h, sum%1000003)
	// Struct keys; the map grows several times; entries deleted while ranging.
	grid := make(map[Pos]int)
	for i := range 200 + w {
		grid[Pos{i % 17, i / 17}] += i
	}
	for p, v := range grid {
		if v%3 == 0 {
			delete(grid, p)
		}
	}
	total := 0
	for p, v := range grid {
		total += (p.X*31 + p.Y) * v
	}
	h = mix(mix(h, len(grid)), total%1000003)
	// Maps of slices, and maps of maps (which share, like Go's).
	byLen := map[int][]string{}
	for _, word := range words {
		byLen[len(word)] = append(byLen[len(word)], word)
	}
	h = mix(h, len(byLen[3])*10+len(byLen[5]))
	nested := map[int]map[int]int{}
	for i := range 30 {
		inner, ok := nested[i%4]
		if !ok {
			inner = map[int]int{}
			nested[i%4] = inner
		}
		inner[i] = i * w
	}
	h = mix(h, len(nested[1])*100+nested[2][6])
	var none map[string]int
	if none == nil && len(none) == 0 && none["x"] == 0 {
		h = mix(h, 3)
	}
	lit := map[string]Pos{"a": {1, 2}, "b": {3, 4}}
	h = mix(h, lit["b"].X*10+lit["a"].Y)
	clear(lit)
	return mix(h, len(lit))
}

// ---- the strings, bytes and unicode/utf8 functions

func textfuncs(w int) int {
	line := "  name=tessera; cores=56 ;  chip=wormhole  "
	h := 0
	for _, field := range strings.Split(line, ";") {
		k, v, ok := strings.Cut(strings.TrimSpace(field), "=")
		if ok {
			h = mix(h, hashStr(k)*7+hashStr(v))
		}
	}
	h = mix(h, strings.Count(line, "e")*100+strings.Index(line, "cores")+strings.LastIndex(line, "o"))
	h = mix(h, hashStr(strings.Repeat("ab", w+1)))
	h = mix(h, hashStr(strings.Join(strings.Fields(" a  bb ccc "), "+")))
	h = mix(h, hashStr(strings.ReplaceAll("a-b-c-d", "-", "::"))+hashStr(strings.Replace("aaaa", "a", "b", 2)))
	if strings.HasPrefix(line, "  n") && strings.HasSuffix(line, "  ") && strings.Contains(line, "chip") {
		h = mix(h, 1)
	}
	h = mix(h, hashStr(strings.Trim("xxhixx", "x"))+hashStr(strings.TrimLeft("¡¡hola!", "¡"))+hashStr(strings.TrimSuffix("file.jpg", ".jpg")))
	h = mix(h, strings.IndexByte("abc", 'c')*100+strings.IndexRune("héllo", 'l')*10+strings.IndexAny("golang", "ln"))
	if after, found := strings.CutPrefix("prefix-body", "prefix-"); found {
		h = mix(h, hashStr(after))
	}
	h = mix(h, len(strings.SplitN("a,b,c,d", ",", 2))*10+len(strings.SplitAfter("a,b,c", ",")))
	h = mix(h, strings.Compare("a", "b")+5)
	s := "Grüße, 世界"
	h = mix(h, utf8.RuneCountInString(s)*100+len(s))
	r, size := utf8.DecodeRuneInString(s[2:])
	h = mix(h, int(r)*10+size)
	r, size = utf8.DecodeLastRuneInString(s)
	h = mix(h, int(r)*10+size)
	if utf8.ValidString(s) && !utf8.ValidString("\xff") {
		h = mix(h, 1)
	}
	buf := []byte("magic: %PDF-1.7 ...")
	h = mix(h, bytes.Index(buf, []byte("%PDF"))*10+bytes.IndexByte(buf, ':'))
	if bytes.HasPrefix(buf[7:], []byte("%PDF")) && bytes.Equal(buf[:5], []byte("magic")) {
		h = mix(h, 2)
	}
	h = mix(h, len(bytes.Fields(buf)))
	if bytes.TrimSpace([]byte("   ")) == nil {
		h = mix(h, 3)
	}
	enc := make([]byte, 4)
	n := utf8.EncodeRune(enc, '界')
	enc = utf8.AppendRune(enc[:n], rune('a'+w))
	h = mix(h, hashStr(string(enc)))
	joined := bytes.Join([][]byte{[]byte("a"), []byte("b")}, []byte(", "))
	return mix(h, hashStr(string(joined)))
}

// ---- integer overflow: Go wraps around, at every size

func overflow(w int) int {
	h := 0
	maxInt := int(^uint(0) >> 1)
	x := maxInt - w
	if x+w+1 < x { // the sum wraps to the most negative int
		h = mix(h, 1)
	}
	acc := int64(w + 1)
	for i := range 40 {
		acc = acc*6364136223846793005 + int64(i) // a hash that overflows every turn
	}
	h = mix(h, int(acc%1000003))
	a32 := int32(2147483600 + w)
	for range 10 {
		a32 += 13
	}
	h = mix(h, int(a32))
	u16 := 65535 - uint16(w)
	u16 *= u16
	h = mix(h, int(u16))
	i8 := int8(120 + w)
	i8 += 10
	h = mix(h, int(i8))
	minInt := -maxInt - 1
	m, d := minInt+w*0, -1+w*0
	h = mix(h, (m/d)%1000003) // the most negative int divided by -1 is itself
	h = mix(h, m%d)
	h = mix(h, (-m)%1000003)
	n := -m
	for i := range 3 {
		n -= maxInt * (i + 1)
	}
	return mix(h, n%1000003)
}

// ---- math/bits, at every width

func bitfuncs(w int) int {
	h := 0
	x := uint64(0x9e3779b97f4a7c15) * uint64(w+1)
	for i := range 12 {
		x ^= x << 13
		x ^= x >> 7
		x ^= x << 17
		v := x >> (i * 5)
		h = mix(h, bits.OnesCount64(v)+bits.OnesCount32(uint32(v))*3+bits.OnesCount8(uint8(v)))
		h = mix(h, bits.Len64(v)+bits.Len16(uint16(v))*7+bits.LeadingZeros32(uint32(v>>40)))
		h = mix(h, bits.TrailingZeros64(v)+bits.TrailingZeros8(uint8(v>>8))*5+bits.TrailingZeros(uint(v<<i)))
		h = mix(h, int(bits.RotateLeft64(v, i-6)%1000003)+int(bits.RotateLeft8(uint8(v), -i)))
		h = mix(h, int(bits.Reverse64(v)%1000003)+int(bits.Reverse16(uint16(v)))+int(bits.ReverseBytes32(uint32(v))%9973))
		sum, carry := bits.Add64(v, x, uint64(i%2))
		diff, borrow := bits.Sub32(uint32(v), uint32(x), uint32(i%2))
		hi, lo := bits.Mul64(v, x|1)
		h = mix(h, int(sum%1000003)+int(carry)*7+int(diff%9973)+int(borrow)*11)
		if y := x>>3 | 1; hi < y {
			q, r := bits.Div64(hi, lo, y)
			h = mix(h, int(q%1000003)+int(r%9973))
		}
		h = mix(h, int(bits.Rem64(hi, lo, v|1)%1000003))
	}
	return h
}

// Empty holds nothing: map[K]Empty is a set.
type Empty struct{}

// pair returns two numbers, for j, a[j] = pair(w).
func pair(w int) (int, int) { return 3, 40 + w }

// evalorder checks what Go's test suite found tessc getting wrong: the order
// in which an assignment to several targets evaluates them, append's values,
// switch string(b) when a case changes b, shifts of 8-, 16-
// and 32-bit integers by a count in a variable, empty structs, and range
// into variables that exist already (and into a map's element).
func evalorder(w int) int {
	h := 0
	// i, m[i] = ...: the key is i from before the assignment.
	m := map[int]int{}
	i := w % 3
	i, m[i] = i+1, 10+w
	h = mix(h, i*100+m[w%3]+len(m))
	// s, s[k] = s[:k], v: the element is the old slice's.
	s := []int{1, 2, 3, w}
	s, s[len(s)-1] = s[:len(s)-1], 7
	h = mix(h, len(s)+s[:4][3]*10)
	// j, a[j] = f(): the element is a[j] for j from before the call.
	var a [4]int
	j := 1
	j, a[j] = pair(w)
	h = mix(h, j*1000+a[0]+a[1]*10+a[2]*100+a[3]*7)
	// append(b[:n], x, b[n]): b[n] is read before x is stored there.
	b := make([]byte, 5, 6)
	copy(b, "1234"+string(rune('a'+w%26)))
	b = append(b[:len(b)-1], '5', b[len(b)-1])
	h = mix(h, int(b[4])*256+int(b[5]))
	// switch string(c) whose case changes c: the tag is c from before.
	c := []byte{'a'}
	switch string(c) {
	case func() string { c[0] = 'b'; return "b" }():
		h = mix(h, 1)
	case "a":
		h = mix(h, 2)
	}
	h = mix(h, int(c[0]))
	// Shifts by a count in a variable, unsigned and signed.
	n := uint(w%9 + 1)
	k := w%5 + 1
	var u8 uint8 = 0xb7
	var i8 int8 = -77
	var u16 uint16 = 0xbeef
	var i32 int32 = -123456
	h = mix(h, int(u8>>n)+int(u8<<n)+int(i8>>n)+int(i8<<n)+int(u16<<n)+int(u16>>(n+7))+int(i32>>n)+int(i32<<(n+20)))
	h = mix(h, int(u8<<k)+int(i8>>k)+int(u16>>k)+int(i32<<k))
	// Blank targets, and comparing a value with itself (fine in Go, a
	// warning in C++).
	_, _ = n, k
	if u8 == u8 && i32 == i32 {
		h = mix(h, 9)
	}
	// Empty structs: values, a slice of them and a set.
	set := map[string]Empty{}
	for r := range 3 + w%4 {
		set[string(rune('a'+r))] = Empty{}
	}
	_, has := set["c"]
	es := []Empty{{}, {}}
	es = append(es, Empty{})
	e1, e2 := Empty{}, es[1]
	if has && e1 == e2 {
		h = mix(h, len(es)+len(set)*10)
	}
	// Range into a map's element, and into a variable that exists: changing
	// the variable in the body does not change the turns, and after the
	// loop it holds the last index.
	last := map[string]int{}
	for last["at"] = range "héllo" {
	}
	idx, turns := 0, 0
	for idx = range s[:3+w%2] {
		turns++
		idx += 10
	}
	h = mix(h, last["at"]*10000+idx*100+turns)
	return h
}

// Log records the order in which calls happen.
type Log struct {
	S []int
}

// Add records v and returns it.
func (l *Log) Add(v int) int {
	l.S = append(l.S, v)
	return v
}

func addThree(a, b, c int) int { return a*100 + b*10 + c }

func name(i int) string { return string(rune('a' + i%26)) }

// callorder checks that calls happen in Go's left-to-right order wherever
// C++ would leave the order open: the operands of an operator, a call's
// arguments, indexes and slices, literals (a struct's fields in the order
// written, not declared), an assignment's target before its value (and
// once for +=), string +, a loop's condition, and a method's receiver
// before its arguments.
func callorder(w int) int {
	var lg Log
	h := 0
	x := lg.Add(1) + lg.Add(2)*lg.Add(3+w%2)
	y := addThree(lg.Add(4), lg.Add(5), lg.Add(6))
	if lg.Add(7) < lg.Add(8) {
		h = mix(h, 1)
	}
	arr := []int{10, 20, 30, 40 + w}
	z := arr[lg.Add(0)] + arr[lg.Add(1):lg.Add(3)][lg.Add(1)]
	p := Point{Y: lg.Add(9), X: lg.Add(10)}
	sl := []int{lg.Add(11), lg.Add(12)}
	mp := map[int]int{lg.Add(13): lg.Add(14)}
	arr[lg.Add(2)] = lg.Add(15)
	mp[lg.Add(16)] = lg.Add(17) + len(mp)
	arr[lg.Add(3)] += lg.Add(18)
	str := name(lg.Add(19)) + name(lg.Add(20)) + name(lg.Add(21+w))
	for n := 0; n < 2 && lg.Add(22)+lg.Add(23) > 0; n++ {
	}
	logs := make([]Log, 2)
	logs[lg.Add(1)].Add(lg.Add(25))
	for _, v := range lg.S {
		h = mix(h, v)
	}
	h = mix(h, x+y+z+p.X*7+p.Y+sl[0]*3+sl[1]+len(mp)*11+mp[16]+arr[2]+arr[3]+len(logs[1].S)*13)
	return mix(h, int(str[0])+int(str[2])*5)
}

func main() {
	tag := "lang" + "check"
	res := tt.Parallel(workers, func(w int, out *tt.List[Result]) {
		out.Append(Result{w, 1, results(w)})
		out.Append(Result{w, 2, methods(w)})
		out.Append(Result{w, 3, control(w)})
		out.Append(Result{w, 4, matmul(w)})
		out.Append(Result{w, 5, grids(w)})
		out.Append(Result{w, 6, shifts(w)})
		out.Append(Result{w, 7, closures(w)})
		out.Append(Result{w, 8, recursion(w)})
		out.Append(Result{w, 9, appends(w)})
		out.Append(Result{w, 10, strs(w)})
		out.Append(Result{w, 11, churn(w, tag)})
		out.Append(Result{w, 12, bags(w)})
		out.Append(Result{w, 13, maps(w)})
		out.Append(Result{w, 14, textfuncs(w)})
		out.Append(Result{w, 15, overflow(w)})
		out.Append(Result{w, 16, bitfuncs(w)})
		out.Append(Result{w, 17, evalorder(w)})
		out.Append(Result{w, 18, callorder(w)})
	})
	for _, r := range res {
		fmt.Printf("worker %d test %d: %d\n", r.Worker, r.Test, r.Value)
	}
}
