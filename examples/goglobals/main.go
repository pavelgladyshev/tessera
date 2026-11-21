// goglobals runs a pipeline of goroutines that use package-level variables.
// Each core has its own copy of the ones its goroutine uses, set up when it
// starts as Go's package initialization would: the declared values, then
// init. Every goroutine reads the tables; the producer alone changes its
// counters, so no goroutine needs to see another's changes (tessc refuses
// a program where one would).
package main

const stages = 4

var (
	weights = makeWeights(16)
	names   = []string{"alpha", "beta", "gamma", "delta"}
	primes  [8]int
	bias    = len(names[2]) * 3 // set before init runs

	sent  int    // changed by the producer alone
	last  string // the same
	spent = map[string]int{}
)

func makeWeights(n int) []int {
	w := make([]int, n)
	for i := range w {
		w[i] = (i*7 + 3) % 11
	}
	return w
}

func init() {
	n := 0
	for k := 2; n < len(primes); k++ {
		prime := true
		for d := 2; d*d <= k; d++ {
			if k%d == 0 {
				prime = false
				break
			}
		}
		if prime {
			primes[n] = k
			n++
		}
	}
	primes[0] += bias
}

func produce(out chan<- int) {
	for i := 0; i < 40; i++ {
		v := weights[i%len(weights)] * primes[i%len(primes)]
		sent++
		last = names[i%len(names)] + "!"
		spent[last] += v
		out <- v
	}
	out <- sent*1000 + len(last)*100 + len(spent)*10 + spent["beta!"]%10
	close(out)
}

func stage(id int, in <-chan int, out chan<- int) {
	for v := range in {
		out <- v*primes[id] + len(names[id%len(names)])
	}
	close(out)
}

func main() {
	links := make([]chan int, stages+1)
	for i := range links {
		links[i] = make(chan int, 4)
	}
	go produce(links[0])
	for i := 0; i < stages; i++ {
		go stage(i, links[i], links[i+1])
	}
	n, sum := 0, 0
	for v := range links[stages] {
		n++
		sum += v
	}
	println(n, sum)
}
