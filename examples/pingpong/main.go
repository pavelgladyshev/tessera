// pingpong sends values both ways between the n300's two chips.
//
// Each of 60 pairs of goroutines has one goroutine on each chip. ping sends
// 1,000 values to pong, which sends their sum back; main collects the sums.
// That is 120 channels between the chips, which go over the one link the
// fabric leaves to programs through a fabric mux on each chip. The //tt:chip
// directives put each ping on the first chip and each pong on the second;
// go run ignores them.
package main

const (
	pairs = 60
	n     = 1000
)

// Sum is what a pair reports.
type Sum struct{ Pair, Total int }

func ping(k int, out chan<- int, back <-chan int, res chan<- Sum) {
	for i := range n {
		out <- i*pairs + k
	}
	close(out)
	res <- Sum{k, <-back}
	close(res)
}

func pong(in <-chan int, back chan<- int) {
	total := 0
	for v := range in {
		total += v
	}
	back <- total
	close(back)
}

func main() {
	var out, back [pairs]chan int
	var res [pairs]chan Sum
	for k := range pairs {
		out[k] = make(chan int, 16)
		back[k] = make(chan int, 1)
		res[k] = make(chan Sum, 1)
	}
	for k := range pairs {
		//tt:chip 0
		go ping(k, out[k], back[k], res[k])
	}
	for k := range pairs {
		//tt:chip 1
		go pong(out[k], back[k])
	}
	all := 0
	for k := range pairs {
		s := <-res[k]
		all += s.Total
		if k%10 == 0 {
			println("pair", s.Pair, "sum", s.Total)
		}
	}
	println("all pairs", all)
}
