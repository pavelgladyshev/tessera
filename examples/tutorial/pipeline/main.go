// pipeline passes numbers through a chain of goroutines: the tutorial's
// goroutine program. Each goroutine runs on a Tensix core of its own, and
// the channels between them carry values from core to core.
package main

const (
	numbers = 1000
	stages  = 8
)

// generate sends the numbers 1 to n.
func generate(n int, out chan<- int) {
	for i := 1; i <= n; i++ {
		out <- i
	}
	close(out)
}

// stage passes on what it receives, with step added.
func stage(step int, in <-chan int, out chan<- int) {
	for x := range in {
		out <- x + step
	}
	close(out)
}

func main() {
	links := make([]chan int, stages+1)
	for i := range links {
		links[i] = make(chan int, 16)
	}
	go generate(numbers, links[0])
	for i := 0; i < stages; i++ {
		go stage(i+1, links[i], links[i+1])
	}
	// main receives on the host.
	count, sum := 0, 0
	for x := range links[stages] {
		count++
		sum += x
	}
	println(count, "numbers came through", stages, "stages; their sum is", sum)
}
