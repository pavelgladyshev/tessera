// ERROR "main cannot send on channels yet"
package main

func double(in <-chan int, out chan<- int) {
	for v := range in {
		out <- 2 * v
	}
	close(out)
}

func main() {
	in, out := make(chan int, 1), make(chan int, 1)
	go double(in, out)
	in <- 21
	close(in)
	println(<-out)
}
