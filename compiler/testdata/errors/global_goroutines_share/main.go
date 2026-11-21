// ERROR "total is changed by goroutine produce and used by goroutine report"
package main

var total int

func produce(out chan<- int) {
	for i := 0; i < 10; i++ {
		total += i
		out <- i
	}
	close(out)
}

func report(in <-chan int, done chan<- int) {
	for range in {
	}
	done <- total
}

func main() {
	ch := make(chan int, 4)
	done := make(chan int, 1)
	go produce(ch)
	go report(ch, done)
	println(<-done)
}
