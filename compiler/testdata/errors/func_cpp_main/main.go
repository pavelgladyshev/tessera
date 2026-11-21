// ERROR "function values are not supported in main"
package main

func work(n int, out chan<- int) {
	out <- n * 2
}

func double(n int) int { return n * 2 }

func main() {
	out := make(chan int, 1)
	f := double
	go work(f(3), out)
	println(<-out)
}
