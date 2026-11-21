// ERROR "defer is not supported in main"
package main

func work(n int, out chan<- int) {
	defer close(out)
	out <- n * 2
}

func main() {
	out := make(chan int, 1)
	defer println("done")
	go work(3, out)
	println(<-out)
}
