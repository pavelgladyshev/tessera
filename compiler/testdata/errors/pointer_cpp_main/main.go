// ERROR "taking addresses is not supported in main"
package main

func work(n int, out chan<- int) {
	out <- n * 2
}

func main() {
	out := make(chan int, 1)
	n := 3
	p := &n
	go work(*p, out)
	println(<-out)
}
