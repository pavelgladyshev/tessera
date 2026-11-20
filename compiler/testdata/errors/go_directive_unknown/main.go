// ERROR "unknown directive //tt:chips"
package main

func work(out chan<- int) {
	out <- 1
	close(out)
}

func main() {
	out := make(chan int, 1)
	//tt:chips 1
	go work(out)
	println(<-out)
}
