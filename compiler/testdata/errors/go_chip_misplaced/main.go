// ERROR "//tt:chip goes on the line right before a go statement in main"
package main

func work(out chan<- int) {
	out <- 1
	close(out)
}

func main() {
	out := make(chan int, 1)
	//tt:chip 1

	go work(out)
	println(<-out)
}
