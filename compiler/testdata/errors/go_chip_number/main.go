// ERROR "write //tt:chip N"
package main

func work(out chan<- int) {
	out <- 1
	close(out)
}

func main() {
	out := make(chan int, 1)
	//tt:chip one
	go work(out)
	println(<-out)
}
