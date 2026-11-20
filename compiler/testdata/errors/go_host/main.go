// ERROR "//tt:host is not supported yet"
package main

func work(out chan<- int) {
	out <- 1
	close(out)
}

func main() {
	out := make(chan int, 1)
	//tt:host
	go work(out)
	println(<-out)
}
