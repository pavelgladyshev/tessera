// ERROR "main uses package-level variable rounds: in a program that starts goroutines, main runs on the host in C++"
package main

var rounds = 3

func work(n int, out chan<- int) {
	out <- n * 2
}

func main() {
	out := make(chan int, 1)
	go work(rounds, out)
	println(<-out)
}
