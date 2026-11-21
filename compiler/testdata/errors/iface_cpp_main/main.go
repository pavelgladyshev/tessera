// ERROR "interfaces are not supported in main"
package main

func work(n int, out chan<- int) {
	out <- n * 2
}

func main() {
	out := make(chan int, 1)
	var x any = 3
	go work(x.(int), out)
	println(<-out)
}
