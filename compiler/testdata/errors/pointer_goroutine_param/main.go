// ERROR "work: parameter p is *int; goroutines take channels"
package main

func work(p *int, out chan<- int) {
	out <- *p
}

func main() {
	out := make(chan int, 1)
	n := 3
	go work(&n, out)
	println(<-out)
}
