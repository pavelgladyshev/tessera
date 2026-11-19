// ERROR "select is not supported yet"
package main

func merge(a, b <-chan int, out chan<- int) {
	select {
	case v := <-a:
		out <- v
	case v := <-b:
		out <- v
	}
	close(out)
}

func gen(out chan<- int) {
	out <- 1
	close(out)
}

func main() {
	a, b, out := make(chan int, 1), make(chan int, 1), make(chan int, 1)
	go gen(a)
	go gen(b)
	go merge(a, b, out)
	println(<-out)
}
