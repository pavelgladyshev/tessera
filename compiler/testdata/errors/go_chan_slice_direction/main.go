// ERROR "declare the direction of channel parameter in"
package main

func sum(in []chan int, out chan<- int) {
	total := 0
	for _, c := range in {
		total += <-c
	}
	out <- total
	close(out)
}

func main() {
	in := make([]chan int, 2)
	in[0], in[1] = make(chan int, 1), make(chan int, 1)
	out := make(chan int, 1)
	go sum(in, out)
	in[0] <- 1
	in[1] <- 2
	println(<-out)
}
