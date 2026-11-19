// ERROR "declare the direction of channel parameter c"
package main

func produce(c chan int) {
	c <- 1
	close(c)
}

func main() {
	c := make(chan int, 1)
	go produce(c)
	println(<-c)
}
