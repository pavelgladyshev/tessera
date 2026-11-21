// ERROR "package-level channels are not supported"
package main

var results = make(chan int, 4)

func work(n int) {
	results <- n
}

func main() {
	go work(1)
	println(1)
}
