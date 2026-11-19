// ERROR "function literals are not supported in goroutine programs"
package main

func main() {
	c := make(chan int, 1)
	go func() { c <- 1 }()
	println(<-c)
}
