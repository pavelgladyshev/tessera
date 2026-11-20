// ERROR "channels cannot carry strings, slices or maps"
package main

func produce(out chan<- string) {
	out <- "hello"
	close(out)
}

func main() {
	ch := make(chan string, 4)
	go produce(ch)
	for s := range ch {
		println(s)
	}
}
