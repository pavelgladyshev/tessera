// ERROR "seen is changed by goroutine work, which runs as more than one goroutine"
package main

var seen int

func work(id int, out chan<- int) {
	seen++
	out <- id + seen
}

func main() {
	var outs [4]chan int
	for i := range outs {
		outs[i] = make(chan int, 1)
		go work(i, outs[i])
	}
	for i := range outs {
		println(<-outs[i])
	}
}
