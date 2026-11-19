// sieve counts the primes below limit with a pipeline of goroutines, a
// static form of the concurrent prime sieve from the Go documentation.
//
// A generator sends 2, 3, 4, ... down a chain of stages. Each stage keeps
// the first number that no earlier stage has divided (it is prime), marks it
// as a prime, and passes on only the numbers it does not divide. On the card
// every goroutine has a Tensix core of its own, and the numbers move from
// core to core over the network-on-chip.
package main

const (
	stages = 54    // the first 54 primes, 2 ... 251, one stage each
	limit  = 63000 // below 251*251, so every number that survives all stages is prime
	depth  = 8     // values each channel can hold
)

// Number is a value on its way down the sieve.
type Number struct {
	N     int
	Prime bool // found to be prime by an earlier stage: pass it on
}

// generate sends 2, 3, ..., limit-1 and closes out.
func generate(out chan<- Number) {
	for n := 2; n < limit; n++ {
		out <- Number{N: n}
	}
	close(out)
}

// stage keeps the first number it receives that is not already known to be
// prime; that number is prime. It passes on known primes, its own prime, and
// the numbers its prime does not divide.
func stage(in <-chan Number, out chan<- Number) {
	p := 0
	for x := range in {
		switch {
		case x.Prime:
			out <- x
		case p == 0:
			p = x.N
			out <- Number{N: p, Prime: true}
		case x.N%p != 0:
			out <- x
		}
	}
	close(out)
}

func main() {
	chans := make([]chan Number, stages+1)
	for i := range chans {
		chans[i] = make(chan Number, depth)
	}
	go generate(chans[0])
	for i := 0; i < stages; i++ {
		go stage(chans[i], chans[i+1])
	}
	count, sum, largest := 0, 0, 0
	for x := range chans[stages] {
		count++
		sum += x.N
		largest = x.N
	}
	println(count, "primes below", limit, "- sum", sum, "- largest", largest)
}
