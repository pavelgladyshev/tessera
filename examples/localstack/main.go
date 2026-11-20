// localstack runs workers whose calls do not recurse but can go deeper than
// the core's own stack (about 2.2 KB of its local memory) has room for: six
// levels of calls that each keep 768 bytes of structs. GCC inlines them into
// one frame of about 4.6 KB. tessc estimates the stack the calls can need
// (about 5.5 KB) and runs the workers on a stack in L1 instead, so that
// every depth gives go run's result on the card. (Before, the frame ran
// through the core's globals and the cores hung.)
//
//	localstack -depth 2
//	localstack -depth 6
package main

import (
	"flag"
	"fmt"

	"github.com/pavelgladyshev/tessera/tt"
)

// Big is the largest struct with an array in it that worker code can have.
type Big struct{ A [32]int64 }

type R struct{ W, V int }

// level1 keeps three Bigs (768 bytes) on the stack while it calls the next level.
func level1(k, depth int) int {
	var a, b, c Big
	for i := range a.A {
		a.A[i] = int64(i*1 + k)
		b.A[i] = a.A[i] * 3
		c.A[i] = b.A[i] ^ int64(k)
	}
	s := 0
	if depth > 1 {
		s = level2(k+1, depth)
	}
	return s + int(a.A[k%32]+b.A[(k+1)%32]+c.A[(k+2)%32])
}

// level2 keeps three Bigs (768 bytes) on the stack while it calls the next level.
func level2(k, depth int) int {
	var a, b, c Big
	for i := range a.A {
		a.A[i] = int64(i*2 + k)
		b.A[i] = a.A[i] * 3
		c.A[i] = b.A[i] ^ int64(k)
	}
	s := 0
	if depth > 2 {
		s = level3(k+1, depth)
	}
	return s + int(a.A[k%32]+b.A[(k+1)%32]+c.A[(k+2)%32])
}

// level3 keeps three Bigs (768 bytes) on the stack while it calls the next level.
func level3(k, depth int) int {
	var a, b, c Big
	for i := range a.A {
		a.A[i] = int64(i*3 + k)
		b.A[i] = a.A[i] * 3
		c.A[i] = b.A[i] ^ int64(k)
	}
	s := 0
	if depth > 3 {
		s = level4(k+1, depth)
	}
	return s + int(a.A[k%32]+b.A[(k+1)%32]+c.A[(k+2)%32])
}

// level4 keeps three Bigs (768 bytes) on the stack while it calls the next level.
func level4(k, depth int) int {
	var a, b, c Big
	for i := range a.A {
		a.A[i] = int64(i*4 + k)
		b.A[i] = a.A[i] * 3
		c.A[i] = b.A[i] ^ int64(k)
	}
	s := 0
	if depth > 4 {
		s = level5(k+1, depth)
	}
	return s + int(a.A[k%32]+b.A[(k+1)%32]+c.A[(k+2)%32])
}

// level5 keeps three Bigs (768 bytes) on the stack while it calls the next level.
func level5(k, depth int) int {
	var a, b, c Big
	for i := range a.A {
		a.A[i] = int64(i*5 + k)
		b.A[i] = a.A[i] * 3
		c.A[i] = b.A[i] ^ int64(k)
	}
	s := 0
	if depth > 5 {
		s = level6(k+1, depth)
	}
	return s + int(a.A[k%32]+b.A[(k+1)%32]+c.A[(k+2)%32])
}

// level6 keeps three Bigs (768 bytes) on the stack while it calls the next level.
func level6(k, depth int) int {
	var a, b, c Big
	for i := range a.A {
		a.A[i] = int64(i*6 + k)
		b.A[i] = a.A[i] * 3
		c.A[i] = b.A[i] ^ int64(k)
	}
	s := 0
	return s + int(a.A[k%32]+b.A[(k+1)%32]+c.A[(k+2)%32])
}

func main() {
	depthFlag := flag.Int("depth", 2, "how many levels of calls")
	flag.Parse()
	depth := *depthFlag
	res := tt.Parallel(2, func(w int, out *tt.List[R]) {
		out.Append(R{w, level1(w, depth)})
	})
	fmt.Println(res)
}
