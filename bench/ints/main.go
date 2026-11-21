// ints times integer arithmetic on one worker core: 64-bit int against
// 32-bit, for addition, multiplication and division, and a loop over an
// array. It uses no interfaces, function values, defer or recursion, so
// it runs on the core's own stack; bench/run.sh also builds it with
// -l1stack and with -overflow=undefined.
//
//	ints -mode M -n N    prints the result and how long tt.Parallel took
package main

import (
	"flag"
	"fmt"
	"time"

	"github.com/pavelgladyshev/tessera/tt"
)

type R struct{ V int }

// The loops depend on their last values, so the compiler cannot drop them.

func add64(n int) int {
	s := 1
	for i := 0; i < n; i++ {
		s += i ^ (s >> 3)
	}
	return s
}

func add32(n int) int {
	var s int32 = 1
	for i := int32(0); i < int32(n); i++ {
		s += i ^ (s >> 3)
	}
	return int(s)
}

func mul64(n int) int {
	s := uint64(1)
	for i := 0; i < n; i++ {
		s = s*6364136223846793005 + 1442695040888963407
	}
	return int(s >> 1)
}

func mul32(n int) int {
	s := uint32(1)
	for i := int32(0); i < int32(n); i++ {
		s = s*1664525 + 1013904223
	}
	return int(s)
}

func div64(n int) int {
	s := 0
	big := 1<<40 + 12345
	for i := 1; i <= n; i++ {
		s += big / i
	}
	return s
}

func div32(n int) int {
	var s int32
	big := int32(1<<30 + 12345)
	for i := int32(1); i <= int32(n); i++ {
		s += big / i
	}
	return int(s)
}

// array walks an array in L1 with the index from the data, so the bounds
// check stays.
func array(n int) int {
	var buf [1024]int32
	for i := range buf {
		buf[i] = int32(i)
	}
	s := int32(0)
	j := int32(0)
	for i := int32(0); i < int32(n); i++ {
		j = (j*5 + buf[j&1023] + 1) & 1023
		s += buf[j]
		buf[j] = s
	}
	return int(s)
}

func run(mode, n int) int {
	switch mode {
	case 1:
		return add64(n)
	case 2:
		return add32(n)
	case 3:
		return mul64(n)
	case 4:
		return mul32(n)
	case 5:
		return div64(n)
	case 6:
		return div32(n)
	case 7:
		return array(n)
	}
	return 0 // the fixed cost of starting the worker
}

func main() {
	mode := flag.Int("mode", 0, "0: nothing, 1: int add, 2: int32 add, 3: uint64 mul, 4: uint32 mul, 5: int div, 6: int32 div, 7: array")
	n := flag.Int("n", 1000, "iterations")
	flag.Parse()
	m, k := *mode, *n
	start := time.Now()
	res := tt.Parallel(1, func(w int, out *tt.List[R]) {
		out.Append(R{run(m, k)})
	})
	fmt.Printf("mode %d n %d result %d ms %.3f\n", m, k, res[0].V, float64(time.Since(start).Microseconds())/1000)
}
