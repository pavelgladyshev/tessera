// ERROR "apply has type parameters of its own: generic methods are not supported on the card"
package main

import "github.com/pavelgladyshev/tessera/tt"

type R struct{ N int }

type List[E any] []E

func (l List[E]) apply[P any](f func(E) P) List[P] {
	out := make(List[P], len(l))
	for i, x := range l {
		out[i] = f(x)
	}
	return out
}

func main() {
	res := tt.Parallel(2, func(w int, out *tt.List[R]) {
		l := List[int]{w, 2}
		out.Append(R{len(l.apply(func(x int) int { return x * 2 }))})
	})
	println(len(res))
}
