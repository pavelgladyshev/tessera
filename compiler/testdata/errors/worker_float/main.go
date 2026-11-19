// ERROR "no floating-point unit"
package main

import "tessera/tt"

type R struct{ V int }

func main() {
	img := tt.LoadBytes("disk.img")
	_ = tt.Parallel(4, func(w int, out *tt.List[R]) {
		x := float32(img.Len()) * 0.5
		out.Append(R{V: int(x)})
	})
}
