// Command tessc compiles a Tessera program (a Go package that uses
// github.com/pavelgladyshev/tenstorrent-go/tt) into a TT-Metalium C++
// project for a Tenstorrent Wormhole card.
//
// Usage:
//
//	tessc [-o dir] [-v] [-host auto|go|cpp] [package dir]
//
// The output directory gets host.cpp, kernels/{reader,compute,writer}.cpp,
// CMakeLists.txt, build.sh and program.json. Build it on the machine with the
// card: TT_METAL_HOME=... ./build.sh
package main

import (
	"flag"
	"fmt"
	"os"
	"path/filepath"

	"github.com/pavelgladyshev/tenstorrent-go/compiler"
)

func main() {
	out := flag.String("o", "", "output directory (default build/<program>)")
	verbose := flag.Bool("v", false, "print the compute kernel's instruction list")
	host := flag.String("host", "auto", "how main runs on the host: go (stays Go), cpp (translated to C++), or auto (go for worker programs, cpp for the rest)")
	heap := flag.Int("heap", compiler.DefaultHeapKiB, "KiB of L1 for each worker's heap (append, make, strings)")
	stack := flag.Int("stack", compiler.DefaultStackKiB, "KiB of L1 for each worker's stack, if it recurses")
	overflow := flag.String("overflow", "wrap", "signed integer overflow on the card: wrap (as in Go) or undefined (as in C++; faster loops, but a program that overflows can give other results)")
	flag.Usage = func() {
		fmt.Fprintf(os.Stderr, "usage: tessc [-o dir] [-v] [-host auto|go|cpp] [-heap KiB] [-stack KiB] [-overflow wrap|undefined] [package dir]\n")
		flag.PrintDefaults()
	}
	flag.Parse()
	dir := "."
	switch flag.NArg() {
	case 0:
	case 1:
		dir = flag.Arg(0)
	default:
		flag.Usage()
		os.Exit(2)
	}

	if *heap < 1 || *stack < 4 || *heap+*stack > 1024 {
		fmt.Fprintf(os.Stderr, "tessc: -heap must be at least 1 and -stack at least 4 KiB, and together at most 1024 (each core has 1.5 MB of L1)\n")
		os.Exit(2)
	}
	if *overflow != "wrap" && *overflow != "undefined" {
		fmt.Fprintf(os.Stderr, "tessc: -overflow is wrap or undefined, not %q\n", *overflow)
		os.Exit(2)
	}
	opts := compiler.Options{HeapKiB: *heap, StackKiB: *stack, UndefinedOverflow: *overflow == "undefined"}
	switch *host {
	case "auto":
	case "go":
		opts.Host = compiler.HostGo
	case "cpp":
		opts.Host = compiler.HostCPP
	default:
		fmt.Fprintf(os.Stderr, "tessc: -host is auto, go or cpp, not %q\n", *host)
		os.Exit(2)
	}
	prog, err := compiler.LoadWith(dir, opts)
	if err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
	if *out == "" {
		*out = filepath.Join("build", prog.Name)
	}
	if prog.Workers != nil {
		files, err := compiler.EmitWorkers(prog.Workers, *out)
		if err != nil {
			fmt.Fprintln(os.Stderr, err)
			os.Exit(1)
		}
		kind := "worker program (tt.Parallel, one worker per Tensix core)"
		if prog.Workers.Goroutines() > 0 {
			kind = fmt.Sprintf("goroutine program (%d goroutine functions, one goroutine per Tensix core)", prog.Workers.Goroutines())
		}
		fmt.Printf("%s: %s\n", prog.Name, kind)
		fmt.Printf("wrote %s (%d files)\n", *out, len(files))
		return
	}
	kernel, err := compiler.Lower(prog)
	if err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
	if *out == "" {
		*out = filepath.Join("build", prog.Name)
	}
	files, err := compiler.Emit(prog, kernel, *out)
	if err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}

	tiles := (prog.Elems() + 1023) / 1024
	fmt.Printf("%s: %dx%d arrays, %d tiles of 32x32; %d inputs, %d outputs\n",
		prog.Name, prog.Rows, prog.Cols, tiles, len(prog.Inputs), len(prog.Outputs))
	sync := "double-buffered"
	if kernel.FullSync {
		sync = "single-buffered"
	}
	fmt.Printf("compute kernel: %d SFPU instructions per tile, %d DEST slots (%s)\n", len(kernel.Instrs), kernel.Slots, sync)
	if *verbose {
		for _, in := range kernel.Instrs {
			fmt.Printf("  %-32s %s\n", in.Fn, in.Comment)
		}
		for _, p := range kernel.Packs {
			fmt.Printf("  %-32s slot %d -> %s\n", "pack_tile", p.Slot, prog.Outputs[p.Output].Name)
		}
	}
	fmt.Printf("wrote %s (%d files)\n", *out, len(files))
}
