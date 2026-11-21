package main

// The documents the program answers from: notes about Tessera itself,
// here so that the example needs no files, and any .md and .txt files of
// the directory given with -docs. They are cut into passages short enough
// for a few to fit in the model's context.

import (
	"os"
	"path/filepath"
	"strings"
)

type passage struct {
	source, text string
}

const notes = `
Tessera is a subset of the Go language that the compiler tessc turns into TT-Metalium programs for Tenstorrent Wormhole cards such as the n300. The same program also runs on an ordinary computer with go run, which is how programs are tested without a card.

An n300 card has two Wormhole chips. Each chip has 64 Tensix cores for workers, so a Tessera program can run up to 128 workers on one card. tt.Cores returns that number.

A worker is the function given to tt.Parallel. Each worker runs on one Tensix core, on the core's RISC-V processor, and appends its results to a list. tt.Parallel returns the results of all workers, worker 0 first.

Worker code is integer Go. It can use structs, arrays, slices, strings, maps, pointers, interfaces, function values, closures, generics, defer, panic and recover. It has no floating point of its own, because the cores' processors have none.

The cores are 32-bit RISC-V processors. Go's int is 64 bits, so arithmetic on int takes several instructions. In hot loops int32 and uint32 are three to five times faster than int.

The main function of a worker program stays ordinary Go and runs on the host computer. It can use the whole standard library. A small helper process written in C++ owns the card and does what main asks.

tt.LoadBytes copies a file to the card's memory for the workers to read. tt.OpenBytes reads only the parts of a file that the workers ask for, which suits a carver that probes a disk image sparsely.

Tensors in worker code use the matrix engine of each core. main makes a tt.Layer or a tt.Matrix from float32 numbers, and a worker fills tensors, runs tt.Linear on them and reads results back as integers.

Package tt/nn lets main call ttnn, Tenstorrent's library of neural network operations, such as matrix products, convolutions, attention and softmax. On the card the numbers are bf16, which keeps 8 bits of mantissa. Under go run the same operations run in float32.

ttnn and the workers take turns on the card. An nn operation called while tt.Parallel is running waits until the workers have finished.

tt.LayerOf makes a layer for the workers from tensors that ttnn computed, without copying them: the workers read ttnn's own buffers.

Each worker has a small stack in its core's local memory, about 3 kilobytes. A program that recurses, uses defer, or needs more stack runs its workers on a stack of 64 kilobytes in L1 memory instead, which is slower. The compiler flag -l1stack forces that choice.

A worker's heap is 256 kilobytes by default. Running out of heap or stack stops the program with a panic that names the line of Go source.

A worker can append at most 4096 results to its list. A result is a struct of integers and booleans.

If workers run for more than ten seconds the helper prints a notice. The environment variable TESSERA_TIMEOUT gives a number of seconds after which a run is stopped. After a stopped run the card needs a reset with tt-smi -r.

Goroutine programs use go statements, channels and select. Each goroutine runs on its own Tensix core, and channels between cores on different chips go through the fabric over the Ethernet links between the chips.

Elementwise programs use tt.Map to apply a function to every element of an array of float32. The compiler turns the function into instructions for the cores' vector units.

The example galaxy simulates two galaxies colliding under gravity with 4096 bodies in fixed point arithmetic. One step takes about 57 milliseconds on 128 cores.

The example carver finds JPEG, PNG, GIF, PDF, ZIP and BMP files in a disk image by their signatures and writes a page with a map of the disk.

The example upscale is a web server that makes pictures four times bigger with the compact Real-ESRGAN network. On the card the network takes about 73 milliseconds for a picture of 240 by 180 pixels.

To build a program for the card, run tessc on its directory and then the build.sh script it writes. The script needs the environment variable TT_METAL_HOME to point at a tt-metal checkout.

Tessera is written by Pavel Gladyshev. It is open source under the MIT licence.
`

// split cuts a document into passages of at most about 60 words. It reads
// Markdown: a passage carries the heading it stands under, and code
// blocks and tables are left out (a model this small makes nothing of
// them).
func split(source, text string) []passage {
	var out []passage
	heading := ""
	var para []string
	flush := func() {
		words := strings.Fields(plain(strings.Join(para, " ")))
		para = para[:0]
		if len(words) < 6 {
			return
		}
		for len(words) > 0 {
			n := min(len(words), 60)
			if n < len(words) { // back to the end of a sentence, if one is near
				for k := n; k > n-25; k-- {
					if strings.HasSuffix(words[k-1], ".") {
						n = k
						break
					}
				}
			}
			t := strings.Join(words[:n], " ")
			if heading != "" {
				t = heading + ": " + t
			}
			out = append(out, passage{source, t})
			words = words[n:]
		}
	}
	code := false
	for _, line := range strings.Split(text, "\n") {
		trimmed := strings.TrimSpace(line)
		switch {
		case strings.HasPrefix(trimmed, "```"):
			flush()
			code = !code
		case code, strings.HasPrefix(trimmed, "|"), strings.HasPrefix(line, "    "), strings.HasPrefix(line, "\t"):
			// code, tables
		case trimmed == "":
			flush()
		case strings.HasPrefix(trimmed, "#"):
			flush()
			heading = plain(strings.TrimSpace(strings.TrimLeft(trimmed, "#")))
		case strings.HasPrefix(trimmed, "- "), strings.HasPrefix(trimmed, "* "):
			flush() // every item of a list is a passage of its own
			para = append(para, trimmed[2:])
		default:
			para = append(para, trimmed)
		}
	}
	flush()
	return out
}

// plain takes Markdown's marks out of a line: [text](link) becomes text,
// and the signs for code and emphasis go.
func plain(s string) string {
	var b strings.Builder
	for i := 0; i < len(s); i++ {
		switch s[i] {
		case '`', '*':
		case '[':
			if end := strings.IndexByte(s[i:], ']'); end > 0 && i+end+1 < len(s) && s[i+end+1] == '(' {
				if close := strings.IndexByte(s[i+end:], ')'); close > 0 {
					b.WriteString(s[i+1 : i+end])
					i += end + close
					continue
				}
			}
			b.WriteByte(s[i])
		default:
			b.WriteByte(s[i])
		}
	}
	return b.String()
}

func loadPassages(dir string) []passage {
	out := split("notes on Tessera", notes)
	if dir == "" {
		return out
	}
	filepath.WalkDir(dir, func(path string, d os.DirEntry, err error) error {
		if err != nil || d.IsDir() || !strings.HasSuffix(path, ".md") && !strings.HasSuffix(path, ".txt") {
			return nil
		}
		if raw, err := os.ReadFile(path); err == nil {
			out = append(out, split(path, string(raw))...)
		}
		return nil
	})
	return out
}
