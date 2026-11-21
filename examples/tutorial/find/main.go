// find looks for a word in a file: the tutorial's second program. The file
// goes to the card's memory, every worker searches a share of it, and main
// prints the lines the word is on.
//
//	go run ./examples/tutorial/find -file README.md -word Tessera
package main

import (
	"flag"
	"fmt"
	"os"
	"sort"
	"strings"

	"github.com/pavelgladyshev/tessera/tt"
)

// Match is one place where the word was found.
type Match struct {
	Offset int
}

func main() {
	path := flag.String("file", "", "the file to search")
	word := flag.String("word", "", "the word to look for")
	flag.Parse()
	if *path == "" || *word == "" || len(*word) > 64 {
		fmt.Fprintln(os.Stderr, "usage: find -file FILE -word WORD")
		os.Exit(2)
	}

	text := tt.LoadBytes(*path) // on the card: copied to its memory
	size, want := text.Len(), *word
	workers := min(tt.Cores(), max(1, size/1024))

	matches := tt.Parallel(workers, func(w int, out *tt.List[Match]) {
		var buf [4096]byte // an array declared with var lives in the core's L1 memory
		first, last := size*w/workers, size*(w+1)/workers
		// Read this worker's share a buffer at a time. Each read starts
		// len(want)-1 bytes before the next one's start, so a word that
		// straddles two buffers, or two workers' shares, is still found.
		for off := first; off < last; off += len(buf) - (len(want) - 1) {
			got := text.Read(buf[:], off)
			for i := 0; i+len(want) <= got && off+i < last; i++ {
				same := true
				for k := 0; k < len(want) && same; k++ {
					same = buf[i+k] == want[k]
				}
				if same {
					out.Append(Match{off + i})
				}
			}
		}
	})

	// The rest is ordinary Go on the host: show each match's line.
	sort.Slice(matches, func(a, b int) bool { return matches[a].Offset < matches[b].Offset })
	raw, err := os.ReadFile(*path)
	if err != nil {
		fmt.Fprintln(os.Stderr, "find:", err)
		os.Exit(1)
	}
	fmt.Printf("%q is in %s %d times\n", want, *path, len(matches))
	for i, m := range matches {
		if i == 5 {
			fmt.Println("...")
			break
		}
		start := strings.LastIndexByte(string(raw[:m.Offset]), '\n') + 1
		end := m.Offset + strings.IndexByte(string(raw[m.Offset:])+"\n", '\n')
		fmt.Printf("line %d: %s\n", strings.Count(string(raw[:m.Offset]), "\n")+1, strings.TrimSpace(string(raw[start:end])))
	}
}
