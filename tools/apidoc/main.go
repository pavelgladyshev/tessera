// apidoc writes a package's exported API as a Markdown page for the
// documentation site, from its doc comments:
//
//	go run ./tools/apidoc -title "Package tt" -order 2 ./tt > docs/reference/tt.md
package main

import (
	"bytes"
	"flag"
	"fmt"
	"go/ast"
	"go/doc"
	"go/parser"
	"go/printer"
	"go/token"
	"os"
	"sort"
	"strings"
)

func main() {
	title := flag.String("title", "", "the page's title")
	order := flag.Int("order", 1, "the page's place among the reference pages")
	flag.Parse()
	dir := flag.Arg(0)
	fset := token.NewFileSet()
	pkgs, err := parser.ParseDir(fset, dir, func(f os.FileInfo) bool { return !strings.HasSuffix(f.Name(), "_test.go") }, parser.ParseComments)
	if err != nil || len(pkgs) != 1 {
		fmt.Fprintln(os.Stderr, "apidoc: cannot read the package in", dir, err)
		os.Exit(1)
	}
	var files []*ast.File
	for _, p := range pkgs {
		for _, f := range p.Files {
			files = append(files, f)
		}
	}
	sort.Slice(files, func(a, b int) bool { return fset.File(files[a].Pos()).Name() < fset.File(files[b].Pos()).Name() })
	p, err := doc.NewFromFiles(fset, files, "github.com/pavelgladyshev/tessera/"+strings.TrimPrefix(dir, "./"))
	if err != nil {
		fmt.Fprintln(os.Stderr, "apidoc:", err)
		os.Exit(1)
	}
	source := func(n any) string {
		var b bytes.Buffer
		printer.Fprint(&b, fset, n)
		return b.String()
	}
	text := func(s string) string { return strings.TrimSpace(s) + "\n\n" }
	fn := func(f *doc.Func, level string) {
		f.Decl.Body = nil
		name := f.Name
		if f.Recv != "" {
			name = "(" + f.Recv + ") " + name
		}
		fmt.Printf("%s %s\n\n```go\n%s\n```\n\n%s", level, name, source(f.Decl), text(f.Doc))
	}
	fmt.Printf("---\ntitle: %s\nparent: Reference\nnav_order: %d\n---\n\n# %s\n\n", *title, *order, *title)
	fmt.Printf("```go\nimport \"%s\"\n```\n\n", p.ImportPath)
	fmt.Print("This page is made from the package's doc comments by `tools/apidoc`.\n{: .fs-2 }\n\n")
	fmt.Print(text(p.Doc))
	if len(p.Funcs) > 0 {
		fmt.Print("## Functions\n\n")
		for _, f := range p.Funcs {
			fn(f, "###")
		}
	}
	if len(p.Types) > 0 {
		fmt.Print("## Types\n\n")
		for _, t := range p.Types {
			fmt.Printf("### type %s\n\n```go\n%s\n```\n\n%s", t.Name, source(t.Decl), text(t.Doc))
			for _, f := range t.Funcs {
				fn(f, "####")
			}
			for _, f := range t.Methods {
				fn(f, "####")
			}
		}
	}
	if len(p.Consts)+len(p.Vars) > 0 {
		fmt.Print("## Constants and variables\n\n")
		for _, v := range append(p.Consts, p.Vars...) {
			fmt.Printf("```go\n%s\n```\n\n%s", source(v.Decl), text(v.Doc))
		}
	}
}
