package compiler

import (
	_ "embed"
	"encoding/json"
	"fmt"
	"go/ast"
	"go/format"
	"os"
	"path/filepath"
	"sort"
	"strings"
)

// Worker programs whose main stays Go (Options.Host). The project gets:
//
//	host/             the program, with its tt.Parallel rewritten into
//	                  tt.Launch, as a Go module that builds on its own (it
//	                  carries a copy of package tt)
//	card.cpp          the card helper: the C++ host runtime and a loop that
//	                  serves the Go program's requests (package tt/card)
//	kernels/          the worker kernel (and the detector's)
//	build.sh          builds the helper with CMake and the program with Go,
//	                  both into build/
//
// The Go program starts the helper the first time it uses the card.

//go:embed runtime/card_main.cpp
var cardMain string

func emitGoHost(wp *WorkerProgram, dir, kernel string, scratch uint32) ([]string, error) {
	exe := cident(wp.Name)
	out := map[string]string{
		"kernels/worker.cpp": kernel,
		"card.cpp": fmt.Sprintf(`// Card helper for %s. %s
//
// The program's main runs as Go (build/%s); this process owns the card and
// does what the Go program asks: it puts files and the detector on the card
// and runs the workers of tt.Launch.
%s
%s
%s
%s`, wp.Name, generatedBy, exe, hostIncludes, hostRuntime, detectorHostRuntime, cardMain),
		"CMakeLists.txt": fmt.Sprintf(`# %s
cmake_minimum_required(VERSION 3.22...4.2)
project(tessera_%s LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(TT-Metalium REQUIRED)

# The card helper, which the Go program starts.
add_executable(%s_card card.cpp)
target_link_libraries(%s_card PRIVATE TT::Metalium)
# It reads the kernel sources at run time from here.
target_compile_definitions(%s_card PRIVATE TSR_KERNEL_DIR="${CMAKE_CURRENT_SOURCE_DIR}/kernels")
`, generatedBy, exe, exe, exe, exe),
		"build.sh": fmt.Sprintf(`#!/usr/bin/env bash
# %s
# Builds the card helper against a tt-metal checkout and the Go program,
# both into build/. Run the program as ./build/%s; it starts
# ./build/%s_card itself.
set -euo pipefail
: "${TT_METAL_HOME:?set TT_METAL_HOME to your tt-metal checkout}"
cd "$(dirname "$0")"
cmake -S . -B build -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CXX_COMPILER="${CXX:-clang++-20}" \
  -DCMAKE_PREFIX_PATH="$TT_METAL_HOME/build" \
  -DTT-Metalium_DIR="$TT_METAL_HOME/build/lib/cmake/tt-metalium"
cmake --build build
(cd host && go build -o ../build/%s .)
echo "built ./build/%s and ./build/%s_card"
`, generatedBy, exe, exe, exe, exe, exe),
	}
	if wp.detector != nil {
		out["kernels/detector.cpp"] = detectorKernel
	}
	manifest, _ := json.MarshalIndent(map[string]any{
		"program":       wp.Name,
		"kind":          "workers",
		"host":          "go",
		"scratch_bytes": scratch,
		"device_funcs":  len(wp.device),
		"stack_need":    wp.sh.stackNeed, // bytes the deepest chain of calls can need (an estimate)
		"stack":         map[bool]string{true: "L1", false: "local"}[wp.sh.l1Stack()],
	}, "", "  ")
	out["program.json"] = string(manifest) + "\n"

	// The program, rewritten, and what it needs to build on its own.
	src, err := rewriteHost(wp)
	if err != nil {
		return nil, err
	}
	for name, text := range src {
		out["host/"+name] = text
	}
	out["host/zz_tessera.go"] = tesseraGo(wp, exe, scratch)
	out["host/go.mod"] = fmt.Sprintf(`// %s
module tessera.local/%s

go 1.24

require github.com/pavelgladyshev/tenstorrent-go v0.0.0

replace github.com/pavelgladyshev/tenstorrent-go => ./tessera
`, generatedBy, exe)
	out["host/tessera/go.mod"] = "module github.com/pavelgladyshev/tenstorrent-go\n\ngo 1.24\n"
	ttDir, err := ttSourceDir(wp)
	if err != nil {
		return nil, err
	}
	for _, sub := range []string{"", "card"} {
		entries, err := os.ReadDir(filepath.Join(ttDir, sub))
		if err != nil {
			return nil, err
		}
		for _, e := range entries {
			n := e.Name()
			if e.IsDir() || !strings.HasSuffix(n, ".go") || strings.HasSuffix(n, "_test.go") {
				continue
			}
			b, err := os.ReadFile(filepath.Join(ttDir, sub, n))
			if err != nil {
				return nil, err
			}
			out[filepath.ToSlash(filepath.Join("host/tessera/tt", sub, n))] = string(b)
		}
	}
	return writeFiles(dir, out)
}

// ttSourceDir is where the tt package the program was checked against lives.
func ttSourceDir(wp *WorkerProgram) (string, error) {
	sel := wp.call.Fun.(*ast.SelectorExpr)
	obj := wp.sh.info.Uses[sel.Sel]
	file := wp.sh.fset.Position(obj.Pos()).Filename
	if file == "" {
		return "", fmt.Errorf("cannot find the source of package tt")
	}
	return filepath.Abs(filepath.Dir(file))
}

// rewriteHost returns the program's files with tt.Parallel(n, work) turned
// into tt.Launch(tesseraKernel, n, work, args) and main releasing the card
// when it returns. The worker function stays: without a card, tt.Launch
// runs it on the CPU.
func rewriteHost(wp *WorkerProgram) (map[string]string, error) {
	fset := wp.sh.fset
	type edit struct {
		at, del int
		text    string
	}
	out := map[string]string{}
	for _, f := range wp.files {
		name := fset.Position(f.Pos()).Filename
		b, err := os.ReadFile(name)
		if err != nil {
			return nil, err
		}
		var edits []edit
		inFile := func(n ast.Node) bool { return fset.Position(n.Pos()).Filename == name }
		if inFile(wp.call) {
			sel := wp.call.Fun.(*ast.SelectorExpr)
			tt := sel.X.(*ast.Ident).Name
			var args []string
			for _, v := range wp.captures {
				switch {
				case isTT(v.Type(), "Bytes"):
					args = append(args, fmt.Sprintf("%s.BytesArg(%s)", tt, v.Name()))
				case isTT(v.Type(), "Detector"):
					args = append(args, fmt.Sprintf("%s.DetectorArg(%s)", tt, v.Name()))
				case isInteger(v.Type()):
					args = append(args, fmt.Sprintf("%s.IntArg(int64(%s))", tt, v.Name()))
				case isString(v.Type()):
					args = append(args, fmt.Sprintf("%s.StringArg(string(%s))", tt, v.Name()))
				default:
					args = append(args, fmt.Sprintf("%s.BoolArg(%s)", tt, v.Name()))
				}
			}
			start := fset.Position(sel.Pos()).Offset
			end := fset.Position(sel.End()).Offset
			edits = append(edits,
				edit{start, end - start, tt + ".Launch"},
				edit{fset.Position(wp.call.Lparen).Offset + 1, 0, "tesseraKernel, "},
				edit{fset.Position(wp.call.Args[1].End()).Offset, 0,
					fmt.Sprintf(", []%s.CardArg{%s}", tt, strings.Join(args, ", "))})
		}
		if inFile(wp.main) {
			edits = append(edits, edit{fset.Position(wp.main.Body.Lbrace).Offset + 1, 0,
				"\n\tdefer tesseraClose() // release the card when main returns\n"})
		}
		sort.Slice(edits, func(i, j int) bool { return edits[i].at > edits[j].at })
		text := string(b)
		for _, e := range edits {
			text = text[:e.at] + e.text + text[e.at+e.del:]
		}
		if len(edits) > 0 {
			text = "// Rewritten by tessc from " + filepath.ToSlash(name) + ": tt.Parallel runs on the card.\n\n" + text
		}
		formatted, err := format.Source([]byte(text))
		if err != nil {
			return nil, fmt.Errorf("rewriting %s: %v", name, err)
		}
		out[filepath.Base(name)] = string(formatted)
	}
	return out, nil
}

// tesseraGo is the generated file that connects the rewritten program to
// the card.
func tesseraGo(wp *WorkerProgram, exe string, scratch uint32) string {
	var b strings.Builder
	fmt.Fprintf(&b, `// Code generated by tessc for %s. DO NOT EDIT.
//
// main runs on the host as ordinary Go. The card helper (build/%s_card)
// runs the worker of tt.Parallel, compiled into kernels/worker.cpp.

package main

import (
	tesseracard "github.com/pavelgladyshev/tenstorrent-go/tt/card"
	tesseratt "github.com/pavelgladyshev/tenstorrent-go/tt"
)

func init() { tesseratt.UseCard(tesseracard.New(%q)) }

func tesseraClose() { tesseratt.CloseCard() }

var tesseraKernel = tesseratt.Kernel{
	Name:     "worker.cpp",
	Scratch:  %d,
	Capacity: %d,
	Positions: []tesseratt.Position{
`, wp.Name, exe, exe, scratch, ListCapacity)
	for _, p := range wp.sh.positions {
		fmt.Fprintf(&b, "\t\t{Where: %q, What: %q},\n", p.where, p.what)
	}
	b.WriteString("\t},\n}\n")
	formatted, err := format.Source([]byte(b.String()))
	if err != nil {
		panic(err)
	}
	return string(formatted)
}
