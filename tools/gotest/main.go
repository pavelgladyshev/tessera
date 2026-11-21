// Command gotest runs Go's own test suite ($GOROOT/test) through tessc.
//
// It takes the suite's self-checking programs, the files whose first line
// is "// run": each must exit normally and print exactly what its .out file
// holds, or nothing if it has none. Each becomes a Tessera program:
//
//   - a test without goroutines becomes a worker program: its main is
//     renamed tsrTestMain and called by the one worker of a tt.Parallel,
//     and the new main prints "tsr: ok" when the worker is done;
//   - a test with goroutines is taken as it is, as a goroutine program.
//
// In a test without a .out file, any output means failure, so its calls
// of print, println and fmt's Print functions become panics: the test
// still fails exactly when it would have printed, and worker code, which
// cannot print, can run it.
//
// Every program runs with go run first (the reference: it must still
// pass), then goes through tessc, and with -card is built and run on the
// card, where its output must equal go run's. The results go to
// results.json and summary.md in the work folder.
//
//	go run ./tools/gotest -work /tmp/gotest              # go run and tessc only
//	go run ./tools/gotest -work /tmp/gotest -card        # and on the card
//
// -card needs TT_METAL_HOME; the card helper of worker programs is built
// once and shared by all of them (TESSERA_CARD, TESSERA_KERNELS).
package main

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"go/ast"
	"go/build/constraint"
	"go/parser"
	"go/token"
	"os"
	"os/exec"
	"path/filepath"
	"regexp"
	"runtime"
	"sort"
	"strconv"
	"strings"
	"sync"
	"time"
)

var (
	goroot   = flag.String("goroot", "", "Go installation whose test folder to use (default: go env GOROOT)")
	work     = flag.String("work", "", "work folder (required)")
	runRE    = flag.String("run", "", "only tests whose name (e.g. ken/chan) matches this regular expression")
	jobs     = flag.Int("j", runtime.NumCPU()/2, "tests prepared at once")
	card     = flag.Bool("card", false, "also build and run the programs on the card")
	cpuLimit = flag.Duration("cpu-timeout", 60*time.Second, "time limit of each go run")
	cardLim  = flag.Duration("card-timeout", 180*time.Second, "time limit of each card run")
	reset    = flag.String("reset", "", "command that resets the card after a hang (e.g. tt-smi -r)")
	only     = flag.String("only", "", "with -card: run only the tests listed (one per line) in this file")
)

// The folders of the suite that hold single-file tests, as Go's test
// runner (cmd/internal/testdir) lists them.
var dirs = []string{".", "ken", "chan", "interface", "fixedbugs", "typeparam", "abi", "dwarf"}

// Test is one test of the suite and what became of it.
type Test struct {
	Name    string `json:"name"`              // e.g. "ken/chan"
	Variant string `json:"variant,omitempty"` // "workers" or "goroutines"
	Golden  bool   `json:"golden,omitempty"`  // has a .out file
	Status  string `json:"status"`            // see statuses
	Detail  string `json:"detail,omitempty"`  // the reason, for anything but a pass
	Kind    string `json:"kind,omitempty"`    // what tessc made (program.json)
	CPUTime string `json:"cpu_time,omitempty"`
	Card    string `json:"card_time,omitempty"`

	src, dir, gen, exe string
	want               string // the output the test must print
	cpuOut             string // go run's output
}

// statuses, in the order of the summary.
var statuses = []string{
	"pass",          // the card printed what go run printed
	"compiled",      // tessc took it (and it was not run on the card)
	"rejected",      // tessc refused it: a feature Tessera lacks
	"tessc-crash",   // tessc itself failed
	"build-fail",    // the generated program did not build
	"card-compile",  // the card's compiler refused the generated kernel
	"card-panic",    // the program stopped with an error on the card where go run did not
	"card-timeout",  // the card run did not finish in time
	"card-mismatch", // it finished, with other output than go run's
	"cpu-fail",      // go run of the wrapped test failed: a limit of this harness
	"skip",          // not run: flags, build constraints, ...
}

func main() {
	flag.Parse()
	if *work == "" {
		fmt.Fprintln(os.Stderr, "gotest: -work is required")
		os.Exit(2)
	}
	if *goroot == "" {
		out, err := exec.Command("go", "env", "GOROOT").Output()
		if err != nil {
			fail(err)
		}
		*goroot = strings.TrimSpace(string(out))
	}
	repo, err := repoRoot()
	if err != nil {
		fail(err)
	}
	*work, _ = filepath.Abs(*work)
	must(os.MkdirAll(filepath.Join(*work, "bin"), 0o755))
	tessc := filepath.Join(*work, "bin", "tessc")
	if out, err := exec.Command("go", "build", "-C", repo, "-o", tessc, "./cmd/tessc").CombinedOutput(); err != nil {
		fail(fmt.Errorf("building tessc: %v\n%s", err, out))
	}

	tests := collect(*goroot)
	if *runRE != "" {
		re := regexp.MustCompile(*runRE)
		var keep []*Test
		for _, t := range tests {
			if re.MatchString(t.Name) {
				keep = append(keep, t)
			}
		}
		tests = keep
	}
	fmt.Fprintf(os.Stderr, "gotest: %d tests from %s/test\n", len(tests), *goroot)

	// go run and tessc, several at a time.
	var wg sync.WaitGroup
	next := make(chan *Test)
	var mu sync.Mutex
	done := 0
	for range max(*jobs, 1) {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for t := range next {
				prepare(t, repo, tessc)
				mu.Lock()
				done++
				if done%100 == 0 {
					fmt.Fprintf(os.Stderr, "gotest: prepared %d\n", done)
				}
				mu.Unlock()
			}
		}()
	}
	for _, t := range tests {
		if t.Status == "" {
			next <- t
		}
	}
	close(next)
	wg.Wait()
	save(tests)

	if *card {
		runCard(tests, repo)
		save(tests)
	}
	summarize(tests)
}

// collect lists the suite's "// run" tests.
func collect(goroot string) []*Test {
	var tests []*Test
	for _, d := range dirs {
		files, _ := filepath.Glob(filepath.Join(goroot, "test", d, "*.go"))
		sort.Strings(files)
		for _, f := range files {
			src, err := os.ReadFile(f)
			if err != nil {
				continue
			}
			first, _, _ := strings.Cut(string(src), "\n")
			first = strings.TrimSpace(first)
			if first != "// run" && !strings.HasPrefix(first, "// run ") {
				continue
			}
			name := strings.TrimSuffix(filepath.ToSlash(filepath.Join(d, filepath.Base(f))), ".go")
			t := &Test{Name: name, src: f}
			if first != "// run" {
				t.Status, t.Detail = "skip", "run with arguments: "+strings.TrimPrefix(first, "// run ")
			} else if !buildable(src) {
				t.Status, t.Detail = "skip", "build constraints exclude linux/amd64"
			}
			if b, err := os.ReadFile(strings.TrimSuffix(f, ".go") + ".out"); err == nil {
				t.Golden, t.want = true, string(b)
			}
			tests = append(tests, t)
		}
	}
	return tests
}

// buildable reports whether the file's //go:build line (if any) holds on
// linux/amd64 with the gc toolchain, where go run runs it.
func buildable(src []byte) bool {
	for _, line := range strings.Split(string(src), "\n") {
		line = strings.TrimSpace(line)
		if strings.HasPrefix(line, "package ") {
			break
		}
		if !constraint.IsGoBuild(line) {
			continue
		}
		e, err := constraint.Parse(line)
		if err != nil {
			return false
		}
		return e.Eval(func(tag string) bool {
			switch tag {
			case "linux", "amd64", "gc", "unix", "cgo":
				return true
			}
			return strings.HasPrefix(tag, "go1.")
		})
	}
	return true
}

// prepare writes the test's program, runs it with go run and compiles it
// with tessc.
func prepare(t *Test, repo, tessc string) {
	id := strings.NewReplacer("/", "_", ".", "_", "-", "_").Replace(t.Name)
	if id[0] >= '0' && id[0] <= '9' {
		id = "t" + id
	}
	t.dir = filepath.Join(*work, "src", id)
	t.gen = filepath.Join(*work, "gen", id)
	t.exe = id
	os.RemoveAll(t.dir)
	os.RemoveAll(t.gen)
	must(os.MkdirAll(t.dir, 0o755))

	src, err := os.ReadFile(t.src)
	if err != nil {
		t.Status, t.Detail = "cpu-fail", err.Error()
		return
	}
	prog, variant, err := transform(src, t.Golden)
	if err != nil {
		t.Status, t.Detail = "skip", err.Error()
		return
	}
	t.Variant = variant
	if variant == "workers" {
		t.want += "tsr: ok\n"
	}
	must(os.WriteFile(filepath.Join(t.dir, "main.go"), prog, 0o644))
	must(os.WriteFile(filepath.Join(t.dir, "go.mod"), []byte(fmt.Sprintf(
		"module gotest.local/%s\n\ngo 1.27\n\nrequire github.com/pavelgladyshev/tessera v0.0.0\n\nreplace github.com/pavelgladyshev/tessera => %q\n",
		id, repo)), 0o644))

	// The reference: go run of the program must still pass the test.
	cpuExe := filepath.Join(t.dir, "cpu")
	cmd := exec.Command("go", "build", "-o", cpuExe, ".")
	cmd.Dir = t.dir
	cmd.Env = append(os.Environ(), "GOFLAGS=-mod=mod")
	if out, err := cmd.CombinedOutput(); err != nil {
		t.Status, t.Detail = "cpu-fail", "go build: "+firstLines(string(out), 3)
		return
	}
	start := time.Now()
	out, code := run(*cpuLimit, t.dir, nil, cpuExe)
	t.CPUTime = time.Since(start).Round(time.Millisecond).String()
	t.cpuOut = out
	if code != 0 || out != t.want {
		t.Status = "cpu-fail"
		t.Detail = fmt.Sprintf("exit %d: %s", code, firstLines(out, 3))
		return
	}

	cmd = exec.Command(tessc, "-o", t.gen, ".")
	cmd.Dir = t.dir
	b, err := cmd.CombinedOutput()
	if err != nil {
		var ee *exec.ExitError
		if errors.As(err, &ee) && ee.ExitCode() == 1 && !strings.Contains(string(b), "panic:") {
			t.Status, t.Detail = "rejected", firstLines(string(b), 1)
		} else {
			t.Status, t.Detail = "tessc-crash", firstLines(string(b), 5)
		}
		return
	}
	var manifest struct {
		Kind string `json:"kind"`
		Host string `json:"host"`
	}
	if mb, err := os.ReadFile(filepath.Join(t.gen, "program.json")); err == nil {
		json.Unmarshal(mb, &manifest)
	}
	t.Kind = manifest.Kind
	if manifest.Host != "" {
		t.Kind += "/" + manifest.Host
	}
	t.Status = "compiled"
	if t.Kind == "workers/go" {
		// The Go side of a worker program; its card helper is shared.
		cmd := exec.Command("go", "build", "-o", filepath.Join(t.gen, "build", t.exe), ".")
		cmd.Dir = filepath.Join(t.gen, "host")
		if out, err := cmd.CombinedOutput(); err != nil {
			t.Status, t.Detail = "build-fail", "go build: "+firstLines(string(out), 3)
		}
	}
}

// transform turns a test into a Tessera program: see the package comment.
func transform(src []byte, golden bool) ([]byte, string, error) {
	fset := token.NewFileSet()
	f, err := parser.ParseFile(fset, "main.go", src, parser.ParseComments)
	if err != nil {
		return nil, "", err
	}
	if f.Name.Name != "main" {
		return nil, "", fmt.Errorf("package %s, not main", f.Name.Name)
	}
	var mainFn *ast.FuncDecl
	hasGo, panicDeclared := false, false
	ast.Inspect(f, func(n ast.Node) bool {
		switch n := n.(type) {
		case *ast.FuncDecl:
			if n.Recv == nil && n.Name.Name == "main" {
				mainFn = n
			}
		case *ast.GoStmt:
			hasGo = true
		case *ast.Ident:
			if n.Name == "panic" && n.Obj != nil {
				panicDeclared = true
			}
		}
		return true
	})
	if mainFn == nil {
		return nil, "", errors.New("no func main")
	}

	// The program is the test's text with a few edits, so that every line
	// stays where it was (some tests check their own line numbers).
	type edit struct {
		from, to int
		text     string
	}
	var edits []edit
	off := func(p token.Pos) int { return fset.Position(p).Offset }
	if !golden && !panicDeclared {
		replaced := unprint(f)
		for _, es := range replaced {
			// Keep the variables the printing used, or Go finds them unused.
			var uses strings.Builder
			ast.Inspect(es, func(n ast.Node) bool {
				switch n := n.(type) {
				case *ast.FuncLit:
					return false
				case *ast.Ident:
					if n.Obj != nil && n.Obj.Kind == ast.Var {
						uses.WriteString("_ = " + n.Name + "; ")
					}
				}
				return true
			})
			text := "{ " + uses.String() + `panic("tsr: the test printed output") }`
			text += strings.Repeat("\n", strings.Count(string(src[off(es.Pos()):off(es.End())]), "\n"))
			edits = append(edits, edit{off(es.Pos()), off(es.End()), text})
		}
		if len(replaced) > 0 {
			// Imports only the printing used become blank imports.
			used := map[string]bool{}
			gone := map[ast.Node]bool{}
			for _, es := range replaced {
				gone[es] = true
			}
			ast.Inspect(f, func(n ast.Node) bool {
				if gone[n] {
					return false
				}
				if sel, ok := n.(*ast.SelectorExpr); ok {
					if id, ok := sel.X.(*ast.Ident); ok && id.Obj == nil {
						used[id.Name] = true
					}
				}
				return true
			})
			for _, is := range f.Imports {
				p, _ := strconv.Unquote(is.Path.Value)
				name := pkgName(p)
				if is.Name != nil {
					name = is.Name.Name
				}
				if name != "_" && name != "." && !used[name] {
					edits = append(edits, edit{off(is.Pos()), off(is.End()), "_ " + is.Path.Value})
				}
			}
		}
	}
	variant := "goroutines"
	if !hasGo {
		variant = "workers"
		obj := mainFn.Name.Obj
		ast.Inspect(f, func(n ast.Node) bool {
			if id, ok := n.(*ast.Ident); ok && id.Name == "main" && (id.Obj == obj || id == mainFn.Name) {
				edits = append(edits, edit{off(id.Pos()), off(id.End()), "tsrTestMain"})
			}
			return true
		})
		edits = append(edits, edit{off(f.Name.End()), off(f.Name.End()),
			`; import tsrtt "github.com/pavelgladyshev/tessera/tt"; import tsros "os"`})
	}
	sort.SliceStable(edits, func(i, j int) bool { return edits[i].from > edits[j].from })
	out := append([]byte(nil), src...)
	last := len(out) + 1
	for _, e := range edits {
		if e.to > last {
			continue // inside an edit already made (a print within a print)
		}
		out = append(out[:e.from:e.from], append([]byte(e.text), out[e.to:]...)...)
		last = e.from
	}
	if variant == "workers" {
		out = append(out, `
// tsrDone is what the worker sends back when the test's main returns.
type tsrDone struct{ W int }

// main runs the test's main as the one worker of a worker program.
func main() {
	r := tsrtt.Parallel(1, func(w int, out *tsrtt.List[tsrDone]) {
		tsrTestMain()
		out.Append(tsrDone{w + 1})
	})
	if len(r) != 1 || r[0].W != 1 {
		panic("tsr: the worker did not finish")
	}
	tsros.Stdout.WriteString("tsr: ok\n")
}
`...)
	}
	return out, variant, nil
}

// unprint finds the statements that print (print, println, fmt.Print,
// fmt.Println, fmt.Printf, and fmt.Fprint* to os.Stdout or os.Stderr).
func unprint(f *ast.File) []*ast.ExprStmt {
	fmtName, osName := importName(f, "fmt"), importName(f, "os")
	printing := func(call *ast.CallExpr) bool {
		switch fn := call.Fun.(type) {
		case *ast.Ident:
			return (fn.Name == "print" || fn.Name == "println") && fn.Obj == nil
		case *ast.SelectorExpr:
			pkg, ok := fn.X.(*ast.Ident)
			if !ok || pkg.Obj != nil || fmtName == "" || pkg.Name != fmtName {
				return false
			}
			switch fn.Sel.Name {
			case "Print", "Println", "Printf":
				return true
			case "Fprint", "Fprintln", "Fprintf":
				if len(call.Args) > 0 {
					if w, ok := call.Args[0].(*ast.SelectorExpr); ok {
						if p, ok := w.X.(*ast.Ident); ok && osName != "" && p.Name == osName && (w.Sel.Name == "Stdout" || w.Sel.Name == "Stderr") {
							return true
						}
					}
				}
			}
		}
		return false
	}
	var found []*ast.ExprStmt
	ast.Inspect(f, func(n ast.Node) bool {
		if es, ok := n.(*ast.ExprStmt); ok {
			if call, ok := es.X.(*ast.CallExpr); ok && printing(call) {
				found = append(found, es)
				return false
			}
		}
		return true
	})
	return found
}

// importName is the name the file uses for the package at path ("" if it
// does not import it).
func importName(f *ast.File, path string) string {
	for _, s := range f.Imports {
		if p, _ := strconv.Unquote(s.Path.Value); p == path {
			if s.Name != nil {
				return s.Name.Name
			}
			return pkgName(p)
		}
	}
	return ""
}

// pkgName guesses a standard package's name from its path.
func pkgName(path string) string {
	parts := strings.Split(path, "/")
	n := parts[len(parts)-1]
	if len(parts) > 1 && regexp.MustCompile(`^v[0-9]+$`).MatchString(n) {
		n = parts[len(parts)-2]
	}
	return n
}

// runCard builds what needs the card and runs every compiled program on
// it, one at a time.
func runCard(tests []*Test, repo string) {
	if os.Getenv("TT_METAL_HOME") == "" {
		fail(errors.New("-card needs TT_METAL_HOME"))
	}
	if os.Getenv("TT_METAL_RUNTIME_ROOT") == "" {
		os.Setenv("TT_METAL_RUNTIME_ROOT", os.Getenv("TT_METAL_HOME"))
	}
	if os.Getenv("TT_METAL_CACHE") == "" {
		os.Setenv("TT_METAL_CACHE", filepath.Join(*work, "jit-cache"))
	}
	var list []*Test
	for _, t := range tests {
		if t.Status == "compiled" {
			list = append(list, t)
		}
	}
	if *only != "" {
		b, err := os.ReadFile(*only)
		if err != nil {
			fail(err)
		}
		want := map[string]bool{}
		for _, l := range strings.Fields(string(b)) {
			want[l] = true
		}
		keep := list[:0]
		for _, t := range list {
			if want[t.Name] {
				keep = append(keep, t)
			}
		}
		list = keep
	}

	// One card helper for every worker program: the helper is the same
	// C++ for all of them, and reads each one's kernel from TESSERA_KERNELS.
	helper := filepath.Join(*work, "bin", "tessera_card")
	for _, t := range list {
		if t.Kind != "workers/go" {
			continue
		}
		if _, err := os.Stat(helper); err == nil {
			break
		}
		fmt.Fprintf(os.Stderr, "gotest: building the card helper in %s\n", t.gen)
		if out, code := run(30*time.Minute, t.gen, nil, "./build.sh"); code != 0 {
			fail(fmt.Errorf("building the card helper: %s", firstLines(out, 20)))
		}
		b, err := os.ReadFile(filepath.Join(t.gen, "build", t.exe+"_card"))
		must(err)
		must(os.WriteFile(helper, b, 0o755))
	}

	for i, t := range list {
		if t.Kind != "workers/go" {
			// A goroutine (or C++) program: its own CMake build.
			if out, code := run(30*time.Minute, t.gen, nil, "./build.sh"); code != 0 {
				t.Status, t.Detail = "build-fail", errorLines(out, 5)
				continue
			}
		}
		env := []string{"TESSERA_CHIPS=1", "TESSERA_CARD=" + helper, "TESSERA_KERNELS=" + filepath.Join(t.gen, "kernels")}
		cardRun(t, env)
		if t.Status == "card-timeout" && *reset != "" {
			resetCard()
			cardRun(t, env)
		}
		fmt.Fprintf(os.Stderr, "gotest: [%d/%d] %s %s %s\n", i+1, len(list), t.Name, t.Status, t.Card)
		if (i+1)%20 == 0 {
			save(tests)
		}
	}
}

func cardRun(t *Test, env []string) {
	start := time.Now()
	out, code := run(*cardLim, t.gen, env, filepath.Join(t.gen, "build", t.exe))
	t.Card = time.Since(start).Round(time.Millisecond).String()
	os.WriteFile(filepath.Join(t.gen, "card.log"), []byte(out), 0o644)
	got := cardOnly(out, t.exe)
	switch {
	case code == 0 && got == cardOnly(t.cpuOut, t.exe):
		t.Status, t.Detail = "pass", ""
	case code == -1:
		t.Status, t.Detail = "card-timeout", "no result in "+cardLim.String()
	case strings.Contains(out, "Failed to generate binaries") || strings.Contains(out, "error: "):
		t.Status, t.Detail = "card-compile", errorLines(out, 5)
	case code != 0:
		t.Status, t.Detail = "card-panic", fmt.Sprintf("exit %d: %s", code, lastLines(got, 3))
	default:
		t.Status, t.Detail = "card-mismatch", "got: "+firstLines(got, 3)
	}
}

func resetCard() {
	fmt.Fprintf(os.Stderr, "gotest: resetting the card: %s\n", *reset)
	cmd := exec.Command("sh", "-c", *reset)
	cmd.Stdout, cmd.Stderr = os.Stderr, os.Stderr
	cmd.Run()
}

// cardOnly drops tt-metal's log lines and the program's own "<exe>: ..."
// lines (timings and warnings) from a card run's output, as
// scripts/card-test.sh does. It also drops the warning tt-metal prints when
// it opens the card after a program whose kernel did not compile, which
// leaves a core's run mailbox at 0x40.
var logLine = regexp.MustCompile(`\| (info|warning|debug|critical) +\||^ *(Always|Metal|Device)|^ \([A-Za-z0-9_./-]+:[0-9]+\)$|^$|^Read unexpected run_mailbox value: `)

func cardOnly(out, exe string) string {
	var b strings.Builder
	for _, l := range strings.SplitAfter(out, "\n") {
		s := strings.TrimRight(l, "\n")
		if logLine.MatchString(s) || strings.HasPrefix(s, exe+": ") || strings.HasPrefix(s, exe+"_card: ") || strings.HasPrefix(s, "tessera_card: ") {
			continue
		}
		b.WriteString(l)
	}
	return b.String()
}

// run runs a program with a time limit and returns its output (stdout and
// stderr together) and exit status, -1 if it ran out of time.
func run(limit time.Duration, dir string, env []string, name string, args ...string) (string, int) {
	ctx, cancel := context.WithTimeout(context.Background(), limit)
	defer cancel()
	cmd := exec.CommandContext(ctx, name, args...)
	cmd.Dir = dir
	cmd.Env = append(os.Environ(), env...)
	var out bytes.Buffer
	cmd.Stdout, cmd.Stderr = &out, &out
	cmd.WaitDelay = 5 * time.Second
	err := cmd.Run()
	if ctx.Err() != nil {
		return out.String(), -1
	}
	if err != nil {
		var ee *exec.ExitError
		if errors.As(err, &ee) {
			return out.String(), max(ee.ExitCode(), 1)
		}
		return out.String() + err.Error(), 1
	}
	return out.String(), 0
}

func save(tests []*Test) {
	b, _ := json.MarshalIndent(tests, "", "  ")
	must(os.WriteFile(filepath.Join(*work, "results.json"), b, 0o644))
}

// summarize writes summary.md: the counts, the reasons tessc gave, and
// every test that went wrong after tessc took it.
func summarize(tests []*Test) {
	count := map[string]int{}
	for _, t := range tests {
		count[t.Status]++
	}
	var b strings.Builder
	fmt.Fprintf(&b, "# Go's test suite through Tessera\n\n%d \"// run\" tests from %s/test, %s.\n\n| status | tests |\n|---|---:|\n",
		len(tests), *goroot, time.Now().Format("2006-01-02 15:04"))
	for _, s := range statuses {
		if count[s] > 0 {
			fmt.Fprintf(&b, "| %s | %d |\n", s, count[s])
		}
	}

	reasons := map[string][]string{}
	for _, t := range tests {
		if t.Status == "rejected" {
			r := reason(t.Detail)
			reasons[r] = append(reasons[r], t.Name)
		}
	}
	keys := make([]string, 0, len(reasons))
	for k := range reasons {
		keys = append(keys, k)
	}
	sort.Slice(keys, func(i, j int) bool {
		if len(reasons[keys[i]]) != len(reasons[keys[j]]) {
			return len(reasons[keys[i]]) > len(reasons[keys[j]])
		}
		return keys[i] < keys[j]
	})
	fmt.Fprintf(&b, "\n## Why tessc refused tests (the first reason in each)\n\n| reason | tests | e.g. |\n|---|---:|---|\n")
	for _, k := range keys {
		fmt.Fprintf(&b, "| %s | %d | %s |\n", strings.ReplaceAll(k, "|", `\|`), len(reasons[k]), strings.Join(reasons[k][:min(3, len(reasons[k]))], ", "))
	}

	for _, s := range []string{"pass", "compiled", "tessc-crash", "build-fail", "card-compile", "card-panic", "card-timeout", "card-mismatch", "cpu-fail", "skip"} {
		if count[s] == 0 {
			continue
		}
		fmt.Fprintf(&b, "\n## %s\n\n", s)
		for _, t := range tests {
			if t.Status != s {
				continue
			}
			line := "- " + t.Name
			if t.Variant != "" {
				line += " (" + t.Variant + ")"
			}
			if t.Card != "" && s == "pass" {
				line += ": card " + t.Card + ", go run " + t.CPUTime
			}
			if t.Detail != "" {
				line += ": " + strings.ReplaceAll(t.Detail, "\n", " / ")
			}
			b.WriteString(line + "\n")
		}
	}
	must(os.WriteFile(filepath.Join(*work, "summary.md"), []byte(b.String()), 0o644))
	for _, s := range statuses {
		if count[s] > 0 {
			fmt.Printf("%-14s %d\n", s, count[s])
		}
	}
	fmt.Printf("details in %s\n", filepath.Join(*work, "summary.md"))
}

// reason turns a tessc message into a category: without its position and
// with names and literals replaced.
var (
	posRE    = regexp.MustCompile(`^[^ ]*:\d+:\d+: ([A-Za-z_][A-Za-z0-9_.]*: )?`)
	quoteRE  = regexp.MustCompile("\"[^\"]*\"|`[^`]*`")
	identRE  = regexp.MustCompile(`\b(variable|function|field|method|type|package|constant|call to|of) [A-Za-z_][A-Za-z0-9_.]*`)
	numberRE = regexp.MustCompile(`\b\d+\b`)
)

func reason(msg string) string {
	msg = posRE.ReplaceAllString(strings.TrimSpace(msg), "")
	msg = quoteRE.ReplaceAllString(msg, "…")
	msg = identRE.ReplaceAllString(msg, "$1 …")
	msg = numberRE.ReplaceAllString(msg, "N")
	return msg
}

func firstLines(s string, n int) string {
	lines := strings.Split(strings.TrimSpace(s), "\n")
	return strings.Join(lines[:min(n, len(lines))], "\n")
}

func lastLines(s string, n int) string {
	lines := strings.Split(strings.TrimSpace(s), "\n")
	return strings.Join(lines[max(0, len(lines)-n):], "\n")
}

// errorLines picks a build log's lines that mention errors: the compiler's
// "file:line:col: error: ..." lines if there are any (not tt-metal's line
// with the whole compiler command), each once.
func errorLines(s string, n int) string {
	var keep []string
	seen := map[string]bool{}
	for _, want := range []string{": error: ", "error"} {
		for _, l := range strings.Split(s, "\n") {
			l = strings.TrimSpace(l)
			if strings.Contains(l, "compile failure -- cmd:") || seen[l] || len(keep) == n {
				continue
			}
			if strings.Contains(l, want) || want == "error" && strings.Contains(l, "Error") {
				keep = append(keep, l)
				seen[l] = true
			}
		}
		if len(keep) > 0 {
			break
		}
	}
	if len(keep) == 0 {
		return lastLines(s, n)
	}
	return strings.Join(keep, "\n")
}

// repoRoot is the Tessera checkout this command is in.
func repoRoot() (string, error) {
	out, err := exec.Command("go", "list", "-m", "-f", "{{.Dir}}").Output()
	if err != nil {
		return "", fmt.Errorf("run gotest from the Tessera repository: %v", err)
	}
	return strings.TrimSpace(string(out)), nil
}

func must(err error) {
	if err != nil {
		fail(err)
	}
}

func fail(err error) {
	fmt.Fprintln(os.Stderr, "gotest:", err)
	os.Exit(1)
}
