// Package compiler turns a Tessera program (a Go package using
// github.com/pavelgladyshev/tenstorrent-go/tt) into a TT-Metalium C++
// program for a Tenstorrent Wormhole card.
//
// The pipeline is:
//
//	Load   parse and type-check the Go package, then walk main and the element
//	       functions, building one hash-consed dataflow graph (frontend.go,
//	       elem.go, ir.go)
//	Lower  schedule the graph into vector-unit (SFPU) instructions on the
//	       compute core's destination registers (lower.go)
//	Emit   write the host program, the three kernels and a CMake project
//	       (emit.go)
package compiler

import (
	"fmt"
	"go/ast"
	"go/constant"
	"go/importer"
	"go/parser"
	"go/token"
	"go/types"
	"os"
	"path/filepath"
	"sort"
	"strings"
)

// TTPath is the import path of the Tessera runtime package.
const TTPath = "github.com/pavelgladyshev/tenstorrent-go/tt"

// Input is an array read from a file with tt.Load.
type Input struct {
	Name string // Go variable name
	Path string // file name given in the source
	Pos  token.Position
}

// Output is an array written to a file with tt.Store.
type Output struct {
	Name  string // Go variable name (made unique)
	Path  string
	Value *Node
	Pos   token.Position
}

// Program is a checked Tessera program.
type Program struct {
	Name       string // the package directory's base name
	Rows, Cols int
	Inputs     []*Input
	Outputs    []*Output

	// Workers is set for a worker program (main calls tt.Parallel); the
	// fields above are then unused.
	Workers *WorkerProgram
}

// Elems returns the number of elements in each array.
func (p *Program) Elems() int { return p.Rows * p.Cols }

// maxElems keeps every buffer size within uint32 on the card (bf16, 2 bytes).
const maxElems = 1 << 29

// reservedFlags are the host program's own command-line flags.
var reservedFlags = map[string]bool{"device": true, "repeat": true, "kernels": true, "help": true, "compile-only": true}

// Options change how a program is compiled.
type Options struct {
	// Host says how main runs on the host. With HostGo main stays Go: tessc
	// rewrites the program's tt.Parallel into tt.Launch and writes a Go host
	// program plus a card helper, so main can use all of Go. With HostCPP
	// main is translated to C++. HostAuto, the default, keeps main as Go for
	// worker programs (tt.Parallel), the only kind that can do so yet.
	Host string
}

// The values of Options.Host.
const (
	HostAuto = ""
	HostGo   = "go"
	HostCPP  = "cpp"
)

// Load parses, type-checks and translates the Tessera program in dir.
func Load(dir string) (*Program, error) { return LoadWith(dir, Options{}) }

// LoadWith is Load with options.
func LoadWith(dir string, opts Options) (prog *Program, err error) {
	abs, err := filepath.Abs(dir)
	if err != nil {
		return nil, err
	}
	fset := token.NewFileSet()
	entries, err := os.ReadDir(abs)
	if err != nil {
		return nil, err
	}
	var files []*ast.File
	for _, e := range entries {
		name := e.Name()
		if e.IsDir() || !strings.HasSuffix(name, ".go") || strings.HasSuffix(name, "_test.go") {
			continue
		}
		// Parse with the path as the user gave it, so messages say
		// examples/x/main.go:3:1 rather than an absolute path.
		f, err := parser.ParseFile(fset, filepath.Join(dir, name), nil, parser.SkipObjectResolution)
		if err != nil {
			return nil, err
		}
		files = append(files, f)
	}
	if len(files) == 0 {
		return nil, fmt.Errorf("%s: no Go files", dir)
	}

	info := &types.Info{
		Types:      map[ast.Expr]types.TypeAndValue{},
		Defs:       map[*ast.Ident]types.Object{},
		Uses:       map[*ast.Ident]types.Object{},
		Selections: map[*ast.SelectorExpr]*types.Selection{},
	}
	var errs ErrorList
	conf := types.Config{
		Importer: importer.ForCompiler(fset, "source", nil),
		Error: func(err error) {
			if te, ok := err.(types.Error); ok {
				errs = append(errs, &Error{Pos: fset.Position(te.Pos), Msg: te.Msg})
			} else {
				errs = append(errs, &Error{Msg: err.Error()})
			}
		},
	}
	pkg, _ := conf.Check("main", fset, files, info)
	if len(errs) > 0 {
		return nil, errs
	}

	c := &checker{
		fset:   fset,
		info:   info,
		pkg:    pkg,
		g:      newGraph(),
		funcs:  map[*types.Func]*ast.FuncDecl{},
		arrays: map[types.Object]*Node{},
		prog:   &Program{Name: filepath.Base(abs)},
		host:   opts.Host,
		files:  files,
	}
	// Errors deep inside element translation are raised with panic(*Error)
	// and caught here, so the translator does not thread errors everywhere.
	defer func() {
		if r := recover(); r != nil {
			e, ok := r.(*Error)
			if !ok {
				panic(r)
			}
			prog, err = nil, ErrorList{e}
		}
	}()
	c.run(files)
	if len(c.errs) > 0 {
		return nil, c.errs
	}
	return c.prog, nil
}

type checker struct {
	fset   *token.FileSet
	info   *types.Info
	pkg    *types.Package
	g      *graph
	funcs  map[*types.Func]*ast.FuncDecl
	arrays map[types.Object]*Node
	prog   *Program
	errs   ErrorList
	shape  token.Position // where the program's array shape was set
	host   string         // Options.Host
	goHost bool           // main stays Go
	files  []*ast.File
}

func (c *checker) pos(n ast.Node) token.Position { return c.fset.Position(n.Pos()) }

func (c *checker) errorf(n ast.Node, format string, args ...any) {
	c.errs = append(c.errs, &Error{Pos: c.pos(n), Msg: fmt.Sprintf(format, args...)})
}

func (c *checker) fail(n ast.Node, format string, args ...any) {
	panic(&Error{Pos: c.pos(n), Msg: fmt.Sprintf(format, args...)})
}

func (c *checker) run(files []*ast.File) {
	var mainDecl *ast.FuncDecl
	for _, f := range files {
		for _, d := range f.Decls {
			if fd, ok := d.(*ast.FuncDecl); ok && fd.Recv == nil && fd.Name.Name == "main" {
				mainDecl = fd
			}
		}
	}
	if mainDecl == nil {
		c.errs = append(c.errs, &Error{Msg: "no func main"})
		return
	}
	// A main that calls tt.Parallel makes a worker program (v0.2); otherwise
	// it is an elementwise program (v0).
	goroutines := c.mainStartsGoroutines(mainDecl)
	workers := c.mainCallsParallel(mainDecl) || goroutines
	switch c.host {
	case HostGo:
		if goroutines || !workers {
			c.errorf(mainDecl.Name, "so far only worker programs (tt.Parallel) can keep main as Go; compile this one without -host=go")
			return
		}
		c.goHost = true
	case HostAuto:
		c.goHost = workers && !goroutines
	}
	for _, f := range files {
		for _, imp := range f.Imports {
			path := strings.Trim(imp.Path.Value, `"`)
			switch {
			case c.goHost:
				// main is Go: it may import anything. Worker code that
				// calls another package is rejected when it is translated.
			case path == TTPath || path == "math":
			case path == "strconv" && workers:
			case workers:
				c.errorf(imp, "import %q: a worker program may import only %q, \"strconv\" and \"math\"", path, TTPath)
			default:
				c.errorf(imp, "import %q: a Tessera program may import only %q (and \"math\" for its constants)", path, TTPath)
			}
		}
		for _, d := range f.Decls {
			switch d := d.(type) {
			case *ast.FuncDecl:
				if d.Recv != nil {
					c.errorf(d, "methods are not supported yet")
					continue
				}
				fn := c.info.Defs[d.Name].(*types.Func)
				c.funcs[fn] = d
			case *ast.GenDecl:
				switch d.Tok {
				case token.VAR:
					c.errorf(d, "package-level variables are not supported; use const, or a local variable in main")
				case token.TYPE:
					if !workers {
						c.errorf(d, "type declarations are not supported in elementwise programs")
					}
				}
			}
		}
	}
	if len(c.errs) > 0 {
		return
	}
	if goroutines {
		c.goroutines(mainDecl)
		return
	}
	if workers {
		c.workers(mainDecl)
		return
	}
	for _, st := range mainDecl.Body.List {
		c.mainStmt(st)
	}
	if len(c.errs) > 0 {
		return
	}
	if len(c.prog.Outputs) == 0 {
		c.errorf(mainDecl.Name, "main stores nothing: add tt.Store(\"out.f32\", array) so the card has something to compute")
		return
	}
	c.dropUnusedInputs(mainDecl)
}

// mainStmt handles one statement of main.
func (c *checker) mainStmt(st ast.Stmt) {
	switch s := st.(type) {
	case *ast.AssignStmt:
		if len(s.Lhs) != 1 || len(s.Rhs) != 1 || (s.Tok != token.DEFINE && s.Tok != token.ASSIGN) {
			c.errorf(s, "in main, assign one array at a time: x := tt.Load(...) or y := tt.Map(...)")
			return
		}
		id, ok := s.Lhs[0].(*ast.Ident)
		if !ok || id.Name == "_" {
			c.errorf(s.Lhs[0], "assign the array to a named variable")
			return
		}
		call, name := c.ttCall(s.Rhs[0])
		var node *Node
		switch name {
		case "Load":
			if s.Tok != token.DEFINE {
				c.errorf(s, "declare loaded arrays with :=")
				return
			}
			node = c.load(call, id)
		case "Map", "Map2", "Map3":
			node = c.mapCall(call)
		default:
			c.errorf(s.Rhs[0], "in main, arrays come from tt.Load or tt.Map/Map2/Map3")
			return
		}
		if node != nil {
			c.arrays[c.info.ObjectOf(id)] = node
		}
	case *ast.ExprStmt:
		call, name := c.ttCall(s.X)
		if name != "Store" {
			c.errorf(s, "in main, the only call statement allowed is tt.Store(path, array)")
			return
		}
		c.store(call)
	case *ast.DeclStmt:
		if gd, ok := s.Decl.(*ast.GenDecl); ok && gd.Tok == token.CONST {
			return // constants are folded by the type checker
		}
		c.errorf(s, "in main, declare arrays with := (var declarations are not supported)")
	case *ast.GoStmt:
		c.errorf(s, "goroutines are planned for v0.2; v0 runs every Map on all cores automatically")
	default:
		c.errorf(s, "v0 main may contain only tt.Load, tt.Map/Map2/Map3, tt.Store and const declarations")
	}
}

// ttCall reports whether e is a call to a function in package tt, and which.
func (c *checker) ttCall(e ast.Expr) (*ast.CallExpr, string) {
	call, ok := e.(*ast.CallExpr)
	if !ok {
		return nil, ""
	}
	fn := c.ttFunc(call.Fun)
	if fn == nil {
		return call, ""
	}
	return call, fn.Name()
}

// ttFunc returns the tt function that e names, or nil.
func (c *checker) ttFunc(e ast.Expr) *types.Func {
	sel, ok := e.(*ast.SelectorExpr)
	if !ok {
		return nil
	}
	fn, ok := c.info.Uses[sel.Sel].(*types.Func)
	if !ok || fn.Pkg() == nil || fn.Pkg().Path() != TTPath {
		return nil
	}
	return fn
}

func (c *checker) constString(e ast.Expr, what string) (string, bool) {
	tv := c.info.Types[e]
	if tv.Value == nil || tv.Value.Kind() != constant.String {
		c.errorf(e, "%s must be a constant string", what)
		return "", false
	}
	return constant.StringVal(tv.Value), true
}

func (c *checker) constInt(e ast.Expr, what string) (int, bool) {
	tv := c.info.Types[e]
	if tv.Value == nil {
		c.errorf(e, "%s must be a constant (the compiler sizes buffers from it)", what)
		return 0, false
	}
	v, ok := constant.Int64Val(constant.ToInt(tv.Value))
	if !ok || v <= 0 || v > maxElems {
		c.errorf(e, "%s must be a positive integer constant", what)
		return 0, false
	}
	return int(v), true
}

func (c *checker) load(call *ast.CallExpr, id *ast.Ident) *Node {
	path, ok1 := c.constString(call.Args[0], "the file name")
	rows, ok2 := c.constInt(call.Args[1], "rows")
	cols, ok3 := c.constInt(call.Args[2], "cols")
	if !ok1 || !ok2 || !ok3 {
		return nil
	}
	p := c.prog
	switch {
	case p.Rows == 0:
		if rows*cols > maxElems {
			c.errorf(call, "%dx%d is too large for v0 (at most %d elements per array)", rows, cols, maxElems)
			return nil
		}
		p.Rows, p.Cols, c.shape = rows, cols, c.pos(call)
	case p.Rows != rows || p.Cols != cols:
		c.errorf(call, "array is %dx%d but %s loaded a %dx%d array; in v0 all arrays must have the same shape", rows, cols, c.shape, p.Rows, p.Cols)
		return nil
	}
	if reservedFlags[id.Name] {
		c.errorf(id, "rename %q: the compiled program uses --%s as its own flag", id.Name, id.Name)
		return nil
	}
	p.Inputs = append(p.Inputs, &Input{Name: id.Name, Path: path, Pos: c.pos(call)})
	return c.g.input(len(p.Inputs)-1, id.Name, c.pos(call))
}

// arrayArg translates an array-valued argument: an array variable or a
// nested tt.Map/Map2/Map3 call.
func (c *checker) arrayArg(e ast.Expr) *Node {
	switch x := ast.Unparen(e).(type) {
	case *ast.Ident:
		if n, ok := c.arrays[c.info.ObjectOf(x)]; ok {
			return n
		}
	case *ast.CallExpr:
		if call, name := c.ttCall(x); name == "Map" || name == "Map2" || name == "Map3" {
			return c.mapCall(call)
		}
	}
	c.errorf(e, "pass an array here: a variable assigned from tt.Load or tt.Map, or a tt.Map call")
	return nil
}

func (c *checker) mapCall(call *ast.CallExpr) *Node {
	args := make([]*Node, 0, len(call.Args)-1)
	for _, a := range call.Args[1:] {
		n := c.arrayArg(a)
		if n == nil {
			return nil
		}
		args = append(args, n)
	}
	return c.apply(call.Args[0], args, call)
}

func (c *checker) store(call *ast.CallExpr) {
	path, ok := c.constString(call.Args[0], "the file name")
	value := c.arrayArg(call.Args[1])
	if !ok || value == nil {
		return
	}
	name := "out"
	if id, ok := ast.Unparen(call.Args[1]).(*ast.Ident); ok {
		name = id.Name
	} else if base := strings.TrimSuffix(filepath.Base(path), filepath.Ext(path)); isGoIdent(base) {
		name = base // tt.Store("y.f32", tt.Map(...)) is called y
	}
	for _, o := range c.prog.Outputs {
		if o.Path == path {
			c.errorf(call, "%q is already written by the tt.Store at %s", path, o.Pos)
			return
		}
	}
	unique := name
	for i := 2; c.hasOutput(unique) || c.hasInput(unique); i++ {
		unique = fmt.Sprintf("%s%d", name, i)
	}
	if reservedFlags[unique] {
		c.errorf(call.Args[1], "rename %q: the compiled program uses --%s as its own flag", name, name)
		return
	}
	c.prog.Outputs = append(c.prog.Outputs, &Output{Name: unique, Path: path, Value: value, Pos: c.pos(call)})
}

func isGoIdent(s string) bool {
	return s != "" && token.IsIdentifier(s)
}

func (c *checker) hasOutput(name string) bool {
	for _, o := range c.prog.Outputs {
		if o.Name == name {
			return true
		}
	}
	return false
}

func (c *checker) hasInput(name string) bool {
	for _, in := range c.prog.Inputs {
		if in.Name == name {
			return true
		}
	}
	return false
}

// dropUnusedInputs removes loaded arrays that no output depends on, so the
// card does not read them, and renumbers the rest.
func (c *checker) dropUnusedInputs(mainDecl *ast.FuncDecl) {
	used := map[int]bool{}
	seen := map[*Node]bool{}
	var walk func(n *Node)
	walk = func(n *Node) {
		if seen[n] {
			return
		}
		seen[n] = true
		if n.Op == OpInput {
			used[n.Input] = true
		}
		for _, a := range n.Args {
			walk(a)
		}
	}
	for _, o := range c.prog.Outputs {
		walk(o.Value)
	}
	if len(used) == 0 {
		c.errorf(mainDecl.Name, "no stored array depends on a loaded array; v0 needs at least one tt.Load feeding a tt.Store")
		return
	}
	idx := make([]int, 0, len(used))
	for i := range used {
		idx = append(idx, i)
	}
	sort.Ints(idx)
	remap := map[int]int{}
	var kept []*Input
	for newIdx, old := range idx {
		remap[old] = newIdx
		kept = append(kept, c.prog.Inputs[old])
	}
	for n := range seen {
		if n.Op == OpInput {
			n.Input = remap[n.Input]
		}
	}
	c.prog.Inputs = kept
}
