package compiler

import (
	"fmt"
	"go/ast"
	"go/token"
	"go/types"
	"path/filepath"
	"sort"
	"strings"
)

// Package-level variables in worker and goroutine programs.
//
// The cores share no memory that they could keep variables in cheaply, so
// each core gets its own copy of the package-level variables its code uses,
// in L1. When its kernel starts, it sets them up as Go's package
// initialization would: the declared values, in Go's order, then the init
// functions that set any of them. The host has its own copy: Go's, when
// main stays Go. That gives the same results as Go's single copy as long as
// no copy is changed where the user of another copy would see the change,
// which tessc checks:
//
//   - main, and what it calls, must not change a variable that the card's
//     code uses, since the card would not see the change;
//   - the card's code must not change a variable that main uses, since main
//     would not see it;
//   - in a program that starts goroutines, a variable that a goroutine
//     changes must not be used by another goroutine, nor by the same
//     function started more than once.
//
// The workers of tt.Parallel each start with a fresh copy. Under go run
// they share one, but they run at the same time, so workers that change a
// variable another worker uses are racing anyway.

// globalRef is how code uses a package-level variable.
type globalRef struct {
	use   ast.Node // the first use
	write ast.Node // the first use that may change it, or nil
}

type globalRefs map[*types.Var]*globalRef

func (r globalRefs) add(v *types.Var, n ast.Node, write bool) {
	g := r[v]
	if g == nil {
		g = &globalRef{use: n}
		r[v] = g
	}
	if write && g.write == nil {
		g.write = n
	}
}

func (r globalRefs) merge(o globalRefs) {
	for v, g := range o {
		r.add(v, g.use, false)
		if g.write != nil {
			r.add(v, g.write, true)
		}
	}
}

// sorted returns the variables in the order they are declared.
func (r globalRefs) sorted() []*types.Var {
	vs := make([]*types.Var, 0, len(r))
	for v := range r {
		vs = append(vs, v)
	}
	sort.Slice(vs, func(i, j int) bool { return vs[i].Pos() < vs[j].Pos() })
	return vs
}

// pkgVar returns the package-level variable that id uses, or nil.
func pkgVar(info *types.Info, pkg *types.Package, id *ast.Ident) *types.Var {
	v, ok := info.Uses[id].(*types.Var)
	if !ok || v.IsField() || v.Pkg() != pkg || v.Parent() != pkg.Scope() {
		return nil
	}
	return v
}

// isPkgVar reports whether obj is a package-level variable.
func isPkgVar(obj types.Object) bool {
	v, ok := obj.(*types.Var)
	return ok && !v.IsField() && v.Pkg() != nil && v.Parent() == v.Pkg().Scope()
}

// mutRef reports whether a copy of a value of t can change what the
// original refers to: whether it holds a slice, a map, a pointer, ... (a
// string cannot be changed).
func mutRef(t types.Type) bool {
	switch u := types.Unalias(t).Underlying().(type) {
	case *types.Slice, *types.Map, *types.Pointer, *types.Chan, *types.Signature, *types.Interface:
		return true
	case *types.Array:
		return mutRef(u.Elem())
	case *types.Struct:
		for i := 0; i < u.NumFields(); i++ {
			if mutRef(u.Field(i).Type()) {
				return true
			}
		}
	}
	return false
}

// globalRefsIn adds the package-level variables that code uses to out, and
// whether it may change them: assign to them, change them through a method
// or a builtin, or copy out a slice or a map through which it could change
// what they hold. Nodes for which skip is true are left out.
func globalRefsIn(info *types.Info, pkg *types.Package, root ast.Node, skip func(ast.Node) bool, out globalRefs) {
	var stack []ast.Node
	ast.Inspect(root, func(n ast.Node) bool {
		if n == nil {
			stack = stack[:len(stack)-1]
			return true
		}
		if skip != nil && skip(n) {
			return false
		}
		if id, ok := n.(*ast.Ident); ok {
			if v := pkgVar(info, pkg, id); v != nil {
				out.add(v, id, globalWrite(info, id, stack))
			}
		}
		stack = append(stack, n)
		return true
	})
}

// globalWrite reports whether the use id of a package-level variable, with
// the nodes around it in stack, may change the variable.
func globalWrite(info *types.Info, id *ast.Ident, stack []ast.Node) bool {
	var child ast.Expr = id
	i := len(stack) - 1
	// Up the selectors and indexes that start at the variable: v.f[i].g.
climb:
	for ; i >= 0; i-- {
		switch p := stack[i].(type) {
		case *ast.ParenExpr:
			child = p
		case *ast.StarExpr:
			child = p
		case *ast.SelectorExpr:
			if s, ok := info.Selections[p]; ok && s.Kind() != types.FieldVal {
				// A method of the variable: through a pointer receiver it can
				// change it.
				recv := s.Obj().Type().(*types.Signature).Recv().Type()
				if _, ptr := types.Unalias(recv).(*types.Pointer); ptr {
					return true
				}
				return mutRef(recv)
			}
			child = p
		case *ast.IndexExpr:
			if p.X != child {
				break climb // the variable is the index
			}
			child = p
		default:
			break climb
		}
	}
	t := info.TypeOf(child)
	if i < 0 {
		return mutRef(t)
	}
	switch p := stack[i].(type) {
	case *ast.AssignStmt:
		for _, l := range p.Lhs {
			if l == child {
				return true
			}
		}
	case *ast.IncDecStmt:
		return true
	case *ast.RangeStmt:
		if p.Key == child || p.Value == child {
			return true
		}
		if p.X == child { // the elements are copied out
			switch u := types.Unalias(t).Underlying().(type) {
			case *types.Slice:
				return mutRef(u.Elem())
			case *types.Array:
				return mutRef(u.Elem())
			case *types.Map:
				return mutRef(u.Key()) || mutRef(u.Elem())
			case *types.Basic:
				return false
			}
			return true
		}
		return false
	case *ast.UnaryExpr:
		return p.Op == token.AND || p.Op == token.ARROW
	case *ast.SliceExpr:
		return p.X == child && !isString(t)
	case *ast.CallExpr:
		if p.Fun == child {
			return false // a function value, called
		}
		if tv := info.Types[p.Fun]; tv.IsType() {
			return mutRef(info.TypeOf(p)) // a conversion
		}
		if fid, ok := ast.Unparen(p.Fun).(*ast.Ident); ok {
			if b, ok := info.Uses[fid].(*types.Builtin); ok {
				switch b.Name() {
				case "len", "cap", "min", "max", "real", "imag", "complex", "print", "println", "panic":
					return false
				case "append":
					if p.Args[0] == child {
						return true
					}
					if p.Ellipsis.IsValid() && p.Args[len(p.Args)-1] == child {
						if elem, ok := sliceElem(t); ok {
							return mutRef(elem)
						}
						return false // append(b, s...) with a string
					}
					return mutRef(t)
				case "copy":
					return p.Args[0] == child
				case "delete", "clear", "close":
					return true
				}
			}
		}
		return mutRef(t)
	case *ast.BinaryExpr, *ast.IfStmt, *ast.SwitchStmt, *ast.CaseClause, *ast.IndexExpr, *ast.ExprStmt, *ast.ForStmt:
		return false
	}
	return mutRef(t)
}

// funcsFrom returns the functions of the program that code reached from
// roots can call or refer to. With methods it also counts every method of
// the program's types that the code has values of, which the standard
// library can call through interfaces (fmt calls String, say).
func (c *checker) funcsFrom(roots []ast.Node, skip func(ast.Node) bool, methods bool) map[*types.Func]bool {
	out := map[*types.Func]bool{}
	seenType := map[types.Type]bool{}
	var work []ast.Node
	addFunc := func(f *types.Func) {
		f = f.Origin()
		if out[f] || c.funcs[f] == nil || c.funcs[f].Body == nil {
			return
		}
		out[f] = true
		work = append(work, c.funcs[f].Body)
	}
	var addType func(t types.Type)
	addType = func(t types.Type) {
		if t == nil || seenType[t] {
			return
		}
		seenType[t] = true
		switch u := types.Unalias(t).(type) {
		case *types.Named:
			if u.Obj().Pkg() == c.pkg {
				for i := 0; i < u.NumMethods(); i++ {
					addFunc(u.Method(i))
				}
			}
			addType(u.Underlying())
		case *types.Pointer:
			addType(u.Elem())
		case *types.Slice:
			addType(u.Elem())
		case *types.Array:
			addType(u.Elem())
		case *types.Map:
			addType(u.Key())
			addType(u.Elem())
		case *types.Chan:
			addType(u.Elem())
		case *types.Struct:
			for i := 0; i < u.NumFields(); i++ {
				addType(u.Field(i).Type())
			}
		}
	}
	work = append(work, roots...)
	for len(work) > 0 {
		n := work[len(work)-1]
		work = work[:len(work)-1]
		ast.Inspect(n, func(n ast.Node) bool {
			if skip != nil && skip(n) {
				return false
			}
			if x, ok := n.(ast.Expr); ok && methods {
				addType(c.info.TypeOf(x))
			}
			if x, ok := n.(*ast.Ident); ok {
				if f, ok := c.info.Uses[x].(*types.Func); ok && f.Pkg() == c.pkg {
					addFunc(f)
				}
			}
			return true
		})
	}
	return out
}

// refsOf collects the package-level variables used by the roots and the
// functions they reach.
func (c *checker) refsOf(roots []ast.Node, skip func(ast.Node) bool, methods bool) (globalRefs, map[*types.Func]bool) {
	refs := globalRefs{}
	funcs := c.funcsFrom(roots, skip, methods)
	for _, r := range roots {
		globalRefsIn(c.info, c.pkg, r, skip, refs)
	}
	for f := range funcs {
		globalRefsIn(c.info, c.pkg, c.funcs[f].Body, nil, refs)
	}
	return refs, funcs
}

// cardCode is code that runs on the card: the workers, or one goroutine
// function and what it calls.
type cardCode struct {
	name  string // "the worker code", "goroutine f"
	roots []ast.Node
	many  bool // a goroutine function started more than once
}

// isInitFunc reports whether f is one of the package's init functions.
func isInitFunc(f *types.Func) bool {
	return f.Name() == "init" && f.Type().(*types.Signature).Recv() == nil && f.Pkg() != nil && f.Pkg().Scope().Lookup("init") != f
}

// planGlobals decides which package-level variables the card needs, and
// which of the package's initializers and init functions it runs to set
// them up (recorded in wp and its cgShared), and reports the uses that
// would make the card's copies differ from Go's single variable. host is
// main's code; hostFuncs says whether main is Go, so that it can call
// anything, or C++, which has no package-level variables. It returns the
// functions the card's initialization calls, for reachable.
func (c *checker) planGlobals(wp *WorkerProgram, card []cardCode, host []ast.Node, skipHost func(ast.Node) bool, goHost bool, cppWhy string) []*types.Func {
	sh := wp.sh
	// What the card's code (not its initialization) uses.
	cardRefs := globalRefs{}
	perCode := make([]globalRefs, len(card))
	for k, cc := range card {
		perCode[k], _ = c.refsOf(cc.roots, nil, false)
		cardRefs.merge(perCode[k])
	}
	// What main uses.
	var hostRefs globalRefs
	if goHost {
		hostRefs, _ = c.refsOf(host, skipHost, true)
	} else {
		hostRefs = globalRefs{}
		for _, r := range host {
			globalRefsIn(c.info, c.pkg, r, skipHost, hostRefs)
		}
		for _, v := range hostRefs.sorted() {
			c.errorf(hostRefs[v].use, "main uses package-level variable %s: %s, which has no package-level variables (yet); use a constant, or a variable of main", v.Name(), cppWhy)
		}
		hostRefs = globalRefs{} // reported
	}

	// The package's initialization, in Go's order: the initializers, then
	// the init functions.
	type initStep struct {
		init  *types.Initializer // or
		fn    *types.Func
		refs  globalRefs
		funcs map[*types.Func]bool
	}
	var steps []*initStep
	for _, in := range c.info.InitOrder {
		refs, funcs := c.refsOf([]ast.Node{in.Rhs}, nil, false)
		steps = append(steps, &initStep{init: in, refs: refs, funcs: funcs})
	}
	var inits []*types.Func
	for f := range c.funcs {
		if isInitFunc(f) {
			inits = append(inits, f)
		}
	}
	sort.Slice(inits, func(i, j int) bool { return inits[i].Pos() < inits[j].Pos() })
	for _, f := range inits {
		refs, funcs := c.refsOf([]ast.Node{c.funcs[f].Body}, nil, false)
		funcs[f] = true
		steps = append(steps, &initStep{fn: f, refs: refs, funcs: funcs})
	}
	// The card needs the variables its code uses, the steps that set them
	// (or change them on the side), and what those steps use in turn.
	need := globalRefs{}
	need.merge(cardRefs)
	run := make([]bool, len(steps))
	for changed := true; changed; {
		changed = false
		for k, st := range steps {
			if run[k] {
				continue
			}
			needed := false
			if st.init != nil {
				for _, v := range st.init.Lhs {
					if need[v] != nil {
						needed = true
					}
				}
			}
			for v, r := range st.refs {
				if r.write != nil && need[v] != nil {
					needed = true
				}
			}
			if !needed {
				continue
			}
			run[k], changed = true, true
			need.merge(st.refs)
			if st.init != nil {
				for _, v := range st.init.Lhs {
					if v.Name() != "_" {
						need.add(v, st.init.Rhs, true)
					}
				}
			}
		}
	}

	// The uses that would tell the copies apart.
	for _, v := range cardRefs.sorted() {
		cr, hr := cardRefs[v], hostRefs[v]
		switch {
		case hr != nil && hr.write != nil:
			c.errorf(hr.write, "%s is changed here, on the host, and used by the card (%s): each core has its own copy of a package-level variable, set up from its declaration, and would not see the change; give the value to the workers in a variable of main instead",
				v.Name(), c.pos(cr.use))
		case hr != nil && cr.write != nil:
			c.errorf(hr.use, "%s is changed on the card (%s) and used here, on the host: each core changes its own copy of a package-level variable, so main would not see the change; send what main needs back in the results",
				v.Name(), c.pos(cr.write))
		}
		if cr.write == nil {
			continue
		}
		var users []int
		writer := -1
		for k := range card {
			if r := perCode[k][v]; r != nil {
				users = append(users, k)
				if r.write != nil && writer < 0 {
					writer = k
				}
			}
		}
		w := card[writer]
		switch {
		case w.many:
			c.errorf(perCode[writer][v].write, "%s is changed by %s, which runs as more than one goroutine: each core has its own copy of a package-level variable, so they would not see each other's changes; send the value over a channel, or keep it in a local variable",
				v.Name(), w.name)
		case len(users) > 1:
			other := users[0]
			if other == writer {
				other = users[1]
			}
			c.errorf(perCode[writer][v].write, "%s is changed by %s and used by %s (%s): each core has its own copy of a package-level variable, so they would not see each other's changes; send the value over a channel",
				v.Name(), w.name, card[other].name, c.pos(perCode[other][v].use))
		}
	}

	// What the card keeps: the variables, set up by the steps it runs.
	var extra []*types.Func
	for _, v := range need.sorted() {
		if why := globalTypeProblem(v.Type()); why != "" {
			c.errorf(declIdent(c.info, v), "%s: %s", v.Name(), why)
		}
		sh.globals = append(sh.globals, v)
	}
	for k, st := range steps {
		if !run[k] {
			continue
		}
		if st.init != nil {
			wp.inits = append(wp.inits, st.init)
			extra = append(extra, callGraph(c.info, c.pkg, st.init.Rhs)...)
		} else {
			wp.initFuncs = append(wp.initFuncs, st.fn)
			extra = append(extra, st.fn)
		}
	}
	sh.globalName = map[*types.Var]string{}
	taken := map[string]int{}
	for _, v := range sh.globals {
		base := cident(v.Name())
		n := base
		if k := taken[base]; k > 0 {
			n = fmt.Sprintf("%s_%d", base, k+1)
		}
		taken[base]++
		sh.globalName[v] = "pkgvar::" + n + "()"
	}
	return extra
}

// globalTypeProblem says why the card cannot keep its own copy of a
// variable of type t, or "".
func globalTypeProblem(t types.Type) string {
	found := ""
	var walk func(t types.Type, seen map[types.Type]bool)
	walk = func(t types.Type, seen map[types.Type]bool) {
		if found != "" || seen[t] {
			return
		}
		seen[t] = true
		switch u := types.Unalias(t).(type) {
		case *types.Named:
			if obj := u.Obj(); obj.Pkg() != nil && obj.Pkg().Path() == TTPath {
				found = fmt.Sprintf("a package-level variable cannot hold a tt.%s; make it in main", obj.Name())
				return
			}
			walk(u.Underlying(), seen)
		case *types.Chan:
			found = "package-level channels are not supported: each core would make its own; make the channels in main and pass them to the goroutines"
		case *types.Slice:
			walk(u.Elem(), seen)
		case *types.Array:
			walk(u.Elem(), seen)
		case *types.Map:
			walk(u.Key(), seen)
			walk(u.Elem(), seen)
		case *types.Struct:
			for i := 0; i < u.NumFields(); i++ {
				walk(u.Field(i).Type(), seen)
			}
		}
	}
	walk(t, map[types.Type]bool{})
	return found
}

// declIdent returns the identifier that declares v.
func declIdent(info *types.Info, v *types.Var) *ast.Ident {
	for id, obj := range info.Defs {
		if obj == v {
			return id
		}
	}
	return &ast.Ident{NamePos: v.Pos(), Name: v.Name()}
}

// goInstances counts how many goroutines each function main starts with go
// can run as: more than one if two go statements start it, or one inside a
// loop (or in a main with labels, which goto can make into loops).
func goInstances(main *ast.FuncDecl, info *types.Info) map[*types.Func]int {
	labels := false
	ast.Inspect(main.Body, func(n ast.Node) bool {
		if _, ok := n.(*ast.LabeledStmt); ok {
			labels = true
		}
		return !labels
	})
	count := map[*types.Func]int{}
	var stack []ast.Node
	ast.Inspect(main.Body, func(n ast.Node) bool {
		if n == nil {
			stack = stack[:len(stack)-1]
			return true
		}
		if s, ok := n.(*ast.GoStmt); ok {
			if id, ok := ast.Unparen(s.Call.Fun).(*ast.Ident); ok {
				if f, ok := info.Uses[id].(*types.Func); ok {
					k := 1
					for _, p := range stack {
						switch p.(type) {
						case *ast.ForStmt, *ast.RangeStmt:
							k = 2
						}
					}
					if labels {
						k = 2
					}
					count[f] += k
				}
			}
		}
		stack = append(stack, n)
		return true
	})
	return count
}

// computeStoresGlobal finds the functions that can store a string, a slice
// or a map into a package-level variable, themselves or through calls:
// what they allocate outlives them.
//
// A call through a function value counts too: the closure can store into
// its variables, which can outlive the call.
func computeStoresGlobal(sh *cgShared, fns []*types.Func) {
	sh.storesGlobal = map[*types.Func]bool{}
	if len(sh.globals) == 0 && !sh.usesFuncs {
		return
	}
	for _, f := range fns {
		if callsValue(sh, sh.funcs[f].Body) {
			sh.storesGlobal[f] = true
			continue
		}
		refs := globalRefs{}
		globalRefsIn(sh.info, sh.pkg, sh.funcs[f].Body, nil, refs)
		for v, r := range refs {
			if r.write != nil && hasRef(v.Type()) {
				sh.storesGlobal[f] = true
			}
		}
	}
	for changed := true; changed; {
		changed = false
		for _, f := range fns {
			if sh.storesGlobal[f] {
				continue
			}
			for _, h := range callGraph(sh.info, sh.pkg, sh.funcs[f].Body) {
				if sh.storesGlobal[h.Origin()] || (isIfaceMethod(h) && sh.ifaceStores(h.Name())) {
					sh.storesGlobal[f], changed = true, true
					break
				}
			}
		}
	}
}

// ifaceStores reports whether a method of this name that a call through an
// interface can reach stores into a package-level variable.
func (sh *cgShared) ifaceStores(name string) bool {
	for m := range sh.viaIface {
		if m.Name() == name && sh.storesGlobal[m] {
			return true
		}
	}
	return false
}

// emitGlobals places the card's package-level variables in L1 scratch. It
// returns their C++ accessors, which go before the functions, and
// init_globals, which zeroes them and runs the initialization steps the
// card needs; kernel_body calls it before the workers or goroutines start.
func emitGlobals(wp *WorkerProgram) (defs, init string) {
	sh := wp.sh
	if len(sh.globals) == 0 {
		return "", ""
	}
	g := newCgen(sh, deviceMode)
	g.initCode = true
	sizes := types.SizesFor("gc", "amd64")
	type slot struct {
		off  int64
		size int64
	}
	slots := make([]slot, len(sh.globals))
	var total int64
	for i, v := range sh.globals {
		size := sizes.Sizeof(v.Type())
		total = (total + 7) / 8 * 8
		slots[i] = slot{total, max(size, 1)}
		total += max(size, 1)
	}
	base := int64(sh.allocScratch(total))
	var d strings.Builder
	d.WriteString("// The package-level variables the card uses: each core has its own copy,\n")
	fmt.Fprintf(&d, "// in %d bytes of L1 scratch, set up by init_globals.\nnamespace pkgvar {\n", total)
	for i, v := range sh.globals {
		at := declIdent(sh.info, v)
		name := strings.TrimSuffix(strings.TrimPrefix(sh.globalName[v], "pkgvar::"), "()")
		off := base + slots[i].off
		if sizes.Sizeof(v.Type()) == 0 {
			// Go's linker puts a variable of no bytes at runtime.zerobase.
			if _, et, isArr := arrayLen(v.Type()); isArr {
				ctype := g.elemCT(et, at)
				fmt.Fprintf(&d, "[[gnu::always_inline]] inline tsr::Ptr<%s> %s() { return tsr::zero_size<%s>(); }  // var %s %s\n", ctype, name, ctype, v.Name(), v.Type())
			} else {
				ctype := g.ct(v.Type(), at)
				fmt.Fprintf(&d, "[[gnu::always_inline]] inline %s& %s() { return *tsr::zero_size<%s>(); }  // var %s %s\n", ctype, name, ctype, v.Name(), v.Type())
			}
			continue
		}
		if _, et, isArr := arrayLen(v.Type()); isArr {
			ctype := g.elemCT(et, at)
			fmt.Fprintf(&d, "[[gnu::always_inline]] inline tsr::Ptr<%s> %s() { return tsr::scratch<%s>(%d); }  // var %s %s\n",
				ctype, name, ctype, off, v.Name(), v.Type())
			continue
		}
		ctype := g.ct(v.Type(), at)
		fmt.Fprintf(&d, "[[gnu::always_inline]] inline %s& %s() { return *tsr::scratch<%s>(%d); }  // var %s %s\n",
			ctype, name, ctype, off, v.Name(), v.Type())
		fmt.Fprintf(&d, "static_assert(sizeof(%s) <= %d, \"%s fits its slot\");\n", ctype, slots[i].size, name)
	}
	d.WriteString("}  // namespace pkgvar\n\n")

	g.line("// Go's package initialization, for the variables the card uses.")
	g.line("static void init_globals() {")
	g.indent++
	calls := len(wp.initFuncs) > 0
	for _, in := range wp.inits {
		calls = calls || len(callGraph(sh.info, sh.pkg, in.Rhs)) > 0
	}
	if calls {
		g.line(fmt.Sprintf("tsr::stack_check(%d);", g.stackPos(wp.inits0(), "the package's initialization")))
	}
	g.line(fmt.Sprintf("tsr::zero(tsr::scratch<uint8_t>(%d), %d);", base, total))
	for _, in := range wp.inits {
		lhs := make([]ast.Expr, len(in.Lhs))
		for i, v := range in.Lhs {
			lhs[i] = declIdent(sh.info, v)
		}
		if len(lhs) > 1 {
			g.assignResults(lhs, in.Rhs, false)
			continue
		}
		v := in.Lhs[0]
		if v.Name() == "_" || sh.globalName[v] == "" {
			g.line("(void)(" + g.expr(in.Rhs) + ");")
			continue
		}
		if n, et, isArr := arrayLen(v.Type()); isArr {
			if _, isLit := ast.Unparen(in.Rhs).(*ast.CompositeLit); isLit {
				g.fail(in.Rhs, "array literals are not supported; declare the array and fill it")
			}
			g.line(fmt.Sprintf("tsr::assign_array<%s>(%s, %s, %d);", g.elemCT(et, in.Rhs), sh.globalName[v], g.expr(in.Rhs), n))
			continue
		}
		g.line(sh.globalName[v] + " = " + g.exprAs(in.Rhs, v.Type()) + ";")
	}
	for _, f := range wp.initFuncs {
		p := sh.fset.Position(f.Pos())
		g.line(fmt.Sprintf("%s();  // func init at %s:%d", funcName(f), filepath.Base(p.Filename), p.Line))
	}
	g.indent--
	g.line("}")
	return d.String(), g.b.String() + "\n"
}

// inits0 is where the card's initialization starts, for its stack check.
func (wp *WorkerProgram) inits0() ast.Node {
	if len(wp.inits) > 0 {
		return wp.inits[0].Rhs
	}
	return wp.sh.funcs[wp.initFuncs[0]].Name
}

// inInit reports whether p is in the package initialization that the card
// runs.
func (wp *WorkerProgram) inInit(p token.Position) bool {
	in := func(n ast.Node) bool {
		a, b := wp.sh.fset.Position(n.Pos()), wp.sh.fset.Position(n.End())
		return p.Filename == a.Filename && p.Offset >= a.Offset && p.Offset < b.Offset
	}
	for _, i := range wp.inits {
		if in(i.Rhs) {
			return true
		}
	}
	for _, f := range wp.initFuncs {
		if in(wp.sh.funcs[f]) {
			return true
		}
	}
	return false
}

// initRoots are the card's initialization code, for computeStackNeed.
func (wp *WorkerProgram) initRoots() []ast.Node {
	var roots []ast.Node
	for _, in := range wp.inits {
		roots = append(roots, in.Rhs)
	}
	for _, f := range wp.initFuncs {
		roots = append(roots, wp.sh.funcs[f].Body)
	}
	return roots
}
