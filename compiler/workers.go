package compiler

import (
	_ "embed"
	"encoding/json"
	"fmt"
	"go/ast"
	"go/constant"
	"go/token"
	"go/types"
	"os"
	"path/filepath"
	"sort"
	"strings"
)

// Worker programs (v0.2): main loads byte arrays with tt.LoadBytes and runs
// one tt.Parallel; each worker is ordinary integer Go code that runs on its
// own Tensix core (RISC-V 0, the BRISC), reading the arrays from DRAM and
// appending results to its tt.List. The rest of main runs on the host.

//go:embed runtime/device.h
var deviceRuntime string

//go:embed runtime/text_device.h
var textDeviceRuntime string

//go:embed runtime/bits_device.h
var bitsDeviceRuntime string

//go:embed runtime/maps_device.h
var mapsDeviceRuntime string

// The device runtime is device.h, then the string functions and the maps,
// which build on it.
func init() {
	deviceRuntime += "\n" + textDeviceRuntime + "\n" + mapsDeviceRuntime + "\n" + bitsDeviceRuntime
}

//go:embed runtime/host.h
var hostRuntime string

//go:embed runtime/detector_device.h
var detectorDeviceRuntime string

//go:embed runtime/detector_host.h
var detectorHostRuntime string

//go:embed runtime/detector.cpp
var detectorKernel string

//go:embed runtime/tensor_device.h
var tensorDeviceRuntime string

//go:embed runtime/tensor_host.h
var tensorHostRuntime string

//go:embed runtime/tensor.cpp
var tensorKernel string

// ListCapacity is how many results one worker can append.
const ListCapacity = 4096

// WorkerProgram is a checked worker program.
type WorkerProgram struct {
	Name     string
	sh       *cgShared
	main     *ast.FuncDecl
	parallel *ast.AssignStmt
	lit      *ast.FuncLit
	elem     types.Type   // T in tt.List[T]
	captures []*types.Var // main's variables the worker uses, by position
	loads    []*types.Var // variables assigned from tt.LoadBytes or tt.LoadDetector, in order
	paths    map[*types.Var]string
	device   []*types.Func // functions the worker calls, directly or not
	detector *types.Var    // the tt.Detector the worker uses, if any
	goFuncs  []*types.Func // goroutine programs: the functions main starts with go
	goHost   bool          // main stays Go (Options.Host)
	call     *ast.CallExpr // the tt.Parallel call
	files    []*ast.File

	// The package initialization the card runs for its copy of the
	// package-level variables (globals.go), in Go's order.
	inits     []*types.Initializer
	initFuncs []*types.Func
}

// ttFunc returns the name of the tt function e names, or "".
func (wp *WorkerProgram) ttFunc(e ast.Expr) string {
	sel, ok := e.(*ast.SelectorExpr)
	if !ok {
		return ""
	}
	if f, ok := wp.sh.info.Uses[sel.Sel].(*types.Func); ok && f.Pkg() != nil && f.Pkg().Path() == TTPath {
		return f.Name()
	}
	return ""
}

// mainCallsParallel reports whether main uses tt.Parallel anywhere.
func (c *checker) mainCallsParallel(main *ast.FuncDecl) bool {
	found := false
	ast.Inspect(main.Body, func(n ast.Node) bool {
		if call, ok := n.(*ast.CallExpr); ok {
			if f := c.ttFunc(call.Fun); f != nil && f.Name() == "Parallel" {
				found = true
			}
		}
		return !found
	})
	return found
}

func (c *checker) workers(main *ast.FuncDecl) {
	wp := &WorkerProgram{
		Name:   c.prog.Name,
		main:   main,
		paths:  map[*types.Var]string{},
		goHost: c.goHost,
		files:  c.files,
		sh: &cgShared{fset: c.fset, info: c.info, pkg: c.pkg, funcs: c.funcs,
			seenType: map[*types.Named]bool{}, recursive: map[*types.Func]bool{},
			heapKiB: c.heapKiB, stackKiB: c.stackKiB, undefinedOverflow: c.undefinedOverflow,
			forceL1Stack: c.l1Stack, tensorAt: map[*ast.CallExpr]int{}},
	}
	if c.goHost {
		// main stays Go, so tt.Parallel may be anywhere in it, and the
		// files main opens are whatever its code says.
		var calls []*ast.CallExpr
		ast.Inspect(main.Body, func(n ast.Node) bool {
			if call, ok := n.(*ast.CallExpr); ok {
				if f := c.ttFunc(call.Fun); f != nil && f.Name() == "Parallel" {
					calls = append(calls, call)
				}
			}
			return true
		})
		if len(calls) > 1 {
			c.errorf(calls[1], "so far a program can call tt.Parallel once")
			return
		}
		call := calls[0]
		lit, ok := call.Args[1].(*ast.FuncLit)
		if !ok {
			c.errorf(call.Args[1], "pass the worker as a function literal: func(w int, out *tt.List[T]) { ... }")
			return
		}
		wp.call, wp.lit = call, lit
		sig := c.info.TypeOf(lit).(*types.Signature)
		wp.elem = sig.Params().At(1).Type().(*types.Pointer).Elem().(*types.Named).TypeArgs().At(0)
	}
	for _, st := range main.Body.List {
		if c.goHost {
			break
		}
		as, ok := st.(*ast.AssignStmt)
		if !ok || len(as.Rhs) != 1 {
			continue
		}
		call, name := c.ttCall(as.Rhs[0])
		switch name {
		case "LoadBytes", "OpenBytes", "LoadDetector":
			id, ok := as.Lhs[0].(*ast.Ident)
			if !ok || as.Tok != token.DEFINE || len(as.Lhs) != 1 {
				c.errorf(as, "declare the variable: x := tt.%s(path)", name)
				continue
			}
			path, ok := c.constString(call.Args[0], "the file name")
			if !ok {
				continue
			}
			v := c.info.Defs[id].(*types.Var)
			if reservedFlags[id.Name] {
				c.errorf(id, "rename %q: the compiled program uses --%s as its own flag", id.Name, id.Name)
			}
			wp.loads = append(wp.loads, v)
			wp.paths[v] = path
		case "Parallel":
			if wp.parallel != nil {
				c.errorf(as, "v0.2 supports one tt.Parallel per program")
				continue
			}
			blank := false
			if id, ok := as.Lhs[0].(*ast.Ident); ok && id.Name == "_" {
				blank = true
			}
			if len(as.Lhs) != 1 || (as.Tok != token.DEFINE && !blank) {
				c.errorf(as, "assign the results: results := tt.Parallel(n, func(w int, out *tt.List[T]) {...})")
				continue
			}
			lit, ok := call.Args[1].(*ast.FuncLit)
			if !ok {
				c.errorf(call.Args[1], "pass the worker as a function literal: func(w int, out *tt.List[T]) { ... }")
				continue
			}
			wp.parallel, wp.lit = as, lit
			sig := c.info.TypeOf(lit).(*types.Signature)
			wp.elem = sig.Params().At(1).Type().(*types.Pointer).Elem().(*types.Named).TypeArgs().At(0)
		}
	}
	// tt.LoadBytes and tt.Parallel anywhere else are reported by the
	// translator, which only accepts them as statements of main.
	if wp.lit == nil || len(c.errs) > 0 {
		return
	}
	if named, ok := types.Unalias(wp.elem).(*types.Named); !ok || named.Obj().Pkg() != c.pkg {
		c.errorf(wp.lit, "the result type of tt.List must be a struct type of this program")
		return
	} else if _, ok := named.Underlying().(*types.Struct); !ok {
		c.errorf(wp.lit, "the result type of tt.List must be a struct type of this program")
		return
	} else if hasRef(named) {
		c.errorf(wp.lit, "the results (%s) cannot hold strings, slices, maps, pointers, interfaces or functions: they are copied to the host, and what those refer to stays in the worker's L1", named.Obj().Name())
		return
	} else if types.SizesFor("gc", "amd64").Sizeof(named) == 0 {
		c.errorf(wp.lit, "the results (%s) hold nothing: give the struct a field", named.Obj().Name())
		return
	}

	// Variables of main that the worker uses become runtime arguments.
	seen := map[*types.Var]bool{}
	ast.Inspect(wp.lit.Body, func(n ast.Node) bool {
		id, ok := n.(*ast.Ident)
		if !ok {
			return true
		}
		v, ok := c.info.Uses[id].(*types.Var)
		if !ok || seen[v] || v.Parent() == c.pkg.Scope() || (v.Pos() >= wp.lit.Pos() && v.Pos() < wp.lit.End()) {
			return true
		}
		if v.Parent() == nil {
			return true // a struct field
		}
		seen[v] = true
		t := v.Type()
		switch {
		case isTT(t, "Detector"):
			if wp.detector != nil {
				c.errorf(id, "a worker can use one tt.Detector (it runs on the core's matrix engine)")
			}
			wp.detector = v
		case isTT(t, "Layer"), isTT(t, "Matrix"):
			if !c.goHost {
				c.errorf(id, "tt.Layer and tt.Matrix need main to stay Go (not -host=cpp)")
			}
		case !isTT(t, "Bytes") && !isInteger(t) && !isString(t):
			if b, ok := basicOf(t); !ok || b.Kind() != types.Bool {
				c.errorf(id, "the worker can use main's tt.Bytes, tt.Detector, tt.Layer, tt.Matrix, integer, bool and string variables; %s is %s", id.Name, t)
			}
		}
		wp.captures = append(wp.captures, v)
		return true
	})
	sort.Slice(wp.captures, func(i, j int) bool { return wp.captures[i].Pos() < wp.captures[j].Pos() })

	// The functions the worker reaches.
	var roots []*types.Func
	for _, f := range callGraph(c.info, c.pkg, wp.lit.Body) {
		if f.Name() == "main" {
			c.errorf(wp.lit, "the worker cannot call main")
			continue
		}
		roots = append(roots, f)
	}
	// The methods that calls through interfaces can reach.
	inits := c.pkgInitBodies()
	via := c.ifaceTargets([]ast.Node{wp.lit.Body}, inits)
	cardRoots := []ast.Node{wp.lit.Body}
	for _, m := range via {
		cardRoots = append(cardRoots, c.funcs[m])
	}
	roots = append(roots, via...)
	// The package-level variables the card keeps, and what sets them up.
	skipLit := func(n ast.Node) bool { return n == wp.lit }
	roots = append(roots, c.planGlobals(wp, []cardCode{{name: "the worker code", roots: cardRoots}},
		[]ast.Node{main.Body}, skipLit, c.goHost, "with -host=cpp main runs on the host in C++")...)
	if len(c.errs) > 0 {
		return
	}
	// And those that the card's initialization reaches.
	via = append(via, c.ifaceTargets(wp.initRoots(), inits)...)
	roots = append(roots, via...)
	wp.sh.viaIface = map[*types.Func]bool{}
	for _, m := range via {
		wp.sh.viaIface[m] = true
	}
	c.via = via
	// The function values the card's code makes and calls.
	fvRoots := append([]ast.Node{wp.lit.Body}, wp.initRoots()...)
	for _, m := range via {
		fvRoots = append(fvRoots, c.funcs[m])
	}
	c.planFuncValues(wp.sh, fvRoots)
	wp.device = c.reachable(roots, wp.lit, wp.sh.recursive)
	c.litRecursion(wp.sh)
	c.prog.Workers = wp
}

// deviceSet is the set of the functions compiled for the card.
func deviceSet(wp *WorkerProgram) map[*types.Func]bool {
	set := map[*types.Func]bool{}
	for _, f := range wp.device {
		set[f] = true
	}
	return set
}

// reachable returns the roots and every function they call, callees first
// (but for calls around a cycle), and marks the functions that recurse:
// those on a cycle of calls, which can have several activations at once.
func (c *checker) reachable(roots []*types.Func, site ast.Node, recursive map[*types.Func]bool) []*types.Func {
	var out []*types.Func
	// Tarjan's algorithm: a function is on a cycle when its strongly
	// connected component has other functions in it, or it calls itself.
	index := map[*types.Func]int{}
	low := map[*types.Func]int{}
	onStack := map[*types.Func]bool{}
	var stack []*types.Func
	var visit func(f *types.Func)
	visit = func(f *types.Func) {
		index[f] = len(index)
		low[f] = index[f]
		sig := f.Type().(*types.Signature)
		if f.Origin() != f || sig.TypeParams().Len() > 0 || sig.RecvTypeParams().Len() > 0 {
			// Before anything asks for the size of a type parameter, which
			// go/types cannot give.
			var at ast.Node = site
			if d := c.funcs[f.Origin()]; d != nil {
				at = d.Name
			}
			c.errorf(at, "%s is generic: generic functions and types are not supported on the card", f.Name())
			return
		}
		var body *ast.BlockStmt
		var name ast.Node = site
		if lit := c.litOf(f); lit != nil {
			// A function literal that becomes a value: what calls through
			// function values can reach.
			body, name = lit.Body, lit
		} else {
			decl := c.funcs[f]
			if decl == nil && sig.Recv() != nil {
				c.errorf(site, "%s: calls through interfaces are not supported on the card", f.Name())
				return
			}
			if decl == nil || decl.Body == nil {
				c.errorf(site, "%s has no body", f.Name())
				return
			}
			body, name = decl.Body, decl.Name
		}
		stack = append(stack, f)
		onStack[f] = true
		var callees []*types.Func
		for _, g := range callGraph(c.info, c.pkg, body) {
			if isIfaceMethod(g) {
				// Through an interface: the methods of that name it can
				// reach, which can make a cycle of calls too.
				for _, m := range c.via {
					if m.Name() == g.Name() {
						callees = append(callees, m)
					}
				}
				continue
			}
			callees = append(callees, g)
		}
		callees = append(callees, c.valueCallees(body)...)
		for _, g := range callees {
			switch {
			case g.Name() == "main":
				c.errorf(name, "%s cannot call main", f.Name())
				continue
			case g == f:
				recursive[f] = true
			}
			if _, seen := index[g]; !seen {
				visit(g)
				low[f] = min(low[f], low[g])
			} else if onStack[g] {
				low[f] = min(low[f], index[g])
			}
		}
		if low[f] == index[f] {
			var scc []*types.Func
			for {
				g := stack[len(stack)-1]
				stack = stack[:len(stack)-1]
				onStack[g] = false
				scc = append(scc, g)
				if g == f {
					break
				}
			}
			for _, g := range scc {
				if len(scc) > 1 {
					recursive[g] = true
				}
			}
		}
		if c.litOf(f) == nil {
			out = append(out, f)
		}
	}
	for _, f := range roots {
		if _, seen := index[f]; !seen {
			visit(f)
		}
	}
	return out
}

// wrapPragma makes the kernel's signed integers wrap around when they
// overflow, as Go's do, unless the program was compiled with
// -overflow=undefined. It also turns off a warning about code that is
// fine in Go, since tt-metal builds kernels with -Werror.
func wrapPragma(sh *cgShared) string {
	return `// Go allows x == x; GCC warns about it, and kernels are built with -Werror.
#pragma GCC diagnostic ignored "-Wtautological-compare"
// A variable Go only needs for unsafe.Sizeof is set and never read.
#pragma GCC diagnostic ignored "-Wunused-but-set-variable"
// A function the code only uses as a function value that nothing calls.
#pragma GCC diagnostic ignored "-Wunused-function"
` + overflowPragma(sh) + deferDefine(sh)
}

// deferDefine turns on the runtime's support for defer and recover, for a
// program with defer statements: panics then go to the innermost function
// with deferred calls.
func deferDefine(sh *cgShared) string {
	if !sh.usesDefer {
		return ""
	}
	return "// The program has defer statements: a panic runs the deferred calls.\n#define TSR_DEFER 1\n"
}

func overflowPragma(sh *cgShared) string {
	if sh.undefinedOverflow {
		return `// Compiled with tessc -overflow=undefined: signed overflow is undefined, as
// in C++, where Go wraps around. Faster loops; programs that overflow can
// give other results than go run.
`
	}
	return `// Go's integers wrap around when they overflow; C++'s signed ones are
// undefined then, and GCC optimizes on that. -fwrapv makes them wrap, for
// everything below (the tt-metal headers above keep their own options).
// It costs speed in tight loops: tessc -overflow=undefined leaves it out.
#if !defined(__clang__)  // GCC builds the kernels; clang is only used to check them
#pragma GCC optimize("wrapv")
#endif
`
}

// kernelEntry is a kernel's kernel_main, which runs kernel_body: it finds
// the scratch area and the heap in L1 and, for a program that recurses,
// puts the stack there too.
func kernelEntry(sh *cgShared) string {
	var b strings.Builder
	b.WriteString("void kernel_main() {\n")
	b.WriteString("    tsr::scratch_base = get_write_ptr(0);\n")
	if sh.allocated {
		bytes := uint32(sh.heapKiB) * 1024
		off := sh.allocScratch(int64(bytes))
		fmt.Fprintf(&b, "    tsr::heap_init(tsr::scratch_base + %d, %d);  // %d KiB for append, make and strings\n", off, bytes, sh.heapKiB)
	}
	if sh.l1Stack() {
		bytes := uint32(sh.stackKiB) * 1024
		off := sh.allocScratch(int64(bytes))
		// Room below the last check for the biggest frame GCC can make (see
		// frameBytes), and 1 KiB at least.
		spare := uint32(min(max(1024, sh.stackNeed+256), int64(bytes)/2))
		why := "they recurse"
		switch {
		case len(sh.recursive) > 0:
		case sh.forceL1Stack:
			why = "tessc was told -l1stack"
		case len(sh.litRecursive) > 0:
		case sh.usesDefer:
			why = "they defer calls, and a function with deferred calls keeps its values on the stack"
		case sh.usesIface || sh.usesFuncs:
			why = fmt.Sprintf("their interfaces and function values, with GCC merging the functions into one, can need about %d bytes of stack, more than the core's own has room for", sh.stackNeed)
		default:
			why = fmt.Sprintf("their calls can need about %d bytes of stack, more than the core's own has room for", sh.stackNeed)
		}
		fmt.Fprintf(&b, "    // The workers run on a stack of %d KiB in L1, because %s (keeping %d bytes spare).\n", sh.stackKiB, why, spare)
		fmt.Fprintf(&b, "    tsr::stack_limit = tsr::scratch_base + %d;\n", off+spare)
		fmt.Fprintf(&b, "    tsr::run_on_stack(kernel_body, tsr::scratch_base + %d);\n", off+bytes)
	} else {
		b.WriteString("    tsr::local_stack_init();  // the core's own stack, with checks and a canary\n")
		b.WriteString("    kernel_body();\n")
	}
	b.WriteString("}\n")
	return b.String()
}

// EmitWorkers writes the Metalium project for a worker program.
func EmitWorkers(wp *WorkerProgram, dir string) (files []string, err error) {
	defer func() {
		if r := recover(); r != nil {
			e, ok := r.(*Error)
			if !ok {
				panic(r)
			}
			if wp.inInit(e.Pos) {
				e.Msg += " (the card runs this to set up its own copy of the package-level variables it uses)"
			}
			files, err = nil, ErrorList{e}
		}
	}()
	sh := wp.sh
	sizes := types.SizesFor("gc", "amd64")
	const bounce = 4096 // tsr::kBounceBytes
	staging := uint32(bounce)
	var recordBytes, slot int64
	if wp.elem != nil { // tt.Parallel's result records
		recordBytes = sizes.Sizeof(wp.elem)
		slot = max(32, (recordBytes+31)/32*32)
	}
	sh.scratch = staging + uint32(slot)
	var detEmb, detCounts uint32
	if wp.detector != nil {
		detEmb = sh.allocScratch(4096)    // the byte embeddings
		detCounts = sh.allocScratch(1024) // the byte histogram
	}

	// Worker kernel: the functions, then the worker itself.
	sh.usesIface = usesIfaces(sh, wp)
	bodies := wp.initRoots()
	if wp.lit != nil {
		bodies = append(bodies, wp.lit)
	}
	var all []ast.Node
	for _, f := range wp.device {
		all = append(all, sh.funcs[f])
	}
	if sh.usesDefer = usesDefers(append(all, bodies...)); sh.usesDefer {
		sh.usesFuncs = true // deferred calls are function values
	}
	computePointers(sh, wp.device, bodies)
	computeMayAlloc(sh, wp.device)
	computeStoresGlobal(sh, wp.device)
	roots := wp.initRoots()
	if wp.goFuncs != nil {
		for _, f := range wp.goFuncs {
			roots = append(roots, sh.funcs[f])
		}
	} else {
		roots = append(roots, wp.lit.Body)
	}
	sh.stackNeed = computeStackNeed(sh, roots)
	var fns strings.Builder
	var protos []string
	for _, f := range wp.device {
		g := newCgen(sh, deviceMode)
		protos = append(protos, g.prototype(f, sh.funcs[f]))
		g = newCgen(sh, deviceMode)
		g.indent = 0
		g.deviceFunc(f, sh.funcs[f])
		fns.WriteString(g.b.String())
		fns.WriteString("\n")
	}
	globalDefs, initGlobals := emitGlobals(wp)
	fns.WriteString(initGlobals)
	if wp.goFuncs != nil {
		fvProtos, fvDefs := emitFuncValues(wp)
		fns.WriteString(fvDefs)
		ifaceProtos, ifaceDefs, ifaceAfter := emitIfaces(wp, deviceSet(wp))
		fns.WriteString(ifaceDefs)
		return emitGoroutines(wp, dir, globalDefs, fns.String(), append(append(protos, fvProtos...), ifaceProtos...), ifaceAfter)
	}
	g := newCgen(sh, deviceMode)
	sig := sh.info.TypeOf(wp.lit).(*types.Signature)
	var params, boxes []string
	param := func(v *types.Var) {
		ct, name := g.ct(v.Type(), wp.lit), g.name(v)
		if g.heapVar(v) {
			// A function value uses it by address (escape.go).
			boxes = append(boxes, fmt.Sprintf("    %s& %s = *tsr::box<%s>(%s_in_, %d);  // its address outlives the call", ct, name, ct, name, g.oom(wp.lit)))
			name += "_in_"
		}
		params = append(params, ct+" "+name)
	}
	param(sig.Params().At(0))
	param(sig.Params().At(1))
	for _, v := range wp.captures {
		param(v)
	}
	g.line("// The worker function given to tt.Parallel.")
	g.line("static void worker(" + strings.Join(params, ", ") + ") {")
	if len(callGraph(sh.info, sh.pkg, wp.lit.Body)) > 0 || sh.usesFuncs {
		g.line(fmt.Sprintf("    tsr::stack_check(%d);", g.stackPos(wp.lit, "the worker")))
	}
	for _, l := range boxes {
		g.line(l)
	}
	g.enter(wp.lit.Body, sig.Results(), "worker", wp.lit)
	g.funcBody(wp.lit.Body, wp.lit)
	g.line("}")
	fns.WriteString(g.b.String())
	fvProtos, fvDefs := emitFuncValues(wp)
	protos = append(protos, fvProtos...)
	fns.WriteString(fvDefs)
	ifaceProtos, ifaceDefs, ifaceAfter := emitIfaces(wp, deviceSet(wp))
	protos = append(protos, ifaceProtos...)
	fns.WriteString(ifaceDefs)

	elemType := g.ct(wp.elem, wp.lit)
	var km strings.Builder
	var strs strings.Builder // captured strings, copied to the heap once failures can be reported
	km.WriteString("static void kernel_body() {\n")
	km.WriteString("    uint32_t a = 0;\n")
	km.WriteString("    const int64_t index = tsr::arg64(a);\n")
	fmt.Fprintf(&km, "    tsr::List<%s> results = tsr::arg_list<%s>(a, tsr::scratch_base + %d);\n", elemType, elemType, staging)
	var callArgs = []string{"index", "&results"}
	var tensorLoads []string // the layers and matrices kept in L1
	for i, v := range wp.captures {
		arg := fmt.Sprintf("c%d", i)
		switch {
		case isTT(v.Type(), "Bytes"):
			fmt.Fprintf(&km, "    const tsr::Bytes %s = tsr::arg_bytes(a);  // %s\n", arg, v.Name())
		case isTT(v.Type(), "Detector"):
			fmt.Fprintf(&km, "    const tsr::Detector %s = tsr::arg_detector(a);  // %s\n", arg, v.Name())
			fmt.Fprintf(&km, "    tsr::det_init(%s, %d, %d);\n", arg, detEmb, detCounts)
		case isTT(v.Type(), "Layer"):
			fmt.Fprintf(&km, "    const tsr::Layer %s = tsr::arg_layer(a);  // %s\n", arg, v.Name())
			tensorLoads = append(tensorLoads, "tsr::layer_load("+arg+");")
		case isTT(v.Type(), "Matrix"):
			fmt.Fprintf(&km, "    const tsr::Matrix %s = tsr::arg_matrix(a);  // %s\n", arg, v.Name())
			tensorLoads = append(tensorLoads, "tsr::matrix_load("+arg+");")
		case isInteger(v.Type()):
			fmt.Fprintf(&km, "    const %s %s = static_cast<%s>(tsr::arg64(a));  // %s\n", g.ct(v.Type(), wp.lit), arg, g.ct(v.Type(), wp.lit), v.Name())
		case isString(v.Type()):
			fmt.Fprintf(&km, "    tsr::Str %s{};  // %s\n", arg, v.Name())
			fmt.Fprintf(&km, "    uint32_t %s_at = a;\n", arg)
			fmt.Fprintf(&km, "    a += 1 + (get_arg_val<uint32_t>(a) + 3) / 4;\n")
			fmt.Fprintf(&strs, "        %s = tsr::arg_str(%s_at, %d);\n", arg, arg, g.oom(wp.lit))
		default:
			fmt.Fprintf(&km, "    const bool %s = tsr::arg64(a) != 0;  // %s\n", arg, v.Name())
		}
		callArgs = append(callArgs, arg)
	}
	if sh.usesTensor {
		if wp.detector != nil {
			return nil, ErrorList{&Error{Pos: sh.fset.Position(wp.lit.Pos()), Msg: "a worker can use tt.Detector or tensors, not both: each has the core's matrix engine to itself"}}
		}
		km.WriteString("    // Tensors: the weights and matrices kept in L1, then each tensor's buffer.\n    tsr::tns_begin();\n")
		for _, l := range tensorLoads {
			km.WriteString("    " + l + "\n")
		}
		km.WriteString("    tsr::tns_ready();\n")
		for i, tiles := range sh.tensors {
			fmt.Fprintf(&km, "    tsr::tns_make(%d, %d);\n", 16+i, tiles)
		}
	}
	km.WriteString("    if (__builtin_setjmp(tsr::jmp) == 0) {\n")
	km.WriteString(strs.String())
	if initGlobals != "" {
		km.WriteString("        prog::init_globals();\n")
	}
	fmt.Fprintf(&km, "        prog::worker(%s);\n", strings.Join(callArgs, ", "))
	km.WriteString("    }\n")
	if wp.detector != nil {
		km.WriteString("    tsr::det_stop();\n")
	}
	if sh.usesTensor {
		km.WriteString("    tsr::tns_stop();\n")
	}
	if !sh.l1Stack() {
		fmt.Fprintf(&km, "    if (tsr::fail_pos < 0 && !tsr::local_stack_intact()) {\n        tsr::fail_pos = %d;  // the stack ran into the globals\n    }\n",
			sh.position(wp.lit, "stack overflow: the worker's stack ran into the core's memory for its globals (its calls go too deep, or hold big structs); tessc -l1stack runs the workers on a stack in L1"))
	}
	km.WriteString("    results.finish(tsr::fail_pos >= 0);\n")
	km.WriteString("}\n\n")
	km.WriteString(kernelEntry(sh))
	runtime := deviceRuntime
	if wp.detector != nil {
		runtime += "\n" + detectorDeviceRuntime
	}
	if sh.usesTensor {
		runtime += "\n" + tensorDeviceRuntime
	}

	// Host program: main, translated (unless main stays Go).
	h := newCgen(sh, hostMode)
	h.indent = 1
	for _, st := range wp.main.Body.List {
		if wp.goHost {
			break
		}
		h.hostStmt(wp, st, slot)
	}
	hostMain := h.b.String()
	structs := structDefs(sh)
	scratch := (sh.scratch + 31) / 32 * 32

	var kernel strings.Builder
	fmt.Fprintf(&kernel, `// Worker kernel for %s. %s
//
// Runs on RISC-V 0 (BRISC) of every Tensix core with a worker: the Go code of
// the function given to tt.Parallel and of everything it calls, translated
// one function at a time. Arrays live in %d bytes of L1 scratch; tt.Bytes
// and the results live in DRAM.
#include <cstdint>
#include "api/dataflow/dataflow_api.h"

%s
%s
namespace prog {

%s%s%s%s%s
%s
}  // namespace prog

%s%s`, wp.Name, generatedBy, scratch, wrapPragma(sh), runtime, literalDefs(sh), structs, structHashes(sh), globalDefs, strings.Join(protos, "\n")+"\n", fns.String(), ifaceAfter, km.String())

	if wp.goHost {
		return emitGoHost(wp, dir, kernel.String(), scratch)
	}

	hostRT := hostRuntime
	for _, v := range wp.loads {
		if isTT(v.Type(), "Detector") {
			hostRT += "\n" + detectorHostRuntime
			break
		}
	}
	queues := "1"
	for _, st := range wp.main.Body.List {
		if as, ok := st.(*ast.AssignStmt); ok && len(as.Rhs) == 1 {
			if call, ok := as.Rhs[0].(*ast.CallExpr); ok {
				if f := wp.ttFunc(call.Fun); f == "OpenBytes" {
					queues = "2"
				}
			}
		}
	}
	host := hostProgram2(wp, hostRT, structs, hostMain, queues,
		"Runs main: loads the byte arrays into the card's DRAM, starts one worker per\n"+
			"// Tensix core for tt.Parallel, collects their results and carries on with the\n"+
			"// rest of main on the host.")
	manifest, _ := json.MarshalIndent(map[string]any{
		"program":       wp.Name,
		"kind":          "workers",
		"record_bytes":  recordBytes,
		"scratch_bytes": scratch,
		"device_funcs":  len(wp.device),
		"stack_need":    wp.sh.stackNeed, // bytes the deepest chain of calls can need (an estimate)
		"stack":         map[bool]string{true: "L1", false: "local"}[wp.sh.l1Stack()],
	}, "", "  ")
	out := map[string]string{
		"CMakeLists.txt":     cmakeLists(&Program{Name: wp.Name}),
		"build.sh":           buildScript(&Program{Name: wp.Name}),
		"host.cpp":           host,
		"kernels/worker.cpp": kernel.String(),
		"program.json":       string(manifest) + "\n",
	}
	if wp.detector != nil {
		out["kernels/detector.cpp"] = detectorKernel
	}
	names := make([]string, 0, len(out))
	for n := range out {
		names = append(names, n)
	}
	sort.Strings(names)
	for _, n := range names {
		p := filepath.Join(dir, n)
		if err := os.MkdirAll(filepath.Dir(p), 0o755); err != nil {
			return nil, err
		}
		mode := os.FileMode(0o644)
		if strings.HasSuffix(n, ".sh") {
			mode = 0o755
		}
		if err := os.WriteFile(p, []byte(out[n]), mode); err != nil {
			return nil, err
		}
		files = append(files, p)
	}
	return files, nil
}

// hostStmt translates one statement of main. tt.LoadBytes and tt.Parallel
// are only allowed here, as `x := ...` statements.
func (g *cgen) hostStmt(wp *WorkerProgram, st ast.Stmt, slot int64) {
	if as, ok := st.(*ast.AssignStmt); ok && len(as.Rhs) == 1 {
		if call, ok := as.Rhs[0].(*ast.CallExpr); ok {
			if sel, ok := call.Fun.(*ast.SelectorExpr); ok {
				if id, ok := sel.X.(*ast.Ident); ok {
					if pn, ok := g.sh.info.Uses[id].(*types.PkgName); ok && pn.Imported().Path() == TTPath {
						switch sel.Sel.Name {
						case "LoadBytes":
							v := g.sh.info.Defs[as.Lhs[0].(*ast.Ident)].(*types.Var)
							g.line(fmt.Sprintf("tsr::Bytes %s = tsr::load_bytes(ctx, %s, paths.at(%s));",
								g.name(v), cstring(v.Name()), cstring(v.Name())))
							return
						case "OpenBytes":
							v := g.sh.info.Defs[as.Lhs[0].(*ast.Ident)].(*types.Var)
							g.line(fmt.Sprintf("tsr::Bytes %s = tsr::open_bytes(ctx, %s, paths.at(%s));",
								g.name(v), cstring(v.Name()), cstring(v.Name())))
							return
						case "LoadDetector":
							v := g.sh.info.Defs[as.Lhs[0].(*ast.Ident)].(*types.Var)
							g.line(fmt.Sprintf("tsr::Detector %s = tsr::load_detector(ctx, %s, paths.at(%s));",
								g.name(v), cstring(v.Name()), cstring(v.Name())))
							return
						case "Parallel":
							g.hostParallel(wp, as, call)
							return
						}
					}
				}
			}
		}
	}
	g.stmt(st)
}

func (g *cgen) hostParallel(wp *WorkerProgram, as *ast.AssignStmt, call *ast.CallExpr) {
	results := g.temp("discarded")
	if v, ok := g.sh.info.Defs[as.Lhs[0].(*ast.Ident)].(*types.Var); ok {
		results = g.name(v)
	}
	elem := g.ct(wp.elem, as)
	g.line(fmt.Sprintf("std::vector<%s> %s;", elem, results))
	g.line("{")
	g.indent++
	g.line(fmt.Sprintf("tsr::Workers job(ctx, %s, sizeof(%s), %d, %d, \"worker.cpp\");",
		g.expr(call.Args[0]), elem, ListCapacity, (g.sh.scratch+31)/32*32))
	if wp.detector != nil {
		g.line(fmt.Sprintf("tsr::add_detector(job, %s);  // its compute kernel runs on the same cores", g.name(wp.detector)))
	}
	for _, c := range wp.captures {
		if isTT(c.Type(), "Bytes") {
			g.line(fmt.Sprintf("job.serve(%s);  // read on demand if it came from tt.OpenBytes", g.name(c)))
		}
	}
	g.line("for (int64_t index = 0; index < job.n; index++) {")
	g.indent++
	g.line("std::vector<uint32_t> captured;")
	for _, c := range wp.captures {
		switch {
		case isTT(c.Type(), "Bytes"):
			g.line(fmt.Sprintf("job.push_bytes(captured, %s, index);", g.name(c)))
		case isTT(c.Type(), "Detector"):
			g.line(fmt.Sprintf("tsr::push_detector(captured, %s);", g.name(c)))
		case isInteger(c.Type()):
			g.line(fmt.Sprintf("tsr::push64(captured, static_cast<int64_t>(%s));", g.name(c)))
		case isString(c.Type()):
			g.line(fmt.Sprintf("tsr::push_string(captured, %s);", g.name(c)))
		default:
			g.line(fmt.Sprintf("tsr::push64(captured, %s ? 1 : 0);", g.name(c)))
		}
	}
	g.line("job.set_args(index, captured);")
	g.indent--
	g.line("}")
	g.line(fmt.Sprintf("%s = job.run<%s>();", results, elem))
	g.indent--
	g.line("}")
}

// unused keeps constant's import when only some paths use it.
var _ = constant.MakeBool

// hostIncludes are what the host runtime needs.
const hostIncludes = `// tt-metal's internal headers, for windows into the cores' memory (see
// tsr::CoreMemory); without them the host uses tt-metal's calls instead.
#if __has_include("impl/context/metal_context.hpp") && __has_include(<umd/device/io_window/io_window.hpp>)
#include "impl/context/metal_context.hpp"
#include "llrt/tt_cluster.hpp"
#include <umd/device/cluster.hpp>
#include <umd/device/io_window/io_window.hpp>
#define TSR_IO_WINDOW 1
#endif
#include <tt-metalium/allocator.hpp>
#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/distributed.hpp>
#include <tt-metalium/experimental/fabric/fabric.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/tt_metal.hpp>
#include <tt-metalium/work_split.hpp>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <fcntl.h>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iostream>
#include <map>
#include <memory>
#include <mutex>
#include <span>
#include <sstream>
#include <string>
#include <sys/stat.h>
#include <thread>
#include <unistd.h>
#include <vector>

using namespace tt;
using namespace tt::tt_metal;
`

// hostProgram2 renders the host program: the runtime, the struct types, the
// failure positions, main as go_main and a C++ main that parses the flags.
func hostProgram2(wp *WorkerProgram, hostRT, structs, hostMain, queues, description string) string {
	var host strings.Builder
	fmt.Fprintf(&host, `// Host program for %s. %s
//
// %s
//
//   ./build.sh
//   ./build/%s [--NAME FILE ...] [--device N] [--kernels DIR] [--compile-only]
%s
%s
namespace prog {

%s}  // namespace prog

// Places in the Go source where a worker can fail at run time.
static const tsr::Position kPositions[] = {
`, wp.Name, generatedBy, description, cident(wp.Name), hostIncludes, hostRT, structs)
	for _, p := range wp.sh.positions {
		fmt.Fprintf(&host, "    {%s, %s},\n", cstring(p.where), cstring(p.what))
	}
	if len(wp.sh.positions) == 0 {
		host.WriteString("    {\"?\", \"?\"},\n")
	}
	host.WriteString("};\n\n")
	host.WriteString("// main, translated from Go.\n")
	host.WriteString("static void go_main(tsr::Context& ctx, const std::map<std::string, std::string>& paths) {\n")
	host.WriteString("    (void)paths;\n")
	host.WriteString(hostMain)
	host.WriteString("}\n\n")
	var loadNames []string
	for _, v := range wp.loads {
		loadNames = append(loadNames, fmt.Sprintf("{%s, %s}", cstring(v.Name()), cstring(wp.paths[v])))
	}
	fmt.Fprintf(&host, `int main(int argc, char** argv) {
    tsr::Context ctx;
    ctx.program = %s;
    ctx.kernel_dir = TSR_KERNEL_DIR;
    ctx.positions = kPositions;
    ctx.n_positions = sizeof kPositions / sizeof kPositions[0];
    tsr::current = &ctx;
    std::map<std::string, std::string> paths = {%s};
    int device_id = -1;  // every chip of the card (--device N: chip N alone)
    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        auto value = [&]() -> std::string {
            if (i + 1 >= argc) {
                tsr::die("missing value after " + arg);
            }
            return argv[++i];
        };
        if (arg == "--device") {
            device_id = std::stoi(value());
        } else if (arg == "--kernels") {
            ctx.kernel_dir = value();
        } else if (arg == "--compile-only") {
            ctx.compile_only = true;
        } else if (arg.rfind("--", 0) == 0 && paths.count(arg.substr(2))) {
            paths[arg.substr(2)] = value();
        } else {
            std::cerr << "usage: " << ctx.program << " [--device N] [--kernels DIR] [--compile-only]";
            for (const auto& [name, path] : paths) {
                std::cerr << " [--" << name << " FILE (default " << path << ")]";
            }
            std::cerr << "\n";
            return arg == "--help" || arg == "-h" ? 0 : 2;
        }
    }
    // A program that reads files on demand uses a second command queue for
    // the data, while the first one runs the workers. Every chip of the
    // card: workers and goroutines fill the first, then the next.
    ctx.mesh = tsr::open_card(%s, device_id, true);
    go_main(ctx, paths);
    tsr::close_card(ctx);
    return 0;
}
`, cstring(cident(wp.Name)), strings.Join(loadNames, ", "), queues)
	return host.String()

}
