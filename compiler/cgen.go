package compiler

import (
	"fmt"
	"go/ast"
	"go/constant"
	"go/token"
	"go/types"
	"math"
	"sort"
	"strconv"
	"strings"
)

// cgen translates the integer Go subset of worker programs into C++. The same
// translator serves the worker kernels (device mode: no floating point, arrays
// in L1 scratch, slices as pointer+length views) and the host program's main
// (host mode: strings, println, the results of tt.Parallel as std::vector).
//
// The translation is close to one-to-one, so the generated code can be read
// next to the Go source. Go semantics that C++ does not share are kept with
// small helpers from the runtime: bounds-checked indexing (tsr::at), shifts
// by the width or more or by a negative amount (tsr::shl), division by zero
// and of the most negative value by -1 (tsr::div). Signed overflow wraps
// around because the kernels are built with -fwrapv (wrapPragma), unless
// tessc is told -overflow=undefined.

type cgMode int

const (
	hostMode cgMode = iota
	deviceMode
)

// cgShared is what all translated functions of one program share.
type cgShared struct {
	fset      *token.FileSet
	info      *types.Info
	pkg       *types.Package
	funcs     map[*types.Func]*ast.FuncDecl
	goFIDs    map[*types.Func]int // goroutine programs: the functions started with go
	goChips   map[*ast.GoStmt]int // and the chip a //tt:chip directive pins a go statement's goroutine to
	positions []posEntry
	structs   []*types.Named // in dependency order
	seenType  map[*types.Named]bool
	scratch   uint32          // bytes of L1 scratch allocated so far (device)
	results   []string        // definitions of the structs that hold several results
	resSeen   map[string]bool // their names

	recursive map[*types.Func]bool // functions on a cycle of calls
	mayAlloc  map[*types.Func]bool // functions that can allocate on the heap, themselves or through calls
	literals  map[string]int       // string literals in worker code, numbered
	allocated bool                 // worker code allocates on the heap
	heapKiB   int                  // the worker's heap in L1, if it allocates
	stackKiB  int                  // the worker's stack in L1, if it recurses

	undefinedOverflow bool  // Options.UndefinedOverflow
	forceL1Stack      bool  // Options.L1Stack
	stackNeed         int64 // what the deepest chain of the workers' calls can need (computeStackNeed)

	// Package-level variables (globals.go).
	globals      []*types.Var          // those the card keeps a copy of, in order
	globalName   map[*types.Var]string // their C++ accessors, pkgvar::x()
	storesGlobal map[*types.Func]bool  // functions that can store an allocation in one

	// Pointers (escape.go).
	heap    map[*types.Var]bool  // variables whose address outlives their call: they live on the heap
	ptrRecv map[*types.Func]bool // methods that take their receiver as a C++ pointer, not a reference

	// Interfaces (iface.go).
	ifaceTypes []types.Type         // the types that go into interface values: number i+1
	ifaceCT    []string             // their C++ types
	ifaceAt    []ast.Node           // where each first goes into one
	ifaceImpls []types.Type         // interface types asserted to: implements_N
	ifaceCalls []*ifaceCall         // methods called through interfaces: their dispatch functions
	viaIface   map[*types.Func]bool // methods that calls through interfaces can reach
	usesIface  bool                 // the card's code has interface values

	// Function values (funcvals.go).
	valueLit     map[*ast.FuncLit]bool         // function literals that become values
	lambdaVar    map[*types.Var]bool           // variables holding a literal that is only called: C++ lambdas
	captures     map[*ast.FuncLit][]litCapture // the variables each such literal uses from outside (escape.go)
	litRecursive map[*ast.FuncLit]bool         // such literals on a cycle of calls
	fnSources    []*fnSource                   // the function values the code makes: number i+1
	fnCalls      []*fnCall                     // the dispatch functions of calls through them
	litProtos    []string                      // the literals' closures and prototypes
	litDefs      []string                      // and their functions
	usesFuncs    bool                          // the card's code has function values
	usesDefer    bool                          // the card's code has defer statements (defer.go)

	// Tensors (tensor.go).
	tensors    []int                 // the tiles of each tt.NewTensor: circular buffer 16+i
	tensorAt   map[*ast.CallExpr]int // their buffers
	usesTensor bool                  // the card's code uses tensors
}

type posEntry struct {
	where string
	what  string
}

func (s *cgShared) position(n ast.Node, what string) int {
	p := s.fset.Position(n.Pos())
	s.positions = append(s.positions, posEntry{fmt.Sprintf("%s:%d:%d", p.Filename, p.Line, p.Column), what})
	return len(s.positions) - 1
}

// allocScratch reserves bytes of L1 scratch, 32-byte aligned for the NoC.
func (s *cgShared) allocScratch(bytes int64) uint32 {
	off := s.scratch
	s.scratch += uint32((bytes + 31) / 32 * 32)
	return off
}

type cgen struct {
	sh     *cgShared
	mode   cgMode
	b      strings.Builder
	indent int
	names  map[types.Object]string
	taken  map[string]int
	temps  int
	ctBusy map[*types.Named]bool // named types ct is translating, to catch type S []S

	// Go's order of calls (order.go).
	subst    map[ast.Expr]string // operands already evaluated into temporaries
	ordering map[ast.Expr]bool   // expressions inOrder is rendering
	callMemo map[ast.Expr]uint8  // what calls reports

	// The function being translated.
	body    *ast.BlockStmt                // for finding the gotos that use a label
	results *types.Tuple                  // its results
	resType string                        // the C++ struct that holds more than one result
	named   []string                      // C++ names of its named results
	refs    map[types.Object]bool         // pointer receivers, which are C++ references
	lambdas map[types.Object]*ast.FuncLit // local variables holding function literals
	recurse bool                          // the function is on a cycle of calls: its arrays go on the stack

	initCode bool // translating the package initialization the card runs

	bodyPre []string // lines the next loop body starts with: its heap variables (loopBody)

	defers     *deferState // the function has defer statements (defer.go)
	canRecover bool        // the function takes can_recover_
}

func newCgen(sh *cgShared, mode cgMode) *cgen {
	return &cgen{sh: sh, mode: mode, names: map[types.Object]string{}, taken: map[string]int{},
		refs: map[types.Object]bool{}, lambdas: map[types.Object]*ast.FuncLit{}}
}

func (g *cgen) fail(n ast.Node, format string, args ...any) {
	panic(&Error{Pos: g.sh.fset.Position(n.Pos()), Msg: fmt.Sprintf(format, args...)})
}

func (g *cgen) line(s string) {
	g.b.WriteString(strings.Repeat("    ", g.indent))
	g.b.WriteString(s)
	g.b.WriteByte('\n')
}

func (g *cgen) where() string {
	if g.initCode {
		return "in the initialization of package-level variables that the card uses"
	}
	if g.mode == deviceMode {
		return "in worker code"
	}
	return "in main"
}

// ------------------------------------------------------------------ names

var cppKeywords = map[string]bool{}

func init() {
	for _, k := range strings.Fields(`alignas alignof and and_eq asm auto bitand bitor bool break case catch char
		char8_t char16_t char32_t class compl concept const consteval constexpr constinit const_cast continue
		co_await co_return co_yield decltype default delete do double dynamic_cast else enum explicit export
		extern false float for friend goto if inline int long mutable namespace new noexcept not not_eq nullptr
		operator or or_eq private protected public register reinterpret_cast requires return short signed
		sizeof static static_assert static_cast struct switch template this thread_local throw true try typedef
		typeid typename union unsigned using virtual void volatile wchar_t while xor xor_eq
		main tsr prog std ctx argc argv paths net kernel_main kernel_body worker index results status fid heap_mark_
		pkgvar init_globals fenv_`) {
		cppKeywords[k] = true
	}
}

// name returns the C++ name of a local variable or parameter. Every object
// gets its own name within a function, so Go's shadowing never turns into
// C++'s "int x = x + 1".
func (g *cgen) name(obj types.Object) string {
	if n, ok := g.names[obj]; ok {
		return n
	}
	if v, ok := obj.(*types.Var); ok && isPkgVar(v) {
		// The card's own copy (globals.go); main in C++ has none.
		if n := g.sh.globalName[v]; n != "" && g.mode == deviceMode {
			return n
		}
		panic(&Error{Pos: g.sh.fset.Position(v.Pos()), Msg: fmt.Sprintf("package-level variable %s cannot be used %s", v.Name(), g.where())})
	}
	base := cident(obj.Name())
	if cppKeywords[base] {
		base += "_"
	}
	n := base
	if k := g.taken[base]; k > 0 {
		n = fmt.Sprintf("%s_%d", base, k+1)
	}
	g.taken[base]++
	g.names[obj] = n
	return n
}

// heapVar reports whether obj is a variable that lives on the heap: its
// address outlives the call (escape.go).
func (g *cgen) heapVar(obj types.Object) bool {
	v, ok := obj.(*types.Var)
	return ok && g.mode == deviceMode && g.sh.heap[v]
}

// declare renders the declaration of the local variable obj, set to init
// ("" for its zero value). A variable on the heap is a reference to it.
func (g *cgen) declare(obj types.Object, n ast.Node, init string) string {
	ctype, name := g.ct(obj.Type(), n), g.name(obj)
	switch {
	case g.heapVar(obj) && init == "":
		return fmt.Sprintf("%s& %s = *tsr::alloc<%s>(1, %d)", ctype, name, ctype, g.oom(n))
	case g.heapVar(obj):
		return fmt.Sprintf("%s& %s = *tsr::box<%s>(%s, %d)", ctype, name, ctype, init, g.oom(n))
	case init == "":
		return ctype + " " + name + "{}"
	}
	return ctype + " " + name + " = " + init
}

// heapBind is for a loop's variable that lives on the heap: the loop sets
// a plain variable of its own, whose name it returns, and each turn starts
// by putting its value in a new variable on the heap, as Go gives each turn
// its own variable.
func (g *cgen) heapBind(obj types.Object, n ast.Node) string {
	tmp := g.temp("turn")
	g.bodyPre = append(g.bodyPre, fmt.Sprintf("%s& %s = *tsr::box<%s>(%s, %d);  // this turn's own %s, on the heap",
		g.ct(obj.Type(), n), g.name(obj), g.ct(obj.Type(), n), tmp, g.oom(n), obj.Name()))
	return tmp
}

// fieldName is the C++ name of a struct field: an embedded int is a field
// named int, which C++ does not allow.
func fieldName(f *types.Var) string {
	n := cident(f.Name())
	if cppKeywords[n] {
		n += "_"
	}
	return n
}

func (g *cgen) temp(prefix string) string {
	g.temps++
	return fmt.Sprintf("%s_%d_", prefix, g.temps)
}

// funcName is the C++ name of a function; a method of T is T_name, and the
// package's init functions, of which there can be several, are init_N.
func funcName(f *types.Func) string {
	if isInitFunc(f) {
		return fmt.Sprintf("init_%d", f.Pos())
	}
	n := cident(f.Name())
	if recv := f.Type().(*types.Signature).Recv(); recv != nil {
		t := recv.Type()
		if p, ok := t.(*types.Pointer); ok {
			t = p.Elem()
		}
		if named, ok := types.Unalias(t).(*types.Named); ok {
			return cident(named.Obj().Name()) + "_" + n
		}
	}
	if cppKeywords[n] {
		n += "_"
	}
	return n
}

// ------------------------------------------------------------------ types

func (g *cgen) typeOf(e ast.Expr) types.Type { return g.sh.info.TypeOf(e) }

func isTT(t types.Type, name string) bool {
	n, ok := types.Unalias(t).(*types.Named)
	return ok && n.Obj().Pkg() != nil && n.Obj().Pkg().Path() == TTPath && n.Obj().Name() == name
}

func basicOf(t types.Type) (*types.Basic, bool) {
	b, ok := types.Unalias(t).Underlying().(*types.Basic)
	return b, ok
}

func isInteger(t types.Type) bool {
	b, ok := basicOf(t)
	return ok && b.Info()&types.IsInteger != 0
}

func isUnsigned(t types.Type) bool {
	b, ok := basicOf(t)
	return ok && b.Info()&types.IsUnsigned != 0
}

func isString(t types.Type) bool {
	b, ok := basicOf(t)
	return ok && b.Info()&types.IsString != 0
}

func isFloat(t types.Type) bool {
	b, ok := basicOf(t)
	return ok && b.Info()&types.IsFloat != 0
}

// intBits returns the width of an integer type (Go int is 64 bits here).
func intBits(t types.Type) int {
	b, _ := basicOf(t)
	switch b.Kind() {
	case types.Int8, types.Uint8:
		return 8
	case types.Int16, types.Uint16:
		return 16
	case types.Int32, types.Uint32:
		return 32
	}
	return 64
}

// narrow reports whether C++ arithmetic on t needs a cast back to t to wrap
// like Go (C++ promotes 8- and 16-bit values to int).
func narrow(t types.Type) bool { return isInteger(t) && intBits(t) < 32 }

func (g *cgen) ct(t types.Type, n ast.Node) string {
	t = types.Unalias(t)
	switch u := t.(type) {
	case *types.Named:
		obj := u.Obj()
		if obj.Pkg() != nil && obj.Pkg().Path() == TTPath {
			switch obj.Name() {
			case "Bytes":
				return "tsr::Bytes"
			case "Detector":
				return "tsr::Detector"
			case "Tensor", "Layer", "Matrix":
				if g.mode != deviceMode || g.sh.goFIDs != nil {
					g.fail(n, "tt.%s is for the worker code of tt.Parallel, with main in Go", obj.Name())
				}
				g.sh.usesTensor = true
				return "tsr::" + obj.Name()
			case "List":
				return "tsr::List<" + g.ct(u.TypeArgs().At(0), n) + ">"
			}
			g.fail(n, "tt.%s cannot be used %s", obj.Name(), g.where())
		}
		if u.TypeArgs().Len() > 0 {
			g.fail(n, "%s is generic: generic functions and types are not supported on the card", obj.Name())
		}
		if _, ok := u.Underlying().(*types.Struct); ok && obj.Pkg() == g.sh.pkg {
			g.useStruct(u, n)
			return "prog::" + cident(obj.Name())
		}
		// Any other named type is its underlying type in C++, which for
		// type S []S would never end.
		if g.ctBusy[u] {
			g.fail(n, "type %s is made of itself; recursive types other than structs are not supported on the card", obj.Name())
		}
		if g.ctBusy == nil {
			g.ctBusy = map[*types.Named]bool{}
		}
		g.ctBusy[u] = true
		defer delete(g.ctBusy, u)
		return g.ct(u.Underlying(), n)
	case *types.Basic:
		switch u.Kind() {
		case types.Bool, types.UntypedBool:
			return "bool"
		case types.Int, types.Int64, types.UntypedInt, types.UntypedRune:
			return "int64_t"
		case types.Int32:
			return "int32_t"
		case types.Int16:
			return "int16_t"
		case types.Int8:
			return "int8_t"
		case types.Uint, types.Uint64, types.Uintptr:
			return "uint64_t"
		case types.Uint32:
			return "uint32_t"
		case types.Uint16:
			return "uint16_t"
		case types.Uint8:
			return "uint8_t"
		case types.String, types.UntypedString:
			if g.mode == hostMode {
				return "std::string"
			}
			return "tsr::Str"
		case types.Float32, types.Float64, types.UntypedFloat:
			if g.mode == hostMode {
				if u.Kind() == types.Float32 {
					return "float"
				}
				return "double"
			}
			g.fail(n, "the worker cores (RISC-V, rv32im) have no floating-point unit; worker code works with integers")
		}
	case *types.Pointer:
		if isTT(u.Elem(), "List") {
			return g.ct(u.Elem(), n) + "*"
		}
		if g.mode == hostMode {
			g.fail(n, "pointers are not supported %s", g.where())
		}
		if _, et, ok := arrayLen(u.Elem()); ok {
			// A pointer to an array points to its first element, as an
			// array variable does.
			return "tsr::Ptr<" + g.elemCT(et, n) + ">"
		}
		return g.ct(u.Elem(), n) + "*"
	case *types.Slice:
		if g.mode == hostMode {
			return "std::vector<" + g.ct(u.Elem(), n) + ">"
		}
		return "tsr::Slice<" + g.elemCT(u.Elem(), n) + ">"
	case *types.Map:
		if g.mode == hostMode {
			g.fail(n, "maps are not supported in main with -host=cpp")
		}
		for _, t := range []types.Type{u.Key(), u.Elem()} {
			if _, _, isArr := arrayLen(t); isArr {
				g.fail(n, "maps of arrays (%s) are not supported on the card; use a struct that holds the array", u)
			}
		}
		return "tsr::Map<" + g.ct(u.Key(), n) + ", " + g.ct(u.Elem(), n) + ">"
	case *types.Chan:
		if g.mode == deviceMode && g.sh.goFIDs == nil {
			g.fail(n, "channels connect goroutines; worker code (tt.Parallel) cannot use them")
		}
		g.ct(u.Elem(), n) // the element type must be supported
		if hasRef(u.Elem()) {
			g.fail(n, "channels cannot carry strings, slices, maps, pointers, interfaces or functions: each goroutine's memory is its own core's")
		}
		if g.mode == hostMode {
			return "tsr::ChanRef"
		}
		return "tsr::Chan"
	case *types.Array:
		g.fail(n, "arrays can only be declared with var, indexed and sliced; pass a slice (a[:]) instead")
	case *types.Signature:
		if g.mode == hostMode {
			g.fail(n, "function values are not supported %s", g.where())
		}
		g.sh.usesFuncs = true
		return "tsr::Func"
	case *types.Interface:
		if g.mode == hostMode {
			g.fail(n, "interfaces are not supported %s", g.where())
		}
		g.sh.usesIface = true
		return "tsr::Iface"
	}
	g.fail(n, "type %s is not supported %s", t, g.where())
	return ""
}

// arrayCT is the C++ type of an array type: int64_t[4], int64_t[4][4], ...
func (g *cgen) arrayCT(t types.Type, n ast.Node) string {
	var dims strings.Builder
	for {
		length, elem, ok := arrayLen(t)
		if !ok {
			return g.ct(t, n) + dims.String()
		}
		fmt.Fprintf(&dims, "[%d]", length)
		t = elem
	}
}

// elemCT is the C++ type of an element: an array element is a C++ array.
func (g *cgen) elemCT(t types.Type, n ast.Node) string {
	if _, _, ok := arrayLen(t); ok {
		return g.arrayCT(t, n)
	}
	return g.ct(t, n)
}

// fieldOK reports whether t can be a struct field on the card: an integer,
// a bool, a string, a struct of the program, or an array or slice of those.
func (g *cgen) fieldOK(t types.Type, n ast.Node) bool {
	switch ft := types.Unalias(t).(type) {
	case *types.Basic:
		return ft.Info()&(types.IsInteger|types.IsBoolean|types.IsString) != 0
	case *types.Slice:
		return g.fieldOK(ft.Elem(), n)
	case *types.Map:
		return g.fieldOK(ft.Key(), n) && g.fieldOK(ft.Elem(), n)
	case *types.Named:
		if ft.TypeArgs().Len() > 0 {
			g.fail(n, "%s is generic: generic functions and types are not supported on the card", ft.Obj().Name())
		}
		if _, ok := ft.Underlying().(*types.Struct); ok && ft.Obj().Pkg() == g.sh.pkg {
			g.useStruct(ft, n)
			return true
		}
		if g.ctBusy[ft] {
			return false // type S []S
		}
		if g.ctBusy == nil {
			g.ctBusy = map[*types.Named]bool{}
		}
		g.ctBusy[ft] = true
		defer delete(g.ctBusy, ft)
		return isInteger(ft) || g.fieldOK(ft.Underlying(), n)
	case *types.Array:
		return g.fieldOK(ft.Elem(), n)
	case *types.Pointer:
		return g.fieldOK(ft.Elem(), n)
	case *types.Interface, *types.Signature:
		return true
	}
	return false
}

// hasArray reports whether a struct type holds an array, directly or not.
func hasArray(t types.Type) bool {
	switch u := types.Unalias(t).Underlying().(type) {
	case *types.Array:
		return true
	case *types.Struct:
		for i := 0; i < u.NumFields(); i++ {
			if hasArray(u.Field(i).Type()) {
				return true
			}
		}
	}
	return false
}

// hasRef reports whether values of t refer to memory elsewhere: whether
// they hold a string, a slice, a map, a pointer, an interface or a function
// value.
func hasRef(t types.Type) bool {
	switch u := types.Unalias(t).Underlying().(type) {
	case *types.Basic:
		return u.Info()&types.IsString != 0
	case *types.Slice, *types.Map, *types.Pointer, *types.Interface, *types.Signature:
		return true
	case *types.Array:
		return hasRef(u.Elem())
	case *types.Struct:
		for i := 0; i < u.NumFields(); i++ {
			if hasRef(u.Field(i).Type()) {
				return true
			}
		}
	}
	return false
}

// storesRef reports whether code handed a value of t can store a string or
// a slice where the code that handed it over sees it: in the array of a
// slice whose elements hold them, or in a map (which also grows into new
// memory when something is added).
func storesRef(t types.Type) bool {
	switch u := types.Unalias(t).Underlying().(type) {
	case *types.Map:
		return true
	case *types.Pointer:
		return hasRef(u.Elem()) // through the pointer, into what it points to
	case *types.Interface:
		return true // it can hold a pointer
	case *types.Signature:
		return true // a closure can store into its variables
	case *types.Slice:
		return hasRef(u.Elem())
	case *types.Array:
		return storesRef(u.Elem())
	case *types.Struct:
		for i := 0; i < u.NumFields(); i++ {
			if storesRef(u.Field(i).Type()) {
				return true
			}
		}
	}
	return false
}

// noscan is the runtime's template argument for slices of t: true when the
// elements hold no strings or slices (Go's allocator rounds those
// differently).
func noscan(t types.Type) string {
	if hasRef(t) {
		return "false"
	}
	return "true"
}

// maxArrayStruct is the largest struct with arrays in it that worker code
// may use: struct values live on the core's stack, in its 4 KB of local
// memory (MEM_BRISC_LOCAL_SIZE on Wormhole), unlike arrays declared with
// var, which live in L1.
const maxArrayStruct = 256

// useStruct records a struct type so its definition is emitted.
func (g *cgen) useStruct(t *types.Named, n ast.Node) {
	if g.sh.seenType[t] {
		return
	}
	g.sh.seenType[t] = true
	st := t.Underlying().(*types.Struct)
	for i := 0; i < st.NumFields(); i++ {
		f := st.Field(i)
		if !g.fieldOK(f.Type(), n) {
			g.fail(n, "field %s.%s: struct fields must be integers, bools, strings, structs, interfaces or function values, or arrays, slices, maps or pointers of those", t.Obj().Name(), f.Name())
		}
		if types.SizesFor("gc", "amd64").Sizeof(f.Type()) == 0 {
			// C++ gives every member at least a byte, so the struct would
			// not have Go's layout.
			g.fail(n, "field %s.%s takes no memory in Go (an empty struct or array); on the card such a field is not supported in a struct", t.Obj().Name(), f.Name())
		}
	}
	if size := types.SizesFor("gc", "amd64").Sizeof(t); g.mode == deviceMode && hasArray(t) && size > maxArrayStruct {
		g.fail(n, "struct %s holds arrays and is %d bytes; on the card such a struct can be at most %d bytes (struct values live on the core's 4 KB stack), so declare big arrays with var instead",
			t.Obj().Name(), size, maxArrayStruct)
	}
	g.sh.structs = append(g.sh.structs, t)
}

// ------------------------------------------------------------------ expressions

func (g *cgen) expr(e ast.Expr) string {
	if t, ok := g.subst[e]; ok {
		return t // evaluated already, in Go's order
	}
	tv := g.sh.info.Types[e]
	if tv.Value != nil {
		return g.constant(tv.Value, tv.Type, e)
	}
	if s, ok := g.inOrder(e, false); ok {
		return s
	}
	switch x := e.(type) {
	case *ast.ParenExpr:
		return g.expr(x.X)
	case *ast.Ident:
		obj := g.sh.info.Uses[x]
		if d := g.sh.info.Defs[x]; obj == nil && isPkgVar(d) {
			obj = d // var a, b = f(): the card's initialization sets a and b
		}
		switch obj := obj.(type) {
		case *types.Var:
			if g.lambdas[obj] != nil {
				g.fail(x, "on the card a function value can only be called: %s(...)", x.Name)
			}
			if g.refs[obj] {
				return "(&" + g.name(obj) + ")" // a pointer receiver, as a pointer
			}
			return g.name(obj)
		case *types.Func:
			if g.mode == hostMode {
				g.fail(x, "function values are not supported %s", g.where())
			}
			if obj.Pkg() == g.sh.pkg {
				return g.funcRef(obj, x) // a function value
			}
		case *types.Nil:
			switch types.Unalias(tv.Type).Underlying().(type) {
			case *types.Chan:
				return g.ct(tv.Type, x) + "{}" // a nil channel
			case *types.Slice, *types.Map:
				if g.mode == deviceMode {
					return g.ct(tv.Type, x) + "{}" // a nil slice or map
				}
			case *types.Pointer:
				if g.mode == deviceMode {
					return "static_cast<" + g.ct(tv.Type, x) + ">(nullptr)"
				}
			case *types.Signature:
				if g.mode == deviceMode {
					return "tsr::Func{}"
				}
			case *types.Basic:
				if g.mode == deviceMode {
					return "nullptr" // p == nil: slices, maps and channels compare to nil on their own
				}
			}
			g.fail(x, "nil is not supported %s", g.where())
		}
		g.fail(x, "%s cannot be used as a value here", x.Name)
	case *ast.BinaryExpr:
		if s, ok := g.nilChanCompare(x); ok {
			return s
		}
		if s, ok := g.ifaceEqual(x); ok {
			return s
		}
		if g.mode == deviceMode {
			if s, ok := g.deviceBinary(x); ok {
				return s
			}
		}
		return g.binary(x.Op, g.expr(x.X), g.expr(x.Y), g.typeOf(x.X), g.typeOf(x), x.Y, x)
	case *ast.UnaryExpr:
		return g.unary(x)
	case *ast.CallExpr:
		return g.call(x)
	case *ast.IndexExpr:
		return g.index(x)
	case *ast.SliceExpr:
		return g.sliceExpr(x)
	case *ast.SelectorExpr:
		if sel, ok := g.sh.info.Selections[x]; ok && sel.Kind() == types.FieldVal {
			return g.fieldPath(g.receiver(x.X, sel.Recv()), sel.Recv(), sel.Index(), x)
		}
		if sel, ok := g.sh.info.Selections[x]; ok && g.mode == deviceMode {
			return g.methodValue(x, sel) // x.M or T.M as a function value
		}
		if _, ok := g.typeOf(x).(*types.Signature); ok && g.mode == deviceMode {
			g.fail(x, "library functions cannot be function values on the card; use a function literal that calls %s", types.ExprString(x))
		}
		g.fail(x, "%s cannot be used as a value here", types.ExprString(x))
	case *ast.CompositeLit:
		return g.composite(x)
	case *ast.TypeAssertExpr:
		return g.typeAssert(x)
	case *ast.StarExpr:
		if id, ok := ast.Unparen(x.X).(*ast.Ident); ok && g.refs[g.sh.info.Uses[id]] {
			return g.name(g.sh.info.Uses[id]) // *p of a pointer receiver
		}
		if g.mode == hostMode {
			g.fail(x, "pointers are not supported %s", g.where())
		}
		if isArray(g.typeOf(x)) {
			return fmt.Sprintf("tsr::deref_arr(%s, %d)", g.expr(x.X), g.nilPos(x))
		}
		return fmt.Sprintf("tsr::deref(%s, %d)", g.expr(x.X), g.nilPos(x))
	case *ast.FuncLit:
		if g.mode == deviceMode && g.sh.valueLit[x] {
			return g.funcLit(x)
		}
		g.fail(x, "function values are not supported %s", g.where())
	}
	g.fail(e, "this expression is not supported %s", g.where())
	return ""
}

func (g *cgen) constant(v constant.Value, t types.Type, n ast.Node) string {
	switch v.Kind() {
	case constant.Bool:
		if constant.BoolVal(v) {
			return "true"
		}
		return "false"
	case constant.String:
		if g.mode != hostMode {
			return g.literal(constant.StringVal(v))
		}
		return "std::string(" + cstring(constant.StringVal(v)) + ")"
	case constant.Int, constant.Float:
		if isFloat(t) {
			if g.mode != hostMode {
				g.fail(n, "the worker cores have no floating-point unit; worker code works with integers")
			}
			f, _ := constant.Float64Val(v)
			return fmt.Sprintf("%v", f)
		}
		v = constant.ToInt(v)
		if v.Kind() != constant.Int {
			g.fail(n, "constant %s is not an integer", v)
		}
		if isUnsigned(t) {
			u, _ := constant.Uint64Val(v)
			if u <= math.MaxInt32 {
				// Small values stay plain ints: comparing a promoted byte
				// with 255u would draw a sign-compare warning (an error
				// in kernel builds).
				return fmt.Sprintf("%d", u)
			}
			if u <= math.MaxUint32 {
				return fmt.Sprintf("%du", u)
			}
			return fmt.Sprintf("%dull", u)
		}
		i, ok := constant.Int64Val(v)
		if !ok {
			g.fail(n, "constant %s does not fit in 64 bits", v)
		}
		switch {
		case i == math.MinInt64:
			return "INT64_MIN"
		case i < math.MinInt32 || i > math.MaxInt32:
			if i < 0 {
				return fmt.Sprintf("(%dll)", i)
			}
			return fmt.Sprintf("%dll", i)
		case i < 0:
			return fmt.Sprintf("(%d)", i)
		}
		return fmt.Sprintf("%d", i)
	}
	g.fail(n, "constant %s is not supported", v)
	return ""
}

// binary builds "l op r" with Go's semantics for the result type res.
func (g *cgen) binary(op token.Token, l, r string, lt, res types.Type, right ast.Expr, n ast.Node) string {
	var e string
	if n, _, ok := arrayLen(lt); ok && (op == token.EQL || op == token.NEQ) {
		e := fmt.Sprintf("tsr::eq_n(%s, %s, %d)", l, r, n)
		if op == token.NEQ {
			e = "!" + e
		}
		return "(" + e + ")"
	}
	switch op {
	case token.LAND, token.LOR, token.EQL, token.NEQ, token.LSS, token.LEQ, token.GTR, token.GEQ:
		return "(" + l + " " + op.String() + " " + r + ")"
	case token.AND_NOT:
		e = "(" + l + " & ~" + r + ")"
	case token.SHL, token.SHR:
		if cv := g.sh.info.Types[right].Value; cv != nil {
			if k, ok := constant.Int64Val(constant.ToInt(cv)); ok && k < int64(intBits(lt)) {
				e = "(" + l + " " + op.String() + " " + r + ")"
				break
			}
		}
		fn := "shl"
		if op == token.SHR {
			fn = "shr"
		}
		if g.mode == deviceMode && !isUnsigned(g.typeOf(right)) && g.sh.info.Types[right].Value == nil {
			// A signed count can be negative, which panics in Go.
			pos := g.sh.position(n, "negative shift amount")
			e = fmt.Sprintf("tsr::%s<%s>(%s, %s, %d)", fn, g.ct(res, n), l, r, pos)
			break
		}
		e = "tsr::" + fn + "<" + g.ct(res, n) + ">(" + l + ", " + r + ")"
	case token.QUO, token.REM:
		if isInteger(res) {
			// By zero (it panics) or by -1 (the most negative value
			// divided by -1 is itself): tsr::div and tsr::rem.
			if cv := g.sh.info.Types[right].Value; cv == nil || constant.Sign(cv) <= 0 {
				fn := "div"
				if op == token.REM {
					fn = "rem"
				}
				pos := g.sh.position(n, "integer divide by zero")
				return fmt.Sprintf("tsr::%s<%s>(%s, %s, %d)", fn, g.ct(res, n), l, r, pos)
			}
		}
		e = "(" + l + " " + op.String() + " " + r + ")"
	case token.ADD:
		if isString(res) && g.mode == deviceMode {
			return fmt.Sprintf("tsr::concat(%d, %s, %s)", g.oom(n), l, r)
		}
		e = "(" + l + " + " + r + ")"
	default:
		e = "(" + l + " " + op.String() + " " + r + ")"
	}
	if narrow(res) {
		if !strings.HasPrefix(e, "(") {
			e = "(" + e + ")" // tsr::shl<int8_t>(...)
		}
		e = g.ct(res, n) + e
	}
	return e
}

// deviceBinary handles what worker code does differently from plain C++
// operators: a + b + ... on strings (one allocation), comparisons of
// string(b) (which do not copy b), and slices compared with nil.
func (g *cgen) deviceBinary(x *ast.BinaryExpr) (string, bool) {
	if x.Op == token.ADD && isString(g.typeOf(x)) {
		var parts []string
		var flat func(e ast.Expr)
		flat = func(e ast.Expr) {
			if _, done := g.subst[e]; done {
				parts = append(parts, g.expr(e))
				return
			}
			if b, ok := ast.Unparen(e).(*ast.BinaryExpr); ok && b.Op == token.ADD && g.sh.info.Types[b].Value == nil {
				flat(b.X)
				flat(b.Y)
				return
			}
			parts = append(parts, g.expr(e))
		}
		flat(x)
		return fmt.Sprintf("tsr::concat(%d, %s)", g.oom(x), strings.Join(parts, ", ")), true
	}
	switch x.Op {
	case token.EQL, token.NEQ, token.LSS, token.LEQ, token.GTR, token.GEQ:
	default:
		return "", false
	}
	for _, pair := range [][2]ast.Expr{{x.X, x.Y}, {x.Y, x.X}} {
		if isNil(g.sh.info, pair[1]) {
			switch types.Unalias(g.typeOf(pair[0])).Underlying().(type) {
			case *types.Slice:
				return "(" + g.expr(pair[0]) + ".ptr " + x.Op.String() + " nullptr)", true
			case *types.Map:
				return "(" + g.expr(pair[0]) + ".d " + x.Op.String() + " nullptr)", true
			}
		}
	}
	if !isString(g.typeOf(x.X)) {
		return "", false
	}
	return "(" + g.compared(x.X) + " " + x.Op.String() + " " + g.compared(x.Y) + ")", true
}

// runsCode reports whether evaluating e can call a function, which could
// change a byte slice that a switch's string(b) tag reads: conversions and
// len, cap, min and max do not count.
func (g *cgen) runsCode(e ast.Expr) bool {
	found := false
	ast.Inspect(e, func(n ast.Node) bool {
		call, ok := n.(*ast.CallExpr)
		if found || !ok {
			return !found
		}
		if g.sh.info.Types[call.Fun].IsType() {
			return true
		}
		if id, ok := ast.Unparen(call.Fun).(*ast.Ident); ok {
			if b, ok := g.sh.info.Uses[id].(*types.Builtin); ok {
				switch b.Name() {
				case "len", "cap", "min", "max":
					return true
				}
			}
		}
		found = true
		return false
	})
	return found
}

// compared renders a string that is only compared: string(b) of a byte
// slice is then a view of b, as in Go, which does not copy it either.
func (g *cgen) compared(e ast.Expr) string {
	if t, ok := g.subst[e]; ok {
		return t
	}
	if call, ok := ast.Unparen(e).(*ast.CallExpr); ok && len(call.Args) == 1 && g.sh.info.Types[call].Value == nil {
		if tv := g.sh.info.Types[call.Fun]; tv.IsType() && isString(tv.Type) && isByteSlice(g.typeOf(call.Args[0])) {
			return "tsr::view(" + g.expr(call.Args[0]) + ")"
		}
	}
	return g.expr(e)
}

// nilChanCompare renders c == nil and c != nil for a channel c.
func (g *cgen) nilChanCompare(x *ast.BinaryExpr) (string, bool) {
	if x.Op != token.EQL && x.Op != token.NEQ {
		return "", false
	}
	for _, pair := range [][2]ast.Expr{{x.X, x.Y}, {x.Y, x.X}} {
		if _, ok := types.Unalias(g.typeOf(pair[0])).Underlying().(*types.Chan); ok && isNil(g.sh.info, pair[1]) {
			if g.mode == hostMode {
				return "(" + g.expr(pair[0]) + ".id " + map[token.Token]string{token.EQL: "<", token.NEQ: ">="}[x.Op] + " 0)", true
			}
			return "(" + g.expr(pair[0]) + ".base " + x.Op.String() + " 0)", true
		}
	}
	return "", false
}

// nilTest is the C++ test that val, of type t, is nil.
func (g *cgen) nilTest(val string, t types.Type, n ast.Node) string {
	switch types.Unalias(t).Underlying().(type) {
	case *types.Slice:
		return "(" + val + ".ptr == nullptr)"
	case *types.Map:
		return "(" + val + ".d == nullptr)"
	case *types.Chan:
		return "(" + val + ".base == 0)"
	case *types.Interface:
		return "(" + val + ".type == 0)"
	case *types.Pointer, *types.Signature:
		return "(" + val + " == nullptr)"
	}
	g.fail(n, "%s cannot be nil", t)
	return ""
}

func isNil(info *types.Info, e ast.Expr) bool {
	id, ok := ast.Unparen(e).(*ast.Ident)
	if !ok {
		return false
	}
	_, ok = info.Uses[id].(*types.Nil)
	return ok
}

func sliceElem(t types.Type) (types.Type, bool) {
	s, ok := types.Unalias(t).Underlying().(*types.Slice)
	if !ok {
		return nil, false
	}
	return s.Elem(), true
}

func isByteSlice(t types.Type) bool {
	e, ok := sliceElem(t)
	if !ok {
		return false
	}
	b, isBasic := basicOf(e)
	return isBasic && b.Kind() == types.Uint8
}

func isRuneSlice(t types.Type) bool {
	e, ok := sliceElem(t)
	if !ok {
		return false
	}
	b, isBasic := basicOf(e)
	return isBasic && b.Kind() == types.Int32
}

// oom is the position of something that allocates on the heap, for when the
// heap is full.
func (g *cgen) oom(n ast.Node) int {
	g.sh.allocated = true
	return g.sh.position(n, fmt.Sprintf("out of memory: the worker's heap (%d KiB of L1) is full; tessc -heap gives it more", g.sh.heapKiB))
}

// literal is a string constant in worker code. Its bytes live with the code,
// in L1 (prog::lit_N), not in .rodata, which would take the core's local
// memory.
func (g *cgen) literal(s string) string {
	if s == "" {
		return "tsr::Str{nullptr, 0}"
	}
	if g.sh.literals == nil {
		g.sh.literals = map[string]int{}
	}
	k, ok := g.sh.literals[s]
	if !ok {
		k = len(g.sh.literals)
		g.sh.literals[s] = k
	}
	return fmt.Sprintf("tsr::lit<prog::lit_%d>(%d)", k, len(s))
}

// literalDefs renders the bytes of the string literals.
func literalDefs(sh *cgShared) string {
	byNum := make([]string, len(sh.literals))
	for s, k := range sh.literals {
		byNum[k] = s
	}
	var b strings.Builder
	for k, s := range byNum {
		fmt.Fprintf(&b, "static const uint8_t lit_%d[%d] TSR_RODATA = {", k, len(s))
		for i := 0; i < len(s); i++ {
			if i > 0 {
				b.WriteString(", ")
			}
			fmt.Fprintf(&b, "%d", s[i])
		}
		q := strconv.Quote(s)
		if len(q) > 60 {
			q = q[:56] + "..."
		}
		fmt.Fprintf(&b, "};  // %s\n", q)
	}
	if b.Len() > 0 {
		b.WriteString("\n")
	}
	return b.String()
}

func (g *cgen) unary(x *ast.UnaryExpr) string {
	if x.Op == token.AND {
		return g.addressOf(x)
	}
	a := g.expr(x.X)
	t := g.typeOf(x)
	switch x.Op {
	case token.ADD:
		return a
	case token.SUB:
		if narrow(t) {
			return g.ct(t, x) + "(-" + a + ")"
		}
		return "(-" + a + ")"
	case token.NOT:
		return "(!" + a + ")"
	case token.XOR:
		if narrow(t) {
			return g.ct(t, x) + "(~" + a + ")"
		}
		return "(~" + a + ")"
	case token.AND:
		return g.addressOf(x)
	case token.ARROW:
		return g.recvValue(x)
	}
	g.fail(x, "operator %s is not supported", x.Op)
	return ""
}

// nilPos is the failure position of going through a nil pointer at n.
func (g *cgen) nilPos(n ast.Node) int {
	return g.sh.position(n, "invalid memory address or nil pointer dereference")
}

// addressOf translates &x: a new value on the heap for &T{...}, else the
// address of a variable (which escape.go put on the heap if the address
// outlives its call), of a field, or of an element.
func (g *cgen) addressOf(x *ast.UnaryExpr) string {
	if g.mode == hostMode {
		g.fail(x, "taking addresses is not supported %s", g.where())
	}
	e := ast.Unparen(x.X)
	if lit, ok := e.(*ast.CompositeLit); ok {
		t := g.typeOf(lit)
		if isArray(t) {
			g.fail(lit, "array literals are not supported; declare the array and fill it")
		}
		if types.SizesFor("gc", "amd64").Sizeof(t) == 0 {
			return fmt.Sprintf("tsr::zero_size<%s>()", g.ct(t, lit)) // Go's runtime.zerobase
		}
		return fmt.Sprintf("tsr::box<%s>(%s, %d)", g.ct(t, lit), g.composite(lit), g.oom(x))
	}
	if id, ok := e.(*ast.Ident); ok && g.refs[g.sh.info.Uses[id]] {
		g.fail(x, "taking the address of a pointer receiver is not supported on the card")
	}
	if isArray(g.typeOf(e)) {
		return g.expr(e) // an array is held as a pointer to its first element
	}
	if st, ok := e.(*ast.StarExpr); ok {
		return fmt.Sprintf("&tsr::deref(%s, %d)", g.expr(st.X), g.nilPos(st)) // &*p
	}
	return "&" + g.lvalue(e)
}

// ------------------------------------------------------------------ channels

// chanElem returns the C++ element type of the channel expression ch.
func (g *cgen) chanElem(ch ast.Expr) string {
	c, ok := types.Unalias(g.typeOf(ch)).Underlying().(*types.Chan)
	if !ok {
		g.fail(ch, "%s is not a channel", types.ExprString(ch))
	}
	return g.ct(c.Elem(), ch)
}

// recvValue renders <-ch as an expression: the next value, or the zero value
// once the channel is closed and drained.
func (g *cgen) recvValue(x *ast.UnaryExpr) string {
	elem := g.chanElem(x.X)
	if g.mode == hostMode {
		return fmt.Sprintf("net.recv_value<%s>(%s)", elem, g.expr(x.X))
	}
	return fmt.Sprintf("tsr::recv_value<%s>(%s)", elem, g.expr(x.X))
}

// recvInto renders a receive that stores into ptr and yields false once the
// channel is closed and drained.
func (g *cgen) recvInto(ch ast.Expr, ptr string) string {
	elem := g.chanElem(ch)
	if g.mode == hostMode {
		return fmt.Sprintf("net.recv<%s>(%s, %s)", elem, g.expr(ch), ptr)
	}
	return fmt.Sprintf("tsr::recv<%s>(%s, %s)", elem, g.expr(ch), ptr)
}

func (g *cgen) send(s *ast.SendStmt) {
	if g.mode == hostMode {
		g.line(fmt.Sprintf("net.send<%s>(%s, %s);", g.chanElem(s.Chan), g.expr(s.Chan), g.expr(s.Value)))
		return
	}
	pos := g.sh.position(s, "send on a closed channel")
	ch := g.expr(s.Chan)
	if c, _ := g.calls(s.Chan); c {
		if v, _ := g.calls(s.Value); v {
			// chans[f()] <- g(): f first.
			t := g.temp("ch")
			g.line("auto " + t + " = " + ch + ";")
			ch = t
		}
	}
	g.line(fmt.Sprintf("tsr::send<%s>(%s, %s, %d);", g.chanElem(s.Chan), ch, g.expr(s.Value), pos))
}

// selectStmt translates select. Its channels and the values to send are
// evaluated once, in order, as in Go; then it looks at the cases in turn,
// from a starting point that moves on each time, until one can proceed (or
// takes the default), and runs that one as a case of a C++ switch, so that
// break leaves the select. In main, it moves relayed channels on between
// rounds, and reports a deadlock once every goroutine is done.
func (g *cgen) selectStmt(s *ast.SelectStmt) {
	type comm struct {
		cc     *ast.CommClause
		ch     string // the channel, evaluated
		val    string // the value to send, evaluated
		elem   string
		send   *ast.SendStmt
		lhs    []ast.Expr // what a receive assigns to
		define bool
	}
	host := g.mode == hostMode
	chanCT := "tsr::Chan"
	if host {
		chanCT = "tsr::ChanRef"
	}
	g.line("{")
	g.indent++
	var comms []comm
	def := (*ast.CommClause)(nil)
	for _, st := range s.Body.List {
		cc := st.(*ast.CommClause)
		if cc.Comm == nil {
			def = cc
			continue
		}
		c := comm{cc: cc}
		var recv ast.Expr
		switch op := cc.Comm.(type) {
		case *ast.SendStmt:
			c.send = op
			c.elem = g.chanElem(op.Chan)
			c.ch = g.temp("sel_ch")
			g.line(fmt.Sprintf("const %s %s = %s;", chanCT, c.ch, g.expr(op.Chan)))
			c.val = g.temp("sel_v")
			g.line(fmt.Sprintf("const %s %s = %s;", c.elem, c.val, g.expr(op.Value)))
		case *ast.ExprStmt:
			recv = op.X
		case *ast.AssignStmt:
			recv, c.lhs, c.define = op.Rhs[0], op.Lhs, op.Tok == token.DEFINE
		}
		if recv != nil {
			u := ast.Unparen(recv).(*ast.UnaryExpr)
			c.elem = g.chanElem(u.X)
			c.ch = g.temp("sel_ch")
			g.line(fmt.Sprintf("const %s %s = %s;", chanCT, c.ch, g.expr(u.X)))
		}
		comms = append(comms, c)
	}
	which := g.temp("sel")
	n := len(comms)
	g.line(fmt.Sprintf("int %s = -1;", which))
	switch {
	case n == 0 && def == nil: // select {} blocks for ever
		if host {
			g.line("net.block_forever();")
		} else {
			g.line("tsr::block_forever();")
		}
	case n == 0:
		g.line(fmt.Sprintf("%s = 0;", which))
	default:
		turn, k, last := g.temp("turn"), g.temp("k"), g.temp("last")
		if host {
			g.line(fmt.Sprintf("const uint32_t %s = net.select_turn();", turn))
			g.line(fmt.Sprintf("for (bool %s = false;;) {", last))
		} else {
			g.line(fmt.Sprintf("const uint32_t %s = tsr::select_turn++;", turn))
			g.line("for (;;) {")
		}
		g.indent++
		g.line(fmt.Sprintf("for (uint32_t %s = 0; %s < %d && %s < 0; %s++) {", k, k, n, which, k))
		g.line(fmt.Sprintf("    switch ((%s + %s) %% %d) {", turn, k, n))
		for i, c := range comms {
			ready := "recv_ready"
			if c.send != nil {
				ready = "send_ready"
			}
			if host {
				ready = "net." + ready
			} else {
				ready = "tsr::" + ready
			}
			g.line(fmt.Sprintf("        case %d: if (%s(%s)) { %s = %d; } break;", i, ready, c.ch, which, i))
		}
		g.line("    }")
		g.line("}")
		g.line(fmt.Sprintf("if (%s >= 0) {", which))
		g.line("    break;")
		g.line("}")
		switch {
		case def != nil:
			g.line(fmt.Sprintf("%s = %d;  // default", which, n))
			g.line("break;")
		case host:
			g.line(fmt.Sprintf("if (%s) {", last))
			g.line("    net.select_deadlock();")
			g.line("}")
			g.line(fmt.Sprintf("%s = net.select_idle();", last))
		}
		g.indent--
		g.line("}")
	}
	g.line(fmt.Sprintf("switch (%s) {", which))
	for i, c := range comms {
		g.line(fmt.Sprintf("case %d: {", i))
		g.indent++
		switch {
		case c.send != nil && host:
			g.line(fmt.Sprintf("net.send<%s>(%s, %s);", c.elem, c.ch, c.val))
		case c.send != nil:
			g.line(fmt.Sprintf("tsr::send<%s>(%s, %s, %d);", c.elem, c.ch, c.val, g.sh.position(c.send, "send on a closed channel")))
		default:
			v, ok := g.temp("v"), g.temp("ok")
			g.line(fmt.Sprintf("%s %s{};", c.elem, v))
			call := fmt.Sprintf("tsr::recv<%s>(%s, &%s)", c.elem, c.ch, v)
			if host {
				call = fmt.Sprintf("net.recv<%s>(%s, &%s)", c.elem, c.ch, v)
			}
			g.line(fmt.Sprintf("const bool %s = %s;", ok, call))
			g.line(fmt.Sprintf("(void)%s;", ok))
			if len(c.lhs) > 0 {
				g.line(g.assignOne(c.lhs[0], v, c.define) + ";")
			}
			if len(c.lhs) > 1 {
				g.line(g.assignOne(c.lhs[1], ok, c.define) + ";")
			}
		}
		g.indent--
		g.block(c.cc.Body)
		g.line("    break;")
		g.line("}")
	}
	if def != nil {
		g.line(fmt.Sprintf("case %d: {", n))
		g.block(def.Body)
		g.line("    break;")
		g.line("}")
	}
	g.line("default:")
	g.line("    break;")
	g.line("}")
	g.indent--
	g.line("}")
}

// commaOkRecv handles v, ok := <-ch and v, ok = <-ch.
func (g *cgen) commaOkRecv(s *ast.AssignStmt, recv *ast.UnaryExpr) {
	elem := g.chanElem(recv.X)
	tmp := g.temp("v")
	okTmp := g.temp("ok")
	g.line(fmt.Sprintf("%s %s{};", elem, tmp))
	g.line(fmt.Sprintf("bool %s = %s;", okTmp, g.recvInto(recv.X, "&"+tmp)))
	elemT := types.Unalias(g.typeOf(recv.X)).Underlying().(*types.Chan).Elem()
	g.line(g.assignOne(s.Lhs[0], g.convertTo(tmp, elemT, s.Lhs[0], s), s.Tok == token.DEFINE) + ";")
	g.line(g.assignOne(s.Lhs[1], g.convertTo(okTmp, types.Typ[types.Bool], s.Lhs[1], s), s.Tok == token.DEFINE) + ";")
}

// goStmt starts a goroutine from main: net.go(function, arguments). The
// directions of its channel parameters say which end each channel is.
func (g *cgen) goStmt(s *ast.GoStmt) {
	if g.mode != hostMode {
		g.fail(s, "goroutines are started from main, one per Tensix core")
	}
	id, ok := ast.Unparen(s.Call.Fun).(*ast.Ident)
	var f *types.Func
	if ok {
		f, _ = g.sh.info.Uses[id].(*types.Func)
	}
	fid, known := g.sh.goFIDs[f]
	if f == nil || !known {
		g.fail(s, "start goroutines with a function of this program: go f(args)")
	}
	sig := f.Type().(*types.Signature)
	var args []string
	for i, a := range s.Call.Args {
		pt := sig.Params().At(i).Type()
		x := g.expr(a)
		switch {
		case isTT(pt, "Bytes"):
			args = append(args, "tsr::arg_bytes("+x+")")
		case isInteger(pt):
			args = append(args, "tsr::arg_int(static_cast<int64_t>("+x+"))")
		default:
			if lc, _, _, isList := chanList(pt); isList {
				if lc.Dir() == types.SendOnly {
					args = append(args, "tsr::arg_sends("+x+")")
				} else {
					args = append(args, "tsr::arg_recvs("+x+")")
				}
			} else if c, ok := types.Unalias(pt).Underlying().(*types.Chan); ok {
				if c.Dir() == types.SendOnly {
					args = append(args, "tsr::arg_send("+x+")")
				} else {
					args = append(args, "tsr::arg_recv("+x+")")
				}
			} else {
				args = append(args, "tsr::arg_int(("+x+") ? 1 : 0)")
			}
		}
	}
	chip := -1 // anywhere
	if c, ok := g.sh.goChips[s]; ok {
		chip = c
	}
	g.line(fmt.Sprintf("net.go(%d, %s, {%s}, %d);", fid, cstring(f.Name()), strings.Join(args, ", "), chip))
}

func (g *cgen) args(list []ast.Expr) string {
	parts := make([]string, len(list))
	for i, a := range list {
		parts[i] = g.expr(a)
	}
	return strings.Join(parts, ", ")
}

// argsAs translates values that go where values of type t(i) go: an
// untyped nil there needs its type in C++.
func (g *cgen) argsAs(list []ast.Expr, t func(i int) types.Type) string {
	parts := make([]string, len(list))
	for i, a := range list {
		if tt := t(i); tt != nil {
			parts[i] = g.exprAs(a, tt)
		} else {
			parts[i] = g.expr(a)
		}
	}
	return strings.Join(parts, ", ")
}

// callArgs translates the arguments of a call of a function of type sig.
// For a variadic function, the values that go to its ... parameter become
// a new slice (nil if there are none), unless the call passes one with s....
func (g *cgen) callArgs(sig *types.Signature, args []ast.Expr, ellipsis bool, n ast.Node) string {
	np := sig.Params().Len()
	param := func(i int) types.Type {
		if i < np {
			return sig.Params().At(i).Type()
		}
		return nil
	}
	if !sig.Variadic() || ellipsis {
		return g.argsAs(args, param)
	}
	var parts []string
	if np > 1 {
		parts = append(parts, g.argsAs(args[:np-1], param))
	}
	st := sig.Params().At(np - 1).Type()
	elem, _ := sliceElem(st)
	if rest := args[np-1:]; len(rest) == 0 {
		parts = append(parts, g.ct(st, n)+"{}")
	} else {
		parts = append(parts, fmt.Sprintf("tsr::literal<%s>(%d, %s)", g.elemCT(elem, n), g.oom(n), g.argsAs(rest, func(int) types.Type { return elem })))
	}
	return strings.Join(parts, ", ")
}

// convertTo is val, of type from, as a value for lhs: an interface value
// if lhs is an interface and from is not.
func (g *cgen) convertTo(val string, from types.Type, lhs ast.Expr, n ast.Node) string {
	lt := g.typeOf(lhs)
	if lt == nil || !types.IsInterface(lt) || types.IsInterface(from) || g.mode != deviceMode {
		return val
	}
	id := g.typeID(from, n)
	return fmt.Sprintf("tsr::to_iface<%s>(%d, %s, %d)", g.sh.ifaceCT[id-1], id, val, g.oom(n))
}

// paramType is the type of argument i of a call of a function of type sig.
func paramType(sig *types.Signature, i int) types.Type {
	n := sig.Params().Len()
	if sig.Variadic() && i >= n-1 {
		if e, ok := sliceElem(sig.Params().At(n - 1).Type()); ok {
			return e
		}
		return nil
	}
	if i < n {
		return sig.Params().At(i).Type()
	}
	return nil
}

func (g *cgen) call(x *ast.CallExpr) string {
	if len(x.Args) == 1 {
		if t, ok := g.typeOf(x.Args[0]).(*types.Tuple); ok && t.Len() > 1 {
			g.fail(x, "pass the results of %s one at a time", types.ExprString(x.Args[0]))
		}
	}
	if tv := g.sh.info.Types[x.Fun]; tv.IsType() {
		if g.mode == deviceMode && isNil(g.sh.info, x.Args[0]) {
			return g.exprAs(x.Args[0], tv.Type) // T(nil)
		}
		if g.mode == deviceMode {
			if s, ok := g.conversion(tv.Type, x); ok {
				return s
			}
			if types.IsInterface(tv.Type) {
				return g.exprAs(x.Args[0], tv.Type) // I(v)
			}
			if _, ok := types.Unalias(tv.Type).Underlying().(*types.Struct); ok && !types.Identical(tv.Type, g.typeOf(x.Args[0])) {
				// T(s) between struct types with the same fields: the same bytes.
				return fmt.Sprintf("tsr::struct_cast<%s>(%s)", g.ct(tv.Type, x), g.expr(x.Args[0]))
			}
			if pt, ok := types.Unalias(tv.Type).Underlying().(*types.Pointer); ok {
				if n, et, isArr := arrayLen(pt.Elem()); isArr {
					if _, fromSlice := types.Unalias(g.typeOf(x.Args[0])).Underlying().(*types.Slice); fromSlice {
						// (*[N]T)(s): s's array, if s has N elements.
						return fmt.Sprintf("tsr::slice_to_array<%s>(%s, %d, %d)", g.elemCT(et, x), g.expr(x.Args[0]), n,
							g.sh.position(x, fmt.Sprintf("cannot convert slice to pointer to array with length %d", n)))
					}
				}
				if isNil(g.sh.info, x.Args[0]) || types.Identical(tv.Type.Underlying(), g.typeOf(x.Args[0]).Underlying()) {
					return "static_cast<" + g.ct(tv.Type, x) + ">(" + g.exprAs(x.Args[0], tv.Type) + ")"
				}
				// (*T)(p) where T and *p's type have the same fields.
				return "reinterpret_cast<" + g.ct(tv.Type, x) + ">(" + g.expr(x.Args[0]) + ")"
			}
		} else if isString(tv.Type) || isString(g.typeOf(x.Args[0])) {
			g.fail(x, "string conversions are not supported; use strconv.Itoa")
		}
		if _, ok := types.Unalias(tv.Type).Underlying().(*types.Signature); ok && g.mode == deviceMode {
			return g.exprAs(x.Args[0], tv.Type) // F(f), F(nil)
		}
		return g.ct(tv.Type, x) + "(" + g.expr(x.Args[0]) + ")"
	}
	switch fun := ast.Unparen(x.Fun).(type) {
	case *ast.FuncLit:
		if g.mode == deviceMode {
			return "(" + g.lambda(fun) + ")(" + g.args(x.Args) + ")"
		}
	case *ast.Ident:
		switch obj := g.sh.info.Uses[fun].(type) {
		case *types.Var:
			if g.lambdas[obj] != nil {
				return g.name(obj) + "(" + g.args(x.Args) + ")"
			}
		case *types.Builtin:
			return g.builtin(obj.Name(), x)
		case *types.Func:
			if obj.Pkg() == g.sh.pkg {
				if g.mode == hostMode {
					g.fail(x, "main can call only tt functions and builtins; %s runs on the card, so call it from the tt.Parallel worker", obj.Name())
				}
				sig := obj.Type().(*types.Signature)
				return "prog::" + funcName(obj) + "(" + g.callArgs(sig, x.Args, x.Ellipsis.IsValid(), x) + ")"
			}
		}
	case *ast.SelectorExpr:
		if id, ok := fun.X.(*ast.Ident); ok {
			if pn, ok := g.sh.info.Uses[id].(*types.PkgName); ok {
				return g.pkgCall(pn.Imported().Path(), fun.Sel.Name, x)
			}
		}
		if sel, ok := g.sh.info.Selections[fun]; ok && sel.Kind() == types.MethodVal {
			if m := sel.Obj().(*types.Func); isIfaceMethod(m) && g.mode == deviceMode {
				return g.ifaceMethodCall(fun, sel, m, x)
			}
			if m := sel.Obj().(*types.Func); m.Pkg() == g.sh.pkg {
				if g.mode == hostMode {
					g.fail(x, "main can call only tt functions and builtins; %s runs on the card, so call it from the tt.Parallel worker", m.Name())
				}
				args := []string{g.methodRecv(fun.X, sel, m, x)}
				if len(x.Args) > 0 {
					sig := m.Type().(*types.Signature)
					args = append(args, g.callArgs(sig, x.Args, x.Ellipsis.IsValid(), x))
				}
				return "prog::" + funcName(m) + "(" + strings.Join(args, ", ") + ")"
			}
			return g.methodCall(sel.Recv(), fun.Sel.Name, g.expr(fun.X), x)
		}
	}
	if g.mode == deviceMode {
		if sig, ok := valueCall(g.sh.info, x, g.sh.lambdaVar); ok {
			return g.valueCall(x, sig) // through a function value
		}
	}
	g.fail(x, "this call is not supported %s", g.where())
	return ""
}

// methodRecv is the receiver that a call x.m(...) passes to method m of
// selection s, found through embedded fields: a copy for a value receiver,
// the value itself for a pointer receiver taken by reference (as before
// pointers), and a pointer for one taken as a C++ pointer (escape.go).
func (g *cgen) methodRecv(x ast.Expr, s *types.Selection, m *types.Func, n ast.Node) string {
	t := g.typeOf(x)
	base, isPtr := "", isPointer(t)
	if id, ok := ast.Unparen(x).(*ast.Ident); ok && g.refs[g.sh.info.Uses[id]] {
		base, isPtr = g.name(g.sh.info.Uses[id]), false // a receiver held as a reference
	} else {
		base = g.expr(x)
	}
	return g.recvArg(base, t, isPtr, s.Index(), m, n)
}

// recvArg is the receiver for method m from base, of Go type t (isPtr: base
// is a C++ pointer), through the embedded fields of idx, whose last entry
// is the method's.
func (g *cgen) recvArg(base string, t types.Type, isPtr bool, idx []int, m *types.Func, n ast.Node) string {
	x := n
	for _, i := range idx[:len(idx)-1] {
		if p, ok := types.Unalias(t).Underlying().(*types.Pointer); ok {
			t = p.Elem()
			if isPtr {
				base = fmt.Sprintf("tsr::deref(%s, %d)", base, g.nilPos(n))
			}
		}
		f := types.Unalias(t).Underlying().(*types.Struct).Field(i)
		base += "." + fieldName(f)
		t = f.Type()
		isPtr = isPointer(t)
	}
	takesPtr := pointerRecv(m) && g.sh.ptrRecv[m]
	switch {
	case takesPtr && isPtr:
		return base
	case takesPtr:
		return "&" + base
	case isPtr:
		if g.mode == hostMode {
			g.fail(x, "pointers are not supported %s", g.where())
		}
		return fmt.Sprintf("tsr::deref(%s, %d)", base, g.nilPos(n))
	}
	return base
}

// conversion translates the conversions that involve strings in worker
// code; ok is false for the others, which are C++ casts.
func (g *cgen) conversion(to types.Type, x *ast.CallExpr) (string, bool) {
	arg := x.Args[0]
	from := g.typeOf(arg)
	switch {
	case types.IsInterface(to):
		return "", false // I(s): into an interface
	case isString(to) && isString(from):
		return g.expr(arg), true
	case isString(to) && isByteSlice(from):
		return fmt.Sprintf("tsr::to_string(%s, %d)", g.expr(arg), g.oom(x)), true
	case isString(to) && isRuneSlice(from):
		return fmt.Sprintf("tsr::runes_string(%s, %d)", g.expr(arg), g.oom(x)), true
	case isString(to) && isInteger(from):
		return fmt.Sprintf("tsr::rune_string(static_cast<int64_t>(%s), %d)", g.expr(arg), g.oom(x)), true
	case isByteSlice(to) && isString(from):
		return fmt.Sprintf("tsr::to_bytes(%s, %d)", g.expr(arg), g.oom(x)), true
	case isRuneSlice(to) && isString(from):
		return fmt.Sprintf("tsr::to_runes(%s, %d)", g.expr(arg), g.oom(x)), true
	case isString(to) || isString(from):
		g.fail(x, "converting %s to %s is not supported in worker code", from, to)
	}
	return "", false
}

func (g *cgen) builtin(name string, x *ast.CallExpr) string {
	switch name {
	case "new":
		if g.mode == hostMode {
			g.fail(x, "the builtin new is not supported %s", g.where())
		}
		t := g.typeOf(x.Args[0])
		if n, et, ok := arrayLen(t); ok {
			return fmt.Sprintf("tsr::alloc<%s>(%d, %d)", g.elemCT(et, x), n, g.oom(x))
		}
		if types.SizesFor("gc", "amd64").Sizeof(t) == 0 {
			return fmt.Sprintf("tsr::zero_size<%s>()", g.ct(t, x)) // Go's runtime.zerobase
		}
		return fmt.Sprintf("tsr::alloc<%s>(1, %d)", g.ct(t, x), g.oom(x))
	case "len", "cap":
		t := types.Unalias(g.typeOf(x.Args[0])).Underlying()
		a := g.expr(x.Args[0])
		switch {
		case g.mode == hostMode && name == "len":
			return "static_cast<int64_t>(" + a + ".size())"
		case g.mode == hostMode:
		case isString(t) && name == "len":
			return a + ".len"
		}
		if _, ok := t.(*types.Slice); ok && g.mode == deviceMode {
			return a + "." + name
		}
		if _, ok := t.(*types.Map); ok && name == "len" {
			return "tsr::map_len(" + a + ")"
		}
	case "delete":
		key := types.Unalias(g.typeOf(x.Args[0])).Underlying().(*types.Map).Key()
		return fmt.Sprintf("tsr::map_delete(%s, %s)", g.expr(x.Args[0]), g.exprAs(x.Args[1], key))
	case "clear":
		if g.mode == deviceMode {
			if _, ok := types.Unalias(g.typeOf(x.Args[0])).Underlying().(*types.Map); ok {
				return "tsr::map_clear(" + g.expr(x.Args[0]) + ")"
			}
			return "tsr::clear_slice(" + g.expr(x.Args[0]) + ")"
		}
	case "append":
		if g.mode == deviceMode {
			return g.appendCall(x)
		}
	case "min", "max":
		return "tsr::" + name + "<" + g.ct(g.typeOf(x), x) + ">(" + g.args(x.Args) + ")"
	case "copy":
		if g.mode == hostMode {
			break
		}
		return "tsr::copy(" + g.args(x.Args) + ")"
	case "println":
		if g.mode != hostMode {
			g.fail(x, "println is not available in worker code; append results to the tt.List instead")
		}
		return "tsr::println(" + g.args(x.Args) + ")"
	case "panic":
		return g.panicCall(x)
	case "recover":
		if g.mode == deviceMode {
			return g.recoverCall()
		}
	case "close":
		if g.mode == hostMode {
			return "net.close(" + g.expr(x.Args[0]) + ")"
		}
		pos := g.sh.position(x, "close of a closed channel")
		return fmt.Sprintf("tsr::close(%s, %d, %d)", g.expr(x.Args[0]), pos, g.sh.position(x, "close of nil channel"))
	case "make":
		if g.mode != hostMode {
			if mt, ok := types.Unalias(g.typeOf(x)).Underlying().(*types.Map); ok {
				g.ct(mt, x) // a map type the card supports
				hint := "0"
				if len(x.Args) > 1 {
					hint = g.expr(x.Args[1])
				}
				return fmt.Sprintf("tsr::make_map<%s, %s>(%s, %d)", g.ct(mt.Key(), x), g.ct(mt.Elem(), x), hint, g.oom(x))
			}
			elem, ok := sliceElem(g.typeOf(x))
			if !ok {
				g.fail(x, "channels are made in main and passed to the goroutines")
			}
			if _, _, isArr := arrayLen(elem); isArr {
				g.fail(x, "slices of arrays are not supported on the card; use a slice of structs")
			}
			ct := g.elemCT(elem, x)
			lenPos := g.sh.position(x, "makeslice: len out of range")
			if len(x.Args) == 2 {
				return fmt.Sprintf("tsr::make<%s>(%s, %d, %d)", ct, g.expr(x.Args[1]), lenPos, g.oom(x))
			}
			capPos := g.sh.position(x, "makeslice: cap out of range")
			return fmt.Sprintf("tsr::make<%s>(%s, %s, %d, %d, %d)", ct, g.expr(x.Args[1]), g.expr(x.Args[2]), lenPos, capPos, g.oom(x))
		}
		switch t := types.Unalias(g.typeOf(x)).Underlying().(type) {
		case *types.Chan:
			capacity := "1" // an unbuffered channel holds one value on the card
			if len(x.Args) > 1 {
				capacity = g.expr(x.Args[1])
			}
			return fmt.Sprintf("net.make_chan(sizeof(%s), %s)", g.ct(t.Elem(), x), capacity)
		case *types.Slice:
			if len(x.Args) > 2 {
				g.fail(x, "make with a capacity is not supported")
			}
			return fmt.Sprintf("std::vector<%s>(static_cast<size_t>(%s))", g.ct(t.Elem(), x), g.expr(x.Args[1]))
		}
	}
	g.fail(x, "the builtin %s is not supported %s", name, g.where())
	return ""
}

// appendCall translates append(s, v...) and append(s, t...) in worker code.
func (g *cgen) appendCall(x *ast.CallExpr) string {
	elem, _ := sliceElem(g.typeOf(x))
	if _, _, isArr := arrayLen(elem); isArr {
		g.fail(x, "slices of arrays are not supported on the card; use a slice of structs")
	}
	s := g.expr(x.Args[0])
	if len(x.Args) == 1 {
		return s
	}
	if x.Ellipsis.IsValid() {
		return fmt.Sprintf("tsr::append_all<%s>(%d, %s, %s)", noscan(elem), g.oom(x), s, g.expr(x.Args[1]))
	}
	return fmt.Sprintf("tsr::append<%s>(%d, %s, %s)", noscan(elem), g.oom(x), s, g.argsAs(x.Args[1:], func(int) types.Type { return elem }))
}

// pkgFunc is a function of the standard library that worker code can call:
// the runtime's C++ for it (compiler/runtime/text_device.h), which takes
// first the position for "out of memory" if it allocates, then the position
// for what else can go wrong (panics), then Go's arguments.
type pkgFunc struct {
	cpp     string
	alloc   bool
	panics  string
	panics2 string // a second way to panic, whose position comes next
}

var pkgFuncs = map[string]pkgFunc{}

func init() {
	simple := map[string][]string{
		"strings": {"Compare", "Contains", "ContainsAny", "ContainsRune", "Count", "Cut", "CutPrefix", "CutSuffix",
			"HasPrefix", "HasSuffix", "Index", "IndexAny", "IndexByte", "IndexRune", "LastIndex", "LastIndexAny",
			"LastIndexByte", "Trim", "TrimLeft", "TrimRight", "TrimPrefix", "TrimSuffix", "TrimSpace"},
		"bytes": {"Compare", "Contains", "ContainsAny", "ContainsRune", "Count", "Cut", "CutPrefix", "CutSuffix", "Equal",
			"HasPrefix", "HasSuffix", "Index", "IndexAny", "IndexByte", "IndexRune", "LastIndex", "LastIndexAny",
			"LastIndexByte", "Trim", "TrimLeft", "TrimRight", "TrimPrefix", "TrimSuffix", "TrimSpace"},
		"unicode/utf8": {"DecodeRune", "DecodeRuneInString", "DecodeLastRune", "DecodeLastRuneInString", "RuneCount",
			"RuneCountInString", "RuneLen", "RuneStart", "Valid", "ValidRune", "ValidString"},
		"unicode": {"IsSpace"},
		"math/bits": {"OnesCount", "OnesCount8", "OnesCount16", "OnesCount32", "OnesCount64",
			"Len", "Len8", "Len16", "Len32", "Len64",
			"LeadingZeros", "LeadingZeros8", "LeadingZeros16", "LeadingZeros32", "LeadingZeros64",
			"TrailingZeros", "TrailingZeros8", "TrailingZeros16", "TrailingZeros32", "TrailingZeros64",
			"RotateLeft", "RotateLeft8", "RotateLeft16", "RotateLeft32", "RotateLeft64",
			"Reverse", "Reverse8", "Reverse16", "Reverse32", "Reverse64",
			"ReverseBytes", "ReverseBytes16", "ReverseBytes32", "ReverseBytes64",
			"Add", "Add32", "Add64", "Sub", "Sub32", "Sub64", "Mul", "Mul32", "Mul64"},
	}
	allocating := map[string][]string{
		"strings":      {"Clone", "Fields", "Join", "Replace", "ReplaceAll", "Split", "SplitAfter", "SplitN", "SplitAfterN"},
		"bytes":        {"Clone", "Fields", "Join", "Replace", "ReplaceAll", "Split", "SplitAfter", "SplitN", "SplitAfterN"},
		"unicode/utf8": {"AppendRune"},
	}
	prefix := func(path string) string { return "tsr::" + path[strings.LastIndex(path, "/")+1:] + "_" }
	for path, names := range simple {
		for _, n := range names {
			pkgFuncs[path+"."+n] = pkgFunc{cpp: prefix(path) + n}
		}
	}
	for path, names := range allocating {
		for _, n := range names {
			pkgFuncs[path+"."+n] = pkgFunc{cpp: prefix(path) + n, alloc: true}
		}
	}
	pkgFuncs["strings.Repeat"] = pkgFunc{cpp: "tsr::strings_Repeat", alloc: true, panics: "strings: negative Repeat count"}
	pkgFuncs["bytes.Repeat"] = pkgFunc{cpp: "tsr::bytes_Repeat", alloc: true, panics: "bytes: negative Repeat count"}
	pkgFuncs["unicode/utf8.EncodeRune"] = pkgFunc{cpp: "tsr::utf8_EncodeRune", panics: "index out of range in utf8.EncodeRune"}
	for _, n := range []string{"Div", "Div32", "Div64"} {
		pkgFuncs["math/bits."+n] = pkgFunc{cpp: "tsr::bits_" + n, panics: "integer divide by zero", panics2: "integer overflow"}
	}
	for _, n := range []string{"Rem", "Rem32", "Rem64"} {
		pkgFuncs["math/bits."+n] = pkgFunc{cpp: "tsr::bits_" + n, panics: "integer divide by zero"}
	}
}

// pkgFuncOf finds the standard-library function that a call calls, if it
// is one that worker code can call.
func pkgFuncOf(info *types.Info, x *ast.CallExpr) (pkgFunc, bool) {
	sel, ok := ast.Unparen(x.Fun).(*ast.SelectorExpr)
	if !ok {
		return pkgFunc{}, false
	}
	id, ok := sel.X.(*ast.Ident)
	if !ok {
		return pkgFunc{}, false
	}
	pn, ok := info.Uses[id].(*types.PkgName)
	if !ok {
		return pkgFunc{}, false
	}
	f, ok := pkgFuncs[pn.Imported().Path()+"."+sel.Sel.Name]
	return f, ok
}

func (g *cgen) pkgCall(path, name string, x *ast.CallExpr) string {
	if path == TTPath {
		if s, ok := g.tensorCall(name, x); ok {
			return s
		}
	}
	if f, ok := pkgFuncs[path+"."+name]; ok && g.mode == deviceMode {
		var args []string
		if f.alloc {
			args = append(args, fmt.Sprint(g.oom(x)))
		}
		if f.panics != "" {
			args = append(args, fmt.Sprint(g.sh.position(x, f.panics)))
		}
		if f.panics2 != "" {
			args = append(args, fmt.Sprint(g.sh.position(x, f.panics2)))
		}
		for _, a := range x.Args {
			args = append(args, g.expr(a))
		}
		return f.cpp + "(" + strings.Join(args, ", ") + ")"
	}
	switch {
	case path == "strconv" && name == "Itoa" && g.mode == hostMode:
		return "std::to_string(" + g.expr(x.Args[0]) + ")"
	case path == TTPath && name == "WriteFile" && g.mode == hostMode:
		return "tsr::write_file(" + g.args(x.Args) + ")"
	case path == TTPath && name == "Cores" && g.mode == hostMode:
		return "tsr::cores(ctx)"
	case path == TTPath && (name == "NewLayer" || name == "NewMatrix"):
		g.fail(x, "tt.%s is for main: worker code has no floating point to make the values with", name)
	case path == TTPath && (name == "LoadBytes" || name == "OpenBytes" || name == "LoadDetector" || name == "Parallel"):
		g.fail(x, "tt.%s must be assigned to a variable in main: x := tt.%s(...)", name, name)
	}
	if g.mode == deviceMode {
		switch path {
		case "strings", "bytes", "unicode/utf8", "unicode", "math/bits":
			g.fail(x, "%s.%s is not available in worker code yet (see SPEC.md for the functions that are)", path, name)
		}
	}
	g.fail(x, "%s.%s is not supported %s", path, name, g.where())
	return ""
}

func (g *cgen) methodCall(recv types.Type, name, r string, x *ast.CallExpr) string {
	if isTT(recv, "Tensor") {
		if s, ok := g.tensorMethod(name, r, x); ok {
			return s
		}
	}
	switch {
	case isTT(recv, "Bytes") && name == "Len":
		return r + ".len"
	case isTT(recv, "Bytes") && name == "BytesRead" && g.mode == hostMode:
		return r + ".bytes_read()"
	case isTT(recv, "Bytes") && name == "Read" && g.mode == deviceMode:
		pos := g.sh.position(x, "negative offset in tt.Bytes.Read")
		return fmt.Sprintf("tsr::read(%s, %s, %s, %d)", r, g.expr(x.Args[0]), g.expr(x.Args[1]), pos)
	case isTT(recv, "Bytes") && name == "ReadStrided" && g.mode == deviceMode:
		posOff := g.sh.position(x, "negative offset in tt.Bytes.ReadStrided")
		posSize := g.sh.position(x, "tt.Bytes.ReadStrided: buffer length not a whole number of pieces")
		return fmt.Sprintf("tsr::read_strided(%s, %s, %s, %s, %s, %d, %d)", r, g.expr(x.Args[0]), g.expr(x.Args[1]),
			g.expr(x.Args[2]), g.expr(x.Args[3]), posOff, posSize)
	case isTT(recv, "Detector") && name == "Classify" && g.mode == deviceMode:
		pos := g.sh.position(x, "tt.Detector.Classify: the block must be BlockSize (4096) bytes")
		return fmt.Sprintf("tsr::classify(%s, %s, %d)", r, g.expr(x.Args[0]), pos)
	case isTT(recv, "Detector") && name == "ClassifyBatch" && g.mode == deviceMode:
		pos := g.sh.position(x, "tt.Detector.ClassifyBatch: blocks must hold len(labels) blocks of 4096 bytes")
		return fmt.Sprintf("tsr::classify_batch(%s, %s, %s, %d)", r, g.expr(x.Args[0]), g.expr(x.Args[1]), pos)
	case isTT(recv, "Detector") && name == "ClassName" && g.mode == hostMode:
		return r + ".class_name(" + g.expr(x.Args[0]) + ")"
	case isTT(recv, "Detector") && name == "Label" && g.mode == hostMode:
		return r + ".label(" + g.expr(x.Args[0]) + ")"
	case isTT(recv, "Detector") && name == "BlockSize":
		return "4096"
	}
	if p, ok := types.Unalias(recv).(*types.Pointer); ok && isTT(p.Elem(), "List") && name == "Append" && g.mode == deviceMode {
		return fmt.Sprintf("%s->append(%s, %d)", r, g.expr(x.Args[0]), g.sh.position(x, "append"))
	}
	g.fail(x, "%s.%s is not supported %s", types.TypeString(recv, nil), name, g.where())
	return ""
}

func arrayLen(t types.Type) (int64, types.Type, bool) {
	a, ok := types.Unalias(t).Underlying().(*types.Array)
	if !ok {
		return 0, nil, false
	}
	return a.Len(), a.Elem(), true
}

func (g *cgen) index(x *ast.IndexExpr) string {
	bt := g.typeOf(x.X)
	if _, ok := types.Unalias(bt).Underlying().(*types.Map); ok && g.mode == deviceMode {
		return fmt.Sprintf("tsr::map_get(%s, %s)", g.expr(x.X), g.exprAs(x.Index, types.Unalias(bt).Underlying().(*types.Map).Key()))
	}
	base, i := g.expr(x.X), g.expr(x.Index)
	if pt, ok := types.Unalias(bt).Underlying().(*types.Pointer); ok && isArray(pt.Elem()) {
		bt, base = pt.Elem(), fmt.Sprintf("tsr::deref_arr(%s, %d)", base, g.nilPos(x)) // p[i] for a pointer to an array
	}
	pos := g.sh.position(x, "index out of range")
	if n, _, ok := arrayLen(bt); ok {
		return fmt.Sprintf("tsr::at(%s, %d, %s, %d)", base, n, i, pos)
	}
	if _, ok := types.Unalias(bt).Underlying().(*types.Slice); ok {
		if g.mode == hostMode {
			return fmt.Sprintf("tsr::at(%s, %s, %d)", base, i, pos)
		}
		return fmt.Sprintf("%s.at(%s, %d)", base, i, pos)
	}
	if isString(bt) && g.mode == hostMode {
		return fmt.Sprintf("static_cast<uint8_t>(%s.at(%s))", base, i)
	}
	if isString(bt) {
		return fmt.Sprintf("%s.at(%s, %d)", base, i, pos)
	}
	g.fail(x, "indexing %s is not supported", bt)
	return ""
}

func (g *cgen) sliceExpr(x *ast.SliceExpr) string {
	if g.mode == hostMode {
		g.fail(x, "slicing is not supported in main")
	}
	bt := g.typeOf(x.X)
	base := g.expr(x.X)
	if pt, ok := types.Unalias(bt).Underlying().(*types.Pointer); ok && isArray(pt.Elem()) {
		bt, base = pt.Elem(), fmt.Sprintf("tsr::deref_arr(%s, %d)", base, g.nilPos(x)) // p[lo:hi] for a pointer to an array
	}
	lo := "0"
	if x.Low != nil {
		lo = g.expr(x.Low)
	}
	pos := g.sh.position(x, "slice bounds out of range")
	if n, _, ok := arrayLen(bt); ok {
		hi := fmt.Sprint(n)
		if x.High != nil {
			hi = g.expr(x.High)
		}
		if x.Slice3 {
			return fmt.Sprintf("tsr::slice3(%s, %d, %s, %s, %s, %d)", base, n, lo, hi, g.expr(x.Max), pos)
		}
		return fmt.Sprintf("tsr::slice(%s, %d, %s, %s, %d)", base, n, lo, hi, pos)
	}
	_, isSlice := types.Unalias(bt).Underlying().(*types.Slice)
	if isSlice || isString(bt) {
		switch {
		case x.Slice3:
			return fmt.Sprintf("%s.sub3(%s, %s, %s, %d)", base, lo, g.expr(x.High), g.expr(x.Max), pos)
		case x.High == nil:
			return fmt.Sprintf("%s.from(%s, %d)", base, lo, pos)
		}
		return fmt.Sprintf("%s.sub(%s, %s, %d)", base, lo, g.expr(x.High), pos)
	}
	g.fail(x, "slicing %s is not supported", bt)
	return ""
}

func (g *cgen) composite(x *ast.CompositeLit) string {
	t := g.typeOf(x)
	if pt, ok := types.Unalias(t).Underlying().(*types.Pointer); ok && g.mode == deviceMode {
		// {...} where a *T goes, as in []*T{{...}}: &T{...}
		return fmt.Sprintf("tsr::box<%s>(%s, %d)", g.ct(pt.Elem(), x), g.compositeOf(x, pt.Elem()), g.oom(x))
	}
	return g.compositeOf(x, t)
}

// compositeOf translates the composite literal x of type t.
func (g *cgen) compositeOf(x *ast.CompositeLit, t types.Type) string {
	if elem, ok := sliceElem(t); ok && g.mode == deviceMode {
		return g.sliceLit(x, elem)
	}
	if mt, ok := types.Unalias(t).Underlying().(*types.Map); ok && g.mode == deviceMode {
		// map[K]V{k: v, ...}: make the map, then set each entry.
		g.ct(mt, x) // a map type the card supports
		tmp := g.temp("lit")
		sets := []string{}
		for _, el := range x.Elts {
			kv := el.(*ast.KeyValueExpr)
			sets = append(sets, fmt.Sprintf("tsr::map_ref(%s, %s, 0, %d) = %s;", tmp, g.exprAs(kv.Key, mt.Key()), g.oom(x), g.exprAs(kv.Value, mt.Elem())))
		}
		return fmt.Sprintf("[&] { auto %s = tsr::make_map<%s, %s>(%d, %d); %s return %s; }()",
			tmp, g.ct(mt.Key(), x), g.ct(mt.Elem(), x), len(x.Elts), g.oom(x), strings.Join(sets, " "), tmp)
	}
	named, ok := types.Unalias(t).(*types.Named)
	st, ok2 := t.Underlying().(*types.Struct)
	if !ok || !ok2 {
		g.fail(x, "only struct literals are supported (not %s)", t)
	}
	vals := make([]string, st.NumFields())
	for i := range vals {
		ft := st.Field(i).Type()
		switch {
		case isArray(ft):
			vals[i] = "{}"
		case isPointer(ft):
			vals[i] = "static_cast<" + g.ct(ft, x) + ">(nullptr)"
		default:
			vals[i] = g.ct(ft, x) + "{}"
		}
	}
	var promoted []string // keys of fields promoted from embedded structs: set afterwards
	for i, el := range x.Elts {
		val := el
		if kv, ok := el.(*ast.KeyValueExpr); ok {
			val = kv.Value
		}
		if _, _, isArr := arrayLen(g.typeOf(val)); isArr {
			g.fail(val, "set the array field after making the struct: s.f = a, or copy into it")
		}
		if kv, ok := el.(*ast.KeyValueExpr); ok {
			name := kv.Key.(*ast.Ident).Name
			found := false
			for j := 0; j < st.NumFields(); j++ {
				if st.Field(j).Name() == name {
					vals[j] = g.exprAs(kv.Value, st.Field(j).Type())
					found = true
				}
			}
			if !found {
				obj, idx, _ := types.LookupFieldOrMethod(named, false, g.sh.pkg, name)
				f, ok := obj.(*types.Var)
				if !ok {
					g.fail(kv.Key, "unknown field %s", name)
				}
				path, t := "", types.Type(named)
				for _, k := range idx {
					if isPointer(t) {
						g.fail(kv.Key, "a field promoted through an embedded pointer cannot be set in a literal on the card")
					}
					fld := types.Unalias(t).Underlying().(*types.Struct).Field(k)
					path += "." + fieldName(fld)
					t = fld.Type()
				}
				promoted = append(promoted, fmt.Sprintf("v%s = %s;", path, g.exprAs(kv.Value, f.Type())))
			}
		} else {
			vals[i] = g.exprAs(el, st.Field(i).Type())
		}
	}
	lit := g.ct(named, x) + "{" + strings.Join(vals, ", ") + "}"
	if len(promoted) > 0 {
		return fmt.Sprintf("[&] { %s v = %s; %s return v; }()", g.ct(named, x), lit, strings.Join(promoted, " "))
	}
	return lit
}

// sliceLit translates a slice literal, []T{...}, which makes a new array on
// the heap.
func (g *cgen) sliceLit(x *ast.CompositeLit, elem types.Type) string {
	if _, _, isArr := arrayLen(elem); isArr {
		g.fail(x, "slices of arrays are not supported on the card; use a slice of structs")
	}
	ct := g.elemCT(elem, x)
	keyed := false
	for _, el := range x.Elts {
		if _, ok := el.(*ast.KeyValueExpr); ok {
			keyed = true
		}
	}
	if !keyed {
		if len(x.Elts) == 0 {
			return fmt.Sprintf("tsr::literal<%s>(%d)", ct, g.oom(x))
		}
		return fmt.Sprintf("tsr::literal<%s>(%d, %s)", ct, g.oom(x), g.argsAs(x.Elts, func(int) types.Type { return elem }))
	}
	// []T{i: v, ...}: make the slice, then set the elements given.
	tmp := g.temp("lit")
	var sets []string
	next, n := int64(0), int64(0)
	for _, el := range x.Elts {
		val := el
		if kv, ok := el.(*ast.KeyValueExpr); ok {
			k, _ := constant.Int64Val(constant.ToInt(g.sh.info.Types[kv.Key].Value))
			next, val = k, kv.Value
		}
		sets = append(sets, fmt.Sprintf("%s.ptr[%d] = %s;", tmp, next, g.exprAs(val, elem)))
		next++
		n = max(n, next)
	}
	return fmt.Sprintf("[&] { auto %s = tsr::make<%s>(%d, %d, %d); %s return %s; }()",
		tmp, ct, n, g.oom(x), g.oom(x), strings.Join(sets, " "), tmp)
}

// ------------------------------------------------------------------ statements

func (g *cgen) block(list []ast.Stmt) {
	g.indent++
	for _, s := range list {
		g.stmt(s)
	}
	g.indent--
}

func (g *cgen) stmt(s ast.Stmt) {
	switch s := s.(type) {
	case *ast.ExprStmt:
		g.line(g.expr(s.X) + ";")
	case *ast.AssignStmt:
		g.assign(s)
	case *ast.IncDecStmt:
		g.line(g.incDec(s) + ";")
	case *ast.DeclStmt:
		g.decl(s)
	case *ast.IfStmt:
		g.ifStmt(s, false)
	case *ast.ForStmt:
		g.forStmt(s, "")
	case *ast.RangeStmt:
		g.rangeStmt(s, "")
	case *ast.BlockStmt:
		g.line("{")
		g.block(s.List)
		g.line("}")
	case *ast.ReturnStmt:
		g.ret(s)
	case *ast.LabeledStmt:
		g.labeled(s)
	case *ast.BranchStmt:
		switch {
		case s.Tok == token.GOTO:
			g.line("goto " + label(s.Label.Name, "") + ";")
		case s.Tok == token.BREAK && s.Label != nil:
			g.line("goto " + label(s.Label.Name, "break") + ";")
		case s.Tok == token.CONTINUE && s.Label != nil:
			g.line("goto " + label(s.Label.Name, "continue") + ";")
		case s.Tok == token.BREAK:
			g.line("break;")
		case s.Tok == token.CONTINUE:
			g.line("continue;")
		default:
			g.fail(s, "fallthrough must be the last statement of a case")
		}
	case *ast.SwitchStmt:
		g.switchStmt(s)
	case *ast.TypeSwitchStmt:
		if g.mode != deviceMode {
			g.fail(s, "type switches are not supported %s", g.where())
		}
		g.typeSwitch(s)
	case *ast.EmptyStmt:
	case *ast.GoStmt:
		if g.sh.goFIDs == nil {
			g.fail(s, "start workers with tt.Parallel, which gives each one a core of its own")
		}
		g.goStmt(s)
	case *ast.SendStmt:
		g.send(s)
	case *ast.SelectStmt:
		g.selectStmt(s)
	case *ast.DeferStmt:
		if g.mode != deviceMode {
			g.fail(s, "defer is not supported %s", g.where())
		}
		g.deferStmt(s)
	default:
		g.fail(s, "this statement is not supported %s", g.where())
	}
}

// lvalue renders e as somewhere to store: m[k] of a map is its slot, made
// if k is not there yet.
func (g *cgen) lvalue(e ast.Expr) string {
	if s, ok := g.inOrder(e, true); ok {
		return s // a[f()][g()]: f, then g
	}
	if x, ok := ast.Unparen(e).(*ast.IndexExpr); ok && g.mode == deviceMode {
		if _, isMap := types.Unalias(g.typeOf(x.X)).Underlying().(*types.Map); isMap {
			return fmt.Sprintf("tsr::map_ref(%s, %s, %d, %d)", g.expr(x.X), g.exprAs(x.Index, types.Unalias(g.typeOf(x.X)).Underlying().(*types.Map).Key()),
				g.sh.position(x, "assignment to entry in nil map"), g.oom(x))
		}
	}
	return g.expr(e)
}

func isMapIndex(info *types.Info, e ast.Expr) bool {
	x, ok := ast.Unparen(e).(*ast.IndexExpr)
	if !ok {
		return false
	}
	_, isMap := types.Unalias(info.TypeOf(x.X)).Underlying().(*types.Map)
	return isMap
}

func (g *cgen) incDec(s *ast.IncDecStmt) string {
	l := g.lvalue(s.X)
	t := g.typeOf(s.X)
	if narrow(t) {
		op := "+"
		if s.Tok == token.DEC {
			op = "-"
		}
		if c, _ := g.calls(s.X); c {
			// a[f()]++ calls f once.
			at := g.temp("at")
			return fmt.Sprintf("[&](auto& %s) { %s = %s(%s %s 1); }(%s)", at, at, g.ct(t, s), at, op, l)
		}
		return l + " = " + g.ct(t, s) + "(" + l + " " + op + " 1)"
	}
	return l + s.Tok.String()
}

var assignOpTok = map[token.Token]token.Token{
	token.ADD_ASSIGN: token.ADD, token.SUB_ASSIGN: token.SUB, token.MUL_ASSIGN: token.MUL,
	token.QUO_ASSIGN: token.QUO, token.REM_ASSIGN: token.REM, token.AND_ASSIGN: token.AND,
	token.OR_ASSIGN: token.OR, token.XOR_ASSIGN: token.XOR, token.SHL_ASSIGN: token.SHL,
	token.SHR_ASSIGN: token.SHR, token.AND_NOT_ASSIGN: token.AND_NOT,
}

// assignExpr renders a single assignment without the semicolon, for use as a
// statement or as a for loop's post statement.
func (g *cgen) assignExpr(s *ast.AssignStmt) string {
	lhsCalls, _ := g.calls(s.Lhs[0])
	if op, ok := assignOpTok[s.Tok]; ok {
		l := g.lvalue(s.Lhs[0])
		t := g.typeOf(s.Lhs[0])
		if lhsCalls {
			// a[f()] += g(): f once, and before g. The place is the
			// lambda's argument, so it is found before the body runs.
			at := g.temp("at")
			return fmt.Sprintf("[&](auto& %s) { %s = %s; }(%s)", at, at, g.binary(op, at, g.expr(s.Rhs[0]), t, t, s.Rhs[0], s), l)
		}
		return l + " = " + g.binary(op, l, g.expr(s.Rhs[0]), t, t, s.Rhs[0], s)
	}
	if len(s.Lhs) != 1 || len(s.Rhs) != 1 {
		g.fail(s, "assign one value here")
	}
	if rhsCalls, _ := g.calls(s.Rhs[0]); lhsCalls && rhsCalls && s.Tok == token.ASSIGN {
		// a[f()] = g(): Go calls f first; C++ evaluates the right side
		// of = first.
		rhs := g.exprAs(s.Rhs[0], g.typeOf(s.Lhs[0]))
		if x, ok := ast.Unparen(s.Lhs[0]).(*ast.IndexExpr); ok && isMapIndex(g.sh.info, x) && g.mode == deviceMode {
			// The map and the key first; the slot (which adding the key
			// makes) after the value, as Go stores it after evaluating it.
			mt := types.Unalias(g.typeOf(x.X)).Underlying().(*types.Map)
			m, k := g.temp("m"), g.temp("k")
			return fmt.Sprintf("[&](auto %s, %s %s) { tsr::map_ref(%s, %s, %d, %d) = %s; }(%s, %s)",
				m, g.ct(mt.Key(), x), k, m, k, g.sh.position(x, "assignment to entry in nil map"), g.oom(x), rhs,
				g.expr(x.X), g.exprAs(x.Index, mt.Key()))
		}
		at := g.temp("at")
		return fmt.Sprintf("[&](auto& %s) { %s = %s; }(%s)", at, at, rhs, g.lvalue(s.Lhs[0]))
	}
	return g.assignOne(s.Lhs[0], g.exprAs(s.Rhs[0], g.typeOf(s.Lhs[0])), s.Tok == token.DEFINE)
}

// exprAs renders e where a value of type t is wanted: there, nil (which
// go/types leaves untyped) is t's nil.
func (g *cgen) exprAs(e ast.Expr, t types.Type) string {
	if t != nil && types.IsInterface(t) && g.mode == deviceMode {
		switch {
		case isNil(g.sh.info, e):
			return "tsr::Iface{}"
		case !types.IsInterface(g.typeOf(e)):
			return g.toIface(e) // a value goes into an interface
		}
		return g.expr(e)
	}
	if isNil(g.sh.info, e) {
		switch types.Unalias(t).Underlying().(type) {
		case *types.Chan:
			return g.ct(t, e) + "{}"
		case *types.Slice, *types.Map:
			if g.mode == deviceMode {
				return g.ct(t, e) + "{}"
			}
		case *types.Pointer:
			if g.mode == deviceMode {
				return "static_cast<" + g.ct(t, e) + ">(nullptr)"
			}
		case *types.Signature:
			if g.mode == deviceMode {
				return "tsr::Func{}"
			}
		}
	}
	return g.expr(e)
}

func (g *cgen) assignOne(lhs ast.Expr, rhs string, define bool) string {
	if id, ok := lhs.(*ast.Ident); ok {
		if id.Name == "_" {
			return "(void)(" + rhs + ")"
		}
		if obj := g.sh.info.Defs[id]; obj != nil && define {
			if _, _, isArr := arrayLen(obj.Type()); isArr {
				g.fail(id, "declare arrays with var, e.g. var %s [n]byte", id.Name)
			}
			return g.declare(obj, id, rhs)
		}
	}
	if _, _, isArr := arrayLen(g.typeOf(lhs)); isArr {
		g.fail(lhs, "assign whole arrays one at a time: a = b")
	}
	return g.lvalue(lhs) + " = " + rhs
}

func (g *cgen) assign(s *ast.AssignStmt) {
	if len(s.Lhs) == 2 && len(s.Rhs) == 1 && isMapIndex(g.sh.info, s.Rhs[0]) && g.mode == deviceMode {
		g.commaOkMap(s.Lhs, s.Rhs[0], s.Tok == token.DEFINE)
		return
	}
	if ta, ok := ast.Unparen(s.Rhs[0]).(*ast.TypeAssertExpr); ok && len(s.Lhs) == 2 && len(s.Rhs) == 1 && g.mode == deviceMode {
		g.commaOkAssert(s.Lhs, ta, s.Tok == token.DEFINE)
		return
	}
	if len(s.Lhs) == 2 && len(s.Rhs) == 1 {
		if u, ok := ast.Unparen(s.Rhs[0]).(*ast.UnaryExpr); ok && u.Op == token.ARROW {
			g.commaOkRecv(s, u)
			return
		}
	}
	if len(s.Lhs) > 1 && len(s.Rhs) == 1 {
		g.assignResults(s.Lhs, s.Rhs[0], s.Tok == token.DEFINE)
		return
	}
	if len(s.Lhs) == 1 && len(s.Rhs) == 1 {
		if lit, ok := ast.Unparen(s.Rhs[0]).(*ast.FuncLit); ok && g.mode == deviceMode && !g.sh.valueLit[lit] {
			id, isID := s.Lhs[0].(*ast.Ident)
			if !isID || s.Tok != token.DEFINE || g.sh.info.Defs[id] == nil {
				g.fail(s, "on the card a function literal can only start a new variable: f := func(...) {...}")
			}
			g.defineLambda(g.sh.info.Defs[id], lit)
			return
		}
	}
	if len(s.Lhs) == 1 && len(s.Rhs) == 1 && (s.Tok == token.ASSIGN || s.Tok == token.DEFINE) {
		if _, _, isArr := arrayLen(g.typeOf(s.Rhs[0])); isArr {
			g.assignArray(s.Lhs[0], s.Rhs[0], s.Tok == token.DEFINE)
			return
		}
	}
	if _, ok := assignOpTok[s.Tok]; ok || (len(s.Lhs) == 1 && len(s.Rhs) == 1) {
		g.line(g.assignExpr(s) + ";")
		return
	}
	// a, b = b, a. Go assigns in two phases: first the operands of the
	// index expressions on the left and then the right-hand sides are
	// evaluated, in order; then the values are stored, left to right. So in
	// i, m[i] = 1, 2 the map's key is the old i, and in s, s[1] = s[:1], 3
	// the element is the old slice's.
	targets := g.bindTargets(s.Lhs)
	tmps := make([]string, len(s.Rhs))
	for i, r := range s.Rhs {
		tmps[i] = g.temp("t")
		t := g.typeOf(r)
		if lt := g.typeOf(s.Lhs[i]); isNil(g.sh.info, r) || (lt != nil && types.IsInterface(lt)) {
			t = lt // nil, or a value that goes into an interface
		}
		g.line(g.ct(t, r) + " " + tmps[i] + " = " + g.exprAs(r, t) + ";")
	}
	for i, l := range s.Lhs {
		if targets[i] != "" {
			g.line(targets[i] + " = " + tmps[i] + ";")
			continue
		}
		g.line(g.assignOne(l, tmps[i], s.Tok == token.DEFINE) + ";")
	}
}

// bindTargets does the first phase of an assignment to several targets for
// the ones that index something: it evaluates their operands now, before
// the values to store, and returns what to store into (or "" for a target
// that needs nothing done first). An element of a slice or an array, or a
// field of one, is bound by reference; for a map, the map and the key are
// kept, and the slot is found when the value is stored (adding another key
// can move the slots).
func (g *cgen) bindTargets(lhs []ast.Expr) []string {
	targets := make([]string, len(lhs))
	if g.mode != deviceMode {
		return targets
	}
	for i, l := range lhs {
		if !hasIndex(l) { // first: _ has no type
			continue
		}
		if _, _, isArr := arrayLen(g.typeOf(l)); isArr {
			continue
		}
		if x, ok := ast.Unparen(l).(*ast.IndexExpr); ok && isMapIndex(g.sh.info, x) {
			mt := types.Unalias(g.typeOf(x.X)).Underlying().(*types.Map)
			m, k := g.temp("m"), g.temp("k")
			g.line("auto " + m + " = " + g.expr(x.X) + ";")
			g.line(g.ct(mt.Key(), x) + " " + k + " = " + g.exprAs(x.Index, mt.Key()) + ";")
			targets[i] = fmt.Sprintf("tsr::map_ref(%s, %s, %d, %d)", m, k,
				g.sh.position(x, "assignment to entry in nil map"), g.oom(x))
			continue
		}
		ref := g.temp("at")
		g.line("auto& " + ref + " = " + g.lvalue(l) + ";")
		targets[i] = ref
	}
	return targets
}

// hasIndex reports whether an assignment's target indexes something: s[i],
// m[k], a[i].f, ... (not a variable, or a field of one).
func hasIndex(e ast.Expr) bool {
	for {
		switch x := ast.Unparen(e).(type) {
		case *ast.IndexExpr:
			return true
		case *ast.SelectorExpr:
			e = x.X
		default:
			return false
		}
	}
}

// defineLambda translates f := func(...) {...} into a C++ lambda that
// captures by reference. Go would let f escape; on the card it can only be
// called, so the lambda never outlives what it captures.
func (g *cgen) defineLambda(obj types.Object, lit *ast.FuncLit) {
	text := g.lambda(lit)
	g.lambdas[obj] = lit
	g.line("auto " + g.name(obj) + " = " + text + ";")
}

// lambda renders a function literal as a C++ lambda expression.
func (g *cgen) lambda(lit *ast.FuncLit) string {
	sig := g.typeOf(lit).(*types.Signature)
	var params, boxes []string
	for i := 0; i < sig.Params().Len(); i++ {
		p := sig.Params().At(i)
		if _, _, ok := arrayLen(p.Type()); ok {
			g.fail(lit, "pass arrays as slices (a[:])")
		}
		name := g.temp("p")
		if p.Name() != "" && p.Name() != "_" {
			name = g.name(p)
		}
		ct := g.ct(p.Type(), lit)
		if g.heapVar(p) {
			// Its address outlives the call (escape.go).
			boxes = append(boxes, fmt.Sprintf("%s& %s = *tsr::box<%s>(%s_in_, %d);  // its address outlives the call", ct, name, ct, name, g.oom(lit)))
			name += "_in_"
		}
		params = append(params, ct+" "+name)
	}
	// Translate the body into its own buffer, as a function of its own.
	savedPre := g.bodyPre
	g.bodyPre = nil
	defer func() { g.bodyPre = savedPre }()
	saved, savedIndent := g.b, g.indent
	body, results, resType, named := g.body, g.results, g.resType, g.named
	defers, canRecover := g.defers, g.canRecover
	g.defers, g.canRecover = nil, false // its returns and recover are its own
	defer func() { g.defers, g.canRecover = defers, canRecover }()
	g.b = strings.Builder{}
	g.indent++
	for _, l := range boxes {
		g.line(l)
	}
	g.enter(lit.Body, sig.Results(), g.temp("lambda"), lit)
	g.indent--
	g.funcBody(lit.Body, lit)
	inner := g.b.String()
	res := g.resType
	g.b, g.indent = saved, savedIndent
	g.body, g.results, g.resType, g.named = body, results, resType, named
	return "[&](" + strings.Join(params, ", ") + ") -> " + res + " {\n" + inner + strings.Repeat("    ", g.indent) + "}"
}

// assignArray translates a = b for arrays, and a := b, which makes a new
// array in L1.
func (g *cgen) assignArray(lhs, rhs ast.Expr, define bool) {
	n, et, _ := arrayLen(g.typeOf(rhs))
	if id, ok := lhs.(*ast.Ident); ok && define && g.sh.info.Defs[id] != nil {
		g.newArray(g.sh.info.Defs[id], id)
	}
	if id, ok := lhs.(*ast.Ident); ok && id.Name == "_" {
		return
	}
	var target string
	if id, ok := lhs.(*ast.Ident); ok && g.sh.info.Defs[id] != nil {
		target = g.name(g.sh.info.Defs[id])
	} else {
		target = g.expr(lhs)
	}
	g.line(fmt.Sprintf("tsr::assign_array<%s>(%s, %s, %d);", g.elemCT(et, rhs), target, g.expr(rhs), n))
}

// newArray declares an array variable, in L1 on the card, set to zero.
func (g *cgen) newArray(obj types.Object, id *ast.Ident) {
	n, et, _ := arrayLen(obj.Type())
	ctype := g.elemCT(et, id)
	name := g.name(obj)
	if g.mode == hostMode {
		full := g.arrayCT(obj.Type(), id)
		k := strings.Index(full, "[")
		g.line(fmt.Sprintf("%s %s%s = {};", full[:k], name, full[k:]))
		return
	}
	if g.heapVar(obj) {
		// Its address outlives the call (escape.go).
		g.line(fmt.Sprintf("tsr::Ptr<%s> %s = tsr::alloc<%s>(%d, %d);  // %s on the heap", ctype, name, ctype, n, g.oom(id), obj.Type()))
		return
	}
	if g.recurse {
		// A function that recurses has an activation per call: its arrays
		// go on the stack (in L1, for programs that recurse).
		full := g.arrayCT(obj.Type(), id)
		k := strings.Index(full, "[")
		mem := name + "_mem_"
		g.line(fmt.Sprintf("%s %s%s;  // %s, on the stack", full[:k], mem, full[k:], obj.Type()))
		g.line(fmt.Sprintf("tsr::Ptr<%s> %s = %s;", ctype, name, mem))
		g.line(fmt.Sprintf("tsr::zero(%s, %d);", name, n))
		return
	}
	size := n * types.SizesFor("gc", "amd64").Sizeof(et)
	off := g.sh.allocScratch(size)
	g.line(fmt.Sprintf("tsr::Ptr<%s> %s = tsr::scratch<%s>(%d);  // %s in L1", ctype, name, ctype, off, obj.Type()))
	g.line(fmt.Sprintf("tsr::zero(%s, %d);", name, n))
}

func (g *cgen) decl(s *ast.DeclStmt) {
	gd := s.Decl.(*ast.GenDecl)
	switch gd.Tok {
	case token.CONST:
		return
	case token.VAR:
	default:
		g.fail(s, "local type declarations are not supported")
	}
	for _, spec := range gd.Specs {
		vs := spec.(*ast.ValueSpec)
		if len(vs.Names) > 1 && len(vs.Values) == 1 {
			lhs := make([]ast.Expr, len(vs.Names))
			for i, id := range vs.Names {
				lhs[i] = id
			}
			g.assignResults(lhs, vs.Values[0], true)
			continue
		}
		for i, id := range vs.Names {
			obj := g.sh.info.Defs[id]
			if obj == nil {
				continue
			}
			if len(vs.Values) > 0 && g.mode == deviceMode {
				if lit, ok := ast.Unparen(vs.Values[i]).(*ast.FuncLit); ok && !g.sh.valueLit[lit] {
					g.defineLambda(obj, lit)
					continue
				}
			}
			if _, _, ok := arrayLen(obj.Type()); ok {
				if len(vs.Values) > 0 {
					if _, isLit := ast.Unparen(vs.Values[i]).(*ast.CompositeLit); isLit {
						g.fail(vs, "array literals are not supported; declare the array and fill it")
					}
					g.assignArray(id, vs.Values[i], true)
					continue
				}
				g.newArray(obj, id)
				continue
			}
			if id.Name == "_" {
				continue
			}
			if len(vs.Values) > 0 {
				g.line(g.declare(obj, id, g.exprAs(vs.Values[i], obj.Type())) + ";")
			} else {
				g.line(g.declare(obj, id, "") + ";")
			}
		}
	}
}

func (g *cgen) ifStmt(s *ast.IfStmt, isElse bool) {
	prefix := "if ("
	if isElse {
		prefix = "} else if ("
	}
	if s.Init != nil {
		if isElse {
			g.line("} else {")
			g.indent++
		}
		g.line("{")
		g.indent++
		g.stmt(s.Init)
		g.ifStmt(&ast.IfStmt{If: s.If, Cond: s.Cond, Body: s.Body, Else: s.Else}, false)
		g.indent--
		g.line("}")
		if isElse {
			g.indent--
			g.line("}")
		}
		return
	}
	g.line(prefix + g.expr(s.Cond) + ") {")
	g.block(s.Body.List)
	switch el := s.Else.(type) {
	case nil:
		g.line("}")
	case *ast.BlockStmt:
		g.line("} else {")
		g.block(el.List)
		g.line("}")
	case *ast.IfStmt:
		if el.Init != nil {
			g.line("} else {")
			g.indent++
			g.ifStmt(el, false)
			g.indent--
			g.line("}")
		} else {
			g.ifStmt(el, true)
		}
	}
}

func (g *cgen) simple(s ast.Stmt) string {
	switch s := s.(type) {
	case nil:
		return ""
	case *ast.AssignStmt:
		if len(s.Lhs) > 1 && s.Tok == token.ASSIGN {
			// i, j = i+1, j-1 as a for loop's post statement: the
			// statement, in a lambda called at once.
			saved, indent := g.b, g.indent
			g.b, g.indent = strings.Builder{}, indent+1
			g.assign(s)
			body := g.b.String()
			g.b, g.indent = saved, indent
			return "[&]() {\n" + body + strings.Repeat("    ", indent) + "}()"
		}
		return g.assignExpr(s)
	case *ast.IncDecStmt:
		return g.incDec(s)
	case *ast.ExprStmt:
		return g.expr(s.X)
	}
	g.fail(s, "use a single assignment or ++ here")
	return ""
}

func (g *cgen) forStmt(s *ast.ForStmt, cont string) {
	if as, ok := s.Init.(*ast.AssignStmt); ok && as.Tok == token.DEFINE && g.mode == deviceMode {
		for _, l := range as.Lhs {
			if id, ok := l.(*ast.Ident); ok && g.heapVar(g.sh.info.Defs[id]) {
				g.forHeapVar(s, cont)
				return
			}
		}
	}
	if as, ok := s.Init.(*ast.AssignStmt); ok && len(as.Lhs) > 1 {
		// for i, j := 0, n; ...: the init in a block of its own, before
		// the loop.
		g.line("{")
		g.indent++
		g.stmt(s.Init)
		loop := *s
		loop.Init = nil
		g.forStmt(&loop, cont)
		g.indent--
		g.line("}")
		return
	}
	cond := "true"
	if s.Cond != nil {
		cond = g.expr(s.Cond)
	}
	if s.Init == nil && s.Post == nil {
		g.line("while (" + cond + ") {")
	} else {
		g.line("for (" + g.simple(s.Init) + "; " + cond + "; " + g.simple(s.Post) + ") {")
	}
	g.loopBody(s.Body, cont)
	g.line("}")
}

// forHeapVar translates a for loop whose variable lives on the heap. Go
// gives each turn its own variable, set before the post statement to the
// value of the last turn's: the loop counts with a plain variable, which
// each turn copies into a new variable on the heap, and copies back.
func (g *cgen) forHeapVar(s *ast.ForStmt, cont string) {
	as := s.Init.(*ast.AssignStmt)
	if len(as.Lhs) != 1 || len(as.Rhs) != 1 {
		g.fail(as, "a for loop whose variable's address outlives the turn can declare only that variable")
	}
	obj := g.sh.info.Defs[as.Lhs[0].(*ast.Ident)]
	ctype, real := g.ct(obj.Type(), as), g.name(obj)
	ctl, cur := g.temp("ctl"), g.temp("cur")
	g.line("{")
	g.indent++
	g.line(fmt.Sprintf("%s %s = %s;", ctype, ctl, g.exprAs(as.Rhs[0], obj.Type())))
	g.line(fmt.Sprintf("%s* %s = &%s;", ctype, cur, ctl))
	g.names[obj] = ctl
	cond, post := "true", ""
	if s.Cond != nil {
		cond = g.expr(s.Cond)
	}
	if s.Post != nil {
		post = g.simple(s.Post)
	}
	g.names[obj] = real
	post = fmt.Sprintf("(%s = *%s%s)", ctl, cur, map[bool]string{true: "", false: ", " + post}[post == ""])
	g.line(fmt.Sprintf("for (; %s; %s) {", cond, post))
	g.bodyPre = append(g.bodyPre,
		fmt.Sprintf("%s = tsr::box<%s>(%s, %d);", cur, ctype, ctl, g.oom(as)),
		fmt.Sprintf("%s& %s = *%s;  // this turn's own %s, on the heap", ctype, real, cur, obj.Name()))
	g.loopBody(s.Body, cont)
	g.line("}")
	g.indent--
	g.line("}")
}

// loopBody translates a loop's body. With a label for continue L, the body
// gets a block of its own, so that the goto does not jump over the body's
// declarations, and the label follows it.
//
// A loop whose turns allocate on the heap but keep nothing they allocate
// gives that memory back at the end of every turn.
func (g *cgen) loopBody(body *ast.BlockStmt, cont string) {
	mark := g.mode == deviceMode && g.mayAllocIn(body, map[*ast.FuncLit]bool{}) && !g.loopKeeps(body)
	pre := g.bodyPre
	g.bodyPre = nil
	if cont != "" {
		g.indent++
		g.line("{")
	}
	if mark {
		g.indent++
		g.line("tsr::HeapMark " + g.temp("mark") + ";  // nothing a turn allocates outlives it")
		g.indent--
	}
	g.indent++
	for _, l := range pre {
		g.line(l) // the turn's own variables on the heap
	}
	g.indent--
	g.block(body.List)
	if cont != "" {
		g.line("}")
		g.line(cont + ":;")
		g.indent--
	}
}

// ------------------------------------------------------------------ the heap

// allocates reports whether code allocates on the heap itself (not through
// the functions it calls): append, make, new, slice literals, &T{...},
// string concatenation, conversions between strings and slices, and
// variables that live on the heap.
func allocates(sh *cgShared, n ast.Node) bool {
	info := sh.info
	found := false
	eachConversion(info, n, func(_ ast.Expr, from, _ types.Type) {
		if types.SizesFor("gc", "amd64").Sizeof(from) > 8 {
			found = true // a copy on the heap for the interface value
		}
	})
	callees := map[ast.Node]bool{}
	ast.Inspect(n, func(n ast.Node) bool {
		switch x := n.(type) {
		case *ast.Ident:
			if v, ok := info.Defs[x].(*types.Var); ok && sh.heap[v] {
				found = true
			}
		case *ast.FuncLit:
			if sh.valueLit[x] && len(sh.captures[x]) > 0 {
				found = true // its closure, on the heap
			}
		case *ast.DeferStmt:
			found = true // the deferred call, on the heap
		case *ast.SelectorExpr:
			if s, ok := info.Selections[x]; ok && s.Kind() == types.MethodVal && !callees[x] {
				if m := s.Obj().(*types.Func); isIfaceMethod(m) || !pointerRecv(m) {
					found = true // a method value's copy of its receiver
				}
			}
		case *ast.UnaryExpr:
			if _, ok := ast.Unparen(x.X).(*ast.CompositeLit); ok && x.Op == token.AND {
				found = true
			}
		case *ast.CompositeLit:
			switch types.Unalias(info.TypeOf(x)).Underlying().(type) {
			case *types.Slice, *types.Map, *types.Pointer:
				found = true
			}
		case *ast.IncDecStmt:
			found = isMapIndex(info, x.X) // adding to a map can grow it
		case *ast.BinaryExpr:
			if x.Op == token.ADD && isString(info.TypeOf(x)) && info.Types[x].Value == nil {
				found = true
			}
		case *ast.AssignStmt:
			if x.Tok == token.ADD_ASSIGN && isString(info.TypeOf(x.Lhs[0])) {
				found = true
			}
			for _, l := range x.Lhs {
				found = found || isMapIndex(info, l)
			}
		case *ast.CallExpr:
			callees[ast.Unparen(x.Fun)] = true
			if tv := info.Types[x.Fun]; tv.IsType() && info.Types[x].Value == nil {
				to, from := tv.Type, info.TypeOf(x.Args[0])
				if isString(to) != isString(from) {
					found = true
				}
			}
			if id, ok := ast.Unparen(x.Fun).(*ast.Ident); ok {
				if b, ok := info.Uses[id].(*types.Builtin); ok && (b.Name() == "append" || b.Name() == "make" || b.Name() == "new") {
					found = true
				}
			}
			if f, ok := pkgFuncOf(info, x); ok && f.alloc {
				found = true
			}
			if sig, ok := types.Unalias(info.TypeOf(x.Fun)).Underlying().(*types.Signature); ok && sig.Variadic() && !x.Ellipsis.IsValid() && len(x.Args) >= sig.Params().Len() {
				found = true // the slice of the ... parameter
			}
		}
		return !found
	})
	return found
}

// computeMayAlloc finds the functions that can allocate on the heap,
// themselves or through the functions they call.
func computeMayAlloc(sh *cgShared, fns []*types.Func) {
	sh.mayAlloc = map[*types.Func]bool{}
	for _, f := range fns {
		d := sh.funcs[f]
		sh.mayAlloc[f] = allocates(sh, d.Body) || allocates(sh, d.Type) || (d.Recv != nil && allocates(sh, d.Recv))
	}
	for changed := true; changed; {
		changed = false
		for _, f := range fns {
			if sh.mayAlloc[f] {
				continue
			}
			for _, h := range callGraph(sh.info, sh.pkg, sh.funcs[f].Body) {
				if sh.mayAlloc[h] {
					sh.mayAlloc[f], changed = true, true
					break
				}
			}
		}
	}
}

// mayAllocIn reports whether code in the function being translated can
// allocate on the heap: itself, or through the functions and function
// literals it calls.
func (g *cgen) mayAllocIn(n ast.Node, seen map[*ast.FuncLit]bool) bool {
	if allocates(g.sh, n) {
		return true
	}
	found := false
	ast.Inspect(n, func(n ast.Node) bool {
		if id, ok := n.(*ast.Ident); ok {
			switch obj := g.sh.info.Uses[id].(type) {
			case *types.Func:
				found = found || g.sh.mayAlloc[obj]
			case *types.Var:
				if lit := g.lambdas[obj]; lit != nil && !seen[lit] {
					seen[lit] = true
					found = found || g.mayAllocIn(lit.Body, seen)
				}
			}
		}
		return !found
	})
	return found
}

// releases reports whether nothing that a call of f allocates can outlive
// the call: f's results hold no strings or slices, and neither its receiver
// nor its parameters, nor a package-level variable, let it store one where
// its caller would see it.
func (sh *cgShared) releases(f *types.Func) bool {
	if d := sh.funcs[f]; d != nil && hasDefer(d.Body) {
		return false // its deferred calls, and what recover gives, come after
	}
	sig := f.Type().(*types.Signature)
	for i := 0; i < sig.Results().Len(); i++ {
		if hasRef(sig.Results().At(i).Type()) {
			return false
		}
	}
	return !sh.stores(f, nil)
}

// stores reports whether a call of f can store a string or a slice where its
// caller sees it. outerRecv, if not nil, says whether the receiver is one.
func (sh *cgShared) stores(f *types.Func, outerRecv func() bool) bool {
	if sh.storesGlobal[f.Origin()] {
		return true
	}
	sig := f.Type().(*types.Signature)
	if recv := sig.Recv(); recv != nil {
		if p, ok := recv.Type().(*types.Pointer); ok {
			if hasRef(p.Elem()) && (outerRecv == nil || outerRecv()) {
				return true
			}
		} else if storesRef(recv.Type()) {
			return true
		}
	}
	for i := 0; i < sig.Params().Len(); i++ {
		if storesRef(sig.Params().At(i).Type()) {
			return true
		}
	}
	return false
}

// loopKeeps reports whether a turn of a loop with this body can keep
// something it allocated for later turns or for after the loop: whether it
// can store a string or a slice in a variable declared outside the body (or
// somewhere it cannot follow), or return one.
func (g *cgen) loopKeeps(body *ast.BlockStmt) bool {
	return g.keeps(body, func(p token.Pos) bool { return p >= body.Pos() && p < body.End() }, true, map[*ast.FuncLit]bool{})
}

func (g *cgen) keeps(root ast.Node, inside func(token.Pos) bool, returns bool, seen map[*ast.FuncLit]bool) bool {
	info := g.sh.info
	// outer reports whether storing into e can store outside the turn.
	outer := func(e ast.Expr) bool {
		for {
			switch x := ast.Unparen(e).(type) {
			case *ast.Ident:
				obj := info.Uses[x]
				if obj == nil {
					obj = info.Defs[x]
				}
				if obj == nil {
					return false
				}
				return g.refs[obj] || !inside(obj.Pos())
			case *ast.SelectorExpr:
				if isPointer(g.typeOf(x.X)) && !g.isRefIdent(x.X) {
					return true // through a pointer: anyone's
				}
				e = x.X
			case *ast.IndexExpr:
				if _, _, ok := arrayLen(g.typeOf(x.X)); !ok {
					return true // a slice's array may be anyone's (or a pointer's)
				}
				e = x.X
			default:
				return true
			}
		}
	}
	found := false
	check := func(e ast.Expr) {
		if e != nil && hasRef(g.typeOf(e)) && outer(e) {
			found = true
		}
	}
	ast.Inspect(root, func(n ast.Node) bool {
		if found {
			return false
		}
		switch x := n.(type) {
		case *ast.FuncLit:
			// A literal's code runs here; its returns come back here.
			lit := x
			found = g.keeps(lit.Body, func(p token.Pos) bool { return inside(p) || (p >= lit.Pos() && p < lit.End()) }, false, seen)
			return false
		case *ast.AssignStmt:
			for _, l := range x.Lhs {
				if id, ok := l.(*ast.Ident); ok && (id.Name == "_" || (x.Tok == token.DEFINE && info.Defs[id] != nil)) {
					continue
				}
				if isMapIndex(info, l) {
					found = true // a map can be anyone's, and grows into new memory
				}
				check(l)
			}
		case *ast.IncDecStmt:
			if isMapIndex(info, x.X) {
				found = true
			}
		case *ast.DeferStmt:
			found = true // the deferred call stays until the function returns
		case *ast.RangeStmt:
			if x.Tok == token.ASSIGN {
				check(x.Key)
				check(x.Value)
			}
		case *ast.ReturnStmt:
			for i := 0; returns && g.results != nil && i < g.results.Len(); i++ {
				if hasRef(g.results.At(i).Type()) {
					found = true
				}
			}
		case *ast.CallExpr:
			found = g.callKeeps(x, outer, inside, seen)
		}
		return !found
	})
	return found
}

// isRefIdent reports whether e is a pointer receiver held as a reference.
func (g *cgen) isRefIdent(e ast.Expr) bool {
	id, ok := ast.Unparen(e).(*ast.Ident)
	return ok && g.refs[g.sh.info.Uses[id]]
}

// callKeeps is keeps for one call: whether it can store a string or a slice
// outside the turn.
func (g *cgen) callKeeps(x *ast.CallExpr, outer func(ast.Expr) bool, inside func(token.Pos) bool, seen map[*ast.FuncLit]bool) bool {
	info := g.sh.info
	if info.Types[x.Fun].IsType() {
		return false
	}
	if _, ok := valueCall(info, x, g.sh.lambdaVar); ok {
		return true // a closure can store into its variables
	}
	switch fun := ast.Unparen(x.Fun).(type) {
	case *ast.Ident:
		switch obj := info.Uses[fun].(type) {
		case *types.Builtin:
			if obj.Name() == "append" || obj.Name() == "copy" {
				// They write into the first slice's array, which may be anyone's.
				elem, _ := sliceElem(g.typeOf(x.Args[0]))
				return elem != nil && hasRef(elem)
			}
		case *types.Var:
			if lit := g.lambdas[obj]; lit != nil && !seen[lit] {
				seen[lit] = true
				return g.keeps(lit.Body, func(p token.Pos) bool { return inside(p) || (p >= lit.Pos() && p < lit.End()) }, false, seen)
			}
		case *types.Func:
			return g.sh.stores(obj, nil)
		}
	case *ast.SelectorExpr:
		if sel, ok := info.Selections[fun]; ok && sel.Kind() == types.MethodVal {
			if m, ok := sel.Obj().(*types.Func); ok && m.Pkg() == g.sh.pkg {
				return g.sh.stores(m, func() bool { return (isPointer(g.typeOf(fun.X)) && !g.isRefIdent(fun.X)) || outer(fun.X) })
			}
		}
	}
	return false
}

// label is the C++ name of a Go label, or of the places that break L and
// continue L go to.
func label(name, what string) string {
	if what == "" {
		return "L_" + cident(name)
	}
	return "L_" + cident(name) + "_" + what
}

// labeled translates "L: statement". Only the labels that some goto, break
// or continue uses are written (an unused label is a warning, and kernel
// builds treat warnings as errors).
func (g *cgen) labeled(s *ast.LabeledStmt) {
	name := s.Label.Name
	uses := func(root ast.Node, tok token.Token) bool {
		found := false
		ast.Inspect(root, func(n ast.Node) bool {
			if b, ok := n.(*ast.BranchStmt); ok && b.Tok == tok && b.Label != nil && b.Label.Name == name {
				found = true
			}
			if _, ok := n.(*ast.FuncLit); ok && n != root {
				return false
			}
			return !found
		})
		return found
	}
	if g.body != nil && uses(g.body, token.GOTO) {
		g.line(label(name, "") + ":;")
	}
	cont := ""
	if uses(s.Stmt, token.CONTINUE) {
		cont = label(name, "continue")
	}
	switch st := s.Stmt.(type) {
	case *ast.ForStmt:
		g.forStmt(st, cont)
	case *ast.RangeStmt:
		g.rangeStmt(st, cont)
	default:
		g.stmt(s.Stmt)
	}
	if uses(s.Stmt, token.BREAK) {
		g.line(label(name, "break") + ":;")
	}
}

func (g *cgen) rangeStmt(s *ast.RangeStmt, cont string) {
	xt := types.Unalias(g.typeOf(s.X)).Underlying()
	key := ""
	keyType := "int64_t"
	declare := s.Tok == token.DEFINE
	bind := func(e ast.Expr) (string, bool) {
		if e == nil {
			return "", false
		}
		if id, ok := e.(*ast.Ident); ok && id.Name == "_" {
			return "", false
		}
		if declare {
			obj := g.sh.info.Defs[e.(*ast.Ident)]
			if g.heapVar(obj) {
				return g.heapBind(obj, e), true
			}
			return g.name(obj), true
		}
		return g.lvalue(e), false // for m[k] = range s stores into the map
	}
	k, kNew := bind(s.Key)
	v, vNew := bind(s.Value)
	if _, ok := xt.(*types.Chan); ok {
		// for v := range ch: receive until the channel is closed and drained.
		elem := g.chanElem(s.X)
		tmp := g.temp("v")
		g.line("{")
		g.indent++
		g.line(fmt.Sprintf("%s %s{};", elem, tmp))
		g.line(fmt.Sprintf("while (%s) {", g.recvInto(s.X, "&"+tmp)))
		g.indent++
		if k != "" {
			if kNew {
				g.line(fmt.Sprintf("%s %s = %s;", elem, k, tmp))
			} else {
				g.line(fmt.Sprintf("%s = %s;", k, tmp))
			}
		}
		g.indent--
		g.loopBody(s.Body, cont)
		g.line("}")
		g.indent--
		g.line("}")
		return
	}
	if b, ok := xt.(*types.Basic); ok && b.Info()&types.IsInteger != 0 {
		keyType = g.ct(g.typeOf(s.X), s)
		key = k
		if key == "" {
			key = g.temp("i")
		} else if !kNew {
			g.fail(s.Key, "range over an integer declares its variable with :=")
		}
		limit := g.temp("n")
		g.line(fmt.Sprintf("for (%s %s = 0, %s = %s; %s < %s; %s++) {", keyType, key, limit, g.expr(s.X), key, limit, key))
		g.loopBody(s.Body, cont)
		g.line("}")
		return
	}
	if b, ok := xt.(*types.Basic); ok && b.Info()&types.IsString != 0 && g.mode == deviceMode {
		g.rangeString(s, k, kNew, v, vNew, cont)
		return
	}
	if mt, ok := xt.(*types.Map); ok && g.mode == deviceMode {
		// for k, v := range m: see tsr::MapIter for what Go promises.
		kt, vt := g.ct(mt.Key(), s), g.ct(mt.Elem(), s)
		it, kk, vv := g.temp("it"), g.temp("k"), g.temp("v")
		g.line("{")
		g.indent++
		g.line(fmt.Sprintf("tsr::MapIter<%s, %s> %s(%s);", kt, vt, it, g.expr(s.X)))
		g.line(fmt.Sprintf("%s %s{};", kt, kk))
		g.line(fmt.Sprintf("%s %s{};", vt, vv))
		g.line(fmt.Sprintf("while (%s.next(%s, %s)) {", it, kk, vv))
		g.indent++
		for _, b := range [][3]string{{k, kt, kk}, {v, vt, vv}} {
			switch name, typ, val := b[0], b[1], b[2]; {
			case name == "":
			case declare:
				g.line(fmt.Sprintf("%s %s = %s;", typ, name, val))
			default:
				g.line(fmt.Sprintf("%s = %s;", name, val))
			}
		}
		g.indent--
		g.loopBody(s.Body, cont)
		g.line("}")
		g.indent--
		g.line("}")
		return
	}
	var length, elem string
	var elemType types.Type
	x := g.temp("x")
	arrayExpr := func() string { return g.expr(s.X) }
	if pt, ok := xt.(*types.Pointer); ok && isArray(pt.Elem()) {
		// A pointer to an array: Go goes through it only for the elements.
		xt = types.Unalias(pt.Elem()).Underlying()
		arrayExpr = func() string { return fmt.Sprintf("tsr::deref_arr(%s, %d)", g.expr(s.X), g.nilPos(s.X)) }
	}
	switch t := xt.(type) {
	case *types.Array:
		length, elemType = fmt.Sprint(t.Len()), t.Elem()
		if _, _, isArr := arrayLen(t.Elem()); isArr && v != "" {
			g.fail(s.Value, "range over an array of arrays with the index only, and use a[i]")
		}
		g.line("{")
		g.indent++
		if v != "" { // an unused variable is a warning, which kernel builds treat as an error
			g.line(fmt.Sprintf("tsr::Ptr<%s> %s = %s;", g.elemCT(t.Elem(), s), x, arrayExpr()))
		}
		elem = x + "[" + "%s" + "]"
	case *types.Slice:
		elemType = t.Elem()
		g.line("{")
		g.indent++
		if g.mode == hostMode {
			g.line(fmt.Sprintf("const auto& %s = %s;", x, g.expr(s.X)))
			length = "static_cast<int64_t>(" + x + ".size())"
		} else {
			g.line(fmt.Sprintf("const auto %s = %s;", x, g.expr(s.X)))
			length = x + ".len"
			x += ".ptr"
		}
		elem = x + "[%s]"
	default:
		g.fail(s, "range over %s is not supported", xt)
	}
	// The loop counts with a variable of its own: for i = range s (or
	// m[k] = range s) assigns i each turn, but the body changing i does
	// not change the turns, and after the loop i is the last index.
	key = k
	if !kNew {
		key = g.temp("i")
	}
	g.line(fmt.Sprintf("for (int64_t %s = 0; %s < %s; %s++) {", key, key, length, key))
	g.indent++
	if !kNew && k != "" {
		g.line(k + " = " + key + ";")
	}
	if v != "" {
		item := fmt.Sprintf(elem, key)
		if vNew {
			g.line(fmt.Sprintf("%s %s = %s;", g.ct(elemType, s), v, item))
		} else {
			g.line(fmt.Sprintf("%s = %s;", v, item))
		}
	}
	g.indent--
	g.loopBody(s.Body, cont)
	g.line("}")
	g.indent--
	g.line("}")
}

// rangeString translates for i, r := range s: the runes of s, decoded from
// UTF-8 as Go does, and where each starts.
func (g *cgen) rangeString(s *ast.RangeStmt, k string, kNew bool, v string, vNew bool, cont string) {
	str, at, width, r := g.temp("s"), g.temp("at"), g.temp("w"), g.temp("r")
	g.line("{")
	g.indent++
	g.line(fmt.Sprintf("const tsr::Str %s = %s;", str, g.expr(s.X)))
	g.line(fmt.Sprintf("for (int64_t %s = 0, %s = 0; %s < %s.len; %s += %s) {", at, width, at, str, at, width))
	g.indent++
	g.line(fmt.Sprintf("int32_t %s;", r))
	g.line(fmt.Sprintf("%s = tsr::decode_rune(%s, %s, %s);", width, str, at, r))
	switch {
	case k != "" && kNew:
		g.line(fmt.Sprintf("int64_t %s = %s;", k, at))
	case k != "":
		g.line(fmt.Sprintf("%s = %s;", k, at))
	}
	switch {
	case v != "" && vNew:
		g.line(fmt.Sprintf("int32_t %s = %s;", v, r))
	case v != "":
		g.line(fmt.Sprintf("%s = %s;", v, r))
	}
	g.indent--
	g.loopBody(s.Body, cont)
	g.line("}")
	g.indent--
	g.line("}")
}

// casesRunCode reports whether any case expression of a switch can call a
// function.
func (g *cgen) casesRunCode(s *ast.SwitchStmt) bool {
	for _, c := range s.Body.List {
		for _, e := range c.(*ast.CaseClause).List {
			if g.runsCode(e) {
				return true
			}
		}
	}
	return false
}

// switchStmt finds the case that matches in Go's order (an if / else if
// chain that sets its number) and then runs it in a C++ switch, where break
// and fallthrough mean what they mean in Go.
func (g *cgen) switchStmt(s *ast.SwitchStmt) {
	g.line("{")
	g.indent++
	if s.Init != nil {
		g.stmt(s.Init)
	}
	tag := ""
	if s.Tag != nil {
		tag = g.temp("tag")
		val := g.expr
		if g.mode == deviceMode && !g.casesRunCode(s) {
			// The tag is only compared: switch string(b) does not copy b,
			// unless a case can call something that changes b.
			val = g.compared
		}
		g.line(g.ct(g.typeOf(s.Tag), s.Tag) + " " + tag + " = " + val(s.Tag) + ";")
	}
	which := g.temp("case")
	def := -1
	for i, c := range s.Body.List {
		if c.(*ast.CaseClause).List == nil {
			def = i
		}
	}
	g.line(fmt.Sprintf("int %s = %d;", which, def))
	first := true
	for i, c := range s.Body.List {
		cc := c.(*ast.CaseClause)
		if cc.List == nil {
			continue
		}
		var conds []string
		for _, e := range cc.List {
			tt := g.typeOf(s.Tag)
			if tag != "" && isNil(g.sh.info, e) && g.mode == deviceMode {
				conds = append(conds, g.nilTest(tag, tt, e))
			} else if tag != "" && !types.IsInterface(tt) && types.IsInterface(g.typeOf(e)) && g.mode == deviceMode {
				// switch t { case i: }: t goes into an interface to be compared.
				id := g.typeID(tt, s.Tag)
				conds = append(conds, fmt.Sprintf("tsr::iface_equal(tsr::to_iface<%s>(%d, %s, %d), %s)", g.sh.ifaceCT[id-1], id, tag, g.oom(e), g.expr(e)))
			} else if tag != "" && types.IsInterface(tt) && g.mode == deviceMode {
				conds = append(conds, "tsr::iface_equal("+tag+", "+g.exprAs(e, g.typeOf(s.Tag))+")")
			} else if tag != "" {
				conds = append(conds, "("+tag+" == "+g.expr(e)+")")
			} else {
				conds = append(conds, g.expr(e))
			}
		}
		kw := "} else if ("
		if first {
			kw = "if ("
			first = false
		}
		g.line(kw + strings.Join(conds, " || ") + ") {")
		g.line(fmt.Sprintf("    %s = %d;", which, i))
	}
	if !first {
		g.line("}")
	}
	g.line("switch (" + which + ") {")
	for i, c := range s.Body.List {
		cc := c.(*ast.CaseClause)
		body, fall := cc.Body, false
		if n := len(body); n > 0 {
			if b, ok := body[n-1].(*ast.BranchStmt); ok && b.Tok == token.FALLTHROUGH {
				body, fall = body[:n-1], true
			}
		}
		g.line(fmt.Sprintf("case %d: {", i))
		g.block(body)
		g.line("}")
		if fall {
			g.line("[[fallthrough]];") // must come right before the next case label
		} else {
			g.line("break;")
		}
	}
	g.line("default:")
	g.line("    break;")
	g.line("}")
	g.indent--
	g.line("}")
}

// ------------------------------------------------------------------ functions

// signature renders a device function's C++ signature.
func (g *cgen) signature(f *types.Func, decl *ast.FuncDecl) string {
	sig := f.Type().(*types.Signature)
	res := g.resultType(funcName(f), sig.Results(), decl)
	var params []string
	if recv := sig.Recv(); recv != nil {
		name := g.temp("recv")
		if recv.Name() != "" && recv.Name() != "_" {
			name = g.name(recv)
		}
		if g.heapVar(recv) {
			name += "_in_" // boxed by deviceFunc
		}
		if p, ok := recv.Type().(*types.Pointer); ok && !g.sh.ptrRecv[f] {
			// A pointer receiver is a C++ reference, unless a call can pass
			// it nil (escape.go): the method changes the caller's value,
			// as in Go.
			g.refs[recv] = true
			params = append(params, g.ct(p.Elem(), decl)+"& "+name)
		} else {
			params = append(params, g.ct(recv.Type(), decl)+" "+name)
		}
	}
	for i := 0; i < sig.Params().Len(); i++ {
		p := sig.Params().At(i)
		if _, _, ok := arrayLen(p.Type()); ok {
			// A goroutine takes an array of channels by reference: the
			// copy main hands it lives on its heap.
			if _, n, _, isList := chanList(p.Type()); isList && g.sh.goFIDs != nil {
				if _, isGo := g.sh.goFIDs[f]; isGo {
					params = append(params, fmt.Sprintf("tsr::Chan (&%s)[%d]", g.name(p), n))
					continue
				}
			}
			g.fail(decl, "%s: pass arrays as slices (a[:])", f.Name())
		}
		name := g.name(p)
		if g.heapVar(p) {
			name += "_in_" // boxed by deviceFunc
		}
		params = append(params, g.ct(p.Type(), decl)+" "+name)
	}
	if g.sh.recovers(f) {
		params = append(params, recoverParam(false)) // false unless a deferred call calls it
	}
	return "static " + res + " " + funcName(f) + "(" + strings.Join(params, ", ") + ")"
}

// prototype renders a device function's prototype: its signature, with the
// default of can_recover_.
func (g *cgen) prototype(f *types.Func, decl *ast.FuncDecl) string {
	return strings.Replace(g.signature(f, decl), recoverParam(false)+")", recoverParam(true)+")", 1) + ";"
}

// deviceFunc translates one top-level function for the worker kernel.
func (g *cgen) deviceFunc(f *types.Func, decl *ast.FuncDecl) {
	g.recurse = g.sh.recursive[f]
	g.canRecover = g.sh.recovers(f)
	g.line(g.signature(f, decl) + " {")
	g.indent++
	switch {
	case g.recurse:
		pos := g.sh.position(decl.Name, fmt.Sprintf("stack overflow: %s recursed deeper than the worker's stack (%d KiB of L1) allows; tessc -stack gives it more",
			f.Name(), g.sh.stackKiB))
		g.line(fmt.Sprintf("tsr::stack_check(%d);  // %s recurses", pos, f.Name()))
	case len(callGraph(g.sh.info, g.sh.pkg, decl.Body)) > 0:
		// It calls others: a chain of calls can run the stack out. (Leaf
		// functions and the runtime's live in the spare kStackSpare.)
		g.line(fmt.Sprintf("tsr::stack_check(%d);", g.stackPos(decl.Name, f.Name())))
	}
	if g.sh.mayAlloc[f] && g.sh.releases(f) {
		g.line("tsr::HeapMark heap_mark_;  // nothing this call allocates outlives it")
	}
	// Parameters whose address outlives the call live on the heap.
	sig := f.Type().(*types.Signature)
	vars := []*types.Var{}
	if sig.Recv() != nil {
		vars = append(vars, sig.Recv())
	}
	for i := 0; i < sig.Params().Len(); i++ {
		vars = append(vars, sig.Params().At(i))
	}
	for _, v := range vars {
		if g.heapVar(v) {
			ctype := g.ct(v.Type(), decl)
			g.line(fmt.Sprintf("%s& %s = *tsr::box<%s>(%s_in_, %d);  // its address outlives the call", ctype, g.name(v), ctype, g.name(v), g.oom(decl)))
		}
	}
	g.enter(decl.Body, f.Type().(*types.Signature).Results(), funcName(f), decl)
	g.indent--
	if !g.funcBody(decl.Body, decl) && f.Type().(*types.Signature).Results().Len() > 0 {
		g.line("    __builtin_unreachable();")
	}
	g.line("}")
}

// stackPos is the position of a stack check in a function that does not
// recurse itself.
func (g *cgen) stackPos(n ast.Node, name string) int {
	if g.sh.l1Stack() {
		return g.sh.position(n, fmt.Sprintf("stack overflow: the calls into %s went deeper than the worker's stack (%d KiB of L1) allows; tessc -stack gives it more",
			name, g.sh.stackKiB))
	}
	return g.sh.position(n, fmt.Sprintf("stack overflow: the calls into %s went deeper than the core's stack (about 3 KB of its local memory) allows; tessc -l1stack runs the workers on a stack of %d KiB in L1",
		name, g.sh.stackKiB))
}

// l1Stack reports whether the workers run on a stack in L1: when they
// recurse or defer, when their calls can need more than the core's own
// stack has room for, or when tessc was told -l1stack. For a program with
// interfaces or function values, the need is that of all its functions
// merged into one (computeStackNeed): small ones stay on the core's own
// stack, which is faster.
func (s *cgShared) l1Stack() bool {
	return len(s.recursive) > 0 || len(s.litRecursive) > 0 || s.forceL1Stack || s.stackNeed > localStackBudget || s.usesDefer
}

// localStackBudget is the most stack that tessc lets a worker's calls need
// on the core's own stack. The core has about 2.2 KB of local memory for its
// stack (measured on Wormhole: kernel_main starts 96 bytes below the top of
// the 4 KB, and tt-metal's firmware and the kernel's globals take the bottom
// 1.8 KB). The budget leaves room for kernel_body's own frame, the canary,
// tsr::kStackSpare and estimates that come out low. A worker that can need
// more runs on a stack in L1; within the budget, the stack checks and the
// canary are there for what the estimate misses.
const localStackBudget = 1536

// frameBytes estimates the stack a call of the function with this body
// takes itself: its struct values and other locals (arrays declared with
// var live in L1, but in a function that recurses they are on the stack
// too), and 96 bytes for saved registers and spills. GCC may inline the
// functions it calls into it, merging their frames into its own; following
// the calls, as stackNeed does, counts those too.
func frameBytes(sh *cgShared, body ast.Node, recursive bool) int64 {
	sizes := types.SizesFor("gc", "amd64")
	bytes := int64(96)
	ast.Inspect(body, func(n ast.Node) bool {
		id, ok := n.(*ast.Ident)
		if !ok {
			return true
		}
		v, ok := sh.info.Defs[id].(*types.Var)
		if !ok || v.IsField() {
			return true
		}
		if _, _, isArr := arrayLen(v.Type()); isArr && !recursive {
			return true // in L1 scratch
		}
		if size := sizes.Sizeof(v.Type()); size > 16 {
			bytes += size
		}
		return true
	})
	// A slice, a string, a tt.Bytes or a struct passed to one of the
	// program's functions is copied for the call, and GCC can give every
	// call in the function a copy of its own.
	ast.Inspect(body, func(n ast.Node) bool {
		call, ok := n.(*ast.CallExpr)
		if !ok {
			return true
		}
		// Only the program's own functions: the runtime's take references.
		var id *ast.Ident
		switch f := call.Fun.(type) {
		case *ast.Ident:
			id = f
		case *ast.SelectorExpr:
			id = f.Sel
		}
		if id == nil {
			return true
		}
		if f, ok := sh.info.Uses[id].(*types.Func); !ok || f.Pkg() != sh.pkg {
			return true
		}
		for _, arg := range call.Args {
			tv, ok := sh.info.Types[arg]
			if !ok || tv.Type == nil {
				continue
			}
			switch tv.Type.Underlying().(type) {
			case *types.Slice, *types.Struct:
				bytes += min(48, sizes.Sizeof(tv.Type))
			case *types.Basic:
				if isString(tv.Type) && tv.Value == nil {
					bytes += 16
				}
			}
		}
		return true
	})
	return bytes
}

// computeStackNeed estimates the stack the deepest chain of calls from the
// worker (or from each goroutine function) can need. Recursion counts one
// call; it runs on a stack in L1 anyway.
func computeStackNeed(sh *cgShared, roots []ast.Node) int64 {
	need := map[*types.Func]int64{}
	visiting := map[*types.Func]bool{}
	var of func(body ast.Node, recursive bool) int64
	var fn func(f *types.Func) int64
	fn = func(f *types.Func) int64 {
		if n, ok := need[f]; ok {
			return n
		}
		if visiting[f] || sh.funcs[f] == nil {
			return 0
		}
		visiting[f] = true
		n := of(sh.funcs[f].Body, sh.recursive[f])
		visiting[f] = false
		need[f] = n
		return n
	}
	of = func(body ast.Node, recursive bool) int64 {
		deepest := int64(256) // what a leaf's calls into the runtime can take
		for _, g := range callGraph(sh.info, sh.pkg, body) {
			if isIfaceMethod(g) { // through an interface's dispatch function
				for m := range sh.viaIface {
					if m.Name() == g.Name() {
						deepest = max(deepest, fn(m)+64)
					}
				}
				continue
			}
			deepest = max(deepest, fn(g))
		}
		return frameBytes(sh, body, recursive) + deepest
	}
	most := int64(0)
	for _, r := range roots {
		most = max(most, of(r, false))
	}
	if sh.usesIface || sh.usesFuncs {
		// GCC can inline all of it into one function, whose frame then
		// holds every function's values at once.
		merged := int64(96)
		for _, r := range roots {
			merged += valueBytes(sh, r)
		}
		for f := range need {
			merged += valueBytes(sh, sh.funcs[f].Body)
		}
		most = max(most, merged+256)
	}
	return most
}

// valueBytes estimates what the values of a function with interfaces or
// function values take in a frame that GCC merged it into: its variables of
// 16 bytes and more, and the temporaries of the values that go into
// interfaces, of the calls that give one, and of the calls through
// interfaces and function values.
func valueBytes(sh *cgShared, body ast.Node) int64 {
	sizes := types.SizesFor("gc", "amd64")
	bytes := int64(0)
	eachConversion(sh.info, body, func(ast.Expr, types.Type, types.Type) { bytes += 16 })
	ast.Inspect(body, func(n ast.Node) bool {
		switch x := n.(type) {
		case *ast.Ident:
			if v, ok := sh.info.Defs[x].(*types.Var); ok && !v.IsField() && !isArray(v.Type()) {
				if size := sizes.Sizeof(v.Type()); size >= 16 {
					bytes += size
				}
			}
		case *ast.CallExpr:
			t := sh.info.TypeOf(x)
			if t != nil && types.IsInterface(t) {
				bytes += 16
			}
			if _, ok := valueCall(sh.info, x, sh.lambdaVar); ok {
				bytes += 32
			} else if sel, ok := ast.Unparen(x.Fun).(*ast.SelectorExpr); ok {
				if s, ok := sh.info.Selections[sel]; ok && s.Kind() == types.MethodVal {
					if m, ok := s.Obj().(*types.Func); ok && isIfaceMethod(m) {
						bytes += 32
					}
				}
			}
		case *ast.FuncLit:
			if sh.valueLit[x] {
				bytes += 16
			}
		case *ast.TypeAssertExpr, *ast.CompositeLit:
			if t := sh.info.TypeOf(x.(ast.Expr)); t != nil && !isArray(t) {
				bytes += min(sizes.Sizeof(t), 64)
			}
		}
		return true
	})
	return bytes
}

// receiver translates the operand x of x.f or x.m(), whose type is t. A
// pointer is allowed only as a pointer receiver, which is a reference.
func (g *cgen) receiver(x ast.Expr, t types.Type) string {
	if isPointer(t) {
		if id, ok := ast.Unparen(x).(*ast.Ident); ok && g.refs[g.sh.info.Uses[id]] {
			return g.name(g.sh.info.Uses[id]) // a pointer receiver, held as a reference
		}
		if g.mode == hostMode {
			g.fail(x, "pointers are not supported %s", g.where())
		}
		return fmt.Sprintf("tsr::deref(%s, %d)", g.expr(x), g.nilPos(x))
	}
	return g.expr(x)
}

// fieldPath follows the field indices of a selection from base, whose type
// is t: one field, or the embedded fields that lead to it, going through
// those that are pointers.
func (g *cgen) fieldPath(base string, t types.Type, path []int, n ast.Node) string {
	for k, i := range path {
		if p, ok := types.Unalias(t).Underlying().(*types.Pointer); ok {
			t = p.Elem()
			if k > 0 {
				base = fmt.Sprintf("tsr::deref(%s, %d)", base, g.nilPos(n))
			}
		}
		f := types.Unalias(t).Underlying().(*types.Struct).Field(i)
		base += "." + fieldName(f)
		t = f.Type()
	}
	return base
}

// resultType is the C++ type a function returns: void, its one result's
// type, or a struct that holds its results (name_results, with fields r0,
// r1, ...).
func (g *cgen) resultType(name string, t *types.Tuple, n ast.Node) string {
	switch t.Len() {
	case 0:
		return "void"
	case 1:
		return g.ct(t.At(0).Type(), n)
	}
	sname := name + "_results"
	if g.sh.resSeen == nil {
		g.sh.resSeen = map[string]bool{}
	}
	if !g.sh.resSeen[sname] {
		g.sh.resSeen[sname] = true
		var b strings.Builder
		fmt.Fprintf(&b, "struct %s {", sname)
		for i := 0; i < t.Len(); i++ {
			fmt.Fprintf(&b, " %s r%d;", g.ct(t.At(i).Type(), n), i)
		}
		b.WriteString(" };\n")
		g.sh.results = append(g.sh.results, b.String())
	}
	return "prog::" + sname
}

// enter starts translating a function body: it records what return needs
// and declares the named results.
func (g *cgen) enter(body *ast.BlockStmt, results *types.Tuple, name string, n ast.Node) {
	g.body, g.results, g.named = body, results, nil
	g.resType = g.resultType(name, results, n)
	for i := 0; i < results.Len(); i++ {
		v := results.At(i)
		if v.Name() == "" {
			break
		}
		cname := g.temp("result")
		if v.Name() != "_" {
			cname = g.name(v)
		}
		if g.heapVar(v) {
			g.line(fmt.Sprintf("%s& %s = *tsr::alloc<%s>(1, %d);", g.ct(v.Type(), n), cname, g.ct(v.Type(), n), g.oom(n)))
		} else {
			g.line(g.ct(v.Type(), n) + " " + cname + "{};")
		}
		g.named = append(g.named, cname)
	}
}

// ret translates a return statement.
func (g *cgen) ret(s *ast.ReturnStmt) {
	if g.defers != nil {
		g.deferRet(s) // the deferred calls first
		return
	}
	n := 0
	if g.results != nil {
		n = g.results.Len()
	}
	switch {
	case len(s.Results) == 0 && len(g.named) == 1:
		g.line("return " + g.named[0] + ";")
	case len(s.Results) == 0 && len(g.named) > 1:
		g.line("return " + g.resType + "{" + strings.Join(g.named, ", ") + "};")
	case len(s.Results) == 0:
		g.line("return;")
	case n == 1:
		g.line("return " + g.exprAs(s.Results[0], g.results.At(0).Type()) + ";")
	case len(s.Results) == n:
		g.line("return " + g.resType + "{" + g.argsAs(s.Results, func(i int) types.Type { return g.results.At(i).Type() }) + "};")
	default: // return f(), where f has the same results
		t := g.temp("r")
		parts := make([]string, n)
		for i := range parts {
			parts[i] = fmt.Sprintf("%s.r%d", t, i)
		}
		g.line("{")
		g.line("    auto " + t + " = " + g.expr(s.Results[0]) + ";")
		g.line("    return " + g.resType + "{" + strings.Join(parts, ", ") + "};")
		g.line("}")
	}
}

// commaOkMap translates v, ok := m[k] and v, ok = m[k].
func (g *cgen) commaOkMap(lhs []ast.Expr, e ast.Expr, define bool) {
	x := ast.Unparen(e).(*ast.IndexExpr)
	mt := types.Unalias(g.typeOf(x.X)).Underlying().(*types.Map)
	v, ok := g.temp("v"), g.temp("ok")
	g.line(fmt.Sprintf("%s %s{};", g.ct(mt.Elem(), x), v))
	g.line(fmt.Sprintf("bool %s = tsr::map_lookup(%s, %s, %s);", ok, g.expr(x.X), g.exprAs(x.Index, mt.Key()), v))
	g.line(g.assignOne(lhs[0], g.convertTo(v, mt.Elem(), lhs[0], e), define) + ";")
	g.line(g.assignOne(lhs[1], g.convertTo(ok, types.Typ[types.Bool], lhs[1], e), define) + ";")
}

// assignResults translates x, y := f() and x, y = f(). New variables are
// declared first, so that they outlive the block that holds the results.
func (g *cgen) assignResults(lhs []ast.Expr, call ast.Expr, define bool) {
	if len(lhs) == 2 && isMapIndex(g.sh.info, call) && g.mode == deviceMode {
		g.commaOkMap(lhs, call, define)
		return
	}
	if ta, ok := ast.Unparen(call).(*ast.TypeAssertExpr); ok && len(lhs) == 2 && g.mode == deviceMode {
		g.commaOkAssert(lhs, ta, define)
		return
	}
	if _, ok := g.typeOf(call).(*types.Tuple); !ok {
		g.fail(call, "assign as many values as there are variables")
	}
	targets := g.bindTargets(lhs) // i, a[i] = f(): a's element is the one before the call
	for i, l := range lhs {
		id, isID := l.(*ast.Ident)
		switch {
		case targets[i] != "":
		case isID && id.Name == "_":
		case isID && define && g.sh.info.Defs[id] != nil:
			obj := g.sh.info.Defs[id]
			if _, _, isArr := arrayLen(obj.Type()); isArr {
				g.fail(id, "functions cannot return arrays on the card")
			}
			targets[i] = g.name(obj)
			g.line(g.declare(obj, id, "") + ";")
		default:
			targets[i] = g.expr(l)
		}
	}
	used := false
	for _, target := range targets {
		used = used || target != ""
	}
	if !used { // _, _ = f(): GCC warns about a temporary that is never read
		g.line("(void)(" + g.expr(call) + ");")
		return
	}
	t := g.temp("r")
	g.line("{")
	g.line("    auto " + t + " = " + g.expr(call) + ";")
	for i, target := range targets {
		if target != "" {
			g.line(fmt.Sprintf("    %s = %s;", target, g.convertTo(fmt.Sprintf("%s.r%d", t, i), g.typeOf(call).(*types.Tuple).At(i).Type(), lhs[i], call)))
		}
	}
	g.line("}")
}

// structDefs renders the definitions of the struct types used.
func structDefs(sh *cgShared) string {
	var b strings.Builder
	sizes := types.SizesFor("gc", "amd64")
	g := newCgen(sh, deviceMode)
	pointing := false
	for _, t := range sh.structs {
		pointing = pointing || holdsPointer(t)
	}
	if pointing && len(sh.structs) > 1 {
		// Structs can point to each other.
		for _, t := range sh.structs {
			fmt.Fprintf(&b, "struct %s;\n", cident(t.Obj().Name()))
		}
		b.WriteString("\n")
	}
	for _, t := range sh.structs {
		st := t.Underlying().(*types.Struct)
		name := cident(t.Obj().Name())
		fmt.Fprintf(&b, "struct %s {\n", name)
		var same []string
		for i := 0; i < st.NumFields(); i++ {
			f := st.Field(i)
			fname := fieldName(f)
			if _, _, isArr := arrayLen(f.Type()); isArr {
				full := g.arrayCT(f.Type(), nil)
				k := strings.Index(full, "[")
				fmt.Fprintf(&b, "    %s %s%s;\n", full[:k], fname, full[k:])
			} else {
				fmt.Fprintf(&b, "    %s %s;\n", g.ct(f.Type(), nil), fname)
			}
			same = append(same, fmt.Sprintf("tsr::eq(%s, o.%s)", fname, fname))
		}
		if len(same) == 0 {
			same = []string{"true"}
		}
		if types.Comparable(t) { // not with slices in it
			// Go's == and != on structs.
			fmt.Fprintf(&b, "    bool operator==(const %s& o) const { return %s; }\n", name, strings.Join(same, " && "))
			fmt.Fprintf(&b, "    bool operator!=(const %s& o) const { return !(*this == o); }\n", name)
		}
		if size := sizes.Sizeof(t); size > 0 && holdsPointer(t) {
			// A pointer has 4 bytes on the card and 8 in Go: such a struct
			// is smaller here (it never goes to the host).
			fmt.Fprintf(&b, "};\nstatic_assert(sizeof(%s) <= %d, \"%s fits in its size in Go\");\n", name, size, t.Obj().Name())
			fmt.Fprintf(&b, "constexpr uint64_t go_size_(const %s*) { return %d; }  // for append's capacities\n\n", name, size)
		} else if size > 0 {
			fmt.Fprintf(&b, "};\nstatic_assert(sizeof(%s) == %d, \"%s must have the same layout on the host and the card\");\n\n",
				cident(t.Obj().Name()), size, t.Obj().Name())
		} else {
			// struct{}: 0 bytes in Go, 1 in C++. Its values hold nothing,
			// and it cannot be a field (useStruct) or a result (workers).
			fmt.Fprintf(&b, "};\nconstexpr uint64_t go_size_(const %s*) { return 0; }  // for append's capacities\n\n", name)
		}
	}
	for _, r := range sh.results {
		b.WriteString(r)
	}
	if len(sh.results) > 0 {
		b.WriteString("\n")
	}
	return b.String()
}

// holdsPointer reports whether a value of t holds a pointer itself (not
// through a slice or a map), which makes its C++ layout differ from Go's.
func holdsPointer(t types.Type) bool {
	switch u := types.Unalias(t).Underlying().(type) {
	case *types.Pointer:
		return true
	case *types.Array:
		return holdsPointer(u.Elem())
	case *types.Struct:
		for i := 0; i < u.NumFields(); i++ {
			if holdsPointer(u.Field(i).Type()) {
				return true
			}
		}
	}
	return false
}

// structHashes renders, for the kernel, a hash for each comparable struct
// type, so that it can be a map's key (tsr::hash_key_, found by
// argument-dependent lookup).
func structHashes(sh *cgShared) string {
	var b strings.Builder
	for _, t := range sh.structs {
		if !types.Comparable(t) {
			continue
		}
		st := t.Underlying().(*types.Struct)
		name := cident(t.Obj().Name())
		fmt.Fprintf(&b, "inline uint32_t hash_key_(const %s& v) {\n    using tsr::hash_key_;\n    uint32_t h = 0;\n", name)
		for i := 0; i < st.NumFields(); i++ {
			fmt.Fprintf(&b, "    h = tsr::hash_more(h, hash_key_(v.%s));\n", fieldName(st.Field(i)))
		}
		b.WriteString("    return h;\n}\n")
	}
	if b.Len() > 0 {
		b.WriteString("\n")
	}
	return b.String()
}

// callGraph returns the package functions each function calls directly.
func callGraph(info *types.Info, pkg *types.Package, body ast.Node) []*types.Func {
	seen := map[*types.Func]bool{}
	var out []*types.Func
	ast.Inspect(body, func(n ast.Node) bool {
		if id, ok := n.(*ast.Ident); ok {
			if f, ok := info.Uses[id].(*types.Func); ok && f.Pkg() == pkg && !seen[f] {
				seen[f] = true
				out = append(out, f)
			}
		}
		return true
	})
	sort.Slice(out, func(i, j int) bool { return out[i].Pos() < out[j].Pos() })
	return out
}
