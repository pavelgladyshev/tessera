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
// by the width or more (tsr::shl), division by zero (tsr::div). Signed
// overflow is the one difference: Go wraps, the kernels' C++ does not promise
// to.

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

	// The function being translated.
	body    *ast.BlockStmt                // for finding the gotos that use a label
	results *types.Tuple                  // its results
	resType string                        // the C++ struct that holds more than one result
	named   []string                      // C++ names of its named results
	refs    map[types.Object]bool         // pointer receivers, which are C++ references
	lambdas map[types.Object]*ast.FuncLit // local variables holding function literals
	recurse bool                          // the function is on a cycle of calls: its arrays go on the stack
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
		main tsr prog std ctx argc argv paths net kernel_main kernel_body worker index results status fid heap_mark_`) {
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

func (g *cgen) temp(prefix string) string {
	g.temps++
	return fmt.Sprintf("%s_%d_", prefix, g.temps)
}

// funcName is the C++ name of a function; a method of T is T_name.
func funcName(f *types.Func) string {
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
			case "List":
				return "tsr::List<" + g.ct(u.TypeArgs().At(0), n) + ">"
			}
			g.fail(n, "tt.%s cannot be used %s", obj.Name(), g.where())
		}
		if _, ok := u.Underlying().(*types.Struct); ok && obj.Pkg() == g.sh.pkg {
			g.useStruct(u, n)
			return "prog::" + cident(obj.Name())
		}
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
		g.fail(n, "pointers are not supported %s (except method receivers and *tt.List)", g.where())
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
		g.ct(u.Elem(), n) // the element type must be supported
		if hasRef(u.Elem()) {
			g.fail(n, "channels cannot carry strings, slices or maps: each goroutine's memory is its own core's")
		}
		if g.mode == hostMode {
			return "tsr::ChanRef"
		}
		return "tsr::Chan"
	case *types.Array:
		g.fail(n, "arrays can only be declared with var, indexed and sliced; pass a slice (a[:]) instead")
	case *types.Signature:
		g.fail(n, "function values cannot be passed around on the card; call a function literal where it is defined")
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
		if _, ok := ft.Underlying().(*types.Struct); ok && ft.Obj().Pkg() == g.sh.pkg {
			g.useStruct(ft, n)
			return true
		}
		return isInteger(ft) || g.fieldOK(ft.Underlying(), n)
	case *types.Array:
		return g.fieldOK(ft.Elem(), n)
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
// they hold a string, a slice or a map.
func hasRef(t types.Type) bool {
	switch u := types.Unalias(t).Underlying().(type) {
	case *types.Basic:
		return u.Info()&types.IsString != 0
	case *types.Slice, *types.Map:
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
			g.fail(n, "field %s.%s: struct fields must be integers, bools, strings, structs, or arrays, slices or maps of those", t.Obj().Name(), f.Name())
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
	tv := g.sh.info.Types[e]
	if tv.Value != nil {
		return g.constant(tv.Value, tv.Type, e)
	}
	switch x := e.(type) {
	case *ast.ParenExpr:
		return g.expr(x.X)
	case *ast.Ident:
		switch obj := g.sh.info.Uses[x].(type) {
		case *types.Var:
			if g.lambdas[obj] != nil {
				g.fail(x, "on the card a function value can only be called: %s(...)", x.Name)
			}
			return g.name(obj)
		case *types.Nil:
			switch types.Unalias(tv.Type).Underlying().(type) {
			case *types.Slice, *types.Map:
				if g.mode == deviceMode {
					return g.ct(tv.Type, x) + "{}" // a nil slice or map
				}
			}
			g.fail(x, "nil is not supported %s", g.where())
		}
		g.fail(x, "%s cannot be used as a value here", x.Name)
	case *ast.BinaryExpr:
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
			return g.fieldPath(g.receiver(x.X, sel.Recv()), sel.Recv(), sel.Index())
		}
		g.fail(x, "%s cannot be used as a value here", types.ExprString(x))
	case *ast.CompositeLit:
		return g.composite(x)
	case *ast.StarExpr:
		if id, ok := x.X.(*ast.Ident); ok && g.refs[g.sh.info.Uses[id]] {
			return g.name(g.sh.info.Uses[id]) // *p of a pointer receiver
		}
		g.fail(x, "pointers are not supported %s (except method receivers)", g.where())
	case *ast.FuncLit:
		g.fail(x, "on the card a function literal must be called at once, or assigned to a variable and called: f := func(...) {...}")
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
		e = "tsr::" + fn + "<" + g.ct(res, n) + ">(" + l + ", " + r + ")"
	case token.QUO, token.REM:
		if isInteger(res) {
			if cv := g.sh.info.Types[right].Value; cv == nil || constant.Sign(cv) == 0 {
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

// compared renders a string that is only compared: string(b) of a byte
// slice is then a view of b, as in Go, which does not copy it either.
func (g *cgen) compared(e ast.Expr) string {
	if call, ok := ast.Unparen(e).(*ast.CallExpr); ok && len(call.Args) == 1 && g.sh.info.Types[call].Value == nil {
		if tv := g.sh.info.Types[call.Fun]; tv.IsType() && isString(tv.Type) && isByteSlice(g.typeOf(call.Args[0])) {
			return "tsr::view(" + g.expr(call.Args[0]) + ")"
		}
	}
	return g.expr(e)
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
		g.fail(x, "taking addresses is not supported %s", g.where())
	case token.ARROW:
		return g.recvValue(x)
	}
	g.fail(x, "operator %s is not supported", x.Op)
	return ""
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
		g.fail(s, "main cannot send on channels yet; pass values to the goroutines as arguments")
	}
	pos := g.sh.position(s, "send on a closed channel")
	g.line(fmt.Sprintf("tsr::send<%s>(%s, %s, %d);", g.chanElem(s.Chan), g.expr(s.Chan), g.expr(s.Value), pos))
}

// commaOkRecv handles v, ok := <-ch and v, ok = <-ch.
func (g *cgen) commaOkRecv(s *ast.AssignStmt, recv *ast.UnaryExpr) {
	elem := g.chanElem(recv.X)
	tmp := g.temp("v")
	okTmp := g.temp("ok")
	g.line(fmt.Sprintf("%s %s{};", elem, tmp))
	g.line(fmt.Sprintf("bool %s = %s;", okTmp, g.recvInto(recv.X, "&"+tmp)))
	g.line(g.assignOne(s.Lhs[0], tmp, s.Tok == token.DEFINE) + ";")
	g.line(g.assignOne(s.Lhs[1], okTmp, s.Tok == token.DEFINE) + ";")
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
			if c, ok := types.Unalias(pt).Underlying().(*types.Chan); ok {
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
	g.line(fmt.Sprintf("net.go(%d, %s, {%s});", fid, cstring(f.Name()), strings.Join(args, ", ")))
}

func (g *cgen) args(list []ast.Expr) string {
	parts := make([]string, len(list))
	for i, a := range list {
		parts[i] = g.expr(a)
	}
	return strings.Join(parts, ", ")
}

func (g *cgen) call(x *ast.CallExpr) string {
	if len(x.Args) == 1 {
		if t, ok := g.typeOf(x.Args[0]).(*types.Tuple); ok && t.Len() > 1 {
			g.fail(x, "pass the results of %s one at a time", types.ExprString(x.Args[0]))
		}
	}
	if tv := g.sh.info.Types[x.Fun]; tv.IsType() {
		if g.mode == deviceMode {
			if s, ok := g.conversion(tv.Type, x); ok {
				return s
			}
		} else if isString(tv.Type) || isString(g.typeOf(x.Args[0])) {
			g.fail(x, "string conversions are not supported; use strconv.Itoa")
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
				return "prog::" + funcName(obj) + "(" + g.args(x.Args) + ")"
			}
		}
	case *ast.SelectorExpr:
		if id, ok := fun.X.(*ast.Ident); ok {
			if pn, ok := g.sh.info.Uses[id].(*types.PkgName); ok {
				return g.pkgCall(pn.Imported().Path(), fun.Sel.Name, x)
			}
		}
		if sel, ok := g.sh.info.Selections[fun]; ok && sel.Kind() == types.MethodVal {
			if m := sel.Obj().(*types.Func); m.Pkg() == g.sh.pkg {
				if g.mode == hostMode {
					g.fail(x, "main can call only tt functions and builtins; %s runs on the card, so call it from the tt.Parallel worker", m.Name())
				}
				// The receiver, found through embedded fields; a pointer
				// receiver takes it by reference.
				idx := sel.Index()
				r := g.fieldPath(g.receiver(fun.X, sel.Recv()), sel.Recv(), idx[:len(idx)-1])
				args := []string{r}
				if len(x.Args) > 0 {
					args = append(args, g.args(x.Args))
				}
				return "prog::" + funcName(m) + "(" + strings.Join(args, ", ") + ")"
			}
			return g.methodCall(sel.Recv(), fun.Sel.Name, g.expr(fun.X), x)
		}
	}
	g.fail(x, "this call is not supported %s", g.where())
	return ""
}

// conversion translates the conversions that involve strings in worker
// code; ok is false for the others, which are C++ casts.
func (g *cgen) conversion(to types.Type, x *ast.CallExpr) (string, bool) {
	arg := x.Args[0]
	from := g.typeOf(arg)
	switch {
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
		return fmt.Sprintf("tsr::map_delete(%s, %s)", g.expr(x.Args[0]), g.expr(x.Args[1]))
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
		pos := g.sh.position(x, "panic("+types.ExprString(x.Args[0])+")")
		return fmt.Sprintf("tsr::fail(%d, 0)", pos)
	case "close":
		if g.mode == hostMode {
			g.fail(x, "a channel is closed by the goroutine that sends on it")
		}
		pos := g.sh.position(x, "close of a closed channel")
		return fmt.Sprintf("tsr::close(%s, %d)", g.expr(x.Args[0]), pos)
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
			rng := g.sh.position(x, "makeslice: len or cap out of range")
			if len(x.Args) == 2 {
				return fmt.Sprintf("tsr::make<%s>(%s, %d, %d)", ct, g.expr(x.Args[1]), rng, g.oom(x))
			}
			return fmt.Sprintf("tsr::make<%s>(%s, %s, %d, %d)", ct, g.expr(x.Args[1]), g.expr(x.Args[2]), rng, g.oom(x))
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
	return fmt.Sprintf("tsr::append<%s>(%d, %s, %s)", noscan(elem), g.oom(x), s, g.args(x.Args[1:]))
}

// pkgFunc is a function of the standard library that worker code can call:
// the runtime's C++ for it (compiler/runtime/text_device.h), which takes
// first the position for "out of memory" if it allocates, then the position
// for what else can go wrong (panics), then Go's arguments.
type pkgFunc struct {
	cpp    string
	alloc  bool
	panics string
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
	if f, ok := pkgFuncs[path+"."+name]; ok && g.mode == deviceMode {
		var args []string
		if f.alloc {
			args = append(args, fmt.Sprint(g.oom(x)))
		}
		if f.panics != "" {
			args = append(args, fmt.Sprint(g.sh.position(x, f.panics)))
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
	case path == TTPath && (name == "LoadBytes" || name == "OpenBytes" || name == "LoadDetector" || name == "Parallel"):
		g.fail(x, "tt.%s must be assigned to a variable in main: x := tt.%s(...)", name, name)
	}
	if g.mode == deviceMode {
		switch path {
		case "strings", "bytes", "unicode/utf8", "unicode":
			g.fail(x, "%s.%s is not available in worker code yet (see SPEC.md for the functions that are)", path, name)
		}
	}
	g.fail(x, "%s.%s is not supported %s", path, name, g.where())
	return ""
}

func (g *cgen) methodCall(recv types.Type, name, r string, x *ast.CallExpr) string {
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
		return fmt.Sprintf("tsr::map_get(%s, %s)", g.expr(x.X), g.expr(x.Index))
	}
	base, i := g.expr(x.X), g.expr(x.Index)
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
			sets = append(sets, fmt.Sprintf("tsr::map_ref(%s, %s, 0, %d) = %s;", tmp, g.expr(kv.Key), g.oom(x), g.expr(kv.Value)))
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
		vals[i] = "{}"
		if _, _, isArr := arrayLen(st.Field(i).Type()); !isArr {
			vals[i] = g.ct(st.Field(i).Type(), x) + "{}"
		}
	}
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
			for j := 0; j < st.NumFields(); j++ {
				if st.Field(j).Name() == name {
					vals[j] = g.expr(kv.Value)
				}
			}
		} else {
			vals[i] = g.expr(el)
		}
	}
	return g.ct(named, x) + "{" + strings.Join(vals, ", ") + "}"
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
		return fmt.Sprintf("tsr::literal<%s>(%d, %s)", ct, g.oom(x), g.args(x.Elts))
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
		sets = append(sets, fmt.Sprintf("%s.ptr[%d] = %s;", tmp, next, g.expr(val)))
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
	case *ast.EmptyStmt:
	case *ast.GoStmt:
		if g.sh.goFIDs == nil {
			g.fail(s, "start workers with tt.Parallel, which gives each one a core of its own")
		}
		g.goStmt(s)
	case *ast.SendStmt:
		g.send(s)
	case *ast.SelectStmt:
		g.fail(s, "select is not supported yet")
	default:
		g.fail(s, "this statement is not supported %s", g.where())
	}
}

// lvalue renders e as somewhere to store: m[k] of a map is its slot, made
// if k is not there yet.
func (g *cgen) lvalue(e ast.Expr) string {
	if x, ok := ast.Unparen(e).(*ast.IndexExpr); ok && g.mode == deviceMode {
		if _, isMap := types.Unalias(g.typeOf(x.X)).Underlying().(*types.Map); isMap {
			return fmt.Sprintf("tsr::map_ref(%s, %s, %d, %d)", g.expr(x.X), g.expr(x.Index),
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
	if op, ok := assignOpTok[s.Tok]; ok {
		l := g.lvalue(s.Lhs[0])
		t := g.typeOf(s.Lhs[0])
		return l + " = " + g.binary(op, l, g.expr(s.Rhs[0]), t, t, s.Rhs[0], s)
	}
	if len(s.Lhs) != 1 || len(s.Rhs) != 1 {
		g.fail(s, "assign one value here")
	}
	return g.assignOne(s.Lhs[0], g.expr(s.Rhs[0]), s.Tok == token.DEFINE)
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
			return g.ct(obj.Type(), id) + " " + g.name(obj) + " = " + rhs
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
		if lit, ok := ast.Unparen(s.Rhs[0]).(*ast.FuncLit); ok && g.mode == deviceMode {
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
	// a, b = b, a: evaluate every right-hand side first.
	tmps := make([]string, len(s.Rhs))
	for i, r := range s.Rhs {
		tmps[i] = g.temp("t")
		g.line(g.ct(g.typeOf(r), r) + " " + tmps[i] + " = " + g.expr(r) + ";")
	}
	for i, l := range s.Lhs {
		g.line(g.assignOne(l, tmps[i], s.Tok == token.DEFINE) + ";")
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
	var params []string
	for i := 0; i < sig.Params().Len(); i++ {
		p := sig.Params().At(i)
		if _, _, ok := arrayLen(p.Type()); ok {
			g.fail(lit, "pass arrays as slices (a[:])")
		}
		name := g.temp("p")
		if p.Name() != "" && p.Name() != "_" {
			name = g.name(p)
		}
		params = append(params, g.ct(p.Type(), lit)+" "+name)
	}
	// Translate the body into its own buffer, as a function of its own.
	saved, savedIndent := g.b, g.indent
	body, results, resType, named := g.body, g.results, g.resType, g.named
	g.b = strings.Builder{}
	g.indent++
	g.enter(lit.Body, sig.Results(), g.temp("lambda"), lit)
	g.indent--
	g.block(lit.Body.List)
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
				if lit, ok := ast.Unparen(vs.Values[i]).(*ast.FuncLit); ok {
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
				g.line(g.ct(obj.Type(), id) + " " + g.name(obj) + " = " + g.expr(vs.Values[i]) + ";")
			} else {
				g.line(g.ct(obj.Type(), id) + " " + g.name(obj) + "{};")
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

// loopBody translates a loop's body. With a label for continue L, the body
// gets a block of its own, so that the goto does not jump over the body's
// declarations, and the label follows it.
//
// A loop whose turns allocate on the heap but keep nothing they allocate
// gives that memory back at the end of every turn.
func (g *cgen) loopBody(body *ast.BlockStmt, cont string) {
	mark := g.mode == deviceMode && g.mayAllocIn(body, map[*ast.FuncLit]bool{}) && !g.loopKeeps(body)
	if cont != "" {
		g.indent++
		g.line("{")
	}
	if mark {
		g.indent++
		g.line("tsr::HeapMark " + g.temp("mark") + ";  // nothing a turn allocates outlives it")
		g.indent--
	}
	g.block(body.List)
	if cont != "" {
		g.line("}")
		g.line(cont + ":;")
		g.indent--
	}
}

// ------------------------------------------------------------------ the heap

// allocates reports whether code allocates on the heap itself (not through
// the functions it calls): append, make, slice literals, string
// concatenation and conversions between strings and slices.
func allocates(info *types.Info, n ast.Node) bool {
	found := false
	ast.Inspect(n, func(n ast.Node) bool {
		switch x := n.(type) {
		case *ast.CompositeLit:
			switch types.Unalias(info.TypeOf(x)).Underlying().(type) {
			case *types.Slice, *types.Map:
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
			if tv := info.Types[x.Fun]; tv.IsType() && info.Types[x].Value == nil {
				to, from := tv.Type, info.TypeOf(x.Args[0])
				if isString(to) != isString(from) {
					found = true
				}
			}
			if id, ok := ast.Unparen(x.Fun).(*ast.Ident); ok {
				if b, ok := info.Uses[id].(*types.Builtin); ok && (b.Name() == "append" || b.Name() == "make") {
					found = true
				}
			}
			if f, ok := pkgFuncOf(info, x); ok && f.alloc {
				found = true
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
		sh.mayAlloc[f] = allocates(sh.info, sh.funcs[f].Body)
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
	if allocates(g.sh.info, n) {
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
// nor its parameters let it store one where its caller would see it.
func releases(f *types.Func) bool {
	sig := f.Type().(*types.Signature)
	for i := 0; i < sig.Results().Len(); i++ {
		if hasRef(sig.Results().At(i).Type()) {
			return false
		}
	}
	return !stores(f, nil)
}

// stores reports whether a call of f can store a string or a slice where its
// caller sees it. outerRecv, if not nil, says whether the receiver is one.
func stores(f *types.Func, outerRecv func() bool) bool {
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
				e = x.X
			case *ast.IndexExpr:
				if _, _, ok := arrayLen(g.typeOf(x.X)); !ok {
					return true // a slice's array may be anyone's
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

// callKeeps is keeps for one call: whether it can store a string or a slice
// outside the turn.
func (g *cgen) callKeeps(x *ast.CallExpr, outer func(ast.Expr) bool, inside func(token.Pos) bool, seen map[*ast.FuncLit]bool) bool {
	info := g.sh.info
	if info.Types[x.Fun].IsType() {
		return false
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
			return stores(obj, nil)
		}
	case *ast.SelectorExpr:
		if sel, ok := info.Selections[fun]; ok && sel.Kind() == types.MethodVal {
			if m, ok := sel.Obj().(*types.Func); ok && m.Pkg() == g.sh.pkg {
				return stores(m, func() bool { return outer(fun.X) })
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
			return g.name(g.sh.info.Defs[e.(*ast.Ident)]), true
		}
		return g.expr(e), false
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
	switch t := xt.(type) {
	case *types.Array:
		length, elemType = fmt.Sprint(t.Len()), t.Elem()
		if _, _, isArr := arrayLen(t.Elem()); isArr && v != "" {
			g.fail(s.Value, "range over an array of arrays with the index only, and use a[i]")
		}
		g.line("{")
		g.indent++
		if v != "" { // an unused variable is a warning, which kernel builds treat as an error
			g.line(fmt.Sprintf("tsr::Ptr<%s> %s = %s;", g.elemCT(t.Elem(), s), x, g.expr(s.X)))
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
	key = k
	if key == "" {
		key = g.temp("i")
	}
	kDecl := ""
	if kNew || k == "" {
		kDecl = "int64_t "
	}
	if !kNew && k != "" {
		g.line(fmt.Sprintf("for (%s = 0; %s < %s; %s++) {", key, key, length, key))
	} else {
		g.line(fmt.Sprintf("for (%s%s = 0; %s < %s; %s++) {", kDecl, key, key, length, key))
	}
	g.indent++
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
		if g.mode == deviceMode {
			val = g.compared // the tag is only compared: switch string(b) does not copy b
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
			if tag != "" {
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
		if p, ok := recv.Type().(*types.Pointer); ok {
			// A pointer receiver is a C++ reference: the method changes the
			// caller's value, as in Go, without pointers on the card.
			g.refs[recv] = true
			params = append(params, g.ct(p.Elem(), decl)+"& "+name)
		} else {
			params = append(params, g.ct(recv.Type(), decl)+" "+name)
		}
	}
	for i := 0; i < sig.Params().Len(); i++ {
		p := sig.Params().At(i)
		if _, _, ok := arrayLen(p.Type()); ok {
			g.fail(decl, "%s: pass arrays as slices (a[:])", f.Name())
		}
		params = append(params, g.ct(p.Type(), decl)+" "+g.name(p))
	}
	return "static " + res + " " + funcName(f) + "(" + strings.Join(params, ", ") + ")"
}

// deviceFunc translates one top-level function for the worker kernel.
func (g *cgen) deviceFunc(f *types.Func, decl *ast.FuncDecl) {
	g.recurse = g.sh.recursive[f]
	g.line(g.signature(f, decl) + " {")
	g.indent++
	if g.recurse {
		pos := g.sh.position(decl.Name, fmt.Sprintf("stack overflow: %s recursed deeper than the worker's stack (%d KiB of L1) allows; tessc -stack gives it more",
			f.Name(), g.sh.stackKiB))
		g.line(fmt.Sprintf("tsr::stack_check(%d);  // %s recurses", pos, f.Name()))
	}
	if g.sh.mayAlloc[f] && releases(f) {
		g.line("tsr::HeapMark heap_mark_;  // nothing this call allocates outlives it")
	}
	g.enter(decl.Body, f.Type().(*types.Signature).Results(), funcName(f), decl)
	g.indent--
	g.block(decl.Body.List)
	if f.Type().(*types.Signature).Results().Len() > 0 {
		g.line("    __builtin_unreachable();")
	}
	g.line("}")
}

// receiver translates the operand x of x.f or x.m(), whose type is t. A
// pointer is allowed only as a pointer receiver, which is a reference.
func (g *cgen) receiver(x ast.Expr, t types.Type) string {
	if _, ok := types.Unalias(t).Underlying().(*types.Pointer); ok {
		id, isID := ast.Unparen(x).(*ast.Ident)
		if !isID || !g.refs[g.sh.info.Uses[id]] {
			g.fail(x, "pointers are not supported %s (except method receivers)", g.where())
		}
	}
	return g.expr(x)
}

// fieldPath follows the field indices of a selection from base, whose type
// is t: one field, or the embedded fields that lead to it.
func (g *cgen) fieldPath(base string, t types.Type, path []int) string {
	for _, i := range path {
		if p, ok := types.Unalias(t).Underlying().(*types.Pointer); ok {
			t = p.Elem()
		}
		f := types.Unalias(t).Underlying().(*types.Struct).Field(i)
		base += "." + cident(f.Name())
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
		g.line(g.ct(v.Type(), n) + " " + cname + "{};")
		g.named = append(g.named, cname)
	}
}

// ret translates a return statement.
func (g *cgen) ret(s *ast.ReturnStmt) {
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
		g.line("return " + g.expr(s.Results[0]) + ";")
	case len(s.Results) == n:
		g.line("return " + g.resType + "{" + g.args(s.Results) + "};")
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
	g.line(fmt.Sprintf("bool %s = tsr::map_lookup(%s, %s, %s);", ok, g.expr(x.X), g.expr(x.Index), v))
	g.line(g.assignOne(lhs[0], v, define) + ";")
	g.line(g.assignOne(lhs[1], ok, define) + ";")
}

// assignResults translates x, y := f() and x, y = f(). New variables are
// declared first, so that they outlive the block that holds the results.
func (g *cgen) assignResults(lhs []ast.Expr, call ast.Expr, define bool) {
	if len(lhs) == 2 && isMapIndex(g.sh.info, call) && g.mode == deviceMode {
		g.commaOkMap(lhs, call, define)
		return
	}
	if _, ok := g.typeOf(call).(*types.Tuple); !ok {
		g.fail(call, "assign as many values as there are variables")
	}
	targets := make([]string, len(lhs))
	for i, l := range lhs {
		id, isID := l.(*ast.Ident)
		switch {
		case isID && id.Name == "_":
		case isID && define && g.sh.info.Defs[id] != nil:
			obj := g.sh.info.Defs[id]
			if _, _, isArr := arrayLen(obj.Type()); isArr {
				g.fail(id, "functions cannot return arrays on the card")
			}
			targets[i] = g.name(obj)
			g.line(g.ct(obj.Type(), id) + " " + targets[i] + "{};")
		default:
			targets[i] = g.expr(l)
		}
	}
	t := g.temp("r")
	g.line("{")
	g.line("    auto " + t + " = " + g.expr(call) + ";")
	for i, target := range targets {
		if target != "" {
			g.line(fmt.Sprintf("    %s = %s.r%d;", target, t, i))
		}
	}
	g.line("}")
}

// structDefs renders the definitions of the struct types used.
func structDefs(sh *cgShared) string {
	var b strings.Builder
	sizes := types.SizesFor("gc", "amd64")
	g := newCgen(sh, deviceMode)
	for _, t := range sh.structs {
		st := t.Underlying().(*types.Struct)
		name := cident(t.Obj().Name())
		fmt.Fprintf(&b, "struct %s {\n", name)
		var same []string
		for i := 0; i < st.NumFields(); i++ {
			f := st.Field(i)
			fname := cident(f.Name())
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
		fmt.Fprintf(&b, "};\nstatic_assert(sizeof(%s) == %d, \"%s must have the same layout on the host and the card\");\n\n",
			cident(t.Obj().Name()), sizes.Sizeof(t), t.Obj().Name())
	}
	for _, r := range sh.results {
		b.WriteString(r)
	}
	if len(sh.results) > 0 {
		b.WriteString("\n")
	}
	return b.String()
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
			fmt.Fprintf(&b, "    h = tsr::hash_more(h, hash_key_(v.%s));\n", cident(st.Field(i).Name()))
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
