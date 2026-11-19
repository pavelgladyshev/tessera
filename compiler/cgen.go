package compiler

import (
	"fmt"
	"go/ast"
	"go/constant"
	"go/token"
	"go/types"
	"math"
	"sort"
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
	scratch   uint32 // bytes of L1 scratch allocated so far (device)
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
}

func newCgen(sh *cgShared, mode cgMode) *cgen {
	return &cgen{sh: sh, mode: mode, names: map[types.Object]string{}, taken: map[string]int{}}
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
		main tsr prog std ctx argc argv paths net kernel_main worker index results status fid`) {
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

func funcName(f *types.Func) string {
	n := cident(f.Name())
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
			g.fail(n, "strings are not supported in worker code")
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
		g.fail(n, "pointers are not supported %s (except *tt.List)", g.where())
	case *types.Slice:
		if g.mode == hostMode {
			return "std::vector<" + g.ct(u.Elem(), n) + ">"
		}
		return "tsr::Slice<" + g.ct(u.Elem(), n) + ">"
	case *types.Chan:
		g.ct(u.Elem(), n) // the element type must be supported
		if g.mode == hostMode {
			return "tsr::ChanRef"
		}
		return "tsr::Chan"
	case *types.Array:
		g.fail(n, "arrays can only be declared with var, indexed and sliced; pass a slice (a[:]) instead")
	}
	g.fail(n, "type %s is not supported %s", t, g.where())
	return ""
}

// useStruct records a struct type so its definition is emitted.
func (g *cgen) useStruct(t *types.Named, n ast.Node) {
	if g.sh.seenType[t] {
		return
	}
	g.sh.seenType[t] = true
	st := t.Underlying().(*types.Struct)
	for i := 0; i < st.NumFields(); i++ {
		f := st.Field(i)
		switch ft := types.Unalias(f.Type()).(type) {
		case *types.Basic:
			if ft.Info()&(types.IsInteger|types.IsBoolean) == 0 {
				g.fail(n, "field %s.%s: struct fields must be integers, bools or structs", t.Obj().Name(), f.Name())
			}
		case *types.Named:
			if _, ok := ft.Underlying().(*types.Struct); ok {
				g.useStruct(ft, n)
			} else if !isInteger(ft) {
				g.fail(n, "field %s.%s: struct fields must be integers, bools or structs", t.Obj().Name(), f.Name())
			}
		default:
			g.fail(n, "field %s.%s: struct fields must be integers, bools or structs", t.Obj().Name(), f.Name())
		}
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
			return g.name(obj)
		case *types.Nil:
			g.fail(x, "nil is not supported %s", g.where())
		}
		g.fail(x, "%s cannot be used as a value here", x.Name)
	case *ast.BinaryExpr:
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
			if _, ok := types.Unalias(sel.Recv()).Underlying().(*types.Pointer); ok {
				g.fail(x, "pointers to structs are not supported")
			}
			return g.expr(x.X) + "." + cident(x.Sel.Name)
		}
		g.fail(x, "%s cannot be used as a value here", types.ExprString(x))
	case *ast.CompositeLit:
		return g.composite(x)
	case *ast.FuncLit:
		g.fail(x, "function literals are only allowed as the worker function of tt.Parallel")
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
			g.fail(n, "strings are not supported in worker code")
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
		if isString(res) {
			return "(" + l + " + " + r + ")"
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
	if tv := g.sh.info.Types[x.Fun]; tv.IsType() {
		if isString(tv.Type) || isString(g.typeOf(x.Args[0])) {
			g.fail(x, "string conversions are not supported; use strconv.Itoa")
		}
		return g.ct(tv.Type, x) + "(" + g.expr(x.Args[0]) + ")"
	}
	switch fun := ast.Unparen(x.Fun).(type) {
	case *ast.Ident:
		switch obj := g.sh.info.Uses[fun].(type) {
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
			return g.methodCall(sel.Recv(), fun.Sel.Name, g.expr(fun.X), x)
		}
	}
	g.fail(x, "this call is not supported %s", g.where())
	return ""
}

func (g *cgen) builtin(name string, x *ast.CallExpr) string {
	switch name {
	case "len":
		t := types.Unalias(g.typeOf(x.Args[0])).Underlying()
		a := g.expr(x.Args[0])
		switch {
		case isString(t), g.mode == hostMode:
			return "static_cast<int64_t>(" + a + ".size())"
		}
		if _, ok := t.(*types.Slice); ok {
			return a + ".len"
		}
	case "min", "max":
		return "tsr::" + name + "<" + g.ct(g.typeOf(x), x) + ">(" + g.args(x.Args) + ")"
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
			g.fail(x, "make is only available in main: channels are made there and passed to the goroutines")
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

func (g *cgen) pkgCall(path, name string, x *ast.CallExpr) string {
	switch {
	case path == "strconv" && name == "Itoa" && g.mode == hostMode:
		return "std::to_string(" + g.expr(x.Args[0]) + ")"
	case path == TTPath && name == "WriteFile" && g.mode == hostMode:
		return "tsr::write_file(" + g.args(x.Args) + ")"
	case path == TTPath && (name == "LoadBytes" || name == "OpenBytes" || name == "LoadDetector" || name == "Parallel"):
		g.fail(x, "tt.%s must be assigned to a variable in main: x := tt.%s(...)", name, name)
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
	g.fail(x, "indexing %s is not supported", bt)
	return ""
}

func (g *cgen) sliceExpr(x *ast.SliceExpr) string {
	if x.Slice3 {
		g.fail(x, "three-index slices are not supported")
	}
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
		return fmt.Sprintf("tsr::slice(%s, %d, %s, %s, %d)", base, n, lo, hi, pos)
	}
	if _, ok := types.Unalias(bt).Underlying().(*types.Slice); ok {
		hi := base + ".len"
		if x.High != nil {
			hi = g.expr(x.High)
		}
		return fmt.Sprintf("%s.sub(%s, %s, %d)", base, lo, hi, pos)
	}
	g.fail(x, "slicing %s is not supported", bt)
	return ""
}

func (g *cgen) composite(x *ast.CompositeLit) string {
	t := g.typeOf(x)
	named, ok := types.Unalias(t).(*types.Named)
	st, ok2 := t.Underlying().(*types.Struct)
	if !ok || !ok2 {
		g.fail(x, "only struct literals are supported (not %s)", t)
	}
	vals := make([]string, st.NumFields())
	for i := range vals {
		vals[i] = g.ct(st.Field(i).Type(), x) + "{}"
	}
	for i, el := range x.Elts {
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
		g.forStmt(s)
	case *ast.RangeStmt:
		g.rangeStmt(s)
	case *ast.BlockStmt:
		g.line("{")
		g.block(s.List)
		g.line("}")
	case *ast.ReturnStmt:
		switch len(s.Results) {
		case 0:
			g.line("return;")
		case 1:
			g.line("return " + g.expr(s.Results[0]) + ";")
		default:
			g.fail(s, "functions return at most one value in v0.2")
		}
	case *ast.BranchStmt:
		if s.Label != nil {
			g.fail(s, "labelled break and continue are not supported")
		}
		switch s.Tok {
		case token.BREAK:
			g.line("break;")
		case token.CONTINUE:
			g.line("continue;")
		default:
			g.fail(s, "%s is not supported", s.Tok)
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

func (g *cgen) incDec(s *ast.IncDecStmt) string {
	l := g.expr(s.X)
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
		l := g.expr(s.Lhs[0])
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
		g.fail(lhs, "assigning whole arrays is not supported; copy the elements")
	}
	return g.expr(lhs) + " = " + rhs
}

func (g *cgen) assign(s *ast.AssignStmt) {
	if len(s.Lhs) == 2 && len(s.Rhs) == 1 {
		if u, ok := ast.Unparen(s.Rhs[0]).(*ast.UnaryExpr); ok && u.Op == token.ARROW {
			g.commaOkRecv(s, u)
			return
		}
	}
	if _, ok := assignOpTok[s.Tok]; ok || (len(s.Lhs) == 1 && len(s.Rhs) == 1) {
		g.line(g.assignExpr(s) + ";")
		return
	}
	if len(s.Lhs) != len(s.Rhs) {
		g.fail(s, "functions return at most one value in v0.2")
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
		for i, id := range vs.Names {
			obj := g.sh.info.Defs[id]
			if obj == nil {
				continue
			}
			if n, et, ok := arrayLen(obj.Type()); ok {
				if len(vs.Values) > 0 {
					g.fail(vs, "array literals are not supported; declare the array and fill it")
				}
				if _, isArr := types.Unalias(et).Underlying().(*types.Array); isArr {
					g.fail(id, "arrays of arrays are not supported")
				}
				ctype := g.ct(et, id)
				name := g.name(obj)
				if g.mode == hostMode {
					g.line(fmt.Sprintf("%s %s[%d] = {};", ctype, name, n))
					continue
				}
				size := n * types.SizesFor("gc", "amd64").Sizeof(et)
				off := g.sh.allocScratch(size)
				g.line(fmt.Sprintf("%s* %s = tsr::scratch<%s>(%d);  // [%d]%s in L1", ctype, name, ctype, off, n, et))
				g.line(fmt.Sprintf("tsr::zero(%s, %d);", name, n))
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

func (g *cgen) forStmt(s *ast.ForStmt) {
	cond := "true"
	if s.Cond != nil {
		cond = g.expr(s.Cond)
	}
	if s.Init == nil && s.Post == nil {
		g.line("while (" + cond + ") {")
	} else {
		g.line("for (" + g.simple(s.Init) + "; " + cond + "; " + g.simple(s.Post) + ") {")
	}
	g.block(s.Body.List)
	g.line("}")
}

func (g *cgen) rangeStmt(s *ast.RangeStmt) {
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
		g.block(s.Body.List)
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
		g.block(s.Body.List)
		g.line("}")
		return
	}
	var length, elem string
	var elemType types.Type
	x := g.temp("x")
	switch t := xt.(type) {
	case *types.Array:
		length, elemType = fmt.Sprint(t.Len()), t.Elem()
		g.line("{")
		g.indent++
		g.line(fmt.Sprintf("%s* %s = %s;", g.ct(t.Elem(), s), x, g.expr(s.X)))
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
	g.block(s.Body.List)
	g.line("}")
	g.indent--
	g.line("}")
}

// switchStmt becomes an if / else if chain. Go's switch has no fallthrough by
// default, so the only thing to watch is break, which would leave the
// enclosing loop in C++.
func (g *cgen) switchStmt(s *ast.SwitchStmt) {
	ast.Inspect(s.Body, func(n ast.Node) bool {
		switch n := n.(type) {
		case *ast.ForStmt, *ast.RangeStmt, *ast.FuncLit:
			return false
		case *ast.BranchStmt:
			if n.Tok == token.BREAK || n.Tok == token.FALLTHROUGH {
				g.fail(n, "%s inside switch is not supported", n.Tok)
			}
		}
		return true
	})
	g.line("{")
	g.indent++
	if s.Init != nil {
		g.stmt(s.Init)
	}
	tag := ""
	if s.Tag != nil {
		tag = g.temp("tag")
		g.line(g.ct(g.typeOf(s.Tag), s.Tag) + " " + tag + " = " + g.expr(s.Tag) + ";")
	}
	var def *ast.CaseClause
	first := true
	for _, c := range s.Body.List {
		cc := c.(*ast.CaseClause)
		if cc.List == nil {
			def = cc
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
		g.block(cc.Body)
	}
	if def != nil {
		if first {
			g.line("{")
		} else {
			g.line("} else {")
		}
		g.block(def.Body)
		first = false
	}
	if !first {
		g.line("}")
	}
	g.indent--
	g.line("}")
}

// ------------------------------------------------------------------ functions

// signature renders a device function's C++ signature.
func (g *cgen) signature(f *types.Func, decl *ast.FuncDecl) string {
	sig := f.Type().(*types.Signature)
	res := "void"
	switch sig.Results().Len() {
	case 0:
	case 1:
		res = g.ct(sig.Results().At(0).Type(), decl)
	default:
		g.fail(decl, "%s: functions return at most one value in v0.2", f.Name())
	}
	var params []string
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
	g.line(g.signature(f, decl) + " {")
	g.block(decl.Body.List)
	if f.Type().(*types.Signature).Results().Len() > 0 {
		g.line("    __builtin_unreachable();")
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
		fmt.Fprintf(&b, "struct %s {\n", cident(t.Obj().Name()))
		for i := 0; i < st.NumFields(); i++ {
			f := st.Field(i)
			fmt.Fprintf(&b, "    %s %s;\n", g.ct(f.Type(), nil), cident(f.Name()))
		}
		fmt.Fprintf(&b, "};\nstatic_assert(sizeof(%s) == %d, \"%s must have the same layout on the host and the card\");\n\n",
			cident(t.Obj().Name()), sizes.Sizeof(t), t.Obj().Name())
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
				if _, isMethod := f.Type().(*types.Signature); isMethod && f.Type().(*types.Signature).Recv() == nil {
					seen[f] = true
					out = append(out, f)
				}
			}
		}
		return true
	})
	sort.Slice(out, func(i, j int) bool { return out[i].Pos() < out[j].Pos() })
	return out
}
