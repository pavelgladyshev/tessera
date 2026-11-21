package compiler

import (
	"go/ast"
	"go/token"
	"go/types"
	"strings"
)

// Go evaluates the function calls, method calls and channel receives of an
// expression in lexical left-to-right order. C++ leaves the order of most
// operands open: the arguments of a call, the two sides of + or ==, and
// (as tessc renders them) the index of s[i] and the fields of a struct
// literal. So when two or more operands of an expression make calls, and
// one of those calls can change memory, tessc evaluates every such operand
// but the last into a temporary, in order, inside a lambda it calls at
// once, and the expression uses the temporaries:
//
//	f(g(), h())   ->   [&]() { auto o1 = g(); return f(o1, h()); }()
//
// The lambda keeps it one expression, so this works in a loop's condition
// and on the right of && too, where the operands may not be evaluated at
// all. Operands that make no calls stay where they are: Go does not order
// reading a variable against a call either.

// callKind is what a call does, as far as order goes.
type callKind int

const (
	noCall     callKind = iota // a conversion, len, cap, min, max, make
	pureCall                   // a library function: it reads its arguments, it changes nothing
	effectCall                 // the program's functions and methods, append, copy, delete, tt's methods, ...
)

func (g *cgen) callKind(x *ast.CallExpr) callKind {
	if g.sh.info.Types[x].Value != nil || g.sh.info.Types[x.Fun].IsType() {
		return noCall
	}
	switch fun := ast.Unparen(x.Fun).(type) {
	case *ast.Ident:
		if b, ok := g.sh.info.Uses[fun].(*types.Builtin); ok {
			switch b.Name() {
			case "len", "cap", "min", "max", "make", "new", "complex", "real", "imag":
				return noCall
			}
			return effectCall // append, copy, delete, clear, close, panic, ...
		}
	case *ast.SelectorExpr:
		if id, ok := fun.X.(*ast.Ident); ok {
			if pn, ok := g.sh.info.Uses[id].(*types.PkgName); ok && pn.Imported().Path() != TTPath {
				return pureCall // strings, bytes, unicode/utf8, math/bits, ...
			}
		}
	}
	return effectCall
}

// calls reports whether evaluating e calls a function or receives from a
// channel, and whether one of those can change memory. The bodies of
// function literals do not count: they run when they are called.
func (g *cgen) calls(e ast.Expr) (call, effect bool) {
	if v, ok := g.callMemo[e]; ok {
		return v&1 != 0, v&2 != 0
	}
	ast.Inspect(e, func(n ast.Node) bool {
		switch x := n.(type) {
		case *ast.FuncLit:
			return false
		case *ast.UnaryExpr:
			if x.Op == token.ARROW {
				call, effect = true, true
			}
		case *ast.CallExpr:
			switch g.callKind(x) {
			case pureCall:
				call = true
			case effectCall:
				call, effect = true, true
			}
		}
		return true
	})
	if g.callMemo == nil {
		g.callMemo = map[ast.Expr]uint8{}
	}
	var v uint8
	if call {
		v |= 1
	}
	if effect {
		v |= 2
	}
	g.callMemo[e] = v
	return call, effect
}

// operands are the parts of e that Go evaluates before e itself, in Go's
// order. && and || have none here: C++ evaluates them in order too, and
// their right side only sometimes.
func (g *cgen) operands(e ast.Expr) []ast.Expr {
	var ops []ast.Expr
	switch x := e.(type) {
	case *ast.BinaryExpr:
		if x.Op != token.LAND && x.Op != token.LOR {
			ops = []ast.Expr{x.X, x.Y}
		}
	case *ast.CallExpr:
		if sel, ok := ast.Unparen(x.Fun).(*ast.SelectorExpr); ok {
			if s, ok := g.sh.info.Selections[sel]; ok && s.Kind() == types.MethodVal {
				ops = append(ops, sel.X) // the receiver
			}
		}
		ops = append(ops, x.Args...)
	case *ast.IndexExpr:
		ops = []ast.Expr{x.X, x.Index}
	case *ast.SliceExpr:
		for _, o := range []ast.Expr{x.X, x.Low, x.High, x.Max} {
			if o != nil {
				ops = append(ops, o)
			}
		}
	case *ast.CompositeLit:
		for _, el := range x.Elts {
			if kv, ok := el.(*ast.KeyValueExpr); ok {
				ops = append(ops, kv.Key, kv.Value) // a struct's keys are names: they make no calls
			} else {
				ops = append(ops, el)
			}
		}
	}
	return ops
}

// inOrder renders e with its calls in Go's order, if they need it (ok is
// false if not). As an lvalue (lv), it stands for the place itself.
func (g *cgen) inOrder(e ast.Expr, lv bool) (string, bool) {
	if g.ordering[e] {
		return "", false // inOrder is rendering e itself
	}
	var calling []ast.Expr
	effect := false
	for _, op := range g.operands(e) {
		if c, ef := g.calls(op); c {
			calling = append(calling, op)
			effect = effect || ef
		}
	}
	if len(calling) < 2 || !effect {
		return "", false
	}
	if g.subst == nil {
		g.subst, g.ordering = map[ast.Expr]string{}, map[ast.Expr]bool{}
	}
	var b strings.Builder
	early := calling[:len(calling)-1]
	for _, op := range early {
		t := g.temp("o")
		decl := "auto"
		if lv || g.byReference(e, op) {
			decl = "auto&&"
		}
		b.WriteString(decl + " " + t + " = " + g.expr(op) + "; ")
		g.subst[op] = t
	}
	g.ordering[e] = true
	var inner string
	if lv {
		inner = g.lvalue(e)
	} else {
		inner = g.expr(e)
	}
	delete(g.ordering, e)
	for _, op := range early {
		delete(g.subst, op)
	}
	if lv {
		return "(*[&]() { " + b.String() + "return &(" + inner + "); }())", true
	}
	return "[&]() { " + b.String() + "return " + inner + "; }()", true
}

// byReference reports whether operand op of e must be kept as a reference,
// not a copy: an array (C++ cannot copy one with auto), and the receiver
// of a method with a pointer receiver, which changes what it is called on.
func (g *cgen) byReference(e, op ast.Expr) bool {
	if _, _, isArr := arrayLen(g.typeOf(op)); isArr {
		return true
	}
	call, ok := e.(*ast.CallExpr)
	if !ok {
		return false
	}
	sel, ok := ast.Unparen(call.Fun).(*ast.SelectorExpr)
	if !ok || sel.X != op {
		return false
	}
	s, ok := g.sh.info.Selections[sel]
	if !ok || s.Kind() != types.MethodVal {
		return false
	}
	recv := s.Obj().Type().(*types.Signature).Recv()
	_, ptr := types.Unalias(recv.Type()).(*types.Pointer)
	return ptr
}
