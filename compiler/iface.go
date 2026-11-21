package compiler

import (
	"fmt"
	"go/ast"
	"go/token"
	"go/types"
	"sort"
	"strings"
)

// Interfaces on the card.
//
// An interface value is a tsr::Iface: the number tessc gives its dynamic
// type (0 for nil) and one word, which holds a value of up to 8 bytes itself
// and a pointer to a copy on the heap for a bigger one. tessc sees the
// whole program, so it knows every type that the card's code puts in an
// interface. A method call through an interface is a switch on the type's
// number that calls the method of each type that can be there (the kernel
// loader cannot fix up tables of code addresses, and a switch is fast). So
// are type assertions, type switches, == and the hash of a map key.
//
// Which methods a call can reach is known before translation: the methods,
// of the called name, of the types that the card's code (or the package's
// initialization) converts to an interface which they implement. Only
// those are compiled for the card.

// isIfaceMethod reports whether f is a method of an interface type.
func isIfaceMethod(f *types.Func) bool {
	recv := f.Type().(*types.Signature).Recv()
	return recv != nil && types.IsInterface(recv.Type())
}

// ifaceOfMethod returns the interface type whose method f is.
func ifaceOfMethod(f *types.Func) types.Type {
	return f.Type().(*types.Signature).Recv().Type()
}

// fromType is the type of a value converted to an interface: an untyped
// constant converts with its default type.
func fromType(info *types.Info, e ast.Expr) types.Type {
	t := info.TypeOf(e)
	if b, ok := t.(*types.Basic); ok && b.Info()&types.IsUntyped != 0 {
		return types.Default(t)
	}
	return t
}

// eachConversion calls f for every value that code converts to an interface
// type, implicitly or not, with the value's type and the interface type.
func eachConversion(info *types.Info, root ast.Node, f func(e ast.Expr, from, to types.Type)) {
	conv := func(e ast.Expr, to types.Type) {
		if e == nil || to == nil || !types.IsInterface(to) {
			return
		}
		tv := info.Types[e]
		if tv.IsNil() || tv.IsType() || tv.Type == nil || types.IsInterface(tv.Type) {
			return
		}
		if _, ok := tv.Type.(*types.Tuple); ok {
			return
		}
		f(e, fromType(info, e), to)
	}
	var results []*types.Tuple
	var stack []ast.Node
	ast.Inspect(root, func(n ast.Node) bool {
		if n == nil {
			switch stack[len(stack)-1].(type) {
			case *ast.FuncDecl, *ast.FuncLit:
				results = results[:len(results)-1]
			}
			stack = stack[:len(stack)-1]
			return true
		}
		stack = append(stack, n)
		switch x := n.(type) {
		case *ast.FuncDecl:
			var res *types.Tuple
			if fn, ok := info.Defs[x.Name].(*types.Func); ok {
				res = fn.Type().(*types.Signature).Results()
			}
			results = append(results, res)
		case *ast.FuncLit:
			var res *types.Tuple
			if sig, ok := info.TypeOf(x).(*types.Signature); ok {
				res = sig.Results()
			}
			results = append(results, res)
		case *ast.AssignStmt:
			if len(x.Lhs) == len(x.Rhs) && x.Tok != token.DEFINE {
				for i := range x.Lhs {
					conv(x.Rhs[i], info.TypeOf(x.Lhs[i]))
				}
			}
		case *ast.ValueSpec:
			if x.Type != nil && len(x.Values) == len(x.Names) {
				t := info.TypeOf(x.Type)
				for _, v := range x.Values {
					conv(v, t)
				}
			}
		case *ast.ReturnStmt:
			if len(results) > 0 && results[len(results)-1] != nil {
				res := results[len(results)-1]
				if len(x.Results) == res.Len() {
					for i, r := range x.Results {
						conv(r, res.At(i).Type())
					}
				}
			}
		case *ast.CallExpr:
			if tv := info.Types[x.Fun]; tv.IsType() {
				if len(x.Args) == 1 {
					conv(x.Args[0], tv.Type)
				}
				break
			}
			if id, ok := ast.Unparen(x.Fun).(*ast.Ident); ok {
				if b, ok := info.Uses[id].(*types.Builtin); ok {
					switch b.Name() {
					case "append":
						if elem, ok := sliceElem(info.TypeOf(x)); ok && !x.Ellipsis.IsValid() {
							for _, a := range x.Args[1:] {
								conv(a, elem)
							}
						}
					case "panic":
						conv(x.Args[0], types.Universe.Lookup("any").Type())
					case "delete":
						if mt, ok := types.Unalias(info.TypeOf(x.Args[0])).Underlying().(*types.Map); ok {
							conv(x.Args[1], mt.Key())
						}
					}
					break
				}
			}
			if sig, ok := types.Unalias(info.TypeOf(x.Fun)).Underlying().(*types.Signature); ok && !x.Ellipsis.IsValid() {
				for i, a := range x.Args {
					conv(a, paramType(sig, i))
				}
			}
		case *ast.CompositeLit:
			t := info.TypeOf(x)
			if p, ok := types.Unalias(t).Underlying().(*types.Pointer); ok {
				t = p.Elem()
			}
			switch u := types.Unalias(t).Underlying().(type) {
			case *types.Struct:
				for i, el := range x.Elts {
					if kv, ok := el.(*ast.KeyValueExpr); ok {
						if id, ok := kv.Key.(*ast.Ident); ok {
							if f, ok := info.Uses[id].(*types.Var); ok && f.IsField() {
								conv(kv.Value, f.Type()) // also a field promoted from an embedded struct
							}
						}
					} else if i < u.NumFields() {
						conv(el, u.Field(i).Type())
					}
				}
			case *types.Slice, *types.Array:
				elem := u.(interface{ Elem() types.Type }).Elem()
				for _, el := range x.Elts {
					if kv, ok := el.(*ast.KeyValueExpr); ok {
						el = kv.Value
					}
					conv(el, elem)
				}
			case *types.Map:
				for _, el := range x.Elts {
					if kv, ok := el.(*ast.KeyValueExpr); ok {
						conv(kv.Key, u.Key())
						conv(kv.Value, u.Elem())
					}
				}
			}
		case *ast.IndexExpr:
			if mt, ok := types.Unalias(info.TypeOf(x.X)).Underlying().(*types.Map); ok {
				conv(x.Index, mt.Key())
			}
		case *ast.BinaryExpr:
			if x.Op == token.EQL || x.Op == token.NEQ {
				conv(x.X, info.TypeOf(x.Y))
				conv(x.Y, info.TypeOf(x.X))
			}
		case *ast.SwitchStmt:
			if x.Tag != nil {
				for _, c := range x.Body.List {
					for _, e := range c.(*ast.CaseClause).List {
						conv(e, info.TypeOf(x.Tag))
					}
				}
			}
		case *ast.SendStmt:
			if ct, ok := types.Unalias(info.TypeOf(x.Chan)).Underlying().(*types.Chan); ok {
				conv(x.Value, ct.Elem())
			}
		}
		return true
	})
}

// eachIfaceCall calls f for every call of a method of an interface value in
// code, and for every method value and method expression of one (which
// calls can go through).
func eachIfaceCall(info *types.Info, root ast.Node, f func(iface types.Type, name string)) {
	ast.Inspect(root, func(n ast.Node) bool {
		if sel, ok := n.(*ast.SelectorExpr); ok {
			if s, ok := info.Selections[sel]; ok && (s.Kind() == types.MethodVal || s.Kind() == types.MethodExpr) {
				if m, ok := s.Obj().(*types.Func); ok && isIfaceMethod(m) {
					f(ifaceOfMethod(m), m.Name())
				}
			}
		}
		return true
	})
}

// ifaceTargets returns the methods that the interface calls of code
// reached from bodies can call: for each method called through an
// interface, the method of that name of each type converted to an
// interface it implements, in that code or in conv (the package's
// initialization, which can fill package-level variables with interfaces).
func (c *checker) ifaceTargets(bodies []ast.Node, conv []ast.Node) []*types.Func {
	var convTypes []types.Type
	addConv := func(_ ast.Expr, from, _ types.Type) {
		for _, t := range convTypes {
			if types.Identical(t, from) {
				return
			}
		}
		convTypes = append(convTypes, from)
	}
	for _, b := range conv {
		eachConversion(c.info, b, addConv)
	}
	// var i I = v: the package's initialization converts v to I.
	for _, in := range c.info.InitOrder {
		t := fromType(c.info, in.Rhs)
		for k, v := range in.Lhs {
			from := t
			if tup, ok := t.(*types.Tuple); ok {
				if k >= tup.Len() {
					break
				}
				from = tup.At(k).Type()
			}
			if from != nil && types.IsInterface(v.Type()) && !types.IsInterface(from) && !isNil(c.info, in.Rhs) {
				addConv(in.Rhs, from, v.Type())
			}
		}
	}
	type call struct {
		iface types.Type
		name  string
	}
	var calls []call
	addCall := func(iface types.Type, name string) {
		for _, k := range calls {
			if k.name == name && types.Identical(k.iface, iface) {
				return
			}
		}
		calls = append(calls, call{iface, name})
	}
	added := map[*types.Func]bool{}
	var out []*types.Func
	for {
		roots := append([]ast.Node{}, bodies...)
		for _, m := range out {
			roots = append(roots, c.funcs[m])
		}
		scan := append([]ast.Node{}, roots...)
		for f := range c.funcsFrom(roots, nil, false) {
			scan = append(scan, c.funcs[f])
		}
		for _, b := range scan {
			eachConversion(c.info, b, addConv)
			eachIfaceCall(c.info, b, addCall)
		}
		grew := false
		for i := 0; i < len(calls); i++ { // calls grows: embedded interfaces
			k := calls[i]
			it, ok := types.Unalias(k.iface).Underlying().(*types.Interface)
			if !ok {
				continue
			}
			for _, t := range convTypes {
				if !types.Implements(t, it) {
					continue
				}
				obj, _, _ := types.LookupFieldOrMethod(t, true, c.pkg, k.name)
				m, ok := obj.(*types.Func)
				if !ok {
					continue
				}
				if isIfaceMethod(m) { // promoted from an embedded interface
					addCall(ifaceOfMethod(m), m.Name())
					continue
				}
				m = m.Origin()
				if m.Pkg() == c.pkg && c.funcs[m] != nil && !added[m] {
					added[m] = true
					out = append(out, m)
					grew = true
				}
			}
		}
		if !grew {
			break
		}
	}
	sort.Slice(out, func(i, j int) bool { return out[i].Pos() < out[j].Pos() })
	return out
}

// pkgInitBodies are the package's initializers and init functions.
func (c *checker) pkgInitBodies() []ast.Node {
	var out []ast.Node
	for _, in := range c.info.InitOrder {
		out = append(out, in.Rhs)
	}
	for f, d := range c.funcs {
		if isInitFunc(f) {
			out = append(out, d)
		}
	}
	return out
}

// ------------------------------------------------------------------ translation

// usesIfaces reports whether the card's code has interface values: such a
// program runs its workers on a stack in L1 (cgShared.l1Stack). Interface
// values are 16 bytes, and with GCC inlining the functions into one, their
// temporaries can make a frame bigger than the core's own stack.
func usesIfaces(sh *cgShared, wp *WorkerProgram) bool {
	var bodies []ast.Node
	for _, f := range wp.device {
		bodies = append(bodies, sh.funcs[f])
	}
	if wp.lit != nil {
		bodies = append(bodies, wp.lit)
	}
	bodies = append(bodies, wp.initRoots()...)
	found := false
	for _, b := range bodies {
		ast.Inspect(b, func(n ast.Node) bool {
			if e, ok := n.(ast.Expr); ok && !found {
				if t := sh.info.TypeOf(e); t != nil && types.IsInterface(t) {
					found = true
				}
			}
			return !found
		})
	}
	return found
}

// ifaceCall is a method called through an interface: the dispatch function
// tessc writes for it, prog::iface_N.
type ifaceCall struct {
	iface types.Type
	name  string
	sig   *types.Signature // the interface method's
	fn    string
}

// typeID returns the number of type t in interface values, numbering it if
// it is new; n is where it first goes into one.
func (g *cgen) typeID(t types.Type, n ast.Node) int {
	for i, u := range g.sh.ifaceTypes {
		if types.Identical(u, t) {
			return i + 1
		}
	}
	if isArray(t) {
		g.fail(n, "arrays cannot go into interfaces on the card (%s); use a slice or a struct", t)
	}
	d := newCgen(g.sh, deviceMode)
	g.sh.ifaceTypes = append(g.sh.ifaceTypes, t)
	g.sh.ifaceCT = append(g.sh.ifaceCT, d.ct(t, n))
	g.sh.ifaceAt = append(g.sh.ifaceAt, n)
	return len(g.sh.ifaceTypes)
}

// toIface translates e, whose type is not an interface, into an interface
// value.
func (g *cgen) toIface(e ast.Expr) string {
	t := fromType(g.sh.info, e)
	id := g.typeID(t, e)
	return fmt.Sprintf("tsr::to_iface<%s>(%d, %s, %d)", g.sh.ifaceCT[id-1], id, g.exprAs(e, t), g.oom(e))
}

// implementsFn returns the name of the function, written by emitIfaces, that
// tells whether a type number's type implements interface t.
func (g *cgen) implementsFn(t types.Type) string {
	for i, u := range g.sh.ifaceImpls {
		if types.Identical(u, t) {
			return fmt.Sprintf("prog::implements_%d", i)
		}
	}
	g.sh.ifaceImpls = append(g.sh.ifaceImpls, t)
	return fmt.Sprintf("prog::implements_%d", len(g.sh.ifaceImpls)-1)
}

// typeAssert translates x.(T).
func (g *cgen) typeAssert(x *ast.TypeAssertExpr) string {
	if x.Type == nil {
		g.fail(x, "use x.(type) in a type switch")
	}
	t := g.typeOf(x.Type)
	pos := g.sh.position(x, fmt.Sprintf("interface conversion: %s is not %s", types.TypeString(g.typeOf(x.X), types.RelativeTo(g.sh.pkg)), types.TypeString(t, types.RelativeTo(g.sh.pkg))))
	if types.IsInterface(t) {
		v := g.temp("x")
		return fmt.Sprintf("[&](const tsr::Iface& %s) { return tsr::iface_check(%s, %s(%s.type), %d); }(%s)", v, v, g.implementsFn(t), v, pos, g.expr(x.X))
	}
	id := g.typeID(t, x)
	return fmt.Sprintf("tsr::iface_assert<%s>(%s, %d, %d)", g.sh.ifaceCT[id-1], g.expr(x.X), id, pos)
}

// commaOkAssert translates v, ok := x.(T) and v, ok = x.(T).
func (g *cgen) commaOkAssert(lhs []ast.Expr, e *ast.TypeAssertExpr, define bool) {
	t := g.typeOf(e.Type)
	x, v, ok := g.temp("x"), g.temp("v"), g.temp("ok")
	g.line(fmt.Sprintf("const tsr::Iface %s = %s;", x, g.expr(e.X)))
	if types.IsInterface(t) {
		g.line(fmt.Sprintf("const bool %s = %s(%s.type);", ok, g.implementsFn(t), x))
		g.line(fmt.Sprintf("const tsr::Iface %s = %s ? %s : tsr::Iface{};", v, ok, x))
	} else {
		id := g.typeID(t, e)
		ct := g.sh.ifaceCT[id-1]
		g.line(fmt.Sprintf("const bool %s = %s.type == %d;", ok, x, id))
		g.line(fmt.Sprintf("%s %s = %s ? tsr::iface_value<%s>(%s) : %s;", ct, v, ok, ct, x, g.zeroValue(t, e)))
	}
	g.line(g.assignOne(lhs[0], g.convertTo(v, t, lhs[0], e), define) + ";")
	g.line(g.assignOne(lhs[1], g.convertTo(ok, types.Typ[types.Bool], lhs[1], e), define) + ";")
}

// zeroValue is a C++ expression for the zero value of t.
func (g *cgen) zeroValue(t types.Type, n ast.Node) string {
	if isPointer(t) {
		return "static_cast<" + g.ct(t, n) + ">(nullptr)"
	}
	return g.ct(t, n) + "{}"
}

// typeSwitch translates switch v := x.(type) {...}: like switchStmt, it
// finds the case first, then runs it in a C++ switch, so that break works.
func (g *cgen) typeSwitch(s *ast.TypeSwitchStmt) {
	g.line("{")
	g.indent++
	if s.Init != nil {
		g.stmt(s.Init)
	}
	var x ast.Expr
	switch a := s.Assign.(type) {
	case *ast.ExprStmt:
		x = a.X.(*ast.TypeAssertExpr).X
	case *ast.AssignStmt:
		x = a.Rhs[0].(*ast.TypeAssertExpr).X
	}
	v := g.temp("ts")
	g.line(fmt.Sprintf("const tsr::Iface %s = %s;", v, g.expr(x)))
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
			switch {
			case isNil(g.sh.info, e):
				conds = append(conds, v+".type == 0")
			case types.IsInterface(g.typeOf(e)):
				conds = append(conds, fmt.Sprintf("%s(%s.type)", g.implementsFn(g.typeOf(e)), v))
			default:
				conds = append(conds, fmt.Sprintf("%s.type == %d", v, g.typeID(g.typeOf(e), e)))
			}
		}
		kw := "} else if ("
		if first {
			kw, first = "if (", false
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
		g.line(fmt.Sprintf("case %d: {", i))
		if obj, ok := g.sh.info.Implicits[cc].(*types.Var); ok && obj.Name() != "_" {
			g.indent++
			t := obj.Type()
			if types.IsInterface(t) {
				g.line(g.declare(obj, cc, v) + ";")
			} else {
				id := g.typeID(t, cc)
				g.line(g.declare(obj, cc, fmt.Sprintf("tsr::iface_value<%s>(%s)", g.sh.ifaceCT[id-1], v)) + ";")
			}
			g.line("(void)" + g.name(obj) + ";")
			g.indent--
		}
		g.block(cc.Body)
		g.line("}")
		g.line("break;")
	}
	g.line("default:")
	g.line("    break;")
	g.line("}")
	g.indent--
	g.line("}")
}

// ifaceMethodCall translates x.m(args) for a method of an interface, after
// the embedded fields of selection s: a call of the dispatch function.
func (g *cgen) ifaceMethodCall(fun *ast.SelectorExpr, s *types.Selection, m *types.Func, x *ast.CallExpr) string {
	base := g.ifaceRecv(fun.X, s, x)
	fn := g.dispatchFn(ifaceOfMethod(m), m)
	sig := m.Type().(*types.Signature)
	args := []string{base}
	if len(x.Args) > 0 {
		args = append(args, g.callArgs(sig, x.Args, x.Ellipsis.IsValid(), x))
	}
	return fn + "(" + strings.Join(args, ", ") + ")"
}

// ifaceRecv is the interface value whose method x.m of selection s is: x,
// or an interface it holds in an embedded field.
func (g *cgen) ifaceRecv(x ast.Expr, s *types.Selection, n ast.Node) string {
	t := g.typeOf(x)
	base, isPtr := g.expr(x), isPointer(t)
	if id, ok := ast.Unparen(x).(*ast.Ident); ok && g.refs[g.sh.info.Uses[id]] {
		base, isPtr = g.name(g.sh.info.Uses[id]), false
	}
	idx := s.Index()
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
	return base
}

// dispatchFn returns the name of the dispatch function for method m of
// interface type iface.
func (g *cgen) dispatchFn(iface types.Type, m *types.Func) string {
	for _, k := range g.sh.ifaceCalls {
		if k.name == m.Name() && types.Identical(k.iface, iface) {
			return k.fn
		}
	}
	fn := fmt.Sprintf("prog::iface_%d_%s", len(g.sh.ifaceCalls), cident(m.Name()))
	g.sh.ifaceCalls = append(g.sh.ifaceCalls, &ifaceCall{iface: iface, name: m.Name(), sig: m.Type().(*types.Signature), fn: fn})
	return fn
}

// ifaceEqual translates a == b (or !=) where one side is an interface.
func (g *cgen) ifaceEqual(x *ast.BinaryExpr) (string, bool) {
	if g.mode != deviceMode || (x.Op != token.EQL && x.Op != token.NEQ) {
		return "", false
	}
	lt, rt := g.typeOf(x.X), g.typeOf(x.Y)
	var it types.Type
	switch {
	case lt != nil && types.IsInterface(lt):
		it = lt
	case rt != nil && types.IsInterface(rt):
		it = rt
	default:
		return "", false
	}
	if isNil(g.sh.info, x.Y) {
		return fmt.Sprintf("(%s.type %s 0)", g.expr(x.X), x.Op), true
	}
	if isNil(g.sh.info, x.X) {
		return fmt.Sprintf("(%s.type %s 0)", g.expr(x.Y), x.Op), true
	}
	s := fmt.Sprintf("tsr::iface_equal(%s, %s)", g.exprAs(x.X, it), g.exprAs(x.Y, it))
	if x.Op == token.NEQ {
		s = "(!" + s + ")"
	}
	return s, true
}

// emitIfaces writes what interface values need, once the card's code is
// translated (so that every type that goes into one has its number): the
// dispatch functions and the implements_N tests (with their prototypes),
// and tsr::iface_equal and tsr::iface_hash, which go after namespace prog.
func emitIfaces(wp *WorkerProgram, device map[*types.Func]bool) (protos []string, defs, after string) {
	sh := wp.sh
	if len(sh.ifaceTypes) == 0 && len(sh.ifaceCalls) == 0 && len(sh.ifaceImpls) == 0 && !sh.usesIface {
		return nil, "", ""
	}
	g := newCgen(sh, deviceMode)
	var d strings.Builder
	at := func() ast.Node {
		if len(sh.ifaceAt) > 0 {
			return sh.ifaceAt[0]
		}
		return wp.main
	}
	// Dispatch: the list can grow while it is written, for methods promoted
	// from embedded interfaces.
	needText := false
	for k := 0; k < len(sh.ifaceCalls); k++ {
		c := sh.ifaceCalls[k]
		h := newCgen(sh, deviceMode)
		res := h.resultType(strings.TrimPrefix(c.fn, "prog::"), c.sig.Results(), at())
		params := []string{"const tsr::Iface& x"}
		var args []string
		for i := 0; i < c.sig.Params().Len(); i++ {
			p := c.sig.Params().At(i)
			params = append(params, fmt.Sprintf("%s p%d", h.ct(p.Type(), at()), i))
			args = append(args, fmt.Sprintf("p%d", i))
		}
		proto := fmt.Sprintf("static %s %s(%s)", res, strings.TrimPrefix(c.fn, "prog::"), strings.Join(params, ", "))
		protos = append(protos, proto+";")
		fmt.Fprintf(&d, "// %s.%s, called through an interface.\n%s {\n    switch (x.type) {\n", types.TypeString(c.iface, types.RelativeTo(sh.pkg)), c.name, proto)
		it, _ := types.Unalias(c.iface).Underlying().(*types.Interface)
		for i, t := range sh.ifaceTypes {
			if it == nil || !types.Implements(t, it) {
				continue
			}
			obj, idx, _ := types.LookupFieldOrMethod(t, true, sh.pkg, c.name)
			m, ok := obj.(*types.Func)
			if !ok {
				continue
			}
			val := fmt.Sprintf("tsr::iface_value<%s>(x)", sh.ifaceCT[i])
			var call string
			if isIfaceMethod(m) {
				// Promoted from an embedded interface: dispatch again.
				inner := h.pathTo(val, t, idx[:len(idx)-1], at())
				call = h.dispatchFn(ifaceOfMethod(m), m) + "(" + strings.Join(append([]string{inner}, args...), ", ") + ")"
			} else {
				m = m.Origin()
				if !device[m] {
					panic(&Error{Pos: sh.fset.Position(sh.ifaceAt[i].Pos()), Msg: fmt.Sprintf("internal error: tessc did not compile %s for the card, which a call through an interface can reach", m.FullName())})
				}
				recv := h.recvArg(val, t, isPointer(t), idx, m, at())
				if p, ok := t.(*types.Pointer); ok && len(idx) == 1 && !pointerRecv(m) {
					if named, ok := types.Unalias(p.Elem()).(*types.Named); ok {
						// Go's text for calling a value method through a nil pointer.
						pos := sh.position(at(), fmt.Sprintf("value method %s.%s.%s called using nil *%s pointer", m.Pkg().Name(), named.Obj().Name(), m.Name(), named.Obj().Name()))
						recv = fmt.Sprintf("tsr::deref(%s, %d)", val, pos)
					}
				}
				call = "prog::" + funcName(m) + "(" + strings.Join(append([]string{recv}, args...), ", ") + ")"
			}
			fmt.Fprintf(&d, "    case %d: {  // %s\n", i+1, types.TypeString(t, types.RelativeTo(sh.pkg)))
			switch n := c.sig.Results().Len(); {
			case n == 0:
				fmt.Fprintf(&d, "        %s;\n        return;\n", call)
			case n == 1:
				fmt.Fprintf(&d, "        return %s;\n", call)
			default:
				var fields []string
				for r := 0; r < n; r++ {
					fields = append(fields, fmt.Sprintf("r.r%d", r))
				}
				fmt.Fprintf(&d, "        auto r = %s;\n        return %s{%s};\n", call, res, strings.Join(fields, ", "))
			}
			d.WriteString("    }\n")
		}
		if sh.usesDefer && c.name == "Error" && isErrorSig(c.sig) {
			// What recover gives for a runtime error is an error too.
			fmt.Fprintf(&d, "    case tsr::kRuntimeError:\n        return runtime_error_text(x.word);\n")
			needText = true
		}
		fmt.Fprintf(&d, "    default:  // nil\n        tsr::fail(%d, 0);\n    }\n}\n\n", g.nilPos(at()))
	}
	// implements_N.
	for k := 0; k < len(sh.ifaceImpls); k++ {
		it, _ := types.Unalias(sh.ifaceImpls[k]).Underlying().(*types.Interface)
		proto := fmt.Sprintf("static bool implements_%d(uint32_t t)", k)
		protos = append(protos, proto+";")
		fmt.Fprintf(&d, "// Whether a value of type number t implements %s.\n%s {\n    switch (t) {\n", types.TypeString(sh.ifaceImpls[k], types.RelativeTo(sh.pkg)), proto)
		cases := 0
		for i, t := range sh.ifaceTypes {
			if it != nil && types.Implements(t, it) {
				fmt.Fprintf(&d, "    case %d:  // %s\n", i+1, types.TypeString(t, types.RelativeTo(sh.pkg)))
				cases++
			}
		}
		if sh.usesDefer && it != nil && types.Implements(types.Universe.Lookup("error").Type(), it) {
			d.WriteString("    case tsr::kRuntimeError:  // a runtime error, from recover\n")
			cases++
		}
		if cases > 0 {
			d.WriteString("        return true;\n")
		}
		d.WriteString("    default:\n        return false;\n    }\n}\n\n")
	}
	if needText {
		protos = append(protos, "static tsr::Str runtime_error_text(uint64_t pos);")
		d.WriteString(runtimeErrorText(g))
	}
	// == and hashing, by dynamic type.
	var a strings.Builder
	a.WriteString("// == on interface values, and their hash as map keys: by dynamic type.\n")
	a.WriteString("bool tsr::iface_equal(const tsr::Iface& a, const tsr::Iface& b) {\n    if (a.type != b.type) {\n        return false;\n    }\n    switch (a.type) {\n    case 0:\n        return true;\n")
	var hb strings.Builder
	hb.WriteString("uint32_t tsr::iface_hash(const tsr::Iface& a) {\n    using tsr::hash_key_;\n    switch (a.type) {\n    case 0:\n        return 0;\n")
	for i, t := range sh.ifaceTypes {
		name := types.TypeString(t, types.RelativeTo(sh.pkg))
		ct := sh.ifaceCT[i]
		if types.Comparable(t) {
			fmt.Fprintf(&a, "    case %d:  // %s\n        return tsr::eq(tsr::iface_value<%s>(a), tsr::iface_value<%s>(b));\n", i+1, name, ct, ct)
			fmt.Fprintf(&hb, "    case %d:\n        return tsr::hash_more(%d, hash_key_(tsr::iface_value<%s>(a)));\n", i+1, i+1, ct)
		} else {
			fmt.Fprintf(&a, "    case %d:  // %s\n        tsr::fail(%d, 0);\n", i+1, name, sh.position(sh.ifaceAt[i], "runtime error: comparing uncomparable type "+name))
			fmt.Fprintf(&hb, "    case %d:\n        tsr::fail(%d, 0);\n", i+1, sh.position(sh.ifaceAt[i], "runtime error: hash of unhashable type "+name))
		}
	}
	a.WriteString("    }\n    return false;\n}\n")
	hb.WriteString("    }\n    return 0;\n}\n")
	after = a.String() + hb.String() + "\n"
	return protos, d.String(), after
}

// isErrorSig reports whether sig is func() string, the type of Error.
func isErrorSig(sig *types.Signature) bool {
	return sig.Params().Len() == 0 && sig.Results().Len() == 1 && isString(sig.Results().At(0).Type())
}

// runtimeErrorText writes runtime_error_text, the Error method of the
// runtime errors that recover gives: Go's text for the failure at a
// position (without the values Go puts in some of them).
func runtimeErrorText(g *cgen) string {
	byText := map[string][]int{}
	var texts []string
	for i, p := range g.sh.positions {
		var t string
		switch m := p.what; {
		case strings.HasPrefix(m, "stack overflow"), strings.HasPrefix(m, "out of memory"):
			continue // fatal: recover never sees them
		case strings.HasPrefix(m, "panic("):
			t = "panic called with nil argument (see issue 25448)" // the only panic that gives a runtime error
		case strings.HasPrefix(m, "runtime error:"), strings.HasPrefix(m, "assignment to entry in nil map"),
			strings.HasPrefix(m, "close of"), strings.HasPrefix(m, "interface conversion"), strings.HasPrefix(m, "value method"):
			t = m
		default:
			t = "runtime error: " + m
		}
		if byText[t] == nil {
			texts = append(texts, t)
		}
		byText[t] = append(byText[t], i)
	}
	var b strings.Builder
	b.WriteString("// Error() of a runtime error that recover gave: the failure's text.\nstatic tsr::Str runtime_error_text(uint64_t pos) {\n    switch (pos) {\n")
	for _, t := range texts {
		for _, i := range byText[t] {
			fmt.Fprintf(&b, "    case %d:\n", i)
		}
		fmt.Fprintf(&b, "        return %s;\n", g.literal(t))
	}
	b.WriteString("    default:\n        return " + g.literal("runtime error") + ";\n    }\n}\n\n")
	return b.String()
}

// pathTo follows embedded fields from base, of type t.
func (g *cgen) pathTo(base string, t types.Type, path []int, n ast.Node) string {
	isPtr := isPointer(t)
	for _, i := range path {
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
	return base
}
