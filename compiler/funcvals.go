package compiler

import (
	"fmt"
	"go/ast"
	"go/token"
	"go/types"
	"sort"
	"strings"
)

// Function values on the card.
//
// A function value is a tsr::Func: the number tessc gives the code it calls
// (0 for nil) and a pointer to its data, 8 bytes as in Go. tessc sees the
// whole program, so it knows every function value the card's code makes:
// the program's functions and methods used as values (f, x.M, T.M) and the
// function literals that do more than get called where they are made. A
// call through a function value is a call of the dispatch function tessc
// writes for its signature, a switch on the number (the kernel loader
// cannot fix up code addresses kept in data, and a switch is fast).
//
// A function literal that becomes a value is a function of its own,
// prog::func_N, which reaches its closure's variables through a pointer to
// a struct on the heap: a copy of each variable that nothing changes once
// it is set, and the address of each other one, which then lives on the
// heap (escape.go), as Go's would. A literal that a variable holds and that
// is only ever called (f := func(...) {...}; f(x)) stays a C++ lambda.

// fvInfo is what the checker finds out about the function values in the
// card's code.
type fvInfo struct {
	lits    map[*ast.FuncLit]bool        // literals that become values
	lambdas map[*types.Var]bool          // variables holding a literal that is only called: C++ lambdas
	sources []fvSource                   // the function values the code makes
	node    map[*ast.FuncLit]*types.Func // a stand-in for each such literal in reachable's graph
	litOf   map[*types.Func]*ast.FuncLit
	used    bool // the code has function values
}

// fvSource is a function value that card code makes, and what a call
// through it reaches.
type fvSource struct {
	sig   *types.Signature
	fn    *types.Func // a function, or the method of a method value or expression
	lit   *ast.FuncLit
	iface string // a method value of an interface: the method's name
}

// isCallee reports whether e (under parentheses) is the function a call
// calls.
func isCallee(parent map[ast.Node]ast.Node, e ast.Node) bool {
	for {
		p := parent[e]
		switch x := p.(type) {
		case *ast.ParenExpr:
			e = x
			continue
		case *ast.CallExpr:
			return x.Fun == e
		}
		return false
	}
}

// valueCall reports whether call calls a function value, and its signature.
func valueCall(info *types.Info, call *ast.CallExpr, lambdas map[*types.Var]bool) (*types.Signature, bool) {
	if tv := info.Types[call.Fun]; tv.IsType() || tv.IsBuiltin() {
		return nil, false
	}
	switch fun := ast.Unparen(call.Fun).(type) {
	case *ast.FuncLit:
		return nil, false
	case *ast.Ident:
		switch obj := info.Uses[fun].(type) {
		case *types.Func, *types.Builtin:
			return nil, false
		case *types.Var:
			if lambdas[obj] {
				return nil, false
			}
		}
	case *ast.SelectorExpr:
		if id, ok := fun.X.(*ast.Ident); ok {
			if _, ok := info.Uses[id].(*types.PkgName); ok {
				return nil, false
			}
		}
		if s, ok := info.Selections[fun]; ok && s.Kind() == types.MethodVal {
			return nil, false
		}
	}
	sig, ok := types.Unalias(info.TypeOf(call.Fun)).Underlying().(*types.Signature)
	return sig, ok
}

// funcValues finds the function values of the card's code: roots and the
// functions they reach.
func (c *checker) funcValues(roots []ast.Node) *fvInfo {
	fv := &fvInfo{lits: map[*ast.FuncLit]bool{}, lambdas: map[*types.Var]bool{},
		node: map[*ast.FuncLit]*types.Func{}, litOf: map[*types.Func]*ast.FuncLit{}}
	bodies := append([]ast.Node{}, roots...)
	var fns []*types.Func
	for f := range c.funcsFrom(roots, nil, false) {
		fns = append(fns, f)
	}
	sort.Slice(fns, func(i, j int) bool { return fns[i].Pos() < fns[j].Pos() })
	for _, f := range fns {
		bodies = append(bodies, c.funcs[f])
	}
	parent := map[ast.Node]ast.Node{}
	uses := map[*types.Var][]*ast.Ident{}
	var lits []*ast.FuncLit
	var idents []*ast.Ident
	var sels []*ast.SelectorExpr
	var calls []*ast.CallExpr
	seen := map[ast.Node]bool{}
	for _, b := range bodies {
		if b == nil || seen[b] {
			continue
		}
		seen[b] = true
		var stack []ast.Node
		ast.Inspect(b, func(n ast.Node) bool {
			if n == nil {
				stack = stack[:len(stack)-1]
				return true
			}
			if len(stack) > 0 {
				parent[n] = stack[len(stack)-1]
			}
			stack = append(stack, n)
			switch x := n.(type) {
			case *ast.FuncLit:
				if !seen[x] {
					seen[x] = true
					lits = append(lits, x)
				}
			case *ast.Ident:
				idents = append(idents, x)
				if v, ok := c.info.Uses[x].(*types.Var); ok {
					uses[v] = append(uses[v], x)
				}
			case *ast.SelectorExpr:
				sels = append(sels, x)
			case *ast.CallExpr:
				calls = append(calls, x)
			}
			return true
		})
	}

	// Which literals are only called: right where they are made, or
	// through a variable that nothing else uses.
	lambdaLit := map[*types.Var]*ast.FuncLit{}
	for _, lit := range lits {
		if isCallee(parent, lit) && !isDeferred(parent, parent[lit]) {
			continue // called where it is made (defer func() {...}() keeps it for later)
		}
		var p ast.Node = lit
		for {
			if q, ok := parent[p].(*ast.ParenExpr); ok {
				p = q
				continue
			}
			break
		}
		var v *types.Var
		switch x := parent[p].(type) {
		case *ast.AssignStmt:
			if x.Tok == token.DEFINE && len(x.Lhs) == 1 && len(x.Rhs) == 1 {
				if id, ok := x.Lhs[0].(*ast.Ident); ok {
					v, _ = c.info.Defs[id].(*types.Var)
				}
			}
		case *ast.ValueSpec:
			if len(x.Names) == 1 && len(x.Values) == 1 {
				v, _ = c.info.Defs[x.Names[0]].(*types.Var)
			}
		}
		if v == nil || isPkgVar(v) {
			fv.lits[lit] = true
			continue
		}
		lambda := true
		for _, id := range uses[v] {
			if !isCallee(parent, id) || isDeferred(parent, parent[id]) {
				lambda = false
			}
		}
		if lambda {
			lambdaLit[v] = lit
		} else {
			fv.lits[lit] = true
		}
	}
	// A C++ lambda cannot be called from another function: a variable that
	// a literal which becomes a value uses cannot hold one.
	usedInValue := func(v *types.Var) bool {
		for _, id := range uses[v] {
			for l := range fv.lits {
				if id.Pos() >= l.Pos() && id.Pos() < l.End() && !(v.Pos() >= l.Pos() && v.Pos() < l.End()) {
					return true
				}
			}
		}
		return false
	}
	for changed := true; changed; {
		changed = false
		for v, lit := range lambdaLit {
			if usedInValue(v) {
				fv.lits[lit] = true
				delete(lambdaLit, v)
				changed = true
			}
		}
	}
	for v := range lambdaLit {
		fv.lambdas[v] = true
	}

	// The function values the code makes.
	sigOf := func(e ast.Expr) *types.Signature {
		s, _ := types.Unalias(c.info.TypeOf(e)).Underlying().(*types.Signature)
		return s
	}
	add := func(s fvSource) {
		if s.sig == nil {
			return
		}
		for _, o := range fv.sources {
			if o.fn == s.fn && o.lit == s.lit && o.iface == s.iface && types.Identical(o.sig, s.sig) {
				return
			}
		}
		fv.sources = append(fv.sources, s)
	}
	for _, id := range idents {
		if f, ok := c.info.Uses[id].(*types.Func); ok && f.Pkg() == c.pkg && f.Type().(*types.Signature).Recv() == nil && !isCallee(parent, id) {
			if sel, ok := parent[id].(*ast.SelectorExpr); ok && sel.Sel == id {
				continue
			}
			add(fvSource{sig: f.Type().(*types.Signature), fn: f})
		}
		if v, ok := c.info.Uses[id].(*types.Var); ok && !fv.lambdas[v] && sigOf(id) != nil {
			fv.used = true
		}
	}
	for _, sel := range sels {
		s, ok := c.info.Selections[sel]
		if !ok {
			continue
		}
		switch {
		case s.Kind() == types.FieldVal && sigOf(sel) != nil:
			fv.used = true
		case s.Kind() == types.MethodExpr || (s.Kind() == types.MethodVal && !isCallee(parent, sel)):
			m := s.Obj().(*types.Func)
			if isIfaceMethod(m) {
				add(fvSource{sig: sigOf(sel), iface: m.Name()})
			} else if m.Pkg() == c.pkg {
				add(fvSource{sig: sigOf(sel), fn: m.Origin()})
			}
		}
	}
	var sorted []*ast.FuncLit
	for lit := range fv.lits {
		sorted = append(sorted, lit)
	}
	sort.Slice(sorted, func(i, j int) bool { return sorted[i].Pos() < sorted[j].Pos() })
	for _, lit := range sorted {
		sig := sigOf(lit)
		add(fvSource{sig: sig, lit: lit})
		f := types.NewFunc(lit.Pos(), c.pkg, "func literal", sig)
		fv.node[lit], fv.litOf[f] = f, lit
	}
	for _, call := range calls {
		if _, ok := valueCall(c.info, call, fv.lambdas); ok {
			fv.used = true
		}
	}
	fv.used = fv.used || len(fv.sources) > 0
	return fv
}

// valueCallees returns what the calls through function values in body can
// call: the functions and methods made into values of the same signature,
// and the stand-ins of the literals.
func (c *checker) valueCallees(body ast.Node) []*types.Func {
	if c.fv == nil {
		return nil
	}
	var out []*types.Func
	ast.Inspect(body, func(n ast.Node) bool {
		call, ok := n.(*ast.CallExpr)
		if !ok {
			return true
		}
		sig, ok := valueCall(c.info, call, c.fv.lambdas)
		if !ok {
			return true
		}
		for _, s := range c.fv.sources {
			if !types.Identical(s.sig, sig) {
				continue
			}
			switch {
			case s.lit != nil:
				out = append(out, c.fv.node[s.lit])
			case s.fn != nil:
				out = append(out, s.fn)
			case s.iface != "":
				for _, m := range c.via {
					if m.Name() == s.iface {
						out = append(out, m)
					}
				}
			}
		}
		return true
	})
	// A literal that a defer statement calls runs when the function returns.
	ast.Inspect(body, func(n ast.Node) bool {
		if d, ok := n.(*ast.DeferStmt); ok {
			if lit, ok := ast.Unparen(d.Call.Fun).(*ast.FuncLit); ok && c.fv.node[lit] != nil {
				out = append(out, c.fv.node[lit])
			}
		}
		return true
	})
	return out
}

// planFuncValues runs funcValues for a program's card code and gives the
// code generator what it needs.
func (c *checker) planFuncValues(sh *cgShared, roots []ast.Node) {
	c.fv = c.funcValues(roots)
	sh.valueLit = c.fv.lits
	sh.lambdaVar = c.fv.lambdas
	sh.usesFuncs = c.fv.used
}

// litOf returns the function literal whose stand-in f is, or nil.
func (c *checker) litOf(f *types.Func) *ast.FuncLit {
	if c.fv == nil {
		return nil
	}
	return c.fv.litOf[f]
}

// litRecursion moves reachable's marks for the literals' stand-ins out of
// recursive.
func (c *checker) litRecursion(sh *cgShared) {
	sh.litRecursive = map[*ast.FuncLit]bool{}
	if c.fv == nil {
		return
	}
	for f, lit := range c.fv.litOf {
		if sh.recursive[f] {
			sh.litRecursive[lit] = true
			delete(sh.recursive, f)
		}
	}
}

// callsValue reports whether code calls a function value.
func callsValue(sh *cgShared, n ast.Node) bool {
	found := false
	ast.Inspect(n, func(n ast.Node) bool {
		if call, ok := n.(*ast.CallExpr); ok && !found {
			_, found = valueCall(sh.info, call, sh.lambdaVar)
		}
		return !found
	})
	return found
}

// ------------------------------------------------------------------ translation

// litCapture is a variable that a function literal which becomes a value
// uses from outside: it holds a copy, or (byRef) the variable's address.
type litCapture struct {
	v     *types.Var
	byRef bool
}

// fnSource is a function value that the translated code makes: case id of
// the dispatch functions of its signature.
type fnSource struct {
	id       int
	key      any
	sig      *types.Signature
	what     string
	recovers bool // the function takes can_recover_ (defer.go)
	// call is the call, in a dispatch function, of function value f with
	// the dispatch function's parameters.
	call func(h *cgen, f string, args []string) string
}

// fnCall is the dispatch function for calls through function values of
// one signature.
type fnCall struct {
	sig      *types.Signature
	fn       string
	deferred bool // for deferred calls: the function called can recover
}

// source returns the function value of key, numbering it if it is new.
func (g *cgen) source(key any, sig *types.Signature, what string, call func(h *cgen, f string, args []string) string) *fnSource {
	for _, s := range g.sh.fnSources {
		if s.key == key && types.Identical(s.sig, sig) {
			return s
		}
	}
	s := &fnSource{id: len(g.sh.fnSources) + 1, key: key, sig: sig, what: what, call: call}
	g.sh.fnSources = append(g.sh.fnSources, s)
	return s
}

// ifaceMethodKey identifies a method value of an interface type.
type ifaceMethodKey struct {
	iface string
	name  string
}

// methodKey identifies a method value or expression of a method.
type methodKey struct {
	m    *types.Func
	expr string // the receiver type of a method expression; "" for a method value
}

func (g *cgen) litWhere(n ast.Node) string {
	p := g.sh.fset.Position(n.Pos())
	return fmt.Sprintf("%s:%d", shortFile(p.Filename), p.Line)
}

func shortFile(name string) string {
	if i := strings.LastIndexByte(name, '/'); i >= 0 {
		return name[i+1:]
	}
	return name
}

// funcRef translates the name of function f used as a value.
func (g *cgen) funcRef(f *types.Func, n ast.Node) string {
	sig := f.Type().(*types.Signature)
	s := g.source(f, sig, "func "+f.Name(), func(h *cgen, _ string, args []string) string {
		return "prog::" + funcName(f) + "(" + strings.Join(args, ", ") + ")"
	})
	s.recovers = g.sh.recovers(f)
	return fmt.Sprintf("tsr::Func{%d, nullptr}", s.id)
}

// methodValue translates x.M and T.M used as values.
func (g *cgen) methodValue(x *ast.SelectorExpr, s *types.Selection) string {
	m := s.Obj().(*types.Func)
	sig, _ := types.Unalias(g.typeOf(x)).Underlying().(*types.Signature)
	if s.Kind() == types.MethodExpr {
		recvT := s.Recv()
		rs := types.TypeString(recvT, types.RelativeTo(g.sh.pkg))
		var dispatch string
		if isIfaceMethod(m) {
			dispatch = g.dispatchFn(ifaceOfMethod(m), m)
		}
		src := g.source(methodKey{m, rs}, sig, "method expression "+rs+"."+m.Name(), func(h *cgen, _ string, args []string) string {
			if dispatch != "" {
				return dispatch + "(" + strings.Join(args, ", ") + ")"
			}
			recv := h.recvArg(args[0], recvT, isPointer(recvT), s.Index(), m, x)
			return "prog::" + funcName(m.Origin()) + "(" + strings.Join(append([]string{recv}, args[1:]...), ", ") + ")"
		})
		src.recovers = dispatch == "" && g.sh.recovers(m)
		return fmt.Sprintf("tsr::Func{%d, nullptr}", src.id)
	}
	if isIfaceMethod(m) {
		it := ifaceOfMethod(m)
		dispatch := g.dispatchFn(it, m)
		base := g.ifaceRecv(x.X, s, x)
		src := g.source(ifaceMethodKey{types.TypeString(it, nil), m.Name()}, sig, "method value ."+m.Name()+" of "+types.TypeString(it, types.RelativeTo(g.sh.pkg)),
			func(h *cgen, f string, args []string) string {
				return dispatch + "(" + strings.Join(append([]string{"*static_cast<const tsr::Iface*>(" + f + ".env)"}, args...), ", ") + ")"
			})
		return fmt.Sprintf("tsr::Func{%d, tsr::box<tsr::Iface>(tsr::iface_nonnil(%s, %d), %d)}", src.id, base, g.nilPos(x), g.oom(x))
	}
	m = m.Origin()
	recvT := m.Type().(*types.Signature).Recv().Type()
	recv := g.methodRecv(x.X, s, m, x)
	if p, ok := types.Unalias(recvT).(*types.Pointer); ok {
		if !g.sh.ptrRecv[m] {
			g.fail(x, "internal error: method %s should take its receiver as a pointer", m.Name())
		}
		elem := g.ct(p.Elem(), x)
		src := g.source(methodKey{m, ""}, sig, "method value "+m.FullName(), func(h *cgen, f string, args []string) string {
			return "prog::" + funcName(m) + "(" + strings.Join(append([]string{"static_cast<" + elem + "*>(" + f + ".env)"}, args...), ", ") + ")"
		})
		src.recovers = g.sh.recovers(m)
		return fmt.Sprintf("tsr::Func{%d, tsr::to_env(%s)}", src.id, recv)
	}
	ct := g.ct(recvT, x)
	src := g.source(methodKey{m, ""}, sig, "method value "+m.FullName(), func(h *cgen, f string, args []string) string {
		return "prog::" + funcName(m) + "(" + strings.Join(append([]string{"*static_cast<const " + ct + "*>(" + f + ".env)"}, args...), ", ") + ")"
	})
	src.recovers = g.sh.recovers(m)
	return fmt.Sprintf("tsr::Func{%d, tsr::box<%s>(%s, %d)}", src.id, ct, recv, g.oom(x))
}

// capKind is how a literal's function reaches a variable it uses.
type capKind int

const (
	capCopy    capKind = iota // a copy (arrays: a copy of the tsr::Ptr to their elements)
	capAddress                // its address
)

// funcLit translates a function literal that becomes a value: the closure
// of its variables, and its own function.
func (g *cgen) funcLit(lit *ast.FuncLit) string {
	caps := g.sh.captures[lit]
	kinds := make([]capKind, len(caps))
	var fields, inits []string
	for i, c := range caps {
		v := c.v
		name := g.name(v)
		switch {
		case g.refs[v]:
			// A pointer receiver held as a C++ reference: its address is the
			// pointer.
			kinds[i] = capAddress
			fields = append(fields, fmt.Sprintf("%s c%d;  // %s", g.ct(v.Type(), lit), i, v.Name()))
			inits = append(inits, "&"+name)
		case isArray(v.Type()):
			_, et, _ := arrayLen(v.Type())
			kinds[i] = capCopy
			fields = append(fields, fmt.Sprintf("tsr::Ptr<%s> c%d;  // %s, on the heap", g.elemCT(et, lit), i, v.Name()))
			inits = append(inits, name)
		case c.byRef:
			kinds[i] = capAddress
			fields = append(fields, fmt.Sprintf("%s* c%d;  // %s, on the heap", g.ct(v.Type(), lit), i, v.Name()))
			inits = append(inits, "&"+name)
		default:
			kinds[i] = capCopy
			fields = append(fields, fmt.Sprintf("%s c%d;  // %s", g.ct(v.Type(), lit), i, v.Name()))
			inits = append(inits, name)
		}
	}
	src := g.litSource(lit, caps, kinds, fields)
	if len(caps) == 0 {
		return fmt.Sprintf("tsr::Func{%d, nullptr}", src.id)
	}
	env := fmt.Sprintf("prog::func_%d_env", src.id)
	return fmt.Sprintf("tsr::Func{%d, tsr::box<%s>(%s{%s}, %d)}", src.id, env, env, strings.Join(inits, ", "), g.oom(lit))
}

// litSource translates a function literal into a function of its own,
// prog::func_N, once.
func (g *cgen) litSource(lit *ast.FuncLit, caps []litCapture, kinds []capKind, fields []string) *fnSource {
	for _, s := range g.sh.fnSources {
		if s.key == lit {
			return s
		}
	}
	sig := g.typeOf(lit).(*types.Signature)
	id := len(g.sh.fnSources) + 1
	name := fmt.Sprintf("func_%d", id)
	env := name + "_env"
	src := g.source(lit, sig, "func literal at "+g.litWhere(lit), func(h *cgen, f string, args []string) string {
		if len(caps) > 0 {
			args = append([]string{"static_cast<" + env + "*>(" + f + ".env)"}, args...)
		}
		return name + "(" + strings.Join(args, ", ") + ")"
	})

	src.recovers = g.sh.usesDefer && hasRecover(g.sh.info, lit.Body)
	h := newCgen(g.sh, deviceMode)
	h.recurse = g.sh.litRecursive[lit]
	h.canRecover = src.recovers
	var params []string
	if len(caps) > 0 {
		var b strings.Builder
		fmt.Fprintf(&b, "// The variables of the func literal at %s.\nstruct %s {\n", g.litWhere(lit), env)
		for _, f := range fields {
			b.WriteString("    " + f + "\n")
		}
		b.WriteString("};")
		g.sh.litProtos = append(g.sh.litProtos, b.String())
		params = append(params, env+"* fenv_")
		for i, c := range caps {
			switch kinds[i] {
			case capAddress:
				h.names[c.v] = fmt.Sprintf("(*fenv_->c%d)", i)
			default:
				h.names[c.v] = fmt.Sprintf("fenv_->c%d", i)
			}
			if g.refs[c.v] {
				h.refs[c.v] = true
			}
		}
	}
	var boxes []string
	for i := 0; i < sig.Params().Len(); i++ {
		p := sig.Params().At(i)
		if isArray(p.Type()) {
			h.fail(lit, "pass arrays as slices (a[:])")
		}
		pname := h.temp("p")
		if p.Name() != "" && p.Name() != "_" {
			pname = h.name(p)
		}
		ct := h.ct(p.Type(), lit)
		if h.heapVar(p) {
			boxes = append(boxes, fmt.Sprintf("%s& %s = *tsr::box<%s>(%s_in_, %d);  // its address outlives the call", ct, pname, ct, pname, h.oom(lit)))
			pname += "_in_"
		}
		params = append(params, ct+" "+pname)
	}
	res := h.resultType(name, sig.Results(), lit)
	if src.recovers {
		g.sh.litProtos = append(g.sh.litProtos, "static "+res+" "+name+"("+strings.Join(append(params, recoverParam(true)), ", ")+");")
		params = append(params, recoverParam(false))
	} else {
		g.sh.litProtos = append(g.sh.litProtos, "static "+res+" "+name+"("+strings.Join(params, ", ")+");")
	}
	proto := "static " + res + " " + name + "(" + strings.Join(params, ", ") + ")"
	h.line(fmt.Sprintf("// The func literal at %s.", g.litWhere(lit)))
	h.line(proto + " {")
	h.indent++
	if h.recurse {
		pos := g.sh.position(lit, fmt.Sprintf("stack overflow: the func literal at %s recursed deeper than the worker's stack (%d KiB of L1) allows; tessc -stack gives it more",
			g.litWhere(lit), g.sh.stackKiB))
		h.line(fmt.Sprintf("tsr::stack_check(%d);  // it recurses", pos))
	} else {
		h.line(fmt.Sprintf("tsr::stack_check(%d);", h.stackPos(lit, "the func literal at "+g.litWhere(lit))))
	}
	for _, l := range boxes {
		h.line(l)
	}
	h.enter(lit.Body, sig.Results(), name, lit)
	h.indent--
	if !h.funcBody(lit.Body, lit) && sig.Results().Len() > 0 {
		h.line("    __builtin_unreachable();")
	}
	h.line("}")
	g.sh.litDefs = append(g.sh.litDefs, h.b.String())
	return src
}

// valueCall translates a call through a function value.
func (g *cgen) valueCall(x *ast.CallExpr, sig *types.Signature) string {
	fn := g.fcallFn(sig)
	args := []string{g.expr(x.Fun), fmt.Sprint(g.nilPos(x))}
	if sig.Params().Len() > 0 {
		args = append(args, g.callArgs(sig, x.Args, x.Ellipsis.IsValid(), x))
	}
	return fn + "(" + strings.Join(args, ", ") + ")"
}

// fcallFn returns the name of the dispatch function for calls through
// function values of signature sig.
func (g *cgen) fcallFn(sig *types.Signature) string {
	for _, k := range g.sh.fnCalls {
		if !k.deferred && types.Identical(k.sig, sig) {
			return k.fn
		}
	}
	fn := fmt.Sprintf("prog::fcall_%d", len(g.sh.fnCalls))
	g.sh.fnCalls = append(g.sh.fnCalls, &fnCall{sig: sig, fn: fn})
	return fn
}

// emitFuncValues writes what function values need, once the card's code is
// translated: the literals' functions and closures, and the dispatch
// functions, with their prototypes. It runs before emitIfaces, as a
// dispatch case can call through an interface.
func emitFuncValues(wp *WorkerProgram) (protos []string, defs string) {
	sh := wp.sh
	if len(sh.fnSources) == 0 && len(sh.fnCalls) == 0 {
		return nil, ""
	}
	protos = append(protos, sh.litProtos...)
	var d strings.Builder
	for _, l := range sh.litDefs {
		d.WriteString(l)
		d.WriteString("\n")
	}
	var at ast.Node = wp.main
	if wp.lit != nil {
		at = wp.lit
	}
	for k := 0; k < len(sh.fnCalls); k++ {
		c := sh.fnCalls[k]
		h := newCgen(sh, deviceMode)
		name := strings.TrimPrefix(c.fn, "prog::")
		res := h.resultType(name, c.sig.Results(), at)
		params := []string{"const tsr::Func& f", "int32_t pos"}
		var args []string
		for i := 0; i < c.sig.Params().Len(); i++ {
			params = append(params, fmt.Sprintf("%s p%d", h.ct(c.sig.Params().At(i).Type(), at), i))
			args = append(args, fmt.Sprintf("p%d", i))
		}
		proto := fmt.Sprintf("static %s %s(%s)", res, name, strings.Join(params, ", "))
		protos = append(protos, proto+";")
		what := "Calls"
		if c.deferred {
			what = "Deferred calls"
		}
		fmt.Fprintf(&d, "// %s through function values of type %s.\n%s {\n    switch (f.id) {\n", what, types.TypeString(c.sig, types.RelativeTo(sh.pkg)), proto)
		for _, s := range sh.fnSources {
			if !types.Identical(s.sig, c.sig) {
				continue
			}
			a := args
			if c.deferred && s.recovers {
				a = append(append([]string{}, args...), "true") // a deferred call: it can recover
			}
			call := s.call(h, "f", a)
			fmt.Fprintf(&d, "    case %d: {  // %s\n", s.id, s.what)
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
		d.WriteString("    default:  // nil\n        tsr::fail(pos, 0);\n    }\n}\n\n")
	}
	return protos, d.String()
}
