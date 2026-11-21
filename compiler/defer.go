package compiler

import (
	"fmt"
	"go/ast"
	"go/types"
	"strings"
)

// defer, panic and recover on the card.
//
// A function with defer statements keeps its deferred calls in a list on
// the heap, last first: each is a function value (funcvals.go) of type
// func(). defer f(x) evaluates f and x at once into a closure, prog::defer_N,
// that makes the call later; defer func() {...}() keeps the literal's own
// closure. Such a function also catches panics: it sets up a
// __builtin_setjmp buffer, and tsr::fail jumps to the innermost one
// (tsr::catcher) instead of straight back to kernel_main. Returning, or
// being jumped to, it runs its deferred calls (a panic in one of them comes
// back to it, and it carries on with the rest), and then returns, or passes
// a panic that no deferred call recovered on to the next such function.
// GCC keeps the values that live across the calls of a function with a
// __builtin_setjmp receiver in memory, so they survive the jump.
//
// recover() stops a panic only in a function that a deferred call calls
// directly, as in Go: such a function takes a hidden parameter,
// can_recover_, false unless the deferred call passes true (the dispatch
// functions for deferred calls do, prog::fcall_N_defer). recover gives the
// value panic was called with; for a runtime error (an index out of range,
// say) it gives a non-nil value that the card's code cannot look into. Out
// of memory and stack overflow stay fatal, as in Go.

// deferState is the function being translated that has defer statements.
type deferState struct {
	frame string // its tsr::DeferFrame
	label string // where its returns go: the deferred calls
	ret   string // the variable that holds its results, if they are not named
}

// hasDefer reports whether a function body has defer statements of its
// own (not in the function literals in it).
func hasDefer(body ast.Node) bool {
	found := false
	ast.Inspect(body, func(n ast.Node) bool {
		switch n.(type) {
		case *ast.FuncLit:
			return n == body
		case *ast.DeferStmt:
			found = true
		}
		return !found
	})
	return found
}

// hasRecover reports whether a function body calls recover itself.
func hasRecover(info *types.Info, body ast.Node) bool {
	if body == nil {
		return false
	}
	found := false
	ast.Inspect(body, func(n ast.Node) bool {
		switch x := n.(type) {
		case *ast.FuncLit:
			return n == body
		case *ast.CallExpr:
			if id, ok := ast.Unparen(x.Fun).(*ast.Ident); ok {
				if b, ok := info.Uses[id].(*types.Builtin); ok && b.Name() == "recover" {
					found = true
				}
			}
		}
		return !found
	})
	return found
}

// usesDefers reports whether the card's code has defer statements.
func usesDefers(bodies []ast.Node) bool {
	found := false
	for _, b := range bodies {
		ast.Inspect(b, func(n ast.Node) bool {
			if _, ok := n.(*ast.DeferStmt); ok {
				found = true
			}
			return !found
		})
	}
	return found
}

// isDeferred reports whether call is the call of a defer statement.
func isDeferred(parent map[ast.Node]ast.Node, call ast.Node) bool {
	d, ok := parent[call].(*ast.DeferStmt)
	return ok && ast.Node(d.Call) == call
}

// recovers reports whether function f calls recover itself (and so takes
// can_recover_).
func (sh *cgShared) recovers(f *types.Func) bool {
	if !sh.usesDefer || f == nil {
		return false
	}
	d := sh.funcs[f.Origin()]
	return d != nil && hasRecover(sh.info, d.Body)
}

// recoverParam is the hidden parameter of a function that calls recover:
// with its default value in a prototype.
func recoverParam(proto bool) string {
	if proto {
		return "bool can_recover_ = false"
	}
	return "bool can_recover_"
}

// funcBody translates the body of a function being translated (after
// enter): with its deferred calls, if it has defer statements. It reports
// whether it wrote the function's last return.
func (g *cgen) funcBody(body *ast.BlockStmt, n ast.Node) bool {
	if g.mode != deviceMode || !hasDefer(body) {
		g.block(body.List)
		return false
	}
	saved := g.defers
	d := &deferState{frame: g.temp("defers"), label: g.temp("deferred")}
	g.indent++
	if g.results != nil && g.results.Len() > 0 && len(g.named) == 0 {
		d.ret = g.temp("ret")
		g.line(g.resType + " " + d.ret + "{};")
	}
	g.line(fmt.Sprintf("tsr::DeferFrame %s{};  // its deferred calls; panics jump here", d.frame))
	g.line(fmt.Sprintf("if (__builtin_setjmp(%s.buf) == 0) {", d.frame))
	g.line(fmt.Sprintf("    %s.enter();", d.frame))
	g.defers = d
	g.block(body.List)
	g.defers = saved
	g.line("    goto " + d.label + ";")
	g.line("}")
	call := g.temp("call")
	g.line(d.label + ":  // the deferred calls, last first")
	g.line(fmt.Sprintf("while (tsr::DeferNode* %s = %s.pop()) {", call, d.frame))
	g.line(fmt.Sprintf("    %s(%s->fn, %d);", g.fcallFnD(funcSig0), call, g.nilPos(n)))
	g.line("}")
	g.line(d.frame + ".leave();  // a panic that nothing recovered goes on")
	switch {
	case len(g.named) == 1:
		g.line("return " + g.named[0] + ";")
	case len(g.named) > 1:
		g.line("return " + g.resType + "{" + strings.Join(g.named, ", ") + "};")
	case d.ret != "":
		g.line("return " + d.ret + ";")
	}
	g.indent--
	return true
}

// funcSig0 is the type func().
var funcSig0 = types.NewSignatureType(nil, nil, nil, nil, nil, false)

// deferRet translates a return statement in a function with defer
// statements: it sets the results and goes to the deferred calls.
func (g *cgen) deferRet(s *ast.ReturnStmt) {
	d := g.defers
	n := 0
	if g.results != nil {
		n = g.results.Len()
	}
	g.line("{")
	g.indent++
	switch {
	case len(s.Results) == 0:
		// The named results hold the values.
	case len(g.named) > 0 && len(s.Results) == n:
		if n == 1 {
			g.line(g.named[0] + " = " + g.exprAs(s.Results[0], g.results.At(0).Type()) + ";")
			break
		}
		tmps := make([]string, n)
		for i, r := range s.Results {
			tmps[i] = g.temp("r")
			g.line(fmt.Sprintf("%s %s = %s;", g.ct(g.results.At(i).Type(), s), tmps[i], g.exprAs(r, g.results.At(i).Type())))
		}
		for i := range tmps {
			g.line(g.named[i] + " = " + tmps[i] + ";")
		}
	case len(g.named) > 0: // return f(), where f has the same results
		t := g.temp("r")
		g.line("auto " + t + " = " + g.expr(s.Results[0]) + ";")
		for i := range g.named {
			g.line(fmt.Sprintf("%s = %s.r%d;", g.named[i], t, i))
		}
	case n == 1:
		g.line(d.ret + " = " + g.exprAs(s.Results[0], g.results.At(0).Type()) + ";")
	case len(s.Results) == n:
		g.line(d.ret + " = " + g.resType + "{" + g.argsAs(s.Results, func(i int) types.Type { return g.results.At(i).Type() }) + "};")
	default:
		t := g.temp("r")
		parts := make([]string, n)
		for i := range parts {
			parts[i] = fmt.Sprintf("%s.r%d", t, i)
		}
		g.line("auto " + t + " = " + g.expr(s.Results[0]) + ";")
		g.line(d.ret + " = " + g.resType + "{" + strings.Join(parts, ", ") + "};")
	}
	g.line("goto " + d.label + ";")
	g.indent--
	g.line("}")
}

// deferStmt translates defer f(args): the function value and the arguments
// are evaluated now, into a closure that the function's deferred calls make.
func (g *cgen) deferStmt(s *ast.DeferStmt) {
	d := g.defers
	if d == nil {
		g.fail(s, "internal error: a defer statement outside a function with deferred calls")
	}
	call := s.Call
	if lit, ok := ast.Unparen(call.Fun).(*ast.FuncLit); ok && len(call.Args) == 0 {
		// defer func() {...}(): the literal's closure is the deferred call.
		g.line(fmt.Sprintf("%s.push(%s, %d);", d.frame, g.funcLit(lit), g.oom(s)))
		return
	}
	// What is evaluated now goes into the closure: field i is fenv_->ai.
	var fields, inits []string
	keep := func(e ast.Expr, ct, val string) string {
		i := len(fields)
		fields = append(fields, fmt.Sprintf("%s a%d;", ct, i))
		inits = append(inits, val)
		return fmt.Sprintf("fenv_->a%d", i)
	}
	subst := map[ast.Expr]string{}
	var sig *types.Signature // the called function's, for a call through a function value
	var callee *types.Func   // a function or method of the program called directly
	builtin := ""
	tv := g.sh.info.Types[call.Fun]
	switch fun := ast.Unparen(call.Fun).(type) {
	case *ast.FuncLit:
		sig = g.typeOf(fun).(*types.Signature)
		subst[call.Fun] = keep(call.Fun, "tsr::Func", g.funcLit(fun))
	case *ast.Ident:
		switch obj := g.sh.info.Uses[fun].(type) {
		case *types.Builtin:
			builtin = obj.Name()
			switch builtin {
			case "close", "delete", "panic", "recover", "copy", "clear":
			default:
				g.fail(s, "defer %s is not supported on the card; defer a function literal that calls it", builtin)
			}
		case *types.Func:
			callee = obj
		}
	case *ast.SelectorExpr:
		if id, ok := fun.X.(*ast.Ident); ok {
			if _, ok := g.sh.info.Uses[id].(*types.PkgName); ok {
				break // a library function: its arguments are kept
			}
		}
		if sel, ok := g.sh.info.Selections[fun]; ok && sel.Kind() == types.MethodVal {
			if m := sel.Obj().(*types.Func); m.Pkg() == g.sh.pkg || isIfaceMethod(m) {
				// x.m(...): the method value, now.
				sig, _ = types.Unalias(g.typeOf(fun)).Underlying().(*types.Signature)
				subst[call.Fun] = keep(call.Fun, "tsr::Func", g.methodValue(fun, sel))
			} else {
				// A method of a tt type: its receiver, now.
				subst[fun.X] = keep(fun.X, g.ct(g.typeOf(fun.X), fun.X), g.expr(fun.X))
			}
		}
	}
	if sig == nil && callee == nil && builtin == "" && !tv.IsType() {
		if s2, ok := valueCall(g.sh.info, call, g.sh.lambdaVar); ok {
			sig = s2 // through a function value
			subst[call.Fun] = keep(call.Fun, "tsr::Func", g.expr(call.Fun))
		}
	}
	// The arguments, in order (constants and nil need no keeping).
	for _, a := range call.Args {
		if tv := g.sh.info.Types[a]; tv.Value != nil || tv.IsNil() {
			continue
		}
		if isArray(g.typeOf(a)) {
			g.fail(a, "deferred calls cannot take arrays on the card; pass a slice (a[:])")
		}
		subst[a] = keep(a, g.ct(g.typeOf(a), a), g.exprAs(a, g.typeOf(a)))
	}

	// The call, made later from the closure's fields.
	saved := g.subst
	g.subst = map[ast.Expr]string{}
	for k, v := range saved {
		g.subst[k] = v
	}
	for k, v := range subst {
		g.subst[k] = v
	}
	var text string
	switch {
	case builtin == "recover":
		text = "(void)0" // recover called by defer itself recovers nothing
	case builtin != "":
		text = g.builtin(builtin, call)
	case callee != nil:
		args := []string{}
		if sg := callee.Type().(*types.Signature); sg.Params().Len() > 0 {
			args = append(args, g.callArgs(sg, call.Args, call.Ellipsis.IsValid(), call))
		}
		if g.sh.recovers(callee) {
			args = append(args, "true")
		}
		text = "prog::" + funcName(callee) + "(" + strings.Join(args, ", ") + ")"
	case sig != nil:
		args := []string{subst[call.Fun], fmt.Sprint(g.nilPos(call))}
		if sig.Params().Len() > 0 {
			args = append(args, g.callArgs(sig, call.Args, call.Ellipsis.IsValid(), call))
		}
		text = g.fcallFnD(sig) + "(" + strings.Join(args, ", ") + ")"
	default:
		text = g.call(call) // a library function, or a method of a tt type
	}
	g.subst = saved

	id := len(g.sh.fnSources) + 1
	name := fmt.Sprintf("defer_%d", id)
	env := name + "_env"
	src := g.source(s, funcSig0, "defer at "+g.litWhere(s), func(h *cgen, f string, args []string) string {
		if len(fields) > 0 {
			return name + "(static_cast<" + env + "*>(" + f + ".env))"
		}
		return name + "()"
	})
	var b, def strings.Builder
	params := "()"
	if len(fields) > 0 {
		fmt.Fprintf(&b, "// What the defer statement at %s keeps.\nstruct %s {\n", g.litWhere(s), env)
		for _, f := range fields {
			b.WriteString("    " + f + "\n")
		}
		b.WriteString("};\n")
		params = "(" + env + "* fenv_)"
	}
	fmt.Fprintf(&b, "static void %s%s;", name, params)
	fmt.Fprintf(&def, "// The call that the defer statement at %s makes.\nstatic void %s%s {\n    %s;\n}\n", g.litWhere(s), name, params, text)
	g.sh.litProtos = append(g.sh.litProtos, b.String())
	g.sh.litDefs = append(g.sh.litDefs, def.String())
	value := fmt.Sprintf("tsr::Func{%d, nullptr}", src.id)
	if len(fields) > 0 {
		value = fmt.Sprintf("tsr::Func{%d, tsr::box<prog::%s>(prog::%s{%s}, %d)}", src.id, env, env, strings.Join(inits, ", "), g.oom(s))
	}
	g.line(fmt.Sprintf("%s.push(%s, %d);", d.frame, value, g.oom(s)))
}

// fcallFnD returns the name of the dispatch function for deferred calls of
// function values of signature sig: it lets the function called recover.
func (g *cgen) fcallFnD(sig *types.Signature) string {
	for _, k := range g.sh.fnCalls {
		if k.deferred && types.Identical(k.sig, sig) {
			return k.fn
		}
	}
	fn := fmt.Sprintf("prog::fcall_%d_defer", len(g.sh.fnCalls))
	g.sh.fnCalls = append(g.sh.fnCalls, &fnCall{sig: sig, fn: fn, deferred: true})
	return fn
}

// recoverCall translates recover().
func (g *cgen) recoverCall() string {
	if !g.sh.usesDefer || !g.canRecover {
		return "tsr::Iface{}" // not in a function a deferred call calls
	}
	g.sh.usesIface = true
	return "tsr::recover(can_recover_)"
}

// panicCall translates panic(v).
func (g *cgen) panicCall(x *ast.CallExpr) string {
	pos := g.sh.position(x, "panic("+types.ExprString(x.Args[0])+")")
	if !g.sh.usesDefer || g.mode != deviceMode {
		return fmt.Sprintf("tsr::fail(%d, 0)", pos)
	}
	// Its value, for recover.
	return fmt.Sprintf("tsr::panic_with(%d, %s)", pos, g.exprAs(x.Args[0], types.Universe.Lookup("any").Type()))
}
