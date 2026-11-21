package compiler

import (
	"go/ast"
	"go/token"
	"go/types"
	"sort"
)

// Pointers on the card.
//
// A Go pointer is a C++ pointer into the core's own memory: the cores share
// none, so pointers never leave the core (they cannot be results, cross a
// channel or come from main). Going through a nil pointer panics, as in Go.
//
// What a pointer points to must live as long as the pointer. new(T) and
// &T{...} allocate on the heap. A variable whose address is taken stays
// where it is (the core's stack, or L1 scratch for an array) if the address
// cannot outlive the call: if it is only passed to functions and methods
// that do not keep it, compared, or gone through at once. Otherwise the
// variable lives on the heap, as Go's escape analysis would put it there.
// The analysis is simpler than Go's and errs towards the heap: a pointer
// parameter that is copied into a variable, returned or stored counts as
// kept.
//
// A method with a pointer receiver takes it as a C++ reference, as before
// pointers: the receiver can then never be nil. If any call passes it a
// pointer that could be nil, or the method assigns to its receiver, the
// method takes a C++ pointer instead, and goes through it with a nil check.

// ptrAnalysis finds what the code generator needs to know about pointers in
// the card's code: which variables live on the heap, which pointer
// parameters can outlive the call, and which methods take a C++ pointer.
type ptrAnalysis struct {
	info   *types.Info
	pkg    *types.Package
	funcs  map[*types.Func]*ast.FuncDecl
	parent map[ast.Node]ast.Node
	uses   map[*types.Var][]*ast.Ident // the uses of each variable, in the card's code
	recvOf map[*types.Var]*types.Func  // pointer receivers: their methods
	leaks  map[*types.Var]bool         // pointer parameters and receivers whose value can outlive the call
	ptrRcv map[*types.Func]bool        // methods that take their receiver as a C++ pointer
	result map[*types.Var]bool         // named results
}

func isPointer(t types.Type) bool {
	if t == nil {
		return false
	}
	_, ok := types.Unalias(t).Underlying().(*types.Pointer)
	return ok
}

func isArray(t types.Type) bool {
	_, _, ok := arrayLen(t)
	return ok
}

// pointerRecv reports whether method m has a pointer receiver.
func pointerRecv(m *types.Func) bool {
	recv := m.Type().(*types.Signature).Recv()
	return recv != nil && isPointer(recv.Type())
}

// computePointers runs the analysis over the card's code: the functions in
// fns and the other bodies (the worker, the initialization).
func computePointers(sh *cgShared, fns []*types.Func, bodies []ast.Node) {
	a := &ptrAnalysis{info: sh.info, pkg: sh.pkg, funcs: sh.funcs, parent: map[ast.Node]ast.Node{},
		uses: map[*types.Var][]*ast.Ident{}, recvOf: map[*types.Var]*types.Func{},
		leaks: map[*types.Var]bool{}, ptrRcv: map[*types.Func]bool{}, result: map[*types.Var]bool{}}
	var roots []ast.Node
	for _, f := range fns {
		roots = append(roots, sh.funcs[f])
		if pointerRecv(f) {
			if r := f.Type().(*types.Signature).Recv(); r != nil {
				a.recvOf[r] = f
			}
		}
	}
	roots = append(roots, bodies...)
	var params []*types.Var // pointer parameters and receivers
	for _, r := range roots {
		var stack []ast.Node
		ast.Inspect(r, func(n ast.Node) bool {
			if n == nil {
				stack = stack[:len(stack)-1]
				return true
			}
			if len(stack) > 0 {
				a.parent[n] = stack[len(stack)-1]
			}
			stack = append(stack, n)
			switch x := n.(type) {
			case *ast.Ident:
				if v, ok := a.info.Uses[x].(*types.Var); ok && !v.IsField() && !isPkgVar(v) {
					a.uses[v] = append(a.uses[v], x)
				}
			case *ast.FuncType:
				for _, fl := range []*ast.FieldList{x.Params} {
					if fl == nil {
						continue
					}
					for _, f := range fl.List {
						for _, id := range f.Names {
							if v, ok := a.info.Defs[id].(*types.Var); ok && isPointer(v.Type()) {
								params = append(params, v)
							}
						}
					}
				}
				if x.Results != nil {
					for _, f := range x.Results.List {
						for _, id := range f.Names {
							if v, ok := a.info.Defs[id].(*types.Var); ok {
								a.result[v] = true
							}
						}
					}
				}
			case *ast.FuncDecl:
				if x.Recv != nil {
					for _, f := range x.Recv.List {
						for _, id := range f.Names {
							if v, ok := a.info.Defs[id].(*types.Var); ok && isPointer(v.Type()) {
								params = append(params, v)
							}
						}
					}
				}
			}
			return true
		})
	}
	sort.Slice(params, func(i, j int) bool { return params[i].Pos() < params[j].Pos() })

	// What the function literals that become values use from outside: a
	// pointer they copy can outlive the call.
	sh.captures = a.captures(sh.valueLit)
	for _, caps := range sh.captures {
		for _, c := range caps {
			if isPointer(c.v.Type()) {
				a.leaks[c.v] = true
			}
		}
	}

	// Which pointer parameters can outlive the call, until nothing changes.
	for changed := true; changed; {
		changed = false
		for _, v := range params {
			if a.leaks[v] {
				continue
			}
			for _, id := range a.uses[v] {
				if a.leakPtr(id) {
					a.leaks[v], changed = true, true
					break
				}
			}
		}
	}

	// Which methods take a C++ pointer: those that change their receiver
	// variable, those called with a pointer that could be nil, and those a
	// call through an interface can reach (which can hold a nil pointer).
	for m := range sh.viaIface {
		if pointerRecv(m) {
			a.ptrRcv[m] = true
		}
	}
	// And those of method values and method expressions, which can hold one.
	for _, r := range roots {
		ast.Inspect(r, func(n ast.Node) bool {
			if sel, ok := n.(*ast.SelectorExpr); ok {
				if s, ok := a.info.Selections[sel]; ok && (s.Kind() == types.MethodExpr || (s.Kind() == types.MethodVal && !a.callee(sel))) {
					if m, ok := s.Obj().(*types.Func); ok && m.Pkg() == a.pkg && pointerRecv(m) {
						a.ptrRcv[m.Origin()] = true
					}
				}
			}
			return true
		})
	}
	for v, m := range a.recvOf {
		for _, id := range a.uses[v] {
			if a.assigned(id) {
				a.ptrRcv[m] = true
			}
		}
	}
	for changed := true; changed; {
		changed = false
		for _, r := range roots {
			ast.Inspect(r, func(n ast.Node) bool {
				call, ok := n.(*ast.CallExpr)
				if !ok {
					return true
				}
				sel, ok := ast.Unparen(call.Fun).(*ast.SelectorExpr)
				if !ok {
					return true
				}
				s, ok := a.info.Selections[sel]
				if !ok || s.Kind() != types.MethodVal {
					return true
				}
				m, ok := s.Obj().(*types.Func)
				if !ok || m.Pkg() != a.pkg || !pointerRecv(m) || a.ptrRcv[m] {
					return true
				}
				if a.nilable(sel.X, s) {
					a.ptrRcv[m], changed = true, true
				}
				return true
			})
		}
	}

	// The variables whose address can outlive the call live on the heap.
	sh.heap = map[*types.Var]bool{}
	for v, ids := range a.uses {
		for _, id := range ids {
			if a.leakVal(id) {
				sh.heap[v] = true
				break
			}
		}
	}
	// So do the variables a function value uses by address, and its arrays.
	for _, caps := range sh.captures {
		for _, c := range caps {
			if c.byRef || isArray(c.v.Type()) {
				sh.heap[c.v] = true
			}
		}
	}
	sh.ptrRecv = a.ptrRcv
}

// callee reports whether e is the function a call calls.
func (a *ptrAnalysis) callee(e ast.Node) bool {
	return isCallee(a.parent, e)
}

// captures finds, for each function literal in lits, the variables it uses
// that are declared outside it: by address if anything assigns to them (or
// to a part of them) after they are declared, or takes their address, else
// as a copy. A named result can change at a return.
func (a *ptrAnalysis) captures(lits map[*ast.FuncLit]bool) map[*ast.FuncLit][]litCapture {
	out := map[*ast.FuncLit][]litCapture{}
	var vars []*types.Var
	for v := range a.uses {
		vars = append(vars, v)
	}
	sort.Slice(vars, func(i, j int) bool { return vars[i].Pos() < vars[j].Pos() })
	written := map[*types.Var]bool{}
	for _, v := range vars {
		for _, id := range a.uses[v] {
			if a.writes(id) {
				written[v] = true
				break
			}
		}
	}
	for lit := range lits {
		in := func(p token.Pos) bool { return p >= lit.Pos() && p < lit.End() }
		for _, v := range vars {
			if in(v.Pos()) {
				continue
			}
			for _, id := range a.uses[v] {
				if in(id.Pos()) {
					out[lit] = append(out[lit], litCapture{v: v, byRef: written[v] || a.result[v]})
					break
				}
			}
		}
	}
	return out
}

// writes reports whether the use id of a variable can change it: assign to
// it or to a field or an element of it, take its address (also for a
// method with a pointer receiver), or make a slice of its array.
func (a *ptrAnalysis) writes(id *ast.Ident) bool {
	var e ast.Node = id
	for {
		switch x := a.parent[e].(type) {
		case *ast.ParenExpr:
			e = x
			continue
		case *ast.SelectorExpr:
			if x.X != e || isPointer(a.info.TypeOf(e.(ast.Expr))) {
				return false // through a pointer: not the variable
			}
			s, ok := a.info.Selections[x]
			if !ok {
				return false
			}
			if s.Kind() == types.FieldVal {
				e = x
				continue
			}
			m, ok := s.Obj().(*types.Func)
			return ok && !isIfaceMethod(m) && pointerRecv(m)
		case *ast.IndexExpr:
			if x.X == e && isArray(a.info.TypeOf(e.(ast.Expr))) {
				e = x
				continue
			}
			return false
		case *ast.SliceExpr:
			return x.X == e && isArray(a.info.TypeOf(e.(ast.Expr)))
		case *ast.AssignStmt:
			for _, l := range x.Lhs {
				if l == e {
					return true
				}
			}
		case *ast.IncDecStmt:
			return true
		case *ast.UnaryExpr:
			return x.Op == token.AND
		case *ast.RangeStmt:
			return x.Key == e || x.Value == e
		}
		return false
	}
}

// nilable reports whether the receiver that a call x.m(...) passes, through
// the embedded fields of selection s, is a pointer that could be nil: not
// an addressable value, and not the receiver of a method that takes it as
// a C++ reference.
func (a *ptrAnalysis) nilable(x ast.Expr, s *types.Selection) bool {
	t := a.info.TypeOf(x)
	idx := s.Index()
	for _, i := range idx[:len(idx)-1] {
		if p, ok := types.Unalias(t).Underlying().(*types.Pointer); ok {
			t = p.Elem()
		}
		t = types.Unalias(t).Underlying().(*types.Struct).Field(i).Type()
	}
	if !isPointer(t) {
		return false
	}
	if len(idx) == 1 {
		if id, ok := ast.Unparen(x).(*ast.Ident); ok {
			if v, ok := a.info.Uses[id].(*types.Var); ok {
				if m, ok := a.recvOf[v]; ok && !a.ptrRcv[m] {
					return false
				}
			}
		}
	}
	return true
}

// assigned reports whether the use id of a variable assigns to it or takes
// its address.
func (a *ptrAnalysis) assigned(id *ast.Ident) bool {
	var e ast.Node = id
	for {
		p := a.parent[e]
		switch x := p.(type) {
		case *ast.ParenExpr:
			e = x
			continue
		case *ast.AssignStmt:
			for _, l := range x.Lhs {
				if l == e {
					return true
				}
			}
		case *ast.IncDecStmt:
			return true
		case *ast.UnaryExpr:
			return x.Op == token.AND
		case *ast.RangeStmt:
			return x.Key == e || x.Value == e
		}
		return false
	}
}

// leakPtr reports whether the pointer value of expression e can outlive the
// call it is in: whether it is stored, returned, copied into a variable, or
// passed where it can be.
func (a *ptrAnalysis) leakPtr(e ast.Node) bool {
	switch x := a.parent[e].(type) {
	case *ast.ParenExpr:
		return a.leakPtr(x)
	case *ast.SelectorExpr:
		if x.X != e {
			return true
		}
		s, ok := a.info.Selections[x]
		if !ok {
			return true
		}
		if s.Kind() == types.FieldVal {
			return a.leakVal(x) // a field of what e points to
		}
		if !a.callee(x) {
			return true // a method value keeps it
		}
		return a.recvLeaks(s) // e.m(...)
	case *ast.StarExpr:
		return a.leakVal(x)
	case *ast.IndexExpr:
		if x.X == e {
			return a.leakVal(x) // p[i] for a pointer to an array
		}
		return true // a map's key
	case *ast.BinaryExpr:
		return false // == and !=
	case *ast.AssignStmt:
		for _, l := range x.Lhs {
			if l == e {
				return false // assigned to: the old value is gone
			}
		}
		if len(x.Lhs) == len(x.Rhs) {
			for i, r := range x.Rhs {
				if r == e {
					if id, ok := x.Lhs[i].(*ast.Ident); ok && id.Name == "_" {
						return false
					}
				}
			}
		}
		return true
	case *ast.CallExpr:
		return a.argLeaks(x, e)
	case *ast.IfStmt, *ast.ExprStmt:
		return false
	}
	return true
}

// leakVal reports whether the address of v, an addressable value (a
// variable, a field, an array element or *p), can outlive the call: taken
// with & where it can, or passed to a method with a pointer receiver that
// keeps it.
func (a *ptrAnalysis) leakVal(v ast.Node) bool {
	switch x := a.parent[v].(type) {
	case *ast.ParenExpr:
		return a.leakVal(x)
	case *ast.SelectorExpr:
		if x.X != v {
			return false
		}
		if isPointer(a.info.TypeOf(v.(ast.Expr))) {
			return false // v.f goes through the pointer v, to memory elsewhere
		}
		s, ok := a.info.Selections[x]
		if !ok {
			return false
		}
		if s.Kind() == types.FieldVal {
			return a.leakVal(x)
		}
		m, ok := s.Obj().(*types.Func)
		if !ok || !pointerRecv(m) {
			return false // a value receiver gets a copy
		}
		// v.m(): the method gets &v, or the address of an embedded field
		// of v (if the path does not go through an embedded pointer).
		t := a.info.TypeOf(v.(ast.Expr))
		idx := s.Index()
		for _, i := range idx[:len(idx)-1] {
			t = types.Unalias(t).Underlying().(*types.Struct).Field(i).Type()
			if isPointer(t) {
				return false
			}
		}
		if !a.callee(x) {
			return true // a method value keeps it
		}
		return a.recvLeaks(s)
	case *ast.IndexExpr:
		if x.X == v && isArray(a.info.TypeOf(v.(ast.Expr))) {
			return a.leakVal(x)
		}
		return false
	case *ast.UnaryExpr:
		if x.Op == token.AND {
			return a.leakPtr(x)
		}
	}
	return false
}

// recvLeaks reports whether the method of selection s keeps its pointer
// receiver.
func (a *ptrAnalysis) recvLeaks(s *types.Selection) bool {
	m, ok := s.Obj().(*types.Func)
	if !ok || !pointerRecv(m) {
		return false
	}
	if m.Pkg() != a.pkg {
		return false // tt.List.Append and the like
	}
	recv := m.Type().(*types.Signature).Recv()
	if _, ok := a.recvOf[recv]; !ok {
		return true // not in the card's code: assume the worst
	}
	return a.leaks[recv]
}

// argLeaks reports whether call keeps its argument e.
func (a *ptrAnalysis) argLeaks(call *ast.CallExpr, e ast.Node) bool {
	if call.Fun == e {
		return true
	}
	if tv := a.info.Types[call.Fun]; tv.IsType() {
		return a.leakPtr(call) // a conversion: the value goes on
	}
	k := -1
	for i, arg := range call.Args {
		if arg == e {
			k = i
		}
	}
	if k < 0 {
		return true
	}
	var sig *types.Signature
	switch fun := ast.Unparen(call.Fun).(type) {
	case *ast.Ident:
		switch obj := a.info.Uses[fun].(type) {
		case *types.Builtin:
			switch obj.Name() {
			case "len", "cap", "min", "max":
				return false
			}
			return true
		case *types.Func:
			if obj.Pkg() != a.pkg {
				return true
			}
			sig = obj.Type().(*types.Signature)
		}
	case *ast.SelectorExpr:
		if s, ok := a.info.Selections[fun]; ok && s.Kind() == types.MethodVal {
			if m, ok := s.Obj().(*types.Func); ok && m.Pkg() == a.pkg {
				sig = m.Type().(*types.Signature)
			}
		}
	case *ast.FuncLit:
		sig, _ = a.info.TypeOf(fun).(*types.Signature)
	}
	if sig == nil || call.Ellipsis.IsValid() {
		return true
	}
	if sig.Variadic() && k >= sig.Params().Len()-1 {
		return true // into the slice of the ... parameter
	}
	// Every function the card's code calls is in the analysis (reachable
	// takes them all); a parameter it never uses never leaks. A pointer
	// that goes into an interface parameter can go anywhere.
	p := sig.Params().At(k)
	if !isPointer(p.Type()) {
		return true
	}
	return a.leaks[p]
}
