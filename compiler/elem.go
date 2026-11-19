package compiler

import (
	"go/ast"
	"go/constant"
	"go/token"
	"go/types"
	"sort"
)

// Element functions are translated by symbolic execution: every float32 or
// bool variable holds a graph node, statements update the environment, and
// the function's result is a single node. An if without a return merges the
// variables it assigns with Where; an if containing a return is handled by
// evaluating the rest of the function once on each side and selecting the
// result with Where. Nothing branches on the card: both sides are computed
// for all lanes and the right one is picked per element.

// builtins are the tt math functions element code may call.
var builtins = map[string]Op{
	"Exp": OpExp, "Log": OpLog, "Sqrt": OpSqrt, "Rsqrt": OpRsqrt, "Recip": OpRecip,
	"Abs": OpAbs, "Relu": OpRelu, "Sigmoid": OpSigmoid, "Tanh": OpTanh,
	"Sin": OpSin, "Cos": OpCos, "Square": OpSquare, "Max": OpMax, "Min": OpMin,
}

const maxInlineDepth = 32

type env map[types.Object]*Node

func (e env) clone() env {
	out := make(env, len(e))
	for k, v := range e {
		out[k] = v
	}
	return out
}

type elemTranslator struct {
	c     *checker
	stack []*types.Func // functions being inlined, to reject recursion
}

// apply translates a call of the element function fn on args. site is the
// call (for positions in messages).
func (c *checker) apply(fn ast.Expr, args []*Node, site ast.Node) *Node {
	t := &elemTranslator{c: c}
	return t.call(fn, args, site)
}

func (t *elemTranslator) call(fn ast.Expr, args []*Node, site ast.Node) *Node {
	c := t.c
	if f := c.ttFunc(fn); f != nil {
		op, ok := builtins[f.Name()]
		if !ok {
			c.fail(fn, "tt.%s cannot be used in element code", f.Name())
		}
		if len(args) != opTable[op].arity {
			c.fail(site, "tt.%s takes %d arguments", f.Name(), opTable[op].arity)
		}
		return c.g.node(op, c.pos(site), types.ExprString(site.(ast.Expr)), args...)
	}

	var ftype *ast.FuncType
	var body *ast.BlockStmt
	var fobj *types.Func
	switch f := ast.Unparen(fn).(type) {
	case *ast.FuncLit:
		ftype, body = f.Type, f.Body
	case *ast.Ident:
		obj, ok := c.info.Uses[f].(*types.Func)
		if !ok || obj.Pkg() != c.pkg {
			c.fail(fn, "%s is not a function of this program; element code can call its own functions and tt math functions", f.Name)
		}
		decl := c.funcs[obj]
		if decl == nil || decl.Body == nil {
			c.fail(fn, "%s has no body", f.Name)
		}
		if obj.Name() == "main" {
			c.fail(fn, "element code cannot call main")
		}
		ftype, body, fobj = decl.Type, decl.Body, obj
	default:
		c.fail(fn, "pass a function name or a func literal here")
	}

	if fobj != nil {
		for _, active := range t.stack {
			if active == fobj {
				c.fail(site, "%s is recursive; element functions cannot recurse on the card", fobj.Name())
			}
		}
		t.stack = append(t.stack, fobj)
		defer func() { t.stack = t.stack[:len(t.stack)-1] }()
	}
	if len(t.stack) > maxInlineDepth {
		c.fail(site, "element functions nest more than %d calls deep", maxInlineDepth)
	}

	// Bind parameters.
	e := env{}
	i := 0
	for _, field := range ftype.Params.List {
		c.checkValueType(field.Type, c.info.TypeOf(field.Type), "parameter")
		if len(field.Names) == 0 {
			i++
			continue
		}
		for _, name := range field.Names {
			if obj := c.info.Defs[name]; obj != nil {
				e[obj] = args[i]
			}
			i++
		}
	}
	if i != len(args) {
		c.fail(site, "function takes %d arguments, called with %d", i, len(args))
	}
	if ftype.Results == nil || len(ftype.Results.List) != 1 || len(ftype.Results.List[0].Names) > 1 {
		c.fail(ftype, "element functions return exactly one float32 or bool value")
	}
	c.checkValueType(ftype.Results.List[0].Type, c.info.TypeOf(ftype.Results.List[0].Type), "result")
	if len(ftype.Results.List[0].Names) == 1 {
		c.fail(ftype.Results.List[0].Names[0], "named results are not supported; use a plain return")
	}

	result, returned := t.stmts(body.List, e)
	if !returned {
		c.fail(body, "missing return")
	}
	return result
}

// checkValueType rejects types other than float32 and bool.
func (c *checker) checkValueType(n ast.Node, typ types.Type, what string) {
	if !isValueType(typ) {
		c.fail(n, "%s has type %s; element code works on float32 and bool values only", what, typ)
	}
}

func isValueType(typ types.Type) bool {
	b, ok := typ.Underlying().(*types.Basic)
	if !ok {
		return false
	}
	switch b.Kind() {
	case types.Float32, types.Bool, types.UntypedBool, types.UntypedFloat, types.UntypedInt, types.UntypedRune:
		return true
	}
	return false
}

// stmts executes a statement list. It returns the function result if the
// list returns.
func (t *elemTranslator) stmts(list []ast.Stmt, e env) (*Node, bool) {
	c := t.c
	for i := 0; i < len(list); i++ {
		switch s := list[i].(type) {
		case *ast.ReturnStmt:
			if len(s.Results) != 1 {
				c.fail(s, "return exactly one value")
			}
			return t.expr(s.Results[0], e), true

		case *ast.AssignStmt:
			t.assign(s, e)

		case *ast.IncDecStmt:
			id := t.lhsIdent(s.X)
			if id == nil {
				continue
			}
			obj := c.info.ObjectOf(id)
			op := OpAdd
			if s.Tok == token.DEC {
				op = OpSub
			}
			e[obj] = c.g.node(op, c.pos(s), types.ExprString(s.X)+s.Tok.String(), t.lookup(id, e), c.g.constant(1, c.pos(s)))

		case *ast.DeclStmt:
			t.decl(s, e)

		case *ast.BlockStmt:
			// Splice the block into the current list; scoping is already
			// resolved by the type checker, so the objects stay distinct.
			rest := append(append([]ast.Stmt{}, s.List...), list[i+1:]...)
			return t.stmts(rest, e)

		case *ast.IfStmt:
			if s.Init != nil {
				if r, ok := t.stmts([]ast.Stmt{s.Init}, e); ok {
					return r, true
				}
			}
			cond := t.expr(s.Cond, e)
			rest := list[i+1:]
			thenList := s.Body.List
			var elseList []ast.Stmt
			switch el := s.Else.(type) {
			case *ast.BlockStmt:
				elseList = el.List
			case *ast.IfStmt:
				elseList = []ast.Stmt{el}
			}
			if containsReturn(s) {
				// Run the rest of the function on each side, then select.
				thenRes, ok1 := t.stmts(concat(thenList, rest), e.clone())
				elseRes, ok2 := t.stmts(concat(elseList, rest), e.clone())
				if !ok1 || !ok2 {
					c.fail(s, "missing return")
				}
				return c.g.node(OpWhere, c.pos(s), "if "+types.ExprString(s.Cond), cond, thenRes, elseRes), true
			}
			thenEnv, elseEnv := e.clone(), e.clone()
			t.stmts(thenList, thenEnv)
			t.stmts(elseList, elseEnv)
			for _, obj := range sortedObjects(e) {
				if a, b := thenEnv[obj], elseEnv[obj]; a != b {
					e[obj] = c.g.node(OpWhere, c.pos(s), obj.Name()+" after if "+types.ExprString(s.Cond), cond, a, b)
				}
			}

		case *ast.EmptyStmt:

		case *ast.ForStmt, *ast.RangeStmt:
			c.fail(s, "loops are not supported in element code yet; unroll the loop, or split it into several Map calls")
		case *ast.SwitchStmt:
			c.fail(s, "switch is not supported in element code yet; use if / else if")
		case *ast.ExprStmt:
			c.fail(s, "this statement has no effect on the card (element code has no side effects)")
		default:
			c.fail(s, "this statement is not supported in element code")
		}
	}
	return nil, false
}

func concat(a, b []ast.Stmt) []ast.Stmt {
	return append(append([]ast.Stmt{}, a...), b...)
}

func containsReturn(n ast.Node) bool {
	found := false
	ast.Inspect(n, func(x ast.Node) bool {
		switch x.(type) {
		case *ast.ReturnStmt:
			found = true
		case *ast.FuncLit:
			return false
		}
		return !found
	})
	return found
}

func sortedObjects(e env) []types.Object {
	objs := make([]types.Object, 0, len(e))
	for obj := range e {
		objs = append(objs, obj)
	}
	sort.Slice(objs, func(i, j int) bool { return objs[i].Pos() < objs[j].Pos() })
	return objs
}

func (t *elemTranslator) lhsIdent(x ast.Expr) *ast.Ident {
	id, ok := ast.Unparen(x).(*ast.Ident)
	if !ok {
		t.c.fail(x, "only plain variables can be assigned in element code")
	}
	if id.Name == "_" {
		return nil
	}
	obj := t.c.info.ObjectOf(id)
	if v, ok := obj.(*types.Var); !ok || v.Parent() == t.c.pkg.Scope() {
		t.c.fail(x, "%s cannot be assigned in element code", id.Name)
	}
	t.c.checkValueType(id, obj.Type(), id.Name)
	return id
}

var assignOps = map[token.Token]Op{
	token.ADD_ASSIGN: OpAdd, token.SUB_ASSIGN: OpSub,
	token.MUL_ASSIGN: OpMul, token.QUO_ASSIGN: OpDiv,
}

func (t *elemTranslator) assign(s *ast.AssignStmt, e env) {
	c := t.c
	switch s.Tok {
	case token.DEFINE, token.ASSIGN:
		if len(s.Lhs) != len(s.Rhs) {
			c.fail(s, "assign one value per variable")
		}
		vals := make([]*Node, len(s.Rhs))
		for i, r := range s.Rhs {
			vals[i] = t.expr(r, e)
		}
		for i, l := range s.Lhs {
			if id := t.lhsIdent(l); id != nil {
				e[c.info.ObjectOf(id)] = vals[i]
			}
		}
	default:
		op, ok := assignOps[s.Tok]
		if !ok {
			c.fail(s, "%s is not supported in element code", s.Tok)
		}
		id := t.lhsIdent(s.Lhs[0])
		if id == nil {
			return
		}
		rhs := t.expr(s.Rhs[0], e)
		e[c.info.ObjectOf(id)] = c.g.node(op, c.pos(s), types.ExprString(s.Lhs[0])+" "+s.Tok.String()+" "+types.ExprString(s.Rhs[0]), t.lookup(id, e), rhs)
	}
}

func (t *elemTranslator) decl(s *ast.DeclStmt, e env) {
	c := t.c
	gd := s.Decl.(*ast.GenDecl)
	switch gd.Tok {
	case token.CONST:
		return // folded by the type checker
	case token.VAR:
		for _, spec := range gd.Specs {
			vs := spec.(*ast.ValueSpec)
			for i, name := range vs.Names {
				obj := c.info.Defs[name]
				if obj == nil {
					continue
				}
				c.checkValueType(name, obj.Type(), name.Name)
				if len(vs.Values) == 0 {
					e[obj] = c.g.constant(0, c.pos(name)) // the zero value
				} else {
					e[obj] = t.expr(vs.Values[i], e)
				}
			}
		}
	default:
		c.fail(s, "type declarations are not supported in element code")
	}
}

func (t *elemTranslator) lookup(id *ast.Ident, e env) *Node {
	n, ok := e[t.c.info.ObjectOf(id)]
	if !ok {
		t.c.fail(id, "%s is not available on the card; element code can use its parameters, its locals and constants", id.Name)
	}
	return n
}

var binaryOps = map[token.Token]Op{
	token.ADD: OpAdd, token.SUB: OpSub, token.MUL: OpMul, token.QUO: OpDiv,
	token.LSS: OpLt, token.GTR: OpGt, token.LEQ: OpLe, token.GEQ: OpGe,
	token.EQL: OpEq, token.NEQ: OpNe,
	token.LAND: OpMul, // on 0/1 masks
	token.LOR:  OpMax,
}

func (t *elemTranslator) expr(x ast.Expr, e env) *Node {
	c := t.c
	tv := c.info.Types[x]
	pos := c.pos(x)
	if tv.Value != nil {
		switch tv.Value.Kind() {
		case constant.Bool:
			return c.g.constant(b2f(constant.BoolVal(tv.Value)), pos)
		case constant.Int, constant.Float:
			v, _ := constant.Float32Val(constant.ToFloat(tv.Value))
			return c.g.constant(v, pos)
		default:
			c.fail(x, "constant of kind %s cannot be used in element code", tv.Value.Kind())
		}
	}
	if tv.Type != nil && !isValueType(tv.Type) {
		c.fail(x, "%s has type %s; element code works on float32 and bool values only", types.ExprString(x), tv.Type)
	}
	src := types.ExprString(x)

	switch x := x.(type) {
	case *ast.ParenExpr:
		return t.expr(x.X, e)
	case *ast.Ident:
		return t.lookup(x, e)
	case *ast.UnaryExpr:
		a := t.expr(x.X, e)
		switch x.Op {
		case token.SUB:
			return c.g.node(OpNeg, pos, src, a)
		case token.ADD:
			return a
		case token.NOT:
			return c.g.node(OpSub, pos, src, c.g.constant(1, pos), a)
		}
		c.fail(x, "operator %s is not supported in element code", x.Op)
	case *ast.BinaryExpr:
		op, ok := binaryOps[x.Op]
		if !ok {
			c.fail(x, "operator %s is not supported in element code", x.Op)
		}
		a, b := t.expr(x.X, e), t.expr(x.Y, e)
		return c.g.node(op, pos, src, a, b)
	case *ast.CallExpr:
		if c.info.Types[x.Fun].IsType() {
			// A conversion such as float32(v): values are float32 already.
			return t.expr(x.Args[0], e)
		}
		if f := c.ttFunc(x.Fun); f != nil && (f.Name() == "Map" || f.Name() == "Map2" || f.Name() == "Map3" || f.Name() == "Load" || f.Name() == "Store") {
			c.fail(x, "tt.%s can only be called from main", f.Name())
		}
		args := make([]*Node, len(x.Args))
		for i, a := range x.Args {
			args[i] = t.expr(a, e)
		}
		return t.call(x.Fun, args, x)
	}
	c.fail(x, "this expression is not supported in element code")
	return nil
}
