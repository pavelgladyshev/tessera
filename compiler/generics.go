package compiler

import (
	"fmt"
	"go/ast"
	"go/printer"
	"go/token"
	"go/types"
	"reflect"
	"strconv"
	"strings"
)

// Generics on the card.
//
// tessc sees the whole program, so it instantiates the program's generic
// functions and types in the source, before anything else: each use of a
// generic function or type with type arguments gets a copy of its
// declaration (and of a type's methods) with the type arguments in place of
// the type parameters, as if written by hand, and the use names the copy.
// The program is then type-checked again, and the rest of tessc sees no
// generics. A copy is named after the generic and its type arguments,
// Mapᐸintˏstringᐳ (letters Go allows in names; Map_int_string_ in C++).
//
// A copy keeps the positions of the declaration it copies, so that
// messages point at the generic code, but in a file of its own (with the
// same name and lines), so that no two nodes share a position. The Go host
// program is built from the source files as written: Go instantiates
// generics there itself.

// monoBound is what a type parameter stands for in a copy: a type, whose
// own type parameters (those of the declaration that instantiated this
// one) stand for what env says.
type monoBound struct {
	t   types.Type
	env monoEnv
}

type monoEnv map[*types.TypeParam]monoBound

// monoInst is one instantiation of a generic function or type.
type monoInst struct {
	obj  types.Object // the generic function or type name
	name string       // the copy's name
	args []monoBound
}

type mono struct {
	fset   *token.FileSet
	info   *types.Info
	pkg    *types.Package
	fileOf map[ast.Decl]*ast.File
	funcs  map[*types.Func]*ast.FuncDecl
	types  map[*types.TypeName]*ast.TypeSpec
	meths  map[*types.TypeName][]*ast.FuncDecl // the methods of each generic type
	insts  map[string]*monoInst                // by key: the generic and its type arguments
	names  map[string]bool
	queue  []*monoInst
	added  map[*ast.File][]ast.Decl
	errs   ErrorList
	parent map[ast.Node]ast.Node

	genMeths map[*types.Func]bool // methods with type parameters of their own
}

// isGenericDecl reports whether d declares a generic function or a method
// of a generic type.
func isGenericDecl(d *ast.FuncDecl) bool {
	if d.Type.TypeParams != nil && len(d.Type.TypeParams.List) > 0 {
		return true
	}
	if d.Recv != nil && len(d.Recv.List) == 1 {
		t := d.Recv.List[0].Type
		if s, ok := t.(*ast.StarExpr); ok {
			t = s.X
		}
		switch ast.Unparen(t).(type) {
		case *ast.IndexExpr, *ast.IndexListExpr:
			return true
		}
	}
	return false
}

// monomorphize instantiates the generic functions and types of pkg that
// files use. It reports whether it changed the files (which must then be
// type-checked again).
func monomorphize(fset *token.FileSet, files []*ast.File, info *types.Info, pkg *types.Package) (bool, error) {
	m := &mono{fset: fset, info: info, pkg: pkg, fileOf: map[ast.Decl]*ast.File{},
		funcs: map[*types.Func]*ast.FuncDecl{}, types: map[*types.TypeName]*ast.TypeSpec{},
		meths: map[*types.TypeName][]*ast.FuncDecl{}, insts: map[string]*monoInst{},
		names: map[string]bool{}, added: map[*ast.File][]ast.Decl{}, parent: map[ast.Node]ast.Node{},
		genMeths: map[*types.Func]bool{}}
	used := false
	for id := range info.Instances {
		if obj := info.Uses[id]; obj != nil && obj.Pkg() == pkg {
			used = true
			break
		}
	}
	if !used {
		return false, nil
	}
	for _, obj := range pkg.Scope().Names() {
		m.names[obj] = true
	}
	for _, f := range files {
		for _, d := range f.Decls {
			m.fileOf[d] = f
			switch d := d.(type) {
			case *ast.FuncDecl:
				fn, _ := info.Defs[d.Name].(*types.Func)
				if fn == nil {
					continue
				}
				if d.Type.TypeParams != nil && len(d.Type.TypeParams.List) > 0 && d.Recv != nil {
					m.genMeths[fn] = true // a method with type parameters of its own (Go 1.27)
				} else if d.Type.TypeParams != nil && len(d.Type.TypeParams.List) > 0 {
					m.funcs[fn] = d
				} else if isGenericDecl(d) {
					if tn := recvTypeName(info, d); tn != nil {
						m.meths[tn] = append(m.meths[tn], d)
					}
				}
			case *ast.GenDecl:
				for _, s := range d.Specs {
					if ts, ok := s.(*ast.TypeSpec); ok && ts.TypeParams != nil && len(ts.TypeParams.List) > 0 {
						if tn, ok := info.Defs[ts.Name].(*types.TypeName); ok {
							m.types[tn] = ts
							m.fileOf[d] = f
						}
					}
				}
			}
		}
		var stack []ast.Node
		ast.Inspect(f, func(n ast.Node) bool {
			if n == nil {
				stack = stack[:len(stack)-1]
				return true
			}
			if len(stack) > 0 {
				m.parent[n] = stack[len(stack)-1]
			}
			stack = append(stack, n)
			return true
		})
	}

	for id, obj := range info.Uses {
		if f, ok := obj.(*types.Func); ok && m.genMeths[f.Origin()] {
			m.errorf(id.Pos(), "%s has type parameters of its own: generic methods are not supported on the card", f.Name())
			return true, m.errs
		}
	}
	// The program's own code: its uses of generics name the copies.
	for _, f := range files {
		for i, d := range f.Decls {
			switch d := d.(type) {
			case *ast.FuncDecl:
				if isGenericDecl(d) {
					continue
				}
			case *ast.GenDecl:
				if d.Tok == token.IMPORT || d.Tok == token.TYPE && m.genericSpecs(d) {
					continue
				}
			}
			f.Decls[i] = m.copyNode(d, 0, nil).(ast.Decl)
		}
	}
	// And the copies, which can instantiate more.
	for len(m.queue) > 0 && len(m.errs) == 0 {
		in := m.queue[0]
		m.queue = m.queue[1:]
		m.instantiate(in)
	}
	if len(m.errs) > 0 {
		return true, m.errs
	}
	for _, f := range files {
		f.Decls = append(f.Decls, m.added[f]...)
	}
	return true, nil
}

// genericSpecs reports whether a type declaration declares generic types
// (tessc keeps such declarations as they are; Go does not group them with
// other types in practice).
func (m *mono) genericSpecs(d *ast.GenDecl) bool {
	for _, s := range d.Specs {
		if ts, ok := s.(*ast.TypeSpec); ok && ts.TypeParams != nil && len(ts.TypeParams.List) > 0 {
			return true
		}
	}
	return false
}

// recvTypeName is the generic type whose method d is.
func recvTypeName(info *types.Info, d *ast.FuncDecl) *types.TypeName {
	t := d.Recv.List[0].Type
	if s, ok := t.(*ast.StarExpr); ok {
		t = s.X
	}
	var id ast.Expr
	switch x := ast.Unparen(t).(type) {
	case *ast.IndexExpr:
		id = x.X
	case *ast.IndexListExpr:
		id = x.X
	}
	if ident, ok := id.(*ast.Ident); ok {
		tn, _ := info.Uses[ident].(*types.TypeName)
		return tn
	}
	return nil
}

// recvTypeParams are the type parameters a method of a generic type
// declares in its receiver.
func recvTypeParams(info *types.Info, d *ast.FuncDecl) []*types.TypeParam {
	t := d.Recv.List[0].Type
	if s, ok := t.(*ast.StarExpr); ok {
		t = s.X
	}
	var idx []ast.Expr
	switch x := ast.Unparen(t).(type) {
	case *ast.IndexExpr:
		idx = []ast.Expr{x.Index}
	case *ast.IndexListExpr:
		idx = x.Indices
	}
	var out []*types.TypeParam
	for _, e := range idx {
		if id, ok := e.(*ast.Ident); ok {
			if tn, ok := info.Defs[id].(*types.TypeName); ok {
				if tp, ok := tn.Type().(*types.TypeParam); ok {
					out = append(out, tp)
				}
			}
		}
	}
	return out
}

// typeParams are the type parameters a field list declares.
func typeParams(info *types.Info, fl *ast.FieldList) []*types.TypeParam {
	var out []*types.TypeParam
	if fl == nil {
		return nil
	}
	for _, f := range fl.List {
		for _, id := range f.Names {
			if tn, ok := info.Defs[id].(*types.TypeName); ok {
				if tp, ok := tn.Type().(*types.TypeParam); ok {
					out = append(out, tp)
				}
			}
		}
	}
	return out
}

func (m *mono) errorf(pos token.Pos, format string, args ...any) {
	m.errs = append(m.errs, &Error{Pos: m.fset.Position(pos), Msg: fmt.Sprintf(format, args...)})
}

// instance returns the name of the copy of generic obj for type arguments
// args (whose type parameters stand for what env says), asking for the copy
// if it is new.
func (m *mono) instance(obj types.Object, args []types.Type, env monoEnv, at token.Pos) string {
	keys := make([]string, len(args))
	bounds := make([]monoBound, len(args))
	for i, a := range args {
		keys[i] = m.exprKey(m.typeExpr(a, env, at))
		bounds[i] = monoBound{a, env}
	}
	key := obj.Name() + "[" + strings.Join(keys, ", ") + "]"
	if in, ok := m.insts[key]; ok {
		return in.name
	}
	if len(m.insts) > 2000 {
		m.errorf(at, "%s: too many instantiations of generic code (does it instantiate itself with ever bigger types?)", key)
		return obj.Name()
	}
	parts := make([]string, len(keys))
	for i, k := range keys {
		parts[i] = mangleType(k)
	}
	name := obj.Name() + "ᐸ" + strings.Join(parts, "ˏ") + "ᐳ"
	for base, n := name, 2; m.names[name]; n++ {
		name = base + strconv.Itoa(n)
	}
	m.names[name] = true
	in := &monoInst{obj: obj, name: name, args: bounds}
	m.insts[key] = in
	m.queue = append(m.queue, in)
	return name
}

// exprKey is a type expression as Go source, struct tags included (which
// types.ExprString leaves out, and which make struct types differ).
func (m *mono) exprKey(e ast.Expr) string {
	var b strings.Builder
	if err := printer.Fprint(&b, m.fset, e); err != nil {
		return types.ExprString(e)
	}
	return strings.Join(strings.Fields(b.String()), " ")
}

// mangleType turns a type, as Go writes it, into letters for a name.
func mangleType(s string) string {
	s = typeWords.Replace(s)
	var b strings.Builder
	for _, r := range s {
		switch {
		case r == '_' || r >= 'a' && r <= 'z' || r >= 'A' && r <= 'Z' || r >= '0' && r <= '9' || r > 127:
			b.WriteRune(r)
		case r == '[':
			b.WriteString("ᐸ")
		case r == ']':
			b.WriteString("ᐳ")
		case r == '*':
			b.WriteString("ptr")
		case r == ',':
			b.WriteString("ˏ")
		case r == ' ':
		default:
			b.WriteString("_")
		}
	}
	return b.String()
}

var typeWords = strings.NewReplacer("(", "", ")", "", "[]", "Slice", "map[", "Mapᐸ", "func", "Func", "chan ", "Chan", "interface{}", "Any", "struct{", "Structᐸ", "}", "ᐳ", "{", "ᐸ", ";", "ˏ")

// instantiate writes the copies for one instantiation: a function, or a
// type and its methods.
func (m *mono) instantiate(in *monoInst) {
	switch obj := in.obj.(type) {
	case *types.Func:
		d := m.funcs[obj]
		if d == nil {
			m.errorf(obj.Pos(), "internal error: no declaration of generic %s", obj.Name())
			return
		}
		env := monoEnv{}
		for i, tp := range typeParams(m.info, d.Type.TypeParams) {
			if i < len(in.args) {
				env[tp] = in.args[i]
			}
		}
		delta := m.newFile(d)
		c := m.copyNode(d, delta, env).(*ast.FuncDecl)
		c.Name = &ast.Ident{NamePos: d.Name.NamePos + delta, Name: in.name}
		c.Type.TypeParams = nil
		c.Doc = nil
		m.added[m.fileOf[d]] = append(m.added[m.fileOf[d]], c)
	case *types.TypeName:
		ts := m.types[obj]
		if ts == nil {
			m.errorf(obj.Pos(), "internal error: no declaration of generic type %s", obj.Name())
			return
		}
		env := monoEnv{}
		for i, tp := range typeParams(m.info, ts.TypeParams) {
			if i < len(in.args) {
				env[tp] = in.args[i]
			}
		}
		var gd ast.Decl
		for d := range m.fileOf {
			if g, ok := d.(*ast.GenDecl); ok {
				for _, s := range g.Specs {
					if s == ts {
						gd = g
					}
				}
			}
		}
		delta := m.newFile(ts)
		c := m.copyNode(ts, delta, env).(*ast.TypeSpec)
		c.Name = &ast.Ident{NamePos: ts.Name.NamePos + delta, Name: in.name}
		c.TypeParams = nil
		c.Doc, c.Comment = nil, nil
		file := m.fileOf[gd]
		m.added[file] = append(m.added[file], &ast.GenDecl{TokPos: ts.Pos() + delta, Tok: token.TYPE, Specs: []ast.Spec{c}})
		for _, md := range m.meths[obj] {
			menv := monoEnv{}
			for i, tp := range recvTypeParams(m.info, md) {
				if i < len(in.args) {
					menv[tp] = in.args[i]
				}
			}
			delta := m.newFile(md)
			mc := m.copyNode(md, delta, menv).(*ast.FuncDecl)
			mc.Doc = nil
			// The receiver: the copy of the type.
			rt := mc.Recv.List[0]
			ident := &ast.Ident{NamePos: md.Recv.List[0].Type.Pos() + delta, Name: in.name}
			if _, ok := md.Recv.List[0].Type.(*ast.StarExpr); ok {
				rt.Type = &ast.StarExpr{Star: md.Recv.List[0].Type.Pos() + delta, X: ident}
			} else {
				rt.Type = ident
			}
			m.added[m.fileOf[md]] = append(m.added[m.fileOf[md]], mc)
		}
	}
}

// monoRepair fixes what the type check of the instantiated program finds
// in the copies that Go allows in generic code: cases of a switch that the
// type arguments make duplicates or impossible (they can never be chosen,
// so they go), and type assertions that they make impossible (on an
// interface{} of the value instead, which fails as Go's would). It reports
// whether it fixed every error.
func monoRepair(files []*ast.File, errs ErrorList) bool {
	at := map[token.Pos]string{} // the exact positions: copies share their lines
	for _, e := range errs {
		switch {
		case strings.HasPrefix(e.Msg, "\t"):
			// The second line of a message ("\tprevious case").
		case strings.HasPrefix(e.Msg, "duplicate case"), strings.HasPrefix(e.Msg, "impossible type switch case"):
			at[e.at] = "case"
		case strings.HasPrefix(e.Msg, "impossible type assertion"):
			at[e.at] = "assert"
		default:
			return false
		}
	}
	fixed := 0
	for _, f := range files {
		ast.Inspect(f, func(n ast.Node) bool {
			switch x := n.(type) {
			case *ast.BlockStmt:
				// A switch's clauses: drop the cases (and clauses with none left).
				var keep []ast.Stmt
				for _, s := range x.List {
					cc, ok := s.(*ast.CaseClause)
					if !ok {
						keep = append(keep, s)
						continue
					}
					var list []ast.Expr
					for _, e := range cc.List {
						if at[e.Pos()] == "case" {
							fixed++
							continue
						}
						list = append(list, e)
					}
					if cc.List != nil && len(list) == 0 {
						continue
					}
					cc.List = list
					keep = append(keep, cc)
				}
				x.List = keep
			case *ast.TypeAssertExpr:
				if at[x.Pos()] == "assert" {
					any := &ast.ParenExpr{Lparen: x.X.Pos(), X: &ast.InterfaceType{Interface: x.X.Pos(), Methods: &ast.FieldList{Opening: x.X.Pos(), Closing: x.X.Pos()}}, Rparen: x.X.Pos()}
					x.X = &ast.CallExpr{Fun: any, Lparen: x.X.Pos(), Args: []ast.Expr{x.X}, Rparen: x.X.Pos()}
					fixed++
				}
			}
			return true
		})
	}
	return fixed >= len(at)
}

// newFile adds a file to the file set for a copy of n: the same name and
// lines as n's, at other positions. It returns how far the copy's
// positions are from n's.
func (m *mono) newFile(n ast.Node) token.Pos {
	f := m.fset.File(n.Pos())
	if f == nil {
		return 0
	}
	c := m.fset.AddFile(f.Name(), -1, f.Size())
	c.SetLines(f.Lines())
	return token.Pos(c.Base() - f.Base())
}

// embeddedName is the name of an embedded field whose type is an
// instantiated generic type: the copy's, which the field takes in the copy
// of the struct. "" for other fields.
func (m *mono) embeddedName(obj types.Object, env monoEnv, at token.Pos) string {
	f, ok := obj.(*types.Var)
	if !ok || !f.IsField() || !f.Embedded() {
		return ""
	}
	t := f.Type()
	if p, ok := t.(*types.Pointer); ok {
		t = p.Elem()
	}
	n, ok := t.(*types.Named)
	if !ok || n.TypeArgs().Len() == 0 || n.Obj().Pkg() != m.pkg || m.types[n.Origin().Obj()] == nil {
		return ""
	}
	return m.instance(n.Origin().Obj(), typeList(n.TypeArgs()), env, at)
}

// copyExpr copies e: its replacement, if it has one.
func (m *mono) copyExpr(e ast.Expr, delta token.Pos, env monoEnv) ast.Expr {
	if r := m.replace(e, env, delta); r != nil {
		return r
	}
	return m.copyNode(e, delta, env).(ast.Expr)
}

// replace is what a copy has in place of e, or nil: the copy of a generic
// for a use with type arguments, and the type a type parameter stands for.
func (m *mono) replace(e ast.Expr, env monoEnv, delta token.Pos) ast.Expr {
	switch x := e.(type) {
	case *ast.SelectorExpr:
		// x.B for an embedded B[T]: the field has the copy's name.
		if s, ok := m.info.Selections[x]; ok && s.Kind() == types.FieldVal {
			if name := m.embeddedName(s.Obj(), env, x.Sel.Pos()); name != "" {
				return &ast.SelectorExpr{X: m.copyExpr(x.X, delta, env), Sel: &ast.Ident{NamePos: x.Sel.Pos() + delta, Name: name}}
			}
		}
	case *ast.Ident:
		if name := m.embeddedName(m.info.Uses[x], env, x.Pos()); name != "" {
			return &ast.Ident{NamePos: x.Pos() + delta, Name: name} // T{B: ...}
		}
		if tn, ok := m.info.Uses[x].(*types.TypeName); ok {
			if tp, ok := tn.Type().(*types.TypeParam); ok {
				if _, bound := env[tp]; bound {
					return m.typeExpr(tp, env, x.Pos()+delta)
				}
			}
		}
		if p, ok := m.parent[x].(*ast.IndexExpr); ok && p.X == x {
			return nil // f[T]: the index expression is replaced
		}
		if p, ok := m.parent[x].(*ast.IndexListExpr); ok && p.X == x {
			return nil
		}
		if inst, ok := m.info.Instances[x]; ok {
			if obj := m.info.Uses[x]; obj != nil && obj.Pkg() == m.pkg && m.isGeneric(obj) {
				return &ast.Ident{NamePos: x.Pos() + delta, Name: m.instance(obj, typeList(inst.TypeArgs), env, x.Pos())}
			}
		}
	case *ast.IndexExpr, *ast.IndexListExpr:
		var X ast.Expr
		if ix, ok := x.(*ast.IndexExpr); ok {
			X = ix.X
		} else {
			X = x.(*ast.IndexListExpr).X
		}
		id, ok := X.(*ast.Ident)
		if !ok {
			return nil
		}
		if inst, ok := m.info.Instances[id]; ok {
			if obj := m.info.Uses[id]; obj != nil && obj.Pkg() == m.pkg && m.isGeneric(obj) {
				return &ast.Ident{NamePos: id.Pos() + delta, Name: m.instance(obj, typeList(inst.TypeArgs), env, id.Pos())}
			}
		}
	}
	return nil
}

func (m *mono) isGeneric(obj types.Object) bool {
	switch o := obj.(type) {
	case *types.Func:
		return m.funcs[o] != nil
	case *types.TypeName:
		return m.types[o] != nil
	}
	return false
}

func typeList(l *types.TypeList) []types.Type {
	out := make([]types.Type, l.Len())
	for i := range out {
		out[i] = l.At(i)
	}
	return out
}

// typeExpr writes type t as Go source, with what env says for its type
// parameters, and the copies' names for instantiated generic types.
func (m *mono) typeExpr(t types.Type, env monoEnv, at token.Pos) ast.Expr {
	id := func(name string) ast.Expr { return &ast.Ident{NamePos: at, Name: name} }
	paren := func(e ast.Expr) ast.Expr { return &ast.ParenExpr{Lparen: at, X: e, Rparen: at} }
	switch u := t.(type) {
	case *types.TypeParam:
		b, ok := env[u]
		if !ok {
			m.errorf(at, "internal error: type parameter %s has no type here", u)
			return id("int")
		}
		return m.typeExpr(b.t, b.env, at)
	case *types.Alias:
		return m.typeExpr(types.Unalias(u), env, at)
	case *types.Basic:
		if u.Kind() == types.UnsafePointer {
			return &ast.SelectorExpr{X: id("unsafe"), Sel: &ast.Ident{NamePos: at, Name: "Pointer"}}
		}
		return id(types.Typ[u.Kind()].Name())
	case *types.Named:
		obj := u.Obj()
		if u.TypeArgs().Len() > 0 {
			if obj.Pkg() == m.pkg && m.types[u.Origin().Obj()] != nil {
				return id(m.instance(u.Origin().Obj(), typeList(u.TypeArgs()), env, at))
			}
			var args []ast.Expr
			for i := 0; i < u.TypeArgs().Len(); i++ {
				args = append(args, m.typeExpr(u.TypeArgs().At(i), env, at))
			}
			base := m.typeExpr(u.Origin(), env, at)
			if len(args) == 1 {
				return &ast.IndexExpr{X: base, Lbrack: at, Index: args[0], Rbrack: at}
			}
			return &ast.IndexListExpr{X: base, Lbrack: at, Indices: args, Rbrack: at}
		}
		if obj.Pkg() == m.pkg && obj.Parent() != nil && obj.Parent() != m.pkg.Scope() {
			m.errorf(at, "type %s is declared in a function: local types cannot be type arguments on the card", obj.Name())
		}
		if obj.Pkg() == nil || obj.Pkg() == m.pkg {
			return id(obj.Name())
		}
		return &ast.SelectorExpr{X: id(obj.Pkg().Name()), Sel: &ast.Ident{NamePos: at, Name: obj.Name()}}
	case *types.Pointer:
		return paren(&ast.StarExpr{Star: at, X: m.typeExpr(u.Elem(), env, at)})
	case *types.Slice:
		return paren(&ast.ArrayType{Lbrack: at, Elt: m.typeExpr(u.Elem(), env, at)})
	case *types.Array:
		return paren(&ast.ArrayType{Lbrack: at, Len: &ast.BasicLit{ValuePos: at, Kind: token.INT, Value: strconv.FormatInt(u.Len(), 10)}, Elt: m.typeExpr(u.Elem(), env, at)})
	case *types.Map:
		return paren(&ast.MapType{Map: at, Key: m.typeExpr(u.Key(), env, at), Value: m.typeExpr(u.Elem(), env, at)})
	case *types.Chan:
		dir := ast.SEND | ast.RECV
		switch u.Dir() {
		case types.SendOnly:
			dir = ast.SEND
		case types.RecvOnly:
			dir = ast.RECV
		}
		return paren(&ast.ChanType{Begin: at, Arrow: at, Dir: dir, Value: m.typeExpr(u.Elem(), env, at)})
	case *types.Signature:
		return paren(m.funcType(u, env, at))
	case *types.Struct:
		fl := &ast.FieldList{Opening: at, Closing: at}
		for i := 0; i < u.NumFields(); i++ {
			f := u.Field(i)
			fd := &ast.Field{Type: m.typeExpr(f.Type(), env, at)}
			if !f.Embedded() {
				fd.Names = []*ast.Ident{{NamePos: at, Name: f.Name()}}
			}
			if tag := u.Tag(i); tag != "" {
				fd.Tag = &ast.BasicLit{ValuePos: at, Kind: token.STRING, Value: strconv.Quote(tag)}
			}
			fl.List = append(fl.List, fd)
		}
		return paren(&ast.StructType{Struct: at, Fields: fl})
	case *types.Interface:
		fl := &ast.FieldList{Opening: at, Closing: at}
		for i := 0; i < u.NumEmbeddeds(); i++ {
			fl.List = append(fl.List, &ast.Field{Type: m.typeExpr(u.EmbeddedType(i), env, at)})
		}
		for i := 0; i < u.NumExplicitMethods(); i++ {
			f := u.ExplicitMethod(i)
			fl.List = append(fl.List, &ast.Field{Names: []*ast.Ident{{NamePos: at, Name: f.Name()}},
				Type: m.funcType(f.Type().(*types.Signature), env, at)})
		}
		return paren(&ast.InterfaceType{Interface: at, Methods: fl})
	case *types.Union:
		var e ast.Expr
		for i := 0; i < u.Len(); i++ {
			term := m.typeExpr(u.Term(i).Type(), env, at)
			if u.Term(i).Tilde() {
				term = &ast.UnaryExpr{OpPos: at, Op: token.TILDE, X: term}
			}
			if e == nil {
				e = term
			} else {
				e = &ast.BinaryExpr{X: e, OpPos: at, Op: token.OR, Y: term}
			}
		}
		return e
	}
	m.errorf(at, "type %s cannot be a type argument on the card", t)
	return id("int")
}

func (m *mono) funcType(sig *types.Signature, env monoEnv, at token.Pos) *ast.FuncType {
	fields := func(tup *types.Tuple, variadic bool) *ast.FieldList {
		fl := &ast.FieldList{Opening: at, Closing: at}
		for i := 0; i < tup.Len(); i++ {
			t := tup.At(i).Type()
			var e ast.Expr
			if variadic && i == tup.Len()-1 {
				e = &ast.Ellipsis{Ellipsis: at, Elt: m.typeExpr(t.(*types.Slice).Elem(), env, at)}
			} else {
				e = m.typeExpr(t, env, at)
			}
			fl.List = append(fl.List, &ast.Field{Type: e})
		}
		return fl
	}
	return &ast.FuncType{Func: at, Params: fields(sig.Params(), sig.Variadic()), Results: fields(sig.Results(), false)}
}

// copyNode copies the syntax tree n with its positions moved by delta and,
// where env is not nil or n is the program's own code, the uses of
// generics and type parameters replaced (replace). The replacements are
// looked up on the original nodes, which the type information is about.
func (m *mono) copyNode(n ast.Node, delta token.Pos, env monoEnv) ast.Node {
	v := m.copyValue(reflect.ValueOf(n), delta, env)
	return v.Interface().(ast.Node)
}

var (
	posType     = reflect.TypeOf(token.Pos(0))
	exprType    = reflect.TypeOf((*ast.Expr)(nil)).Elem()
	objectType  = reflect.TypeOf((*ast.Object)(nil))
	commentType = reflect.TypeOf((*ast.CommentGroup)(nil))
	scopeType   = reflect.TypeOf((*ast.Scope)(nil))
)

func (m *mono) copyValue(v reflect.Value, delta token.Pos, env monoEnv) reflect.Value {
	switch v.Kind() {
	case reflect.Interface:
		if v.IsNil() {
			return reflect.Zero(v.Type())
		}
		if e, ok := v.Interface().(ast.Expr); ok {
			if r := m.replace(e, env, delta); r != nil {
				out := reflect.New(v.Type()).Elem()
				out.Set(reflect.ValueOf(r))
				return out
			}
		}
		c := m.copyValue(v.Elem(), delta, env)
		out := reflect.New(v.Type()).Elem()
		out.Set(c)
		return out
	case reflect.Ptr:
		if v.IsNil() {
			return reflect.Zero(v.Type())
		}
		switch v.Type() {
		case objectType, commentType, scopeType:
			return reflect.Zero(v.Type())
		}
		if v.Elem().Kind() != reflect.Struct {
			return v
		}
		c := reflect.New(v.Elem().Type())
		for i := 0; i < v.Elem().NumField(); i++ {
			f := v.Elem().Field(i)
			if !c.Elem().Field(i).CanSet() {
				continue
			}
			c.Elem().Field(i).Set(m.copyValue(f, delta, env))
		}
		return c
	case reflect.Slice:
		if v.IsNil() {
			return reflect.Zero(v.Type())
		}
		c := reflect.MakeSlice(v.Type(), v.Len(), v.Len())
		for i := 0; i < v.Len(); i++ {
			c.Index(i).Set(m.copyValue(v.Index(i), delta, env))
		}
		return c
	default:
		if v.Type() == posType && v.Int() != 0 {
			return reflect.ValueOf(token.Pos(v.Int()) + delta)
		}
		return v
	}
}
