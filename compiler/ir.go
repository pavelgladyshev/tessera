package compiler

import (
	"fmt"
	"go/token"
	"math"
	"strings"

	"github.com/pavelgladyshev/tessera/tt"
)

// Op is an elementwise operation in the dataflow graph.
type Op int

const (
	OpInput Op = iota // an input array (Node.Input says which)
	OpConst           // a float32 constant (Node.Val)

	// unary
	OpNeg
	OpExp
	OpLog
	OpSqrt
	OpRsqrt
	OpRecip
	OpAbs
	OpRelu
	OpSigmoid
	OpTanh
	OpSin
	OpCos
	OpSquare

	// binary
	OpAdd
	OpSub
	OpMul
	OpDiv
	OpMax
	OpMin
	OpLt
	OpGt
	OpLe
	OpGe
	OpEq
	OpNe

	// ternary: Where(cond, a, b) is a where cond != 0, else b
	OpWhere
)

type opInfo struct {
	name  string
	arity int
	fold  func(a []float32) float32 // CPU semantics, used for constant folding
}

func b2f(b bool) float32 {
	if b {
		return 1
	}
	return 0
}

var opTable = map[Op]opInfo{
	OpNeg:     {"neg", 1, func(a []float32) float32 { return -a[0] }},
	OpExp:     {"exp", 1, func(a []float32) float32 { return tt.Exp(a[0]) }},
	OpLog:     {"log", 1, func(a []float32) float32 { return tt.Log(a[0]) }},
	OpSqrt:    {"sqrt", 1, func(a []float32) float32 { return tt.Sqrt(a[0]) }},
	OpRsqrt:   {"rsqrt", 1, func(a []float32) float32 { return tt.Rsqrt(a[0]) }},
	OpRecip:   {"recip", 1, func(a []float32) float32 { return tt.Recip(a[0]) }},
	OpAbs:     {"abs", 1, func(a []float32) float32 { return tt.Abs(a[0]) }},
	OpRelu:    {"relu", 1, func(a []float32) float32 { return tt.Relu(a[0]) }},
	OpSigmoid: {"sigmoid", 1, func(a []float32) float32 { return tt.Sigmoid(a[0]) }},
	OpTanh:    {"tanh", 1, func(a []float32) float32 { return tt.Tanh(a[0]) }},
	OpSin:     {"sin", 1, func(a []float32) float32 { return tt.Sin(a[0]) }},
	OpCos:     {"cos", 1, func(a []float32) float32 { return tt.Cos(a[0]) }},
	OpSquare:  {"square", 1, func(a []float32) float32 { return tt.Square(a[0]) }},
	OpAdd:     {"add", 2, func(a []float32) float32 { return a[0] + a[1] }},
	OpSub:     {"sub", 2, func(a []float32) float32 { return a[0] - a[1] }},
	OpMul:     {"mul", 2, func(a []float32) float32 { return a[0] * a[1] }},
	OpDiv:     {"div", 2, func(a []float32) float32 { return a[0] / a[1] }},
	OpMax:     {"max", 2, func(a []float32) float32 { return tt.Max(a[0], a[1]) }},
	OpMin:     {"min", 2, func(a []float32) float32 { return tt.Min(a[0], a[1]) }},
	OpLt:      {"lt", 2, func(a []float32) float32 { return b2f(a[0] < a[1]) }},
	OpGt:      {"gt", 2, func(a []float32) float32 { return b2f(a[0] > a[1]) }},
	OpLe:      {"le", 2, func(a []float32) float32 { return b2f(a[0] <= a[1]) }},
	OpGe:      {"ge", 2, func(a []float32) float32 { return b2f(a[0] >= a[1]) }},
	OpEq:      {"eq", 2, func(a []float32) float32 { return b2f(a[0] == a[1]) }},
	OpNe:      {"ne", 2, func(a []float32) float32 { return b2f(a[0] != a[1]) }},
	OpWhere: {"where", 3, func(a []float32) float32 {
		if a[0] != 0 {
			return a[1]
		}
		return a[2]
	}},
}

// Node is one value in the elementwise dataflow graph. Nodes are hash-consed:
// building the same operation on the same operands twice returns the same
// node, which gives common-subexpression elimination for free.
type Node struct {
	Op    Op
	Args  []*Node
	Input int     // OpInput: index into Program.Inputs
	Val   float32 // OpConst
	Pos   token.Position
	Src   string // source text that first produced this node, for comments
	id    int
}

func (n *Node) IsLeaf() bool { return n.Op == OpInput || n.Op == OpConst }

func (n *Node) isConst(v float32) bool { return n.Op == OpConst && n.Val == v }

// maxNodes bounds the graph size, so that a program whose ifs multiply out
// fails with a clear message instead of exhausting memory.
const maxNodes = 20000

type graph struct {
	memo  map[string]*Node
	count int
}

func newGraph() *graph { return &graph{memo: map[string]*Node{}} }

func (g *graph) add(key string, n *Node) *Node {
	if old, ok := g.memo[key]; ok {
		return old
	}
	if g.count >= maxNodes {
		panic(&Error{Pos: n.Pos, Msg: fmt.Sprintf("element code is too large for v0 (more than %d operations after inlining); split it into several Map calls", maxNodes)})
	}
	n.id = g.count
	g.count++
	g.memo[key] = n
	return n
}

func (g *graph) constant(v float32, pos token.Position) *Node {
	if v == 0 {
		v = 0 // fold -0 into +0 so that 0 and -0 share a node
	}
	return g.add(fmt.Sprintf("c%08x", math.Float32bits(v)), &Node{Op: OpConst, Val: v, Pos: pos, Src: formatFloat(v)})
}

func (g *graph) input(i int, name string, pos token.Position) *Node {
	return g.add(fmt.Sprintf("i%d", i), &Node{Op: OpInput, Input: i, Pos: pos, Src: name})
}

// node builds op(args...), folding constants and applying a few identities
// that are exact in IEEE arithmetic.
func (g *graph) node(op Op, pos token.Position, src string, args ...*Node) *Node {
	info := opTable[op]
	if len(args) != info.arity {
		panic(fmt.Sprintf("internal error: %s takes %d operands, got %d", info.name, info.arity, len(args)))
	}
	allConst := true
	for _, a := range args {
		if a.Op != OpConst {
			allConst = false
		}
	}
	if allConst {
		vals := make([]float32, len(args))
		for i, a := range args {
			vals[i] = a.Val
		}
		return g.constant(info.fold(vals), pos)
	}
	a := args[0]
	switch op {
	case OpWhere:
		if a.Op == OpConst {
			if a.Val != 0 {
				return args[1]
			}
			return args[2]
		}
		if args[1] == args[2] {
			return args[1]
		}
	case OpAdd:
		if args[1].isConst(0) {
			return a
		}
		if a.isConst(0) {
			return args[1]
		}
	case OpSub:
		if args[1].isConst(0) {
			return a
		}
	case OpMul:
		if args[1].isConst(1) {
			return a
		}
		if a.isConst(1) {
			return args[1]
		}
	case OpDiv:
		if args[1].isConst(1) {
			return a
		}
	case OpNeg:
		if a.Op == OpNeg {
			return a.Args[0]
		}
	}
	var key strings.Builder
	fmt.Fprintf(&key, "o%d", op)
	for _, a := range args {
		fmt.Fprintf(&key, ",%d", a.id)
	}
	return g.add(key.String(), &Node{Op: op, Args: args, Pos: pos, Src: src})
}

// formatFloat prints v as a C++ float literal that round-trips exactly.
func formatFloat(v float32) string {
	switch {
	case math.IsInf(float64(v), 1):
		return "INFINITY"
	case math.IsInf(float64(v), -1):
		return "-INFINITY"
	case math.IsNaN(float64(v)):
		return "NAN"
	}
	s := fmt.Sprintf("%g", v)
	if !strings.ContainsAny(s, ".e") {
		s += ".0"
	}
	return s + "f"
}
