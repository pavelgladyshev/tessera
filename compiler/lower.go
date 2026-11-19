package compiler

import (
	"fmt"
	"math"
	"sort"
)

// The compute kernel works on one 32×32 tile per input at a time. Its values
// live in the destination register file ("DEST"), which holds a few tiles,
// and every operation is a vector-unit (SFPU) call on DEST slots. Lowering
// therefore schedules the graph into a list of SFPU calls and assigns DEST
// slots to values, like register allocation in an ordinary compiler.
//
// DEST is used in 32-bit mode (fp32_dest_acc_en) so intermediate results
// keep float32 precision. That leaves 4 slots with double buffering (the
// packer drains one half while the vector unit fills the other) or 8 slots
// without it. Lowering uses double buffering when the program fits in 4.

const (
	slotsHalfSync = 4
	slotsFullSync = 8
)

// InstrKind is the kind of a compute-kernel instruction.
type InstrKind int

const (
	IkCopyIn   InstrKind = iota // copy_tile(cb, 0, Dst): input tile -> slot
	IkFill                      // fill_tile(Dst, Val)
	IkUnary                     // Fn(Dst), in place
	IkScalar                    // Fn(Dst, bits(Val)), in place
	IkBinary                    // Fn(A, B, Dst)
	IkWhere                     // where_tile(A, B, C, Dst)
	IkCopyDest                  // copy_dest_values(A, Dst)
)

// Instr is one compute-kernel instruction.
type Instr struct {
	Kind    InstrKind
	Fn      string // SFPU function, e.g. "exp_tile"
	Init    string // its init call, e.g. "exp_tile_init()"
	Header  string // header that declares it
	Dst     int
	A, B, C int
	Input   int // IkCopyIn: input index (= circular buffer index)
	Val     float32
	Comment string
}

// Pack moves a finished output tile from a DEST slot to its circular buffer.
type Pack struct {
	Slot   int
	Output int
}

// Kernel is the lowered compute kernel.
type Kernel struct {
	Instrs   []Instr
	Packs    []Pack
	Slots    int  // peak number of DEST slots used
	FullSync bool // true if more than slotsHalfSync slots are needed
}

type sfpuFn struct{ fn, init, header string }

var unaryFns = map[Op]sfpuFn{
	OpNeg:     {"negative_tile", "negative_tile_init()", "api/compute/eltwise_unary/negative.h"},
	OpExp:     {"exp_tile", "exp_tile_init()", "api/compute/eltwise_unary/exp.h"},
	OpLog:     {"log_tile", "log_tile_init()", "api/compute/compute_kernel_api.h"},
	OpSqrt:    {"sqrt_tile", "sqrt_tile_init()", "api/compute/eltwise_unary/sqrt.h"},
	OpRsqrt:   {"rsqrt_tile", "rsqrt_tile_init()", "api/compute/eltwise_unary/rsqrt.h"},
	OpRecip:   {"recip_tile", "recip_tile_init()", "api/compute/eltwise_unary/recip.h"},
	OpAbs:     {"abs_tile", "abs_tile_init()", "api/compute/compute_kernel_api.h"},
	OpRelu:    {"relu_tile", "relu_tile_init()", "api/compute/eltwise_unary/relu.h"},
	OpSigmoid: {"sigmoid_tile", "sigmoid_tile_init()", "api/compute/compute_kernel_api.h"},
	OpTanh:    {"tanh_tile", "tanh_tile_init()", "api/compute/compute_kernel_api.h"},
	OpSin:     {"sin_tile", "sin_tile_init()", "api/compute/eltwise_unary/trigonometry.h"},
	OpCos:     {"cos_tile", "cos_tile_init()", "api/compute/eltwise_unary/trigonometry.h"},
	OpSquare:  {"square_tile", "square_tile_init()", "api/compute/compute_kernel_api.h"},
}

const (
	hdrBinary  = "api/compute/eltwise_binary_sfpu.h"
	hdrMaxMin  = "api/compute/binary_max_min.h"
	hdrScalar  = "api/compute/eltwise_unary/binop_with_scalar.h"
	hdrFill    = "api/compute/eltwise_unary/fill.h"
	hdrWhere   = "api/compute/eltwise_unary/where.h"
	hdrCopyDst = "api/compute/copy_dest_values.h"
)

var binaryFns = map[Op]sfpuFn{
	OpAdd: {"add_binary_tile", "add_binary_tile_init()", hdrBinary},
	OpSub: {"sub_binary_tile", "sub_binary_tile_init()", hdrBinary},
	OpMul: {"mul_binary_tile", "mul_binary_tile_init()", hdrBinary},
	OpDiv: {"div_binary_tile", "div_binary_tile_init()", hdrBinary},
	OpMax: {"binary_max_tile", "binary_max_tile_init()", hdrMaxMin},
	OpMin: {"binary_min_tile", "binary_min_tile_init()", hdrMaxMin},
	OpLt:  {"lt_binary_tile", "lt_binary_tile_init()", hdrBinary},
	OpGt:  {"gt_binary_tile", "gt_binary_tile_init()", hdrBinary},
	OpLe:  {"le_binary_tile", "le_binary_tile_init()", hdrBinary},
	OpGe:  {"ge_binary_tile", "ge_binary_tile_init()", hdrBinary},
	OpEq:  {"eq_binary_tile", "eq_binary_tile_init()", hdrBinary},
	OpNe:  {"ne_binary_tile", "ne_binary_tile_init()", hdrBinary},
}

var commutative = map[Op]bool{OpAdd: true, OpMul: true, OpMax: true, OpMin: true, OpEq: true, OpNe: true}

// swapped gives the comparison with its operands exchanged: a < b == b > a.
var swapped = map[Op]Op{OpLt: OpGt, OpGt: OpLt, OpLe: OpGe, OpGe: OpLe}

// step is one in-place operation applied to a single slot.
type step struct {
	unary  Op      // a unary op, or
	scalar string  // a scalar op such as "mul_unary_tile"
	val    float32 // its constant
}

// inPlaceSteps returns how to compute n as in-place operations on one
// operand, or nil if n needs a two-operand instruction.
func inPlaceSteps(n *Node) (operand *Node, steps []step) {
	if _, ok := unaryFns[n.Op]; ok {
		return n.Args[0], []step{{unary: n.Op}}
	}
	if len(n.Args) != 2 {
		return nil, nil
	}
	a, b := n.Args[0], n.Args[1]
	switch n.Op {
	case OpAdd:
		if b.Op == OpConst {
			return a, []step{{scalar: "add_unary_tile", val: b.Val}}
		}
		if a.Op == OpConst {
			return b, []step{{scalar: "add_unary_tile", val: a.Val}}
		}
	case OpMul:
		if b.Op == OpConst {
			return a, []step{{scalar: "mul_unary_tile", val: b.Val}}
		}
		if a.Op == OpConst {
			return b, []step{{scalar: "mul_unary_tile", val: a.Val}}
		}
	case OpSub:
		if b.Op == OpConst { // x - c
			return a, []step{{scalar: "sub_unary_tile", val: b.Val}}
		}
		if a.Op == OpConst { // c - x = (-x) + c
			return b, []step{{unary: OpNeg}, {scalar: "add_unary_tile", val: a.Val}}
		}
	case OpDiv:
		if b.Op == OpConst { // x / c
			return a, []step{{scalar: "div_unary_tile", val: b.Val}}
		}
		if a.Op == OpConst { // c / x = recip(x) * c
			if a.Val == 1 {
				return b, []step{{unary: OpRecip}}
			}
			return b, []step{{unary: OpRecip}, {scalar: "mul_unary_tile", val: a.Val}}
		}
	}
	return nil, nil
}

type lowerer struct {
	prog      *Program
	k         *Kernel
	slotOf    map[*Node]int
	remaining map[*Node]int // uses not yet consumed, for computed nodes
	busy      []bool
	need      map[*Node]int
}

// Lower schedules the program's graph into a compute kernel.
func Lower(p *Program) (*Kernel, error) {
	l := &lowerer{
		prog:      p,
		k:         &Kernel{},
		slotOf:    map[*Node]int{},
		remaining: map[*Node]int{},
		busy:      make([]bool, 64),
		need:      map[*Node]int{},
	}
	order := l.schedule()
	for _, n := range order {
		l.emit(n)
	}
	for j, o := range p.Outputs {
		n := o.Value
		s, ok := l.slotOf[n]
		if !ok {
			s = l.materialize(n, "store "+o.Name)
		}
		l.k.Packs = append(l.k.Packs, Pack{Slot: s, Output: j})
	}
	if l.k.Slots > slotsFullSync {
		return nil, &Error{Pos: p.Outputs[0].Pos, Msg: fmt.Sprintf(
			"the fused computation needs %d tiles of DEST registers at once, but the card has %d and v0 cannot "+
				"spill them to memory yet; store fewer arrays in one program, or simplify the element functions", l.k.Slots, slotsFullSync)}
	}
	l.k.FullSync = l.k.Slots > slotsHalfSync
	return l.k, nil
}

// schedule orders the computed nodes so that operands come first, visiting
// the operand that needs more registers first (Sethi-Ullman), and counts the
// uses of every computed node.
func (l *lowerer) schedule() []*Node {
	var order []*Node
	seen := map[*Node]bool{}
	var visit func(n *Node)
	visit = func(n *Node) {
		if seen[n] || n.IsLeaf() {
			return
		}
		seen[n] = true
		args := append([]*Node{}, n.Args...)
		sort.SliceStable(args, func(i, j int) bool { return l.needOf(args[i]) > l.needOf(args[j]) })
		for _, a := range args {
			visit(a)
		}
		order = append(order, n)
		for _, a := range distinct(n.Args) {
			if !a.IsLeaf() {
				l.remaining[a]++
			}
		}
	}
	for _, o := range l.prog.Outputs {
		visit(o.Value)
		if !o.Value.IsLeaf() {
			l.remaining[o.Value]++
		}
	}
	return order
}

// needOf estimates how many slots computing n takes.
func (l *lowerer) needOf(n *Node) int {
	if n.IsLeaf() {
		return 1
	}
	if v, ok := l.need[n]; ok {
		return v
	}
	l.need[n] = 1 // guard; the graph is acyclic
	var needs []int
	for _, a := range n.Args {
		needs = append(needs, l.needOf(a))
	}
	sort.Sort(sort.Reverse(sort.IntSlice(needs)))
	v := 0
	for i, x := range needs {
		v = max(v, x+i)
	}
	l.need[n] = v
	return v
}

func distinct(ns []*Node) []*Node {
	var out []*Node
	for _, n := range ns {
		dup := false
		for _, m := range out {
			if m == n {
				dup = true
			}
		}
		if !dup {
			out = append(out, n)
		}
	}
	return out
}

func (l *lowerer) alloc() int {
	for i, b := range l.busy {
		if !b {
			l.busy[i] = true
			l.k.Slots = max(l.k.Slots, i+1)
			return i
		}
	}
	panic("internal error: out of slots")
}

func (l *lowerer) free(s int) { l.busy[s] = false }

func comment(n *Node) string {
	if n.Pos.IsValid() {
		return fmt.Sprintf("%s:%d  %s", baseName(n.Pos.Filename), n.Pos.Line, n.Src)
	}
	return n.Src
}

// materialize puts a leaf (an input tile or a constant) into a fresh slot.
func (l *lowerer) materialize(n *Node, why string) int {
	s := l.alloc()
	switch n.Op {
	case OpInput:
		l.k.Instrs = append(l.k.Instrs, Instr{Kind: IkCopyIn, Dst: s, Input: n.Input,
			Comment: fmt.Sprintf("load %s (%s)", l.prog.Inputs[n.Input].Name, why)})
	case OpConst:
		l.k.Instrs = append(l.k.Instrs, Instr{Kind: IkFill, Fn: "fill_tile", Init: "fill_tile_init()", Header: hdrFill,
			Dst: s, Val: n.Val, Comment: fmt.Sprintf("constant %s (%s)", formatFloat(n.Val), why)})
	default:
		panic("internal error: materialize of a computed node")
	}
	return s
}

// operand returns the slot holding n for use by the current instruction, and
// whether that slot may be overwritten or freed afterwards (n is dead or it
// is a temporary copy of a leaf).
func (l *lowerer) operand(n *Node, user *Node) (slot int, dead bool) {
	if n.IsLeaf() {
		return l.materialize(n, "for "+user.Src), true
	}
	s, ok := l.slotOf[n]
	if !ok {
		panic("internal error: operand not computed yet")
	}
	return s, l.remaining[n] == 1
}

// consume records that user has used n.
func (l *lowerer) consume(n *Node) {
	if n.IsLeaf() {
		return
	}
	l.remaining[n]--
	if l.remaining[n] == 0 {
		delete(l.slotOf, n)
	}
}

func (l *lowerer) emit(n *Node) {
	if src, steps := inPlaceSteps(n); steps != nil {
		var d int
		if src.IsLeaf() {
			d = l.materialize(src, "for "+n.Src)
		} else if s, dead := l.operand(src, n); dead {
			d = s
			l.consume(src) // slot ownership passes to n
		} else {
			d = l.alloc()
			// DEST is fp32, so the copy must move all 32 bits of each value.
			l.k.Instrs = append(l.k.Instrs, Instr{Kind: IkCopyDest, Fn: "copy_dest_values<DataFormat::Float32>", Init: "copy_dest_values_init()",
				Header: hdrCopyDst, A: s, Dst: d, Comment: "keep a copy: " + src.Src + " is used again"})
			l.consume(src)
		}
		for i, st := range steps {
			c := ""
			if i == 0 {
				c = comment(n)
			}
			if st.scalar != "" {
				l.k.Instrs = append(l.k.Instrs, Instr{Kind: IkScalar, Fn: st.scalar, Init: "binop_with_scalar_tile_init()",
					Header: hdrScalar, Dst: d, Val: st.val, Comment: c})
			} else {
				f := unaryFns[st.unary]
				l.k.Instrs = append(l.k.Instrs, Instr{Kind: IkUnary, Fn: f.fn, Init: f.init, Header: f.header, Dst: d, Comment: c})
			}
		}
		l.slotOf[n] = d
		return
	}

	args := append([]*Node{}, n.Args...)
	op := n.Op
	// Put a dead operand first, so its slot can hold the result: the SFPU
	// calls are known to support out == first operand.
	if len(args) == 2 && !l.reusable(args[0]) && l.reusable(args[1]) {
		if commutative[op] {
			args[0], args[1] = args[1], args[0]
		} else if sw, ok := swapped[op]; ok {
			op = sw
			args[0], args[1] = args[1], args[0]
		}
	}

	slots := make([]int, len(args))
	dead := make([]bool, len(args))
	materialized := map[*Node]int{}
	for i, a := range args {
		if s, ok := materialized[a]; ok { // x*x on a leaf: one copy is enough
			slots[i], dead[i] = s, true
			continue
		}
		slots[i], dead[i] = l.operand(a, n)
		if a.IsLeaf() {
			materialized[a] = slots[i]
		}
	}
	out := -1
	if dead[0] {
		out = slots[0]
	} else {
		out = l.alloc()
	}

	if op == OpWhere {
		l.k.Instrs = append(l.k.Instrs, Instr{Kind: IkWhere, Fn: "where_tile<DataFormat::Float32>", Init: "where_tile_init()",
			Header: hdrWhere, A: slots[0], B: slots[1], C: slots[2], Dst: out, Comment: comment(n)})
	} else {
		f, ok := binaryFns[op]
		if !ok {
			panic(fmt.Sprintf("internal error: no SFPU function for %s", opTable[op].name))
		}
		l.k.Instrs = append(l.k.Instrs, Instr{Kind: IkBinary, Fn: f.fn, Init: f.init, Header: f.header,
			A: slots[0], B: slots[1], Dst: out, Comment: comment(n)})
	}

	// Release operand slots that are dead (other than the one now holding
	// the result), then record the uses.
	freed := map[int]bool{out: true}
	for i := range args {
		if dead[i] && !freed[slots[i]] {
			l.free(slots[i])
			freed[slots[i]] = true
		}
	}
	for _, a := range distinct(args) {
		l.consume(a)
	}
	l.slotOf[n] = out
}

// reusable reports whether n's slot can be overwritten by the instruction
// that is about to use it.
func (l *lowerer) reusable(n *Node) bool {
	return n.IsLeaf() || l.remaining[n] == 1
}

func floatBits(v float32) uint32 { return math.Float32bits(v) }
