package compiler

import (
	"fmt"
	"go/ast"
	"go/constant"
)

// Tensors in worker code (tt.Tensor, tt.Layer, tt.Matrix).
//
// A tt.NewTensor(rows, cols) in the card's code is one tensor, with a
// circular buffer of its own on every worker's core: rows and cols must be
// constants, so that the buffers can be made before the workers start. The
// operations become calls of the runtime (runtime/tensor_device.h), which
// reads and writes the tensors in place and sends the matrix products to
// the tensor kernel (runtime/tensor.cpp) on the same core.

// tensorCall translates tt.NewTensor, tt.Linear, tt.LinearReLU and
// tt.MaxRows; ok is false for other functions.
func (g *cgen) tensorCall(name string, x *ast.CallExpr) (string, bool) {
	switch name {
	case "NewTensor", "Linear", "LinearReLU", "LinearCat", "LinearReLUCat", "MaxRows", "MeanRows":
	default:
		return "", false
	}
	if g.mode != deviceMode || g.sh.goFIDs != nil {
		g.fail(x, "tt.%s is for the worker code of tt.Parallel", name)
	}
	g.sh.usesTensor = true
	switch name {
	case "NewTensor":
		var dims [2]int64
		for i, a := range x.Args {
			v := g.sh.info.Types[a].Value
			if v == nil {
				g.fail(a, "the size of a tensor on the card must be a constant")
			}
			dims[i], _ = constant.Int64Val(constant.ToInt(v))
			if dims[i] <= 0 || dims[i] > 1<<16 {
				g.fail(a, "a tensor cannot have %d rows or columns", dims[i])
			}
		}
		cb, ok := g.sh.tensorAt[x]
		if !ok {
			if len(g.sh.tensors) == len(tensorCBs) {
				g.fail(x, "a worker program can have %d tt.NewTensor: each has a buffer of its own on the core", len(tensorCBs))
			}
			cb = tensorCBs[len(g.sh.tensors)]
			g.sh.tensorAt[x] = cb
			g.sh.tensors = append(g.sh.tensors, int(((dims[0]+31)/32)*((dims[1]+31)/32)))
		}
		return fmt.Sprintf("tsr::tns_new(%d, %d, %d)", cb, dims[0], dims[1]), true
	case "Linear", "LinearReLU":
		pos := g.sh.position(x, "tt."+name+": the tensors and the layer do not fit each other")
		return fmt.Sprintf("tsr::tns_linear(%s, %s, %v, %d, %s)", g.expr(x.Args[0]), g.expr(x.Args[2]),
			name == "LinearReLU", pos, g.expr(x.Args[1])), true
	case "LinearCat", "LinearReLUCat":
		if x.Ellipsis.IsValid() || len(x.Args) < 3 || len(x.Args) > 9 {
			g.fail(x, "tt.%s takes 1 to 7 tensors, written out", name)
		}
		pos := g.sh.position(x, "tt."+name+": the tensors and the layer do not fit each other (every tensor but the last needs a multiple of 32 columns)")
		return fmt.Sprintf("tsr::tns_linear(%s, %s, %v, %d, %s)", g.expr(x.Args[0]), g.expr(x.Args[1]),
			name == "LinearReLUCat", pos, g.args(x.Args[2:])), true
	}
	pos := g.sh.position(x, "tt."+name+": the result must have one row and the argument's columns (and for MaxRows the argument a multiple of 32 rows)")
	op := 2
	if name == "MeanRows" {
		op = 4
	}
	return fmt.Sprintf("tsr::tns_reduce(%d, %s, %s, %d)", op, g.expr(x.Args[0]), g.expr(x.Args[1]), pos), true
}

// tensorCBs are the circular buffers of the tensors, in order (the card
// helper uses the same list, runtime/tensor_host.h).
var tensorCBs = []int{16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 8, 9, 10, 11, 12, 13, 15, 2, 3, 4}

// tensorMethod translates the methods of tt.Tensor; ok is false for others.
func (g *cgen) tensorMethod(name, r string, x *ast.CallExpr) (string, bool) {
	if g.mode != deviceMode {
		return "", false
	}
	pos := g.sh.position(x, "tt.Tensor."+name+": index out of range")
	switch name {
	case "Zero":
		return fmt.Sprintf("tsr::tns_clear(%s, %d)", r, pos), true
	case "Rows":
		return fmt.Sprintf("tsr::tns_rows(%s, %s, %d)", r, g.args(x.Args),
			g.sh.position(x, "tt.Tensor.Rows: the rows must be in the tensor and start at a multiple of 32 (and be a multiple of 32, or go to its end)")), true
	case "Set":
		zero := g.sh.position(x, "tt.Tensor.Set: division by zero")
		return fmt.Sprintf("tsr::tns_set(%s, %s, %d, %d)", r, g.args(x.Args), pos, zero), true
	case "CopyRow":
		return fmt.Sprintf("tsr::tns_copy_row(%s, %s, %d)", r, g.args(x.Args), pos), true
	case "CopyRowOf":
		return fmt.Sprintf("tsr::tns_copy_row_of(%s, %s, %d)", r, g.args(x.Args), pos), true
	case "ArgMax":
		return fmt.Sprintf("tsr::tns_argmax(%s, %s, %d)", r, g.args(x.Args), pos), true
	case "Scaled":
		return fmt.Sprintf("tsr::tns_scaled(%s, %s, %d)", r, g.args(x.Args), pos), true
	}
	return "", false
}
