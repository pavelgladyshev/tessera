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
	case "NewTensor", "Linear", "LinearReLU", "MaxRows":
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
			if len(g.sh.tensors) == 16 {
				g.fail(x, "a worker program can have 16 tt.NewTensor: each has a buffer of its own on the core")
			}
			cb = 16 + len(g.sh.tensors)
			g.sh.tensorAt[x] = cb
			g.sh.tensors = append(g.sh.tensors, int(((dims[0]+31)/32)*((dims[1]+31)/32)))
		}
		return fmt.Sprintf("tsr::tns_new(%d, %d, %d)", cb, dims[0], dims[1]), true
	case "Linear", "LinearReLU":
		pos := g.sh.position(x, "tt."+name+": the tensors and the layer do not fit each other")
		return fmt.Sprintf("tsr::tns_linear(%s, %s, %s, %v, %d)", g.expr(x.Args[0]), g.expr(x.Args[1]), g.expr(x.Args[2]),
			name == "LinearReLU", pos), true
	}
	pos := g.sh.position(x, "tt.MaxRows: the result must have one row and the argument's columns, and the argument a multiple of 32 rows")
	return fmt.Sprintf("tsr::tns_max_rows(%s, %s, %d)", g.expr(x.Args[0]), g.expr(x.Args[1]), pos), true
}

// tensorMethod translates the methods of tt.Tensor; ok is false for others.
func (g *cgen) tensorMethod(name, r string, x *ast.CallExpr) (string, bool) {
	if g.mode != deviceMode {
		return "", false
	}
	pos := g.sh.position(x, "tt.Tensor."+name+": index out of range")
	switch name {
	case "Zero":
		return "tsr::tns_clear(" + r + ")", true
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
