package compiler

import (
	_ "embed"
	"encoding/json"
	"fmt"
	"go/ast"
	"go/token"
	"go/types"
	"os"
	"path/filepath"
	"sort"
	"strconv"
	"strings"
)

// Goroutine programs (v0.3): main makes channels and starts goroutines with
// go f(args); every goroutine runs on a Tensix core of its own and the
// channels between them are ring buffers in L1 fed over the NoC. main runs on
// the host and may receive from channels that goroutines send on.

//go:embed runtime/chan_device.h
var chanDeviceRuntime string

//go:embed runtime/chan_host.h
var chanHostRuntime string

// Goroutines returns how many functions main starts with go (0 for tt.Parallel programs).
func (wp *WorkerProgram) Goroutines() int { return len(wp.goFuncs) }

// mainStartsGoroutines reports whether main contains a go statement.
func (c *checker) mainStartsGoroutines(main *ast.FuncDecl) bool {
	found := false
	ast.Inspect(main.Body, func(n ast.Node) bool {
		if _, ok := n.(*ast.GoStmt); ok {
			found = true
		}
		return !found
	})
	return found
}

func (c *checker) goroutines(main *ast.FuncDecl) {
	wp := &WorkerProgram{
		Name:  c.prog.Name,
		main:  main,
		paths: map[*types.Var]string{},
		sh: &cgShared{fset: c.fset, info: c.info, pkg: c.pkg, funcs: c.funcs,
			seenType: map[*types.Named]bool{}, goFIDs: map[*types.Func]int{},
			recursive: map[*types.Func]bool{}, heapKiB: c.heapKiB, stackKiB: c.stackKiB,
			undefinedOverflow: c.undefinedOverflow, forceL1Stack: c.l1Stack, goChips: c.goChips},
	}
	for _, st := range main.Body.List {
		as, ok := st.(*ast.AssignStmt)
		if !ok || len(as.Rhs) != 1 {
			continue
		}
		call, name := c.ttCall(as.Rhs[0])
		if name != "LoadBytes" && name != "OpenBytes" {
			continue
		}
		id, ok := as.Lhs[0].(*ast.Ident)
		if !ok || as.Tok != token.DEFINE || len(as.Lhs) != 1 {
			c.errorf(as, "declare the variable: x := tt.%s(path)", name)
			continue
		}
		path, ok := c.constString(call.Args[0], "the file name")
		if !ok {
			continue
		}
		v := c.info.Defs[id].(*types.Var)
		wp.loads = append(wp.loads, v)
		wp.paths[v] = path
	}
	ast.Inspect(main.Body, func(n ast.Node) bool {
		switch x := n.(type) {
		case *ast.CallExpr:
			if _, name := c.ttCall(x); name == "Parallel" || name == "LoadDetector" {
				c.errorf(x, "tt.%s cannot be used in a program that starts goroutines (yet)", name)
			}
		case *ast.FuncLit:
			c.errorf(x, "function literals are not supported in goroutine programs; start a named function: go f(args)")
			return false
		case *ast.GoStmt:
			c.goFunc(wp, x)
		}
		return true
	})
	if len(c.errs) > 0 {
		return
	}
	wp.device = c.reachable(wp.goFuncs, main, wp.sh.recursive)
	c.prog.Workers = wp
}

// directives reads the program's //tt: directives, comments that go run
// ignores. So far there is one: //tt:chip N, on the line right before a go
// statement in main, runs that goroutine on a core of chip N.
func (c *checker) directives(files []*ast.File, main *ast.FuncDecl, goroutines bool) map[*ast.GoStmt]int {
	type line struct {
		file string
		n    int
	}
	goAt := map[line]*ast.GoStmt{}
	if goroutines {
		ast.Inspect(main.Body, func(n ast.Node) bool {
			if g, ok := n.(*ast.GoStmt); ok {
				p := c.fset.Position(g.Pos())
				goAt[line{p.Filename, p.Line}] = g
			}
			return true
		})
	}
	chips := map[*ast.GoStmt]int{}
	for _, f := range files {
		for _, group := range f.Comments {
			for _, cm := range group.List {
				if !strings.HasPrefix(cm.Text, "//tt:") {
					continue
				}
				words := strings.Fields(strings.TrimPrefix(cm.Text, "//tt:"))
				name := ""
				if len(words) > 0 {
					name = words[0]
				}
				switch name {
				case "chip":
					p := c.fset.Position(cm.Pos())
					g := goAt[line{p.Filename, p.Line + 1}]
					if g == nil {
						c.errorf(cm, "//tt:chip goes on the line right before a go statement in main")
						continue
					}
					n, err := strconv.Atoi(strings.Join(words[1:], ""))
					if len(words) != 2 || err != nil || n < 0 {
						c.errorf(cm, "write //tt:chip N, with the number of the chip (0 or 1 on an n300)")
						continue
					}
					chips[g] = n
				case "host":
					c.errorf(cm, "//tt:host is not supported yet: goroutines run on the card's cores")
				default:
					c.errorf(cm, "unknown directive //tt:%s; Tessera knows //tt:chip N", name)
				}
			}
		}
	}
	return chips
}

// goFunc checks the function a go statement starts and gives it an id.
func (c *checker) goFunc(wp *WorkerProgram, s *ast.GoStmt) {
	id, ok := ast.Unparen(s.Call.Fun).(*ast.Ident)
	var f *types.Func
	if ok {
		f, _ = c.info.Uses[id].(*types.Func)
	}
	if f == nil || f.Pkg() != c.pkg || f.Type().(*types.Signature).Recv() != nil {
		c.errorf(s, "start goroutines with a function of this program: go f(args)")
		return
	}
	if _, seen := wp.sh.goFIDs[f]; seen {
		return
	}
	sig := f.Type().(*types.Signature)
	if sig.Results().Len() > 0 {
		c.errorf(s, "%s returns a value; a goroutine function has no results (send them on a channel)", f.Name())
	}
	for i := 0; i < sig.Params().Len(); i++ {
		p := sig.Params().At(i)
		t := p.Type()
		ch, ok := types.Unalias(t).Underlying().(*types.Chan)
		if lc, _, _, isList := chanList(t); isList {
			ch, ok = lc, true
		}
		switch {
		case isTT(t, "Bytes"), isInteger(t):
		case ok:
			if ch.Dir() == types.SendRecv {
				c.errorf(s, "%s: declare the direction of channel parameter %s (chan<- %s to send, <-chan %s to receive)",
					f.Name(), p.Name(), ch.Elem(), ch.Elem())
			}
		default:
			if b, ok := basicOf(t); ok && b.Kind() == types.Bool {
				continue
			}
			c.errorf(s, "%s: parameter %s is %s; goroutines take channels (also in arrays and slices), integers, bools and tt.Bytes",
				f.Name(), p.Name(), t)
		}
	}
	wp.sh.goFIDs[f] = len(wp.goFuncs)
	wp.goFuncs = append(wp.goFuncs, f)
}

// chanList reports whether t is an array or a slice of channels: the
// channel type, the array's length, and whether it is an array.
func chanList(t types.Type) (*types.Chan, int64, bool, bool) {
	var elem types.Type
	var n int64
	isArray := false
	switch u := types.Unalias(t).Underlying().(type) {
	case *types.Array:
		elem, n, isArray = u.Elem(), u.Len(), true
	case *types.Slice:
		elem = u.Elem()
	default:
		return nil, 0, false, false
	}
	ch, ok := types.Unalias(elem).Underlying().(*types.Chan)
	return ch, n, isArray, ok
}

// emitGoroutines writes the project for a goroutine program; fns and protos
// are the translated functions.
func emitGoroutines(wp *WorkerProgram, dir, fns string, protos []string) ([]string, error) {
	sh := wp.sh
	g := newCgen(sh, deviceMode)
	// A goroutine's arrays and slices of channels go on its heap.
	chansPos := -1
	for _, f := range wp.goFuncs {
		sig := f.Type().(*types.Signature)
		for k := 0; k < sig.Params().Len(); k++ {
			if _, _, _, isList := chanList(sig.Params().At(k).Type()); isList {
				sh.allocated = true
				chansPos = sh.position(wp.main, "out of memory: a goroutine's channels do not fit in its heap (tessc -heap gives it more)")
			}
		}
	}
	var km strings.Builder
	km.WriteString(`static void kernel_body() {
    uint32_t a = 0;
    const uint32_t fid = tsr::arg32(a);
    volatile uint32_t* status = reinterpret_cast<volatile uint32_t*>(tsr::arg32(a));
    const uint32_t fabric_at = tsr::arg32(a);  // where the fabric connection's arguments start; 0: none
#if defined(TSR_FABRIC)
    // A goroutine at an end of a channel to the other chip talks to it
    // through the fabric: an Ethernet router, or the fabric mux.
    tsr::FabricConn connection;
    if (fabric_at != 0) {
        size_t i = fabric_at;
        connection = tsr::fabric_connect(i);
        connection.open();
        tsr::fabric_init(&connection);
    }
#else
    (void)fabric_at;
#endif
    if (__builtin_setjmp(tsr::jmp) == 0) {
        switch (fid) {
`)
	for i, f := range wp.goFuncs {
		sig := f.Type().(*types.Signature)
		fmt.Fprintf(&km, "            case %d: {  // go %s(...)\n", i, f.Name())
		var args []string
		for k := 0; k < sig.Params().Len(); k++ {
			t := sig.Params().At(k).Type()
			arg := fmt.Sprintf("p%d", k)
			_, n, isArray, isList := chanList(t)
			switch {
			case isList && isArray:
				fmt.Fprintf(&km, "                tsr::Chan (&%s)[%d] = *reinterpret_cast<tsr::Chan (*)[%d]>(tsr::arg_chans(a, %d).ptr);\n",
					arg, n, n, chansPos)
			case isList:
				fmt.Fprintf(&km, "                const tsr::Slice<tsr::Chan> %s = tsr::arg_chans(a, %d);\n", arg, chansPos)
			case isTT(t, "Bytes"):
				fmt.Fprintf(&km, "                const tsr::Bytes %s = tsr::arg_bytes(a);\n", arg)
			case isInteger(t):
				fmt.Fprintf(&km, "                const %s %s = static_cast<%s>(tsr::arg64(a));\n", g.ct(t, wp.main), arg, g.ct(t, wp.main))
			default:
				if _, ok := types.Unalias(t).Underlying().(*types.Chan); ok {
					fmt.Fprintf(&km, "                const tsr::Chan %s = tsr::arg_chan(a);\n", arg)
				} else {
					fmt.Fprintf(&km, "                const bool %s = tsr::arg64(a) != 0;\n", arg)
				}
			}
			args = append(args, arg)
		}
		fmt.Fprintf(&km, "                prog::%s(%s);\n                break;\n            }\n", funcName(f), strings.Join(args, ", "))
	}
	km.WriteString(`        }
    }
#if defined(TSR_FABRIC)
    if (fabric_at != 0) {
        connection.close();
    }
#endif
`)
	if !sh.l1Stack() {
		fmt.Fprintf(&km, "    if (tsr::fail_pos < 0 && !tsr::local_stack_intact()) {\n        tsr::fail_pos = %d;  // the stack ran into the globals\n    }\n",
			sh.position(wp.main, "stack overflow: a goroutine's stack ran into the core's memory for its globals (its calls go too deep, or hold big structs); tessc -l1stack runs them on a stack in L1"))
	}
	km.WriteString(`    // {state, pad, position, value}: 1 = done, 2 = failed
    *reinterpret_cast<volatile int64_t*>(status + 2) = tsr::fail_pos;
    *reinterpret_cast<volatile int64_t*>(status + 4) = tsr::fail_value;
    status[0] = tsr::fail_pos >= 0 ? 2 : 1;
}

`)
	km.WriteString(kernelEntry(sh))
	structs := structDefs(sh)
	scratch := (sh.scratch + 31) / 32 * 32
	var kernel strings.Builder
	fmt.Fprintf(&kernel, `// Goroutine kernel for %s. %s
//
// Runs on RISC-V 0 (BRISC) of every Tensix core with a goroutine; the first
// runtime argument says which function to run. Arrays live in %d bytes of L1
// scratch; channels are rings in a region at the same L1 address on every core.
#include <cstdint>
#include "api/dataflow/dataflow_api.h"
#if defined(TSR_FABRIC)  // the host defines it when channels cross between chips
#include "tt_metal/fabric/hw/inc/tt_fabric_api.h"
#include "fabric/fabric_edm_packet_header.hpp"
#include "tt_metal/fabric/hw/inc/edm_fabric/fabric_connection_manager.hpp"
#include "tt_metal/fabric/hw/inc/packet_header_pool.h"
#if defined(TSR_FABRIC_MUX)  // and this when they go through the fabric mux
#include "tt_metal/fabric/hw/inc/tt_fabric_mux_v2_sender.hpp"
#endif
#endif

%s
%s
%s
namespace prog {

%s%s%s%s
%s
}  // namespace prog

%s`, wp.Name, generatedBy, scratch, wrapPragma(sh), deviceRuntime, chanDeviceRuntime, literalDefs(sh), structs, structHashes(sh), strings.Join(protos, "\n")+"\n", fns, km.String())

	h := newCgen(sh, hostMode)
	h.indent = 1
	h.line(fmt.Sprintf("tsr::Net net(ctx, %d);", scratch))
	for _, st := range wp.main.Body.List {
		h.hostStmt(wp, st, 0)
	}
	h.line("net.wait();")
	hostMain := h.b.String()

	host := hostProgram2(wp, hostRuntime+"\n"+chanHostRuntime, structs, hostMain, "1",
		"Runs main: makes the channels, starts each goroutine on a Tensix core of its own,\n"+
			"// and receives from the channels that goroutines send to main.")
	manifest, _ := json.MarshalIndent(map[string]any{
		"program":       wp.Name,
		"kind":          "goroutines",
		"scratch_bytes": scratch,
		"goroutines":    len(wp.goFuncs),
		"device_funcs":  len(wp.device),
		"stack_need":    wp.sh.stackNeed, // bytes the deepest chain of calls can need (an estimate)
		"stack":         map[bool]string{true: "L1", false: "local"}[wp.sh.l1Stack()],
	}, "", "  ")
	out := map[string]string{
		"CMakeLists.txt":         cmakeLists(&Program{Name: wp.Name}),
		"build.sh":               buildScript(&Program{Name: wp.Name}),
		"host.cpp":               host,
		"kernels/goroutines.cpp": kernel.String(),
		"program.json":           string(manifest) + "\n",
	}
	return writeFiles(dir, out)
}

func writeFiles(dir string, out map[string]string) ([]string, error) {
	names := make([]string, 0, len(out))
	for n := range out {
		names = append(names, n)
	}
	sort.Strings(names)
	var files []string
	for _, n := range names {
		p := filepath.Join(dir, n)
		if err := os.MkdirAll(filepath.Dir(p), 0o755); err != nil {
			return nil, err
		}
		mode := os.FileMode(0o644)
		if strings.HasSuffix(n, ".sh") {
			mode = 0o755
		}
		if err := os.WriteFile(p, []byte(out[n]), mode); err != nil {
			return nil, err
		}
		files = append(files, p)
	}
	return files, nil
}
