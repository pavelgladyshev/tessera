package compiler

import (
	_ "embed"
	"encoding/json"
	"fmt"
	"go/ast"
	"go/constant"
	"go/token"
	"go/types"
	"os"
	"path/filepath"
	"sort"
	"strings"
)

// Worker programs (v0.2): main loads byte arrays with tt.LoadBytes and runs
// one tt.Parallel; each worker is ordinary integer Go code that runs on its
// own Tensix core (RISC-V 0, the BRISC), reading the arrays from DRAM and
// appending results to its tt.List. The rest of main runs on the host.

//go:embed runtime/device.h
var deviceRuntime string

//go:embed runtime/host.h
var hostRuntime string

//go:embed runtime/detector_device.h
var detectorDeviceRuntime string

//go:embed runtime/detector_host.h
var detectorHostRuntime string

//go:embed runtime/detector.cpp
var detectorKernel string

// ListCapacity is how many results one worker can append.
const ListCapacity = 4096

// WorkerProgram is a checked worker program.
type WorkerProgram struct {
	Name     string
	sh       *cgShared
	main     *ast.FuncDecl
	parallel *ast.AssignStmt
	lit      *ast.FuncLit
	elem     types.Type   // T in tt.List[T]
	captures []*types.Var // main's variables the worker uses, by position
	loads    []*types.Var // variables assigned from tt.LoadBytes or tt.LoadDetector, in order
	paths    map[*types.Var]string
	device   []*types.Func // functions the worker calls, directly or not
	detector *types.Var    // the tt.Detector the worker uses, if any
	goFuncs  []*types.Func // goroutine programs: the functions main starts with go
	goHost   bool          // main stays Go (Options.Host)
	call     *ast.CallExpr // the tt.Parallel call
	files    []*ast.File
}

// ttFunc returns the name of the tt function e names, or "".
func (wp *WorkerProgram) ttFunc(e ast.Expr) string {
	sel, ok := e.(*ast.SelectorExpr)
	if !ok {
		return ""
	}
	if f, ok := wp.sh.info.Uses[sel.Sel].(*types.Func); ok && f.Pkg() != nil && f.Pkg().Path() == TTPath {
		return f.Name()
	}
	return ""
}

// mainCallsParallel reports whether main uses tt.Parallel anywhere.
func (c *checker) mainCallsParallel(main *ast.FuncDecl) bool {
	found := false
	ast.Inspect(main.Body, func(n ast.Node) bool {
		if call, ok := n.(*ast.CallExpr); ok {
			if f := c.ttFunc(call.Fun); f != nil && f.Name() == "Parallel" {
				found = true
			}
		}
		return !found
	})
	return found
}

func (c *checker) workers(main *ast.FuncDecl) {
	wp := &WorkerProgram{
		Name:   c.prog.Name,
		main:   main,
		paths:  map[*types.Var]string{},
		goHost: c.goHost,
		files:  c.files,
		sh: &cgShared{fset: c.fset, info: c.info, pkg: c.pkg, funcs: c.funcs,
			seenType: map[*types.Named]bool{}},
	}
	if c.goHost {
		// main stays Go, so tt.Parallel may be anywhere in it, and the
		// files main opens are whatever its code says.
		var calls []*ast.CallExpr
		ast.Inspect(main.Body, func(n ast.Node) bool {
			if call, ok := n.(*ast.CallExpr); ok {
				if f := c.ttFunc(call.Fun); f != nil && f.Name() == "Parallel" {
					calls = append(calls, call)
				}
			}
			return true
		})
		if len(calls) > 1 {
			c.errorf(calls[1], "so far a program can call tt.Parallel once")
			return
		}
		call := calls[0]
		lit, ok := call.Args[1].(*ast.FuncLit)
		if !ok {
			c.errorf(call.Args[1], "pass the worker as a function literal: func(w int, out *tt.List[T]) { ... }")
			return
		}
		wp.call, wp.lit = call, lit
		sig := c.info.TypeOf(lit).(*types.Signature)
		wp.elem = sig.Params().At(1).Type().(*types.Pointer).Elem().(*types.Named).TypeArgs().At(0)
	}
	for _, st := range main.Body.List {
		if c.goHost {
			break
		}
		as, ok := st.(*ast.AssignStmt)
		if !ok || len(as.Rhs) != 1 {
			continue
		}
		call, name := c.ttCall(as.Rhs[0])
		switch name {
		case "LoadBytes", "OpenBytes", "LoadDetector":
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
			if reservedFlags[id.Name] {
				c.errorf(id, "rename %q: the compiled program uses --%s as its own flag", id.Name, id.Name)
			}
			wp.loads = append(wp.loads, v)
			wp.paths[v] = path
		case "Parallel":
			if wp.parallel != nil {
				c.errorf(as, "v0.2 supports one tt.Parallel per program")
				continue
			}
			blank := false
			if id, ok := as.Lhs[0].(*ast.Ident); ok && id.Name == "_" {
				blank = true
			}
			if len(as.Lhs) != 1 || (as.Tok != token.DEFINE && !blank) {
				c.errorf(as, "assign the results: results := tt.Parallel(n, func(w int, out *tt.List[T]) {...})")
				continue
			}
			lit, ok := call.Args[1].(*ast.FuncLit)
			if !ok {
				c.errorf(call.Args[1], "pass the worker as a function literal: func(w int, out *tt.List[T]) { ... }")
				continue
			}
			wp.parallel, wp.lit = as, lit
			sig := c.info.TypeOf(lit).(*types.Signature)
			wp.elem = sig.Params().At(1).Type().(*types.Pointer).Elem().(*types.Named).TypeArgs().At(0)
		}
	}
	// tt.LoadBytes and tt.Parallel anywhere else are reported by the
	// translator, which only accepts them as statements of main.
	if wp.lit == nil || len(c.errs) > 0 {
		return
	}
	if named, ok := types.Unalias(wp.elem).(*types.Named); !ok || named.Obj().Pkg() != c.pkg {
		c.errorf(wp.lit, "the result type of tt.List must be a struct type of this program")
		return
	} else if _, ok := named.Underlying().(*types.Struct); !ok {
		c.errorf(wp.lit, "the result type of tt.List must be a struct type of this program")
		return
	}

	// Variables of main that the worker uses become runtime arguments.
	seen := map[*types.Var]bool{}
	ast.Inspect(wp.lit.Body, func(n ast.Node) bool {
		id, ok := n.(*ast.Ident)
		if !ok {
			return true
		}
		v, ok := c.info.Uses[id].(*types.Var)
		if !ok || seen[v] || v.Parent() == c.pkg.Scope() || (v.Pos() >= wp.lit.Pos() && v.Pos() < wp.lit.End()) {
			return true
		}
		if v.Parent() == nil {
			return true // a struct field
		}
		seen[v] = true
		t := v.Type()
		switch {
		case isTT(t, "Detector"):
			if wp.detector != nil {
				c.errorf(id, "a worker can use one tt.Detector (it runs on the core's matrix engine)")
			}
			wp.detector = v
		case !isTT(t, "Bytes") && !isInteger(t):
			if b, ok := basicOf(t); !ok || b.Kind() != types.Bool {
				c.errorf(id, "the worker can use main's tt.Bytes, tt.Detector, integer and bool variables; %s is %s", id.Name, t)
			}
		}
		wp.captures = append(wp.captures, v)
		return true
	})
	sort.Slice(wp.captures, func(i, j int) bool { return wp.captures[i].Pos() < wp.captures[j].Pos() })

	// The functions the worker reaches.
	var roots []*types.Func
	for _, f := range callGraph(c.info, c.pkg, wp.lit.Body) {
		if f.Name() == "main" {
			c.errorf(wp.lit, "the worker cannot call main")
			continue
		}
		roots = append(roots, f)
	}
	wp.device = c.reachable(roots, wp.lit)
	c.prog.Workers = wp
}

// reachable returns the roots and every function they call, callees first.
// There is no recursion, so every function has at most one activation and
// its arrays a fixed place in L1.
func (c *checker) reachable(roots []*types.Func, site ast.Node) []*types.Func {
	var out []*types.Func
	done := map[*types.Func]bool{}
	var visit func(f *types.Func, path []*types.Func)
	visit = func(f *types.Func, path []*types.Func) {
		for _, p := range path {
			if p == f {
				names := []string{}
				for _, q := range append(path, f) {
					names = append(names, q.Name())
				}
				c.errorf(c.funcs[f].Name, "recursion (%s): functions on the card keep their arrays in fixed places in L1, so they cannot recurse", strings.Join(names, " → "))
				return
			}
		}
		if done[f] {
			return
		}
		decl := c.funcs[f]
		if decl == nil || decl.Body == nil {
			c.errorf(site, "%s has no body", f.Name())
			return
		}
		for _, g := range callGraph(c.info, c.pkg, decl.Body) {
			if g.Name() == "main" {
				c.errorf(decl.Name, "%s cannot call main", f.Name())
				continue
			}
			visit(g, append(path, f))
		}
		done[f] = true
		out = append(out, f)
	}
	for _, f := range roots {
		visit(f, nil)
	}
	return out
}

// EmitWorkers writes the Metalium project for a worker program.
func EmitWorkers(wp *WorkerProgram, dir string) (files []string, err error) {
	defer func() {
		if r := recover(); r != nil {
			e, ok := r.(*Error)
			if !ok {
				panic(r)
			}
			files, err = nil, ErrorList{e}
		}
	}()
	sh := wp.sh
	sizes := types.SizesFor("gc", "amd64")
	const bounce = 4096 // tsr::kBounceBytes
	staging := uint32(bounce)
	var recordBytes, slot int64
	if wp.elem != nil { // tt.Parallel's result records
		recordBytes = sizes.Sizeof(wp.elem)
		slot = max(32, (recordBytes+31)/32*32)
	}
	sh.scratch = staging + uint32(slot)
	var detEmb, detCounts uint32
	if wp.detector != nil {
		detEmb = sh.allocScratch(4096)    // the byte embeddings
		detCounts = sh.allocScratch(1024) // the byte histogram
	}

	// Worker kernel: the functions, then the worker itself.
	var fns strings.Builder
	var protos []string
	for _, f := range wp.device {
		g := newCgen(sh, deviceMode)
		protos = append(protos, g.signature(f, sh.funcs[f])+";")
		g = newCgen(sh, deviceMode)
		g.indent = 0
		g.deviceFunc(f, sh.funcs[f])
		fns.WriteString(g.b.String())
		fns.WriteString("\n")
	}
	if wp.goFuncs != nil {
		return emitGoroutines(wp, dir, fns.String(), protos)
	}
	g := newCgen(sh, deviceMode)
	sig := sh.info.TypeOf(wp.lit).(*types.Signature)
	params := []string{
		g.ct(sig.Params().At(0).Type(), wp.lit) + " " + g.name(sig.Params().At(0)),
		g.ct(sig.Params().At(1).Type(), wp.lit) + " " + g.name(sig.Params().At(1)),
	}
	for _, v := range wp.captures {
		params = append(params, g.ct(v.Type(), wp.lit)+" "+g.name(v))
	}
	g.line("// The worker function given to tt.Parallel.")
	g.line("static void worker(" + strings.Join(params, ", ") + ") {")
	g.block(wp.lit.Body.List)
	g.line("}")
	fns.WriteString(g.b.String())

	elemType := g.ct(wp.elem, wp.lit)
	var km strings.Builder
	km.WriteString("void kernel_main() {\n")
	km.WriteString("    uint32_t a = 0;\n")
	km.WriteString("    tsr::scratch_base = get_write_ptr(0);\n")
	km.WriteString("    const int64_t index = tsr::arg64(a);\n")
	fmt.Fprintf(&km, "    tsr::List<%s> results = tsr::arg_list<%s>(a, tsr::scratch_base + %d);\n", elemType, elemType, staging)
	var callArgs = []string{"index", "&results"}
	for i, v := range wp.captures {
		arg := fmt.Sprintf("c%d", i)
		switch {
		case isTT(v.Type(), "Bytes"):
			fmt.Fprintf(&km, "    const tsr::Bytes %s = tsr::arg_bytes(a);  // %s\n", arg, v.Name())
		case isTT(v.Type(), "Detector"):
			fmt.Fprintf(&km, "    const tsr::Detector %s = tsr::arg_detector(a);  // %s\n", arg, v.Name())
			fmt.Fprintf(&km, "    tsr::det_init(%s, %d, %d);\n", arg, detEmb, detCounts)
		case isInteger(v.Type()):
			fmt.Fprintf(&km, "    const %s %s = static_cast<%s>(tsr::arg64(a));  // %s\n", g.ct(v.Type(), wp.lit), arg, g.ct(v.Type(), wp.lit), v.Name())
		default:
			fmt.Fprintf(&km, "    const bool %s = tsr::arg64(a) != 0;  // %s\n", arg, v.Name())
		}
		callArgs = append(callArgs, arg)
	}
	km.WriteString("    if (__builtin_setjmp(tsr::jmp) == 0) {\n")
	fmt.Fprintf(&km, "        prog::worker(%s);\n", strings.Join(callArgs, ", "))
	km.WriteString("    }\n")
	if wp.detector != nil {
		km.WriteString("    tsr::det_stop();\n")
	}
	km.WriteString("    results.finish(tsr::fail_pos >= 0);\n")
	km.WriteString("}\n")
	runtime := deviceRuntime
	if wp.detector != nil {
		runtime += "\n" + detectorDeviceRuntime
	}

	// Host program: main, translated (unless main stays Go).
	h := newCgen(sh, hostMode)
	h.indent = 1
	for _, st := range wp.main.Body.List {
		if wp.goHost {
			break
		}
		h.hostStmt(wp, st, slot)
	}
	hostMain := h.b.String()
	structs := structDefs(sh)
	scratch := (sh.scratch + 31) / 32 * 32

	var kernel strings.Builder
	fmt.Fprintf(&kernel, `// Worker kernel for %s. %s
//
// Runs on RISC-V 0 (BRISC) of every Tensix core with a worker: the Go code of
// the function given to tt.Parallel and of everything it calls, translated
// one function at a time. Arrays live in %d bytes of L1 scratch; tt.Bytes
// and the results live in DRAM.
#include <cstdint>
#include "api/dataflow/dataflow_api.h"

%s
namespace prog {

%s%s
%s
}  // namespace prog

%s`, wp.Name, generatedBy, scratch, runtime, structs, strings.Join(protos, "\n")+"\n", fns.String(), km.String())

	if wp.goHost {
		return emitGoHost(wp, dir, kernel.String(), scratch)
	}

	hostRT := hostRuntime
	for _, v := range wp.loads {
		if isTT(v.Type(), "Detector") {
			hostRT += "\n" + detectorHostRuntime
			break
		}
	}
	queues := "1"
	for _, st := range wp.main.Body.List {
		if as, ok := st.(*ast.AssignStmt); ok && len(as.Rhs) == 1 {
			if call, ok := as.Rhs[0].(*ast.CallExpr); ok {
				if f := wp.ttFunc(call.Fun); f == "OpenBytes" {
					queues = "2"
				}
			}
		}
	}
	host := hostProgram2(wp, hostRT, structs, hostMain, queues,
		"Runs main: loads the byte arrays into the card's DRAM, starts one worker per\n"+
			"// Tensix core for tt.Parallel, collects their results and carries on with the\n"+
			"// rest of main on the host.")
	manifest, _ := json.MarshalIndent(map[string]any{
		"program":       wp.Name,
		"kind":          "workers",
		"record_bytes":  recordBytes,
		"scratch_bytes": scratch,
		"device_funcs":  len(wp.device),
	}, "", "  ")
	out := map[string]string{
		"CMakeLists.txt":     cmakeLists(&Program{Name: wp.Name}),
		"build.sh":           buildScript(&Program{Name: wp.Name}),
		"host.cpp":           host,
		"kernels/worker.cpp": kernel.String(),
		"program.json":       string(manifest) + "\n",
	}
	if wp.detector != nil {
		out["kernels/detector.cpp"] = detectorKernel
	}
	names := make([]string, 0, len(out))
	for n := range out {
		names = append(names, n)
	}
	sort.Strings(names)
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

// hostStmt translates one statement of main. tt.LoadBytes and tt.Parallel
// are only allowed here, as `x := ...` statements.
func (g *cgen) hostStmt(wp *WorkerProgram, st ast.Stmt, slot int64) {
	if as, ok := st.(*ast.AssignStmt); ok && len(as.Rhs) == 1 {
		if call, ok := as.Rhs[0].(*ast.CallExpr); ok {
			if sel, ok := call.Fun.(*ast.SelectorExpr); ok {
				if id, ok := sel.X.(*ast.Ident); ok {
					if pn, ok := g.sh.info.Uses[id].(*types.PkgName); ok && pn.Imported().Path() == TTPath {
						switch sel.Sel.Name {
						case "LoadBytes":
							v := g.sh.info.Defs[as.Lhs[0].(*ast.Ident)].(*types.Var)
							g.line(fmt.Sprintf("tsr::Bytes %s = tsr::load_bytes(ctx, %s, paths.at(%s));",
								g.name(v), cstring(v.Name()), cstring(v.Name())))
							return
						case "OpenBytes":
							v := g.sh.info.Defs[as.Lhs[0].(*ast.Ident)].(*types.Var)
							g.line(fmt.Sprintf("tsr::Bytes %s = tsr::open_bytes(ctx, %s, paths.at(%s));",
								g.name(v), cstring(v.Name()), cstring(v.Name())))
							return
						case "LoadDetector":
							v := g.sh.info.Defs[as.Lhs[0].(*ast.Ident)].(*types.Var)
							g.line(fmt.Sprintf("tsr::Detector %s = tsr::load_detector(ctx, %s, paths.at(%s));",
								g.name(v), cstring(v.Name()), cstring(v.Name())))
							return
						case "Parallel":
							g.hostParallel(wp, as, call)
							return
						}
					}
				}
			}
		}
	}
	g.stmt(st)
}

func (g *cgen) hostParallel(wp *WorkerProgram, as *ast.AssignStmt, call *ast.CallExpr) {
	results := g.temp("discarded")
	if v, ok := g.sh.info.Defs[as.Lhs[0].(*ast.Ident)].(*types.Var); ok {
		results = g.name(v)
	}
	elem := g.ct(wp.elem, as)
	g.line(fmt.Sprintf("std::vector<%s> %s;", elem, results))
	g.line("{")
	g.indent++
	g.line(fmt.Sprintf("tsr::Workers job(ctx, %s, sizeof(%s), %d, %d, \"worker.cpp\");",
		g.expr(call.Args[0]), elem, ListCapacity, (g.sh.scratch+31)/32*32))
	if wp.detector != nil {
		g.line(fmt.Sprintf("tsr::add_detector(job, %s);  // its compute kernel runs on the same cores", g.name(wp.detector)))
	}
	for _, c := range wp.captures {
		if isTT(c.Type(), "Bytes") {
			g.line(fmt.Sprintf("job.serve(%s);  // read on demand if it came from tt.OpenBytes", g.name(c)))
		}
	}
	g.line("for (int64_t index = 0; index < job.n; index++) {")
	g.indent++
	g.line("std::vector<uint32_t> captured;")
	for _, c := range wp.captures {
		switch {
		case isTT(c.Type(), "Bytes"):
			g.line(fmt.Sprintf("job.push_bytes(captured, %s, index);", g.name(c)))
		case isTT(c.Type(), "Detector"):
			g.line(fmt.Sprintf("tsr::push_detector(captured, %s);", g.name(c)))
		case isInteger(c.Type()):
			g.line(fmt.Sprintf("tsr::push64(captured, static_cast<int64_t>(%s));", g.name(c)))
		default:
			g.line(fmt.Sprintf("tsr::push64(captured, %s ? 1 : 0);", g.name(c)))
		}
	}
	g.line("job.set_args(index, captured);")
	g.indent--
	g.line("}")
	g.line(fmt.Sprintf("%s = job.run<%s>();", results, elem))
	g.indent--
	g.line("}")
}

// unused keeps constant's import when only some paths use it.
var _ = constant.MakeBool

// hostIncludes are what the host runtime needs.
const hostIncludes = `#include <tt-metalium/allocator.hpp>
#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/distributed.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/tt_metal.hpp>
#include <tt-metalium/work_split.hpp>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <fcntl.h>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iostream>
#include <map>
#include <memory>
#include <mutex>
#include <span>
#include <sstream>
#include <string>
#include <sys/stat.h>
#include <thread>
#include <unistd.h>
#include <vector>

using namespace tt;
using namespace tt::tt_metal;
`

// hostProgram2 renders the host program: the runtime, the struct types, the
// failure positions, main as go_main and a C++ main that parses the flags.
func hostProgram2(wp *WorkerProgram, hostRT, structs, hostMain, queues, description string) string {
	var host strings.Builder
	fmt.Fprintf(&host, `// Host program for %s. %s
//
// %s
//
//   ./build.sh
//   ./build/%s [--NAME FILE ...] [--device N] [--kernels DIR] [--compile-only]
%s
%s
namespace prog {

%s}  // namespace prog

// Places in the Go source where a worker can fail at run time.
static const tsr::Position kPositions[] = {
`, wp.Name, generatedBy, description, cident(wp.Name), hostIncludes, hostRT, structs)
	for _, p := range wp.sh.positions {
		fmt.Fprintf(&host, "    {%s, %s},\n", cstring(p.where), cstring(p.what))
	}
	if len(wp.sh.positions) == 0 {
		host.WriteString("    {\"?\", \"?\"},\n")
	}
	host.WriteString("};\n\n")
	host.WriteString("// main, translated from Go.\n")
	host.WriteString("static void go_main(tsr::Context& ctx, const std::map<std::string, std::string>& paths) {\n")
	host.WriteString("    (void)paths;\n")
	host.WriteString(hostMain)
	host.WriteString("}\n\n")
	var loadNames []string
	for _, v := range wp.loads {
		loadNames = append(loadNames, fmt.Sprintf("{%s, %s}", cstring(v.Name()), cstring(wp.paths[v])))
	}
	fmt.Fprintf(&host, `int main(int argc, char** argv) {
    tsr::Context ctx;
    ctx.program = %s;
    ctx.kernel_dir = TSR_KERNEL_DIR;
    ctx.positions = kPositions;
    ctx.n_positions = sizeof kPositions / sizeof kPositions[0];
    tsr::current = &ctx;
    std::map<std::string, std::string> paths = {%s};
    int device_id = 0;
    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        auto value = [&]() -> std::string {
            if (i + 1 >= argc) {
                tsr::die("missing value after " + arg);
            }
            return argv[++i];
        };
        if (arg == "--device") {
            device_id = std::stoi(value());
        } else if (arg == "--kernels") {
            ctx.kernel_dir = value();
        } else if (arg == "--compile-only") {
            ctx.compile_only = true;
        } else if (arg.rfind("--", 0) == 0 && paths.count(arg.substr(2))) {
            paths[arg.substr(2)] = value();
        } else {
            std::cerr << "usage: " << ctx.program << " [--device N] [--kernels DIR] [--compile-only]";
            for (const auto& [name, path] : paths) {
                std::cerr << " [--" << name << " FILE (default " << path << ")]";
            }
            std::cerr << "\n";
            return arg == "--help" || arg == "-h" ? 0 : 2;
        }
    }
    // A program that reads files on demand uses a second command queue for
    // the data, while the first one runs the workers.
    ctx.mesh = distributed::MeshDevice::create_unit_mesh(device_id, DEFAULT_L1_SMALL_SIZE, DEFAULT_TRACE_REGION_SIZE, %s);
    go_main(ctx, paths);
    ctx.mesh->close();
    return 0;
}
`, cstring(cident(wp.Name)), strings.Join(loadNames, ", "), queues)
	return host.String()

}
