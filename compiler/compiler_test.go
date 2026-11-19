package compiler

import (
	"os"
	"path/filepath"
	"regexp"
	"strings"
	"testing"
)

// TestExamples compiles every example and checks the generated project.
func TestExamples(t *testing.T) {
	dirs, _ := filepath.Glob("../examples/*")
	if len(dirs) == 0 {
		t.Fatal("no examples found")
	}
	for _, dir := range dirs {
		t.Run(filepath.Base(dir), func(t *testing.T) {
			p, err := Load(dir)
			if err != nil {
				t.Fatal(err)
			}
			if p.Workers != nil {
				out := t.TempDir()
				files, err := EmitWorkers(p.Workers, out)
				if err != nil {
					t.Fatal(err)
				}
				kernel, wants := "kernels/worker.cpp", []string{"kernel_main", "prog::worker(", "__builtin_setjmp"}
				if p.Workers.Goroutines() > 0 {
					kernel, wants = "kernels/goroutines.cpp", []string{"kernel_main", "switch (fid)", "tsr::recv<", "tsr::send<"}
				}
				worker, _ := os.ReadFile(filepath.Join(out, kernel))
				for _, want := range wants {
					if !strings.Contains(string(worker), want) {
						t.Errorf("worker kernel lacks %s", want)
					}
				}
				t.Logf("worker program: %d files", len(files))
				return
			}
			k, err := Lower(p)
			if err != nil {
				t.Fatal(err)
			}
			out := t.TempDir()
			if _, err := Emit(p, k, out); err != nil {
				t.Fatal(err)
			}
			compute, _ := os.ReadFile(filepath.Join(out, "kernels/compute.cpp"))
			for _, want := range []string{"compute_kernel_hw_startup", "tile_regs_acquire", "pack_tile"} {
				if !strings.Contains(string(compute), want) {
					t.Errorf("compute kernel lacks %s", want)
				}
			}
			if k.Slots > slotsFullSync {
				t.Errorf("uses %d DEST slots", k.Slots)
			}
		})
	}
}

// TestErrors checks that each program in testdata/errors is rejected with
// the message named in its first-line comment: // ERROR "text".
func TestErrors(t *testing.T) {
	dirs, _ := filepath.Glob("testdata/errors/*")
	want := regexp.MustCompile(`// ERROR "([^"]+)"`)
	for _, dir := range dirs {
		t.Run(filepath.Base(dir), func(t *testing.T) {
			src, err := os.ReadFile(filepath.Join(dir, "main.go"))
			if err != nil {
				t.Fatal(err)
			}
			m := want.FindSubmatch(src)
			if m == nil {
				t.Fatal("missing // ERROR comment")
			}
			p, err := Load(dir)
			if err == nil && p.Workers != nil {
				_, err = EmitWorkers(p.Workers, t.TempDir())
			} else if err == nil {
				_, err = Lower(p)
			}
			if err == nil {
				t.Fatalf("compiled without error; want %q", m[1])
			}
			if !strings.Contains(err.Error(), string(m[1])) {
				t.Fatalf("error %q does not mention %q", err, m[1])
			}
			t.Log(err)
		})
	}
}

// TestFolding checks constant folding and the exact identities.
func TestFolding(t *testing.T) {
	g := newGraph()
	var pos = Node{}.Pos
	x := g.input(0, "x", pos)
	two := g.constant(2, pos)
	if n := g.node(OpMul, pos, "", two, g.constant(3, pos)); n.Op != OpConst || n.Val != 6 {
		t.Errorf("2*3 folded to %+v", n)
	}
	if n := g.node(OpAdd, pos, "", x, g.constant(0, pos)); n != x {
		t.Error("x+0 not simplified")
	}
	if a, b := g.node(OpExp, pos, "", x), g.node(OpExp, pos, "", x); a != b {
		t.Error("exp(x) not shared")
	}
	if n := g.node(OpWhere, pos, "", g.constant(1, pos), x, two); n != x {
		t.Error("where(true, x, 2) not simplified")
	}
}
