// Tensor kernel: the matrix engine at the service of worker code.
// Written by tessc for worker programs that use tt.Tensor.
//
// Runs on the three compute RISC-V cores (unpack, math, pack) of every core
// with a worker, and carries out the worker's tensor operations, one command
// at a time, until the worker sends the stop command. Values are bf16 in L1
// and fp32 in the destination registers.
//
// A tensor of mt x nt tiles lives in a circular buffer of its own (16 and
// up), always full (this kernel pushes its pages, first with the make
// command and then with every result): tile (m, n) is tile m*nt + n. A layer's weights are, for
// every output column tile n, the kt tiles of W for that column and then a
// tile with the column's biases in row 0. They are either kept in L1 (cb 14,
// from tile wbase) or streamed by the worker for each call (cb 5, one column
// per block of 40 pages).
#include <cstdint>
#include "api/compute/common.h"
#include "api/compute/compute_kernel_api.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/matmul.h"
#include "api/compute/reduce.h"
#include "api/compute/eltwise_unary/relu.h"

namespace {

constexpr uint32_t cb_cmd = 1, cb_w = 5, cb_const = 6, cb_done = 7, cb_arena = 14;
constexpr uint32_t kOnes = 0, kScaler = 1;  // constant tiles in cb_const
constexpr uint32_t kWBlock = 40;
constexpr uint32_t kStop = 0, kLinear = 1, kMaxRows = 2, kMake = 3;
constexpr uint32_t kStreamed = 0xffffffffu;

// y[m][n] = act(sum over k of x[m][k] W[k][n] + bias[n]).
void linear(uint32_t y, uint32_t x, uint32_t mt, uint32_t kt, uint32_t nt, uint32_t wbase, bool relu) {
    const bool streamed = wbase == kStreamed;
    const uint32_t wcb = streamed ? cb_w : cb_arena;
    cb_wait_front(x, mt * kt);
    cb_pop_front(y, mt * nt);  // the old values go; the same pages come back
    cb_reserve_back(y, mt * nt);
    for (uint32_t n = 0; n < nt; n++) {
        uint32_t w0 = wbase + n * (kt + 1);
        if (streamed) {
            cb_wait_front(cb_w, kWBlock);
            w0 = 0;
        }
        for (uint32_t m = 0; m < mt; m++) {
            tile_regs_acquire();
            matmul_init(x, wcb);
            for (uint32_t k = 0; k < kt; k++) {
                matmul_tiles(x, wcb, m * kt + k, w0 + k, 0);
            }
            matmul_init(cb_const, wcb);
            matmul_tiles(cb_const, wcb, kOnes, w0 + kt, 0);  // + bias
            if (relu) {
                relu_tile_init();
                relu_tile(0);
            }
            tile_regs_commit();
            tile_regs_wait();
            pack_tile<true>(0, y, m * nt + n);
            tile_regs_release();
        }
        if (streamed) {
            cb_pop_front(cb_w, kWBlock);
        }
    }
    cb_push_back(y, mt * nt);
}

// Row 0 of y: the largest element of each column of x (mt x nt tiles).
void max_rows(uint32_t y, uint32_t x, uint32_t mt, uint32_t nt) {
    cb_wait_front(x, mt * nt);
    cb_pop_front(y, nt);
    cb_reserve_back(y, nt);
    reduce_init<PoolType::MAX, ReduceDim::REDUCE_COL>(x, cb_const, y);
    for (uint32_t n = 0; n < nt; n++) {
        tile_regs_acquire();
        for (uint32_t m = 0; m < mt; m++) {
            reduce_tile<PoolType::MAX, ReduceDim::REDUCE_COL>(x, cb_const, m * nt + n, kScaler, 0);
        }
        tile_regs_commit();
        tile_regs_wait();
        pack_tile<true>(0, y, n);
        tile_regs_release();
    }
    reduce_uninit(x);
    cb_push_back(y, nt);
}

}  // namespace

void kernel_main() {
    compute_kernel_hw_startup<SrcOrder::Reverse>(cb_const, cb_arena, cb_const);
    cb_wait_front(cb_const, 2);  // kept for the whole run
    cb_wait_front(cb_arena, 1);  // likewise (all of it is pushed at once)
    while (true) {
        cb_wait_front(cb_cmd, 1);
        const uint32_t op = read_tile_value(cb_cmd, 0, 0);
        const uint32_t y = read_tile_value(cb_cmd, 0, 1);
        const uint32_t x = read_tile_value(cb_cmd, 0, 2);
        const uint32_t mt = read_tile_value(cb_cmd, 0, 3);
        const uint32_t kt = read_tile_value(cb_cmd, 0, 4);
        const uint32_t nt = read_tile_value(cb_cmd, 0, 5);
        const uint32_t wbase = read_tile_value(cb_cmd, 0, 6);
        const uint32_t relu = read_tile_value(cb_cmd, 0, 7);
        cb_pop_front(cb_cmd, 1);
        if (op == kStop) {
            break;
        }
        if (op == kLinear) {
            linear(y, x, mt, kt, nt, wbase, relu != 0);
        } else if (op == kMaxRows) {
            max_rows(y, x, mt, nt);
        } else if (op == kMake) {
            // A tensor's buffer becomes full, for good: this kernel is the
            // one that pushes its pages, now and with every result.
            cb_reserve_back(y, mt);
            cb_push_back(y, mt);
        }
        // Tell the worker that the result is there.
        cb_reserve_back(cb_done, 1);
        cb_push_back(cb_done, 1);
    }
}
