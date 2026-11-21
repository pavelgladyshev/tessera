// Tensor kernel: the matrix engine at the service of worker code.
// Written by tessc for worker programs that use tt.Tensor.
//
// Runs on the three compute RISC-V cores (unpack, math, pack) of every core
// with a worker, and carries out the worker's tensor operations, one command
// at a time, until the worker sends the stop command. Values are bf16 in L1
// and fp32 in the destination registers.
//
// A tensor of mt x nt tiles lives in a circular buffer of its own, always
// full (this kernel pushes its pages, first with the make
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
constexpr uint32_t kOnes = 0, kScaler = 1, kAvg = 2;  // constant tiles in cb_const
constexpr uint32_t kWBlock = 40;
constexpr uint32_t kStop = 0, kLinear = 1, kMaxRows = 2, kMake = 3, kMeanRows = 4;
constexpr uint32_t kStreamed = 0xffffffffu;
constexpr uint32_t kMaxIn = 7;

// An input of a layer: kt column tiles of a tensor, from its row tile x0.
struct Input {
    uint32_t cb, x0, kt;
};

// Rows y0.. of y: act(sum over the inputs and k of x[m][k] W[k][n] + bias[n]).
// The inputs stand side by side: their columns take the layer's weights one
// input after another.
void linear(uint32_t y, uint32_t y0, uint32_t ytiles, uint32_t mt, uint32_t nt, uint32_t wbase, bool relu,
            const Input* in, uint32_t nin) {
    const bool streamed = wbase == kStreamed;
    const uint32_t wcb = streamed ? cb_w : cb_arena;
    uint32_t kt = 0;
    for (uint32_t i = 0; i < nin; i++) {
        kt += in[i].kt;
    }
    cb_pop_front(y, ytiles);  // the old values go; the same pages come back
    cb_reserve_back(y, ytiles);
    for (uint32_t n = 0; n < nt; n++) {
        uint32_t w0 = wbase + n * (kt + 1);
        if (streamed) {
            cb_wait_front(cb_w, kWBlock);
            w0 = 0;
        }
        for (uint32_t m = 0; m < mt; m++) {
            tile_regs_acquire();
            uint32_t w = w0;
            for (uint32_t i = 0; i < nin; i++) {
                matmul_init(in[i].cb, wcb);
                for (uint32_t k = 0; k < in[i].kt; k++) {
                    matmul_tiles(in[i].cb, wcb, (in[i].x0 + m) * in[i].kt + k, w++, 0);
                }
            }
            matmul_init(cb_const, wcb);
            matmul_tiles(cb_const, wcb, kOnes, w, 0);  // + bias
            if (relu) {
                relu_tile_init();
                relu_tile(0);
            }
            tile_regs_commit();
            tile_regs_wait();
            pack_tile<true>(0, y, (y0 + m) * nt + n);
            tile_regs_release();
        }
        if (streamed) {
            cb_pop_front(cb_w, kWBlock);
        }
    }
    cb_push_back(y, ytiles);
}

// Row 0 of y: the largest element of each column of x (mt row tiles from x0).
void max_rows(uint32_t y, uint32_t ytiles, uint32_t x, uint32_t x0, uint32_t mt, uint32_t nt) {
    cb_pop_front(y, ytiles);
    cb_reserve_back(y, ytiles);
    reduce_init<PoolType::MAX, ReduceDim::REDUCE_COL>(x, cb_const, y);
    for (uint32_t n = 0; n < nt; n++) {
        tile_regs_acquire();
        for (uint32_t m = 0; m < mt; m++) {
            reduce_tile<PoolType::MAX, ReduceDim::REDUCE_COL>(x, cb_const, (x0 + m) * nt + n, kScaler, 0);
        }
        tile_regs_commit();
        tile_regs_wait();
        pack_tile<true>(0, y, n);
        tile_regs_release();
    }
    reduce_uninit(x);
    cb_push_back(y, ytiles);
}

// Row 0 of y: the mean of each column of x. The worker has put 1/rows in
// row 0 of the constant tile kAvg: times a tile of x, it gives the sum of
// the tile's rows over the number of rows.
void mean_rows(uint32_t y, uint32_t ytiles, uint32_t x, uint32_t x0, uint32_t mt, uint32_t nt) {
    cb_pop_front(y, ytiles);
    cb_reserve_back(y, ytiles);
    matmul_init(cb_const, x);
    for (uint32_t n = 0; n < nt; n++) {
        tile_regs_acquire();
        for (uint32_t m = 0; m < mt; m++) {
            matmul_tiles(cb_const, x, kAvg, (x0 + m) * nt + n, 0);
        }
        tile_regs_commit();
        tile_regs_wait();
        pack_tile<true>(0, y, n);
        tile_regs_release();
    }
    cb_push_back(y, ytiles);
}

}  // namespace

void kernel_main() {
    compute_kernel_hw_startup<SrcOrder::Reverse>(cb_const, cb_arena, cb_const);
    cb_wait_front(cb_const, 3);  // kept for the whole run
    cb_wait_front(cb_arena, 1);  // likewise (all of it is pushed at once)
    while (true) {
        // A command: 16 words.
        cb_wait_front(cb_cmd, 1);
        uint32_t c[16];
        for (uint32_t i = 0; i < 16; i++) {
            c[i] = read_tile_value(cb_cmd, 0, i);
        }
        cb_pop_front(cb_cmd, 1);
        const uint32_t op = c[0], y = c[1], y0 = c[2], mt = c[3], nt = c[4], wbase = c[5], relu = c[6], nin = c[7],
                       ytiles = c[8];
        if (op == kStop) {
            break;
        }
        Input in[kMaxIn];
        for (uint32_t i = 0; i < kMaxIn; i++) {
            in[i] = {c[9 + i] & 0xff, (c[9 + i] >> 8) & 0xfff, c[9 + i] >> 20};
        }
        if (op == kLinear) {
            linear(y, y0, ytiles, mt, nt, wbase, relu != 0, in, nin);
        } else if (op == kMaxRows) {
            max_rows(y, ytiles, in[0].cb, in[0].x0, mt, nt);
        } else if (op == kMeanRows) {
            mean_rows(y, ytiles, in[0].cb, in[0].x0, mt, nt);
        } else if (op == kMake) {
            // A tensor's buffer becomes full, for good: this kernel is the
            // one that pushes its pages, now and with every result.
            cb_reserve_back(y, ytiles);
            cb_push_back(y, ytiles);
        }
        // Tell the worker that the result is there.
        cb_reserve_back(cb_done, 1);
        cb_push_back(cb_done, 1);
    }
}
