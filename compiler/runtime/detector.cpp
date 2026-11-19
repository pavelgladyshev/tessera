// Detector kernel: ndeca's PatchMLP block classifier on the matrix engine.
// Written by tessc for worker programs that use tt.Detector.
//
// Runs on the three compute RISC-V cores (unpack, math, pack) of every core
// with a worker, and serves the worker's Classify calls one batch at a time
// until the worker sends 0 on the command buffer. Values are bf16 in L1 and
// fp32 in the destination registers.
//
// For each block of a batch the kernel runs levels 1-4 with weights that
// stay in L1 (cb 14, loaded once) and pools them into row 0 of 26 tiles
// (cb 12). The worker moves each block's row to its own row of the batch
// (cb 15). The head then runs once for the whole batch, one block per row:
// the prefix branch (resident weights), the hidden layer (weights streamed
// from DRAM, cb 5, one output column per block of 40 pages) and the logits
// (resident weights).
#include <cstdint>
#include "api/compute/common.h"
#include "api/compute/compute_kernel_api.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/matmul.h"
#include "api/compute/reduce.h"
#include "api/compute/eltwise_unary/relu.h"

namespace {

constexpr uint32_t cb_cmd = 1, cb_xe = 2, cb_pre = 3, cb_hist = 4, cb_w = 5, cb_const = 6, cb_logits = 7;
constexpr uint32_t cb_t1 = 8, cb_h = 9, cb_j = 10, cb_k = 11, cb_pool = 12, cb_h1 = 13, cb_wres = 14, cb_z = 15,
                   cb_pre_out = 16;
constexpr uint32_t kOnes = 0, kAvg = 1, kScaler = 2;  // constant tiles in cb_const
constexpr uint32_t kWBlock = 40;
// Resident weights (cb_wres): where each layer's columns start.
constexpr uint32_t kResL1 = 0, kResL2 = 3, kResL3 = 15, kResL4 = 69, kResPre = 173, kResH2 = 191, kResTiles = 204;

// An input of a layer: tile (g, m, k) is base + g*group_stride + m*row_stride
// + k in its circular buffer.
struct LayerInput {
    uint32_t cb, base, group_stride, row_stride, kt;
};

// out[g][m][n] = act(sum over inputs and k of A[g][m][k] W[k][n] + bias[n]).
// Each output column's weights (the k tiles of every input, then the bias)
// are either resident in cb_wres from tile res_base, or streamed in cb_w.
template <bool kRelu, bool kStreamed>
void layer(const LayerInput* ins, uint32_t n_ins, uint32_t groups, uint32_t mt, uint32_t nt, uint32_t cb_out,
           uint32_t out_base, uint32_t out_group_stride, uint32_t out_row_stride, uint32_t res_base) {
    uint32_t per_col = 1;
    for (uint32_t i = 0; i < n_ins; i++) {
        per_col += ins[i].kt;
    }
    const uint32_t wcb = kStreamed ? cb_w : cb_wres;
    for (uint32_t n = 0; n < nt; n++) {
        uint32_t wbase = res_base + n * per_col;
        if constexpr (kStreamed) {
            cb_wait_front(cb_w, kWBlock);
            wbase = 0;
        }
        for (uint32_t g = 0; g < groups; g++) {
            for (uint32_t m = 0; m < mt; m++) {
                tile_regs_acquire();
                uint32_t w = wbase;
                for (uint32_t i = 0; i < n_ins; i++) {
                    const LayerInput& in = ins[i];
                    matmul_init(in.cb, wcb);
                    for (uint32_t k = 0; k < in.kt; k++) {
                        matmul_tiles(in.cb, wcb, in.base + g * in.group_stride + m * in.row_stride + k, w++, 0);
                    }
                }
                matmul_init(cb_const, wcb);
                matmul_tiles(cb_const, wcb, kOnes, w, 0);  // + bias
                if constexpr (kRelu) {
                    relu_tile_init();
                    relu_tile(0);
                }
                tile_regs_commit();
                tile_regs_wait();
                pack_tile<true>(0, cb_out, out_base + g * out_group_stride + m * out_row_stride + n);
                tile_regs_release();
            }
        }
        if constexpr (kStreamed) {
            cb_pop_front(cb_w, kWBlock);
        }
    }
}

// Column-wise max over all row tiles of each column n, into row 0 of cb_pool.
void pool_max(uint32_t cb_in, uint32_t groups, uint32_t group_stride, uint32_t mt, uint32_t row_stride, uint32_t nt,
              uint32_t out_base) {
    reduce_init<PoolType::MAX, ReduceDim::REDUCE_COL>(cb_in, cb_const, cb_pool);
    for (uint32_t n = 0; n < nt; n++) {
        tile_regs_acquire();
        for (uint32_t g = 0; g < groups; g++) {
            for (uint32_t m = 0; m < mt; m++) {
                reduce_tile<PoolType::MAX, ReduceDim::REDUCE_COL>(
                    cb_in, cb_const, g * group_stride + m * row_stride + n, kScaler, 0);
            }
        }
        tile_regs_commit();
        tile_regs_wait();
        pack_tile<true>(0, cb_pool, out_base + n);
        tile_regs_release();
    }
    reduce_uninit(cb_in);
}

// Levels 1-4 and pooling of one block.
void block() {
    // Level 1: 8 groups (token index mod 8) x 2 row tiles; 64 inputs.
    cb_wait_front(cb_xe, 32);
    cb_reserve_back(cb_t1, 16);
    {
        const LayerInput a[] = {{cb_xe, 0, 4, 2, 2}};
        layer<true, false>(a, 1, 8, 2, 1, cb_t1, 0, 2, 1, kResL1);
    }
    cb_push_back(cb_t1, 16);
    cb_pop_front(cb_xe, 32);

    // Level 2: group s pairs groups 2s and 2s+1 of level 1.
    cb_wait_front(cb_t1, 16);
    cb_reserve_back(cb_h, 32);
    {
        const LayerInput a[] = {{cb_t1, 0, 4, 1, 1}, {cb_t1, 2, 4, 1, 1}};
        layer<true, false>(a, 2, 4, 2, 4, cb_h, 0, 8, 4, kResL2);
    }
    cb_push_back(cb_h, 32);
    cb_pop_front(cb_t1, 16);

    // Level 3: group u pairs groups 2u and 2u+1 of level 2.
    cb_wait_front(cb_h, 32);
    cb_reserve_back(cb_j, 24);
    {
        const LayerInput a[] = {{cb_h, 0, 16, 4, 4}, {cb_h, 8, 16, 4, 4}};
        layer<true, false>(a, 2, 2, 2, 6, cb_j, 0, 12, 6, kResL3);
    }
    cb_push_back(cb_j, 24);

    // Level 4 pairs the two groups of level 3.
    cb_wait_front(cb_j, 24);
    cb_reserve_back(cb_k, 16);
    {
        const LayerInput a[] = {{cb_j, 0, 0, 6, 6}, {cb_j, 12, 0, 6, 6}};
        layer<true, false>(a, 2, 1, 2, 8, cb_k, 0, 0, 8, kResL4);
    }
    cb_push_back(cb_k, 16);
    cb_wait_front(cb_k, 16);

    // [max level 2, max level 3, max level 4, mean level 4], in row 0.
    cb_reserve_back(cb_pool, 26);
    pool_max(cb_h, 4, 8, 2, 4, 4, 0);
    pool_max(cb_j, 2, 12, 2, 6, 6, 4);
    pool_max(cb_k, 1, 0, 2, 8, 8, 10);
    matmul_init(cb_const, cb_k);
    for (uint32_t n = 0; n < 8; n++) {
        tile_regs_acquire();
        for (uint32_t m = 0; m < 2; m++) {
            matmul_tiles(cb_const, cb_k, kAvg, m * 8 + n, 0);
        }
        tile_regs_commit();
        tile_regs_wait();
        pack_tile<true>(0, cb_pool, 18 + n);
        tile_regs_release();
    }
    cb_push_back(cb_pool, 26);
    cb_pop_front(cb_h, 32);
    cb_pop_front(cb_j, 24);
    cb_pop_front(cb_k, 16);
}

}  // namespace

void kernel_main() {
    compute_kernel_hw_startup<SrcOrder::Reverse>(cb_xe, cb_wres, cb_t1);
    cb_wait_front(cb_const, 3);         // kept for the whole run
    cb_wait_front(cb_wres, kResTiles);  // likewise
    while (true) {
        cb_wait_front(cb_cmd, 1);
        const uint32_t n = read_tile_value(cb_cmd, 0, 0);  // blocks in this batch, 0 = stop
        cb_pop_front(cb_cmd, 1);
        if (n == 0) {
            break;
        }
        for (uint32_t b = 0; b < n; b++) {
            block();
        }

        // Head, one block per row: z (from the worker), the prefix branch and
        // the histogram -> 384 hidden units -> logits.
        cb_wait_front(cb_z, 26);
        cb_wait_front(cb_pre, 8);
        cb_wait_front(cb_hist, 8);
        cb_reserve_back(cb_pre_out, 2);
        {
            const LayerInput a[] = {{cb_pre, 0, 0, 0, 8}};
            layer<true, false>(a, 1, 1, 1, 2, cb_pre_out, 0, 0, 0, kResPre);
        }
        cb_push_back(cb_pre_out, 2);
        cb_wait_front(cb_pre_out, 2);
        cb_reserve_back(cb_h1, 12);
        {
            const LayerInput a[] = {{cb_z, 0, 0, 0, 26}, {cb_pre_out, 0, 0, 0, 2}, {cb_hist, 0, 0, 0, 8}};
            layer<true, true>(a, 3, 1, 1, 12, cb_h1, 0, 0, 0, 0);
        }
        cb_push_back(cb_h1, 12);
        cb_pop_front(cb_z, 26);
        cb_pop_front(cb_pre, 8);
        cb_pop_front(cb_hist, 8);
        cb_pop_front(cb_pre_out, 2);

        cb_wait_front(cb_h1, 12);
        cb_reserve_back(cb_logits, 1);
        {
            const LayerInput a[] = {{cb_h1, 0, 0, 0, 12}};
            layer<false, false>(a, 1, 1, 1, 1, cb_logits, 0, 0, 0, kResH2);
        }
        cb_push_back(cb_logits, 1);
        cb_pop_front(cb_h1, 12);
    }
}
