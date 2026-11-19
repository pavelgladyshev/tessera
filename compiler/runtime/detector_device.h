// ---------------------------------------------------------------------------
// tt.Detector on the card (pasted in by tessc when a worker uses one).
//
// ndeca's PatchMLP runs on the matrix engine of the worker's own core: the
// compute kernel (kernels/detector.cpp) runs on the three TRISCs and serves
// one batch of up to kBatch blocks at a time. This side, on the BRISC, turns
// blocks into the network's inputs, gathers each block's pooled features into
// its own row of the batch, streams the hidden layer's weights from DRAM once
// per batch, and reads back one row of logits per block.
//
// Network shape (ndeca model4096_large): 4096-byte blocks, 8-dim byte
// embeddings, levels of 32, 128, 192 and 256 channels, a 64-wide prefix
// branch, 384 hidden units. Tokens are grouped by index mod 8 (token 8q+r is
// row q of group r), so every level is a plain 64-row matrix product.
// ---------------------------------------------------------------------------
namespace tsr {

constexpr uint32_t kTile = 2048;  // a 32x32 bf16 tile
constexpr uint32_t kDetBlock = 4096;
constexpr uint32_t kBatch = 16;  // blocks per run of the head (at most 32: one row each)
constexpr uint32_t kCbCmd = 1, kCbXe = 2, kCbPre = 3, kCbHist = 4, kCbW = 5, kCbConst = 6, kCbLogits = 7;
constexpr uint32_t kCbPool = 12, kCbWres = 14, kCbZ = 15;
constexpr uint32_t kWBlock = 40;       // pages per output column of streamed weights (the widest needs 37)
constexpr uint32_t kResTiles = 204;    // weights kept in L1: levels 1-4, prefix branch, logits
constexpr uint32_t kHeadCols = 12, kHeadTiles = 37;  // the hidden layer, streamed once per batch
constexpr uint32_t kPooled = 26;       // tiles of pooled features per block

struct Detector {
    uint32_t resident;   // DRAM: the weights kept in L1, one bf16 tile per page
    uint32_t head;       // DRAM: the hidden layer's weights, one tile per page
    uint32_t emb;        // DRAM: the byte embeddings, 256 x 8 bf16 in one 4 KiB page
    uint32_t n_classes;
};

inline Detector arg_detector(uint32_t& i) {
    Detector d;
    d.resident = arg32(i);
    d.head = arg32(i);
    d.emb = arg32(i);
    d.n_classes = arg32(i);
    return d;
}

// Element offset of row r, column c in a tile (four 16x16 faces).
constexpr uint32_t tile_elem(uint32_t r, uint32_t c) {
    return ((r >> 4) * 2 + (c >> 4)) * 256 + (r & 15) * 16 + (c & 15);
}

inline uint32_t det_emb = 0;            // L1 copy of the embeddings
inline uint32_t* det_counts = nullptr;  // byte histogram

inline void copy16(uint32_t dst, uint32_t src) {
    uint32_t* d = reinterpret_cast<uint32_t*>(dst);
    const uint32_t* s = reinterpret_cast<const uint32_t*>(src);
    d[0] = s[0];
    d[1] = s[1];
    d[2] = s[2];
    d[3] = s[3];
}

inline void zero_l1(uint32_t addr, uint32_t bytes) {
    uint32_t* p = reinterpret_cast<uint32_t*>(addr);
    for (uint32_t i = 0; i < bytes / 4; i++) {
        p[i] = 0;
    }
}

// count / 4096 as bf16, rounded to nearest even.
inline uint16_t bf16_of_count(uint32_t count) {
    if (count == 0) {
        return 0;
    }
    const uint32_t e = 31 - static_cast<uint32_t>(__builtin_clz(count));
    const uint32_t m = count << (31 - e);
    const uint32_t mant = (m >> 24) & 0x7f;
    const uint32_t rest = m & 0xffffff;
    uint32_t bits = ((127 + e - 12) << 7) | mant;
    if (rest > 0x800000 || (rest == 0x800000 && (mant & 1))) {
        bits++;
    }
    return static_cast<uint16_t>(bits);
}

// A bf16 value as an int that orders like the float.
inline int32_t bf16_key(uint16_t v) {
    return (v & 0x8000) ? -static_cast<int32_t>(v & 0x7fff) : static_cast<int32_t>(v);
}

// Reads tiles [first, first+n) of a DRAM buffer into L1, 8 at a time.
inline void read_tiles(uint32_t buffer, uint32_t first, uint32_t n, uint32_t l1) {
    const InterleavedAddrGen<true> g = {buffer, kTile};
    for (uint32_t t = 0; t < n; t++) {
        noc_async_read(g.get_noc_addr(first + t), l1 + t * kTile, kTile);
        if ((t & 7) == 7) {
            noc_async_read_barrier();
        }
    }
    noc_async_read_barrier();
}

// Runs once, before the worker: loads the embeddings and the resident
// weights, and hands the compute kernel its constant tiles.
inline void det_init(const Detector& d, uint32_t emb_offset, uint32_t counts_offset) {
    det_emb = scratch_base + emb_offset;
    det_counts = scratch<uint32_t>(counts_offset);
    const InterleavedAddrGen<true> g = {d.emb, 4096};
    noc_async_read(g.get_noc_addr(0), det_emb, 4096);
    noc_async_read_barrier();

    cb_reserve_back(kCbWres, kResTiles);
    read_tiles(d.resident, 0, kResTiles, get_write_ptr(kCbWres));
    cb_push_back(kCbWres, kResTiles);

    // Tile 0: ones in column 0 (times a bias tile, adds the bias to every
    // row); tile 1: 1/64 in row 0 (a mean over 64 rows); tile 2: all ones
    // (the scaler of the max pooling).
    cb_reserve_back(kCbConst, 3);
    const uint32_t c = get_write_ptr(kCbConst);
    zero_l1(c, 3 * kTile);
    uint16_t* t = reinterpret_cast<uint16_t*>(c);
    for (uint32_t i = 0; i < 32; i++) {
        t[tile_elem(i, 0)] = 0x3f80;
        t[1024 + tile_elem(0, i)] = 0x3c80;
    }
    for (uint32_t i = 0; i < 1024; i++) {
        t[2048 + i] = 0x3f80;
    }
    cb_push_back(kCbConst, 3);

    // The batch inputs hold one row per block; rows of unused blocks stay
    // as they are (they are computed and ignored). Each batch reserves the
    // whole buffer, so it always gets these same pages.
    cb_reserve_back(kCbPre, 8);
    zero_l1(get_write_ptr(kCbPre), 8 * kTile);
    cb_reserve_back(kCbHist, 8);
    zero_l1(get_write_ptr(kCbHist), 8 * kTile);
    cb_reserve_back(kCbZ, kPooled);
    zero_l1(get_write_ptr(kCbZ), kPooled * kTile);
}

// Moves row 0 of every pooled tile of one block into row b of the batch.
inline void take_pooled(uint32_t z, uint32_t b) {
    cb_wait_front(kCbPool, kPooled);
    const uint32_t pool = get_read_ptr(kCbPool);
    for (uint32_t t = 0; t < kPooled; t++) {
        // Row 0 is the first 16 values of faces 0 and 1; row b is in faces
        // (b/16)*2 and (b/16)*2 + 1, at row b%16.
        const uint32_t from = pool + t * kTile;
        const uint32_t to = z + t * kTile + tile_elem(b, 0) * 2;
        for (uint32_t half = 0; half < 2; half++) {
            copy16(to + half * 512, from + half * 512);
            copy16(to + half * 512 + 16, from + half * 512 + 16);
        }
    }
    cb_pop_front(kCbPool, kPooled);
}

// Labels n blocks (n <= kBatch) laid out one after another at x.
inline void classify_some(const Detector& d, const uint8_t* x0, uint32_t n, int64_t* labels) {
    cb_reserve_back(kCbCmd, 1);
    *reinterpret_cast<volatile uint32_t*>(get_write_ptr(kCbCmd)) = n;
    cb_push_back(kCbCmd, 1);

    cb_reserve_back(kCbPre, 8);
    cb_reserve_back(kCbHist, 8);
    cb_reserve_back(kCbZ, kPooled);
    const uint32_t pre = get_write_ptr(kCbPre);
    uint16_t* h = reinterpret_cast<uint16_t*>(get_write_ptr(kCbHist));
    const uint32_t z = get_write_ptr(kCbZ);

    for (uint32_t b = 0; b < n; b++) {
        const uint8_t* x = x0 + b * kDetBlock;
        // Level-1 inputs: the 8 embeddings of each 8-byte token, as a row of 64.
        cb_reserve_back(kCbXe, 32);
        const uint32_t xe = get_write_ptr(kCbXe);
        for (uint32_t j = 0; j < 512; j++) {
            const uint32_t q = j >> 3, r = j & 7, m = q >> 5, rr = q & 31;
            for (uint32_t s = 0; s < 8; s++) {
                const uint32_t tile = r * 4 + m * 2 + (s >> 2);
                copy16(xe + tile * kTile + tile_elem(rr, (s & 3) * 8) * 2, det_emb + x[8 * j + s] * 16u);
            }
        }
        cb_push_back(kCbXe, 32);

        // Row b of the prefix input: the first 32 bytes' embeddings.
        for (uint32_t i = 0; i < 32; i++) {
            copy16(pre + (i >> 2) * kTile + tile_elem(b, (i & 3) * 8) * 2, det_emb + x[i] * 16u);
        }
        // Row b of the histogram input: byte counts / 4096.
        for (uint32_t i = 0; i < 256; i++) {
            det_counts[i] = 0;
        }
        for (uint32_t i = 0; i < kDetBlock; i++) {
            det_counts[x[i]]++;
        }
        for (uint32_t v = 0; v < 256; v++) {
            h[(v >> 5) * 1024 + tile_elem(b, v & 31)] = bf16_of_count(det_counts[v]);
        }
        // Collect the previous block's pooled features while this one runs.
        if (b > 0) {
            take_pooled(z, b - 1);
        }
    }
    take_pooled(z, n - 1);
    cb_push_back(kCbPre, 8);
    cb_push_back(kCbHist, 8);
    cb_push_back(kCbZ, kPooled);

    // The hidden layer's weights, one output column per block of kWBlock pages.
    for (uint32_t col = 0; col < kHeadCols; col++) {
        for (uint32_t k = 0; k < kHeadTiles;) {
            const uint32_t m = kHeadTiles - k < 8 ? kHeadTiles - k : 8;
            cb_reserve_back(kCbW, m);
            read_tiles(d.head, col * kHeadTiles + k, m, get_write_ptr(kCbW));
            cb_push_back(kCbW, m);
            k += m;
        }
        cb_reserve_back(kCbW, kWBlock - kHeadTiles);
        cb_push_back(kCbW, kWBlock - kHeadTiles);
    }

    // Row b of the logits tile is block b's.
    cb_wait_front(kCbLogits, 1);
    const uint16_t* lg = reinterpret_cast<const uint16_t*>(get_read_ptr(kCbLogits));
    for (uint32_t b = 0; b < n; b++) {
        int64_t best = 0;
        int32_t best_key = bf16_key(lg[tile_elem(b, 0)]);
        for (uint32_t c = 1; c < d.n_classes; c++) {
            const int32_t k = bf16_key(lg[tile_elem(b, c)]);
            if (k > best_key) {
                best = c;
                best_key = k;
            }
        }
        labels[b] = best;
    }
    cb_pop_front(kCbLogits, 1);
}

inline void classify_batch(const Detector& d, Slice<uint8_t> blocks, Slice<int64_t> labels, int64_t pos) {
    if (blocks.len != labels.len * static_cast<int64_t>(kDetBlock)) {
        fail(pos, blocks.len);
    }
    for (int64_t i = 0; i < labels.len; i += kBatch) {
        const uint32_t n = labels.len - i < kBatch ? static_cast<uint32_t>(labels.len - i) : kBatch;
        classify_some(d, blocks.ptr + i * kDetBlock, n, labels.ptr + i);
    }
}

inline int64_t classify(const Detector& d, Slice<uint8_t> block, int64_t pos) {
    if (block.len != kDetBlock) {
        fail(pos, block.len);
    }
    int64_t label = 0;
    classify_some(d, block.ptr, 1, &label);
    return label;
}

// Tells the compute kernel that the worker is done.
inline void det_stop() {
    cb_reserve_back(kCbCmd, 1);
    *reinterpret_cast<volatile uint32_t*>(get_write_ptr(kCbCmd)) = 0;
    cb_push_back(kCbCmd, 1);
}

}  // namespace tsr
// ---------------------------------------------------------------------------
