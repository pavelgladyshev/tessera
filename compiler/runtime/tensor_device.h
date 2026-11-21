// ---------------------------------------------------------------------------
// tt.Tensor, tt.Layer and tt.Matrix on the card (pasted in by tessc when
// worker code uses them).
//
// The numbers are bf16 in 32x32 tiles in L1. A tensor has a circular buffer
// of its own (16 and up) that is always full, so its tiles stay where they
// are (the kernel fills it once): the worker reads and writes them in place, and the tensor kernel
// (kernels/tensor.cpp, on the core's three compute processors) reads them
// with the matrix engine's unpacker and writes results with its packer. The
// worker sends the kernel one command per operation and waits for it.
//
// Layers' weights and matrices are in DRAM. Those that fit are copied once
// into L1 (cb 14); a bigger layer's weights are streamed for each call.
// ---------------------------------------------------------------------------
namespace tsr {

constexpr uint32_t kTnsTile = 2048;  // a 32x32 bf16 tile
constexpr uint32_t kTnsCmd = 1, kTnsW = 5, kTnsConst = 6, kTnsDone = 7, kTnsArena = 14;
constexpr uint32_t kTnsWBlock = 40;  // pages per output column of streamed weights
constexpr uint32_t kTnsStreamed = 0xffffffffu;

struct Layer {
    uint32_t dram;  // its tiles: for each output column tile, kt tiles of W, then the biases
    uint32_t in, out;
    uint32_t base;  // its first tile in cb 14, or kTnsStreamed
};
struct Matrix {
    uint32_t dram;  // its values, row by row
    uint32_t rows, cols;
    uint32_t base;  // its first page in cb 14
};
struct Tensor {
    uint32_t cb;
    uint32_t rows, cols;
};

inline uint32_t tns_pages = 1;  // pages of cb 14
inline uint32_t tns_arena = 0;  // where they are

inline Layer arg_layer(uint32_t& i) {
    Layer l;
    l.dram = arg32(i);
    l.in = arg32(i);
    l.out = arg32(i);
    l.base = arg32(i);
    tns_pages = arg32(i);
    return l;
}

inline Matrix arg_matrix(uint32_t& i) {
    Matrix m;
    m.dram = arg32(i);
    m.rows = arg32(i);
    m.cols = arg32(i);
    m.base = arg32(i);
    tns_pages = arg32(i);
    return m;
}

// Element offset of row r, column c in a tile (four 16x16 faces).
constexpr uint32_t tns_elem(uint32_t r, uint32_t c) {
    return ((r >> 4) * 2 + (c >> 4)) * 256 + (r & 15) * 16 + (c & 15);
}

// Reads tiles [first, first+n) of a DRAM buffer into L1, 8 at a time.
inline void tns_read(uint32_t buffer, uint32_t first, uint32_t n, uint32_t l1) {
    const InterleavedAddrGen<true> g = {buffer, kTnsTile};
    for (uint32_t t = 0; t < n; t++) {
        noc_async_read(g.get_noc_addr(first + t), l1 + t * kTnsTile, kTnsTile);
        if ((t & 7) == 7) {
            noc_async_read_barrier();
        }
    }
    noc_async_read_barrier();
}

inline void tns_zero(uint32_t addr, uint32_t bytes) {
    uint32_t* p = reinterpret_cast<uint32_t*>(addr);
    for (uint32_t i = 0; i < bytes / 4; i++) {
        p[i] = 0;
    }
}

// Before the worker: the weights and matrices kept in L1...
inline void tns_begin() {
    cb_reserve_back(kTnsArena, tns_pages);
    tns_arena = get_write_ptr(kTnsArena);
}

inline void layer_load(const Layer& l) {
    if (l.base != kTnsStreamed) {
        const uint32_t kt = (l.in + 31) / 32, nt = (l.out + 31) / 32;
        tns_read(l.dram, 0, nt * (kt + 1), tns_arena + l.base * kTnsTile);
    }
}

inline void matrix_load(const Matrix& m) {
    tns_read(m.dram, 0, (m.rows * m.cols * 2 + kTnsTile - 1) / kTnsTile, tns_arena + m.base * kTnsTile);
}

// ... and the kernel's constant tiles. Tile 0: ones in column 0 (times a
// bias tile, it adds the bias to every row); tile 1: all ones (the scaler
// of the max).
inline void tns_ready() {
    cb_push_back(kTnsArena, tns_pages);
    cb_reserve_back(kTnsConst, 2);
    const uint32_t c = get_write_ptr(kTnsConst);
    tns_zero(c, 2 * kTnsTile);
    uint16_t* t = reinterpret_cast<uint16_t*>(c);
    for (uint32_t i = 0; i < 32; i++) {
        t[tns_elem(i, 0)] = 0x3f80;
    }
    for (uint32_t i = 0; i < 1024; i++) {
        t[1024 + i] = 0x3f80;
    }
    cb_push_back(kTnsConst, 2);
}

inline void tns_command(uint32_t op, uint32_t y, uint32_t x, uint32_t mt, uint32_t kt, uint32_t nt, uint32_t wbase,
                        uint32_t relu) {
    cb_reserve_back(kTnsCmd, 1);
    volatile uint32_t* p = reinterpret_cast<volatile uint32_t*>(get_write_ptr(kTnsCmd));
    p[0] = op;
    p[1] = y;
    p[2] = x;
    p[3] = mt;
    p[4] = kt;
    p[5] = nt;
    p[6] = wbase;
    p[7] = relu;
    cb_push_back(kTnsCmd, 1);
}

inline void tns_wait() {
    cb_wait_front(kTnsDone, 1);
    cb_pop_front(kTnsDone, 1);
}

// A tensor's buffer, filled once and for all. The kernel pushes its pages:
// its packer writes the results there later, and a buffer's pages can be
// counted by one producer only. The worker reads and writes them in place.
inline void tns_make(uint32_t cb, uint32_t tiles) {
    tns_zero(get_write_ptr(cb), tiles * kTnsTile);
    tns_command(3, cb, 0, tiles, 0, 0, 0, 0);
    tns_wait();
}

// Tells the tensor kernel that the worker is done.
inline void tns_stop() { tns_command(0, 0, 0, 0, 0, 0, 0, 0); }

// tt.NewTensor: the tensor of one place in the program, zeroed.
inline Tensor tns_new(uint32_t cb, uint32_t rows, uint32_t cols) {
    tns_zero(get_read_ptr(cb), ((rows + 31) / 32) * ((cols + 31) / 32) * kTnsTile);
    return Tensor{cb, rows, cols};
}

inline void tns_clear(const Tensor& t) { tns_new(t.cb, t.rows, t.cols); }

inline uint16_t* tns_at(const Tensor& t, int64_t r, int64_t c, int64_t pos) {
    if (static_cast<uint64_t>(r) >= t.rows || static_cast<uint64_t>(c) >= t.cols) {
        fail(pos, r);
    }
    const uint32_t nt = (t.cols + 31) / 32, ru = static_cast<uint32_t>(r), cu = static_cast<uint32_t>(c);
    return reinterpret_cast<uint16_t*>(get_read_ptr(t.cb) + ((ru >> 5) * nt + (cu >> 5)) * kTnsTile) +
           tns_elem(ru & 31, cu & 31);
}

// num/den as bf16, rounded to nearest even.
inline uint16_t bf16_ratio(int64_t num, int64_t den) {
    if (num == 0) {
        return 0;
    }
    const bool neg = (num < 0) != (den < 0);
    uint64_t a = static_cast<uint64_t>(num < 0 ? -num : num), b = static_cast<uint64_t>(den < 0 ? -den : den);
    int32_t e = 0;
    while (a < b) {
        a <<= 1;
        e--;
    }
    while (a >= 2 * b) {
        b <<= 1;
        e++;
    }
    uint64_t rem = a - b;  // a/b is in [1, 2)
    uint32_t mant = 0;
    for (int i = 0; i < 7; i++) {
        rem <<= 1;
        mant <<= 1;
        if (rem >= b) {
            rem -= b;
            mant |= 1;
        }
    }
    rem <<= 1;
    if (rem > b || (rem == b && (mant & 1))) {
        if (++mant == 128) {
            mant = 0;
            e++;
        }
    }
    e += 127;
    if (e < 1) {
        return 0;
    }
    if (e > 254) {
        e = 254;
        mant = 127;
    }
    return static_cast<uint16_t>((neg ? 0x8000u : 0u) | (static_cast<uint32_t>(e) << 7) | mant);
}

// A bf16 value as an int that orders like the float.
inline int32_t tns_key(uint16_t v) {
    return (v & 0x8000) ? -static_cast<int32_t>(v & 0x7fff) : static_cast<int32_t>(v);
}

inline void tns_set(const Tensor& t, int64_t r, int64_t c, int64_t num, int64_t den, int64_t pos, int64_t pos_zero) {
    if (den == 0) {
        fail(pos_zero, 0);
    }
    *tns_at(t, r, c, pos) = bf16_ratio(num, den);
}

inline void tns_copy_row(const Tensor& t, int64_t r, int64_t c, const Matrix& m, int64_t i, int64_t pos) {
    if (static_cast<uint64_t>(i) >= m.rows || c < 0 || c + m.cols > t.cols) {
        fail(pos, i);
    }
    const uint16_t* src = reinterpret_cast<const uint16_t*>(tns_arena + m.base * kTnsTile) + i * m.cols;
    for (uint32_t j = 0; j < m.cols; j++) {
        *tns_at(t, r, c + j, pos) = src[j];
    }
}

inline void tns_copy_row_of(const Tensor& t, int64_t r, int64_t c, const Tensor& s, int64_t i, int64_t pos) {
    if (static_cast<uint64_t>(i) >= s.rows || c < 0 || c + s.cols > t.cols) {
        fail(pos, i);
    }
    for (uint32_t j = 0; j < s.cols; j++) {
        *tns_at(t, r, c + j, pos) = *tns_at(s, i, j, pos);
    }
}

inline int64_t tns_argmax(const Tensor& t, int64_t r, int64_t pos) {
    int64_t best = 0;
    int32_t best_key = tns_key(*tns_at(t, r, 0, pos));
    for (uint32_t c = 1; c < t.cols; c++) {
        const int32_t k = tns_key(*tns_at(t, r, c, pos));
        if (k > best_key) {
            best = c;
            best_key = k;
        }
    }
    return best;
}

// round(value * scale), halves away from zero.
inline int64_t tns_scaled(const Tensor& t, int64_t r, int64_t c, int64_t scale, int64_t pos) {
    const uint16_t v = *tns_at(t, r, c, pos);
    const int32_t e = (v >> 7) & 0xff;
    if (e == 0 || scale == 0) {
        return 0;
    }
    int64_t p = static_cast<int64_t>(128 | (v & 0x7f)) * scale;  // times 2^(e - 134)
    const bool neg = ((v & 0x8000) != 0) != (p < 0);
    uint64_t a = static_cast<uint64_t>(p < 0 ? -p : p);
    const int32_t sh = e - 134;
    if (sh >= 0) {
        a <<= sh;
    } else if (sh < -63) {
        a = 0;
    } else {
        a = (a + (uint64_t(1) << (-sh - 1))) >> -sh;
    }
    return neg ? -static_cast<int64_t>(a) : static_cast<int64_t>(a);
}

// tt.Linear and tt.LinearReLU.
inline void tns_linear(const Tensor& y, const Tensor& x, const Layer& l, bool relu, int64_t pos) {
    if (x.cols != l.in || y.cols != l.out || x.rows != y.rows || x.cb == y.cb) {
        fail(pos, l.in);
    }
    const uint32_t mt = (x.rows + 31) / 32, kt = (l.in + 31) / 32, nt = (l.out + 31) / 32;
    tns_command(1, y.cb, x.cb, mt, kt, nt, l.base, relu ? 1 : 0);
    if (l.base == kTnsStreamed) {
        // The weights, one output column per block of kTnsWBlock pages.
        for (uint32_t n = 0; n < nt; n++) {
            for (uint32_t k = 0; k < kt + 1;) {
                const uint32_t m = kt + 1 - k < 8 ? kt + 1 - k : 8;
                cb_reserve_back(kTnsW, m);
                tns_read(l.dram, n * (kt + 1) + k, m, get_write_ptr(kTnsW));
                cb_push_back(kTnsW, m);
                k += m;
            }
            if (kt + 1 < kTnsWBlock) {
                cb_reserve_back(kTnsW, kTnsWBlock - (kt + 1));
                cb_push_back(kTnsW, kTnsWBlock - (kt + 1));
            }
        }
    }
    tns_wait();
}

// tt.MaxRows.
inline void tns_max_rows(const Tensor& y, const Tensor& x, int64_t pos) {
    if (y.rows != 1 || y.cols != x.cols || x.cb == y.cb || (x.rows & 31) != 0) {
        fail(pos, x.cols);
    }
    tns_command(2, y.cb, x.cb, (x.rows + 31) / 32, 0, (x.cols + 31) / 32, 0, 0);
    tns_wait();
}

}  // namespace tsr
// ---------------------------------------------------------------------------
