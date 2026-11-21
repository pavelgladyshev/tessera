// ---------------------------------------------------------------------------
// tt.Tensor, tt.Layer and tt.Matrix on the card (pasted in by tessc when
// worker code uses them).
//
// The numbers are bf16 in 32x32 tiles in L1. A tensor has a circular buffer
// of its own that is always full, so its tiles stay where they
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
    uint32_t row0;   // a view (Rows) starts at this row of the buffer's tensor
    uint32_t tiles;  // of the whole buffer
};

inline uint32_t tns_pages = 1;  // pages of cb 14
inline uint32_t tns_arena = 0;  // where they are
inline uint32_t tns_const = 0;  // the kernel's constant tiles

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
// of the max); tile 2: 1/rows in row 0, set for each MeanRows.
inline void tns_ready() {
    cb_push_back(kTnsArena, tns_pages);
    cb_reserve_back(kTnsConst, 3);
    const uint32_t c = get_write_ptr(kTnsConst);
    tns_const = c;
    tns_zero(c, 3 * kTnsTile);
    uint16_t* t = reinterpret_cast<uint16_t*>(c);
    for (uint32_t i = 0; i < 32; i++) {
        t[tns_elem(i, 0)] = 0x3f80;
    }
    for (uint32_t i = 0; i < 1024; i++) {
        t[1024 + i] = 0x3f80;
    }
    cb_push_back(kTnsConst, 3);
}

// A command for the tensor kernel: 16 words.
inline void tns_command(const uint32_t* c) {
    cb_reserve_back(kTnsCmd, 1);
    volatile uint32_t* p = reinterpret_cast<volatile uint32_t*>(get_write_ptr(kTnsCmd));
    for (uint32_t i = 0; i < 16; i++) {
        p[i] = c[i];
    }
    cb_push_back(kTnsCmd, 1);
}

inline void tns_wait() {
    cb_wait_front(kTnsDone, 1);
    cb_pop_front(kTnsDone, 1);
}

// An input of an operation, as the kernel takes it: buffer, first row tile,
// column tiles.
inline uint32_t tns_input(const Tensor& x) { return x.cb | (x.row0 / 32) << 8 | ((x.cols + 31) / 32) << 20; }

// A tensor's buffer, filled once and for all. The kernel pushes its pages:
// its packer writes the results there later, and a buffer's pages can be
// counted by one producer only. The worker reads and writes them in place.
inline void tns_make(uint32_t cb, uint32_t tiles) {
    tns_zero(get_write_ptr(cb), tiles * kTnsTile);
    const uint32_t c[16] = {3, cb, 0, 0, 0, 0, 0, 0, tiles};
    tns_command(c);
    tns_wait();
}

// Tells the tensor kernel that the worker is done.
inline void tns_stop() {
    const uint32_t c[16] = {0};
    tns_command(c);
}

// tt.NewTensor: the tensor of one place in the program, zeroed.
inline Tensor tns_new(uint32_t cb, uint32_t rows, uint32_t cols) {
    const uint32_t tiles = ((rows + 31) / 32) * ((cols + 31) / 32);
    tns_zero(get_read_ptr(cb), tiles * kTnsTile);
    return Tensor{cb, rows, cols, 0, tiles};
}

inline uint16_t* tns_at(const Tensor& t, int64_t r, int64_t c, int64_t pos) {
    if (static_cast<uint64_t>(r) >= t.rows || static_cast<uint64_t>(c) >= t.cols) {
        fail(pos, r);
    }
    const uint32_t nt = (t.cols + 31) / 32, ru = static_cast<uint32_t>(r) + t.row0, cu = static_cast<uint32_t>(c);
    return reinterpret_cast<uint16_t*>(get_read_ptr(t.cb) + ((ru >> 5) * nt + (cu >> 5)) * kTnsTile) +
           tns_elem(ru & 31, cu & 31);
}

// t.Zero(): a whole tensor at once, a view row by row.
inline void tns_clear(const Tensor& t, int64_t pos) {
    if (t.row0 == 0 && ((t.rows + 31) / 32) * ((t.cols + 31) / 32) == t.tiles) {
        tns_zero(get_read_ptr(t.cb), t.tiles * kTnsTile);
        return;
    }
    for (uint32_t r = 0; r < t.rows; r++) {
        for (uint32_t c = 0; c < t.cols; c++) {
            *tns_at(t, r, c, pos) = 0;
        }
    }
}

// t.Rows(first, n): whole tiles of rows, so that the matrix engine can work
// on them.
inline Tensor tns_rows(const Tensor& t, int64_t first, int64_t n, int64_t pos) {
    if (first < 0 || n <= 0 || first + n > t.rows || (first & 31) != 0 || ((n & 31) != 0 && first + n != t.rows)) {
        fail(pos, first);
    }
    return Tensor{t.cb, static_cast<uint32_t>(n), t.cols, t.row0 + static_cast<uint32_t>(first), t.tiles};
}

// Copies n values to row r of t from column c. The values of a row that are
// in one 16-column face of a tile lie one after another.
inline void tns_put(const Tensor& t, int64_t r, int64_t c, const uint16_t* src, uint32_t n, int64_t pos) {
    for (uint32_t j = 0; j < n;) {
        uint16_t* dst = tns_at(t, r, c + j, pos);
        uint32_t run = 16 - (static_cast<uint32_t>(c + j) & 15);
        if (run > n - j) {
            run = n - j;
        }
        for (uint32_t k = 0; k < run; k++) {
            dst[k] = src[j + k];
        }
        j += run;
    }
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
    tns_put(t, r, c, reinterpret_cast<const uint16_t*>(tns_arena + m.base * kTnsTile) + i * m.cols, m.cols, pos);
}

inline void tns_copy_row_of(const Tensor& t, int64_t r, int64_t c, const Tensor& s, int64_t i, int64_t pos) {
    if (static_cast<uint64_t>(i) >= s.rows || c < 0 || c + s.cols > t.cols) {
        fail(pos, i);
    }
    for (uint32_t j = 0; j < s.cols;) {
        uint32_t run = 16 - (j & 15);
        if (run > s.cols - j) {
            run = s.cols - j;
        }
        tns_put(t, r, c + j, tns_at(s, i, j, pos), run, pos);
        j += run;
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

// tt.Linear, tt.LinearReLU, tt.LinearCat and tt.LinearReLUCat: y from the
// n tensors of in side by side.
inline void tns_linear_n(const Tensor& y, const Layer& l, bool relu, int64_t pos, const Tensor* in, uint32_t n) {
    uint32_t cols = 0, kt = 0;
    uint32_t c[16] = {1, y.cb, y.row0 / 32, (y.rows + 31) / 32, (l.out + 31) / 32, l.base, relu ? 1u : 0u, n, y.tiles};
    for (uint32_t i = 0; i < n; i++) {
        // Every input but the last fills whole tiles of columns, so that
        // the next one's weights start at a tile too.
        if (n > 7 || in[i].rows != y.rows || in[i].cb == y.cb || (i + 1 < n && (in[i].cols & 31) != 0)) {
            fail(pos, i);
        }
        cols += in[i].cols;
        kt += (in[i].cols + 31) / 32;
        c[9 + i] = tns_input(in[i]);
    }
    if (n == 0 || cols != l.in || y.cols != l.out || kt != (l.in + 31) / 32) {
        fail(pos, l.in);
    }
    tns_command(c);
    if (l.base == kTnsStreamed) {
        // The weights, one output column per block of kTnsWBlock pages.
        const uint32_t nt = (l.out + 31) / 32;
        for (uint32_t col = 0; col < nt; col++) {
            for (uint32_t k = 0; k < kt + 1;) {
                const uint32_t m = kt + 1 - k < 8 ? kt + 1 - k : 8;
                cb_reserve_back(kTnsW, m);
                tns_read(l.dram, col * (kt + 1) + k, m, get_write_ptr(kTnsW));
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

template <typename... X>
inline void tns_linear(const Tensor& y, const Layer& l, bool relu, int64_t pos, const X&... xs) {
    const Tensor in[] = {xs...};
    tns_linear_n(y, l, relu, pos, in, sizeof...(X));
}

// tt.MaxRows (op 2) and tt.MeanRows (op 4): row 0 of y from the columns of x.
inline void tns_reduce(uint32_t op, const Tensor& y, const Tensor& x, int64_t pos) {
    if (y.rows != 1 || y.row0 != 0 || y.cols != x.cols || x.cb == y.cb || (op == 2 && (x.rows & 31) != 0)) {
        fail(pos, x.cols);
    }
    if (op == 4) {
        // 1/rows in row 0 of the kernel's third constant tile.
        uint16_t* t = reinterpret_cast<uint16_t*>(tns_const + 2 * kTnsTile);
        const uint16_t v = bf16_ratio(1, x.rows);
        for (uint32_t i = 0; i < 32; i++) {
            t[tns_elem(0, i)] = v;
        }
    }
    const uint32_t c[16] = {op, y.cb, 0, (x.rows + 31) / 32, (x.cols + 31) / 32, 0, 0, 1, y.tiles, tns_input(x)};
    tns_command(c);
    tns_wait();
}

}  // namespace tsr
// ---------------------------------------------------------------------------
