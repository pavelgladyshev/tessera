// ---------------------------------------------------------------------------
// Tessera device runtime for worker kernels (pasted in by tessc).
//
// A worker runs on RISC-V 0 (BRISC) of one Tensix core. It has no floating
// point unit and a few KB of local memory, so Go arrays live in a scratch area
// of the core's 1.5 MB L1, and tt.Bytes / tt.List live in DRAM and are
// reached over the network-on-chip.
// ---------------------------------------------------------------------------
namespace tsr {

// Runtime failures (an index out of range, say) jump back to kernel_main,
// which reports them to the host with the Go source position.
inline void* jmp[5];
inline int64_t fail_pos = -1;
inline int64_t fail_value = 0;

[[noreturn]] inline void fail(int64_t pos, int64_t value) {
    fail_pos = pos;
    fail_value = value;
    __builtin_longjmp(jmp, 1);
}

// The scratch area: one circular buffer's worth of L1, carved up by tessc.
inline uint32_t scratch_base = 0;
template <typename T>
inline T* scratch(uint32_t offset) {
    return reinterpret_cast<T*>(scratch_base + offset);
}
template <typename T>
inline void zero(T* p, uint32_t n) {
    uint32_t* w = reinterpret_cast<uint32_t*>(p);
    for (uint32_t i = 0, words = (n * sizeof(T) + 3) / 4; i < words; i++) {
        w[i] = 0;
    }
}

// Go slices.
template <typename T>
struct Slice {
    T* ptr;
    int64_t len;
    T& at(int64_t i, int64_t pos) const {
        if (static_cast<uint64_t>(i) >= static_cast<uint64_t>(len)) {
            fail(pos, i);
        }
        return ptr[i];
    }
    Slice sub(int64_t lo, int64_t hi, int64_t pos) const {
        if (lo < 0 || hi < lo || hi > len) {
            fail(pos, hi);
        }
        return Slice{ptr + lo, hi - lo};
    }
};

template <typename T>
inline T& at(T* p, int64_t n, int64_t i, int64_t pos) {
    if (static_cast<uint64_t>(i) >= static_cast<uint64_t>(n)) {
        fail(pos, i);
    }
    return p[i];
}

template <typename T>
inline Slice<T> slice(T* p, int64_t n, int64_t lo, int64_t hi, int64_t pos) {
    if (lo < 0 || hi < lo || hi > n) {
        fail(pos, hi);
    }
    return Slice<T>{p + lo, hi - lo};
}

// Arrays and copies. A Go array declared with var lives in L1 as a pointer
// to its first element (Ptr<T>, which also spells pointers to arrays of
// arrays); arrays inside structs are C++ arrays.
template <typename T>
using Ptr = T*;

// Moves n bytes, as memmove does (Go's copy allows overlapping slices).
inline void move_bytes(uint8_t* to, const uint8_t* from, int64_t n) {
    if (to == from || n <= 0) {
        return;
    }
    const bool words = ((reinterpret_cast<uintptr_t>(to) | reinterpret_cast<uintptr_t>(from) | n) & 3) == 0;
    if (to < from || to >= from + n) {
        if (words) {
            uint32_t* t = reinterpret_cast<uint32_t*>(to);
            const uint32_t* f = reinterpret_cast<const uint32_t*>(from);
            for (int64_t k = 0; k < n / 4; k++) {
                t[k] = f[k];
            }
        } else {
            for (int64_t k = 0; k < n; k++) {
                to[k] = from[k];
            }
        }
        return;
    }
    for (int64_t k = n - 1; k >= 0; k--) {
        to[k] = from[k];
    }
}

// a = b for arrays of n elements.
template <typename T>
inline void assign_array(T* to, const T* from, int64_t n) {
    move_bytes(reinterpret_cast<uint8_t*>(to), reinterpret_cast<const uint8_t*>(from),
               n * static_cast<int64_t>(sizeof(T)));
}

// Go's copy builtin.
template <typename T>
inline int64_t copy(Slice<T> to, Slice<T> from) {
    const int64_t n = to.len < from.len ? to.len : from.len;
    move_bytes(reinterpret_cast<uint8_t*>(to.ptr), reinterpret_cast<const uint8_t*>(from.ptr),
               n * static_cast<int64_t>(sizeof(T)));
    return n;
}

// Go's == for values made of integers, bools, structs and arrays.
template <typename T>
inline bool eq(const T& a, const T& b) {
    return a == b;
}
template <typename T, decltype(sizeof(0)) N>
inline bool eq(const T (&a)[N], const T (&b)[N]) {
    for (decltype(sizeof(0)) i = 0; i < N; i++) {
        if (!eq(a[i], b[i])) {
            return false;
        }
    }
    return true;
}
template <typename T>
inline bool eq_n(const T* a, const T* b, int64_t n) {
    for (int64_t i = 0; i < n; i++) {
        if (!eq(a[i], b[i])) {
            return false;
        }
    }
    return true;
}

// Go's min and max builtins.
template <typename T>
constexpr T min(T a) { return a; }
template <typename T, typename... R>
constexpr T min(T a, T b, R... rest) { return min<T>(b < a ? b : a, rest...); }
template <typename T>
constexpr T max(T a) { return a; }
template <typename T, typename... R>
constexpr T max(T a, T b, R... rest) { return max<T>(b > a ? b : a, rest...); }

// Integer division and shifts with Go's rules.
template <typename T>
inline T div(T a, T b, int64_t pos) {
    if (b == 0) {
        fail(pos, 0);
    }
    return a / b;
}
template <typename T>
inline T rem(T a, T b, int64_t pos) {
    if (b == 0) {
        fail(pos, 0);
    }
    return a % b;
}
template <typename T, typename S>
inline T shl(T a, S n) {
    return static_cast<uint64_t>(n) >= sizeof(T) * 8 ? T(0) : T(a << n);
}
template <typename T, typename S>
inline T shr(T a, S n) {
    if (static_cast<uint64_t>(n) >= sizeof(T) * 8) {
        return a < T(0) ? T(-1) : T(0);
    }
    return T(a >> n);
}

// Runtime arguments are 32-bit words; 64-bit values take two.
inline uint32_t arg32(uint32_t& i) { return get_arg_val<uint32_t>(i++); }
inline int64_t arg64(uint32_t& i) {
    uint64_t lo = arg32(i);
    uint64_t hi = arg32(i);
    return static_cast<int64_t>(lo | (hi << 32));
}

// tt.Bytes, two kinds:
//   kInDram    (tt.LoadBytes): the whole array, in an interleaved DRAM buffer
//              with power-of-two pages.
//   kFromHost  (tt.OpenBytes): the file stays on the host. The worker asks
//              for a range through a mailbox in its own L1 and the host
//              writes it into the window that follows the mailbox.
constexpr uint32_t kInDram = 0, kFromHost = 1;

struct Bytes {
    uint32_t kind;
    uint32_t addr;       // kInDram: the DRAM buffer; kFromHost: the mailbox in L1
    uint32_t page_log2;  // kInDram: page size; kFromHost: window size
    int64_t len;
    uint64_t table;      // kFromHost: NoC address of this worker's slot in the request table
    uint32_t staged;     // kFromHost: DRAM buffer where the host puts the data (4 KiB pages)
    uint32_t staged_page;  // kFromHost: this worker's first page in it
};

inline Bytes arg_bytes(uint32_t& i) {
    Bytes b;
    b.kind = arg32(i);
    b.addr = arg32(i);
    b.page_log2 = arg32(i);
    b.len = arg64(i);
    b.table = 0;
    b.staged = 0;
    b.staged_page = 0;
    if (b.kind == 1) {
        const uint32_t x = arg32(i), y = arg32(i), slot = arg32(i);
        b.table = get_noc_addr(x, y, slot);
        b.staged = arg32(i);
        b.staged_page = arg32(i);
    }
    return b;
}

// The mailbox, 64 bytes at the start of the worker's L1 region; the host
// zeroes it before the workers start. Its first 32 bytes are the request,
// which the worker copies into the table. What the window holds is recorded
// here rather than in Bytes, because Bytes is passed around by value.
struct Mailbox {
    volatile uint32_t req_seq;     // worker: number of the latest request
    volatile uint32_t req_count;   // worker: 0 = one range; else pieces of a strided read
    volatile uint32_t req_len;     // worker: bytes wanted (of each piece)
    uint32_t unused;
    volatile int64_t req_off;      // worker: file offset wanted (of the first piece)
    volatile int64_t req_stride;   // worker: from one piece to the next
    volatile uint32_t ack_seq;     // host: number of the latest request served
    volatile uint32_t got_len;     // host: bytes delivered
    int64_t win_off;               // what the window holds now
    int64_t win_len;
    uint32_t pre_seq;              // the read-ahead request for the next window, 0 if none
    uint32_t unused2;
};
static_assert(sizeof(Mailbox) <= 64, "the mailbox is 64 bytes");
constexpr uint32_t kMailboxBytes = 64;
constexpr uint32_t kFetchAlign = 4096;  // the host reads whole 4 KiB blocks

inline void copy_bytes(uint8_t* to, const uint8_t* from, uint32_t n) {
    if (((reinterpret_cast<uintptr_t>(to) | reinterpret_cast<uintptr_t>(from) | n) & 3) == 0) {
        uint32_t* t = reinterpret_cast<uint32_t*>(to);
        const uint32_t* f = reinterpret_cast<const uint32_t*>(from);
        for (uint32_t k = 0; k < n / 4; k++) {
            t[k] = f[k];
        }
        return;
    }
    for (uint32_t k = 0; k < n; k++) {
        to[k] = from[k];
    }
}

// Asks the host for data: len bytes at off when count is 0, else count
// pieces of len bytes, piece i at off + i*stride. Returns the request's
// number. A newer request replaces one the host has not served yet, so a
// worker never waits for a read-ahead it no longer needs.
inline uint32_t post(const Bytes& b, int64_t off, uint32_t len, uint32_t count, int64_t stride) {
    Mailbox* mb = reinterpret_cast<Mailbox*>(b.addr);
    mb->req_off = off;
    mb->req_len = len;
    mb->req_count = count;
    mb->req_stride = stride;
    const uint32_t seq = mb->req_seq + 1;
    mb->req_seq = seq;
    // Post the request in the table on worker 0's core, which the host
    // reads in one go.
    noc_async_write(b.addr, b.table, 32);
    noc_async_write_barrier();
    return seq;
}

// Waits for the answer to request seq in this core's mailbox. The host has
// put the data in this worker's part of the staging buffer in DRAM (its fast
// path); pull it into the window, the pieces of a strided read one after
// another. Returns the bytes delivered.
inline uint32_t wait_pull(const Bytes& b, uint32_t seq) {
    Mailbox* mb = reinterpret_cast<Mailbox*>(b.addr);
    while (mb->ack_seq != seq) {
    }
    const uint32_t got = mb->got_len;
    const InterleavedAddrGen<true> g = {b.staged, kFetchAlign};
    const uint32_t to = b.addr + kMailboxBytes;
    for (uint32_t k = 0; k * kFetchAlign < got; k++) {
        uint32_t c = got - k * kFetchAlign;
        if (c > kFetchAlign) {
            c = kFetchAlign;
        }
        noc_async_read(g.get_noc_addr(b.staged_page + k), to + k * kFetchAlign, (c + 31) & ~31u);
    }
    noc_async_read_barrier();
    return got;
}

inline uint32_t fetch(const Bytes& b, int64_t off, uint32_t len, uint32_t count, int64_t stride) {
    reinterpret_cast<Mailbox*>(b.addr)->pre_seq = 0;  // a pending read-ahead is dropped
    return wait_pull(b, post(b, off, len, count, stride));
}

// Reads n bytes at off from the host's file. A read that misses the window
// asks for the 4 KiB blocks it needs; one that continues where the window
// ends asks for a whole window ahead (sequential read-ahead). After a whole
// window, the worker asks for the next one straight away, so that the host
// reads it while the worker uses this one.
inline int64_t read_from_host(const Bytes& b, Slice<uint8_t> dst, int64_t off, int64_t n) {
    Mailbox* mb = reinterpret_cast<Mailbox*>(b.addr);
    const uint8_t* win = reinterpret_cast<const uint8_t*>(b.addr + kMailboxBytes);
    const int64_t window = int64_t(1) << b.page_log2;
    int64_t done = 0;
    while (done < n) {
        const int64_t o = off + done;
        if (o >= mb->win_off && o < mb->win_off + mb->win_len) {
            int64_t c = mb->win_off + mb->win_len - o;
            if (c > n - done) {
                c = n - done;
            }
            copy_bytes(dst.ptr + done, win + (o - mb->win_off), static_cast<uint32_t>(c));
            done += c;
            continue;
        }
        const int64_t start = o & ~int64_t(kFetchAlign - 1);
        const bool sequential = mb->win_len != 0 && o == mb->win_off + mb->win_len;
        uint32_t got;
        if (sequential && mb->pre_seq != 0) {
            got = wait_pull(b, mb->pre_seq);  // the read-ahead of this window
            mb->pre_seq = 0;
        } else {
            int64_t want = window;
            if (!sequential) {
                want = (o + (n - done) - start + kFetchAlign - 1) & ~int64_t(kFetchAlign - 1);
                if (want > window) {
                    want = window;
                }
            }
            got = fetch(b, start, static_cast<uint32_t>(want), 0, 0);
        }
        mb->win_off = start;
        mb->win_len = got;
        if (got == 0) {
            break;
        }
        if (sequential && got == window) {
            mb->pre_seq = post(b, start + window, static_cast<uint32_t>(window), 0, 0);
        }
    }
    return done;
}

// Reads that are not 32-byte aligned go through this bounce buffer.
constexpr uint32_t kBounceOffset = 0;
constexpr uint32_t kBounceBytes = 4096;

inline int64_t read(const Bytes& b, Slice<uint8_t> dst, int64_t off, int64_t pos) {
    if (off < 0) {
        fail(pos, off);
    }
    if (off >= b.len) {
        return 0;
    }
    const int64_t n = dst.len < b.len - off ? dst.len : b.len - off;
    if (b.kind == kFromHost) {
        return read_from_host(b, dst, off, n);
    }
    const uint32_t page = 1u << b.page_log2;
    const InterleavedAddrGen<true> gen = {b.addr, page};
    const uint32_t l1 = static_cast<uint32_t>(reinterpret_cast<uintptr_t>(dst.ptr));
    int64_t done = 0;
    while (done < n) {
        const uint64_t o = static_cast<uint64_t>(off + done);
        const uint32_t id = static_cast<uint32_t>(o >> b.page_log2);
        const uint32_t in_page = static_cast<uint32_t>(o) & (page - 1);
        uint32_t chunk = page - in_page;
        if (static_cast<int64_t>(chunk) > n - done) {
            chunk = static_cast<uint32_t>(n - done);
        }
        const uint32_t to = l1 + static_cast<uint32_t>(done);
        if (((in_page | to | chunk) & 31u) == 0) {
            noc_async_read(gen.get_noc_addr(id, in_page), to, chunk);
            noc_async_read_barrier();
        } else {
            const uint32_t a0 = in_page & ~31u;
            uint32_t a1 = (in_page + chunk + 31u) & ~31u;
            if (a1 > page) {
                a1 = page;
            }
            const uint32_t bounce = scratch_base + kBounceOffset;
            noc_async_read(gen.get_noc_addr(id, a0), bounce, a1 - a0);
            noc_async_read_barrier();
            const uint8_t* from = reinterpret_cast<const uint8_t*>(bounce + (in_page - a0));
            uint8_t* into = dst.ptr + done;
            for (uint32_t k = 0; k < chunk; k++) {
                into[k] = from[k];
            }
        }
        done += chunk;
    }
    return n;
}

// tt.Bytes.ReadStrided: dst.len/size pieces, piece i the size bytes at
// off + i*stride. From the host's file, as many pieces as fit in the window
// come in one request.
inline int64_t read_strided(const Bytes& b, Slice<uint8_t> dst, int64_t off, int64_t stride, int64_t size,
                            int64_t pos_off, int64_t pos_size) {
    if (size <= 0 || dst.len % size != 0) {
        fail(pos_size, dst.len);
    }
    const int64_t count = dst.len / size;
    for (int64_t i = 0; i < count; i++) {
        if (off + i * stride < 0) {
            fail(pos_off, off + i * stride);
        }
    }
    const int64_t window = int64_t(1) << b.page_log2;
    int64_t done = 0;
    if (b.kind != kFromHost || size > window) {
        for (int64_t i = 0; i < count; i++) {
            done += read(b, Slice<uint8_t>{dst.ptr + i * size, size}, off + i * stride, pos_off);
        }
        return done;
    }
    Mailbox* mb = reinterpret_cast<Mailbox*>(b.addr);
    const uint8_t* win = reinterpret_cast<const uint8_t*>(b.addr + kMailboxBytes);
    const int64_t per = window / size;
    for (int64_t i = 0; i < count; i += per) {
        const int64_t k = count - i < per ? count - i : per;
        fetch(b, off + i * stride, static_cast<uint32_t>(size), static_cast<uint32_t>(k), stride);
        mb->win_len = 0;  // the window holds the pieces, not one range of the file
        for (int64_t j = 0; j < k; j++) {
            const int64_t o = off + (i + j) * stride;
            if (o >= b.len) {
                continue;
            }
            const int64_t c = size < b.len - o ? size : b.len - o;
            copy_bytes(dst.ptr + (i + j) * size, win + j * size, static_cast<uint32_t>(c));
            done += c;
        }
    }
    return done;
}

// tt.List: this worker's records in DRAM, one page each, after a header page
// that kernel_main writes at the end: {count, failed, position, value}.
struct ListHeader {
    int64_t count;
    int64_t failed;
    int64_t pos;
    int64_t value;
};

template <typename T>
struct List {
    uint32_t addr;        // DRAM buffer
    uint32_t slot;        // bytes per page (record size rounded up to 32)
    uint32_t first_page;  // this worker's header page
    uint32_t staging;     // L1 address of one page for writes
    int64_t cap;
    int64_t count;

    void write_page(uint32_t id) const {
        const InterleavedAddrGen<true> gen = {addr, slot};
        noc_async_write(staging, gen.get_noc_addr(id), slot);
        noc_async_write_barrier();
    }
    void append(const T& v, int64_t pos) {
        (void)pos;
        if (count < cap) {
            *reinterpret_cast<T*>(staging) = v;
            write_page(first_page + 1 + static_cast<uint32_t>(count));
        }
        count++;  // the host reports an overflow if count > cap
    }
    void finish(bool failed) const {
        ListHeader* h = reinterpret_cast<ListHeader*>(staging);
        h->count = count;
        h->failed = failed ? 1 : 0;
        h->pos = fail_pos;
        h->value = fail_value;
        write_page(first_page);
    }
};

template <typename T>
inline List<T> arg_list(uint32_t& i, uint32_t staging) {
    List<T> l;
    l.addr = arg32(i);
    l.slot = arg32(i);
    l.first_page = arg32(i);
    l.cap = arg64(i);
    l.count = 0;
    l.staging = staging;
    return l;
}

}  // namespace tsr
// ---------------------------------------------------------------------------
