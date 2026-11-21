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

#if defined(TSR_DEFER)
// A program with defer statements: a panic jumps to the innermost function
// with deferred calls (catcher), which runs them; recover stops it.
// panic_type and panic_word are the value panic was called with, as an
// interface value's (a runtime error's is kRuntimeError).
inline void** catcher = nullptr;
inline uint32_t panic_type = 0;
inline uint64_t panic_word = 0;
constexpr uint32_t kRuntimeError = 0xffffffffu;
[[noreturn]] __attribute__((noinline)) inline void unwind() {
    __builtin_longjmp(catcher != nullptr ? catcher : jmp, 1);
}
#endif

[[noreturn]] inline void fail(int64_t pos, int64_t value) {
    fail_pos = pos;
    fail_value = value;
#if defined(TSR_DEFER)
    panic_type = kRuntimeError;
    unwind();
#else
    __builtin_longjmp(jmp, 1);
#endif
}

// Running out of memory or stack is fatal, as in Go: no deferred call runs.
[[noreturn]] inline void fatal(int64_t pos, int64_t value) {
    fail_pos = pos;
    fail_value = value;
    __builtin_longjmp(jmp, 1);
}

// Runtime functions that are big and used in many places are kept out of
// line: the kernel's code shares the core's kernel config buffer (about
// 69 KB) with its arguments, and inlining them at every use would not fit.
#define TSR_OUTLINE __attribute__((noinline))

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

// Go slices: a view of an array in L1 (or of the heap), with Go's length
// and capacity.
template <typename T>
struct Slice {
    T* ptr;
    int64_t len;
    int64_t cap;
    T& at(int64_t i, int64_t pos) const {
        if (static_cast<uint64_t>(i) >= static_cast<uint64_t>(len)) {
            fail(pos, i);
        }
        return ptr[i];
    }
    // s[lo:hi], s[lo:] and s[lo:hi:max].
    Slice sub(int64_t lo, int64_t hi, int64_t pos) const { return sub3(lo, hi, cap, pos); }
    Slice from(int64_t lo, int64_t pos) const { return sub3(lo, len, cap, pos); }
    Slice sub3(int64_t lo, int64_t hi, int64_t max, int64_t pos) const {
        if (lo < 0 || hi < lo || max < hi || max > cap) {
            fail(pos, max > cap ? max : hi);
        }
        return Slice{ptr + lo, hi - lo, max - lo};
    }
};

template <typename T>
inline T& at(T* p, int64_t n, int64_t i, int64_t pos) {
    if (static_cast<uint64_t>(i) >= static_cast<uint64_t>(n)) {
        fail(pos, i);
    }
    return p[i];
}

// a[lo:hi] and a[lo:hi:max] of an array of n elements.
template <typename T>
inline Slice<T> slice(T* p, int64_t n, int64_t lo, int64_t hi, int64_t pos) {
    if (lo < 0 || hi < lo || hi > n) {
        fail(pos, hi);
    }
    return Slice<T>{p + lo, hi - lo, n - lo};
}
template <typename T>
inline Slice<T> slice3(T* p, int64_t n, int64_t lo, int64_t hi, int64_t max, int64_t pos) {
    return Slice<T>{p, n, n}.sub3(lo, hi, max, pos);
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

// Integer division and shifts with Go's rules: dividing by zero panics,
// the most negative value divided by -1 is itself (remainder 0), a shift by
// the width or more gives 0 (or -1), and a shift by a negative amount
// panics.
template <typename T>
inline T div(T a, T b, int64_t pos) {
    if (b == 0) {
        fail(pos, 0);
    }
    if (T(-1) < T(0) && b == T(-1)) {
        return T(-a);  // wraps for the most negative value
    }
    return a / b;
}
template <typename T>
inline T rem(T a, T b, int64_t pos) {
    if (b == 0) {
        fail(pos, 0);
    }
    if (T(-1) < T(0) && b == T(-1)) {
        return T(0);
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
// The same, for a count of a signed type, which may be negative.
template <typename T, typename S>
inline T shl(T a, S n, int64_t pos) {
    if (n < S(0)) {
        fail(pos, static_cast<int64_t>(n));
    }
    return shl<T>(a, n);
}
template <typename T, typename S>
inline T shr(T a, S n, int64_t pos) {
    if (n < S(0)) {
        fail(pos, static_cast<int64_t>(n));
    }
    return shr<T>(a, n);
}

// ---------------------------------------------------------------- the heap
// What append, make, slice literals and strings allocate comes from a
// region of L1 after the scratch arrays, handed out from the bottom up.
// Nothing is freed piece by piece: a function whose results, receiver and
// parameters cannot carry an allocation out releases what it allocated when
// it returns (HeapMark), and so does each turn of a loop that keeps nothing
// it allocates.

// Read-only data the compiler would put in .rodata, which the kernels'
// linker script places in the core's 4 KB of local memory, next to the
// stack: string literals and tables go with the code, in L1, instead.
#if defined(__riscv)
#define TSR_RODATA __attribute__((section(".text.tsr_rodata")))
#else
#define TSR_RODATA
#endif

// The address of read-only data kept with the code (TSR_RODATA), worked out
// relative to the program counter. tt-metal moves a kernel's code into
// place when it loads it, and fixes up absolute references to the code
// only in simple cases: it pairs each %lo with the last %hi of the same
// symbol, which goes wrong when GCC addresses two objects through one
// section anchor. A pc-relative address needs no fixing up, and as the
// asm's result it cannot be folded into data either (where the loader could
// not fix it at all).
template <typename T, const T* P>
[[gnu::always_inline]] inline const T* text_addr() {
#if defined(__riscv)
    const T* p;
    asm("lla %0, %1" : "=r"(p) : "S"(P));
    return p;
#else
    return P;
#endif
}

inline uintptr_t heap_top = 0, heap_end = 0;

inline void heap_init(uintptr_t base, uintptr_t bytes) {
    heap_top = base;
    heap_end = base + bytes;
}

// Where Go's runtime puts every allocation of no bytes (runtime.zerobase):
// they all have the same address.
inline uint64_t zerobase = 0;
template <typename T>
[[gnu::always_inline]] inline T* zero_size() {
    return reinterpret_cast<T*>(&zerobase);
}

// n elements of size bytes, zeroed and 8-byte aligned. An empty allocation
// is not nil: it is zerobase, as in Go.
TSR_OUTLINE inline void* alloc_bytes(int64_t n, uint32_t size, int64_t pos) {
    if (n < 0 || static_cast<uint64_t>(n) > (heap_end - heap_top) / size) {
        fatal(pos, n < 0 ? n : static_cast<int64_t>(static_cast<uint64_t>(n) * size));
    }
    if (n == 0) {
        return &zerobase;
    }
    const uintptr_t bytes = (static_cast<uintptr_t>(n) * size + 7) & ~uintptr_t(7);
    void* p = reinterpret_cast<void*>(heap_top);
    heap_top += bytes;
    zero(reinterpret_cast<uint8_t*>(p), static_cast<uint32_t>(bytes));
    return p;
}
template <typename T>
inline T* alloc(int64_t n, int64_t pos) {
    return static_cast<T*>(alloc_bytes(n, static_cast<uint32_t>(sizeof(T)), pos));
}

// Releases what was allocated after it was made, when it goes out of scope.
struct HeapMark {
    uintptr_t top = heap_top;
    ~HeapMark() { heap_top = top; }
};

// ---------------------------------------------------------------- pointers
// A Go pointer is a C++ pointer into the core's own memory (the cores share
// none, so pointers stay on their core). Going through nil panics, as in Go:
// address 0 of L1 is the firmware's, so it must not be written.
template <typename T>
[[gnu::always_inline]] inline T& deref(T* p, int64_t pos) {
    if (p == nullptr) {
        fail(pos, 0);
    }
    return *p;
}

// A pointer to an array is a pointer to its first element, as an array
// variable is (Ptr): p[i], p[lo:hi] and *p check it here.
template <typename T>
[[gnu::always_inline]] inline T* deref_arr(T* p, int64_t pos) {
    if (p == nullptr) {
        fail(pos, 0);
    }
    return p;
}

// (*[N]T)(s): the array of s, which must have N elements at least (a nil
// slice gives a nil pointer).
template <typename T>
inline T* slice_to_array(const Slice<T>& s, int64_t n, int64_t pos) {
    if (s.len < n) {
        fail(pos, s.len);
    }
    return s.ptr;
}

// new(T), &T{...} and variables whose address outlives their call: a T on
// the heap, set to v.
template <typename T>
TSR_OUTLINE inline T* box(const T& v, int64_t pos) {
    T* p = alloc<T>(1, pos);
    *p = v;
    return p;
}

// ---------------------------------------------------------------- interfaces
// A Go interface value: the number tessc gives its dynamic type (0: nil)
// and one word, which holds a value of up to 8 bytes itself and a pointer
// to a copy on the heap for a bigger one. 16 bytes, like Go's. tessc knows
// every type the program puts in an interface, so a method call through
// one is a switch on the number (no tables of code addresses, which the
// kernel loader could not fix up), and so are type assertions, ==, and
// hashing for map keys (iface_equal and iface_hash, which tessc writes).
struct alignas(8) Iface {
    uint32_t type;
    uint32_t pad_;
    uint64_t word;
};
bool iface_equal(const Iface& a, const Iface& b);
uint32_t iface_hash(const Iface& a);
inline bool operator==(const Iface& a, const Iface& b) { return iface_equal(a, b); }
inline bool operator!=(const Iface& a, const Iface& b) { return !iface_equal(a, b); }

template <typename T>
inline Iface to_iface(uint32_t type, const T& v, int64_t pos) {
    Iface x{type, 0, 0};
    if constexpr (sizeof(T) <= sizeof(uint64_t)) {
        __builtin_memcpy(&x.word, &v, sizeof(T));
        (void)pos;
    } else {
        x.word = reinterpret_cast<uintptr_t>(box<T>(v, pos));
    }
    return x;
}

template <typename T>
inline T iface_value(const Iface& x) {
    if constexpr (sizeof(T) <= sizeof(uint64_t)) {
        T v;
        __builtin_memcpy(&v, &x.word, sizeof(T));
        return v;
    } else {
        return *reinterpret_cast<const T*>(static_cast<uintptr_t>(x.word));
    }
}

// T(s) between struct types with the same fields (and so the same layout).
template <typename To, typename From>
inline To struct_cast(const From& f) {
    static_assert(sizeof(To) == sizeof(From), "the same fields");
    To t;
    __builtin_memcpy(&t, &f, sizeof(To));
    return t;
}

// x.(T) for a concrete type T: its value, or a panic.
template <typename T>
inline T iface_assert(const Iface& x, uint32_t type, int64_t pos) {
    if (x.type != type) {
        fail(pos, x.type);
    }
    return iface_value<T>(x);
}

// x.(I) for an interface type I: x, if its dynamic type implements I.
inline Iface iface_check(const Iface& x, bool implements, int64_t pos) {
    if (!implements) {
        fail(pos, x.type);
    }
    return x;
}

// x.M of an interface value x: Go panics there if x is nil.
inline Iface iface_nonnil(const Iface& x, int64_t pos) {
    if (x.type == 0) {
        fail(pos, 0);
    }
    return x;
}

// ---------------------------------------------------------------- function values
// A Go function value: the number tessc gives the code it calls (0: nil)
// and that code's data: a closure's variables, or a method value's
// receiver. 8 bytes, like Go's. A call through one is a call of the
// dispatch function tessc writes for its signature, a switch on the number
// (no code addresses in data, which the kernel loader could not fix up).
struct alignas(8) Func {
    uint32_t id;
    void* env;
};
inline bool operator==(const Func& f, decltype(nullptr)) { return f.id == 0; }
inline bool operator!=(const Func& f, decltype(nullptr)) { return f.id != 0; }
inline bool operator==(decltype(nullptr), const Func& f) { return f.id == 0; }
inline bool operator!=(decltype(nullptr), const Func& f) { return f.id != 0; }

// A method value's receiver pointer, as a Func's data.
template <typename T>
inline void* to_env(const T* p) {
    return const_cast<T*>(p);
}

#if defined(TSR_DEFER)
// ---------------------------------------------------------------- defer
// A function with defer statements keeps its deferred calls, last first, as
// function values on the heap, and catches the panics in it: tsr::fail
// jumps to its buffer.
struct DeferNode {
    Func fn;
    DeferNode* next;
};
struct DeferFrame {
    void* buf[5];
    void** prev;
    DeferNode* head;
    void enter() {
        prev = catcher;
        head = nullptr;
        catcher = buf;
    }
    void push(const Func& f, int64_t pos) { head = box<DeferNode>(DeferNode{f, head}, pos); }
    DeferNode* pop() {
        DeferNode* n = head;
        if (n != nullptr) {
            head = n->next;
        }
        return n;
    }
    // After the deferred calls: a panic that none of them recovered goes on
    // to the caller's catcher.
    void leave() {
        catcher = prev;
        if (fail_pos >= 0) {
            unwind();
        }
    }
};

// recover() in a function that a deferred call called (can): the panic's
// value, which stops it; nil if there is none. panic(nil) gives a non-nil
// value, as Go's *runtime.PanicNilError.
inline Iface recover(bool can) {
    if (!can || fail_pos < 0) {
        return Iface{};
    }
    Iface v{panic_type, 0, panic_word};
    if (panic_type == 0 || panic_type == kRuntimeError) {
        v = Iface{kRuntimeError, 0, static_cast<uint64_t>(fail_pos)};  // its Error() is the position's text
    }
    fail_pos = -1;
    return v;
}

// panic(v).
[[noreturn]] inline void panic_with(int64_t pos, const Iface& v) {
    fail_pos = pos;
    fail_value = 0;
    panic_type = v.type;
    panic_word = v.word;
    unwind();
}
#endif

// Go's size of a T, for the capacity append gives: a pointer has 8 bytes in
// Go and 4 on the card, so a struct that holds one is smaller here (tessc
// writes a go_size_ for each such struct, found by argument-dependent
// lookup).
template <typename T>
constexpr uint64_t go_size_(const T*) {
    return sizeof(T);
}
template <typename T>
constexpr uint64_t go_size_(T* const*) {
    return 8;
}
template <typename T>
constexpr uint64_t go_size() {
    return go_size_(static_cast<const T*>(nullptr));
}

// The Go allocator's size classes (internal/runtime/gc.SizeClassToSize).
static const uint16_t size_classes[] TSR_RODATA = {
    8, 16, 24, 32, 48, 64, 80, 96, 112, 128, 144, 160, 176, 192, 208, 224, 240, 256, 288, 320, 352, 384, 416,
    448, 480, 512, 576, 640, 704, 768, 896, 1024, 1152, 1280, 1408, 1536, 1792, 2048, 2304, 2688, 3072, 3200,
    3456, 4096, 4864, 5376, 6144, 6528, 6784, 6912, 8192, 9472, 9728, 10240, 10880, 12288, 13568, 14336, 16384,
    18432, 19072, 20480, 21760, 24576, 27264, 28672, 32768};

// runtime.roundupsize on a 64-bit host: what a request of size bytes really
// gets (with 8-byte malloc headers for big objects that hold pointers).
TSR_OUTLINE inline uint64_t roundupsize(uint64_t size, bool noscan) {
    if (size <= 32768 - 8) {
        uint64_t req = size;
        if (!noscan && req > 512) {
            req += 8;
        }
        const uint16_t* classes = text_addr<uint16_t, size_classes>();
        for (unsigned i = 0; i < sizeof size_classes / sizeof size_classes[0]; i++) {
            if (classes[i] >= req) {
                return classes[i] - (req - size);
            }
        }
    }
    return (size + 8191) & ~uint64_t(8191);
}

// runtime.growslice: the capacity append gives a slice of capacity old that
// must now hold need elements of size bytes. (Go itself may give a slice
// that does not escape a stack buffer instead, so cap after append is the
// one thing that can differ from go run.)
TSR_OUTLINE inline int64_t grow_cap(int64_t need, int64_t old, uint64_t size, bool noscan) {
    if (size == 0) {
        return need;  // struct{}: Go gives just the room needed
    }
    int64_t cap = old;
    if (need > 2 * old) {
        cap = need;
    } else if (old < 256) {
        cap = 2 * old;
    } else {
        while (cap < need) {
            cap += (cap + 3 * 256) >> 2;
        }
    }
    return static_cast<int64_t>(roundupsize(static_cast<uint64_t>(cap) * size, noscan) / size);
}

// s with room for extra more elements: s itself if it has the capacity,
// else a copy in a new, bigger array. Noscan: T holds no strings or slices.
template <bool Noscan, typename T>
TSR_OUTLINE inline Slice<T> grow(const Slice<T>& s, int64_t extra, int64_t pos) {
    const int64_t need = s.len + extra;
    if (need <= s.cap) {
        return Slice<T>{s.ptr, need, s.cap};
    }
    const int64_t cap = grow_cap(need, s.cap, go_size<T>(), Noscan);
    T* p = alloc<T>(cap, pos);
    move_bytes(reinterpret_cast<uint8_t*>(p), reinterpret_cast<const uint8_t*>(s.ptr),
               s.len * static_cast<int64_t>(sizeof(T)));
    return Slice<T>{p, need, cap};
}

// append(s, v1, v2, ...). The values come by value, as Go evaluates them
// before appending: in append(s[:4], '5', s[4]), s[4] is read before '5'
// is stored where it is.
template <bool Noscan, typename T, typename... V>
inline Slice<T> append(int64_t pos, const Slice<T>& s, V... v) {
    Slice<T> r = grow<Noscan>(s, static_cast<int64_t>(sizeof...(V)), pos);
    T* q = r.ptr + s.len;
    ((*q++ = static_cast<T>(v)), ...);
    (void)q;
    return r;
}

// append(s, t...).
template <bool Noscan, typename T>
inline Slice<T> append_all(int64_t pos, const Slice<T>& s, const Slice<T>& t) {
    Slice<T> r = grow<Noscan>(s, t.len, pos);
    move_bytes(reinterpret_cast<uint8_t*>(r.ptr + s.len), reinterpret_cast<const uint8_t*>(t.ptr),
               t.len * static_cast<int64_t>(sizeof(T)));
    return r;
}

// make([]T, len, cap), with Go's checks: makeslice panics for a length or
// a capacity below zero or of more than maxAlloc (2^48) bytes, which no
// heap has, and for a length above the capacity.
constexpr uint64_t kMaxAlloc = uint64_t(1) << 48;
template <typename T>
inline Slice<T> make(int64_t len, int64_t cap, int64_t pos_len, int64_t pos_cap, int64_t pos) {
    constexpr uint64_t most = go_size<T>() > 0 ? kMaxAlloc / go_size<T>() : uint64_t(INT64_MAX);
    if (cap < 0 || static_cast<uint64_t>(cap) > most || len < 0 || len > cap) {
        if (len < 0 || static_cast<uint64_t>(len) > most) {
            fail(pos_len, len);
        }
        fail(pos_cap, cap);
    }
    return Slice<T>{alloc<T>(cap, pos), len, cap};
}

// clear(s) for a slice: its elements become zero.
template <typename T>
inline void clear_slice(const Slice<T>& s) {
    for (int64_t i = 0; i < s.len; i++) {
        s.ptr[i] = T{};
    }
}

// make([]T, len).
template <typename T>
inline Slice<T> make(int64_t len, int64_t pos_len, int64_t pos) {
    return make<T>(len, len, pos_len, pos_len, pos);
}

// []T{v1, v2, ...}.
template <typename T, typename... V>
inline Slice<T> literal(int64_t pos, const V&... v) {
    constexpr int64_t n = static_cast<int64_t>(sizeof...(V));
    T* p = alloc<T>(n, pos);
    T* q = p;
    ((*q++ = static_cast<T>(v)), ...);
    (void)q;
    return Slice<T>{p, n, n};
}

// ---------------------------------------------------------------- strings
// A Go string: immutable bytes, in L1 with the code (literals) or on the
// heap.
struct Str {
    const uint8_t* ptr;
    int64_t len;
    uint8_t at(int64_t i, int64_t pos) const {
        if (static_cast<uint64_t>(i) >= static_cast<uint64_t>(len)) {
            fail(pos, i);
        }
        return ptr[i];
    }
    Str sub(int64_t lo, int64_t hi, int64_t pos) const {
        if (lo < 0 || hi < lo || hi > len) {
            fail(pos, hi);
        }
        return Str{ptr + lo, hi - lo};
    }
    Str from(int64_t lo, int64_t pos) const { return sub(lo, len, pos); }
};

// A string literal, prog::lit_N, which is kept with the code (see
// text_addr).
template <const uint8_t* P>
[[gnu::always_inline]] inline Str lit(int64_t n) {
    return Str{text_addr<uint8_t, P>(), n};
}

TSR_OUTLINE inline int64_t compare(const Str& a, const Str& b) {
    const int64_t n = a.len < b.len ? a.len : b.len;
    for (int64_t i = 0; i < n; i++) {
        if (a.ptr[i] != b.ptr[i]) {
            return a.ptr[i] < b.ptr[i] ? -1 : 1;
        }
    }
    return a.len < b.len ? -1 : (a.len > b.len ? 1 : 0);
}
inline bool operator==(const Str& a, const Str& b) {
    if (a.len != b.len) {
        return false;
    }
    for (int64_t i = 0; i < a.len && a.ptr != b.ptr; i++) {
        if (a.ptr[i] != b.ptr[i]) {
            return false;
        }
    }
    return true;
}
inline bool operator!=(const Str& a, const Str& b) { return !(a == b); }
inline bool operator<(const Str& a, const Str& b) { return compare(a, b) < 0; }
inline bool operator<=(const Str& a, const Str& b) { return compare(a, b) <= 0; }
inline bool operator>(const Str& a, const Str& b) { return compare(a, b) > 0; }
inline bool operator>=(const Str& a, const Str& b) { return compare(a, b) >= 0; }

// a + b + ...
template <typename... S>
TSR_OUTLINE inline Str concat(int64_t pos, const S&... parts) {
    int64_t n = 0;
    ((n += parts.len), ...);
    uint8_t* p = alloc<uint8_t>(n, pos);
    uint8_t* q = p;
    ((move_bytes(q, parts.ptr, parts.len), q += parts.len), ...);
    return Str{p, n};
}

// string(b) copies; view(b) is string(b) where Go does not copy either: in
// a comparison or a switch.
TSR_OUTLINE inline Str to_string(const Slice<uint8_t>& b, int64_t pos) {
    uint8_t* p = alloc<uint8_t>(b.len, pos);
    move_bytes(p, b.ptr, b.len);
    return Str{p, b.len};
}
inline Str view(const Slice<uint8_t>& b) { return Str{b.ptr, b.len}; }

// []byte(s).
TSR_OUTLINE inline Slice<uint8_t> to_bytes(const Str& s, int64_t pos) {
    const int64_t cap = static_cast<int64_t>(roundupsize(static_cast<uint64_t>(s.len), true));
    uint8_t* p = alloc<uint8_t>(cap, pos);
    move_bytes(p, s.ptr, s.len);
    return Slice<uint8_t>{p, s.len, cap};
}

// append(b, s...) and copy(b, s).
template <bool Noscan>
inline Slice<uint8_t> append_all(int64_t pos, const Slice<uint8_t>& s, const Str& t) {
    return append_all<Noscan>(pos, s, Slice<uint8_t>{const_cast<uint8_t*>(t.ptr), t.len, t.len});
}
inline int64_t copy(Slice<uint8_t> to, const Str& from) {
    return copy(to, Slice<uint8_t>{const_cast<uint8_t*>(from.ptr), from.len, from.len});
}

// UTF-8, with Go's rules: an invalid encoding decodes as U+FFFD, one byte
// wide, and an invalid code point encodes as U+FFFD.
constexpr int32_t kRuneError = 0xFFFD;

TSR_OUTLINE inline int64_t decode_rune(const Str& s, int64_t i, int32_t& r) {
    const uint8_t* p = s.ptr + i;
    const int64_t n = s.len - i;
    const uint32_t c = p[0];
    r = kRuneError;
    if (c < 0x80) {
        r = static_cast<int32_t>(c);
        return 1;
    }
    if (c < 0xC2 || c > 0xF4) {
        return 1;
    }
    const int64_t width = c < 0xE0 ? 2 : (c < 0xF0 ? 3 : 4);
    uint32_t lo = 0x80, hi = 0xBF;  // the second byte's range
    if (c == 0xE0) {
        lo = 0xA0;
    } else if (c == 0xED) {
        hi = 0x9F;
    } else if (c == 0xF0) {
        lo = 0x90;
    } else if (c == 0xF4) {
        hi = 0x8F;
    }
    if (n < 2 || p[1] < lo || p[1] > hi) {
        return 1;
    }
    uint32_t v = c & (width == 2 ? 0x1F : (width == 3 ? 0x0F : 0x07));
    v = (v << 6) | (p[1] & 0x3F);
    for (int64_t k = 2; k < width; k++) {
        if (n <= k || (p[k] & 0xC0) != 0x80) {
            return 1;
        }
        v = (v << 6) | (p[k] & 0x3F);
    }
    r = static_cast<int32_t>(v);
    return width;
}

TSR_OUTLINE inline int64_t encode_rune(uint8_t* p, int64_t r) {
    if (r < 0 || r > 0x10FFFF || (r >= 0xD800 && r <= 0xDFFF)) {
        r = kRuneError;
    }
    const uint32_t v = static_cast<uint32_t>(r);
    if (v < 0x80) {
        p[0] = static_cast<uint8_t>(v);
        return 1;
    }
    if (v < 0x800) {
        p[0] = static_cast<uint8_t>(0xC0 | (v >> 6));
        p[1] = static_cast<uint8_t>(0x80 | (v & 0x3F));
        return 2;
    }
    if (v < 0x10000) {
        p[0] = static_cast<uint8_t>(0xE0 | (v >> 12));
        p[1] = static_cast<uint8_t>(0x80 | ((v >> 6) & 0x3F));
        p[2] = static_cast<uint8_t>(0x80 | (v & 0x3F));
        return 3;
    }
    p[0] = static_cast<uint8_t>(0xF0 | (v >> 18));
    p[1] = static_cast<uint8_t>(0x80 | ((v >> 12) & 0x3F));
    p[2] = static_cast<uint8_t>(0x80 | ((v >> 6) & 0x3F));
    p[3] = static_cast<uint8_t>(0x80 | (v & 0x3F));
    return 4;
}

// string(r) for an integer r.
TSR_OUTLINE inline Str rune_string(int64_t r, int64_t pos) {
    uint8_t buf[4];
    const int64_t n = encode_rune(buf, r);
    uint8_t* p = alloc<uint8_t>(n, pos);
    move_bytes(p, buf, n);
    return Str{p, n};
}

// []rune(s) and string(runes).
TSR_OUTLINE inline Slice<int32_t> to_runes(const Str& s, int64_t pos) {
    int64_t n = 0;
    int32_t r;
    for (int64_t i = 0; i < s.len; i += decode_rune(s, i, r)) {
        n++;
    }
    const int64_t cap = static_cast<int64_t>(roundupsize(static_cast<uint64_t>(n) * 4, true) / 4);
    int32_t* p = alloc<int32_t>(cap, pos);
    for (int64_t i = 0, k = 0; i < s.len; k++) {
        i += decode_rune(s, i, p[k]);
    }
    return Slice<int32_t>{p, n, cap};
}
TSR_OUTLINE inline Str runes_string(const Slice<int32_t>& rs, int64_t pos) {
    uint8_t buf[4];
    int64_t n = 0;
    for (int64_t k = 0; k < rs.len; k++) {
        n += encode_rune(buf, rs.ptr[k]);
    }
    uint8_t* p = alloc<uint8_t>(n, pos);
    for (int64_t k = 0, at = 0; k < rs.len; k++) {
        at += encode_rune(p + at, rs.ptr[k]);
    }
    return Str{p, n};
}

// A string of main's that the worker uses: its length, then its bytes
// packed four to a runtime argument; it is copied to the heap.
inline Str arg_str(uint32_t& i, int64_t pos) {
    const uint32_t n = get_arg_val<uint32_t>(i++);
    uint8_t* p = alloc<uint8_t>(n, pos);
    for (uint32_t k = 0; k < n; k += 4) {
        const uint32_t w = get_arg_val<uint32_t>(i++);
        for (uint32_t j = 0; j < 4 && k + j < n; j++) {
            p[k + j] = static_cast<uint8_t>(w >> (8 * j));
        }
    }
    return Str{p, n};
}

// ---------------------------------------------------------------- the stack
// A worker runs on the core's own stack, in its 4 KB of local memory, above
// the kernel's globals; a program that recurses (or is compiled with tessc
// -l1stack, or needs more stack than the core's own has) runs on a stack
// in L1 instead: run_on_stack switches to it. tessc estimates the stack each
// worker needs and keeps it within what the stack has. Each function that
// calls others also checks on entry that room is left, so that running out
// stops the worker with "stack overflow" instead of overwriting the
// globals; and a canary at the bottom of the local stack, checked when the
// worker is done, catches what slips past the checks.
inline uint32_t stack_limit = 0;
constexpr uint32_t kStackSpare = 256;  // on the core's own stack: tessc keeps the frames within it
constexpr uint32_t kCanary = 0x57ac57acu;
constexpr int kCanaryWords = 16;

#if defined(__riscv)
extern "C" uint8_t __stack_base[];  // the end of the kernel's globals (tt-metal's linker script)
#endif

inline uint32_t* local_stack_bottom() {
#if defined(__riscv)
    return reinterpret_cast<uint32_t*>((reinterpret_cast<uintptr_t>(__stack_base) + 15) & ~uintptr_t(15));
#else
    return nullptr;
#endif
}

inline void local_stack_init() {
    volatile uint32_t* w = local_stack_bottom();
    if (w == nullptr) {
        return;
    }
    stack_limit = static_cast<uint32_t>(reinterpret_cast<uintptr_t>(w)) + 4 * kCanaryWords + kStackSpare;
    for (int k = 0; k < kCanaryWords; k++) {
        w[k] = kCanary;
    }
}

inline bool local_stack_intact() {
    volatile uint32_t* w = local_stack_bottom();
    for (int k = 0; w != nullptr && k < kCanaryWords; k++) {
        if (w[k] != kCanary) {
            return false;
        }
    }
    return true;
}

inline void run_on_stack(void (*fn)(), uint32_t top) {
#if defined(__riscv)
    asm volatile(
        "mv t0, sp\n\t"
        "mv sp, %0\n\t"
        "addi sp, sp, -16\n\t"
        "sw t0, 0(sp)\n\t"
        "jalr %1\n\t"
        "lw t0, 0(sp)\n\t"
        "mv sp, t0\n\t"
        :
        : "r"(top), "r"(fn)
        : "ra", "t0", "t1", "t2", "t3", "t4", "t5", "t6", "a0", "a1", "a2", "a3", "a4", "a5", "a6", "a7",
          "memory");
#else
    (void)top;
    fn();
#endif
}

[[gnu::always_inline]] inline void stack_check(int64_t pos) {
#if defined(__riscv)
    uint32_t sp;
    asm volatile("mv %0, sp" : "=r"(sp));
    if (sp < stack_limit) {
        fatal(pos, 0);
    }
#else
    (void)pos;
#endif
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
            done += read(b, Slice<uint8_t>{dst.ptr + i * size, size, size}, off + i * stride, pos_off);
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
