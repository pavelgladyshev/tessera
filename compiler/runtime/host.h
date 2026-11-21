// ---------------------------------------------------------------------------
// Tessera host runtime for worker programs (pasted in by tessc).
// ---------------------------------------------------------------------------
namespace tsr {

struct Position {
    const char* where;  // file:line:col in the Go source
    const char* what;   // what can go wrong there
};

struct Context {
    std::shared_ptr<distributed::MeshDevice> mesh;
    std::string program;
    std::string kernel_dir;
    bool compile_only = false;
    const Position* positions = nullptr;
    size_t n_positions = 0;
};

inline Context* current = nullptr;
inline void close_card(Context& ctx);

// Small reads and writes of a core's L1, on a chip the host reaches over
// PCIe, go through a window that UMD maps into the host's address space,
// pointed at the core for each access. A 64-byte read takes about 2 us that
// way, against about 235 us through detail::ReadFromDeviceL1, most of which
// is a fixed cost; so reads over 8 KiB, and writes over 1 KiB, still go
// through tt-metal. So does everything on the second chip, and everything in
// a build without tt-metal's internal headers (TSR_IO_WINDOW unset).
class CoreMemory {
public:
    void read(IDevice* dev, const CoreCoord& core, uint32_t addr, std::span<uint8_t> out) {
#if defined(TSR_IO_WINDOW)
        if (out.size() <= kReadMax) {
            std::lock_guard<std::mutex> lock(m_);
            if (tt::umd::IoWindow* w = point(dev, core, addr, out.size())) {
                w->read_block(0, out.data(), out.size());
                return;
            }
        }
#endif
        detail::ReadFromDeviceL1(dev, core, addr, out);
    }
    void write(IDevice* dev, const CoreCoord& core, uint32_t addr, std::span<const uint8_t> in) {
#if defined(TSR_IO_WINDOW)
        if (in.size() <= kWriteMax) {
            std::lock_guard<std::mutex> lock(m_);
            if (tt::umd::IoWindow* w = point(dev, core, addr, in.size())) {
                w->write_block(0, in.data(), in.size());
                return;
            }
        }
#endif
        detail::WriteToDeviceL1(dev, core, addr, in);
    }
    // Before the card closes: a window must not outlive UMD's driver.
    void close() {
#if defined(TSR_IO_WINDOW)
        std::lock_guard<std::mutex> lock(m_);
        windows_.clear();
#endif
    }

private:
#if defined(TSR_IO_WINDOW)
    static constexpr size_t kReadMax = 8192, kWriteMax = 1024;
    struct Window {
        std::unique_ptr<tt::umd::IoWindow> w;
        bool tried = false;
        CoreCoord core{~0u, ~0u};  // where it points now (NoC coordinates)
        uint32_t addr = 0;
    };
    // The window for dev, pointed at addr on core; nullptr if there is none.
    tt::umd::IoWindow* point(IDevice* dev, const CoreCoord& logical, uint32_t addr, size_t size) {
        if (!dev->is_mmio_capable()) {
            return nullptr;
        }
        Window& win = windows_[dev->id()];
        const CoreCoord at = dev->worker_core_from_logical_core(logical);
        if (!win.tried) {
            win.tried = true;
            try {
                win.w = MetalContext::instance().get_cluster().get_driver()->create_io_window(
                    dev->id(), tt::umd::CoreCoord(at.x, at.y, tt::CoreType::TENSIX, tt::CoordSystem::TRANSLATED), addr,
                    tt::umd::HostIoWindowConfig{tt::umd::HostMemoryCaching::UC, 0});
            } catch (...) {
                win.w.reset();  // then tt-metal's calls, as without the window
            }
            win.core = at;
            win.addr = addr;
        }
        if (!win.w) {
            return nullptr;
        }
        if (at != win.core || addr != win.addr) {
            tt::umd::TargetIoWindowConfig target = win.w->get_target_config();
            target.core_start = tt::xy_pair(at.x, at.y);
            target.addr = addr;
            win.w->configure(target);
            win.core = at;
            win.addr = addr;
        }
        return win.w->get_size() >= size ? win.w.get() : nullptr;
    }
    std::map<int, Window> windows_;  // by chip
    std::mutex m_;
#endif
};
inline CoreMemory core_memory;

inline void read_l1(IDevice* dev, const CoreCoord& core, uint32_t addr, std::span<uint8_t> out) {
    core_memory.read(dev, core, addr, out);
}
inline void write_l1(IDevice* dev, const CoreCoord& core, uint32_t addr, std::span<const uint8_t> in) {
    core_memory.write(dev, core, addr, in);
}

// Closes the card (and the windows into it).
inline void close_card(Context& ctx) {
    core_memory.close();
    ctx.mesh->close();
}

// Stops the program with a message. The card is closed first: exiting with
// it open leaves tt-metal's background threads running into freed memory.
[[noreturn]] inline void die(const std::string& msg) {
    std::cerr << msg << "\n";
    static bool dying = false;
    if (!dying && current && current->mesh) {
        dying = true;
        try {
            close_card(*current);
        } catch (...) {
        }
    }
    std::exit(1);
}

[[noreturn]] inline void fail(int64_t pos, int64_t value) {
    std::string where = "?", what = "runtime error";
    if (current && pos >= 0 && static_cast<size_t>(pos) < current->n_positions) {
        where = current->positions[pos].where;
        what = current->positions[pos].what;
    }
    die("panic: " + what + " (value " + std::to_string(value) + ") at " + where);
}

// Opens the card: all its chips, as one mesh of 1 x chips (an n300 has
// two), with tt-metal's dispatchers on Ethernet cores. That leaves 64 Tensix
// cores per chip for the program instead of 56, and the card opens in about
// 0.15 s instead of 0.7 s. TESSERA_DEVICE=N opens chip N alone,
// TESSERA_CHIPS=K the first K chips, and all_chips = false the first chip
// (for programs that use one).
// Whether open_card turned on the fabric for programs. (tt-metal turns it on
// by itself to dispatch to the second chip, but then its routers take no
// connections from Tensix cores, so GetFabricConfig cannot tell.)
inline bool fabric_for_programs = false;

// Bytes of every core's L1 set aside as ttnn's "small" region, which its
// convolutions and pools need. The workers lose them, so it is 0 unless the
// program uses those (tessc defines TSR_L1_SMALL), or TESSERA_L1_SMALL says.
#ifndef TSR_L1_SMALL
#define TSR_L1_SMALL DEFAULT_L1_SMALL_SIZE
#endif
inline size_t l1_small() {
    if (const char* s = std::getenv("TESSERA_L1_SMALL")) {
        return static_cast<size_t>(std::atol(s));
    }
    return TSR_L1_SMALL;
}

inline std::shared_ptr<distributed::MeshDevice> open_card(size_t queues, int device, bool all_chips) {
    const DispatchCoreConfig dispatch{DispatchCoreType::ETH};
    if (const char* d = std::getenv("TESSERA_DEVICE")) {
        device = std::atoi(d);
    }
    if (device >= 0) {
        return distributed::MeshDevice::create_unit_mesh(device, l1_small(), DEFAULT_TRACE_REGION_SIZE,
                                                         queues, dispatch);
    }
    size_t chips = all_chips ? GetNumAvailableDevices() : 1;
    if (const char* k = std::getenv("TESSERA_CHIPS")) {
        chips = std::min<size_t>(chips, std::max(1, std::atoi(k)));
    }
    const char* fabric = std::getenv("TESSERA_FABRIC");  // 0: no fabric; channels between the chips are relayed
    if (chips > 1 && (fabric == nullptr || std::string(fabric) != "0")) {
        // The fabric over the Ethernet links between the chips, for channels
        // between goroutines on different chips. (tt-metal runs its dispatch
        // to the second chip over it anyway, and the setup stays the same
        // from one program to the next, which keeps the card's start fast.)
        tt::tt_fabric::SetFabricConfig(tt::tt_fabric::FabricConfig::FABRIC_1D);
        fabric_for_programs = true;
    }
    return distributed::MeshDevice::create(distributed::MeshDeviceConfig(distributed::MeshShape(1, chips)),
                                           l1_small(), DEFAULT_TRACE_REGION_SIZE, queues, dispatch);
}

// tt.Cores(): the Tensix cores for workers on all the card's chips.
inline int64_t cores(Context& ctx) {
    const CoreCoord grid = ctx.mesh->compute_with_storage_grid_size();
    return std::min<int64_t>(static_cast<int64_t>(grid.x) * grid.y, 64 /* kMaxWorkers */) * static_cast<int64_t>(ctx.mesh->shape()[1]);
}

// The chip at (0, index) of the card's mesh.
inline IDevice* device(Context& ctx, int index) {
    return ctx.mesh->get_view().get_devices(
        distributed::MeshCoordinateRange(distributed::MeshCoordinate(0, index), distributed::MeshCoordinate(0, index)))[0];
}

using seconds = std::chrono::duration<double>;
inline double since(std::chrono::steady_clock::time_point t0) {
    return seconds(std::chrono::steady_clock::now() - t0).count();
}

// Go slices and arrays, with Go's bounds checks. (Slices in main are
// std::vector; Slice and Str are here for the struct types that worker code
// shares with main, which may hold them.)
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
    Slice sub(int64_t lo, int64_t hi, int64_t pos) const {
        if (lo < 0 || hi < lo || hi > cap) {
            fail(pos, hi);
        }
        return Slice{ptr + lo, hi - lo, cap - lo};
    }
};
struct Str {
    const uint8_t* ptr;
    int64_t len;
    bool operator==(const Str& o) const { return len == o.len && std::memcmp(ptr, o.ptr, static_cast<size_t>(len)) == 0; }
    bool operator!=(const Str& o) const { return !(*this == o); }
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
    return Slice<T>{p + lo, hi - lo, n - lo};
}
template <typename T>
inline T& at(std::vector<T>& v, int64_t i, int64_t pos) {
    if (static_cast<uint64_t>(i) >= v.size()) {
        fail(pos, i);
    }
    return v[i];
}
template <typename T>
inline const T& at(const std::vector<T>& v, int64_t i, int64_t pos) {
    if (static_cast<uint64_t>(i) >= v.size()) {
        fail(pos, i);
    }
    return v[i];
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

template <typename T>
constexpr T min(T a) { return a; }
template <typename T, typename... R>
constexpr T min(T a, T b, R... rest) { return min<T>(b < a ? b : a, rest...); }
template <typename T>
constexpr T max(T a) { return a; }
template <typename T, typename... R>
constexpr T max(T a, T b, R... rest) { return max<T>(b > a ? b : a, rest...); }
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

// println, as Go's builtin: arguments separated by spaces, to standard error.
inline void print_one(const std::string& s) { std::cerr << s; }
inline void print_one(bool b) { std::cerr << (b ? "true" : "false"); }
template <typename T>
inline void print_one(T v) { std::cerr << +v; }
inline void println() { std::cerr << "\n"; }
template <typename T, typename... R>
inline void println(const T& first, const R&... rest) {
    print_one(first);
    ((std::cerr << ' ', print_one(rest)), ...);
    std::cerr << "\n";
}

// tt.Bytes: the bytes on the host (for tt.WriteFile) and in DRAM on the card.
constexpr uint32_t kPageLog2 = 12;  // 4 KiB pages
constexpr uint32_t kPage = 1u << kPageLog2;

// tt.OpenBytes: every worker core gets a 64-byte mailbox and a window at
// the same L1 address; the host serves the workers' requests from the file.
// The workers also copy each request into a table on worker 0's core, so
// the host sees all of them with one read.
constexpr uint32_t kWindowLog2 = 17;  // 128 KiB
constexpr uint32_t kMailboxBytes = 64;
constexpr uint32_t kTableOffset = kMailboxBytes + (1u << kWindowLog2);
constexpr uint32_t kMaxWorkers = 64;
constexpr uint32_t kTableBytes = 32 * kMaxWorkers;

struct Bytes {
    std::string name;
    std::vector<uint8_t> host;  // tt.LoadBytes: padded with zeros to whole pages
    int64_t len = 0;
    std::shared_ptr<distributed::MeshBuffer> dev;
    // tt.OpenBytes
    bool from_host = false;
    int fd = -1;
    std::shared_ptr<distributed::MeshBuffer> l1;      // the mailbox and window on every core
    std::shared_ptr<distributed::MeshBuffer> staged;  // DRAM: one window's worth per worker
    std::shared_ptr<std::atomic<int64_t>> read = std::make_shared<std::atomic<int64_t>>(0);  // bytes read from the file

    int64_t bytes_read() const { return read->load(); }

    // Reads n bytes at off from the file into out; returns how many.
    int64_t pread_at(uint8_t* out, int64_t off, int64_t n) const {
        int64_t got = 0;
        while (got < n) {
            ssize_t r = ::pread(fd, out + got, static_cast<size_t>(n - got), static_cast<off_t>(off + got));
            if (r <= 0) {
                break;
            }
            got += r;
        }
        *read += got;
        return got;
    }
};

inline Bytes load_bytes(Context& ctx, const std::string& name, const std::string& path) {
    auto t0 = std::chrono::steady_clock::now();
    Bytes b;
    b.name = name;
    if (!ctx.compile_only) {
        std::ifstream f(path, std::ios::binary | std::ios::ate);
        if (!f) {
            die("cannot open " + path + " (" + name + ", pass --" + name + " FILE)");
        }
        b.len = static_cast<int64_t>(f.tellg());
        b.host.resize(static_cast<size_t>(b.len));
        f.seekg(0);
        f.read(reinterpret_cast<char*>(b.host.data()), static_cast<std::streamsize>(b.len));
        if (!f) {
            die("cannot read " + path);
        }
    }
    size_t padded = (static_cast<size_t>(b.len) + kPage - 1) / kPage * kPage;
    b.host.resize(padded > 0 ? padded : kPage, 0);
    distributed::DeviceLocalBufferConfig local{.page_size = kPage, .buffer_type = BufferType::DRAM};
    distributed::ReplicatedBufferConfig global{.size = b.host.size()};
    b.dev = distributed::MeshBuffer::create(global, local, ctx.mesh.get());
    double read_s = since(t0);
    auto t1 = std::chrono::steady_clock::now();
    distributed::EnqueueWriteMeshBuffer(ctx.mesh->mesh_command_queue(), b.dev, b.host, true);
    std::fprintf(stderr, "%s: %s: %.1f MB read in %.2f s, copied to the card in %.2f s\n", ctx.program.c_str(),
                 name.c_str(), b.len / 1e6, read_s, since(t1));
    *b.read = b.len;
    return b;
}

inline Bytes open_bytes(Context& ctx, const std::string& name, const std::string& path) {
    Bytes b;
    b.name = name;
    b.from_host = true;
    if (!ctx.compile_only) {
        b.fd = ::open(path.c_str(), O_RDONLY);
        if (b.fd < 0) {
            die("cannot open " + path + " (" + name + ", pass --" + name + " FILE)");
        }
        struct stat st;
        if (::fstat(b.fd, &st) != 0) {
            die("cannot stat " + path);
        }
        b.len = static_cast<int64_t>(st.st_size);
    }
    // One page per L1 bank puts the region at the same address on every core.
    const uint32_t page = kTableOffset + kTableBytes;
    const uint32_t banks = ctx.mesh->allocator()->get_num_banks(BufferType::L1);
    distributed::DeviceLocalBufferConfig local{.page_size = page, .buffer_type = BufferType::L1};
    distributed::ReplicatedBufferConfig global{.size = static_cast<uint64_t>(banks) * page};
    b.l1 = distributed::MeshBuffer::create(global, local, ctx.mesh.get());
    // The data goes to the card through the second command queue (its fast
    // DMA path) into a staging area in DRAM, and the worker pulls it from there.
    distributed::DeviceLocalBufferConfig slocal{.page_size = 4096, .buffer_type = BufferType::DRAM};
    distributed::ReplicatedBufferConfig sglobal{.size = static_cast<uint64_t>(kMaxWorkers) << kWindowLog2};
    b.staged = distributed::MeshBuffer::create(sglobal, slocal, ctx.mesh.get());
    std::fprintf(stderr, "%s: %s: %.1f MB, read on demand\n", ctx.program.c_str(), name.c_str(), b.len / 1e6);
    return b;
}

inline void push32(std::vector<uint32_t>& a, uint32_t v) { a.push_back(v); }
inline void push64(std::vector<uint32_t>& a, int64_t v) {
    a.push_back(static_cast<uint32_t>(static_cast<uint64_t>(v)));
    a.push_back(static_cast<uint32_t>(static_cast<uint64_t>(v) >> 32));
}
// A string of main's that the workers use: its length, then its bytes,
// four to a runtime argument (tsr::arg_str on the card).
inline void push_string(std::vector<uint32_t>& a, const std::string& s) {
    if (s.size() > 1024) {
        die("a string the workers use is " + std::to_string(s.size()) +
            " bytes; it travels as runtime arguments, so it can be at most 1024");
    }
    a.push_back(static_cast<uint32_t>(s.size()));
    for (size_t k = 0; k < s.size(); k += 4) {
        uint32_t w = 0;
        for (size_t j = 0; j < 4 && k + j < s.size(); j++) {
            w |= static_cast<uint32_t>(static_cast<uint8_t>(s[k + j])) << (8 * j);
        }
        a.push_back(w);
    }
}
inline void push_bytes(std::vector<uint32_t>& a, const Bytes& b) {
    if (b.from_host) {
        die("internal error: on-demand bytes are passed with Workers::push_bytes");
    }
    a.push_back(0);  // tsr::kInDram
    a.push_back(static_cast<uint32_t>(b.dev->address()));
    a.push_back(kPageLog2);
    push64(a, b.len);
}

inline void write_file(const std::string& path, const Bytes& b, int64_t off, int64_t n) {
    if (off < 0 || n < 0) {
        die("tt.WriteFile(" + path + "): bad range");
    }
    int64_t end = std::min(off + n, b.len);
    off = std::min(off, end);
    std::filesystem::path p(path);
    if (p.has_parent_path()) {
        std::filesystem::create_directories(p.parent_path());
    }
    std::vector<uint8_t> data;
    const uint8_t* from = b.host.data() + off;
    if (b.from_host) {
        data.resize(static_cast<size_t>(end - off));
        b.pread_at(data.data(), off, end - off);
        from = data.data();
    }
    std::ofstream f(path, std::ios::binary);
    f.write(reinterpret_cast<const char*>(from), static_cast<std::streamsize>(end - off));
    if (!f) {
        die("cannot write " + path);
    }
}

// A few threads that read the file in parallel: an SSD serves many
// requests at once much faster than one at a time.
class ReadPool {
public:
    explicit ReadPool(int threads) {
        for (int i = 0; i < threads; i++) {
            threads_.emplace_back([this] { loop(); });
        }
    }
    ~ReadPool() {
        {
            std::lock_guard<std::mutex> lk(mu_);
            stop_ = true;
        }
        cv_.notify_all();
        for (auto& t : threads_) {
            t.join();
        }
    }
    // Runs job(0) ... job(n-1) on the pool and waits for all of them.
    void run(size_t n, const std::function<void(size_t)>& job) {
        if (n == 0) {
            return;
        }
        std::unique_lock<std::mutex> lk(mu_);
        job_ = &job;
        count_ = n;
        next_ = 0;
        finished_ = 0;
        generation_++;
        cv_.notify_all();
        done_.wait(lk, [&] { return finished_ == count_; });
        job_ = nullptr;
    }

private:
    void loop() {
        uint64_t seen = 0;
        std::unique_lock<std::mutex> lk(mu_);
        for (;;) {
            cv_.wait(lk, [&] { return stop_ || generation_ != seen; });
            if (stop_) {
                return;
            }
            seen = generation_;
            while (job_ && next_ < count_) {
                size_t i = next_++;
                const std::function<void(size_t)>* job = job_;
                lk.unlock();
                (*job)(i);
                lk.lock();
                if (++finished_ == count_) {
                    done_.notify_all();
                }
            }
        }
    }
    std::vector<std::thread> threads_;
    std::mutex mu_;
    std::condition_variable cv_, done_;
    const std::function<void(size_t)>* job_ = nullptr;
    size_t count_ = 0, next_ = 0, finished_ = 0;
    uint64_t generation_ = 0;
    bool stop_ = false;
};

struct ListHeader {
    int64_t count;
    int64_t failed;
    int64_t pos;
    int64_t value;
};

// tt.Parallel: one worker per Tensix core, each with its own result list.
// The workers on one chip of the card, with the program that runs them.
struct Chip {
    int index = 0;  // its place in the mesh: (0, index)
    IDevice* dev = nullptr;
    int64_t first = 0, count = 0;  // workers first, ..., first+count-1
    std::vector<CoreCoord> cores;  // one per worker, in order
    CoreRangeSet set;
    Program program;
    KernelHandle kernel{};
};

struct Workers {
    Context& ctx;
    int64_t n;
    uint32_t slot;
    int64_t cap;
    // The chips in use. Workers fill the first chip's cores, then the
    // next chip's: a program with no more workers than one chip has cores
    // runs on the first chip alone.
    std::vector<Chip> chips;
    int64_t per_chip = 0;
    std::shared_ptr<distributed::MeshBuffer> results;  // on every chip; each writes its workers' part

    // What the program is made of, so that it can be made again: the worker
    // kernel is built for speed (-O2) unless its code does not fit the
    // core's kernel config buffer, and then for size (-Os).
    uint32_t scratch_bytes;
    std::string kernel_src;
    KernelBuildOptLevel opt = KernelBuildOptLevel::O2;
    std::vector<std::function<void(Workers&, Chip&)>> extras;  // more kernels on the same cores (the detector)
    std::vector<std::vector<uint32_t>> args;            // each worker's runtime arguments

    Chip& chip_of(int64_t w) { return chips[static_cast<size_t>(w / per_chip)]; }
    const Chip& chip_of(int64_t w) const { return chips[static_cast<size_t>(w / per_chip)]; }

    void build() {
        for (Chip& c : chips) {
            build(c);
        }
    }
    void build(Chip& c) {
        c.program = CreateProgram();
        CircularBufferConfig scratch =
            CircularBufferConfig(scratch_bytes, {{tt::CBIndex::c_0, tt::DataFormat::UInt8}})
                .set_page_size(tt::CBIndex::c_0, scratch_bytes);
        CreateCircularBuffer(c.program, c.set, scratch);
        c.kernel = CreateKernelFromString(
            c.program,
            kernel_src,
            c.set,
            DataMovementConfig{
                .processor = DataMovementProcessor::RISCV_0,
                .noc = NOC::RISCV_0_default,
                // tt-metal's kernel cache does not tell the optimization levels
                // apart (v0.80 hashes the source and the defines, not opt_level),
                // so a define does.
                .defines = {{"TSR_OPT", opt == KernelBuildOptLevel::Os ? "s" : "2"}},
                .opt_level = opt});
        for (auto& add : extras) {
            add(*this, c);
        }
        for (int64_t w = c.first; w < c.first + c.count; w++) {
            if (!args[static_cast<size_t>(w)].empty()) {
                SetRuntimeArgs(c.program, c.kernel, c.cores[static_cast<size_t>(w - c.first)], args[static_cast<size_t>(w)]);
            }
        }
    }

    Workers(Context& c, int64_t workers, uint32_t record_bytes, int64_t capacity, uint32_t scratch,
            const std::string& kernel_file)
        : ctx(c), n(workers), cap(capacity), scratch_bytes(scratch) {
        CoreCoord grid = ctx.mesh->compute_with_storage_grid_size();  // each chip's
        per_chip = std::min<int64_t>(static_cast<int64_t>(grid.x) * grid.y, kMaxWorkers);
        const int64_t nchips = static_cast<int64_t>(ctx.mesh->shape()[1]);
        if (n < 1 || n > per_chip * nchips) {
            die("tt.Parallel(" + std::to_string(n) + "): the card has " + std::to_string(per_chip * nchips) +
                " Tensix cores for workers (" + std::to_string(nchips) + " chip" + (nchips > 1 ? "s" : "") + " of " +
                std::to_string(per_chip) + "), one per worker");
        }
        slot = std::max<uint32_t>(32, (record_bytes + 31) / 32 * 32);
        for (int64_t first = 0; first < n; first += per_chip) {
            Chip c;
            c.index = static_cast<int>(chips.size());
            c.dev = device(ctx, c.index);
            c.first = first;
            c.count = std::min(per_chip, n - first);
            c.set = num_cores_to_corerangeset(static_cast<uint32_t>(c.count), grid, true);
            c.cores = corerange_to_cores(c.set, static_cast<uint32_t>(c.count), true);
            chips.push_back(std::move(c));
        }
        args.resize(static_cast<size_t>(n));

        distributed::DeviceLocalBufferConfig local{.page_size = slot, .buffer_type = BufferType::DRAM};
        distributed::ReplicatedBufferConfig global{.size = static_cast<uint64_t>(n) * (cap + 1) * slot};
        results = distributed::MeshBuffer::create(global, local, ctx.mesh.get());

        std::ifstream f(ctx.kernel_dir + "/" + kernel_file);
        if (!f) {
            die("cannot read kernel source " + ctx.kernel_dir + "/" + kernel_file + " (pass --kernels DIR)");
        }
        std::stringstream src;
        src << f.rdbuf();
        kernel_src = src.str();
        if (const char* o = std::getenv("TESSERA_OPT"); o != nullptr && std::string(o) == "s") {
            opt = KernelBuildOptLevel::Os;
        }
        build();
    }

    // Adds a kernel on the workers' cores of every chip, now and whenever
    // the programs are made again.
    void add_kernel(std::function<void(Workers&, Chip&)> add) {
        for (Chip& c : chips) {
            add(*this, c);
        }
        extras.push_back(std::move(add));
    }

    // Byte arrays the workers read from the host's file while they run.
    std::vector<const Bytes*> from_host;
    void serve(const Bytes& b) {
        if (b.from_host) {
            from_host.push_back(&b);
        }
    }

    // A worker's runtime arguments for a tt.Bytes; an on-demand one also
    // gets the NoC address of its slot in the request table.
    void push_bytes(std::vector<uint32_t>& a, const Bytes& b, int64_t w) const {
        if (!b.from_host) {
            tsr::push_bytes(a, b);
            return;
        }
        // Each chip has a request table of its own, in its first worker's L1,
        // with an entry (and a staging window) per worker on the chip.
        const Chip& c = chip_of(w);
        const uint32_t local = static_cast<uint32_t>(w - c.first);
        const CoreCoord table = c.dev->worker_core_from_logical_core(c.cores[0]);
        a.push_back(1);  // tsr::kFromHost
        a.push_back(static_cast<uint32_t>(b.l1->address()));
        a.push_back(kWindowLog2);
        push64(a, b.len);
        a.push_back(static_cast<uint32_t>(table.x));
        a.push_back(static_cast<uint32_t>(table.y));
        a.push_back(static_cast<uint32_t>(b.l1->address()) + kTableOffset + 32 * local);
        a.push_back(static_cast<uint32_t>(b.staged->address()));
        a.push_back(local << (kWindowLog2 - 12));  // first 4 KiB page
    }

    // Serves the workers' requests until the workload is done. A worker
    // numbers each request and posts it in the table on worker 0's core.
    // Each sweep the host reads the table, reads every new range from the
    // file, sends them all to the workers' staging areas in DRAM through the
    // second command queue, waits for that, and then answers each worker in
    // its mailbox with the request's number. A strided request (from
    // tt.Bytes.ReadStrided) asks for several pieces of the file at once;
    // they go to the worker one after another. Returns the requests served.
    int64_t serve_until(const distributed::MeshEvent& done) {
        auto& io = ctx.mesh->mesh_command_queue(1);
        constexpr uint32_t window = 1u << kWindowLog2;
        std::vector<uint32_t> last(from_host.size() * static_cast<size_t>(n), 0);
        std::vector<std::vector<uint8_t>> bufs(static_cast<size_t>(n), std::vector<uint8_t>(window + 32));
        std::vector<uint8_t> table(32 * static_cast<size_t>(per_chip));
        struct Answer {
            int64_t w;
            uint32_t seq, got;
        };
        struct Piece {  // one range of the file, for one worker
            size_t answer;
            uint32_t at;  // where it goes in the worker's data
            int64_t off;
            uint32_t len, got;
        };
        ReadPool pool(16);
        std::vector<Answer> answers;
        std::vector<Piece> pieces;
        int64_t served = 0;
        using clock = std::chrono::steady_clock;
        while (!distributed::EventQuery(done)) {
            for (size_t i = 0; i < from_host.size(); i++) {
                for (const Chip& c : chips) {
                    const Bytes& b = *from_host[i];
                    const uint32_t addr = static_cast<uint32_t>(b.l1->address());
                    auto tp = clock::now();
                    read_l1(c.dev, c.cores[0], addr + kTableOffset,
                                             std::span<uint8_t>(table.data(), 32 * static_cast<size_t>(c.count)));
                    t_poll += since(tp);
                    answers.clear();
                    pieces.clear();
                    for (int64_t w = c.first; w < c.first + c.count; w++) {
                        // {req_seq, req_count, req_len, -, req_off, req_stride}: see tsr::Mailbox
                        const uint8_t* mb = table.data() + 32 * (w - c.first);
                        uint32_t seq, count, len;
                        int64_t off, stride;
                        std::memcpy(&seq, mb, 4);
                        std::memcpy(&count, mb + 4, 4);
                        std::memcpy(&len, mb + 8, 4);
                        std::memcpy(&off, mb + 16, 8);
                        std::memcpy(&stride, mb + 24, 8);
                        uint32_t& seen = last[i * static_cast<size_t>(n) + static_cast<size_t>(w)];
                        if (seq == seen) {
                            continue;
                        }
                        seen = seq;
                        const size_t k = answers.size();
                        if (count == 0) {
                            answers.push_back({w, seq, 0});
                            pieces.push_back({k, 0, off, std::min(len, window), 0});
                            continue;
                        }
                        if (len == 0 || static_cast<uint64_t>(len) * count > window) {
                            die("internal error: worker " + std::to_string(w) + " asked for " + std::to_string(count) +
                                " pieces of " + std::to_string(len) + " bytes");
                        }
                        answers.push_back({w, seq, len * count});  // the worker pulls all the pieces
                        for (uint32_t j = 0; j < count; j++) {
                            pieces.push_back({k, j * len, off + static_cast<int64_t>(j) * stride, len, 0});
                        }
                    }
                    if (answers.empty()) {
                        continue;
                    }
                    sweeps++;
                    auto tf = clock::now();
                    pool.run(pieces.size(), [&](size_t j) {
                        Piece& p = pieces[j];
                        uint8_t* buf = bufs[static_cast<size_t>(answers[p.answer].w)].data() + p.at;
                        if (p.off >= 0 && p.off < b.len) {
                            p.got = static_cast<uint32_t>(b.pread_at(buf, p.off, std::min<int64_t>(p.len, b.len - p.off)));
                        }
                        std::fill(buf + p.got, buf + p.len, 0);  // a piece cut short by the end of the file
                    });
                    t_file += since(tf);
                    for (const Piece& p : pieces) {
                        bytes_served += p.got;
                        if (answers[p.answer].got == 0) {
                            answers[p.answer].got = p.got;  // one range: what the file had
                        }
                    }
                    std::vector<distributed::ShardDataTransfer> transfers;
                    for (const Answer& a : answers) {
                        const uint32_t padded = (a.got + 31) & ~31u;
                        std::vector<uint8_t>& buf = bufs[static_cast<size_t>(a.w)];
                        std::fill(buf.begin() + a.got, buf.begin() + padded, 0);
                        if (padded > 0) {
                            transfers.push_back(
                                distributed::ShardDataTransfer{distributed::MeshCoordinate(0, c.index)}
                                    .host_data(buf.data())
                                    .region(BufferRegion(static_cast<uint64_t>(a.w - c.first) << kWindowLog2, padded)));
                        }
                    }
                    auto tw = clock::now();
                    if (!transfers.empty()) {
                        io.enqueue_write_shards(b.staged, transfers, true);
                    }
                    for (const Answer& a : answers) {
                        // {ack_seq, got_len} at byte 32 of the mailbox; the number last.
                        const CoreCoord core = c.cores[static_cast<size_t>(a.w - c.first)];
                        write_l1(c.dev, core, addr + 36,
                                                std::span<const uint8_t>(reinterpret_cast<const uint8_t*>(&a.got), 4));
                        write_l1(c.dev, core, addr + 32,
                                                std::span<const uint8_t>(reinterpret_cast<const uint8_t*>(&a.seq), 4));
                    }
                    t_card += since(tw);
                    served += static_cast<int64_t>(answers.size());
                }
            }
        }
        return served;
    }
    int64_t sweeps = 0;
    double t_poll = 0, t_file = 0, t_card = 0;
    int64_t bytes_served = 0;

    // Sets worker w's runtime arguments: its index and result list, then the
    // variables its function captures.
    void set_args(int64_t w, const std::vector<uint32_t>& captured) {
        std::vector<uint32_t> a;
        push64(a, w);
        push32(a, static_cast<uint32_t>(results->address()));
        push32(a, slot);
        push32(a, static_cast<uint32_t>(w * (cap + 1)));
        push64(a, cap);
        a.insert(a.end(), captured.begin(), captured.end());
        args[static_cast<size_t>(w)] = a;
        Chip& c = chip_of(w);
        SetRuntimeArgs(c.program, c.kernel, c.cores[static_cast<size_t>(w - c.first)], a);
    }

    // Runs the workers and returns their results, record_bytes each, one
    // worker's after another.
    // Waits for the workers, and says so when it takes long: a worker that
    // never ends (a loop, or memory damaged by a stack that overflowed
    // unnoticed) would otherwise look like a program doing nothing. With
    // TESSERA_TIMEOUT=seconds a run that takes longer stops the program.
    void watch(const distributed::MeshEvent& done, std::chrono::steady_clock::time_point t0) {
        double limit = 0, next = 10;
        if (const char* t = std::getenv("TESSERA_TIMEOUT")) {
            limit = std::atof(t);
        }
        while (!distributed::EventQuery(done)) {
            std::this_thread::sleep_for(std::chrono::microseconds(100));
            const double s = since(t0);
            if (limit > 0 && s > limit) {
                std::fprintf(stderr,
                             "%s: the workers have not finished after %.0f s (TESSERA_TIMEOUT). A worker may be in a "
                             "loop that does not end, or its stack may have overflowed unnoticed: try tessc "
                             "-l1stack. The card needs a reset before its next use (tt-smi -r)\n",
                             ctx.program.c_str(), s);
                std::_Exit(1);  // at once: closing the card would wait for the workers too
            }
            if (s > next) {
                std::fprintf(stderr,
                             "%s: the workers have been running for %.0f s. If that is not expected, a worker may be "
                             "in a loop that does not end, or its stack may have overflowed unnoticed (try tessc "
                             "-l1stack); TESSERA_TIMEOUT=seconds stops such a run\n",
                             ctx.program.c_str(), s);
                next = next < 60 ? next * 3 : next + 60;
            }
        }
    }

    std::vector<uint8_t> run_raw(size_t record_bytes) {
        std::vector<uint8_t> out;
        if (ctx.compile_only) {
            for (Chip& c : chips) {
                detail::CompileProgram(c.dev, c.program);
            }
            std::fprintf(stderr, "%s: worker kernel compiled for %lld cores\n", ctx.program.c_str(),
                         static_cast<long long>(n));
            return out;
        }
        auto& cq = ctx.mesh->mesh_command_queue();
        distributed::Finish(cq);  // earlier writes to DRAM are done
        for (const Bytes* b : from_host) {
            const std::vector<uint8_t> zeros(kTableBytes, 0);
            const uint32_t addr = static_cast<uint32_t>(b->l1->address());
            for (const Chip& c : chips) {
                for (const CoreCoord& core : c.cores) {
                    write_l1(c.dev, core, addr, std::span<const uint8_t>(zeros.data(), kMailboxBytes));
                }
                write_l1(c.dev, c.cores[0], addr + kTableOffset, std::span<const uint8_t>(zeros));
            }
        }
        // One program per chip, each on its own part of the mesh.
        auto workload_of = [&] {
            distributed::MeshWorkload w;
            for (Chip& c : chips) {
                const distributed::MeshCoordinate at(0, c.index);
                w.add_program(distributed::MeshCoordinateRange(at, at), std::move(c.program));
            }
            return w;
        };
        distributed::MeshWorkload workload = workload_of();
        auto t0 = std::chrono::steady_clock::now();
        try {
            distributed::EnqueueMeshWorkload(cq, workload, false);
        } catch (const std::exception& e) {
            if (opt == KernelBuildOptLevel::Os || std::strstr(e.what(), "too large for kernel config buffer") == nullptr) {
                throw;
            }
            // The worker's code does not fit: build it for size, and try again.
            std::fprintf(stderr, "%s: the worker's code is too big for the kernel config buffer at -O2; building it with -Os\n",
                         ctx.program.c_str());
            opt = KernelBuildOptLevel::Os;
            build();
            workload = workload_of();
            t0 = std::chrono::steady_clock::now();
            distributed::EnqueueMeshWorkload(cq, workload, false);
        }
        int64_t requests = 0;
        if (!from_host.empty()) {
            requests = serve_until(cq.enqueue_record_event_to_host());
        } else {
            watch(cq.enqueue_record_event_to_host(), t0);
        }
        distributed::Finish(cq);
        double run_s = since(t0);
        if (!from_host.empty()) {
            std::fprintf(stderr,
                         "%s: served %lld read requests (%.1f MB) from the host's file in %lld sweeps: %.2f s "
                         "reading the file, %.2f s sending to the card, %.2f s polling\n",
                         ctx.program.c_str(), static_cast<long long>(requests), bytes_served / 1e6,
                         static_cast<long long>(sweeps), t_file, t_card, t_poll);
        }
        std::vector<uint8_t> raw;
        for (int64_t w = 0; w < n; w++) {
            const Chip& c = chip_of(w);
            if (w == c.first) {  // each chip's copy holds its own workers' lists
                distributed::ReadShard(cq, raw, results, distributed::MeshCoordinate(0, c.index), true);
            }
            const uint8_t* base = raw.data() + static_cast<size_t>(w) * (cap + 1) * slot;
            ListHeader h;
            std::memcpy(&h, base, sizeof h);
            if (h.failed) {
                std::cerr << "worker " << w << ": ";
                fail(h.pos, h.value);
            }
            if (h.count > cap) {
                die("worker " + std::to_string(w) + " appended " + std::to_string(h.count) +
                    " results; a worker can hold " + std::to_string(cap));
            }
            for (int64_t i = 0; i < h.count; i++) {
                const uint8_t* record = base + static_cast<size_t>(i + 1) * slot;
                out.insert(out.end(), record, record + record_bytes);
            }
        }
        std::fprintf(stderr, "%s: %lld workers ran in %.2f ms on %zu chip%s (kernel compile included on the first run)\n",
                     ctx.program.c_str(), static_cast<long long>(n), run_s * 1e3, chips.size(), chips.size() > 1 ? "s" : "");
        return out;
    }

    template <typename T>
    std::vector<T> run() {
        const std::vector<uint8_t> raw = run_raw(sizeof(T));
        std::vector<T> out(raw.size() / sizeof(T));
        if (!raw.empty()) {
            std::memcpy(static_cast<void*>(out.data()), raw.data(), raw.size());
        }
        return out;
    }
};

}  // namespace tsr
// ---------------------------------------------------------------------------
