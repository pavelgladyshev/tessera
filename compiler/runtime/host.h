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

[[noreturn]] inline void die(const std::string& msg) {
    std::cerr << msg << "\n";
    std::exit(1);
}

inline Context* current = nullptr;

[[noreturn]] inline void fail(int64_t pos, int64_t value) {
    std::string where = "?", what = "runtime error";
    if (current && pos >= 0 && static_cast<size_t>(pos) < current->n_positions) {
        where = current->positions[pos].where;
        what = current->positions[pos].what;
    }
    die("panic: " + what + " (value " + std::to_string(value) + ") at " + where);
}

using seconds = std::chrono::duration<double>;
inline double since(std::chrono::steady_clock::time_point t0) {
    return seconds(std::chrono::steady_clock::now() - t0).count();
}

// Go slices and arrays, with Go's bounds checks.
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
struct Workers {
    Context& ctx;
    int64_t n;
    uint32_t slot;
    int64_t cap;
    Program program;
    KernelHandle kernel{};
    std::vector<CoreCoord> cores;
    std::shared_ptr<distributed::MeshBuffer> results;

    Workers(Context& c, int64_t workers, uint32_t record_bytes, int64_t capacity, uint32_t scratch_bytes,
            const std::string& kernel_file)
        : ctx(c), n(workers), cap(capacity), program(CreateProgram()) {
        CoreCoord grid = ctx.mesh->compute_with_storage_grid_size();
        int64_t available = static_cast<int64_t>(grid.x) * grid.y;
        if (n < 1 || n > available || n > static_cast<int64_t>(kMaxWorkers)) {
            die("tt.Parallel(" + std::to_string(n) + "): the card has " + std::to_string(available) +
                " Tensix cores, one per worker");
        }
        slot = std::max<uint32_t>(32, (record_bytes + 31) / 32 * 32);
        CoreRangeSet set = num_cores_to_corerangeset(static_cast<uint32_t>(n), grid, true);
        cores = corerange_to_cores(set, static_cast<uint32_t>(n), true);

        CircularBufferConfig scratch =
            CircularBufferConfig(scratch_bytes, {{tt::CBIndex::c_0, tt::DataFormat::UInt8}})
                .set_page_size(tt::CBIndex::c_0, scratch_bytes);
        CreateCircularBuffer(program, set, scratch);

        distributed::DeviceLocalBufferConfig local{.page_size = slot, .buffer_type = BufferType::DRAM};
        distributed::ReplicatedBufferConfig global{.size = static_cast<uint64_t>(n) * (cap + 1) * slot};
        results = distributed::MeshBuffer::create(global, local, ctx.mesh.get());

        std::ifstream f(ctx.kernel_dir + "/" + kernel_file);
        if (!f) {
            die("cannot read kernel source " + ctx.kernel_dir + "/" + kernel_file + " (pass --kernels DIR)");
        }
        std::stringstream src;
        src << f.rdbuf();
        kernel = CreateKernelFromString(
            program,
            src.str(),
            set,
            DataMovementConfig{.processor = DataMovementProcessor::RISCV_0, .noc = NOC::RISCV_0_default});
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
        const CoreCoord table = ctx.mesh->worker_core_from_logical_core(cores[0]);
        a.push_back(1);  // tsr::kFromHost
        a.push_back(static_cast<uint32_t>(b.l1->address()));
        a.push_back(kWindowLog2);
        push64(a, b.len);
        a.push_back(static_cast<uint32_t>(table.x));
        a.push_back(static_cast<uint32_t>(table.y));
        a.push_back(static_cast<uint32_t>(b.l1->address()) + kTableOffset + 32 * static_cast<uint32_t>(w));
        a.push_back(static_cast<uint32_t>(b.staged->address()));
        a.push_back(static_cast<uint32_t>(w) << (kWindowLog2 - 12));  // first 4 KiB page
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
        IDevice* dev = ctx.mesh->get_devices().at(0);
        auto& io = ctx.mesh->mesh_command_queue(1);
        constexpr uint32_t window = 1u << kWindowLog2;
        std::vector<uint32_t> last(from_host.size() * static_cast<size_t>(n), 0);
        std::vector<std::vector<uint8_t>> bufs(static_cast<size_t>(n), std::vector<uint8_t>(window + 32));
        std::vector<uint8_t> table(32 * static_cast<size_t>(n));
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
                const Bytes& b = *from_host[i];
                const uint32_t addr = static_cast<uint32_t>(b.l1->address());
                auto tp = clock::now();
                detail::ReadFromDeviceL1(dev, cores[0], addr + kTableOffset, std::span<uint8_t>(table));
                t_poll += since(tp);
                answers.clear();
                pieces.clear();
                for (int64_t w = 0; w < n; w++) {
                    // {req_seq, req_count, req_len, -, req_off, req_stride}: see tsr::Mailbox
                    const uint8_t* mb = table.data() + 32 * w;
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
                        transfers.push_back(distributed::ShardDataTransfer{distributed::MeshCoordinate(0, 0)}
                                                .host_data(buf.data())
                                                .region(BufferRegion(static_cast<uint64_t>(a.w) << kWindowLog2, padded)));
                    }
                }
                auto tw = clock::now();
                if (!transfers.empty()) {
                    io.enqueue_write_shards(b.staged, transfers, true);
                }
                for (const Answer& a : answers) {
                    // {ack_seq, got_len} at byte 32 of the mailbox; the number last.
                    const CoreCoord core = cores[static_cast<size_t>(a.w)];
                    detail::WriteToDeviceL1(dev, core, addr + 36,
                                            std::span<const uint8_t>(reinterpret_cast<const uint8_t*>(&a.got), 4));
                    detail::WriteToDeviceL1(dev, core, addr + 32,
                                            std::span<const uint8_t>(reinterpret_cast<const uint8_t*>(&a.seq), 4));
                }
                t_card += since(tw);
                served += static_cast<int64_t>(answers.size());
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
        SetRuntimeArgs(program, kernel, cores[static_cast<size_t>(w)], a);
    }

    template <typename T>
    std::vector<T> run() {
        std::vector<T> out;
        if (ctx.compile_only) {
            detail::CompileProgram(ctx.mesh->get_devices().at(0), program);
            std::fprintf(stderr, "%s: worker kernel compiled for %lld cores\n", ctx.program.c_str(),
                         static_cast<long long>(n));
            return out;
        }
        auto& cq = ctx.mesh->mesh_command_queue();
        distributed::Finish(cq);  // earlier writes to DRAM are done
        IDevice* dev = ctx.mesh->get_devices().at(0);
        for (const Bytes* b : from_host) {
            const std::vector<uint8_t> zeros(kTableBytes, 0);
            const uint32_t addr = static_cast<uint32_t>(b->l1->address());
            for (const CoreCoord& core : cores) {
                detail::WriteToDeviceL1(dev, core, addr, std::span<const uint8_t>(zeros.data(), kMailboxBytes));
            }
            detail::WriteToDeviceL1(dev, cores[0], addr + kTableOffset, std::span<const uint8_t>(zeros));
        }
        distributed::MeshWorkload workload;
        workload.add_program(distributed::MeshCoordinateRange(ctx.mesh->shape()), std::move(program));
        auto t0 = std::chrono::steady_clock::now();
        distributed::EnqueueMeshWorkload(cq, workload, false);
        int64_t requests = 0;
        if (!from_host.empty()) {
            requests = serve_until(cq.enqueue_record_event_to_host());
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
        distributed::EnqueueReadMeshBuffer(cq, raw, results, true);
        for (int64_t w = 0; w < n; w++) {
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
                T v;
                std::memcpy(&v, base + static_cast<size_t>(i + 1) * slot, sizeof(T));
                out.push_back(v);
            }
        }
        std::fprintf(stderr, "%s: %lld workers ran in %.2f ms (kernel compile included on the first run)\n",
                     ctx.program.c_str(), static_cast<long long>(n), run_s * 1e3);
        return out;
    }
};

}  // namespace tsr
// ---------------------------------------------------------------------------
