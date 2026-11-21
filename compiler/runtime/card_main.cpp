// ---------------------------------------------------------------------------
// The card helper (pasted in by tessc). A Go program compiled by tessc runs
// main itself, on the host, and starts this process to reach the card: it
// asks it to put files and the detector on the card and to run the workers
// of tt.Launch. Requests arrive on file descriptor 3 and answers leave on
// descriptor 4; standard output and error are shared with the Go program.
//
// A request is {u32 op, u32 length} and the payload; an answer is
// {u32 status (0 = ok), u32 length} and the payload. Numbers are little
// endian, a string is {u32 length} and its bytes. The protocol's other end
// is package tt/card.
// ---------------------------------------------------------------------------
namespace card {

enum Op : uint32_t {
    kLoadBytes = 1,     // name, path -> u32 handle
    kOpenBytes = 2,     // name, path -> u32 handle
    kLoadDetector = 3,  // name, path -> u32 handle
    kPutLayer = 8,      // in, out, weights, biases -> u32 handle
    kPutMatrix = 9,     // rows, cols, values -> u32 handle
    kBytesRead = 4,     // u32 handle -> u64 bytes read from the file by the card's workers
    kLaunch = 5,        // see launch() -> u32 records, then the records
    kClose = 6,         // -> nothing; the helper closes the card and exits
    kCores = 7,         // -> u32 Tensix cores for workers, on all the chips
    kNN = 10,           // a request of package tt/nn, see nn(); an error is an answer with status 1
};

constexpr int kRequests = 3, kAnswers = 4;

// Reads n bytes; false if the other end closed before the first one.
inline bool read_full(void* buf, size_t n) {
    auto* p = static_cast<uint8_t*>(buf);
    size_t got = 0;
    while (got < n) {
        ssize_t r = ::read(kRequests, p + got, n - got);
        if (r == 0 && got == 0) {
            return false;
        }
        if (r <= 0) {
            tsr::die("card helper: the request pipe broke");
        }
        got += static_cast<size_t>(r);
    }
    return true;
}

inline void write_full(const void* buf, size_t n) {
    const auto* p = static_cast<const uint8_t*>(buf);
    while (n > 0) {
        ssize_t w = ::write(kAnswers, p, n);
        if (w <= 0) {
            tsr::die("card helper: the answer pipe broke");
        }
        p += w;
        n -= static_cast<size_t>(w);
    }
}

struct Reader {
    const std::vector<uint8_t>& b;
    size_t at = 0;
    const uint8_t* take(size_t n) {
        if (at + n > b.size()) {
            tsr::die("card helper: malformed request");
        }
        const uint8_t* p = b.data() + at;
        at += n;
        return p;
    }
    uint32_t u32() {
        uint32_t v;
        std::memcpy(&v, take(4), 4);
        return v;
    }
    uint64_t u64() {
        uint64_t v;
        std::memcpy(&v, take(8), 8);
        return v;
    }
    std::string str() {
        uint32_t n = u32();
        const uint8_t* p = take(n);
        return std::string(reinterpret_cast<const char*>(p), n);
    }
    std::vector<float> floats() {
        uint32_t n = u32();
        std::vector<float> v(n);
        if (n > 0) {
            std::memcpy(v.data(), take(static_cast<size_t>(n) * 4), static_cast<size_t>(n) * 4);
        }
        return v;
    }
};

struct Writer {
    std::vector<uint8_t> b;
    void u32(uint32_t v) { b.insert(b.end(), reinterpret_cast<uint8_t*>(&v), reinterpret_cast<uint8_t*>(&v) + 4); }
    void u64(uint64_t v) { b.insert(b.end(), reinterpret_cast<uint8_t*>(&v), reinterpret_cast<uint8_t*>(&v) + 8); }
    void bytes(const std::vector<uint8_t>& v) { b.insert(b.end(), v.begin(), v.end()); }
};

inline void answer(const Writer& w) {
    const uint32_t head[2] = {0, static_cast<uint32_t>(w.b.size())};
    write_full(head, sizeof head);
    write_full(w.b.data(), w.b.size());
}

// An answer that says why the request failed. The helper goes on.
inline void refuse(std::string why) {
    // Of a TT_FATAL ("TT_FATAL @ file:line: condition", "info:", the text,
    // "backtrace:", ...) the text is what the program's author needs.
    if (const size_t bt = why.find("\nbacktrace:"); bt != std::string::npos) {
        why.resize(bt);
    }
    if (const size_t info = why.find("\ninfo:\n"); info != std::string::npos) {
        why.erase(0, info + 7);
    }
    const uint32_t head[2] = {1, static_cast<uint32_t>(why.size())};
    write_full(head, sizeof head);
    write_full(why.data(), why.size());
}

// What a worker's argument is (tt.CardArg kinds).
enum ArgKind : uint32_t { kInt = 0, kBool = 1, kBytes = 2, kDetector = 3, kString = 4, kLayer = 5, kMatrix = 6 };

struct Arg {
    uint32_t kind;
    uint64_t value;
    std::string text;  // kString
};

struct State {
    tsr::Context& ctx;
    std::deque<tsr::Bytes> bytes;  // a deque: the workers keep pointers to them
    std::deque<tsr::Detector> detectors;
    std::deque<tsr::TLayer> layers;
    std::deque<tsr::TMatrix> matrices;
    std::deque<std::string> text;  // the strings the failure positions point to
    std::vector<tsr::Position> positions;

    template <typename T>
    T& get(std::deque<T>& from, uint64_t handle, const char* what) {
        if (handle >= from.size()) {
            tsr::die(std::string("card helper: no such ") + what);
        }
        return from[handle];
    }

    // kernel file, workers, record bytes, capacity, scratch bytes, failure
    // positions {where, what}..., arguments {kind, value[, string]}...
    void launch(Reader& r, Writer& w) {
        const std::string kernel = r.str();
        const uint32_t n = r.u32(), record = r.u32(), capacity = r.u32(), scratch = r.u32();
        text.clear();
        positions.clear();
        for (uint32_t i = 0, np = r.u32(); i < np; i++) {
            text.push_back(r.str());
            text.push_back(r.str());
        }
        for (size_t i = 0; i + 1 < text.size(); i += 2) {
            positions.push_back({text[i].c_str(), text[i + 1].c_str()});
        }
        ctx.positions = positions.data();
        ctx.n_positions = positions.size();
        std::vector<uint32_t> tensors;  // the tiles of each tt.NewTensor
        for (uint32_t i = 0, nt = r.u32(); i < nt; i++) {
            tensors.push_back(r.u32());
        }
        std::vector<Arg> args;
        for (uint32_t i = 0, na = r.u32(); i < na; i++) {
            Arg a;
            a.kind = r.u32();
            a.value = r.u64();
            if (a.kind == kString) {
                a.text = r.str();
            }
            args.push_back(a);
        }
        if (record == 0) {
            tsr::die("card helper: records of 0 bytes");
        }
        tsr::Workers job(ctx, n, record, capacity, scratch, kernel);
        // Where each layer and matrix goes in the cores' L1: one after
        // another while they fit; a layer that does not is streamed.
        std::vector<uint32_t> base(args.size(), tsr::kTnsStreamed);
        uint32_t arena = 0;
        bool tensor_args = false;
        for (size_t i = 0; i < args.size(); i++) {
            if (args[i].kind == kMatrix) {
                const tsr::TMatrix& m = get(matrices, args[i].value, "matrix");
                if (arena + m.pages > tsr::kTnsArena) {
                    tsr::die("tt.Matrix: the matrices the workers use do not fit in a core's memory");
                }
                base[i] = arena;
                arena += m.pages;
                tensor_args = true;
            }
        }
        for (size_t i = 0; i < args.size(); i++) {
            if (args[i].kind == kLayer) {
                const tsr::TLayer& l = get(layers, args[i].value, "layer");
                if (arena + l.tiles <= tsr::kTnsArena) {
                    base[i] = arena;
                    arena += l.tiles;
                } else if ((l.in + 31) / 32 + 1 > tsr::kTnsWBlock) {
                    tsr::die("tt.Layer: a layer of " + std::to_string(l.in) + " inputs is too wide for the card (" +
                             std::to_string((tsr::kTnsWBlock - 1) * 32) + " at most)");
                }
                tensor_args = true;
            }
        }
        arena = std::max<uint32_t>(arena, 1);
        if (!tensors.empty() || tensor_args) {
            tsr::add_tensors(job, tensors, arena);  // the tensor kernel runs on the same cores
        }
        for (const Arg& a : args) {
            if (a.kind == kDetector) {
                tsr::add_detector(job, get(detectors, a.value, "detector"));  // runs on the same cores
            } else if (a.kind == kBytes) {
                job.serve(get(bytes, a.value, "byte array"));  // read on demand if it came from tt.OpenBytes
            }
        }
        for (int64_t index = 0; index < job.n; index++) {
            std::vector<uint32_t> captured;
            for (size_t i = 0; i < args.size(); i++) {
                const Arg& a = args[i];
                switch (a.kind) {
                case kInt:
                case kBool:
                    tsr::push64(captured, static_cast<int64_t>(a.value));
                    break;
                case kBytes:
                    job.push_bytes(captured, get(bytes, a.value, "byte array"), index);
                    break;
                case kDetector:
                    tsr::push_detector(captured, get(detectors, a.value, "detector"));
                    break;
                case kString:
                    tsr::push_string(captured, a.text);
                    break;
                case kLayer: {
                    const tsr::TLayer& l = get(layers, a.value, "layer");
                    for (uint32_t v : {l.address(), l.in, l.out, base[i], arena, l.bias}) {
                        captured.push_back(v);
                    }
                    break;
                }
                case kMatrix: {
                    const tsr::TMatrix& m = get(matrices, a.value, "matrix");
                    for (uint32_t v : {static_cast<uint32_t>(m.buf->address()), m.rows, m.cols, base[i], arena}) {
                        captured.push_back(v);
                    }
                    break;
                }
                default:
                    tsr::die("card helper: unknown argument kind");
                }
            }
            job.set_args(index, captured);
        }
        const std::vector<uint8_t> raw = job.run_raw(record);
        w.u32(static_cast<uint32_t>(raw.size() / record));
        w.bytes(raw);
    }
};

#if defined(TSR_NN)
// A request of package tt/nn (the kinds are in tt/nn/nn.go; 6 and 7 are
// tt.LayerOf's and tt.MatrixOf's).
inline void nn(State& st, Reader& r, Writer& w) {
    tsr::Context& ctx = st.ctx;
    auto dims = [&] {
        std::vector<uint32_t> d(r.u32());
        for (uint32_t& x : d) {
            x = r.u32();
        }
        return d;
    };
    switch (r.u32()) {
    case 1: {  // dims, floats -> handle
        const std::vector<uint32_t> d = dims();
        w.u32(tsr::nn::put(ctx, d, r.floats()));
        break;
    }
    case 2: {  // handle -> floats
        const std::vector<float> v = tsr::nn::floats(r.u32());
        w.u32(static_cast<uint32_t>(v.size()));
        w.b.insert(w.b.end(), reinterpret_cast<const uint8_t*>(v.data()),
                   reinterpret_cast<const uint8_t*>(v.data() + v.size()));
        break;
    }
    case 3:  // handle
        tsr::nn::free(r.u32());
        break;
    case 4: {  // operation, handles -> handle
        const uint32_t op = r.u32();
        std::vector<uint32_t> in(r.u32());
        for (uint32_t& h : in) {
            h = r.u32();
        }
        w.u32(tsr::nn::run(op, in));
        break;
    }
    case 5: {  // dims, value -> handle
        const std::vector<uint32_t> d = dims();
        float value;
        std::memcpy(&value, r.take(4), 4);
        w.u32(tsr::nn::full(ctx, d, value));
        break;
    }
    case 6: {  // weights, biases (or 0xffffffff) -> layer
        const uint32_t weights = r.u32(), bias = r.u32();
        st.layers.push_back(tsr::nn::layer_of(ctx, weights, bias));
        w.u32(static_cast<uint32_t>(st.layers.size() - 1));
        break;
    }
    case 7:  // handle -> matrix
        st.matrices.push_back(tsr::nn::matrix_of(ctx, r.u32()));
        w.u32(static_cast<uint32_t>(st.matrices.size() - 1));
        break;
    default:
        throw std::runtime_error("unknown request");
    }
}
#endif

}  // namespace card

int main(int argc, char** argv) {
    tsr::Context ctx;
    ctx.program = argc > 1 ? argv[1] : "tessera";
    ctx.kernel_dir = TSR_KERNEL_DIR;
    if (const char* dir = std::getenv("TESSERA_KERNELS")) {
        ctx.kernel_dir = dir;
    }
    ctx.compile_only = std::getenv("TESSERA_COMPILE_ONLY") != nullptr;
    tsr::current = &ctx;
    // All the card's chips (TESSERA_DEVICE or TESSERA_CHIPS can say
    // otherwise), with two command queues: the second carries the files read
    // on demand while the first runs the workers.
    ctx.mesh = tsr::open_card(2, -1, true);
    card::State st{ctx};
    for (bool open = true; open;) {
        uint32_t head[2];
        if (!card::read_full(head, sizeof head)) {
            break;  // the Go program has finished
        }
        std::vector<uint8_t> body(head[1]);
        if (!body.empty() && !card::read_full(body.data(), body.size())) {
            tsr::die("card helper: the request pipe broke");
        }
        card::Reader r{body};
        card::Writer w;
        switch (head[0]) {
        case card::kLoadBytes: {
            const std::string name = r.str(), path = r.str();
            st.bytes.push_back(tsr::load_bytes(ctx, name, path));
            w.u32(static_cast<uint32_t>(st.bytes.size() - 1));
            break;
        }
        case card::kOpenBytes: {
            const std::string name = r.str(), path = r.str();
            st.bytes.push_back(tsr::open_bytes(ctx, name, path));
            w.u32(static_cast<uint32_t>(st.bytes.size() - 1));
            break;
        }
        case card::kLoadDetector: {
            const std::string name = r.str(), path = r.str();
            st.detectors.push_back(tsr::load_detector(ctx, name, path));
            w.u32(static_cast<uint32_t>(st.detectors.size() - 1));
            break;
        }
        case card::kPutLayer: {
            const uint32_t in = r.u32(), out = r.u32();
            const std::vector<float> wts = r.floats(), bias = r.floats();
            st.layers.push_back(tsr::put_layer(ctx, in, out, wts, bias));
            w.u32(static_cast<uint32_t>(st.layers.size() - 1));
            break;
        }
        case card::kPutMatrix: {
            const uint32_t rows = r.u32(), cols = r.u32();
            st.matrices.push_back(tsr::put_matrix(ctx, rows, cols, r.floats()));
            w.u32(static_cast<uint32_t>(st.matrices.size() - 1));
            break;
        }
        case card::kBytesRead:
            w.u64(static_cast<uint64_t>(st.get(st.bytes, r.u32(), "byte array").bytes_read()));
            break;
        case card::kLaunch:
            st.launch(r, w);
            break;
        case card::kClose:
            open = false;
            break;
        case card::kCores:
            w.u32(static_cast<uint32_t>(tsr::cores(ctx)));
            break;
#if defined(TSR_NN)
        case card::kNN:
            // ttnn reports an error as an exception: the Go program gets it
            // as the answer, and the card stays open.
            try {
                card::nn(st, r, w);
            } catch (const std::exception& e) {
                card::refuse(e.what());
                continue;
            }
            break;
#endif
        default:
            tsr::die("card helper: unknown request " + std::to_string(head[0]));
        }
        card::answer(w);
    }
#if defined(TSR_NN)
    st.layers.clear();  // they hold tensors
    tsr::nn::clear();
#endif
    tsr::close_card(ctx);
    return 0;
}
