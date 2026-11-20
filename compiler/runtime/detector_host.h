// ---------------------------------------------------------------------------
// tt.Detector on the host (pasted in by tessc when a worker uses one):
// reads the network, lays its weights out as bf16 tiles in the order the
// compute kernel uses them, and adds the detector to every worker's core.
// ---------------------------------------------------------------------------
namespace tsr {

struct Detector {
    std::vector<std::string> classes;
    std::shared_ptr<distributed::MeshBuffer> resident;  // kept in L1: levels 1-4, prefix, logits
    std::shared_ptr<distributed::MeshBuffer> head;      // streamed once per batch: the hidden layer
    std::shared_ptr<distributed::MeshBuffer> emb;       // 256 x 8 bf16

    const std::string& class_name(int64_t i) const {
        if (i < 0 || static_cast<size_t>(i) >= classes.size()) {
            die("tt.Detector.ClassName: no class " + std::to_string(i));
        }
        return classes[static_cast<size_t>(i)];
    }

    int64_t label(const std::string& name) const {
        for (size_t i = 0; i < classes.size(); i++) {
            if (classes[i] == name) {
                return static_cast<int64_t>(i);
            }
        }
        die("tt.Detector.Label: no class " + name);
    }
};

inline uint16_t bf16(float f) {
    uint32_t b;
    std::memcpy(&b, &f, 4);
    if ((b & 0x7f800000u) == 0x7f800000u) {
        return static_cast<uint16_t>(b >> 16);
    }
    b += 0x7fffu + ((b >> 16) & 1u);
    return static_cast<uint16_t>(b >> 16);
}

constexpr uint32_t tile_elem(uint32_t r, uint32_t c) {
    return ((r >> 4) * 2 + (c >> 4)) * 256 + (r & 15) * 16 + (c & 15);
}

inline Detector load_detector(Context& ctx, const std::string& name, const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) {
        die("cannot open " + path + " (" + name + ", pass --" + name + " FILE)");
    }
    auto u32 = [&]() {
        uint32_t v = 0;
        f.read(reinterpret_cast<char*>(&v), 4);
        return v;
    };
    char magic[4];
    f.read(magic, 4);
    if (std::memcmp(magic, "TSW1", 4) != 0) {
        die(path + " is not a Tessera detector file");
    }
    uint32_t hdr[9];
    for (auto& v : hdr) {
        v = u32();
    }
    const uint32_t want[9] = {4096, hdr[1], 8, 32, 128, 192, 256, 64, 384};
    for (int i = 0; i < 9; i++) {
        if (hdr[i] != want[i] || hdr[1] > 32) {
            die(path + ": the card's detector kernel is written for ndeca's 4 KiB model "
                       "(8, 32, 128, 192, 256, 64, 384) with at most 32 classes");
        }
    }
    Detector d;
    for (uint32_t i = 0; i < hdr[1]; i++) {
        std::string s(u32(), '\0');
        f.read(s.data(), static_cast<std::streamsize>(s.size()));
        d.classes.push_back(s);
    }
    auto floats = [&](size_t n) {
        if (u32() != n) {
            die(path + ": unexpected array size");
        }
        std::vector<float> v(n);
        f.read(reinterpret_cast<char*>(v.data()), static_cast<std::streamsize>(4 * n));
        return v;
    };
    const uint32_t C = hdr[1];
    auto emb = floats(256 * 8);
    auto l1w = floats(64 * 32), l1b = floats(32);
    auto l2w = floats(64 * 128), l2b = floats(128);
    auto l3w = floats(256 * 192), l3b = floats(192);
    auto l4w = floats(384 * 256), l4b = floats(256);
    auto lpw = floats(256 * 64), lpb = floats(64);
    auto h1w = floats(1152 * 384), h1b = floats(384);
    auto h2w = floats(384 * C), h2b = floats(C);
    if (!f) {
        die(path + " is truncated");
    }

    // For every output column n: the k tiles of W[in][out], then a bias
    // tile with the column's biases in row 0. The layers kept in L1 come in
    // one stream, the hidden layer in another.
    std::vector<uint16_t> resident, head;
    std::vector<uint16_t>* target = &resident;
    auto layer = [&](const std::vector<float>& w, uint32_t in, uint32_t out, const std::vector<float>& b) {
        std::vector<uint16_t>& stream = *target;
        for (uint32_t n = 0; n < (out + 31) / 32; n++) {
            for (uint32_t k = 0; k < in / 32; k++) {
                size_t base = stream.size();
                stream.resize(base + 1024, 0);
                for (uint32_t r = 0; r < 32; r++) {
                    for (uint32_t c = 0; c < 32 && n * 32 + c < out; c++) {
                        stream[base + tile_elem(r, c)] = bf16(w[(k * 32 + r) * out + n * 32 + c]);
                    }
                }
            }
            size_t base = stream.size();
            stream.resize(base + 1024, 0);
            for (uint32_t c = 0; c < 32 && n * 32 + c < out; c++) {
                stream[base + tile_elem(0, c)] = bf16(b[n * 32 + c]);
            }
        }
    };
    layer(l1w, 64, 32, l1b);
    layer(l2w, 64, 128, l2b);
    layer(l3w, 256, 192, l3b);
    layer(l4w, 384, 256, l4b);
    layer(lpw, 256, 64, lpb);
    layer(h2w, 384, C, h2b);
    target = &head;
    layer(h1w, 1152, 384, h1b);
    if (resident.size() != 204 * 1024 || head.size() != 444 * 1024) {
        die("internal error: unexpected weight layout");
    }

    std::vector<uint16_t> e(256 * 8);
    for (size_t i = 0; i < e.size(); i++) {
        e[i] = bf16(emb[i]);
    }
    distributed::DeviceLocalBufferConfig wl{.page_size = 2048, .buffer_type = BufferType::DRAM};
    distributed::ReplicatedBufferConfig rg{.size = resident.size() * 2};
    distributed::ReplicatedBufferConfig hg{.size = head.size() * 2};
    d.resident = distributed::MeshBuffer::create(rg, wl, ctx.mesh.get());
    d.head = distributed::MeshBuffer::create(hg, wl, ctx.mesh.get());
    distributed::DeviceLocalBufferConfig el{.page_size = 4096, .buffer_type = BufferType::DRAM};
    distributed::ReplicatedBufferConfig eg{.size = 4096};
    d.emb = distributed::MeshBuffer::create(eg, el, ctx.mesh.get());
    auto& cq = ctx.mesh->mesh_command_queue();
    distributed::EnqueueWriteMeshBuffer(cq, d.resident, resident, true);
    distributed::EnqueueWriteMeshBuffer(cq, d.head, head, true);
    distributed::EnqueueWriteMeshBuffer(cq, d.emb, e, true);
    std::fprintf(stderr, "%s: %s: %zu classes, %zu weight tiles kept in L1, %zu streamed per batch\n",
                 ctx.program.c_str(), name.c_str(), d.classes.size(), resident.size() / 1024, head.size() / 1024);
    return d;
}

inline void push_detector(std::vector<uint32_t>& a, const Detector& d) {
    a.push_back(static_cast<uint32_t>(d.resident->address()));
    a.push_back(static_cast<uint32_t>(d.head->address()));
    a.push_back(static_cast<uint32_t>(d.emb->address()));
    a.push_back(static_cast<uint32_t>(d.classes.size()));
}

// Adds the detector's circular buffers and compute kernel to the workers' cores.
inline void add_detector(Workers& workers, const Detector& d) {
    (void)d;
    workers.add_kernel([](Workers& job) {
        CoreRangeSet set = num_cores_to_corerangeset(static_cast<uint32_t>(job.n),
                                                     job.ctx.mesh->compute_with_storage_grid_size(), true);
        auto cb = [&](uint32_t index, uint32_t pages, uint32_t page_size) {
            auto i = static_cast<tt::CBIndex>(index);
            CircularBufferConfig c =
                CircularBufferConfig(pages * page_size, {{i, tt::DataFormat::Float16_b}}).set_page_size(i, page_size);
            CreateCircularBuffer(job.program, set, c);
        };
        cb(1, 1, 32);      // command
        cb(2, 64, 2048);   // level-1 inputs, two blocks
        cb(3, 8, 2048);    // prefix rows of the batch
        cb(4, 8, 2048);    // histogram rows of the batch
        cb(5, 40, 2048);   // hidden-layer weights, one column at a time
        cb(6, 3, 2048);    // constants
        cb(7, 1, 2048);    // logits, one row per block
        cb(8, 16, 2048);   // level 1
        cb(9, 32, 2048);   // level 2
        cb(10, 24, 2048);  // level 3
        cb(11, 16, 2048);  // level 4
        cb(12, 26, 2048);  // pooled features of one block, in row 0
        cb(13, 12, 2048);  // hidden layer of the batch
        cb(14, 204, 2048); // resident weights
        cb(15, 26, 2048);  // pooled features of the batch, one row per block
        cb(16, 2, 2048);   // prefix branch of the batch
        std::ifstream f(job.ctx.kernel_dir + "/detector.cpp");
        if (!f) {
            die("cannot read kernel source " + job.ctx.kernel_dir + "/detector.cpp");
        }
        std::stringstream src;
        src << f.rdbuf();
        CreateKernelFromString(
            job.program,
            src.str(),
            set,
            ComputeConfig{.math_fidelity = MathFidelity::HiFi4, .fp32_dest_acc_en = true, .math_approx_mode = false});
    });
}

}  // namespace tsr
// ---------------------------------------------------------------------------
