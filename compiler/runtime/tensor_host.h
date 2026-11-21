// ---------------------------------------------------------------------------
// tt.Layer, tt.Matrix and tt.Tensor on the host (in the card helper): lays a
// layer's weights out as bf16 tiles in the order the tensor kernel uses
// them, puts them and the matrices in DRAM, and adds the tensor kernel and
// its circular buffers to the workers' cores.
// ---------------------------------------------------------------------------
namespace tsr {

struct TLayer {
    uint32_t in, out, tiles;
    std::shared_ptr<distributed::MeshBuffer> buf;
};
struct TMatrix {
    uint32_t rows, cols, pages;
    std::shared_ptr<distributed::MeshBuffer> buf;
};

constexpr uint32_t kTnsArena = 240;   // most tiles of weights and matrices kept in a core's L1
constexpr uint32_t kTnsWBlock = 40;   // pages per output column of streamed weights
constexpr uint32_t kTnsStreamed = 0xffffffffu;

inline std::shared_ptr<distributed::MeshBuffer> tns_buffer(Context& ctx, std::vector<uint16_t>& v) {
    v.resize((v.size() + 1023) / 1024 * 1024, 0);
    distributed::DeviceLocalBufferConfig local{.page_size = 2048, .buffer_type = BufferType::DRAM};
    distributed::ReplicatedBufferConfig global{.size = v.size() * 2};
    auto buf = distributed::MeshBuffer::create(global, local, ctx.mesh.get());
    distributed::EnqueueWriteMeshBuffer(ctx.mesh->mesh_command_queue(), buf, v, true);
    return buf;
}

// For every output column tile n: the kt tiles of W for it, then a tile
// with the column's biases in row 0.
inline TLayer put_layer(Context& ctx, uint32_t in, uint32_t out, const std::vector<float>& w,
                        const std::vector<float>& b) {
    if (w.size() != static_cast<size_t>(in) * out || b.size() != out) {
        die("tt.NewLayer: unexpected number of weights");
    }
    const uint32_t kt = (in + 31) / 32, nt = (out + 31) / 32;
    std::vector<uint16_t> stream(static_cast<size_t>(nt) * (kt + 1) * 1024, 0);
    for (uint32_t n = 0; n < nt; n++) {
        for (uint32_t k = 0; k < kt; k++) {
            const size_t base = (static_cast<size_t>(n) * (kt + 1) + k) * 1024;
            for (uint32_t r = 0; r < 32 && k * 32 + r < in; r++) {
                for (uint32_t c = 0; c < 32 && n * 32 + c < out; c++) {
                    stream[base + tile_elem(r, c)] = bf16(w[static_cast<size_t>(k * 32 + r) * out + n * 32 + c]);
                }
            }
        }
        const size_t base = (static_cast<size_t>(n) * (kt + 1) + kt) * 1024;
        for (uint32_t c = 0; c < 32 && n * 32 + c < out; c++) {
            stream[base + tile_elem(0, c)] = bf16(b[n * 32 + c]);
        }
    }
    TLayer l{in, out, nt * (kt + 1), nullptr};
    l.buf = tns_buffer(ctx, stream);
    return l;
}

inline TMatrix put_matrix(Context& ctx, uint32_t rows, uint32_t cols, const std::vector<float>& v) {
    if (v.size() != static_cast<size_t>(rows) * cols) {
        die("tt.NewMatrix: unexpected number of values");
    }
    std::vector<uint16_t> raw(v.size());
    for (size_t i = 0; i < v.size(); i++) {
        raw[i] = bf16(v[i]);
    }
    TMatrix m{rows, cols, 0, nullptr};
    m.buf = tns_buffer(ctx, raw);
    m.pages = static_cast<uint32_t>(raw.size() / 1024);
    return m;
}

// Adds the tensor kernel and its circular buffers to the workers' cores:
// arena pages of weights and matrices kept in L1, and a buffer per tensor.
inline void add_tensors(Workers& workers, const std::vector<uint32_t>& tensor_tiles, uint32_t arena) {
    // The circular buffers of the tensors, in order (tessc uses the same list).
    static const uint32_t ids[] = {16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28,
                                   29, 30, 31, 8,  9,  10, 11, 12, 13, 15, 2,  3,  4};
    if (tensor_tiles.size() > 26) {
        die("a worker program can have 26 tt.NewTensor, this one has " + std::to_string(tensor_tiles.size()));
    }
    workers.add_kernel([tensor_tiles, arena](Workers& job, Chip& chip) {
        const CoreRangeSet& set = chip.set;
        auto cb = [&](uint32_t index, uint32_t pages, uint32_t page_size) {
            auto i = static_cast<tt::CBIndex>(index);
            CircularBufferConfig c =
                CircularBufferConfig(pages * page_size, {{i, tt::DataFormat::Float16_b}}).set_page_size(i, page_size);
            CreateCircularBuffer(chip.program, set, c);
        };
        cb(1, 1, 64);               // command
        cb(5, kTnsWBlock, 2048);    // streamed weights, one output column at a time
        cb(6, 3, 2048);             // constants
        cb(7, 1, 32);               // done
        cb(14, arena, 2048);        // weights and matrices kept in L1
        for (size_t i = 0; i < tensor_tiles.size(); i++) {
            cb(ids[i], tensor_tiles[i], 2048);
        }
        std::ifstream f(job.ctx.kernel_dir + "/tensor.cpp");
        if (!f) {
            die("cannot read kernel source " + job.ctx.kernel_dir + "/tensor.cpp");
        }
        std::stringstream src;
        src << f.rdbuf();
        CreateKernelFromString(
            chip.program,
            src.str(),
            set,
            ComputeConfig{.math_fidelity = MathFidelity::HiFi4, .fp32_dest_acc_en = true, .math_approx_mode = false});
    });
}

}  // namespace tsr
// ---------------------------------------------------------------------------
