// ---------------------------------------------------------------------------
// ttnn for main (pasted in by tessc for programs that import tt/nn). The
// card helper keeps the tensors that package tt/nn makes, and runs its
// operations with ttnn on the card the helper has open. ttnn and the
// workers take turns: a request is served between launches.
//
// Tensors are bf16 in tiles, in DRAM (the workers take most of each core's
// L1), and the same on every chip of the mesh. A ttnn error is a C++
// exception, which the helper's loop turns into an answer.
// ---------------------------------------------------------------------------
#if defined(TSR_NN)
namespace tsr::nn {

// The operations, as in tt/nn/nn.go.
enum Op : uint32_t {
    kAdd = 1,
    kSub,
    kMul,
    kDiv,
    kReLU,
    kSigmoid,
    kTanh,
    kExp,
    kLog,
    kSqrt,
    kGELU,
    kMatMul,
};

inline std::vector<std::optional<ttnn::Tensor>> tensors;
inline std::vector<uint32_t> unused;  // handles to give out again

inline ttnn::Tensor& get(uint32_t handle) {
    if (handle >= tensors.size() || !tensors[handle]) {
        throw std::runtime_error("no such tensor");
    }
    return *tensors[handle];
}

// Keeps a tensor, in DRAM, and returns its handle.
inline uint32_t keep(ttnn::Tensor t) {
    if (!t.memory_config().is_dram()) {
        t = ttnn::to_memory_config(t, ttnn::DRAM_MEMORY_CONFIG);
    }
    if (!unused.empty()) {
        const uint32_t h = unused.back();
        unused.pop_back();
        tensors[h] = std::move(t);
        return h;
    }
    tensors.emplace_back(std::move(t));
    return static_cast<uint32_t>(tensors.size() - 1);
}

inline ttnn::Shape shape_of(const std::vector<uint32_t>& dims) {
    return ttnn::Shape(ttsl::SmallVector<uint32_t>(dims.begin(), dims.end()));
}

inline uint32_t put(tsr::Context& ctx, const std::vector<uint32_t>& dims, const std::vector<float>& v) {
    const tt::tt_metal::TensorSpec spec(
        shape_of(dims),
        tt::tt_metal::TensorLayout(ttnn::DataType::BFLOAT16, tt::tt_metal::PageConfig(ttnn::TILE_LAYOUT),
                                   ttnn::DRAM_MEMORY_CONFIG));
    return keep(ttnn::Tensor::from_vector(v, spec, ctx.mesh.get()));
}

inline uint32_t full(tsr::Context& ctx, const std::vector<uint32_t>& dims, float value) {
    return keep(ttnn::full(shape_of(dims), value, ttnn::DataType::BFLOAT16, ttnn::TILE_LAYOUT, std::ref(*ctx.mesh),
                           ttnn::DRAM_MEMORY_CONFIG));
}

// The numbers of a tensor: the first chip's copy.
inline std::vector<float> floats(uint32_t handle) {
    return ttnn::distributed::get_device_tensors(get(handle).cpu())[0].to_vector<float>();
}

inline void free(uint32_t handle) {
    get(handle);
    tensors[handle].reset();
    unused.push_back(handle);
}

inline uint32_t run(uint32_t op, const std::vector<uint32_t>& in) {
    auto arg = [&](size_t i) -> ttnn::Tensor& {
        if (i >= in.size()) {
            throw std::runtime_error("an operation with too few tensors");
        }
        return get(in[i]);
    };
    switch (op) {
    case kAdd: return keep(ttnn::add(arg(0), arg(1)));
    case kSub: return keep(ttnn::subtract(arg(0), arg(1)));
    case kMul: return keep(ttnn::multiply(arg(0), arg(1)));
    case kDiv: return keep(ttnn::divide(arg(0), arg(1)));
    case kReLU: return keep(ttnn::relu(arg(0)));
    case kSigmoid: return keep(ttnn::sigmoid(arg(0)));
    case kTanh: return keep(ttnn::tanh(arg(0)));
    case kExp: return keep(ttnn::exp(arg(0)));
    case kLog: return keep(ttnn::log(arg(0)));
    case kSqrt: return keep(ttnn::sqrt(arg(0)));
    case kGELU: return keep(ttnn::gelu(arg(0)));
    case kMatMul: return keep(ttnn::matmul(arg(0), arg(1)));
    }
    throw std::runtime_error("unknown operation " + std::to_string(op));
}

// A 2-D bf16 tensor in tiles on the card, for the workers to read.
inline const ttnn::Tensor& for_workers(uint32_t handle, const char* what) {
    const ttnn::Tensor& t = get(handle);
    if (t.dtype() != ttnn::DataType::BFLOAT16 || t.layout() != ttnn::TILE_LAYOUT) {
        throw std::runtime_error(std::string(what) + " must be a bf16 tensor in tiles");
    }
    return t;
}

// tt.LayerOf: the workers read the tensors' own buffers. bias is a handle,
// or 0xffffffff for none (a row of zero tiles then).
inline TLayer layer_of(tsr::Context& ctx, uint32_t weights, uint32_t bias) {
    const ttnn::Tensor& w = for_workers(weights, "the weights");
    if (w.logical_shape().rank() != 2) {
        throw std::runtime_error("the weights must have 2 dimensions");
    }
    TLayer l{w.logical_shape()[0], w.logical_shape()[1], 0, nullptr};
    const uint32_t kt = (l.in + 31) / 32, nt = (l.out + 31) / 32;
    l.tiles = nt * (kt + 1);
    l.w = static_cast<uint32_t>(w.mesh_buffer().address());
    if (bias == 0xffffffffu) {
        std::vector<uint16_t> zeros(static_cast<size_t>(nt) * 1024, 0);
        auto buf = tns_buffer(ctx, zeros);
        l.bias = static_cast<uint32_t>(buf->address());
        l.hold = std::make_shared<std::pair<ttnn::Tensor, std::shared_ptr<distributed::MeshBuffer>>>(w, buf);
    } else {
        const ttnn::Tensor& b = for_workers(bias, "the biases");
        if (b.logical_shape().volume() != l.out) {
            throw std::runtime_error("the biases do not fit the weights");
        }
        l.bias = static_cast<uint32_t>(b.mesh_buffer().address());
        l.hold = std::make_shared<std::pair<ttnn::Tensor, ttnn::Tensor>>(w, b);
    }
    return l;
}

// tt.MatrixOf: a copy of the tensor row by row, which is how worker code
// reads a matrix (ttnn keeps tiles).
inline TMatrix matrix_of(tsr::Context& ctx, uint32_t handle) {
    const ttnn::Tensor& t = get(handle);
    if (t.logical_shape().rank() != 2) {
        throw std::runtime_error("the tensor must have 2 dimensions");
    }
    return put_matrix(ctx, t.logical_shape()[0], t.logical_shape()[1], floats(handle));
}

// Frees every tensor: before the card closes.
inline void clear() {
    tensors.clear();
    unused.clear();
}

}  // namespace tsr::nn
#endif
