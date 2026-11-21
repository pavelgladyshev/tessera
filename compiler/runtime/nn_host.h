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

// Frees every tensor: before the card closes.
inline void clear() {
    tensors.clear();
    unused.clear();
}

}  // namespace tsr::nn
#endif
