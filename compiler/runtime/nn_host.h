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
    kSum,  // attributes: the dimension
    kMean,
    kMax,
    kSoftmax,
    kLayerNorm,  // a, gamma, beta; the bits of epsilon
    kLinear,     // a, w and perhaps b
    kReshape,    // the new shape
    kTranspose,  // two dimensions
    kConcat,     // the dimension
    kSlice,      // where each dimension begins, then where each ends
    kConv2d,     // x, w and perhaps b; stride, padding, dilation (2 each), groups, the result's shape
    kMaxPool2d,  // kernel, stride, padding (2 each), the result's shape
    kAvgPool2d,
    kAttention,  // q, k, v; 1 if causal
};

inline std::vector<std::optional<ttnn::Tensor>> tensors;
inline std::vector<uint32_t> unused;  // handles to give out again

inline ttnn::Tensor& get(uint32_t handle) {
    if (handle >= tensors.size() || !tensors[handle]) {
        throw std::runtime_error("no such tensor");
    }
    return *tensors[handle];
}

// Keeps a tensor, in tiles in DRAM, and returns its handle.
inline uint32_t keep(ttnn::Tensor t) {
    if (!t.memory_config().is_dram() || t.memory_config().is_sharded()) {
        t = ttnn::to_memory_config(t, ttnn::DRAM_MEMORY_CONFIG);
    }
    if (t.layout() != ttnn::TILE_LAYOUT) {
        t = ttnn::to_layout(t, ttnn::TILE_LAYOUT);
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

// nn.ArgMax: for each line along the last dimension, where its largest
// element is.
inline std::vector<uint32_t> argmax(uint32_t handle) {
    const ttnn::Tensor r = ttnn::argmax(get(handle), -1, true);
    return ttnn::distributed::get_device_tensors(r.cpu())[0].to_vector<uint32_t>();
}

inline void free(uint32_t handle) {
    get(handle);
    tensors[handle].reset();
    unused.push_back(handle);
}

// A tensor of floats on the host, row by row: what ttnn's convolution
// wants its weights as.
inline ttnn::Tensor on_host(uint32_t handle, const std::vector<uint32_t>& dims) {
    const tt::tt_metal::TensorSpec spec(
        shape_of(dims), tt::tt_metal::TensorLayout(ttnn::DataType::FLOAT32,
                                                   tt::tt_metal::PageConfig(ttnn::ROW_MAJOR_LAYOUT), ttnn::MemoryConfig{}));
    return ttnn::Tensor::from_vector(floats(handle), spec);
}

// nn.Embedding: the rows of a table that the indices name.
inline uint32_t embedding(tsr::Context& ctx, uint32_t table, const std::vector<uint32_t>& ids) {
    const ttnn::Tensor& t = get(table);
    const uint32_t n = static_cast<uint32_t>(ids.size());
    const tt::tt_metal::TensorSpec spec(
        shape_of({1, n}), tt::tt_metal::TensorLayout(ttnn::DataType::UINT32,
                                                     tt::tt_metal::PageConfig(ttnn::ROW_MAJOR_LAYOUT), ttnn::DRAM_MEMORY_CONFIG));
    const ttnn::Tensor rows = ttnn::embedding(ttnn::Tensor::from_vector(ids, spec, ctx.mesh.get()), t);
    return keep(ttnn::reshape(rows, shape_of({n, t.logical_shape()[1]})));
}

// nn.Attention, from ttnn's own operations: softmax(q k' / sqrt(d)) v,
// with what lies ahead hidden if causal.
inline ttnn::Tensor attention(tsr::Context& ctx, const ttnn::Tensor& q, const ttnn::Tensor& k, const ttnn::Tensor& v,
                              bool causal) {
    const int r = static_cast<int>(q.logical_shape().rank());
    const uint32_t s = q.logical_shape()[r - 2], t = k.logical_shape()[r - 2], d = q.logical_shape()[r - 1];
    ttnn::Tensor scores = ttnn::multiply(ttnn::matmul(q, ttnn::transpose(k, r - 2, r - 1)), 1.0f / std::sqrt(float(d)));
    if (causal) {
        std::vector<float> mask(static_cast<size_t>(s) * t, 0.0f);
        for (uint32_t i = 0; i < s; i++) {
            for (uint32_t j = i + 1; j < t; j++) {
                mask[static_cast<size_t>(i) * t + j] = -10000.0f;
            }
        }
        const tt::tt_metal::TensorSpec spec(
            shape_of({s, t}), tt::tt_metal::TensorLayout(ttnn::DataType::BFLOAT16, tt::tt_metal::PageConfig(ttnn::TILE_LAYOUT),
                                                         ttnn::DRAM_MEMORY_CONFIG));
        scores = ttnn::add(scores, ttnn::Tensor::from_vector(mask, spec, ctx.mesh.get()));
    }
    return ttnn::matmul(ttnn::softmax(scores, -1), v);
}

inline uint32_t run(tsr::Context& ctx, uint32_t op, const std::vector<uint32_t>& in, const std::vector<uint32_t>& attrs) {
    auto arg = [&](size_t i) -> ttnn::Tensor& {
        if (i >= in.size()) {
            throw std::runtime_error("an operation with too few tensors");
        }
        return get(in[i]);
    };
    auto attr = [&](size_t i) -> int {
        if (i >= attrs.size()) {
            throw std::runtime_error("an operation with too few attributes");
        }
        return static_cast<int>(attrs[i]);
    };
    auto u = [&](size_t i) { return static_cast<uint32_t>(attr(i)); };
    switch (op) {
    case kConv2d: {
        const ttnn::Tensor& x = arg(0);
        const auto xs = x.logical_shape(), ws = arg(1).logical_shape();
        std::optional<ttnn::Tensor> bias;
        if (in.size() > 2) {
            bias = on_host(in[2], {1, 1, 1, ws[0]});
        }
        auto result = ttnn::conv2d(x, on_host(in[1], {ws[0], ws[1], ws[2], ws[3]}), ctx.mesh.get(), xs[3], ws[0], xs[0],
                                   xs[1], xs[2], std::array<uint32_t, 2>{ws[2], ws[3]}, std::array<uint32_t, 2>{u(0), u(1)},
                                   std::array<uint32_t, 2>{u(2), u(3)}, std::array<uint32_t, 2>{u(4), u(5)}, u(6),
                                   std::nullopt, bias);
        return keep(ttnn::reshape(std::get<ttnn::Tensor>(result), shape_of({u(7), u(8), u(9), u(10)})));
    }
    case kMaxPool2d:
    case kAvgPool2d: {
        // ttnn's pools take their input row by row, not in tiles.
        const ttnn::Tensor x = ttnn::to_layout(arg(0), ttnn::ROW_MAJOR_LAYOUT);
        const auto xs = x.logical_shape();
        const std::array<uint32_t, 2> kernel{u(0), u(1)}, stride{u(2), u(3)}, padding{u(4), u(5)};
        ttnn::Tensor out = op == kMaxPool2d ? ttnn::max_pool2d(x, xs[0], xs[1], xs[2], xs[3], kernel, stride, padding,
                                                               std::array<uint32_t, 2>{1, 1})
                                                  .at(0)
                                            : ttnn::avg_pool2d(x, xs[0], xs[1], xs[2], xs[3], kernel, stride, padding);
        // The result comes in shards in L1: out of there before anything else.
        out = ttnn::to_memory_config(out, ttnn::DRAM_MEMORY_CONFIG);
        return keep(ttnn::reshape(out, shape_of({u(6), u(7), u(8), u(9)})));
    }
    case kAttention: return keep(attention(ctx, arg(0), arg(1), arg(2), attr(0) != 0));
    case kSum: return keep(ttnn::sum(arg(0), attr(0), true));
    case kMean: return keep(ttnn::mean(arg(0), attr(0), true));
    case kMax: return keep(ttnn::max(arg(0), attr(0), true));
    case kSoftmax: return keep(ttnn::softmax(arg(0), attr(0)));
    case kLayerNorm: {
        float eps;
        const uint32_t bits = static_cast<uint32_t>(attr(0));
        std::memcpy(&eps, &bits, 4);
        return keep(ttnn::layer_norm(arg(0), eps, arg(1), arg(2)));
    }
    case kLinear:
        if (in.size() > 2) {
            return keep(ttnn::linear(arg(0), arg(1), arg(2)));
        }
        return keep(ttnn::linear(arg(0), arg(1)));
    case kReshape: return keep(ttnn::reshape(arg(0), shape_of(attrs)));
    case kTranspose: return keep(ttnn::transpose(arg(0), attr(0), attr(1)));
    case kConcat: {
        std::vector<ttnn::Tensor> ts;
        for (size_t i = 0; i < in.size(); i++) {
            ts.push_back(arg(i));
        }
        return keep(ttnn::concat(ts, attr(0)));
    }
    case kSlice: {
        const size_t n = attrs.size() / 2;
        const ttsl::SmallVector<uint32_t> begins(attrs.begin(), attrs.begin() + n), ends(attrs.begin() + n, attrs.end()),
            step(n, 1);
        return keep(ttnn::slice(arg(0), begins, ends, step));
    }
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
