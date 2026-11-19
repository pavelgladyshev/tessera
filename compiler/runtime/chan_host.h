// ---------------------------------------------------------------------------
// Goroutines and channels on the host side (pasted in by tessc): main makes
// channels and starts goroutines; the network starts on the card the first
// time main receives, or when main ends; main receives from channels whose
// sender is a goroutine.
// ---------------------------------------------------------------------------
namespace tsr {

struct ChanRef {
    int id = -1;
};

struct Arg {
    enum Kind { Int, Send, Recv, BytesArg } kind;
    int64_t value;
    int chan;
    const Bytes* bytes;
};
inline Arg arg_int(int64_t v) { return {Arg::Int, v, -1, nullptr}; }
inline Arg arg_send(ChanRef c) { return {Arg::Send, 0, c.id, nullptr}; }
inline Arg arg_recv(ChanRef c) { return {Arg::Recv, 0, c.id, nullptr}; }
inline Arg arg_bytes(const Bytes& b) { return {Arg::BytesArg, 0, -1, &b}; }

// Block layout, as in the device runtime.
constexpr uint32_t kChWritten = 0, kChClosed = 16, kChConsumed = 32, kChBlockExtra = 64;
constexpr uint32_t kStatusBytes = 32;  // per core: {state, pad, position, value}

class Net {
public:
    Net(Context& c, uint32_t scratch_bytes) : ctx_(c), scratch_bytes_(scratch_bytes) {}

    ChanRef make_chan(uint32_t elem_bytes, int64_t capacity) {
        if (started_) {
            die("make(chan) after the goroutines started: make every channel before main first receives");
        }
        if (capacity < 1) {
            capacity = 1;  // an unbuffered channel holds one value on the card
        }
        Channel ch;
        ch.cap = static_cast<uint32_t>(capacity);
        ch.slot = (elem_bytes + 15) / 16 * 16;
        ch.ring = ch.cap * ch.slot;
        chans_.push_back(ch);
        return ChanRef{static_cast<int>(chans_.size()) - 1};
    }

    void go(int fid, const char* name, std::vector<Arg> args) {
        if (started_) {
            die(std::string("go ") + name + ": start every goroutine before main first receives from a channel");
        }
        for (const Arg& a : args) {
            if ((a.kind == Arg::Send || a.kind == Arg::Recv) && (a.chan < 0 || a.chan >= static_cast<int>(chans_.size()))) {
                die(std::string("go ") + name + ": a channel argument was never made (nil channel)");
            }
            if (a.kind == Arg::BytesArg && a.bytes->from_host) {
                die(std::string("go ") + name + ": goroutines cannot read tt.OpenBytes yet; use tt.LoadBytes");
            }
        }
        procs_.push_back({fid, name, std::move(args)});
    }

    template <typename T>
    bool recv(ChanRef c, T* out) {
        start();
        if (c.id < 0 || c.id >= static_cast<int>(chans_.size())) {
            die("receive from a nil channel");
        }
        Channel& ch = chans_[static_cast<size_t>(c.id)];
        if (ch.sender < 0) {
            die("main receives from a channel that no goroutine sends on: it would block forever");
        }
        if (ch.receiver >= 0) {
            die("main receives from a channel that goroutine " + procs_[static_cast<size_t>(ch.receiver)].name +
                " also receives from; v0.3 channels have one receiver");
        }
        if (ch.next < ch.batch.size()) {  // values read from the card earlier
            std::memcpy(out, ch.batch[ch.next++].data(), sizeof(T));
            received_++;
            return true;
        }
        IDevice* dev = ctx_.mesh->get_devices().at(0);
        const CoreCoord core = cores_[static_cast<size_t>(ch.sender)];
        const uint32_t block = region_addr_ + ch.offset;
        for (;;) {
            uint32_t words[5];  // written, pad x3, closed
            detail::ReadFromDeviceL1(dev, core, block + ch.ring,
                                     std::span<uint8_t>(reinterpret_cast<uint8_t*>(words), sizeof words));
            const uint32_t written = words[0], closed = words[4];
            if (written != ch.host_consumed) {
                // Take every value waiting in the ring with one read, then
                // give the slots back with one write.
                std::vector<uint8_t> ring(ch.ring);
                detail::ReadFromDeviceL1(dev, core, block, std::span<uint8_t>(ring));
                ch.batch.clear();
                ch.next = 0;
                for (uint32_t k = ch.host_consumed; k != written; k++) {
                    const uint8_t* slot = ring.data() + (k % ch.cap) * ch.slot;
                    ch.batch.emplace_back(slot, slot + ch.slot);
                }
                ch.host_consumed = written;
                detail::WriteToDeviceL1(dev, core, block + ch.ring + kChConsumed,
                                        std::span<const uint8_t>(reinterpret_cast<const uint8_t*>(&ch.host_consumed), 4));
                std::memcpy(out, ch.batch[ch.next++].data(), sizeof(T));
                received_++;
                return true;
            }
            if (closed) {
                return false;
            }
            if (finished()) {
                // Re-check: the last values may have landed just before the end.
                detail::ReadFromDeviceL1(dev, core, block + ch.ring,
                                         std::span<uint8_t>(reinterpret_cast<uint8_t*>(words), sizeof words));
                if (words[0] == ch.host_consumed && !words[4]) {
                    die("all goroutines are done but main is still receiving from a channel that was never "
                        "closed (the CPU version reports this as 'all goroutines are asleep - deadlock')");
                }
            }
        }
    }

    template <typename T>
    T recv_value(ChanRef c) {
        T v{};
        recv(c, &v);
        return v;
    }

    // At the end of main: waits for every goroutine.
    void wait() {
        if (procs_.empty()) {
            return;
        }
        start();
        while (!finished()) {
        }
        distributed::Finish(ctx_.mesh->mesh_command_queue());
        check_status(true);
        std::fprintf(stderr, "%s: %zu goroutines on %zu cores ran in %.2f ms; main received %lld values\n",
                     ctx_.program.c_str(), procs_.size(), procs_.size(), since(t0_) * 1e3,
                     static_cast<long long>(received_));
    }

private:
    struct Channel {
        uint32_t cap = 0, slot = 0, ring = 0, offset = 0, host_consumed = 0;
        int sender = -1, receiver = -1;
        std::vector<std::vector<uint8_t>> batch;  // values read from the ring, not yet received by main
        size_t next = 0;
    };
    struct Proc {
        int fid;
        std::string name;
        std::vector<Arg> args;
    };

    // Has the workload finished? Also checks the goroutines' status now and
    // then, so that a failed goroutine is reported instead of hanging main.
    bool finished() {
        if (distributed::EventQuery(*done_)) {
            return true;
        }
        if (since(last_check_) > 0.1) {
            check_status(false);
            last_check_ = std::chrono::steady_clock::now();
            if (since(t0_) > next_hint_) {
                std::fprintf(stderr, "%s: still running after %.0f s (on the CPU, go run reports a deadlock as "
                             "'all goroutines are asleep')\n", ctx_.program.c_str(), since(t0_));
                next_hint_ *= 2;
            }
        }
        return false;
    }

    void check_status(bool final) {
        IDevice* dev = ctx_.mesh->get_devices().at(0);
        for (size_t p = 0; p < procs_.size(); p++) {
            uint8_t st[kStatusBytes];
            detail::ReadFromDeviceL1(dev, cores_[p], region_addr_ + status_offset_, std::span<uint8_t>(st, sizeof st));
            uint32_t state;
            int64_t pos, value;
            std::memcpy(&state, st, 4);
            std::memcpy(&pos, st + 8, 8);
            std::memcpy(&value, st + 16, 8);
            if (state == 2) {
                std::cerr << "goroutine " << procs_[p].name << " (core " << p << "): ";
                fail(pos, value);
            }
            if (final && state != 1) {
                die("goroutine " + procs_[p].name + " did not finish");
            }
        }
    }

    void start() {
        if (started_) {
            return;
        }
        started_ = true;
        const size_t n = procs_.size();
        if (n == 0) {
            die("main receives from a channel but started no goroutines");
        }
        CoreCoord grid = ctx_.mesh->compute_with_storage_grid_size();
        if (n > static_cast<size_t>(grid.x) * grid.y) {
            die(std::to_string(n) + " goroutines, but the card has " + std::to_string(grid.x * grid.y) +
                " Tensix cores, one per goroutine");
        }
        CoreRangeSet set = num_cores_to_corerangeset(static_cast<uint32_t>(n), grid, true);
        cores_ = corerange_to_cores(set, static_cast<uint32_t>(n), true);

        // Which goroutine is at each end of each channel.
        for (size_t p = 0; p < n; p++) {
            for (const Arg& a : procs_[p].args) {
                if (a.kind != Arg::Send && a.kind != Arg::Recv) {
                    continue;
                }
                Channel& ch = chans_[static_cast<size_t>(a.chan)];
                int& end = a.kind == Arg::Send ? ch.sender : ch.receiver;
                if (end >= 0 && end != static_cast<int>(p)) {
                    die("goroutines " + procs_[static_cast<size_t>(end)].name + " and " + procs_[p].name + " both " +
                        (a.kind == Arg::Send ? "send on" : "receive from") +
                        " one channel; v0.3 channels connect one sender to one receiver (use a channel each)");
                }
                if (ch.sender >= 0 && ch.sender == ch.receiver) {
                    die("goroutine " + procs_[p].name + " both sends on and receives from one channel");
                }
                end = static_cast<int>(p);
            }
        }
        uint32_t offset = 0;
        for (Channel& ch : chans_) {
            if (ch.receiver >= 0 && ch.sender < 0) {
                die("goroutine " + procs_[static_cast<size_t>(ch.receiver)].name +
                    " receives from a channel that no goroutine sends on; main cannot send on channels yet");
            }
            ch.offset = offset;
            offset += (ch.ring + 15) / 16 * 16 + kChBlockExtra;
        }
        status_offset_ = offset;
        const uint32_t region_bytes = (offset + kStatusBytes + 31) / 32 * 32;

        // The region, at the same L1 address on every core.
        const uint32_t banks = ctx_.mesh->allocator()->get_num_banks(BufferType::L1);
        distributed::DeviceLocalBufferConfig local{.page_size = region_bytes, .buffer_type = BufferType::L1};
        distributed::ReplicatedBufferConfig global{.size = static_cast<uint64_t>(banks) * region_bytes};
        region_ = distributed::MeshBuffer::create(global, local, ctx_.mesh.get());
        region_addr_ = static_cast<uint32_t>(region_->address());

        program_ = CreateProgram();
        CircularBufferConfig scratch =
            CircularBufferConfig(scratch_bytes_, {{tt::CBIndex::c_0, tt::DataFormat::UInt8}})
                .set_page_size(tt::CBIndex::c_0, scratch_bytes_);
        CreateCircularBuffer(program_, set, scratch);
        std::ifstream f(ctx_.kernel_dir + "/goroutines.cpp");
        if (!f) {
            die("cannot read kernel source " + ctx_.kernel_dir + "/goroutines.cpp (pass --kernels DIR)");
        }
        std::stringstream src;
        src << f.rdbuf();
        KernelHandle kernel = CreateKernelFromString(
            program_,
            src.str(),
            set,
            DataMovementConfig{.processor = DataMovementProcessor::RISCV_0, .noc = NOC::RISCV_0_default});

        for (size_t p = 0; p < n; p++) {
            std::vector<uint32_t> a;
            a.push_back(static_cast<uint32_t>(procs_[p].fid));
            a.push_back(region_addr_ + status_offset_);
            for (const Arg& x : procs_[p].args) {
                switch (x.kind) {
                    case Arg::Int:
                        push64(a, x.value);
                        break;
                    case Arg::BytesArg:
                        push_bytes(a, *x.bytes);
                        break;
                    case Arg::Send:
                    case Arg::Recv: {
                        const Channel& ch = chans_[static_cast<size_t>(x.chan)];
                        const bool to_host = ch.receiver < 0;
                        const int other = x.kind == Arg::Send ? (to_host ? static_cast<int>(p) : ch.receiver) : ch.sender;
                        const CoreCoord peer = ctx_.mesh->worker_core_from_logical_core(cores_[static_cast<size_t>(other)]);
                        a.push_back(region_addr_ + ch.offset);
                        a.push_back(static_cast<uint32_t>(peer.x));
                        a.push_back(static_cast<uint32_t>(peer.y));
                        a.push_back(ch.cap);
                        a.push_back(ch.slot);
                        a.push_back((ch.ring + 15) / 16 * 16);
                        a.push_back(to_host ? 1 : 0);
                        break;
                    }
                }
            }
            SetRuntimeArgs(program_, kernel, cores_[p], a);
        }
        for (Channel& ch : chans_) {
            ch.ring = (ch.ring + 15) / 16 * 16;
        }
        if (ctx_.compile_only) {
            detail::CompileProgram(ctx_.mesh->get_devices().at(0), program_);
            std::fprintf(stderr, "%s: goroutine kernel compiled for %zu cores\n", ctx_.program.c_str(), n);
            ctx_.mesh->close();
            std::exit(0);
        }

        auto& cq = ctx_.mesh->mesh_command_queue();
        distributed::Finish(cq);
        IDevice* dev = ctx_.mesh->get_devices().at(0);
        const std::vector<uint8_t> zeros(region_bytes, 0);
        for (const CoreCoord& core : cores_) {
            detail::WriteToDeviceL1(dev, core, region_addr_, std::span<const uint8_t>(zeros));
        }
        workload_.add_program(distributed::MeshCoordinateRange(ctx_.mesh->shape()), std::move(program_));
        t0_ = last_check_ = std::chrono::steady_clock::now();
        distributed::EnqueueMeshWorkload(cq, workload_, false);
        done_ = cq.enqueue_record_event_to_host();
    }

    Context& ctx_;
    uint32_t scratch_bytes_;
    std::vector<Channel> chans_;
    std::vector<Proc> procs_;
    std::vector<CoreCoord> cores_;
    bool started_ = false;
    std::shared_ptr<distributed::MeshBuffer> region_;
    uint32_t region_addr_ = 0, status_offset_ = 0;
    Program program_;
    distributed::MeshWorkload workload_;
    std::optional<distributed::MeshEvent> done_;
    std::chrono::steady_clock::time_point t0_, last_check_;
    double next_hint_ = 10;
    int64_t received_ = 0;
};

}  // namespace tsr
// ---------------------------------------------------------------------------
