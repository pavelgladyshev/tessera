// ---------------------------------------------------------------------------
// Goroutines and channels on the host side (pasted in by tessc): main makes
// channels and starts goroutines; the network starts on the card the first
// time main receives, or when main ends; main receives from channels whose
// sender is a goroutine.
//
// Goroutines fill the first chip's cores, then the second's. A channel
// between goroutines on different chips cannot use the NoC, which ends at
// the chip's edge. It goes over tt-metal's fabric instead, the Ethernet
// links between the chips: each of its two cores gets a connection to a
// fabric router, and sends and credits travel as fabric packets. A router
// takes one core's connection, and tt-metal's dispatch holds one link, so on
// an n300 one core per chip can connect (it may carry several channels). A
// channel whose cores cannot both connect is relayed by the host: the
// sender treats it as a channel to main (its ring stays on its own core),
// the receiver as an ordinary channel whose ring and counters the host
// fills, and the host moves the values across while main waits or receives.
// TESSERA_FABRIC=0 relays them all.
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
        IDevice* dev = devs_[static_cast<size_t>(ch.sender)];
        const CoreCoord core = cores_[static_cast<size_t>(ch.sender)];
        const uint32_t block = region_addr_ + ch.offset;
        for (;;) {
            relay();
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
                // The sender sets written before closed, but one read of both
                // can catch them apart (on the second chip it goes over
                // Ethernet): look at written again before taking "closed"
                // to mean that nothing is left.
                if (written_of(dev, core, block, ch) != ch.host_consumed) {
                    continue;
                }
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
            relay();
        }
        distributed::Finish(ctx_.mesh->mesh_command_queue());
        check_status(true);
        std::fprintf(stderr, "%s: %zu goroutines on %zu cores of %zu chip%s ran in %.2f ms; main received %lld values",
                     ctx_.program.c_str(), procs_.size(), procs_.size(), chips_, chips_ > 1 ? "s" : "",
                     since(t0_) * 1e3, static_cast<long long>(received_));
        size_t over_fabric = 0;
        for (const Channel& ch : chans_) {
            over_fabric += ch.fabric;
        }
        if (over_fabric > 0) {
            std::fprintf(stderr, "; %zu channel%s between the chips over the fabric", over_fabric, over_fabric > 1 ? "s" : "");
        }
        if (relayed_ > 0) {
            std::fprintf(stderr, "; the host relayed %lld values between the chips", static_cast<long long>(relayed_));
        }
        std::fprintf(stderr, "\n");
    }

private:
    struct Channel {
        uint32_t cap = 0, slot = 0, ring = 0, offset = 0, host_consumed = 0;
        int sender = -1, receiver = -1;
        bool fabric = false;   // between chips, over the fabric
        bool relayed = false;  // between chips, moved by the host
        uint32_t delivered = 0;
        bool closed_sent = false;
        std::vector<std::vector<uint8_t>> batch;  // values read from the ring, not yet received by main
        size_t next = 0;
    };
    struct Proc {
        int fid;
        std::string name;
        std::vector<Arg> args;
    };

    // The written counter of channel ch on the core that holds its ring.
    uint32_t written_of(IDevice* dev, const CoreCoord& core, uint32_t block, const Channel& ch) {
        uint32_t w;
        detail::ReadFromDeviceL1(dev, core, block + ch.ring + kChWritten, std::span<uint8_t>(reinterpret_cast<uint8_t*>(&w), 4));
        return w;
    }

    // The fabric's name for the chip at (0, c) of the mesh.
    tt::tt_fabric::FabricNodeId node(size_t c) {
        return tt::tt_fabric::get_fabric_node_id_from_physical_chip_id(device(ctx_, static_cast<int>(c))->id());
    }

    // Moves what senders have put in the channels between chips to their
    // receivers, as far as the receivers' rings have room, and frees the
    // senders' slots; passes a close on once everything sent has arrived.
    void relay() {
        for (Channel& ch : chans_) {
            if (!ch.relayed || ch.closed_sent) {
                continue;
            }
            IDevice* from = devs_[static_cast<size_t>(ch.sender)];
            IDevice* to = devs_[static_cast<size_t>(ch.receiver)];
            const CoreCoord sc = cores_[static_cast<size_t>(ch.sender)], rc = cores_[static_cast<size_t>(ch.receiver)];
            const uint32_t block = region_addr_ + ch.offset;
            uint32_t words[5];  // the sender's written, pad x3, closed
            detail::ReadFromDeviceL1(from, sc, block + ch.ring,
                                     std::span<uint8_t>(reinterpret_cast<uint8_t*>(words), sizeof words));
            const uint32_t written = words[0], closed = words[4];
            if (written != ch.delivered) {
                uint32_t consumed;  // what the receiver has taken, counted in its own block
                detail::ReadFromDeviceL1(to, rc, block + ch.ring + kChConsumed,
                                         std::span<uint8_t>(reinterpret_cast<uint8_t*>(&consumed), 4));
                const uint32_t room = ch.cap - (ch.delivered - consumed);
                const uint32_t move = std::min(room, written - ch.delivered);
                if (move > 0) {
                    std::vector<uint8_t> ring(ch.ring);
                    detail::ReadFromDeviceL1(from, sc, block, std::span<uint8_t>(ring));
                    // The slots in one write, or two where they wrap around
                    // the ring's end: each write to the other chip is a round
                    // trip over Ethernet.
                    for (uint32_t k = ch.delivered; k != ch.delivered + move;) {
                        const uint32_t first = k % ch.cap;
                        const uint32_t run = std::min(ch.cap - first, ch.delivered + move - k);
                        detail::WriteToDeviceL1(to, rc, block + first * ch.slot,
                                                std::span<const uint8_t>(ring.data() + first * ch.slot, run * ch.slot));
                        k += run;
                    }
                    ch.delivered += move;
                    relayed_ += move;
                    // The values first, then the count that lets the receiver take them.
                    detail::WriteToDeviceL1(to, rc, block + ch.ring + kChWritten,
                                            std::span<const uint8_t>(reinterpret_cast<const uint8_t*>(&ch.delivered), 4));
                    detail::WriteToDeviceL1(from, sc, block + ch.ring + kChConsumed,
                                            std::span<const uint8_t>(reinterpret_cast<const uint8_t*>(&ch.delivered), 4));
                }
            }
            if (closed && ch.delivered == written && written_of(from, sc, block, ch) == written) {  // see recv
                const uint32_t one = 1;
                detail::WriteToDeviceL1(to, rc, block + ch.ring + kChClosed,
                                        std::span<const uint8_t>(reinterpret_cast<const uint8_t*>(&one), 4));
                ch.closed_sent = true;
            }
        }
    }

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
        for (size_t p = 0; p < procs_.size(); p++) {
            uint8_t st[kStatusBytes];
            detail::ReadFromDeviceL1(devs_[p], cores_[p], region_addr_ + status_offset_, std::span<uint8_t>(st, sizeof st));
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
        const size_t per_chip = std::min<size_t>(static_cast<size_t>(grid.x) * grid.y, 64);
        const size_t nchips = ctx_.mesh->shape()[1];
        if (n > per_chip * nchips) {
            die(std::to_string(n) + " goroutines, but the card has " + std::to_string(per_chip * nchips) +
                " Tensix cores, one per goroutine");
        }
        // Goroutines fill the first chip, then the next.
        std::vector<CoreRangeSet> sets;
        for (size_t first = 0; first < n; first += per_chip) {
            const size_t count = std::min(per_chip, n - first);
            sets.push_back(num_cores_to_corerangeset(static_cast<uint32_t>(count), grid, true));
            IDevice* dev = device(ctx_, static_cast<int>(sets.size() - 1));
            for (const CoreCoord& c : corerange_to_cores(sets.back(), static_cast<uint32_t>(count), true)) {
                cores_.push_back(c);
                devs_.push_back(dev);
                chip_.push_back(static_cast<int>(sets.size() - 1));
            }
        }
        chips_ = sets.size();

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
        // Channels between chips: over the fabric while its links last, then
        // relayed by the host. A link's router takes one connection from a
        // Tensix core, and tt-metal's dispatch to the other chip holds the
        // last link's (RelayMux::get_dispatch_link_index), so a core that
        // connects there hangs. An n300 has two links: one is left.
        std::vector<int> link(n, -1);  // each core's fabric link to the other chip, or -1
        std::vector<std::vector<uint32_t>> free_links(chips_);
        if (fabric_for_programs && chips_ == 2) {
            for (size_t c = 0; c < 2; c++) {
                free_links[c] = tt::tt_fabric::get_forwarding_link_indices(node(c), node(1 - c));
                if (!free_links[c].empty()) {
                    free_links[c].pop_back();
                }
            }
        }
        auto connect = [&](int p) {  // gives core p a link if it has none; false if none is left
            if (link[static_cast<size_t>(p)] >= 0) {
                return true;
            }
            auto& left = free_links[static_cast<size_t>(chip_[static_cast<size_t>(p)])];
            if (left.empty()) {
                return false;
            }
            link[static_cast<size_t>(p)] = static_cast<int>(left.front());
            left.erase(left.begin());
            return true;
        };
        for (Channel& ch : chans_) {
            if (ch.sender < 0 || ch.receiver < 0 ||
                chip_[static_cast<size_t>(ch.sender)] == chip_[static_cast<size_t>(ch.receiver)]) {
                continue;
            }
            const bool had_s = link[static_cast<size_t>(ch.sender)] >= 0, had_r = link[static_cast<size_t>(ch.receiver)] >= 0;
            if (connect(ch.sender) && connect(ch.receiver)) {
                ch.fabric = true;
                fabric_used_ = true;
                continue;
            }
            // Give back what this channel took, and relay it.
            for (auto [p, had] : {std::pair{ch.sender, had_s}, std::pair{ch.receiver, had_r}}) {
                if (!had && link[static_cast<size_t>(p)] >= 0) {
                    free_links[static_cast<size_t>(chip_[static_cast<size_t>(p)])].insert(
                        free_links[static_cast<size_t>(chip_[static_cast<size_t>(p)])].begin(),
                        static_cast<uint32_t>(link[static_cast<size_t>(p)]));
                    link[static_cast<size_t>(p)] = -1;
                }
            }
            ch.relayed = true;
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

        std::ifstream f(ctx_.kernel_dir + "/goroutines.cpp");
        if (!f) {
            die("cannot read kernel source " + ctx_.kernel_dir + "/goroutines.cpp (pass --kernels DIR)");
        }
        std::stringstream src;
        src << f.rdbuf();
        // A program per chip.
        std::vector<Program> programs;
        std::vector<KernelHandle> kernels;
        std::map<std::string, std::string> defines;
        if (fabric_used_) {
            defines["TSR_FABRIC"] = "1";
        }
        for (const CoreRangeSet& set : sets) {
            programs.push_back(CreateProgram());
            CircularBufferConfig scratch =
                CircularBufferConfig(scratch_bytes_, {{tt::CBIndex::c_0, tt::DataFormat::UInt8}})
                    .set_page_size(tt::CBIndex::c_0, scratch_bytes_);
            CreateCircularBuffer(programs.back(), set, scratch);
            kernels.push_back(CreateKernelFromString(
                programs.back(),
                src.str(),
                set,
                DataMovementConfig{
                    .processor = DataMovementProcessor::RISCV_0, .noc = NOC::RISCV_0_default, .defines = defines}));
        }
        for (size_t p = 0; p < n; p++) {
            std::vector<uint32_t> a;
            a.push_back(static_cast<uint32_t>(procs_[p].fid));
            a.push_back(region_addr_ + status_offset_);
            a.push_back(0);  // where the fabric connection's arguments start, set below
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
                        // A channel to main, or the sending end of one the host
                        // relays, keeps its ring on the sender's core; the
                        // receiving end of a relayed one counts what it takes in
                        // its own block. Either way the "other end" is itself.
                        const bool to_host = ch.receiver < 0 || (ch.relayed && x.kind == Arg::Send);
                        int other = x.kind == Arg::Send ? ch.receiver : ch.sender;
                        if (to_host || ch.relayed) {
                            other = static_cast<int>(p);
                        }
                        const uint32_t via = to_host ? 1 : (ch.fabric ? 2 : 0);  // tsr::kToHost, kFabric, kNoc
                        const CoreCoord peer = devs_[static_cast<size_t>(other)]->worker_core_from_logical_core(
                            cores_[static_cast<size_t>(other)]);
                        a.push_back(region_addr_ + ch.offset);
                        a.push_back(static_cast<uint32_t>(peer.x));
                        a.push_back(static_cast<uint32_t>(peer.y));
                        a.push_back(ch.cap);
                        a.push_back(ch.slot);
                        a.push_back((ch.ring + 15) / 16 * 16);
                        a.push_back(via);
                        break;
                    }
                }
            }
            if (link[p] >= 0) {
                const size_t c = static_cast<size_t>(chip_[p]);
                a[2] = static_cast<uint32_t>(a.size());
                tt::tt_fabric::append_fabric_connection_rt_args(node(c), node(1 - c), static_cast<uint32_t>(link[p]),
                                                                programs[c], cores_[p], a);
            }
            SetRuntimeArgs(programs[static_cast<size_t>(chip_[p])], kernels[static_cast<size_t>(chip_[p])], cores_[p], a);
        }
        for (Channel& ch : chans_) {
            ch.ring = (ch.ring + 15) / 16 * 16;
        }
        if (ctx_.compile_only) {
            for (size_t c = 0; c < programs.size(); c++) {
                detail::CompileProgram(device(ctx_, static_cast<int>(c)), programs[c]);
            }
            std::fprintf(stderr, "%s: goroutine kernel compiled for %zu cores\n", ctx_.program.c_str(), n);
            ctx_.mesh->close();
            std::exit(0);
        }

        auto& cq = ctx_.mesh->mesh_command_queue();
        distributed::Finish(cq);
        const std::vector<uint8_t> zeros(region_bytes, 0);
        for (size_t p = 0; p < n; p++) {
            detail::WriteToDeviceL1(devs_[p], cores_[p], region_addr_, std::span<const uint8_t>(zeros));
        }
        for (size_t c = 0; c < programs.size(); c++) {
            const distributed::MeshCoordinate at(0, static_cast<uint32_t>(c));
            workload_.add_program(distributed::MeshCoordinateRange(at, at), std::move(programs[c]));
        }
        t0_ = last_check_ = std::chrono::steady_clock::now();
        distributed::EnqueueMeshWorkload(cq, workload_, false);
        done_ = cq.enqueue_record_event_to_host();
    }

    Context& ctx_;
    uint32_t scratch_bytes_;
    std::vector<Channel> chans_;
    std::vector<Proc> procs_;
    std::vector<CoreCoord> cores_;  // each goroutine's core,
    std::vector<IDevice*> devs_;    // its chip,
    std::vector<int> chip_;         // and that chip's place in the mesh
    size_t chips_ = 0;
    int64_t relayed_ = 0;
    bool fabric_used_ = false;
    bool started_ = false;
    std::shared_ptr<distributed::MeshBuffer> region_;
    uint32_t region_addr_ = 0, status_offset_ = 0;
    distributed::MeshWorkload workload_;
    std::optional<distributed::MeshEvent> done_;
    std::chrono::steady_clock::time_point t0_, last_check_;
    double next_hint_ = 10;
    int64_t received_ = 0;
};

}  // namespace tsr
// ---------------------------------------------------------------------------
