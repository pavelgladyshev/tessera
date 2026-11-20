// ---------------------------------------------------------------------------
// Goroutines and channels on the host side (pasted in by tessc): main makes
// channels and starts goroutines; the network starts on the card the first
// time main receives, or when main ends; main receives from channels whose
// sender is a goroutine.
//
// Goroutines fill the first chip's cores, then the second's. A channel
// between goroutines on different chips cannot use the NoC, which ends at
// the chip's edge. It goes over tt-metal's fabric instead, the Ethernet
// links between the chips, and sends and credits travel as fabric packets.
// A router takes one core's connection, and tt-metal's dispatch holds one
// of the n300's two links, so goroutines leave each chip's last core to
// tt-metal's fabric mux, which forwards the packets of every core that
// needs the fabric to the router. A program that needs every core connects
// one core per chip to the router directly. A channel that cannot go over
// the fabric is relayed by the host, with a warning: the sender treats it
// as a channel to main (its ring stays on its own core), the receiver as an
// ordinary channel whose ring and counters the host fills, and the host
// moves the values across while main waits or receives. TESSERA_FABRIC=0
// relays them all.
// ---------------------------------------------------------------------------
namespace tsr {

struct ChanRef {
    int id = -1;
};

struct Arg {
    enum Kind { Int, Send, Recv, BytesArg, Sends, Recvs } kind;
    int64_t value;
    int chan;
    const Bytes* bytes;
    std::vector<int> chans;  // Sends, Recvs: an array or slice of channels
};
inline Arg arg_int(int64_t v) { return {Arg::Int, v, -1, nullptr, {}}; }
inline Arg arg_send(ChanRef c) { return {Arg::Send, 0, c.id, nullptr, {}}; }
inline Arg arg_recv(ChanRef c) { return {Arg::Recv, 0, c.id, nullptr, {}}; }
inline Arg arg_bytes(const Bytes& b) { return {Arg::BytesArg, 0, -1, &b, {}}; }
inline Arg arg_chans(Arg::Kind k, const ChanRef* first, const ChanRef* end) {
    Arg a{k, 0, -1, nullptr, {}};
    for (const ChanRef* c = first; c != end; c++) {
        a.chans.push_back(c->id);
    }
    return a;
}
template <size_t N>
inline Arg arg_sends(const ChanRef (&cs)[N]) { return arg_chans(Arg::Sends, cs, cs + N); }
template <size_t N>
inline Arg arg_recvs(const ChanRef (&cs)[N]) { return arg_chans(Arg::Recvs, cs, cs + N); }
inline Arg arg_sends(const std::vector<ChanRef>& cs) { return arg_chans(Arg::Sends, cs.data(), cs.data() + cs.size()); }
inline Arg arg_recvs(const std::vector<ChanRef>& cs) { return arg_chans(Arg::Recvs, cs.data(), cs.data() + cs.size()); }

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
        if (capacity > 65535) {
            die("make(chan T, " + std::to_string(capacity) + "): a channel on the card holds at most 65535 values");
        }
        Channel ch;
        ch.cap = static_cast<uint32_t>(capacity);
        ch.slot = (elem_bytes + 15) / 16 * 16;
        if (ch.slot > 65535) {
            die("a channel's values are at most 65535 bytes on the card");
        }
        ch.ring = ch.cap * ch.slot;
        chans_.push_back(ch);
        return ChanRef{static_cast<int>(chans_.size()) - 1};
    }

    // go f(args) in main; chip is the one a //tt:chip directive pins it to,
    // or -1 for anywhere.
    void go(int fid, const char* name, std::vector<Arg> args, int chip = -1) {
        if (started_) {
            die(std::string("go ") + name +
                ": start every goroutine before main first receives, selects, or sends on a full channel");
        }
        for (const Arg& a : args) {
            if ((a.kind == Arg::Send || a.kind == Arg::Recv) && (a.chan < 0 || a.chan >= static_cast<int>(chans_.size()))) {
                die(std::string("go ") + name + ": a channel argument was never made (nil channel)");
            }
            for (int c : a.chans) {
                if (c < 0 || c >= static_cast<int>(chans_.size())) {
                    die(std::string("go ") + name + ": a channel in an array or slice argument was never made (nil channel)");
                }
            }
            if (a.kind == Arg::BytesArg && a.bytes->from_host) {
                die(std::string("go ") + name + ": goroutines cannot read tt.OpenBytes yet; use tt.LoadBytes");
            }
        }
        procs_.push_back({fid, name, std::move(args), chip});
    }

    // ch <- v in main. Before the goroutines start, values wait on the host,
    // as many as the channel holds; a send on a full channel starts them.
    template <typename T>
    void send(ChanRef c, const T& v) {
        Channel& ch = sent_by_main(c, "send on");
        if (ch.main_closed) {
            die("send on closed channel");
        }
        std::vector<uint8_t> slot(ch.slot, 0);
        std::memcpy(slot.data(), &v, sizeof(T));
        if (!started_ && ch.pending.size() < ch.cap) {
            ch.pending.push_back(std::move(slot));
            return;
        }
        start();
        for (bool last = false; !has_room(ch);) {
            if (last) {
                die("all goroutines are done but main is still sending on a full channel (the CPU version reports "
                    "this as 'all goroutines are asleep - deadlock')");
            }
            relay();
            last = finished();
        }
        put(ch, slot);
    }

    // close(ch) in main.
    void close(ChanRef c) {
        Channel& ch = sent_by_main(c, "close");
        if (ch.main_closed) {
            die("close of closed channel");
        }
        ch.main_closed = true;
        if (started_) {
            put_closed(ch);
        }
    }

    // For select in main: whether a receive or a send would proceed now.
    bool recv_ready(ChanRef c) {
        if (c.id < 0) {
            return false;  // a nil channel
        }
        Channel& ch = main_receives(c);
        if (ch.next < ch.batch.size()) {
            return true;
        }
        uint32_t words[5];  // written, pad x3, closed
        read_l1(devs_[static_cast<size_t>(ch.sender)], cores_[static_cast<size_t>(ch.sender)],
                                 region_addr_ + ch.offset + ch.ring,
                                 std::span<uint8_t>(reinterpret_cast<uint8_t*>(words), sizeof words));
        return words[0] != ch.host_consumed || words[4] != 0;
    }
    bool send_ready(ChanRef c) {
        if (c.id < 0) {
            return false;
        }
        Channel& ch = sent_by_main(c, "send on");
        start();
        return ch.main_closed || has_room(ch);
    }
    uint32_t select_turn() {
        start();
        return select_turn_++;
    }
    // Between rounds of a select in main that found nothing ready: moves the
    // relayed channels on, and says once every goroutine is done (then one
    // more round decides).
    bool select_idle() {
        relay();
        return finished();
    }
    [[noreturn]] void select_deadlock() {
        die("all goroutines are done but main's select can never proceed (the CPU version reports this as 'all "
            "goroutines are asleep - deadlock')");
    }
    [[noreturn]] void block_forever() {  // select {} in main
        start();
        while (!finished()) {
            relay();
        }
        die("all goroutines are done and main blocks for ever in select {} (the CPU version reports this as 'all "
            "goroutines are asleep - deadlock')");
    }

    template <typename T>
    bool recv(ChanRef c, T* out) {
        start();
        if (c.id < 0) {
            die("main receives from a nil channel: it would block forever");
        }
        Channel& ch = main_receives(c);
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
            read_l1(dev, core, block + ch.ring,
                                     std::span<uint8_t>(reinterpret_cast<uint8_t*>(words), sizeof words));
            const uint32_t written = words[0], closed = words[4];
            if (written != ch.host_consumed) {
                // Take every value waiting in the ring with one read, then
                // give the slots back with one write.
                std::vector<uint8_t> ring(ch.ring);
                read_l1(dev, core, block, std::span<uint8_t>(ring));
                ch.batch.clear();
                ch.next = 0;
                for (uint32_t k = ch.host_consumed; k != written; k++) {
                    const uint8_t* slot = ring.data() + (k % ch.cap) * ch.slot;
                    ch.batch.emplace_back(slot, slot + ch.slot);
                }
                ch.host_consumed = written;
                write_l1(dev, core, block + ch.ring + kChConsumed,
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
                read_l1(dev, core, block + ch.ring,
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
        std::string where;  // how many on each chip, when there are two or more
        for (size_t c = 0; chips_ > 1 && c < chips_; c++) {
            where += (c == 0 ? " (" : ", ") + std::to_string(on_chip_[c]) + " on chip " + std::to_string(c) +
                     (c + 1 == chips_ ? ")" : "");
        }
        std::fprintf(stderr, "%s: %zu goroutines on %zu cores of %zu chip%s%s ran in %.2f ms; main received %lld values",
                     ctx_.program.c_str(), procs_.size(), procs_.size(), chips_, chips_ > 1 ? "s" : "", where.c_str(),
                     since(t0_) * 1e3, static_cast<long long>(received_));
        size_t over_fabric = 0;
        for (const Channel& ch : chans_) {
            over_fabric += ch.fabric;
        }
        if (over_fabric > 0) {
            std::fprintf(stderr, "; %zu channel%s between the chips over the fabric%s", over_fabric, over_fabric > 1 ? "s" : "",
                         mux_ ? " (through a fabric mux on each chip)" : "");
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
        // Sent by main: values sent before the goroutines started, how many
        // main has sent and how many of those the receiver has taken (as last
        // read), and whether main closed it.
        std::vector<std::vector<uint8_t>> pending;
        uint32_t main_sent = 0, main_credit = 0;
        bool main_sends = false, main_closed = false;
        std::vector<std::vector<uint8_t>> batch;  // values read from the ring, not yet received by main
        size_t next = 0;
    };
    struct Proc {
        int fid;
        std::string name;
        std::vector<Arg> args;
        int chip;  // -1: anywhere
    };

    // The written counter of channel ch on the core that holds its ring.
    uint32_t written_of(IDevice* dev, const CoreCoord& core, uint32_t block, const Channel& ch) {
        uint32_t w;
        read_l1(dev, core, block + ch.ring + kChWritten, std::span<uint8_t>(reinterpret_cast<uint8_t*>(&w), 4));
        return w;
    }

    // The channel main receives from, checked.
    Channel& main_receives(ChanRef c) {
        start();
        if (c.id >= static_cast<int>(chans_.size())) {
            die("receive from a channel that was never made");
        }
        Channel& ch = chans_[static_cast<size_t>(c.id)];
        if (ch.main_sends) {
            die("main both sends on and receives from one channel");
        }
        if (ch.sender < 0) {
            die("main receives from a channel that no goroutine sends on: it would block forever");
        }
        if (ch.receiver >= 0) {
            die("main receives from a channel that goroutine " + procs_[static_cast<size_t>(ch.receiver)].name +
                " also receives from; a channel has one receiver on the card");
        }
        return ch;
    }

    // The channel main sends on (or closes), checked as far as it can be yet.
    Channel& sent_by_main(ChanRef c, const char* what) {
        if (c.id < 0) {
            die(std::string("main ") + what + " a nil channel" +
                (std::string(what) == "close" ? "" : ": it would block forever"));
        }
        if (c.id >= static_cast<int>(chans_.size())) {
            die("a channel that was never made");
        }
        Channel& ch = chans_[static_cast<size_t>(c.id)];
        ch.main_sends = true;
        if (started_) {
            check_main_sends(ch);
        }
        return ch;
    }
    void check_main_sends(const Channel& ch) {
        if (ch.sender >= 0) {
            die("main sends on (or closes) a channel that goroutine " + procs_[static_cast<size_t>(ch.sender)].name +
                " sends on; a channel has one sender on the card");
        }
        if (ch.receiver < 0) {
            die("main sends on a channel that no goroutine receives from: it would block forever");
        }
    }

    // Whether the receiver of a channel main sends on has a free slot.
    bool has_room(Channel& ch) {
        if (ch.main_sent - ch.main_credit < ch.cap) {
            return true;
        }
        uint32_t consumed;
        read_l1(devs_[static_cast<size_t>(ch.receiver)], cores_[static_cast<size_t>(ch.receiver)],
                                 region_addr_ + ch.offset + ch.ring + kChConsumed,
                                 std::span<uint8_t>(reinterpret_cast<uint8_t*>(&consumed), 4));
        ch.main_credit = consumed;
        return ch.main_sent - ch.main_credit < ch.cap;
    }

    // Writes a value main sends into its slot in the receiver's ring, then the
    // count that lets the receiver take it.
    void put(Channel& ch, const std::vector<uint8_t>& slot) {
        IDevice* dev = devs_[static_cast<size_t>(ch.receiver)];
        const CoreCoord core = cores_[static_cast<size_t>(ch.receiver)];
        const uint32_t block = region_addr_ + ch.offset;
        write_l1(dev, core, block + (ch.main_sent % ch.cap) * ch.slot, std::span<const uint8_t>(slot));
        ch.main_sent++;
        write_l1(dev, core, block + ch.ring + kChWritten,
                                std::span<const uint8_t>(reinterpret_cast<const uint8_t*>(&ch.main_sent), 4));
    }
    void put_closed(Channel& ch) {
        const uint32_t one = 1;
        write_l1(devs_[static_cast<size_t>(ch.receiver)], cores_[static_cast<size_t>(ch.receiver)],
                                region_addr_ + ch.offset + ch.ring + kChClosed,
                                std::span<const uint8_t>(reinterpret_cast<const uint8_t*>(&one), 4));
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
            read_l1(from, sc, block + ch.ring,
                                     std::span<uint8_t>(reinterpret_cast<uint8_t*>(words), sizeof words));
            const uint32_t written = words[0], closed = words[4];
            if (written != ch.delivered) {
                uint32_t consumed;  // what the receiver has taken, counted in its own block
                read_l1(to, rc, block + ch.ring + kChConsumed,
                                         std::span<uint8_t>(reinterpret_cast<uint8_t*>(&consumed), 4));
                const uint32_t room = ch.cap - (ch.delivered - consumed);
                const uint32_t move = std::min(room, written - ch.delivered);
                if (move > 0) {
                    std::vector<uint8_t> ring(ch.ring);
                    read_l1(from, sc, block, std::span<uint8_t>(ring));
                    // The slots in one write, or two where they wrap around
                    // the ring's end: each write to the other chip is a round
                    // trip over Ethernet.
                    for (uint32_t k = ch.delivered; k != ch.delivered + move;) {
                        const uint32_t first = k % ch.cap;
                        const uint32_t run = std::min(ch.cap - first, ch.delivered + move - k);
                        write_l1(to, rc, block + first * ch.slot,
                                                std::span<const uint8_t>(ring.data() + first * ch.slot, run * ch.slot));
                        k += run;
                    }
                    ch.delivered += move;
                    relayed_ += move;
                    // The values first, then the count that lets the receiver take them.
                    write_l1(to, rc, block + ch.ring + kChWritten,
                                            std::span<const uint8_t>(reinterpret_cast<const uint8_t*>(&ch.delivered), 4));
                    write_l1(from, sc, block + ch.ring + kChConsumed,
                                            std::span<const uint8_t>(reinterpret_cast<const uint8_t*>(&ch.delivered), 4));
                }
            }
            if (closed && ch.delivered == written && written_of(from, sc, block, ch) == written) {  // see recv
                const uint32_t one = 1;
                write_l1(to, rc, block + ch.ring + kChClosed,
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
            read_l1(devs_[p], cores_[p], region_addr_ + status_offset_, std::span<uint8_t>(st, sizeof st));
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
        std::vector<size_t> pinned(nchips, 0);
        for (const Proc& pr : procs_) {
            if (pr.chip < 0) {
                continue;
            }
            if (static_cast<size_t>(pr.chip) >= nchips) {
                die("go " + pr.name + ": //tt:chip " + std::to_string(pr.chip) + ", but the program has " +
                    std::to_string(nchips) + " chip" + (nchips > 1 ? "s" : "") +
                    " of the card open (chips are numbered from 0; TESSERA_CHIPS and TESSERA_DEVICE limit them)");
            }
            if (++pinned[static_cast<size_t>(pr.chip)] > per_chip) {
                die("//tt:chip " + std::to_string(pr.chip) + " pins more goroutines than the chip's " +
                    std::to_string(per_chip) + " Tensix cores");
            }
        }
        // Which goroutine is at each end of each channel.
        for (size_t p = 0; p < n; p++) {
            for (const Arg& a : procs_[p].args) {
                const bool sends = a.kind == Arg::Send || a.kind == Arg::Sends;
                std::vector<int> ids = a.chans;
                if (a.kind == Arg::Send || a.kind == Arg::Recv) {
                    ids.push_back(a.chan);
                } else if (a.kind != Arg::Sends && a.kind != Arg::Recvs) {
                    continue;
                }
                for (int id : ids) {
                    Channel& ch = chans_[static_cast<size_t>(id)];
                    int& end = sends ? ch.sender : ch.receiver;
                    if (end >= 0 && end != static_cast<int>(p)) {
                        die("goroutines " + procs_[static_cast<size_t>(end)].name + " and " + procs_[p].name + " both " +
                            (sends ? "send on" : "receive from") +
                            " one channel; a channel connects one sender to one receiver on the card (use a channel each)");
                    }
                    if (ch.sender >= 0 && ch.sender == ch.receiver) {
                        die("goroutine " + procs_[p].name + " both sends on and receives from one channel");
                    }
                    end = static_cast<int>(p);
                }
            }
        }
        // A channel that a goroutine receives from and none sends on is main's
        // to send on, now or later.
        for (Channel& ch : chans_) {
            if (ch.main_sends) {
                check_main_sends(ch);
            }
            if (ch.receiver >= 0 && ch.sender < 0) {
                ch.main_sends = true;
            }
        }
        // Channels between the chips go over the fabric where they can. A
        // router takes one connection from a Tensix core, and tt-metal's
        // dispatch to the other chip holds the last link
        // (RelayMux::get_dispatch_link_index): a core that connects there
        // hangs. On an n300 that leaves one link. So a program on both chips
        // leaves each chip's last core free if it can, and if any of its
        // channels cross, that core runs tt-metal's fabric mux, which takes
        // a connection from every core at an end of such a channel to the
        // one router. A program that needs every core connects one core per
        // chip to the router directly, and the host relays its other
        // channels between the chips, as it does any channel whose values
        // are bigger than a fabric packet's payload.
        const size_t max_payload = tt::tt_fabric::get_tt_fabric_max_payload_size_bytes();
        // Each goroutine's chip, with cap cores per chip: a goroutine that
        // main pinned to a chip (//tt:chip) is there, and the others go to
        // the first chip with a free core, in the order main started them.
        // Empty if they do not fit.
        auto place = [&](size_t cap) {
            std::vector<int> at(n, -1);
            std::vector<size_t> used(nchips, 0);
            for (size_t p = 0; p < n; p++) {
                if (const int c = procs_[p].chip; c >= 0) {
                    if (used[static_cast<size_t>(c)]++ == cap) {
                        return std::vector<int>{};
                    }
                    at[p] = c;
                }
            }
            size_t c = 0;
            for (size_t p = 0; p < n; p++) {
                if (at[p] >= 0) {
                    continue;
                }
                while (c < nchips && used[c] == cap) {
                    c++;
                }
                if (c == nchips) {
                    return std::vector<int>{};
                }
                used[c]++;
                at[p] = static_cast<int>(c);
            }
            return at;
        };
        auto crosses = [&](const std::vector<int>& at) {  // whether a channel the fabric can carry crosses chips
            for (const Channel& ch : chans_) {
                if (ch.sender >= 0 && ch.receiver >= 0 && ch.slot <= max_payload &&
                    at[static_cast<size_t>(ch.sender)] != at[static_cast<size_t>(ch.receiver)]) {
                    return true;
                }
            }
            return false;
        };
        std::vector<int> at = place(per_chip);
        if (fabric_for_programs && nchips == 2 && std::find(at.begin(), at.end(), 1) != at.end()) {
            if (std::vector<int> spare = place(per_chip - 1); !spare.empty()) {
                at = spare;
                mux_ = crosses(at);
            }
        }
        // The first cores of each chip, in the order of the goroutines there.
        chips_ = static_cast<size_t>(*std::max_element(at.begin(), at.end())) + 1;
        std::vector<CoreRangeSet> sets(chips_);
        std::vector<std::vector<CoreCoord>> free(chips_);
        on_chip_.assign(chips_, 0);
        for (size_t p = 0; p < n; p++) {
            on_chip_[static_cast<size_t>(at[p])]++;
        }
        for (size_t c = 0; c < chips_; c++) {
            if (on_chip_[c] > 0) {
                sets[c] = num_cores_to_corerangeset(static_cast<uint32_t>(on_chip_[c]), grid, true);
                free[c] = corerange_to_cores(sets[c], static_cast<uint32_t>(on_chip_[c]), true);
                std::reverse(free[c].begin(), free[c].end());
            }
        }
        for (size_t p = 0; p < n; p++) {
            const size_t c = static_cast<size_t>(at[p]);
            cores_.push_back(free[c].back());
            free[c].pop_back();
            devs_.push_back(device(ctx_, static_cast<int>(c)));
            chip_.push_back(static_cast<int>(c));
        }

        // Each core's link to the other chip, or (with the mux) its channel of
        // its chip's mux; -1: none.
        std::vector<int> link(n, -1);
        std::vector<std::vector<uint32_t>> free_links(chips_);
        if (fabric_for_programs && chips_ == 2 && !mux_) {
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
            if (mux_ && ch.slot <= max_payload) {
                for (int p : {ch.sender, ch.receiver}) {
                    if (link[static_cast<size_t>(p)] < 0) {
                        link[static_cast<size_t>(p)] = mux_clients_[static_cast<size_t>(chip_[static_cast<size_t>(p)])]++;
                    }
                }
                ch.fabric = true;
                fabric_used_ = true;
                continue;
            }
            const bool had_s = link[static_cast<size_t>(ch.sender)] >= 0, had_r = link[static_cast<size_t>(ch.receiver)] >= 0;
            if (!mux_ && ch.slot <= max_payload && connect(ch.sender) && connect(ch.receiver)) {
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
        if (fabric_for_programs && chips_ == 2) {
            size_t crowded = 0, big = 0;
            for (const Channel& ch : chans_) {
                if (ch.relayed) {
                    (ch.slot > max_payload ? big : crowded)++;
                }
            }
            if (crowded > 0) {
                std::string why = std::to_string(n) + " goroutines leave no core free for the fabric mux (at most " +
                                  std::to_string(2 * (per_chip - 1)) + " would)";
                for (size_t c = 0; c < nchips; c++) {
                    if (pinned[c] == per_chip) {
                        why = "//tt:chip " + std::to_string(c) + " pins a goroutine to every core of chip " +
                              std::to_string(c) + ", which leaves none free for the fabric mux";
                    }
                }
                std::fprintf(stderr,
                             "%s: warning: %zu channel%s between the chips through the host, at about 30 µs a value "
                             "instead of 1.5 µs: %s\n",
                             ctx_.program.c_str(), crowded, crowded > 1 ? "s go" : " goes", why.c_str());
            }
            if (big > 0) {
                std::fprintf(stderr,
                             "%s: warning: %zu channel%s between the chips through the host: %s values are bigger "
                             "than the %zu bytes a fabric packet holds\n",
                             ctx_.program.c_str(), big, big > 1 ? "s go" : " goes", big > 1 ? "their" : "its",
                             max_payload);
            }
        }
        uint32_t offset = 0;
        for (Channel& ch : chans_) {
            ch.offset = offset;
            offset += (ch.ring + 15) / 16 * 16 + kChBlockExtra;
        }
        status_offset_ = offset;
        uint32_t region_bytes = (offset + kStatusBytes + 31) / 32 * 32;
        // The mux keeps its buffers at the region's address on its core: a
        // header and the largest value per buffer.
        uint32_t mux_buffer = 0, mux_buffers = 8;
        if (mux_) {
            uint32_t largest = 16;
            for (const Channel& ch : chans_) {
                if (ch.fabric) {
                    largest = std::max(largest, ch.slot);
                }
            }
            mux_buffer = static_cast<uint32_t>(tt::tt_fabric::get_tt_fabric_packet_header_size_bytes()) + largest;
            mux_buffer = (mux_buffer + 15) / 16 * 16;
            auto mux_bytes = [&](uint32_t buffers) {
                size_t most = 0;
                for (uint32_t clients : mux_clients_) {
                    if (clients > 0) {
                        most = std::max(most, tt::tt_fabric::FabricMuxV2Config(static_cast<uint8_t>(clients),
                                                                               static_cast<uint8_t>(buffers), mux_buffer, 0)
                                                  .get_memory_map_end_address());
                    }
                }
                return most;
            };
            while (mux_buffers > 2 && mux_bytes(mux_buffers) > 128 * 1024) {
                mux_buffers /= 2;
            }
            region_bytes = std::max(region_bytes, static_cast<uint32_t>((mux_bytes(mux_buffers) + 31) / 32 * 32));
        }

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
        if (mux_) {
            defines["TSR_FABRIC_MUX"] = "1";
        }
        for (const CoreRangeSet& set : sets) {
            programs.push_back(CreateProgram());
            if (set.empty()) {  // no goroutine on this chip
                kernels.push_back(0);
                continue;
            }
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
        // The mux on each chip's last core, connected to the link that
        // dispatch leaves, and the two semaphores each of its clients needs.
        std::vector<tt::tt_fabric::FabricMuxV2Config> muxes;
        muxes.reserve(2);
        std::vector<CoreCoord> mux_at;  // each mux's core, as NoC coordinates
        std::vector<tt::tt_fabric::FabricMuxV2Config::ClientSemaphores> mux_sems;
        if (mux_) {
            const CoreCoord core = corerange_to_cores(num_cores_to_corerangeset(static_cast<uint32_t>(per_chip), grid, true),
                                                      static_cast<uint32_t>(per_chip), true)
                                       .back();
            for (size_t c = 0; c < 2; c++) {
                muxes.emplace_back(static_cast<uint8_t>(std::max(1, mux_clients_[c])), static_cast<uint8_t>(mux_buffers),
                                   mux_buffer, region_addr_);
                mux_at.push_back(device(ctx_, static_cast<int>(c))->worker_core_from_logical_core(core));
                mux_sems.push_back({CreateSemaphore(programs[c], sets[c], 0), CreateSemaphore(programs[c], sets[c], 0)});
                const uint32_t first_link = tt::tt_fabric::get_forwarding_link_indices(node(c), node(1 - c)).front();
                tt::tt_fabric::add_fabric_mux_v2_to_program(programs[c], muxes[c], core, node(c), node(1 - c), first_link);
            }
        }
        // A channel as the device runtime's arg_chan reads it. A channel to
        // main, or the sending end of one the host relays, keeps its ring on
        // the sender's core; the receiving end of one that main sends on or
        // the host relays counts what it takes in its own block. Either way
        // the "other end" is itself.
        auto push_chan = [&](std::vector<uint32_t>& a, size_t p, int id, bool sends) {
            const Channel& ch = chans_[static_cast<size_t>(id)];
            const bool to_host = ch.receiver < 0 || (ch.relayed && sends);
            int other = sends ? ch.receiver : ch.sender;
            if (to_host || ch.relayed || ch.main_sends) {
                other = static_cast<int>(p);
            }
            const uint32_t via = to_host ? 1 : (ch.fabric ? 2 : 0);  // tsr::kToHost, kFabric, kNoc
            const CoreCoord peer = devs_[static_cast<size_t>(other)]->worker_core_from_logical_core(
                cores_[static_cast<size_t>(other)]);
            a.push_back(region_addr_ + ch.offset);
            a.push_back(static_cast<uint32_t>(peer.x) | (static_cast<uint32_t>(peer.y) << 8) | (via << 16));
            a.push_back(ch.cap | (ch.slot << 16));
        };
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
                    case Arg::Recv:
                        push_chan(a, p, x.chan, x.kind == Arg::Send);
                        break;
                    case Arg::Sends:
                    case Arg::Recvs:
                        a.push_back(static_cast<uint32_t>(x.chans.size()));
                        for (int id : x.chans) {
                            push_chan(a, p, id, x.kind == Arg::Sends);
                        }
                        break;
                }
            }
            if (link[p] >= 0) {
                const size_t c = static_cast<size_t>(chip_[p]);
                a[2] = static_cast<uint32_t>(a.size());
                if (mux_) {
                    muxes[c].append_client_connection_rt_args(mux_at[c], static_cast<uint8_t>(link[p]), mux_sems[c], a);
                } else {
                    tt::tt_fabric::append_fabric_connection_rt_args(node(c), node(1 - c), static_cast<uint32_t>(link[p]),
                                                                    programs[c], cores_[p], a);
                }
            }
            SetRuntimeArgs(programs[static_cast<size_t>(chip_[p])], kernels[static_cast<size_t>(chip_[p])], cores_[p], a);
        }
        for (Channel& ch : chans_) {
            ch.ring = (ch.ring + 15) / 16 * 16;
        }
        if (ctx_.compile_only) {
            for (size_t c = 0; c < programs.size(); c++) {
                if (on_chip_[c] > 0) {
                    detail::CompileProgram(device(ctx_, static_cast<int>(c)), programs[c]);
                }
            }
            std::fprintf(stderr, "%s: goroutine kernel compiled for %zu cores\n", ctx_.program.c_str(), n);
            close_card(ctx_);
            std::exit(0);
        }

        auto& cq = ctx_.mesh->mesh_command_queue();
        distributed::Finish(cq);
        const std::vector<uint8_t> zeros(region_bytes, 0);
        for (size_t p = 0; p < n; p++) {
            write_l1(devs_[p], cores_[p], region_addr_, std::span<const uint8_t>(zeros));
        }
        for (Channel& ch : chans_) {
            for (const std::vector<uint8_t>& slot : ch.pending) {
                put(ch, slot);
            }
            ch.pending.clear();
            if (ch.main_closed) {
                put_closed(ch);
            }
        }
        for (size_t c = 0; c < programs.size(); c++) {
            if (on_chip_[c] > 0) {
                const distributed::MeshCoordinate at(0, static_cast<uint32_t>(c));
                workload_.add_program(distributed::MeshCoordinateRange(at, at), std::move(programs[c]));
            }
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
    uint32_t select_turn_ = 0;
    std::vector<size_t> on_chip_;  // how many goroutines run on each chip
    bool mux_ = false;                       // channels between the chips go through a fabric mux on each
    std::vector<int> mux_clients_ = {0, 0};  // and so many cores of each chip connect to it
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
