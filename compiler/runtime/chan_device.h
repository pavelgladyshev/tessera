// ---------------------------------------------------------------------------
// Go channels between goroutines on different Tensix cores (pasted in by
// tessc for programs that start goroutines).
//
// Every core has the same channel region at the same L1 address; channel c
// has a block at the same offset in every core's region:
//
//   [ring: cap slots] [written] [closed] [consumed] [sent|received|closed by me]
//
// The ring and written/closed are used on the receiver's core, consumed and
// sent on the sender's. To send, the sender waits until it has a free slot
// (sent - consumed < cap), writes the value into the receiver's ring over
// the NoC, then increments the receiver's written counter with a NoC atomic.
// The receiver reads the slot from its own L1 and gives it back by
// incrementing the sender's consumed counter. Nothing is shared but these
// counters, so a value is never read before it has landed.
//
// A channel whose receiver is main keeps its ring on the sender's core; the
// host reads it over PCIe and writes consumed back.
//
// A channel between goroutines on different chips goes over tt-metal's
// fabric, the Ethernet links between the chips: the sender writes the slot
// and bumps the receiver's written counter with one fused packet, and the
// receiver bumps the sender's consumed counter with another. (If the fabric
// has no connection left for a core, the host relays the channel instead:
// the sender sees a channel to main, the receiver one whose other end is
// itself.)
// ---------------------------------------------------------------------------
namespace tsr {

constexpr uint32_t kWritten = 0, kClosed = 16, kConsumed = 32, kSent = 48, kReceived = 52, kClosedByMe = 56;

constexpr uint32_t kNoc = 0, kToHost = 1, kFabric = 2;  // how a channel reaches its other end

struct Chan {
    uint32_t base;  // L1: this channel's block on this core
    uint64_t peer;  // NoC address of the block on the other end's core (on its chip)
    uint32_t cap;   // slots
    uint32_t slot;  // bytes per slot (a multiple of 16)
    uint32_t ring;  // bytes of the ring
    uint32_t via;   // kNoc; kToHost: main receives (the ring is on this, the sender's, core); kFabric
};

#if defined(TSR_FABRIC)
// A core's connection to the fabric: straight to an Ethernet router, or
// (TSR_FABRIC_MUX) to a channel of the fabric mux on its chip's last core,
// which forwards the packets of many cores to one router.
#if defined(TSR_FABRIC_MUX)
using FabricConn = tt::tt_fabric::FabricMuxV2Sender<>;
inline FabricConn fabric_connect(size_t& i) { return FabricConn::build_from_args(i); }
#else
using FabricConn = tt::tt_fabric::WorkerToFabricEdmSender;
inline FabricConn fabric_connect(size_t& i) { return FabricConn::build_from_args<ProgrammableCoreType::TENSIX>(i); }
#endif
inline FabricConn* fabric_conn = nullptr;
inline volatile tt_l1_ptr PACKET_HEADER_TYPE* fabric_header = nullptr;

inline void fabric_init(FabricConn* c) {
    fabric_conn = c;
    fabric_header = PacketHeaderPool::allocate_header();
    fabric_set_unicast_route<false>((LowLatencyPacketHeader*)fabric_header, 1);  // the next chip
}

// Sends the packet header (after any payload) and waits until the header
// has left this core's L1, so that the next packet can reuse it.
inline void fabric_send_header() {
    fabric_conn->send_payload_flush_non_blocking_from_address(reinterpret_cast<uint32_t>(fabric_header),
                                                              sizeof(PACKET_HEADER_TYPE));
    noc_async_writes_flushed();
}

// Writes len bytes from src to dst on the other chip, then adds 1 to the
// counter there.
inline void fabric_write_inc(uint32_t src, uint32_t len, uint64_t dst, uint64_t counter) {
    fabric_header->to_noc_fused_unicast_write_atomic_inc(
        tt::tt_fabric::NocUnicastAtomicIncFusedCommandHeader{dst, counter, 1, true}, len);
    fabric_conn->wait_for_empty_write_slot();
    fabric_conn->send_payload_without_header_non_blocking_from_address(src, len);
    fabric_send_header();
}

// Adds 1 to a counter on the other chip.
inline void fabric_inc(uint64_t counter) {
    fabric_header->to_noc_unicast_atomic_inc(tt::tt_fabric::NocUnicastAtomicIncCommandHeader{counter, 1, true});
    fabric_conn->wait_for_empty_write_slot();
    fabric_send_header();
}
#endif

inline Chan arg_chan(uint32_t& i) {
    Chan c;
    c.base = arg32(i);
    const uint32_t x = arg32(i), y = arg32(i);
    c.peer = get_noc_addr(x, y, c.base);
    c.cap = arg32(i);
    c.slot = arg32(i);
    c.ring = arg32(i);
    c.via = arg32(i);
    return c;
}

inline volatile uint32_t* chan_word(const Chan& c, uint32_t off) {
    return reinterpret_cast<volatile uint32_t*>(c.base + c.ring + off);
}

template <typename T>
inline void send(const Chan& c, const T& v, int64_t pos) {
    if (*chan_word(c, kClosedByMe)) {
        fail(pos, 0);  // send on a closed channel
    }
    volatile uint32_t* sent = chan_word(c, kSent);
    volatile uint32_t* consumed = chan_word(c, kConsumed);
    const uint32_t s = *sent;
    while (s - *consumed >= c.cap) {
    }
    // The sender's own copy of the ring is free, so the value is staged there.
    const uint32_t at = c.base + (s % c.cap) * c.slot;
    *reinterpret_cast<T*>(at) = v;
    if (c.via == kToHost) {
        *chan_word(c, kWritten) = s + 1;
#if defined(TSR_FABRIC)
    } else if (c.via == kFabric) {
        fabric_write_inc(at, c.slot, c.peer + (s % c.cap) * c.slot, c.peer + c.ring + kWritten);
#endif
    } else {
        noc_async_write(at, c.peer + (s % c.cap) * c.slot, c.slot);
        noc_async_write_barrier();
        noc_semaphore_inc(c.peer + c.ring + kWritten, 1);
        noc_async_atomic_barrier();
    }
    *sent = s + 1;
}

// Receives the next value into *out; false once the channel is closed and
// every value sent has been received.
template <typename T>
inline bool recv(const Chan& c, T* out) {
    volatile uint32_t* written = chan_word(c, kWritten);
    volatile uint32_t* closed = chan_word(c, kClosed);
    volatile uint32_t* received = chan_word(c, kReceived);
    const uint32_t r = *received;
    while (*written == r) {
        if (*closed && *written == r) {
            return false;
        }
    }
    *out = *reinterpret_cast<const T*>(c.base + (r % c.cap) * c.slot);
    *received = r + 1;
#if defined(TSR_FABRIC)
    if (c.via == kFabric) {
        fabric_inc(c.peer + c.ring + kConsumed);
        return true;
    }
#endif
    noc_semaphore_inc(c.peer + c.ring + kConsumed, 1);
    noc_async_atomic_barrier();
    return true;
}

template <typename T>
inline T recv_value(const Chan& c) {
    T v{};
    recv(c, &v);
    return v;
}

inline void close(const Chan& c, int64_t pos) {
    volatile uint32_t* mine = chan_word(c, kClosedByMe);
    if (*mine) {
        fail(pos, 0);  // close of a closed channel
    }
    *mine = 1;
    if (c.via == kToHost) {
        *chan_word(c, kClosed) = 1;
#if defined(TSR_FABRIC)
    } else if (c.via == kFabric) {
        fabric_inc(c.peer + c.ring + kClosed);
#endif
    } else {
        noc_semaphore_inc(c.peer + c.ring + kClosed, 1);
        noc_async_atomic_barrier();
    }
}

}  // namespace tsr
// ---------------------------------------------------------------------------
