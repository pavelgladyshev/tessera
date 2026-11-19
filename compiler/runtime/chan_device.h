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
// ---------------------------------------------------------------------------
namespace tsr {

constexpr uint32_t kWritten = 0, kClosed = 16, kConsumed = 32, kSent = 48, kReceived = 52, kClosedByMe = 56;

struct Chan {
    uint32_t base;     // L1: this channel's block on this core
    uint64_t peer;     // NoC address of the block on the other end's core
    uint32_t cap;      // slots
    uint32_t slot;     // bytes per slot (a multiple of 16)
    uint32_t ring;     // bytes of the ring
    uint32_t to_host;  // 1: main receives (the ring is on this, the sender's, core)
};

inline Chan arg_chan(uint32_t& i) {
    Chan c;
    c.base = arg32(i);
    const uint32_t x = arg32(i), y = arg32(i);
    c.peer = get_noc_addr(x, y, c.base);
    c.cap = arg32(i);
    c.slot = arg32(i);
    c.ring = arg32(i);
    c.to_host = arg32(i);
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
    if (c.to_host) {
        *chan_word(c, kWritten) = s + 1;
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
    if (c.to_host) {
        *chan_word(c, kClosed) = 1;
    } else {
        noc_semaphore_inc(c.peer + c.ring + kClosed, 1);
        noc_async_atomic_barrier();
    }
}

}  // namespace tsr
// ---------------------------------------------------------------------------
