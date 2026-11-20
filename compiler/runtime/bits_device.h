// ---------------------------------------------------------------------------
// math/bits in worker code (pasted in by tessc).
//
// The worker cores (rv32im) have no instructions for counting or reversing
// bits, and GCC's builtins for them call libgcc versions that use lookup
// tables, which would land in the core's 4 KB of local memory. So these are
// plain integer arithmetic. Go's int results are int64_t, and its uint is
// uint64_t, as everywhere in worker code; two results come back as .r0 and
// .r1.
// ---------------------------------------------------------------------------
namespace tsr {

template <typename T>
struct BitsPair {
    T r0, r1;
};

// --- OnesCount

inline int64_t bits_ones32(uint32_t x) {
    x = x - ((x >> 1) & 0x55555555u);
    x = (x & 0x33333333u) + ((x >> 2) & 0x33333333u);
    x = (x + (x >> 4)) & 0x0f0f0f0fu;
    return static_cast<int64_t>((x * 0x01010101u) >> 24);
}
inline int64_t bits_OnesCount8(uint8_t x) { return bits_ones32(x); }
inline int64_t bits_OnesCount16(uint16_t x) { return bits_ones32(x); }
inline int64_t bits_OnesCount32(uint32_t x) { return bits_ones32(x); }
inline int64_t bits_OnesCount64(uint64_t x) {
    return bits_ones32(static_cast<uint32_t>(x)) + bits_ones32(static_cast<uint32_t>(x >> 32));
}
inline int64_t bits_OnesCount(uint64_t x) { return bits_OnesCount64(x); }

// --- Len, LeadingZeros: the bits needed to hold x (0 for 0), and the zeros above them

inline int64_t bits_len32(uint32_t x) {
    int64_t n = 0;
    if (x >= 1u << 16) {
        x >>= 16;
        n += 16;
    }
    if (x >= 1u << 8) {
        x >>= 8;
        n += 8;
    }
    if (x >= 1u << 4) {
        x >>= 4;
        n += 4;
    }
    if (x >= 1u << 2) {
        x >>= 2;
        n += 2;
    }
    if (x >= 1u << 1) {
        x >>= 1;
        n += 1;
    }
    return n + static_cast<int64_t>(x);  // x is 0 or 1 by now
}
inline int64_t bits_Len8(uint8_t x) { return bits_len32(x); }
inline int64_t bits_Len16(uint16_t x) { return bits_len32(x); }
inline int64_t bits_Len32(uint32_t x) { return bits_len32(x); }
inline int64_t bits_Len64(uint64_t x) {
    const uint32_t hi = static_cast<uint32_t>(x >> 32);
    return hi != 0 ? 32 + bits_len32(hi) : bits_len32(static_cast<uint32_t>(x));
}
inline int64_t bits_Len(uint64_t x) { return bits_Len64(x); }

inline int64_t bits_LeadingZeros8(uint8_t x) { return 8 - bits_Len8(x); }
inline int64_t bits_LeadingZeros16(uint16_t x) { return 16 - bits_Len16(x); }
inline int64_t bits_LeadingZeros32(uint32_t x) { return 32 - bits_Len32(x); }
inline int64_t bits_LeadingZeros64(uint64_t x) { return 64 - bits_Len64(x); }
inline int64_t bits_LeadingZeros(uint64_t x) { return bits_LeadingZeros64(x); }

// --- TrailingZeros: the zeros below the lowest one (all of them for 0)

inline int64_t bits_tz32(uint32_t x) { return x == 0 ? 32 : bits_ones32((x & (0u - x)) - 1); }
inline int64_t bits_TrailingZeros8(uint8_t x) { return x == 0 ? 8 : bits_tz32(x); }
inline int64_t bits_TrailingZeros16(uint16_t x) { return x == 0 ? 16 : bits_tz32(x); }
inline int64_t bits_TrailingZeros32(uint32_t x) { return bits_tz32(x); }
inline int64_t bits_TrailingZeros64(uint64_t x) {
    const uint32_t lo = static_cast<uint32_t>(x);
    return lo != 0 ? bits_tz32(lo) : 32 + bits_tz32(static_cast<uint32_t>(x >> 32));
}
inline int64_t bits_TrailingZeros(uint64_t x) { return bits_TrailingZeros64(x); }

// --- RotateLeft: by k mod the width; a negative k rotates right

// W is wide enough to shift x without C++'s promotion to int.
template <typename T, typename W, uint32_t N>
inline T bits_rotate(T x, int64_t k) {
    const uint32_t s = static_cast<uint32_t>(static_cast<uint64_t>(k) & (N - 1));
    if (s == 0) {
        return x;
    }
    const W v = x;
    return static_cast<T>((v << s) | (v >> (N - s)));
}
inline uint8_t bits_RotateLeft8(uint8_t x, int64_t k) { return bits_rotate<uint8_t, uint32_t, 8>(x, k); }
inline uint16_t bits_RotateLeft16(uint16_t x, int64_t k) { return bits_rotate<uint16_t, uint32_t, 16>(x, k); }
inline uint32_t bits_RotateLeft32(uint32_t x, int64_t k) { return bits_rotate<uint32_t, uint32_t, 32>(x, k); }
inline uint64_t bits_RotateLeft64(uint64_t x, int64_t k) { return bits_rotate<uint64_t, uint64_t, 64>(x, k); }
inline uint64_t bits_RotateLeft(uint64_t x, int64_t k) { return bits_RotateLeft64(x, k); }

// --- Reverse (the bits) and ReverseBytes

inline uint32_t bits_rev32(uint32_t x) {
    x = ((x >> 1) & 0x55555555u) | ((x & 0x55555555u) << 1);
    x = ((x >> 2) & 0x33333333u) | ((x & 0x33333333u) << 2);
    x = ((x >> 4) & 0x0f0f0f0fu) | ((x & 0x0f0f0f0fu) << 4);
    return (x >> 24) | ((x >> 8) & 0xff00u) | ((x << 8) & 0xff0000u) | (x << 24);
}
inline uint8_t bits_Reverse8(uint8_t x) { return static_cast<uint8_t>(bits_rev32(x) >> 24); }
inline uint16_t bits_Reverse16(uint16_t x) { return static_cast<uint16_t>(bits_rev32(x) >> 16); }
inline uint32_t bits_Reverse32(uint32_t x) { return bits_rev32(x); }
inline uint64_t bits_Reverse64(uint64_t x) {
    return (static_cast<uint64_t>(bits_rev32(static_cast<uint32_t>(x))) << 32) | bits_rev32(static_cast<uint32_t>(x >> 32));
}
inline uint64_t bits_Reverse(uint64_t x) { return bits_Reverse64(x); }

inline uint16_t bits_ReverseBytes16(uint16_t x) { return static_cast<uint16_t>((x >> 8) | (x << 8)); }
inline uint32_t bits_ReverseBytes32(uint32_t x) {
    return (x >> 24) | ((x >> 8) & 0xff00u) | ((x << 8) & 0xff0000u) | (x << 24);
}
inline uint64_t bits_ReverseBytes64(uint64_t x) {
    return (static_cast<uint64_t>(bits_ReverseBytes32(static_cast<uint32_t>(x))) << 32) |
           bits_ReverseBytes32(static_cast<uint32_t>(x >> 32));
}
inline uint64_t bits_ReverseBytes(uint64_t x) { return bits_ReverseBytes64(x); }

// --- Add, Sub (with a carry or borrow of 0 or 1 in and out), Mul (the full product)

inline BitsPair<uint32_t> bits_Add32(uint32_t x, uint32_t y, uint32_t carry) {
    const uint32_t sum = x + y + carry;
    return {sum, ((x & y) | ((x | y) & ~sum)) >> 31};
}
inline BitsPair<uint64_t> bits_Add64(uint64_t x, uint64_t y, uint64_t carry) {
    const uint64_t sum = x + y + carry;
    return {sum, ((x & y) | ((x | y) & ~sum)) >> 63};
}
inline BitsPair<uint64_t> bits_Add(uint64_t x, uint64_t y, uint64_t carry) { return bits_Add64(x, y, carry); }

inline BitsPair<uint32_t> bits_Sub32(uint32_t x, uint32_t y, uint32_t borrow) {
    const uint32_t diff = x - y - borrow;
    return {diff, ((~x & y) | (~(x ^ y) & diff)) >> 31};
}
inline BitsPair<uint64_t> bits_Sub64(uint64_t x, uint64_t y, uint64_t borrow) {
    const uint64_t diff = x - y - borrow;
    return {diff, ((~x & y) | (~(x ^ y) & diff)) >> 63};
}
inline BitsPair<uint64_t> bits_Sub(uint64_t x, uint64_t y, uint64_t borrow) { return bits_Sub64(x, y, borrow); }

inline BitsPair<uint32_t> bits_Mul32(uint32_t x, uint32_t y) {
    const uint64_t p = static_cast<uint64_t>(x) * y;
    return {static_cast<uint32_t>(p >> 32), static_cast<uint32_t>(p)};
}
TSR_OUTLINE inline BitsPair<uint64_t> bits_Mul64(uint64_t x, uint64_t y) {
    const uint64_t x0 = x & 0xffffffffu, x1 = x >> 32, y0 = y & 0xffffffffu, y1 = y >> 32;
    const uint64_t w0 = x0 * y0;
    const uint64_t t = x1 * y0 + (w0 >> 32);
    const uint64_t w1 = (t & 0xffffffffu) + x0 * y1;
    return {x1 * y1 + (t >> 32) + (w1 >> 32), x * y};
}
inline BitsPair<uint64_t> bits_Mul(uint64_t x, uint64_t y) { return bits_Mul64(x, y); }

// --- Div, Rem: the 128- (or 64-) bit number hi:lo divided by y. Div panics
// as Go's does when y is 0 or the quotient does not fit (y <= hi); Rem only
// when y is 0.

TSR_OUTLINE inline BitsPair<uint32_t> bits_Div32(int64_t pos_zero, int64_t pos_overflow, uint32_t hi, uint32_t lo,
                                                 uint32_t y) {
    if (y == 0) {
        fail(pos_zero, 0);
    }
    if (y <= hi) {
        fail(pos_overflow, 0);
    }
    const uint64_t z = (static_cast<uint64_t>(hi) << 32) | lo;
    return {static_cast<uint32_t>(z / y), static_cast<uint32_t>(z % y)};
}
// Long division a bit at a time: hi < y, so the quotient fits in 64 bits.
inline BitsPair<uint64_t> bits_div128(uint64_t hi, uint64_t lo, uint64_t y) {
    uint64_t q = 0, r = hi;
    for (int i = 63; i >= 0; i--) {
        const uint64_t top = r >> 63;
        r = (r << 1) | ((lo >> i) & 1);
        q <<= 1;
        if (top != 0 || r >= y) {
            r -= y;
            q |= 1;
        }
    }
    return {q, r};
}
TSR_OUTLINE inline BitsPair<uint64_t> bits_Div64(int64_t pos_zero, int64_t pos_overflow, uint64_t hi, uint64_t lo,
                                                 uint64_t y) {
    if (y == 0) {
        fail(pos_zero, 0);
    }
    if (y <= hi) {
        fail(pos_overflow, 0);
    }
    return bits_div128(hi, lo, y);
}
inline BitsPair<uint64_t> bits_Div(int64_t pos_zero, int64_t pos_overflow, uint64_t hi, uint64_t lo, uint64_t y) {
    return bits_Div64(pos_zero, pos_overflow, hi, lo, y);
}

TSR_OUTLINE inline uint32_t bits_Rem32(int64_t pos_zero, uint32_t hi, uint32_t lo, uint32_t y) {
    if (y == 0) {
        fail(pos_zero, 0);
    }
    return static_cast<uint32_t>(((static_cast<uint64_t>(hi) << 32) | lo) % y);
}
TSR_OUTLINE inline uint64_t bits_Rem64(int64_t pos_zero, uint64_t hi, uint64_t lo, uint64_t y) {
    if (y == 0) {
        fail(pos_zero, 0);
    }
    return bits_div128(hi % y, lo, y).r1;
}
inline uint64_t bits_Rem(int64_t pos_zero, uint64_t hi, uint64_t lo, uint64_t y) { return bits_Rem64(pos_zero, hi, lo, y); }

}  // namespace tsr
