// Runs the device runtime's math/bits functions on the host, for
// compiler/bits_test.go, which compares the answers with Go's.
//
// Each line read is a call, "Name arg...", with the arguments in decimal
// (RotateLeft's count may be negative); each line written is its result or
// results, in decimal. Calls that would panic are not sent.
#include <cstdio>
#include <iostream>
#include <sstream>
#include <string>

using namespace tsr;

static uint8_t heap_memory[1 << 16];

int main() {
    heap_init(reinterpret_cast<uintptr_t>(heap_memory), sizeof heap_memory);
    std::string line;
    while (std::getline(std::cin, line)) {
        std::istringstream in(line);
        std::string f;
        unsigned long long a = 0, b = 0, c = 0;
        long long k = 0;
        in >> f;
        std::ostringstream out;
#define ONE(name, T) else if (f == #name) { in >> a; out << bits_##name(static_cast<T>(a)); }
#define UONE(name, T) else if (f == #name) { in >> a; out << static_cast<unsigned long long>(bits_##name(static_cast<T>(a))); }
#define ROT(name, T) else if (f == #name) { in >> a >> k; out << static_cast<unsigned long long>(bits_##name(static_cast<T>(a), k)); }
#define PAIR3(name, T) else if (f == #name) { in >> a >> b >> c; auto r = bits_##name(static_cast<T>(a), static_cast<T>(b), static_cast<T>(c)); \
        out << static_cast<unsigned long long>(r.r0) << " " << static_cast<unsigned long long>(r.r1); }
#define PAIR2(name, T) else if (f == #name) { in >> a >> b; auto r = bits_##name(static_cast<T>(a), static_cast<T>(b)); \
        out << static_cast<unsigned long long>(r.r0) << " " << static_cast<unsigned long long>(r.r1); }
#define DIV(name, T) else if (f == #name) { in >> a >> b >> c; auto r = bits_##name(0, 0, static_cast<T>(a), static_cast<T>(b), static_cast<T>(c)); \
        out << static_cast<unsigned long long>(r.r0) << " " << static_cast<unsigned long long>(r.r1); }
#define REM(name, T) else if (f == #name) { in >> a >> b >> c; out << static_cast<unsigned long long>(bits_##name(0, static_cast<T>(a), static_cast<T>(b), static_cast<T>(c))); }
        if (false) {
        }
        ONE(OnesCount, uint64_t) ONE(OnesCount8, uint8_t) ONE(OnesCount16, uint16_t) ONE(OnesCount32, uint32_t) ONE(OnesCount64, uint64_t)
        ONE(Len, uint64_t) ONE(Len8, uint8_t) ONE(Len16, uint16_t) ONE(Len32, uint32_t) ONE(Len64, uint64_t)
        ONE(LeadingZeros, uint64_t) ONE(LeadingZeros8, uint8_t) ONE(LeadingZeros16, uint16_t) ONE(LeadingZeros32, uint32_t)
        ONE(LeadingZeros64, uint64_t)
        ONE(TrailingZeros, uint64_t) ONE(TrailingZeros8, uint8_t) ONE(TrailingZeros16, uint16_t) ONE(TrailingZeros32, uint32_t)
        ONE(TrailingZeros64, uint64_t)
        ROT(RotateLeft, uint64_t) ROT(RotateLeft8, uint8_t) ROT(RotateLeft16, uint16_t) ROT(RotateLeft32, uint32_t)
        ROT(RotateLeft64, uint64_t)
        UONE(Reverse, uint64_t) UONE(Reverse8, uint8_t) UONE(Reverse16, uint16_t) UONE(Reverse32, uint32_t) UONE(Reverse64, uint64_t)
        UONE(ReverseBytes, uint64_t) UONE(ReverseBytes16, uint16_t) UONE(ReverseBytes32, uint32_t) UONE(ReverseBytes64, uint64_t)
        PAIR3(Add, uint64_t) PAIR3(Add32, uint32_t) PAIR3(Add64, uint64_t)
        PAIR3(Sub, uint64_t) PAIR3(Sub32, uint32_t) PAIR3(Sub64, uint64_t)
        PAIR2(Mul, uint64_t) PAIR2(Mul32, uint32_t) PAIR2(Mul64, uint64_t)
        DIV(Div, uint64_t) DIV(Div32, uint32_t) DIV(Div64, uint64_t)
        REM(Rem, uint64_t) REM(Rem32, uint32_t) REM(Rem64, uint64_t)
        else {
            out << "unknown " << f;
        }
        std::cout << out.str() << "\n";
    }
    return 0;
}
