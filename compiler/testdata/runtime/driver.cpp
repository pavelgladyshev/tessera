// Runs the device runtime's string functions and maps on the host, for
// compiler/runtime_test.go, which compares the answers with Go's.
//
// Each line read is a call, "name arg...", and each line written is its
// result. Values are written as runtime_test.go writes Go's:
//   s:HEX            a string (HEX may be empty)
//   b:HEX/CAP        a []byte, or nil
//   i:N, t, f        an integer, true, false
//   [v;v;...]/CAP    a []string or [][]byte, or nil
// Maps are tested with "map OPS...", a program of operations (see
// runtime_test.go).
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <map>
#include <sstream>
#include <string>
#include <vector>

using namespace tsr;

static uint8_t heap_memory[256 << 20];

static std::string hex(const uint8_t* p, int64_t n) {
    static const char* digits = "0123456789abcdef";
    std::string h;
    for (int64_t i = 0; i < n; i++) {
        h += digits[p[i] >> 4];
        h += digits[p[i] & 15];
    }
    return h;
}

static std::vector<uint8_t> unhex(const std::string& h) {
    std::vector<uint8_t> b;
    for (size_t i = 0; i + 1 < h.size(); i += 2) {
        b.push_back(static_cast<uint8_t>(std::stoi(h.substr(i, 2), nullptr, 16)));
    }
    return b;
}

static Bytes8 heap_bytes(const std::vector<uint8_t>& v) {
    Bytes8 b = make<uint8_t>(static_cast<int64_t>(v.size()), 0, 0);
    for (size_t i = 0; i < v.size(); i++) {
        b.ptr[i] = v[i];
    }
    return b;
}

static Str arg_s(const std::string& t) { return view(heap_bytes(unhex(t.substr(2)))); }
static Bytes8 arg_b(const std::string& t) { return t == "nil" ? nil_bytes() : heap_bytes(unhex(t.substr(2))); }
static int64_t arg_i(const std::string& t) { return std::stoll(t.substr(2)); }

static std::vector<std::string> items(const std::string& t) {  // l:xHEX,xHEX,... (l: is empty)
    std::vector<std::string> out;
    std::string body = t.substr(2), cur;
    if (body.empty()) {
        return out;
    }
    std::stringstream ss(body);
    while (std::getline(ss, cur, ',')) {
        out.push_back(cur.substr(1));
    }
    return out;
}
static Slice<Str> arg_ls(const std::string& t) {
    auto v = items(t);
    Slice<Str> a = make<Str>(static_cast<int64_t>(v.size()), 0, 0);
    for (size_t i = 0; i < v.size(); i++) {
        a.ptr[i] = view(heap_bytes(unhex(v[i])));
    }
    return a;
}
static Slice<Bytes8> arg_lb(const std::string& t) {
    auto v = items(t);
    Slice<Bytes8> a = make<Bytes8>(static_cast<int64_t>(v.size()), 0, 0);
    for (size_t i = 0; i < v.size(); i++) {
        a.ptr[i] = heap_bytes(unhex(v[i]));
    }
    return a;
}

static std::string out(const Str& s) { return "s:" + hex(s.ptr, s.len); }
static std::string out(const Bytes8& b) {
    if (b.ptr == nullptr) {
        return "nil";
    }
    return "b:" + hex(b.ptr, b.len) + "/" + std::to_string(b.cap);
}
static std::string out(int64_t v) { return "i:" + std::to_string(v); }
static std::string out(bool v) { return v ? "t" : "f"; }
template <typename T>
static std::string out(const Slice<T>& a) {
    if (a.ptr == nullptr) {
        return "nil";
    }
    std::string s = "[";
    for (int64_t i = 0; i < a.len; i++) {
        s += (i > 0 ? ";" : "") + out(a.ptr[i]);
    }
    return s + "]/" + std::to_string(a.cap);
}
static std::string out(const RuneSize& r) { return out(static_cast<int64_t>(r.r0)) + " " + out(r.r1); }
static std::string out(const StrCut& c) { return out(c.r0) + " " + out(c.r1) + " " + out(c.r2); }
static std::string out(const StrFound& c) { return out(c.r0) + " " + out(c.r1); }
static std::string out(const BytesCut& c) { return out(c.r0) + " " + out(c.r1) + " " + out(c.r2); }
static std::string out(const BytesFound& c) { return out(c.r0) + " " + out(c.r1); }

// map OPS...: a program on a map[int64]int64 (keys "k:N") or a
// map[string]int64 (keys "s:HEX"), one result per operation:
//   set K V   m[K] = V                    -> -
//   add K V   m[K] += V                   -> -
//   del K     delete(m, K)                -> -
//   get K     v, ok := m[K]               -> i:V t|f
//   len       len(m)                      -> i:N
//   sum       sum of hash(k)*v over range -> i:S (order does not matter)
//   rdel V    range, deleting entries whose value is V; -> i:visits
//   radd N    range, adding N new keys per entry visited; -> i:(sum of the old entries visited)
//   clear     clear(m)                    -> -
template <typename K>
static std::string run_map(std::istringstream& in, K (*key)(const std::string&), int64_t (*weight)(const K&),
                           std::string (*name)(const K&)) {
    Map<K, int64_t> m = make_map<K, int64_t>(0, 0);
    std::string op, results;
    int64_t fresh = 1000000;
    auto emit = [&](const std::string& r) { results += (results.empty() ? "" : " ") + r; };
    while (in >> op) {
        std::string a, b;
        if (op == "set" || op == "add") {
            in >> a >> b;
            int64_t& slot = map_ref(m, key(a), 0, 0);
            slot = op == "set" ? std::stoll(b) : slot + std::stoll(b);
            emit("-");
        } else if (op == "del") {
            in >> a;
            map_delete(m, key(a));
            emit("-");
        } else if (op == "get") {
            in >> a;
            int64_t v;
            const bool ok = map_lookup(m, key(a), v);
            emit(out(v) + " " + out(ok) + (map_get(m, key(a)) == v ? "" : "!get"));
        } else if (op == "len") {
            emit(out(map_len(m)));
        } else if (op == "sum") {
            int64_t s = 0, n = 0;
            MapIter<K, int64_t> it(m);
            K k{};
            int64_t v;
            while (it.next(k, v)) {
                s += weight(k) * v;
                n++;
            }
            emit(out(s) + (n == map_len(m) ? "" : "!count"));
        } else if (op == "rdel") {
            in >> a;
            const int64_t target = std::stoll(a);
            int64_t visits = 0;
            MapIter<K, int64_t> it(m);
            K k{};
            int64_t v;
            while (it.next(k, v)) {
                visits++;
                if (v == target) {
                    map_delete(m, k);
                }
            }
            emit(out(visits));
        } else if (op == "radd") {
            in >> a;
            const int64_t per = std::stoll(a);
            // The entries there before the loop, which it must visit once each.
            std::map<std::string, int64_t> before;
            {
                MapIter<K, int64_t> it(m);
                K k{};
                int64_t v;
                while (it.next(k, v)) {
                    before[name(k)] = v;
                }
            }
            int64_t s = 0;
            MapIter<K, int64_t> it(m);
            K k{};
            int64_t v;
            while (it.next(k, v)) {
                auto f = before.find(name(k));
                if (f == before.end()) {
                    continue;  // added by this loop: Go may or may not visit it
                }
                s += weight(k) * v;
                before.erase(f);
                for (int64_t j = 0; j < per; j++) {  // so the map grows while the loop runs
                    map_ref(m, key("k:" + std::to_string(fresh++)), 0, 0) = 1;
                }
            }
            emit(out(s) + (before.empty() ? "" : "!missed"));
        } else if (op == "clear") {
            map_clear(m);
            emit("-");
        }
    }
    return results;
}

static int64_t int_key(const std::string& t) { return std::stoll(t.substr(2)); }
static Str str_key(const std::string& t) {
    if (t.rfind("k:", 0) == 0) {  // a fresh key from radd
        std::string d = t.substr(2);
        return view(heap_bytes(std::vector<uint8_t>(d.begin(), d.end())));
    }
    return arg_s(t);
}
static int64_t int_weight(const int64_t& k) { return k; }
static std::string int_name(const int64_t& k) { return std::to_string(k); }
static std::string str_name(const Str& k) { return std::string(reinterpret_cast<const char*>(k.ptr), static_cast<size_t>(k.len)); }
static int64_t str_weight(const Str& k) {  // the same as runtime_test.go's
    int64_t h = 7;
    for (int64_t i = 0; i < k.len; i++) {
        h = h * 31 + k.ptr[i];
    }
    return h % 1000003;
}

int main() {
    heap_init(reinterpret_cast<uintptr_t>(heap_memory), sizeof heap_memory);
    std::string line;
    while (std::getline(std::cin, line)) {
        HeapMark mark;
        std::istringstream in(line);
        std::string f;
        in >> f;
        std::vector<std::string> a;
        if (f != "map" && f != "smap") {
            std::string t;
            while (in >> t) {
                a.push_back(t);
            }
        }
        std::string r;
#define S(i) arg_s(a.at(i))
#define B(i) arg_b(a.at(i))
#define I(i) arg_i(a.at(i))
#define CALL(name, expr)  \
    else if (f == name) { \
        r = out(expr);    \
    }
        if (f == "map") {
            r = run_map<int64_t>(in, int_key, int_weight, int_name);
        } else if (f == "smap") {
            r = run_map<Str>(in, str_key, str_weight, str_name);
        }
        CALL("strings.Compare", strings_Compare(S(0), S(1)))
        CALL("strings.Contains", strings_Contains(S(0), S(1)))
        CALL("strings.ContainsAny", strings_ContainsAny(S(0), S(1)))
        CALL("strings.ContainsRune", strings_ContainsRune(S(0), I(1)))
        CALL("strings.Count", strings_Count(S(0), S(1)))
        CALL("strings.Cut", strings_Cut(S(0), S(1)))
        CALL("strings.CutPrefix", strings_CutPrefix(S(0), S(1)))
        CALL("strings.CutSuffix", strings_CutSuffix(S(0), S(1)))
        CALL("strings.Fields", strings_Fields(0, S(0)))
        CALL("strings.HasPrefix", strings_HasPrefix(S(0), S(1)))
        CALL("strings.HasSuffix", strings_HasSuffix(S(0), S(1)))
        CALL("strings.Index", strings_Index(S(0), S(1)))
        CALL("strings.IndexAny", strings_IndexAny(S(0), S(1)))
        CALL("strings.IndexByte", strings_IndexByte(S(0), static_cast<uint8_t>(I(1))))
        CALL("strings.IndexRune", strings_IndexRune(S(0), I(1)))
        CALL("strings.Join", strings_Join(0, arg_ls(a.at(0)), S(1)))
        CALL("strings.LastIndex", strings_LastIndex(S(0), S(1)))
        CALL("strings.LastIndexAny", strings_LastIndexAny(S(0), S(1)))
        CALL("strings.LastIndexByte", strings_LastIndexByte(S(0), static_cast<uint8_t>(I(1))))
        CALL("strings.Repeat", strings_Repeat(0, 0, S(0), I(1)))
        CALL("strings.Replace", strings_Replace(0, S(0), S(1), S(2), I(3)))
        CALL("strings.ReplaceAll", strings_ReplaceAll(0, S(0), S(1), S(2)))
        CALL("strings.Split", strings_Split(0, S(0), S(1)))
        CALL("strings.SplitAfter", strings_SplitAfter(0, S(0), S(1)))
        CALL("strings.SplitN", strings_SplitN(0, S(0), S(1), I(2)))
        CALL("strings.SplitAfterN", strings_SplitAfterN(0, S(0), S(1), I(2)))
        CALL("strings.Trim", strings_Trim(S(0), S(1)))
        CALL("strings.TrimLeft", strings_TrimLeft(S(0), S(1)))
        CALL("strings.TrimRight", strings_TrimRight(S(0), S(1)))
        CALL("strings.TrimPrefix", strings_TrimPrefix(S(0), S(1)))
        CALL("strings.TrimSuffix", strings_TrimSuffix(S(0), S(1)))
        CALL("strings.TrimSpace", strings_TrimSpace(S(0)))
        CALL("strings.Clone", strings_Clone(0, S(0)))
        CALL("bytes.Clone", bytes_Clone(0, B(0)))
        CALL("bytes.Compare", bytes_Compare(B(0), B(1)))
        CALL("bytes.Contains", bytes_Contains(B(0), B(1)))
        CALL("bytes.ContainsAny", bytes_ContainsAny(B(0), S(1)))
        CALL("bytes.ContainsRune", bytes_ContainsRune(B(0), I(1)))
        CALL("bytes.Count", bytes_Count(B(0), B(1)))
        CALL("bytes.Cut", bytes_Cut(B(0), B(1)))
        CALL("bytes.CutPrefix", bytes_CutPrefix(B(0), B(1)))
        CALL("bytes.CutSuffix", bytes_CutSuffix(B(0), B(1)))
        CALL("bytes.Equal", bytes_Equal(B(0), B(1)))
        CALL("bytes.Fields", bytes_Fields(0, B(0)))
        CALL("bytes.HasPrefix", bytes_HasPrefix(B(0), B(1)))
        CALL("bytes.HasSuffix", bytes_HasSuffix(B(0), B(1)))
        CALL("bytes.Index", bytes_Index(B(0), B(1)))
        CALL("bytes.IndexAny", bytes_IndexAny(B(0), S(1)))
        CALL("bytes.IndexByte", bytes_IndexByte(B(0), static_cast<uint8_t>(I(1))))
        CALL("bytes.IndexRune", bytes_IndexRune(B(0), I(1)))
        CALL("bytes.Join", bytes_Join(0, arg_lb(a.at(0)), B(1)))
        CALL("bytes.LastIndex", bytes_LastIndex(B(0), B(1)))
        CALL("bytes.LastIndexAny", bytes_LastIndexAny(B(0), S(1)))
        CALL("bytes.LastIndexByte", bytes_LastIndexByte(B(0), static_cast<uint8_t>(I(1))))
        CALL("bytes.Repeat", bytes_Repeat(0, 0, B(0), I(1)))
        CALL("bytes.Replace", bytes_Replace(0, B(0), B(1), B(2), I(3)))
        CALL("bytes.ReplaceAll", bytes_ReplaceAll(0, B(0), B(1), B(2)))
        CALL("bytes.Split", bytes_Split(0, B(0), B(1)))
        CALL("bytes.SplitAfter", bytes_SplitAfter(0, B(0), B(1)))
        CALL("bytes.SplitN", bytes_SplitN(0, B(0), B(1), I(2)))
        CALL("bytes.SplitAfterN", bytes_SplitAfterN(0, B(0), B(1), I(2)))
        CALL("bytes.Trim", bytes_Trim(B(0), S(1)))
        CALL("bytes.TrimLeft", bytes_TrimLeft(B(0), S(1)))
        CALL("bytes.TrimRight", bytes_TrimRight(B(0), S(1)))
        CALL("bytes.TrimPrefix", bytes_TrimPrefix(B(0), B(1)))
        CALL("bytes.TrimSuffix", bytes_TrimSuffix(B(0), B(1)))
        CALL("bytes.TrimSpace", bytes_TrimSpace(B(0)))
        CALL("utf8.DecodeRuneInString", utf8_DecodeRuneInString(S(0)))
        CALL("utf8.DecodeLastRuneInString", utf8_DecodeLastRuneInString(S(0)))
        CALL("utf8.DecodeRune", utf8_DecodeRune(B(0)))
        CALL("utf8.DecodeLastRune", utf8_DecodeLastRune(B(0)))
        CALL("utf8.RuneCountInString", utf8_RuneCountInString(S(0)))
        CALL("utf8.RuneCount", utf8_RuneCount(B(0)))
        CALL("utf8.ValidString", utf8_ValidString(S(0)))
        CALL("utf8.Valid", utf8_Valid(B(0)))
        CALL("utf8.ValidRune", utf8_ValidRune(I(0)))
        CALL("utf8.RuneLen", utf8_RuneLen(I(0)))
        CALL("utf8.AppendRune", utf8_AppendRune(0, B(0), I(1)))
        CALL("unicode.IsSpace", unicode_IsSpace(I(0)))
        CALL("range", [&] {  // for i, r := range s
            std::string got;
            for (int64_t i = 0, w = 0; i < S(0).len; i += w) {
                const Str s = S(0);
                int32_t c;
                w = decode_rune(s, i, c);
                got += std::to_string(i) + ":" + std::to_string(c) + ",";
            }
            return view(heap_bytes(std::vector<uint8_t>(got.begin(), got.end())));
        }())
        else {
            r = "unknown " + f;
        }
        std::cout << r << "\n";
    }
    return 0;
}
