// ---------------------------------------------------------------------------
// The strings, bytes and unicode/utf8 functions that worker code can call
// (pasted in by tessc). Each follows Go's own code closely enough to give
// the same results: the same indices, the same pieces, and for bytes the
// same nil or empty slices and capacities. compiler/runtime_test.go checks
// them against Go on the host.
// ---------------------------------------------------------------------------
namespace tsr {

// ---- UTF-8 (the decoding itself is decode_rune, with the strings)

constexpr int64_t kMaxRune = 0x10FFFF;

inline bool utf8_ValidRune(int64_t r) { return (0 <= r && r < 0xD800) || (0xDFFF < r && r <= kMaxRune); }

TSR_OUTLINE inline int64_t utf8_RuneLen(int64_t r) {
    if (r < 0) {
        return -1;
    }
    if (r < 0x80) {
        return 1;
    }
    if (r < 0x800) {
        return 2;
    }
    if (0xD800 <= r && r <= 0xDFFF) {
        return -1;
    }
    if (r < 0x10000) {
        return 3;
    }
    return r <= kMaxRune ? 4 : -1;
}

inline bool utf8_RuneStart(uint8_t b) { return (b & 0xC0) != 0x80; }

struct RuneSize {
    int32_t r0;
    int64_t r1;
};

TSR_OUTLINE inline RuneSize utf8_DecodeRuneInString(const Str& s) {
    if (s.len == 0) {
        return RuneSize{kRuneError, 0};
    }
    int32_t r;
    const int64_t n = decode_rune(s, 0, r);
    return RuneSize{r, n};
}

// The last rune of s and its width, as utf8.DecodeLastRuneInString finds it.
TSR_OUTLINE inline RuneSize utf8_DecodeLastRuneInString(const Str& s) {
    const int64_t end = s.len;
    if (end == 0) {
        return RuneSize{kRuneError, 0};
    }
    int64_t start = end - 1;
    if (s.ptr[start] < 0x80) {
        return RuneSize{s.ptr[start], 1};
    }
    const int64_t lim = end - 4 < 0 ? 0 : end - 4;
    for (start--; start >= lim; start--) {
        if (utf8_RuneStart(s.ptr[start])) {
            break;
        }
    }
    if (start < 0) {
        start = 0;
    }
    int32_t r;
    const int64_t size = decode_rune(Str{s.ptr + start, end - start}, 0, r);
    if (start + size != end) {
        return RuneSize{kRuneError, 1};
    }
    return RuneSize{r, size};
}

TSR_OUTLINE inline int64_t utf8_RuneCountInString(const Str& s) {
    int64_t n = 0;
    int32_t r;
    for (int64_t i = 0; i < s.len; i += decode_rune(s, i, r)) {
        n++;
    }
    return n;
}

TSR_OUTLINE inline bool utf8_ValidString(const Str& s) {
    int32_t r;
    for (int64_t i = 0; i < s.len;) {
        const int64_t w = decode_rune(s, i, r);
        if (r == kRuneError && w == 1) {
            return false;
        }
        i += w;
    }
    return true;
}

inline RuneSize utf8_DecodeRune(const Slice<uint8_t>& p) { return utf8_DecodeRuneInString(view(p)); }
inline RuneSize utf8_DecodeLastRune(const Slice<uint8_t>& p) { return utf8_DecodeLastRuneInString(view(p)); }
inline int64_t utf8_RuneCount(const Slice<uint8_t>& p) { return utf8_RuneCountInString(view(p)); }
inline bool utf8_Valid(const Slice<uint8_t>& p) { return utf8_ValidString(view(p)); }

// utf8.EncodeRune: like Go, it panics (index out of range) when p is short.
TSR_OUTLINE inline int64_t utf8_EncodeRune(int64_t pos, const Slice<uint8_t>& p, int64_t r) {
    uint8_t buf[4];
    const int64_t n = encode_rune(buf, r);
    for (int64_t i = n - 1; i >= 0; i--) {  // Go writes the last byte first
        p.at(i, pos) = buf[i];
    }
    return n;
}

TSR_OUTLINE inline Slice<uint8_t> utf8_AppendRune(int64_t pos, const Slice<uint8_t>& p, int64_t r) {
    uint8_t buf[4];
    const int64_t n = encode_rune(buf, r);
    return append_all<true>(pos, p, Slice<uint8_t>{buf, n, n});
}

// unicode.IsSpace: Unicode's White_Space property.
inline bool unicode_IsSpace(int64_t r) {
    switch (r) {
    case '\t':
    case '\n':
    case '\v':
    case '\f':
    case '\r':
    case ' ':
    case 0x85:
    case 0xA0:
    case 0x1680:
    case 0x2028:
    case 0x2029:
    case 0x202F:
    case 0x205F:
    case 0x3000:
        return true;
    }
    return 0x2000 <= r && r <= 0x200A;
}

// ---- searching, on byte strings (both packages)

TSR_OUTLINE inline int64_t text_index_byte(const Str& s, uint8_t c) {
    for (int64_t i = 0; i < s.len; i++) {
        if (s.ptr[i] == c) {
            return i;
        }
    }
    return -1;
}

TSR_OUTLINE inline int64_t text_last_index_byte(const Str& s, uint8_t c) {
    for (int64_t i = s.len - 1; i >= 0; i--) {
        if (s.ptr[i] == c) {
            return i;
        }
    }
    return -1;
}

inline bool text_has_prefix(const Str& s, const Str& p) { return s.len >= p.len && Str{s.ptr, p.len} == p; }
inline bool text_has_suffix(const Str& s, const Str& p) {
    return s.len >= p.len && Str{s.ptr + s.len - p.len, p.len} == p;
}

TSR_OUTLINE inline int64_t text_index(const Str& s, const Str& sep) {
    const int64_t n = sep.len;
    if (n == 0) {
        return 0;
    }
    if (n == 1) {
        return text_index_byte(s, sep.ptr[0]);
    }
    const uint8_t c0 = sep.ptr[0];
    for (int64_t i = 0; i + n <= s.len; i++) {
        if (s.ptr[i] == c0 && Str{s.ptr + i, n} == sep) {
            return i;
        }
    }
    return -1;
}

TSR_OUTLINE inline int64_t text_last_index(const Str& s, const Str& sep) {
    const int64_t n = sep.len;
    if (n == 0) {
        return s.len;
    }
    if (n == 1) {
        return text_last_index_byte(s, sep.ptr[0]);
    }
    for (int64_t i = s.len - n; i >= 0; i--) {
        if (Str{s.ptr + i, n} == sep) {
            return i;
        }
    }
    return -1;
}

TSR_OUTLINE inline int64_t text_count(const Str& s, const Str& sep) {
    if (sep.len == 0) {
        return utf8_RuneCountInString(s) + 1;
    }
    int64_t n = 0;
    if (sep.len == 1) {
        for (int64_t i = 0; i < s.len; i++) {
            n += s.ptr[i] == sep.ptr[0];
        }
        return n;
    }
    Str t = s;
    for (;;) {
        const int64_t i = text_index(t, sep);
        if (i < 0) {
            return n;
        }
        n++;
        t = Str{t.ptr + i + sep.len, t.len - i - sep.len};
    }
}

TSR_OUTLINE inline int64_t text_index_rune(const Str& s, int64_t r) {
    if (0 <= r && r < 0x80) {
        return text_index_byte(s, static_cast<uint8_t>(r));
    }
    if (r == kRuneError) {
        int32_t c;
        for (int64_t i = 0, w = 0; i < s.len; i += w) {
            w = decode_rune(s, i, c);
            if (c == kRuneError) {
                return i;
            }
        }
        return -1;
    }
    if (!utf8_ValidRune(r)) {
        return -1;
    }
    uint8_t buf[4];
    const int64_t n = encode_rune(buf, r);
    return text_index(s, Str{buf, n});
}

TSR_OUTLINE inline int64_t text_index_any(const Str& s, const Str& chars) {
    if (chars.len == 0) {
        return -1;
    }
    int32_t c;
    for (int64_t i = 0, w = 0; i < s.len; i += w) {
        w = decode_rune(s, i, c);
        if (text_index_rune(chars, c) >= 0) {
            return i;
        }
    }
    return -1;
}

TSR_OUTLINE inline int64_t text_last_index_any(const Str& s, const Str& chars) {
    if (chars.len == 0) {
        return -1;
    }
    for (int64_t i = s.len; i > 0;) {
        const RuneSize rs = utf8_DecodeLastRuneInString(Str{s.ptr, i});
        i -= rs.r1;
        if (text_index_rune(chars, rs.r0) >= 0) {
            return i;
        }
    }
    return -1;
}

// What a trim keeps of s: [lo, hi).
struct Span {
    int64_t lo, hi;
};

TSR_OUTLINE inline Span text_trim_left(const Str& s, const Str& cutset, Span sp) {
    while (sp.lo < sp.hi) {
        int32_t r = s.ptr[sp.lo];
        int64_t n = 1;
        if (r >= 0x80) {
            n = decode_rune(Str{s.ptr, sp.hi}, sp.lo, r);
        }
        if (text_index_rune(cutset, r) < 0) {
            break;
        }
        sp.lo += n;
    }
    return sp;
}

TSR_OUTLINE inline Span text_trim_right(const Str& s, const Str& cutset, Span sp) {
    while (sp.lo < sp.hi) {
        int32_t r = s.ptr[sp.hi - 1];
        int64_t n = 1;
        if (r >= 0x80) {
            const RuneSize rs = utf8_DecodeLastRuneInString(Str{s.ptr + sp.lo, sp.hi - sp.lo});
            r = rs.r0;
            n = rs.r1;
        }
        if (text_index_rune(cutset, r) < 0) {
            break;
        }
        sp.hi -= n;
    }
    return sp;
}

TSR_OUTLINE inline Span text_trim_space(const Str& s) {
    Span sp{0, s.len};
    int32_t r;
    while (sp.lo < sp.hi) {
        const int64_t n = decode_rune(Str{s.ptr, sp.hi}, sp.lo, r);
        if (!unicode_IsSpace(r)) {
            break;
        }
        sp.lo += n;
    }
    while (sp.lo < sp.hi) {
        const RuneSize rs = utf8_DecodeLastRuneInString(Str{s.ptr + sp.lo, sp.hi - sp.lo});
        if (!unicode_IsSpace(rs.r0)) {
            break;
        }
        sp.hi -= rs.r1;
    }
    return sp;
}

// The fields of s: the runs of runes that are not spaces.
template <typename F>
TSR_OUTLINE inline int64_t text_fields(const Str& s, F&& field) {
    int64_t n = 0, start = -1;
    int32_t r;
    for (int64_t i = 0, w = 0; i < s.len; i += w) {
        w = decode_rune(s, i, r);
        if (unicode_IsSpace(r)) {
            if (start >= 0) {
                field(n++, start, i);
                start = -1;
            }
        } else if (start < 0) {
            start = i;
        }
    }
    if (start >= 0) {
        field(n++, start, s.len);
    }
    return n;
}

// ---- package strings

inline int64_t strings_Compare(const Str& a, const Str& b) { return compare(a, b); }
inline bool strings_Contains(const Str& s, const Str& sub) { return text_index(s, sub) >= 0; }
inline bool strings_ContainsAny(const Str& s, const Str& chars) { return text_index_any(s, chars) >= 0; }
inline bool strings_ContainsRune(const Str& s, int64_t r) { return text_index_rune(s, r) >= 0; }
inline int64_t strings_Count(const Str& s, const Str& sep) { return text_count(s, sep); }
inline bool strings_HasPrefix(const Str& s, const Str& p) { return text_has_prefix(s, p); }
inline bool strings_HasSuffix(const Str& s, const Str& p) { return text_has_suffix(s, p); }
inline int64_t strings_Index(const Str& s, const Str& sep) { return text_index(s, sep); }
inline int64_t strings_IndexAny(const Str& s, const Str& chars) { return text_index_any(s, chars); }
inline int64_t strings_IndexByte(const Str& s, uint8_t c) { return text_index_byte(s, c); }
inline int64_t strings_IndexRune(const Str& s, int64_t r) { return text_index_rune(s, r); }
inline int64_t strings_LastIndex(const Str& s, const Str& sep) { return text_last_index(s, sep); }
inline int64_t strings_LastIndexAny(const Str& s, const Str& chars) { return text_last_index_any(s, chars); }
inline int64_t strings_LastIndexByte(const Str& s, uint8_t c) { return text_last_index_byte(s, c); }

inline Str strings_Clone(int64_t pos, const Str& s) {
    return s.len == 0 ? Str{nullptr, 0} : to_string(Slice<uint8_t>{const_cast<uint8_t*>(s.ptr), s.len, s.len}, pos);
}

struct StrCut {
    Str r0, r1;
    bool r2;
};
struct StrFound {
    Str r0;
    bool r1;
};

TSR_OUTLINE inline StrCut strings_Cut(const Str& s, const Str& sep) {
    const int64_t i = text_index(s, sep);
    if (i >= 0) {
        return StrCut{Str{s.ptr, i}, Str{s.ptr + i + sep.len, s.len - i - sep.len}, true};
    }
    return StrCut{s, Str{nullptr, 0}, false};
}
TSR_OUTLINE inline StrFound strings_CutPrefix(const Str& s, const Str& p) {
    if (!text_has_prefix(s, p)) {
        return StrFound{s, false};
    }
    return StrFound{Str{s.ptr + p.len, s.len - p.len}, true};
}
TSR_OUTLINE inline StrFound strings_CutSuffix(const Str& s, const Str& p) {
    if (!text_has_suffix(s, p)) {
        return StrFound{s, false};
    }
    return StrFound{Str{s.ptr, s.len - p.len}, true};
}

inline Str strings_TrimPrefix(const Str& s, const Str& p) { return strings_CutPrefix(s, p).r0; }
inline Str strings_TrimSuffix(const Str& s, const Str& p) { return strings_CutSuffix(s, p).r0; }

inline Str text_span(const Str& s, Span sp) { return Str{s.ptr + sp.lo, sp.hi - sp.lo}; }

TSR_OUTLINE inline Str strings_Trim(const Str& s, const Str& cutset) {
    if (s.len == 0 || cutset.len == 0) {
        return s;
    }
    return text_span(s, text_trim_left(s, cutset, text_trim_right(s, cutset, Span{0, s.len})));
}
TSR_OUTLINE inline Str strings_TrimLeft(const Str& s, const Str& cutset) {
    if (s.len == 0 || cutset.len == 0) {
        return s;
    }
    return text_span(s, text_trim_left(s, cutset, Span{0, s.len}));
}
TSR_OUTLINE inline Str strings_TrimRight(const Str& s, const Str& cutset) {
    if (s.len == 0 || cutset.len == 0) {
        return s;
    }
    return text_span(s, text_trim_right(s, cutset, Span{0, s.len}));
}
inline Str strings_TrimSpace(const Str& s) { return text_span(s, text_trim_space(s)); }

TSR_OUTLINE inline Slice<Str> strings_Fields(int64_t pos, const Str& s) {
    const int64_t n = text_fields(s, [](int64_t, int64_t, int64_t) {});
    Slice<Str> a = make<Str>(n, pos, pos);
    text_fields(s, [&](int64_t k, int64_t lo, int64_t hi) { a.ptr[k] = Str{s.ptr + lo, hi - lo}; });
    return a;
}

TSR_OUTLINE inline Str strings_Repeat(int64_t pos, int64_t pos_neg, const Str& s, int64_t count) {
    if (count == 0) {
        return Str{nullptr, 0};
    }
    if (count < 0) {
        fail(pos_neg, count);
    }
    if (s.len == 0) {
        return Str{nullptr, 0};
    }
    if (count > INT64_MAX / s.len) {
        fail(pos, count);
    }
    uint8_t* p = alloc<uint8_t>(s.len * count, pos);
    for (int64_t k = 0; k < count; k++) {
        move_bytes(p + k * s.len, s.ptr, s.len);
    }
    return Str{p, s.len * count};
}

TSR_OUTLINE inline Str strings_Join(int64_t pos, const Slice<Str>& elems, const Str& sep) {
    if (elems.len == 0) {
        return Str{nullptr, 0};
    }
    if (elems.len == 1) {
        return elems.ptr[0];
    }
    int64_t n = sep.len * (elems.len - 1);
    for (int64_t i = 0; i < elems.len; i++) {
        n += elems.ptr[i].len;
    }
    uint8_t* p = alloc<uint8_t>(n, pos);
    int64_t at = 0;
    for (int64_t i = 0; i < elems.len; i++) {
        if (i > 0) {
            move_bytes(p + at, sep.ptr, sep.len);
            at += sep.len;
        }
        move_bytes(p + at, elems.ptr[i].ptr, elems.ptr[i].len);
        at += elems.ptr[i].len;
    }
    return Str{p, n};
}

// strings.Replace: how many replacements, and where, as Go decides.
template <typename Out>
TSR_OUTLINE inline void text_replace(const Str& s, const Str& old, const Str& repl, int64_t n, Out&& out) {
    int64_t start = 0;
    for (int64_t i = 0; i < n; i++) {
        int64_t j = start;
        if (old.len == 0) {
            if (i > 0) {
                int32_t r;
                j += decode_rune(Str{s.ptr, s.len}, start, r);
            }
        } else {
            j += text_index(Str{s.ptr + start, s.len - start}, old);
        }
        out(Str{s.ptr + start, j - start});
        out(repl);
        start = j + old.len;
    }
    out(Str{s.ptr + start, s.len - start});
}

TSR_OUTLINE inline Str strings_Replace(int64_t pos, const Str& s, const Str& old, const Str& repl, int64_t n) {
    if (old == repl || n == 0) {
        return s;
    }
    const int64_t m = text_count(s, old);
    if (m == 0) {
        return s;
    }
    if (n < 0 || m < n) {
        n = m;
    }
    uint8_t* p = alloc<uint8_t>(s.len + n * (repl.len - old.len), pos);
    int64_t at = 0;
    text_replace(s, old, repl, n, [&](const Str& part) {
        move_bytes(p + at, part.ptr, part.len);
        at += part.len;
    });
    return Str{p, at};
}
inline Str strings_ReplaceAll(int64_t pos, const Str& s, const Str& old, const Str& repl) {
    return strings_Replace(pos, s, old, repl, -1);
}

// strings.Split and its relatives (Go's genSplit and explode).
TSR_OUTLINE inline Slice<Str> strings_split(int64_t pos, Str s, const Str& sep, int64_t save, int64_t n) {
    if (n == 0) {
        return Slice<Str>{nullptr, 0, 0};
    }
    if (sep.len == 0) {
        const int64_t l = utf8_RuneCountInString(s);
        if (n < 0 || n > l) {
            n = l;
        }
        Slice<Str> a = make<Str>(n, pos, pos);
        for (int64_t i = 0; i < n - 1; i++) {
            int32_t r;
            const int64_t size = decode_rune(s, 0, r);
            a.ptr[i] = Str{s.ptr, size};
            s = Str{s.ptr + size, s.len - size};
        }
        if (n > 0) {
            a.ptr[n - 1] = s;
        }
        return a;
    }
    if (n < 0) {
        n = text_count(s, sep) + 1;
    }
    if (n > s.len + 1) {
        n = s.len + 1;
    }
    Slice<Str> a = make<Str>(n, pos, pos);
    n--;
    int64_t i = 0;
    while (i < n) {
        const int64_t m = text_index(s, sep);
        if (m < 0) {
            break;
        }
        a.ptr[i] = Str{s.ptr, m + save};
        s = Str{s.ptr + m + sep.len, s.len - m - sep.len};
        i++;
    }
    a.ptr[i] = s;
    return Slice<Str>{a.ptr, i + 1, a.cap};
}
inline Slice<Str> strings_Split(int64_t pos, const Str& s, const Str& sep) { return strings_split(pos, s, sep, 0, -1); }
inline Slice<Str> strings_SplitAfter(int64_t pos, const Str& s, const Str& sep) {
    return strings_split(pos, s, sep, sep.len, -1);
}
inline Slice<Str> strings_SplitN(int64_t pos, const Str& s, const Str& sep, int64_t n) {
    return strings_split(pos, s, sep, 0, n);
}
inline Slice<Str> strings_SplitAfterN(int64_t pos, const Str& s, const Str& sep, int64_t n) {
    return strings_split(pos, s, sep, sep.len, n);
}

// ---- package bytes: the same, on slices, with Go's nil and empty results

using Bytes8 = Slice<uint8_t>;

inline Bytes8 nil_bytes() { return Bytes8{nullptr, 0, 0}; }
// s[lo:hi] and s[lo:hi:hi].
inline Bytes8 bsub(const Bytes8& s, int64_t lo, int64_t hi) { return Bytes8{s.ptr + lo, hi - lo, s.cap - lo}; }
inline Bytes8 bsub3(const Bytes8& s, int64_t lo, int64_t hi) { return Bytes8{s.ptr + lo, hi - lo, hi - lo}; }

inline bool bytes_Equal(const Bytes8& a, const Bytes8& b) { return view(a) == view(b); }
inline int64_t bytes_Compare(const Bytes8& a, const Bytes8& b) { return compare(view(a), view(b)); }
inline bool bytes_Contains(const Bytes8& s, const Bytes8& sub) { return text_index(view(s), view(sub)) >= 0; }
inline bool bytes_ContainsAny(const Bytes8& s, const Str& chars) { return text_index_any(view(s), chars) >= 0; }
inline bool bytes_ContainsRune(const Bytes8& s, int64_t r) { return text_index_rune(view(s), r) >= 0; }
inline int64_t bytes_Count(const Bytes8& s, const Bytes8& sep) { return text_count(view(s), view(sep)); }
inline bool bytes_HasPrefix(const Bytes8& s, const Bytes8& p) { return text_has_prefix(view(s), view(p)); }
inline bool bytes_HasSuffix(const Bytes8& s, const Bytes8& p) { return text_has_suffix(view(s), view(p)); }
inline int64_t bytes_Index(const Bytes8& s, const Bytes8& sep) { return text_index(view(s), view(sep)); }
inline int64_t bytes_IndexAny(const Bytes8& s, const Str& chars) { return text_index_any(view(s), chars); }
inline int64_t bytes_IndexByte(const Bytes8& s, uint8_t c) { return text_index_byte(view(s), c); }
inline int64_t bytes_IndexRune(const Bytes8& s, int64_t r) { return text_index_rune(view(s), r); }
inline int64_t bytes_LastIndex(const Bytes8& s, const Bytes8& sep) { return text_last_index(view(s), view(sep)); }
inline int64_t bytes_LastIndexAny(const Bytes8& s, const Str& chars) { return text_last_index_any(view(s), chars); }
inline int64_t bytes_LastIndexByte(const Bytes8& s, uint8_t c) { return text_last_index_byte(view(s), c); }

TSR_OUTLINE inline Bytes8 bytes_Clone(int64_t pos, const Bytes8& b) {
    if (b.ptr == nullptr) {
        return nil_bytes();
    }
    return append_all<true>(pos, literal<uint8_t>(pos), b);
}

struct BytesCut {
    Bytes8 r0, r1;
    bool r2;
};
struct BytesFound {
    Bytes8 r0;
    bool r1;
};

TSR_OUTLINE inline BytesCut bytes_Cut(const Bytes8& s, const Bytes8& sep) {
    const int64_t i = text_index(view(s), view(sep));
    if (i >= 0) {
        return BytesCut{bsub(s, 0, i), bsub(s, i + sep.len, s.len), true};
    }
    return BytesCut{s, nil_bytes(), false};
}
TSR_OUTLINE inline BytesFound bytes_CutPrefix(const Bytes8& s, const Bytes8& p) {
    if (!text_has_prefix(view(s), view(p))) {
        return BytesFound{s, false};
    }
    return BytesFound{bsub(s, p.len, s.len), true};
}
TSR_OUTLINE inline BytesFound bytes_CutSuffix(const Bytes8& s, const Bytes8& p) {
    if (!text_has_suffix(view(s), view(p))) {
        return BytesFound{s, false};
    }
    return BytesFound{bsub(s, 0, s.len - p.len), true};
}
inline Bytes8 bytes_TrimPrefix(const Bytes8& s, const Bytes8& p) { return bytes_CutPrefix(s, p).r0; }
inline Bytes8 bytes_TrimSuffix(const Bytes8& s, const Bytes8& p) { return bytes_CutSuffix(s, p).r0; }

// What bytes' trims return: nil instead of an empty slice where Go does.
inline Bytes8 bspan(const Bytes8& s, Span sp, bool nil_if_empty) {
    if (sp.lo == sp.hi && nil_if_empty) {
        return nil_bytes();
    }
    return bsub(s, sp.lo, sp.hi);
}

TSR_OUTLINE inline Bytes8 bytes_Trim(const Bytes8& s, const Str& cutset) {
    if (s.len == 0) {
        return nil_bytes();
    }
    if (cutset.len == 0) {
        return s;
    }
    return bspan(s, text_trim_left(view(s), cutset, text_trim_right(view(s), cutset, Span{0, s.len})), true);
}
TSR_OUTLINE inline Bytes8 bytes_TrimLeft(const Bytes8& s, const Str& cutset) {
    if (s.len == 0) {
        return nil_bytes();
    }
    if (cutset.len == 0) {
        return s;
    }
    return bspan(s, text_trim_left(view(s), cutset, Span{0, s.len}), true);
}
TSR_OUTLINE inline Bytes8 bytes_TrimRight(const Bytes8& s, const Str& cutset) {
    if (s.len == 0 || cutset.len == 0) {
        return s;
    }
    return bspan(s, text_trim_right(view(s), cutset, Span{0, s.len}), false);
}
inline Bytes8 bytes_TrimSpace(const Bytes8& s) { return bspan(s, text_trim_space(view(s)), true); }

TSR_OUTLINE inline Slice<Bytes8> bytes_Fields(int64_t pos, const Bytes8& s) {
    const int64_t n = text_fields(view(s), [](int64_t, int64_t, int64_t) {});
    Slice<Bytes8> a = make<Bytes8>(n, pos, pos);
    text_fields(view(s), [&](int64_t k, int64_t lo, int64_t hi) { a.ptr[k] = bsub3(s, lo, hi); });
    return a;
}

TSR_OUTLINE inline Bytes8 bytes_Repeat(int64_t pos, int64_t pos_neg, const Bytes8& b, int64_t count) {
    if (count == 0) {
        return literal<uint8_t>(pos);
    }
    if (count < 0) {
        fail(pos_neg, count);
    }
    if (b.len != 0 && count > INT64_MAX / b.len) {
        fail(pos, count);
    }
    if (b.len == 0) {
        return literal<uint8_t>(pos);
    }
    const int64_t n = b.len * count;
    Bytes8 r = make<uint8_t>(n, pos, pos);
    for (int64_t k = 0; k < count; k++) {
        move_bytes(r.ptr + k * b.len, b.ptr, b.len);
    }
    return r;
}

TSR_OUTLINE inline Bytes8 bytes_Join(int64_t pos, const Slice<Bytes8>& s, const Bytes8& sep) {
    if (s.len == 0) {
        return literal<uint8_t>(pos);
    }
    if (s.len == 1) {
        return append_all<true>(pos, nil_bytes(), s.ptr[0]);
    }
    int64_t n = sep.len * (s.len - 1);
    for (int64_t i = 0; i < s.len; i++) {
        n += s.ptr[i].len;
    }
    Bytes8 b = make<uint8_t>(n, pos, pos);
    int64_t at = 0;
    for (int64_t i = 0; i < s.len; i++) {
        if (i > 0) {
            move_bytes(b.ptr + at, sep.ptr, sep.len);
            at += sep.len;
        }
        move_bytes(b.ptr + at, s.ptr[i].ptr, s.ptr[i].len);
        at += s.ptr[i].len;
    }
    return b;
}

TSR_OUTLINE inline Bytes8 bytes_Replace(int64_t pos, const Bytes8& s, const Bytes8& old, const Bytes8& repl, int64_t n) {
    int64_t m = 0;
    if (n != 0) {
        m = text_count(view(s), view(old));
    }
    if (m == 0) {
        return append_all<true>(pos, nil_bytes(), s);  // a copy
    }
    if (n < 0 || m < n) {
        n = m;
    }
    Bytes8 t = make<uint8_t>(s.len + n * (repl.len - old.len), pos, pos);
    int64_t w = 0;
    text_replace(view(s), view(old), view(repl), n, [&](const Str& part) {
        move_bytes(t.ptr + w, part.ptr, part.len);
        w += part.len;
    });
    return Bytes8{t.ptr, w, t.cap};
}
inline Bytes8 bytes_ReplaceAll(int64_t pos, const Bytes8& s, const Bytes8& old, const Bytes8& repl) {
    return bytes_Replace(pos, s, old, repl, -1);
}

TSR_OUTLINE inline Slice<Bytes8> bytes_split(int64_t pos, Bytes8 s, const Bytes8& sep, int64_t save, int64_t n) {
    if (n == 0) {
        return Slice<Bytes8>{nullptr, 0, 0};
    }
    if (sep.len == 0) {
        if (n <= 0 || n > s.len) {
            n = s.len;
        }
        Slice<Bytes8> a = make<Bytes8>(n, pos, pos);
        int64_t na = 0;
        while (s.len > 0) {
            if (na + 1 >= n) {
                a.ptr[na++] = s;
                break;
            }
            int32_t r;
            const int64_t size = decode_rune(view(s), 0, r);
            a.ptr[na++] = bsub3(s, 0, size);
            s = bsub(s, size, s.len);
        }
        return Slice<Bytes8>{a.ptr, na, a.cap};
    }
    if (n < 0) {
        n = text_count(view(s), view(sep)) + 1;
    }
    if (n > s.len + 1) {
        n = s.len + 1;
    }
    Slice<Bytes8> a = make<Bytes8>(n, pos, pos);
    n--;
    int64_t i = 0;
    while (i < n) {
        const int64_t m = text_index(view(s), view(sep));
        if (m < 0) {
            break;
        }
        a.ptr[i] = bsub3(s, 0, m + save);
        s = bsub(s, m + sep.len, s.len);
        i++;
    }
    a.ptr[i] = s;
    return Slice<Bytes8>{a.ptr, i + 1, a.cap};
}
inline Slice<Bytes8> bytes_Split(int64_t pos, const Bytes8& s, const Bytes8& sep) { return bytes_split(pos, s, sep, 0, -1); }
inline Slice<Bytes8> bytes_SplitAfter(int64_t pos, const Bytes8& s, const Bytes8& sep) {
    return bytes_split(pos, s, sep, sep.len, -1);
}
inline Slice<Bytes8> bytes_SplitN(int64_t pos, const Bytes8& s, const Bytes8& sep, int64_t n) {
    return bytes_split(pos, s, sep, 0, n);
}
inline Slice<Bytes8> bytes_SplitAfterN(int64_t pos, const Bytes8& s, const Bytes8& sep, int64_t n) {
    return bytes_split(pos, s, sep, sep.len, n);
}

}  // namespace tsr
