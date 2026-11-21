// ---------------------------------------------------------------------------
// Go maps in worker code (pasted in by tessc): hash tables with open
// addressing on the worker's heap. A map value is a handle to its table, so
// copies share it, as in Go; the zero handle is a nil map.
//
// Iteration follows the table, so its order is fixed where Go's changes
// from run to run; Go promises no order, so programs cannot rely on either.
// ---------------------------------------------------------------------------
namespace tsr {

// Hashes of map keys: integers, bools, strings, arrays of those, and the
// program's structs (tessc writes a hash_key_ for each comparable struct).
inline uint32_t mix32(uint32_t h) {  // MurmurHash3's finalizer
    h ^= h >> 16;
    h *= 0x85ebca6bu;
    h ^= h >> 13;
    h *= 0xc2b2ae35u;
    h ^= h >> 16;
    return h;
}
inline uint32_t hash_more(uint32_t h, uint32_t v) { return mix32(h + 0x9e3779b9u + v); }
inline uint32_t hash_key_(uint64_t v) {
    return mix32(static_cast<uint32_t>(v) ^ mix32(static_cast<uint32_t>(v >> 32) + 0x9e3779b9u));
}
inline uint32_t hash_key_(int64_t v) { return hash_key_(static_cast<uint64_t>(v)); }
inline uint32_t hash_key_(uint32_t v) { return hash_key_(static_cast<uint64_t>(v)); }
inline uint32_t hash_key_(int32_t v) { return hash_key_(static_cast<uint64_t>(static_cast<int64_t>(v))); }
inline uint32_t hash_key_(uint16_t v) { return hash_key_(static_cast<uint64_t>(v)); }
inline uint32_t hash_key_(int16_t v) { return hash_key_(static_cast<uint64_t>(static_cast<int64_t>(v))); }
inline uint32_t hash_key_(uint8_t v) { return hash_key_(static_cast<uint64_t>(v)); }
inline uint32_t hash_key_(int8_t v) { return hash_key_(static_cast<uint64_t>(static_cast<int64_t>(v))); }
inline uint32_t hash_key_(bool v) { return hash_key_(static_cast<uint64_t>(v)); }
inline uint32_t hash_key_(const Iface& x) { return iface_hash(x); }
template <typename T>
inline uint32_t hash_key_(T* const& p) {  // a reference: an array must not decay to match
    return hash_key_(static_cast<uint64_t>(reinterpret_cast<uintptr_t>(p)));
}
TSR_OUTLINE inline uint32_t hash_key_(const Str& s) {
    uint32_t h = 2166136261u;  // FNV-1a
    for (int64_t i = 0; i < s.len; i++) {
        h = (h ^ s.ptr[i]) * 16777619u;
    }
    return hash_more(h, static_cast<uint32_t>(s.len));
}
template <typename T, decltype(sizeof(0)) N>
inline uint32_t hash_key_(const T (&a)[N]) {
    uint32_t h = 0;
    for (decltype(sizeof(0)) i = 0; i < N; i++) {
        h = hash_more(h, hash_key_(a[i]));
    }
    return h;
}

// T itself, where it must not be deduced: a map's key type comes from the
// map, and the key given converts to it (m[5] on a map[uint8]int).
template <typename T>
struct same {
    using type = T;
};

constexpr uint8_t kEmpty = 0, kFull = 1, kDeleted = 2;

template <typename K, typename V>
struct MapData {
    int64_t count;   // entries
    int64_t used;    // slots that are not empty: the entries and the deleted ones
    int64_t cap;     // slots: 0 or a power of two
    uint8_t* state;  // each slot's: kEmpty, kFull or kDeleted
    K* keys;
    V* vals;
};

template <typename K, typename V>
struct alignas(8) Map {  // 8 bytes, like a Go map, so structs keep Go's layout
    MapData<K, V>* d;
};

// The slot that holds k, or -1.
template <typename K, typename V>
TSR_OUTLINE inline int64_t map_find(const MapData<K, V>* d, const K& k) {
    if (d == nullptr || d->count == 0) {
        return -1;
    }
    const uint64_t mask = static_cast<uint64_t>(d->cap) - 1;
    uint64_t i = hash_key_(k) & mask;
    for (int64_t n = 0; n < d->cap; n++, i = (i + 1) & mask) {
        if (d->state[i] == kEmpty) {
            return -1;
        }
        if (d->state[i] == kFull && d->keys[i] == k) {
            return static_cast<int64_t>(i);
        }
    }
    return -1;
}

// Moves the entries to a table of cap slots (its old one stays where it is,
// for a range that began before).
template <typename K, typename V>
TSR_OUTLINE inline void map_rehash(MapData<K, V>* d, int64_t cap, int64_t pos) {
    const uint8_t* state = d->state;
    const K* keys = d->keys;
    const V* vals = d->vals;
    const int64_t old = d->cap;
    d->state = alloc<uint8_t>(cap, pos);
    d->keys = alloc<K>(cap, pos);
    d->vals = alloc<V>(cap, pos);
    d->cap = cap;
    d->used = d->count;
    const uint64_t mask = static_cast<uint64_t>(cap) - 1;
    for (int64_t j = 0; j < old; j++) {
        if (state[j] != kFull) {
            continue;
        }
        uint64_t i = hash_key_(keys[j]) & mask;
        while (d->state[i] == kFull) {
            i = (i + 1) & mask;
        }
        d->state[i] = kFull;
        d->keys[i] = keys[j];
        d->vals[i] = vals[j];
    }
}

// make(map[K]V, hint). Go ignores a hint below zero, and so does this.
template <typename K, typename V>
TSR_OUTLINE inline Map<K, V> make_map(int64_t hint, int64_t pos) {
    MapData<K, V>* d = alloc<MapData<K, V>>(1, pos);
    if (hint > 0) {
        int64_t cap = 8;
        while (hint * 4 > cap * 3) {
            cap *= 2;
        }
        map_rehash(d, cap, pos);
    }
    return Map<K, V>{d};
}

// m[k], read: the zero value if k is not in m.
template <typename K, typename V>
TSR_OUTLINE inline V map_get(const Map<K, V>& m, const typename same<K>::type& k) {
    const int64_t at = map_find(m.d, k);
    return at < 0 ? V{} : m.d->vals[at];
}

// v, ok := m[k].
template <typename K, typename V>
TSR_OUTLINE inline bool map_lookup(const Map<K, V>& m, const typename same<K>::type& k, V& v) {
    const int64_t at = map_find(m.d, k);
    v = at < 0 ? V{} : m.d->vals[at];
    return at >= 0;
}

// m[k] as a place to store: k's slot, made (holding the zero value) if k is
// not in m yet. Storing in a nil map panics, as in Go.
template <typename K, typename V>
TSR_OUTLINE inline V& map_ref(const Map<K, V>& m, const typename same<K>::type& k, int64_t pos_nil, int64_t pos) {
    MapData<K, V>* d = m.d;
    if (d == nullptr) {
        fail(pos_nil, 0);
    }
    const int64_t at = map_find(d, k);
    if (at >= 0) {
        return d->vals[at];
    }
    if ((d->used + 1) * 4 > d->cap * 3) {
        // Grow to at most half full, or (with many deleted slots) just tidy up.
        int64_t cap = d->cap == 0 ? 8 : d->cap;
        while ((d->count + 1) * 2 > cap) {
            cap *= 2;
        }
        map_rehash(d, cap, pos);
    }
    const uint64_t mask = static_cast<uint64_t>(d->cap) - 1;
    uint64_t i = hash_key_(k) & mask;
    while (d->state[i] == kFull) {
        i = (i + 1) & mask;
    }
    if (d->state[i] == kEmpty) {
        d->used++;
    }
    d->state[i] = kFull;
    d->keys[i] = k;
    d->vals[i] = V{};
    d->count++;
    return d->vals[i];
}

// delete(m, k): nothing if k is not in m, or m is nil.
template <typename K, typename V>
TSR_OUTLINE inline void map_delete(const Map<K, V>& m, const typename same<K>::type& k) {
    const int64_t at = map_find(m.d, k);
    if (at >= 0) {
        m.d->state[at] = kDeleted;
        m.d->vals[at] = V{};
        m.d->count--;
    }
}

template <typename K, typename V>
inline int64_t map_len(const Map<K, V>& m) {
    return m.d == nullptr ? 0 : m.d->count;
}

// clear(m).
template <typename K, typename V>
TSR_OUTLINE inline void map_clear(const Map<K, V>& m) {
    MapData<K, V>* d = m.d;
    if (d == nullptr) {
        return;
    }
    for (int64_t i = 0; i < d->cap; i++) {
        d->state[i] = kEmpty;
        d->vals[i] = V{};
    }
    d->count = 0;
    d->used = 0;
}

// for k, v := range m. Go's rules: an entry deleted before it is reached is
// not produced, one added during the loop may or may not be, and every other
// entry is produced once, even if the map grows meanwhile. So the iterator
// walks the table the map had when the loop began; once the map has moved
// to a new table, it looks each key up there.
template <typename K, typename V>
struct MapIter {
    MapData<K, V>* d;
    int64_t cap = 0;
    const uint8_t* state = nullptr;
    const K* keys = nullptr;
    const V* vals = nullptr;
    int64_t at = 0;
    explicit MapIter(const Map<K, V>& m) : d(m.d) {
        if (d != nullptr) {
            cap = d->cap;
            state = d->state;
            keys = d->keys;
            vals = d->vals;
        }
    }
    TSR_OUTLINE bool next(K& k, V& v) {
        while (at < cap) {
            const int64_t i = at++;
            if (d->keys == keys) {
                if (state[i] == kFull) {
                    k = keys[i];
                    v = vals[i];
                    return true;
                }
                continue;
            }
            if (state[i] != kFull) {
                continue;
            }
            const int64_t j = map_find(d, keys[i]);
            if (j >= 0) {
                k = keys[i];
                v = d->vals[j];
                return true;
            }
        }
        return false;
    }
};

}  // namespace tsr
