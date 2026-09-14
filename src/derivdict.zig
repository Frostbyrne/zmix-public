//! Arm E′ dict-builder (`-Dderivdict`, comptime-dead at default): rebuild
//! english.dic BIT-EXACTLY from the corpus + a small recipe, so comp9 can
//! DERIVE the dictionary it currently CARRIES (Form-1 charges carried assets
//! twice, derived assets once —,
//! Arm E′: |R_E′| = 65,764 coded vs comp_dict 100,838).
//!
//! Recipe blob layout (= the measured artifact from the jul-31 lab, make_e2.py):
//!   [0, K/8)   membership bitmap over the K=200,000 candidate pool
//!              (pool = `[a-z]+` tokens of ASCII-case-folded corpus, ordered
//!              count desc then token asc, first K; bit i = pool[i] ∈ dict)
//!   then       varint straggler count, then per straggler varint(len) ++ bytes
//!              (dict words outside the pool, in dict order)
//!   then       permutation stream over S = sorted(selected ∪ stragglers)
//!              byte-lexicographically: token 0x00 ⇒ restart, varint(v+1);
//!              otherwise varint(v − prev) (ascending-run continuation)
//! Output = words of S in permuted order, each '\n'-terminated == english.dic.

const std = @import("std");

pub const K: usize = 200_000;
const BITMAP_LEN: usize = K / 8;

// ---- Frozen bit-exactness fingerprint of the dictionary this recipe rebuilds.
// The whole Arm E′ trade is "derive instead of carry", and it is only legal if
// the derived bytes are the SAME bytes: the payload was coded against
// english.dic, so a one-byte reconstruction drift produces an archive9 that
// cannot decode. The E7 functional gate measured the real builder against the
// real golden ("Functional gate —
// PASS": 411,996 B, sha256 4c8568cc…d36215a, `cmp`-identical); this is that
// same identity, re-derived from the tracked golden at integration time and
// baked in so the SHIPPED encoder re-proves it on every run instead of
// trusting a historical report.
//
//   src/models/fxcm26/goldens/english.dic
//   len    411,996
//   sha256 4c8568cca9343b9a6212477880f56f8efd162f8784224a25edd043097d36215a
//
// Wyhash (not sha256) is the on-line check because Wyhash is ALREADY in this
// binary (the census map above uses it) — the assert therefore costs one u64
// constant and a call, not a whole hash implementation. It guards against
// reconstruction drift, not against an adversary.
pub const GOLDEN_DICT_LEN: usize = 411_996;
pub const GOLDEN_DICT_WYHASH: u64 = 0x02e3e91dc8c4784e;

pub const DerivDictError = error{
    RecipeCorrupt,
    CorpusVocabTooSmall,
    /// The rebuilt dictionary is not the golden english.dic. NEVER continue:
    /// the payload would be coded against a dictionary the decoder cannot
    /// reproduce, i.e. a silently unrecoverable archive9.
    DerivDictMismatch,
    OutOfMemory,
};

// Fixed-capacity open-addressed census map instead of std.StringHashMap:
// the generic map machinery (getOrPut/grow/iterator instantiations) measured
// ~+1.4K packed on the ship binary. enwik9 has 1,171,106 distinct [a-z]+
// tokens; 2^22 slots = 3.58x headroom, load-guarded (CorpusVocabTooSmall is
// impossible to confuse with overflow: we fail hard at 3/4 load instead of
// growing). Keys are (offset,len) into the folded corpus; hash = Wyhash
// (already in the binary via dictionary_lex's StringHashMap).
const MAP_BITS: u6 = 22;
const MAP_CAP: usize = 1 << MAP_BITS;
const MAP_MASK: u64 = MAP_CAP - 1;
const Slot = struct { off: u32, len: u32, cnt: u32 }; // len == 0 => empty

fn readVarint(buf: []const u8, pos: *usize) DerivDictError!u64 {
    var v: u64 = 0;
    var sh: u6 = 0;
    while (true) {
        if (pos.* >= buf.len) return error.RecipeCorrupt;
        const b = buf[pos.*];
        pos.* += 1;
        v |= @as(u64, b & 0x7F) << sh;
        if (b & 0x80 == 0) return v;
        if (sh >= 56) return error.RecipeCorrupt;
        sh += 7;
    }
}

// Tiny handwritten heapsort instead of std.sort.pdq: the two pdq
// instantiations measured +33,416 raw / ~+16K packed on the ship binary
// (attribution arm armATTR2 in the E7 lab) — 91% of the whole fee. Heapsort
// is O(n log n), in-place, ~30 lines; the comparator is a TOTAL order (all
// keys distinct), so the sorted result is unique regardless of algorithm.
fn siftDown(comptime T: type, items: []T, start: usize, end: usize, comptime less: fn (void, T, T) bool) void {
    var root = start;
    while (true) {
        var child = root * 2 + 1;
        if (child >= end) return;
        if (child + 1 < end and less({}, items[child], items[child + 1])) child += 1;
        if (!less({}, items[root], items[child])) return;
        std.mem.swap(T, &items[root], &items[child]);
        root = child;
    }
}

fn heapSort(comptime T: type, items: []T, comptime less: fn (void, T, T) bool) void {
    if (items.len < 2) return;
    var start = items.len / 2;
    while (start > 0) {
        start -= 1;
        siftDown(T, items, start, items.len, less);
    }
    var end = items.len;
    while (end > 1) {
        end -= 1;
        std.mem.swap(T, &items[0], &items[end]);
        siftDown(T, items, 0, end, less);
    }
}

pub const Entry = struct { tok: []const u8, cnt: u32 };

fn entryLess(_: void, a: Entry, b: Entry) bool {
    if (a.cnt != b.cnt) return a.cnt > b.cnt; // count desc
    return std.mem.lessThan(u8, a.tok, b.tok); // then token asc
}

/// Census `corpus` (FOLDED IN PLACE: A–Z → a–z; the caller owns and must not
/// reuse the buffer as original content) and return the canonical vocabulary
/// order — count desc, token asc, a TOTAL order, so the result is unique
/// regardless of sort algorithm. The returned `tok` slices BORROW `corpus`.
///
/// ★ This is THE vocabulary definition for Arm E′, and it is deliberately a
/// separate, exported function: the recipe's PRODUCER (`emitRecipe`, driven by
/// `tools/derivdict_emit.zig`) and its CONSUMER (`buildDictFromVocab`, the
/// shipped decoder) must agree on every token and every index, and the only way
/// to guarantee that is for them to call the same code. A second
/// implementation of this function — in Python, or anywhere else — is a
/// producer/consumer drift bug waiting to be found by a gate.
pub fn buildVocab(gpa: std.mem.Allocator, corpus: []u8) ![]Entry {
    // ---- 1) ASCII case fold, in place (byte-level, no locale) ----
    for (corpus) |*b| {
        if (b.* >= 'A' and b.* <= 'Z') b.* += 32;
    }

    // ---- 2) token census: count every [a-z]+ run (keys = corpus slices) ----
    const slots = try gpa.alloc(Slot, MAP_CAP);
    defer gpa.free(slots);
    @memset(slots, .{ .off = 0, .len = 0, .cnt = 0 });
    var distinct: usize = 0;
    var i: usize = 0;
    while (i < corpus.len) {
        if (corpus[i] >= 'a' and corpus[i] <= 'z') {
            var j = i + 1;
            while (j < corpus.len and corpus[j] >= 'a' and corpus[j] <= 'z') j += 1;
            const tok = corpus[i..j];
            var idx: usize = @intCast(std.hash.Wyhash.hash(0, tok) & MAP_MASK);
            while (true) {
                const s = &slots[idx];
                if (s.len == 0) {
                    if (distinct >= MAP_CAP / 4 * 3) return error.CorpusVocabTooSmall;
                    s.* = .{ .off = @intCast(i), .len = @intCast(tok.len), .cnt = 1 };
                    distinct += 1;
                    break;
                }
                if (s.len == tok.len and std.mem.eql(u8, corpus[s.off..][0..s.len], tok)) {
                    s.cnt += 1;
                    break;
                }
                idx = (idx + 1) & MAP_MASK;
            }
            i = j;
        } else {
            i += 1;
        }
    }
    if (distinct < K) return error.CorpusVocabTooSmall;

    // ---- 3) canonical order (count desc, token asc — a total order, so the
    //         sorted result is unique regardless of algorithm) ----
    const entries = try gpa.alloc(Entry, distinct);
    errdefer gpa.free(entries);
    {
        var n: usize = 0;
        for (slots) |s| {
            if (s.len != 0) {
                entries[n] = .{ .tok = corpus[s.off..][0..s.len], .cnt = s.cnt };
                n += 1;
            }
        }
    }
    heapSort(Entry, entries, entryLess);
    return entries;
}

const WordSet = struct {
    /// S — the recipe's index space, byte-lexicographically sorted.
    words: []Entry,
    /// Offset of the permutation stream in the recipe (just past the stragglers).
    pos: usize,
};

/// Decode the recipe's bitmap + straggler sections against `entries` and return
/// S, the permutation's index space. Shared by the reader and the emitter so
/// the index space cannot be built two different ways.
fn readWordSet(gpa: std.mem.Allocator, entries: []const Entry, recipe: []const u8) !WordSet {
    if (recipe.len < BITMAP_LEN + 1) return error.RecipeCorrupt;
    if (entries.len < K) return error.CorpusVocabTooSmall;

    // ---- membership bitmap over the top-K pool + stragglers ⇒ word set ----
    var pos: usize = BITMAP_LEN;
    var selected_n: usize = 0;
    for (recipe[0..BITMAP_LEN]) |b| selected_n += @popCount(b);
    const straggler_n: usize = @intCast(try readVarint(recipe, &pos));
    // Reuse the Entry type + the ONE heapSort(Entry) instantiation for the
    // alphabetical sort too: with cnt uniformly 0, entryLess degenerates to
    // the pure byte-lexicographic order (words distinct => total order).
    const words = try gpa.alloc(Entry, selected_n + straggler_n);
    errdefer gpa.free(words);
    {
        var n: usize = 0;
        var k: usize = 0;
        while (k < K) : (k += 1) {
            if (recipe[k >> 3] & (@as(u8, 1) << @intCast(k & 7)) != 0) {
                words[n] = .{ .tok = entries[k].tok, .cnt = 0 };
                n += 1;
            }
        }
        var s: usize = 0;
        while (s < straggler_n) : (s += 1) {
            const wlen: usize = @intCast(try readVarint(recipe, &pos));
            if (wlen == 0 or pos + wlen > recipe.len) return error.RecipeCorrupt;
            words[n] = .{ .tok = recipe[pos .. pos + wlen], .cnt = 0 };
            n += 1;
            pos += wlen;
        }
    }

    // ---- S = the word set sorted byte-lexicographically (the permutation's
    //      index space; all words distinct => unique order) ----
    heapSort(Entry, words, entryLess);
    return .{ .words = words, .pos = pos };
}

/// Build the dictionary bytes from an already-censused vocabulary + the recipe
/// blob. Returns the dictionary file bytes, caller-owned.
pub fn buildDictFromVocab(gpa: std.mem.Allocator, entries: []const Entry, recipe: []const u8) ![]u8 {
    const ws = try readWordSet(gpa, entries, recipe);
    const words = ws.words;
    defer gpa.free(words);
    var pos = ws.pos;

    // ---- decode the permutation, emitting words as we go ----
    var out_len: usize = 0;
    for (words) |w| out_len += w.tok.len + 1;
    const out = try gpa.alloc(u8, out_len);
    errdefer gpa.free(out);
    const used = try gpa.alloc(bool, words.len);
    defer gpa.free(used);
    @memset(used, false);
    var cursor: usize = 0;
    var emitted: usize = 0;
    var prev: u64 = 0;
    var have_prev = false;
    while (pos < recipe.len) {
        var v: u64 = undefined;
        if (recipe[pos] == 0) {
            pos += 1;
            const raw = try readVarint(recipe, &pos);
            if (raw == 0) return error.RecipeCorrupt;
            v = raw - 1;
        } else {
            if (!have_prev) return error.RecipeCorrupt;
            v = prev + try readVarint(recipe, &pos);
        }
        if (v >= words.len) return error.RecipeCorrupt;
        const vi: usize = @intCast(v);
        if (used[vi]) return error.RecipeCorrupt;
        used[vi] = true;
        const w = words[vi].tok;
        @memcpy(out[cursor .. cursor + w.len], w);
        out[cursor + w.len] = '\n';
        cursor += w.len + 1;
        emitted += 1;
        prev = v;
        have_prev = true;
    }
    if (emitted != words.len or cursor != out.len) return error.RecipeCorrupt;
    return out;
}

/// Build the dictionary bytes from `corpus` (FOLDED IN PLACE) and the recipe
/// blob. Returns the dictionary file bytes, caller-owned.
pub fn buildDict(gpa: std.mem.Allocator, corpus: []u8, recipe: []const u8) ![]u8 {
    // Cheap shape check BEFORE the ~1 GB census, not after: a wrong file in the
    // recipe slot should cost 0 s, not a full vocabulary scan. (the
    // LSTM-era packaging fed english.dic here and paid 97 s of CPU before the
    // permutation stream rejected it — see
    if (recipe.len < BITMAP_LEN + 1) return error.RecipeCorrupt;
    const entries = try buildVocab(gpa, corpus);
    defer gpa.free(entries);
    return buildDictFromVocab(gpa, entries, recipe);
}

// ---------------------------------------------------------------------------
// PRODUCER SIDE — not referenced by any shipped binary (Zig analyses only what
// is reached, so this costs `decomp_bin` zero bytes; asserted by the packed
// prefix re-measure in the aug-17 report). Driven by `tools/derivdict_emit.zig`
// via `zig build derivdict-emit`.
// ---------------------------------------------------------------------------

const PoolRef = struct { tok: []const u8, idx: u32 };

fn poolLess(_: void, a: PoolRef, b: PoolRef) bool {
    return std.mem.lessThan(u8, a.tok, b.tok);
}

fn writeVarint(buf: []u8, pos: *usize, x_in: u64) void {
    var x = x_in;
    while (true) {
        const b: u8 = @intCast(x & 0x7F);
        x >>= 7;
        if (x != 0) {
            buf[pos.*] = b | 0x80;
            pos.* += 1;
        } else {
            buf[pos.*] = b;
            pos.* += 1;
            return;
        }
    }
}

/// Byte-lexicographic lookup in a sorted `[]Entry` (S). Returns the index, or
/// null if the word is absent.
fn lookupSorted(words: []const Entry, tok: []const u8) ?usize {
    var lo: usize = 0;
    var hi: usize = words.len;
    while (lo < hi) {
        const mid = lo + (hi - lo) / 2;
        const c = std.mem.order(u8, words[mid].tok, tok);
        switch (c) {
            .lt => lo = mid + 1,
            .gt => hi = mid,
            .eq => return mid,
        }
    }
    return null;
}

/// ★ THE PRODUCER. Emit the recipe blob that rebuilds `dict` from `entries`.
///
/// This is the counterpart of `buildDictFromVocab`, and it exists so that ONE
/// implementation owns both sides of the format. It shares:
///   * `buildVocab`   — the corpus tokenisation, census and canonical order,
///   * `readWordSet`  — the bitmap/straggler decode AND the byte-lexicographic
///                      sort that defines the permutation's index space,
///   * `buildDictFromVocab` — run at the end as an in-process self-check.
/// The emitter therefore cannot disagree with the reader about tokenisation,
/// tie-breaking, the top-K cut, or the index space; the only thing it adds is
/// the varint ENCODING of a permutation the reader already knows how to decode.
///
/// Refuses to emit a blob that does not reproduce `dict` byte-for-byte.
pub fn emitRecipe(gpa: std.mem.Allocator, entries: []const Entry, dict: []const u8) ![]u8 {
    if (entries.len < K) return error.CorpusVocabTooSmall;

    // ---- 1) the dictionary's own word order (D), '\n'-separated ----
    var word_n: usize = 0;
    {
        var it = std.mem.splitScalar(u8, dict, '\n');
        while (it.next()) |w| {
            if (w.len != 0) word_n += 1;
        }
    }
    if (word_n == 0) return error.RecipeCorrupt;
    const d = try gpa.alloc([]const u8, word_n);
    defer gpa.free(d);
    {
        var n: usize = 0;
        var it = std.mem.splitScalar(u8, dict, '\n');
        while (it.next()) |w| {
            if (w.len != 0) {
                d[n] = w;
                n += 1;
            }
        }
    }

    // ---- 2) the candidate pool = entries[0..K], indexed by token ----
    const pool = try gpa.alloc(PoolRef, K);
    defer gpa.free(pool);
    for (pool, 0..) |*p, k| p.* = .{ .tok = entries[k].tok, .idx = @intCast(k) };
    heapSort(PoolRef, pool, poolLess);

    // ---- 3) membership bitmap (OR-ing bits: order-independent) + stragglers
    //         (dict words outside the pool, IN DICT ORDER) ----
    const bitmap = try gpa.alloc(u8, BITMAP_LEN);
    defer gpa.free(bitmap);
    @memset(bitmap, 0);
    var outside_n: usize = 0;
    var outside_bytes: usize = 0;
    for (d) |w| {
        if (poolIndexOf(pool, w)) |k| {
            bitmap[k >> 3] |= @as(u8, 1) << @intCast(k & 7);
        } else {
            outside_n += 1;
            outside_bytes += w.len;
        }
    }

    // Upper bound: bitmap + varint(count) + per straggler varint(len)+bytes +
    // per dict word a restart token (1) plus a 10-byte varint.
    const cap = BITMAP_LEN + 10 + outside_n * 10 + outside_bytes + word_n * 11;
    const blob = try gpa.alloc(u8, cap);
    errdefer gpa.free(blob);
    @memcpy(blob[0..BITMAP_LEN], bitmap);
    var pos: usize = BITMAP_LEN;
    writeVarint(blob, &pos, outside_n);
    for (d) |w| {
        if (poolIndexOf(pool, w) == null) {
            writeVarint(blob, &pos, w.len);
            @memcpy(blob[pos .. pos + w.len], w);
            pos += w.len;
        }
    }

    // ---- 4) S — built by the READER's own code, from the prefix just written ----
    const ws = try readWordSet(gpa, entries, blob[0..pos]);
    defer gpa.free(ws.words);
    if (ws.pos != pos) return error.RecipeCorrupt; // prefix framing disagreement

    // ---- 5) the permutation, in S's alphabetical index space. english.dic's
    //         alphabetical runs become ASCENDING runs the delta coder captures
    //         (55,152 B vs 85,379 in frequency-rank space —
    //         feedback-permutation-index-space). ----
    var prev: u64 = 0;
    var have_prev = false;
    for (d) |w| {
        const vi = lookupSorted(ws.words, w) orelse return error.RecipeCorrupt;
        const v: u64 = @intCast(vi);
        if (!have_prev or v <= prev) {
            blob[pos] = 0;
            pos += 1;
            writeVarint(blob, &pos, v + 1);
        } else {
            writeVarint(blob, &pos, v - prev);
        }
        prev = v;
        have_prev = true;
    }
    const out = try gpa.realloc(blob, pos);

    // ---- 6) SELF-CHECK with the shipped reader: refuse to emit a blob that
    //         does not reproduce the dictionary byte-for-byte. ----
    const rebuilt = try buildDictFromVocab(gpa, entries, out);
    defer gpa.free(rebuilt);
    if (!std.mem.eql(u8, rebuilt, dict)) return error.DerivDictMismatch;
    return out;
}

fn poolIndexOf(pool: []const PoolRef, tok: []const u8) ?u32 {
    var lo: usize = 0;
    var hi: usize = pool.len;
    while (lo < hi) {
        const mid = lo + (hi - lo) / 2;
        switch (std.mem.order(u8, pool[mid].tok, tok)) {
            .lt => lo = mid + 1,
            .gt => hi = mid,
            .eq => return pool[mid].idx,
        }
    }
    return null;
}

/// The SHIP entry point for Arm E′ (`-Dderivdict`): rebuild english.dic from
/// the corpus + recipe and PROVE it is the golden, byte for byte, before any
/// coder sees it.
///
/// `corpus` is borrowed read-only — this takes its own working copy because
/// `buildDict` case-folds in place and keeps slices into that buffer, while the
/// caller still needs the original enwik9 for the payload. The copy is a
/// ~1 GB transient that lives only across the vocabulary scan; it is released
/// before the coder is constructed, so it sits ~7.4 GB below the encode's own
/// predictor peak and cannot move MaxRSS. (Owed as a measurement on the
/// imposed-cap e9 leg, not assumed.)
pub fn rebuildShipDict(gpa: std.mem.Allocator, corpus: []const u8, recipe: []const u8) ![]u8 {
    const work = try gpa.dupe(u8, corpus);
    defer gpa.free(work);
    const dict = try buildDict(gpa, work, recipe);
    errdefer gpa.free(dict);
    if (dict.len != GOLDEN_DICT_LEN) return error.DerivDictMismatch;
    if (std.hash.Wyhash.hash(0, dict) != GOLDEN_DICT_WYHASH) return error.DerivDictMismatch;
    return dict;
}
