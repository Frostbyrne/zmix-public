//! SentenceContext — bit-exact Zig port of cmix-lex `fxcmv1.cpp` `struct
//! SentenceContext`. Holds up to 64 past sentences (each a WordsContext snapshot) and
//! finds the most similar one by codeword-overlap. Feeds the simiwor contexts.
//!
//! Ported 1:1 from sentence_context_v26.rs. Validated bit-exact vs
//! v26-port-tools/sentence_oracle.cpp over synthetic sentences (folded into a checksum,
//! same drive reproduced in the test below).
//!
//! Zig notes: Rust has a field `sentence` alongside a method `sentence`; Zig does not
//! allow that, so the accessor method is renamed `sentence_at` (the integrator should
//! call `sc.sentence_at(i)` where the Rust used `sc.sentence(i)`). The `Vec<WordsContext>`
//! of fixed length 64 is a fixed `[SIMILARWORDS]WordsContext` array (no allocator needed).

const std = @import("std");
const WordsContext = @import("words_context.zig").WordsContext;

const SIMILARWORDS: usize = 64;

// AVX2-width integer SIMD for the similar_sentence_idx membership scan (the O(n^2)
// codeword-overlap loop). Integer equality has no reassociation freedom, so the
// vectorized count is BIT-IDENTICAL to the scalar break-on-first-hit scan.
const VLEN = 8;
const VU = @Vector(VLEN, u32);

pub const SentenceContext = struct {
    sentence: [SIMILARWORDS]WordsContext,
    empty: WordsContext,
    sindex: usize,
    total: u32,

    pub fn new() SentenceContext {
        var sc = SentenceContext{
            .sentence = undefined,
            .empty = WordsContext.new(),
            .sindex = 0,
            .total = 0,
        };
        var i: usize = 0;
        while (i < SIMILARWORDS) : (i += 1) {
            sc.sentence[i] = WordsContext.new();
        }
        return sc;
    }

    pub fn reset(self: *SentenceContext) void {
        for (&self.sentence) |*s| {
            s.reset();
        }
        self.sindex = 0;
        self.total = 0;
    }

    pub fn update(self: *SentenceContext, w: *const WordsContext) void {
        if (w.wordcount != 0) {
            self.sentence[self.sindex] = w.*;
            self.sindex = (self.sindex + 1) & (SIMILARWORDS - 1);
            self.total = self.total +% 1;
        }
    }

    /// Renamed from Rust `sentence` to avoid the field/method name clash in Zig.
    pub fn sentence_at(self: *const SentenceContext, i: i32) *const WordsContext {
        const idx: usize = @intCast((@as(u32, @intCast(self.sindex)) -% @as(u32, @bitCast(i))) & (@as(u32, SIMILARWORDS) - 1));
        return &self.sentence[idx];
    }

    /// Returns the index of the most-similar stored sentence, or null for `empty`.
    pub fn similar_sentence_idx(self: *const SentenceContext, wor: *const WordsContext, wcount: i32, pres: u32) ?usize {
        var is_similar: u32 = 0;
        var is_similar_idx: u32 = 0;
        var codesum: u32 = 0;
        if (wcount >= 1) {
            const MAXC: usize = 64 * 4;
            var curcode: [MAXC]u32 = undefined; // written [0..curcount), read [0..curcount) — zero-fill was dead
            var curcount: usize = 0;
            var k: i32 = 0;
            while (k < wcount) : (k += 1) {
                const code = wor.code(wcount -% k);
                if (code != 0 and curcount < MAXC) {
                    curcode[curcount] = code;
                    curcount += 1;
                }
            }
            var i: usize = 0;
            while (i < SIMILARWORDS) : (i += 1) {
                const wc = self.sentence[i].wordcount;
                if (wc != 0 and wc <= wcount and wc > @divTrunc(wcount, 2)) {
                    // Padded to a whole number of VLEN lanes so the membership test
                    // below can compare full @Vector chunks without a scalar tail.
                    // Codes are non-zero by construction (guarded at push, both here
                    // and in curcode), so 0 is a match-impossible pad sentinel — the
                    // count is bit-IDENTICAL to the scalar `break`-on-first-hit scan
                    // (a per-kk membership test; integer equality, no reassociation).
                    var testcode: [MAXC + VLEN]u32 = undefined;
                    var testcount: usize = 0;
                    var j: i32 = 0;
                    while (j < wc) : (j += 1) {
                        const code = self.sentence[i].code(wc -% j);
                        if (code != 0 and testcount < MAXC) {
                            testcode[testcount] = code;
                            testcount += 1;
                        }
                    }
                    const testpad = (testcount + (VLEN - 1)) & ~@as(usize, VLEN - 1);
                    var pj = testcount;
                    while (pj < testpad) : (pj += 1) testcode[pj] = 0;
                    var kk: usize = 0;
                    while (kk < curcount) : (kk += 1) {
                        const needle: VU = @splat(curcode[kk]);
                        var found = false;
                        var jj: usize = 0;
                        while (jj < testpad) : (jj += VLEN) {
                            const hay: VU = testcode[jj..][0..VLEN].*;
                            if (@reduce(.Or, needle == hay)) {
                                found = true;
                                break;
                            }
                        }
                        if (found) codesum +%= 1;
                    }
                    if (codesum > is_similar) {
                        is_similar = codesum;
                        is_similar_idx = @intCast(i + 1);
                    }
                    codesum = 0;
                }
            }
            if (is_similar != 0) {
                is_similar = is_similar *% 100 / @as(u32, @bitCast(wcount));
                if (is_similar < pres) {
                    is_similar_idx = 0;
                }
            }
        }
        if (is_similar_idx != 0) {
            return @intCast(is_similar_idx - 1);
        } else {
            return null;
        }
    }

    pub fn similar_sentence(self: *const SentenceContext, wor: *const WordsContext, wcount: i32, pres: u32) *const WordsContext {
        if (self.similar_sentence_idx(wor, wcount, pres)) |i| {
            return &self.sentence[i];
        } else {
            return &self.empty;
        }
    }
};

const Rng = struct {
    s: u32,
    fn next(self: *Rng) u32 {
        self.s = self.s *% 1664525 +% 1013904223;
        return self.s;
    }
};

test "sentence_context_matches_cpp_oracle" {
    var sc = SentenceContext.new();
    var cur = WordsContext.new();
    var rng = Rng{ .s = 0x9e3779b1 };
    var cs: u64 = 0;
    var s: usize = 0;
    while (s < 400) : (s += 1) {
        cur.reset();
        const n = rng.next() % 8 + 1;
        var it: u32 = 0;
        while (it < n) : (it += 1) {
            const b_set: u8 = @truncate(rng.next());
            const a_set: i32 = @intCast(rng.next() & 1);
            const g_set: u8 = @truncate(rng.next());
            cur.set(b_set, a_set, g_set);
            const w = rng.next() & 0xffff;
            const t = rng.next() & 0x3ff;
            const st = rng.next() & 0xffff;
            const cw = rng.next() & 0x3f;
            const wi: i32 = @intCast(rng.next() & 1);
            const b_upd: u8 = @truncate(rng.next());
            cur.update(w, b_upd, t, st, cw, wi);
        }
        const sim = sc.similar_sentence(&cur, cur.wordcount, 53);
        cs = cs *% 1000003 +% @as(u64, @as(u32, @bitCast(sim.wordcount)));
        var k: i32 = 1;
        while (k <= 4) : (k += 1) {
            cs = cs *% 1000003 +% @as(u64, sim.word(k));
            cs = cs *% 1000003 +% @as(u64, sim.code(k));
            cs = cs *% 1000003 +% @as(u64, sim.type_at(k));
        }
        sc.update(&cur);
        var i: i32 = 0;
        while (i < 3) : (i += 1) {
            const se = sc.sentence_at(i);
            cs = cs *% 1000003 +% @as(u64, @as(u32, @bitCast(se.wordcount)));
            cs = cs *% 1000003 +% @as(u64, se.word(1));
            cs = cs *% 1000003 +% @as(u64, se.code(1));
        }
        cs = cs *% 1000003 +% @as(u64, sc.total);
        cs = cs *% 1000003 +% @as(u64, @as(u32, @intCast(sc.sindex)));
        if (s % 37 == 0) {
            sc.reset();
        }
    }
    try std.testing.expectEqual(@as(u64, 0x674a2f85a88c33ab), cs);
}
