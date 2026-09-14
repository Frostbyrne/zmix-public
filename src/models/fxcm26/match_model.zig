//! MatchModel2 — bit-exact Zig port of cmix-lex `fxcmv1.cpp` match model
//! (slots 7-13, 7 model outputs). Ported 1:1 from `match_model_v26.rs`.
//!
//! Ports the byte buffer (buffer/pos), the position-hash `mhashtable`, the
//! `cand[4]` state machine (MatchInfo recovery/delta, prio/is_better_than,
//! register_match, add_candidates/is_m_match), update_model/mix, and the three
//! `StateMap1` predictors `sma[0..2]` (imported from state_map1.zig).
//!
//! The 4th candidate hash source is `worcxt.Word(1)` (from the not-yet-ported
//! preamble); like the oracle, it's a synthetic input `word1` here (identical
//! both sides). The v26 Rust source has NO IDR (indel-recovery) path, so this
//! faithful port omits it too; validated bit-exact vs the C++ oracle checksum.

const std = @import("std");
const Sink = @import("emit_sink.zig").Sink;
// This file lives in the `cmfast` module (see cm_fast.zig); byte_core.zig and
// state_map1.zig (std-only) are module-local; zero_alloc is the module copy.
const X = @import("cmcold").X;
const StateMap1 = @import("state_map1.zig").StateMap1;
const zero_alloc = @import("cmcold").zero_alloc;

const BMASK: u32 = 0xffffff;
const MAXLEN: u32 = 62;
const LEN1: u32 = 5;
const LEN2: u32 = 7;
const LEN3: u32 = 9;
const MINLEN_RM: u32 = 3;
const MATCH_N: u32 = 4;
const MHASH_N: usize = 4;
const NST: usize = 3;

/// Decay-table container so the imported `StateMap1` (which reads `tables.dt`)
/// can be threaded the file-global `dt` the C++ used. `dt[i] = 4096/(i+2)`,
/// `dt[1023] = 1`.
const SmTables = struct {
    dt: [1024]i32,

    fn build() SmTables {
        var t: SmTables = undefined;
        var o: i32 = 2;
        var i: usize = 0;
        while (i < 1024) : (i += 1) {
            t.dt[i] = @divTrunc(4096, o);
            o += 1;
        }
        t.dt[1023] = 1;
        return t;
    }
};

inline fn hash3(a: u32, b: u32, c: u32) u32 {
    const h = a *% 110002499 +% b *% 30005491 +% c *% 50004239;
    return h ^ (h >> 9) ^ (a >> 3) ^ (b >> 3) ^ (c >> 4);
}

// -Didr: the banked IDR stack — INDEL_RECOVERY ±1/±2 probes (IDR_DEPTH=2) +
// IDR_PATIENCE=3 multi-round backup keep-alive. Ground truth:
// submission_record/src_edits/fxcmv1.DEPTH2-PATIENCE3.cpp:3966-4007 (md5
// 2940aa8d); fleet-banked −11,137 @100 MB post-WRT ≈ −62 K enwik9, RAM/time-
// neutral. GATE_DEEP and BUDGET2 were sweep-REJECTED and are not ported.
// Comptime-gated OFF by default so the stock-v26 oracle checksum gate holds.
const IDR = @import("build_options").idr;
const IDR_PATIENCE: u16 = 3;

const MatchInfo = struct {
    length: u32 = 0,
    index: u32 = 0,
    length_bak: u32 = 0,
    index_bak: u32 = 0,
    expected_byte: u8 = 0,
    delta: bool = false,
    // Patience counter: consecutive failed pre-recovery rounds (IDR only).
    pat: u16 = 0,

    fn init() MatchInfo {
        return MatchInfo{};
    }

    inline fn is_in_no_match_mode(self: *const MatchInfo) bool {
        return self.length == 0 and !self.delta and self.length_bak == 0;
    }
    inline fn is_in_pre_recovery_mode(self: *const MatchInfo) bool {
        return self.length == 0 and !self.delta and self.length_bak != 0;
    }
    inline fn is_in_recovery_mode(self: *const MatchInfo) bool {
        return self.length != 0 and self.length_bak != 0;
    }
    inline fn recovery_mode_pos(self: *const MatchInfo) u32 {
        return self.length -% self.length_bak;
    }
    inline fn prio(self: *const MatchInfo) u32 {
        return (@as(u32, @intFromBool(self.length != 0)) << 31) |
            (@as(u32, @intFromBool(self.delta)) << 30) |
            ((if (self.delta) (self.length_bak >> 1) else (self.length >> 1)) << 24) |
            (self.index & 0x00ffffff);
    }
    inline fn is_better_than(self: *const MatchInfo, other: *const MatchInfo) bool {
        return self.prio() > other.prio();
    }
    inline fn register_match(self: *MatchInfo, pos: u32, len: u32) void {
        self.length = len -% LEN1 +% 1;
        self.index = pos;
        self.length_bak = 0;
        self.index_bak = 0;
        self.expected_byte = 0;
        self.delta = false;
    }

    fn update(self: *MatchInfo, x: *const X, buffer: []const u8, c1: u8) void {
        if (self.length != 0) {
            const sh: u3 = @intCast((8 - x.bpos) & 7);
            const expected_bit: i32 = (@as(i32, self.expected_byte) >> sh) & 1;
            if (x.y != expected_bit) {
                if (self.is_in_recovery_mode()) {
                    self.length_bak = 0;
                    self.index_bak = 0;
                } else {
                    self.length_bak = self.length;
                    self.index_bak = self.index;
                    self.delta = true;
                    if (comptime IDR) self.pat = 0; // fresh pre-recovery episode (C++ :3952)
                }
                self.length = 0;
            }
        }
        if (x.bpos == 0) {
            if (self.is_in_pre_recovery_mode()) {
                self.index_bak +%= 1;
                if (self.length_bak < MAXLEN) {
                    self.length_bak += 1;
                }
                // bufr(index_bak) == c1
                if (buffer[@intCast(self.index_bak & BMASK)] == c1) {
                    self.length = self.length_bak;
                    self.index = self.index_bak;
                } else if (comptime IDR) {
                    // Indel probes, C++ :3983-3995 order: +1 (deletion), −1
                    // (insertion, guarded), +2, −2 (guarded). Unsigned wrap on
                    // index arithmetic matches the C++ `unsigned int`.
                    if (buffer[@intCast((self.index_bak +% 1) & BMASK)] == c1) {
                        self.length = self.length_bak;
                        self.index = self.index_bak +% 1;
                    } else if (self.index_bak >= 1 and buffer[@intCast((self.index_bak -% 1) & BMASK)] == c1) {
                        self.length = self.length_bak;
                        self.index = self.index_bak -% 1;
                    } else if (buffer[@intCast((self.index_bak +% 2) & BMASK)] == c1) {
                        self.length = self.length_bak;
                        self.index = self.index_bak +% 2;
                    } else if (self.index_bak >= 2 and buffer[@intCast((self.index_bak -% 2) & BMASK)] == c1) {
                        self.length = self.length_bak;
                        self.index = self.index_bak -% 2;
                    } else {
                        // Patience (C++ :3998-4002): keep the backup alive and
                        // retry next byte; purge only after IDR_PATIENCE rounds.
                        self.pat += 1;
                        if (self.pat >= IDR_PATIENCE) {
                            self.length_bak = 0;
                            self.index_bak = 0;
                            self.pat = 0;
                        }
                    }
                } else {
                    self.length_bak = 0;
                    self.index_bak = 0;
                }
            }
            if (self.length != 0) {
                self.index +%= 1;
                if (self.length < MAXLEN) {
                    self.length += 1;
                }
                if (self.is_in_recovery_mode() and self.recovery_mode_pos() >= MINLEN_RM) {
                    self.length_bak = 0;
                    self.index_bak = 0;
                    if (comptime IDR) self.pat = 0; // stable-recovery exit (C++ :4014)
                }
            }
            self.delta = false;
        }
    }
};

pub const MatchModel2 = struct {
    allocator: std.mem.Allocator,
    buffer: []u8,
    pos: u32,
    c1: u8, // the GLOBAL (override-adjusted) c1 in the full predictor
    t: [16]u32, // order-hash array (only LEN1/LEN2/LEN3 used here)
    mhashtable: [][MHASH_N]u32,
    mhashtablemask: u32,
    cand: [4]MatchInfo,
    active: u32,
    ctx: [NST]u32,
    sma: [NST]StateMap1,
    sm_tables: SmTables,
    // Shared pointer (C++ has one global strt; embedded 8KB copies thrash L1).
    strt: *const [4096]i16,
    // Fixed per-mix slot sink (C++ writes into a fixed array; ArrayList kept an
    // extra pointer indirection + capacity bookkeeping in the hot loop).

    pub fn new(allocator: std.mem.Allocator) MatchModel2 {
        return MatchModel2{
            .allocator = allocator,
            .buffer = &[_]u8{},
            .pos = 0,
            .c1 = 0,
            .t = [_]u32{0} ** 16,
            .mhashtable = &[_][MHASH_N]u32{},
            .mhashtablemask = 0,
            .cand = [_]MatchInfo{MatchInfo.init()} ** 4,
            .active = 0,
            .ctx = [_]u32{0} ** NST,
            .sma = undefined,
            .sm_tables = SmTables.build(),
            // Placeholder until init; points at the shared module table (fix9)
            // instead of a dedicated 8KB zero blob (S1: rodata is LZMA'd into the dist).
            .strt = @import("direct_state_map.zig").STRT_PTR,
        };
    }

    pub fn deinit(self: *MatchModel2) void {
        if (self.buffer.len != 0) zero_alloc.free(self.allocator, self.buffer);
        if (self.mhashtable.len != 0) zero_alloc.free(self.allocator, self.mhashtable);
        var i: usize = 0;
        while (i < NST) : (i += 1) self.sma[i].deinit(self.allocator);
    }

    inline fn stretch(self: *const MatchModel2, p: i32) i32 {
        const cp = std.math.clamp(p, 0, 4095);
        return self.strt[@intCast(cp)];
    }

    pub fn init(self: *MatchModel2, mhashtablemask: u32, sma_bits: [3]usize, strt: *const [4096]i16) !void {
        // calloc-parity (C++ alloc1): kernel-zeroed lazy pages for the 16 MB
        // byte ring and the 64 MB position-hash table.
        self.buffer = try zero_alloc.alloc(self.allocator, u8, @as(usize, BMASK + 1));
        self.pos = 0;
        self.c1 = 0;
        self.t = [_]u32{0} ** 16;
        self.mhashtablemask = mhashtablemask;
        self.mhashtable = try zero_alloc.alloc(self.allocator, [MHASH_N]u32, @as(usize, mhashtablemask + 1));
        self.cand = [_]MatchInfo{MatchInfo.init()} ** 4;
        self.active = 0;
        self.ctx = [_]u32{0} ** NST;
        self.sma_tables_init();
        self.sma[0] = try StateMap1.new(self.allocator, sma_bits[0], 1023);
        self.sma[1] = try StateMap1.new(self.allocator, sma_bits[1], 1023);
        self.sma[2] = try StateMap1.new(self.allocator, sma_bits[2], 1023);
        self.strt = strt;
    }

    inline fn sma_tables_init(self: *MatchModel2) void {
        self.sm_tables = SmTables.build();
    }

    /// parseByte-substitute: write the just-completed byte and compute order hashes.
    pub fn parse_byte(self: *MatchModel2, byte: u8) void {
        self.c1 = byte;
        self.buffer[@intCast(self.pos & BMASK)] = byte;
        self.pos = self.pos +% 1;
        self.t[LEN1] = self.ordhash(LEN1);
        self.t[LEN2] = self.ordhash(LEN2);
        self.t[LEN3] = self.ordhash(LEN3);
    }

    /// Debug: the three StateMap1 prs.
    pub fn sma_prs(self: *const MatchModel2) [3]i32 {
        return [3]i32{ self.sma[0].pr, self.sma[1].pr, self.sma[2].pr };
    }

    /// Override the order-hash sources with the REAL parseByte t[] values.
    pub inline fn set_order_hashes(self: *MatchModel2, t5: u32, t7: u32, t9: u32) void {
        self.t[LEN1] = t5;
        self.t[LEN2] = t7;
        self.t[LEN3] = t9;
    }

    inline fn ordhash(self: *const MatchModel2, k: u32) u32 {
        var h = k;
        var j: u32 = 1;
        while (j <= k) : (j += 1) {
            h = hash3(h, @as(u32, self.buffer[@intCast((self.pos -% j) & BMASK)]), 0xffffffff);
        }
        return h;
    }

    inline fn is_m_match(self: *const MatchModel2, pos: u32, minlen: u32) bool {
        var length: u32 = 1;
        while (length <= minlen) : (length += 1) {
            // buf(length) != bufr(pos - length)
            const a = self.buffer[@intCast((self.pos -% length) & BMASK)];
            const b = self.buffer[@intCast((pos -% length) & BMASK)];
            if (a != b) {
                return false;
            }
        }
        return true;
    }

    fn add_candidates(self: *MatchModel2, matches: *const [MHASH_N]u32, len: u32) void {
        var i: usize = 0;
        while (self.active < MATCH_N and i < MHASH_N) : (i += 1) {
            const matchpos = matches[i];
            if (matchpos == 0) {
                break;
            }
            if (self.is_m_match(matchpos, len)) {
                var is_same = false;
                var j: usize = 0;
                while (j < self.active) : (j += 1) {
                    if (self.cand[j].index == matchpos) {
                        is_same = true;
                        break;
                    }
                }
                if (!is_same) {
                    self.cand[@intCast(self.active)].register_match(matchpos, len);
                    self.active += 1;
                }
            }
        }
    }

    inline fn bucket_add(bucket: *[MHASH_N]u32, pos: u32) void {
        bucket[3] = bucket[2];
        bucket[2] = bucket[1];
        bucket[1] = bucket[0];
        bucket[0] = pos;
    }

    fn update_model(self: *MatchModel2, x: *const X, word1: u32) void {
        const n = @max(self.active, 1);
        var i: u32 = 0;
        while (i < n) : (i +%= 1) {
            // borrow-free: take, update, store
            var c = self.cand[@intCast(i)];
            c.update(x, self.buffer, self.c1);
            self.cand[@intCast(i)] = c;
            if (self.active != 0 and self.cand[@intCast(i)].is_in_no_match_mode()) {
                self.active -= 1;
                if (self.active == i) {
                    break;
                }
                var j: usize = @intCast(i);
                while (j < self.active) : (j += 1) {
                    self.cand[j] = self.cand[j + 1];
                }
                i = i -% 1;
            }
        }
        if (x.bpos == 0) {
            const sources = [4]u32{ self.t[LEN3], self.t[LEN2], self.t[LEN1], word1 };
            const lens = [4]u32{ LEN3, LEN2, LEN1, LEN1 };
            var s: usize = 0;
            while (s < 4) : (s += 1) {
                const idx: usize = @intCast(sources[s] & self.mhashtablemask);
                const bvals = self.mhashtable[idx];
                if (self.active < MATCH_N) {
                    self.add_candidates(&bvals, lens[s]);
                }
                const pos = self.pos;
                bucket_add(&self.mhashtable[idx], pos);
            }
            var k: usize = 0;
            while (k < self.active) : (k += 1) {
                // expectedByte = bufr(index)
                self.cand[k].expected_byte = self.buffer[@intCast(self.cand[k].index & BMASK)];
            }
        }
    }

    /// MatchModel2mix — returns match length; emits the 7 model slots.
    pub fn mix(self: *MatchModel2, x: *const X, word1: u32, sink: *const Sink) u32 {
        self.update_model(x, word1);
        var i: usize = 0;
        while (i < NST) : (i += 1) {
            self.ctx[i] = 0;
        }
        var best: usize = 0;
        var m: usize = 1;
        while (m < self.active) : (m += 1) {
            if (self.cand[m].is_better_than(&self.cand[best])) {
                best = m;
            }
        }
        const length = self.cand[best].length;
        const expected_byte = self.cand[best].expected_byte;
        const is_delta = self.cand[best].delta;
        const expected_bit: i32 = if (length != 0)
            ((@as(i32, expected_byte) >> @as(u5, @intCast(7 - x.bpos))) & 1)
        else
            0;
        if (length != 0) {
            const denselength: u32 = if (length <= 16) length - 1 else 12 + (length >> 2);
            self.ctx[0] = (denselength << 4) | (@as(u32, @intCast(expected_bit)) << 3) | @as(u32, @intCast(x.bpos));
            self.ctx[1] = (@as(u32, expected_byte) << 11) | (@as(u32, @intCast(x.bpos)) << 8) | @as(u32, self.c1);
            const sign: i32 = 2 * expected_bit - 1;
            self.emit(sink, sign * @as(i32, @intCast(length << 5)));
        } else {
            self.emit(sink, 0);
        }
        if (is_delta) {
            self.ctx[2] = (@as(u32, expected_byte) << 8) | @as(u32, @intCast(x.c0));
        }
        var k: usize = 0;
        while (k < NST) : (k += 1) {
            const c = self.ctx[k];
            if (c != 0) {
                const p1 = self.sma[k].set(&self.sm_tables, x.y, @intCast(c));
                const st = self.stretch(p1);
                self.emit(sink, st >> 2);
                self.emit(sink, (p1 - 2048) >> 3);
            } else {
                self.emit(sink, 0);
                self.emit(sink, 0);
            }
        }
        return length;
    }

    inline fn emit(self: *MatchModel2, sink: *const Sink, p: i32) void {
        _ = self;
        const clipped: i16 = if (p < -2047)
            -2047
        else if (p > 2047)
            2047
        else
            @intCast(p);
        sink.push(clipped);
    }
};

// ---- test ----

const STRT_BYTES = @embedFile("goldens/cm3_STRT.bin");

inline fn gsbl(bpos: i32, c0: i32) i32 {
    const smask: i32 = @intCast((@as(u32, 0x31031010) >> @as(u5, @intCast(bpos << 2))) & 0x0F);
    return smask + (c0 & smask);
}

test "match_model_matches_cpp_oracle" {
    const allocator = std.testing.allocator;

    // Load STRT (fx::STRT == cm3_STRT.bin, 4096 little-endian i16).
    var strt: [4096]i16 = undefined;
    {
        try std.testing.expectEqual(@as(usize, 8192), STRT_BYTES.len);
        var i: usize = 0;
        while (i < 4096) : (i += 1) {
            const lo: u16 = STRT_BYTES[i * 2];
            const hi: u16 = STRT_BYTES[i * 2 + 1];
            strt[i] = @bitCast(lo | (hi << 8));
        }
    }

    var mm = MatchModel2.new(allocator);
    defer mm.deinit();
    try mm.init(0xffff, [3]usize{ 1 << 9, 1 << 19, 1 << 16 }, &strt);

    // Test-local sink stands in for the predictor's in1/slots (reset per mix).
    var in1_buf: [544]i16 = undefined;
    var slot_buf: [560]i16 = undefined;
    var in1_n: usize = 0;
    var slot_n: usize = 0;
    const sink = Sink{ .in1 = &in1_buf, .in1_n = &in1_n, .slots = &slot_buf, .slot_n = &slot_n };

    // repeat-rich stream: base period 128 over 8-symbol alphabet, 1/8 mutation.
    const NB: usize = 8000;
    const P: usize = 128;
    const al = [8]u8{ 'a', 'b', 'c', 'd', 'e', ' ', '.', 't' };
    var base: [P]u8 = undefined;
    var rb: u32 = 0xC0FFEE;
    for (&base) |*bi| {
        rb = rb *% 1664525 +% 1013904223;
        bi.* = al[@as(usize, (rb >> 16) & 7)];
    }

    var x = X{ .c0 = 1 };
    var cs: u64 = 0;
    var r: u32 = 0x12345678;
    var ii: usize = 0;
    while (ii < NB) : (ii += 1) {
        r = r *% 1664525 +% 1013904223;
        const by: u8 = if (((r >> 13) & 7) == 0)
            al[@as(usize, (r >> 16) & 7)]
        else
            base[ii % P];
        var b: i32 = 7;
        while (b >= 0) : (b -= 1) {
            x.y = @intCast((by >> @as(u3, @intCast(b))) & 1);
            x.c0 += x.c0 + x.y;
            if (x.c0 >= 256) {
                x.c4 = (x.c4 << 8) +% (@as(u32, @intCast(x.c0)) & 0xff);
                x.c0 = 1;
            }
            x.bpos = (x.bpos + 1) & 7;
            x.bposshift = 7 - x.bpos;
            x.c0shift_bpos = (x.c0 << 1) ^ (@as(i32, 256) >> @as(u5, @intCast(x.bposshift)));
            x.cm_bit_state = gsbl(x.bpos, x.c0);
            if (x.bpos == 0) {
                mm.parse_byte(@intCast(x.c4 & 0xff));
            }
            in1_n = 0;
            slot_n = 0;
            _ = mm.mix(&x, 0, &sink);
            for (in1_buf[0..in1_n]) |v| {
                cs = cs *% 1000003 +% @as(u64, @as(u16, @bitCast(v)));
            }
        }
    }
    try std.testing.expectEqual(@as(u64, 0x19072cabcfc9c6e2), cs);
}
