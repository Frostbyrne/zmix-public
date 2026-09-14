//! ContextMap4 — bit-exact Zig port of cmix-lex `fxcmv1.cpp` `struct ContextMap4`.
//! Same family as ContextMap3 but with the half-size `E<3,32>` bucket (3 checksums,
//! 32-byte element), NO internal statemap, and `st9`/`st32` emission. Owns ~70 of
//! the 525 active slots (cmC4[*], cmCR[*]).
//!
//! Ported 1:1 from cm4.rs. Validated bit-exact against the C++ oracle across 9
//! configs (see `test`). X shared bit-state comes from byte_core.zig.

const std = @import("std");
const Sink = @import("emit_sink.zig").Sink;
const X = @import("cmcold").X;
const zero_alloc = @import("cmcold").zero_alloc;
const cm3_mask = @import("cm3.zig");
const MASK_ANY: bool = cm3_mask.MASK_ANY;
/// -Dfxcm-slotdelete: the masked instances emit NOTHING (see cm3.zig/slotdel.zig).
const SLOTDEL: bool = cm3_mask.SLOTDEL;

const MAXCXT: usize = 8;

/// Placeholder target for the shared-table pointers before `init` (never read).
const NN_INIT: [1024]u8 = .{0} ** 1024;

// E<3,32>: chk[3] u16-LE at [0..6), last at [6], bh[3][7] at [7..28). elem = 32 bytes.
const ELEM: usize = 32;
const BH0: usize = 7; // offset of bh[0][0]
const A: usize = 3; // checksum slots

inline fn prefetch_t(t: []const u8, off: usize) void {
    @prefetch(&t[off], .{});
}

inline fn clp(z: i32) i16 {
    if (z < -2047) {
        return -2047;
    } else if (z > 2047) {
        return 2047;
    } else {
        return @intCast(z);
    }
}

inline fn sc(p: i32) i32 {
    if (p > 0) {
        return p >> 7;
    } else {
        return (p + 127) >> 7;
    }
}

pub const ContextMap4 = struct {
    c: usize,
    cn: usize,
    result: i32,
    cxt_mask: u16,
    /// -Dfxcm-slotmask: this instance is DROPPED (see cm3.zig).
    masked: bool,
    skip2: i32,
    kep: u8,
    tmask: u32,
    cp: [MAXCXT]?usize,
    cp0: [MAXCXT]?usize,
    runp: [MAXCXT]usize,
    cxt: [MAXCXT]u32,
    t: []u8,
    st32: [256]i16,
    st9: [256]i16,
    // Shared read-only lookup tables (one C++ global each, fxcmv1.cpp:1084-1086;
    // st32/st9 stay per-instance — they are per-instance members in C++ too).
    nn: *const [1024]u8,
    rcpr: []const i16,
    alloc: std.mem.Allocator,

    pub fn new() ContextMap4 {
        return ContextMap4{
            .masked = false,
            .c = 0,
            .cn = 0,
            .result = 0,
            .cxt_mask = 0,
            .skip2 = 0,
            .kep = 0,
            .tmask = 0,
            .cp = [_]?usize{null} ** MAXCXT,
            .cp0 = [_]?usize{null} ** MAXCXT,
            .runp = [_]usize{0} ** MAXCXT,
            .cxt = [_]u32{0} ** MAXCXT,
            .t = &.{},
            .st32 = [_]i16{0} ** 256,
            .st9 = [_]i16{0} ** 256,
            .nn = &NN_INIT,
            .rcpr = &.{},
            .alloc = undefined,
        };
    }

    pub fn deinit(self: *ContextMap4) void {
        if (self.t.len > 0) zero_alloc.free(self.alloc, self.t);
        self.t = &.{};
        self.rcpr = &.{};
    }

    inline fn next(self: *const ContextMap4, i: i32, y: i32) u8 {
        return self.nn[@intCast(y + i * 4)];
    }

    inline fn pre(self: *const ContextMap4, state: i32) i32 {
        const n0: u32 = @as(u32, self.next(state, 2)) * 3 + 1;
        const n1: u32 = @as(u32, self.next(state, 3)) * 3 + 1;
        return @intCast((n1 << 12) / (n0 + n1));
    }

    inline fn chk_get(self: *const ContextMap4, eb: usize, i: usize) u16 {
        const o = eb + i * 2;
        return @as(u16, self.t[o]) | (@as(u16, self.t[o + 1]) << 8);
    }

    inline fn chk_set(self: *ContextMap4, eb: usize, i: usize, v: u16) void {
        const o = eb + i * 2;
        self.t[o] = @intCast(v & 0xff);
        self.t[o + 1] = @intCast(v >> 8);
    }

    fn bucket_get(self: *ContextMap4, eb: usize, ch: u16, keep: u8) usize {
        const last = self.t[eb + 6];
        const recent0: usize = @as(usize, last & 15);
        const recent1: usize = @as(usize, last >> 4);
        if (recent0 < A and self.chk_get(eb, recent0) == ch) {
            return eb + BH0 + recent0 * 7;
        }
        var b: i32 = 0xffff;
        var bi: usize = 0;
        var i: usize = 0;
        while (i < A) : (i += 1) {
            if (self.chk_get(eb, i) == ch) {
                self.t[eb + 6] = @as(u8, @truncate(@as(u32, last) << 4)) | @as(u8, @intCast(i));
                return eb + BH0 + i * 7;
            }
            if (i != recent0 and i != recent1) {
                const pri: i32 = @as(i32, self.t[eb + BH0 + i * 7]);
                if (pri < b) {
                    b = pri;
                    bi = i;
                }
            }
        }
        self.t[eb + 6] = @as(u8, @truncate(@as(u32, last) << 4)) | @as(u8, @intCast(bi)) | keep;
        self.chk_set(eb, bi, ch);
        var j: usize = 0;
        while (j < 7) : (j += 1) {
            self.t[eb + BH0 + bi * 7 + j] = 0;
        }
        return eb + BH0 + bi * 7;
    }

    pub fn init(
        self: *ContextMap4,
        a: std.mem.Allocator,
        m: u32,
        c1: usize,
        cms: i32,
        cms3: i32,
        cms4: i32,
        nn: *const [1024]u8,
        kep: u8,
        skip2: i32,
        rcpr: []const i16,
        strt: *const [4096]i16,
    ) !void {
        _ = cms; // CM4 has no st1 table; cms is consumed only by CM3.
        self.alloc = a;
        self.c = @min(c1, MAXCXT);
        self.tmask = (m >> 6) -% 1;
        self.cn = 0;
        self.result = 0;
        self.cxt_mask = if (self.c > 1) 0 else 0xfffe;
        self.kep = kep;
        self.skip2 = skip2;
        const nelem: usize = @as(usize, m >> 6) + 64;
        if (self.t.len > 0) zero_alloc.free(self.alloc, self.t);
        // calloc-parity (C++ alloc1): kernel-zeroed lazy pages, no up-front commit.
        self.t = try zero_alloc.alloc(a, u8, nelem * ELEM);
        self.nn = nn;
        self.rcpr = rcpr;
        var i: usize = 0;
        while (i < self.c) : (i += 1) {
            self.cp0[i] = BH0;
            self.cp[i] = BH0;
            self.runp[i] = BH0 + 3;
        }
        i = 0;
        while (i < 256) : (i += 1) {
            self.st9[i] = clp(sc(18 * (self.pre(@intCast(i)) - 2048)));
        }
        var s: usize = 0;
        while (s < 256) : (s += 1) {
            const n0: i32 = -@as(i32, @intFromBool(self.next(@intCast(s), 2) == 0));
            const n1: i32 = -@as(i32, @intFromBool(self.next(@intCast(s), 3) == 0));
            var r: i32 = 0;
            var sp0: i32 = 0;
            if (n1 - n0 == 1) {
                sp0 = 0;
                r = 1;
            }
            if (n1 - n0 == -1) {
                sp0 = 4095;
                r = 1;
            }
            if (r == 1) {
                const pre_s = self.pre(@intCast(s));
                const st8 = clp(sc(cms4 * (pre_s - sp0)));
                const pre_str: i32 = @as(i32, strt[@intCast(std.math.clamp(pre_s, 0, 4095))]);
                const st32v = clp(sc(cms3 * pre_str));
                if (s < 8) {
                    self.st32[s] = st8;
                } else {
                    self.st32[s] = @intCast((@as(i32, st8) + @as(i32, st32v)) >> 1);
                }
            } else {
                self.st32[s] = 0;
            }
        }
    }

    /// fxcmv1.cpp ContextMap4::reset — wipe the table and re-arm at element 0.
    /// Fired at `</page>` (parse_byte.page_reset) on cmC4[1,2,4,7,8,6].
    pub fn reset(self: *ContextMap4) void {
        @memset(self.t[0 .. (@as(usize, self.tmask) + 1) * ELEM], 0);
        for (0..self.c) |i| {
            self.cp0[i] = BH0;
            self.cp[i] = BH0;
            self.runp[i] = BH0 + 3;
        }
        self.cn = 0;
        self.result = 0;
        self.cxt_mask = if (self.c > 1) 0 else 0xfffe;
    }

    pub fn set(self: *ContextMap4, cx_in: u32) void {
        // -Dfxcm-slotdelete: `cn` stays 0 forever (see cm3.set).
        if (comptime SLOTDEL) {
            if (self.masked) return;
        }
        if (self.cn >= self.c) {
            return;
        }
        if (comptime MASK_ANY) {
            if (self.masked) {
                self.cn += 1;
                self.cxt_mask = self.cxt_mask *% 2;
                return;
            }
        }
        const i = self.cn;
        self.cn += 1;
        var cx = cx_in *% 987654323 +% @as(u32, @intCast(i));
        cx = std.math.rotl(u32, cx, 16); // (cx << 16) | (cx >> 16)
        self.cxt[i] = cx *% 123456791 +% @as(u32, @intCast(i));
        self.cxt_mask = self.cxt_mask *% 2;
        // Prefetch the bpos-0 bucket (c0 == 1 at byte start); elem is 32 bytes.
        const idx: usize = @as(usize, (self.cxt[i] +% 1) & self.tmask);
        prefetch_t(self.t, idx * ELEM);
    }

    /// Prefetch the bpos-2/5 buckets for every live context (cache hint only;
    /// mirrors mix's skip conditions).
    pub fn prefetch_c0(self: *const ContextMap4, c0: u32) void {
        if (comptime MASK_ANY) {
            if (self.masked) return;
        }
        var i: usize = 0;
        while (i < self.cn) : (i += 1) {
            if ((self.cxt_mask >> @as(u4, @intCast(self.cn - i))) & 1 != 0) {
                continue;
            }
            if (self.t[self.runp[i]] == 0) {
                continue;
            }
            const idx: usize = @as(usize, (self.cxt[i] +% c0) & self.tmask);
            prefetch_t(self.t, idx * ELEM);
        }
    }

    pub fn sets(self: *ContextMap4) void {
        if (comptime SLOTDEL) {
            if (self.masked) return;
        }
        if (self.cn >= self.c) {
            return;
        }
        self.cn += 1;
        self.cxt_mask = self.cxt_mask +% 1;
        self.cxt_mask = self.cxt_mask *% 2;
    }

    inline fn add(self: *ContextMap4, sink: *const Sink, p: i32) void {
        _ = self;
        sink.push(clp(p));
    }

    inline fn mix3(self: *ContextMap4, sink: *const Sink, s: i32) i32 {
        if (s == 0) {
            if (self.skip2 == 1) {
                self.add(sink, 0);
            }
            self.add(sink, 0);
            return 0;
        } else {
            if (self.skip2 == 1) {
                self.add(sink, @as(i32, self.st9[@intCast(s)]));
            }
            self.add(sink, @as(i32, self.st32[@intCast(s)]));
            return 1;
        }
    }

    inline fn mix4(self: *ContextMap4, sink: *const Sink) void {
        if (self.skip2 == 1) {
            self.add(sink, 0);
        }
        self.add(sink, 0);
        self.add(sink, 0);
    }

    pub fn mix(self: *ContextMap4, x: *const X, sink: *const Sink) i32 {
        self.result = 0;
        // -Dfxcm-slotdelete: TRUE DELETION — no emission (see cm3.mix).
        if (comptime SLOTDEL) {
            if (self.masked) return 0;
        }
        if (comptime MASK_ANY) {
            if (self.masked) {
                var q: usize = 0;
                while (q < self.cn) : (q += 1) self.mix4(sink);
                if (x.bpos == 7) {
                    self.cn = 0;
                    self.cxt_mask = 0;
                }
                return 0;
            }
        }
        // sm32 (cm4): prefetch each live context's carried-over bucket before the
        // loop reads t[cp[i]]/t[runp[i]] — same rationale + same-line coverage as
        // cm3.mix (runp[i] = cp base + 3). Pure cache hint, bit-identical.
        {
            var j: usize = 0;
            while (j < self.cn) : (j += 1) prefetch_t(self.t, self.runp[j]);
        }
        var i: usize = 0;
        while (i < self.cn) : (i += 1) {
            if ((self.cxt_mask >> @as(u4, @intCast(self.cn - i))) & 1 != 0) {
                self.mix4(sink);
            } else {
                if (self.cp[i]) |cpi| {
                    const st: i32 = @as(i32, self.t[cpi]);
                    self.t[cpi] = self.next(st, x.y);
                }
                var s: i32 = 0;
                if (x.bpos > 1 and self.t[self.runp[i]] == 0) {
                    self.cp[i] = null;
                } else {
                    const chksum: u16 = @truncate((self.cxt[i] >> 16) ^ @as(u32, @intCast(i)));
                    if (x.bpos != 0) {
                        if (x.bpos == 2 or x.bpos == 5) {
                            const idx: usize = @as(usize, (self.cxt[i] +% @as(u32, @intCast(x.c0))) & self.tmask);
                            const off = self.bucket_get(idx * ELEM, chksum, self.kep);
                            self.cp0[i] = off;
                            self.cp[i] = off;
                        } else {
                            self.cp[i] = self.cp0[i].? + @as(usize, @intCast(x.cm_bit_state));
                        }
                    } else {
                        const idx: usize = @as(usize, (self.cxt[i] +% @as(u32, @intCast(x.c0))) & self.tmask);
                        const off = self.bucket_get(idx * ELEM, chksum, self.kep);
                        self.cp0[i] = off;
                        self.cp[i] = off;
                        if (self.t[off + 3] == 2) {
                            const c: i32 = @as(i32, self.t[off + 4]) + 256;
                            const idx1: usize = @as(usize, (self.cxt[i] +% @as(u32, @intCast(c >> 6))) & self.tmask);
                            const idx2p: usize = @as(usize, (self.cxt[i] +% @as(u32, @intCast(c >> 3))) & self.tmask);
                            prefetch_t(self.t, idx2p * ELEM);
                            const p = self.bucket_get(idx1 * ELEM, chksum, self.kep);
                            self.t[p] = @intCast(1 + ((c >> 5) & 1));
                            self.t[p + 1 + @as(usize, @intCast((c >> 5) & 1))] = @intCast(1 + ((c >> 4) & 1));
                            self.t[p + 3 + @as(usize, @intCast((c >> 4) & 3))] = @intCast(1 + ((c >> 3) & 1));
                            const idx2: usize = @as(usize, (self.cxt[i] +% @as(u32, @intCast(c >> 3))) & self.tmask);
                            const p2 = self.bucket_get(idx2 * ELEM, chksum, self.kep);
                            self.t[p2] = @intCast(1 + ((c >> 2) & 1));
                            self.t[p2 + 1 + @as(usize, @intCast((c >> 2) & 1))] = @intCast(1 + ((c >> 1) & 1));
                            self.t[p2 + 3 + @as(usize, @intCast((c >> 1) & 3))] = @intCast(1 + (c & 1));
                            self.t[off + 6] = 0;
                        }
                        const c1: u8 = @truncate(x.c4 & 0xff);
                        const rp = self.runp[i];
                        if (self.t[rp] == 0) {
                            self.t[rp] = 2;
                            self.t[rp + 1] = c1;
                        } else if (self.t[rp + 1] != c1) {
                            self.t[rp] = 1;
                            self.t[rp + 1] = c1;
                        } else if (self.t[rp] < 254) {
                            self.t[rp] += 2;
                        }
                        self.runp[i] = off + 3;
                    }
                    s = @as(i32, self.t[self.cp[i].?]);
                }
                self.result += self.mix3(sink, s);
                const b = x.c0shift_bpos ^ (@as(i32, self.t[self.runp[i] + 1]) >> @as(u5, @intCast(x.bposshift)));
                if (b <= 1) {
                    const bb = b * 256;
                    self.add(sink, @as(i32, self.rcpr[@intCast(@as(i32, self.t[self.runp[i]]) + bb)]));
                } else {
                    self.add(sink, 0);
                }
            }
        }
        if (x.bpos == 7) {
            self.cn = 0;
            self.cxt_mask = 0;
        }
        return self.result;
    }
};

// ---------------------------------------------------------------------------
// Tests (ported from cm4.rs `mod tests`; goldens from the C++ oracle).
// Fixtures: STA1/2/4/5/6/7 come from state_table_tables.bin (same order as the
// cm4_fixtures stamap: 1,2,4,5,6,7). STRT reuses dsm_STRT.bin. RCPR is the
// cm3_tables.rs RCPR array dumped little-endian to cm4_RCPR.bin.
// ---------------------------------------------------------------------------

const testing = std.testing;

const G_STA = @embedFile("goldens/state_table_tables.bin"); // 6 * 1024 bytes
const G_STRT = @embedFile("goldens/dsm_STRT.bin"); // 4096 * i16 LE
const G_RCPR = @embedFile("goldens/cm4_RCPR.bin"); // 512 * i16 LE

fn i16arr(comptime raw: anytype, comptime N: usize) [N]i16 {
    @setEvalBranchQuota(4 * N + 1000);
    var out: [N]i16 = undefined;
    for (0..N) |i| {
        out[i] = std.mem.readInt(i16, raw[i * 2 ..][0..2], .little);
    }
    return out;
}

const STRT: [4096]i16 = i16arr(G_STRT, 4096);
const RCPR: [512]i16 = i16arr(G_RCPR, 512);

// STA blocks in state_table_tables.bin order: [0]=STA1,[1]=STA2,[2]=STA4,[3]=STA5,[4]=STA6,[5]=STA7.
fn sta(s: i32) *const [1024]u8 {
    const idx: usize = switch (s) {
        1 => 0,
        2 => 1,
        4 => 2,
        5 => 3,
        6 => 4,
        else => 5,
    };
    return G_STA[idx * 1024 ..][0..1024];
}

inline fn get_state_byte_location(bpos: i32, c0: i32) i32 {
    const smask: i32 = @intCast((@as(u32, 0x31031010) >> @as(u5, @intCast(bpos << 2))) & 0x0F);
    return smask + (c0 & smask);
}

fn run_cfg(a: std.mem.Allocator, mem: u32, c: usize, cms: i32, cms3: i32, sta_id: i32, cms4: i32, kep: u8, skip2: i32) !u64 {
    var cm = ContextMap4.new();
    try cm.init(a, mem, c, cms, cms3, cms4, sta(sta_id), kep, skip2, &RCPR, &STRT);
    defer cm.deinit();
    var x = X{ .c0 = 1 };
    var cs: u64 = 0;
    var r: u32 = 0x12345678;
    var in1_buf: [544]i16 = undefined;
    var slot_buf: [560]i16 = undefined;
    var in1_n: usize = 0;
    var slot_n: usize = 0;
    const sink = Sink{ .in1 = &in1_buf, .in1_n = &in1_n, .slots = &slot_buf, .slot_n = &slot_n };
    var iter: usize = 0;
    while (iter < 4000) : (iter += 1) {
        r = r *% 1664525 +% 1013904223;
        const by: i32 = @intCast((r >> 16) & 0xff);
        var b: i32 = 7;
        while (b >= 0) : (b -= 1) {
            const bit = (by >> @as(u5, @intCast(b))) & 1;
            x.y = bit;
            x.c0 += x.c0 + x.y;
            if (x.c0 >= 256) {
                x.c4 = (x.c4 << 8) +% (@as(u32, @intCast(x.c0)) & 0xff);
                x.c0 = 1;
            }
            x.bpos = (x.bpos + 1) & 7;
            x.bposshift = 7 - x.bpos;
            x.c0shift_bpos = (x.c0 << 1) ^ (@as(i32, 256) >> @as(u5, @intCast(x.bposshift)));
            x.cm_bit_state = get_state_byte_location(x.bpos, x.c0);
            if (x.bpos == 0) {
                var j: usize = 0;
                while (j < c) : (j += 1) {
                    const salt = (@as(u32, @intCast(j)) *% 0x9e3779b1) ^ x.c4;
                    if (salt & 7 == 0) {
                        cm.sets();
                    } else {
                        cm.set(((x.c4 & 0x3ff) *% 1000003) +% @as(u32, @intCast(j)) *% 40503);
                    }
                }
            }
            in1_n = 0;
            slot_n = 0;
            _ = cm.mix(&x, &sink);
            for (in1_buf[0..in1_n]) |v| {
                cs = cs *% 1000003 +% @as(u64, @as(u16, @bitCast(v)));
            }
        }
    }
    return cs;
}

test "cm4_matches_cpp_oracle" {
    const a = testing.allocator;
    const Case = struct {
        name: []const u8,
        mem: u32,
        c: usize,
        cms: i32,
        cms3: i32,
        sta_id: i32,
        cms4: i32,
        kep: u8,
        skip2: i32,
        golden: u64,
    };
    const cases = [_]Case{
        .{ .name = "cmC4[0]", .mem = 1 << 16, .c = 2, .cms = 35, .cms3 = 35, .sta_id = 6, .cms4 = 12, .kep = 0x00, .skip2 = 0, .golden = 0xa0655a946b6d4dae },
        .{ .name = "cmC4[1]", .mem = 1 << 16, .c = 3, .cms = 29, .cms3 = 33, .sta_id = 7, .cms4 = 10, .kep = 0x00, .skip2 = 1, .golden = 0xbaf986da774e9470 },
        .{ .name = "cmC4[2]", .mem = 1 << 16, .c = 4, .cms = 32, .cms3 = 28, .sta_id = 2, .cms4 = 7, .kep = 0x00, .skip2 = 1, .golden = 0x484b54b91f530066 },
        .{ .name = "cmC4[4]", .mem = 1 << 16, .c = 6, .cms = 33, .cms3 = 31, .sta_id = 7, .cms4 = 7, .kep = 0x00, .skip2 = 1, .golden = 0x71ab380d17fe21c8 },
        .{ .name = "cmC4[3]", .mem = 1 << 16, .c = 2, .cms = 31, .cms3 = 33, .sta_id = 1, .cms4 = 13, .kep = 0x00, .skip2 = 0, .golden = 0xddc02bbae8f158c1 },
        .{ .name = "cmCR[0]", .mem = 1 << 16, .c = 1, .cms = 29, .cms3 = 33, .sta_id = 7, .cms4 = 10, .kep = 0x00, .skip2 = 1, .golden = 0xe9f2b19bde740215 },
        .{ .name = "cmCR[2]", .mem = 1 << 16, .c = 1, .cms = 29, .cms3 = 33, .sta_id = 6, .cms4 = 10, .kep = 0x00, .skip2 = 0, .golden = 0x5b3f12a5a90a6248 },
        .{ .name = "cmC4[7]", .mem = 1 << 16, .c = 4, .cms = 33, .cms3 = 31, .sta_id = 2, .cms4 = 7, .kep = 0x00, .skip2 = 1, .golden = 0x4aa1085ffe170440 },
        .{ .name = "big-mem", .mem = 1 << 22, .c = 3, .cms = 29, .cms3 = 33, .sta_id = 7, .cms4 = 10, .kep = 0x00, .skip2 = 1, .golden = 0x27d547af1b59c6a0 },
    };
    for (cases) |cse| {
        const got = try run_cfg(a, cse.mem, cse.c, cse.cms, cse.cms3, cse.sta_id, cse.cms4, cse.kep, cse.skip2);
        try testing.expectEqual(cse.golden, got);
    }
}
