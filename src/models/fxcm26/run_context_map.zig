//! RunContextMap — port of `struct RunContextMap` in
//! `src/cmix/src/models/fxcmv1.cpp` (~lines 756-819).
//!
//! Maps a context (via a 4-way set-associative, move-to-front hash table) onto the
//! next predicted byte and a repeat count up to ~254. Pure integer; the only float-
//! derived input is `ilog` (threaded in via `tables.Tables`), and the runtime
//! prediction is a table lookup of clamped `ilog`-scaled values. Bit-exact by
//! construction (validated against the compiled C++ oracle).
//!
//! ## Hash-table entry layout (B = 4 bytes per slot)
//! - bytes `[0,1]`: 16-bit checksum (little-endian, written by `find`)
//! - byte `[2]`: repeat count — also the "slot in use" flag (`cp[0]`)
//! - byte `[3]`: the stored/predicted byte (`cp[1]`)
//!
//! `cp` is a *byte index* into `t` pointing at entry-byte-2 (count): `find` returns
//! `i*B + 1` and `set` adds one more, so `cp = i*B + 2`. The sole exception is the
//! initial `cp = 1` (the C++ `cp = &t[0] + 1`), a one-time quirk before any `find`.

const std = @import("std");
// In the `cmfast` module: private copies (see cm_fast.zig / cmf_tables.zig).
const Tables = @import("cmcold").Tables;
const zero_alloc = @import("cmcold").zero_alloc;

const B: usize = 4; // bytes per slot
const M: usize = 4; // associativity

/// `clp` — clamp to the stretched-logit range `[-2047, 2047]` (fxcmv1.cpp::clp).
inline fn clp(z: i32) i16 {
    if (z < -2047) {
        return -2047;
    } else if (z > 2047) {
        return 2047;
    } else {
        return @intCast(z);
    }
}

pub const RunContextMap = struct {
    t: []u8, // hash table (m bytes)
    cp: usize, // byte index into `t`: cp[0]=count, cp[1]=stored byte
    rc: [512]i16,
    n: u32, // slot-index mask: m/B - 1

    /// `Init(m, rcm_ml=8)` — `m` is the table size in bytes (a power of two; with
    /// `M == B` the probe `i+j` can never leave the table). `rcm_ml` is the
    /// even-context multiplier numerator (default 8).
    pub fn new(a: std.mem.Allocator, tables: *const Tables, m: usize, rcm_ml: i32) !RunContextMap {
        const t = try zero_alloc.alloc(a, u8, m); // calloc(m): zero, lazily committed
        const n: u32 = @intCast(m / B - 1);
        var rc = [_]i16{0} ** 512;
        var r: usize = 0;
        while (r < 256) : (r += 1) {
            var c: i32 = @as(i32, @intCast(tables.ilog[r])) * 8;
            if ((r & 1) == 0) {
                c = @divTrunc(c * rcm_ml, 4);
            }
            rc[r + 256] = clp(c);
            rc[r] = clp(-c);
        }
        return RunContextMap{ .t = t, .cp = 1, .rc = rc, .n = n };
    }

    pub fn deinit(self: *RunContextMap, a: std.mem.Allocator) void {
        zero_alloc.free(a, self.t);
    }

    /// `set(cx, c1)` — finalise the previously-selected slot's count/byte, then move
    /// to (and reorder toward the front) the slot for context `cx`.
    pub fn set(self: *RunContextMap, cx: u32, c1: u8) void {
        const cp = self.cp;
        if (self.t[cp] == 0) {
            self.t[cp] = 2;
            self.t[cp + 1] = c1;
        } else if (self.t[cp + 1] != c1) {
            self.t[cp] = 1;
            self.t[cp + 1] = c1;
        } else if (self.t[cp] < 254) {
            self.t[cp] = self.t[cp] +% 2;
        }
        self.cp = self.find(cx) + 1;
    }

    /// `p` — predict the next bit. `c0shift_bpos` and `bposshift` are the global
    /// `x.c0shift_bpos` / `x.bposshift` (`bposshift = 7 - bpos`,
    /// `c0shift_bpos = (c0 << 1) ^ (256 >> bposshift)`).
    pub fn predict(self: *const RunContextMap, c0shift_bpos: i32, bposshift: i32) i32 {
        const cp = self.cp;
        const b = c0shift_bpos ^ (@as(i32, @intCast(self.t[cp + 1])) >> @intCast(bposshift));
        if (b <= 1) {
            return @as(i32, self.rc[@intCast(b * 256 + @as(i32, @intCast(self.t[cp])))]);
        } else {
            return 0;
        }
    }

    /// `mix` run-length component: whether the current slot has a nonzero count.
    pub inline fn run_length(self: *const RunContextMap) bool {
        return self.t[self.cp] != 0;
    }

    /// `find(cx)` — locate the slot for `cx` in its bucket, with move-to-front /
    /// LRU-ish eviction. Returns the byte index `i*B + 1` of the front slot.
    fn find(self: *RunContextMap, cx: u32) usize {
        const chk: u16 = @truncate((cx >> 16) ^ cx); // (cx>>16 ^ cx) & 0xffff
        const i: usize = @intCast((cx *% @as(u32, M)) & self.n);

        var j: usize = 0;
        while (j < M) {
            const base = (i + j) * B;
            if (self.t[base + 2] == 0) {
                // empty slot: write checksum (little-endian) and stop here
                self.t[base] = @truncate(chk & 0xff);
                self.t[base + 1] = @truncate(chk >> 8);
                break;
            }
            const stored: u16 = @as(u16, self.t[base]) | (@as(u16, self.t[base + 1]) << 8);
            if (stored == chk) {
                break; // found
            }
            j += 1;
        }

        if (j == 0) {
            return i * B + 1; // already at front
        }

        var tmp = [_]u8{0} ** B;
        if (j == M) {
            // no slot free/matching: evict. tmp becomes a fresh (checksum-only) slot.
            j -= 1;
            tmp[0] = @truncate(chk & 0xff);
            tmp[1] = @truncate(chk >> 8);
            // keep the slot with the larger count (favour evicting the smaller one)
            if (M > 2 and self.t[(i + j) * B + 2] > self.t[(i + j - 1) * B + 2]) {
                j -= 1;
            }
        } else {
            // matched/empty at j>0: pull that slot's bytes to the front
            const base = (i + j) * B;
            @memcpy(tmp[0..B], self.t[base .. base + B]);
        }

        // shift slots [i .. i+j) down by one, then drop tmp in at the front (memmove).
        std.mem.copyBackwards(u8, self.t[(i + 1) * B .. (i + 1) * B + j * B], self.t[i * B .. i * B + j * B]);
        @memcpy(self.t[i * B .. i * B + B], tmp[0..B]);

        return i * B + 1;
    }
};

// ==== Goldens (from run_context_map_ref.rs; generated from the C++ oracle). ====
const RC_CS: u64 = 208565612243664896;
const P_CS: u64 = 3060242295023227840;
const RL_CS: u64 = 13711932181370575184;
const T_CS: u64 = 5467677365614695368;
const NONZERO: i64 = 616;
const PMIN: i32 = -400;
const PMAX: i32 = 400;
const SAMPLES = [_]i32{
    400, -400, -400, -400, -400, 400, 400, -400, -400, -400, 400, 400, 400, -400, 400, 400,
    400, 400, 400, -400, 400, 400, -400, 400, -400, 400, -400, 400, 400, 400, -400, -400,
    -400, 128, 128, 128, 128, 400, -400, -400, 400, -400, 400, -400, -400, 400, 400, -400,
    400, 400, 400, -400, -400, 400, -400, 400, 400, -400, 400, 400, 400, 400, 400, -400,
};

const MBYTES: usize = 4096; // 1024 slots; n = 1023

test "run_context_map matches cpp oracle" {
    const a = std.testing.allocator;
    var tables = try @import("cmcold").tables.newFromGoldensForTests(a);
    defer tables.deinit(a);
    var rcm = try RunContextMap.new(a, &tables, MBYTES, 8);
    defer rcm.deinit(a);

    // rc[] checksum (validates Init/clp/ilog/rcm_ml).
    var rc_cs: u64 = 0;
    for (rcm.rc) |v| {
        rc_cs = rc_cs *% 1000003 +% @as(u64, @as(u16, @bitCast(v)));
    }
    try std.testing.expectEqual(RC_CS, rc_cs);

    // Same deterministic drive as the oracle.
    var s: u32 = 0x1234_5678;

    var p_cs: u64 = 0;
    var rl_cs: u64 = 0;
    var samples = std.ArrayList(i32){};
    defer samples.deinit(a);
    var nonzero: i64 = 0;
    var pmin: i32 = std.math.maxInt(i32);
    var pmax: i32 = std.math.minInt(i32);

    var c1: i32 = 0;
    var c2: i32 = 0;

    var step: usize = 0;
    while (step < 20000) : (step += 1) {
        // xorshift32
        s ^= s << 13;
        s ^= s >> 17;
        s ^= s << 5;
        const byte: i32 = @intCast(s & 0xff);

        const ctx2: u32 = (@as(u32, @bitCast(c2)) << 8) | @as(u32, @bitCast(c1));
        const cx: u32 = ctx2 *% 2_654_435_761;
        rcm.set(cx, @truncate(@as(u32, @bitCast(c1))));

        var c0: i32 = 1;
        var bp: i32 = 0;
        while (bp < 8) : (bp += 1) {
            const bposshift: i32 = 7 - bp;
            const c0shift_bpos: i32 = (c0 << 1) ^ (@as(i32, 256) >> @intCast(bposshift));
            const pr = rcm.predict(c0shift_bpos, bposshift);
            p_cs = p_cs *% 1000003 +% @as(u64, @as(u16, @bitCast(@as(i16, @truncate(pr)))));
            const rl: u32 = if (rcm.run_length()) 1 else 0;
            rl_cs = rl_cs *% 1000003 +% @as(u64, rl);
            if (pr != 0) {
                nonzero += 1;
                if (samples.items.len < 64) {
                    try samples.append(a, pr);
                }
            }
            if (pr < pmin) pmin = pr;
            if (pr > pmax) pmax = pr;
            const y = (byte >> @intCast(7 - bp)) & 1;
            c0 = (c0 << 1) | y;
        }
        c2 = c1;
        c1 = byte;
    }

    var t_cs: u64 = 0;
    for (rcm.t) |v| {
        t_cs = t_cs *% 1000003 +% @as(u64, v);
    }

    try std.testing.expectEqual(P_CS, p_cs);
    try std.testing.expectEqual(RL_CS, rl_cs);
    try std.testing.expectEqual(T_CS, t_cs);
    try std.testing.expectEqual(NONZERO, nonzero);
    try std.testing.expectEqual(PMIN, pmin);
    try std.testing.expectEqual(PMAX, pmax);
    try std.testing.expectEqualSlices(i32, &SAMPLES, samples.items);
}
