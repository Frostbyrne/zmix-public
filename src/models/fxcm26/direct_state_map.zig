//! DirectStateMap (+ its StateMap and Mix deps) — bit-exact Zig port of cmix-lex
//! `fxcmv1.cpp`. Owns slots 489-493 (dcsm/dcsm0/dcsm1/dcsm2/dcsmN, 1 model slot each;
//! the per-bank `set` emits an `add_internal(pre1[state])` that is mixer-only).
//!
//! Cadence (from update): per bit, `set` is called `count` times (index 0..count),
//! then `mix` emits `pu>>2` and resets index. Validated bit-exact vs
//! v26-port-tools/dsm_oracle.cpp.
//!
//! Ported 1:1 from direct_state_map_v26.rs. The private `StateMap`/`Mix` here differ
//! from the sibling `state_map.zig`/mixer modules (this StateMap uses `>>14`/`<<18`
//! leaks, the Mix is a 2-input per-context combiner), so they are kept local.

const std = @import("std");
// This file lives in the `cmfast` module (see cm_fast.zig); byte_core.zig is
// module-local, zero_alloc is the module-private copy (the root module also
// imports src/zero_alloc.zig, and a file cannot belong to two modules).
const X = @import("cmcold").X;
const zero_alloc = @import("cmcold").zero_alloc;

/// C++ `clp`: clamp to the stretched-logit range. Returns i32 (callers promote).
inline fn clp(z: i32) i32 {
    if (z < -2047) {
        return -2047;
    } else if (z > 2047) {
        return 2047;
    } else {
        return z;
    }
}

/// Placeholder targets for the shared-table pointers before `init` (never read).
/// Zero-filled, NOT the golden tables: pointing the placeholders at the embedded
/// goldens dragged 17KB of .bin dumps into the shipping binary for pointers that
/// `init` immediately overwrites.
const NN_INIT: [1024]u8 = .{0} ** 1024;
const STRT_INIT: [4096]i16 = .{0} ** 4096;
const SQT_INIT: [4096]i16 = .{0} ** 4096;
const PRE1_INIT: [256]i16 = .{0} ** 256;

/// -Ddsm-sm-rate — the third member of the state->probability DECODER class (see the audit's
/// §12y-c). Same shape as `ContextMap3.ts[]`: a fixed-rate 256-entry EMA whose two shifts are
/// locked to sum to 32, set once by cmix and never re-derived. At e9 each entry receives
/// 4.697e9/256 = 18.3 M updates against τ = 2^14 ⇒ **1,120 time constants** of unused
/// statistical strength. `ts[]`'s own ladder says the shipped rate there is too slow.
/// Default 14 = shipped, bit-identical.
const SM_RATE: u32 = @import("build_options").dsm_sm_rate;
const SM_SHR: u5 = @intCast(SM_RATE);
const SM_SHL: u5 = @intCast(32 - SM_RATE);
comptime {
    if (SM_RATE < 10 or SM_RATE > 20) @compileError("-Ddsm-sm-rate must be in 10..20");
}

/// StateMap (no update limit). `t`: cxt -> prediction in high 22 bits, count low 10.
const StateMap = struct {
    t: []u32,
    cxt: usize,
    pr: i32,
    nn: *const [1024]u8,

    fn new() StateMap {
        return StateMap{ .t = &[_]u32{}, .cxt = 0, .pr = 2048, .nn = &NN_INIT };
    }

    inline fn next(self: *const StateMap, i: i32, y: i32) u8 {
        return self.nn[@intCast(y + i * 4)];
    }

    fn init(self: *StateMap, a: std.mem.Allocator, n: usize, nn: *const [1024]u8) !void {
        self.cxt = 0;
        self.pr = 2048;
        self.nn = nn;
        self.t = try a.alloc(u32, n);
        var i: usize = 0;
        while (i < n) : (i += 1) {
            const n0 = @as(u32, self.next(@intCast(i), 2)) *% 3 +% 1;
            const n1 = @as(u32, self.next(@intCast(i), 3)) *% 3 +% 1;
            self.t[i] = ((n1 << 20) / (n0 +% n1)) << 12;
        }
    }

    inline fn update(self: *StateMap, y: i32) void {
        const p0 = self.t[self.cxt];
        const pr1: i32 = @intCast(p0 >> SM_SHR);
        const delta: u32 = @bitCast((y << SM_SHL) - pr1);
        self.t[self.cxt] = p0 +% delta;
    }

    inline fn set(self: *StateMap, c: usize, y: i32) void {
        self.update(y);
        self.cxt = c;
        self.pr = @intCast(self.t[c] >> 20);
    }

    fn deinit(self: *StateMap, a: std.mem.Allocator) void {
        a.free(self.t);
    }
};

/// Mix — 2-input per-context adaptive combiner (weights scaled 24 bits).
const Mix = struct {
    wt: []i32,
    x1: i32,
    x2: i32,
    cxt: usize,
    pr: i32,
    // Shared pointer, NOT an embedded copy: C++ has ONE global table; per-
    // instance 8KB copies were evicting each other (45% of seton one load).
    sqt: *const [4096]i16,

    fn new() Mix {
        return Mix{ .wt = &[_]i32{}, .x1 = 0, .x2 = 0, .cxt = 0, .pr = 0, .sqt = &SQT_INIT };
    }

    fn init(self: *Mix, a: std.mem.Allocator, n: usize, sqt: *const [4096]i16) !void {
        self.x1 = 0;
        self.x2 = 0;
        self.cxt = 0;
        self.pr = 0;
        self.wt = try a.alloc(i32, n * 2);
        for (self.wt) |*w| {
            w.* = @as(i32, 1) << 23;
        }
        self.sqt = sqt;
    }

    inline fn squash(self: *const Mix, d: i32) i32 {
        if (d < -2047) {
            return 1;
        } else if (d > 2047) {
            return 4095;
        } else {
            return @as(i32, self.sqt[@intCast(d + 2047)]);
        }
    }

    inline fn pp(self: *Mix, p1: i32, p2: i32, cx: usize) i32 {
        self.cxt = cx * 2;
        self.x1 = p1;
        self.x2 = p2;
        const v = ((p1 *% (self.wt[self.cxt] >> 16)) +%
            (p2 *% (self.wt[self.cxt + 1] >> 16)) +% 128) >> 8;
        self.pr = v;
        return v;
    }

    inline fn update(self: *Mix, y: i32) void {
        var err = (y << 12) -% self.squash(self.pr);
        if ((self.wt[self.cxt] & 3) < 3) {
            self.wt[self.cxt] = self.wt[self.cxt] +% 1;
            err = err *% (4 - (self.wt[self.cxt] & 3));
        }
        err = (err +% 8) >> 4;
        self.wt[self.cxt] = self.wt[self.cxt] +% ((self.x1 *% err) & @as(i32, -4));
        self.wt[self.cxt + 1] = self.wt[self.cxt + 1] +% (self.x2 *% err);
    }

    fn deinit(self: *Mix, a: std.mem.Allocator) void {
        a.free(self.wt);
    }
};

/// Shared placeholder pointer for pre-init struct fields elsewhere.
pub const STRT_PTR: *const [4096]i16 = &STRT;

/// -Ddsm-tag — COLLISION DETECTION for the five DirectStateMap banks.
///
/// `cxt_state` is 2^m bytes of bit-history state indexed by `cx & mask` with NO
/// checksum, NO bucket and NO owner — the same structure `indirect-tag` measured
/// at 37.90 % foreign reads on the (much smaller) 102 MB shared map, where one tag
/// byte per state was Pareto-dominant. These five banks are 639 MB — 6× the shared
/// map — and every ContextMap in the same binary tags its entries.
///
/// ON: positions halve, every state gets an interleaved tag byte, and a read whose
/// tag does not match claims the position and returns a fresh state 0. TOTAL MEMORY
/// IS UNCHANGED, so this is an exchange of positions for detection, exactly as
/// `-Dind-tag=2`. Decoder-symmetric (the tag is a function of decoded bytes only).
/// Default 0 ⇒ every field and branch is comptime-dead and the layout is stock.
/// 0 = stock, 1 = interleaved tag at equal RAM, 2 = halved-positions ISOLATING CONTROL
/// (same position count as mode 1, no tag) — without it a mode-1 reading confounds
/// "detection helps" with "half the positions helps/hurts".
const DSM_MODE: u32 = @import("build_options").dsm_tag;
/// True whenever the POSITION COUNT is halved (modes 1 and 2).
const DSM_HALF: bool = DSM_MODE != 0;
/// True only when tags are stored and checked (mode 1).
const DSM_TAG: bool = DSM_MODE == 1;
/// -Ddsm-div=2^k — shrink every DirectStateMap bank by 2^k positions. The LOAD-FACTOR
/// knob: at e9 these banks run at load factor >>1, but a 1 MB screen runs them at ~0.15,
/// so a tag can find nothing to detect. Dividing the table reproduces the e9 aliasing
/// regime at a small tier instead of waiting for 1e9 bytes — the instrument
/// `indirect-admission` banked (`-Dshared-map-div`). Default 0 = stock size.
const DSM_DIV_LOG2: u32 = @import("build_options").dsm_div_log2;

pub const DirectStateMap = struct {
    a: std.mem.Allocator,
    sm: []StateMap,
    mmm: []Mix,
    cxt: []u32,
    cxt_state: []u8,
    /// -Ddsm-tag only: the tag of the context each slot currently points at.
    cxt_tag: if (DSM_TAG) []u8 else void,
    mask: u32,
    index: usize,
    count: usize,
    // Shared read-only state-advance table (C++ `const U8 *nn` into one STA*
    // global) — hot in setvia next; was an embedded per-instance copy.
    nn: *const [1024]u8,
    pu: i32,
    strt: *const [4096]i16,
    pre1: *const [256]i16,
    pub fn new() DirectStateMap {
        return DirectStateMap{
            .a = undefined,
            .sm = &[_]StateMap{},
            .mmm = &[_]Mix{},
            .cxt = &[_]u32{},
            .cxt_state = &[_]u8{},
            .cxt_tag = if (DSM_TAG) &[_]u8{} else {},
            .mask = 0,
            .index = 0,
            .count = 0,
            .nn = &NN_INIT,
            .pu = 0,
            .strt = &STRT_INIT,
            .pre1 = &PRE1_INIT,
        };
    }

    inline fn next(self: *const DirectStateMap, state: i32, y: i32) u8 {
        return self.nn[@intCast(state * 4 + y)];
    }

    inline fn stretch(self: *const DirectStateMap, p: i32) i32 {
        return @as(i32, self.strt[@intCast(std.math.clamp(p, 0, 4095))]);
    }

    pub fn init(
        self: *DirectStateMap,
        a: std.mem.Allocator,
        m: u32,
        c: usize,
        nn: *const [1024]u8,
        strt: *const [4096]i16,
        sqt: *const [4096]i16,
        pre1: *const [256]i16,
    ) !void {
        self.a = a;
        self.pu = 0;
        // -Ddsm-tag: halve the POSITION count so [state,tag] pairs cost exactly the
        // stock allocation. m>=1 for every shipped bank (20..28).
        const eff_m: u32 = @max(@as(u32, 8), (if (DSM_HALF) m - 1 else m) - DSM_DIV_LOG2);
        self.mask = (@as(u32, 1) << @intCast(eff_m)) -% 1;
        self.index = 0;
        self.count = c;
        self.nn = nn;
        self.strt = strt;
        self.pre1 = pre1;
        self.cxt = try a.alloc(u32, c);
        @memset(self.cxt, 0);
        if (comptime DSM_TAG) {
            self.cxt_tag = try a.alloc(u8, c);
            @memset(self.cxt_tag, 0);
        }
        // calloc-parity (C++ alloc): the two m=28 banks alone are 512 MB;
        // kernel-zeroed lazy pages instead of an up-front commit.
        self.cxt_state = try zero_alloc.alloc(a, u8, (@as(usize, self.mask) + 1) * (if (DSM_HALF) @as(usize, 2) else 1));
        self.sm = try a.alloc(StateMap, c);
        for (self.sm) |*s| {
            s.* = StateMap.new();
            try s.init(a, 256, nn);
        }
        self.mmm = try a.alloc(Mix, c);
        for (self.mmm) |*mx| {
            mx.* = Mix.new();
            try mx.init(a, 256, sqt);
        }
    }

    pub fn deinit(self: *DirectStateMap) void {
        const a = self.a;
        for (self.sm) |*s| s.deinit(a);
        for (self.mmm) |*mx| mx.deinit(a);
        a.free(self.sm);
        a.free(self.mmm);
        a.free(self.cxt);
        if (comptime DSM_TAG) a.free(self.cxt_tag);
        zero_alloc.free(a, self.cxt_state);
    }

    /// Prefetch the state byte `cx` will select (speed hint only).
    /// `inline`: tiny; a cross-module out-of-line call costs more than the body.
    pub inline fn prefetch(self: *const DirectStateMap, cx: u32) void {
        const p: usize = if (comptime DSM_HALF) @as(usize, cx & self.mask) * 2 else @as(usize, cx & self.mask);
        @prefetch(&self.cxt_state[p], .{});
    }

    /// sm33: prefetch each slot's CARRIED-OVER (previous-bit) context state.
    /// setreads cxt_state[cxt[j]] (line ~248) for the prior context before
    /// overwriting cxt[j]; that byte was last touched a full predict/perceive ago
    /// (likely evicted from the 512 MB bank) and `prefetch(cx)` only covers the
    /// NEW context. cxt[j] is already masked. Pure cache hint, bit-identical.
    pub inline fn prefetchPrev(self: *const DirectStateMap) void {
        for (self.cxt) |c| @prefetch(&self.cxt_state[@as(usize, c)], .{});
    }

    /// Returns the `add_internal(pre1[state])` mixer-only value (clipped), which the
    /// predictor pushes into mxInputs1 at this exact call position (fxcmv1.cpp 3574).
    /// -Ddsm-tag: 8 fresh bits from a multiplicative mix of the FULL context, so
    /// the tag is ~independent of the low index bits (same construction as
    /// `-Dcmc2-hash64`). `| 1` keeps 0 reserved for "never claimed".
    inline fn tagOf(cx: u32) u8 {
        return @as(u8, @truncate(((cx ^ (cx >> 15)) *% 2654435761) >> 24)) | 1;
    }

    pub fn set(self: *DirectStateMap, cx: u32, x: *const X) i16 {
        const ci: usize = @intCast(self.cxt[self.index]);
        self.cxt_state[ci] = self.next(@as(i32, self.cxt_state[ci]), x.y);
        if (comptime DSM_HALF) {
            const off: u32 = (cx & self.mask) *% 2;
            self.cxt[self.index] = off;
            if (comptime DSM_TAG) {
                const tg = tagOf(cx);
                self.cxt_tag[self.index] = tg;
                const to: usize = @as(usize, off) + 1;
                if (self.cxt_state[to] != tg) { // foreign owner => claim + honest fresh state
                    self.cxt_state[to] = tg;
                    self.cxt_state[@intCast(off)] = 0;
                }
            }
        } else {
            self.cxt[self.index] = cx & self.mask;
        }
        const state: i32 = @as(i32, self.cxt_state[@as(usize, @intCast(self.cxt[self.index]))]);
        self.sm[self.index].set(@intCast(state), x.y);
        const stretched_pr = self.stretch(self.sm[self.index].pr);
        if (self.index == 0) {
            self.pu = stretched_pr;
        } else {
            self.mmm[self.index - 1].update(x.y);
            self.pu = clp(self.mmm[self.index - 1].pp(self.pu, stretched_pr, @intCast(state)));
        }
        // add_internal(pre1[state]) is mixer-only; not a model slot — returned to
        // the caller for mxInputs1.
        self.index += 1;
        return @intCast(clp(@as(i32, self.pre1[@intCast(state)])));
    }

    /// Returns the mixed slot value directly (a single value per mix — the
    /// buffer round-trip through a 1-slot emitted array cost a store+load per
    /// bit per instance for nothing; C++ writes its fixed array once, we return).
    pub inline fn mix(self: *DirectStateMap) i16 {
        const v: i16 = @intCast(clp(self.pu >> 2));
        self.index = 0;
        self.pu = 0;
        return v;
    }
};

// ---------------------------------------------------------------------------
// Tests (ported from direct_state_map_v26.rs `mod tests`).
// Fixtures STA7/STRT/SQT/PRE1 are the exact cm3_tables.rs arrays, dumped to
// little-endian binary goldens.
// ---------------------------------------------------------------------------

const G_STA7 = @embedFile("goldens/dsm_STA7.bin");
const G_STRT = @embedFile("goldens/dsm_STRT.bin");
const G_SQT = @embedFile("goldens/dsm_SQT.bin");
const G_PRE1 = @embedFile("goldens/dsm_PRE1.bin");

fn i16arr(comptime raw: anytype, comptime N: usize) [N]i16 {
    @setEvalBranchQuota(4 * N + 1000);
    var out: [N]i16 = undefined;
    for (0..N) |i| {
        out[i] = std.mem.readInt(i16, raw[i * 2 ..][0..2], .little);
    }
    return out;
}

const STA7: [1024]u8 = G_STA7[0..1024].*;
const STRT: [4096]i16 = i16arr(G_STRT, 4096);
const SQT: [4096]i16 = i16arr(G_SQT, 4096);
const PRE1: [256]i16 = i16arr(G_PRE1, 256);

inline fn gsbl(bpos: i32, c0: i32) i32 {
    const smask: i32 = @intCast((@as(u32, 0x31031010) >> @intCast(bpos << 2)) & 0x0F);
    return smask + (c0 & smask);
}

fn run_dsm(a: std.mem.Allocator, m: u32, count: usize) !u64 {
    var d = DirectStateMap.new();
    try d.init(a, m, count, &STA7, &STRT, &SQT, &PRE1);
    defer d.deinit();

    var x = X{ .c0 = 1 };
    var cs: u64 = 0;
    var r: u32 = 0x12345678;
    var iter: usize = 0;
    while (iter < 4000) : (iter += 1) {
        r = r *% 1664525 +% 1013904223;
        const by: i32 = @intCast((r >> 16) & 0xff);
        var b: i32 = 7;
        while (b >= 0) : (b -= 1) {
            x.y = (by >> @intCast(b)) & 1;
            x.c0 = x.c0 +% (x.c0 +% x.y);
            if (x.c0 >= 256) {
                x.c4 = (x.c4 << 8) +% (@as(u32, @intCast(x.c0)) & 0xff);
                x.c0 = 1;
            }
            x.bpos = (x.bpos + 1) & 7;
            x.bposshift = 7 - x.bpos;
            x.c0shift_bpos = (x.c0 << 1) ^ (@as(i32, 256) >> @intCast(x.bposshift));
            x.cm_bit_state = gsbl(x.bpos, x.c0);
            var j: usize = 0;
            while (j < count) : (j += 1) {
                const cx = (((x.c4 & 0xff) *% 191 +% (@as(u32, @intCast(j)) *% 7)) *% 256) +%
                    @as(u32, @intCast(x.c0));
                _ = d.set(cx, &x);
            }
            const v = d.mix();
            cs = cs *% 1000003 +% @as(u64, @as(u16, @bitCast(v)));
        }
    }
    return cs;
}

test "dsm_matches_cpp_oracle" {
    const a = std.testing.allocator;
    try std.testing.expectEqual(@as(u64, 0x6d8c69e05672845b), try run_dsm(a, 12, 5)); // dcsm(m12,count5)
    try std.testing.expectEqual(@as(u64, 0x85bd18801934e232), try run_dsm(a, 12, 6)); // dcsm0(m12,count6)
    try std.testing.expectEqual(@as(u64, 0xd1c80b6429908aee), try run_dsm(a, 12, 2)); // dcsm1(m12,count2)
    try std.testing.expectEqual(@as(u64, 0x265a40ad398c96c8), try run_dsm(a, 12, 3)); // dcsm2(m12,count3)
    try std.testing.expectEqual(@as(u64, 0xa3b0c158d91731fa), try run_dsm(a, 16, 5)); // dcsm(m16,count5)
}
