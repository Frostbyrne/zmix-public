//! fxcm_v26 precomputed tables — port of tables.rs (the `Predictor::Predictor`
//! precalc block in fxcmv1.cpp, VERSION 26).
//!
//! `squashc`/`stretchc` derive integer tables from `exp`/`log` then `round` to an
//! integer. `ilog`/`dt` are pure integer. Verified against the compiled C++ via
//! checksums over all entries (see tests).
const std = @import("std");
// Pinned float math: the SAME implementations detm_math.zig exports as the
// binary's libm symbols (musl logf, ARM optimized-routines exp). Calling them
// directly (instead of @log/@exp libcalls) makes table generation bit-identical
// in every build — including `zig test`, which does not link the detm_math
// overrides and would otherwise resolve @log/@exp against the host libm.
const crt_log = @import("../../vendor/crt/log.zig");
const fast_math = @import("../../vendored_math.zig");

/// `squashc(d)` — 12-bit squash of an 8-bit-scaled logit.
fn squashc(d: i32) i16 {
    if (d < -2047) return 1;
    if (d > 2047) return 4095;
    // p = 1/(1+exp(-d/256.0)); arg is double, so exp is the double exp. Stored to float.
    const dd: f64 = @floatFromInt(d);
    const p1: f32 = @floatCast(1.0 / (1.0 + fast_math.exp(-dd / 256.0)));
    const p2: f32 = @floatCast(@as(f64, p1) * 4096.0); // p *= 4096.0 (double), back to float
    var pi: u32 = @intFromFloat(@round(p2)); // (U32)round(p)
    if (pi > 4095) pi = 4095;
    if (pi < 1) pi = 1;
    return @intCast(pi);
}

/// `stretchc(p)` — inverse of squash. `log` here has a float arg → `logf`.
fn stretchc(p_in: i32) i16 {
    const p: i32 = if (p_in == 0) 1 else p_in;
    const f: f32 = @as(f32, @floatFromInt(p)) / 4096.0;
    const d: f32 = crt_log.logf(f / (1.0 - f)) * 256.0;
    var di: i32 = @intFromFloat(@round(d));
    if (di > 2047) di = 2047;
    if (di < -2047) di = -2047;
    return @intCast(di);
}

/// `sc(p)` — fxcmv1.cpp:1013 (arithmetic shift by 7, rounding toward zero).
fn sc(p: i32) i32 {
    if (p > 0) return p >> 7;
    return (p + 127) >> 7;
}

/// `InitIlog` — fxcmv1.cpp:258-266, pure integer. `ilog[0]` stays 0 (the C++
/// global array is zero-initialized and the loop starts at index 1).
fn genIlog() [256]u8 {
    var ilog = [_]u8{0} ** 256;
    var x: u32 = 14_155_776;
    var i: u32 = 2;
    while (i < 257) : (i += 1) {
        x +%= 774_541_002 / (i * 2 - 1); // numerator is 2^29/ln 2
        ilog[i - 1] = @intCast(x >> 24);
    }
    return ilog;
}

// ---------------------------------------------------------------------------
// Startup regeneration of the Predictor-ctor lookup tables (fxcmv1.cpp:
// 5841-5859). These replace the ~25KB of embedded golden .bin dumps in
// fxcm_v26.zig; each generator is verified byte-identical to its golden by the
// tests below (the goldens stay in the repo as test-only references).
// ---------------------------------------------------------------------------

/// `strt[i] = stretchc(i)` for i in 0..=4095 (golden: cm3_STRT.bin/dsm_STRT.bin).
pub fn genStrt() [4096]i16 {
    var strt: [4096]i16 = undefined;
    for (&strt, 0..) |*slot, i| slot.* = stretchc(@intCast(i));
    return strt;
}

/// `sqt[i+2047] = squashc(i)` for i in -2047..=2047 (golden: dsm_SQT.bin).
/// `sqt[4095]` stays 0: the C++ global array is zero-initialized and the ctor
/// loop never writes that slot.
pub fn genSqt() [4096]i16 {
    var sqt = [_]i16{0} ** 4096;
    var d: i32 = -2047;
    while (d <= 2047) : (d += 1) {
        sqt[@intCast(d + 2047)] = squashc(d);
    }
    return sqt;
}

/// `st2_p1[i] = clp(sc(13*(i - 2048)))` — pure integer (golden: cm3_ST2_P1.bin).
pub fn genSt2P1() [4096]i16 {
    var st2: [4096]i16 = undefined;
    for (&st2, 0..) |*slot, i| {
        slot.* = clp(sc(13 * (@as(i32, @intCast(i)) - 2048)));
    }
    return st2;
}

/// `rcpr[rc+256] = clp(ilog[rc] << (2+(~rc&1)))`, `rcpr[rc] = clp(-that)` —
/// pure integer (golden: cm3_RCPR.bin/cm4_RCPR.bin).
pub fn genRcpr() [512]i16 {
    const ilog = genIlog();
    var rcpr: [512]i16 = undefined;
    for (0..256) |rc| {
        var c: i32 = ilog[rc];
        c = c << @intCast(2 + (~@as(u32, @intCast(rc)) & 1));
        rcpr[rc + 256] = clp(c);
        rcpr[rc] = clp(-c);
    }
    return rcpr;
}

/// Runtime clamp helpers (fxcmv1.cpp `clp`/`clp1`).
pub inline fn clp(z: i32) i16 {
    if (z < -2047) return -2047;
    if (z > 2047) return 2047;
    return @intCast(z);
}

pub inline fn clp1(z: i32) i16 {
    if (z < 0) return 0;
    if (z > 4095) return 4095;
    return @intCast(z);
}

/// The fxcm tables.
pub const Tables = struct {
    sqt: []i16, // 4095 entries
    strt: []i16, // 4096 entries
    ilog: []u8, // 256 entries
    dt: []i32, // 1024 entries

    pub fn new(a: std.mem.Allocator) !Tables {
        const sqt = try a.alloc(i16, 4095);
        const strt = try a.alloc(i16, 4096);
        const ilog = try a.alloc(u8, 256);
        const dt = try a.alloc(i32, 1024);
        @memset(sqt, 0);
        @memset(strt, 0);
        @memset(ilog, 0);
        @memset(dt, 0);

        // dt[i] = 4096/(i+2), then dt[1023]=1
        var o: i32 = 2;
        for (dt) |*slot| {
            slot.* = @divTrunc(4096, o);
            o += 1;
        }
        dt[1023] = 1;

        for (strt, 0..) |*slot, i| {
            slot.* = stretchc(@intCast(i));
        }
        var d: i32 = -2047;
        while (d <= 2047) : (d += 1) {
            sqt[@intCast(d + 2047)] = squashc(d);
        }

        // InitIlog: x += 2^29/ln2 / (2i-1), pure integer.
        var x: u32 = 14_155_776;
        var i: i32 = 2;
        while (i < 257) : (i += 1) {
            x +%= 774_541_002 / @as(u32, @intCast(i * 2 - 1));
            ilog[@intCast(i - 1)] = @intCast(x >> 24);
        }

        return Tables{ .sqt = sqt, .strt = strt, .ilog = ilog, .dt = dt };
    }

    pub fn deinit(self: *Tables, a: std.mem.Allocator) void {
        a.free(self.sqt);
        a.free(self.strt);
        a.free(self.ilog);
        a.free(self.dt);
    }

    /// `squash(d)` runtime lookup with clamping (`fxcmv1.cpp::squash`).
    pub inline fn squash(self: *const Tables, d: i32) i32 {
        if (d < -2047) return 1;
        if (d > 2047) return 4095;
        return self.sqt[@intCast(d + 2047)];
    }

    /// `stretch(p)` runtime lookup.
    pub inline fn stretch(self: *const Tables, p: i32) i16 {
        return self.strt[@intCast(p)];
    }
};

// Checksums (cs = cs*1000003 + value) from the compiled C++ (fxtables_ref.cpp).
const SQT_CS: u64 = 10603081325546183832;
const STRT_CS: u64 = 10697758682949088048;
const ILOG_CS: u64 = 11347391304047397951;
const DT_CS: u64 = 10175721186995242730;

test "tables_match_cpp" {
    const a = std.testing.allocator;
    var t = try Tables.new(a);
    defer t.deinit(a);

    var sqt: u64 = 0;
    for (t.sqt) |v| {
        sqt = sqt *% 1000003 +% @as(u64, @as(u16, @bitCast(v)));
    }
    var strt: u64 = 0;
    for (t.strt) |v| {
        strt = strt *% 1000003 +% @as(u64, @as(u16, @bitCast(v)));
    }
    var ilog: u64 = 0;
    for (t.ilog) |v| {
        ilog = ilog *% 1000003 +% @as(u64, v);
    }
    var dt: u64 = 0;
    for (t.dt) |v| {
        dt = dt *% 1000003 +% @as(u64, @as(u32, @bitCast(v)));
    }
    try std.testing.expectEqual(SQT_CS, sqt);
    try std.testing.expectEqual(STRT_CS, strt);
    try std.testing.expectEqual(ILOG_CS, ilog);
    try std.testing.expectEqual(DT_CS, dt);
}

test "table_samples" {
    const a = std.testing.allocator;
    var t = try Tables.new(a);
    defer t.deinit(a);

    try std.testing.expectEqual(@as(i16, 2048), t.sqt[2047]);
    try std.testing.expectEqual(@as(i16, 2052), t.sqt[2048]);
    try std.testing.expectEqual(@as(i16, 2056), t.sqt[2049]);
    try std.testing.expectEqual(@as(i16, -2047), t.strt[0]);
    try std.testing.expectEqual(@as(i16, 0), t.strt[2048]);
    try std.testing.expectEqual(@as(i16, 2047), t.strt[4095]);
    try std.testing.expectEqual(@as(i32, 2048), t.dt[0]);
    try std.testing.expectEqual(@as(i32, 1365), t.dt[1]);
    try std.testing.expectEqual(@as(i32, 1), t.dt[1023]);
    try std.testing.expectEqual(@as(i32, 2048), t.squash(0));
}

// ---------------------------------------------------------------------------
// Golden parity for the startup-regenerated tables. The .bin goldens were
// captured from the C++ oracle during porting; they are test-only references
// now (embedded here, in test-only decls — NOT in the shipping binary).
// ---------------------------------------------------------------------------

fn goldenI16(comptime path: []const u8, comptime n: usize) [n]i16 {
    @setEvalBranchQuota(n * 8 + 1000);
    const bytes = @embedFile(path);
    var arr: [n]i16 = undefined;
    for (0..n) |i| arr[i] = std.mem.readInt(i16, bytes[i * 2 ..][0..2], .little);
    return arr;
}

test "genStrt matches golden cm3_STRT.bin (== dsm_STRT.bin)" {
    const want = goldenI16("goldens/cm3_STRT.bin", 4096);
    const got = genStrt();
    try std.testing.expectEqualSlices(i16, &want, &got);
}

test "genSqt matches golden dsm_SQT.bin (incl. the sqt[4095]=0 quirk)" {
    const want = goldenI16("goldens/dsm_SQT.bin", 4096);
    const got = genSqt();
    try std.testing.expectEqualSlices(i16, &want, &got);
}

test "genSt2P1 matches golden cm3_ST2_P1.bin" {
    const want = goldenI16("goldens/cm3_ST2_P1.bin", 4096);
    const got = genSt2P1();
    try std.testing.expectEqualSlices(i16, &want, &got);
}

test "genRcpr matches golden cm3_RCPR.bin (== cm4_RCPR.bin)" {
    const want = goldenI16("goldens/cm3_RCPR.bin", 512);
    const got = genRcpr();
    try std.testing.expectEqualSlices(i16, &want, &got);
}
