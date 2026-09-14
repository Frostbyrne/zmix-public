//! cmf_tables — cmfast-module-private LAYOUT copy of tables.zig's `Tables`
//! (a file cannot belong to two modules; tables.zig stays in the root module,
//! which also owns construction — newand the float-generation helpers are
//! deliberately NOT copied). Instances are built by the root module and cross
//! the boundary as @ptrCast'd pointers (same slice layout). The runtime
//! lookups + the pure-integer generators are copied verbatim.

const std = @import("std");

/// `InitIlog` — fxcmv1.cpp:258-266, pure integer (copy of tables.zig genIlog).
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

/// `sc(p)` — fxcmv1.cpp:1013 (arithmetic shift by 7, rounding toward zero).
fn sc(p: i32) i32 {
    if (p > 0) return p >> 7;
    return (p + 127) >> 7;
}

/// Runtime clamp helper (fxcmv1.cpp `clp`), copy of tables.zig's.
pub inline fn clp(z: i32) i16 {
    if (z < -2047) return -2047;
    if (z > 2047) return 2047;
    return @intCast(z);
}

/// `st2_p1[i] = clp(sc(13*(i - 2048)))` — pure integer (copy of tables.zig's).
pub fn genSt2P1() [4096]i16 {
    var st2: [4096]i16 = undefined;
    for (&st2, 0..) |*slot, i| {
        slot.* = clp(sc(13 * (@as(i32, @intCast(i)) - 2048)));
    }
    return st2;
}

/// `rcpr[rc+256] = clp(ilog[rc] << (2+(~rc&1)))`, `rcpr[rc] = clp(-that)` —
/// pure integer (copy of tables.zig's).
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

/// TEST-ONLY: Tables built from the embedded dsm goldens — value-identical to
/// the root module's float-generated Tables.new (the root tests.zig
/// `tables_match_cpp` gate proves generator==golden; dsm_STRT/dsm_SQT are the
/// same dumps). Tables.sqt is 4095 long: golden slot 4095 is 0/never written.
/// Never referenced by shipping code (analyzed only from test blocks).
pub fn newFromGoldensForTests(a: std.mem.Allocator) !Tables {
    const g_strt = @embedFile("goldens/dsm_STRT.bin");
    const g_sqt = @embedFile("goldens/dsm_SQT.bin");
    const sqt = try a.alloc(i16, 4095);
    const strt = try a.alloc(i16, 4096);
    const ilog_s = try a.alloc(u8, 256);
    const dt = try a.alloc(i32, 1024);
    for (strt, 0..) |*slot, i| slot.* = std.mem.readInt(i16, g_strt[i * 2 ..][0..2], .little);
    for (sqt, 0..) |*slot, i| slot.* = std.mem.readInt(i16, g_sqt[i * 2 ..][0..2], .little);
    const il = genIlog();
    @memcpy(ilog_s, &il);
    var o: i32 = 2;
    for (dt) |*slot| {
        slot.* = @divTrunc(4096, o);
        o += 1;
    }
    dt[1023] = 1;
    return .{ .sqt = sqt, .strt = strt, .ilog = ilog_s, .dt = dt };
}

pub const Tables = struct {
    sqt: []i16, // 4095 entries
    strt: []i16, // 4096 entries
    ilog: []u8, // 256 entries
    dt: []i32, // 1024 entries

    /// Frees the (root-allocated) slices — verbatim copy of tables.zig's.
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
