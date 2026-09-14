//! Leaf maps — bit-exact port of `StationaryMap` from cmix-lex `fxcmv1.cpp`
//! (VERSION 26), translated 1:1 from leaf_maps.rs.
//!
//!   StationaryMap -> slots 0-3 (maps1/maps2, 2 model slots each): direct U32
//!   prediction with `dt[]` adaptation.
//!
//! (SmallStationaryContextMap and RunContextMap from the same .rs live in their
//! own sibling modules; RunContextMap is in run_context_map.zig.)
//!
//! Validated bit-exact vs v26-port-tools/leaf_oracle.cpp (same synthetic drive).

const std = @import("std");
// In the `cmcold` module: private copies (see cm_cold.zig).
const X = @import("cmcold").X;
const zero_alloc = @import("cmcold").zero_alloc;

/// `clp` — clamp to the stretched-logit range [-2047, 2047] (fxcmv1.cpp::clp).
inline fn clp(z: i32) i16 {
    if (z < -2047) {
        return -2047;
    } else if (z > 2047) {
        return 2047;
    } else {
        return @intCast(z);
    }
}

// ===================== StationaryMap (U32 data) =====================

/// Adaptation-rate table, one shared copy (C++ file-scope `static int dt[1024]`,
/// fxcmv1.cpp:779): dt[i] = 4096/(i+2), dt[1023] = 1.
const DT: [1024]i32 = blk: {
    @setEvalBranchQuota(4096);
    var t: [1024]i32 = undefined;
    for (&t, 0..) |*d, i| d.* = @intCast(4096 / (i + 2));
    t[1023] = 1;
    break :blk t;
};

pub const StationaryMap = struct {
    /// Stored offset by `init_val` (`data[i] = true -% init_val`), so the
    /// fresh map is all-zeros on lazy kernel zero pages instead of an eager
    /// init_val fill; `mix` adds it back at every touch (exact u32 bijection).
    data: []u32,
    /// The C++ seed word `(0x7FF << 20) | min(rate, 1023)`.
    init_val: u32,
    context: usize,
    mask: u32,
    stride: usize,
    b_count: i32,
    b_total: i32,
    b: usize,
    multiplier: i32,
    strt: []const i16, // shared table slice, not an embedded copy
    cp: usize,

    pub fn new() StationaryMap {
        return StationaryMap{
            .data = @constCast(&[_]u32{}),
            .init_val = 0,
            .context = 0,
            .mask = 0,
            .stride = 0,
            .b_count = 0,
            .b_total = 0,
            .b = 0,
            .multiplier = 0,
            .strt = &.{},
            .cp = 0,
        };
    }

    pub fn deinit(self: *StationaryMap, a: std.mem.Allocator) void {
        if (self.data.len != 0) zero_alloc.free(a, self.data);
    }

    inline fn stretch(self: *const StationaryMap, p: i32) i32 {
        return @as(i32, self.strt[@intCast(std.math.clamp(p, 0, 4095))]);
    }

    pub fn init(
        self: *StationaryMap,
        a: std.mem.Allocator,
        bits_of_context: u32,
        input_bits: u32,
        mul: i32,
        rate: i32,
        strt: []const i16,
    ) !void {
        self.multiplier = mul;
        const n = (@as(usize, 1) << @intCast(bits_of_context)) * ((@as(usize, 1) << @intCast(input_bits)) - 1);
        self.context = 0;
        self.mask = (@as(u32, 1) << @intCast(bits_of_context)) - 1;
        self.stride = (@as(usize, 1) << @intCast(input_bits)) - 1;
        self.b_count = 0;
        self.b_total = @intCast(input_bits);
        self.b = 0;
        self.init_val = (@as(u32, 0x7FF) << 20) | @as(u32, @intCast(@min(rate, @as(i32, 1023))));
        // Offset storage (see `data` doc): allocate zeroed, never filled.
        self.data = try zero_alloc.alloc(a, u32, n);
        self.cp = 0;
        self.strt = strt[0..4096];
    }

    pub fn set(self: *StationaryMap, ctx: u32) void {
        self.context = @as(usize, ctx & self.mask) * self.stride;
        self.b_count = 0;
        self.b = 0;
    }

    pub fn mix(self: *StationaryMap, x: *const X) [2]i16 {
        var p0 = self.data[self.cp] +% self.init_val;
        const n: usize = @intCast(p0 & 1023);
        const pr: i32 = @intCast(p0 >> 13);
        p0 = p0 +% @as(u32, @intFromBool(n < 1023));
        const delta: i64 = @as(i64, (x.y << 19) - pr) * @as(i64, DT[n]);
        p0 = p0 +% (@as(u32, @truncate(@as(u64, @bitCast(delta)))) & 0xfffffc00);
        self.data[self.cp] = p0 -% self.init_val;
        self.b += @as(usize, @intFromBool(x.y != 0 and self.b > 0));
        self.cp = self.context + self.b;
        const prediction: i32 = @intCast((self.data[self.cp] +% self.init_val) >> 20);
        const e0 = clp(@divTrunc(self.stretch(prediction) * self.multiplier, 32));
        const e1 = clp(@divTrunc((prediction - 2048) * self.multiplier, 32 * 2));
        self.b_count += 1;
        self.b = self.b * 2 + 1;
        if (self.b_count == self.b_total) {
            self.b_count = 0;
            self.b = 0;
        }
        return .{ e0, e1 };
    }
};

// NOTE: the StationaryMap-vs-cpp-oracle tests live in src/tests.zig (root
// module): this file is in the cmfast module, whose in-file tests are not
// collected by `zig test`, and its test-only tables.zig import would put that
// file in two modules.
