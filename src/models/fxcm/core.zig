//! fxcm_v26 core mixer + state map, ported from reference/fxcmv1_v26.cpp.
//!
//! Mixer1: integer logistic mixer; the reference's AVX2 dot_product/train are
//! expressed with 16-lane @Vector ops (identical semantics). StateMap: the
//! update-limited state->probability map.
const std = @import("std");
const prim = @import("primitives.zig");

const V16i16 = @Vector(16, i16);
const V8i32 = @Vector(8, i32);

fn dotProduct(t: [*]const i16, w: [*]const i16, n_in: usize) i32 {
    var n = n_in;
    var sum: V8i32 = @splat(0);
    while (n >= 16) {
        n -= 16;
        const tv: V16i16 = t[n..][0..16].*;
        const wv: V16i16 = w[n..][0..16].*;
        const tw: @Vector(16, i32) = @as(@Vector(16, i32), tv) * @as(@Vector(16, i32), wv);
        var pair: V8i32 = undefined;
        inline for (0..8) |i| pair[i] = (tw[2 * i] + tw[2 * i + 1]) >> 8;
        sum += pair;
    }
    return @reduce(.Add, sum);
}

fn train(t: [*]const i16, w: [*]i16, n_in: usize, e: i32) void {
    if (e == 0) return;
    var n = n_in;
    const err: V16i16 = @splat(@intCast(e));
    const one: V16i16 = @splat(1);
    while (n >= 16) {
        n -= 16;
        const tv: V16i16 = t[n..][0..16].*;
        var tmp = tv +| tv;
        const prod: @Vector(16, i32) = @as(@Vector(16, i32), tmp) * @as(@Vector(16, i32), err);
        tmp = @intCast(prod >> @as(@Vector(16, i32), @splat(16)));
        tmp +|= one;
        tmp = tmp >> @as(V16i16, @splat(1));
        const wv: V16i16 = w[n..][0..16].*;
        tmp +|= wv;
        w[n..][0..16].* = tmp;
    }
}

pub const Mixer1 = struct {
    n: usize = 0,
    m: usize = 0,
    tx: [*]i16 = undefined,
    wx: []align(32) i16 = undefined,
    cxt: usize = 0,
    pr: i32 = 2048,
    shift1: i32 = 0,
    elim: i32 = 0,
    uperr: i32 = 0,
    err: i32 = 0,
    alloc: std.mem.Allocator = undefined,

    pub fn init(self: *Mixer1, a: std.mem.Allocator, m: usize, s: i32, e: i32, ue: i32) void {
        self.alloc = a;
        self.m = m;
        self.cxt = 0;
        self.shift1 = s;
        self.elim = e;
        self.uperr = ue;
        self.err = 0;
        self.pr = 2048;
    }

    pub fn setTxWx(self: *Mixer1, n: usize, mn: [*]i16) void {
        self.n = n;
        self.wx = self.alloc.alignedAlloc(i16, .@"32", self.n * self.m + 32) catch unreachable;
        @memset(self.wx, 0); // fxcm uses calloc: weights start at 0
        self.tx = mn;
    }

    pub fn update(self: *Mixer1, y: i32) void {
        self.err = @divTrunc(((y << 12) - self.pr) * self.uperr, 4);
        if (self.err > 32767) self.err = 32767;
        if (self.err < -32768) self.err = -32768;
        if (self.err >= -self.elim and self.err <= self.elim) self.err = 0;
        train(self.tx, self.wx.ptr + self.cxt * self.n, self.n, self.err);
    }

    pub fn p(self: *Mixer1) i32 {
        const dp = (dotProduct(self.tx, self.wx.ptr + self.cxt * self.n, self.n) * self.shift1) >> 11;
        self.pr = prim.squash(dp);
        return self.pr;
    }

    pub fn p1(self: *Mixer1) i32 {
        var dp = (dotProduct(self.tx, self.wx.ptr + self.cxt * self.n, self.n) * self.shift1) >> 11;
        if (dp < -2047) dp = -2047 else if (dp > 2047) dp = 2047;
        self.pr = prim.squash(dp);
        return dp;
    }
};

pub const StateMap = struct {
    n: usize = 0,
    cxt: usize = 0,
    t: []u32 = undefined,
    pr: i32 = 2048,
    mask: usize = 0,
    limit: u32 = 0,
    nn: [*]const u8 = undefined,
    alloc: std.mem.Allocator = undefined,

    inline fn next(self: *StateMap, i: usize, y: usize) u32 {
        return self.nn[y + i * 4];
    }

    pub fn init(self: *StateMap, a: std.mem.Allocator, n: usize, lim: u32, nn1: [*]const u8) void {
        self.alloc = a;
        self.nn = nn1;
        self.n = n;
        self.cxt = 0;
        self.pr = 2048;
        self.mask = n - 1;
        self.limit = lim;
        self.t = a.alloc(u32, n) catch unreachable;
        if (n == 256) {
            for (0..n) |i| {
                const n0: u32 = self.next(i, 2) * 3 + 1;
                const n1: u32 = self.next(i, 3) * 3 + 1;
                self.t[i] = (((n1 << 20) / (n0 + n1)) << 12);
            }
        } else {
            for (self.t) |*e| e.* = @as(u32, 1) << 31;
        }
    }

    inline fn update(self: *StateMap, y: i32) void {
        var p0 = self.t[self.cxt];
        const nc: u32 = p0 & 1023;
        const pr1: u32 = p0 >> 12;
        p0 +%= @intFromBool(nc < self.limit);
        const dtv: u32 = @bitCast(prim.dt[@intCast(nc)]);
        p0 +%= ((((@as(u32, @intCast(y)) << 20) -% pr1)) *% dtv +% 512) & 0xfffffc00;
        self.t[self.cxt] = p0;
    }

    pub fn set(self: *StateMap, c: usize, y: i32) void {
        self.update(y);
        self.cxt = c & self.mask;
        self.pr = @intCast(self.t[self.cxt] >> 20);
    }
};
