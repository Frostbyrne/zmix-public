//! paq8 IMAGE and AUDIO models, ported faithfully from
//! reference/cmix-src/models/paq8.cpp (the cmix PAQ8 model).
//!
//! Builds ON src/models/paq8/core.zig (shared predictor state, Mixer, StateMap,
//! APM, tables) and src/models/paq8/maps.zig (hash, ContextMap, StationaryMap,
//! SmallStationaryContextMap, IndirectMap, StateMap32, OLS, IndirectContext).
//!
//! Faithfulness notes:
//!   * C `unsigned` wrap -> `+%`/`-%`/`*%`; U8/U16/U32 truncating stores -> masks.
//!   * Pixel neighbours and the *Ctxs[] arrays are kept as U8 to reproduce the
//!     exact C++ truncation (e.g. ((W+N)*3-NW*2)/4 can exceed 255).
//!   * `long double` in wavModel is modelled with Zig `f80` (x86 80-bit extended,
//!     matching gcc/clang long double).
//!   * Signed pixel/sample arithmetic uses i32; `/` -> @divTrunc (C truncates
//!     toward zero); signed `>>` is arithmetic (matches g++/clang).
const std = @import("std");
const core = @import("core.zig");
const maps = @import("maps.zig");

const Mixer = core.Mixer;
const Allocator = std.mem.Allocator;

// ============================== small helpers ==============================

/// buf(i): the byte `i` positions before global pos (0..255), returned as int.
/// C++ passes an `int` index that is implicitly converted to U32 in operator.
inline fn bget(i: i32) i32 {
    return core.buf.get(@bitCast(i));
}

/// Truncate a signed int to U8 (matches C++ `(U8)x` / assignment to a U8 var).
inline fn u8t(v: i32) u8 {
    return @truncate(@as(u32, @bitCast(v)));
}

/// Truncate a signed int to int8_t.
inline fn toI8(v: i32) i8 {
    return @bitCast(@as(u8, @truncate(@as(u32, @bitCast(v)))));
}

inline fn Clip(px: i32) i32 {
    return @min(255, @max(0, px));
}

inline fn Clamp4(px: i32, n1: i32, n2: i32, n3: i32, n4: i32) i32 {
    const mx = @max(n1, @max(n2, @max(n3, n4)));
    const mn = @min(n1, @min(n2, @min(n3, n4)));
    return @min(mx, @max(mn, px));
}

inline fn LogMeanDiffQt(a: i32, b: i32, limit: i32) i32 {
    if (a == b) return 0;
    const hi: i32 = @as(i32, @intFromBool(a > b)) << 3;
    const denom: i32 = @max(2, @as(i32, @intCast(@abs(a - b))) * 2);
    const arg: u32 = @intCast(@divTrunc(a + b, denom) + 1);
    const l: i32 = @intCast(maps.ilog2(arg));
    return hi | @min(limit, l);
}

inline fn LogQt(px: i32, bits: i32) u32 {
    const l: i32 = @intCast(maps.ilog2(@intCast(px)));
    const sh: i32 = @max(0, l - bits);
    return (@as(u32, @intCast(0x100 | px))) >> @intCast(sh);
}

inline fn BitCount(v0: u32) u32 {
    var v = v0;
    v -%= ((v >> 1) & 0x55555555);
    v = ((v >> 2) & 0x33333333) +% (v & 0x33333333);
    v = ((v >> 4) +% v) & 0x0f0f0f0f;
    v = ((v >> 8) +% v) & 0x00ff00ff;
    v = ((v >> 16) +% v) & 0x0000ffff;
    return v;
}

// U32 buffer readers used by BMP/TGA/WAV header detection and wav samples.
// (named rd4/rd2 to avoid shadowing Zig's i4/i2 integer primitives)
inline fn rd4(i: i32) u32 {
    return @as(u32, @intCast(bget(i))) +%
        (@as(u32, @intCast(bget(i - 1))) << 8) +%
        (@as(u32, @intCast(bget(i - 2))) << 16) +%
        (@as(u32, @intCast(bget(i - 3))) << 24);
}
inline fn rd2(i: i32) i32 {
    return bget(i) + 256 * bget(i - 1);
}
inline fn m4(i: i32) u32 {
    return @as(u32, @intCast(bget(i - 3))) +%
        (@as(u32, @intCast(bget(i - 2))) << 8) +%
        (@as(u32, @intCast(bget(i - 1))) << 16) +%
        (@as(u32, @intCast(bget(i))) << 24);
}
inline fn m2(i: i32) i32 {
    return bget(i) * 256 + bget(i - 1);
}

// signed 16-bit sample readers (wav)
inline fn s2(i: i32) i32 {
    const v: u16 = @intCast(bget(i) + 256 * bget(i - 1));
    return @as(i32, @as(i16, @bitCast(v)));
}
inline fn t2(i: i32) i32 {
    const v: u16 = @intCast(bget(i - 1) + 256 * bget(i));
    return @as(i32, @as(i16, @bitCast(v)));
}

inline fn signedClip8(i: i32) i32 {
    return @max(-128, @min(127, i));
}

inline fn floorToI32(x: f64) i32 {
    const f = @floor(x);
    if (f >= 2147483647.0) return 2147483647;
    if (f <= -2147483648.0) return -2147483648;
    return @intFromFloat(f);
}

// Shared wav/audio globals (paq8 file-scope statics S, D, wmode).
var S: i32 = 0;
var D: i32 = 0;
var wmode: i32 = 0;
pub fn resetShared() void {
    S = 0;
    D = 0;
    wmode = 0;
}

fn X1(i: i32) i32 {
    return switch (wmode) {
        0 => bget(i) - 128,
        1 => bget(i << 1) - 128,
        2 => s2(i << 1),
        3 => s2(i << 2),
        4 => (bget(i) ^ 128) - 128,
        5 => (bget(i << 1) ^ 128) - 128,
        6 => t2(i << 1),
        7 => t2(i << 2),
        else => 0,
    };
}

fn X2(i: i32) i32 {
    return switch (wmode) {
        0 => bget(i + S) - 128,
        1 => bget((i << 1) - 1) - 128,
        2 => s2((i + S) << 1),
        3 => s2((i << 2) - 2),
        4 => (bget(i + S) ^ 128) - 128,
        5 => (bget((i << 1) - 1) ^ 128) - 128,
        6 => t2((i + S) << 1),
        7 => t2((i << 2) - 2),
        else => 0,
    };
}

const OlsU8 = maps.OLS(f64, u8, true);
const OlsI8 = maps.OLS(f64, i8, true);

// ================================ im1bitModel ==============================
// Model for 1-bit image data.
pub const Im1Bit = struct {
    r0: u32 = 0,
    r1: u32 = 0,
    r2: u32 = 0,
    r3: u32 = 0,
    t: core.Array(u8, 0),
    cxt: [11]i32 = .{0} ** 11,
    sm: [11]core.StateMap,
    alloc: Allocator,

    pub fn init(a: Allocator) Im1Bit {
        var self = Im1Bit{
            .t = core.Array(u8, 0).initSize(a, 0x23000),
            .sm = undefined,
            .alloc = a,
        };
        for (0..11) |i| self.sm[i] = core.StateMap.init(a);
        return self;
    }

    pub fn deinit(self: *Im1Bit) void {
        self.t.deinit();
        for (0..11) |i| self.sm[i].deinit();
    }

    pub fn im1bitModel(self: *Im1Bit, m: *Mixer, w: i32) void {
        const N = 11;
        const y: u32 = @intCast(core.y);
        const bpos = core.bpos;

        // update the model
        for (0..N) |i| {
            const idx: u32 = @intCast(self.cxt[i]);
            self.t.at(idx).* = core.nex(self.t.get(idx), y);
        }

        // update the contexts (pixels surrounding the predicted one)
        self.r0 +%= self.r0 +% y;
        self.r1 +%= self.r1 +% (@as(u32, @intCast((bget(w - 1) >> @intCast(7 - bpos)) & 1)));
        self.r2 +%= self.r2 +% (@as(u32, @intCast((bget(w + w - 1) >> @intCast(7 - bpos)) & 1)));
        self.r3 +%= self.r3 +% (@as(u32, @intCast((bget(w + w + w - 1) >> @intCast(7 - bpos)) & 1)));
        const r0 = self.r0;
        const r1 = self.r1;
        const r2 = self.r2;
        const r3 = self.r3;
        self.cxt[0] = @bitCast((r0 & 0x7) | ((r1 >> 4) & 0x38) | ((r2 >> 3) & 0xc0));
        self.cxt[1] = @bitCast(0x100 +% ((r0 & 1) | ((r1 >> 4) & 0x3e) | ((r2 >> 2) & 0x40) | ((r3 >> 1) & 0x80)));
        self.cxt[2] = @bitCast(0x200 +% ((r0 & 1) | ((r1 >> 4) & 0x1d) | ((r2 >> 1) & 0x60) | (r3 & 0xC0)));
        self.cxt[3] = @bitCast(0x300 +% (y | ((r0 << 1) & 4) | ((r1 >> 1) & 0xF0) | ((r2 >> 3) & 0xA)));
        self.cxt[4] = @bitCast(0x400 +% (((r0 >> 4) & 0x2AC) | (r1 & 0xA4) | (r2 & 0x349) | (@as(u32, @intFromBool((r3 & 0x14D) == 0)))));
        self.cxt[5] = @bitCast(0x800 +% (y | ((r1 >> 4) & 0xE) | ((r2 >> 1) & 0x70) | ((r3 << 2) & 0x380)));
        self.cxt[6] = @bitCast(0xC00 +% (((r1 & 0x30) ^ (r3 & 0x0c0c)) | (r0 & 3)));
        self.cxt[7] = @bitCast(0x1000 +% ((@as(u32, @intFromBool((r0 & 0x444) == 0))) | (r1 & 0xC0C) | (r2 & 0xAE3) | (r3 & 0x51C)));
        self.cxt[8] = @bitCast(0x2000 +% ((r0 & 7) | ((r1 >> 1) & 0x3F8) | ((r2 << 5) & 0xC00)));
        self.cxt[9] = @bitCast(0x3000 +% ((r0 & 0x3f) ^ (r1 & 0x3ffe) ^ ((r2 << 2) & 0x7f00) ^ ((r3 << 5) & 0xf800)));
        self.cxt[10] = @bitCast(0x13000 +% ((r0 & 0x3e) ^ (r1 & 0x0c0c) ^ (r2 & 0xc800)));

        // predict
        for (0..N) |i| {
            const idx: u32 = @intCast(self.cxt[i]);
            m.add(core.stretch(self.sm[i].p(self.t.get(idx))));
        }

        m.set(@bitCast((r0 & 7) | ((r1 & 0x3E) >> 2) | ((r2 & 0x1C0) << 2)), 2048);
        m.set(@bitCast(y | ((r1 & 0x1C0) >> 5) | ((r2 & 0x1C0) >> 2) | ((r3 & 0x1C0) << 1)), 1024);
        m.set(@bitCast(((r1 >> 5) & 0xFE) | y), 256);
        m.set(@bitCast((r0 & 0x3) | ((r1 & 0xF80) >> 5)), 128);
    }
};

// ================================ im4bitModel ==============================
// Model for 4-bit image data.
pub const Im4Bit = struct {
    t: maps.HashTable(16),
    cp: [14][*]u8 = undefined,
    cp_inited: bool = false,
    sm: [14]core.StateMap,
    map: core.StateMap32,
    WW: u8 = 0,
    W: u8 = 0,
    NWW: u8 = 0,
    NW: u8 = 0,
    N: u8 = 0,
    NE: u8 = 0,
    NEE: u8 = 0,
    NNWW: u8 = 0,
    NNW: u8 = 0,
    NN: u8 = 0,
    NNE: u8 = 0,
    NNEE: u8 = 0,
    col: i32 = 0,
    line: i32 = 0,
    run: i32 = 0,
    prevColor: i32 = 0,
    px: i32 = 0,
    alloc: Allocator,

    pub fn init(a: Allocator) Im4Bit {
        var self = Im4Bit{
            .t = maps.HashTable(16).init(a, @intCast(core.MEM() / 2)),
            .sm = undefined,
            .map = core.StateMap32.init(a, 16, true),
            .alloc = a,
        };
        for (0..14) |i| self.sm[i] = core.StateMap.init(a);
        return self;
    }

    pub fn deinit(self: *Im4Bit) void {
        self.t.deinit();
        self.map.deinit();
        for (0..14) |i| self.sm[i].deinit();
    }

    pub fn im4bitModel(self: *Im4Bit, m: *Mixer, w: i32) void {
        @setEvalBranchQuota(2000000);
        const Sn = 14;
        const y: u32 = @intCast(core.y);
        const bpos = core.bpos;

        if (!self.cp_inited) {
            for (0..Sn) |i| self.cp[i] = self.t.get(@intCast(263 * i)) + 1;
            self.cp_inited = true;
        }
        for (0..Sn) |i| self.cp[i][0] = core.nex(self.cp[i][0], y);

        if (bpos == 0 or bpos == 4) {
            self.WW = self.W;
            self.NWW = self.NW;
            self.NW = self.N;
            self.N = self.NE;
            self.NE = self.NEE;
            self.NNWW = self.NWW;
            self.NNW = self.NN;
            self.NN = self.NNE;
            self.NNE = self.NNEE;
            if (bpos == 0) {
                self.W = @intCast(core.c4 & 0xF);
                self.NEE = @intCast(bget(w - 1) >> 4);
                self.NNEE = @intCast(bget(w * 2 - 1) >> 4);
            } else {
                self.W = @intCast(core.c0 & 0xF);
                self.NEE = @intCast(bget(w - 1) & 0xF);
                self.NNEE = @intCast(bget(w * 2 - 1) & 0xF);
            }
            const WW: i32 = self.WW;
            const W: i32 = self.W;
            const NWW: i32 = self.NWW;
            const NW: i32 = self.NW;
            const N: i32 = self.N;
            const NE: i32 = self.NE;
            const NEE: i32 = self.NEE;
            const NNWW: i32 = self.NNWW;
            const NNW: i32 = self.NNW;
            const NN: i32 = self.NN;
            const NNE: i32 = self.NNE;
            const NNEE: i32 = self.NNEE;
            if (W != WW or self.col == 0) {
                self.prevColor = WW;
                self.run = 0;
            } else {
                self.run = @min(0xFFF, self.run + 1);
            }
            const run = self.run;
            const prevColor = self.prevColor;
            const col = self.col;
            const line = self.line;
            self.px = 1;
            var i: u64 = 0;
            self.cp[0] = self.t.get(maps.hash(.{ i, W, NW, N }));
            i += 1;
            self.cp[1] = self.t.get(maps.hash(.{ i, N, @min(0xFFF, @divTrunc(col, 8)) }));
            i += 1;
            self.cp[2] = self.t.get(maps.hash(.{ i, W, NW, N, NN, NE }));
            i += 1;
            self.cp[3] = self.t.get(maps.hash(.{ i, W, N, NE + NNE * 16, NEE + NNEE * 16 }));
            i += 1;
            self.cp[4] = self.t.get(maps.hash(.{ i, W, N, NW + NNW * 16, NWW + NNWW * 16 }));
            i += 1;
            self.cp[5] = self.t.get(maps.hash(.{ i, W, @as(i32, @intCast(maps.ilog2(@intCast(run + 1)))), prevColor, @divTrunc(col, @max(1, @divTrunc(w, 2))) }));
            i += 1;
            self.cp[6] = self.t.get(maps.hash(.{ i, NE, @min(0x3FF, @divTrunc(col + line, @max(1, w * 8))) }));
            i += 1;
            self.cp[7] = self.t.get(maps.hash(.{ i, NW, @divTrunc(col - line, @max(1, w * 8)) }));
            i += 1;
            self.cp[8] = self.t.get(maps.hash(.{ i, WW * 16 + W, NN * 16 + N, NNWW * 16 + NW }));
            i += 1;
            self.cp[9] = self.t.get(maps.hash(.{ i, N, NN }));
            i += 1;
            self.cp[10] = self.t.get(maps.hash(.{ i, W, WW }));
            i += 1;
            self.cp[11] = self.t.get(maps.hash(.{ i, W, NE }));
            i += 1;
            self.cp[12] = self.t.get(maps.hash(.{ i, WW, NN, NEE }));
            i += 1;
            self.cp[13] = self.t.get(std.math.maxInt(u64));
            self.col += 1;
            self.col *= @intFromBool(self.col < w * 2);
            self.line += @intFromBool(self.col == 0);
        } else {
            self.px += self.px + @as(i32, @intCast(y));
            const j: usize = @intCast((@as(i32, @intCast(y)) + 1) << @intCast(bpos & 3));
            for (0..Sn) |i| self.cp[i] += j;
        }

        // predict
        for (0..Sn) |i| {
            const s = self.cp[i][0];
            const n0: i32 = -@as(i32, @intFromBool(core.nex(s, 2) == 0));
            const n1: i32 = -@as(i32, @intFromBool(core.nex(s, 3) == 0));
            const p1 = self.sm[i].p(s);
            const st = core.stretch(p1) >> 1;
            m.add(st);
            m.add((p1 - 2047) >> 2);
            m.add(st * @as(i32, @intCast(@abs(n1 - n0))));
        }
        m.add(core.stretch(self.map.p(self.px, 1023)) >> 1);

        const W: i32 = self.W;
        const N: i32 = self.N;
        const NE: i32 = self.NE;
        m.set(W * 16 + self.px, 256);
        m.set(@min(31, @divTrunc(self.col, @max(1, @divTrunc(w, 16)))) + N * 32, 512);
        m.set((bpos & 3) + 4 * W + 64 * @min(7, @as(i32, @intCast(maps.ilog2(@intCast(self.run + 1))))), 512);
        m.set(W + NE * 16 + (bpos & 3) * 256, 1024);
        m.set(self.px, 16);
        m.set(0, 1);
    }
};

// ================================ im8bitModel ==============================
// Model for 8-bit image data (palette / grayscale).
const im8_nOLS = 5;
const im8_nMaps0 = 2;
const im8_nMaps1 = 55;
const im8_nMaps = im8_nMaps0 + im8_nMaps1 + im8_nOLS; // 62
const im8_nPltMaps = 4;
const im8_lambda = [im8_nOLS]f64{ 0.996, 0.87, 0.93, 0.8, 0.9 };
const im8_num = [im8_nOLS]usize{ 32, 12, 15, 10, 14 };

pub const Im8Bit = struct {
    cm: maps.ContextMap,
    Map: [im8_nMaps]maps.StationaryMap,
    pltMap: [im8_nPltMaps]maps.SmallStationaryContextMap,
    iCtx: [im8_nPltMaps]maps.IndirectContext(u8),
    ols: [im8_nOLS]OlsU8,
    // pixel neighbourhood (WWWWWW is declared but never loaded in C++ -> 0)
    WWWWWW: u8 = 0,
    WWWWW: u8 = 0,
    WWWW: u8 = 0,
    WWW: u8 = 0,
    WW: u8 = 0,
    W: u8 = 0,
    NWWWW: u8 = 0,
    NWWW: u8 = 0,
    NWW: u8 = 0,
    NW: u8 = 0,
    N: u8 = 0,
    NE: u8 = 0,
    NEE: u8 = 0,
    NEEE: u8 = 0,
    NEEEE: u8 = 0,
    NNWWW: u8 = 0,
    NNWW: u8 = 0,
    NNW: u8 = 0,
    NN: u8 = 0,
    NNE: u8 = 0,
    NNEE: u8 = 0,
    NNEEE: u8 = 0,
    NNNWW: u8 = 0,
    NNNW: u8 = 0,
    NNN: u8 = 0,
    NNNE: u8 = 0,
    NNNEE: u8 = 0,
    NNNNW: u8 = 0,
    NNNN: u8 = 0,
    NNNNE: u8 = 0,
    NNNNN: u8 = 0,
    NNNNNN: u8 = 0,
    ctx: i32 = 0,
    lastPos: i32 = 0,
    col: i32 = 0,
    x: i32 = 0,
    line: i32 = 0,
    columns: [2]i32 = .{ 1, 1 },
    column: [2]i32 = .{ 0, 0 },
    MapCtxs: [im8_nMaps1]u8 = .{0} ** im8_nMaps1,
    pOLS: [im8_nOLS]u8 = .{0} ** im8_nOLS,
    alloc: Allocator,

    pub fn init(a: Allocator) Im8Bit {
        var self = Im8Bit{
            .cm = maps.ContextMap.init(a, core.MEM() * 4, 48 + im8_nPltMaps),
            .Map = undefined,
            .pltMap = undefined,
            .iCtx = undefined,
            .ols = undefined,
            .alloc = a,
        };
        self.Map[0] = maps.StationaryMap.init(a, 0, 8, 0);
        self.Map[1] = maps.StationaryMap.init(a, 15, 1, 0);
        for (2..im8_nMaps) |i| self.Map[i] = maps.StationaryMap.init(a, 11, 1, 0);
        for (0..im8_nPltMaps) |i| self.pltMap[i] = maps.SmallStationaryContextMap.init(a, 11, 1);
        for (0..im8_nPltMaps) |i| self.iCtx[i] = maps.IndirectContext(u8).init(a, 16, 8);
        for (0..im8_nOLS) |i| self.ols[i] = OlsU8.init(a, im8_num[i], 1, im8_lambda[i], 0.001);
        return self;
    }

    pub fn deinit(self: *Im8Bit) void {
        self.cm.deinit();
        for (0..im8_nMaps) |i| self.Map[i].deinit();
        for (0..im8_nPltMaps) |i| self.pltMap[i].deinit();
        for (0..im8_nPltMaps) |i| self.iCtx[i].deinit();
        for (0..im8_nOLS) |i| self.ols[i].deinit();
    }

    pub fn im8bitModel(self: *Im8Bit, m: *Mixer, w: i32, stats: ?*core.ModelStats, gray: i32) void {
        @setEvalBranchQuota(2000000);
        const bpos = core.bpos;
        if (bpos == 0) {
            if (core.pos != self.lastPos + 1) {
                self.x = 0;
                self.line = 0;
                self.columns[0] = @max(1, @divTrunc(w, @max(1, @as(i32, @intCast(maps.ilog2(@intCast(w)))) * 2)));
                self.columns[1] = @max(1, @divTrunc(self.columns[0], @max(1, @as(i32, @intCast(maps.ilog2(@intCast(self.columns[0])))))));
            } else {
                self.x += 1;
                self.x *= @intFromBool(self.x < w);
                self.line += @intFromBool(self.x == 0);
            }
            self.lastPos = core.pos;
            self.column[0] = @divTrunc(self.x, self.columns[0]);
            self.column[1] = @divTrunc(self.x, self.columns[1]);

            self.WWWWW = u8t(bget(5));
            self.WWWW = u8t(bget(4));
            self.WWW = u8t(bget(3));
            self.WW = u8t(bget(2));
            self.W = u8t(bget(1));
            self.NWWWW = u8t(bget(w + 4));
            self.NWWW = u8t(bget(w + 3));
            self.NWW = u8t(bget(w + 2));
            self.NW = u8t(bget(w + 1));
            self.N = u8t(bget(w));
            self.NE = u8t(bget(w - 1));
            self.NEE = u8t(bget(w - 2));
            self.NEEE = u8t(bget(w - 3));
            self.NEEEE = u8t(bget(w - 4));
            self.NNWWW = u8t(bget(w * 2 + 3));
            self.NNWW = u8t(bget(w * 2 + 2));
            self.NNW = u8t(bget(w * 2 + 1));
            self.NN = u8t(bget(w * 2));
            self.NNE = u8t(bget(w * 2 - 1));
            self.NNEE = u8t(bget(w * 2 - 2));
            self.NNEEE = u8t(bget(w * 2 - 3));
            self.NNNWW = u8t(bget(w * 3 + 2));
            self.NNNW = u8t(bget(w * 3 + 1));
            self.NNN = u8t(bget(w * 3));
            self.NNNE = u8t(bget(w * 3 - 1));
            self.NNNEE = u8t(bget(w * 3 - 2));
            self.NNNNW = u8t(bget(w * 4 + 1));
            self.NNNN = u8t(bget(w * 4));
            self.NNNNE = u8t(bget(w * 4 - 1));
            self.NNNNN = u8t(bget(w * 5));
            self.NNNNNN = u8t(bget(w * 6));

            const W: i32 = self.W;
            const WW: i32 = self.WW;
            const WWW: i32 = self.WWW;
            const N: i32 = self.N;
            const NW: i32 = self.NW;
            const NWW: i32 = self.NWW;
            const NE: i32 = self.NE;
            const NEE: i32 = self.NEE;
            const NN: i32 = self.NN;
            const NNW: i32 = self.NNW;
            const NNWW: i32 = self.NNWW;
            const NNE: i32 = self.NNE;
            const NNEE: i32 = self.NNEE;
            const NNN: i32 = self.NNN;
            const column0 = self.column[0];
            const column1 = self.column[1];

            var j: usize = 0;
            self.MapCtxs[j] = u8t(Clamp4(W + N - NW, W, NW, N, NE));
            j += 1;
            self.MapCtxs[j] = u8t(Clip(W + N - NW));
            j += 1;
            self.MapCtxs[j] = u8t(Clamp4(W + NE - N, W, NW, N, NE));
            j += 1;
            self.MapCtxs[j] = u8t(Clip(W + NE - N));
            j += 1;
            self.MapCtxs[j] = u8t(Clamp4(N + NW - NNW, W, NW, N, NE));
            j += 1;
            self.MapCtxs[j] = u8t(Clip(N + NW - NNW));
            j += 1;
            self.MapCtxs[j] = u8t(Clamp4(N + NE - NNE, W, N, NE, NEE));
            j += 1;
            self.MapCtxs[j] = u8t(Clip(N + NE - NNE));
            j += 1;
            self.MapCtxs[j] = u8t(@divTrunc(W + NEE, 2));
            j += 1;
            self.MapCtxs[j] = u8t(Clip(N * 3 - NN * 3 + NNN));
            j += 1;
            self.MapCtxs[j] = u8t(Clip(W * 3 - WW * 3 + WWW));
            j += 1;
            self.MapCtxs[j] = u8t(@divTrunc(W + Clip(NE * 3 - NNE * 3 + bget(w * 3 - 1)), 2));
            j += 1;
            self.MapCtxs[j] = u8t(@divTrunc(W + Clip(NEE * 3 - bget(w * 2 - 3) * 3 + bget(w * 3 - 4)), 2));
            j += 1;
            self.MapCtxs[j] = u8t(Clip(NN + bget(w * 4) - bget(w * 6)));
            j += 1;
            self.MapCtxs[j] = u8t(Clip(WW + bget(4) - bget(6)));
            j += 1;
            self.MapCtxs[j] = u8t(Clip(@divTrunc(bget(w * 5) - 6 * bget(w * 4) + 15 * NNN - 20 * NN + 15 * N + Clamp4(W * 2 - NWW, W, NW, N, NN), 6)));
            j += 1;
            self.MapCtxs[j] = u8t(Clip(@divTrunc(-3 * WW + 8 * W + Clamp4(NEE * 3 - NNEE * 3 + bget(w * 3 - 2), NE, NEE, bget(w - 3), bget(w - 4)), 6)));
            j += 1;
            self.MapCtxs[j] = u8t(Clip(NN + NW - bget(w * 3 + 1)));
            j += 1;
            self.MapCtxs[j] = u8t(Clip(NN + NE - bget(w * 3 - 1)));
            j += 1;
            self.MapCtxs[j] = u8t(Clip((W * 2 + NW) - (WW + 2 * NWW) + bget(w + 3)));
            j += 1;
            self.MapCtxs[j] = u8t(Clip(@divTrunc(@divTrunc(NW + NWW, 2) * 3 - bget(w * 2 + 3) * 3 + @divTrunc(bget(w * 3 + 4) + bget(w * 3 + 5), 2), 1)));
            j += 1;
            self.MapCtxs[j] = u8t(Clip(NEE + NE - bget(w * 2 - 3)));
            j += 1;
            self.MapCtxs[j] = u8t(Clip(NWW + WW - bget(w + 4)));
            j += 1;
            self.MapCtxs[j] = u8t(Clip(@divTrunc((W + NW) * 3 - NWW * 6 + bget(w + 3) + bget(w * 2 + 3), 2)));
            j += 1;
            self.MapCtxs[j] = u8t(Clip((NE * 2 + NNE) - (NNEE + bget(w * 3 - 2) * 2) + bget(w * 4 - 3)));
            j += 1;
            self.MapCtxs[j] = u8t(bget(w * 6));
            j += 1;
            self.MapCtxs[j] = u8t(@divTrunc(bget(w - 4) + bget(w - 6), 2));
            j += 1;
            self.MapCtxs[j] = u8t(@divTrunc(bget(4) + bget(6), 2));
            j += 1;
            self.MapCtxs[j] = u8t(@divTrunc(W + N + bget(w - 5) + bget(w - 7), 4));
            j += 1;
            self.MapCtxs[j] = u8t(Clip(bget(w - 3) + W - NEE));
            j += 1;
            self.MapCtxs[j] = u8t(Clip(4 * NNN - 3 * bget(w * 4)));
            j += 1;
            self.MapCtxs[j] = u8t(Clip(N + NN - NNN));
            j += 1;
            self.MapCtxs[j] = u8t(Clip(W + WW - WWW));
            j += 1;
            self.MapCtxs[j] = u8t(Clip(W + NEE - NE));
            j += 1;
            self.MapCtxs[j] = u8t(Clip(WW + NEE - N));
            j += 1;
            self.MapCtxs[j] = u8t(@divTrunc(Clip(W * 2 - NW) + Clip(W * 2 - NWW) + N + NE, 4));
            j += 1;
            self.MapCtxs[j] = u8t(Clamp4(N * 2 - NN, W, N, NE, NEE));
            j += 1;
            self.MapCtxs[j] = u8t(@divTrunc(N + NNN, 2));
            j += 1;
            self.MapCtxs[j] = u8t(Clip(NN + W - NNW));
            j += 1;
            self.MapCtxs[j] = u8t(Clip(NWW + N - NNWW));
            j += 1;
            self.MapCtxs[j] = u8t(Clip(@divTrunc(4 * WWW - 15 * WW + 20 * W + Clip(NEE * 2 - NNEE), 10)));
            j += 1;
            self.MapCtxs[j] = u8t(Clip(@divTrunc(bget(w * 3 - 3) - 4 * NNEE + 6 * NE + Clip(W * 3 - NW * 3 + NNW), 4)));
            j += 1;
            self.MapCtxs[j] = u8t(Clip((N * 2 + NE) - (NN + 2 * NNE) + bget(w * 3 - 1)));
            j += 1;
            self.MapCtxs[j] = u8t(Clip((NW * 2 + NNW) - (NNWW + bget(w * 3 + 2) * 2) + bget(w * 4 + 3)));
            j += 1;
            self.MapCtxs[j] = u8t(Clip(NNWW + W - bget(w * 2 + 3)));
            j += 1;
            self.MapCtxs[j] = u8t(Clip(@divTrunc(-bget(w * 4) + 5 * NNN - 10 * NN + 10 * N + Clip(W * 4 - NWW * 6 + bget(w * 2 + 3) * 4 - bget(w * 3 + 4)), 5)));
            j += 1;
            self.MapCtxs[j] = u8t(Clip(NEE + Clip(bget(w - 3) * 2 - bget(w * 2 - 4)) - bget(w - 4)));
            j += 1;
            self.MapCtxs[j] = u8t(Clip(NW + W - NWW));
            j += 1;
            self.MapCtxs[j] = u8t(Clip((N * 2 + NW) - (NN + 2 * NNW) + bget(w * 3 + 1)));
            j += 1;
            self.MapCtxs[j] = u8t(Clip(NN + Clip(NEE * 2 - bget(w * 2 - 3)) - NNE));
            j += 1;
            self.MapCtxs[j] = u8t(Clip(@divTrunc(-bget(4) + 5 * WWW - 10 * WW + 10 * W + Clip(NE * 2 - NNE), 5)));
            j += 1;
            self.MapCtxs[j] = u8t(Clip(@divTrunc(-bget(5) + 4 * bget(4) - 5 * WWW + 5 * W + Clip(NE * 2 - NNE), 4)));
            j += 1;
            self.MapCtxs[j] = u8t(Clip(@divTrunc(WWW - 4 * WW + 6 * W + Clip(NE * 3 - NNE * 3 + bget(w * 3 - 1)), 4)));
            j += 1;
            self.MapCtxs[j] = u8t(Clip(@divTrunc(-NNEE + 3 * NE + Clip(W * 4 - NW * 6 + NNW * 4 - bget(w * 3 + 1)), 3)));
            j += 1;
            self.MapCtxs[j] = u8t(@divTrunc((W + N) * 3 - NW * 2, 4));
            // j is now 54 -> total 55 assigned (0..54)

            // OLS predictors
            const ols_ctx1 = [32]*const u8{ &self.WWWWWW, &self.WWWWW, &self.WWWW, &self.WWW, &self.WW, &self.W, &self.NWWWW, &self.NWWW, &self.NWW, &self.NW, &self.N, &self.NE, &self.NEE, &self.NEEE, &self.NEEEE, &self.NNWWW, &self.NNWW, &self.NNW, &self.NN, &self.NNE, &self.NNEE, &self.NNEEE, &self.NNNWW, &self.NNNW, &self.NNN, &self.NNNE, &self.NNNEE, &self.NNNNW, &self.NNNN, &self.NNNNE, &self.NNNNN, &self.NNNNNN };
            const ols_ctx2 = [12]*const u8{ &self.WWW, &self.WW, &self.W, &self.NWW, &self.NW, &self.N, &self.NE, &self.NEE, &self.NNW, &self.NN, &self.NNE, &self.NNN };
            const ols_ctx3 = [15]*const u8{ &self.N, &self.NE, &self.NEE, &self.NEEE, &self.NEEEE, &self.NN, &self.NNE, &self.NNEE, &self.NNEEE, &self.NNN, &self.NNNE, &self.NNNEE, &self.NNNN, &self.NNNNE, &self.NNNNN };
            const ols_ctx4 = [10]*const u8{ &self.N, &self.NE, &self.NEE, &self.NEEE, &self.NN, &self.NNE, &self.NNEE, &self.NNN, &self.NNNE, &self.NNNN };
            const ols_ctx5 = [14]*const u8{ &self.WWWW, &self.WWW, &self.WW, &self.W, &self.NWWW, &self.NWW, &self.NW, &self.N, &self.NNWW, &self.NNW, &self.NN, &self.NNNW, &self.NNN, &self.NNNN };
            self.ols[0].update(self.W);
            self.pOLS[0] = u8t(Clip(floorToI32(self.ols[0].predictWith(&ols_ctx1))));
            self.ols[1].update(self.W);
            self.pOLS[1] = u8t(Clip(floorToI32(self.ols[1].predictWith(&ols_ctx2))));
            self.ols[2].update(self.W);
            self.pOLS[2] = u8t(Clip(floorToI32(self.ols[2].predictWith(&ols_ctx3))));
            self.ols[3].update(self.W);
            self.pOLS[3] = u8t(Clip(floorToI32(self.ols[3].predictWith(&ols_ctx4))));
            self.ols[4].update(self.W);
            self.pOLS[4] = u8t(Clip(floorToI32(self.ols[4].predictWith(&ols_ctx5))));

            for (0..im8_nPltMaps) |k| self.iCtx[k].add(@intCast(W));
            self.iCtx[0].setCtx(@intCast(W | (NE << 8)));
            self.iCtx[1].setCtx(@intCast(W | (N << 8)));
            self.iCtx[2].setCtx(@intCast(W | (WW << 8)));
            self.iCtx[3].setCtx(@intCast(N | (NN << 8)));

            var i: u64 = 0;
            if (gray == 0) {
                i += 1;
                self.cm.set(maps.hash(.{ i, W }));
                i += 1;
                self.cm.set(maps.hash(.{ i, W, column0 }));
                i += 1;
                self.cm.set(maps.hash(.{ i, N }));
                i += 1;
                self.cm.set(maps.hash(.{ i, N, column0 }));
                i += 1;
                self.cm.set(maps.hash(.{ i, NW }));
                i += 1;
                self.cm.set(maps.hash(.{ i, NW, column0 }));
                i += 1;
                self.cm.set(maps.hash(.{ i, NE }));
                i += 1;
                self.cm.set(maps.hash(.{ i, NE, column0 }));
                i += 1;
                self.cm.set(maps.hash(.{ i, NWW }));
                i += 1;
                self.cm.set(maps.hash(.{ i, NEE }));
                i += 1;
                self.cm.set(maps.hash(.{ i, WW }));
                i += 1;
                self.cm.set(maps.hash(.{ i, NN }));
                i += 1;
                self.cm.set(maps.hash(.{ i, W, N }));
                i += 1;
                self.cm.set(maps.hash(.{ i, W, NW }));
                i += 1;
                self.cm.set(maps.hash(.{ i, W, NE }));
                i += 1;
                self.cm.set(maps.hash(.{ i, W, NEE }));
                i += 1;
                self.cm.set(maps.hash(.{ i, W, NWW }));
                i += 1;
                self.cm.set(maps.hash(.{ i, N, NW }));
                i += 1;
                self.cm.set(maps.hash(.{ i, N, NE }));
                i += 1;
                self.cm.set(maps.hash(.{ i, NW, NE }));
                i += 1;
                self.cm.set(maps.hash(.{ i, W, WW }));
                i += 1;
                self.cm.set(maps.hash(.{ i, N, NN }));
                i += 1;
                self.cm.set(maps.hash(.{ i, NW, NNWW }));
                i += 1;
                self.cm.set(maps.hash(.{ i, NE, NNEE }));
                i += 1;
                self.cm.set(maps.hash(.{ i, NW, NWW }));
                i += 1;
                self.cm.set(maps.hash(.{ i, NW, NNW }));
                i += 1;
                self.cm.set(maps.hash(.{ i, NE, NEE }));
                i += 1;
                self.cm.set(maps.hash(.{ i, NE, NNE }));
                i += 1;
                self.cm.set(maps.hash(.{ i, N, NNW }));
                i += 1;
                self.cm.set(maps.hash(.{ i, N, NNE }));
                i += 1;
                self.cm.set(maps.hash(.{ i, N, NNN }));
                i += 1;
                self.cm.set(maps.hash(.{ i, W, WWW }));
                i += 1;
                self.cm.set(maps.hash(.{ i, WW, NEE }));
                i += 1;
                self.cm.set(maps.hash(.{ i, WW, NN }));
                i += 1;
                self.cm.set(maps.hash(.{ i, W, bget(w - 3) }));
                i += 1;
                self.cm.set(maps.hash(.{ i, W, bget(w - 4) }));
                i += 1;
                self.cm.set(maps.hash(.{ i, W, N, NW }));
                i += 1;
                self.cm.set(maps.hash(.{ i, N, NN, NNN }));
                i += 1;
                self.cm.set(maps.hash(.{ i, W, NE, NEE }));
                i += 1;
                self.cm.set(maps.hash(.{ i, W, NW, N, NE }));
                i += 1;
                self.cm.set(maps.hash(.{ i, N, NE, NN, NNE }));
                i += 1;
                self.cm.set(maps.hash(.{ i, N, NW, NNW, NN }));
                i += 1;
                self.cm.set(maps.hash(.{ i, W, WW, NWW, NW }));
                i += 1;
                self.cm.set(maps.hash(.{ i, W, NW, N, WW, NWW }));
                i += 1;
                self.cm.set(maps.hash(.{ i, column0 }));
                i += 1;
                self.cm.set(maps.hash(.{ i, N, column1 }));
                i += 1;
                self.cm.set(maps.hash(.{ i, W, column1 }));
                i += 1;
                self.cm.set(i);
                for (0..im8_nPltMaps) |k| {
                    i += 1;
                    self.cm.set(maps.hash(.{ i, self.iCtx[k].value() }));
                }
                self.ctx = @min(0x1F, @divTrunc(self.x, @min(0x20, self.columns[0])));
            } else {
                i += 1;
                self.cm.set(maps.hash(.{ i, N }));
                i += 1;
                self.cm.set(maps.hash(.{ i, W }));
                i += 1;
                self.cm.set(maps.hash(.{ i, NW }));
                i += 1;
                self.cm.set(maps.hash(.{ i, NE }));
                i += 1;
                self.cm.set(maps.hash(.{ i, N, NN }));
                i += 1;
                self.cm.set(maps.hash(.{ i, W, WW }));
                i += 1;
                self.cm.set(maps.hash(.{ i, NE, NNEE }));
                i += 1;
                self.cm.set(maps.hash(.{ i, NW, NNWW }));
                i += 1;
                self.cm.set(maps.hash(.{ i, W, NEE }));
                i += 1;
                self.cm.set(maps.hash(.{ i, @divTrunc(Clamp4(W + N - NW, W, NW, N, NE), 2), LogMeanDiffQt(Clip(N + NE - NNE), Clip(N + NW - NNW), 7) }));
                i += 1;
                self.cm.set(maps.hash(.{ i, @divTrunc(W, 4), @divTrunc(NE, 4), column0 }));
                i += 1;
                self.cm.set(maps.hash(.{ i, @divTrunc(Clip(W * 2 - WW), 4), @divTrunc(Clip(N * 2 - NN), 4) }));
                i += 1;
                self.cm.set(maps.hash(.{ i, @divTrunc(Clamp4(N + NE - NNE, W, N, NE, NEE), 4), column0 }));
                i += 1;
                self.cm.set(maps.hash(.{ i, @divTrunc(Clamp4(N + NW - NNW, W, NW, N, NE), 4), column0 }));
                i += 1;
                self.cm.set(maps.hash(.{ i, @divTrunc(W + NEE, 4), column0 }));
                i += 1;
                self.cm.set(maps.hash(.{ i, Clip(W + N - NW), column0 }));
                i += 1;
                self.cm.set(maps.hash(.{ i, Clamp4(N * 3 - NN * 3 + NNN, W, N, NN, NE), LogMeanDiffQt(W, Clip(NW * 2 - NNW), 7) }));
                i += 1;
                self.cm.set(maps.hash(.{ i, Clamp4(W * 3 - WW * 3 + WWW, W, N, NE, NEE), LogMeanDiffQt(N, Clip(NW * 2 - NWW), 7) }));
                i += 1;
                self.cm.set(maps.hash(.{ i, @divTrunc(W + Clamp4(NE * 3 - NNE * 3 + bget(w * 3 - 1), W, N, NE, NEE), 2), LogMeanDiffQt(N, @divTrunc(NW + NE, 2), 7) }));
                i += 1;
                self.cm.set(maps.hash(.{ i, @divTrunc(N + NNN, 8), @divTrunc(Clip(N * 3 - NN * 3 + NNN), 4) }));
                i += 1;
                self.cm.set(maps.hash(.{ i, @divTrunc(W + WWW, 8), @divTrunc(Clip(W * 3 - WW * 3 + WWW), 4) }));
                i += 1;
                self.cm.set(maps.hash(.{ i, Clip(@divTrunc(-bget(4) + 5 * WWW - 10 * WW + 10 * W + Clamp4(NE * 4 - NNE * 6 + bget(w * 3 - 1) * 4 - bget(w * 4 - 1), N, NE, bget(w - 2), bget(w - 3)), 5)) }));
                i += 1;
                self.cm.set(maps.hash(.{ i, Clip(N * 2 - NN), LogMeanDiffQt(N, Clip(NN * 2 - NNN), 7) }));
                i += 1;
                self.cm.set(maps.hash(.{ i, Clip(W * 2 - WW), LogMeanDiffQt(NE, Clip(N * 2 - NW), 7) }));
                i += 1;
                self.cm.set(~@as(u64, 0xde7ec7ed));
                self.ctx = @min(0x1F, @divTrunc(self.x, @max(1, @divTrunc(w, @min(32, self.columns[0]))))) | ((((@as(i32, @intFromBool(@abs(W - N) * 16 > W + N)) << 1) | @as(i32, @intFromBool(@abs(N - NW) > 8))) << 5)) | ((W + N) & 0x180);
            }
            if (stats) |st| {
                st.Image.pixels.W = self.W;
                st.Image.pixels.N = self.N;
                st.Image.pixels.NN = self.NN;
                st.Image.pixels.WW = self.WW;
                // C++ `ctx>>gray`; gray can be >=32 (BMP grayscale detect passes
                // 0x1xx). On x86 the shift count is masked to 5 bits — replicate.
                const gsh: u5 = @truncate(@as(u32, @bitCast(gray)));
                st.Image.ctx = @truncate(@as(u32, @bitCast(self.ctx >> gsh)));
            }
        }

        const B: i32 = (@as(i32, @bitCast(@as(u32, @intCast(core.c0)) << @intCast(8 - bpos)))) & 0xFF;
        {
            const W: i32 = self.W;
            const N: i32 = self.N;
            const NW: i32 = self.NW;
            const NE: i32 = self.NE;
            const NNE: i32 = self.NNE;
            const NNW: i32 = self.NNW;
            var i: usize = 1;
            self.Map[i].setDirect(@bitCast((@as(i32, u8t(Clip(W + N - NW) - B)) * 8 + bpos) | (LogMeanDiffQt(Clip(N + NE - NNE), Clip(N + NW - NNW), 7) << 11)));
            i += 1;
            var jj: usize = 0;
            while (jj < im8_nMaps1) : ({
                i += 1;
                jj += 1;
            }) {
                self.Map[i].setDirect(@bitCast((@as(i32, self.MapCtxs[jj]) - B) * 8 + bpos));
            }
            jj = 0;
            while (i < im8_nMaps) : ({
                i += 1;
                jj += 1;
            }) {
                self.Map[i].setDirect(@bitCast((@as(i32, self.pOLS[jj]) - B) * 8 + bpos));
            }
        }

        _ = self.cm.mix(m);
        if (gray != 0) {
            for (0..im8_nMaps) |i| self.Map[i].mix(m, 1, 4, 1023);
        } else {
            for (0..im8_nPltMaps) |i| {
                self.pltMap[i].set(@intCast((@as(u32, @intCast(bpos)) << 8) | self.iCtx[i].value()));
                self.pltMap[i].mix(m, 7, 1, 4);
            }
        }
        self.col = (self.col + 1) & 7;
        const W: i32 = self.W;
        const N: i32 = self.N;
        const NW: i32 = self.NW;
        const NE: i32 = self.NE;
        const WW: i32 = self.WW;
        const NN: i32 = self.NN;
        const NNWW: i32 = self.NNWW;
        const NNEE: i32 = self.NNEE;
        m.set(self.ctx, 2048);
        m.set(self.col, 8);
        m.set((N + W) >> 4, 32);
        m.set(core.c0, 256);
        m.set(((@as(i32, @intFromBool(@abs(W - N) > 4)) << 9) | (@as(i32, @intFromBool(@abs(N - NE) > 4)) << 8) | (@as(i32, @intFromBool(@abs(W - NW) > 4)) << 7) | (@as(i32, @intFromBool(W > N)) << 6) | (@as(i32, @intFromBool(N > NE)) << 5) | (@as(i32, @intFromBool(W > NW)) << 4) | (@as(i32, @intFromBool(W > WW)) << 3) | (@as(i32, @intFromBool(N > NN)) << 2) | (@as(i32, @intFromBool(NW > NNWW)) << 1) | @as(i32, @intFromBool(NE > NNEE))), 1024);
        m.set(@min(63, self.column[0]), 64);
        m.set(@min(127, self.column[1]), 128);
        m.set(@min(255, @divTrunc(self.x +% self.line, 32)), 256);
    }
};

// =============================== im24bitModel ==============================
// Model for 24/32-bit image data.
const im24_nMaps0 = 18;
const im24_nMaps1 = 76;
const im24_nOLS = 6;
const im24_nMaps = im24_nMaps0 + im24_nMaps1 + im24_nOLS; // 100
const im24_nSCMaps = 59;
const im24_lambda = [im24_nOLS]f64{ 0.98, 0.87, 0.9, 0.8, 0.9, 0.7 };
const im24_num = [im24_nOLS]usize{ 32, 12, 15, 10, 14, 8 };

pub const Im24Bit = struct {
    cm: maps.ContextMap,
    SCMap: [im24_nSCMaps]maps.SmallStationaryContextMap,
    Map: [im24_nMaps]maps.StationaryMap,
    ols: [im24_nOLS][4]OlsU8,
    // pixel neighbourhood
    WWWWWW: u8 = 0,
    WWWWW: u8 = 0,
    WWWW: u8 = 0,
    WWW: u8 = 0,
    WW: u8 = 0,
    W: u8 = 0,
    NWWWW: u8 = 0,
    NWWW: u8 = 0,
    NWW: u8 = 0,
    NW: u8 = 0,
    N: u8 = 0,
    NE: u8 = 0,
    NEE: u8 = 0,
    NEEE: u8 = 0,
    NEEEE: u8 = 0,
    NNWWW: u8 = 0,
    NNWW: u8 = 0,
    NNW: u8 = 0,
    NN: u8 = 0,
    NNE: u8 = 0,
    NNEE: u8 = 0,
    NNEEE: u8 = 0,
    NNNWW: u8 = 0,
    NNNW: u8 = 0,
    NNN: u8 = 0,
    NNNE: u8 = 0,
    NNNEE: u8 = 0,
    NNNNW: u8 = 0,
    NNNN: u8 = 0,
    NNNNE: u8 = 0,
    NNNNN: u8 = 0,
    NNNNNN: u8 = 0,
    WWp1: u8 = 0,
    Wp1: u8 = 0,
    p1: u8 = 0,
    NWp1: u8 = 0,
    Np1: u8 = 0,
    NEp1: u8 = 0,
    NNp1: u8 = 0,
    WWp2: u8 = 0,
    Wp2: u8 = 0,
    p2: u8 = 0,
    NWp2: u8 = 0,
    Np2: u8 = 0,
    NEp2: u8 = 0,
    NNp2: u8 = 0,
    color: i32 = -1,
    stride: i32 = 3,
    ctxa: [2]i32 = .{ 0, 0 },
    padding: i32 = 0,
    lastPos: i32 = 0,
    x: i32 = 0,
    line: i32 = 0,
    columns: [2]i32 = .{ 1, 1 },
    column: [2]i32 = .{ 0, 0 },
    col24: i32 = 0,
    MapCtxs: [im24_nMaps1]u8 = .{0} ** im24_nMaps1,
    SCMapCtxs: [im24_nSCMaps - 1]u8 = .{0} ** (im24_nSCMaps - 1),
    pOLS: [im24_nOLS]u8 = .{0} ** im24_nOLS,
    alloc: Allocator,

    pub fn init(a: Allocator) Im24Bit {
        var self = Im24Bit{
            .cm = maps.ContextMap.init(a, core.MEM() * 4, 47),
            .SCMap = undefined,
            .Map = undefined,
            .ols = undefined,
            .alloc = a,
        };
        for (0..im24_nSCMaps - 1) |i| self.SCMap[i] = maps.SmallStationaryContextMap.init(a, 11, 1);
        self.SCMap[im24_nSCMaps - 1] = maps.SmallStationaryContextMap.init(a, 0, 8);
        // StationaryMap layout: {8,8},{8,8},{8,8},{2,8},{0,8},{15,1}x5,{17,1}x4,{13,1}x4,{11,1}...
        const boc = [18]u32{ 8, 8, 8, 2, 0, 15, 15, 15, 15, 15, 17, 17, 17, 17, 13, 13, 13, 13 } ++ ([_]u32{11} ** (im24_nMaps - 18));
        const bpc = [18]u32{ 8, 8, 8, 8, 8, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1 } ++ ([_]u32{1} ** (im24_nMaps - 18));
        for (0..im24_nMaps) |i| self.Map[i] = maps.StationaryMap.init(a, boc[i], bpc[i], 0);
        for (0..im24_nOLS) |i| {
            for (0..4) |k| self.ols[i][k] = OlsU8.init(a, im24_num[i], 1, im24_lambda[i], 0.001);
        }
        return self;
    }

    pub fn deinit(self: *Im24Bit) void {
        self.cm.deinit();
        for (0..im24_nSCMaps) |i| self.SCMap[i].deinit();
        for (0..im24_nMaps) |i| self.Map[i].deinit();
        for (0..im24_nOLS) |i| {
            for (0..4) |k| self.ols[i][k].deinit();
        }
    }

    pub fn im24bitModel(self: *Im24Bit, m: *Mixer, w: i32, stats: ?*core.ModelStats, alpha: i32) void {
        @setEvalBranchQuota(4000000);
        const bpos = core.bpos;
        if (bpos == 0) {
            if ((self.color < 0) or (core.pos - self.lastPos != 1)) {
                self.stride = 3 + alpha;
                self.padding = @mod(w, self.stride);
                self.x = 0;
                self.line = 0;
                self.columns[0] = @max(1, @divTrunc(w, @max(1, @as(i32, @intCast(maps.ilog2(@intCast(w)))) * 3)));
                self.columns[1] = @max(1, @divTrunc(self.columns[0], @max(1, @as(i32, @intCast(maps.ilog2(@intCast(self.columns[0])))))));
            }
            self.lastPos = core.pos;
            self.x += 1;
            self.x *= @intFromBool(self.x < w);
            self.line += @intFromBool(self.x == 0);
            if (self.x + self.padding < w) {
                self.color += 1;
                self.color *= @intFromBool(self.color < self.stride);
            } else {
                self.color = @as(i32, @intFromBool(self.padding > 0)) * (self.stride + 1);
            }
            self.column[0] = @divTrunc(self.x, self.columns[0]);
            self.column[1] = @divTrunc(self.x, self.columns[1]);
            const stride = self.stride;

            self.WWWWWW = u8t(bget(6 * stride));
            self.WWWWW = u8t(bget(5 * stride));
            self.WWWW = u8t(bget(4 * stride));
            self.WWW = u8t(bget(3 * stride));
            self.WW = u8t(bget(2 * stride));
            self.W = u8t(bget(stride));
            self.NWWWW = u8t(bget(w + 4 * stride));
            self.NWWW = u8t(bget(w + 3 * stride));
            self.NWW = u8t(bget(w + 2 * stride));
            self.NW = u8t(bget(w + stride));
            self.N = u8t(bget(w));
            self.NE = u8t(bget(w - stride));
            self.NEE = u8t(bget(w - 2 * stride));
            self.NEEE = u8t(bget(w - 3 * stride));
            self.NEEEE = u8t(bget(w - 4 * stride));
            self.NNWWW = u8t(bget(w * 2 + stride * 3));
            self.NNWW = u8t(bget((w + stride) * 2));
            self.NNW = u8t(bget(w * 2 + stride));
            self.NN = u8t(bget(w * 2));
            self.NNE = u8t(bget(w * 2 - stride));
            self.NNEE = u8t(bget((w - stride) * 2));
            self.NNEEE = u8t(bget(w * 2 - stride * 3));
            self.NNNWW = u8t(bget(w * 3 + stride * 2));
            self.NNNW = u8t(bget(w * 3 + stride));
            self.NNN = u8t(bget(w * 3));
            self.NNNE = u8t(bget(w * 3 - stride));
            self.NNNEE = u8t(bget(w * 3 - stride * 2));
            self.NNNNW = u8t(bget(w * 4 + stride));
            self.NNNN = u8t(bget(w * 4));
            self.NNNNE = u8t(bget(w * 4 - stride));
            self.NNNNN = u8t(bget(w * 5));
            self.NNNNNN = u8t(bget(w * 6));
            self.WWp1 = u8t(bget(stride * 2 + 1));
            self.Wp1 = u8t(bget(stride + 1));
            self.p1 = u8t(bget(1));
            self.NWp1 = u8t(bget(w + stride + 1));
            self.Np1 = u8t(bget(w + 1));
            self.NEp1 = u8t(bget(w - stride + 1));
            self.NNp1 = u8t(bget(w * 2 + 1));
            self.WWp2 = u8t(bget(stride * 2 + 2));
            self.Wp2 = u8t(bget(stride + 2));
            self.p2 = u8t(bget(2));
            self.NWp2 = u8t(bget(w + stride + 2));
            self.Np2 = u8t(bget(w + 2));
            self.NEp2 = u8t(bget(w - stride + 2));
            self.NNp2 = u8t(bget(w * 2 + 2));

            const WWWWWW: i32 = self.WWWWWW;
            const WWWW: i32 = self.WWWW;
            const WWW: i32 = self.WWW;
            const WW: i32 = self.WW;
            const W: i32 = self.W;
            const NWWW: i32 = self.NWWW;
            const NWW: i32 = self.NWW;
            const NW: i32 = self.NW;
            const N: i32 = self.N;
            const NE: i32 = self.NE;
            const NEE: i32 = self.NEE;
            const NEEE: i32 = self.NEEE;
            const NEEEE: i32 = self.NEEEE;
            const NNWWW: i32 = self.NNWWW;
            const NNWW: i32 = self.NNWW;
            const NNW: i32 = self.NNW;
            const NN: i32 = self.NN;
            const NNE: i32 = self.NNE;
            const NNEE: i32 = self.NNEE;
            const NNEEE: i32 = self.NNEEE;
            const NNNWW: i32 = self.NNNWW;
            const NNNW: i32 = self.NNNW;
            const NNN: i32 = self.NNN;
            const NNNE: i32 = self.NNNE;
            const NNNEE: i32 = self.NNNEE;
            const NNNN: i32 = self.NNNN;
            const NNNNE: i32 = self.NNNNE;
            const NNNNN: i32 = self.NNNNN;
            const NNNNNN: i32 = self.NNNNNN;
            const WWp1: i32 = self.WWp1;
            const Wp1: i32 = self.Wp1;
            const p1: i32 = self.p1;
            const NWp1: i32 = self.NWp1;
            const Np1: i32 = self.Np1;
            const NEp1: i32 = self.NEp1;
            const NNp1: i32 = self.NNp1;
            const WWp2: i32 = self.WWp2;
            const Wp2: i32 = self.Wp2;
            const p2: i32 = self.p2;
            const NWp2: i32 = self.NWp2;
            const Np2: i32 = self.Np2;
            const NEp2: i32 = self.NEp2;
            const NNp2: i32 = self.NNp2;
            const color = self.color;
            const column0 = self.column[0];
            const column1 = self.column[1];

            var j: usize = 0;
            self.MapCtxs[j] = u8t(Clamp4(N + p1 - Np1, W, NW, N, NE));
            j += 1;
            self.MapCtxs[j] = u8t(Clamp4(N + p2 - Np2, W, NW, N, NE));
            j += 1;
            self.MapCtxs[j] = u8t(@divTrunc(W + Clamp4(NE * 3 - NNE * 3 + NNNE, W, N, NE, NEE), 2));
            j += 1;
            self.MapCtxs[j] = u8t(Clamp4(@divTrunc(W + Clip(NE * 2 - NNE), 2), W, NW, N, NE));
            j += 1;
            self.MapCtxs[j] = u8t(@divTrunc(W + NEE, 2));
            j += 1;
            self.MapCtxs[j] = u8t(Clip(@divTrunc(WWW - 4 * WW + 6 * W + Clip(NE * 4 - NNE * 6 + NNNE * 4 - NNNNE), 4)));
            j += 1;
            self.MapCtxs[j] = u8t(Clip(@divTrunc(-WWWW + 5 * WWW - 10 * WW + 10 * W + Clamp4(NE * 4 - NNE * 6 + NNNE * 4 - NNNNE, N, NE, NEE, NEEE), 5)));
            j += 1;
            self.MapCtxs[j] = u8t(Clip(@divTrunc(-4 * WW + 15 * W + 10 * Clip(NE * 3 - NNE * 3 + NNNE) - Clip(NEEE * 3 - NNEEE * 3 + bget(w * 3 - 3 * stride)), 20)));
            j += 1;
            self.MapCtxs[j] = u8t(Clip(@divTrunc(-3 * WW + 8 * W + Clamp4(NEE * 3 - NNEE * 3 + NNNEE, NE, NEE, NEEE, NEEEE), 6)));
            j += 1;
            self.MapCtxs[j] = u8t(Clip(@divTrunc(W + Clip(NE * 2 - NNE), 2) + p1 - @divTrunc(Wp1 + Clip(NEp1 * 2 - bget(w * 2 - stride + 1)), 2)));
            j += 1;
            self.MapCtxs[j] = u8t(Clip(@divTrunc(W + Clip(NE * 2 - NNE), 2) + p2 - @divTrunc(Wp2 + Clip(NEp2 * 2 - bget(w * 2 - stride + 2)), 2)));
            j += 1;
            self.MapCtxs[j] = u8t(Clip(@divTrunc(-3 * WW + 8 * W + Clip(NEE * 2 - NNEE), 6) + p1 - @divTrunc(-3 * WWp1 + 8 * Wp1 + Clip(bget(w - stride * 2 + 1) * 2 - bget(w * 2 - stride * 2 + 1)), 6)));
            j += 1;
            self.MapCtxs[j] = u8t(Clip(@divTrunc(-3 * WW + 8 * W + Clip(NEE * 2 - NNEE), 6) + p2 - @divTrunc(-3 * WWp2 + 8 * Wp2 + Clip(bget(w - stride * 2 + 2) * 2 - bget(w * 2 - stride * 2 + 2)), 6)));
            j += 1;
            self.MapCtxs[j] = u8t(Clip(@divTrunc(W + NEE, 2) + p1 - @divTrunc(Wp1 + bget(w - stride * 2 + 1), 2)));
            j += 1;
            self.MapCtxs[j] = u8t(Clip(@divTrunc(W + NEE, 2) + p2 - @divTrunc(Wp2 + bget(w - stride * 2 + 2), 2)));
            j += 1;
            self.MapCtxs[j] = u8t(Clip(@divTrunc(WW + Clip(NEE * 2 - NNEE), 2) + p1 - @divTrunc(WWp1 + Clip(bget(w - stride * 2 + 1) * 2 - bget(w * 2 - stride * 2 + 1)), 2)));
            j += 1;
            self.MapCtxs[j] = u8t(Clip(@divTrunc(WW + Clip(NEE * 2 - NNEE), 2) + p2 - @divTrunc(WWp2 + Clip(bget(w - stride * 2 + 2) * 2 - bget(w * 2 - stride * 2 + 2)), 2)));
            j += 1;
            self.MapCtxs[j] = u8t(Clip(WW + NEE - N + p1 - Clip(WWp1 + bget(w - stride * 2 + 1) - Np1)));
            j += 1;
            self.MapCtxs[j] = u8t(Clip(WW + NEE - N + p2 - Clip(WWp2 + bget(w - stride * 2 + 2) - Np2)));
            j += 1;
            self.MapCtxs[j] = u8t(Clip(W + N - NW));
            j += 1;
            self.MapCtxs[j] = u8t(Clip(W + N - NW + p1 - Clip(Wp1 + Np1 - NWp1)));
            j += 1;
            self.MapCtxs[j] = u8t(Clip(W + N - NW + p2 - Clip(Wp2 + Np2 - NWp2)));
            j += 1;
            self.MapCtxs[j] = u8t(Clip(W + NE - N));
            j += 1;
            self.MapCtxs[j] = u8t(Clip(N + NW - NNW));
            j += 1;
            self.MapCtxs[j] = u8t(Clip(N + NW - NNW + p1 - Clip(Np1 + NWp1 - bget(w * 2 + stride + 1))));
            j += 1;
            self.MapCtxs[j] = u8t(Clip(N + NW - NNW + p2 - Clip(Np2 + NWp2 - bget(w * 2 + stride + 2))));
            j += 1;
            self.MapCtxs[j] = u8t(Clip(N + NE - NNE));
            j += 1;
            self.MapCtxs[j] = u8t(Clip(N + NE - NNE + p1 - Clip(Np1 + NEp1 - bget(w * 2 - stride + 1))));
            j += 1;
            self.MapCtxs[j] = u8t(Clip(N + NE - NNE + p2 - Clip(Np2 + NEp2 - bget(w * 2 - stride + 2))));
            j += 1;
            self.MapCtxs[j] = u8t(Clip(N + NN - NNN));
            j += 1;
            self.MapCtxs[j] = u8t(Clip(N + NN - NNN + p1 - Clip(Np1 + NNp1 - bget(w * 3 + 1))));
            j += 1;
            self.MapCtxs[j] = u8t(Clip(N + NN - NNN + p2 - Clip(Np2 + NNp2 - bget(w * 3 + 2))));
            j += 1;
            self.MapCtxs[j] = u8t(Clip(W + WW - WWW));
            j += 1;
            self.MapCtxs[j] = u8t(Clip(W + WW - WWW + p1 - Clip(Wp1 + WWp1 - bget(stride * 3 + 1))));
            j += 1;
            self.MapCtxs[j] = u8t(Clip(W + WW - WWW + p2 - Clip(Wp2 + WWp2 - bget(stride * 3 + 2))));
            j += 1;
            self.MapCtxs[j] = u8t(Clip(W + NEE - NE));
            j += 1;
            self.MapCtxs[j] = u8t(Clip(W + NEE - NE + p1 - Clip(Wp1 + bget(w - stride * 2 + 1) - NEp1)));
            j += 1;
            self.MapCtxs[j] = u8t(Clip(W + NEE - NE + p2 - Clip(Wp2 + bget(w - stride * 2 + 2) - NEp2)));
            j += 1;
            self.MapCtxs[j] = u8t(Clip(NN + p1 - NNp1));
            j += 1;
            self.MapCtxs[j] = u8t(Clip(NN + p2 - NNp2));
            j += 1;
            self.MapCtxs[j] = u8t(Clip(NN + W - NNW));
            j += 1;
            self.MapCtxs[j] = u8t(Clip(NN + W - NNW + p1 - Clip(NNp1 + Wp1 - bget(w * 2 + stride + 1))));
            j += 1;
            self.MapCtxs[j] = u8t(Clip(NN + W - NNW + p2 - Clip(NNp2 + Wp2 - bget(w * 2 + stride + 2))));
            j += 1;
            self.MapCtxs[j] = u8t(Clip(NN + NW - NNNW));
            j += 1;
            self.MapCtxs[j] = u8t(Clip(NN + NW - NNNW + p1 - Clip(NNp1 + NWp1 - bget(w * 3 + stride + 1))));
            j += 1;
            self.MapCtxs[j] = u8t(Clip(NN + NW - NNNW + p2 - Clip(NNp2 + NWp2 - bget(w * 3 + stride + 2))));
            j += 1;
            self.MapCtxs[j] = u8t(Clip(NN + NE - NNNE));
            j += 1;
            self.MapCtxs[j] = u8t(Clip(NN + NE - NNNE + p1 - Clip(NNp1 + NEp1 - bget(w * 3 - stride + 1))));
            j += 1;
            self.MapCtxs[j] = u8t(Clip(NN + NE - NNNE + p2 - Clip(NNp2 + NEp2 - bget(w * 3 - stride + 2))));
            j += 1;
            self.MapCtxs[j] = u8t(Clip(NN + NNNN - NNNNNN));
            j += 1;
            self.MapCtxs[j] = u8t(Clip(NN + NNNN - NNNNNN + p1 - Clip(NNp1 + bget(w * 4 + 1) - bget(w * 6 + 1))));
            j += 1;
            self.MapCtxs[j] = u8t(Clip(NN + NNNN - NNNNNN + p2 - Clip(NNp2 + bget(w * 4 + 2) - bget(w * 6 + 2))));
            j += 1;
            self.MapCtxs[j] = u8t(Clip(WW + p1 - WWp1));
            j += 1;
            self.MapCtxs[j] = u8t(Clip(WW + p2 - WWp2));
            j += 1;
            self.MapCtxs[j] = u8t(Clip(WW + WWWW - WWWWWW));
            j += 1;
            self.MapCtxs[j] = u8t(Clip(WW + WWWW - WWWWWW + p1 - Clip(WWp1 + bget(stride * 4 + 1) - bget(stride * 6 + 1))));
            j += 1;
            self.MapCtxs[j] = u8t(Clip(WW + WWWW - WWWWWW + p2 - Clip(WWp2 + bget(stride * 4 + 2) - bget(stride * 6 + 2))));
            j += 1;
            self.MapCtxs[j] = u8t(Clip(N * 2 - NN + p1 - Clip(Np1 * 2 - NNp1)));
            j += 1;
            self.MapCtxs[j] = u8t(Clip(N * 2 - NN + p2 - Clip(Np2 * 2 - NNp2)));
            j += 1;
            self.MapCtxs[j] = u8t(Clip(W * 2 - WW + p1 - Clip(Wp1 * 2 - WWp1)));
            j += 1;
            self.MapCtxs[j] = u8t(Clip(W * 2 - WW + p2 - Clip(Wp2 * 2 - WWp2)));
            j += 1;
            self.MapCtxs[j] = u8t(Clip(N * 3 - NN * 3 + NNN));
            j += 1;
            self.MapCtxs[j] = u8t(Clamp4(N * 3 - NN * 3 + NNN, W, NW, N, NE));
            j += 1;
            self.MapCtxs[j] = u8t(Clamp4(W * 3 - WW * 3 + WWW, W, NW, N, NE));
            j += 1;
            self.MapCtxs[j] = u8t(Clamp4(N * 2 - NN, W, NW, N, NE));
            j += 1;
            self.MapCtxs[j] = u8t(Clip(@divTrunc(NNNNN - 6 * NNNN + 15 * NNN - 20 * NN + 15 * N + Clamp4(W * 4 - NWW * 6 + NNWWW * 4 - bget(w * 3 + 4 * stride), W, NW, N, NN), 6)));
            j += 1;
            self.MapCtxs[j] = u8t(Clip(@divTrunc(bget(w * 3 - 3 * stride) - 4 * NNEE + 6 * NE + Clip(W * 4 - NW * 6 + NNW * 4 - NNNW), 4)));
            j += 1;
            self.MapCtxs[j] = u8t(Clip(@divTrunc(N + 3 * NW, 4) * 3 - @divTrunc(NNW + NNWW, 2) * 3 + @divTrunc(NNNWW * 3 + bget(w * 3 + 3 * stride), 4)));
            j += 1;
            self.MapCtxs[j] = u8t(Clip((W * 2 + NW) - (WW + 2 * NWW) + NWWW));
            j += 1;
            self.MapCtxs[j] = u8t(@divTrunc(Clip(W * 2 - NW) + Clip(W * 2 - NWW) + N + NE, 4));
            j += 1;
            self.MapCtxs[j] = u8t(NNNNNN);
            j += 1;
            self.MapCtxs[j] = u8t(@divTrunc(NEEEE + bget(w - 6 * stride), 2));
            j += 1;
            self.MapCtxs[j] = u8t(@divTrunc(WWWWWW + WWWW, 2));
            j += 1;
            self.MapCtxs[j] = u8t(@divTrunc((W + N) * 3 - NW * 2, 4));
            j += 1;
            self.MapCtxs[j] = u8t(N);
            j += 1;
            self.MapCtxs[j] = u8t(NN);
            // j == 75 -> 76 assigned (0..75)

            var jj: usize = 0;
            self.SCMapCtxs[jj] = u8t(N + p1 - Np1);
            jj += 1;
            self.SCMapCtxs[jj] = u8t(N + p2 - Np2);
            jj += 1;
            self.SCMapCtxs[jj] = u8t(W + p1 - Wp1);
            jj += 1;
            self.SCMapCtxs[jj] = u8t(W + p2 - Wp2);
            jj += 1;
            self.SCMapCtxs[jj] = u8t(NW + p1 - NWp1);
            jj += 1;
            self.SCMapCtxs[jj] = u8t(NW + p2 - NWp2);
            jj += 1;
            self.SCMapCtxs[jj] = u8t(NE + p1 - NEp1);
            jj += 1;
            self.SCMapCtxs[jj] = u8t(NE + p2 - NEp2);
            jj += 1;
            self.SCMapCtxs[jj] = u8t(NN + p1 - NNp1);
            jj += 1;
            self.SCMapCtxs[jj] = u8t(NN + p2 - NNp2);
            jj += 1;
            self.SCMapCtxs[jj] = u8t(WW + p1 - WWp1);
            jj += 1;
            self.SCMapCtxs[jj] = u8t(WW + p2 - WWp2);
            jj += 1;
            self.SCMapCtxs[jj] = u8t(W + N - NW);
            jj += 1;
            self.SCMapCtxs[jj] = u8t(W + N - NW + p1 - Wp1 - Np1 + NWp1);
            jj += 1;
            self.SCMapCtxs[jj] = u8t(W + N - NW + p2 - Wp2 - Np2 + NWp2);
            jj += 1;
            self.SCMapCtxs[jj] = u8t(W + NE - N);
            jj += 1;
            self.SCMapCtxs[jj] = u8t(W + NE - N + p1 - Wp1 - NEp1 + Np1);
            jj += 1;
            self.SCMapCtxs[jj] = u8t(W + NE - N + p2 - Wp2 - NEp2 + Np2);
            jj += 1;
            self.SCMapCtxs[jj] = u8t(W + NEE - NE);
            jj += 1;
            self.SCMapCtxs[jj] = u8t(W + NEE - NE + p1 - Wp1 - bget(w - stride * 2 + 1) + NEp1);
            jj += 1;
            self.SCMapCtxs[jj] = u8t(W + NEE - NE + p2 - Wp2 - bget(w - stride * 2 + 2) + NEp2);
            jj += 1;
            self.SCMapCtxs[jj] = u8t(N + NN - NNN);
            jj += 1;
            self.SCMapCtxs[jj] = u8t(N + NN - NNN + p1 - Np1 - NNp1 + bget(w * 3 + 1));
            jj += 1;
            self.SCMapCtxs[jj] = u8t(N + NN - NNN + p2 - Np2 - NNp2 + bget(w * 3 + 2));
            jj += 1;
            self.SCMapCtxs[jj] = u8t(N + NE - NNE);
            jj += 1;
            self.SCMapCtxs[jj] = u8t(N + NE - NNE + p1 - Np1 - NEp1 + bget(w * 2 - stride + 1));
            jj += 1;
            self.SCMapCtxs[jj] = u8t(N + NE - NNE + p2 - Np2 - NEp2 + bget(w * 2 - stride + 2));
            jj += 1;
            self.SCMapCtxs[jj] = u8t(N + NW - NNW);
            jj += 1;
            self.SCMapCtxs[jj] = u8t(N + NW - NNW + p1 - Np1 - NWp1 + bget(w * 2 + stride + 1));
            jj += 1;
            self.SCMapCtxs[jj] = u8t(N + NW - NNW + p2 - Np2 - NWp2 + bget(w * 2 + stride + 2));
            jj += 1;
            self.SCMapCtxs[jj] = u8t(NE + NW - NN);
            jj += 1;
            self.SCMapCtxs[jj] = u8t(NE + NW - NN + p1 - NEp1 - NWp1 + NNp1);
            jj += 1;
            self.SCMapCtxs[jj] = u8t(NE + NW - NN + p2 - NEp2 - NWp2 + NNp2);
            jj += 1;
            self.SCMapCtxs[jj] = u8t(NW + W - NWW);
            jj += 1;
            self.SCMapCtxs[jj] = u8t(NW + W - NWW + p1 - NWp1 - Wp1 + bget(w + stride * 2 + 1));
            jj += 1;
            self.SCMapCtxs[jj] = u8t(NW + W - NWW + p2 - NWp2 - Wp2 + bget(w + stride * 2 + 2));
            jj += 1;
            self.SCMapCtxs[jj] = u8t(W * 2 - WW);
            jj += 1;
            self.SCMapCtxs[jj] = u8t(W * 2 - WW + p1 - Wp1 * 2 + WWp1);
            jj += 1;
            self.SCMapCtxs[jj] = u8t(W * 2 - WW + p2 - Wp2 * 2 + WWp2);
            jj += 1;
            self.SCMapCtxs[jj] = u8t(N * 2 - NN);
            jj += 1;
            self.SCMapCtxs[jj] = u8t(N * 2 - NN + p1 - Np1 * 2 + NNp1);
            jj += 1;
            self.SCMapCtxs[jj] = u8t(N * 2 - NN + p2 - Np2 * 2 + NNp2);
            jj += 1;
            self.SCMapCtxs[jj] = u8t(NW * 2 - NNWW);
            jj += 1;
            self.SCMapCtxs[jj] = u8t(NW * 2 - NNWW + p1 - NWp1 * 2 + bget(w * 2 + stride * 2 + 1));
            jj += 1;
            self.SCMapCtxs[jj] = u8t(NW * 2 - NNWW + p2 - NWp2 * 2 + bget(w * 2 + stride * 2 + 2));
            jj += 1;
            self.SCMapCtxs[jj] = u8t(NE * 2 - NNEE);
            jj += 1;
            self.SCMapCtxs[jj] = u8t(NE * 2 - NNEE + p1 - NEp1 * 2 + bget(w * 2 - stride * 2 + 1));
            jj += 1;
            self.SCMapCtxs[jj] = u8t(NE * 2 - NNEE + p2 - NEp2 * 2 + bget(w * 2 - stride * 2 + 2));
            jj += 1;
            self.SCMapCtxs[jj] = u8t(N * 3 - NN * 3 + NNN + p1 - Np1 * 3 + NNp1 * 3 - bget(w * 3 + 1));
            jj += 1;
            self.SCMapCtxs[jj] = u8t(N * 3 - NN * 3 + NNN + p2 - Np2 * 3 + NNp2 * 3 - bget(w * 3 + 2));
            jj += 1;
            self.SCMapCtxs[jj] = u8t(N * 3 - NN * 3 + NNN);
            jj += 1;
            self.SCMapCtxs[jj] = u8t(@divTrunc(W + NE * 2 - NNE, 2));
            jj += 1;
            self.SCMapCtxs[jj] = u8t(@divTrunc(W + NE * 3 - NNE * 3 + NNNE, 2));
            jj += 1;
            self.SCMapCtxs[jj] = u8t(@divTrunc(W + NE * 2 - NNE, 2) + p1 - @divTrunc(Wp1 + NEp1 * 2 - bget(w * 2 - stride + 1), 2));
            jj += 1;
            self.SCMapCtxs[jj] = u8t(@divTrunc(W + NE * 2 - NNE, 2) + p2 - @divTrunc(Wp2 + NEp2 * 2 - bget(w * 2 - stride + 2), 2));
            jj += 1;
            self.SCMapCtxs[jj] = u8t(NNE + NE - NNNE);
            jj += 1;
            self.SCMapCtxs[jj] = u8t(NNE + W - NN);
            jj += 1;
            self.SCMapCtxs[jj] = u8t(NNW + W - NNWW);
            // jj == 57 -> 58 assigned (0..57)

            // OLS predictors (indexed by color plane)
            const ols_ctx1 = [32]*const u8{ &self.WWWWWW, &self.WWWWW, &self.WWWW, &self.WWW, &self.WW, &self.W, &self.NWWWW, &self.NWWW, &self.NWW, &self.NW, &self.N, &self.NE, &self.NEE, &self.NEEE, &self.NEEEE, &self.NNWWW, &self.NNWW, &self.NNW, &self.NN, &self.NNE, &self.NNEE, &self.NNEEE, &self.NNNWW, &self.NNNW, &self.NNN, &self.NNNE, &self.NNNEE, &self.NNNNW, &self.NNNN, &self.NNNNE, &self.NNNNN, &self.NNNNNN };
            const ols_ctx2 = [12]*const u8{ &self.WWW, &self.WW, &self.W, &self.NWW, &self.NW, &self.N, &self.NE, &self.NEE, &self.NNW, &self.NN, &self.NNE, &self.NNN };
            const ols_ctx3 = [15]*const u8{ &self.N, &self.NE, &self.NEE, &self.NEEE, &self.NEEEE, &self.NN, &self.NNE, &self.NNEE, &self.NNEEE, &self.NNN, &self.NNNE, &self.NNNEE, &self.NNNN, &self.NNNNE, &self.NNNNN };
            const ols_ctx4 = [10]*const u8{ &self.N, &self.NE, &self.NEE, &self.NEEE, &self.NN, &self.NNE, &self.NNEE, &self.NNN, &self.NNNE, &self.NNNN };
            const ols_ctx5 = [14]*const u8{ &self.WWWW, &self.WWW, &self.WW, &self.W, &self.NWWW, &self.NWW, &self.NW, &self.N, &self.NNWW, &self.NNW, &self.NN, &self.NNNW, &self.NNN, &self.NNNN };
            const ols_ctx6 = [8]*const u8{ &self.WWW, &self.WW, &self.W, &self.NNN, &self.NN, &self.N, &self.p1, &self.p2 };
            const ci: usize = @intCast(@min(color, 3));
            const kk: usize = @intCast(@min(if (color > 0) color - 1 else self.stride - 1, 3));
            self.pOLS[0] = u8t(Clip(floorToI32(self.ols[0][ci].predictWith(&ols_ctx1))));
            self.ols[0][kk].update(self.p1);
            self.pOLS[1] = u8t(Clip(floorToI32(self.ols[1][ci].predictWith(&ols_ctx2))));
            self.ols[1][kk].update(self.p1);
            self.pOLS[2] = u8t(Clip(floorToI32(self.ols[2][ci].predictWith(&ols_ctx3))));
            self.ols[2][kk].update(self.p1);
            self.pOLS[3] = u8t(Clip(floorToI32(self.ols[3][ci].predictWith(&ols_ctx4))));
            self.ols[3][kk].update(self.p1);
            self.pOLS[4] = u8t(Clip(floorToI32(self.ols[4][ci].predictWith(&ols_ctx5))));
            self.ols[4][kk].update(self.p1);
            self.pOLS[5] = u8t(Clip(floorToI32(self.ols[5][ci].predictWith(&ols_ctx6))));
            self.ols[5][kk].update(self.p1);

            var mean: i32 = W + NW + N + NE;
            const vari: i32 = (W * W + NW * NW + N * N + NE * NE - @divTrunc(mean * mean, 4)) >> 2;
            mean >>= 2;
            const logvar = core.ilog(@intCast(vari));

            self.ctxa[0] = (@min(color, stride - 1) << 9) | (@as(i32, @intFromBool(@abs(W - N) > 3)) << 8) | (@as(i32, @intFromBool(W > N)) << 7) | (@as(i32, @intFromBool(W > NW)) << 6) | (@as(i32, @intFromBool(@abs(N - NW) > 3)) << 5) | (@as(i32, @intFromBool(N > NW)) << 4) | (@as(i32, @intFromBool(@abs(N - NE) > 3)) << 3) | (@as(i32, @intFromBool(N > NE)) << 2) | (@as(i32, @intFromBool(W > WW)) << 1) | @as(i32, @intFromBool(N > NN));
            self.ctxa[1] = ((LogMeanDiffQt(bget(1), Clip(bget(w + 1) + bget(w - stride + 1) - bget(w * 2 - stride + 1)), 7) >> 1) << 5) | ((LogMeanDiffQt(Clip(N + NE - NNE), Clip(N + NW - NNW), 7) >> 1) << 2) | @min(color, stride - 1);

            var i: u64 = 0;
            i += 1;
            self.cm.set(maps.hash(.{ i, (N + 1) >> 1, LogMeanDiffQt(N, Clip(NN * 2 - NNN), 7) }));
            i += 1;
            self.cm.set(maps.hash(.{ i, (W + 1) >> 1, LogMeanDiffQt(W, Clip(WW * 2 - WWW), 7) }));
            i += 1;
            self.cm.set(maps.hash(.{ i, Clamp4(W + N - NW, W, NW, N, NE), LogMeanDiffQt(Clip(N + NE - NNE), Clip(N + NW - NNW), 7) }));
            i += 1;
            self.cm.set(maps.hash(.{ i, @divTrunc(NNN + N + 4, 8), Clip(N * 3 - NN * 3 + NNN) >> 1 }));
            i += 1;
            self.cm.set(maps.hash(.{ i, @divTrunc(WWW + W + 4, 8), Clip(W * 3 - WW * 3 + WWW) >> 1 }));
            i += 1;
            self.cm.set(maps.hash(.{ i, color, @divTrunc(W + Clip(NE * 3 - NNE * 3 + NNNE), 4), LogMeanDiffQt(N, @divTrunc(NW + NE, 2), 7) }));
            i += 1;
            self.cm.set(maps.hash(.{ i, color, @divTrunc(Clip(@divTrunc(-WWWW + 5 * WWW - 10 * WW + 10 * W + Clamp4(NE * 4 - NNE * 6 + NNNE * 4 - NNNNE, N, NE, NEE, NEEE), 5)), 4) }));
            i += 1;
            self.cm.set(maps.hash(.{ i, Clip(NEE + N - NNEE), LogMeanDiffQt(W, Clip(NW + NE - NNE), 7) }));
            i += 1;
            self.cm.set(maps.hash(.{ i, Clip(NN + W - NNW), LogMeanDiffQt(W, Clip(NNW + WW - NNWW), 7) }));
            i += 1;
            self.cm.set(maps.hash(.{ i, color, p1 }));
            i += 1;
            self.cm.set(maps.hash(.{ i, color, p2 }));
            i += 1;
            self.cm.set(maps.hash(.{ i, color, @divTrunc(Clip(W + N - NW), 2), @divTrunc(Clip(W + p1 - Wp1), 2) }));
            i += 1;
            self.cm.set(maps.hash(.{ i, @divTrunc(Clip(N * 2 - NN), 2), LogMeanDiffQt(N, Clip(NN * 2 - NNN), 7) }));
            i += 1;
            self.cm.set(maps.hash(.{ i, @divTrunc(Clip(W * 2 - WW), 2), LogMeanDiffQt(W, Clip(WW * 2 - WWW), 7) }));
            i += 1;
            self.cm.set(maps.hash(.{ i, @divTrunc(Clamp4(N * 3 - NN * 3 + NNN, W, NW, N, NE), 2) }));
            i += 1;
            self.cm.set(maps.hash(.{ i, @divTrunc(Clamp4(W * 3 - WW * 3 + WWW, W, N, NE, NEE), 2) }));
            i += 1;
            self.cm.set(maps.hash(.{ i, color, LogMeanDiffQt(W, Wp1, 7), Clamp4(@divTrunc(p1 * W, if (Wp1 < 1) 1 else Wp1), W, N, NE, NEE) }));
            i += 1;
            self.cm.set(maps.hash(.{ i, color, Clamp4(N + p2 - Np2, W, NW, N, NE) }));
            i += 1;
            self.cm.set(maps.hash(.{ i, color, Clip(W + N - NW), column0 }));
            i += 1;
            self.cm.set(maps.hash(.{ i, color, Clip(N * 2 - NN), LogMeanDiffQt(W, Clip(NW * 2 - NNW), 7) }));
            i += 1;
            self.cm.set(maps.hash(.{ i, color, Clip(W * 2 - WW), LogMeanDiffQt(N, Clip(NW * 2 - NWW), 7) }));
            i += 1;
            self.cm.set(maps.hash(.{ i, @divTrunc(W + NEE, 2), LogMeanDiffQt(W, @divTrunc(WW + NE, 2), 7) }));
            i += 1;
            self.cm.set(maps.hash(.{ i, Clamp4(Clip(W * 2 - WW) + Clip(N * 2 - NN) - Clip(NW * 2 - NNWW), W, NW, N, NE) }));
            i += 1;
            self.cm.set(maps.hash(.{ i, color, W, p2 }));
            i += 1;
            self.cm.set(maps.hash(.{ i, N, NN & 0x1F, NNN & 0x1F }));
            i += 1;
            self.cm.set(maps.hash(.{ i, W, WW & 0x1F, WWW & 0x1F }));
            i += 1;
            self.cm.set(maps.hash(.{ i, color, N, column0 }));
            i += 1;
            self.cm.set(maps.hash(.{ i, color, Clip(W + NEE - NE), LogMeanDiffQt(W, Clip(WW + NE - N), 7) }));
            i += 1;
            self.cm.set(maps.hash(.{ i, NN, NNNN & 0x1F, NNNNNN & 0x1F, column1 }));
            i += 1;
            self.cm.set(maps.hash(.{ i, WW, WWWW & 0x1F, WWWWWW & 0x1F, column1 }));
            i += 1;
            self.cm.set(maps.hash(.{ i, NNN, NNNNNN & 0x1F, bget(w * 9) & 0x1F, column1 }));
            i += 1;
            self.cm.set(maps.hash(.{ i, color, column1 }));
            i += 1;
            self.cm.set(maps.hash(.{ i, color, W, LogMeanDiffQt(W, WW, 7) }));
            i += 1;
            self.cm.set(maps.hash(.{ i, color, W, p1 }));
            i += 1;
            self.cm.set(maps.hash(.{ i, color, @divTrunc(W, 4), LogMeanDiffQt(W, p1, 7), LogMeanDiffQt(W, p2, 7) }));
            i += 1;
            self.cm.set(maps.hash(.{ i, color, N, LogMeanDiffQt(N, NN, 7) }));
            i += 1;
            self.cm.set(maps.hash(.{ i, color, N, p1 }));
            i += 1;
            self.cm.set(maps.hash(.{ i, color, @divTrunc(N, 4), LogMeanDiffQt(N, p1, 7), LogMeanDiffQt(N, p2, 7) }));
            i += 1;
            self.cm.set(maps.hash(.{ i, color, (W + N) >> 3, p1 >> 4, p2 >> 4 }));
            i += 1;
            self.cm.set(maps.hash(.{ i, color, @divTrunc(p1, 2), @divTrunc(p2, 2) }));
            i += 1;
            self.cm.set(maps.hash(.{ i, color, W, p1 - Wp1 }));
            i += 1;
            self.cm.set(maps.hash(.{ i, color, W + p1 - Wp1 }));
            i += 1;
            self.cm.set(maps.hash(.{ i, color, N, p1 - Np1 }));
            i += 1;
            self.cm.set(maps.hash(.{ i, color, N + p1 - Np1 }));
            i += 1;
            self.cm.set(maps.hash(.{ i, bget(w * 3 - stride), bget(w * 3 - 2 * stride) }));
            i += 1;
            self.cm.set(maps.hash(.{ i, bget(w * 3 + stride), bget(w * 3 + 2 * stride) }));
            i += 1;
            self.cm.set(maps.hash(.{ i, color, mean, logvar >> 4 }));

            self.Map[0].setDirect(@bitCast((W & 0xC0) | ((N & 0xC0) >> 2) | ((WW & 0xC0) >> 4) | (NN >> 6)));
            self.Map[1].setDirect(@bitCast((N & 0xC0) | ((NN & 0xC0) >> 2) | ((NE & 0xC0) >> 4) | (NEE >> 6)));
            self.Map[2].setDirect(@bitCast(bget(1)));
            self.Map[3].setDirect(@bitCast(@min(color, stride - 1)));
            if (stats) |st| {
                st.Image.plane = @intCast(@min(color, stride - 1));
                st.Image.pixels.W = self.W;
                st.Image.pixels.N = self.N;
                st.Image.pixels.NN = self.NN;
                st.Image.pixels.WW = self.WW;
                st.Image.pixels.Wp1 = self.Wp1;
                st.Image.pixels.Np1 = self.Np1;
                st.Image.ctx = @truncate(@as(u32, @bitCast(self.ctxa[0] >> 3)));
            }
        }

        const B: i32 = (@as(i32, @bitCast(@as(u32, @intCast(core.c0)) << @intCast(8 - bpos)))) & 0xFF;
        {
            const W: i32 = self.W;
            const WW: i32 = self.WW;
            const N: i32 = self.N;
            const NN: i32 = self.NN;
            const NW: i32 = self.NW;
            const NE: i32 = self.NE;
            const NNE: i32 = self.NNE;
            const NNW: i32 = self.NNW;
            const NWW: i32 = self.NWW;
            const p1: i32 = self.p1;
            const p2: i32 = self.p2;
            const Wp1: i32 = self.Wp1;
            const Np1: i32 = self.Np1;
            const NWp1: i32 = self.NWp1;
            const Wp2: i32 = self.Wp2;
            const Np2: i32 = self.Np2;
            const NWp2: i32 = self.NWp2;
            const color = self.color;
            const stride = self.stride;
            const bposu: u64 = @intCast(bpos);
            var i: usize = 5;
            self.Map[i].setDirect(@bitCast((@as(i32, u8t(Clip(W + N - NW) - B)) * 8 + bpos) | (LogMeanDiffQt(Clip(N + NE - NNE), Clip(N + NW - NNW), 7) << 11)));
            i += 1;
            self.Map[i].setDirect(@bitCast((@as(i32, u8t(Clip(N * 2 - NN) - B)) * 8 + bpos) | (LogMeanDiffQt(W, Clip(NW * 2 - NNW), 7) << 11)));
            i += 1;
            self.Map[i].setDirect(@bitCast((@as(i32, u8t(Clip(W * 2 - WW) - B)) * 8 + bpos) | (LogMeanDiffQt(N, Clip(NW * 2 - NWW), 7) << 11)));
            i += 1;
            self.Map[i].setDirect(@bitCast((@as(i32, u8t(Clip(W + N - NW) - B)) * 8 + bpos) | (LogMeanDiffQt(p1, Clip(Wp1 + Np1 - NWp1), 7) << 11)));
            i += 1;
            self.Map[i].setDirect(@bitCast((@as(i32, u8t(Clip(W + N - NW) - B)) * 8 + bpos) | (LogMeanDiffQt(p2, Clip(Wp2 + Np2 - NWp2), 7) << 11)));
            i += 1;
            self.Map[i].set(maps.hash(.{ W - B, N - B }) *% 8 +% bposu);
            i += 1;
            self.Map[i].set(maps.hash(.{ W - B, WW - B }) *% 8 +% bposu);
            i += 1;
            self.Map[i].set(maps.hash(.{ N - B, NN - B }) *% 8 +% bposu);
            i += 1;
            self.Map[i].set(maps.hash(.{ Clip(N + NE - NNE) - B, Clip(N + NW - NNW) - B }) *% 8 +% bposu);
            i += 1;
            self.Map[i].setDirect(@bitCast((@min(color, stride - 1) << 11) | (@as(i32, u8t(Clip(N + p1 - Np1) - B)) * 8 + bpos)));
            i += 1;
            self.Map[i].setDirect(@bitCast((@min(color, stride - 1) << 11) | (@as(i32, u8t(Clip(N + p2 - Np2) - B)) * 8 + bpos)));
            i += 1;
            self.Map[i].setDirect(@bitCast((@min(color, stride - 1) << 11) | (@as(i32, u8t(Clip(W + p1 - Wp1) - B)) * 8 + bpos)));
            i += 1;
            self.Map[i].setDirect(@bitCast((@min(color, stride - 1) << 11) | (@as(i32, u8t(Clip(W + p2 - Wp2) - B)) * 8 + bpos)));
            i += 1;
            var jm: usize = 0;
            while (jm < im24_nMaps1) : ({
                i += 1;
                jm += 1;
            }) {
                self.Map[i].setDirect(@bitCast((@as(i32, self.MapCtxs[jm]) - B) * 8 + bpos));
            }
            jm = 0;
            while (i < im24_nMaps) : ({
                i += 1;
                jm += 1;
            }) {
                self.Map[i].setDirect(@bitCast((@as(i32, self.pOLS[jm]) - B) * 8 + bpos));
            }
            for (0..im24_nSCMaps - 1) |k| {
                self.SCMap[k].set(@bitCast((@as(i32, self.SCMapCtxs[k]) - B) * 8 + bpos));
            }
        }

        _ = self.cm.mix(m);
        for (0..im24_nMaps) |i| self.Map[i].mix(m, 1, 3, 1023);
        for (0..im24_nSCMaps) |i| self.SCMap[i].mix(m, 9, 1, 3);
        self.col24 += 1;
        if (self.col24 >= self.stride * 8) self.col24 = 0;
        const W: i32 = self.W;
        const WW: i32 = self.WW;
        const N: i32 = self.N;
        const NN: i32 = self.NN;
        const stride = self.stride;
        const c0 = core.c0;
        m.set(0, 1);
        m.set(@min(63, self.column[0]) + ((self.ctxa[0] >> 3) & 0xC0), 256);
        m.set(@min(127, self.column[1]) + ((self.ctxa[0] >> 2) & 0x180), 512);
        m.set((self.ctxa[0] & 0x7FC) | (bpos >> 1), 2048);
        m.set(self.col24, stride * 8);
        m.set(@mod(self.x, stride), stride);
        m.set(c0, 256);
        m.set((self.ctxa[1] << 2) | (bpos >> 1), 1024);
        m.set(@bitCast(maps.finalize64(maps.hash(.{ LogMeanDiffQt(W, WW, 5), LogMeanDiffQt(N, NN, 5), LogMeanDiffQt(W, N, 5), maps.ilog2(@intCast(W)), self.color }), 13)), 8192);
        m.set(@bitCast(maps.finalize64(maps.hash(.{ self.ctxa[0], @divTrunc(self.column[0], 8) }), 13)), 8192);
        m.set(@bitCast(maps.finalize64(maps.hash(.{ LogQt(N, 5), LogMeanDiffQt(N, NN, 3), c0 }), 13)), 8192);
        m.set(@bitCast(maps.finalize64(maps.hash(.{ LogQt(W, 5), LogMeanDiffQt(W, WW, 3), c0 }), 13)), 8192);
        m.set(@min(255, @divTrunc(self.x +% self.line, 32)), 256);
    }
};

// ================================= imgModel ================================
const BMPImage = struct {
    Header: u32 = 0,
    Offset: u32 = 0,
    Bpp: u32 = 0,
    Size: u32 = 0,
    Palette: u32 = 0,
    HdrLess: u32 = 0,
    Width: u32 = 0,
    Height: u32 = 0,
    BitMask: u32 = 0,
};
const TGAImage = struct {
    Header: u32 = 0,
    IdLength: u32 = 0,
    Bpp: u32 = 0,
    ImgType: u32 = 0,
    MapSize: u32 = 0,
    Width: u32 = 0,
    Height: u32 = 0,
};

pub const ImgModel = struct {
    im1: Im1Bit,
    im4: Im4Bit,
    im8: Im8Bit,
    im24: Im24Bit,
    w: i32 = 0,
    bpp: i32 = 0,
    eoi: i32 = 0,
    BMP: BMPImage = .{},
    TGA: TGAImage = .{},
    alpha: i32 = 0,
    gray: i32 = 0,
    pltorder: i32 = 0,
    alloc: Allocator,

    pub fn init(a: Allocator) ImgModel {
        return .{
            .im1 = Im1Bit.init(a),
            .im4 = Im4Bit.init(a),
            .im8 = Im8Bit.init(a),
            .im24 = Im24Bit.init(a),
            .alloc = a,
        };
    }

    pub fn deinit(self: *ImgModel) void {
        self.im1.deinit();
        self.im4.deinit();
        self.im8.deinit();
        self.im24.deinit();
    }

    fn checkIfGrayscale(self: *ImgModel, x: u32, a: i32, stats: ?*core.ModelStats) void {
        if (self.w == 0 and self.gray != 0 and @mod(x, @as(u32, @intCast(3 + a))) == 0) {
            var i: i32 = 0;
            while (i < (3 + a) and self.gray != 0) : (i += 1) {
                const Bv: i32 = bget(4 - i);
                if (self.gray >> 9 != 0) {
                    self.gray = 0x100 | Bv;
                    self.pltorder = 1 - 2 * @as(i32, @intFromBool(Bv > 0));
                    if (stats) |st| st.Record = (st.Record & 0xFFFF) | (@as(u32, @intCast(3 + a)) << 16);
                    continue;
                }
                if (i == 0) {
                    self.gray = self.gray & (@as(i32, @intFromBool(Bv - (self.gray & 0xFF) == self.pltorder)) << 8);
                    self.gray |= if (self.gray != 0) Bv else 0;
                } else if (i == 3) {
                    self.gray &= @as(i32, @intFromBool(Bv == 0 or Bv == 0xFF)) * 0x1FF;
                } else {
                    self.gray &= @as(i32, @intFromBool(Bv == (self.gray & 0xFF))) * 0x1FF;
                }
            }
        }
    }

    pub fn imgModel(self: *ImgModel, m: *Mixer, stats: ?*core.ModelStats) i32 {
        const pos = core.pos;
        if (core.bpos == 0) {
            // detect .BMP/DIB images
            var trig = false;
            if (pos >= (self.eoi + 40) and self.BMP.Header == 0) {
                if (bget(54) == 'B' and bget(53) == 'M') {
                    self.BMP.Offset = rd4(44);
                    if ((self.BMP.Offset & 0xFFFFFBF7) == 0x36 and rd4(40) == 0x28) trig = true;
                }
                if (!trig) {
                    self.BMP.HdrLess = @intFromBool(rd4(40) == 0x28);
                    if (self.BMP.HdrLess != 0) trig = true;
                }
            }
            if (trig) {
                self.BMP.Width = rd4(36);
                self.BMP.Height = @intCast(@abs(@as(i32, @bitCast(rd4(32)))));
                self.BMP.Bpp = @intCast(rd2(26));
                self.BMP.Size = rd4(20);
                self.BMP.Palette = rd4(4);
                const bpp = self.BMP.Bpp;
                self.BMP.Header = @intFromBool(rd4(24) == 0 and rd2(28) == 1 and
                    (bpp == 1 or bpp == 4 or bpp == 8 or bpp == 24 or bpp == 32) and
                    self.BMP.Width < 30000 and self.BMP.Height < 10000 and
                    (self.BMP.Palette == 0 or (@as(u64, 1) << @intCast(bpp)) >= self.BMP.Palette));
                if (self.BMP.Header != 0) {
                    self.BMP.Offset = if (self.BMP.HdrLess != 0)
                        (if (bpp < 24) (if (self.BMP.Palette != 0) self.BMP.Palette * 4 else @as(u32, 4) << @intCast(bpp)) else 0)
                    else
                        self.BMP.Offset - 54;
                    self.gray = if (bpp == 8) 0x300 else 0;
                    if (self.BMP.HdrLess != 0 and (self.BMP.Width * 2 == self.BMP.Height) and bpp > 1 and blk: {
                        const W = self.BMP.Width;
                        const okSize = self.BMP.Size > 0 and self.BMP.Size == ((self.BMP.Width * self.BMP.Height * (bpp + 1)) >> 4);
                        const smallSize = (self.BMP.Size == 0 or self.BMP.Size < ((self.BMP.Width * self.BMP.Height * bpp) >> 3)) and
                            (W == 8 or W == 10 or W == 14 or W == 16 or W == 20 or W == 22 or W == 24 or W == 32 or
                            W == 40 or W == 48 or W == 60 or W == 64 or W == 72 or W == 80 or W == 96 or W == 128 or W == 256);
                        break :blk okSize or smallSize;
                    }) {
                        self.BMP.Height = self.BMP.Width;
                        self.BMP.BitMask = self.BMP.Width;
                    }
                }
            } else {
                self.BMP.Offset -%= @intFromBool(self.BMP.Offset > 0);
                self.checkIfGrayscale(self.BMP.Offset, 1, stats);
            }

            if (self.BMP.Offset == 0 and (self.BMP.Header > 0 or self.BMP.BitMask > 0) and pos >= self.eoi) {
                if (self.BMP.Header == 0 and self.BMP.BitMask != 0) {
                    self.BMP.Header = 1;
                    self.BMP.Bpp = 1;
                    self.BMP.Width = self.BMP.BitMask;
                    self.BMP.BitMask = 0;
                }
                self.bpp = @intCast(self.BMP.Bpp);
                const bpp = self.bpp;
                const Wd = self.BMP.Width;
                self.w = if (bpp > 4)
                    @bitCast((Wd *% @as(u32, @intCast(bpp >> 3)) +% 3) & 0xFFFFFFFC)
                else if (bpp == 1)
                    @bitCast(((((Wd -% 1) >> 5) +% 1) *% 4))
                else
                    @bitCast(((Wd *% 4 +% 31) >> 5) *% 4);
                self.alpha = @intFromBool(bpp == 32);
                self.eoi = @bitCast(@as(u32, @bitCast(self.w)) *% self.BMP.Height);
                if (self.eoi > 64) {
                    self.eoi = self.eoi + pos;
                } else {
                    self.BMP.Header = 0;
                    self.w = 0;
                    self.eoi = 0;
                }
            }

            // detect .TGA images
            if (pos >= (self.eoi + 8) and self.TGA.Header == 0) {
                if ((m4(8) & 0xFFFFFF) == 0x010100 and (m4(4) & 0xFFFFFFC7) == 0x00000100 and
                    (bget(1) == 16 or bget(1) == 24 or bget(1) == 32))
                {
                    self.TGA.Header = @intCast(pos);
                    self.TGA.IdLength = @intCast(bget(8));
                    self.TGA.MapSize = @intCast(@divTrunc(bget(1), 8));
                    self.TGA.Bpp = 8;
                    self.TGA.ImgType = 1;
                } else if ((m4(8) & 0xFFFEFF) == 0x000200 and m4(4) == 0) {
                    self.TGA.Header = @intCast(pos);
                    self.TGA.IdLength = @intCast(bget(8));
                    self.TGA.ImgType = @intCast(bget(6));
                    self.TGA.Bpp = if (self.TGA.ImgType == 2) 24 else 8;
                }
            } else if (self.w == 0 and self.TGA.Header != 0) {
                const p: u32 = @as(u32, @intCast(pos)) -% self.TGA.Header;
                if (p == 8) {
                    self.TGA.Width = @intCast(rd2(4));
                    self.TGA.Height = @intCast(rd2(2));
                    self.TGA.Header *%= @intFromBool(rd4(8) == 0 and self.TGA.Width != 0 and self.TGA.Width < 0x3FFF and self.TGA.Height != 0 and self.TGA.Height < 0x3FFF);
                } else if (p == 10) {
                    const iv: u16 = @intCast(m2(2));
                    if ((iv & 0xFFF7) == (32 << 8)) self.TGA.Bpp = 32;
                    if ((iv & 0xFFD7) != (self.TGA.Bpp << 8)) self.TGA = .{};
                }
                if (self.TGA.Header != 0 and p == 10 + self.TGA.IdLength + self.TGA.MapSize * 256) {
                    self.w = @bitCast((self.TGA.Width *% self.TGA.Bpp) >> 3);
                    self.gray = @intFromBool(self.TGA.ImgType == 3);
                    self.bpp = @intCast(self.TGA.Bpp);
                    self.alpha = @intFromBool(self.bpp == 32);
                    self.eoi = @bitCast(@as(u32, @bitCast(self.w)) *% self.TGA.Height);
                    if (self.eoi > 64) {
                        self.eoi = self.eoi + pos;
                    } else {
                        self.TGA.Header = 0;
                        self.w = 0;
                        self.eoi = 0;
                    }
                }
            }
        }
        if (pos > self.eoi) {
            self.w = 0;
            return 0;
        }
        if (self.w != 0) {
            switch (self.bpp) {
                1 => self.im1.im1bitModel(m, self.w),
                4 => self.im4.im4bitModel(m, self.w),
                8 => {
                    self.im8.im8bitModel(m, self.w, stats, self.gray);
                    if (stats) |st| st.Type = if (self.gray != 0) core.FT_IMAGE8GRAY else core.FT_IMAGE8;
                },
                else => {
                    self.im24.im24bitModel(m, self.w, stats, self.alpha);
                    if (stats) |st| st.Type = if (self.alpha != 0) core.FT_IMAGE32 else core.FT_IMAGE24;
                },
            }
        }
        if (core.bpos == 7 and (pos + 1) == self.eoi) {
            self.TGA = .{};
            self.BMP.Header = 0;
            self.gray = 0;
            self.alpha = 0;
        }
        return self.w;
    }
};

// =============================== audio8bModel ==============================
const a8_nOLS = 8;
const a8_nLnrPrd = a8_nOLS + 3; // 11
const a8_num = [a8_nOLS]usize{ 128, 90, 90, 90, 90, 90, 28, 28 };
const a8_kmax = [a8_nOLS]usize{ 24, 30, 31, 32, 33, 34, 4, 3 };
const a8_lambda = [a8_nOLS]f64{ 0.9975, 0.9965, 0.996, 0.995, 0.995, 0.9985, 0.98, 0.992 };

pub const Audio8b = struct {
    sMap1b: [a8_nLnrPrd][3]maps.SmallStationaryContextMap,
    ols: [a8_nOLS][2]OlsI8,
    prd: [a8_nLnrPrd][2][2]i32 = .{.{.{ 0, 0 }} ** 2} ** a8_nLnrPrd,
    residuals: [a8_nLnrPrd][2]i32 = .{.{ 0, 0 }} ** a8_nLnrPrd,
    stereo: i32 = 0,
    ch: i32 = 0,
    rpos: i32 = 0,
    lastPos: i32 = 0,
    mask: u32 = 0,
    errLog: u32 = 0,
    mxCtx: u32 = 0,
    alloc: Allocator,

    pub fn init(a: Allocator) Audio8b {
        var self = Audio8b{ .sMap1b = undefined, .ols = undefined, .alloc = a };
        for (0..a8_nLnrPrd) |i| {
            for (0..3) |k| self.sMap1b[i][k] = maps.SmallStationaryContextMap.init(a, 11, 1);
        }
        for (0..a8_nOLS) |i| {
            for (0..2) |k| self.ols[i][k] = OlsI8.init(a, a8_num[i], a8_kmax[i], a8_lambda[i], 0.001);
        }
        return self;
    }

    pub fn deinit(self: *Audio8b) void {
        for (0..a8_nLnrPrd) |i| {
            for (0..3) |k| self.sMap1b[i][k].deinit();
        }
        for (0..a8_nOLS) |i| {
            for (0..2) |k| self.ols[i][k].deinit();
        }
    }

    pub fn audio8bModel(self: *Audio8b, m: *Mixer, info: i32, stats: ?*core.ModelStats) void {
        const bpos = core.bpos;
        const B: i8 = toI8(core.c0 << @intCast(8 - bpos));

        if (bpos == 0) {
            self.rpos = if (core.pos == self.lastPos + 1) self.rpos + 1 else 0;
            self.lastPos = core.pos;
            if (self.rpos == 0) {
                self.stereo = info & 1;
                self.mask = 0;
                if (stats) |st| st.Record = (@as(u32, @intCast(self.stereo + 1)) << 16) | (st.Record & 0xFFFF);
                wmode = info;
            }
            self.ch = if (self.stereo != 0) core.blpos & 1 else 0;
            const s: i8 = toI8((if ((info & 4) > 0) bget(1) ^ 128 else bget(1)) - 128);
            const si: i32 = s;
            const pCh: usize = @intCast(self.ch ^ self.stereo);
            const ch: usize = @intCast(self.ch);
            const stereo = self.stereo;

            self.errLog = 0;
            var oi: usize = 0;
            while (oi < a8_nOLS) : (oi += 1) {
                self.ols[oi][pCh].update(s);
                self.residuals[oi][pCh] = si - self.prd[oi][pCh][0];
                const absResidual: u32 = @intCast(@abs(self.residuals[oi][pCh]));
                self.mask +%= self.mask +% @intFromBool(absResidual > 4);
                self.errLog +%= absResidual *% absResidual;
            }
            while (oi < a8_nLnrPrd) : (oi += 1) {
                self.residuals[oi][pCh] = si - self.prd[oi][pCh][0];
            }
            self.errLog = @min(0xF, maps.ilog2(self.errLog));
            self.mxCtx = maps.ilog2(@min(0x1F, BitCount(self.mask))) *% 2 +% @as(u32, @intCast(self.ch));

            var k1: i32 = 90;
            var k2: i32 = k1 - 12 * stereo;
            {
                var j: i32 = 1;
                var i: i32 = 1;
                while (j <= k1) : ({
                    j += 1;
                    i += @as(i32, 1) << @intCast(@as(i32, @intFromBool(j > 8)) + @intFromBool(j > 16) + @intFromBool(j > 64));
                }) self.ols[1][ch].add(toI8(X1(i)));
            }
            {
                var j: i32 = 1;
                var i: i32 = 1;
                while (j <= k2) : ({
                    j += 1;
                    i += @as(i32, 1) << @intCast(@as(i32, @intFromBool(j > 5)) + @intFromBool(j > 10) + @intFromBool(j > 17) + @intFromBool(j > 26) + @intFromBool(j > 37));
                }) self.ols[2][ch].add(toI8(X1(i)));
            }
            {
                var j: i32 = 1;
                var i: i32 = 1;
                while (j <= k2) : ({
                    j += 1;
                    i += @as(i32, 1) << @intCast(@as(i32, @intFromBool(j > 3)) + @intFromBool(j > 7) + @intFromBool(j > 14) + @intFromBool(j > 20) + @intFromBool(j > 33) + @intFromBool(j > 49));
                }) self.ols[3][ch].add(toI8(X1(i)));
            }
            {
                var j: i32 = 1;
                var i: i32 = 1;
                while (j <= k2) : ({
                    j += 1;
                    i += 1 + @as(i32, @intFromBool(j > 4)) + @intFromBool(j > 8);
                }) self.ols[4][ch].add(toI8(X1(i)));
            }
            {
                var j: i32 = 1;
                var i: i32 = 1;
                while (j <= k1) : ({
                    j += 1;
                    i += 2 + (@as(i32, @intFromBool(j > 3)) + @intFromBool(j > 9) + @intFromBool(j > 19) + @intFromBool(j > 36) + @intFromBool(j > 61));
                }) self.ols[5][ch].add(toI8(X1(i)));
            }
            if (stereo != 0) {
                var i: i32 = 1;
                while (i <= k1 - k2) : (i += 1) {
                    const sv: f64 = @floatFromInt(X2(i));
                    self.ols[2][ch].addFloat(sv);
                    self.ols[3][ch].addFloat(sv);
                    self.ols[4][ch].addFloat(sv);
                }
            }
            k1 = 28;
            k2 = k1 - 6 * stereo;
            var i: i32 = 1;
            while (i <= k2) : (i += 1) {
                const sv: f64 = @floatFromInt(X1(i));
                self.ols[0][ch].addFloat(sv);
                self.ols[6][ch].addFloat(sv);
                self.ols[7][ch].addFloat(sv);
            }
            while (i <= 96) : (i += 1) self.ols[0][ch].add(toI8(X1(i)));
            if (stereo != 0) {
                i = 1;
                while (i <= k1 - k2) : (i += 1) {
                    const sv: f64 = @floatFromInt(X2(i));
                    self.ols[0][ch].addFloat(sv);
                    self.ols[6][ch].addFloat(sv);
                    self.ols[7][ch].addFloat(sv);
                }
                i = k1 - k2 + 1;
                while (i <= 32) : (i += 1) self.ols[0][ch].add(toI8(X2(i)));
            } else {
                while (i <= 128) : (i += 1) self.ols[0][ch].add(toI8(X1(i)));
            }

            var pi: usize = 0;
            while (pi < a8_nOLS) : (pi += 1) {
                self.prd[pi][ch][0] = signedClip8(floorToI32(self.ols[pi][ch].predict()));
                self.prd[pi][ch][1] = signedClip8(self.prd[pi][ch][0] + self.residuals[pi][pCh]);
            }
            self.prd[8][ch][0] = signedClip8(X1(1) * 2 - X1(2));
            self.prd[9][ch][0] = signedClip8(X1(1) * 3 - X1(2) * 3 + X1(3));
            self.prd[10][ch][0] = signedClip8(X1(1) * 4 - X1(2) * 6 + X1(3) * 4 - X1(4));
            pi = a8_nOLS;
            while (pi < a8_nLnrPrd) : (pi += 1) {
                self.prd[pi][ch][1] = signedClip8(self.prd[pi][ch][0] + self.residuals[pi][pCh]);
            }
        }

        const ch: usize = @intCast(self.ch);
        const Bi: i32 = B;
        for (0..a8_nLnrPrd) |ii| {
            const ctx: u32 = @bitCast((self.prd[ii][ch][0] - Bi) * 8 + bpos);
            self.sMap1b[ii][0].set(ctx);
            self.sMap1b[ii][1].set(ctx);
            self.sMap1b[ii][2].set(@bitCast((self.prd[ii][ch][1] - Bi) * 8 + bpos));
            const div: i32 = 2 + @as(i32, @intFromBool(ii >= a8_nOLS));
            self.sMap1b[ii][0].mix(m, 6, 1, div);
            self.sMap1b[ii][1].mix(m, 9, 1, div);
            self.sMap1b[ii][2].mix(m, 7, 1, 3);
        }
        m.set(@bitCast((self.errLog << 8) | @as(u32, @intCast(core.c0))), 4096);
        m.set(@bitCast((@as(u32, u8t(@bitCast(self.mask))) << 3) | (@as(u32, @intCast(self.ch)) << 2) | (@as(u32, @intCast(bpos)) >> 1)), 2048);
        m.set(@bitCast((self.mxCtx << 7) | @as(u32, @intCast(bget(1) >> 1))), 1280);
        m.set(@bitCast((self.errLog << 4) | (@as(u32, @intCast(self.ch)) << 3) | @as(u32, @intCast(bpos))), 256);
        m.set(@bitCast(self.mxCtx), 10);
    }
};

// ================================ wavModel =================================
// Model a 16/8-bit stereo/mono uncompressed .wav file (Ghido predictor).
// `long double sum` is modelled with f80; F/L are f64 (double).
pub const Wav = struct {
    pr: [3][2]i32 = .{.{ 0, 0 }} ** 3,
    n: [2]i32 = .{ 0, 0 },
    counter: [2]i32 = .{ 0, 0 },
    F: [49][49][2]f64 = std.mem.zeroes([49][49][2]f64),
    L: [49][49]f64 = std.mem.zeroes([49][49]f64),
    rpos: i32 = 0,
    lastPos: i32 = 0,
    scm1: maps.SmallStationaryContextMap,
    scm2: maps.SmallStationaryContextMap,
    scm3: maps.SmallStationaryContextMap,
    scm4: maps.SmallStationaryContextMap,
    scm5: maps.SmallStationaryContextMap,
    scm6: maps.SmallStationaryContextMap,
    scm7: maps.SmallStationaryContextMap,
    cm: maps.ContextMap,
    bits: i32 = 0,
    channels: i32 = 0,
    w: i32 = 0,
    ch: i32 = 0,
    col: i32 = 0,
    z1: i32 = 0,
    z2: i32 = 0,
    z3: i32 = 0,
    z4: i32 = 0,
    z5: i32 = 0,
    z6: i32 = 0,
    z7: i32 = 0,
    alloc: Allocator,

    pub fn init(a: Allocator) Wav {
        return .{
            .scm1 = maps.SmallStationaryContextMap.init(a, 8, 8),
            .scm2 = maps.SmallStationaryContextMap.init(a, 8, 8),
            .scm3 = maps.SmallStationaryContextMap.init(a, 8, 8),
            .scm4 = maps.SmallStationaryContextMap.init(a, 8, 8),
            .scm5 = maps.SmallStationaryContextMap.init(a, 8, 8),
            .scm6 = maps.SmallStationaryContextMap.init(a, 8, 8),
            .scm7 = maps.SmallStationaryContextMap.init(a, 8, 8),
            .cm = maps.ContextMap.init(a, core.MEM() * 2, 10 + 1),
            .alloc = a,
        };
    }

    pub fn deinit(self: *Wav) void {
        self.scm1.deinit();
        self.scm2.deinit();
        self.scm3.deinit();
        self.scm4.deinit();
        self.scm5.deinit();
        self.scm6.deinit();
        self.scm7.deinit();
        self.cm.deinit();
    }

    pub fn wavModel(self: *Wav, m: *Mixer, info: i32, stats: ?*core.ModelStats) void {
        @setEvalBranchQuota(2000000);
        const bpos = core.bpos;
        const a: f64 = 0.996;
        const a2: f64 = 1.0 / a;

        if (bpos == 0) {
            self.rpos = if (core.pos == self.lastPos + 1) self.rpos + 1 else 0;
            self.lastPos = core.pos;
        }

        if (bpos == 0 and self.rpos == 0) {
            self.bits = @divTrunc(@mod(info, 4), 2) * 8 + 8;
            self.channels = @mod(info, 2) + 1;
            self.col = 0;
            self.w = self.channels * (self.bits >> 3);
            wmode = info;
            if (self.channels == 1) {
                S = 48;
                D = 0;
            } else {
                S = 36;
                D = 12;
            }
            const SD: usize = @intCast(S + D);
            var jc: usize = 0;
            while (jc < @as(usize, @intCast(self.channels))) : (jc += 1) {
                for (0..SD + 1) |k| {
                    for (0..SD + 1) |l| {
                        self.F[k][l][jc] = 0;
                        self.L[k][l] = 0;
                    }
                }
                self.F[1][0][jc] = 1;
                self.n[jc] = 0;
                self.counter[jc] = 0;
                self.pr[2][jc] = 0;
                self.pr[1][jc] = 0;
                self.pr[0][jc] = 0;
                self.z1 = 0;
                self.z2 = 0;
                self.z3 = 0;
                self.z4 = 0;
                self.z5 = 0;
                self.z6 = 0;
                self.z7 = 0;
            }
        }

        if (bpos == 0 and self.rpos >= self.w) {
            self.ch = @mod(self.rpos, self.w);
            const bytes = self.bits >> 3;
            const msb = @mod(self.ch, bytes);
            const chn: usize = @intCast(@divTrunc(self.ch, bytes));
            const Sc = S;
            const Dc = D;
            if (msb == 0) {
                self.z1 = X1(1);
                self.z2 = X1(2);
                self.z3 = X1(3);
                self.z4 = X1(4);
                self.z5 = X1(5);
                var k: i32 = X1(1);
                {
                    var l: i32 = 0;
                    const lim = @min(Sc, self.counter[chn] - 1);
                    while (l <= lim) : (l += 1) {
                        self.F[0][@intCast(l)][chn] *= a;
                        self.F[0][@intCast(l)][chn] += @as(f64, @floatFromInt(X1(l + 1) * k));
                    }
                }
                {
                    var l: i32 = 1;
                    const lim = @min(Dc, self.counter[chn]);
                    while (l <= lim) : (l += 1) {
                        self.F[0][@intCast(l + Sc)][chn] *= a;
                        self.F[0][@intCast(l + Sc)][chn] += @as(f64, @floatFromInt(X2(l + 1) * k));
                    }
                }
                if (self.channels == 2) {
                    k = X2(2);
                    {
                        var l: i32 = 1;
                        const lim = @min(Dc, self.counter[chn]);
                        while (l <= lim) : (l += 1) {
                            self.F[@intCast(Sc + 1)][@intCast(l + Sc)][chn] *= a;
                            self.F[@intCast(Sc + 1)][@intCast(l + Sc)][chn] += @as(f64, @floatFromInt(X2(l + 1) * k));
                        }
                    }
                    {
                        var l: i32 = 1;
                        const lim = @min(Sc, self.counter[chn] - 1);
                        while (l <= lim) : (l += 1) {
                            self.F[@intCast(l)][@intCast(Sc + 1)][chn] *= a;
                            self.F[@intCast(l)][@intCast(Sc + 1)][chn] += @as(f64, @floatFromInt(X1(l + 1) * k));
                        }
                    }
                    self.z6 = X2(1) + X1(1) - X2(2);
                    self.z7 = X2(1);
                } else {
                    self.z6 = 2 * X1(1) - X1(2);
                    self.z7 = X1(1);
                }
                self.n[chn] += 1;
                if (self.n[chn] == 1) {
                    const SD = Sc + Dc;
                    if (self.channels == 1) {
                        var kk: i32 = 1;
                        while (kk <= SD) : (kk += 1) {
                            var l: i32 = kk;
                            while (l <= SD) : (l += 1) {
                                self.F[@intCast(kk)][@intCast(l)][chn] = (self.F[@intCast(kk - 1)][@intCast(l - 1)][chn] - @as(f64, @floatFromInt(X1(kk) * X1(l)))) * a2;
                            }
                        }
                    } else {
                        var kk: i32 = 1;
                        while (kk <= SD) : (kk += 1) {
                            if (kk != Sc + 1) {
                                var l: i32 = kk;
                                while (l <= SD) : (l += 1) {
                                    if (l != Sc + 1) {
                                        const xk = if (kk - 1 <= Sc) X1(kk) else X2(kk - Sc);
                                        const xl = if (l - 1 <= Sc) X1(l) else X2(l - Sc);
                                        self.F[@intCast(kk)][@intCast(l)][chn] = (self.F[@intCast(kk - 1)][@intCast(l - 1)][chn] - @as(f64, @floatFromInt(xk * xl))) * a2;
                                    }
                                }
                            }
                        }
                    }
                    var i: i32 = 1;
                    while (i <= SD) : (i += 1) {
                        var sum: f80 = @as(f80, self.F[@intCast(i)][@intCast(i)][chn]);
                        var kk: i32 = 1;
                        while (kk < i) : (kk += 1) sum -= @as(f80, self.L[@intCast(i)][@intCast(kk)] * self.L[@intCast(i)][@intCast(kk)]);
                        sum = @floor(sum + 0.5);
                        sum = 1.0 / sum;
                        if (sum > 0) {
                            self.L[@intCast(i)][@intCast(i)] = @floatCast(@sqrt(sum));
                            var j: i32 = i + 1;
                            while (j <= SD) : (j += 1) {
                                var sum2: f80 = @as(f80, self.F[@intCast(i)][@intCast(j)][chn]);
                                var k2: i32 = 1;
                                while (k2 < i) : (k2 += 1) sum2 -= @as(f80, self.L[@intCast(j)][@intCast(k2)] * self.L[@intCast(i)][@intCast(k2)]);
                                sum2 = @floor(sum2 + 0.5);
                                self.L[@intCast(j)][@intCast(i)] = @floatCast(sum2 * @as(f80, self.L[@intCast(i)][@intCast(i)]));
                            }
                        } else break;
                    }
                    if (i > SD and self.counter[chn] > Sc + 1) {
                        var kk: i32 = 1;
                        while (kk <= SD) : (kk += 1) {
                            self.F[@intCast(kk)][0][chn] = self.F[0][@intCast(kk)][chn];
                            var j: i32 = 1;
                            while (j < kk) : (j += 1) self.F[@intCast(kk)][0][chn] -= self.L[@intCast(kk)][@intCast(j)] * self.F[@intCast(j)][0][chn];
                            self.F[@intCast(kk)][0][chn] *= self.L[@intCast(kk)][@intCast(kk)];
                        }
                        var kk2: i32 = SD;
                        while (kk2 > 0) : (kk2 -= 1) {
                            var j: i32 = kk2 + 1;
                            while (j <= SD) : (j += 1) self.F[@intCast(kk2)][0][chn] -= self.L[@intCast(j)][@intCast(kk2)] * self.F[@intCast(j)][0][chn];
                            self.F[@intCast(kk2)][0][chn] *= self.L[@intCast(kk2)][@intCast(kk2)];
                        }
                    }
                    self.n[chn] = 0;
                }
                var sum: f80 = 0;
                var l: i32 = 1;
                while (l <= Sc + Dc) : (l += 1) {
                    const xv = if (l <= Sc) X1(l) else X2(l - Sc);
                    sum += @as(f80, self.F[@intCast(l)][0][chn] * @as(f64, @floatFromInt(xv)));
                }
                self.pr[2][chn] = self.pr[1][chn];
                self.pr[1][chn] = self.pr[0][chn];
                self.pr[0][chn] = floorToI32(@floatCast(sum));
                self.counter[chn] += 1;
            }
            const y1 = self.pr[0][chn];
            const y2 = self.pr[1][chn];
            const y3 = self.pr[2][chn];
            var x1 = bget(1);
            var x2 = bget(2);
            const x3 = bget(3);
            if (wmode == 4 or wmode == 5) {
                x1 ^= 128;
                x2 ^= 128;
            }
            if (self.bits == 8) {
                x1 -= 128;
                x2 -= 128;
            }
            const t: i32 = @intFromBool(self.bits == 8 or ((@as(i32, @intFromBool(msb == 0)) ^ @as(i32, @intFromBool(wmode < 6))) != 0));
            const z1 = self.z1;
            const z2 = self.z2;
            const z6 = self.z6;
            var i: i32 = self.ch << 4;
            if ((@as(i32, @intFromBool(msb != 0)) ^ @as(i32, @intFromBool(wmode < 6))) != 0) {
                i += 1;
                self.cm.set(maps.hash(.{ i, y1 & 0xff }));
                i += 1;
                self.cm.set(maps.hash(.{ i, y1 & 0xff, ((z1 - y2 + z2 - y3) >> 1) & 0xff }));
                i += 1;
                self.cm.set(maps.hash(.{ i, x1, y1 & 0xff }));
                i += 1;
                self.cm.set(maps.hash(.{ i, x1, x2 >> 3, x3 }));
                if (self.bits == 8) {
                    i += 1;
                    self.cm.set(maps.hash(.{ i, y1 & 0xFE, @as(i32, @intCast(maps.ilog2(@intCast(@abs(z1 - y2))))) * 2 + @as(i32, @intFromBool(z1 > y2)) }));
                } else {
                    i += 1;
                    self.cm.set(maps.hash(.{ i, (y1 + z1 - y2) & 0xff }));
                }
                i += 1;
                self.cm.set(maps.hash(.{ i, x1 }));
                i += 1;
                self.cm.set(maps.hash(.{ i, x1, x2 }));
                i += 1;
                self.cm.set(maps.hash(.{ i, z1 & 0xff }));
                i += 1;
                self.cm.set(maps.hash(.{ i, (z1 * 2 - z2) & 0xff }));
                i += 1;
                self.cm.set(maps.hash(.{ i, z6 & 0xff }));
                i += 1;
                self.cm.set(maps.hash(.{ i, y1 & 0xFF, @divTrunc(z1 - y2 + z2 - y3, self.bits >> 3) & 0xFF }));
            } else {
                i += 1;
                self.cm.set(maps.hash(.{ i, (y1 - x1 + z1 - y2) >> 8 }));
                i += 1;
                self.cm.set(maps.hash(.{ i, (y1 - x1) >> 8 }));
                i += 1;
                self.cm.set(maps.hash(.{ i, (y1 - x1 + z1 * 2 - y2 * 2 - z2 + y3) >> 8 }));
                i += 1;
                self.cm.set(maps.hash(.{ i, (y1 - x1) >> 8, (z1 - y2 + z2 - y3) >> 9 }));
                i += 1;
                self.cm.set(maps.hash(.{ i, z1 >> 12 }));
                i += 1;
                self.cm.set(maps.hash(.{ i, x1 }));
                i += 1;
                self.cm.set(maps.hash(.{ i, x1 >> 7, x2, x3 >> 7 }));
                i += 1;
                self.cm.set(maps.hash(.{ i, z1 >> 8 }));
                i += 1;
                self.cm.set(maps.hash(.{ i, (z1 * 2 - z2) >> 8 }));
                i += 1;
                self.cm.set(maps.hash(.{ i, y1 >> 8 }));
                i += 1;
                self.cm.set(maps.hash(.{ i, (y1 - x1) >> 6 }));
            }
            self.scm1.set(@bitCast(t * self.ch));
            self.scm2.set(@bitCast((t * ((z1 - x1 + y1) >> 9)) & 0xff));
            self.scm3.set(@bitCast((t * ((z1 * 2 - z2 - x1 + y1) >> 8)) & 0xff));
            self.scm4.set(@bitCast((t * ((z1 * 3 - z2 * 3 + self.z3 - x1) >> 7)) & 0xff));
            self.scm5.set(@bitCast((t * ((z1 + self.z7 - x1 + y1 * 2) >> 10)) & 0xff));
            self.scm6.set(@bitCast((t * ((z1 * 4 - z2 * 6 + self.z3 * 4 - self.z4 - x1) >> 7)) & 0xff));
            self.scm7.set(@bitCast((t * ((z1 * 5 - z2 * 10 + self.z3 * 10 - self.z4 * 5 + self.z5 - x1 + y1) >> 9)) & 0xff));
        }

        self.scm1.mix(m, 7, 1, 4);
        self.scm2.mix(m, 7, 1, 4);
        self.scm3.mix(m, 7, 1, 4);
        self.scm4.mix(m, 7, 1, 4);
        self.scm5.mix(m, 7, 1, 4);
        self.scm6.mix(m, 7, 1, 4);
        self.scm7.mix(m, 7, 1, 4);
        _ = self.cm.mix(m);
        if (stats) |st| st.Record = (@as(u32, @bitCast(self.w)) << 16) | (st.Record & 0xFFFF);
        self.col += 1;
        if (self.col >= self.w * 8) self.col = 0;
        m.set(self.ch + 4 * @as(i32, @intCast(maps.ilog2(@intCast(self.col & (self.bits - 1))))), 4 * 8);
        m.set(@intFromBool(@mod(self.col, self.bits) < 8), 2);
        m.set(@mod(self.col, self.bits), self.bits);
        m.set(self.col, self.w * 8);
        m.set(core.c0, 256);
    }
};

// ================================ audioModel ===============================
const WAVAudio = struct {
    Header: u32 = 0,
    Size: u32 = 0,
    Channels: u32 = 0,
    BitsPerSample: u32 = 0,
    Chunk: u32 = 0,
    Data: u32 = 0,
};

/// recordModel is a separate paq8 model (not in this file's scope). The
/// integrator passes a pointer to their ported recordModel so audioModel can
/// invoke it exactly as the C++ dispatch does.
pub const RecordModelFn = *const fn (m: *Mixer, filetype: i32, stats: ?*core.ModelStats) void;

pub const AudioModel = struct {
    audio8: Audio8b,
    wav: Wav,
    eoi: u32 = 0,
    length: u32 = 0,
    info: u32 = 0,
    WAV: WAVAudio = .{},
    alloc: Allocator,

    pub fn init(a: Allocator) AudioModel {
        return .{ .audio8 = Audio8b.init(a), .wav = Wav.init(a), .alloc = a };
    }

    pub fn deinit(self: *AudioModel) void {
        self.audio8.deinit();
        self.wav.deinit();
    }

    pub fn audioModel(self: *AudioModel, m: *Mixer, stats: ?*core.ModelStats, record_model_fn: ?RecordModelFn) i32 {
        const pos = core.pos;
        if (core.bpos == 0) {
            if (pos >= @as(i32, @bitCast(self.eoi +% 4)) and self.WAV.Header == 0 and m4(4) == 0x52494646) {
                self.WAV.Header = @intCast(pos);
                self.WAV.Chunk = 0;
                self.length = 0;
            } else if (self.WAV.Header != 0) {
                const p: i32 = pos - @as(i32, @bitCast(self.WAV.Header));
                if (p == 4) {
                    self.WAV.Size = rd4(4);
                    self.WAV.Header *%= @intFromBool(self.WAV.Size <= 0x3FFFFFFF);
                } else if (p == 8) {
                    self.WAV.Header *%= @intFromBool(m4(4) == 0x57415645);
                } else if (p == @as(i32, @bitCast(16 +% self.length))) {
                    var cond = false;
                    if (m4(8) != 0x666d7420) {
                        cond = true;
                    } else {
                        self.WAV.Chunk = rd4(4) -% 16;
                        cond = (self.WAV.Chunk & 0xFFFFFFFD) != 0;
                    }
                    if (cond) {
                        self.length = ((rd4(4) +% 1) & 0xFFFFFFFE) +% 8;
                        self.WAV.Header *%= @intFromBool(!(m4(8) == 0x666d7420 and (rd4(4) & 0xFFFFFFFD) != 16));
                    }
                } else if (p == @as(i32, @bitCast(20 +% self.length))) {
                    self.WAV.Channels = @intCast(bget(2));
                    self.WAV.Header *%= @intFromBool((self.WAV.Channels == 1 or self.WAV.Channels == 2) and (m4(4) & 0xFFFFFCFF) == 0x01000000);
                } else if (p == @as(i32, @bitCast(32 +% self.length))) {
                    self.WAV.BitsPerSample = @intCast(bget(2));
                    self.WAV.Header *%= @intFromBool((self.WAV.BitsPerSample == 8 or self.WAV.BitsPerSample == 16) and (m2(2) & 0xE7FF) == 0);
                } else if (p == @as(i32, @bitCast(40 +% self.length +% self.WAV.Chunk)) and m4(8) != 0x64617461) {
                    self.WAV.Chunk +%= ((rd4(4) +% 1) & 0xFFFFFFFE) +% 8;
                    self.WAV.Header *%= @intFromBool(self.WAV.Chunk <= 0xFFFFF);
                } else if (p == @as(i32, @bitCast(40 +% self.length +% self.WAV.Chunk))) {
                    self.WAV.Data = (rd4(4) +% 1) & 0xFFFFFFFE;
                    if (self.WAV.Data != 0 and @mod(self.WAV.Data, self.WAV.Channels *% (self.WAV.BitsPerSample / 8)) == 0) {
                        self.info = (self.WAV.Channels +% self.WAV.BitsPerSample / 4 -% 3) +% 1;
                        self.eoi = @as(u32, @intCast(pos)) +% self.WAV.Data;
                    }
                }
            }
        }

        if (pos > @as(i32, @bitCast(self.eoi))) {
            self.info = 0;
            return 0;
        }

        if (self.info != 0) {
            const infom1: i32 = @bitCast(self.info -% 1);
            if ((infom1 & 2) == 0) {
                self.audio8.audio8bModel(m, infom1, stats);
            } else {
                self.wav.wavModel(m, infom1, stats);
            }
            if (record_model_fn) |rf| rf(m, core.FT_AUDIO, stats);
        }

        if (core.bpos == 7 and (pos + 1) == @as(i32, @bitCast(self.eoi))) {
            self.WAV = .{};
        }
        return @bitCast(self.info);
    }
};

// IntBuf (paq8 helper). Only used by jpegModel (out of scope for this file);
// provided for completeness.
pub const IntBuf = struct {
    b: core.Array(i32, 0),
    pub fn init(a: Allocator, i: u32) IntBuf {
        return .{ .b = core.Array(i32, 0).initSize(a, if (i == 0) 1 else i) };
    }
    pub fn at(self: *IntBuf, i: i32) *i32 {
        return self.b.at(@as(u32, @bitCast(i)) & (self.b.size() - 1));
    }
    pub fn deinit(self: *IntBuf) void {
        self.b.deinit();
    }
};
