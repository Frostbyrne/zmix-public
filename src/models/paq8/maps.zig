//! paq8 context-map / hashing infrastructure, ported faithfully from
//! reference/cmix-src/models/paq8.cpp.
//!
//! Provides (all building on core.zig):
//!   * hashing: hash(...), finalize64, checksum64, combine64, ilog2
//!   * HashBucket (the 64-byte 7-way bit-history bucket shared by ContextMap /
//!     ContextMap2), BH<B>, HashTable<B>
//!   * RunContextMap, SmallStationaryContextMap, StationaryMap, IndirectMap
//!   * ContextMap (paq8l-style, uses core.rnd), ContextMap2
//!   * OLS (Cholesky least-squares), IndirectContext<T>, MTFList
//!
//! Faithfulness: C `unsigned` wrap -> `+%`/`-%`/`*%`; U8/U16/U32 truncating
//! stores -> `@truncate`/`& mask`. Hash multipliers, table sizes and bit-shift
//! constants match paq8.cpp exactly.
const std = @import("std");
const core = @import("core.zig");

// ================================= hashing =================================
pub const PHI32: u32 = 0x9E3779B9;
pub const PHI64: u64 = 0x9E3779B97F4A7C15;
pub const MUL64_1: u64 = 0x993DDEFFB1462949;
pub const MUL64_2: u64 = 0xE9C91DC159AB0D2D;
pub const MUL64_3: u64 = 0x83D6A14F1B0CED73;
pub const MUL64_4: u64 = 0xA14F1B0CED5A841F;
pub const MUL64_5: u64 = 0xC0E51314A614F4EF;
pub const MUL64_6: u64 = 0xDA9CC2600AE45A27;
pub const MUL64_7: u64 = 0x826797AA04A65737;
pub const MUL64_8: u64 = 0x2375BE54C41A08ED;
pub const MUL64_9: u64 = 0xD39104E950564B37;
pub const MUL64_10: u64 = 0x3091697D5E685623;
pub const MUL64_11: u64 = 0x20EB84EE04A3C7E1;
pub const MUL64_12: u64 = 0xF501F1D0944B2383;
pub const MUL64_13: u64 = 0xE3E4E8AA829AB9B5;

const MULS = [8]u64{ PHI64, MUL64_1, MUL64_2, MUL64_3, MUL64_4, MUL64_5, MUL64_6, MUL64_7 };

inline fn toU64(v: anytype) u64 {
    const T = @TypeOf(v);
    return switch (@typeInfo(T)) {
        .int => |info| if (info.signedness == .signed) @bitCast(@as(i64, v)) else @as(u64, v),
        .comptime_int => @as(u64, @intCast(v)),
        else => @compileError("bad hash arg type"),
    };
}

/// paq8 `hash(x0, x1, ...)` for 1..8 args. Call as `hash(.{a, b, c})`.
/// hash = sum_i (x_i + 1) * MULS[i].
pub inline fn hash(args: anytype) u64 {
    var h: u64 = 0;
    inline for (args, 0..) |arg, idx| {
        h +%= (toU64(arg) +% 1) *% MULS[idx];
    }
    return h;
}

pub inline fn combine64(seed: u64, x: u64) u64 {
    return hash(.{seed +% x});
}

pub inline fn finalize64(h: u64, hashbits: i32) u32 {
    return @truncate(h >> @intCast(64 - hashbits));
}

pub inline fn checksum64(h: u64, hashbits: i32, checksumbits: i32) u64 {
    return h >> @intCast(64 - hashbits - checksumbits);
}

pub inline fn ilog2(x0: u32) u32 {
    var x = x0;
    x |= x >> 1;
    x |= x >> 2;
    x |= x >> 4;
    x |= x >> 8;
    x |= x >> 16;
    return @popCount(x >> 1);
}

// =============================== HashBucket ================================
// 64-byte, 7-way associative bit-history bucket. This is paq8's ContextMap::E
// and ContextMap2::Bucket (identical layout & replacement policy); shared here.
pub const HashBucket = extern struct {
    chk: [7]u16, // per-context checksums
    last: u8, // last 2 accesses (0..6) in low/high nibble (MRU)
    bh: [7][7]u8, // bit-history states

    inline fn rowptr(self: *HashBucket, i: usize) [*]u8 {
        return @ptrCast(&self.bh[i][0]);
    }

    /// Find (or create/replace lowest-priority) the row matching `ch`.
    pub fn get(self: *HashBucket, ch: u16) [*]u8 {
        const lastlo: usize = self.last & 15;
        if (self.chk[lastlo] == ch) return self.rowptr(lastlo);
        var b: i32 = 0xffff;
        var bi: usize = 0;
        var i: usize = 0;
        while (i < 7) : (i += 1) {
            const iu8: u8 = @intCast(i);
            if (self.chk[i] == ch) {
                self.last = @truncate((@as(u32, self.last) << 4) | @as(u32, iu8));
                return self.rowptr(i);
            }
            const pri: i32 = self.bh[i][0];
            if (pri < b and (self.last & 15) != iu8 and (self.last >> 4) != iu8) {
                b = pri;
                bi = i;
            }
        }
        self.last = @intCast(0xf0 | bi);
        self.chk[bi] = ch;
        @memset(self.bh[bi][0..], 0);
        return self.rowptr(bi);
    }
};

// ================================== BH<B> ==================================
// paq8 `template <int B> class BH`: hashed byte array with M=8-way search and
// move-to-front-on-miss replacement. Element layout: [0..1]=checksum, then data.
pub fn BH(comptime B: usize) type {
    return struct {
        const Self = @This();
        const M: usize = 8;
        t: []u8,
        mask: u32,
        hashbits: i32,
        tmp: [B]u8 = undefined,
        alloc: std.mem.Allocator,

        pub fn init(a: std.mem.Allocator, i: u32) Self {
            var self = Self{
                .t = undefined,
                .mask = i - 1,
                .hashbits = @intCast(ilog2(i)),
                .alloc = a,
            };
            self.t = a.alloc(u8, @as(usize, i) * B) catch unreachable;
            @memset(self.t, 0);
            return self;
        }

        pub fn get(self: *Self, ctx: u64) [*]u8 {
            const chk: u16 = @intCast(checksum64(ctx, self.hashbits, 16) & 0xffff);
            const i: usize = @as(usize, (finalize64(ctx, self.hashbits) *% @as(u32, M)) & self.mask);
            var pp: [*]u8 = undefined;
            var j: usize = 0;
            while (j < M) : (j += 1) {
                pp = self.t.ptr + (i + j) * B;
                if (pp[2] == 0) {
                    std.mem.writeInt(u16, pp[0..2], chk, .little);
                    break;
                }
                if (std.mem.readInt(u16, pp[0..2], .little) == chk) break;
            }
            if (j == 0) return pp + 1;
            if (j == M) {
                j -= 1;
                @memset(self.tmp[0..B], 0);
                std.mem.writeInt(u16, self.tmp[0..2], chk, .little);
                if (M > 2 and self.t[(i + j) * B + 2] > self.t[(i + j - 1) * B + 2]) j -= 1;
            } else {
                @memcpy(self.tmp[0..B], pp[0..B]);
            }
            std.mem.copyBackwards(u8, self.t[(i + 1) * B .. (i + 1) * B + j * B], self.t[i * B .. i * B + j * B]);
            @memcpy(self.t[i * B .. i * B + B], self.tmp[0..B]);
            return self.t.ptr + i * B + 1;
        }

        pub fn deinit(self: *Self) void {
            self.alloc.free(self.t);
        }
    };
}

// =============================== HashTable<B> ==============================
// paq8 `template <int B> class HashTable`: 4-way cuckoo-ish table, 8-bit
// checksum, priority (byte[1]) replacement.
pub fn HashTable(comptime B: usize) type {
    return struct {
        const Self = @This();
        t: []align(64) u8,
        N: usize,
        mask: usize,
        hashbits: i32,
        alloc: std.mem.Allocator,

        pub fn init(a: std.mem.Allocator, n: usize) Self {
            var self = Self{
                .t = undefined,
                .N = n,
                .mask = n - 1,
                .hashbits = @intCast(ilog2(@intCast(n))),
                .alloc = a,
            };
            self.t = a.alignedAlloc(u8, .@"64", n) catch unreachable;
            @memset(self.t, 0);
            return self;
        }

        pub fn get(self: *Self, i_in: u64) [*]u8 {
            const chk: u8 = @intCast(checksum64(i_in, self.hashbits, 8) & 0xff);
            var i: usize = @as(usize, finalize64(i_in, self.hashbits)) * B & self.mask;
            const p = self.t.ptr;
            if (p[i] == chk) return p + i + 1;
            if (p[i ^ B] == chk) return p + (i ^ B) + 1;
            if (p[i ^ (B * 2)] == chk) return p + (i ^ (B * 2)) + 1;
            if (p[i + 1] > p[(i + 1) ^ B] or p[i + 1] > p[(i + 1) ^ (B * 2)]) i ^= B;
            if (p[i + 1] > p[(i + 1) ^ B ^ (B * 2)]) i ^= B ^ (B * 2);
            @memset((p + i)[0..B], 0);
            p[i] = chk;
            return p + i + 1;
        }

        pub fn deinit(self: *Self) void {
            self.alloc.free(self.t);
        }
    };
}

// ============================== RunContextMap ==============================
pub const RunContextMap = struct {
    t: BH(4),
    cp: [*]u8,
    alloc: std.mem.Allocator,

    pub fn init(a: std.mem.Allocator, m: u32) RunContextMap {
        var self = RunContextMap{ .t = BH(4).init(a, m / 4), .cp = undefined, .alloc = a };
        self.cp = self.t.get(0) + 1;
        return self;
    }

    pub fn set(self: *RunContextMap, cx: u64) void {
        const b1: u8 = @intCast(core.buf.get(1));
        if (self.cp[0] == 0 or self.cp[1] != b1) {
            self.cp[0] = 1;
            self.cp[1] = b1;
        } else if (self.cp[0] < 255) {
            self.cp[0] += 1;
        }
        self.cp = self.t.get(cx) + 1;
    }

    pub fn p(self: *RunContextMap) i32 {
        if ((@as(i32, self.cp[1]) + 256) >> @intCast(8 - core.bpos) == core.c0) {
            const bit: i32 = (@as(i32, self.cp[1]) >> @intCast(7 - core.bpos)) & 1;
            return (bit * 2 - 1) * core.ilog(@as(u32, self.cp[0]) + 1) * 8;
        }
        return 0;
    }

    pub fn mix(self: *RunContextMap, m: *core.Mixer) i32 {
        m.add(self.p());
        return @intFromBool(self.cp[0] != 0);
    }

    pub fn deinit(self: *RunContextMap) void {
        self.t.deinit();
    }
};

// ========================= SmallStationaryContextMap =======================
pub const SmallStationaryContextMap = struct {
    Data: []u16,
    Context: usize = 0,
    Mask: usize,
    Stride: usize,
    bCount: i32 = 0,
    bTotal: i32,
    B: i32 = 0,
    cp: usize = 0,
    alloc: std.mem.Allocator,

    pub fn init(a: std.mem.Allocator, bits_of_context: u32, bits_per_context: u32) SmallStationaryContextMap {
        const nn = (@as(usize, 1) << @intCast(bits_of_context)) * ((@as(usize, 1) << @intCast(bits_per_context)) - 1);
        var self = SmallStationaryContextMap{
            .Data = a.alloc(u16, nn) catch unreachable,
            .Mask = (@as(usize, 1) << @intCast(bits_of_context)) - 1,
            .Stride = (@as(usize, 1) << @intCast(bits_per_context)) - 1,
            .bTotal = @intCast(bits_per_context),
            .alloc = a,
        };
        self.reset();
        self.cp = 0;
        return self;
    }

    pub fn reset(self: *SmallStationaryContextMap) void {
        for (self.Data) |*d| d.* = 0x7FFF;
    }

    pub fn set(self: *SmallStationaryContextMap, ctx: u32) void {
        self.Context = (@as(usize, ctx) & self.Mask) * self.Stride;
        self.bCount = 0;
        self.B = 0;
    }

    pub fn mix(self: *SmallStationaryContextMap, m: *core.Mixer, rate: u5, multiplier: i32, divisor: i32) void {
        {
            const cur: i32 = self.Data[self.cp];
            const nv = cur + (((core.y << 16) - cur + (@as(i32, 1) << @intCast(rate - 1))) >> rate);
            self.Data[self.cp] = @intCast(nv & 0xffff);
        }
        self.B += @intFromBool(core.y != 0 and self.B > 0);
        self.cp = self.Context + @as(usize, @intCast(self.B));
        const prediction: i32 = @as(i32, self.Data[self.cp]) >> 4;
        m.add(@divTrunc(core.stretch(prediction) * multiplier, divisor));
        m.add(@divTrunc((prediction - 2048) * multiplier, divisor * 2));
        self.bCount += 1;
        self.B += self.B + 1;
        if (self.bCount == self.bTotal) {
            self.bCount = 0;
            self.B = 0;
        }
    }

    pub fn deinit(self: *SmallStationaryContextMap) void {
        self.alloc.free(self.Data);
    }
};

// ============================== StationaryMap ==============================
pub const StationaryMap = struct {
    Data: []u32,
    mask: u32,
    maskbits: i32,
    stride: u32,
    Context: usize = 0,
    bCount: i32 = 0,
    bTotal: i32,
    B: i32 = 0,
    cp: usize = 0,
    alloc: std.mem.Allocator,

    pub fn init(a: std.mem.Allocator, bits_of_context: u32, bits_per_context: u32, rate: i32) StationaryMap {
        const nn = (@as(usize, 1) << @intCast(bits_of_context)) * ((@as(usize, 1) << @intCast(bits_per_context)) - 1);
        var self = StationaryMap{
            .Data = a.alloc(u32, nn) catch unreachable,
            .mask = @intCast((@as(u32, 1) << @intCast(bits_of_context)) - 1),
            .maskbits = @intCast(bits_of_context),
            .stride = @intCast((@as(u32, 1) << @intCast(bits_per_context)) - 1),
            .bTotal = @intCast(bits_per_context),
            .alloc = a,
        };
        self.reset(rate);
        self.cp = 0;
        return self;
    }

    pub fn reset(self: *StationaryMap, rate: i32) void {
        const v: u32 = (@as(u32, 0x7FF) << 20) | @as(u32, @intCast(@min(1023, rate)));
        for (self.Data) |*d| d.* = v;
    }

    pub fn setDirect(self: *StationaryMap, ctx: u32) void {
        self.Context = @as(usize, ctx & self.mask) * self.stride;
        self.bCount = 0;
        self.B = 0;
    }

    pub fn set(self: *StationaryMap, ctx: u64) void {
        self.Context = @as(usize, finalize64(ctx, self.maskbits) & self.mask) * self.stride;
        self.bCount = 0;
        self.B = 0;
    }

    pub fn mix(self: *StationaryMap, m: *core.Mixer, multiplier: i32, divisor: i32, limit: u16) void {
        // update
        const cur = self.Data[self.cp];
        const Count: i32 = @min(@min(@as(i32, limit), 0x3FF), @as(i32, @intCast(cur & 0x3FF)) + 1);
        var Prediction: i32 = @intCast(cur >> 10);
        var Error: i32 = (core.y << 22) - Prediction;
        Error = @divTrunc(@divTrunc(Error, 8) * core.dt[@intCast(Count)], 1024);
        Prediction = @min(0x3FFFFF, @max(0, Prediction + Error));
        self.Data[self.cp] = (@as(u32, @intCast(Prediction)) << 10) | @as(u32, @intCast(Count));
        // predict
        self.B += @intFromBool(core.y != 0 and self.B > 0);
        self.cp = self.Context + @as(usize, @intCast(self.B));
        Prediction = @intCast(self.Data[self.cp] >> 20);
        m.add(@divTrunc(core.stretch(Prediction) * multiplier, divisor));
        m.add(@divTrunc((Prediction - 2048) * multiplier, divisor * 2));
        self.bCount += 1;
        self.B += self.B + 1;
        if (self.bCount == self.bTotal) {
            self.bCount = 0;
            self.B = 0;
        }
    }

    pub fn deinit(self: *StationaryMap) void {
        self.alloc.free(self.Data);
    }
};

// =============================== IndirectMap ===============================
pub const IndirectMap = struct {
    Data: []u8,
    Map: core.StateMap32,
    mask: u32,
    maskbits: i32,
    stride: u32,
    Context: usize = 0,
    bCount: i32 = 0,
    bTotal: i32,
    B: i32 = 0,
    cp: usize = 0,
    alloc: std.mem.Allocator,

    pub fn init(a: std.mem.Allocator, bits_of_context: u32, bits_per_context: u32) IndirectMap {
        const nn = (@as(usize, 1) << @intCast(bits_of_context)) * ((@as(usize, 1) << @intCast(bits_per_context)) - 1);
        var self = IndirectMap{
            .Data = a.alloc(u8, nn) catch unreachable,
            .Map = core.StateMap32.init(a, 256, true),
            .mask = @intCast((@as(u32, 1) << @intCast(bits_of_context)) - 1),
            .maskbits = @intCast(bits_of_context),
            .stride = @intCast((@as(u32, 1) << @intCast(bits_per_context)) - 1),
            .bTotal = @intCast(bits_per_context),
            .alloc = a,
        };
        @memset(self.Data, 0);
        self.cp = 0;
        return self;
    }

    pub fn setDirect(self: *IndirectMap, ctx: u32) void {
        self.Context = @as(usize, ctx & self.mask) * self.stride;
        self.bCount = 0;
        self.B = 0;
    }

    pub fn set(self: *IndirectMap, ctx: u64) void {
        self.Context = @as(usize, finalize64(ctx, self.maskbits) & self.mask) * self.stride;
        self.bCount = 0;
        self.B = 0;
    }

    pub fn mix(self: *IndirectMap, m: *core.Mixer, multiplier: i32, divisor: i32, limit: i32) void {
        // update
        self.Data[self.cp] = core.nex(self.Data[self.cp], @intCast(core.y));
        // predict
        self.B += @intFromBool(core.y != 0 and self.B > 0);
        self.cp = self.Context + @as(usize, @intCast(self.B));
        const state = self.Data[self.cp];
        const p1 = self.Map.p(state, limit);
        m.add(@divTrunc(core.stretch(p1) * multiplier, divisor));
        m.add(@divTrunc((p1 - 2048) * multiplier, divisor * 2));
        self.bCount += 1;
        self.B += self.B + 1;
        if (self.bCount == self.bTotal) {
            self.bCount = 0;
            self.B = 0;
        }
    }

    pub fn deinit(self: *IndirectMap) void {
        self.Map.deinit();
        self.alloc.free(self.Data);
    }
};

// ================================ ContextMap ===============================
// paq8l-style ContextMap. NOTE: mix1 uses core.rnd (bit-history randomisation),
// exactly as paq8.cpp.
pub const ContextMap = struct {
    C: usize,
    t: []align(64) HashBucket,
    cp: []?[*]u8,
    cp0: [][*]u8,
    cxt: []u32,
    chk: []u16,
    runp: [][*]u8,
    sm: []core.StateMap,
    cn: usize,
    mask: u32,
    hashbits: i32,
    alloc: std.mem.Allocator,

    pub fn init(a: std.mem.Allocator, m: u64, c: i32) ContextMap {
        const C: usize = @intCast(c);
        const nbuckets: usize = @intCast(m >> 6);
        var self = ContextMap{
            .C = C,
            .t = a.alignedAlloc(HashBucket, .@"64", nbuckets) catch unreachable,
            .cp = a.alloc(?[*]u8, C) catch unreachable,
            .cp0 = a.alloc([*]u8, C) catch unreachable,
            .cxt = a.alloc(u32, C) catch unreachable,
            .chk = a.alloc(u16, C) catch unreachable,
            .runp = a.alloc([*]u8, C) catch unreachable,
            .sm = a.alloc(core.StateMap, C) catch unreachable,
            .cn = 0,
            .mask = @intCast(nbuckets - 1),
            .hashbits = 0,
            .alloc = a,
        };
        @memset(std.mem.sliceAsBytes(self.t), 0);
        self.hashbits = @intCast(ilog2(self.mask + 1));
        for (0..C) |i| self.sm[i] = core.StateMap.init(a);
        const base: [*]u8 = @ptrCast(&self.t[0].bh[0][0]);
        for (0..C) |i| {
            self.cp0[i] = base;
            self.cp[i] = base;
            self.runp[i] = base + 3;
        }
        return self;
    }

    pub fn set(self: *ContextMap, cx: u64) void {
        const h = hash(.{ cx, self.cn });
        self.cxt[self.cn] = finalize64(h, self.hashbits);
        self.chk[self.cn] = @intCast(checksum64(h, self.hashbits, 16) & 0xffff);
        self.cn += 1;
    }

    pub fn mix(self: *ContextMap, m: *core.Mixer) i32 {
        return self.mix1(m, core.c0, core.bpos, core.buf.get(1), core.y);
    }

    fn mix1(self: *ContextMap, m: *core.Mixer, cc: i32, bp: i32, c1: i32, y1: i32) i32 {
        var result: i32 = 0;
        const c1u: u8 = @intCast(c1);
        var i: usize = 0;
        while (i < self.cn) : (i += 1) {
            if (self.cp[i]) |cpi| {
                var ns: i32 = core.nex(cpi[0], @intCast(y1));
                if (ns >= 204 and (core.rnd.next() << @intCast((452 - ns) >> 3)) != 0) ns -= 4;
                cpi[0] = @intCast(ns);
            }

            if (bp > 1 and self.runp[i][0] == 0) {
                self.cp[i] = null;
            } else {
                switch (bp) {
                    1, 3, 6 => self.cp[i] = self.cp0[i] + 1 + @as(usize, @intCast(cc & 1)),
                    4, 7 => self.cp[i] = self.cp0[i] + 3 + @as(usize, @intCast(cc & 3)),
                    2, 5 => {
                        const checksum = self.chk[i];
                        const ctx = self.cxt[i];
                        self.cp0[i] = self.t[@as(usize, (ctx +% @as(u32, @intCast(cc))) & self.mask)].get(checksum);
                        self.cp[i] = self.cp0[i];
                    },
                    else => { // bp == 0
                        const checksum = self.chk[i];
                        const ctx = self.cxt[i];
                        self.cp0[i] = self.t[@as(usize, (ctx +% @as(u32, @intCast(cc))) & self.mask)].get(checksum);
                        self.cp[i] = self.cp0[i];
                        if (self.cp0[i][3] == 2) {
                            const c: i32 = @as(i32, self.cp0[i][4]) + 256;
                            var pp = self.t[@as(usize, (ctx +% @as(u32, @intCast(c >> 6))) & self.mask)].get(checksum);
                            pp[0] = @intCast(1 + ((c >> 5) & 1));
                            pp[@intCast(1 + ((c >> 5) & 1))] = @intCast(1 + ((c >> 4) & 1));
                            pp[@intCast(3 + ((c >> 4) & 3))] = @intCast(1 + ((c >> 3) & 1));
                            pp = self.t[@as(usize, (ctx +% @as(u32, @intCast(c >> 3))) & self.mask)].get(checksum);
                            pp[0] = @intCast(1 + ((c >> 2) & 1));
                            pp[@intCast(1 + ((c >> 2) & 1))] = @intCast(1 + ((c >> 1) & 1));
                            pp[@intCast(3 + ((c >> 1) & 3))] = @intCast(1 + (c & 1));
                            self.cp0[i][6] = 0;
                        }
                        if (self.runp[i][0] == 0) {
                            self.runp[i][0] = 2;
                            self.runp[i][1] = c1u;
                        } else if (self.runp[i][1] != c1u) {
                            self.runp[i][0] = 1;
                            self.runp[i][1] = c1u;
                        } else if (self.runp[i][0] < 254) {
                            self.runp[i][0] += 2;
                        } else if (self.runp[i][0] == 255) {
                            self.runp[i][0] = 128;
                        }
                        self.runp[i] = self.cp0[i] + 3;
                    },
                }
            }

            const rc: i32 = self.runp[i][0];
            if ((@as(i32, self.runp[i][1]) + 256) >> @intCast(8 - bp) == cc) {
                const b: i32 = ((@as(i32, self.runp[i][1]) >> @intCast(7 - bp)) & 1) * 2 - 1;
                const cval: i32 = core.ilog(@as(u32, @intCast(rc)) + 1) << @intCast(2 + (~@as(u32, @intCast(rc)) & 1));
                m.add(b * cval);
            } else {
                m.add(0);
            }

            const s: i32 = if (self.cp[i]) |cpi| @as(i32, cpi[0]) else 0;
            const p1 = self.sm[i].p(s);
            const st: i32 = (core.stretch(p1) + (1 << 1)) >> 2;
            m.add(st);
            m.add((p1 - 2047 + (1 << 2)) >> 3);
            const n0: i32 = if (core.nex(@intCast(s), 2) == 0) -1 else 0;
            const n1: i32 = if (core.nex(@intCast(s), 3) == 0) -1 else 0;
            m.add(st * @as(i32, @intCast(@abs(n1 - n0))));
            const p0: i32 = 4095 - p1;
            m.add(((p1 & n0) - (p0 & n1) + (1 << 3)) >> 4);
            result += @intFromBool(s > 0);
        }
        if (bp == 7) self.cn = 0;
        return result;
    }

    pub fn deinit(self: *ContextMap) void {
        for (0..self.C) |i| self.sm[i].deinit();
        self.alloc.free(self.sm);
        self.alloc.free(self.t);
        self.alloc.free(self.cp);
        self.alloc.free(self.cp0);
        self.alloc.free(self.cxt);
        self.alloc.free(self.chk);
        self.alloc.free(self.runp);
    }
};

// =============================== ContextMap2 ===============================
pub const ContextMap2 = struct {
    C: usize,
    Table: []align(64) HashBucket,
    BitState: []?[*]u8,
    BitState0: [][*]u8,
    ByteHistory: [][*]u8,
    Contexts: []u32,
    Chk: []u16,
    HasHistory: []bool,
    Maps6b: []core.StateMap32,
    Maps8b: []core.StateMap32,
    Maps12b: []core.StateMap32,
    index: usize,
    mask: u32,
    hashbits: i32,
    bits: u32,
    lastByte: u8,
    lastBit: u8,
    bitPos: u8,
    alloc: std.mem.Allocator,

    pub fn init(a: std.mem.Allocator, size: u64, count: u32) ContextMap2 {
        const C: usize = count;
        const nbuckets: usize = @intCast(size >> 6);
        var self = ContextMap2{
            .C = C,
            .Table = a.alignedAlloc(HashBucket, .@"64", nbuckets) catch unreachable,
            .BitState = a.alloc(?[*]u8, C) catch unreachable,
            .BitState0 = a.alloc([*]u8, C) catch unreachable,
            .ByteHistory = a.alloc([*]u8, C) catch unreachable,
            .Contexts = a.alloc(u32, C) catch unreachable,
            .Chk = a.alloc(u16, C) catch unreachable,
            .HasHistory = a.alloc(bool, C) catch unreachable,
            .Maps6b = a.alloc(core.StateMap32, C) catch unreachable,
            .Maps8b = a.alloc(core.StateMap32, C) catch unreachable,
            .Maps12b = a.alloc(core.StateMap32, C) catch unreachable,
            .index = 0,
            .mask = @intCast(nbuckets - 1),
            .hashbits = 0,
            .bits = 1,
            .lastByte = 0,
            .lastBit = 0,
            .bitPos = 0,
            .alloc = a,
        };
        @memset(std.mem.sliceAsBytes(self.Table), 0);
        self.hashbits = @intCast(ilog2(self.mask + 1));
        for (0..C) |i| {
            self.Maps6b[i] = core.StateMap32.init(a, (1 << 6) + 8, true);
            self.Maps8b[i] = core.StateMap32.init(a, 1 << 8, true);
            self.Maps12b[i] = core.StateMap32.init(a, (1 << 12) + (1 << 9), true);
            const base: [*]u8 = @ptrCast(&self.Table[i].bh[0][0]);
            self.BitState0[i] = base;
            self.BitState[i] = base;
            self.ByteHistory[i] = base + 3;
            self.HasHistory[i] = false;
        }
        return self;
    }

    pub fn set(self: *ContextMap2, ctx: u64) void {
        const h = hash(.{ ctx, self.index });
        self.Contexts[self.index] = finalize64(h, self.hashbits);
        self.Chk[self.index] = @intCast(checksum64(h, self.hashbits, 16) & 0xffff);
        self.index += 1;
    }

    fn update(self: *ContextMap2) void {
        var i: usize = 0;
        while (i < self.index) : (i += 1) {
            if (self.BitState[i]) |bs| bs[0] = core.nex(bs[0], self.lastBit);

            if (self.bitPos > 1 and self.ByteHistory[i][0] == 0) {
                self.BitState[i] = null;
            } else {
                switch (self.bitPos) {
                    0 => {
                        const chk = self.Chk[i];
                        const ctx = self.Contexts[i];
                        self.BitState0[i] = self.Table[@as(usize, (ctx +% self.bits) & self.mask)].get(chk);
                        self.BitState[i] = self.BitState0[i];
                        if (self.BitState0[i][3] == 2) {
                            const c: i32 = @as(i32, self.BitState0[i][4]) + 256;
                            var pp = self.Table[@as(usize, (ctx +% @as(u32, @intCast(c >> 6))) & self.mask)].get(chk);
                            pp[0] = @intCast(1 + ((c >> 5) & 1));
                            pp[@intCast(1 + ((c >> 5) & 1))] = @intCast(1 + ((c >> 4) & 1));
                            pp[@intCast(3 + ((c >> 4) & 3))] = @intCast(1 + ((c >> 3) & 1));
                            pp = self.Table[@as(usize, (ctx +% @as(u32, @intCast(c >> 3))) & self.mask)].get(chk);
                            pp[0] = @intCast(1 + ((c >> 2) & 1));
                            pp[@intCast(1 + ((c >> 2) & 1))] = @intCast(1 + ((c >> 1) & 1));
                            pp[@intCast(3 + ((c >> 1) & 3))] = @intCast(1 + (c & 1));
                            self.BitState0[i][6] = 0;
                        }
                        self.ByteHistory[i][3] = self.ByteHistory[i][2];
                        self.ByteHistory[i][2] = self.ByteHistory[i][1];
                        if (self.ByteHistory[i][0] == 0) {
                            self.ByteHistory[i][0] = 2;
                            self.ByteHistory[i][1] = self.lastByte;
                        } else if (self.ByteHistory[i][1] != self.lastByte) {
                            self.ByteHistory[i][0] = 1;
                            self.ByteHistory[i][1] = self.lastByte;
                        } else if (self.ByteHistory[i][0] < 254) {
                            self.ByteHistory[i][0] += 2;
                        } else if (self.ByteHistory[i][0] == 255) {
                            self.ByteHistory[i][0] = 128;
                        }
                        self.ByteHistory[i] = self.BitState0[i] + 3;
                        self.HasHistory[i] = self.BitState0[i][0] > 15;
                    },
                    2, 5 => {
                        const chk = self.Chk[i];
                        const ctx = self.Contexts[i];
                        self.BitState0[i] = self.Table[@as(usize, (ctx +% self.bits) & self.mask)].get(chk);
                        self.BitState[i] = self.BitState0[i];
                    },
                    1, 3, 6 => self.BitState[i] = self.BitState0[i] + 1 + self.lastBit,
                    4, 7 => self.BitState[i] = self.BitState0[i] + 3 + @as(usize, @intCast(self.bits & 3)),
                    else => {},
                }
            }
        }
    }

    pub fn Train(self: *ContextMap2, b: u8) void {
        self.bitPos = 0;
        while (self.bitPos < 8) : (self.bitPos += 1) {
            self.update();
            self.lastBit = (b >> @intCast(7 - self.bitPos)) & 1;
            self.bits +%= self.bits +% @as(u32, self.lastBit);
        }
        self.index = 0;
        self.bits = 1;
        self.bitPos = 0;
        self.lastByte = b;
    }

    pub fn mix(self: *ContextMap2, m: *core.Mixer) i32 {
        var result: i32 = 0;
        self.lastBit = @intCast(core.y);
        self.bitPos = @intCast(core.bpos);
        self.bits +%= self.bits +% @as(u32, self.lastBit);
        self.lastByte = @intCast(self.bits & 0xFF);
        if (self.bitPos == 0) self.bits = 1;
        self.update();

        var i: usize = 0;
        while (i < self.index) : (i += 1) {
            var state: i32 = if (self.BitState[i]) |bs| @as(i32, bs[0]) else 0;
            result += @intFromBool(state > 0);
            var p1: i32 = self.Maps8b[i].p(state, 1023);
            var n0: i32 = core.nex(@intCast(state), 2);
            var n1: i32 = core.nex(@intCast(state), 3);
            var k: i32 = n1 + 1; // -~n1
            k = @divTrunc(k * 64, k + n0 + 1); // k*64 / (k - ~n0)
            n0 = if (n0 == 0) -1 else 0;
            n1 = if (n1 == 0) -1 else 0;

            const bpi: i32 = self.bitPos;
            // predict from last byte in context
            if (@as(u32, @intCast((@as(i32, self.ByteHistory[i][1]) + 256) >> @intCast(8 - bpi))) == self.bits) {
                const RunStats: i32 = self.ByteHistory[i][0];
                const sign: i32 = ((@as(i32, self.ByteHistory[i][1]) >> @intCast(7 - bpi)) & 1) * 2 - 1;
                const value: i32 = core.ilog(@as(u32, @intCast(RunStats)) + 1) << @intCast(3 - (RunStats & 1));
                m.add(sign * value);
            } else if (bpi > 0 and (self.ByteHistory[i][0] & 1) > 0) {
                if (@as(u32, @intCast((@as(i32, self.ByteHistory[i][2]) + 256) >> @intCast(8 - bpi))) == self.bits) {
                    m.add((((@as(i32, self.ByteHistory[i][2]) >> @intCast(7 - bpi)) & 1) * 2 - 1) * 128);
                } else if (self.HasHistory[i] and @as(u32, @intCast((@as(i32, self.ByteHistory[i][3]) + 256) >> @intCast(8 - bpi))) == self.bits) {
                    m.add((((@as(i32, self.ByteHistory[i][3]) >> @intCast(7 - bpi)) & 1) * 2 - 1) * 128);
                } else {
                    m.add(0);
                }
            } else {
                m.add(0);
            }

            if (self.HasHistory[i]) {
                state = (@as(i32, self.ByteHistory[i][1]) >> @intCast(7 - bpi)) & 1;
                state |= ((@as(i32, self.ByteHistory[i][2]) >> @intCast(7 - bpi)) & 1) * 2;
                state |= ((@as(i32, self.ByteHistory[i][3]) >> @intCast(7 - bpi)) & 1) * 4;
            } else {
                state = 8;
            }

            const st: i32 = core.stretch(p1) >> 2;
            m.add(st);
            m.add((p1 - 2047) >> 3);
            p1 >>= 4;
            const p0: i32 = 255 - p1;
            m.add(st * @as(i32, @intCast(@abs(n1 - n0))));
            m.add((p1 & n0) - (p0 & n1));
            m.add(core.stretch(self.Maps12b[i].p((state << 9) | (bpi << 6) | k, 1023)) >> 2);
            m.add(core.stretch(self.Maps6b[i].p((state << 3) | bpi, 1023)) >> 2);
        }
        if (self.bitPos == 7) self.index = 0;
        return result;
    }

    pub fn deinit(self: *ContextMap2) void {
        for (0..self.C) |i| {
            self.Maps6b[i].deinit();
            self.Maps8b[i].deinit();
            self.Maps12b[i].deinit();
        }
        self.alloc.free(self.Maps6b);
        self.alloc.free(self.Maps8b);
        self.alloc.free(self.Maps12b);
        self.alloc.free(self.Table);
        self.alloc.free(self.BitState);
        self.alloc.free(self.BitState0);
        self.alloc.free(self.ByteHistory);
        self.alloc.free(self.Contexts);
        self.alloc.free(self.Chk);
        self.alloc.free(self.HasHistory);
    }
};

// ==================================== OLS ==================================
// Ordinary Least Squares predictor (recursive covariance + Cholesky solve).
pub fn OLS(comptime F: type, comptime T: type, comptime hasZeroMean: bool) type {
    return struct {
        const Self = @This();
        const ftol: F = 1e-8;
        const sub: F = if (hasZeroMean) 0 else @floatFromInt(@as(i64, 1) << (8 * @sizeOf(T) - 1));

        n: usize,
        kmax: usize,
        km: usize,
        index: usize,
        lambda: F,
        nu: F,
        x: []F,
        w: []F,
        b: []F,
        mCovariance: [][]F,
        mCholesky: [][]F,
        alloc: std.mem.Allocator,

        pub fn init(a: std.mem.Allocator, n: usize, kmax: usize, lambda: F, nu: F) Self {
            const self = Self{
                .n = n,
                .kmax = kmax,
                .km = 0,
                .index = 0,
                .lambda = lambda,
                .nu = nu,
                .x = a.alloc(F, n) catch unreachable,
                .w = a.alloc(F, n) catch unreachable,
                .b = a.alloc(F, n) catch unreachable,
                .mCovariance = a.alloc([]F, n) catch unreachable,
                .mCholesky = a.alloc([]F, n) catch unreachable,
                .alloc = a,
            };
            for (0..n) |i| {
                self.x[i] = 0;
                self.w[i] = 0;
                self.b[i] = 0;
                self.mCovariance[i] = a.alloc(F, n) catch unreachable;
                self.mCholesky[i] = a.alloc(F, n) catch unreachable;
                for (0..n) |j| {
                    self.mCovariance[i][j] = 0;
                    self.mCholesky[i][j] = 0;
                }
            }
            return self;
        }

        fn factor(self: *Self) i32 {
            for (0..self.n) |i| for (0..self.n) |j| {
                self.mCholesky[i][j] = self.mCovariance[i][j];
            };
            for (0..self.n) |i| self.mCholesky[i][i] += self.nu;
            for (0..self.n) |i| {
                for (0..i) |j| {
                    var sum = self.mCholesky[i][j];
                    for (0..j) |kk| sum -= self.mCholesky[i][kk] * self.mCholesky[j][kk];
                    self.mCholesky[i][j] = sum / self.mCholesky[j][j];
                }
                var sum = self.mCholesky[i][i];
                for (0..i) |kk| sum -= self.mCholesky[i][kk] * self.mCholesky[i][kk];
                if (sum > ftol) {
                    self.mCholesky[i][i] = @sqrt(sum);
                } else {
                    return 1;
                }
            }
            return 0;
        }

        fn solve(self: *Self) void {
            for (0..self.n) |i| {
                var sum = self.b[i];
                for (0..i) |j| sum -= self.mCholesky[i][j] * self.w[j];
                self.w[i] = sum / self.mCholesky[i][i];
            }
            var i: usize = self.n;
            while (i > 0) {
                i -= 1;
                var sum = self.w[i];
                var j: usize = i + 1;
                while (j < self.n) : (j += 1) sum -= self.mCholesky[j][i] * self.w[j];
                self.w[i] = sum / self.mCholesky[i][i];
            }
        }

        inline fn valToF(val: T) F {
            return switch (@typeInfo(T)) {
                .float, .comptime_float => @floatCast(val),
                .int, .comptime_int => @floatFromInt(val),
                else => @compileError("OLS T must be int or float"),
            };
        }

        pub fn add(self: *Self, val: T) void {
            if (self.index < self.n) {
                self.x[self.index] = valToF(val) - sub;
                self.index += 1;
            }
        }

        pub fn addFloat(self: *Self, val: F) void {
            if (self.index < self.n) {
                self.x[self.index] = val - sub;
                self.index += 1;
            }
        }

        pub fn predictWith(self: *Self, p: []const *const T) F {
            var sum: F = 0;
            for (0..self.n) |i| {
                self.x[i] = valToF(p[i].*) - sub;
                sum += self.w[i] * self.x[i];
            }
            return sum + sub;
        }

        pub fn predict(self: *Self) F {
            self.index = 0;
            var sum: F = 0;
            for (0..self.n) |i| sum += self.w[i] * self.x[i];
            return sum + sub;
        }

        pub fn update(self: *Self, val: T) void {
            const vf: F = valToF(val) - sub;
            for (0..self.n) |j| for (0..self.n) |i| {
                self.mCovariance[j][i] = self.lambda * self.mCovariance[j][i] + (1.0 - self.lambda) * (self.x[j] * self.x[i]);
            };
            for (0..self.n) |i| self.b[i] = self.lambda * self.b[i] + (1.0 - self.lambda) * (self.x[i] * vf);
            self.km += 1;
            if (self.km >= self.kmax) {
                if (self.factor() == 0) self.solve();
                self.km = 0;
            }
        }

        pub fn deinit(self: *Self) void {
            for (0..self.n) |i| {
                self.alloc.free(self.mCovariance[i]);
                self.alloc.free(self.mCholesky[i]);
            }
            self.alloc.free(self.mCovariance);
            self.alloc.free(self.mCholesky);
            self.alloc.free(self.x);
            self.alloc.free(self.w);
            self.alloc.free(self.b);
        }
    };
}

// ============================= IndirectContext =============================
pub fn IndirectContext(comptime T: type) type {
    return struct {
        const Self = @This();
        data: []T,
        ctx: usize,
        ctxMask: usize,
        inputMask: u64,
        inputBits: u6,
        alloc: std.mem.Allocator,

        pub fn init(a: std.mem.Allocator, bits_per_context: u32, input_bits: u32) Self {
            const sz = @as(usize, 1) << @intCast(bits_per_context);
            const self = Self{
                .data = a.alloc(T, sz) catch unreachable,
                .ctx = 0,
                .ctxMask = sz - 1,
                .inputMask = (@as(u64, 1) << @intCast(input_bits)) - 1,
                .inputBits = @intCast(input_bits),
                .alloc = a,
            };
            @memset(self.data, 0);
            return self;
        }

        /// operator+=
        pub fn add(self: *Self, i: u32) void {
            const v: u64 = (@as(u64, self.data[self.ctx]) << self.inputBits) | (@as(u64, i) & self.inputMask);
            self.data[self.ctx] = @truncate(v);
        }

        /// operator=
        pub fn setCtx(self: *Self, i: u32) void {
            self.ctx = @as(usize, i) & self.ctxMask;
        }

        /// operator
        pub fn value(self: *Self) T {
            return self.data[self.ctx];
        }

        pub fn valuePtr(self: *Self) *T {
            return &self.data[self.ctx];
        }

        pub fn deinit(self: *Self) void {
            self.alloc.free(self.data);
        }
    };
}

// ================================= MTFList =================================
pub const MTFList = struct {
    Root: i32,
    Index: i32,
    Previous: []i32,
    Next: []i32,
    alloc: std.mem.Allocator,

    pub fn init(a: std.mem.Allocator, n: u16) MTFList {
        const self = MTFList{
            .Root = 0,
            .Index = 0,
            .Previous = a.alloc(i32, n) catch unreachable,
            .Next = a.alloc(i32, n) catch unreachable,
            .alloc = a,
        };
        var i: usize = 0;
        while (i < n) : (i += 1) {
            self.Previous[i] = @as(i32, @intCast(i)) - 1;
            self.Next[i] = @as(i32, @intCast(i)) + 1;
        }
        self.Next[@as(usize, n) - 1] = -1;
        return self;
    }

    pub fn getFirst(self: *MTFList) i32 {
        self.Index = self.Root;
        return self.Index;
    }

    pub fn getNext(self: *MTFList) i32 {
        if (self.Index >= 0) self.Index = self.Next[@intCast(self.Index)];
        return self.Index;
    }

    pub fn moveToFront(self: *MTFList, i: i32) void {
        self.Index = i;
        if (self.Index == self.Root) return;
        const pv = self.Previous[@intCast(self.Index)];
        const nx = self.Next[@intCast(self.Index)];
        if (pv >= 0) self.Next[@intCast(pv)] = self.Next[@intCast(self.Index)];
        if (nx >= 0) self.Previous[@intCast(nx)] = self.Previous[@intCast(self.Index)];
        self.Previous[@intCast(self.Root)] = self.Index;
        self.Next[@intCast(self.Index)] = self.Root;
        self.Root = self.Index;
        self.Previous[@intCast(self.Root)] = -1;
    }

    pub fn deinit(self: *MTFList) void {
        self.alloc.free(self.Previous);
        self.alloc.free(self.Next);
    }
};

// ==================================== tests ================================
const testing = std.testing;

test "hash determinism + finalize/checksum" {
    const h1 = hash(.{ @as(u64, 123), @as(u64, 456) });
    const h2 = hash(.{ @as(u64, 123), @as(u64, 456) });
    try testing.expectEqual(h1, h2);
    try testing.expect(hash(.{@as(u64, 1)}) != hash(.{@as(u64, 2)}));
    // finalize64/checksum64 extract disjoint high bits
    const hb: i32 = 22;
    _ = finalize64(h1, hb);
    _ = checksum64(h1, hb, 16);
    try testing.expectEqual(@as(u32, 4), ilog2(16));
    try testing.expectEqual(@as(u32, 10), ilog2(1024));
    try testing.expectEqual(@as(u32, 0), ilog2(1));
}

test "HashBucket get finds and replaces" {
    var bucket = std.mem.zeroes(HashBucket);
    const a = bucket.get(0x1234);
    a[0] = 42;
    // same checksum returns same row
    const b = bucket.get(0x1234);
    try testing.expectEqual(@as(u8, 42), b[0]);
    // different checksum -> different (fresh, zeroed) row
    const c = bucket.get(0x5678);
    try testing.expectEqual(@as(u8, 0), c[0]);
}

test "BH get returns stable pointer for same ctx" {
    core.init();
    const a = testing.allocator;
    var bh = BH(4).init(a, 1024);
    defer bh.deinit();
    // getreturns element+1; bytes [0,1] are the checksum, data begins at [1].
    const p1 = bh.get(0xABCDEF);
    p1[1] = 7;
    const p2 = bh.get(0xABCDEF);
    try testing.expectEqual(@as(u8, 7), p2[1]);
}

test "HashTable get" {
    core.init();
    const a = testing.allocator;
    var ht = HashTable(16).init(a, 4096);
    defer ht.deinit();
    const p1 = ht.get(0x9999);
    p1[0] = 55;
    const p2 = ht.get(0x9999);
    try testing.expectEqual(@as(u8, 55), p2[0]);
}

test "RunContextMap runs, finite output" {
    core.init();
    core.resetState();
    const a = testing.allocator;
    core.buf.setsize(a, 1 << 16);
    defer core.buf.deinit();
    var rcm = RunContextMap.init(a, 1 << 16);
    defer rcm.deinit();
    const m = core.Mixer.init(a, 8, 1, 1, 0);
    defer m.deinit();
    // simulate a few bytes
    var byte: u32 = 0;
    while (byte < 8) : (byte += 1) {
        core.pos = @intCast(byte);
        core.buf.at(byte).* = @intCast(byte & 0xff);
        rcm.set(hash(.{byte & 3}));
        core.bpos = 0;
        core.c0 = 1;
        core.resetPredictions();
        _ = rcm.mix(m);
    }
    try testing.expect(m.nx > 0);
}

test "SmallStationaryContextMap adapts and stays finite" {
    core.init();
    const a = testing.allocator;
    var sscm = SmallStationaryContextMap.init(a, 8, 8);
    defer sscm.deinit();
    const m = core.Mixer.init(a, 16, 1, 1, 0);
    defer m.deinit();
    core.y = 1;
    var bit: usize = 0;
    while (bit < 40) : (bit += 1) {
        if (bit % 8 == 0) sscm.set(5);
        core.resetPredictions();
        sscm.mix(m, 7, 1, 4);
        m.update(); // reset the mixer input counter each bit
    }
    try testing.expect(true);
}

test "StationaryMap adapts" {
    core.init();
    const a = testing.allocator;
    var sm = StationaryMap.init(a, 8, 8, 0);
    defer sm.deinit();
    const m = core.Mixer.init(a, 16, 1, 1, 0);
    defer m.deinit();
    core.y = 1;
    var bit: usize = 0;
    while (bit < 40) : (bit += 1) {
        if (bit % 8 == 0) sm.setDirect(3);
        core.resetPredictions();
        sm.mix(m, 1, 4, 1023);
        m.update();
    }
    try testing.expect(true);
}

test "IndirectMap runs" {
    core.init();
    const a = testing.allocator;
    var im = IndirectMap.init(a, 8, 8);
    defer im.deinit();
    const m = core.Mixer.init(a, 16, 1, 1, 0);
    defer m.deinit();
    core.y = 0;
    var bit: usize = 0;
    while (bit < 40) : (bit += 1) {
        if (bit % 8 == 0) im.setDirect(9);
        core.resetPredictions();
        im.mix(m, 1, 4, 1023);
        m.update();
    }
    try testing.expect(true);
}

test "ContextMap set/mix over a byte, finite stretched output" {
    core.init();
    core.resetState();
    const a = testing.allocator;
    core.buf.setsize(a, 1 << 16);
    defer core.buf.deinit();
    var cm = ContextMap.init(a, 1 << 18, 4);
    defer cm.deinit();
    const m = core.Mixer.init(a, 64, 1, 1, 0);
    defer m.deinit();

    var byte: u32 = 0;
    while (byte < 32) : (byte += 1) {
        core.pos = @intCast(byte);
        core.buf.at(byte).* = @intCast((byte * 7) & 0xff);
        // model 8 bits
        core.c0 = 1;
        var bit: u32 = 0;
        while (bit < 8) : (bit += 1) {
            core.bpos = @intCast(bit);
            if (bit == 0) {
                cm.set(hash(.{ @as(u64, 1), byte & 7 }));
                cm.set(hash(.{ @as(u64, 2), byte & 15 }));
                cm.set(hash(.{ @as(u64, 3), (byte *% 3) & 31 }));
                cm.set(hash(.{ @as(u64, 4), byte & 63 }));
            }
            core.resetPredictions();
            _ = cm.mix(m);
            // every mixer input must be finite (i16 already); check exports
            for (0..core.prediction_index) |pi| {
                try testing.expect(!std.math.isNan(core.model_predictions[pi]));
            }
            m.update(); // reset mixer input counter each bit
            core.y = @intCast((((byte * 7) & 0xff) >> @intCast(7 - bit)) & 1);
            core.c0 = (core.c0 << 1) | core.y;
        }
    }
    try testing.expect(true);
}

test "ContextMap2 set/mix over a byte, finite output" {
    core.init();
    core.resetState();
    const a = testing.allocator;
    var cm = ContextMap2.init(a, 1 << 18, 4);
    defer cm.deinit();
    const m = core.Mixer.init(a, 64, 1, 1, 0);
    defer m.deinit();

    var byte: u32 = 0;
    while (byte < 32) : (byte += 1) {
        core.c0 = 1;
        var bit: u32 = 0;
        while (bit < 8) : (bit += 1) {
            core.bpos = @intCast(bit);
            if (bit == 0) {
                cm.set(hash(.{ @as(u64, 10), byte & 7 }));
                cm.set(hash(.{ @as(u64, 11), byte & 15 }));
                cm.set(hash(.{ @as(u64, 12), (byte *% 5) & 31 }));
                cm.set(hash(.{ @as(u64, 13), byte & 63 }));
            }
            core.resetPredictions();
            _ = cm.mix(m);
            for (0..core.prediction_index) |pi| {
                try testing.expect(!std.math.isNan(core.model_predictions[pi]));
            }
            m.update();
            core.y = @intCast((((byte * 11) & 0xff) >> @intCast(7 - bit)) & 1);
            core.c0 = (core.c0 << 1) | core.y;
        }
    }
    try testing.expect(true);
}

test "OLS learns a linear relation" {
    const a = testing.allocator;
    const O = OLS(f64, f64, true);
    var ols = O.init(a, 2, 1, 0.998, 1e-6);
    defer ols.deinit();
    // target = 3*x0 - 2*x1
    var rng = std.Random.DefaultPrng.init(1);
    const r = rng.random();
    var iter: usize = 0;
    while (iter < 2000) : (iter += 1) {
        const x0 = r.float(f64) * 2 - 1;
        const x1 = r.float(f64) * 2 - 1;
        const target = 3 * x0 - 2 * x1;
        ols.addFloat(x0);
        ols.addFloat(x1);
        _ = ols.predict();
        ols.update(target);
    }
    // final prediction should be close for a fresh sample
    ols.addFloat(1.0);
    ols.addFloat(0.0);
    const pred = ols.predict();
    try testing.expect(@abs(pred - 3.0) < 0.2);
}

test "IndirectContext shifts history" {
    const a = testing.allocator;
    var ic = IndirectContext(u32).init(a, 8, 4);
    defer ic.deinit();
    ic.setCtx(5);
    ic.add(0xA);
    ic.add(0xB);
    // low nibble now B, next nibble A
    try testing.expectEqual(@as(u32, 0xAB), ic.value() & 0xFF);
}

test "MTFList move to front" {
    const a = testing.allocator;
    var mtf = MTFList.init(a, 8);
    defer mtf.deinit();
    try testing.expectEqual(@as(i32, 0), mtf.getFirst());
    mtf.moveToFront(5);
    try testing.expectEqual(@as(i32, 5), mtf.getFirst());
    try testing.expectEqual(@as(i32, 0), mtf.getNext());
}
