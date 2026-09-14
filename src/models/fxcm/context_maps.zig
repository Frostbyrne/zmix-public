//! fxcm_v26 context maps, ported from reference/fxcmv1_v26.cpp.
//!   RunContextMap, SmallStationaryContextMap, ContextMap (the big one), APM.
const std = @import("std");
const prim = @import("primitives.zig");
const st = @import("state.zig");
const core = @import("core.zig");

const squash = prim.squash;
const stretch = prim.stretch;
const clp = prim.clp;

inline fn sc(p: i32) i32 {
    if (p > 0) return p >> 7;
    return (p + 127) >> 7;
}

pub inline fn getStateByteLocation(bpos: i32, c0: i32) u32 {
    const smask: u32 = (@as(u32, 0x31031010) >> @intCast(bpos << 2)) & 0x0F;
    return smask + (@as(u32, @intCast(c0)) & smask);
}

pub const RunContextMap = struct {
    t: []align(64) u8,
    cp: [*]u8,
    rc: [512]i16 = undefined,
    tmp: [4]u8 = .{0} ** 4,
    n: u32,
    alloc: std.mem.Allocator,
    const B: usize = 4;
    const M: u32 = 4;

    pub fn init(self: *RunContextMap, a: std.mem.Allocator, m: usize, rcm_ml: i32) void {
        self.alloc = a;
        self.t = a.alignedAlloc(u8, .@"64", m) catch unreachable;
        @memset(self.t, 0);
        self.n = @intCast(m / B - 1);
        self.cp = self.t.ptr + 1;
        for (0..256) |r| {
            var c: i32 = @as(i32, prim.ilog[r]) * 8;
            if ((r & 1) == 0) c = @divTrunc(c * rcm_ml, 4);
            self.rc[r + 256] = clp(c);
            self.rc[r] = clp(-c);
        }
    }

    fn find(self: *RunContextMap, i_in: u32) [*]u8 {
        const chk: u16 = @intCast((i_in >> 16 ^ i_in) & 0xffff);
        const i: u32 = i_in *% M & self.n;
        var pp: [*]u8 = undefined;
        var j: usize = 0;
        var cp1: *align(1) u16 = undefined;
        while (j < M) : (j += 1) {
            pp = self.t.ptr + (i + j) * B;
            cp1 = @ptrCast(pp);
            if (pp[2] == 0) {
                cp1.* = chk;
                break;
            }
            if (cp1.* == chk) break;
        }
        if (j == 0) return pp + 1;
        if (j == M) {
            j -= 1;
            @memset(self.tmp[0..B], 0);
            std.mem.writeInt(u16, self.tmp[0..2], chk, .little);
            if (M > 2 and self.t[(i + j) * B + 2] > self.t[(i + j - 1) * B + 2]) j -= 1;
        } else {
            @memcpy(self.tmp[0..B], @as([*]u8, @ptrCast(cp1))[0..B]);
        }
        // memmove t[(i+1)*B] <- t[i*B], j*B bytes: dest>src overlapping upward
        // move, so copy back-to-front (copyForwards would clobber the overlap).
        std.mem.copyBackwards(u8, self.t[(i + 1) * B .. (i + 1) * B + j * B], self.t[i * B .. i * B + j * B]);
        @memcpy(self.t[i * B .. i * B + B], self.tmp[0..B]);
        return self.t.ptr + i * B + 1;
    }

    pub fn set(self: *RunContextMap, cx: u32, c1: u8) void {
        if (self.cp[0] == 0) {
            self.cp[0] = 2;
            self.cp[1] = c1;
        } else if (self.cp[1] != c1) {
            self.cp[0] = 1;
            self.cp[1] = c1;
        } else if (self.cp[0] < 254) {
            self.cp[0] = self.cp[0] + 2;
        }
        self.cp = self.find(cx) + 1;
    }

    fn p(self: *RunContextMap) i32 {
        const b: i32 = st.x.c0shift_bpos ^ (@as(i32, self.cp[1]) >> @intCast(st.x.bposshift));
        if (b <= 1) return self.rc[@intCast(b * 256 + self.cp[0])];
        return 0;
    }

    pub fn mix(self: *RunContextMap, m: usize) i32 {
        st.x.mxInputs[m].add(self.p());
        return @intFromBool(self.cp[0] != 0);
    }
};

pub const SmallStationaryContextMap = struct {
    data: []u16,
    context: usize = 0,
    mask: usize,
    stride: usize,
    bcount: i32 = 0,
    btotal: i32,
    b: i32 = 0,
    cp: usize = 0,
    alloc: std.mem.Allocator,

    pub fn init(self: *SmallStationaryContextMap, a: std.mem.Allocator, bits_of_context: u32, input_bits: u32) void {
        self.alloc = a;
        self.mask = (@as(usize, 1) << @intCast(bits_of_context)) - 1;
        self.stride = (@as(usize, 1) << @intCast(input_bits)) - 1;
        self.btotal = @intCast(input_bits);
        const nn = (@as(usize, 1) << @intCast(bits_of_context)) * ((@as(usize, 1) << @intCast(input_bits)) - 1);
        self.data = a.alloc(u16, nn) catch unreachable;
        for (self.data) |*d| d.* = 0x7FFF;
        self.cp = 0;
        // These are per-run state; reset them so a reused module-level instance
        // does not inherit the previous predictor's residual (mixreads them
        // before the first seteach byte).
        self.context = 0;
        self.bcount = 0;
        self.b = 0;
    }

    pub fn set(self: *SmallStationaryContextMap, ctx: u32) void {
        self.context = (@as(usize, ctx) & self.mask) * self.stride;
        self.bcount = 0;
        self.b = 0;
    }

    pub fn mix(self: *SmallStationaryContextMap, m: usize) void {
        const rate: u4 = 7;
        // *cp += ((y<<16) - *cp + (1<<(rate-1))) >> rate
        {
            const cur: i32 = self.data[self.cp];
            const nv: i32 = cur + (((st.x.y << 16) - cur + (1 << (rate - 1))) >> rate);
            self.data[self.cp] = @intCast(nv & 0xffff);
        }
        self.b += @intFromBool(st.x.y != 0 and self.b > 0);
        self.cp = self.context + @as(usize, @intCast(self.b));
        const prediction: i32 = @as(i32, self.data[self.cp]) >> 4;
        st.x.mxInputs[m].add(@divTrunc(stretch(prediction) * 1, 4));
        st.x.mxInputs[m].add(@divTrunc((prediction - 2048) * 1, 4 * 2));
        self.bcount += 1;
        self.b += self.b + 1;
        if (self.bcount == self.btotal) {
            self.bcount = 0;
            self.b = 0;
        }
    }
};

pub const APM = struct {
    index: usize = 0,
    t: []u16,
    alloc: std.mem.Allocator,

    pub fn init(self: *APM, a: std.mem.Allocator, n: usize) void {
        self.alloc = a;
        self.index = 0;
        self.t = a.alloc(u16, n * 33) catch unreachable;
        for (0..33) |j| self.t[j] = @intCast(squash((@as(i32, @intCast(j)) - 16) * 128) * 16);
        for (33..n * 33) |i| self.t[i] = self.t[i - 33];
    }

    pub fn p(self: *APM, pr_in: i32, cxt: usize, rate: u5, y: i32) i32 {
        const pr = stretch(pr_in);
        const g: i32 = (y << 16) + (y << @intCast(rate)) - y * 2;
        self.t[self.index] = @intCast(@as(i32, self.t[self.index]) + ((g - @as(i32, self.t[self.index])) >> rate));
        self.t[self.index + 1] = @intCast(@as(i32, self.t[self.index + 1]) + ((g - @as(i32, self.t[self.index + 1])) >> rate));
        const w: i32 = pr & 127;
        self.index = @as(usize, @intCast((@as(i32, pr) + 2048) >> 7)) + cxt * 33;
        return (@as(i32, self.t[self.index]) * (128 - w) + @as(i32, self.t[self.index + 1]) * w) >> 11;
    }
};

// Hash element bucket, 64 bytes.
pub const E = extern struct {
    chk: [7]u16,
    last: u8,
    bh: [7][7]u8,

    inline fn rowptr(self: *E, i: usize) [*]u8 {
        return @ptrCast(&self.bh[i][0]);
    }

    pub fn get(self: *E, ch: u16, keep: i32) [*]u8 {
        const lastlo: usize = self.last & 15;
        if (self.chk[lastlo] == ch) return self.rowptr(lastlo);
        var b: i32 = 0xffff;
        var bi: usize = 0;
        var i: usize = 0;
        while (i < 7) : (i += 1) {
            if (self.chk[i] == ch) {
                // ref `last=last<<4|i` stores into U8 `last`, truncating to 8 bits.
                self.last = @intCast(((@as(u32, self.last) << 4) | i) & 0xff);
                return self.rowptr(i);
            }
            const pri: i32 = self.bh[i][0];
            if (pri < b and (self.last & 15) != i and (self.last >> 4) != i) {
                b = pri;
                bi = i;
            }
        }
        self.last = @intCast(((@as(u32, self.last) << 4) | bi | @as(u32, @intCast(keep))) & 0xff);
        self.chk[bi] = ch;
        @memset(&self.bh[bi], 0);
        return self.rowptr(bi);
    }
};

pub const MAXCXT = 8;

pub const ContextMap = struct {
    c: usize = 0,
    cp: [MAXCXT]?[*]u8 = .{null} ** MAXCXT,
    cp0: [MAXCXT][*]u8 = undefined,
    cxt: [MAXCXT]u32 = undefined,
    runp: [MAXCXT][*]u8 = undefined,
    sm: []core.StateMap = undefined,
    cn: usize = 0,
    result: i32 = 0,
    rc1: [512]i16 = undefined,
    st1: [4096]i16 = undefined,
    st2: [4096]i16 = undefined,
    st32: [256]i16 = undefined,
    st8: [256]i16 = undefined,
    cms: i32 = 0,
    cms2: i32 = 0,
    cms3: i32 = 0,
    cms4: i32 = 0,
    kep: i32 = 0,
    nn: [*]const u8 = undefined,
    t: []align(64) E = undefined,
    tmask: u32 = 0,
    skip2: i32 = 0,
    alloc: std.mem.Allocator = undefined,

    inline fn next(self: *ContextMap, i: i32, y: i32) u8 {
        return self.nn[@intCast(y + i * 4)];
    }

    inline fn pre(self: *ContextMap, state: i32) i32 {
        const n0: u32 = @as(u32, self.next(state, 2)) * 3 + 1;
        const n1: u32 = @as(u32, self.next(state, 3)) * 3 + 1;
        return @intCast((n1 << 12) / (n0 + n1));
    }

    pub fn init(self: *ContextMap, a: std.mem.Allocator, m: u32, c: i32, s3: i32, nn1: [*]const u8, cs4: i32, k: i32, u: i32) void {
        self.alloc = a;
        self.c = @intCast(c & 255);
        self.tmask = (m >> 6) - 1;
        self.cn = 0;
        self.result = 0;
        self.kep = k;
        self.t = a.alignedAlloc(E, .@"64", (m >> 6) + 64) catch unreachable;
        @memset(std.mem.sliceAsBytes(self.t), 0);
        self.nn = nn1;
        const cmul: i32 = (c >> 8) & 255;
        self.cms = (c >> 16) & 255;
        self.cms2 = @intCast((@as(u32, @bitCast(c)) >> 24) & 255);
        self.cms4 = cs4;
        self.cms3 = s3;
        self.skip2 = u;
        self.sm = a.alloc(core.StateMap, self.c) catch unreachable;
        for (0..self.c) |i| self.sm[i].init(a, 256, 1023, nn1);
        for (0..self.c) |i| {
            self.cp0[i] = @ptrCast(&self.t[0].bh[0][0]);
            self.cp[i] = self.cp0[i];
            self.runp[i] = self.cp0[i] + 3;
        }
        for (0..256) |rc| {
            var cc: i32 = prim.ilog[rc];
            cc = cc << @intCast(2 + (~@as(u32, @intCast(rc)) & 1));
            if ((rc & 1) == 0) cc = @divTrunc(cc * cmul, 4);
            self.rc1[rc + 256] = clp(cc);
            self.rc1[rc] = clp(-cc);
        }
        for (0..4096) |i| {
            self.st1[i] = clp(sc(self.cms * stretch(@intCast(i))));
            self.st2[i] = clp(sc(self.cms2 * (@as(i32, @intCast(i)) - 2048))) * @as(i16, @intCast(u));
        }
        for (0..256) |s| {
            const si: i32 = @intCast(s);
            const n0: i32 = -@as(i32, @intFromBool(self.next(si, 2) == 0));
            const n1: i32 = -@as(i32, @intFromBool(self.next(si, 3) == 0));
            var r: bool = false;
            var sp0: i32 = 0;
            if ((n1 - n0) == 1) {
                sp0 = 0;
                r = true;
            }
            if ((n1 - n0) == -1) {
                sp0 = 4095;
                r = true;
            }
            if (r) {
                self.st8[s] = clp(sc(self.cms4 * (self.pre(si) - sp0)));
                self.st32[s] = clp(sc(self.cms3 * stretch(self.pre(si))));
                if (s < 8) self.st32[s] = 0;
            } else {
                self.st8[s] = 0;
                self.st32[s] = 0;
            }
        }
    }

    pub fn set(self: *ContextMap, cx_in: u32) void {
        const i = self.cn;
        self.cn += 1;
        var cx = cx_in;
        cx = cx *% 987654323 +% @as(u32, @intCast(i));
        cx = cx << 16 | cx >> 16;
        self.cxt[i] = cx *% 123456791 +% @as(u32, @intCast(i));
    }

    inline fn mix3(self: *ContextMap, y: i32, m: usize, s: i32, sm: *core.StateMap) i32 {
        if (s == 0) {
            st.x.mxInputs[m].add(0);
            if (self.skip2 == 1) st.x.mxInputs[m].add(0);
            st.x.mxInputs[m].add(0);
            st.x.mxInputs[m].add(0);
            st.x.mxInputs[m].add(32 * 2);
            return 0;
        } else {
            sm.set(@intCast(s), y);
            const p1 = sm.pr;
            st.x.mxInputs[m].add(self.st1[@intCast(p1)]);
            if (self.skip2 == 1) st.x.mxInputs[m].add(self.st2[@intCast(p1)]);
            st.x.mxInputs[m].add(self.st8[@intCast(s)]);
            st.x.mxInputs[m].add(self.st32[@intCast(s)]);
            st.x.mxInputs[m].add(0);
            return 1;
        }
    }

    pub fn mix(self: *ContextMap, m: usize) i32 {
        return self.mix1(m, st.x.c0, st.x.bpos, @intCast(st.x.c4 & 0xff), st.x.y);
    }

    /// Escape-byte fast path (ref `mix4`): emit a fixed 6-input, no-update
    /// contribution. Only ever called on skip2==1 maps, whose per-context
    /// export count is 6, so this preserves the deterministic export layout.
    pub fn mix4(self: *ContextMap, m: usize) void {
        _ = self;
        st.x.mxInputs[m].add(0);
        st.x.mxInputs[m].add(0);
        st.x.mxInputs[m].add(0);
        st.x.mxInputs[m].add(0);
        st.x.mxInputs[m].add(32 * 2);
        st.x.mxInputs[m].add(0);
    }

    fn mix1(self: *ContextMap, m: usize, cc: i32, bp: i32, c1: i32, y1: i32) i32 {
        self.result = 0;
        var i: usize = 0;
        while (i < self.cn) : (i += 1) {
            if (self.cp[i]) |cpi| {
                cpi[0] = self.next(cpi[0], y1);
            }
            var s: i32 = 0;
            if (bp > 1 and self.runp[i][0] == 0) {
                self.cp[i] = null;
            } else {
                const chksum: u16 = @intCast((self.cxt[i] >> 16) ^ @as(u32, @intCast(i)));
                if (bp != 0) {
                    if (bp == 2 or bp == 5) {
                        self.cp0[i] = self.t[(self.cxt[i] +% @as(u32, @intCast(cc))) & self.tmask].get(chksum, self.kep);
                        self.cp[i] = self.cp0[i];
                    } else {
                        self.cp[i] = self.cp0[i] + getStateByteLocation(bp, cc);
                    }
                } else {
                    self.cp0[i] = self.t[(self.cxt[i] +% @as(u32, @intCast(cc))) & self.tmask].get(chksum, self.kep);
                    self.cp[i] = self.cp0[i];
                    if (self.cp0[i][3] == 2) {
                        const c: i32 = @as(i32, self.cp0[i][4]) + 256;
                        var pp = self.t[(self.cxt[i] +% @as(u32, @intCast(c >> 6))) & self.tmask].get(chksum, self.kep);
                        pp[0] = @intCast(1 + ((c >> 5) & 1));
                        pp[@intCast(1 + ((c >> 5) & 1))] = @intCast(1 + ((c >> 4) & 1));
                        pp[@intCast(3 + ((c >> 4) & 3))] = @intCast(1 + ((c >> 3) & 1));
                        pp = self.t[(self.cxt[i] +% @as(u32, @intCast(c >> 3))) & self.tmask].get(chksum, self.kep);
                        pp[0] = @intCast(1 + ((c >> 2) & 1));
                        pp[@intCast(1 + ((c >> 2) & 1))] = @intCast(1 + ((c >> 1) & 1));
                        pp[@intCast(3 + ((c >> 1) & 3))] = @intCast(1 + (c & 1));
                        self.cp0[i][6] = 0;
                    }
                    if (self.runp[i][0] == 0) {
                        self.runp[i][0] = 2;
                        self.runp[i][1] = @intCast(c1);
                    } else if (self.runp[i][1] != c1) {
                        self.runp[i][0] = 1;
                        self.runp[i][1] = @intCast(c1);
                    } else if (self.runp[i][0] < 254) {
                        self.runp[i][0] += 2;
                    }
                    self.runp[i] = self.cp0[i] + 3;
                }
                s = self.cp[i].?[0];
            }
            self.result += self.mix3(y1, m, s, &self.sm[i]);
            var b: i32 = st.x.c0shift_bpos ^ (@as(i32, self.runp[i][1]) >> @intCast(st.x.bposshift));
            if (b <= 1) {
                b = b * 256;
                st.x.mxInputs[m].add(self.rc1[@intCast(@as(i32, self.runp[i][0]) + b)]);
            } else {
                st.x.mxInputs[m].add(0);
            }
        }
        if (bp == 7) self.cn = 0;
        return self.result;
    }
};
