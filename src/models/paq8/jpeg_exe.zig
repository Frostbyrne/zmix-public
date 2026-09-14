//! Faithful Zig port of cmix PAQ8's jpegModel and exeModel (x86/x64) from
//! reference/cmix-src/models/paq8.cpp, built on core.zig + maps.zig.
//!
//! Public API (for the integrator):
//!   * init(allocator)                 -- must be called once before use
//!   * jpegModel(m: *core.Mixer) i32   -- non-zero if the model engaged
//!   * exeModel(m, Forced, Stats) bool -- x86/x64 code model
//!
//! Faithfulness notes:
//!   * C `unsigned` wrap -> `+%`/`-%`/`*%`.  Signed int overflow in the JPEG
//!     adv-pred accumulators is reproduced with wrapping i32 ops (x86 two's
//!     complement), then C `/` -> @divTrunc.
//!   * Every m.add/m.setcall is reproduced in the exact order & count.
//!   * jpeg's `static` locals become persistent fields on a heap Jpeg instance;
//!     likewise exe's statics live on a heap Exe instance.
//!
//! ModelStats: exeModel writes only `Stats.x86_64` (present in core.ModelStats).
//! No ModelStats fields are kept locally.

const std = @import("std");
const core = @import("core.zig");
const maps = @import("maps.zig");
const xt = @import("x86_tables.zig");

// ============================ module allocator =============================
var galloc: ?std.mem.Allocator = null;

/// Must be called once (with the same allocator used for core.buf, etc.) before
/// calling jpegModel/exeModel. Mirrors the "static, constructed on first use"
/// semantics of the C++ by lazily allocating each model on first call.
pub fn init(a: std.mem.Allocator) void {
    galloc = a;
    // Reset the lazy singletons so a fresh predictor re-allocates them from the
    // current arena. Without this, the second predictor in a process reuses the
    // first's freed Jpeg/Exe (dangling ContextMap2 -> segfault).
    jpeg_ptr = null;
    exe_ptr = null;
}

inline fn A() std.mem.Allocator {
    return galloc.?;
}

// =============================== small helpers =============================
inline fn imin(a: i32, b: i32) i32 {
    return if (a < b) a else b;
}
inline fn iabs(x: i32) i32 {
    return if (x < 0) -x else x;
}
inline fn b2i(b: bool) i32 {
    return @intFromBool(b);
}
/// U8-truncating store (C `U8& = int`).
inline fn stU8(v: i32) u8 {
    return @truncate(@as(u32, @bitCast(v)));
}
/// paq8 ilog(U16 x): argument truncated to 16 bits by the table lookup.
inline fn ilg(v: i32) i32 {
    return core.ilog(@as(u32, @bitCast(v)));
}

/// buf[p] : absolute (masked) byte read from the global ring buffer.
inline fn bufAbs(p: i32) i32 {
    return @as(i32, core.buf.at(@bitCast(p)).*);
}
/// buf(i) : byte `i` positions before pos.
inline fn bufRel(i: i32) i32 {
    return core.buf.get(@intCast(i));
}

// =============================== JPEG markers ==============================
const M_SOF0: i32 = 0xc0;
const M_DHT: i32 = 0xc4;
const M_RST0: i32 = 0xd0;
const M_SOI: i32 = 0xd8;
const M_EOI: i32 = 0xd9;
const M_DQT: i32 = 0xdb;
const M_FF: i32 = 0xff;

// =============================== zigzag tables =============================
const zzu = [64]u8{
    0, 1, 0, 0, 1, 2, 3, 2, 1, 0, 0, 1, 2, 3, 4, 5, 4, 3, 2, 1, 0, 0, 1, 2, 3, 4, 5, 6, 7, 6, 5, 4,
    3, 2, 1, 0, 1, 2, 3, 4, 5, 6, 7, 7, 6, 5, 4, 3, 2, 3, 4, 5, 6, 7, 7, 6, 5, 4, 5, 6, 7, 7, 6, 7,
};
const zzv = [64]u8{
    0, 0, 1, 2, 1, 0, 0, 1, 2, 3, 4, 3, 2, 1, 0, 0, 1, 2, 3, 4, 5, 6, 5, 4, 3, 2, 1, 0, 0, 1, 2, 3,
    4, 5, 6, 7, 7, 6, 5, 4, 3, 2, 1, 2, 3, 4, 5, 6, 7, 7, 6, 5, 4, 3, 4, 5, 6, 7, 7, 6, 5, 6, 7, 7,
};

inline fn ZU(i: i32) i32 {
    return @as(i32, zzu[@intCast(i)]);
}
inline fn ZV(i: i32) i32 {
    return @as(i32, zzv[@intCast(i)]);
}

// Standard Huffman tables (JPEG standard K.3), 8-bit precision only.
const bits_dc_luminance = [16]u8{ 0, 1, 5, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0 };
const values_dc_luminance = [12]u8{ 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11 };
const bits_dc_chrominance = [16]u8{ 0, 3, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0 };
const values_dc_chrominance = [12]u8{ 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11 };
const bits_ac_luminance = [16]u8{ 0, 2, 1, 3, 3, 2, 4, 3, 5, 5, 4, 4, 0, 0, 1, 0x7d };
const values_ac_luminance = [162]u8{
    0x01, 0x02, 0x03, 0x00, 0x04, 0x11, 0x05, 0x12, 0x21, 0x31, 0x41, 0x06, 0x13, 0x51, 0x61, 0x07,
    0x22, 0x71, 0x14, 0x32, 0x81, 0x91, 0xa1, 0x08, 0x23, 0x42, 0xb1, 0xc1, 0x15, 0x52, 0xd1, 0xf0,
    0x24, 0x33, 0x62, 0x72, 0x82, 0x09, 0x0a, 0x16, 0x17, 0x18, 0x19, 0x1a, 0x25, 0x26, 0x27, 0x28,
    0x29, 0x2a, 0x34, 0x35, 0x36, 0x37, 0x38, 0x39, 0x3a, 0x43, 0x44, 0x45, 0x46, 0x47, 0x48, 0x49,
    0x4a, 0x53, 0x54, 0x55, 0x56, 0x57, 0x58, 0x59, 0x5a, 0x63, 0x64, 0x65, 0x66, 0x67, 0x68, 0x69,
    0x6a, 0x73, 0x74, 0x75, 0x76, 0x77, 0x78, 0x79, 0x7a, 0x83, 0x84, 0x85, 0x86, 0x87, 0x88, 0x89,
    0x8a, 0x92, 0x93, 0x94, 0x95, 0x96, 0x97, 0x98, 0x99, 0x9a, 0xa2, 0xa3, 0xa4, 0xa5, 0xa6, 0xa7,
    0xa8, 0xa9, 0xaa, 0xb2, 0xb3, 0xb4, 0xb5, 0xb6, 0xb7, 0xb8, 0xb9, 0xba, 0xc2, 0xc3, 0xc4, 0xc5,
    0xc6, 0xc7, 0xc8, 0xc9, 0xca, 0xd2, 0xd3, 0xd4, 0xd5, 0xd6, 0xd7, 0xd8, 0xd9, 0xda, 0xe1, 0xe2,
    0xe3, 0xe4, 0xe5, 0xe6, 0xe7, 0xe8, 0xe9, 0xea, 0xf1, 0xf2, 0xf3, 0xf4, 0xf5, 0xf6, 0xf7, 0xf8,
    0xf9, 0xfa,
};
const bits_ac_chrominance = [16]u8{ 0, 2, 1, 2, 4, 4, 3, 4, 7, 5, 4, 4, 0, 1, 2, 0x77 };
const values_ac_chrominance = [162]u8{
    0x00, 0x01, 0x02, 0x03, 0x11, 0x04, 0x05, 0x21, 0x31, 0x06, 0x12, 0x41, 0x51, 0x07, 0x61, 0x71,
    0x13, 0x22, 0x32, 0x81, 0x08, 0x14, 0x42, 0x91, 0xa1, 0xb1, 0xc1, 0x09, 0x23, 0x33, 0x52, 0xf0,
    0x15, 0x62, 0x72, 0xd1, 0x0a, 0x16, 0x24, 0x34, 0xe1, 0x25, 0xf1, 0x17, 0x18, 0x19, 0x1a, 0x26,
    0x27, 0x28, 0x29, 0x2a, 0x35, 0x36, 0x37, 0x38, 0x39, 0x3a, 0x43, 0x44, 0x45, 0x46, 0x47, 0x48,
    0x49, 0x4a, 0x53, 0x54, 0x55, 0x56, 0x57, 0x58, 0x59, 0x5a, 0x63, 0x64, 0x65, 0x66, 0x67, 0x68,
    0x69, 0x6a, 0x73, 0x74, 0x75, 0x76, 0x77, 0x78, 0x79, 0x7a, 0x82, 0x83, 0x84, 0x85, 0x86, 0x87,
    0x88, 0x89, 0x8a, 0x92, 0x93, 0x94, 0x95, 0x96, 0x97, 0x98, 0x99, 0x9a, 0xa2, 0xa3, 0xa4, 0xa5,
    0xa6, 0xa7, 0xa8, 0xa9, 0xaa, 0xb2, 0xb3, 0xb4, 0xb5, 0xb6, 0xb7, 0xb8, 0xb9, 0xba, 0xc2, 0xc3,
    0xc4, 0xc5, 0xc6, 0xc7, 0xc8, 0xc9, 0xca, 0xd2, 0xd3, 0xd4, 0xd5, 0xd6, 0xd7, 0xd8, 0xd9, 0xda,
    0xe2, 0xe3, 0xe4, 0xe5, 0xe6, 0xe7, 0xe8, 0xe9, 0xea, 0xf2, 0xf3, 0xf4, 0xf5, 0xf6, 0xf7, 0xf8,
    0xf9, 0xfa,
};

// ============================== jpeg structs ===============================
const HUF = struct { min: u32 = 0, max: u32 = 0, val: i32 = 0 };

const JPEGImage = struct {
    offset: i32 = 0,
    jpeg: i32 = 0, // 1 header detected, 2 image data
    next_jpeg: i32 = 0,
    app: i32 = 0,
    sof: i32 = 0,
    sos: i32 = 0,
    data: i32 = 0,
    htsize: i32 = 0,
    ht: [8]i32 = .{0} ** 8,
    qtab: [256]u8 = .{0} ** 256,
    qmap: [10]i32 = .{0} ** 10,
};

const N_JCTX: usize = 32; // number of jpeg bit-history contexts

pub const Jpeg = struct {
    // ---- parser / decode state ----
    images: [3]JPEGImage,
    idx: i32,
    lastPos: i32,
    huffcode: u32,
    huffbits: i32,
    huffsize: i32,
    rs: i32,
    mcupos: i32,
    huf: [128]HUF,
    mcusize: i32,
    hufsel: [2][10]i32,
    hbuf: [2048]u8,
    color: [10]i32,
    pred: [4]i32,
    dc: i32,
    width: i32,
    row: i32,
    column: i32,
    cbuf: [0x20000]u8, // rotating buffer of coded coefficients
    cpos: i32,
    rs1: i32,
    rstpos: i32,
    rstlen: i32,
    ssum: i32,
    ssum1: i32,
    ssum2: i32,
    ssum3: i32,
    cbuf2: [0x20000]i32,
    adv_pred: [4]i32,
    sumu: [8]i32,
    sumv: [8]i32,
    run_pred: [6]i32,
    prev_coef: i32,
    prev_coef2: i32,
    prev_coef_rs: i32,
    ls: [10]i32,
    blockW: [10]i32,
    blockN: [10]i32,
    SamplingFactors: [4]i32,
    lcp: [7]i32,
    zpos: [64]i32,
    dqt_state: i32,
    dqt_end: i32,
    qnum: i32,

    // ---- context model state ----
    t: maps.BH(9),
    cxt: [N_JCTX]u64,
    cp: [N_JCTX][*]u8,
    cp_set: bool,
    sm: [N_JCTX]core.StateMap,
    m1: *core.Mixer,
    a1: core.APM,
    a2: core.APM,
    hbcount: i32,

    fn create(a: std.mem.Allocator) *Jpeg {
        const self = a.create(Jpeg) catch unreachable;
        @memset(std.mem.asBytes(self), 0);
        self.idx = -1;
        self.rs = -1;
        self.dqt_state = -1;
        self.hbcount = 2;
        self.cp_set = false;
        self.t = maps.BH(9).init(a, @intCast(core.MEM()));
        self.m1 = core.Mixer.init(a, N_JCTX + 1, 2050, 3, 0);
        self.a1 = core.APM.init(a, 0x8000);
        self.a2 = core.APM.init(a, 0x20000);
        for (&self.sm) |*s| s.* = core.StateMap.init(a);
        return self;
    }

    // cbuf / cbuf2 masked ring accessors (index & 0x1FFFF like Buf/IntBuf).
    inline fn cbufAt(self: *Jpeg, i: i32) *u8 {
        return &self.cbuf[@as(u32, @bitCast(i)) & 0x1FFFF];
    }
    inline fn cbuf2At(self: *Jpeg, i: i32) *i32 {
        return &self.cbuf2[@as(u32, @bitCast(i)) & 0x1FFFF];
    }

    inline fn img(self: *Jpeg) *JPEGImage {
        return &self.images[@intCast(self.idx)];
    }

    fn finish(self: *Jpeg) void {
        const length = core.pos - self.images[@intCast(self.idx)].offset;
        self.images[@intCast(self.idx)] = .{};
        self.mcusize = 0;
        self.dqt_state = -1;
        self.idx -= b2i(self.idx > 0);
        self.images[@intCast(self.idx)].app -= length;
        if (self.images[@intCast(self.idx)].app < 0) self.images[@intCast(self.idx)].app = 0;
    }

    fn jfail(self: *Jpeg) i32 {
        if (self.idx > 0) {
            self.finish();
        } else {
            self.images[@intCast(self.idx)].jpeg = 0;
        }
        return self.images[@intCast(self.idx)].next_jpeg;
    }

    fn run(self: *Jpeg, m: *core.Mixer) i32 {
        const FF = M_FF;
        const bpos = core.bpos;
        const pos = core.pos;
        const y = core.y;

        if (self.idx < 0) {
            self.images = .{ .{}, .{}, .{} };
            self.idx = 0;
            self.lastPos = pos;
        }

        // Be sure to quit on a byte boundary
        if (bpos == 0) self.img().next_jpeg = b2i(self.img().jpeg > 1);
        if (bpos != 0 and self.img().jpeg == 0) return self.img().next_jpeg;
        if (bpos == 0 and self.img().app > 0) {
            self.img().app -= 1;
            if (self.idx < 3 and bufRel(4) == FF and bufRel(3) == M_SOI and bufRel(2) == FF and
                ((bufRel(1) & 0xFE) == 0xC0 or bufRel(1) == 0xC4 or (bufRel(1) >= 0xDB and bufRel(1) <= 0xFE)))
            {
                self.idx += 1;
                self.images[@intCast(self.idx)] = .{};
            }
        }
        if (self.img().app > 0) return self.img().next_jpeg;

        if (bpos == 0) {
            // Detect JPEG (SOI followed by a valid marker)
            if (self.img().jpeg == 0 and bufRel(4) == FF and bufRel(3) == M_SOI and bufRel(2) == FF and
                ((bufRel(1) & 0xFE) == 0xC0 or bufRel(1) == 0xC4 or (bufRel(1) >= 0xDB and bufRel(1) <= 0xFE)))
            {
                self.img().jpeg = 1;
                self.img().offset = pos - 4;
                self.img().sos = 0;
                self.img().sof = 0;
                self.img().htsize = 0;
                self.img().data = 0;
                self.img().app = b2i((bufRel(1) >> 4) == 0xE) * 2;
                self.mcusize = 0;
                self.huffcode = 0;
                self.huffbits = 0;
                self.huffsize = 0;
                self.mcupos = 0;
                self.cpos = 0;
                self.rs = -1;
                @memset(&self.huf, .{});
                @memset(&self.pred, 0);
                self.rstpos = 0;
                self.rstlen = 0;
            }

            // Detect end of JPEG when data contains a non-RSTx marker / jump
            if (self.img().jpeg != 0 and self.img().data != 0 and
                ((bufRel(2) == FF and bufRel(1) != 0 and (bufRel(1) & 0xf8) != M_RST0) or (pos - self.lastPos > 1)))
            {
                if (!((bufRel(1) == M_EOI) or (pos - self.lastPos > 1))) return self.jfail();
                self.finish();
            }
            self.lastPos = pos;
            if (self.img().jpeg == 0) return self.img().next_jpeg;

            // Detect APPx, COM or other markers to skip
            if (self.img().data == 0 and self.img().app == 0 and bufRel(4) == FF and
                (((bufRel(3) > 0xC1) and (bufRel(3) <= 0xCF) and (bufRel(3) != M_DHT)) or ((bufRel(3) >= 0xDC) and (bufRel(3) <= 0xFE))))
            {
                self.img().app = bufRel(2) * 256 + bufRel(1) + 2;
                if (self.idx > 0) {
                    if (!(pos + self.img().app < self.img().offset + self.images[@intCast(self.idx - 1)].app)) return self.jfail();
                }
            }

            // Save pointers to sof, ht, sos, data
            if (bufRel(5) == FF and bufRel(4) == 0xDA) {
                const len = bufRel(3) * 256 + bufRel(2);
                if (len == 6 + 2 * bufRel(1) and bufRel(1) != 0 and bufRel(1) <= 4) {
                    self.img().sos = pos - 5;
                    self.img().data = self.img().sos + len + 2;
                    self.img().jpeg = 2;
                }
            }
            if (bufRel(4) == FF and bufRel(3) == M_DHT and self.img().htsize < 8) {
                self.img().ht[@intCast(self.img().htsize)] = pos - 4;
                self.img().htsize += 1;
            }
            if (bufRel(4) == FF and (bufRel(3) & 0xFE) == M_SOF0) self.img().sof = pos - 4;

            // Parse Quantization tables
            if (bufRel(4) == FF and bufRel(3) == M_DQT) {
                self.dqt_end = pos + bufRel(2) * 256 + bufRel(1) - 1;
                self.dqt_state = 0;
            } else if (self.dqt_state >= 0) {
                if (pos >= self.dqt_end) {
                    self.dqt_state = -1;
                } else {
                    if (@rem(self.dqt_state, 65) == 0) {
                        self.qnum = bufRel(1);
                    } else {
                        if (!(bufRel(1) > 0)) return self.jfail();
                        if (!(self.qnum >= 0 and self.qnum < 4)) return self.jfail();
                        self.img().qtab[@intCast(self.qnum * 64 + (@rem(self.dqt_state, 65) - 1))] = @intCast(bufRel(1) - 1);
                    }
                    self.dqt_state += 1;
                }
            }

            // Restart
            if (bufRel(2) == FF and (bufRel(1) & 0xf8) == M_RST0) {
                self.huffcode = 0;
                self.huffbits = 0;
                self.huffsize = 0;
                self.mcupos = 0;
                self.rs = -1;
                @memset(&self.pred, 0);
                self.rstlen = self.column + self.row * self.width - self.rstpos;
                self.rstpos = self.column + self.row * self.width;
            }
        }

        // ---- Build Huffman tables ----
        if (pos == self.img().data and bpos == 1) {
            var i: i32 = 0;
            while (i < self.img().htsize) : (i += 1) {
                var p = self.img().ht[@intCast(i)] + 4;
                const end = p + bufAbs(p - 2) * 256 + bufAbs(p - 1) - 2;
                var count: i32 = 0;
                while (p < end and end < pos and end < p + 2100) {
                    count += 1;
                    if (!(count < 10)) break;
                    const tc = bufAbs(p) >> 4;
                    const th = bufAbs(p) & 15;
                    if (tc >= 2 or th >= 4) break;
                    if (!(tc >= 0 and tc < 2 and th >= 0 and th < 4)) return self.jfail();
                    const base: usize = @intCast(tc * 64 + th * 16);
                    var val = p + 17;
                    var hval = tc * 1024 + th * 256;
                    var j: i32 = 0;
                    while (j < 256) : (j += 1) self.hbuf[@intCast(hval + j)] = @intCast(bufAbs(val + j));
                    var code: i32 = 0;
                    j = 0;
                    while (j < 16) : (j += 1) {
                        const cnt = bufAbs(p + j + 1);
                        self.huf[base + @as(usize, @intCast(j))].min = @intCast(code);
                        code += cnt;
                        self.huf[base + @as(usize, @intCast(j))].max = @intCast(code);
                        self.huf[base + @as(usize, @intCast(j))].val = hval;
                        val += cnt;
                        hval += cnt;
                        code *= 2;
                    }
                    p = val;
                    if (!(hval >= 0 and hval < 2048)) return self.jfail();
                }
                if (!(p == end)) return self.jfail();
            }
            self.huffcode = 0;
            self.huffbits = 0;
            self.huffsize = 0;
            self.rs = -1;

            // load default tables
            if (self.img().htsize == 0) {
                var tc: i32 = 0;
                while (tc < 2) : (tc += 1) {
                    var th: i32 = 0;
                    while (th < 2) : (th += 1) {
                        const base: usize = @intCast(tc * 64 + th * 16);
                        var hval = tc * 1024 + th * 256;
                        var code: i32 = 0;
                        var c: i32 = 0;
                        var x: i32 = 0;
                        var ib: i32 = 0;
                        while (ib < 16) : (ib += 1) {
                            x = switch (tc * 2 + th) {
                                0 => @as(i32, bits_dc_luminance[@intCast(ib)]),
                                1 => @as(i32, bits_dc_chrominance[@intCast(ib)]),
                                2 => @as(i32, bits_ac_luminance[@intCast(ib)]),
                                else => @as(i32, bits_ac_chrominance[@intCast(ib)]),
                            };
                            self.huf[base + @as(usize, @intCast(ib))].min = @intCast(code);
                            code += x;
                            self.huf[base + @as(usize, @intCast(ib))].max = @intCast(code);
                            self.huf[base + @as(usize, @intCast(ib))].val = hval;
                            hval += x;
                            code += code;
                            c += x;
                        }
                        hval = tc * 1024 + th * 256;
                        c -= 1;
                        while (c >= 0) {
                            x = switch (tc * 2 + th) {
                                0 => @as(i32, values_dc_luminance[@intCast(c)]),
                                1 => @as(i32, values_dc_chrominance[@intCast(c)]),
                                2 => @as(i32, values_ac_luminance[@intCast(c)]),
                                else => @as(i32, values_ac_chrominance[@intCast(c)]),
                            };
                            self.hbuf[@intCast(hval + c)] = @intCast(x);
                            c -= 1;
                        }
                    }
                }
                self.img().htsize = 4;
            }

            // Build Huffman table selection table (indexed by mcupos).
            if (self.img().sof == 0 and self.img().sos != 0) return self.img().next_jpeg;
            const ns = bufAbs(self.img().sos + 4);
            const nf = bufAbs(self.img().sof + 9);
            if (!(ns <= 4 and nf <= 4)) return self.jfail();
            self.mcusize = 0;
            var hmax: i32 = 0;
            i = 0;
            while (i < ns) : (i += 1) {
                var j: i32 = 0;
                while (j < nf) : (j += 1) {
                    if (bufAbs(self.img().sos + 2 * i + 5) == bufAbs(self.img().sof + 3 * j + 10)) {
                        var hv = bufAbs(self.img().sof + 3 * j + 11);
                        self.SamplingFactors[@intCast(j)] = hv;
                        if (hv >> 4 > hmax) hmax = hv >> 4;
                        hv = (hv & 15) * (hv >> 4);
                        if (!(hv >= 1 and hv + self.mcusize <= 10)) return self.jfail();
                        while (hv != 0) {
                            if (!(self.mcusize < 10)) return self.jfail();
                            self.hufsel[0][@intCast(self.mcusize)] = (bufAbs(self.img().sos + 2 * i + 6) >> 4) & 15;
                            self.hufsel[1][@intCast(self.mcusize)] = bufAbs(self.img().sos + 2 * i + 6) & 15;
                            if (!(self.hufsel[0][@intCast(self.mcusize)] < 4 and self.hufsel[1][@intCast(self.mcusize)] < 4)) return self.jfail();
                            self.color[@intCast(self.mcusize)] = i;
                            const tq = bufAbs(self.img().sof + 3 * j + 12);
                            if (!(tq >= 0 and tq < 4)) return self.jfail();
                            self.img().qmap[@intCast(self.mcusize)] = tq;
                            hv -= 1;
                            self.mcusize += 1;
                        }
                    }
                }
            }
            if (!(hmax >= 1 and hmax <= 10)) return self.jfail();
            var j: i32 = 0;
            while (j < self.mcusize) : (j += 1) {
                self.ls[@intCast(j)] = 0;
                var ii: i32 = 1;
                while (ii < self.mcusize) : (ii += 1) {
                    if (self.color[@intCast(@rem(j + ii, self.mcusize))] == self.color[@intCast(j)]) self.ls[@intCast(j)] = ii;
                }
                self.ls[@intCast(j)] = (self.mcusize - self.ls[@intCast(j)]) << 6;
            }
            j = 0;
            while (j < 64) : (j += 1) self.zpos[@intCast(ZU(j) + 8 * ZV(j))] = j;
            self.width = bufAbs(self.img().sof + 7) * 256 + bufAbs(self.img().sof + 8);
            self.width = @divTrunc(self.width - 1, hmax * 8) + 1;
            if (!(self.width > 0)) return self.jfail();
            self.mcusize *= 64;
            self.row = 0;
            self.column = 0;

            // subsampling: blockW / blockN
            var xx: i32 = 0;
            var yy: i32 = 0;
            j = 0;
            while (j < (self.mcusize >> 6)) : (j += 1) {
                const ic = self.color[@intCast(j)];
                const w = self.SamplingFactors[@intCast(ic)] >> 4;
                const h = self.SamplingFactors[@intCast(ic)] & 0xf;
                self.blockW[@intCast(j)] = if (xx == 0) self.mcusize - 64 * (w - 1) else 64;
                self.blockN[@intCast(j)] = if (yy == 0) self.mcusize * self.width - 64 * w * (h - 1) else w * 64;
                xx += 1;
                if (xx >= w) {
                    xx = 0;
                    yy += 1;
                }
                if (yy >= h) {
                    xx = 0;
                    yy = 0;
                }
            }
        }

        // ---- Decode Huffman ----
        if (self.mcusize != 0 and bufRel(1 + b2i(bpos == 0)) != FF) { // skip stuffed byte
            if (!(self.huffbits <= 32)) return self.jfail();
            self.huffcode = self.huffcode +% self.huffcode +% @as(u32, @intCast(y));
            self.huffbits += 1;
            if (self.rs < 0) {
                if (!(self.huffbits >= 1 and self.huffbits <= 16)) return self.jfail();
                const ac = b2i((self.mcupos & 63) > 0);
                if (!(self.mcupos >= 0 and (self.mcupos >> 6) < 10)) return self.jfail();
                const sel = self.hufsel[@intCast(ac)][@intCast(self.mcupos >> 6)];
                if (!(sel >= 0 and sel < 4)) return self.jfail();
                const i = self.huffbits - 1;
                if (!(i >= 0 and i < 16)) return self.jfail();
                const hbase: usize = @intCast(ac * 64 + sel * 16);
                const h = &self.huf[hbase + @as(usize, @intCast(i))];
                if (!(h.min <= h.max and h.val < 2048 and self.huffbits > 0)) return self.jfail();
                if (self.huffcode < h.max) {
                    if (!(self.huffcode >= h.min)) return self.jfail();
                    const k = @as(u32, @bitCast(h.val)) +% self.huffcode -% h.min;
                    if (!(k < 2048)) return self.jfail();
                    self.rs = self.hbuf[k];
                    self.huffsize = self.huffbits;
                }
            }
            if (self.rs >= 0) {
                if (self.huffsize + (self.rs & 15) == self.huffbits) { // done decoding
                    self.rs1 = self.rs;
                    var x: i32 = 0; // decoded extra bits
                    if ((self.mcupos & 63) != 0) { // AC
                        if (self.rs == 0) { // EOB
                            self.mcupos = (self.mcupos + 63) & -64;
                            if (!(self.mcupos >= 0 and self.mcupos <= self.mcusize and self.mcupos <= 640)) return self.jfail();
                            while ((self.cpos & 63) != 0) {
                                self.cbuf2At(self.cpos).* = 0;
                                self.cbufAt(self.cpos).* = if (self.rs == 0) 0 else stU8((63 - (self.cpos & 63)) << 4);
                                self.cpos += 1;
                                self.rs += 1; // (!rs) picks 0 on the first cell only
                            }
                        } else { // rs = r zeros + s extra bits
                            if (!((self.rs & 15) <= 10)) return self.jfail();
                            const r = self.rs >> 4;
                            const s = self.rs & 15;
                            if (!(self.mcupos >> 6 == (self.mcupos + r) >> 6)) return self.jfail();
                            self.mcupos += r + 1;
                            x = @bitCast(self.huffcode & @as(u32, @intCast((@as(i32, 1) << @intCast(s)) - 1)));
                            if (s != 0 and (x >> @intCast(s - 1)) == 0) x -= (@as(i32, 1) << @intCast(s)) - 1;
                            var ri: i32 = r;
                            while (ri >= 1) : (ri -= 1) {
                                self.cbuf2At(self.cpos).* = 0;
                                self.cbufAt(self.cpos).* = stU8((ri << 4) | s);
                                self.cpos += 1;
                            }
                            self.cbuf2At(self.cpos).* = x;
                            self.cbufAt(self.cpos).* = stU8((s << 4) | @as(i32, @bitCast((self.huffcode << 2 >> @intCast(s)) & 3)) | 12);
                            self.cpos += 1;
                            self.ssum += s;
                        }
                    } else { // DC: rs = 0S, s<12
                        if (!(self.rs < 12)) return self.jfail();
                        self.mcupos += 1;
                        x = @bitCast(self.huffcode & @as(u32, @intCast((@as(i32, 1) << @intCast(self.rs)) - 1)));
                        if (self.rs != 0 and (x >> @intCast(self.rs - 1)) == 0) x -= (@as(i32, 1) << @intCast(self.rs)) - 1;
                        if (!(self.mcupos >= 0 and self.mcupos >> 6 < 10)) return self.jfail();
                        const comp = self.color[@intCast(self.mcupos >> 6)];
                        if (!(comp >= 0 and comp < 4)) return self.jfail();
                        self.pred[@intCast(comp)] +%= x;
                        self.dc = self.pred[@intCast(comp)];
                        if (!((self.cpos & 63) == 0)) return self.jfail();
                        self.cbuf2At(self.cpos).* = self.dc;
                        self.cbufAt(self.cpos).* = stU8((self.dc + 1023) >> 3);
                        self.cpos += 1;
                        if ((self.mcupos >> 6) == 0) {
                            self.ssum1 = 0;
                            self.ssum2 = self.ssum3;
                        } else {
                            if (self.color[@intCast((self.mcupos >> 6) - 1)] == self.color[0]) {
                                self.ssum3 = self.ssum;
                                self.ssum1 += self.ssum3;
                            }
                            self.ssum2 = self.ssum1;
                        }
                        self.ssum = self.rs;
                    }
                    if (!(self.mcupos >= 0 and self.mcupos <= self.mcusize)) return self.jfail();
                    if (self.mcupos >= self.mcusize) {
                        self.mcupos = 0;
                        self.column += 1;
                        if (self.column == self.width) {
                            self.column = 0;
                            self.row += 1;
                        }
                    }
                    self.huffcode = 0;
                    self.huffsize = 0;
                    self.huffbits = 0;
                    self.rs = -1;

                    // ---- UPDATE_ADV_PRED ----
                    self.updateAdvPred();
                }
            }
        }

        // ---- Estimate next bit probability ----
        if (self.img().jpeg == 0 or self.img().data == 0) return self.img().next_jpeg;
        if (bufRel(1 + b2i(bpos == 0)) == FF) {
            m.add(128);
            m.set(0, 9);
            m.set(0, 1025);
            m.set(bufRel(1), 1024);
            return 1;
        }
        if (self.rstlen > 0 and self.rstlen == self.column + self.row * self.width - self.rstpos and self.mcupos == 0 and
            @as(u64, self.huffcode) == (@as(u64, 1) << @intCast(self.huffbits)) - 1)
        {
            m.add(4095);
            m.set(0, 9);
            m.set(0, 1025);
            m.set(bufRel(1), 1024);
            return 1;
        }

        // ---- Context model ----
        // Update model (bit histories) with the just-observed bit y.
        if (self.cp_set) {
            var i: usize = 0;
            while (i < N_JCTX) : (i += 1) self.cp[i][0] = core.nex(@as(u32, self.cp[i][0]), @as(u32, @intCast(y)));
        }
        self.m1.update();

        // Update context
        const comp = self.color[@intCast(self.mcupos >> 6)];
        const coef = (self.mcupos & 63) | (comp << 6);
        const hc_u: u32 = (self.huffcode *% 4 +%
            @as(u32, @bitCast(b2i((self.mcupos & 63) == 0) * 2 + b2i(comp == 0)))) |
            (@as(u32, 1) << @intCast(self.huffbits + 2));
        const hc: i32 = @bitCast(hc_u);
        const firstcol = self.column == 0 and self.blockW[@intCast(self.mcupos >> 6)] > self.mcupos;
        self.hbcount += 1;
        if (self.hbcount > 2 or self.huffbits == 0) self.hbcount = 0;
        if (!(coef >= 0 and coef < 256)) return self.jfail();
        const zu = ZU(self.mcupos & 63);
        const zv = ZV(self.mcupos & 63);

        if (self.hbcount == 0) {
            var n: u64 = @bitCast(@as(i64, hc *% 32));
            const ap = &self.adv_pred;
            const rp = &self.run_pred;
            const lc = &self.lcp;
            n +%= 1;
            self.cxt[0] = maps.hash(.{ n, coef, @divTrunc(ap[2], 12) + (rp[2] << 8), self.ssum2 >> 6, @divTrunc(self.prev_coef, 72) });
            n +%= 1;
            self.cxt[1] = maps.hash(.{ n, coef, @divTrunc(ap[0], 12) + (rp[0] << 8), self.ssum2 >> 6, @divTrunc(self.prev_coef, 72) });
            n +%= 1;
            self.cxt[2] = maps.hash(.{ n, coef, @divTrunc(ap[1], 11) + (rp[1] << 8), self.ssum2 >> 6 });
            n +%= 1;
            self.cxt[3] = maps.hash(.{ n, self.rs1, @divTrunc(ap[2], 7), @divTrunc(rp[5], 2), @divTrunc(self.prev_coef, 10) });
            n +%= 1;
            self.cxt[4] = maps.hash(.{ n, self.rs1, @divTrunc(ap[0], 7), @divTrunc(rp[3], 2), @divTrunc(self.prev_coef, 10) });
            n +%= 1;
            self.cxt[5] = maps.hash(.{ n, self.rs1, @divTrunc(ap[1], 11), rp[4] });
            n +%= 1;
            self.cxt[6] = maps.hash(.{ n, @divTrunc(ap[2], 14), rp[2], @divTrunc(ap[0], 14), rp[0] });
            n +%= 1;
            self.cxt[7] = maps.hash(.{ n, @as(i32, self.cbufAt(self.cpos - self.blockN[@intCast(self.mcupos >> 6)]).*) >> 4, @divTrunc(ap[3], 17), rp[1], rp[5] });
            n +%= 1;
            self.cxt[8] = maps.hash(.{ n, @as(i32, self.cbufAt(self.cpos - self.blockW[@intCast(self.mcupos >> 6)]).*) >> 4, @divTrunc(ap[3], 17), rp[1], rp[3] });
            n +%= 1;
            self.cxt[9] = maps.hash(.{ n, @divTrunc(lc[0], 22), @divTrunc(lc[1], 22), @divTrunc(ap[1], 7), rp[1] });
            n +%= 1;
            self.cxt[10] = maps.hash(.{ n, @divTrunc(lc[0], 22), @divTrunc(lc[1], 22), self.mcupos & 63, @divTrunc(lc[4], 30) });
            n +%= 1;
            self.cxt[11] = maps.hash(.{ n, @divTrunc(zu, 2), @divTrunc(lc[0], 13), @divTrunc(lc[2], 30), @divTrunc(self.prev_coef, 40) + (@divTrunc(self.prev_coef2, 28) << 20) });
            n +%= 1;
            self.cxt[12] = maps.hash(.{ n, @divTrunc(zv, 2), @divTrunc(lc[1], 13), @divTrunc(lc[3], 30), @divTrunc(self.prev_coef, 40) + (@divTrunc(self.prev_coef2, 28) << 20) });
            n +%= 1;
            self.cxt[13] = maps.hash(.{ n, self.rs1, @divTrunc(self.prev_coef, 42), @divTrunc(self.prev_coef2, 34), @divTrunc(lc[0], 60), @divTrunc(lc[2], 14), @divTrunc(lc[1], 60), @divTrunc(lc[3], 14) });
            n +%= 1;
            self.cxt[14] = maps.hash(.{ n, self.mcupos & 63, self.column >> 1 });
            n +%= 1;
            self.cxt[15] = maps.hash(.{ n, self.column >> 3, imin(5 + 2 * b2i(comp == 0), zu + zv), @divTrunc(lc[0], 10), @divTrunc(lc[2], 40), @divTrunc(lc[1], 10), @divTrunc(lc[3], 40) });
            n +%= 1;
            self.cxt[16] = maps.hash(.{ n, self.ssum >> 3, self.mcupos & 63 });
            n +%= 1;
            self.cxt[17] = maps.hash(.{ n, self.rs1, self.mcupos & 63, rp[1] });
            n +%= 1;
            const arg18: u64 = if (comp != 0)
                maps.hash(.{ @divTrunc(self.prev_coef, 22), @divTrunc(self.prev_coef2, 50) })
            else
                @bitCast(@as(i64, @divTrunc(self.ssum, (self.mcupos & 0x3F) + 1)));
            self.cxt[18] = maps.hash(.{ n, coef, self.ssum2 >> 5, @divTrunc(ap[3], 30), arg18 });
            n +%= 1;
            const arg19: i32 = if (comp != 0)
                @divTrunc(self.prev_coef, 40) + (@divTrunc(self.prev_coef2, 40) << 20)
            else
                @divTrunc(lc[4], 22);
            self.cxt[19] = maps.hash(.{ n, @divTrunc(lc[0], 40), @divTrunc(lc[1], 40), @divTrunc(ap[1], 28), arg19, imin(7, zu + zv), @divTrunc(self.ssum, 2 * (zu + zv) + 1) });
            n +%= 1;
            self.cxt[20] = maps.hash(.{ n, zv, @as(i32, self.cbufAt(self.cpos - self.blockN[@intCast(self.mcupos >> 6)]).*), @divTrunc(ap[2], 28), rp[2] });
            n +%= 1;
            self.cxt[21] = maps.hash(.{ n, zu, @as(i32, self.cbufAt(self.cpos - self.blockW[@intCast(self.mcupos >> 6)]).*), @divTrunc(ap[0], 28), rp[0] });
            n +%= 1;
            self.cxt[22] = maps.hash(.{ n, @divTrunc(ap[2], 7), rp[2] });
            self.cxt[23] = maps.hash(.{ n, @divTrunc(ap[0], 7), rp[0] });
            self.cxt[24] = maps.hash(.{ n, @divTrunc(ap[1], 7), rp[1] });
            n +%= 1;
            self.cxt[25] = maps.hash(.{ n, zv, @divTrunc(lc[1], 14), @divTrunc(ap[2], 16), rp[5] });
            n +%= 1;
            self.cxt[26] = maps.hash(.{ n, zu, @divTrunc(lc[0], 14), @divTrunc(ap[0], 16), rp[3] });
            n +%= 1;
            self.cxt[27] = maps.hash(.{ n, @divTrunc(lc[0], 14), @divTrunc(lc[1], 14), @divTrunc(ap[3], 16) });
            n +%= 1;
            self.cxt[28] = maps.hash(.{ n, coef, @divTrunc(self.prev_coef, 10), @divTrunc(self.prev_coef2, 20) });
            n +%= 1;
            self.cxt[29] = maps.hash(.{ n, coef, self.ssum >> 2, self.prev_coef_rs });
            n +%= 1;
            self.cxt[30] = maps.hash(.{ n, coef, @divTrunc(ap[1], 17), @divTrunc(lc[@intCast(b2i(zu < zv))], 24), @divTrunc(lc[2], 20), @divTrunc(lc[3], 24) });
            n +%= 1;
            self.cxt[31] = maps.hash(.{ n, coef, @divTrunc(ap[3], 11), @divTrunc(lc[@intCast(b2i(zu < zv))], 50), @divTrunc(lc[@intCast(2 + 3 * b2i(zu * zv > 1))], 50), @divTrunc(lc[@intCast(3 + 3 * b2i(zu * zv > 1))], 50) });
        }

        // Predict next bit
        self.m1.add(128);
        var p: i32 = undefined;
        switch (self.hbcount) {
            0 => {
                self.cp_set = true;
                var i: usize = 0;
                while (i < N_JCTX) : (i += 1) {
                    self.cp[i] = self.t.get(self.cxt[i]) + 1;
                    p = self.sm[i].p(@as(i32, self.cp[i][0]));
                    m.add((p - 2048) >> 2);
                    p = core.stretch(p);
                    self.m1.add(p);
                    m.add(p);
                }
            },
            1 => {
                const hc2: i32 = 1 + @as(i32, @bitCast(self.huffcode & 1)) * 3;
                var i: usize = 0;
                while (i < N_JCTX) : (i += 1) {
                    self.cp[i] += @as(usize, @intCast(hc2));
                    p = self.sm[i].p(@as(i32, self.cp[i][0]));
                    m.add((p - 2048) >> 2);
                    p = core.stretch(p);
                    self.m1.add(p);
                    m.add(p);
                }
            },
            else => {
                const hc2: i32 = 1 + @as(i32, @bitCast(self.huffcode & 1));
                var i: usize = 0;
                while (i < N_JCTX) : (i += 1) {
                    self.cp[i] += @as(usize, @intCast(hc2));
                    p = self.sm[i].p(@as(i32, self.cp[i][0]));
                    m.add((p - 2048) >> 2);
                    p = core.stretch(p);
                    self.m1.add(p);
                    m.add(p);
                }
            },
        }

        self.m1.set(b2i(firstcol), 2);
        self.m1.set(coef + 256 * imin(3, self.huffbits), 1024);
        self.m1.set((hc & 0x1FE) * 2 + imin(3, @intCast(maps.ilog2(@intCast(zu + zv)))), 1024);
        var pr = self.m1.p();
        m.add(core.stretch(pr));
        m.add(pr - 2048);
        pr = self.a1.p(pr, (hc & 511) | ((@divTrunc(self.adv_pred[1], 16) & 63) << 9), 1023);
        m.add(core.stretch(pr));
        m.add(pr - 2048);
        pr = self.a2.p(pr, (hc & 511) | (coef << 9), 1023);
        m.add(core.stretch(pr));
        m.add(pr - 2048);
        m.set(1 + b2i(zu + zv < 5) + b2i(self.huffbits > 8) * 2 + b2i(firstcol) * 4, 9);
        m.set(1 + (hc & 0xFF) + 256 * imin(3, @divTrunc(zu + zv, 3)), 1025);
        m.set(coef + 256 * imin(3, @divTrunc(self.huffbits, 2)), 1024);
        return 1;
    }

    fn updateAdvPred(self: *Jpeg) void {
        const acomp = self.mcupos >> 6;
        const q = 64 * self.img().qmap[@intCast(acomp)];
        const zz = self.mcupos & 63;
        const cpos_dc = self.cpos - zz;
        const norst = self.rstpos != self.column + self.row * self.width;
        const qtab = &self.img().qtab;
        var x: i32 = 0;

        if (zz == 0) {
            var i: i32 = 0;
            while (i < 8) : (i += 1) {
                self.sumu[@intCast(i)] = 0;
                self.sumv[@intCast(i)] = 0;
            }
            const offset_DC_W = cpos_dc - self.blockW[@intCast(acomp)];
            const offset_DC_N = cpos_dc - self.blockN[@intCast(acomp)];
            i = 0;
            while (i < 64) : (i += 1) {
                const su = (if ((ZV(i) & 1) != 0) @as(i32, -1) else 1) *%
                    (if (ZV(i) != 0) 16 *% (16 + ZV(i)) else 185) *%
                    (@as(i32, qtab[@intCast(q + i)]) + 1) *% self.cbuf2At(offset_DC_N + i).*;
                self.sumu[@intCast(ZU(i))] +%= su;
                const sv = (if ((ZU(i) & 1) != 0) @as(i32, -1) else 1) *%
                    (if (ZU(i) != 0) 16 *% (16 + ZU(i)) else 185) *%
                    (@as(i32, qtab[@intCast(q + i)]) + 1) *% self.cbuf2At(offset_DC_W + i).*;
                self.sumv[@intCast(ZV(i))] +%= sv;
            }
        } else {
            self.sumu[@intCast(ZU(zz - 1))] -%= (if (ZV(zz - 1) != 0) 16 *% (16 + ZV(zz - 1)) else 185) *%
                (@as(i32, qtab[@intCast(q + zz - 1)]) + 1) *% self.cbuf2At(self.cpos - 1).*;
            self.sumv[@intCast(ZV(zz - 1))] -%= (if (ZU(zz - 1) != 0) 16 *% (16 + ZU(zz - 1)) else 185) *%
                (@as(i32, qtab[@intCast(q + zz - 1)]) + 1) *% self.cbuf2At(self.cpos - 1).*;
        }

        var i: i32 = 0;
        while (i < 3) : (i += 1) {
            self.run_pred[@intCast(i)] = 0;
            self.run_pred[@intCast(i + 3)] = 0;
            var st: i32 = 0;
            while (st < 10 and zz + st < 64) : (st += 1) {
                const zz2 = zz + st;
                var p: i32 = self.sumu[@intCast(ZU(zz2))] *% i +% self.sumv[@intCast(ZV(zz2))] *% (2 - i);
                const denom = @divTrunc((@as(i32, qtab[@intCast(q + zz2)]) + 1) *% 185 *% (16 + ZV(zz2)) *% (16 + ZU(zz2)), 128);
                p = @divTrunc(p, denom);
                if (zz2 == 0 and (norst or self.ls[@intCast(acomp)] == 64)) p -= self.cbuf2At(cpos_dc - self.ls[@intCast(acomp)]).*;
                p = (if (p < 0) @as(i32, -1) else 1) * ilg(iabs(p) +% 1);
                if (st == 0) {
                    self.adv_pred[@intCast(i)] = p;
                } else if (iabs(p) > iabs(self.adv_pred[@intCast(i)]) + 2 and iabs(self.adv_pred[@intCast(i)]) < 210) {
                    if (self.run_pred[@intCast(i)] == 0) self.run_pred[@intCast(i)] = st * 2 + b2i(p > 0);
                    if (iabs(p) > iabs(self.adv_pred[@intCast(i)]) + 21 and self.run_pred[@intCast(i + 3)] == 0) self.run_pred[@intCast(i + 3)] = st * 2 + b2i(p > 0);
                }
            }
        }
        x = 0;
        i = 0;
        while (i < 8) : (i += 1) x +%= b2i(ZU(zz) < i) *% self.sumu[@intCast(i)] +% b2i(ZV(zz) < i) *% self.sumv[@intCast(i)];
        x = @divTrunc((self.sumu[@intCast(ZU(zz))] *% (2 + ZU(zz)) +% self.sumv[@intCast(ZV(zz))] *% (2 + ZV(zz)) -% x *% 2) *% 4, ZU(zz) + ZV(zz) + 16);
        x = @divTrunc(x, (@as(i32, qtab[@intCast(q + zz)]) + 1) *% 185);
        if (zz == 0 and (norst or self.ls[@intCast(acomp)] == 64)) x -= self.cbuf2At(cpos_dc - self.ls[@intCast(acomp)]).*;
        self.adv_pred[3] = (if (x < 0) @as(i32, -1) else 1) * ilg(iabs(x) +% 1);

        i = 0;
        while (i < 4) : (i += 1) {
            const a: i32 = if ((i & 1) != 0) ZV(zz) else ZU(zz);
            const b: i32 = if ((i & 2) != 0) 2 else 1;
            if (a < b) {
                x = 65535;
            } else {
                const zz2 = self.zpos[@intCast(ZU(zz) + 8 * ZV(zz) - (if ((i & 1) != 0) @as(i32, 8) else 1) * b)];
                x = @divTrunc((@as(i32, qtab[@intCast(q + zz2)]) + 1) *% self.cbuf2At(cpos_dc + zz2).*, @as(i32, qtab[@intCast(q + zz)]) + 1);
                x = (if (x < 0) @as(i32, -1) else 1) * (ilg(iabs(x) +% 1) + (if (x != 0) @as(i32, 17) else 0));
            }
            self.lcp[@intCast(i)] = x;
        }
        if ((ZU(zz) * ZV(zz)) != 0) {
            var zz2 = self.zpos[@intCast(ZU(zz) + 8 * ZV(zz) - 9)];
            x = @divTrunc((@as(i32, qtab[@intCast(q + zz2)]) + 1) *% self.cbuf2At(cpos_dc + zz2).*, @as(i32, qtab[@intCast(q + zz)]) + 1);
            self.lcp[4] = (if (x < 0) @as(i32, -1) else 1) * (ilg(iabs(x) +% 1) + (if (x != 0) @as(i32, 17) else 0));

            zz2 = self.zpos[@intCast(8 * ZV(zz))];
            x = @divTrunc((@as(i32, qtab[@intCast(q + zz2)]) + 1) *% self.cbuf2At(cpos_dc + zz2).*, @as(i32, qtab[@intCast(q + zz)]) + 1);
            self.lcp[5] = (if (x < 0) @as(i32, -1) else 1) * (ilg(iabs(x) +% 1) + (if (x != 0) @as(i32, 17) else 0));

            zz2 = self.zpos[@intCast(ZU(zz))];
            x = @divTrunc((@as(i32, qtab[@intCast(q + zz2)]) + 1) *% self.cbuf2At(cpos_dc + zz2).*, @as(i32, qtab[@intCast(q + zz)]) + 1);
            self.lcp[6] = (if (x < 0) @as(i32, -1) else 1) * (ilg(iabs(x) +% 1) + (if (x != 0) @as(i32, 17) else 0));
        } else {
            self.lcp[4] = 65535;
            self.lcp[5] = 65535;
            self.lcp[6] = 65535;
        }

        var prev1: i32 = 0;
        var prev2: i32 = 0;
        var cnt1: i32 = 0;
        var cnt2: i32 = 0;
        var r: i32 = 0;
        var s: i32 = 0;
        self.prev_coef_rs = @as(i32, self.cbufAt(self.cpos - 64).*);
        i = 0;
        while (i < acomp) : (i += 1) {
            x = 0;
            x += self.cbuf2At(self.cpos - (acomp - i) * 64).*;
            if (zz == 0 and (norst or self.ls[@intCast(i)] == 64)) x -= self.cbuf2At(cpos_dc - (acomp - i) * 64 - self.ls[@intCast(i)]).*;
            if (self.color[@intCast(i)] == self.color[@intCast(acomp)] - 1) {
                prev1 += x;
                cnt1 += 1;
                r += @as(i32, self.cbufAt(self.cpos - (acomp - i) * 64).*) >> 4;
                s += @as(i32, self.cbufAt(self.cpos - (acomp - i) * 64).*) & 0xF;
            }
            if (self.color[@intCast(acomp)] > 1 and self.color[@intCast(i)] == self.color[0]) {
                prev2 += x;
                cnt2 += 1;
            }
        }
        if (cnt1 > 0) {
            prev1 = @divTrunc(prev1, cnt1);
            r = @divTrunc(r, cnt1);
            s = @divTrunc(s, cnt1);
            self.prev_coef_rs = (r << 4) | s;
        }
        if (cnt2 > 0) prev2 = @divTrunc(prev2, cnt2);
        self.prev_coef = (if (prev1 < 0) @as(i32, -1) else 1) * ilg(11 *% iabs(prev1) +% 1) + (cnt1 << 20);
        self.prev_coef2 = (if (prev2 < 0) @as(i32, -1) else 1) * ilg(11 *% iabs(prev2) +% 1);

        if (self.column == 0 and self.blockW[@intCast(acomp)] > 64 * acomp) {
            self.run_pred[1] = self.run_pred[2];
            self.run_pred[0] = 0;
            self.adv_pred[1] = self.adv_pred[2];
            self.adv_pred[0] = 0;
        }
        if (self.row == 0 and self.blockN[@intCast(acomp)] > 64 * acomp) {
            self.run_pred[1] = self.run_pred[0];
            self.run_pred[2] = 0;
            self.adv_pred[1] = self.adv_pred[0];
            self.adv_pred[2] = 0;
        }
    }
};

var jpeg_ptr: ?*Jpeg = null;

pub fn jpegModel(m: *core.Mixer) i32 {
    if (jpeg_ptr == null) jpeg_ptr = Jpeg.create(A());
    return jpeg_ptr.?.run(m);
}

// ================================ exeModel =================================

// Instruction field-packing constants (paq8 #defines).
const CodeShift: u5 = 3;
const CodeMask: u32 = 0xFF << CodeShift; // 0x7F8
const ClearCodeMask: u32 = ~CodeMask;
const PrefixMask: u32 = (1 << CodeShift) - 1; // 7
const OperandSizeOverride: u32 = 0x01 << (8 + CodeShift); // 0x800
const MultiByteOpcode: u32 = 0x02 << (8 + CodeShift); // 0x1000
const PrefixREX: u32 = 0x04 << (8 + CodeShift); // 0x2000
const Prefix38: u32 = 0x08 << (8 + CodeShift); // 0x4000
const Prefix3A: u32 = 0x10 << (8 + CodeShift); // 0x8000
const HasExtraFlags: u32 = 0x20 << (8 + CodeShift); // 0x10000
const HasModRM: u32 = 0x40 << (8 + CodeShift); // 0x20000
const ModRMShift: u5 = 7 + 8 + CodeShift; // 18
const SIBScaleShift: u5 = @intCast(@as(u32, ModRMShift) + 8 - 6); // 20
const RegDWordDisplacement: u32 = 0x01 << (8 + SIBScaleShift); // 0x10000000
const AddressMode: u32 = 0x02 << (8 + SIBScaleShift); // 0x20000000
const TypeShift: u5 = @intCast(2 + 8 + @as(u32, SIBScaleShift)); // 30
const CategoryShift: u5 = 5;
const CategoryMask: u32 = (1 << CategoryShift) - 1; // 31
const ModRM_mod: u32 = 0xC0;
const ModRM_reg: u32 = 0x38;
const ModRM_rm: u32 = 0x07;
const SIB_scale: u32 = 0xC0;
const SIB_base: u32 = 0x07;
const REX_w: u8 = 0x08;
const MinRequired: u32 = 8;
const CacheSize: u32 = 1 << 5;

// Prefixes / opcodes
const ES_OVERRIDE: u8 = 0x26;
const CS_OVERRIDE: u8 = 0x2E;
const SS_OVERRIDE: u8 = 0x36;
const DS_OVERRIDE: u8 = 0x3E;
const FS_OVERRIDE: u8 = 0x64;
const GS_OVERRIDE: u8 = 0x65;
const AD_OVERRIDE: u8 = 0x67;
const WAIT_FPU: u8 = 0x9B;
const LOCK: u8 = 0xF0;
const REP_N_STR: u8 = 0xF2;
const REP_STR: u8 = 0xF3;
const OP_2BYTE: u8 = 0x0f;
const OP_OSIZE: u8 = 0x66;
const OP_CALLF: u8 = 0x9a;
const OP_RETN: u8 = 0xc3;
const OP_ENTER: u8 = 0xc8;
const OP_CALLN: u8 = 0xe8;
const OP_JMPF: u8 = 0xea;

// ExeState values
const St_Start: i32 = 0;
const St_Pref_Op_Size: i32 = 1;
const St_Pref_MultiByte_Op: i32 = 2;
const St_ParseFlags: i32 = 3;
const St_ExtraFlags: i32 = 4;
const St_ReadModRM: i32 = 5;
const St_Read_OP3_38: i32 = 6;
const St_Read_OP3_3A: i32 = 7;
const St_ReadSIB: i32 = 8;
const St_Read8: i32 = 9;
const St_Read16: i32 = 10;
const St_Read32: i32 = 11;
const St_Read8_ModRM: i32 = 12;
const St_Read16_f: i32 = 13;
const St_Read32_ModRM: i32 = 14;
const St_Error: i32 = 15;

const Instruction = struct {
    Data: u32 = 0,
    Prefix: u8 = 0,
    Code: u8 = 0,
    ModRM: u8 = 0,
    SIB: u8 = 0,
    REX: u8 = 0,
    Flags: u8 = 0,
    BytesRead: u8 = 0,
    Size: u8 = 0,
    Category: u8 = 0,
    MustCheckREX: bool = false,
    Decoding: bool = false,
    o16: bool = false,
    imm8: bool = false,
};

fn isInvalidX64Op(op: u8) bool {
    for (xt.InvalidX64Ops) |v| if (op == v) return true;
    return false;
}
fn isValidX64Prefix(prefix: u8) bool {
    for (xt.X64Prefixes) |v| if (prefix == v) return true;
    return ((prefix >= 0x40 and prefix <= 0x4F) or (prefix >= 0x64 and prefix <= 0x67));
}

fn processMode(op: *Instruction, state: *i32) void {
    if ((op.Flags & xt.fMODE) == xt.fAM) {
        op.Data |= AddressMode;
        op.BytesRead = 0;
        switch (op.Flags & xt.fTYPE) {
            xt.fDR => {
                op.Data |= (2 << TypeShift);
                op.Data |= (1 << TypeShift);
                state.* = St_Read32;
            },
            xt.fDA => {
                op.Data |= (1 << TypeShift);
                state.* = St_Read32;
            },
            xt.fAD => {
                state.* = St_Read32;
            },
            xt.fBR => {
                op.Data |= (2 << TypeShift);
                state.* = St_Read8;
            },
            else => {},
        }
    } else {
        switch (op.Flags & xt.fTYPE) {
            xt.fBI => state.* = St_Read8,
            xt.fWI => {
                state.* = St_Read16;
                op.Data |= (1 << TypeShift);
                op.BytesRead = 0;
            },
            xt.fDI => {
                op.imm8 = ((op.REX & REX_w) > 0 and (op.Code & 0xF8) == 0xB8);
                if (!op.o16 or op.imm8) {
                    state.* = St_Read32;
                    op.Data |= (2 << TypeShift);
                } else {
                    state.* = St_Read16;
                    op.Data |= (3 << TypeShift);
                }
                op.BytesRead = 0;
            },
            else => state.* = St_Start,
        }
    }
}

fn processFlags2(op: *Instruction, state: *i32) void {
    if ((op.Flags & xt.fMODE) == xt.fMR and state.* != St_ExtraFlags) {
        state.* = St_ReadModRM;
        return;
    }
    processMode(op, state);
}

fn processFlags(op: *Instruction, state: *i32) void {
    if (op.Code == OP_CALLF or op.Code == OP_JMPF or op.Code == OP_ENTER) {
        op.BytesRead = 0;
        state.* = St_Read16_f;
        return;
    }
    processFlags2(op, state);
}

fn checkFlags(op: *Instruction, state: *i32) void {
    if (op.Flags == xt.fMEXTRA) {
        state.* = St_ExtraFlags;
    } else if (op.Flags == xt.fERR) {
        op.* = .{};
        state.* = St_Error;
    } else {
        processFlags(op, state);
    }
}

fn readFlags(op: *Instruction, state: *i32) void {
    op.Flags = xt.Table1[op.Code];
    op.Category = xt.TypeOp1[op.Code];
    checkFlags(op, state);
}

fn processModRM(op: *Instruction, state: *i32) void {
    if ((@as(u32, op.ModRM) & ModRM_mod) == 0x40) {
        state.* = St_Read8_ModRM;
    } else if ((@as(u32, op.ModRM) & ModRM_mod) == 0x80 or (@as(u32, op.ModRM) & (ModRM_mod | ModRM_rm)) == 0x05 or
        (op.ModRM < 0x40 and (@as(u32, op.SIB) & SIB_base) == 0x05))
    {
        state.* = St_Read32_ModRM;
        op.BytesRead = 0;
    } else {
        processMode(op, state);
    }
}

fn applyCodeAndSetFlag(op: *Instruction, flag: u32) void {
    op.Data &= ClearCodeMask;
    op.Data |= (@as(u32, op.Code) << CodeShift) | flag;
}

pub const Exe = struct {
    cm: maps.ContextMap2,
    Cache_Op: [CacheSize]u32,
    Cache_Index: u32,
    StateBH: [256]u32,
    pState: i32,
    State: i32,
    Op: Instruction,
    TotalOps: u32,
    OpMask: u32,
    OpCategMask: u32,
    Context: u32,
    BrkPoint: u32,
    BrkCtx: u32,
    Valid: bool,

    const N1: i32 = 10;
    const N2: i32 = 10;

    fn create(a: std.mem.Allocator) *Exe {
        const self = a.create(Exe) catch unreachable;
        @memset(std.mem.asBytes(self), 0);
        self.pState = St_Start;
        self.State = St_Start;
        self.Op = .{};
        self.Valid = false;
        self.cm = maps.ContextMap2.init(a, core.MEM() * 2, @intCast(N1 + N2));
        return self;
    }

    inline fn opN(self: *Exe, n: u32) u32 {
        return self.Cache_Op[(self.Cache_Index -% n) & (CacheSize - 1)];
    }

    // pref(i) : x86 prefix class of buf(i)
    inline fn pref(i: i32) i32 {
        return b2i(bufRel(i) == 0x0f) + 2 * b2i(bufRel(i) == 0x66) + 3 * b2i(bufRel(i) == 0x67);
    }

    // execxt(i, x) : sparse parsing context at buf(i)
    fn execxt(i_in: i32, x: i32) u32 {
        var i = i_in;
        var prefix: i32 = 0;
        var opcode: i32 = 0;
        var modrm: i32 = 0;
        var sib: i32 = 0;
        if (i != 0) {
            prefix += 4 * pref(i);
            i -= 1;
        }
        if (i != 0) {
            prefix += pref(i);
            i -= 1;
        }
        if (i != 0) {
            opcode += bufRel(i);
            i -= 1;
        }
        if (i != 0) {
            modrm += bufRel(i) & @as(i32, @intCast(ModRM_mod | ModRM_rm));
            i -= 1;
        }
        if (i != 0 and (modrm & @as(i32, @intCast(ModRM_rm))) == 4 and modrm < @as(i32, @intCast(ModRM_mod))) {
            sib = bufRel(i) & @as(i32, @intCast(SIB_scale));
        }
        return @bitCast(prefix | opcode << 4 | modrm << 12 | x << 20 | sib << (28 - 6));
    }

    fn run(self: *Exe, m: *core.Mixer, Forced: bool, Stats: ?*core.ModelStats) bool {
        const bpos = core.bpos;
        const c4 = core.c4;
        const blpos = core.blpos;
        const c0 = core.c0;

        if (bpos == 0) {
            self.pState = self.State;
            const B: u8 = @truncate(c4);
            self.Op.Size +%= 1;
            switch (self.State) {
                St_Start, St_Error => {
                    var Skip = false;
                    if (self.Op.MustCheckREX) {
                        self.Op.MustCheckREX = false;
                        if (!isInvalidX64Op(B) and !isValidX64Prefix(B)) {
                            self.Op.REX = self.Op.Code;
                            self.Op.Code = B;
                            self.Op.Data = PrefixREX | (@as(u32, self.Op.Code) << CodeShift) | (self.Op.Data & PrefixMask);
                            Skip = true;
                        }
                    }

                    self.Op.ModRM = 0;
                    self.Op.SIB = 0;
                    self.Op.REX = 0;
                    self.Op.Flags = 0;
                    self.Op.BytesRead = 0;
                    var did_break = false;
                    if (!Skip) {
                        self.Op.Code = B;
                        self.Op.MustCheckREX = ((self.Op.Code & 0xF0) == 0x40) and (!(self.Op.Decoding and ((self.Op.Data & PrefixMask) == 1)));

                        self.Op.Prefix = @intCast(b2i(self.Op.Code == ES_OVERRIDE or self.Op.Code == CS_OVERRIDE or self.Op.Code == SS_OVERRIDE or self.Op.Code == DS_OVERRIDE) +
                            b2i(self.Op.Code == FS_OVERRIDE) * 2 +
                            b2i(self.Op.Code == GS_OVERRIDE) * 3 +
                            b2i(self.Op.Code == AD_OVERRIDE) * 4 +
                            b2i(self.Op.Code == WAIT_FPU) * 5 +
                            b2i(self.Op.Code == LOCK) * 6 +
                            b2i(self.Op.Code == REP_N_STR or self.Op.Code == REP_STR) * 7);

                        if (!self.Op.Decoding) {
                            self.TotalOps +%= @as(u32, @bitCast(b2i(self.Op.Data != 0) -
                                b2i(self.Cache_Index != 0 and self.Cache_Op[self.Cache_Index & (CacheSize - 1)] != 0)));
                            self.OpMask = (self.OpMask << 1) | @as(u32, @intCast(b2i(self.State != St_Error)));
                            self.OpCategMask = (self.OpCategMask << CategoryShift) | @as(u32, self.Op.Category);
                            self.Op.Size = 0;

                            self.Cache_Op[self.Cache_Index & (CacheSize - 1)] = self.Op.Data;
                            self.Cache_Index +%= 1;

                            if (self.Op.Prefix == 0) {
                                self.Op.Data = @as(u32, self.Op.Code) << CodeShift;
                            } else {
                                self.Op.Data = @as(u32, self.Op.Prefix);
                                self.Op.Category = xt.TypeOp1[self.Op.Code];
                                self.Op.Decoding = true;
                                self.BrkPoint = 0;
                                self.BrkCtx = @truncate(maps.hash(.{ 1 + self.BrkPoint, self.Op.Prefix, self.OpCategMask & CategoryMask }));
                                did_break = true;
                            }
                        } else {
                            if (self.Op.Prefix == 0) {
                                self.Op.Data |= (@as(u32, self.Op.Code) << CodeShift);
                                self.Op.Decoding = false;
                            } else {
                                self.Op.Data = @as(u32, self.Op.Prefix);
                                self.Op.Category = xt.TypeOp1[self.Op.Code];
                                self.BrkPoint = 1;
                                self.BrkCtx = @truncate(maps.hash(.{ 1 + self.BrkPoint, self.Op.Prefix, self.OpCategMask & CategoryMask }));
                                did_break = true;
                            }
                        }
                    }

                    if (!did_break) {
                        self.Op.o16 = (self.Op.Code == OP_OSIZE);
                        if (self.Op.o16) {
                            self.State = St_Pref_Op_Size;
                        } else if (self.Op.Code == OP_2BYTE) {
                            self.State = St_Pref_MultiByte_Op;
                        } else {
                            readFlags(&self.Op, &self.State);
                        }
                        self.BrkPoint = 2;
                        self.BrkCtx = @truncate(maps.hash(.{ 1 + self.BrkPoint, self.State, self.Op.Code, (self.OpCategMask & CategoryMask), self.opN(1) & ((ModRM_mod | ModRM_reg | ModRM_rm) << ModRMShift) }));
                    }
                },
                St_Pref_Op_Size => {
                    self.Op.Code = B;
                    applyCodeAndSetFlag(&self.Op, OperandSizeOverride);
                    readFlags(&self.Op, &self.State);
                    self.BrkPoint = 3;
                    self.BrkCtx = @truncate(maps.hash(.{ 1 + self.BrkPoint, self.State }));
                },
                St_Pref_MultiByte_Op => {
                    self.Op.Code = B;
                    self.Op.Data |= MultiByteOpcode;
                    if (self.Op.Code == 0x38) {
                        self.State = St_Read_OP3_38;
                    } else if (self.Op.Code == 0x3A) {
                        self.State = St_Read_OP3_3A;
                    } else {
                        applyCodeAndSetFlag(&self.Op, 0);
                        self.Op.Flags = xt.Table2[self.Op.Code];
                        self.Op.Category = xt.TypeOp2[self.Op.Code];
                        checkFlags(&self.Op, &self.State);
                    }
                    self.BrkPoint = 4;
                    self.BrkCtx = @truncate(maps.hash(.{ 1 + self.BrkPoint, self.State }));
                },
                St_ParseFlags => {
                    processFlags(&self.Op, &self.State);
                    self.BrkPoint = 5;
                    self.BrkCtx = @truncate(maps.hash(.{ 1 + self.BrkPoint, self.State }));
                },
                St_ExtraFlags, St_ReadModRM => {
                    self.Op.ModRM = B;
                    self.Op.Data |= (@as(u32, self.Op.ModRM) << ModRMShift) | HasModRM;
                    self.Op.SIB = 0;
                    var handled = false;
                    if (self.Op.Flags == xt.fMEXTRA) {
                        self.Op.Data |= HasExtraFlags;
                        const ei = ((@as(i32, self.Op.ModRM) >> 3) & 0x07) | ((@as(i32, self.Op.Code) & 0x01) << 3) | ((@as(i32, self.Op.Code) & 0x08) << 1);
                        self.Op.Flags = xt.TableX[@intCast(ei)];
                        self.Op.Category = xt.TypeOpX[@intCast(ei)];
                        if (self.Op.Flags == xt.fERR) {
                            self.Op = .{};
                            self.State = St_Error;
                            self.BrkPoint = 6;
                            self.BrkCtx = @truncate(maps.hash(.{ 1 + self.BrkPoint, self.State }));
                            handled = true;
                        } else {
                            processFlags(&self.Op, &self.State);
                            self.BrkPoint = 7;
                            self.BrkCtx = @truncate(maps.hash(.{ 1 + self.BrkPoint, self.State }));
                            handled = true;
                        }
                    }
                    if (!handled) {
                        if ((self.Op.ModRM & @as(u8, @intCast(ModRM_rm))) == 4 and self.Op.ModRM < @as(u8, @intCast(ModRM_mod))) {
                            self.State = St_ReadSIB;
                            self.BrkPoint = 8;
                            self.BrkCtx = @truncate(maps.hash(.{ 1 + self.BrkPoint, self.State }));
                        } else {
                            processModRM(&self.Op, &self.State);
                            self.BrkPoint = 9;
                            self.BrkCtx = @truncate(maps.hash(.{ 1 + self.BrkPoint, self.State, self.Op.Code }));
                        }
                    }
                },
                St_Read_OP3_38, St_Read_OP3_3A => {
                    self.Op.Code = B;
                    applyCodeAndSetFlag(&self.Op, Prefix38 << @intCast(self.State - St_Read_OP3_38));
                    if (self.State == St_Read_OP3_38) {
                        self.Op.Flags = xt.Table3_38[self.Op.Code];
                        self.Op.Category = xt.TypeOp3_38[self.Op.Code];
                    } else {
                        self.Op.Flags = xt.Table3_3A[self.Op.Code];
                        self.Op.Category = xt.TypeOp3_3A[self.Op.Code];
                    }
                    checkFlags(&self.Op, &self.State);
                    self.BrkPoint = 10;
                    self.BrkCtx = @truncate(maps.hash(.{ 1 + self.BrkPoint, self.State }));
                },
                St_ReadSIB => {
                    self.Op.SIB = B;
                    self.Op.Data |= ((@as(u32, self.Op.SIB) & SIB_scale) << SIBScaleShift);
                    processModRM(&self.Op, &self.State);
                    self.BrkPoint = 11;
                    self.BrkCtx = @truncate(maps.hash(.{ 1 + self.BrkPoint, self.State, self.Op.SIB & @as(u8, @intCast(SIB_scale)) }));
                },
                St_Read8, St_Read16, St_Read32 => {
                    self.Op.BytesRead +%= 1;
                    const thresh: i32 = (2 * (self.State - St_Read8)) << @intCast(b2i(self.Op.imm8));
                    if (@as(i32, self.Op.BytesRead) >= thresh) {
                        self.Op.BytesRead = 0;
                        self.Op.imm8 = false;
                        self.State = St_Start;
                    }
                    const extra: i32 = (if (self.Op.BytesRead > 1) (bufRel(@as(i32, self.Op.BytesRead)) << 8) else 0) | (if (self.Op.BytesRead != 0) @as(i32, B) else 0);
                    self.BrkPoint = 12;
                    self.BrkCtx = @truncate(maps.hash(.{ 1 + self.BrkPoint, self.State, @as(u32, self.Op.Flags) & xt.fMODE, self.Op.BytesRead, extra }));
                },
                St_Read8_ModRM => {
                    processMode(&self.Op, &self.State);
                    self.BrkPoint = 13;
                    self.BrkCtx = @truncate(maps.hash(.{ 1 + self.BrkPoint, self.State }));
                },
                St_Read16_f => {
                    self.Op.BytesRead +%= 1;
                    if (self.Op.BytesRead == 2) {
                        self.Op.BytesRead = 0;
                        processFlags2(&self.Op, &self.State);
                    }
                    self.BrkPoint = 14;
                    self.BrkCtx = @truncate(maps.hash(.{ 1 + self.BrkPoint, self.State }));
                },
                St_Read32_ModRM => {
                    self.Op.Data |= RegDWordDisplacement;
                    self.Op.BytesRead +%= 1;
                    if (self.Op.BytesRead == 4) {
                        self.Op.BytesRead = 0;
                        processMode(&self.Op, &self.State);
                    }
                    self.BrkPoint = 15;
                    self.BrkCtx = @truncate(maps.hash(.{ 1 + self.BrkPoint, self.State }));
                },
                else => {},
            }

            self.Valid = (self.TotalOps > 2 * MinRequired) and ((self.OpMask & ((@as(u32, 1) << @intCast(MinRequired)) - 1)) == ((@as(u32, 1) << @intCast(MinRequired)) - 1));
            self.Context = @as(u32, @intCast(self.State)) +% 16 *% @as(u32, self.Op.BytesRead) +% 16 *% @as(u32, self.Op.REX & REX_w);
            self.StateBH[self.Context] = (self.StateBH[self.Context] << 8) | @as(u32, B);

            if (self.Valid or Forced) {
                var mask: i32 = 0;
                var count0: i32 = 0;
                var i: i32 = 0;
                while (i < N1) : (i += 1) {
                    if (i > 1) {
                        mask = mask * 2 + b2i(bufRel(i - 1) == 0);
                        count0 += mask & 1;
                    }
                    const j: i32 = if (i < 4) i + 1 else 5 + (i - 4) * (2 + b2i(i > 6));
                    self.cm.set(maps.hash(.{ i, execxt(j, bufRel(1) * b2i(j > 6)), ((@as(i32, 1) << @intCast(N1)) | mask) * b2i(@divTrunc(count0 * N1, 2) >= i), (0x08 | (blpos & 0x07)) * b2i(i < 4) }));
                }

                self.cm.set(@as(u64, self.BrkCtx));

                var hidx: i32 = N1; // continues as ++i in C++ (i was N1 after loop)
                var maskU: u32 = PrefixMask | (0xF8 << CodeShift) | MultiByteOpcode | Prefix38 | Prefix3A;
                hidx += 1;
                self.cm.set(maps.hash(.{ hidx, self.opN(1) & (maskU | RegDWordDisplacement | AddressMode), self.State + 16 * @as(i32, self.Op.BytesRead), self.Op.Data & maskU, self.Op.REX, self.Op.Category }));

                maskU = 0x04 | (0xFE << CodeShift) | MultiByteOpcode | Prefix38 | Prefix3A | ((ModRM_mod | ModRM_reg) << ModRMShift);
                hidx += 1;
                self.cm.set(maps.hash(.{ hidx, self.opN(1) & maskU, self.opN(2) & maskU, self.opN(3) & maskU, self.Context +% 256 *% @as(u32, @intCast(b2i((self.Op.ModRM & @as(u8, @intCast(ModRM_mod))) == @as(u8, @intCast(ModRM_mod))))), self.Op.Data & ((maskU | PrefixREX) ^ (ModRM_mod << ModRMShift)) }));

                maskU = 0x04 | CodeMask;
                hidx += 1;
                self.cm.set(maps.hash(.{ hidx, self.opN(1) & maskU, self.opN(2) & maskU, self.opN(3) & maskU, self.opN(4) & maskU, (self.Op.Data & maskU) | (@as(u32, @intCast(self.State)) << 11) | (@as(u32, self.Op.BytesRead) << 15) }));

                maskU = 0x04 | (0xFC << CodeShift) | MultiByteOpcode | Prefix38 | Prefix3A;
                hidx += 1;
                self.cm.set(maps.hash(.{ hidx, self.State + 16 * @as(i32, self.Op.BytesRead), self.Op.Data & maskU, @as(i32, self.Op.Category) * 8 + @as(i32, @bitCast(self.OpMask & 0x07)), self.Op.Flags, b2i((self.Op.SIB & @as(u8, @intCast(SIB_base))) == 5) * 4 + b2i((self.Op.ModRM & @as(u8, @intCast(ModRM_reg))) == @as(u8, @intCast(ModRM_reg))) * 2 + b2i((self.Op.ModRM & @as(u8, @intCast(ModRM_mod))) == 0) }));

                maskU = PrefixMask | CodeMask | OperandSizeOverride | MultiByteOpcode | PrefixREX | Prefix38 | Prefix3A | HasExtraFlags | HasModRM | ((ModRM_mod | ModRM_rm) << ModRMShift);
                hidx += 1;
                self.cm.set(maps.hash(.{ hidx, self.Op.Data & maskU, self.State + 16 * @as(i32, self.Op.BytesRead), self.Op.Flags }));

                maskU = PrefixMask | CodeMask | OperandSizeOverride | MultiByteOpcode | Prefix38 | Prefix3A | HasExtraFlags | HasModRM;
                hidx += 1;
                self.cm.set(maps.hash(.{ hidx, self.opN(1) & maskU, self.State, @as(i32, self.Op.BytesRead) * 2 + b2i((self.Op.REX & REX_w) > 0), self.Op.Data & @as(u32, @as(u16, @truncate(maskU ^ OperandSizeOverride))) }));

                maskU = 0x04 | (0xFE << CodeShift) | MultiByteOpcode | Prefix38 | Prefix3A | (ModRM_reg << ModRMShift);
                hidx += 1;
                self.cm.set(maps.hash(.{ hidx, self.opN(1) & maskU, self.opN(2) & maskU, self.State + 16 * @as(i32, self.Op.BytesRead), self.Op.Data & (maskU | PrefixMask | CodeMask) }));

                hidx += 1;
                self.cm.set(maps.hash(.{ hidx, self.State + 16 * @as(i32, self.Op.BytesRead) }));

                hidx += 1;
                self.cm.set(maps.hash(.{
                    hidx,
                    (0x100 | @as(i32, B)) * b2i(self.Op.BytesRead > 0),
                    self.State + 16 * self.pState + 256 * @as(i32, self.Op.BytesRead),
                    b2i((self.Op.Flags & xt.fMODE) == xt.fAM) * 16 + @as(i32, self.Op.REX & REX_w) + b2i(self.Op.o16) * 4 + b2i((self.Op.Code & 0xFE) == 0xE8) * 2 + b2i((self.Op.Data & MultiByteOpcode) != 0 and (self.Op.Code & 0xF0) == 0x80),
                }));
            }
        }

        if (self.Valid or Forced) {
            _ = self.cm.mix(m);
        } else {
            var i: i32 = 0;
            while (i < (N1 + N2) * 7) : (i += 1) m.add(0);
        }
        const sh = self.StateBH[self.Context];
        const s: u8 = @truncate(((sh >> @intCast(28 - bpos)) & 0x08) |
            ((sh >> @intCast(21 - bpos)) & 0x04) |
            ((sh >> @intCast(14 - bpos)) & 0x02) |
            ((sh >> @intCast(7 - bpos)) & 0x01) |
            (@as(u32, @intCast(b2i(self.Op.Category == xt.OP_GEN_BRANCH))) << 4) |
            (@as(u32, @intCast(b2i((c0 & ((@as(i32, 1) << @intCast(bpos)) - 1)) == 0))) << 5));

        m.set(@as(i32, @bitCast(self.Context *% 4 +% @as(u32, s >> 4))), 1024);
        m.set(self.State * 64 + bpos * 8 + b2i(self.Op.BytesRead > 0) * 4 + @as(i32, s >> 4), 1024);
        m.set(@as(i32, @bitCast((self.BrkCtx & 0x1FF) | (@as(u32, s & 0x20) << 4))), 1024);
        m.set(@bitCast(maps.finalize64(maps.hash(.{ self.Op.Code, self.State, self.opN(1) & CodeMask }), 13)), 8192);
        m.set(@bitCast(maps.finalize64(maps.hash(.{ self.State, bpos, self.Op.Code, self.Op.BytesRead }), 13)), 8192);
        m.set(@bitCast(maps.finalize64(maps.hash(.{ self.State, (bpos << 2) | (c0 & 3), self.OpCategMask & CategoryMask, (b2i(self.Op.Category == xt.OP_GEN_BRANCH) << 2) | (b2i((self.Op.Flags & xt.fMODE) == xt.fAM) << 1) | b2i(self.Op.BytesRead > 0) }), 13)), 8192);

        if (Stats) |st| st.x86_64 = @as(u32, @intCast(b2i(self.Valid))) | (self.Context << 1) | (@as(u32, s) << 9);
        return self.Valid;
    }
};

var exe_ptr: ?*Exe = null;

pub fn exeModel(m: *core.Mixer, Forced: bool, Stats: ?*core.ModelStats) bool {
    if (exe_ptr == null) exe_ptr = Exe.create(A());
    return exe_ptr.?.run(m, Forced, Stats);
}

/// debug: last-computed BrkCtx / BrkPoint / State / Context / OpMask / TotalOps
pub fn exeDbg(out: *[6]u64) void {
    const e = exe_ptr.?;
    out[0] = @as(u64, e.BrkCtx);
    out[1] = @as(u64, e.BrkPoint);
    out[2] = @bitCast(@as(i64, e.State));
    out[3] = @as(u64, e.Context);
    out[4] = @as(u64, e.OpMask);
    out[5] = @as(u64, e.TotalOps);
}
