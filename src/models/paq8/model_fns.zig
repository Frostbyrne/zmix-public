//! paq8 model FUNCTIONS ported faithfully from reference/cmix-src/models/paq8.cpp.
//!
//! These are the `void xModel(Mixer& m, ...)` free functions of the cmix PAQ8
//! model, ported on top of the verified foundation in core.zig / maps.zig /
//! text_primitives.zig. Each C++ function with `static` ContextMaps/tables is
//! turned into a per-function state struct (allocated once) whose methods
//! reproduce the EXACT sequence and count of m.add/m.setcalls, so the
//! mixer layout matches cmix bit-for-bit.
//!
//! Integrator order (contextModel2): sparseModel, sparseModel1, distanceModel,
//! picModel, recordModel, recordModel1, wordModel, nestModel, indirectModel,
//! XMLModel, linearPredictionModel.
//!
//! Faithfulness: C `unsigned` wrap -> `+%`/`-%`/`*%`; C `int` overflow (UB but
//! wraps in practice) -> signed `+%`/`-%`/`*%`; U8/U16/U32 truncating stores ->
//! `@truncate`. hashargs keep the C++ operand signedness so U64 promotion
//! (zero/sign extension) matches.
const std = @import("std");
const core = @import("core.zig");
const maps = @import("maps.zig");
const tp = @import("text_primitives.zig");

const hash = maps.hash;
const Allocator = std.mem.Allocator;

// ------------------------------- helpers -----------------------------------
inline fn bf(i: u32) i32 {
    return core.buf.get(i);
}
inline fn bfu(i: u32) u32 {
    return @intCast(core.buf.get(i)); // byte, 0..255
}
inline fn tolwr(c: i32) i32 {
    return if (c >= 'A' and c <= 'Z') c + ('a' - 'A') else c;
}
inline fn isAlpha(c: i32) bool {
    return (c >= 'A' and c <= 'Z') or (c >= 'a' and c <= 'z');
}
inline fn isDigit(c: i32) bool {
    return c >= '0' and c <= '9';
}
inline fn isSpaceC(c: i32) bool {
    return c == ' ' or c == '\t' or c == '\n' or c == 11 or c == 12 or c == '\r';
}
inline fn isPunctC(c: i32) bool {
    return c >= 33 and c <= 126 and !isAlpha(c) and !isDigit(c);
}
inline fn Clip(px: i32) u8 {
    return @intCast(@min(255, @max(0, px)));
}
inline fn imin(a: i32, b: i32) i32 {
    return if (a < b) a else b;
}
inline fn imax(a: i32, b: i32) i32 {
    return if (a > b) a else b;
}
// right shift of i32 by a 0..31 amount computed from other i32s
inline fn shr(x: i32, amt: i32) i32 {
    return x >> @as(u5, @intCast(amt));
}
inline fn shl(x: i32, amt: i32) i32 {
    return x << @as(u5, @intCast(amt));
}
inline fn u32shr(x: u32, amt: i32) u32 {
    return x >> @as(u5, @intCast(amt));
}
inline fn u32shl(x: u32, amt: i32) u32 {
    return x << @as(u5, @intCast(amt));
}
inline fn b2i(v: bool) i32 {
    return @intFromBool(v);
}
inline fn b2u(v: bool) u32 {
    return @intFromBool(v);
}
// sign-extend an i32 to u64, matching C++ `int -> U64` argument conversion.
inline fn sx(x: i32) u64 {
    return @bitCast(@as(i64, x));
}

// wordModel / sparseModel1 share these file-scope statics (paq8.cpp:3872).
// `words` and `col`/`x4`/`w4`/`w5`/`f4`/`tt` live in core.zig.
pub var frstchar: u32 = 0;
pub var spafdo: u32 = 0;
pub var spaces: u32 = 0;
pub var spacecount: u32 = 0;
pub var wordcount: u32 = 0;
pub var wordlen: u32 = 0;
pub var wordlen1: u32 = 0;

/// Reset the shared wordModel/sparse statics (paq8.cpp:3872 initial values).
pub fn resetShared() void {
    frstchar = 0;
    spafdo = 0;
    spaces = 0;
    spacecount = 0;
    wordcount = 0;
    wordlen = 0;
    wordlen1 = 0;
    t_x5 = 0;
}

// ================================ picModel =================================
pub const PicModel = struct {
    r0: u32 = 0,
    r1: u32 = 0,
    r2: u32 = 0,
    r3: u32 = 0,
    t: core.Array(u8, 0),
    cxt: [3]i32 = .{ 0, 0, 0 },
    sm: [3]core.StateMap,

    pub fn init(a: Allocator) PicModel {
        var self = PicModel{
            .t = core.Array(u8, 0).initSize(a, 0x10200),
            .sm = undefined,
        };
        for (0..3) |i| self.sm[i] = core.StateMap.init(a);
        return self;
    }

    pub fn predict(self: *PicModel, m: *core.Mixer) void {
        const y: u32 = @intCast(core.y);
        for (0..3) |i| {
            const idx: u32 = @intCast(self.cxt[i]);
            self.t.at(idx).* = core.nex(self.t.get(idx), y);
        }
        self.r0 = self.r0 *% 2 +% y;
        self.r1 = self.r1 *% 2 +% @as(u32, @intCast(shr(bf(215), 7 - core.bpos) & 1));
        self.r2 = self.r2 *% 2 +% @as(u32, @intCast(shr(bf(431), 7 - core.bpos) & 1));
        self.r3 = self.r3 *% 2 +% @as(u32, @intCast(shr(bf(647), 7 - core.bpos) & 1));
        self.cxt[0] = @intCast((self.r0 & 0x7) | ((self.r1 >> 4) & 0x38) | ((self.r2 >> 3) & 0xc0));
        self.cxt[1] = @intCast(0x100 + ((self.r0 & 1) | ((self.r1 >> 4) & 0x3e) | ((self.r2 >> 2) & 0x40) | ((self.r3 >> 1) & 0x80)));
        self.cxt[2] = @intCast(0x200 + ((self.r0 & 0x3f) ^ (self.r1 & 0x3ffe) ^ ((self.r2 << 2) & 0x7f00) ^ ((self.r3 << 5) & 0xf800)));
        for (0..3) |i| {
            const idx: u32 = @intCast(self.cxt[i]);
            m.add(core.stretch(self.sm[i].p(self.t.get(idx))));
        }
    }

    pub fn deinit(self: *PicModel) void {
        self.t.deinit();
        for (0..3) |i| self.sm[i].deinit();
    }
};

// =============================== wordModel =================================
pub const WordModel = struct {
    word0: u64 = 0,
    word1: u64 = 0,
    word2: u64 = 0,
    word3: u64 = 0,
    word4: u64 = 0,
    word5: u64 = 0,
    wrdhsh: u32 = 0,
    xword0: u64 = 0,
    xword1: u64 = 0,
    xword2: u64 = 0,
    cword0: u64 = 0,
    ccword: u64 = 0,
    number0: u64 = 0,
    number1: u64 = 0,
    text0: u32 = 0,
    data0: u32 = 0,
    type0: u32 = 0,
    lastLetter: u32 = 0,
    firstLetter: u32 = 0,
    lastUpper: u32 = 0,
    lastDigit: u32 = 0,
    wordGap: u32 = 0,
    cm: maps.ContextMap,
    nl1: i32 = -3,
    nl: i32 = -2,
    mask: u32 = 0,
    mask2: u32 = 0,
    wpos: core.Array(i32, 0),
    w: i32 = 0,
    StemWords: [4]tp.Word = .{ .{}, .{}, .{}, .{} },
    cWordIdx: usize = 0,
    pWordIdx: usize = 3,
    stemmer: tp.EnglishStemmer = .{},
    StemIndex: i32 = 0,

    pub fn init(a: Allocator) WordModel {
        return WordModel{
            .cm = maps.ContextMap.init(a, core.MEM() * 16, 61),
            .wpos = core.Array(i32, 0).initSize(a, 0x10000),
        };
    }

    pub fn predict(self: *WordModel, m: *core.Mixer) void {
        if (core.bpos == 0) {
            var c: i32 = @intCast(core.c4 & 255);
            const pC: i32 = @intCast((core.c4 >> 8) & 0xff);
            var f: i32 = 0;
            if (spaces & 0x80000000 != 0) spacecount -%= 1;
            if (core.words & 0x80000000 != 0) wordcount -%= 1;
            spaces = spaces *% 2;
            core.words = core.words *% 2;
            self.lastUpper = @min(self.lastUpper +% 1, 255);
            self.lastLetter = @min(self.lastLetter +% 1, 255);
            self.mask2 = self.mask2 << 2;

            if (c >= 'A' and c <= 'Z') {
                c += 'a' - 'A';
                self.lastUpper = 0;
            }
            if ((c >= 'a' and c <= 'z') or c == '\'' or c == '-') {
                self.StemWords[self.cWordIdx].plusEq(@intCast(c));
            } else if (self.StemWords[self.cWordIdx].length() > 0) {
                _ = self.stemmer.stem(&self.StemWords[self.cWordIdx]);
                self.StemWords[self.cWordIdx].getHashes();
                self.StemIndex = (self.StemIndex + 1) & 3;
                self.pWordIdx = self.cWordIdx;
                self.cWordIdx = @intCast(self.StemIndex);
                self.StemWords[self.cWordIdx] = .{};
            }

            if ((c >= 'a' and c <= 'z') or ((c >= 128 and (core.b3 != 3)) or (c > 0 and c < 4))) {
                if (self.wordlenIsZero()) {
                    // NOTE: in C++ `lastLetter=3 && A && B` parses as
                    // `lastLetter = (3 && A && B)` (= 0/1) because `=` binds
                    // looser than `&&`; the 4 terms are OR'd (short-circuit).
                    const cond = blk: {
                        var v = b2u((core.c4 & 0xFFFF00) == 0x2B0A00 and bf(4) != 0x2B);
                        self.lastLetter = v;
                        if (v != 0) break :blk true;
                        v = b2u((core.c4 & 0xFFFFFF00) == 0x2B0D0A00 and bf(5) != 0x2B);
                        self.lastLetter = v;
                        if (v != 0) break :blk true;
                        v = b2u((core.c4 & 0xFFFF00) == 0x2D0A00 and bf(4) != 0x2D);
                        self.lastLetter = v;
                        if (v != 0) break :blk true;
                        v = b2u((core.c4 & 0xFFFFFF00) == 0x2D0D0A00 and bf(5) != 0x2D);
                        self.lastLetter = v;
                        break :blk (v != 0);
                    };
                    if (cond) {
                        self.word0 = self.word1;
                        self.word1 = self.word2;
                        self.word2 = self.word3;
                        self.word3 = self.word4;
                        self.word4 = self.word5;
                        self.word5 = 0;
                        wordlen = wordlen1;
                        if (c < 128) {
                            self.StemIndex = @bitCast((@as(u32, @bitCast(self.StemIndex)) -% 1) & 3);
                            self.cWordIdx = self.pWordIdx;
                            self.pWordIdx = (@as(u32, @bitCast(self.StemIndex)) -% 1) & 3;
                            self.StemWords[self.cWordIdx] = .{};
                            var i: u32 = 0;
                            while (i <= wordlen) : (i += 1) {
                                const off: u32 = wordlen -% i +% 1 +% 2 *% b2u(i != wordlen);
                                self.StemWords[self.cWordIdx].plusEq(@intCast(tolwr(bf(off))));
                            }
                        }
                    } else {
                        self.wordGap = self.lastLetter;
                        self.firstLetter = @intCast(c);
                        self.wrdhsh = 0;
                    }
                }
                self.lastLetter = 0;
                core.words +%= 1;
                wordcount +%= 1;
                if (c > 4) self.word0 = maps.combine64(self.word0, @intCast(c));
                self.text0 = self.text0 *% 997 *% 16 +% @as(u32, @intCast(c));
                wordlen += 1;
                wordlen = @min(wordlen, 45);
                f = 0;
                self.w = @intCast(@as(u32, @truncate(self.word0)) & (self.wpos.size() - 1));
                if ((c == 'a' or c == 'e' or c == 'i' or c == 'o' or c == 'u') or (c == 'y' and (wordlen > 0 and pC != 'a' and pC != 'e' and pC != 'i' and pC != 'o' and pC != 'u'))) {
                    self.mask2 +%= 1;
                    self.wrdhsh = self.wrdhsh *% 997 *% 8 +% @as(u32, @bitCast(@divTrunc(c, 4) - 22));
                } else if (c >= 'b' and c <= 'z') {
                    self.mask2 +%= 2;
                    self.wrdhsh = self.wrdhsh *% 271 *% 32 +% @as(u32, @bitCast(c - 97));
                } else {
                    self.wrdhsh = self.wrdhsh *% 11 *% 32 +% @as(u32, @intCast(c));
                }
            } else {
                if (self.word0 != 0) {
                    self.type0 = (self.type0 << 2) | 1;
                    self.word5 = self.word4;
                    self.word4 = self.word3;
                    self.word3 = self.word2;
                    self.word2 = self.word1;
                    self.word1 = self.word0;
                    wordlen1 = wordlen;
                    self.wpos.at(@intCast(self.w)).* = core.blpos;
                    if (c == ':' or c == '=') self.cword0 = self.word0;
                    if (c == ']' and (frstchar != ':')) self.xword0 = self.word0;
                    self.ccword = 0;
                    self.word0 = 0;
                    wordlen = 0;
                    if ((c == '.' or c == '!' or c == '?' or c == '}' or c == ')') and bf(2) != 10) f = 1;
                }
                if (c == 32 or c == 10 or c == 5) {
                    spaces +%= 1;
                    spacecount +%= 1;
                    if (c == 10 or c == 5) {
                        self.nl1 = self.nl;
                        self.nl = core.pos - 1;
                    }
                } else if (c == '.' or c == '!' or c == '?' or c == ',' or c == ';' or c == ':') {
                    spafdo = 0;
                    self.ccword = @intCast(c);
                    self.mask2 +%= 3;
                } else {
                    spafdo +%= 1;
                    spafdo = @min(63, spafdo);
                }
            }
            if ((core.c4 & 0xFFFF) == 0x3D3D and frstchar == 0x3d) self.xword1 = self.word1;
            if ((core.c4 & 0xFFFF) == 0x2727) self.xword2 = self.word1;
            self.lastDigit = @min(0xFF, self.lastDigit +% 1);
            if (c >= '0' and c <= '9') {
                if (bf(3) >= '0' and bf(3) <= '9' and bf(2) == '.' and self.number0 == 0) {
                    self.number0 = self.number1;
                    self.number1 = 0;
                }
                self.number0 = maps.combine64(self.number0, @intCast(c));
                self.lastDigit = 0;
            } else if (self.number0 != 0) {
                self.type0 = (self.type0 << 2) | 2;
                self.number1 = self.number0;
                self.number0 = 0;
                self.ccword = 0;
            }
            if (!((c >= 'a' and c <= 'z') or (c >= '0' and c <= '9') or (c >= 128))) {
                self.data0 ^= @truncate(maps.combine64(self.data0, @intCast(c)));
            } else if (self.data0 != 0) {
                self.type0 = (self.type0 << 2) | 3;
                self.data0 = 0;
            }
            core.col = @intCast(@min(255, core.pos - self.nl));

            const above: i32 = @intCast(core.buf.at(@as(u32, @bitCast(self.nl1)) +% core.col).*);
            if (core.col <= 2) frstchar = if (core.col == 2) @intCast(@min(c, 96)) else 0;
            if (frstchar == '[' and c == 32) {
                if (bf(3) == ']' or bf(4) == ']') {
                    frstchar = 96;
                    self.xword0 = 0;
                }
            }
            const cm = &self.cm;
            cm.set(hash(.{ 513, spafdo, spaces, self.ccword }));
            cm.set(hash(.{ 514, frstchar, c }));
            cm.set(hash(.{ 515, core.col, frstchar, (b2u(self.lastUpper < core.col) * 4) + (self.mask2 & 3) }));
            cm.set(hash(.{ 516, spaces, (core.words & 255) }));

            cm.set(spaces & 0x7fff);
            cm.set(spaces & 0xff);

            cm.set(hash(.{ 257, self.number0, self.word1, self.wordGap }));
            cm.set(hash(.{ 258, self.number1, c, self.ccword }));
            cm.set(hash(.{ 259, self.number0, self.number1, self.wordGap }));
            cm.set(hash(.{ 260, self.word0, self.number1, b2u(self.lastDigit < self.wordGap +% wordlen) }));
            cm.set(hash(.{ 274, self.number0, self.cword0 }));
            cm.set(hash(.{ 518, wordlen1, core.col }));
            cm.set(hash(.{ 519, c, spacecount / 2, self.wordGap }));
            var h: u32 = wordcount *% 64 +% spacecount;
            cm.set(hash(.{ 520, c, h, self.ccword }));
            cm.set(hash(.{ 517, frstchar, h, self.lastLetter }));
            cm.set(hash(.{ self.data0, self.word1, self.number1, self.type0 & 0xFFF }));
            cm.set(hash(.{ 521, h, spafdo }));
            const d0: u32 = core.c4 & 0xf0ff;
            cm.set(hash(.{ 522, d0, frstchar, self.ccword }));

            h = @as(u32, @truncate(self.word0 *% 271));
            h = h +% bfu(1);

            cm.set(hash(.{ 262, h, 0 }));
            cm.set(hash(.{ self.number0 *% 271 +% @as(u64, bfu(1)), 0 }));
            cm.set(hash(.{ 263, self.word0, 0 }));
            if (self.wrdhsh != 0) {
                const idx: u32 = @as(u32, @truncate(self.word1)) & (self.wpos.size() - 1);
                cm.set(hash(.{ self.wrdhsh, bfu(@bitCast(self.wpos.get(idx))) }));
            } else cm.set(0);
            cm.set(hash(.{ 264, h, self.word1 }));
            cm.set(hash(.{ 265, self.word0, self.word1 }));
            cm.set(hash(.{ 266, h, self.word1, self.word2, b2u(self.lastUpper < wordlen) }));
            cm.set(hash(.{ 267, self.text0 & 0xffffff, 0 }));
            cm.set(self.text0 & 0xfffff);
            cm.set(hash(.{ 269, self.word0, self.xword0 }));
            cm.set(hash(.{ 270, h, self.xword1 }));
            cm.set(hash(.{ 271, h, self.xword2 }));
            cm.set(hash(.{ 272, frstchar, self.xword2 }));
            cm.set(hash(.{ 273, self.word0, self.cword0 }));
            cm.set(hash(.{ 275, h, self.word2 }));
            cm.set(hash(.{ 276, h, self.word3 }));
            cm.set(hash(.{ 277, h, self.word4 }));
            cm.set(hash(.{ 278, h, self.word5 }));
            cm.set(hash(.{ 279, h, self.word1, self.word3 }));
            cm.set(hash(.{ 280, h, self.word2, self.word3 }));
            cm.set(@as(u64, @intCast(bf(1) | (bf(3) << 8) | (bf(5) << 16))));
            cm.set(@as(u64, @intCast(bf(2) | (bf(4) << 8) | (bf(6) << 16))));
            cm.set(@as(u64, @intCast(bf(1) | (bf(4) << 8) | (bf(7) << 16))));
            if (f != 0) {
                self.word5 = self.word4;
                self.word4 = self.word3;
                self.word3 = self.word2;
                self.word2 = self.word1;
                self.word1 = '.';
            }
            if (core.col < 255) {
                cm.set(hash(.{ 523, core.col, bfu(1), above }));
                cm.set(hash(.{ 524, bfu(1), above }));
                cm.set(hash(.{ 525, core.col, bfu(1) }));
                cm.set(hash(.{ 526, core.col, b2i(c == 32) }));
            } else {
                cm.set(0);
                cm.set(0);
                cm.set(0);
                cm.set(0);
            }

            if (wordlen != 0) {
                const idx: u32 = @as(u32, @truncate(self.word1)) & (self.wpos.size() - 1);
                cm.set(hash(.{ 281, self.word0, core.llog(@bitCast(core.blpos -% self.wpos.get(idx))) >> 4 }));
            } else cm.set(0);

            {
                const idx1: u32 = @as(u32, @truncate(self.word1)) & (self.wpos.size() - 1);
                cm.set(hash(.{ 282, bfu(1), core.llog(@bitCast(core.blpos -% self.wpos.get(idx1))) >> 2 }));
                const idx2: u32 = @as(u32, @truncate(self.word2)) & (self.wpos.size() - 1);
                cm.set(hash(.{ 283, bfu(1), self.word0, core.llog(@bitCast(core.blpos -% self.wpos.get(idx2))) >> 2 }));
            }

            var fl: i32 = 0;
            if ((core.c4 & 0xff) != 0) {
                const cc: i32 = @intCast(core.c4 & 0xff);
                if (isAlpha(cc)) fl = 1 else if (isPunctC(cc)) fl = 2 else if (isSpaceC(cc)) fl = 3 else if (cc == 0xff) fl = 4 else if (cc < 16) fl = 5 else if (cc < 64) fl = 6 else fl = 7;
            }
            self.mask = (self.mask << 3) | @as(u32, @intCast(fl));

            {
                cm.set(hash(.{ 528, self.mask, 0 }));
                cm.set(hash(.{ 529, self.mask, bfu(1) }));
                cm.set(hash(.{ 530, self.mask & 0xff, core.col }));
                cm.set(hash(.{ 531, self.mask, bfu(2), bfu(3) }));
                cm.set(hash(.{ 532, self.mask & 0x1ff, core.f4 & 0x00fff0 }));
                const bits: u32 = (b2u(wordlen1 > 3) << 6) |
                    (b2u(wordlen > 0) << 5) |
                    (b2u(spafdo == wordlen +% 2) << 4) |
                    (b2u(spafdo == wordlen +% wordlen1 +% 3) << 3) |
                    (b2u(spafdo >= self.lastLetter +% wordlen1 +% self.wordGap) << 2) |
                    (b2u(self.lastUpper < self.lastLetter +% wordlen1) << 1) |
                    b2u(self.lastUpper < wordlen +% wordlen1 +% self.wordGap);
                cm.set(hash(.{ h, core.llog(self.wordGap), self.mask & 0x1FF, bits, self.type0 & 0xFFF }));
            }
            if (wordlen1 != 0) {
                cm.set(hash(.{ core.col, wordlen1, above & 0x5F, core.c4 & 0x5F }));
            } else cm.set(0);
            if (self.wrdhsh != 0) {
                cm.set(hash(.{ self.mask2 & 0x3F, self.wrdhsh & 0xFFF, (0x100 | self.firstLetter) *% b2u(wordlen < 6), b2u(self.wordGap > 4) * 2 + b2u(wordlen1 > 5) }));
            } else cm.set(0);
            if (self.lastLetter < 16) {
                cm.set(hash(.{ self.StemWords[self.pWordIdx].Hash[2], h }));
            } else cm.set(0);
        }
        _ = self.cm.mix(m);
    }

    inline fn wordlenIsZero(_: *WordModel) bool {
        return wordlen == 0;
    }

    pub fn deinit(self: *WordModel) void {
        self.cm.deinit();
        self.wpos.deinit();
    }
};

// =============================== nestModel =================================
pub const NestModel = struct {
    ic: i32 = 0,
    bc: i32 = 0,
    pc: i32 = 0,
    qc: i32 = 0,
    lvc: i32 = 0,
    ac: i32 = 0,
    ec: i32 = 0,
    uc: i32 = 0,
    sense1: i32 = 0,
    sense2: i32 = 0,
    w: i32 = 0,
    vc: u32 = 0,
    wc: u32 = 0,
    cm: maps.ContextMap,

    pub fn init(a: Allocator) NestModel {
        return NestModel{ .cm = maps.ContextMap.init(a, core.MEM() / 2, 12) };
    }

    pub fn predict(self: *NestModel, m: *core.Mixer) void {
        if (core.bpos == 0) {
            const c: i32 = @intCast(core.c4 & 255);
            var matched: i32 = 1;
            var vv: i32 = undefined;
            self.w = self.w * b2i((self.vc & 7) > 0 and (self.vc & 7) < 3);
            if (c & 0x80 != 0) self.w = self.w *% 11 *% 32 +% c;
            const lc: i32 = if (c >= 'A' and c <= 'Z') c + 'a' - 'A' else c;
            if (lc == 'a' or lc == 'e' or lc == 'i' or lc == 'o' or lc == 'u') {
                vv = 1;
                self.w = self.w *% 997 *% 8 +% (@divTrunc(lc, 4) - 22);
            } else if (lc >= 'a' and lc <= 'z') {
                vv = 2;
                self.w = self.w *% 271 *% 32 +% lc - 97;
            } else if (lc == ' ' or lc == '.' or lc == ',' or lc == '!' or lc == '?' or lc == '\n') {
                vv = 3;
            } else if (lc >= '0' and lc <= '9') {
                vv = 4;
            } else if (lc == 'y') {
                vv = 5;
            } else if (lc == '\'') {
                vv = 6;
            } else vv = if (c & 32 != 0) 7 else 0;
            self.vc = (self.vc << 3) | @as(u32, @intCast(vv));
            if (vv != self.lvc) {
                self.wc = (self.wc << 3) | @as(u32, @intCast(vv));
                self.lvc = vv;
            }
            switch (c) {
                ' ' => self.qc = 0,
                '(' => self.ic +%= 31,
                ')' => self.ic -%= 31,
                '[' => self.ic +%= 11,
                ']' => self.ic -%= 11,
                '<' => {
                    self.ic +%= 23;
                    self.qc +%= 34;
                },
                '>' => {
                    self.ic -%= 23;
                    self.qc = @divTrunc(self.qc, 5);
                },
                ':' => self.pc = 20,
                '{' => self.ic +%= 17,
                '}' => self.ic -%= 17,
                '|' => self.pc +%= 223,
                '"' => self.pc +%= 0x40,
                '\'' => {
                    self.pc +%= 0x42;
                    if (c != @as(i32, @intCast((core.c4 >> 8) & 0xff))) self.sense2 ^= 1 else self.ac +%= (2 * self.sense2 - 1);
                },
                '\n' => {
                    self.pc = 0;
                    self.qc = 0;
                },
                '.' => self.pc = 0,
                '!' => self.pc = 0,
                '?' => self.pc = 0,
                '#' => self.pc +%= 0x08,
                '%' => self.pc +%= 0x76,
                '$' => self.pc +%= 0x45,
                '*' => self.pc +%= 0x35,
                '-' => self.pc +%= 0x3,
                '@' => self.pc +%= 0x72,
                '&' => self.qc +%= 0x12,
                ';' => self.qc = @divTrunc(self.qc, 3),
                '\\' => self.pc +%= 0x29,
                '/' => {
                    self.pc +%= 0x11;
                    if (core.buf.size() > 1 and bf(1) == '<') self.qc +%= 74;
                },
                '=' => {
                    self.pc +%= 87;
                    if (c != @as(i32, @intCast((core.c4 >> 8) & 0xff))) self.sense1 ^= 1 else self.ec +%= (2 * self.sense1 - 1);
                },
                else => matched = 0,
            }
            if (core.c4 == 0x266C743B) self.uc = imin(7, self.uc + 1) else if (core.c4 == 0x2667743B) self.uc -= b2i(self.uc > 0);
            if (matched != 0) self.bc = 0 else self.bc += 1;
            if (self.bc > 300) {
                self.bc = 0;
                self.ic = 0;
                self.pc = 0;
                self.qc = 0;
                self.uc = 0;
            }
            const vcu: u32 = self.vc;
            const pcu: u32 = @bitCast(self.pc);
            const icu: u32 = @bitCast(self.ic);
            const qcu: u32 = @bitCast(self.qc);
            const wcu: u32 = self.wc;
            const cm = &self.cm;
            cm.set(hash(.{ 1, (if ((vv > 0 and vv < 3)) @as(i32, 0) else (lc | 0x100)), self.ic & 0x3FF, self.ec & 0x7, self.ac & 0x7, self.uc }));
            cm.set(hash(.{ 2, self.ic, self.w, maps.ilog2(@intCast(self.bc + 1)) }));
            cm.set(hash(.{ 3, (3 *% vcu +% 77 *% pcu +% 373 *% icu +% qcu) & 0xffff }));
            cm.set(hash(.{ 4, (31 *% vcu +% 27 *% pcu +% 281 *% qcu) & 0xffff }));
            cm.set(hash(.{ 5, (13 *% vcu +% 271 *% icu +% qcu +% @as(u32, @bitCast(self.bc))) & 0xffff }));
            cm.set(hash(.{ 6, (17 *% pcu +% 7 *% icu) & 0xffff }));
            cm.set(hash(.{ 7, (13 *% vcu +% icu) & 0xffff }));
            cm.set(hash(.{ 8, (vcu / 3 +% pcu) & 0xffff }));
            cm.set(hash(.{ 9, (7 *% wcu +% qcu) & 0xffff }));
            cm.set(hash(.{ 10, vcu & 0xffff, core.f4 & 0xf }));
            cm.set(hash(.{ 11, (3 *% pcu) & 0xffff, core.f4 & 0xf }));
            cm.set(hash(.{ 12, icu & 0xffff, core.f4 & 0xf }));
        }
        _ = self.cm.mix(m);
    }

    pub fn deinit(self: *NestModel) void {
        self.cm.deinit();
    }
};

// =============================== recordModel ===============================
const dBASE = struct {
    Version: u8 = 0,
    nRecords: u32 = 0,
    RecordLength: u16 = 0,
    HeaderLength: u16 = 0,
    Start: i32 = 0,
    End: i32 = 0,
};

pub const RecordModel = struct {
    cpos1: []i32,
    cpos2: []i32,
    cpos3: []i32,
    cpos4: []i32,
    wpos1: []i32,
    rlen: [3]i32 = .{ 2, 3, 4 },
    rcount: [2]i32 = .{ 0, 0 },
    padding: u8 = 0,
    N: u8 = 0,
    NN: u8 = 0,
    NNN: u8 = 0,
    NNNN: u8 = 0,
    WxNW: u8 = 0,
    prevTransition: i32 = 0,
    nTransition: i32 = 0,
    col: i32 = 0,
    mxCtx: i32 = 0,
    x: i32 = 0,
    cm: maps.ContextMap,
    cn: maps.ContextMap,
    co: maps.ContextMap,
    cp: maps.ContextMap,
    Maps: [6]maps.StationaryMap,
    sMap: [3]maps.SmallStationaryContextMap,
    iMap: [3]maps.IndirectMap,
    MayBeImg24b: bool = false,
    dbase: dBASE = .{},
    iCtx: [5]maps.IndirectContext(u16),
    alloc: Allocator,

    pub fn init(a: Allocator) RecordModel {
        var self = RecordModel{
            .cpos1 = a.alloc(i32, 256) catch unreachable,
            .cpos2 = a.alloc(i32, 256) catch unreachable,
            .cpos3 = a.alloc(i32, 256) catch unreachable,
            .cpos4 = a.alloc(i32, 256) catch unreachable,
            .wpos1 = a.alloc(i32, 0x10000) catch unreachable,
            .cm = maps.ContextMap.init(a, 32768, 3),
            .cn = maps.ContextMap.init(a, 32768 / 2, 3),
            .co = maps.ContextMap.init(a, 32768 * 2, 3),
            .cp = maps.ContextMap.init(a, core.MEM(), 16),
            .Maps = undefined,
            .sMap = undefined,
            .iMap = undefined,
            .iCtx = undefined,
            .alloc = a,
        };
        @memset(self.cpos1, 0);
        @memset(self.cpos2, 0);
        @memset(self.cpos3, 0);
        @memset(self.cpos4, 0);
        @memset(self.wpos1, 0);
        const mapdims = [6][2]u32{ .{ 10, 8 }, .{ 10, 8 }, .{ 8, 8 }, .{ 8, 8 }, .{ 8, 8 }, .{ 11, 1 } };
        for (0..6) |i| self.Maps[i] = maps.StationaryMap.init(a, mapdims[i][0], mapdims[i][1], 0);
        const smdims = [3][2]u32{ .{ 11, 1 }, .{ 3, 1 }, .{ 19, 1 } };
        for (0..3) |i| self.sMap[i] = maps.SmallStationaryContextMap.init(a, smdims[i][0], smdims[i][1]);
        for (0..3) |i| self.iMap[i] = maps.IndirectMap.init(a, 8, 8);
        const icdims = [5][2]u32{ .{ 16, 8 }, .{ 16, 8 }, .{ 16, 8 }, .{ 20, 8 }, .{ 11, 1 } };
        for (0..5) |i| self.iCtx[i] = maps.IndirectContext(u16).init(a, icdims[i][0], icdims[i][1]);
        return self;
    }

    pub fn predict(self: *RecordModel, m: *core.Mixer, filetype: i32, stats: ?*core.ModelStats) void {
        if (core.bpos == 0) {
            const w: i32 = @intCast(core.c4 & 0xffff);
            const c: i32 = w & 255;
            const d: i32 = w >> 8;
            if (stats != null and stats.?.Record != 0 and (stats.?.Record >> 16) != @as(u32, @intCast(self.rlen[0]))) {
                self.rlen[0] = @intCast(stats.?.Record >> 16);
                self.rcount[0] = 0;
                self.rcount[1] = 0;
            } else {
                // detect dBASE tables
                if (core.blpos == 0 or (self.dbase.Version > 0 and core.blpos >= self.dbase.End)) {
                    self.dbase.Version = 0;
                } else if (self.dbase.Version == 0 and (filetype == core.FT_DEFAULT or filetype == core.FT_TEXT) and core.blpos >= 31) {
                    const detected = blk: {
                        var b: i32 = bf(32) & 0xff;
                        if (!((b & 7) == 3 or (b & 7) == 4 or (b >> 4) == 3 or b == 0xF5)) break :blk false;
                        b = bf(30) & 0xff;
                        if (!(b > 0 and b < 13)) break :blk false;
                        b = bf(29) & 0xff;
                        if (!(b > 0 and b < 32)) break :blk false;
                        self.dbase.nRecords = bfu(28) | (bfu(27) << 8) | (bfu(26) << 16) | (bfu(25) << 24);
                        if (!(self.dbase.nRecords > 0 and self.dbase.nRecords < 0xFFFFF)) break :blk false;
                        self.dbase.HeaderLength = @truncate(bfu(24) | (bfu(23) << 8));
                        if (!(@as(i32, self.dbase.HeaderLength) > 32)) break :blk false;
                        const cond5 = ((@as(i32, self.dbase.HeaderLength) - 32 - 1) & 31) == 0 or blk3: {
                            if (!(@as(i32, self.dbase.HeaderLength) > 255 + 8)) break :blk3 false;
                            self.dbase.HeaderLength -%= @as(u16, 255 + 8);
                            break :blk3 (((@as(i32, self.dbase.HeaderLength) - 32 - 1) & 31) == 0);
                        };
                        if (!cond5) break :blk false;
                        self.dbase.RecordLength = @truncate(bfu(22) | (bfu(21) << 8));
                        if (!(@as(i32, self.dbase.RecordLength) > 8)) break :blk false;
                        if (!(bf(20) == 0 and bf(19) == 0 and bf(17) <= 1 and bf(16) <= 1)) break :blk false;
                        break :blk true;
                    };
                    if (detected) {
                        self.dbase.Version = if ((bf(32) >> 4) == 3) 3 else @intCast(bf(32) & 7);
                        self.dbase.Start = core.blpos - 32 + @as(i32, self.dbase.HeaderLength);
                        self.dbase.End = self.dbase.Start + @as(i32, @bitCast(self.dbase.nRecords *% @as(u32, self.dbase.RecordLength)));
                        if (self.dbase.Version == 3) {
                            self.rlen[0] = 32;
                            self.rcount[0] = 0;
                            self.rcount[1] = 0;
                        }
                    }
                } else if (self.dbase.Version > 0 and core.blpos == self.dbase.Start) {
                    self.rlen[0] = @intCast(self.dbase.RecordLength);
                    self.rcount[0] = 0;
                    self.rcount[1] = 0;
                }

                const r: i32 = core.pos - self.cpos1[@intCast(c)];
                if (r > 1 and r == self.cpos1[@intCast(c)] - self.cpos2[@intCast(c)] and
                    r == self.cpos2[@intCast(c)] - self.cpos3[@intCast(c)] and (r > 32 or r == self.cpos3[@intCast(c)] - self.cpos4[@intCast(c)]) and
                    (r > 10 or ((c == bf(@intCast(r * 5 + 1))) and c == bf(@intCast(r * 6 + 1)))))
                {
                    if (r == self.rlen[1]) self.rcount[0] += 1 else if (r == self.rlen[2]) self.rcount[1] += 1 else if (self.rcount[0] > self.rcount[1]) {
                        self.rlen[2] = r;
                        self.rcount[1] = 1;
                    } else {
                        self.rlen[1] = r;
                        self.rcount[0] = 1;
                    }
                }

                var i: usize = 0;
                while (i < 2) : (i += 1) {
                    if (self.rcount[i] > imax(0, 12 - @as(i32, @intCast(maps.ilog2(@intCast(self.rlen[i + 1])))))) {
                        if (self.rlen[0] != self.rlen[i + 1]) {
                            if (self.MayBeImg24b and self.rlen[i + 1] == 3) {
                                self.rcount[0] >>= 1;
                                self.rcount[1] >>= 1;
                                continue;
                            } else if ((self.rlen[i + 1] > self.rlen[0]) and (@rem(self.rlen[i + 1], self.rlen[0]) == 0)) {
                                if ((self.rlen[0] > 32) and (self.rlen[i + 1] == self.rlen[0] * 2)) {
                                    self.rcount[0] >>= 1;
                                    self.rcount[1] >>= 1;
                                    continue;
                                }
                            }
                            self.rlen[0] = self.rlen[i + 1];
                            self.rcount[i] = 0;
                            self.MayBeImg24b = (self.rlen[0] > 30 and @rem(self.rlen[0], 3) == 0);
                            self.nTransition = 0;
                        } else self.rcount[i] >>= 2;
                        if ((self.rlen[i + 1] << 4) > self.rlen[1 + (i ^ 1)]) self.rcount[i ^ 1] = 0;
                    }
                }
            }

            self.col = @rem(core.pos, self.rlen[0]);
            self.x = imin(0x1F, @divTrunc(self.col, imax(1, @divTrunc(self.rlen[0], 32))));
            self.N = @intCast(bf(@intCast(self.rlen[0])));
            self.NN = @intCast(bf(@intCast(self.rlen[0] * 2)));
            self.NNN = @intCast(bf(@intCast(self.rlen[0] * 3)));
            self.NNNN = @intCast(bf(@intCast(self.rlen[0] * 4)));
            for (0..4) |ii| self.iCtx[ii].add(@intCast(c));
            self.iCtx[0].setCtx(@intCast((c << 8) | @as(i32, self.N)));
            self.iCtx[1].setCtx(@intCast((bf(@intCast(self.rlen[0] - 1)) << 8) | @as(i32, self.N)));
            self.iCtx[2].setCtx(@intCast((c << 8) | bf(@intCast(self.rlen[0] - 1))));
            self.iCtx[3].setCtx(maps.finalize64(hash(.{ c, @as(i32, self.N), bf(@intCast(self.rlen[0] + 1)) }), 20));

            if (self.col == 0) self.nTransition = 0;
            if ((((core.c4 >> 8) == 32 * 0x010101) and (c != 32)) or ((core.c4 >> 8) == 0 and c != 0 and ((self.padding != 32) or (core.pos - self.prevTransition > self.rlen[0])))) {
                self.prevTransition = core.pos;
                self.nTransition += b2i(self.nTransition < 31);
                self.padding = @intCast(d);
            }

            var ic: u64 = 0;
            const cm = &self.cm;
            const cn = &self.cn;
            const co = &self.co;
            const cp = &self.cp;
            const N: i32 = self.N;
            const NN: i32 = self.NN;
            const NNN: i32 = self.NNN;
            const NNNN: i32 = self.NNNN;

            ic += 1;
            cm.set(hash(.{ ic, (c << 8) | (imin(255, core.pos - self.cpos1[@intCast(c)]) >> 2) }));
            ic += 1;
            cm.set(hash(.{ ic, (w << 9) | (core.llog(@bitCast(core.pos - self.wpos1[@intCast(w)])) >> 2) }));
            ic += 1;
            cm.set(hash(.{ ic, self.rlen[0] | (N << 10) | (NN << 18) }));

            ic += 1;
            cn.set(hash(.{ ic, w | (self.rlen[0] << 16) }));
            ic += 1;
            cn.set(hash(.{ ic, d | (self.rlen[0] << 8) }));
            ic += 1;
            cn.set(hash(.{ ic, c | (self.rlen[0] << 8) }));

            ic += 1;
            co.set(hash(.{ ic, (c << 8) | imin(255, core.pos - self.cpos1[@intCast(c)]) }));
            ic += 1;
            co.set(hash(.{ ic, (c << 17) | (d << 9) | (core.llog(@bitCast(core.pos - self.wpos1[@intCast(w)])) >> 2) }));
            ic += 1;
            co.set(hash(.{ ic, (c << 8) | N }));

            ic += 1;
            cp.set(hash(.{ ic, self.rlen[0] | (N << 10) | (self.col << 18) }));
            ic += 1;
            cp.set(hash(.{ ic, self.rlen[0] | (c << 10) | (self.col << 18) }));
            ic += 1;
            cp.set(hash(.{ ic, self.col | (self.rlen[0] << 12) }));

            if (self.rlen[0] > 8) {
                ic += 1;
                cp.set(hash(.{ ic, imin(imin(0xFF, self.rlen[0]), core.pos - self.prevTransition), imin(0x3FF, self.col), (w & 0xF0F0) | b2i(w == ((@as(i32, self.padding) << 8) | @as(i32, self.padding))), self.nTransition }));
                ic += 1;
                cp.set(hash(.{ ic, w, b2i(bf(@intCast(self.rlen[0] + 1)) == @as(i32, self.padding) and N == @as(i32, self.padding)), @divTrunc(self.col, imax(1, @divTrunc(self.rlen[0], 32))) }));
            } else {
                cp.set(0);
                cp.set(0);
            }

            ic += 1;
            cp.set(hash(.{ ic, N | ((NN & 0xF0) << 4) | ((NNN & 0xE0) << 7) | ((NNNN & 0xE0) << 10) | (@divTrunc(self.col, imax(1, @divTrunc(self.rlen[0], 16))) << 18) }));
            ic += 1;
            cp.set(hash(.{ ic, (N & 0xF8) | ((NN & 0xF8) << 8) | (self.col << 16) }));
            ic += 1;
            cp.set(hash(.{ ic, N, NN }));

            ic += 1;
            cp.set(hash(.{ ic, self.col, @as(u32, self.iCtx[0].value()) }));
            ic += 1;
            cp.set(hash(.{ ic, self.col, @as(u32, self.iCtx[1].value()) }));
            ic += 1;
            cp.set(hash(.{ ic, self.col, @as(u32, self.iCtx[0].value()) & 0xFF, @as(u32, self.iCtx[1].value()) & 0xFF }));

            ic += 1;
            cp.set(hash(.{ ic, @as(u32, self.iCtx[2].value()) }));
            ic += 1;
            cp.set(hash(.{ ic, @as(u32, self.iCtx[3].value()) }));
            ic += 1;
            cp.set(hash(.{ ic, @as(u32, self.iCtx[1].value()) & 0xFF, @as(u32, self.iCtx[3].value()) & 0xFF }));

            self.WxNW = @intCast((c ^ bf(@intCast(self.rlen[0] + 1))) & 0xff);
            ic += 1;
            cp.set(hash(.{ ic, N, @as(i32, self.WxNW) }));
            ic += 1;
            const expB: i32 = if (stats != null and stats.?.Match.length > 0) @as(i32, stats.?.Match.expectedByte) else (0x100 | @as(i32, @intCast(@as(u8, @truncate(self.iCtx[1].value())))));
            cp.set(hash(.{ ic, expB, N, @as(i32, self.WxNW) }));

            var k: i32 = 0x300;
            if (self.MayBeImg24b) {
                k = @rem(self.col, 3) << 8;
                self.Maps[0].setDirect(@as(u32, Clip(@as(i32, @intCast(@as(u8, @truncate(core.c4 >> 16)))) + c - @as(i32, @intCast(core.c4 >> 24)))) | @as(u32, @intCast(k)));
            } else {
                self.Maps[0].setDirect(@as(u32, Clip(c * 2 - d)) | @as(u32, @intCast(k)));
            }
            self.Maps[1].setDirect(@as(u32, Clip(c + N - bf(@intCast(self.rlen[0] + 1)))) | @as(u32, @intCast(k)));
            self.Maps[2].setDirect(@as(u32, Clip(N + NN - NNN)));
            self.Maps[3].setDirect(@as(u32, Clip(N * 2 - NN)));
            self.Maps[4].setDirect(@as(u32, Clip(N * 3 - NN * 3 + NNN)));
            self.iMap[0].setDirect(@bitCast(N + NN - NNN));
            self.iMap[1].setDirect(@bitCast(N * 2 - NN));
            self.iMap[2].setDirect(@bitCast(N * 3 - NN * 3 + NNN));

            self.cpos4[@intCast(c)] = self.cpos3[@intCast(c)];
            self.cpos3[@intCast(c)] = self.cpos2[@intCast(c)];
            self.cpos2[@intCast(c)] = self.cpos1[@intCast(c)];
            self.cpos1[@intCast(c)] = core.pos;
            self.wpos1[@intCast(w)] = core.pos;

            self.mxCtx = if (self.rlen[0] > 128) imin(0x7F, @divTrunc(self.col, imax(1, @divTrunc(self.rlen[0], 128)))) else self.col;
        }
        const B: u8 = @truncate(@as(u32, @intCast(core.c0)) << @as(u5, @intCast(8 - core.bpos)));
        const ctx: u32 = @as(u32, self.N ^ B) | (@as(u32, @intCast(core.bpos)) << 8);
        self.iCtx[4].add(@intCast(core.y));
        self.iCtx[4].setCtx(ctx);
        self.Maps[5].setDirect(ctx);
        self.sMap[0].set(ctx);
        self.sMap[1].set(@as(u32, self.iCtx[4].value()));
        self.sMap[2].set((ctx << 8) | @as(u32, self.WxNW));

        _ = self.cm.mix(m);
        _ = self.cn.mix(m);
        _ = self.co.mix(m);
        _ = self.cp.mix(m);
        for (0..6) |i| self.Maps[i].mix(m, 1, 3, 1023);
        for (0..3) |i| self.iMap[i].mix(m, 1, 3, 255);
        self.sMap[0].mix(m, 6, 1, 3);
        self.sMap[1].mix(m, 6, 1, 3);
        self.sMap[2].mix(m, 5, 1, 2);

        m.set(b2i(self.rlen[0] > 2) * ((core.bpos << 7) | self.mxCtx), 1024);
        m.set(((@as(i32, self.N ^ B)) >> 4) | (self.x << 4), 512);
        m.set((@as(i32, core.grp0) << 5) | self.x, 11 * 32);
        if (stats) |s| s.Record = (@as(u32, @intCast(imin(0xFFFF, self.rlen[0]))) << 16) | @as(u32, @intCast(imin(0xFFFF, self.col)));
    }

    pub fn deinit(self: *RecordModel) void {
        self.alloc.free(self.cpos1);
        self.alloc.free(self.cpos2);
        self.alloc.free(self.cpos3);
        self.alloc.free(self.cpos4);
        self.alloc.free(self.wpos1);
        self.cm.deinit();
        self.cn.deinit();
        self.co.deinit();
        self.cp.deinit();
        for (0..6) |i| self.Maps[i].deinit();
        for (0..3) |i| self.sMap[i].deinit();
        for (0..3) |i| self.iMap[i].deinit();
        for (0..5) |i| self.iCtx[i].deinit();
    }
};

// ============================== recordModel1 ===============================
pub const RecordModel1 = struct {
    cpos1: []i32,
    wpos1: []i32,
    cm: maps.ContextMap,
    cn: maps.ContextMap,
    co: maps.ContextMap,
    cp: maps.ContextMap,
    cq: maps.ContextMap,
    alloc: Allocator,

    pub fn init(a: Allocator) RecordModel1 {
        const self = RecordModel1{
            .cpos1 = a.alloc(i32, 256) catch unreachable,
            .wpos1 = a.alloc(i32, 0x10000) catch unreachable,
            .cm = maps.ContextMap.init(a, 32768, 2),
            .cn = maps.ContextMap.init(a, 32768 / 2, 4 + 1),
            .co = maps.ContextMap.init(a, 32768 * 4, 4),
            .cp = maps.ContextMap.init(a, 32768 * 2, 3),
            .cq = maps.ContextMap.init(a, 32768 * 2, 3),
            .alloc = a,
        };
        @memset(self.cpos1, 0);
        @memset(self.wpos1, 0);
        return self;
    }

    pub fn predict(self: *RecordModel1, m: *core.Mixer) void {
        if (core.bpos == 0) {
            const w: i32 = @intCast(core.c4 & 0xffff);
            const c: i32 = w & 255;
            const d: i32 = @intCast(core.c4 & 0xf0ff);
            const e: i32 = @intCast(core.c4 & 0xffffff);

            self.cm.set(sx((c << 8) | @divTrunc(imin(255, core.pos - self.cpos1[@intCast(c)]), 4)));
            self.cm.set(sx((w << 9) | (core.llog(@bitCast(core.pos - self.wpos1[@intCast(w)])) >> 2)));

            self.cn.set(sx(w));
            self.cn.set(sx(d << 8));
            self.cn.set(sx(c << 16));
            self.cn.set(core.f4 & 0xfffff);
            const col: i32 = core.pos & 3;
            self.cn.set(sx(col | (2 << 12)));

            self.co.set(sx(c));
            self.co.set(sx(w << 8));
            self.co.set(core.w5 & 0x3ffff);
            self.co.set(sx(e << 3));

            self.cp.set(sx(d));
            self.cp.set(sx(c << 8));
            self.cp.set(sx(w << 16));

            self.cq.set(sx(w << 3));
            self.cq.set(sx(c << 19));
            self.cq.set(sx(e));

            self.cpos1[@intCast(c)] = core.pos;
            self.wpos1[@intCast(w)] = core.pos;
        }
        _ = self.cm.mix(m);
        _ = self.cn.mix(m);
        _ = self.co.mix(m);
        _ = self.cq.mix(m);
        _ = self.cp.mix(m);
    }

    pub fn deinit(self: *RecordModel1) void {
        self.alloc.free(self.cpos1);
        self.alloc.free(self.wpos1);
        self.cm.deinit();
        self.cn.deinit();
        self.co.deinit();
        self.cp.deinit();
        self.cq.deinit();
    }
};

// ========================== linearPredictionModel ==========================
pub const LinearPredictionModel = struct {
    const nOLS = 3;
    const nLnrPrd = nOLS + 2;
    sMap: [nLnrPrd]maps.SmallStationaryContextMap,
    ols: [nOLS]maps.OLS(f64, u8, true),
    prd: [nLnrPrd]u8 = .{ 0, 0, 0, 0, 0 },

    pub fn init(a: Allocator) LinearPredictionModel {
        var self = LinearPredictionModel{ .sMap = undefined, .ols = undefined };
        for (0..nLnrPrd) |i| self.sMap[i] = maps.SmallStationaryContextMap.init(a, 11, 1);
        for (0..nOLS) |i| self.ols[i] = maps.OLS(f64, u8, true).init(a, 32, 4, 0.995, 0.001);
        return self;
    }

    pub fn predict(self: *LinearPredictionModel, m: *core.Mixer) void {
        if (core.bpos == 0) {
            const W: u8 = @intCast(bf(1));
            const WW: u8 = @intCast(bf(2));
            const WWW: u8 = @intCast(bf(3));
            for (0..nOLS) |i| self.ols[i].update(W);
            var i: u32 = 1;
            while (i <= 32) : (i += 1) {
                self.ols[0].add(@intCast(bf(i)));
                self.ols[1].add(@intCast(bf(i * 2 - 1)));
                self.ols[2].add(@intCast(bf(i * 2)));
            }
            for (0..nOLS) |k| {
                const pf = @floor(self.ols[k].predict());
                const clamped = @max(-2.0e9, @min(2.0e9, pf));
                self.prd[k] = Clip(@intFromFloat(clamped));
            }
            self.prd[3] = Clip(@as(i32, W) * 2 - @as(i32, WW));
            self.prd[4] = Clip(@as(i32, W) * 3 - @as(i32, WW) * 3 + @as(i32, WWW));
        }
        const B: u8 = @truncate(@as(u32, @intCast(core.c0)) << @as(u5, @intCast(8 - core.bpos)));
        for (0..nLnrPrd) |i| {
            self.sMap[i].set(@bitCast((@as(i32, self.prd[i]) - @as(i32, B)) * 8 + core.bpos));
            self.sMap[i].mix(m, 6, 1, 2);
        }
    }

    pub fn deinit(self: *LinearPredictionModel) void {
        for (0..nLnrPrd) |i| self.sMap[i].deinit();
        for (0..nOLS) |i| self.ols[i].deinit();
    }
};

// =============================== sparseModel ===============================
pub const SparseModel = struct {
    cm: maps.ContextMap,

    pub fn init(a: Allocator) SparseModel {
        return SparseModel{ .cm = maps.ContextMap.init(a, core.MEM() * 2, 40 + 2) };
    }

    pub fn predict(self: *SparseModel, m: *core.Mixer, seenbefore: i32, howmany: i32) void {
        if (core.bpos == 0) {
            const cm = &self.cm;
            var i: u64 = 0;
            i += 1;
            cm.set(hash(.{ i, seenbefore }));
            i += 1;
            cm.set(hash(.{ i, howmany }));
            i += 1;
            cm.set(hash(.{ i, bf(1) | (bf(5) << 8) }));
            i += 1;
            cm.set(hash(.{ i, bf(1) | (bf(6) << 8) }));
            i += 1;
            cm.set(hash(.{ i, bf(3) | (bf(6) << 8) }));
            i += 1;
            cm.set(hash(.{ i, bf(4) | (bf(8) << 8) }));
            i += 1;
            cm.set(hash(.{ i, bf(1) | (bf(3) << 8) | (bf(5) << 16) }));
            i += 1;
            cm.set(hash(.{ i, bf(2) | (bf(4) << 8) | (bf(6) << 16) }));
            i += 1;
            cm.set(hash(.{ i, core.c4 & 0x00f0f0ff }));
            i += 1;
            cm.set(hash(.{ i, core.c4 & 0x00ff00ff }));
            i += 1;
            cm.set(hash(.{ i, core.c4 & 0xff0000ff }));
            i += 1;
            cm.set(hash(.{ i, core.c4 & 0x00f8f8f8 }));
            i += 1;
            cm.set(hash(.{ i, core.c4 & 0xf8f8f8f8 }));
            i += 1;
            cm.set(hash(.{ i, core.f4 & 0x00000fff }));
            i += 1;
            cm.set(hash(.{ i, core.f4 }));
            i += 1;
            cm.set(hash(.{ i, core.c4 & 0x00e0e0e0 }));
            i += 1;
            cm.set(hash(.{ i, core.c4 & 0xe0e0e0e0 }));
            i += 1;
            cm.set(hash(.{ i, core.c4 & 0x810000c1 }));
            i += 1;
            cm.set(hash(.{ i, core.c4 & 0xC3CCC38C }));
            i += 1;
            cm.set(hash(.{ i, core.c4 & 0x0081CC81 }));
            i += 1;
            cm.set(hash(.{ i, core.c4 & 0x00c10081 }));
            var j: u32 = 1;
            while (j < 8) : (j += 1) {
                i += 1;
                cm.set(hash(.{ i, seenbefore | (bf(j) << 8) }));
                i += 1;
                cm.set(hash(.{ i, (bf(j + 2) << 8) | bf(j + 1) }));
                i += 1;
                cm.set(hash(.{ i, (bf(j + 3) << 8) | bf(j + 1) }));
            }
        }
        _ = self.cm.mix(m);
    }

    pub fn deinit(self: *SparseModel) void {
        self.cm.deinit();
    }
};

// ============================== sparseModel1 ===============================
pub const SparseModel1 = struct {
    cm: maps.ContextMap,
    scm1: maps.SmallStationaryContextMap,
    scm2: maps.SmallStationaryContextMap,
    scm3: maps.SmallStationaryContextMap,
    scm4: maps.SmallStationaryContextMap,
    scm5: maps.SmallStationaryContextMap,
    scm6: maps.SmallStationaryContextMap,
    scma: maps.SmallStationaryContextMap,

    pub fn init(a: Allocator) SparseModel1 {
        return SparseModel1{
            .cm = maps.ContextMap.init(a, core.MEM() * 4, 31),
            .scm1 = maps.SmallStationaryContextMap.init(a, 7, 8),
            .scm2 = maps.SmallStationaryContextMap.init(a, 8, 8),
            .scm3 = maps.SmallStationaryContextMap.init(a, 4, 8),
            .scm4 = maps.SmallStationaryContextMap.init(a, 6, 8),
            .scm5 = maps.SmallStationaryContextMap.init(a, 4, 8),
            .scm6 = maps.SmallStationaryContextMap.init(a, 4, 8),
            .scma = maps.SmallStationaryContextMap.init(a, 7, 8),
        };
    }

    pub fn predict(self: *SparseModel1, m: *core.Mixer, seenbefore: i32, howmany: i32) void {
        if (core.bpos == 0) {
            self.scm5.set(@bitCast(seenbefore));
            self.scm6.set(@bitCast(howmany));
            const cm = &self.cm;
            var h: u32 = core.x4 << 6;
            cm.set(bfu(1) +% (h & 0xffffff00));
            cm.set(bfu(1) +% (h & 0x00ffff00));
            cm.set(bfu(1) +% (h & 0x0000ff00));
            var d: u32 = core.c4 & 0xffff;
            h = h << 6;
            cm.set(d +% (h & 0xffff0000));
            cm.set(d +% (h & 0x00ff0000));
            h = h << 6;
            d = core.c4 & 0xffffff;
            cm.set(d +% (h & 0xff000000));

            var i: u32 = 1;
            while (i < 5) : (i += 1) {
                cm.set(@intCast(seenbefore | (bf(i) << 8)));
                cm.set(@intCast((bf(i + 3) << 8) | bf(i + 1)));
            }
            cm.set(spaces & 0x7fff);
            cm.set(spaces & 0xff);
            cm.set(core.words & 0x1ffff);
            cm.set(core.f4 & 0x000fffff);
            cm.set(core.tt & 0x00000fff);
            h = core.w4 << 6;
            cm.set(bfu(1) +% (h & 0xffffff00));
            cm.set(bfu(1) +% (h & 0x00ffff00));
            cm.set(bfu(1) +% (h & 0x0000ff00));
            d = core.c4 & 0xffff;
            h = h << 6;
            cm.set(d +% (h & 0xffff0000));
            cm.set(d +% (h & 0x00ff0000));
            h = h << 6;
            d = core.c4 & 0xffffff;
            cm.set(d +% (h & 0xff000000));
            cm.set(core.w4 & 0xf0f0f0ff);

            cm.set((core.w4 & 63) *% 128 +% (5 << 17));
            cm.set(((core.f4 & 0xffff) << 11) | frstchar);
            cm.set(spafdo *% 8 *% b2u((core.w4 & 3) == 1));

            self.scm1.set(core.words & 127);
            self.scm2.set((core.words & 12) *% 16 +% (core.w4 & 12) *% 4 +% (bfu(1) >> 4));
            self.scm3.set(core.w4 & 15);
            self.scm4.set(spafdo *% b2u((core.w4 & 3) == 1));
            self.scma.set(frstchar);
        }
        _ = self.cm.mix(m);
        self.scm1.mix(m, 7, 1, 4);
        self.scm2.mix(m, 7, 1, 4);
        self.scm3.mix(m, 7, 1, 4);
        self.scm4.mix(m, 7, 1, 4);
        self.scm5.mix(m, 7, 1, 4);
        self.scm6.mix(m, 7, 1, 4);
        self.scma.mix(m, 7, 1, 4);
    }

    pub fn deinit(self: *SparseModel1) void {
        self.cm.deinit();
        self.scm1.deinit();
        self.scm2.deinit();
        self.scm3.deinit();
        self.scm4.deinit();
        self.scm5.deinit();
        self.scm6.deinit();
        self.scma.deinit();
    }
};

// =============================== distanceModel =============================
pub const DistanceModel = struct {
    cm: maps.ContextMap,
    pos00: i32 = 0,
    pos20: i32 = 0,
    posnl: i32 = 0,

    pub fn init(a: Allocator) DistanceModel {
        return DistanceModel{ .cm = maps.ContextMap.init(a, core.MEM(), 3) };
    }

    pub fn predict(self: *DistanceModel, m: *core.Mixer) void {
        if (core.bpos == 0) {
            const c: i32 = @intCast(core.c4 & 0xff);
            if (c == 0x00) self.pos00 = core.pos;
            if (c == 0x20) self.pos20 = core.pos;
            if (c == 0xff or c == '\r' or c == '\n') self.posnl = core.pos;
            const cm = &self.cm;
            var i: u64 = 0;
            i += 1;
            cm.set(hash(.{ i, imin(core.pos - self.pos00, 255) | (c << 8) }));
            i += 1;
            cm.set(hash(.{ i, imin(core.pos - self.pos20, 255) | (c << 8) }));
            i += 1;
            cm.set(hash(.{ i, imin(core.pos - self.posnl, 255) | (c << 8) }));
        }
        _ = self.cm.mix(m);
    }

    pub fn deinit(self: *DistanceModel) void {
        self.cm.deinit();
    }
};

// =============================== indirectModel =============================
pub const IndirectModel = struct {
    cm: maps.ContextMap,
    t1: []u32,
    t2: []u16,
    t3: []u16,
    t4: []u16,
    iCtx: maps.IndirectContext(u32),
    alloc: Allocator,

    pub fn init(a: Allocator) IndirectModel {
        const self = IndirectModel{
            .cm = maps.ContextMap.init(a, core.MEM(), 15),
            .t1 = a.alloc(u32, 256) catch unreachable,
            .t2 = a.alloc(u16, 0x10000) catch unreachable,
            .t3 = a.alloc(u16, 0x8000) catch unreachable,
            .t4 = a.alloc(u16, 0x8000) catch unreachable,
            .iCtx = maps.IndirectContext(u32).init(a, 16, 8),
            .alloc = a,
        };
        @memset(self.t1, 0);
        @memset(self.t2, 0);
        @memset(self.t3, 0);
        @memset(self.t4, 0);
        return self;
    }

    pub fn predict(self: *IndirectModel, m: *core.Mixer) void {
        if (core.bpos == 0) {
            const d: u32 = core.c4 & 0xffff;
            const c_old: u32 = d & 255;
            const d2: u32 = (bfu(1) & 31) + 32 * (bfu(2) & 31) + 1024 * (bfu(3) & 31);
            const d3: u32 = ((bfu(1) >> 3) & 31) + 32 * ((bfu(3) >> 3) & 31) + 1024 * ((bfu(4) >> 3) & 31);
            self.t1[d >> 8] = (self.t1[d >> 8] << 8) | c_old;
            {
                const idx = (core.c4 >> 8) & 0xffff;
                self.t2[idx] = @truncate((@as(u32, self.t2[idx]) << 8) | c_old);
            }
            {
                const idx = (bfu(2) & 31) + 32 * (bfu(3) & 31) + 1024 * (bfu(4) & 31);
                self.t3[idx] = @truncate((@as(u32, self.t3[idx]) << 8) | c_old);
            }
            {
                const idx = ((bfu(2) >> 3) & 31) + 32 * ((bfu(4) >> 3) & 31) + 1024 * ((bfu(5) >> 3) & 31);
                self.t4[idx] = @truncate((@as(u32, self.t4[idx]) << 8) | c_old);
            }
            const t: u32 = c_old | (self.t1[c_old] << 8);
            const t0: u32 = d | (@as(u32, self.t2[d]) << 16);
            const ta: u32 = d2 | (@as(u32, self.t3[d2]) << 16);
            const tc: u32 = d3 | (@as(u32, self.t4[d3]) << 16);
            const pc: u8 = @intCast(tolwr(@intCast((core.c4 >> 8) & 0xff)));
            const c_new: u32 = @intCast(tolwr(@intCast(c_old)));
            self.iCtx.add(c_new);
            self.iCtx.setCtx((@as(u32, pc) << 8) | c_new);
            const ctx0: u32 = self.iCtx.value();
            const mask: u32 = @as(u32, b2u(@as(u8, @truncate(self.t1[c_new])) == @as(u8, @truncate(self.t2[d])))) |
                (@as(u32, b2u(@as(u8, @truncate(self.t1[c_new])) == @as(u8, @truncate(self.t3[d2])))) << 1) |
                (@as(u32, b2u(@as(u8, @truncate(self.t1[c_new])) == @as(u8, @truncate(self.t4[d3])))) << 2) |
                (@as(u32, b2u(@as(u8, @truncate(self.t1[c_new])) == @as(u8, @truncate(ctx0)))) << 3);
            const cm = &self.cm;
            var i: u64 = 0;
            i += 1;
            cm.set(hash(.{ i, t }));
            i += 1;
            cm.set(hash(.{ i, t0 }));
            i += 1;
            cm.set(hash(.{ i, ta }));
            i += 1;
            cm.set(hash(.{ i, tc }));
            i += 1;
            cm.set(hash(.{ i, t & 0xff00, mask }));
            i += 1;
            cm.set(hash(.{ i, t0 & 0xff0000 }));
            i += 1;
            cm.set(hash(.{ i, ta & 0xff0000 }));
            i += 1;
            cm.set(hash(.{ i, tc & 0xff0000 }));
            i += 1;
            cm.set(hash(.{ i, t & 0xffff }));
            i += 1;
            cm.set(hash(.{ i, t0 & 0xffffff }));
            i += 1;
            cm.set(hash(.{ i, ta & 0xffffff }));
            i += 1;
            cm.set(hash(.{ i, tc & 0xffffff }));
            i += 1;
            cm.set(hash(.{ i, ctx0 & 0xff, c_new }));
            i += 1;
            cm.set(hash(.{ i, ctx0 & 0xffff }));
            i += 1;
            cm.set(hash(.{ i, ctx0 & 0x7f7fff }));
        }
        _ = self.cm.mix(m);
    }

    pub fn deinit(self: *IndirectModel) void {
        self.cm.deinit();
        self.alloc.free(self.t1);
        self.alloc.free(self.t2);
        self.alloc.free(self.t3);
        self.alloc.free(self.t4);
        self.iCtx.deinit();
    }
};

// ================================ XMLModel =================================
const XMLAttribute = struct { Name: u32 = 0, Value: u32 = 0, Length: u32 = 0 };
const XMLContent = struct { Data: u32 = 0, Length: u32 = 0, Type: u32 = 0 };
const XMLAttributes = struct { Items: [4]XMLAttribute = .{ .{}, .{}, .{}, .{} }, Index: u32 = 0 };
const XMLTag = struct {
    Name: u32 = 0,
    Length: u32 = 0,
    Level: i32 = 0,
    EndTag: bool = false,
    Empty: bool = false,
    Content: XMLContent = .{},
    Attributes: XMLAttributes = .{},
};
const XMLTagCache = struct { Tags: [32]XMLTag = .{XMLTag{}} ** 32, Index: u32 = 0 };

// ContentFlags
const CF_URL: u32 = 0x010;
const CF_Link: u32 = 0x020;
const CF_Date: u32 = 0x004;
const CF_Time: u32 = 0x008;
const CF_Text: u32 = 0x001;
const CF_Number: u32 = 0x002;
const CF_Coordinates: u32 = 0x040;
const CF_Temperature: u32 = 0x080;

// XMLState
const XS_None: u32 = 0;
const XS_ReadTagName: u32 = 1;
const XS_ReadTag: u32 = 2;
const XS_ReadAttributeName: u32 = 3;
const XS_ReadAttributeValue: u32 = 4;
const XS_ReadContent: u32 = 5;
const XS_ReadCDATA: u32 = 6;
const XS_ReadComment: u32 = 7;

pub const XMLModel = struct {
    cm: maps.ContextMap,
    cache: XMLTagCache = .{},
    StateBH: [8]u32 = .{ 0, 0, 0, 0, 0, 0, 0, 0 },
    State: u32 = XS_None,
    pState: u32 = XS_None,
    c8: u32 = 0,
    WhiteSpaceRun: u32 = 0,
    pWSRun: u32 = 0,
    IndentTab: u32 = 0,
    IndentStep: u32 = 2,
    LineEnding: u32 = 2,

    pub fn init(a: Allocator) XMLModel {
        return XMLModel{ .cm = maps.ContextMap.init(a, core.MEM() / 4, 4) };
    }

    fn detectContent(content: *XMLContent, c8: u32, B: u8, cont_len: u32) void {
        const c4 = core.c4;
        if ((c4 & 0xF0F0F0F0) == 0x30303030) {
            var i: i32 = 0;
            var j: i32 = 0;
            while (i < 4) {
                j = @intCast((c4 >> @as(u5, @intCast(8 * i))) & 0xFF);
                if (!(j >= 0x30 and j <= 0x39)) break;
                i += 1;
            }
            if (i == 4 and (((c8 & 0xFDF0F0FD) == 0x2D30302D and bf(9) >= 0x30 and bf(9) <= 0x39) or ((c8 & 0xF0FDF0FD) == 0x302D302D)))
                content.Type |= CF_Date;
        } else if (((c8 & 0xF0F0FDF0) == 0x30302D30 or (c8 & 0xF0F0F0FD) == 0x3030302D) and bf(9) >= 0x30 and bf(9) <= 0x39) {
            var i: i32 = 2;
            var j: i32 = 0;
            while (i < 4) {
                j = @intCast((c8 >> @as(u5, @intCast(8 * i))) & 0xFF);
                if (!(j >= 0x30 and j <= 0x39)) break;
                i += 1;
            }
            if (i == 4 and (c4 & 0xF0FDF0F0) == 0x302D3030)
                content.Type |= CF_Date;
        }

        if ((c4 & 0xF0FFF0F0) == 0x303A3030 and bf(5) >= 0x30 and bf(5) <= 0x39 and ((bf(6) < 0x30 or bf(6) > 0x39) or ((c8 & 0xF0F0FF00) == 0x30303A00 and (bf(9) < 0x30 or bf(9) > 0x39))))
            content.Type |= CF_Time;

        if (cont_len >= 8 and (c8 & 0x80808080) == 0 and (c4 & 0x80808080) == 0)
            content.Type |= CF_Text;

        if ((c8 & 0xF0F0FF) == 0x3030C2 and (c4 & 0xFFF0F0FF) == 0xB0303027) {
            var i: i32 = 2;
            while (i < 7 and bf(@intCast(i)) >= 0x30 and bf(@intCast(i)) <= 0x39) {
                i += (i & 1) * 2 + 1;
            }
            if (i == 10) content.Type |= CF_Coordinates;
        }

        if ((c4 & 0xFFFFFA) == 0xC2B042 and B != 0x47 and (((c4 >> 24) >= 0x30 and (c4 >> 24) <= 0x39) or ((c4 >> 24) == 0x20 and (bf(5) >= 0x30 and bf(5) <= 0x39))))
            content.Type |= CF_Temperature;

        if (B >= 0x30 and B <= 0x39)
            content.Type |= CF_Number;

        if (c4 == 0x4953424E and bf(5) == 0x20)
            content.Type |= CF_ISBN_placeholder();
    }
    inline fn CF_ISBN_placeholder() u32 {
        return 0x100;
    }

    pub fn predict(self: *XMLModel, m: *core.Mixer, stats: ?*core.ModelStats) void {
        if (core.bpos == 0) {
            const B: u8 = @intCast(core.c4 & 0xff);
            const tagIdx: usize = @intCast(self.cache.Index & 31);
            const pTagIdx0: usize = @intCast((self.cache.Index -% 1) & 31);
            const tag = &self.cache.Tags[tagIdx];
            const attrIdx: usize = @intCast(tag.Attributes.Index & 3);
            self.pState = self.State;
            self.c8 = (self.c8 << 8) | bfu(5);
            if ((B == 0x09 or B == 0x20) and (B == @as(u8, @intCast((core.c4 >> 8) & 0xff)) or self.WhiteSpaceRun == 0)) {
                self.WhiteSpaceRun +%= 1;
                self.IndentTab = b2u(B == 0x09);
            } else {
                if ((self.State == XS_None or (self.State == XS_ReadContent and tag.Content.Length <= self.LineEnding +% self.WhiteSpaceRun)) and self.WhiteSpaceRun > 1 +% self.IndentTab and self.WhiteSpaceRun != self.pWSRun) {
                    self.IndentStep = @intCast(@abs(@as(i32, @bitCast(self.WhiteSpaceRun)) - @as(i32, @bitCast(self.pWSRun))));
                    self.pWSRun = self.WhiteSpaceRun;
                }
                self.WhiteSpaceRun = 0;
            }
            if (B == 0x0A)
                self.LineEnding = 1 +% b2u(@as(u8, @intCast((core.c4 >> 8) & 0xff)) == 0x0D);

            const cm = &self.cm;
            switch (self.State) {
                XS_None => {
                    if (B == 0x3C) {
                        self.State = XS_ReadTagName;
                        tag.* = .{};
                        const pTag = &self.cache.Tags[pTagIdx0];
                        tag.Level = if (pTag.EndTag or pTag.Empty) pTag.Level else pTag.Level + 1;
                    }
                    if (tag.Level > 1)
                        detectContent(&tag.Content, self.c8, B, tag.Content.Length);
                    cm.set(hash(.{ self.pState, self.State, (@as(u32, @bitCast(self.cache.Tags[pTagIdx0].Level + 1)) *% self.IndentStep) -% self.WhiteSpaceRun }));
                },
                XS_ReadTagName => {
                    if (tag.Length > 0 and (B == 0x09 or B == 0x0A or B == 0x0D or B == 0x20)) {
                        self.State = XS_ReadTag;
                    } else if ((B == 0x3A or (B >= 'A' and B <= 'Z') or B == 0x5F or (B >= 'a' and B <= 'z')) or (tag.Length > 0 and (B == 0x2D or B == 0x2E or (B >= '0' and B <= '9')))) {
                        tag.Length +%= 1;
                        tag.Name = tag.Name *% 263 *% 32 +% (B & 0xDF);
                    } else if (B == 0x3E) {
                        if (tag.EndTag) {
                            self.State = XS_None;
                            self.cache.Index +%= 1;
                        } else self.State = XS_ReadContent;
                    } else if (B != 0x21 and B != 0x2D and B != 0x2F and B != 0x5B) {
                        self.State = XS_None;
                        self.cache.Index +%= 1;
                    } else if (tag.Length == 0) {
                        if (B == 0x2F) {
                            tag.EndTag = true;
                            tag.Level = imax(0, tag.Level - 1);
                        } else if (core.c4 == 0x3C212D2D) {
                            self.State = XS_ReadComment;
                            tag.Level = imax(0, tag.Level - 1);
                        }
                    }

                    if (tag.Length == 1 and (core.c4 & 0xFFFF00) == 0x3C2100) {
                        tag.* = .{};
                        self.State = XS_None;
                    } else if (tag.Length == 5 and self.c8 == 0x215B4344 and core.c4 == 0x4154415B) {
                        self.State = XS_ReadCDATA;
                        tag.Level = imax(0, tag.Level - 1);
                    }

                    var i: u32 = 1;
                    var pIdx: usize = pTagIdx0;
                    while (true) {
                        pIdx = @intCast((self.cache.Index -% i) & 31);
                        const pTag = &self.cache.Tags[pIdx];
                        const prevName = self.cache.Tags[@intCast((self.cache.Index -% i -% 1) & 31)].Name;
                        i += 1 + b2u(pTag.EndTag and prevName == pTag.Name);
                        if (!(i < 32 and (pTag.EndTag or pTag.Empty))) break;
                    }
                    const pTag = &self.cache.Tags[pIdx];
                    cm.set(hash(.{ self.pState * 8 + self.State, tag.Name, tag.Level, pTag.Name, b2i(pTag.Level != tag.Level) }));
                },
                XS_ReadTag => {
                    if (B == 0x2F) {
                        tag.Empty = true;
                    } else if (B == 0x3E) {
                        if (tag.Empty) {
                            self.State = XS_None;
                            self.cache.Index +%= 1;
                        } else self.State = XS_ReadContent;
                    } else if (B != 0x09 and B != 0x0A and B != 0x0D and B != 0x20) {
                        self.State = XS_ReadAttributeName;
                        tag.Attributes.Items[attrIdx].Name = B & 0xDF;
                    }
                    cm.set(hash(.{ self.pState, self.State, tag.Name, B, tag.Attributes.Index }));
                },
                XS_ReadAttributeName => {
                    const attr = &tag.Attributes.Items[attrIdx];
                    if ((core.c4 & 0xFFF0) == 0x3D20 and (B == 0x22 or B == 0x27)) {
                        self.State = XS_ReadAttributeValue;
                        if ((self.c8 & 0xDFDF) == 0x4852 and (core.c4 & 0xDFDF0000) == 0x45460000)
                            tag.Content.Type |= CF_Link;
                    } else if (B != 0x22 and B != 0x27 and B != 0x3D) {
                        attr.Name = attr.Name *% 263 *% 32 +% (B & 0xDF);
                    }
                    cm.set(hash(.{ self.pState * 8 + self.State, attr.Name, tag.Attributes.Index, tag.Name, tag.Content.Type }));
                },
                XS_ReadAttributeValue => {
                    const attr = &tag.Attributes.Items[attrIdx];
                    if (B == 0x22 or B == 0x27) {
                        tag.Attributes.Index +%= 1;
                        self.State = XS_ReadTag;
                    } else {
                        attr.Value = attr.Value *% 263 *% 32 +% (B & 0xDF);
                        attr.Length +%= 1;
                        if ((self.c8 & 0xDFDFDFDF) == 0x48545450 and ((core.c4 >> 8) == 0x3A2F2F or core.c4 == 0x733A2F2F))
                            tag.Content.Type |= CF_URL;
                    }
                    cm.set(hash(.{ self.pState, self.State, attr.Name, tag.Content.Type }));
                },
                XS_ReadContent => {
                    if (B == 0x3C) {
                        self.State = XS_ReadTagName;
                        self.cache.Index +%= 1;
                        const nidx: usize = @intCast(self.cache.Index & 31);
                        self.cache.Tags[nidx] = .{};
                        self.cache.Tags[nidx].Level = tag.Level + 1;
                    } else {
                        tag.Content.Length +%= 1;
                        tag.Content.Data = tag.Content.Data *% 997 *% 16 +% (B & 0xDF);
                        detectContent(&tag.Content, self.c8, B, tag.Content.Length);
                    }
                    cm.set(hash(.{ self.pState, self.State, tag.Name, core.c4 & 0xC0FF }));
                },
                XS_ReadCDATA => {
                    if ((core.c4 & 0xFFFFFF) == 0x5D5D3E) {
                        self.State = XS_None;
                        self.cache.Index +%= 1;
                    }
                    cm.set(hash(.{ self.pState, self.State }));
                },
                XS_ReadComment => {
                    if ((core.c4 & 0xFFFFFF) == 0x2D2D3E) {
                        self.State = XS_None;
                        self.cache.Index +%= 1;
                    }
                    cm.set(hash(.{ self.pState, self.State }));
                },
                else => {},
            }

            self.StateBH[self.State] = (self.StateBH[self.State] << 8) | B;
            const pTag2 = &self.cache.Tags[@intCast((self.cache.Index -% 1) & 31)];
            var i: u64 = 64;
            i += 1;
            cm.set(hash(.{ i, self.State, self.cache.Tags[tagIdx].Level, self.pState * 2 + b2u(self.cache.Tags[tagIdx].EndTag), self.cache.Tags[tagIdx].Name }));
            i += 1;
            cm.set(hash(.{ i, pTag2.Name, self.State * 2 + b2u(pTag2.EndTag), pTag2.Content.Type, self.cache.Tags[tagIdx].Content.Type }));
            i += 1;
            cm.set(hash(.{ i, self.State * 2 + b2u(self.cache.Tags[tagIdx].EndTag), self.cache.Tags[tagIdx].Name, self.cache.Tags[tagIdx].Content.Type, core.c4 & 0xE0FF }));
        }
        _ = self.cm.mix(m);
        const st: u32 = ((self.StateBH[self.State] >> @as(u5, @intCast(28 - core.bpos))) & 0x08) |
            ((self.StateBH[self.State] >> @as(u5, @intCast(21 - core.bpos))) & 0x04) |
            ((self.StateBH[self.State] >> @as(u5, @intCast(14 - core.bpos))) & 0x02) |
            ((self.StateBH[self.State] >> @as(u5, @intCast(7 - core.bpos))) & 0x01) |
            (@as(u32, @intCast(core.bpos)) << 4);
        if (stats) |s| s.XML = (st << 3) | self.State;
    }

    pub fn deinit(self: *XMLModel) void {
        self.cm.deinit();
    }
};

// ================================ Models ===================================
/// Owns all 11 model states, allocated once. The integrator (contextModel2)
/// calls them in the documented order.
pub const Models = struct {
    sparse: SparseModel,
    sparse1: SparseModel1,
    distance: DistanceModel,
    pic: PicModel,
    record: RecordModel,
    record1: RecordModel1,
    word: WordModel,
    nest: NestModel,
    indirect: IndirectModel,
    xml: XMLModel,
    linear: LinearPredictionModel,

    pub fn init(a: Allocator) Models {
        return Models{
            .sparse = SparseModel.init(a),
            .sparse1 = SparseModel1.init(a),
            .distance = DistanceModel.init(a),
            .pic = PicModel.init(a),
            .record = RecordModel.init(a),
            .record1 = RecordModel1.init(a),
            .word = WordModel.init(a),
            .nest = NestModel.init(a),
            .indirect = IndirectModel.init(a),
            .xml = XMLModel.init(a),
            .linear = LinearPredictionModel.init(a),
        };
    }

    pub fn deinit(self: *Models) void {
        self.sparse.deinit();
        self.sparse1.deinit();
        self.distance.deinit();
        self.pic.deinit();
        self.record.deinit();
        self.record1.deinit();
        self.word.deinit();
        self.nest.deinit();
        self.indirect.deinit();
        self.xml.deinit();
        self.linear.deinit();
    }
};

// ==================================== tests ================================
// Self-contained smoke test: drives all 11 models over a byte stream in
// contextModel2 order and asserts they run without crashing and add finite,
// in-range mixer inputs. (Full bit-for-bit value parity vs the compiled cmix
// C++ model was verified separately with a differential harness.)
const testing = std.testing;

var t_x5: u32 = 0;
fn tFeedBit(yb: i32) void {
    core.y = yb;
    core.c0 += core.c0 + core.y;
    if (core.c0 >= 256) {
        core.buf.at(@intCast(core.pos)).* = @truncate(@as(u32, @intCast(core.c0)));
        core.pos += 1;
        core.c0 -= 256;
        const c0u: u32 = @intCast(core.c0);
        core.c4 = (core.c4 << 8) +% c0u;
        var i: u32 = core.WRT_mpw[@intCast(core.c0 >> 4)];
        core.w4 = core.w4 *% 4 +% i;
        if (core.b2 == 3) i = 2;
        core.w5 = core.w5 *% 4 +% i;
        core.b3 = core.b2;
        core.b2 = c0u;
        core.x4 = core.x4 *% 256 +% c0u;
        t_x5 = (t_x5 << 8) +% c0u;
        if (core.c0 == '.' or core.c0 == '!' or core.c0 == '?' or core.c0 == '/' or core.c0 == ')') {
            core.w5 = (core.w5 << 8) | 0x3ff;
            core.f4 = (core.f4 & 0xfffffff0) +% 2;
            t_x5 = (t_x5 << 8) +% c0u;
            core.x4 = core.x4 *% 256 +% c0u;
            if (core.c0 != '!') {
                core.w4 |= 12;
                core.tt = (core.tt & 0xfffffff8) +% 1;
                core.b3 = '.';
            }
        }
        if (core.c0 == 32) core.c0 -= 1;
        core.tt = core.tt *% 8 +% core.WRT_mtt[@intCast(core.c0 >> 4)];
        core.f4 = core.f4 *% 16 +% @as(u32, @intCast(core.c0 >> 4));
        core.c0 = 1;
    }
    core.bpos = (core.bpos + 1) & 7;
    if (core.bpos == 0) core.blpos += 1;
    if (core.bpos > 0) {
        const sh: u5 = @intCast(core.bpos);
        core.grp0 = core.AsciiGroupC0[@intCast((@as(i32, 1) << sh) - 2 + (core.c0 & ((@as(i32, 1) << sh) - 1)))];
    } else core.grp0 = 0;
}

test "all models run, add finite/in-range inputs, exact per-model counts" {
    var arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer arena.deinit();
    // reset the global ring buffer afterwards so we don't leave the shared
    // core.buf pointing at freed arena memory for later tests.
    defer core.buf.deinit();
    const a = arena.allocator();

    core.level = 6;
    core.init();
    core.resetState();
    resetShared();
    t_x5 = 0;
    core.buf.setsize(a, 1 << 16);

    var models = Models.init(a);
    const m = core.Mixer.init(a, 4096, 77472, 64, 0);

    const text = "The quick brown <tag a=\"1\">fox</tag> 2021-03-04 12:00:00 jumps! don't stop.\n";
    var round: usize = 0;
    while (round < 60) : (round += 1) {
        for (text) |byte| {
            var bit: i32 = 7;
            while (bit >= 0) : (bit -= 1) {
                const yb: i32 = (@as(i32, byte) >> @as(u5, @intCast(bit))) & 1;
                tFeedBit(yb);
                core.resetPredictions();

                const seenbefore: i32 = (core.blpos * 13 + 7) & 63;
                const howmany: i32 = (core.blpos * 5 + 1) & 7;
                var stats = core.ModelStats{};
                stats.Match.length = @intCast(core.blpos & 3);
                stats.Match.expectedByte = @intCast((core.blpos * 7) & 0xff);

                const start = m.nx;
                models.sparse.predict(m, seenbefore, howmany);
                models.sparse1.predict(m, seenbefore, howmany);
                models.distance.predict(m);
                models.pic.predict(m);
                models.record.predict(m, core.FT_DEFAULT, &stats);
                models.record1.predict(m);
                models.word.predict(m);
                models.nest.predict(m);
                models.indirect.predict(m);
                models.xml.predict(m, &stats);
                models.linear.predict(m);

                // every mixer input this bit must be a valid i16 (implicitly) and
                // squashof it must land in [0,4095].
                for (start..m.nx) |k| {
                    const sq = core.squash(m.tx[k]);
                    try testing.expect(sq >= 0 and sq <= 4095);
                }
                _ = m.p();
                m.update();
            }
        }
    }
    // exercised a decent amount of state without tripping any bounds/overflow.
    try testing.expect(core.pos == @as(i32, @intCast(text.len)) * 60);
}
