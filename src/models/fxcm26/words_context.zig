//! WordsContext — bit-exact Zig port of cmix-lex `fxcmv1.cpp` `struct WordsContext`
//! (+ vec). A word-history ring buffer (sbytes/type/stem/capital/codeword) whose
//! accessors (Word/Type/Code/sBytes/Capital/Word0/Last*/...) supply the args to many
//! cm.set calls in the preamble.
//!
//! Ported 1:1 from words_context_v26.rs. Validated bit-exact vs
//! v26-port-tools/words_oracle.cpp over a synthetic drive (folded into a checksum,
//! same drive reproduced in the test below).
//!
//! Zig notes: Rust allows a field `capital` alongside a method `capital`; Zig does
//! not, so the private field is renamed `capital_v`. Likewise the `V::size` method
//! (which shadows the `size` field in Zig) is renamed `len`.

const std = @import("std");

const CAP: usize = 256; // 64*4
const LF: u32 = 10;

inline fn imin(a: i32, b: i32) i32 {
    return if (a < b) a else b;
}

/// fxcmv1.cpp `vec<T,256>` over u32 storage (u16/u8 values fit; masked on push).
const V = struct {
    data: [CAP]u32,
    size: usize,

    fn new() V {
        return V{ .data = [_]u32{0} ** CAP, .size = 0 };
    }

    inline fn push(self: *V, e: u32) void {
        if (self.size >= CAP) {
            self.size = CAP - 1;
        }
        self.data[self.size] = e;
        self.size += 1;
    }

    inline fn at(self: *const V, index: i32) u32 {
        if (index < 0 or @as(usize, @intCast(index)) >= CAP) {
            return 0;
        } else {
            return self.data[@intCast(index)];
        }
    }

    inline fn len(self: *const V) i32 {
        return @intCast(self.size);
    }

    inline fn pop(self: *V) void {
        if (self.size > 0) {
            self.size -= 1;
            self.data[self.size] = 0;
        }
    }

    inline fn reset(self: *V) void {
        self.data[0] = 0;
        self.size = 0;
    }
};

pub const WordsContext = struct {
    sbytes: V,
    type_: V,
    stem: V,
    capital_v: V,
    codeword: V,
    fword: u32,
    ftype: u32,
    pbyte: u8,
    tpbyte: u8,
    wordcount: i32,
    upper: i32,
    codesum: u32,
    paragraph: bool,
    wor_in_par: i32,
    wor_in_link: i32,

    pub fn new() WordsContext {
        var w = WordsContext{
            .sbytes = V.new(),
            .type_ = V.new(),
            .stem = V.new(),
            .capital_v = V.new(),
            .codeword = V.new(),
            .fword = 0,
            .ftype = 0,
            .pbyte = 0,
            .tpbyte = 0,
            .wordcount = 0,
            .upper = 0,
            .codesum = 0,
            .paragraph = false,
            .wor_in_par = 0,
            .wor_in_link = 0,
        };
        w.reset();
        return w;
    }

    pub fn reset(self: *WordsContext) void {
        self.sbytes.reset();
        self.type_.reset();
        self.stem.reset();
        self.capital_v.reset();
        self.codeword.reset();
        self.fword = 0;
        self.pbyte = 0;
        self.wordcount = 0;
        self.upper = 0;
        self.ftype = 0;
        self.codesum = 0;
        self.tpbyte = 0;
        self.paragraph = false;
        self.wor_in_par = 0;
        self.wor_in_link = 0;
    }

    pub fn set(self: *WordsContext, b: u8, a: i32, g: u8) void {
        self.pbyte = b;
        self.upper = a;
        self.tpbyte = g;
    }

    pub inline fn tpbyteVal(self: *const WordsContext) u8 {
        return self.tpbyte;
    }

    pub fn update(self: *WordsContext, w: u32, b: u8, t: u32, s: u32, cw: u32, wor_in: i32) void {
        if (self.fword == 0) {
            self.fword = w;
        }
        self.sbytes.push((@as(u32, self.pbyte) *% 256 +% @as(u32, b)) & 0xffff);
        self.type_.push(t);
        self.stem.push(s);
        self.capital_v.push(@as(u32, @bitCast(self.upper)) & 0xff);
        self.codeword.push(cw);
        self.pbyte = 0;
        self.tpbyte = 0;
        self.wordcount += 1;
        self.codesum = self.codesum +% cw;
        if (self.ftype == 0 and t != 0) {
            self.ftype = t;
        }
        if (wor_in == 1) {
            self.wor_in_link += 1;
        } else if (wor_in == 0) {
            self.wor_in_par += 1;
        }
    }

    pub fn remove(self: *WordsContext) void {
        if (self.stem.len() != 0) {
            self.sbytes.pop();
            self.type_.pop();
            self.stem.pop();
            self.capital_v.pop();
            self.codeword.pop();
            if (self.wordcount != 0) {
                self.wordcount -= 1;
            }
        }
    }

    pub fn word(self: *const WordsContext, i: i32) u32 {
        const num = self.stem.len();
        if (i <= 0) {
            return 0;
        }
        if (num >= i) {
            return self.stem.at(num -% i);
        } else {
            return 0;
        }
    }

    pub fn word_r(self: *const WordsContext, i: i32) u32 {
        const low = imin(self.wordcount, i);
        return self.word(self.wordcount -% low);
    }

    pub fn s_bytes(self: *const WordsContext, i: i32) u16 {
        const num = self.sbytes.len();
        if (i <= 0) {
            return 0;
        }
        if (num >= i) {
            return @truncate(self.sbytes.at(num -% i));
        } else {
            return 0;
        }
    }

    pub fn type_at(self: *const WordsContext, i: i32) u32 {
        const num = self.type_.len();
        if (i <= 0) {
            return 0;
        }
        if (num >= i) {
            return self.type_.at(num -% i);
        } else {
            return 0;
        }
    }

    pub fn capital(self: *const WordsContext, i: i32) u8 {
        const num = self.capital_v.len();
        if (i <= 0) {
            return 0;
        }
        if (num >= i) {
            return @truncate(self.capital_v.at(num -% i));
        } else {
            return 0;
        }
    }

    pub inline fn code(self: *const WordsContext, i: i32) u32 {
        const num = self.codeword.len();
        if (i <= 0) {
            return 0;
        }
        if (num >= i) {
            return self.codeword.at(num -% i);
        } else {
            return 0;
        }
    }

    pub inline fn code_r(self: *const WordsContext, i: i32, j: i32) u32 {
        const low = imin(self.wordcount, j);
        return self.code(i +% (self.wordcount -% low));
    }

    pub fn last_r(self: *const WordsContext, i: i32, j: i32, t: u32) u32 {
        const low = imin(self.wordcount, j);
        return self.last(i +% (self.wordcount -% low), t);
    }

    pub fn last(self: *const WordsContext, j: i32, t: u32) u32 {
        const num = self.type_.len();
        if (j <= 0) {
            return 0;
        }
        if (t == 0) {
            return self.word(j);
        }
        if (num >= j) {
            var i = j;
            while (i < num) : (i += 1) {
                const typ = self.type_at(i);
                if (typ & t != 0) {
                    return self.word(i);
                }
            }
        }
        return self.word(j);
    }

    pub fn last_if(self: *const WordsContext, j: i32, t: u32) u32 {
        const num = self.type_.len();
        if (j <= 0) {
            return 0;
        }
        if (t == 0) {
            return self.word(j);
        }
        if (num >= j) {
            var i = j;
            while (i < num) : (i += 1) {
                const typ = self.type_at(i);
                if (typ & t != 0) {
                    return self.word(i);
                }
            }
        }
        return 0;
    }

    pub fn last_idx(self: *const WordsContext, j: i32, t: u32) u32 {
        const num = self.type_.len();
        if (t == 0) {
            return 0;
        }
        if (num >= j) {
            var i = j;
            while (i < num) : (i += 1) {
                const typ = self.type_at(i);
                if (typ & t != 0) {
                    return @intCast(i);
                }
            }
        }
        return 0;
    }

    pub fn word0(self: *const WordsContext, i: i32) u32 {
        const num = self.stem.len();
        if (i <= 0) {
            return 0;
        }
        const lb: u32 = @as(u32, self.s_bytes(i) & 0xff);
        var idx: u32 = switch (i) {
            4 => 37 * 47 * 53 * 83,
            3 => 47 * 53 * 83,
            2 => 53 * 83,
            1 => 83,
            else => 0,
        };
        if (lb == LF) {
            idx = switch (i) {
                4 => idx *% 37,
                3 => idx *% 47,
                2 => idx *% 53,
                1 => idx *% 83,
                else => idx,
            };
        }
        if (num >= i) {
            return self.stem.at(num -% i) *% idx;
        } else {
            return 0;
        }
    }

    pub fn remove_words_l(self: *WordsContext, len_: i32, c: u8, d: u8, f: bool) void {
        if (@as(u8, @truncate(self.s_bytes(1) & 0xff)) == d) {
            var i: i32 = 1;
            while (i < len_) : (i += 1) {
                if (@as(u8, @truncate(self.s_bytes(i) >> 8)) == c) {
                    while (@as(u8, @truncate(self.s_bytes(1) >> 8)) != c) {
                        self.remove();
                    }
                    if (f) {
                        self.remove();
                    }
                    break;
                }
            }
        }
    }

    pub fn remove_words_r(self: *WordsContext, len_: i32, c: u8, d: u8, f: bool) void {
        if (@as(u8, @truncate(self.s_bytes(1) & 0xff)) == d) {
            var i: i32 = 1;
            while (i < len_) : (i += 1) {
                if (@as(u8, @truncate(self.s_bytes(i) & 0xff)) == c) {
                    while (@as(u8, @truncate(self.s_bytes(1) & 0xff)) != c) {
                        self.remove();
                    }
                    if (f) {
                        self.remove();
                    }
                    break;
                }
            }
        }
    }
};

test "words_context_matches_cpp_oracle" {
    var wc = WordsContext.new();
    var cs: u64 = 0;
    var r: u32 = 0x12345678;
    var iter: usize = 0;
    while (iter < 6000) : (iter += 1) {
        r = r *% 1664525 +% 1013904223;
        const op = r & 15;
        if (op == 0) {
            wc.reset();
        } else if (op == 1) {
            wc.remove();
        } else if (op == 2) {
            wc.remove_words_l(8, 80, 82, true);
        } else if (op == 3) {
            wc.remove_words_r(8, 80, 82, true);
        } else {
            wc.set(@truncate(r >> 8), @intCast((r >> 16) & 1), @truncate(r >> 20));
            r = r *% 1664525 +% 1013904223;
            const w = r *% 2654435761 & 0xfffff;
            const t = (r >> 5) & 0x3ff;
            const s = (r >> 10) & 0xffff;
            const cw = (r >> 2) & 0xff;
            wc.update(w, @truncate(r >> 3), t, s, cw, @intCast((r >> 14) & 1));
        }
        var k: i32 = 1;
        while (k <= 4) : (k += 1) {
            cs = cs *% 1000003 +% @as(u64, wc.word(k));
            cs = cs *% 1000003 +% @as(u64, wc.type_at(k));
            cs = cs *% 1000003 +% @as(u64, wc.code(k));
            cs = cs *% 1000003 +% @as(u64, wc.s_bytes(k));
            cs = cs *% 1000003 +% @as(u64, wc.capital(k));
            cs = cs *% 1000003 +% @as(u64, wc.word0(k));
        }
        cs = cs *% 1000003 +% @as(u64, @as(u32, @bitCast(wc.wordcount)));
        cs = cs *% 1000003 +% @as(u64, wc.fword);
        cs = cs *% 1000003 +% @as(u64, wc.ftype);
        cs = cs *% 1000003 +% @as(u64, wc.codesum);
        cs = cs *% 1000003 +% @as(u64, wc.word_r(1));
        cs = cs *% 1000003 +% @as(u64, wc.word_r(2));
        cs = cs *% 1000003 +% @as(u64, wc.code_r(1, 1));
        cs = cs *% 1000003 +% @as(u64, wc.last(1, (r >> 7) & 0x3ff));
        cs = cs *% 1000003 +% @as(u64, wc.last_r(1, 1, (r >> 9) & 0x3ff));
        cs = cs *% 1000003 +% @as(u64, wc.last_if(1, (r >> 11) & 0x3ff));
        cs = cs *% 1000003 +% @as(u64, wc.last_idx(1, (r >> 13) & 0x3ff));
    }
    try std.testing.expectEqual(@as(u64, 0x540e068f6e5d1dd7), cs);
}
