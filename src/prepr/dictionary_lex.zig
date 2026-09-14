//! Port of cmix-lex's preprocessor `Dictionary` class
//! (`ref/cmix-lex/src/preprocess/dictionary.{h,cpp}`).
//!
//! This is the WRT word-transform / english.dic substitution used by the `-c`/`-s`
//! preprocessing pipeline. It is DISTINCT from `models/fxcm26/dictionary.zig` (which
//! is fxcm's parallel codeword→index context consumer). Here we physically transform
//! the byte stream: words → 1/2/3-byte codewords, with case markers + escaping.
//!
//! Ground-truth C++ line refs are in comments (dictionary.cpp).
//!
//! NOTE: `/tmp/cmix_full` on the fleet is the STANDARD cmix build (TEXT filetype=4,
//! mode byte=1, NO final byte permutation, an extra 4th codeword tier, and a `kQuote`
//! escape) — it is NOT cmix-lex. This module targets cmix-lex (TEXT=7, mode byte=7,
//! final involutive permutation, 3 codeword tiers, escape set {0x06,0x07,0x0C,0x40,>=0x80}).

const std = @import("std");
const prepr_options = @import("prepr_options");

/// LAB frontend-earnings pessimum control (-Dwrt-earn). See build.zig.
/// `stock` is the shipped path and is comptime-dead-identical.
pub const WRT_EARN = prepr_options.wrt_earn;
pub const NOSUBST: bool = std.mem.eql(u8, WRT_EARN, "nosubst");
pub const NOCASE: bool = std.mem.eql(u8, WRT_EARN, "nocase");
pub const NOWRT: bool = std.mem.eql(u8, WRT_EARN, "nowrt");
/// LAB CEILING probe (-Dwrt-earn=nomark). Case folding is IDENTICAL to stock, so
/// every substitution that fires in stock fires here byte for byte; only the three
/// marker emissions are suppressed. The emitted stream is therefore stock's stream
/// with exactly the case-marker bytes DELETED. NOT lossless — that is the point.
pub const NOMARK: bool = std.mem.eql(u8, WRT_EARN, "nomark");

/// LAB per-temp-byte class + coded-cost dump (-Dwrt-markmap). Empty = comptime-dead.
pub const MARKMAP_PATH: []const u8 = prepr_options.wrt_markmap;
pub const MARKMAP: bool = MARKMAP_PATH.len > 0;

// Class codes written to `<prefix>.cls`, one byte per post-WRT temp byte.
pub const CLS_FRAME: u8 = 0; // segment type + 4-byte length + TEXT mode byte
pub const CLS_RAWBLOCK: u8 = 1; // DEFAULT block / raw-fallback TEXT block payload
pub const CLS_LITERAL: u8 = 2; // ordinary unescaped literal inside a WRT block
pub const CLS_ESCAPE: u8 = 3; // the kEscape byte itself
pub const CLS_ESCAPED: u8 = 4; // the literal byte following a kEscape
pub const CLS_CAP: u8 = 5; // kCapitalized marker
pub const CLS_UPPER: u8 = 6; // kUppercase marker
pub const CLS_ENDUPPER: u8 = 7; // kEndUpper marker
pub const CLS_CODEWORD: u8 = 8; // whole-word codeword byte
pub const CLS_SUBST: u8 = 9; // affix-substituted word (codeword + raw remainder)
pub const CLS_RAWWORD: u8 = 10; // word emitted as letters (dictionary miss)
pub const CLS_N: usize = 11;
/// Any non-stock arm forces the TEXT mode byte to 7 so `encode_text`'s
/// `size > len-50` raw fallback cannot silently turn an arm into `nowrt`
/// (a nosubst stream is LARGER than its input, so the fallback would always
/// fire and the measurement would collapse to a different question).
pub const FORCE_MODE7: bool = NOSUBST or NOCASE;

// Markers (dictionary.cpp:10-13).
pub const kCapitalized: u8 = 0x40;
pub const kUppercase: u8 = 0x07;
pub const kEndUpper: u8 = 0x06;
pub const kEscape: u8 = 0x0C;

/// Simple forward byte reader with EOF (-1) semantics (mirrors `getc(FILE*)`).
pub const Reader = struct {
    data: []const u8,
    pos: usize = 0,

    pub fn getc(self: *Reader) i32 {
        if (self.pos >= self.data.len) return -1;
        const b = self.data[self.pos];
        self.pos += 1;
        return b;
    }
};

/// The final byte permutation applied to every dict-encoded byte on the way out
/// (preprocessor.cpp encode_text:136-139) and re-applied on the way in
/// (`NextChar`, dictionary.cpp:297-304). It is an involution (self-inverse):
///   0x7B..0x7E ({|}~) <-> 0x50..0x53 (PQRS)
///   0x3A..0x3F (;<=>?) <-> 0x4A..0x4F (JKLMNO)   (xor 0x70)
///   0x58 ('X') <-> 0x60 ('`')                      (xor 0x38)
/// Operates in i32 to match C++ int arithmetic exactly (incl. EOF=-1 passthrough).
pub fn permuteI32(c_in: i32) i32 {
    var c = c_in;
    if (c >= '{' and c < 127) {
        c += @as(i32, 'P') - @as(i32, '{'); // += -43
    } else if (c >= 'P' and c < 'T') {
        c -= @as(i32, 'P') - @as(i32, '{'); // += 43
    } else if ((c >= ':' and c <= '?') or (c >= 'J' and c <= 'O')) {
        c ^= 0x70;
    }
    if (c == 'X' or c == '`') {
        c ^= @as(i32, 'X') ^ @as(i32, '`'); // ^= 0x38
    }
    return c;
}

pub fn permuteByte(c: u8) u8 {
    return @truncate(@as(u32, @bitCast(permuteI32(c))));
}

/// `NextChar` = permuted `getc` (dictionary.cpp:297-304). Returns 0xFF at EOF, matching
/// the `(unsigned char)(-1)` fallthrough in the C++.
fn nextChar(r: *Reader) u8 {
    return @truncate(@as(u32, @bitCast(permuteI32(r.getc()))));
}

/// C++ `(unsigned char)((int)c - 'a' + 'A')` — uppercasing that wraps like the C++ cast.
inline fn toUpper(c: u8) u8 {
    return c -% 32; // 'A'-'a' = -32; u8 wrapping matches the C++ unsigned-char result.
}

/// Compute the variable-length codeword `bytes` for the n-th dictionary word
/// (dictionary.cpp:63-80). Three tiers; english.dic (44,515 words < kBoundary3=44880)
/// never reaches an uninitialized case. Exposed so callers/tests can assert it is the
/// exact inverse of fxcm's `decode_code_word` (the §3 consistency guarantee).
pub fn computeCodeword(line_count: u32) u32 {
    const kBoundary1: u32 = 80;
    const kBoundary2: u32 = kBoundary1 + 3840; // 3920
    const kBoundary3: u32 = kBoundary2 + 40960; // 44880
    if (line_count < kBoundary1) {
        return 0x80 + line_count;
    } else if (line_count < kBoundary2) {
        var bytes: u32 = 0xD0 + ((line_count - kBoundary1) / 80);
        bytes += (0x80 + ((line_count - kBoundary1) % 80)) << 8;
        return bytes;
    } else if (line_count < kBoundary3) {
        var bytes: u32 = 0xF0 + (((line_count - kBoundary2) / 80) / 32);
        bytes += (0xD0 + (((line_count - kBoundary2) / 80) % 32)) << 8;
        bytes += (0x80 + ((line_count - kBoundary2) % 80)) << 16;
        return bytes;
    }
    return 0; // unreachable for english.dic (matches C++ never-hit branch)
}

pub const Dictionary = struct {
    allocator: std.mem.Allocator,
    /// Owned copy of the dictionary bytes; word keys/values are stable slices into it.
    dict_copy: []u8,
    /// word -> codeword bytes (dictionary.cpp:81 `byte_map_`).
    byte_map: std.StringHashMap(u32),
    /// codeword bytes -> word (dictionary.cpp:82 `reverse_map_`).
    reverse_map: std.AutoHashMap(u32, []const u8),
    longest_word: usize = 0,
    // Decode state (dictionary.cpp:31 `decode_upper_`, `decode_capital_`, `output_buffer_`).
    decode_upper: bool = false,
    decode_capital: bool = false,
    decode_buf: std.ArrayList(u8) = .empty,
    decode_head: usize = 0,
    /// LAB (-Dwrt-markmap): one class byte per byte appended to the encode scratch.
    /// Mirrors `out` exactly; `encodeText` translates it into final-stream coords.
    cls: std.ArrayList(u8) = .empty,

    /// Load english.dic → codeword maps (dictionary.cpp ctor:57-87). Builds BOTH the
    /// encode (`byte_map_`) and decode (`reverse_map_`) maps so one loaded Dictionary
    /// serves both directions.
    pub fn load(allocator: std.mem.Allocator, dict_bytes: []const u8) !Dictionary {
        var self = Dictionary{
            .allocator = allocator,
            .dict_copy = try allocator.dupe(u8, dict_bytes),
            .byte_map = std.StringHashMap(u32).init(allocator),
            .reverse_map = std.AutoHashMap(u32, []const u8).init(allocator),
        };
        errdefer {
            self.byte_map.deinit();
            self.reverse_map.deinit();
            allocator.free(self.dict_copy);
        }
        var line_count: u32 = 0;
        var run_start: usize = 0;
        var run_len: usize = 0;
        for (self.dict_copy, 0..) |c, idx| {
            if (c >= 'a' and c <= 'z') {
                if (run_len == 0) run_start = idx;
                run_len += 1;
            } else if (run_len != 0) {
                const word = self.dict_copy[run_start .. run_start + run_len];
                if (word.len > self.longest_word) self.longest_word = word.len;
                const bytes = computeCodeword(line_count);
                try self.byte_map.put(word, bytes);
                try self.reverse_map.put(bytes, word);
                line_count += 1;
                run_len = 0;
            }
        }
        // A trailing word with no terminator is dropped (matches the C++ loop, which
        // only flushes `line` on a non-a-z byte). english.dic ends with '\n'.
        return self;
    }

    pub fn deinit(self: *Dictionary) void {
        self.byte_map.deinit();
        self.reverse_map.deinit();
        self.decode_buf.deinit(self.allocator);
        if (comptime MARKMAP) self.cls.deinit(self.allocator);
        self.allocator.free(self.dict_copy);
    }

    /// LAB (-Dwrt-markmap): extend the class mirror up to `n` scratch bytes with `c`.
    /// Comptime-dead (and therefore free) when the dump path is empty.
    inline fn clsPad(self: *Dictionary, n: usize, c: u8) void {
        if (comptime !MARKMAP) return;
        while (self.cls.items.len < n) self.cls.append(self.allocator, c) catch @panic("markmap OOM");
    }

    /// LAB: the class mirror for the block just encoded (scratch coords).
    pub fn clsItems(self: *const Dictionary) []const u8 {
        return self.cls.items;
    }

    pub fn size(self: *const Dictionary) usize {
        return self.byte_map.count();
    }

    // ---- encode side (dictionary.cpp:15-33, 89-141, 197-258) ----

    /// `EncodeByte` (dictionary.cpp:15-21): escape control/high bytes that would collide
    /// with the transform's own markers.
    fn encodeByte(self: *Dictionary, c: u8, out: *std.ArrayList(u8)) !void {
        if (c == kEndUpper or c == kEscape or c == kUppercase or c == kCapitalized or c >= 0x80) {
            try out.append(self.allocator, kEscape);
            self.clsPad(out.items.len, CLS_ESCAPE);
            try out.append(self.allocator, c);
            self.clsPad(out.items.len, CLS_ESCAPED);
            return;
        }
        try out.append(self.allocator, c);
        self.clsPad(out.items.len, CLS_LITERAL);
    }

    /// `EncodeBytes` (dictionary.cpp:23-33): little-endian, content-length-tagged.
    fn encodeBytes(self: *Dictionary, bytes: u32, out: *std.ArrayList(u8)) !void {
        try out.append(self.allocator, @intCast(bytes & 0xFF));
        if (bytes & 0xFF00 != 0) {
            try out.append(self.allocator, @intCast((bytes & 0xFF00) >> 8));
        } else return;
        if (bytes & 0xFF0000 != 0) {
            try out.append(self.allocator, @intCast((bytes & 0xFF0000) >> 16));
        }
    }

    /// `EncodeSubstring` (dictionary.cpp:229-258): affix substitution for words > 7 chars.
    fn encodeSubstring(self: *Dictionary, word: []const u8, out: *std.ArrayList(u8)) !bool {
        if (word.len <= 7) return false;
        var sz = word.len - 1;
        if (sz > self.longest_word) sz = self.longest_word;
        // Suffix pass: trailing `sz` chars, shrink from the front.
        var suffix = word[word.len - sz .. word.len];
        while (suffix.len >= 7) {
            if (self.byte_map.get(suffix)) |bytes| {
                try out.appendSlice(self.allocator, word[0 .. word.len - suffix.len]);
                try self.encodeBytes(bytes, out);
                return true;
            }
            suffix = suffix[1..];
        }
        // Prefix pass: leading `sz` chars, shrink from the back.
        var prefix = word[0..sz];
        while (prefix.len >= 7) {
            if (self.byte_map.get(prefix)) |bytes| {
                try self.encodeBytes(bytes, out);
                try out.appendSlice(self.allocator, word[prefix.len..word.len]);
                return true;
            }
            prefix = prefix[0 .. prefix.len - 1];
        }
        return false;
    }

    /// `EncodeWord` (dictionary.cpp:197-212): case prefix + body (codeword / substring / raw).
    fn encodeWord(self: *Dictionary, word: []const u8, num_upper: i32, next_lower: bool, out: *std.ArrayList(u8)) !void {
        if (comptime !NOCASE and !NOMARK) {
            if (num_upper > 1) {
                try out.append(self.allocator, kUppercase);
                self.clsPad(out.items.len, CLS_UPPER);
            } else if (num_upper == 1) {
                try out.append(self.allocator, kCapitalized);
                self.clsPad(out.items.len, CLS_CAP);
            }
        }
        if (comptime NOSUBST) {
            try out.appendSlice(self.allocator, word);
            self.clsPad(out.items.len, CLS_RAWWORD);
        } else if (self.byte_map.get(word)) |bytes| {
            try self.encodeBytes(bytes, out);
            self.clsPad(out.items.len, CLS_CODEWORD);
        } else if (!try self.encodeSubstring(word, out)) {
            try out.appendSlice(self.allocator, word);
            self.clsPad(out.items.len, CLS_RAWWORD);
        } else {
            self.clsPad(out.items.len, CLS_SUBST);
        }
        if (comptime !NOCASE and !NOMARK) {
            if (num_upper > 1 and next_lower) {
                try out.append(self.allocator, kEndUpper);
                self.clsPad(out.items.len, CLS_ENDUPPER);
            }
        }
    }

    /// `Dictionary::Encode` (dictionary.cpp:89-141): word tokenizer + case handling.
    /// Appends the transformed bytes (pre-permutation) to `out`.
    pub fn encode(self: *Dictionary, input: []const u8, out: *std.ArrayList(u8)) !void {
        var word: std.ArrayList(u8) = .empty;
        defer word.deinit(self.allocator);
        var num_upper: i32 = 0;
        var num_lower: i32 = 0;
        if (comptime MARKMAP) self.cls.clearRetainingCapacity();
        const len = input.len;
        if (len == 0) return;
        var pos: usize = 0;
        while (pos < len) : (pos += 1) {
            const c = input[pos];
            var advance = false;
            if (word.items.len > self.longest_word) {
                advance = true;
            } else if (c >= 'a' and c <= 'z') {
                if (num_upper > 1) {
                    advance = true;
                } else {
                    num_lower += 1;
                    try word.append(self.allocator, c);
                }
            } else if (c >= 'A' and c <= 'Z') {
                if (num_lower > 0) {
                    advance = true;
                } else {
                    num_upper += 1;
                    try word.append(self.allocator, if (comptime NOCASE) c else c - 'A' + 'a');
                }
            } else {
                advance = true;
            }
            if (pos == len - 1 and !advance) {
                try self.encodeWord(word.items, num_upper, false, out);
            }
            if (advance) {
                if (word.items.len == 0) {
                    try self.encodeByte(c, out);
                } else {
                    const next_lower = (c >= 'a' and c <= 'z');
                    try self.encodeWord(word.items, num_upper, next_lower, out);
                    num_lower = 0;
                    num_upper = 0;
                    word.clearRetainingCapacity();
                    if (next_lower) {
                        num_lower += 1;
                        try word.append(self.allocator, c);
                    } else if (c >= 'A' and c <= 'Z') {
                        num_upper += 1;
                        try word.append(self.allocator, if (comptime NOCASE) c else c - 'A' + 'a');
                    } else {
                        try self.encodeByte(c, out);
                    }
                    if (pos == len - 1 and word.items.len != 0) {
                        try self.encodeWord(word.items, num_upper, false, out);
                    }
                }
            }
        }
        // Catch-all: must be a no-op (every append site above classes itself).
        self.clsPad(out.items.len, CLS_LITERAL);
    }

    // ---- decode side (dictionary.cpp:288-348) ----

    /// Reset the streaming decode state (fresh `output_buffer_`, cleared upper/capital).
    pub fn resetDecodeState(self: *Dictionary) void {
        self.decode_upper = false;
        self.decode_capital = false;
        self.decode_buf.clearRetainingCapacity();
        self.decode_head = 0;
    }

    /// `AddToBuffer` (dictionary.cpp:306-348): read one transform-undone byte and expand.
    fn addToBuffer(self: *Dictionary, r: *Reader) !void {
        const c0 = nextChar(r);
        if (c0 == kEscape) {
            self.decode_upper = false;
            try self.decode_buf.append(self.allocator, nextChar(r));
        } else if (c0 == kUppercase) {
            self.decode_upper = true;
        } else if (c0 == kCapitalized) {
            self.decode_capital = true;
        } else if (c0 == kEndUpper) {
            self.decode_upper = false;
        } else if (c0 >= 0x80) {
            var bytes: u32 = c0;
            if (c0 > 0xCF) {
                const c1 = nextChar(r);
                bytes += @as(u32, c1) << 8;
                if (c1 > 0xCF) {
                    const c2 = nextChar(r);
                    bytes += @as(u32, c2) << 16;
                }
            }
            if (self.reverse_map.get(bytes)) |wordc| {
                for (wordc, 0..) |wc, i| {
                    var ch = wc;
                    if (i == 0 and self.decode_capital) {
                        ch = toUpper(ch);
                        self.decode_capital = false;
                    }
                    if (self.decode_upper) {
                        ch = toUpper(ch);
                    }
                    try self.decode_buf.append(self.allocator, ch);
                }
            }
            // reverse_map miss => C++ operator[] default-constructs "" => pushes nothing.
        } else {
            var ch = c0;
            if (!((ch >= 'a' and ch <= 'z') or (ch >= 'A' and ch <= 'Z'))) {
                self.decode_upper = false;
            }
            if (self.decode_capital or self.decode_upper) {
                ch = toUpper(ch);
            }
            if (self.decode_capital) self.decode_capital = false;
            try self.decode_buf.append(self.allocator, ch);
        }
    }

    /// `Dictionary::Decode` (dictionary.cpp:288-295): pull one output byte, refilling.
    pub fn decodeByte(self: *Dictionary, r: *Reader) !u8 {
        while (self.decode_head >= self.decode_buf.items.len) {
            self.decode_buf.clearRetainingCapacity();
            self.decode_head = 0;
            try self.addToBuffer(r);
        }
        const b = self.decode_buf.items[self.decode_head];
        self.decode_head += 1;
        return b;
    }
};
