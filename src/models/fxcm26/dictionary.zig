//! fxcm dictionary / word decoder — port of the `decodeCodeWord`/`loaddict`/`dosym`/
//! `decodeWord` block in `src/cmix/src/models/fxcmv1.cpp` (~354-438), translated
//! 1:1 from `dictionary.rs`.
//!
//! `decode_code_word` is a pure-integer map from a 1-3 byte WRT codeword to a
//! dictionary index (bit-exact, golden-tested vs C++). The `Dictionary` struct wraps
//! the word list + the `codeword2sym` table and resolves a codeword to a word.

const std = @import("std");

const DICT1SIZE: i32 = 80;
const DICT2SIZE: i32 = 32;
const DICT12SIZE: i32 = DICT1SIZE * DICT2SIZE; // 2560

/// `decodeCodeWord(cw)` — map a packed codeword to a dictionary index.
pub fn decode_code_word(cw: i32, codeword2sym: *const [256]i32) i32 {
    var c: usize = @intCast(cw & 255);
    if (codeword2sym[c] < DICT1SIZE) {
        return codeword2sym[c];
    }
    var i: i32 = DICT1SIZE *% (codeword2sym[c] -% DICT1SIZE);
    c = @intCast((cw >> 8) & 255);
    if (codeword2sym[c] < DICT1SIZE) {
        return i +% codeword2sym[c] +% DICT1SIZE;
    }
    i = (i -% DICT12SIZE) *% DICT2SIZE;
    i +%= DICT1SIZE *% (codeword2sym[c] -% DICT1SIZE);
    c = @intCast((cw >> 16) & 255);
    i +%= codeword2sym[c];
    return i +% 80 * 49;
}

/// Build the standard `codeword2sym` table (`dosym`): 0 for bytes < 128, and
/// `c - 128` for bytes 128..255.
pub fn standard_codeword2sym() [256]i32 {
    var t = [_]i32{0} ** 256;
    var used: i32 = 0;
    var c: usize = 0;
    while (c < 256) : (c += 1) {
        if (c >= 128) {
            t[c] = used;
            used += 1;
        }
    }
    return t;
}

/// The fxcm word dictionary: the loaded word list + the codeword→symbol table.
pub const Dictionary = struct {
    words: [][]const u8,
    codeword2sym: [256]i32,
    last_cw: usize,

    /// `loaddict` + `dosym`: build from the dictionary lines (each a word, NUL-trimmed).
    pub fn from_words(words: [][]const u8) Dictionary {
        return Dictionary{
            .words = words,
            .codeword2sym = standard_codeword2sym(),
            .last_cw = 0,
        };
    }

    /// Load from a dictionary file (one word per line), mirroring `loaddict`/`wfgets`.
    /// std-only convenience; dead code in the shipped decode path. Caller owns the
    /// returned words (and each word slice), which are allocated from `allocator`.
    pub fn load_from_file(allocator: std.mem.Allocator, path: []const u8) !Dictionary {
        const data = try std.fs.cwd().readFileAlloc(allocator, path, std.math.maxInt(usize));
        defer allocator.free(data);
        var words = std.ArrayList([]const u8){};
        errdefer words.deinit(allocator);
        var it = std.mem.splitScalar(u8, data, '\n');
        while (it.next()) |line| {
            // wfgets stops at the first \n and NUL-terminates; trailing empty line is skipped.
            if (line.len != 0) {
                const owned = try allocator.dupe(u8, line);
                try words.append(allocator, owned);
            }
        }
        return Dictionary.from_words(try words.toOwnedSlice(allocator));
    }

    pub fn size(self: *const Dictionary) usize {
        return self.words.len;
    }

    /// `decodeWord(c)` — resolve a codeword to a word index. Returns `null` (and leaves
    /// `last_cw` unchanged) when the index is out of range, matching the C++ guard
    /// `if (j <= 0 || j >= sizeDict) return;`.
    pub fn decode_word(self: *Dictionary, c: i32) ?usize {
        const j = decode_code_word(c, &self.codeword2sym);
        if (j <= 0 or j >= @as(i32, @intCast(self.words.len))) {
            return null;
        }
        self.last_cw = @intCast(j);
        return @intCast(j);
    }

    /// The last successfully-decoded word's bytes.
    pub fn last_word(self: *const Dictionary) []const u8 {
        return self.words[self.last_cw];
    }
};

// ---- tests (ported from `dictionary.rs` `mod tests` + `dictionary_ref.rs`) ----

const DECODE_CS: u64 = 3196375604773819714;
const DECODE_N: i32 = 5008;
const DECODE_SAMPLES = [_][2]i32{
    .{ 187, 59 },      .{ 191, 63 },        .{ 249, 3360 },    .{ 239, 2560 },     .{ 11659487, -37631 },
    .{ 202, 74 },      .{ 8568265, 73 },    .{ 212, 400 },     .{ 55968, 32 },     .{ 135, 7 },
    .{ 11921835, 43 }, .{ 64643, 3 },       .{ 14408332, 12 }, .{ 13603297, 1457 }, .{ 9146776, 24 },
    .{ 11447492, 68 }, .{ 51895, 55 },      .{ 11379843, 3 },  .{ 36021, 53 },     .{ 14138300, 60 },
    .{ 12121508, 36 }, .{ 42666, 42 },      .{ 37795, 35 },    .{ 158, 30 },       .{ 45512, 72 },
    .{ 43984, 123 },   .{ 228, 1680 },      .{ 12903309, 13 }, .{ 51135, 63 },     .{ 57226, 10 },
    .{ 223, 1280 },    .{ 13212365, 77 },   .{ 173, 45 },      .{ 151, 23 },       .{ 225, 1440 },
    .{ 12637097, 41 }, .{ 8780179, 19 },    .{ 13734024, 8 },  .{ 14149344, -35193 }, .{ 15174597, 69 },
    .{ 12633055, 1347 }, .{ 65004, -2720 }, .{ 50424, 3348 },  .{ 137, 9 },        .{ 227, 1600 },
    .{ 14474916, 36 }, .{ 200, 72 },        .{ 36314, 893 },   .{ 9742212, 4 },    .{ 195, 67 },
    .{ 187, 59 },      .{ 204, 76 },        .{ 12180895, 31 }, .{ 155, 27 },       .{ 204, 76 },
    .{ 231, 1920 },    .{ 44010, 2203 },    .{ 35012, 68 },    .{ 214, 560 },      .{ 232, 2000 },
    .{ 9481165, 77 },  .{ 140, 12 },        .{ 41687, 674 },   .{ 211, 320 },      .{ 0, 0 },
    .{ 17, 0 },        .{ 34, 0 },          .{ 51, 0 },
};

const Xs = struct {
    s: u32,
    fn next(self: *Xs) u32 {
        self.s ^= self.s << 13;
        self.s ^= self.s >> 17;
        self.s ^= self.s << 5;
        return self.s;
    }
};

test "decode_code_word matches cpp" {
    const sym = standard_codeword2sym();
    var x = Xs{ .s = 0xD1C7 };
    var cs: u64 = 0;
    var n: i32 = 0;
    var samp_idx: usize = 0;
    var t: i32 = 0;
    while (t < 5000) : (t += 1) {
        const r = x.next();
        const b0: i32 = 128 + @as(i32, @intCast(r % 128));
        const b1: i32 = 128 + @as(i32, @intCast((r >> 8) % 128));
        const b2: i32 = 128 + @as(i32, @intCast((r >> 16) % 128));
        const nb: u32 = 1 + (r >> 24) % 3;
        var cw: i32 = b0;
        if (nb >= 2) {
            cw |= b1 << 8;
        }
        if (nb >= 3) {
            cw |= b2 << 16;
        }
        const idx = decode_code_word(cw, &sym);
        cs = cs *% 1000003 +% @as(u64, @bitCast(@as(i64, idx)));
        n += 1;
        if (t < 40 or @rem(t, 200) == 0) {
            try std.testing.expectEqual(cw, DECODE_SAMPLES[samp_idx][0]);
            try std.testing.expectEqual(idx, DECODE_SAMPLES[samp_idx][1]);
            samp_idx += 1;
        }
    }
    var b: i32 = 0;
    while (b < 128) : (b += 17) {
        const idx = decode_code_word(b, &sym);
        cs = cs *% 1000003 +% @as(u64, @bitCast(@as(i64, idx)));
        n += 1;
        if (b < 60) {
            try std.testing.expectEqual(b, DECODE_SAMPLES[samp_idx][0]);
            try std.testing.expectEqual(idx, DECODE_SAMPLES[samp_idx][1]);
            samp_idx += 1;
        }
    }
    try std.testing.expectEqual(DECODE_CS, cs);
    try std.testing.expectEqual(DECODE_N, n);
}

test "dictionary decode_word plumbing" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();

    // Synthetic dict: indices 0..200 -> a word each.
    const words = try a.alloc([]const u8, 200);
    for (words, 0..) |*w, i| {
        w.* = try std.fmt.allocPrint(a, "w{d}", .{i});
    }
    var d = Dictionary.from_words(words);
    // A single-byte codeword < 128 decodes to index 0 -> guarded out (j<=0).
    try std.testing.expectEqual(@as(?usize, null), d.decode_word(5));
    // A two-byte codeword in range resolves and sets last_word.
    const cw: i32 = 187; // from the oracle: decodes to 59
    try std.testing.expectEqual(@as(?usize, 59), d.decode_word(cw));
    try std.testing.expectEqualStrings("w59", d.last_word());
    // Out-of-range index returns null and leaves last_cw unchanged.
    try std.testing.expectEqual(@as(?usize, null), d.decode_word(255 | (255 << 8) | (255 << 16)));
    try std.testing.expectEqualStrings("w59", d.last_word());
}
