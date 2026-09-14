//! Decoded-text buffer (cwbuf) + charSwap — the foundation the stemmer feeds on.
//! Ported from cmix-lex `fxcmv1.cpp` setbuf/buffer1/charSwap (4241/4352/2289).
//! In no-dict, `setbuf(charSwap(c1))` fires for every printable c1 (10,9,32-127) and
//! the dict-word setbuf path degenerates — so the decoded buffer is purely byte-driven.
//!
//! Validated bit-exact vs v26-port-tools/ctxdump.cpp (no-dict) DECBUF_CS over all
//! 15007 bytes (folds cwpos + buffer1(1..16) per byte).

const std = @import("std");

const CBMASK: u32 = 0xfff;

/// fxcmv1.cpp charSwap.
pub fn char_swap(c_in: i32) i32 {
    var c: i32 = c_in;
    if (c >= 123 and c < 127) {
        c += 80 - 123; // 'P'-'{' = -43
    } else if (c >= 80 and c < 84) {
        c -= 80 - 123; // -= -43  => +43
    } else if ((c >= 58 and c <= 63) or (c >= 74 and c <= 79)) {
        c ^= 0x70;
    }
    if (c == 88 or c == 96) {
        c ^= 88 ^ 96;
    }
    return c;
}

pub const DecodedBuffer = struct {
    cwbuf: [0x1000]u8,
    cwpos: u32,

    pub fn new() DecodedBuffer {
        return DecodedBuffer{ .cwbuf = [_]u8{0} ** 0x1000, .cwpos = 0 };
    }

    pub fn setbuf(self: *DecodedBuffer, c: u8) void {
        self.cwbuf[@as(usize, @intCast(self.cwpos & CBMASK))] = c;
        self.cwpos = self.cwpos +% 1;
    }

    pub fn buffer1(self: *const DecodedBuffer, i: u32) u8 {
        return self.cwbuf[@as(usize, @intCast((self.cwpos -% i) & CBMASK))];
    }

    /// The parseByte gate: setbuf(charSwap(c1)) for printable c1.
    pub fn feed_raw_byte(self: *DecodedBuffer, c1: i32) void {
        if (c1 == 10 or c1 == 9 or (c1 > 31 and c1 < 128)) {
            self.setbuf(@as(u8, @truncate(@as(u32, @bitCast(char_swap(c1))))));
        }
    }
};

test "decoded_buffer_matches_ctxdump_nodict" {
    const stream: []const u8 = @embedFile("goldens/stream.bin");
    try std.testing.expectEqual(@as(usize, 15007), stream.len);
    var db = DecodedBuffer.new();
    var cs: u64 = 0;
    for (stream) |byte| {
        db.feed_raw_byte(@as(i32, @intCast(byte)));
        cs = cs *% 1000003 +% @as(u64, db.cwpos);
        var q: u32 = 1;
        while (q <= 16) : (q += 1) {
            cs = cs *% 1000003 +% @as(u64, db.buffer1(q));
        }
    }
    try std.testing.expectEqual(@as(u64, 0xfa26e1db5e41f79b), cs);
}
