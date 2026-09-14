//! parseByte byte-core — the truly-uncoupled foundation of the updatepreamble:
//! the byte history (c1/c2/c3), the x4 packed-byte register, and the order-hash
//! array t[1..13] (`t[i]=t[i-1]*primes[i]+c1+i*256`, with the end-marker pre-update).
//!
//! These are computed unconditionally per byte, independent of the word-parsing
//! core, so they validate standalone.
//!
//! Validated bit-exact vs v26-port-tools/ctxdump.cpp (no-dict) over the real
//! 15007-byte STREAM: x4 and t[1..13] match for ALL 15007 bytes.

const std = @import("std");

const PRIMES = [14]u32{ 0, 257, 251, 241, 239, 233, 229, 227, 223, 211, 199, 197, 193, 191 };

/// Post-WRT end-marker bytes: '$'(36) ')'(41) VERTICALBAR('Q'=81) '['(91) ']'(93).
inline fn is_end_marker(c: u32) bool {
    return switch (c) {
        36, 41, 81, 91, 93 => true,
        else => false,
    };
}

/// Per-bit shared block state (subset of fxcmv1.cpp `BlockData x`).
pub const X = struct {
    y: i32 = 0,
    c0: i32 = 0,
    c4: u32 = 0,
    bpos: i32 = 0,
    bposshift: i32 = 0,
    c0shift_bpos: i32 = 0,
    cm_bit_state: i32 = 0,
};

pub const ByteCore = struct {
    c1: u32,
    c2: u32,
    c3: u32,
    x4: u32,
    t: [16]u32,

    pub fn new() ByteCore {
        return ByteCore{ .c1 = 0, .c2 = 0, .c3 = 0, .x4 = 0, .t = [_]u32{0} ** 16 };
    }

    /// Advance by one completed byte (mirrors the uncoupled parts of parseByte).
    pub fn parse_byte(self: *ByteCore, byte: u8) void {
        self.c3 = self.c2;
        self.c2 = self.c1;
        self.c1 = @as(u32, byte);
        if (is_end_marker(self.c1)) {
            if (self.c1 != self.c2) {
                // order-X end-marker pre-update
                var i: usize = 13;
                while (i > 0) : (i -= 1) {
                    self.t[i] = self.t[i - 1] *% PRIMES[i];
                }
            }
            self.x4 = (self.x4 << 8) +% self.c2;
        }
        self.x4 = (self.x4 << 8) +% self.c1;
        var i: usize = 13;
        while (i > 0) : (i -= 1) {
            self.t[i] = self.t[i - 1] *% PRIMES[i] +% self.c1 +% (@as(u32, @intCast(i)) *% 256);
        }
    }
};

test "byte_core_matches_ctxdump_nodict" {
    const stream: []const u8 = @embedFile("goldens/stream.bin");
    try std.testing.expectEqual(@as(usize, 15007), stream.len);
    var bc = ByteCore.new();
    var cs: u64 = 0;
    for (stream) |byte| {
        bc.parse_byte(byte);
        cs = cs *% 1000003 +% @as(u64, bc.x4);
        var k: usize = 1;
        while (k <= 13) : (k += 1) {
            cs = cs *% 1000003 +% @as(u64, bc.t[k]);
        }
    }
    try std.testing.expectEqual(@as(u64, 0xbf486c2c0880d7fa), cs);
}
