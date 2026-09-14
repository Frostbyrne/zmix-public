//! Binary arithmetic coder, ported from cmix `coder/encoder.{h,cpp}` and
//! `coder/decoder.{h,cpp}`.
//!
//! A 32-bit range coder driven by the Predictor's float probability, discretised
//! to 16 bits. Encoder and decoder run the identical Predictor, so the stream is
//! a faithful bit-for-bit port. Output/input are in-memory (ArrayList / slice)
//! to stay independent of Zig std I/O API churn.
const std = @import("std");
const Predictor = @import("predictor.zig").Predictor;

fn discretize(p: f32) u32 {
    const d: f32 = 1.0 + 65534.0 * p;
    return @intFromFloat(d);
}

pub const Encoder = struct {
    out: *std.ArrayList(u8),
    gpa: std.mem.Allocator,
    p: *Predictor,
    x1: u32 = 0,
    x2: u32 = 0xffffffff,

    pub fn init(gpa: std.mem.Allocator, out: *std.ArrayList(u8), p: *Predictor) Encoder {
        return .{ .out = out, .gpa = gpa, .p = p };
    }

    pub fn encode(self: *Encoder, bit: i32) !void {
        const p: u32 = discretize(self.p.predict());
        const span = self.x2 - self.x1;
        const xmid = self.x1 + (span >> 16) * p + (((span & 0xffff) * p) >> 16);
        if (bit != 0) self.x2 = xmid else self.x1 = xmid + 1;
        self.p.perceive(bit);
        while (((self.x1 ^ self.x2) & 0xff000000) == 0) {
            try self.out.append(self.gpa, @intCast(self.x2 >> 24));
            self.x1 <<= 8;
            self.x2 = (self.x2 << 8) + 255;
        }
    }

    pub fn flush(self: *Encoder) !void {
        while (((self.x1 ^ self.x2) & 0xff000000) == 0) {
            try self.out.append(self.gpa, @intCast(self.x2 >> 24));
            self.x1 <<= 8;
            self.x2 = (self.x2 << 8) + 255;
        }
        try self.out.append(self.gpa, @intCast(self.x2 >> 24));
    }
};

pub const Decoder = struct {
    data: []const u8,
    pos: usize = 0,
    p: *Predictor,
    x1: u32 = 0,
    x2: u32 = 0xffffffff,
    x: u32 = 0,

    pub fn init(data: []const u8, p: *Predictor) Decoder {
        var d: Decoder = .{ .data = data, .p = p };
        for (0..4) |_| d.x = (d.x << 8) + (d.readByte() & 0xff);
        return d;
    }

    fn readByte(self: *Decoder) u32 {
        if (self.pos >= self.data.len) return 0;
        const b = self.data[self.pos];
        self.pos += 1;
        return b;
    }

    pub fn decode(self: *Decoder) i32 {
        const p: u32 = discretize(self.p.predict());
        const span = self.x2 - self.x1;
        const xmid = self.x1 + (span >> 16) * p + (((span & 0xffff) * p) >> 16);
        var bit: i32 = 0;
        if (self.x <= xmid) {
            bit = 1;
            self.x2 = xmid;
        } else {
            self.x1 = xmid + 1;
        }
        self.p.perceive(bit);
        while (((self.x1 ^ self.x2) & 0xff000000) == 0) {
            self.x1 <<= 8;
            self.x2 = (self.x2 << 8) + 255;
            self.x = (self.x << 8) + self.readByte();
        }
        return bit;
    }
};
