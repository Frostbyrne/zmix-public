//! Direct model, ported from cmix `models/direct.{h,cpp}`.
//!
//! A direct probability table indexed by (byte context, partial-byte). Each cell
//! adapts toward the observed bit with a count-limited learning rate.
const std = @import("std");
const Model = @import("../model.zig").Model;
const vtableFor = @import("../model.zig").vtableFor;
const zero_alloc = @import("../zero_alloc.zig");

pub const Direct = struct {
    byte_context: *const u64,
    bit_context: *const u32,
    limit: i32,
    delta: f32,
    divisor: f32,
    predictions: []f32, // size*256, init 0.5
    counts: []u8, // size*256, init 0
    outputs: [1]f32 = .{0.5},
    alloc: std.mem.Allocator,

    pub fn create(a: std.mem.Allocator, byte_context: *const u64, bit_context: *const u32, limit: i32, delta: f32, size: usize) *Direct {
        const self = a.create(Direct) catch unreachable;
        self.* = .{
            .byte_context = byte_context,
            .bit_context = bit_context,
            .limit = limit,
            .delta = delta,
            .divisor = 1.0 / (@as(f32, @floatFromInt(limit)) + delta),
            .predictions = a.alloc(f32, size * 256) catch unreachable,
            // counts is 0-init and (with a 257*256 bracket context that stays ~0)
            // almost entirely untouched -> lazy zero_alloc pages drop the ~16.8 MB
            // eager commit + memset. Bit-identical (0 == 0). predictions stays eager
            // (its 0.5 fill isn't a lazy-safe offset).
            .counts = zero_alloc.alloc(a, u8, size * 256) catch unreachable,
            .alloc = a,
        };
        for (self.predictions) |*p| p.* = 0.5;
        return self;
    }

    pub fn destroy(self: *Direct) void {
        self.alloc.free(self.predictions);
        zero_alloc.free(self.alloc, self.counts);
        self.alloc.destroy(self);
    }

    inline fn idx(self: *Direct) usize {
        return @as(usize, @intCast(self.byte_context.*)) * 256 + self.bit_context.*;
    }

    pub fn predict(self: *Direct) []const f32 {
        self.outputs[0] = self.predictions[self.idx()];
        return self.outputs[0..1];
    }

    pub fn perceive(self: *Direct, bit: i32) void {
        const i = self.idx();
        var divisor = self.divisor;
        if (self.counts[i] < self.limit) {
            self.counts[i] += 1;
            divisor = 1.0 / (@as(f32, @floatFromInt(self.counts[i])) + self.delta);
        }
        self.predictions[i] += (@as(f32, @floatFromInt(bit)) - self.predictions[i]) * divisor;
    }

    pub fn byteUpdate(self: *Direct) void {
        _ = self;
    }
    pub fn numOutputs(self: *Direct) usize {
        _ = self;
        return 1;
    }

    pub fn model(self: *Direct) Model {
        return .{ .ptr = self, .vtable = vtableFor(Direct) };
    }
};
