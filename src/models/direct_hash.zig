//! DirectHash model, ported from cmix `models/direct-hash.{h,cpp}`.
//!
//! Like Direct, but the byte context is hashed into a fixed number of slots with
//! a 20-way linear-probe checksum check, so it can key on high-order contexts.
const std = @import("std");
const Model = @import("../model.zig").Model;
const vtableFor = @import("../model.zig").vtableFor;

pub const DirectHash = struct {
    byte_context: *const u64,
    bit_context: *const u32,
    index: u64 = 0,
    limit: i32,
    delta: f32,
    divisor: f32,
    size: usize,
    predictions: []f32, // size*256
    counts: []u8, // size*256
    checksums: []u64,
    outputs: [1]f32 = .{0.5},
    alloc: std.mem.Allocator,

    pub fn create(a: std.mem.Allocator, byte_context: *const u64, bit_context: *const u32, limit: i32, delta: f32, size: usize) *DirectHash {
        const self = a.create(DirectHash) catch unreachable;
        self.* = .{
            .byte_context = byte_context,
            .bit_context = bit_context,
            .limit = limit,
            .delta = delta,
            .divisor = 1.0 / (@as(f32, @floatFromInt(limit)) + delta),
            .size = size,
            .predictions = a.alloc(f32, size * 256) catch unreachable,
            .counts = a.alloc(u8, size * 256) catch unreachable,
            .checksums = a.alloc(u64, size) catch unreachable,
            .alloc = a,
        };
        for (self.predictions) |*p| p.* = 0.5;
        @memset(self.counts, 0);
        @memset(self.checksums, 0);
        return self;
    }

    pub fn destroy(self: *DirectHash) void {
        self.alloc.free(self.predictions);
        self.alloc.free(self.counts);
        self.alloc.free(self.checksums);
        self.alloc.destroy(self);
    }

    inline fn idx(self: *DirectHash) usize {
        return @as(usize, @intCast(self.index)) * 256 + self.bit_context.*;
    }

    pub fn predict(self: *DirectHash) []const f32 {
        self.outputs[0] = self.predictions[self.idx()];
        return self.outputs[0..1];
    }

    pub fn perceive(self: *DirectHash, bit: i32) void {
        const i = self.idx();
        var divisor = self.divisor;
        if (self.counts[i] < self.limit) {
            self.counts[i] += 1;
            divisor = 1.0 / (@as(f32, @floatFromInt(self.counts[i])) + self.delta);
        }
        self.predictions[i] += (@as(f32, @floatFromInt(bit)) - self.predictions[i]) * divisor;
    }

    pub fn byteUpdate(self: *DirectHash) void {
        self.index = self.byte_context.* % self.size;
        var i: usize = 0;
        while (i < 20) : (i += 1) {
            if (self.checksums[@intCast(self.index)] == 0) {
                self.checksums[@intCast(self.index)] = self.byte_context.*;
                break;
            }
            if (self.checksums[@intCast(self.index)] == self.byte_context.*) break;
            if (i == 19) {
                const base: usize = @as(usize, @intCast(self.index)) * 256;
                for (self.predictions[base .. base + 256]) |*p| p.* = 0.5;
                @memset(self.counts[base .. base + 256], 0);
                self.checksums[@intCast(self.index)] = self.byte_context.*;
                break;
            }
            self.index += 1;
            if (self.index == self.size) self.index = 0;
        }
    }

    pub fn numOutputs(self: *DirectHash) usize {
        _ = self;
        return 1;
    }

    pub fn model(self: *DirectHash) Model {
        return .{ .ptr = self, .vtable = vtableFor(DirectHash) };
    }
};
