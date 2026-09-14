//! Match model, ported from cmix `models/match.{h,cpp}`.
//!
//! Predicts the next bit from the byte that followed the last occurrence of the
//! current context in the history buffer, with confidence growing as the match
//! length grows.
const std = @import("std");
const Model = @import("../model.zig").Model;
const vtableFor = @import("../model.zig").vtableFor;
const zero_alloc = @import("../zero_alloc.zig");

pub const Match = struct {
    history: []const u8,
    byte_context: *const u64,
    bit_context: *const u32,
    history_pos: u64 = 0,
    cur_match: u64 = 0,
    cur_byte: u8 = 0,
    bit_pos: u8 = 128,
    match_length: u8 = 0,
    longest_match: *u64,
    limit: i32,
    delta: f32,
    divisor: f32,
    map: []u32,
    predictions: [256]f32,
    counts: [256]i32 = .{0} ** 256,
    outputs: [1]f32 = .{0.5},
    alloc: std.mem.Allocator,

    pub fn create(a: std.mem.Allocator, history: []const u8, byte_context: *const u64, bit_context: *const u32, limit: i32, delta: f32, map_size: usize, longest_match: *u64) *Match {
        const self = a.create(Match) catch unreachable;
        self.* = .{
            .history = history,
            .byte_context = byte_context,
            .bit_context = bit_context,
            .longest_match = longest_match,
            .limit = limit,
            .delta = delta,
            .divisor = 1.0 / (@as(f32, @floatFromInt(limit)) + delta),
            // zero_alloc: zero like the C++ vector, committed lazily (10 maps x 8 MB).
            .map = zero_alloc.alloc(a, u32, map_size) catch unreachable,
            .predictions = undefined,
            .alloc = a,
        };
        for (0..256) |i| self.predictions[i] = 0.5 + (@as(f32, @floatFromInt(i)) + 0.5) / 512.0;
        return self;
    }

    pub fn destroy(self: *Match) void {
        zero_alloc.free(self.alloc, self.map);
        self.alloc.destroy(self);
    }

    pub fn predict(self: *Match) []const f32 {
        if (self.cur_byte & self.bit_pos != 0) {
            self.outputs[0] = self.predictions[self.match_length];
        } else {
            self.outputs[0] = 1 - self.predictions[self.match_length];
        }
        return self.outputs[0..1];
    }

    pub fn perceive(self: *Match, bit: i32) void {
        const cur_bit: i32 = @intFromBool((self.cur_byte & self.bit_pos) != 0);
        const match: i32 = @intFromBool(bit == cur_bit);
        self.bit_pos /= 2;

        var divisor = self.divisor;
        if (self.counts[self.match_length] < self.limit) {
            self.counts[self.match_length] += 1;
            divisor = 1.0 / (@as(f32, @floatFromInt(self.counts[self.match_length])) + self.delta);
        }
        self.predictions[self.match_length] += (@as(f32, @floatFromInt(match)) - self.predictions[self.match_length]) * divisor;

        if (match != 0) {
            if (self.match_length < 255) self.match_length += 1;
        } else {
            self.match_length = 0;
        }

        if (self.bit_context.* >= 128) {
            self.map[@intCast(self.byte_context.* % self.map.len)] = @intCast(self.history_pos);
            self.history_pos += 1;
        }
    }

    pub fn byteUpdate(self: *Match) void {
        if (self.match_length < 8) {
            self.cur_match = self.map[@intCast(self.byte_context.* % self.map.len)];
        } else {
            self.cur_match += 1;
        }
        self.cur_match %= self.history.len;
        self.cur_byte = self.history[@intCast(self.cur_match)];
        self.bit_pos = 128;
        const match_context: u64 = self.match_length / 32;
        self.longest_match.* = @max(self.longest_match.*, match_context);
    }

    pub fn numOutputs(self: *Match) usize {
        _ = self;
        return 1;
    }

    pub fn model(self: *Match) Model {
        return .{ .ptr = self, .vtable = vtableFor(Match) };
    }
};
