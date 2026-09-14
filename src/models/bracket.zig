//! Bracket byte model, ported from cmix `models/bracket.{h,cpp}`.
//!
//! A ByteModel that predicts the matching close bracket after an open bracket,
//! with per-(bracket,distance) statistics. Added as a normal (1-output) model.
const std = @import("std");
const ByteModel = @import("byte_model.zig").ByteModel;
const Model = @import("../model.zig").Model;
const vtableFor = @import("../model.zig").vtableFor;

const Pair = struct { first: u32, second: u32 };

pub const Bracket = struct {
    base: ByteModel,
    byte: *const u32,
    distance_limit: u32,
    stack_limit: u32,
    stats_limit: u32,
    active: std.ArrayList(u32) = .empty,
    distance: std.ArrayList(u32) = .empty,
    stats: [][]Pair, // [256][distance_limit]
    alloc: std.mem.Allocator,
    // false = zmix ASCII pairs (v21 path); true = cmix-lex WRT-space pairs
    // (models/bracket.cpp: {'(',')'},{'P','R'},{'[',']'},{'L','N'},{'\'','\''},{'"','"'}).
    lex: bool = false,

    fn isBracket(self: *const Bracket, c: u32) bool {
        if (self.lex) {
            // C++ brackets_ KEY set (open chars). ':9-10' bracket.cpp.
            return c == '(' or c == 'P' or c == '[' or c == 'L' or c == '\'' or c == '"';
        }
        return c == '(' or c == '{' or c == '[' or c == '<' or c == '\'' or c == '"';
    }
    fn closeOf(self: *const Bracket, c: u32) u32 {
        if (self.lex) {
            // C++ brackets_[open] -> close. ':9-10' bracket.cpp.
            return switch (c) {
                '(' => ')',
                'P' => 'R',
                '[' => ']',
                'L' => 'N',
                '\'' => '\'',
                '"' => '"',
                else => 0,
            };
        }
        return switch (c) {
            '(' => ')',
            '{' => '}',
            '[' => ']',
            '<' => '>',
            '\'' => '\'',
            '"' => '"',
            else => 0,
        };
    }

    pub fn create(a: std.mem.Allocator, bit_context: *const u32, distance_limit: u32, stack_limit: u32, stats_limit: u32, vocab: *const [256]bool) *Bracket {
        return createImpl(a, bit_context, distance_limit, stack_limit, stats_limit, vocab, false);
    }

    /// cmix-lex WRT-space variant. predictor_lex constructs:
    ///   Bracket.createLex(a, &mgr.bit_context, 200, 10, 100000, &vocab)
    /// mirroring predictor.cpp:59 `Bracket(bit_ctx, 200, 10, 100000, vocab)`.
    pub fn createLex(a: std.mem.Allocator, bit_context: *const u32, distance_limit: u32, stack_limit: u32, stats_limit: u32, vocab: *const [256]bool) *Bracket {
        return createImpl(a, bit_context, distance_limit, stack_limit, stats_limit, vocab, true);
    }

    fn createImpl(a: std.mem.Allocator, bit_context: *const u32, distance_limit: u32, stack_limit: u32, stats_limit: u32, vocab: *const [256]bool, lex: bool) *Bracket {
        const self = a.create(Bracket) catch unreachable;
        const stats = a.alloc([]Pair, 256) catch unreachable;
        for (stats) |*row| {
            row.* = a.alloc(Pair, distance_limit) catch unreachable;
            for (row.*) |*p| p.* = .{ .first = 1, .second = 256 };
        }
        self.* = .{
            .base = ByteModel.init(vocab),
            .byte = bit_context,
            .distance_limit = distance_limit,
            .stack_limit = stack_limit,
            .stats_limit = stats_limit,
            .stats = stats,
            .alloc = a,
            .lex = lex,
        };
        return self;
    }

    pub fn destroy(self: *Bracket) void {
        for (self.stats) |row| self.alloc.free(row);
        self.alloc.free(self.stats);
        self.alloc.destroy(self);
    }

    pub fn predict(self: *Bracket) []const f32 {
        return self.base.predict();
    }
    pub fn perceive(self: *Bracket, bit: i32) void {
        self.base.perceive(bit);
    }
    pub fn numOutputs(self: *Bracket) usize {
        _ = self;
        return 1;
    }

    fn fill(self: *Bracket, v: f32) void {
        for (&self.base.probs) |*p| p.* = v;
    }

    pub fn byteUpdate(self: *Bracket) void {
        const b = self.byte.*;
        self.fill(1.0 / 256.0);
        const active_len = self.active.items.len;
        const top = if (active_len > 0) self.active.items[active_len - 1] else 0;
        if (active_len == 0 or (self.isBracket(b) and !(active_len > 0 and top == b and self.closeOf(b) == b))) {
            if (self.isBracket(b)) {
                self.active.append(self.alloc, b) catch unreachable;
                self.distance.append(self.alloc, 0) catch unreachable;
                if (self.active.items.len > self.stack_limit) {
                    _ = self.active.orderedRemove(0);
                    _ = self.distance.orderedRemove(0);
                }
                const st = &self.stats[b][0];
                const p = @as(f32, @floatFromInt(st.first)) / @as(f32, @floatFromInt(st.second));
                self.fill((1 - p) / 255);
                self.base.probs[self.closeOf(b)] = p;
            }
        } else {
            const active = self.active.items[self.active.items.len - 1];
            const distance = self.distance.items[self.distance.items.len - 1];
            self.stats[active][distance].second += 1;
            if (self.closeOf(active) == b) self.stats[active][distance].first += 1;
            if (self.stats[active][distance].second > self.stats_limit) {
                self.stats[active][distance].first /= 2;
                self.stats[active][distance].second /= 2;
            }
            if (self.closeOf(active) == b or distance >= self.distance_limit - 1) {
                _ = self.active.pop();
                _ = self.distance.pop();
                if (self.active.items.len > 0) {
                    const a2 = self.active.items[self.active.items.len - 1];
                    const d2 = self.distance.items[self.distance.items.len - 1];
                    const st = self.stats[a2][d2];
                    const p = @as(f32, @floatFromInt(st.first)) / @as(f32, @floatFromInt(st.second));
                    self.fill((1 - p) / 255);
                    self.base.probs[self.closeOf(a2)] = p;
                }
            } else {
                self.distance.items[self.distance.items.len - 1] += 1;
                const d2 = distance + 1;
                const st = self.stats[active][d2];
                const p = @as(f32, @floatFromInt(st.first)) / @as(f32, @floatFromInt(st.second));
                self.fill((1 - p) / 255);
                self.base.probs[self.closeOf(active)] = p;
            }
        }
        self.base.byteUpdate();
    }

    pub fn model(self: *Bracket) Model {
        return .{ .ptr = self, .vtable = vtableFor(Bracket) };
    }
};

// --- tests ---
const testing = std.testing;

// Drive a byte through the model exactly as the predictor does: set bit_context
// to the completed byte, run byteUpdate (probs for NEXT byte), then reset.
fn driveByte(b: *Bracket, bit_context: *u32, byte: u8) void {
    bit_context.* = byte;
    b.byteUpdate();
}

test "bracket lex pair set: open char elevates its close char" {
    const a = testing.allocator;
    var vocab: [256]bool = .{true} ** 256;
    var bc: u32 = 1;
    const b = Bracket.createLex(a, &bc, 200, 10, 100000, &vocab);
    defer {
        for (b.stats) |row| a.free(row);
        a.free(b.stats);
        b.active.deinit(a);
        b.distance.deinit(a);
        a.destroy(b);
    }
    // Cold model has p == 1/256 == uniform fill; train each pair so the close
    // char's stat rises above uniform, then confirm the elevated slot is the
    // WRT-lex close char (NOT the ASCII one).
    const Pairs = [_][2]u8{ .{ '(', ')' }, .{ 'P', 'R' }, .{ 'L', 'N' } };
    for (Pairs) |pr| {
        b.active.clearRetainingCapacity();
        b.distance.clearRetainingCapacity();
        var n: usize = 0;
        while (n < 300) : (n += 1) {
            driveByte(b, &bc, pr[0]); // open -> push
            driveByte(b, &bc, pr[1]); // close -> match++, pop
        }
        driveByte(b, &bc, pr[0]); // open again: probs[close] = learned p
        try testing.expect(b.base.probs[pr[1]] > b.base.probs['a']);
    }
}

test "bracket lex: ASCII-only brackets are inert in lex mode" {
    const a = testing.allocator;
    var vocab: [256]bool = .{true} ** 256;
    var bc: u32 = 1;
    const b = Bracket.createLex(a, &bc, 200, 10, 100000, &vocab);
    defer {
        for (b.stats) |row| a.free(row);
        a.free(b.stats);
        b.active.deinit(a);
        b.distance.deinit(a);
        a.destroy(b);
    }
    // '{' and '<' are lex brackets in the ASCII variant but NOT keys in lex.
    driveByte(b, &bc, '{');
    try testing.expectEqual(@as(usize, 0), b.active.items.len);
    driveByte(b, &bc, '<');
    try testing.expectEqual(@as(usize, 0), b.active.items.len);
    // A lex key IS pushed.
    driveByte(b, &bc, '[');
    try testing.expectEqual(@as(usize, 1), b.active.items.len);
}
