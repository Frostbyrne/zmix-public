//! ByteModel base, ported from cmix `models/byte-model.{h,cpp}`.
//!
//! Holds a 256-way byte probability distribution and converts it to a per-bit
//! prediction via a binary search over the byte value (top_/bot_). Used as the
//! base for the LSTM byte-mixer (and later PPMd/Bracket).
const std = @import("std");

pub const ByteModel = struct {
    probs: [256]f32 = .{1.0 / 256.0} ** 256,
    top: i32 = 255,
    mid: i32 = 0,
    bot: i32 = 0,
    ex: i32 = 0,
    output: [1]f32 = .{0.5},
    vocab: *const [256]bool,

    pub fn init(vocab: *const [256]bool) ByteModel {
        return .{ .vocab = vocab };
    }

    pub fn predict(self: *ByteModel) []const f32 {
        const mid = self.bot + @divTrunc(self.top - self.bot, 2);
        var num: f32 = 0;
        var i: i32 = mid + 1;
        while (i <= self.top) : (i += 1) num += self.probs[@intCast(i)];
        var denom: f32 = num;
        i = self.bot;
        while (i <= mid) : (i += 1) denom += self.probs[@intCast(i)];
        // NOTE: cmix's ByteModel::Predict also computes `ex` = argmax byte over
        // [bot,top] here, but `ex` is dead in the lex predictor (write-only —
        // nothing reads it), so the ~O(top-bot) argmax loop is pure per-bit dead
        // work. Dropped (bit-identical: output[0] below does not depend on ex).
        self.output[0] = if (denom == 0) 0.5 else num / denom;
        return self.output[0..1];
    }

    pub fn bytePredict(self: *ByteModel) *const [256]f32 {
        return &self.probs;
    }

    pub fn perceive(self: *ByteModel, bit: i32) void {
        self.mid = self.bot + @divTrunc(self.top - self.bot, 2);
        if (bit != 0) {
            self.bot = self.mid + 1;
        } else {
            self.top = self.mid;
        }
    }

    pub fn byteUpdate(self: *ByteModel) void {
        self.top = 255;
        self.bot = 0;
        for (0..256) |i| {
            if (!self.vocab[i]) self.probs[i] = 0;
        }
    }
};
