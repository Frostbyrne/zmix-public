//! End-to-end size bench for PredictorLex: drive the real arithmetic coder over a
//! file and report the compressed size (plain-byte drive, no container header — the
//! raw predictor+coder cost, comparable to the oracle entropy / cmix -n).
const std = @import("std");
const PredictorLex = @import("predictor_lex.zig").PredictorLex;

fn discretize(p: f32) u32 {
    return @intFromFloat(1.0 + 65534.0 * p);
}

pub fn main() !void {
    var gpa_impl = std.heap.GeneralPurposeAllocator(.{}){};
    const gpa = gpa_impl.allocator();
    const args = try std.process.argsAlloc(gpa);
    if (args.len < 2) {
        std.debug.print("usage: bench_lex <file>\n", .{});
        return;
    }
    const data = try std.fs.cwd().readFileAlloc(gpa, args[1], 1 << 30);
    var vocab: [256]bool = .{false} ** 256;
    for (data) |b| vocab[b] = true;

    const pred = try PredictorLex.init(gpa, vocab);
    var x1: u32 = 0;
    var x2: u32 = 0xffffffff;
    var out: usize = 0;
    for (data) |c| {
        var b: i32 = 7;
        while (b >= 0) : (b -= 1) {
            const bit = (@as(i32, c) >> @intCast(b)) & 1;
            const p: u32 = discretize(pred.predict());
            const span = x2 - x1;
            const xmid = x1 + (span >> 16) * p + (((span & 0xffff) * p) >> 16);
            if (bit != 0) x2 = xmid else x1 = xmid + 1;
            pred.perceive(bit);
            while (((x1 ^ x2) & 0xff000000) == 0) {
                out += 1;
                x1 <<= 8;
                x2 = (x2 << 8) + 255;
            }
        }
    }
    out += 4; // flush tail
    const bpc = @as(f64, @floatFromInt(out)) * 8.0 / @as(f64, @floatFromInt(data.len));
    std.debug.print("PredictorLex: {d} -> {d} bytes = {d:.4} bpc\n", .{ data.len, out, bpc });
}
