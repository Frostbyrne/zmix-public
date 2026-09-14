//! zmix — a Zig port of cmix (byronknoll/cmix) with the fxcm_v26 model.
//!
//! Usage:
//!   zmix -c input output   compress
//!   zmix -d input output   decompress
//!
//! The container is a small zmix header (distinct from cmix's while output is not
//! yet byte-compatible): "zC", version, then the 8-byte original length.
const std = @import("std");
const Predictor = @import("predictor.zig").Predictor;
const coder = @import("coder.zig");

const SIG0: u8 = 'z';
const SIG1: u8 = 'C';
const VERSION: u8 = 1;

/// Vocab = bytes present (cmix ExtractVocab); all-true for inputs < 10000 bytes.
fn computeVocab(data: []const u8) [256]bool {
    var vocab: [256]bool = .{false} ** 256;
    if (data.len < 10000) {
        vocab = .{true} ** 256;
    } else {
        for (data) |c| vocab[c] = true;
    }
    return vocab;
}

fn usage() void {
    std.debug.print(
        \\zmix — Zig port of cmix (with fxcm_v26)
        \\
        \\Compress:   zmix -c input output
        \\Decompress: zmix -d input output
        \\
    , .{});
}

pub fn main() !void {
    var gpa_state = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa_state.deinit();
    const gpa = gpa_state.allocator();

    const args = try std.process.argsAlloc(gpa);
    defer std.process.argsFree(gpa, args);

    if (args.len != 4 or args[1].len != 2 or args[1][0] != '-' or
        (args[1][1] != 'c' and args[1][1] != 'd'))
    {
        usage();
        return;
    }

    const in_data = try std.fs.cwd().readFileAlloc(gpa, args[2], std.math.maxInt(usize));
    defer gpa.free(in_data);

    var out_buf: std.ArrayList(u8) = .empty;
    defer out_buf.deinit(gpa);

    // Header: SIG0 SIG1 VERSION, 8-byte size, 32-byte vocab bitmask, then stream.
    const HEADER = 3 + 8 + 32;

    if (args[1][1] == 'c') {
        var vocab = computeVocab(in_data);
        const pred = try Predictor.init(gpa, &vocab);
        defer pred.deinit();

        try out_buf.append(gpa, SIG0);
        try out_buf.append(gpa, SIG1);
        try out_buf.append(gpa, VERSION);
        var i: i32 = 56;
        while (i >= 0) : (i -= 8) {
            try out_buf.append(gpa, @intCast((in_data.len >> @intCast(i)) & 255));
        }
        for (0..32) |bidx| {
            var c: u8 = 0;
            for (0..8) |j| {
                if (vocab[bidx * 8 + j]) c |= @as(u8, 1) << @intCast(j);
            }
            try out_buf.append(gpa, c);
        }
        var enc = coder.Encoder.init(gpa, &out_buf, pred);
        for (in_data) |c| {
            var b: i32 = 7;
            while (b >= 0) : (b -= 1) try enc.encode((@as(i32, c) >> @intCast(b)) & 1);
        }
        try enc.flush();
    } else {
        if (in_data.len < HEADER or in_data[0] != SIG0 or in_data[1] != SIG1 or in_data[2] != VERSION) {
            std.debug.print("Not a zmix file.\n", .{});
            return;
        }
        var size: u64 = 0;
        for (in_data[3..11]) |b| size = (size << 8) | b;
        var vocab: [256]bool = undefined;
        for (0..32) |bidx| {
            const c = in_data[11 + bidx];
            for (0..8) |j| vocab[bidx * 8 + j] = (c & (@as(u8, 1) << @intCast(j))) != 0;
        }
        const pred = try Predictor.init(gpa, &vocab);
        defer pred.deinit();
        var dec = coder.Decoder.init(in_data[HEADER..], pred);
        try out_buf.ensureTotalCapacity(gpa, @intCast(size));
        var remaining = size;
        while (remaining > 0) : (remaining -= 1) {
            var c: i32 = 0;
            for (0..8) |_| c += c + dec.decode();
            out_buf.appendAssumeCapacity(@intCast(c & 0xff));
        }
    }

    const out_file = try std.fs.cwd().createFile(args[3], .{});
    defer out_file.close();
    try out_file.writeAll(out_buf.items);

    std.debug.print("{s}: {d} -> {d} bytes\n", .{ args[1], in_data.len, out_buf.items.len });
}
