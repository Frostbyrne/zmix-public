//! Decode-only entry point — the shippable Hutter decompressor.
//!
//! The prize archive is produced ONCE by `runner_lex -e`; the decompressor that
//! ships only ever DECODES. This root references only the decode path
//! (runDecompression / runDecompressionD), so Zig dead-code-eliminates the entire
//! encoder: the arithmetic encoder, and the phda9 / reorder / WRT / split ENCODE
//! transforms. Same PredictorLex + Pretrain + decode transforms as `-d`/`-D`.
const std = @import("std");
const rl = @import("runner_lex.zig");

fn readFile(gpa: std.mem.Allocator, path: []const u8) ![]u8 {
    return std.fs.cwd().readFileAlloc(gpa, path, 1 << 34);
}
fn writeFile(path: []const u8, data: []const u8) !void {
    const f = try std.fs.cwd().createFile(path, .{});
    defer f.close();
    try f.writeAll(data);
}
fn getSeed() i32 {
    const v = std.process.getEnvVarOwned(std.heap.page_allocator, "ZMIX_SEED") catch return 923;
    defer std.heap.page_allocator.free(v);
    return std.fmt.parseInt(i32, std.mem.trim(u8, v, " \n\r\t"), 10) catch 923;
}

pub fn main() !void {
    var gpa_impl = std.heap.GeneralPurposeAllocator(.{}){};
    const gpa = gpa_impl.allocator();
    const args = try std.process.argsAlloc(gpa);
    if (args.len < 3) {
        std.debug.print("usage: -D <dict> <order> <payload> <out>  |  -d [dict] <in> <out>\n", .{});
        return;
    }
    const m = args[1];
    const seed = getSeed();

    if (std.mem.eql(u8, m, "-D")) {
        if (args.len != 6) return;
        const dict = try readFile(gpa, args[2]);
        // args[3]=order: unused on a stock decode; under -Dr1v2 the r1 v2
        // restore derives the block permutation from it.
        const order: ?[]u8 = if (comptime @import("build_options").r1v2)
            try readFile(gpa, args[3])
        else
            null;
        const payload = try readFile(gpa, args[4]);
        const enwik9 = try rl.runDecompressionD(gpa, payload, dict, order, seed);
        try writeFile(args[5], enwik9);
        return;
    }
    if (m.len == 2 and m[1] == 'd') {
        const has_dict = args.len == 5;
        const dict: ?[]u8 = if (has_dict) try readFile(gpa, args[2]) else null;
        const in_path = if (has_dict) args[3] else args[2];
        const out_path = if (has_dict) args[4] else args[3];
        const input = try readFile(gpa, in_path);
        const out = try rl.runDecompression(gpa, input, dict, seed);
        try writeFile(out_path, out);
        return;
    }
    std.debug.print("decode-only: use -d or -D\n", .{});
}
