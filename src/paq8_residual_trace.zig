//! LAB-ONLY level-4 PAQ8 scalar trace exporter for the frozen residual gate.
//!
//! This is deliberately a separate executable root. It does not participate in
//! runner_lex, ship, or any production build. The exported scalar is the PAQ8
//! predictor's branch-independent `last_prediction`, sampled immediately before
//! the corresponding target bit is perceived.

const std = @import("std");
const PAQ8 = @import("models/paq8/paq8.zig").PAQ8;
const paq_core = @import("models/paq8/core.zig");

const level: i32 = 4;
const stride: u64 = 17;
const header_bytes: usize = 64;
const record_bytes: usize = 3; // u16 LE raw PAQ probability ++ u8 target bit
const flags: u32 = 1 | 2 | 4; // dict-pretrained | MSB-first | forced-DEFAULT PAQ port

fn usage() void {
    std.debug.print(
        \\usage: paq8_residual_trace DICT POST_WRT_TEMP TRACE_OUT [MAX_TEMP_BYTES]
        \\
        \\The optional byte limit exists only for prefix verification. Full scoring
        \\requires the complete 12,427,640-byte frozen post-WRT stream.
        \\
    , .{});
}

inline fn perceiveByte(paq: *PAQ8, byte: u8) void {
    var bit_index: i32 = 7;
    while (bit_index >= 0) : (bit_index -= 1) {
        const bit = (@as(i32, byte) >> @intCast(bit_index)) & 1;
        paq.perceive(bit);
    }
}

fn pretrainDictionary(paq: *PAQ8, dict: []const u8) void {
    const n: u32 = @intCast(dict.len);
    const header = [5]u8{
        0, // preprocessor::DEFAULT
        @intCast((n >> 24) & 0xff),
        @intCast((n >> 16) & 0xff),
        @intCast((n >> 8) & 0xff),
        @intCast(n & 0xff),
    };
    for (header) |byte| perceiveByte(paq, byte);
    for (dict) |raw| perceiveByte(paq, if (raw == '\n') ' ' else raw);
}

fn makeHeader(temp_bytes: u64, records: u64, dict_bytes: u64) [header_bytes]u8 {
    var header = [_]u8{0} ** header_bytes;
    @memcpy(header[0..8], "ZPAQRES\x00");
    std.mem.writeInt(u32, header[8..12], 1, .little);
    std.mem.writeInt(u32, header[12..16], @intCast(level), .little);
    std.mem.writeInt(u32, header[16..20], @intCast(stride), .little);
    std.mem.writeInt(u32, header[20..24], flags, .little);
    std.mem.writeInt(u64, header[24..32], temp_bytes, .little);
    std.mem.writeInt(u64, header[32..40], temp_bytes * 8, .little);
    std.mem.writeInt(u64, header[40..48], records, .little);
    std.mem.writeInt(u64, header[48..56], dict_bytes, .little);
    std.mem.writeInt(u32, header[56..60], record_bytes, .little);
    std.mem.writeInt(u32, header[60..64], 0, .little);
    return header;
}

pub fn main() !void {
    var gpa_state = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa_state.deinit();
    const gpa = gpa_state.allocator();

    const args = try std.process.argsAlloc(gpa);
    defer std.process.argsFree(gpa, args);
    if (args.len != 4 and args.len != 5) {
        usage();
        return error.InvalidArguments;
    }

    const dict = try std.fs.cwd().readFileAlloc(gpa, args[1], std.math.maxInt(usize));
    defer gpa.free(dict);

    const temp_file = try std.fs.cwd().openFile(args[2], .{});
    defer temp_file.close();
    const stat = try temp_file.stat();
    var temp_bytes: u64 = stat.size;
    if (args.len == 5) {
        const requested = try std.fmt.parseUnsigned(u64, args[4], 10);
        temp_bytes = @min(temp_bytes, requested);
    }
    if (temp_bytes > std.math.maxInt(u64) / 8) return error.InputTooLarge;
    const temp_bits = temp_bytes * 8;
    const records = if (temp_bits == 0) 0 else (temp_bits + stride - 1) / stride;
    if (records > (std.math.maxInt(usize) - header_bytes) / record_bytes)
        return error.InputTooLarge;

    var trace: std.ArrayList(u8) = .empty;
    defer trace.deinit(gpa);
    try trace.ensureTotalCapacity(gpa, header_bytes + @as(usize, @intCast(records)) * record_bytes);
    const header = makeHeader(temp_bytes, records, dict.len);
    try trace.appendSlice(gpa, &header);

    var arena = std.heap.ArenaAllocator.init(gpa);
    defer arena.deinit();
    const paq = PAQ8.create(arena.allocator(), level);
    pretrainDictionary(paq, dict);

    var timer = try std.time.Timer.start();
    var bit_position: u64 = 0;
    var emitted: u64 = 0;
    var raw_min: i32 = 4096;
    var raw_max: i32 = -1;
    var remaining = temp_bytes;
    var buffer: [1 << 20]u8 = undefined;
    while (remaining > 0) {
        const want: usize = @intCast(@min(remaining, buffer.len));
        const n = try temp_file.read(buffer[0..want]);
        if (n == 0) return error.UnexpectedEndOfFile;
        remaining -= n;
        for (buffer[0..n]) |byte| {
            var bit_index: i32 = 7;
            while (bit_index >= 0) : (bit_index -= 1) {
                const bit: u8 = @intCast((@as(i32, byte) >> @intCast(bit_index)) & 1);
                const raw = paq_core.last_prediction;
                if (raw < 0 or raw > 4095) return error.InvalidPaqProbability;
                if (bit_position % stride == 0) {
                    raw_min = @min(raw_min, raw);
                    raw_max = @max(raw_max, raw);
                    try trace.append(gpa, @intCast(raw & 0xff));
                    try trace.append(gpa, @intCast((raw >> 8) & 0xff));
                    try trace.append(gpa, bit);
                    emitted += 1;
                }
                paq.perceive(bit);
                bit_position += 1;
            }
        }
    }

    if (bit_position != temp_bits or emitted != records) return error.TraceGeometryMismatch;
    if (records > 1 and raw_min == raw_max) return error.ConstantPaqTrace;
    if (trace.items.len != header_bytes + @as(usize, @intCast(records)) * record_bytes)
        return error.TraceGeometryMismatch;

    const out = try std.fs.cwd().createFile(args[3], .{});
    defer out.close();
    try out.writeAll(trace.items);

    const elapsed_ms = timer.read() / std.time.ns_per_ms;
    std.debug.print(
        "paq8-residual-trace: level={d} dict={d} temp={d} bits={d} stride={d} records={d} raw=[{d},{d}] bytes={d} ms={d}\n",
        .{ level, dict.len, temp_bytes, bit_position, stride, emitted, raw_min, raw_max, trace.items.len, elapsed_ms },
    );
}
