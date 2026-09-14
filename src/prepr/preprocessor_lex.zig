//! Port of cmix-lex's `preprocessor` namespace
//! (`ref/cmix-lex/src/preprocess/preprocessor.{h,cpp}`).
//!
//! Segment framing + TEXT/DEFAULT detection wrapping the WRT `Dictionary` transform.
//! This produces exactly the byte stream that `cmix-lex -s` emits AFTER its 5-byte
//! storage header (WriteStorageHeader, runner.cpp:79-85), and `-c` feeds to the coder.
//!
//! Public API (for runner_lex.zig):
//!   Dictionary.load(alloc, dict_bytes) -> Dictionary        (see dictionary_lex.zig)
//!   encode(alloc, in, ?*Dictionary)    -> []u8              (preprocessor::Encode)
//!   decode(alloc, in, ?*Dictionary)    -> []u8              (preprocessor::Decode)
//!   noPreprocess(alloc, in)            -> []u8              (preprocessor::NoPreprocess, the -n body)
//!
//! Ground-truth C++ line refs are in comments (preprocessor.cpp).

const std = @import("std");
const dict_mod = @import("dictionary_lex.zig");

pub const Dictionary = dict_mod.Dictionary;
pub const Reader = dict_mod.Reader;
pub const permuteByte = dict_mod.permuteByte;

// Filetype (preprocessor.h:11 `typedef enum {DEFAULT, TEXT=7}`).
pub const DEFAULT: u8 = 0;
pub const TEXT: u8 = 7;

// kMaxSegment (preprocessor.cpp:218).
pub const kMaxSegment: u64 = 0x80000000 - 1;

// ---------------------------------------------------------------------------
// LAB (-Dwrt-markmap): one class byte per byte of the emitted temp stream.
// Comptime-dead when the dump path is empty, so the stock build is unchanged.
pub const MARKMAP = dict_mod.MARKMAP;
var g_cls: std.ArrayList(u8) = .empty;

inline fn clsPadG(alloc: std.mem.Allocator, n: usize, c: u8) void {
    if (comptime !MARKMAP) return;
    while (g_cls.items.len < n) g_cls.append(alloc, c) catch @panic("markmap OOM");
}

/// LAB: the class stream for the most recent `encode` call. Same length as its output.
pub fn clsStream() []const u8 {
    return g_cls.items;
}

/// `IsAscii` (preprocessor.cpp:12-16).
inline fn isAscii(c: u8) bool {
    if (c >= 9 and c <= 13) return true;
    if (c >= 32 and c <= 126) return true;
    return false;
}

const DetectResult = struct {
    /// -1 = EOF (Filetype(-1)); else DEFAULT or TEXT.
    typ: i32,
    /// New read cursor (mirrors `ftell(in)` after the C++ `detect`).
    pos: usize,
};

/// `detect` (preprocessor.cpp:52-97). Scans up to `n` bytes from `start_pos`; may return
/// early with a repositioned cursor on a DEFAULT<->TEXT transition.
fn detect(in: []const u8, start_pos: usize, n: usize, typ: u8) DetectResult {
    var ascii_start: i64 = -1;
    var ascii_run: i32 = 0;
    var space_count: i32 = 0;
    var pos = start_pos;
    var i: usize = 0;
    while (i < n) : (i += 1) {
        if (pos >= in.len) return .{ .typ = -1, .pos = pos }; // c == EOF
        const c = in[pos];
        pos += 1;
        if (typ == DEFAULT) {
            if (isAscii(c)) {
                if (ascii_start == -1) {
                    ascii_start = @intCast(i);
                    ascii_run = 0;
                    space_count = 0;
                }
                if (c == ' ') space_count += 1;
                ascii_run += 1;
                if (ascii_run > 500) {
                    if (space_count < 5) {
                        ascii_start = -1;
                    } else {
                        return .{ .typ = TEXT, .pos = start_pos + @as(usize, @intCast(ascii_start)) };
                    }
                }
            } else {
                ascii_start = -1;
            }
        } else if (typ == TEXT) {
            if (isAscii(c)) {
                ascii_run -= 2;
                if (ascii_run < 0) ascii_run = 0;
            } else {
                ascii_run += 3;
                if (ascii_run > 300) {
                    return .{ .typ = DEFAULT, .pos = pos -| 100 };
                }
            }
        }
    }
    return .{ .typ = @as(i32, typ), .pos = pos };
}

fn appendMarker(alloc: std.mem.Allocator, out: *std.ArrayList(u8), typ: u8, len: usize) !void {
    try out.append(alloc, typ);
    try out.append(alloc, @intCast((len >> 24) & 0xFF));
    try out.append(alloc, @intCast((len >> 16) & 0xFF));
    try out.append(alloc, @intCast((len >> 8) & 0xFF));
    try out.append(alloc, @intCast(len & 0xFF));
}

/// `encode_text` (preprocessor.cpp:107-146): the two-pass dict encode (in-memory here),
/// with the `size > len-50` raw fallback and the final byte permutation.
fn encodeText(alloc: std.mem.Allocator, in: []const u8, begin: usize, len: usize, dict: ?*Dictionary, out: *std.ArrayList(u8)) !void {
    // LAB `-Dwrt-earn=nowrt`: skip the word transform entirely — emit the block
    // exactly as the no-dictionary path does (mode byte 0 + raw bytes). The
    // dictionary object is still loaded and Pretrain still runs, so this is the
    // TRANSFORM pessimum, not `-c` without a dict.
    if (comptime dict_mod.NOWRT) {
        try out.append(alloc, 0);
        clsPadG(alloc, out.items.len, dict_mod.CLS_FRAME);
        try out.appendSlice(alloc, in[begin .. begin + len]);
        clsPadG(alloc, out.items.len, dict_mod.CLS_RAWBLOCK);
        return;
    }
    const d = dict orelse {
        try out.append(alloc, 0);
        clsPadG(alloc, out.items.len, dict_mod.CLS_FRAME);
        try out.appendSlice(alloc, in[begin .. begin + len]);
        clsPadG(alloc, out.items.len, dict_mod.CLS_RAWBLOCK);
        return;
    };
    var scratch: std.ArrayList(u8) = .empty;
    defer scratch.deinit(alloc);
    try d.encode(in[begin .. begin + len], &scratch);
    const size = scratch.items.len;
    // LAB: a nosubst/nocase stream can be LARGER than its input, which would
    // trip the raw fallback on every block and silently turn the arm into
    // `nowrt`. Forcing mode 7 keeps the framing identical to stock; the mode
    // byte is read back out of the stream by `reset_text_decoder`, so the
    // stream stays self-describing and STOCK-decodable.
    if (comptime dict_mod.FORCE_MODE7) {
        try out.append(alloc, 7);
        clsPadG(alloc, out.items.len, dict_mod.CLS_FRAME);
        for (scratch.items) |c| try out.append(alloc, permuteByte(c));
        if (comptime MARKMAP) try g_cls.appendSlice(alloc, d.clsItems());
        return;
    }
    if (@as(i64, @intCast(size)) > @as(i64, @intCast(len)) - 50) {
        try out.append(alloc, 0);
        clsPadG(alloc, out.items.len, dict_mod.CLS_FRAME);
        try out.appendSlice(alloc, in[begin .. begin + len]);
        clsPadG(alloc, out.items.len, dict_mod.CLS_RAWBLOCK);
    } else {
        try out.append(alloc, 7); // mode byte = dict used
        clsPadG(alloc, out.items.len, dict_mod.CLS_FRAME);
        for (scratch.items) |c| try out.append(alloc, permuteByte(c));
        if (comptime MARKMAP) try g_cls.appendSlice(alloc, d.clsItems());
    }
}

/// `EncodeSegment` (preprocessor.cpp:167-216): stats pass, the 95%-text shortcut, then the
/// multi-block DEFAULT loop (note: cmix-lex's multi-block switch always encode_default's).
fn encodeSegment(alloc: std.mem.Allocator, in: []const u8, begin: usize, n: usize, dict: ?*Dictionary, out: *std.ArrayList(u8)) !void {
    // Stats pass.
    var text_bytes: f64 = 0;
    {
        var typ: u8 = DEFAULT;
        var b = begin;
        var remainder = n;
        while (remainder > 0) {
            const r = detect(in, b, remainder, typ);
            if (r.typ < 0) break; // EOF (does not occur for a well-formed segment)
            const len = r.pos - b;
            if (typ == TEXT) text_bytes += @floatFromInt(len);
            remainder -= len;
            typ = @intCast(r.typ);
            b = r.pos;
            // NOTE: no len==0 break — a zero-length DEFAULT->TEXT transition (text at the
            // window start, ascii_start==0) flips `typ` to TEXT; the next detect(TEXT) then
            // consumes the run. detect(TEXT) always makes forward progress (>=1), so this
            // terminates. (preprocessor.cpp:174-186)
        }
    }

    // 95%-text shortcut: one TEXT block over the whole segment.
    if (text_bytes / @as(f64, @floatFromInt(n)) > 0.95) {
        try appendMarker(alloc, out, TEXT, n);
        clsPadG(alloc, out.items.len, dict_mod.CLS_FRAME);
        try encodeText(alloc, in, begin, n, dict, out);
        return;
    }

    // Multi-block path.
    var typ: u8 = DEFAULT;
    var b = begin;
    var remaining = n;
    while (remaining > 0) {
        const r = detect(in, b, remaining, typ);
        const end = if (r.typ < 0) b else r.pos;
        const len = end - b;
        if (len > 0) {
            try appendMarker(alloc, out, typ, len);
            clsPadG(alloc, out.items.len, dict_mod.CLS_FRAME);
            // switch(type){ default: encode_default } — always a raw copy in cmix-lex.
            try out.appendSlice(alloc, in[b..end]);
            clsPadG(alloc, out.items.len, dict_mod.CLS_RAWBLOCK);
        }
        remaining -= len;
        if (r.typ < 0) break;
        typ = @intCast(r.typ);
        b = end;
        // NOTE: no len==0 break (see stats-pass note) — the zero-length transition flips
        // `typ`, then detect makes progress. (preprocessor.cpp:199-215)
    }
}

/// `preprocessor::Encode` (preprocessor.cpp:220-232). Returns the segment-framed WRT/dict
/// stream (WITHOUT the 5-byte storage header — that belongs to the runner/`-s`/`-c` path).
pub fn encode(alloc: std.mem.Allocator, in: []const u8, dict: ?*Dictionary) ![]u8 {
    var out: std.ArrayList(u8) = .empty;
    errdefer out.deinit(alloc);
    if (comptime MARKMAP) g_cls.clearRetainingCapacity();
    var n: u64 = in.len;
    var offset: usize = 0;
    while (n > 0) {
        var segment: usize = @intCast(@min(n, kMaxSegment));
        try encodeSegment(alloc, in, offset, segment, dict, &out);
        offset += segment;
        n -= segment;
        segment = 0;
    }
    return out.toOwnedSlice(alloc);
}

/// `preprocessor::NoPreprocess` (preprocessor.cpp:234-243): the `-n` body — DEFAULT-framed
/// raw copy (which is NOT what a naive "raw bytes" path emits).
pub fn noPreprocess(alloc: std.mem.Allocator, in: []const u8) ![]u8 {
    var out: std.ArrayList(u8) = .empty;
    errdefer out.deinit(alloc);
    var n: u64 = in.len;
    var offset: usize = 0;
    while (n > 0) {
        const segment: usize = @intCast(@min(n, kMaxSegment));
        try appendMarker(alloc, &out, DEFAULT, segment);
        try out.appendSlice(alloc, in[offset .. offset + segment]);
        offset += segment;
        n -= segment;
    }
    return out.toOwnedSlice(alloc);
}

/// `preprocessor::Decode` (preprocessor.cpp:245-278) incl. `DecodeByte`/`reset_text_decoder`/
/// `decode_text`/`decode_default`. Inverts `encode`. `dict` must be provided iff the stream
/// used the dictionary (matches the `-s`/`-d` contract).
pub fn decode(alloc: std.mem.Allocator, in: []const u8, dict: ?*Dictionary) ![]u8 {
    var out: std.ArrayList(u8) = .empty;
    errdefer out.deinit(alloc);
    var r = Reader{ .data = in };

    // DecodeByte statics (preprocessor.cpp:246-247, 149).
    var typ: u8 = DEFAULT;
    var len: i64 = 0;
    var wrt_enabled: bool = true;
    if (dict) |d| d.resetDecodeState();

    while (true) {
        // Refill header(s).
        while (len == 0) {
            const c = r.getc();
            if (c == -1) return out.toOwnedSlice(alloc); // EOF
            typ = @intCast(c);
            const b0: u32 = @bitCast(r.getc());
            const b1: u32 = @bitCast(r.getc());
            const b2: u32 = @bitCast(r.getc());
            const b3: u32 = @bitCast(r.getc());
            const lu: u32 = ((b0 & 0xFF) << 24) | ((b1 & 0xFF) << 16) | ((b2 & 0xFF) << 8) | (b3 & 0xFF);
            var l: i32 = @bitCast(lu);
            if (l < 0) l = 1;
            len = l;
            if (typ == TEXT) {
                // reset_text_decoder (preprocessor.cpp:151-159): mode byte via RAW getc.
                const mode = r.getc();
                wrt_enabled = (mode != 0);
            }
        }
        len -= 1;
        var outbyte: u8 = undefined;
        if (typ == TEXT) {
            if (!wrt_enabled) {
                outbyte = @truncate(@as(u32, @bitCast(r.getc())));
            } else {
                outbyte = try dict.?.decodeByte(&r);
            }
        } else {
            outbyte = @truncate(@as(u32, @bitCast(r.getc())));
        }
        try out.append(alloc, outbyte);
    }
}
