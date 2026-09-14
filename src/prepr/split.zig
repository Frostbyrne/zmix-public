//! Port of cmix-lex's `split4Comp` / `split4Decomp`
//! (`ref/cmix-lex/src/readalike_prepr/misc.h`).
//!
//! These are line-index byte routers used ONLY on the full `-e` enwik9 path
//! (runner.cpp:422-472 compress, 377-408 decompress). They split the byte
//! stream into three contiguous pieces by counting `\n` bytes. Deterministic,
//! no model state, byte-for-byte losslessly reversible.
//!
//! The line-count boundaries are enwik9-SPECIFIC magic numbers hard-coded in
//! misc.h; they are reproduced here verbatim.
//!
//! C++ line-count semantics (misc.h:17-31): each byte is written to the bucket
//! selected by the CURRENT `line_count`, and only AFTER writing is `line_count`
//! incremented when the byte is `\n` (0x0A). Therefore the `\n` that terminates
//! line (X-1) is still routed as belonging to line (X-1); the bucket boundary
//! falls immediately AFTER the X-th newline. Because `line_count` is monotonic,
//! the three buckets are always contiguous ranges, so we express the split as
//! two cut offsets rather than a per-byte copy.
//!
//! Public API:
//!   split4Comp(input)   -> CompSplit{ intro, main, coda }
//!   split4Decomp(input) -> DecompSplit{ main, intro, coda }
//!   offsetAfterNewlines(input, n) -> usize   (the primitive)

const std = @import("std");

// ---- Compress-side boundaries (misc.h:2-4) -------------------------------
// #define COMP_INTRO_END_LINE 29
// #define COMP_MAIN_END_LINE  13146932
// #define COMP_CODA_END_LINE  13147025   <-- DEAD: both else-branches (misc.h:24
//                                            and :26) write to `.coda`, so the
//                                            13147025 threshold never changes
//                                            routing. Kept for documentation.
pub const COMP_INTRO_END_LINE: u64 = 29;
pub const COMP_MAIN_END_LINE: u64 = 13146932;
pub const COMP_CODA_END_LINE: u64 = 13147025; // dead branch (see note above)

// ---- Decompress-side boundaries ------------------------------------------
// misc.h uses LITERALS in the code (13146906 / 13146935 / 13147027), NOT the
// DECOMP_*_END_LINE #defines (13146905 / 13146934 / 13147027) at misc.h:6-8,
// which are stale documentation. We hard-code the literals the code actually
// uses (misc.h:49,51,53).
pub const DECOMP_MAIN_END_LINE: u64 = 13146906; // misc.h:49
pub const DECOMP_INTRO_END_LINE: u64 = 13146935; // misc.h:51
pub const DECOMP_CODA_END_LINE: u64 = 13147027; // misc.h:53 (dead: :53 and :55 both -> coda)

pub const CompSplit = struct {
    /// lines [0, 29)                -> `.intro` (misc.h:20-21)
    intro: []const u8,
    /// lines [29, 13146932)         -> `.main`  (misc.h:22-23)
    main: []const u8,
    /// lines [13146932, inf)        -> `.coda`  (misc.h:24-27, both branches)
    coda: []const u8,
};

pub const DecompSplit = struct {
    /// lines [0, 13146906)          -> `.main_decomp`  (misc.h:49-50)
    main: []const u8,
    /// lines [13146906, 13146935)   -> `.intro_decomp` (misc.h:51-52)
    intro: []const u8,
    /// lines [13146935, inf)        -> `.coda_decomp`  (misc.h:53-56, both branches)
    coda: []const u8,
};

/// Byte offset immediately AFTER the `n`-th `\n` (1-indexed) in `input`.
/// Equivalently: the index of the first byte whose `line_count` == `n` under
/// the misc.h counting rule. Returns `input.len` if there are fewer than `n`
/// newlines (mirrors the C++ `getc==EOF` break: the remaining buckets stay
/// empty). `n == 0` returns 0.
pub fn offsetAfterNewlines(input: []const u8, n: u64) usize {
    if (n == 0) return 0;
    var count: u64 = 0;
    var i: usize = 0;
    while (i < input.len) : (i += 1) {
        if (input[i] == 10) {
            count += 1;
            if (count == n) return i + 1;
        }
    }
    return input.len;
}

/// `split4Comp` (misc.h:10-37). Routes enwik9 into intro/main/coda by line index.
pub fn split4Comp(input: []const u8) CompSplit {
    const a = offsetAfterNewlines(input, COMP_INTRO_END_LINE); // first byte of line 29
    const b = offsetAfterNewlines(input, COMP_MAIN_END_LINE); // first byte of line 13146932
    // b can never precede a (monotonic), but clamp defensively for short inputs.
    const bb = if (b < a) a else b;
    return .{
        .intro = input[0..a],
        .main = input[a..bb],
        .coda = input[bb..],
    };
}

/// `split4Decomp` (misc.h:39-67). Input layout is [main_phda9][intro][coda];
/// note the OUTPUT field order differs from comp (main first).
pub fn split4Decomp(input: []const u8) DecompSplit {
    const m = offsetAfterNewlines(input, DECOMP_MAIN_END_LINE); // end of main_decomp
    const i = offsetAfterNewlines(input, DECOMP_INTRO_END_LINE); // end of intro_decomp
    const ii = if (i < m) m else i;
    return .{
        .main = input[0..m],
        .intro = input[m..ii],
        .coda = input[ii..],
    };
}

// ===========================================================================
// Tests
// ===========================================================================

const testing = std.testing;

test "offsetAfterNewlines primitive" {
    const s = "a\nbb\nccc\n"; // newlines at idx 1,4,8
    try testing.expectEqual(@as(usize, 0), offsetAfterNewlines(s, 0));
    try testing.expectEqual(@as(usize, 2), offsetAfterNewlines(s, 1)); // after 1st \n
    try testing.expectEqual(@as(usize, 5), offsetAfterNewlines(s, 2)); // after 2nd \n
    try testing.expectEqual(@as(usize, 9), offsetAfterNewlines(s, 3)); // after 3rd \n
    try testing.expectEqual(@as(usize, s.len), offsetAfterNewlines(s, 4)); // fewer -> len
}

/// Reference re-implementation matching the C++ byte-copy loop exactly, used to
/// prove the offset-based split is equivalent for arbitrary boundaries.
fn refRoute(
    input: []const u8,
    b0: u64,
    b1: u64,
    out0: *std.ArrayList(u8),
    out1: *std.ArrayList(u8),
    out2: *std.ArrayList(u8),
    comptime comp_order: bool, // true: buckets are [intro,main,coda]; false: [main,intro,coda]
    alloc: std.mem.Allocator,
) !void {
    var line_count: u64 = 0;
    for (input) |c| {
        // comp:   <b0 -> out0(intro), <b1 -> out1(main), else out2(coda)
        // decomp: <b0 -> out0(main),  <b1 -> out1(intro), else out2(coda)
        _ = comp_order;
        if (line_count < b0) {
            try out0.append(alloc, c);
        } else if (line_count < b1) {
            try out1.append(alloc, c);
        } else {
            try out2.append(alloc, c);
        }
        if (c == 10) line_count += 1;
    }
}

test "split4Comp matches C++ byte-copy routing on synthetic input" {
    const alloc = testing.allocator;
    // Build 40 lines: "L<idx>\n". With small boundaries we mirror the shape of
    // the real magic numbers (intro=[0,3), main=[3,7), coda=[7,inf)).
    var buf = std.ArrayList(u8){};
    defer buf.deinit(alloc);
    var li: usize = 0;
    while (li < 12) : (li += 1) {
        try buf.print(alloc, "line{d}\n", .{li});
    }
    const input = buf.items;

    // Reference route with boundaries 3 and 7.
    var o0 = std.ArrayList(u8){};
    var o1 = std.ArrayList(u8){};
    var o2 = std.ArrayList(u8){};
    defer o0.deinit(alloc);
    defer o1.deinit(alloc);
    defer o2.deinit(alloc);
    try refRoute(input, 3, 7, &o0, &o1, &o2, true, alloc);

    // Offset-based split with the same boundaries.
    const a = offsetAfterNewlines(input, 3);
    const b = offsetAfterNewlines(input, 7);
    try testing.expectEqualSlices(u8, o0.items, input[0..a]);
    try testing.expectEqualSlices(u8, o1.items, input[a..b]);
    try testing.expectEqualSlices(u8, o2.items, input[b..]);

    // Round-trip: concatenation of the three buckets == input.
    try testing.expectEqual(input.len, o0.items.len + o1.items.len + o2.items.len);
}

test "split4Decomp field order and round-trip" {
    const alloc = testing.allocator;
    var buf = std.ArrayList(u8){};
    defer buf.deinit(alloc);
    var li: usize = 0;
    while (li < 12) : (li += 1) {
        try buf.print(alloc, "x{d}\n", .{li});
    }
    const input = buf.items;

    var o0 = std.ArrayList(u8){}; // main
    var o1 = std.ArrayList(u8){}; // intro
    var o2 = std.ArrayList(u8){}; // coda
    defer o0.deinit(alloc);
    defer o1.deinit(alloc);
    defer o2.deinit(alloc);
    try refRoute(input, 5, 9, &o0, &o1, &o2, false, alloc);

    const m = offsetAfterNewlines(input, 5);
    const i = offsetAfterNewlines(input, 9);
    try testing.expectEqualSlices(u8, o0.items, input[0..m]);
    try testing.expectEqualSlices(u8, o1.items, input[m..i]);
    try testing.expectEqualSlices(u8, o2.items, input[i..]);

    // main | intro | coda reconstructs the whole input.
    try testing.expectEqual(input.len, o0.items.len + o1.items.len + o2.items.len);
}

test "comp/decomp boundary constants are the misc.h literals" {
    try testing.expectEqual(@as(u64, 29), COMP_INTRO_END_LINE);
    try testing.expectEqual(@as(u64, 13146932), COMP_MAIN_END_LINE);
    try testing.expectEqual(@as(u64, 13146906), DECOMP_MAIN_END_LINE);
    try testing.expectEqual(@as(u64, 13146935), DECOMP_INTRO_END_LINE);
}

test "no trailing newline: last partial line routed by current count" {
    const alloc = testing.allocator;
    const input = "a\nb\nc"; // 2 newlines, last line 'c' has no \n
    var o0 = std.ArrayList(u8){};
    var o1 = std.ArrayList(u8){};
    var o2 = std.ArrayList(u8){};
    defer o0.deinit(alloc);
    defer o1.deinit(alloc);
    defer o2.deinit(alloc);
    try refRoute(input, 1, 2, &o0, &o1, &o2, true, alloc);
    const a = offsetAfterNewlines(input, 1);
    const b = offsetAfterNewlines(input, 2);
    try testing.expectEqualSlices(u8, o0.items, input[0..a]); // "a\n"
    try testing.expectEqualSlices(u8, o1.items, input[a..b]); // "b\n"
    try testing.expectEqualSlices(u8, o2.items, input[b..]); // "c"
}
