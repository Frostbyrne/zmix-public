//! fieldcodec — the ts-before-revid null-swap (T1 arm of the `fieldcodec-composite`
//! family), as a deployable post-WRT temp-stream stage.
//!
//! MEASURED:
//! emitting each revid record's TIMESTAMP line BEFORE its revid line — a frozen
//! rule with NO residual stream and NO table — saves −1,624 B on the [560,562) MB
//! region of the e9 temp under the ship engine (−1,795 B July engine), ≈ +11 K e9
//! at the frozen ×6.687 pool rule (+3.7 K at ×record scaling). The incumbent then
//! exploits ts→revid correlation implicitly. Byte-count preserving.
//!
//! Record grammar (this implementation is verified byte-exact against the
//! reference transform: encode(prep.temp 587,138,826 B) == T1.temp sha16
//! 8d14c24582761c3e, decode(T1.temp) == prep.temp sha16 7826ff63dedd526c,
//! 243,425 records swapped — see the port report):
//!
//!   id line:  DF 86 4E  d{8}        0A                      (fixed 12 bytes)
//!   ts line:  DF CD 4E  d{2} d{1,3} 4A d{1,} 0A             (variable)
//!
//! (d = ASCII '0'..'9'. The 8-digit id width is an asserted property of the
//! record grammar: every one of the 243,425 e9 records has an exactly-8-digit revid.)
//!
//! A RECORD is an id line immediately followed by a ts line. encodeInPlace
//! reorders every record to (ts line)(id line); decodeInPlace reorders every
//! (ts line)(id line) adjacency back. Both are single left-to-right passes that
//! resume after each swapped block, mirroring the arm builder's re.finditer
//! semantics (proven equivalent: no line grammar can contain an interior 0xDF,
//! so no match can start inside a consumed line).
//!
//! Applied to the WHOLE temp buffer the coder codes — data region plus the r1
//! side footer at e9, exactly as the measured arm was built. Inversion is exact
//! on every stream the gates measure; it is NOT a universal-inverse pair (a
//! natural (ts)(id) adjacency in a source stream that the encoder did not
//! create would be falsely un-swapped), so losslessness is owed to — and
//! settled by — the roundtrip gates on the actual shipped content, never
//! assumed. On the full e9 temp it is settled by the frozen-sha check above.
//!
//! Placement (runner_lex.zig): encode = …→WRT→[r1 at -e]→encodeInPlace→coder;
//! decode = coder→decodeInPlace→[r1 restore at -D]→WRT decode→… . The r1 side
//! blob is minted and consumed on the un-transformed stream, and the swap is
//! the LAST transform before the coder.

const std = @import("std");

const ID_DIGITS = 8;
/// DF 86 4E + 8 digits + 0A.
const ID_LINE_LEN = 3 + ID_DIGITS + 1;

inline fn isDigit(c: u8) bool {
    return c -% '0' < 10;
}

/// Length (always ID_LINE_LEN) of the id line starting at `p`, or null.
fn idLineLen(buf: []const u8, p: usize) ?usize {
    if (buf.len - p < ID_LINE_LEN) return null;
    if (buf[p] != 0xdf or buf[p + 1] != 0x86 or buf[p + 2] != 0x4e) return null;
    for (buf[p + 3 ..][0..ID_DIGITS]) |c| {
        if (!isDigit(c)) return null;
    }
    if (buf[p + 3 + ID_DIGITS] != 0x0a) return null;
    return ID_LINE_LEN;
}

/// Length of the ts line starting at `p`, or null. Deterministic equivalent of
/// the arm builder's regex `DF CD 4E ([0-9]{2})([0-9]{1,3}) 4A ([0-9]+) 0A`:
/// the digit run after the marker must end EXACTLY at 0x4A with total length
/// 3..5 (2 for yy + 1..3 for month*31+day), then >=1 digit ending EXACTLY at 0x0A.
fn tsLineLen(buf: []const u8, p: usize) ?usize {
    if (buf.len - p < 3) return null;
    if (buf[p] != 0xdf or buf[p + 1] != 0xcd or buf[p + 2] != 0x4e) return null;
    var i = p + 3;
    const date0 = i;
    while (i < buf.len and isDigit(buf[i])) i += 1;
    const date_digits = i - date0;
    if (date_digits < 3 or date_digits > 5) return null;
    if (i >= buf.len or buf[i] != 0x4a) return null;
    i += 1;
    const sec0 = i;
    while (i < buf.len and isDigit(buf[i])) i += 1;
    if (i == sec0) return null;
    if (i >= buf.len or buf[i] != 0x0a) return null;
    return i + 1 - p;
}

/// ENCODE: reorder every record (id line)(ts line) to (ts line)(id line),
/// in place. Returns the number of records swapped.
pub fn encodeInPlace(buf: []u8) usize {
    var swaps: usize = 0;
    var i: usize = 0;
    while (std.mem.indexOfScalarPos(u8, buf, i, 0xdf)) |j| {
        const il = idLineLen(buf, j) orelse {
            i = j + 1;
            continue;
        };
        const tl = tsLineLen(buf, j + il) orelse {
            i = j + 1;
            continue;
        };
        var id_line: [ID_LINE_LEN]u8 = undefined;
        @memcpy(&id_line, buf[j..][0..ID_LINE_LEN]);
        std.mem.copyForwards(u8, buf[j..][0..tl], buf[j + il ..][0..tl]);
        @memcpy(buf[j + tl ..][0..ID_LINE_LEN], &id_line);
        swaps += 1;
        i = j + il + tl;
    }
    return swaps;
}

/// DECODE: reorder every (ts line)(id line) adjacency back to
/// (id line)(ts line), in place. Returns the number of records un-swapped.
pub fn decodeInPlace(buf: []u8) usize {
    var swaps: usize = 0;
    var i: usize = 0;
    while (std.mem.indexOfScalarPos(u8, buf, i, 0xdf)) |j| {
        const tl = tsLineLen(buf, j) orelse {
            i = j + 1;
            continue;
        };
        const il = idLineLen(buf, j + tl) orelse {
            i = j + 1;
            continue;
        };
        var id_line: [ID_LINE_LEN]u8 = undefined;
        @memcpy(&id_line, buf[j + tl ..][0..ID_LINE_LEN]);
        std.mem.copyBackwards(u8, buf[j + il ..][0..tl], buf[j..][0..tl]);
        @memcpy(buf[j..][0..ID_LINE_LEN], &id_line);
        swaps += 1;
        i = j + il + tl;
    }
    return swaps;
}

// ---------------------------------------------------------------------------

const t = std.testing;

fn roundtrip(original: []const u8, expect_swaps: usize) !void {
    var work: [256]u8 = undefined;
    const w = work[0..original.len];
    @memcpy(w, original);
    try t.expectEqual(expect_swaps, encodeInPlace(w));
    if (expect_swaps == 0) try t.expectEqualSlices(u8, original, w);
    try t.expectEqual(expect_swaps, decodeInPlace(w));
    try t.expectEqualSlices(u8, original, w);
}

test "fieldcodec: single record swaps and inverts" {
    const rec = "x\xdf\x86\x4e" ++ "31886450" ++ "\x0a" ++ "\xdf\xcd\x4e" ++ "05" ++ "123" ++ "\x4a" ++ "45678" ++ "\x0a" ++ "y";
    const want = "x" ++ "\xdf\xcd\x4e" ++ "05" ++ "123" ++ "\x4a" ++ "45678" ++ "\x0a" ++ "\xdf\x86\x4e" ++ "31886450" ++ "\x0a" ++ "y";
    var work: [rec.len]u8 = rec.*;
    try t.expectEqual(@as(usize, 1), encodeInPlace(&work));
    try t.expectEqualSlices(u8, want, &work);
    try t.expectEqual(@as(usize, 1), decodeInPlace(&work));
    try t.expectEqualSlices(u8, rec, &work);
}

test "fieldcodec: consecutive records both swap" {
    const rec = "\xdf\x86\x4e" ++ "00000001" ++ "\x0a" ++ "\xdf\xcd\x4e" ++ "051" ++ "\x4a" ++ "1" ++ "\x0a" ++
        "\xdf\x86\x4e" ++ "00000002" ++ "\x0a" ++ "\xdf\xcd\x4e" ++ "0512" ++ "\x4a" ++ "86399" ++ "\x0a";
    try roundtrip(rec, 2);
}

test "fieldcodec: non-records untouched" {
    // id without ts, ts without id, wrong digit counts, truncated tails
    try roundtrip("\xdf\x86\x4e" ++ "12345678" ++ "\x0a" ++ "plain", 0); // id, no ts after
    try roundtrip("\xdf\xcd\x4e" ++ "05123" ++ "\x4a" ++ "1" ++ "\x0a" ++ "tail", 0); // lone ts
    try roundtrip("\xdf\x86\x4e" ++ "1234567" ++ "\x0a" ++ "\xdf\xcd\x4e" ++ "051" ++ "\x4a" ++ "1" ++ "\x0a", 0); // 7-digit id
    try roundtrip("\xdf\x86\x4e" ++ "123456789" ++ "\x0a" ++ "\xdf\xcd\x4e" ++ "051" ++ "\x4a" ++ "1" ++ "\x0a", 0); // 9-digit id
    try roundtrip("\xdf\x86\x4e" ++ "12345678" ++ "\x0a" ++ "\xdf\xcd\x4e" ++ "123456" ++ "\x4a" ++ "1" ++ "\x0a", 0); // 6 date digits
    try roundtrip("\xdf\x86\x4e" ++ "12345678" ++ "\x0a" ++ "\xdf\xcd\x4e" ++ "05" ++ "\x4a" ++ "1" ++ "\x0a", 0); // 2 date digits
    try roundtrip("\xdf\x86\x4e" ++ "12345678" ++ "\x0a" ++ "\xdf\xcd\x4e" ++ "051" ++ "\x4a" ++ "\x0a", 0); // no seconds digit
    try roundtrip("\xdf\x86\x4e" ++ "12345678" ++ "\x0a" ++ "\xdf\xcd\x4e" ++ "051" ++ "\x4a" ++ "12", 0); // no newline / EOF
    try roundtrip("\xdf\x86\x4e" ++ "1234", 0); // truncated id at EOF
}

test "fieldcodec: ts-then-record chain (the tricky decode ordering)" {
    // natural [ts_N][id_M][ts_M]: encode must produce [ts_N][ts_M][id_M] and
    // decode must restore — the lone leading ts line must not derail the scan.
    const rec = "\xdf\xcd\x4e" ++ "051" ++ "\x4a" ++ "7" ++ "\x0a" ++
        "\xdf\x86\x4e" ++ "00000009" ++ "\x0a" ++ "\xdf\xcd\x4e" ++ "052" ++ "\x4a" ++ "8" ++ "\x0a";
    const enc = "\xdf\xcd\x4e" ++ "051" ++ "\x4a" ++ "7" ++ "\x0a" ++
        "\xdf\xcd\x4e" ++ "052" ++ "\x4a" ++ "8" ++ "\x0a" ++ "\xdf\x86\x4e" ++ "00000009" ++ "\x0a";
    var work: [rec.len]u8 = rec.*;
    try t.expectEqual(@as(usize, 1), encodeInPlace(&work));
    try t.expectEqualSlices(u8, enc, &work);
    try t.expectEqual(@as(usize, 1), decodeInPlace(&work));
    try t.expectEqualSlices(u8, rec, &work);
}

test "fieldcodec: empty and marker-free buffers" {
    var empty: [0]u8 = .{};
    try t.expectEqual(@as(usize, 0), encodeInPlace(&empty));
    try t.expectEqual(@as(usize, 0), decodeInPlace(&empty));
    try roundtrip("no markers at all 0123456789\x0a", 0);
}

test "fieldcodec: frozen e9 arm identity (env-gated)" {
    // The port's ground truth: encode(prep.temp) must equal the frozen T1 arm
    // byte-for-byte (sha16 8d14c24582761c3e, 243,425 swaps) and decode must
    // invert to T0 (7826ff63dedd526c) — the full 587,138,826-byte e9 temp,
    // footer included. Gated on the arm paths so `zig build test` stays cheap:
    //   ZMIX_FIELDCODEC_T0=<prep.temp> ZMIX_FIELDCODEC_T1=<T1.temp> zig build test
    const a = t.allocator;
    const t0_path = std.process.getEnvVarOwned(a, "ZMIX_FIELDCODEC_T0") catch return error.SkipZigTest;
    defer a.free(t0_path);
    const t1_path = std.process.getEnvVarOwned(a, "ZMIX_FIELDCODEC_T1") catch return error.SkipZigTest;
    defer a.free(t1_path);
    const t0 = try std.fs.cwd().readFileAlloc(a, t0_path, 1 << 31);
    defer a.free(t0);
    const t1 = try std.fs.cwd().readFileAlloc(a, t1_path, 1 << 31);
    defer a.free(t1);
    const work = try a.dupe(u8, t0);
    defer a.free(work);
    try t.expectEqual(@as(usize, 243_425), encodeInPlace(work));
    try t.expectEqualSlices(u8, t1, work);
    try t.expectEqual(@as(usize, 243_425), decodeInPlace(work));
    try t.expectEqualSlices(u8, t0, work);
}
