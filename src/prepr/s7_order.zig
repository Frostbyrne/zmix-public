//! s7_runlabel re-encoding of the article-order asset (`comp_order`).
//!
//! The shipped `new_article_order` is a permutation of N compact article ids,
//! decimal text, one id per line. Structurally it is R maximal DESCENDING runs
//! (enwik9: N = 172,277, R = 3,884, mean run ~44): readalike clusters emitted
//! id-descending. Cluster MEMBERSHIP is ~97.9 % of the permutation's
//! information; within-run order is forced (descending) and therefore free.
//!
//! s7 re-keys the stream BY ARTICLE ID instead of by output position: one
//! decimal line per present id, in ascending id order, carrying the label
//! (0-based, in order of first appearance) of the run that id belongs to.
//! Measured on the real CM this codes ~13 % smaller than the raw permutation
//! (173,210 vs 199,904
//! on the pinned fourth-E vehicle; six of nine alternative spaces beat the
//! incumbent, s7 best; MTF and complement variants measured WORSE).
//!
//! Because s7 emits one line per PRESENT id, the id set itself becomes side
//! information (the raw permutation carried it implicitly). The id space is
//! near-dense — enwik9: ids 0..172,314 with 38 interior ids absent — so the
//! file is made self-contained by appending the exceptions as a TRAILER:
//!
//!   [N lines]  run label of id, for each present id ascending   (the body)
//!   [M lines]  the missing (absent) interior ids, ascending
//!   [1 line ]  M   (missing-id count)
//!   [1 line ]  R   (run count; consistency check, R == max label + 1)
//!
//! All lines are minimal decimal ASCII, '\n'-terminated, same alphabet as the
//! body (the trailer parses backwards from EOF: last line R, then M, then M
//! missing-id lines; everything before is body). Trailer-at-end keeps the body
//! byte-identical to the measured representation and places the exception
//! digits where the CM's models are warm.
//!
//! encode: one linear pass to cut runs + one ascending id walk. NO SORT — no
//! std.sort/Fenwick/MTF instantiations (the structures that dominated the
//! comparable derivdict build's naive code fee).
//! decode: count labels, prefix-sum, bucket ids ascending, emit each bucket
//! REVERSED (within-run order is forced descending), print decimal. Exact
//! byte-inverse of the original asset text — asserted at mint time
//! (`zmix_ship --s7-encode` refuses to write an asset whose inverse is not
//! byte-identical) and by the golden test below on the real asset.
//!
//! Consumers: ship.zig's asset-container path ONLY (under -Ds7order the
//! in-image `comp_order` segment holds engine-compressed s7 text; ship.zig
//! reconstructs the exact raw order text before any downstream code sees it).
//! The codec, r1 reorder, and dev runner_lex paths are untouched by design.

const std = @import("std");

pub const S7Error = error{
    EmptyOrder,
    MalformedLine,
    MissingTrailingNewline,
    ValueOverflow,
    DuplicateId,
    TruncatedTrailer,
    BadRunCount,
    EmptyRun,
    LabelOutOfRange,
    LabelNotDense,
    MissingIdsNotAscending,
    MissingIdPresent,
    MissingIdUnconsumed,
    OutOfMemory,
};

/// Parse '\n'-terminated minimal-decimal lines into u32 values. Strict: every
/// line non-empty, digits only, no leading zeros (so printing is the exact
/// byte-inverse of parsing), file must end with '\n'.
fn parseLines(gpa: std.mem.Allocator, text: []const u8) S7Error![]u32 {
    if (text.len == 0) return S7Error.EmptyOrder;
    if (text[text.len - 1] != '\n') return S7Error.MissingTrailingNewline;
    var vals: std.ArrayList(u32) = .empty;
    errdefer vals.deinit(gpa);
    var i: usize = 0;
    while (i < text.len) {
        const start = i;
        while (text[i] != '\n') : (i += 1) {
            const c = text[i];
            if (c < '0' or c > '9') return S7Error.MalformedLine;
        }
        const line = text[start..i];
        i += 1; // consume '\n'
        if (line.len == 0) return S7Error.MalformedLine;
        if (line.len > 1 and line[0] == '0') return S7Error.MalformedLine; // leading zero
        var v: u64 = 0;
        for (line) |c| {
            v = v * 10 + (c - '0');
            if (v > std.math.maxInt(u32)) return S7Error.ValueOverflow;
        }
        vals.append(gpa, @intCast(v)) catch return S7Error.OutOfMemory;
    }
    return vals.toOwnedSlice(gpa) catch return S7Error.OutOfMemory;
}

fn printLine(gpa: std.mem.Allocator, out: *std.ArrayList(u8), v: u32) S7Error!void {
    var buf: [12]u8 = undefined;
    const s = std.fmt.bufPrint(&buf, "{d}\n", .{v}) catch unreachable;
    out.appendSlice(gpa, s) catch return S7Error.OutOfMemory;
}

/// Raw order text (decimal id per line) -> s7 text (body + exception trailer).
/// Encode-side only; runs at asset-mint time (construct_ship.sh), never inside
/// a judged operation.
pub fn encode(gpa: std.mem.Allocator, order_text: []const u8) S7Error![]u8 {
    const v = try parseLines(gpa, order_text);
    defer gpa.free(v);
    if (v.len == 0) return S7Error.EmptyOrder;

    var max_id: u32 = 0;
    for (v) |id| max_id = @max(max_id, id);
    const universe: usize = @as(usize, max_id) + 1;

    const lab = gpa.alloc(u32, universe) catch return S7Error.OutOfMemory;
    defer gpa.free(lab);
    const present = gpa.alloc(bool, universe) catch return S7Error.OutOfMemory;
    defer gpa.free(present);
    @memset(present, false);

    // Cut maximal descending runs in one pass; label them 0.. in order of
    // appearance; key the label by id.
    var label: u32 = 0;
    for (v, 0..) |id, k| {
        if (k > 0 and v[k] >= v[k - 1]) label += 1; // ascending step = new run
        if (present[id]) return S7Error.DuplicateId; // not a permutation
        present[id] = true;
        lab[id] = label;
    }
    const run_count: u32 = label + 1;

    var out: std.ArrayList(u8) = .empty;
    errdefer out.deinit(gpa);
    out.ensureTotalCapacity(gpa, universe * 5 + 64) catch return S7Error.OutOfMemory;

    // Body: label per present id, ascending id walk (no sort). Interior absent
    // ids become the missing-exception list.
    var missing: std.ArrayList(u32) = .empty;
    defer missing.deinit(gpa);
    var id: u32 = 0;
    while (id <= max_id) : (id += 1) {
        if (present[id]) {
            try printLine(gpa, &out, lab[id]);
        } else {
            missing.append(gpa, id) catch return S7Error.OutOfMemory;
        }
    }
    // Trailer: missing ids, then M, then R.
    for (missing.items) |mid| try printLine(gpa, &out, mid);
    try printLine(gpa, &out, @intCast(missing.items.len));
    try printLine(gpa, &out, run_count);
    return out.toOwnedSlice(gpa) catch return S7Error.OutOfMemory;
}

/// s7 text -> the EXACT raw order text (decimal id per line). This is the
/// shipped inverse: ship.zig calls it on the decompressed in-image segment, so
/// every downstream consumer (article reorder, r1v2 restore, --extract-assets)
/// sees bytes identical to the stock asset.
pub fn decode(gpa: std.mem.Allocator, s7_text: []const u8) S7Error![]u8 {
    const vals = try parseLines(gpa, s7_text);
    defer gpa.free(vals);
    if (vals.len < 2) return S7Error.TruncatedTrailer;

    const run_count = vals[vals.len - 1];
    const m_count: usize = vals[vals.len - 2];
    if (vals.len < 2 + m_count + 1) return S7Error.TruncatedTrailer; // >= 1 body line
    const n: usize = vals.len - 2 - m_count;
    const labels = vals[0..n];
    const missing = vals[n .. n + m_count];
    if (run_count == 0 or run_count > n) return S7Error.BadRunCount;

    // Missing ids strictly ascending (parse guard; also rejects duplicates).
    for (missing, 0..) |mid, k| {
        if (k > 0 and mid <= missing[k - 1]) return S7Error.MissingIdsNotAscending;
    }

    // Count per label, then prefix-sum into bucket offsets.
    const counts = gpa.alloc(u32, run_count) catch return S7Error.OutOfMemory;
    defer gpa.free(counts);
    @memset(counts, 0);
    for (labels) |l| {
        if (l >= run_count) return S7Error.LabelOutOfRange;
        counts[l] += 1;
    }
    const offsets = gpa.alloc(u32, run_count) catch return S7Error.OutOfMemory;
    defer gpa.free(offsets);
    var acc: u32 = 0;
    for (counts, 0..) |c, l| {
        if (c == 0) return S7Error.LabelNotDense; // labels are dense by construction
        offsets[l] = acc;
        acc += c;
    }

    // Ascending id walk, skipping missing ids: bucket ids per label (each
    // bucket fills ascending).
    const bucket_ids = gpa.alloc(u32, n) catch return S7Error.OutOfMemory;
    defer gpa.free(bucket_ids);
    const fill = gpa.alloc(u32, run_count) catch return S7Error.OutOfMemory;
    defer gpa.free(fill);
    @memset(fill, 0);
    var id: u32 = 0;
    var mi: usize = 0;
    for (labels) |l| {
        while (mi < missing.len and id == missing[mi]) {
            id += 1;
            mi += 1;
        }
        bucket_ids[offsets[l] + fill[l]] = id;
        fill[l] += 1;
        id = std.math.add(u32, id, 1) catch return S7Error.ValueOverflow;
    }
    // Every declared-missing id must be an INTERIOR gap the walk consumed; a
    // trailing exception would mean the asset lies about its id universe.
    if (mi != missing.len) return S7Error.MissingIdUnconsumed;

    // Emit: buckets in label order (= order of first appearance = original run
    // order), each bucket REVERSED (within-run order is forced descending).
    var out: std.ArrayList(u8) = .empty;
    errdefer out.deinit(gpa);
    out.ensureTotalCapacity(gpa, n * 7 + 16) catch return S7Error.OutOfMemory;
    for (0..run_count) |l| {
        const c = counts[l];
        var j: u32 = 0;
        while (j < c) : (j += 1) {
            try printLine(gpa, &out, bucket_ids[offsets[l] + (c - 1 - j)]);
        }
    }
    return out.toOwnedSlice(gpa) catch return S7Error.OutOfMemory;
}

// ===========================================================================
// Tests
// ===========================================================================
const testing = std.testing;

test "s7: synthetic roundtrip with gaps, singleton runs, run at EOF" {
    const gpa = testing.allocator;
    // ids {0,1,2,4,5,7}: runs = [2,1,0], [5,4], [7]  (3 runs, missing {3,6})
    const order = "2\n1\n0\n5\n4\n7\n";
    const enc = try encode(gpa, order);
    defer gpa.free(enc);
    // body: id0..7 labels 0,0,0,[3 missing],1,1,[6 missing],2 ; trailer: 3,6 ; M=2 ; R=3
    try testing.expectEqualStrings("0\n0\n0\n1\n1\n2\n3\n6\n2\n3\n", enc);
    const dec = try decode(gpa, enc);
    defer gpa.free(dec);
    try testing.expectEqualStrings(order, dec);
}

test "s7: single fully-descending run, dense ids" {
    const gpa = testing.allocator;
    const order = "3\n2\n1\n0\n";
    const enc = try encode(gpa, order);
    defer gpa.free(enc);
    try testing.expectEqualStrings("0\n0\n0\n0\n0\n1\n", enc); // 4 labels, M=0, R=1
    const dec = try decode(gpa, enc);
    defer gpa.free(dec);
    try testing.expectEqualStrings(order, dec);
}

test "s7: strictly ascending input = all singleton runs" {
    const gpa = testing.allocator;
    const order = "0\n1\n2\n";
    const enc = try encode(gpa, order);
    defer gpa.free(enc);
    try testing.expectEqualStrings("0\n1\n2\n0\n3\n", enc); // labels 0,1,2; M=0; R=3
    const dec = try decode(gpa, enc);
    defer gpa.free(dec);
    try testing.expectEqualStrings(order, dec);
}

test "s7: malformed inputs are rejected" {
    const gpa = testing.allocator;
    try testing.expectError(S7Error.MissingTrailingNewline, encode(gpa, "1\n0"));
    try testing.expectError(S7Error.MalformedLine, encode(gpa, "1\n\n0\n"));
    try testing.expectError(S7Error.MalformedLine, encode(gpa, "01\n0\n")); // leading zero
    try testing.expectError(S7Error.MalformedLine, encode(gpa, "1x\n0\n"));
    try testing.expectError(S7Error.DuplicateId, encode(gpa, "1\n1\n"));
    try testing.expectError(S7Error.EmptyOrder, encode(gpa, ""));
    // decode-side: label 5 with R=1
    try testing.expectError(S7Error.LabelOutOfRange, decode(gpa, "5\n0\n1\n"));
    // decode-side: declared-missing id beyond the walked range
    try testing.expectError(S7Error.MissingIdUnconsumed, decode(gpa, "0\n9\n1\n1\n"));
}

// Golden: the REAL shipped asset round-trips byte-exactly through the s7
// container, and its structural constants match the measured report
// N=172,277, R=3,884, M=38 interior
// missing ids, raw s7 size = body 787,025 B + trailer 241 B = 787,266 B
// (trailer: 38 missing-id lines 233 B + "38\n" + "3884\n").
const order_asset = @embedFile("new_article_order_asset");

test "s7: real order asset — exact byte roundtrip + structural goldens" {
    const gpa = testing.allocator;
    const enc = try encode(gpa, order_asset);
    defer gpa.free(enc);
    try testing.expectEqual(@as(usize, 787_266), enc.len);
    const vals = try parseLines(gpa, enc);
    defer gpa.free(vals);
    try testing.expectEqual(@as(u32, 3_884), vals[vals.len - 1]); // R
    try testing.expectEqual(@as(u32, 38), vals[vals.len - 2]); // M
    try testing.expectEqual(@as(usize, 172_277 + 38 + 2), vals.len);
    const dec = try decode(gpa, enc);
    defer gpa.free(dec);
    try testing.expect(std.mem.eql(u8, dec, order_asset));
}
