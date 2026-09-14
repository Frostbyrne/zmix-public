//! Port of cmix-lex's post-WRT payload_lex regime-1 reorder
//! (`ref/cmix-lex/src/r1_reorder_transform.cpp`).
//!
//! This runs on the full-enwik9 `-e` path INSIDE `RunCompression`, AFTER WRT/dict
//! Encode and BEFORE the arithmetic coder (runner.cpp:235-239), and its inverse
//! runs on decode AFTER arith-decode and BEFORE WRT Decode (runner.cpp:322-328).
//!
//! It is a fixed, submission-specific transform: it only accepts a stream of the
//! exact size `kEncodedTailStart + kEncodedTailLen = 586,459,321` bytes (the real
//! enwik9 post-WRT stream). It:
//!   1. Splits the trailing `kEncodedTailLen` region into newline-delimited lines,
//!      partitioned into three regimes r0/r1/r2 by byte offset.
//!   2. Within regime 1, groups lines into "blocks" (each starting with an exact
//!      `DF 99 'N'` line), reorders the blocks by their body sort-key, and records
//!      the inverse permutation (a d86a-ordered Lehmer code) in a compact `side`
//!      channel that is appended to the stream (magic + u64 length footer).
//!   3. On restore, the side is peeled off the footer and the blocks are put back.
//!
//! Because it materially reorders bytes AND appends the side, the post-r1 stream is
//! what `FX_PREPARE_ONLY` dumps and what the coder actually codes, so byte-exact
//! parity with cmix-lex requires this stage.
//!
//! Pure/deterministic; operates over in-memory buffers (the C++ uses temp files).
//!
//! -Dr1v2 adds the R1ORD4 "derived side" codec beside v1 (reference:
//! an absolute build-machine path, C++ line refs "v2:N";
//! marker detection from the fork's r1_reorder_transform.h, refs "mk:N"; plan
//! The regime-1 region is MARKER-DETECTED (longest
//! run of regular D99/D86a blocks — the v2 scheme's definition; NOT v1's
//! fixed-offset parse, whose block set differs on the real stream: sorted
//! mains diverge at byte 557,242,677), same payload-lex sort, and the side
//! blob is derived: most of the sorted->original permutation is reconstructed
//! at decode time from the article-order asset (carried in the shipped binary
//! as comp_order) + the per-block `N`-line page-id delta chains that ride
//! inside the stream. The side shrinks 679,489 (v1) -> 573,228 B raw and is
//! restructured to be far cheaper to code (banked pair a fleet pair:
//! -170,309 B @r1region16m). Encode SELF-VERIFIES (full v2 restore in memory)
//! and hard-fails rather than emit a stream it cannot restore. Flags-off
//! builds compile the v2 paths away.

const std = @import("std");
const build_options = @import("build_options");
const prepr_options = @import("prepr_options");
const A = std.mem.Allocator;

pub const Error = error{ R1Refused, R1Corrupt, R1OrderMissing, R1SelfVerifyFailed } || A.Error;

/// -Dphda9-nullpred re-mints prep.temp, so these fixed offsets — which are
/// properties of the SHIPPED post-WRT stream layout, not of the transform —
/// move with it. The transform itself is untouched: the same 243,425 exact-D99
/// blocks, the same content, the same (sort_key, original_index) sort; only
/// their address changes. See the const block below.
const NULLPRED: bool = prepr_options.nullpred;
const TEST_NO_R1: bool = prepr_options.test_no_r1;

// r1_reorder_transform.cpp:37-38
//
// The -Dphda9-nullpred column was MEASURED on the exact pipeline
// (`--stages` + `--wrt` over enwik9 with the shipped order asset and the
// 4c8568cc english.dic; the stock arm reproduces post-WRT sha cb466004… — the
// frozen cpp1 substrate — so both columns are anchored on the same run):
//
//   quantity                       stock         -Dphda9-nullpred
//   post-WRT stream length         586,459,321   586,459,314   (−7: the phda9
//                                                              tail's lang-blob
//                                                              length line goes
//                                                              from 8 digits to "0")
//   first exact-DF 99 'N' line     554,867,470   570,735,144   (+15,867,674 — the
//   last  exact-DF 99 'N' line     570,585,304   586,452,978    lang blob is no
//   D99 line count                 243,425       243,425        longer segregated
//   header-blob span               15,717,834    15,717,834     ahead of it)
//   max inter-D99 line gap         8             8             (⇒ every block < 16
//                                                               lines, so the block
//                                                               walk is unchanged)
//
// Under nullpred regime 1 is set to start exactly ON the first block (empty
// prelude) and to run to the end of the tail (empty r2). prelude/suffix/r0/r2
// are emitted VERBATIM either way, so this changes no output byte of the
// transform — only which of the five verbatim buckets a line is counted in, and
// hence a few varints in the side header.
pub const kEncodedTailStart: usize = 541126651;
pub const kEncodedTailLen: usize = if (NULLPRED) 45332663 else 45332670;
pub const kExpectedStreamLen: usize = kEncodedTailStart + kEncodedTailLen; // 586459321
const kEncodedRegime1Start: usize = if (NULLPRED) 29608493 else 13599801;
const kEncodedRegime2Start: usize = if (NULLPRED) 45332663 else 30372888;

// :32-35
const kSideMagicD86 = [_]u8{ 'R', '1', 'O', 'R', 'D', '3', '\n' };
const kFooterMagic = [_]u8{ 'R', '1', 'O', 'R', 'D', 'F', 'T', 'R' };
const kD99Line = [_]u8{ 0xDF, 0x99, 'N' };
const kD86Prefix = [_]u8{ 0xDF, 0x86, 'N' };
// v2:68-69
const kSideMagicV2 = [_]u8{ 'R', '1', 'O', 'R', 'D', '4', '\n' };
const kFooterMagicV2 = [_]u8{ 'R', '1', 'O', 'R', 'D', 'F', 'T', '4' };

const LineRef = struct { start: usize = 0, len: usize = 0 };

const TailBlock = struct {
    original_index: usize = 0,
    lines: []LineRef, // slice into the r1 line array (no copy)
    sort_key: []const u8, // owned
};

// ---------------------------------------------------------------------------
// primitives (r1_reorder_transform.cpp:65-194)
// ---------------------------------------------------------------------------

fn hasBytesAt(data: []const u8, pos: usize, bytes: []const u8) bool {
    return pos <= data.len and data.len - pos >= bytes.len and
        std.mem.eql(u8, data[pos .. pos + bytes.len], bytes);
}

fn bodyEnd(data: []const u8, line: LineRef) usize {
    if (line.start > data.len or line.len > data.len - line.start) return line.start;
    var end = line.start + line.len;
    if (end > line.start and data[end - 1] == '\n') end -= 1;
    if (end > line.start and data[end - 1] == '\r') end -= 1;
    return end;
}

fn splitLines(alloc: A, start: usize, len: usize, data: []const u8) ![]LineRef {
    var lines: std.ArrayList(LineRef) = .empty;
    errdefer lines.deinit(alloc);
    if (start > data.len or len > data.len - start) return lines.toOwnedSlice(alloc);
    const end = start + len;
    var line_start = start;
    var i = start;
    while (i < end) : (i += 1) {
        if (data[i] == '\n') {
            try lines.append(alloc, .{ .start = line_start, .len = i + 1 - line_start });
            line_start = i + 1;
        }
    }
    if (line_start < end) try lines.append(alloc, .{ .start = line_start, .len = end - line_start });
    return lines.toOwnedSlice(alloc);
}

fn isExactD99Line(data: []const u8, line: LineRef) bool {
    const end = bodyEnd(data, line);
    return end - line.start == kD99Line.len and
        std.mem.eql(u8, data[line.start .. line.start + kD99Line.len], &kD99Line);
}

fn isAsciiNumberLine(data: []const u8, line: LineRef) bool {
    const end = bodyEnd(data, line);
    if (line.start == end) return false;
    var i = line.start;
    while (i < end) : (i += 1) {
        const c = data[i];
        if ((c < '0' or c > '9') and c != '-') return false;
    }
    return true;
}

fn parseUnsignedExact(data: []const u8, pos_in: usize, end: usize, value: *u64) bool {
    var pos = pos_in;
    if (pos >= end or data[pos] < '0' or data[pos] > '9') return false;
    var result: u64 = 0;
    while (pos < end and data[pos] >= '0' and data[pos] <= '9') {
        const digit: u64 = data[pos] - '0';
        pos += 1;
        if (result > (std.math.maxInt(u64) - digit) / 10) return false;
        result = result * 10 + digit;
    }
    if (pos != end) return false;
    value.* = result;
    return true;
}

fn parseFirstD86a(data: []const u8, block: TailBlock, value: *u64) bool {
    if (block.lines.len < 2) return false;
    const line = block.lines[1];
    const end = bodyEnd(data, line);
    if (end < line.start + kD86Prefix.len or
        !std.mem.eql(u8, data[line.start .. line.start + kD86Prefix.len], &kD86Prefix))
    {
        return false;
    }
    return parseUnsignedExact(data, line.start + kD86Prefix.len, end, value);
}

fn appendLine(out: *std.ArrayList(u8), alloc: A, data: []const u8, line: LineRef) !void {
    try out.appendSlice(alloc, data[line.start .. line.start + line.len]);
}

fn appendLines(out: *std.ArrayList(u8), alloc: A, data: []const u8, lines: []const LineRef) !void {
    for (lines) |line| try appendLine(out, alloc, data, line);
}

fn appendVarint(out: *std.ArrayList(u8), alloc: A, value_in: u64) !void {
    var value = value_in;
    while (value >= 0x80) {
        try out.append(alloc, @intCast((value & 0x7F) | 0x80));
        value >>= 7;
    }
    try out.append(alloc, @intCast(value));
}

fn readVarint(data: []const u8, pos: *usize, value: *u64) bool {
    var result: u64 = 0;
    var shift: u6 = 0;
    while (pos.* < data.len and shift <= 63) {
        const byte = data[pos.*];
        pos.* += 1;
        if (shift == 63 and (byte & 0x7F) > 1) return false;
        result |= @as(u64, byte & 0x7F) << shift;
        if ((byte & 0x80) == 0) {
            value.* = result;
            return true;
        }
        if (shift > 56) return false; // shift+7 would exceed 63
        shift += 7;
    }
    return false;
}

fn appendU64LE(out: *std.ArrayList(u8), alloc: A, value: u64) !void {
    var i: usize = 0;
    while (i < 8) : (i += 1) try out.append(alloc, @intCast((value >> @intCast(8 * i)) & 0xFF));
}

fn readU64LE(data: []const u8, pos: usize, value: *u64) bool {
    if (pos > data.len or data.len - pos < 8) return false;
    var result: u64 = 0;
    var i: usize = 0;
    while (i < 8) : (i += 1) result |= @as(u64, data[pos + i]) << @intCast(8 * i);
    value.* = result;
    return true;
}

// ---------------------------------------------------------------------------
// Fenwick tree (196-231) for the Lehmer permutation codec
// ---------------------------------------------------------------------------
const Fenwick = struct {
    tree: []i64,

    fn init(alloc: A, n: usize) !Fenwick {
        const tree = try alloc.alloc(i64, n + 1);
        @memset(tree, 0);
        var f = Fenwick{ .tree = tree };
        var i: usize = 0;
        while (i < n) : (i += 1) f.add(i, 1);
        return f;
    }
    fn deinit(self: *Fenwick, alloc: A) void {
        alloc.free(self.tree);
    }
    fn add(self: *Fenwick, index_in: usize, delta: i64) void {
        var index = index_in + 1;
        while (index < self.tree.len) : (index += index & (~index +% 1)) {
            self.tree[index] += delta;
        }
    }
    fn sumLessThan(self: *const Fenwick, index_in: usize) usize {
        var index = index_in;
        var result: i64 = 0;
        while (index > 0) {
            result += self.tree[index];
            index -= index & (~index +% 1);
        }
        return @intCast(result);
    }
    fn findByOrder(self: *const Fenwick, rank_in: usize) usize {
        var rank = rank_in;
        var index: usize = 0;
        var bit: usize = 1;
        while ((bit << 1) < self.tree.len) bit <<= 1;
        while (bit != 0) : (bit >>= 1) {
            const next = index + bit;
            if (next < self.tree.len and @as(usize, @intCast(self.tree[next])) <= rank) {
                index = next;
                rank -= @intCast(self.tree[next]);
            }
        }
        return index;
    }
};

fn appendLehmerPermutation(out: *std.ArrayList(u8), alloc: A, permutation: []const usize) !bool {
    var fenwick = try Fenwick.init(alloc, permutation.len);
    defer fenwick.deinit(alloc);
    const seen = try alloc.alloc(bool, permutation.len);
    defer alloc.free(seen);
    @memset(seen, false);
    for (permutation) |value| {
        if (value >= permutation.len or seen[value]) return false;
        seen[value] = true;
        try appendVarint(out, alloc, fenwick.sumLessThan(value));
        fenwick.add(value, -1);
    }
    return true;
}

fn readLehmerPermutation(alloc: A, data: []const u8, pos: *usize, count: usize, permutation: *[]usize) !bool {
    var fenwick = try Fenwick.init(alloc, count);
    defer fenwick.deinit(alloc);
    permutation.* = try alloc.alloc(usize, count);
    @memset(permutation.*, 0);
    var i: usize = 0;
    while (i < count) : (i += 1) {
        var rank64: u64 = 0;
        if (!readVarint(data, pos, &rank64) or rank64 >= count - i) return false;
        const value = fenwick.findByOrder(@intCast(rank64));
        permutation.*[i] = value;
        fenwick.add(value, -1);
    }
    return true;
}

// ---------------------------------------------------------------------------
// block parsing (260-415)
// ---------------------------------------------------------------------------

fn partitionTailLines(lines: []const LineRef, r0: *usize, r1: *usize, r2: *usize) bool {
    r0.* = 0;
    r1.* = 0;
    r2.* = 0;
    var pos: usize = 0;
    for (lines) |line| {
        if (pos < kEncodedRegime1Start) r0.* += 1 else if (pos < kEncodedRegime2Start) r1.* += 1 else r2.* += 1;
        if (line.len > kEncodedTailLen - pos) return false;
        pos += line.len;
    }
    return pos == kEncodedTailLen;
}

fn guessLastHugeBlockEnd(data: []const u8, lines: []const LineRef, start: usize, limit: usize) usize {
    var end = @min(start + 7, limit);
    if (end < limit and isAsciiNumberLine(data, lines[end])) {
        end += 1;
        if (end < limit and !isExactD99Line(data, lines[end])) {
            const be = bodyEnd(data, lines[end]);
            if (be > lines[end].start and be - lines[end].start <= 16 and data[lines[end].start] >= 0x80) {
                end += 1;
            }
        }
    }
    return end;
}

const ParsedTail = struct {
    prelude: []LineRef,
    blocks: []TailBlock,
    suffix: []LineRef,

    fn deinit(self: *ParsedTail, alloc: A) void {
        for (self.blocks) |b| alloc.free(b.sort_key);
        alloc.free(self.blocks);
        // prelude/suffix are sub-slices of r1_lines, not owned here.
    }
};

/// ParseTailBlocks (291-322). `r1_lines` is owned by the caller and must outlive
/// the returned ParsedTail (blocks reference sub-slices of it).
fn parseTailBlocks(alloc: A, data: []const u8, r1_lines: []LineRef) !?ParsedTail {
    var starts: std.ArrayList(usize) = .empty;
    defer starts.deinit(alloc);
    for (r1_lines, 0..) |line, i| {
        if (isExactD99Line(data, line)) try starts.append(alloc, i);
    }
    if (starts.items.len == 0) return null;

    const prelude = r1_lines[0..starts.items[0]];
    var blocks: std.ArrayList(TailBlock) = .empty;
    errdefer {
        for (blocks.items) |b| alloc.free(b.sort_key);
        blocks.deinit(alloc);
    }
    try blocks.ensureTotalCapacity(alloc, starts.items.len);

    var cursor = starts.items[0];
    for (starts.items, 0..) |start, block_index| {
        if (start != cursor) return null;
        const next_start = if (block_index + 1 < starts.items.len) starts.items[block_index + 1] else r1_lines.len;
        const end = if (next_start - start <= 16) next_start else guessLastHugeBlockEnd(data, r1_lines, start, next_start);
        const blk_lines = r1_lines[start..end];
        // sort_key = concat of full lines[2..]
        var key: std.ArrayList(u8) = .empty;
        errdefer key.deinit(alloc);
        var i: usize = 2;
        while (i < blk_lines.len) : (i += 1) {
            try key.appendSlice(alloc, data[blk_lines[i].start .. blk_lines[i].start + blk_lines[i].len]);
        }
        try blocks.append(alloc, .{
            .original_index = block_index,
            .lines = blk_lines,
            .sort_key = try key.toOwnedSlice(alloc),
        });
        cursor = end;
    }
    const suffix = r1_lines[cursor..];
    return ParsedTail{
        .prelude = prelude,
        .blocks = try blocks.toOwnedSlice(alloc),
        .suffix = suffix,
    };
}

fn computeD86aOrder(alloc: A, data: []const u8, blocks: []const TailBlock, order: *[]usize) !bool {
    const d86a = try alloc.alloc(u64, blocks.len);
    defer alloc.free(d86a);
    for (blocks, 0..) |b, i| {
        if (!parseFirstD86a(data, b, &d86a[i])) return false;
    }
    order.* = try alloc.alloc(usize, blocks.len);
    for (order.*, 0..) |*o, i| o.* = i;
    const Ctx = struct { d: []const u64 };
    std.sort.block(usize, order.*, Ctx{ .d = d86a }, struct {
        fn lt(ctx: Ctx, a: usize, b: usize) bool {
            return if (ctx.d[a] != ctx.d[b]) ctx.d[a] < ctx.d[b] else a < b;
        }
    }.lt);
    return true;
}

// ---------------------------------------------------------------------------
// Public: reorder (encode side) and restore (decode side), in-memory.
// ---------------------------------------------------------------------------

pub const ReorderResult = struct {
    /// The reordered stream WITH the appended side + footer (== `out.cmix.temp`).
    stream: []u8,
    /// The compact side channel (== `.r1_payload_lex_side`).
    side: []u8,
};

/// ReorderEncodedTailFile (419-472), in memory. `input` is the post-WRT stream.
pub fn reorderEncodedTail(alloc: A, input: []const u8) Error!ReorderResult {
    if (input.len != kExpectedStreamLen) return error.R1Refused;

    const lines = try splitLines(alloc, kEncodedTailStart, kEncodedTailLen, input);
    defer alloc.free(lines);
    var r0c: usize = 0;
    var r1c: usize = 0;
    var r2c: usize = 0;
    if (!partitionTailLines(lines, &r0c, &r1c, &r2c)) return error.R1Corrupt;
    const r0 = lines[0..r0c];
    const r1 = lines[r0c .. r0c + r1c];
    const r2 = lines[r0c + r1c ..];

    var parsed = (try parseTailBlocks(alloc, input, r1)) orelse return error.R1Corrupt;
    defer parsed.deinit(alloc);
    const blocks = parsed.blocks;

    // sorted_indices: stable sort of [0..n) by (sort_key, original_index).
    const sorted_indices = try alloc.alloc(usize, blocks.len);
    defer alloc.free(sorted_indices);
    for (sorted_indices, 0..) |*s, i| s.* = i;
    std.sort.block(usize, sorted_indices, blocks, struct {
        fn lt(bl: []TailBlock, a: usize, b: usize) bool {
            const ord = std.mem.order(u8, bl[a].sort_key, bl[b].sort_key);
            return if (ord != .eq) ord == .lt else bl[a].original_index < bl[b].original_index;
        }
    }.lt);

    // side
    var side_al: std.ArrayList(u8) = .empty;
    errdefer side_al.deinit(alloc);
    if (!try makeSide(alloc, &side_al, input, blocks, sorted_indices, r0.len, r1.len, r2.len, parsed.prelude.len, parsed.suffix.len))
        return error.R1Corrupt;
    const side = try side_al.toOwnedSlice(alloc);
    errdefer alloc.free(side);

    // output = head ++ r0 ++ prelude ++ (blocks in sorted order) ++ suffix ++ r2
    var out: std.ArrayList(u8) = .empty;
    errdefer out.deinit(alloc);
    try out.ensureTotalCapacity(alloc, input.len + side.len + kFooterMagic.len + 8);
    try out.appendSlice(alloc, input[0..kEncodedTailStart]);
    try appendLines(&out, alloc, input, r0);
    try appendLines(&out, alloc, input, parsed.prelude);
    for (sorted_indices) |idx| try appendLines(&out, alloc, input, blocks[idx].lines);
    try appendLines(&out, alloc, input, parsed.suffix);
    try appendLines(&out, alloc, input, r2);
    if (out.items.len != input.len) return error.R1Corrupt;
    try out.appendSlice(alloc, side);
    try out.appendSlice(alloc, &kFooterMagic);
    try appendU64LE(&out, alloc, side.len);

    return .{ .stream = try out.toOwnedSlice(alloc), .side = side };
}

fn makeSide(
    alloc: A,
    side: *std.ArrayList(u8),
    data: []const u8,
    blocks: []const TailBlock,
    sorted_indices: []const usize,
    r0_count: usize,
    r1_count: usize,
    r2_count: usize,
    prelude_count: usize,
    suffix_count: usize,
) !bool {
    // MakeSide (338-375)
    const d86a = try alloc.alloc(u64, sorted_indices.len);
    defer alloc.free(d86a);
    for (sorted_indices, 0..) |index, sorted_pos| {
        if (index >= blocks.len or !parseFirstD86a(data, blocks[index], &d86a[sorted_pos])) return false;
    }
    const d86_order = try alloc.alloc(usize, sorted_indices.len);
    defer alloc.free(d86_order);
    for (d86_order, 0..) |*o, i| o.* = i;
    const Ctx = struct { d: []const u64 };
    std.sort.block(usize, d86_order, Ctx{ .d = d86a }, struct {
        fn lt(ctx: Ctx, a: usize, b: usize) bool {
            return if (ctx.d[a] != ctx.d[b]) ctx.d[a] < ctx.d[b] else a < b;
        }
    }.lt);

    const original_by_d86a = try alloc.alloc(usize, d86_order.len);
    defer alloc.free(original_by_d86a);
    for (d86_order, 0..) |sorted_pos, i| original_by_d86a[i] = sorted_indices[sorted_pos];

    try side.appendSlice(alloc, &kSideMagicD86);
    try appendVarint(side, alloc, kEncodedTailLen);
    try appendVarint(side, alloc, r0_count);
    try appendVarint(side, alloc, r1_count);
    try appendVarint(side, alloc, r2_count);
    try appendVarint(side, alloc, prelude_count);
    try appendVarint(side, alloc, suffix_count);
    try appendVarint(side, alloc, original_by_d86a.len);
    return appendLehmerPermutation(side, alloc, original_by_d86a);
}

const SideMeta = struct {
    r0_count: usize = 0,
    r1_count: usize = 0,
    r2_count: usize = 0,
    prelude_count: usize = 0,
    suffix_count: usize = 0,
    sorted_to_original: []usize = &.{},
};

fn parseSide(alloc: A, side: []const u8, meta: *SideMeta) !bool {
    if (!hasBytesAt(side, 0, &kSideMagicD86)) return false;
    var pos: usize = kSideMagicD86.len;
    var tail_len: u64 = 0;
    var r0: u64 = 0;
    var r1: u64 = 0;
    var r2: u64 = 0;
    var prelude: u64 = 0;
    var suffix: u64 = 0;
    var count: u64 = 0;
    if (!readVarint(side, &pos, &tail_len) or tail_len != kEncodedTailLen or
        !readVarint(side, &pos, &r0) or !readVarint(side, &pos, &r1) or
        !readVarint(side, &pos, &r2) or !readVarint(side, &pos, &prelude) or
        !readVarint(side, &pos, &suffix) or !readVarint(side, &pos, &count))
    {
        return false;
    }
    meta.r0_count = @intCast(r0);
    meta.r1_count = @intCast(r1);
    meta.r2_count = @intCast(r2);
    meta.prelude_count = @intCast(prelude);
    meta.suffix_count = @intCast(suffix);
    if (!try readLehmerPermutation(alloc, side, &pos, @intCast(count), &meta.sorted_to_original)) return false;
    return pos == side.len;
}

/// Restore entry point, dispatching on the side footer magic:
///   `R1ORDFTR` -> v1 R1ORD3 path (order asset unused),
///   `R1ORDFT4` -> v2 R1ORD4 path (raw order asset REQUIRED; `error.R1OrderMissing`
///                 if null) — only compiled under -Dr1v2.
/// Flags-ON binaries therefore still decode v1 archives (needed for A/Bs and for
/// decode-verifying the official C1-L bhm with a v2 binary). Flags-OFF builds
/// compile the v2 path away; an FT4 footer then falls through to the v1 parser,
/// which rejects it (no FTR magic -> error.R1Corrupt).
pub fn restoreEncodedTail(alloc: A, stream_with_footer: []const u8, order_bytes: ?[]const u8) Error![]u8 {
    if (comptime build_options.r1v2) {
        if (stream_with_footer.len >= kFooterMagicV2.len + 8 and
            hasBytesAt(stream_with_footer, stream_with_footer.len - kFooterMagicV2.len - 8, &kFooterMagicV2))
        {
            const order = order_bytes orelse return error.R1OrderMissing;
            return restoreEncodedTailV2(alloc, stream_with_footer, order);
        }
    }
    if (comptime TEST_NO_R1) {
        // ⛔ TEST ONLY. A -Dtest-no-r1 encode emits a stream with NO r1 footer;
        // recognise that and pass it through unchanged instead of R1Corrupt.
        if (stream_with_footer.len < kFooterMagic.len + 8 or
            !hasBytesAt(stream_with_footer, stream_with_footer.len - kFooterMagic.len - 8, &kFooterMagic))
            return alloc.dupe(u8, stream_with_footer);
    }
    return restoreEncodedTailV1(alloc, stream_with_footer);
}

/// ExtractSideFromFile (474-495) + RestoreEncodedTailFile (497-568), in memory.
/// `stream_with_footer` is the post-r1 stream (as produced by reorderEncodedTail).
/// Returns the restored pre-r1 stream (== the original WRT output).
fn restoreEncodedTailV1(alloc: A, stream_with_footer: []const u8) Error![]u8 {
    // ExtractSideFromFile
    if (stream_with_footer.len < kFooterMagic.len + 8) return error.R1Corrupt;
    const footer_pos = stream_with_footer.len - kFooterMagic.len - 8;
    if (!hasBytesAt(stream_with_footer, footer_pos, &kFooterMagic)) return error.R1Corrupt;
    var side_len64: u64 = 0;
    if (!readU64LE(stream_with_footer, footer_pos + kFooterMagic.len, &side_len64) or side_len64 > footer_pos)
        return error.R1Corrupt;
    const side_len: usize = @intCast(side_len64);
    const side_pos = footer_pos - side_len;
    const side = stream_with_footer[side_pos..footer_pos];
    const input = stream_with_footer[0..side_pos];

    // RestoreEncodedTailFile
    if (input.len != kExpectedStreamLen) return error.R1Refused;
    var meta = SideMeta{};
    if (!try parseSide(alloc, side, &meta)) return error.R1Corrupt;
    defer alloc.free(meta.sorted_to_original);

    const lines = try splitLines(alloc, kEncodedTailStart, kEncodedTailLen, input);
    defer alloc.free(lines);
    if (lines.len != meta.r0_count + meta.r1_count + meta.r2_count) return error.R1Corrupt;

    const r0 = lines[0..meta.r0_count];
    const r1 = lines[meta.r0_count .. meta.r0_count + meta.r1_count];
    const r2 = lines[meta.r0_count + meta.r1_count ..];
    if (meta.prelude_count + meta.suffix_count > r1.len) return error.R1Corrupt;
    const prelude = r1[0..meta.prelude_count];
    const suffix = r1[r1.len - meta.suffix_count ..];
    const middle = r1[meta.prelude_count .. r1.len - meta.suffix_count];

    var parsed = (try parseTailBlocks(alloc, input, middle)) orelse return error.R1Corrupt;
    defer parsed.deinit(alloc);
    if (parsed.prelude.len != 0 or parsed.suffix.len != 0 or
        parsed.blocks.len != meta.sorted_to_original.len) return error.R1Corrupt;
    const sorted_blocks = parsed.blocks;

    var d86_order: []usize = &.{};
    if (!try computeD86aOrder(alloc, input, sorted_blocks, &d86_order)) return error.R1Corrupt;
    defer alloc.free(d86_order);

    const original_blocks = try alloc.alloc(?*const TailBlock, sorted_blocks.len);
    defer alloc.free(original_blocks);
    @memset(original_blocks, null);
    for (d86_order, 0..) |sorted_pos, d86_pos| {
        const original_index = meta.sorted_to_original[d86_pos];
        if (original_index >= original_blocks.len or original_blocks[original_index] != null) return error.R1Corrupt;
        original_blocks[original_index] = &sorted_blocks[sorted_pos];
    }

    var out: std.ArrayList(u8) = .empty;
    errdefer out.deinit(alloc);
    try out.ensureTotalCapacity(alloc, input.len);
    try out.appendSlice(alloc, input[0..kEncodedTailStart]);
    try appendLines(&out, alloc, input, r0);
    try appendLines(&out, alloc, input, prelude);
    for (original_blocks) |maybe_block| {
        const block = maybe_block orelse return error.R1Corrupt;
        try appendLines(&out, alloc, input, block.lines);
    }
    try appendLines(&out, alloc, input, suffix);
    try appendLines(&out, alloc, input, r2);
    if (out.items.len != input.len) return error.R1Corrupt;
    return out.toOwnedSlice(alloc);
}

// ===========================================================================
// R1ORD4 v2 "derived side" codec (r1_reorder_v2.h, refs "v2:N"). All internal
// helpers take an ARENA allocator for scratch (nothing is freed piecemeal; the
// public entry points own the arena) and mirror the C++ bool-returning
// functions as `!?T` / `!bool` (null/false == the C++ `return false`).
// ===========================================================================

/// v2:74-96 — signed page-id delta: the LAST line of the block whose body is
/// exactly `N[-]<digits>`. Keeps the last match, like the C++ (the real N-line
/// is the final line of every regime-1 block; an earlier contributor line that
/// happens to match is overwritten).
fn parseDelta(data: []const u8, block: TailBlock, delta: *i64) bool {
    var found = false;
    for (block.lines) |line| {
        const end = bodyEnd(data, line);
        var p = line.start;
        if (p >= end or data[p] != 'N') continue;
        p += 1;
        var neg = false;
        if (p < end and data[p] == '-') {
            neg = true;
            p += 1;
        }
        if (p >= end) continue;
        var v: u64 = 0;
        var ok = true;
        var i = p;
        while (i < end) : (i += 1) {
            if (data[i] < '0' or data[i] > '9') {
                ok = false;
                break;
            }
            // The C++ accumulates uint64_t with no overflow check (real deltas
            // are tiny page-id steps); wrap identically rather than trap.
            v = v *% 10 +% @as(u64, data[i] - '0');
        }
        if (!ok) continue;
        const sv: i64 = @bitCast(v);
        delta.* = if (neg) 0 -% sv else sv;
        found = true; // keep last match
    }
    return found;
}

/// v2:98-101
const BlockFields = struct { delta: i64 = 0, d86a: u64 = 0 };

/// v2:103-111 — every block must yield BOTH its D86a value (line 1) and its
/// signed N-line delta, in the given (original or sorted-stream) block order.
fn parseAllFields(arena: A, data: []const u8, blocks: []const TailBlock) !?[]BlockFields {
    const fields = try arena.alloc(BlockFields, blocks.len);
    for (blocks, 0..) |b, i| {
        if (!parseFirstD86a(data, b, &fields[i].d86a)) return null;
        if (!parseDelta(data, b, &fields[i].delta)) return null;
    }
    return fields;
}

/// v2:115-124 ReadOrderFile: one i64 per line, empty lines skipped, must be
/// non-empty. The C++ reads with std::getline + std::stoll; the shipped order
/// asset is clean `[-]<digits>\n` lines, on which this strict parse is
/// value-identical (a nonconforming line is a loud null here vs an uncaught
/// throw there — both refuse). A lone trailing '\r' is tolerated the way
/// stoll's digit scan stops before it.
fn readOrderLines(arena: A, order_bytes: []const u8) !?[]i64 {
    var out: std.ArrayList(i64) = .empty;
    var it = std.mem.splitScalar(u8, order_bytes, '\n');
    while (it.next()) |line_raw| {
        if (line_raw.len == 0) continue; // C++ `line.empty()` check
        var line = line_raw;
        if (line[line.len - 1] == '\r') line = line[0 .. line.len - 1];
        if (line.len == 0) return null; // "\r" alone: stoll would throw
        const v = std.fmt.parseInt(i64, line, 10) catch return null;
        try out.append(arena, v);
    }
    if (out.items.len == 0) return null;
    return out.items;
}

/// Exact-match binary search in a strictly-ascending array (the C++ builds a
/// std::map<int64_t,size_t> rank_of; identical lookups, no allocation).
fn indexOfSorted(sorted: []const i64, needle: i64) ?usize {
    var lo: usize = 0;
    var hi: usize = sorted.len;
    while (lo < hi) {
        const mid = lo + (hi - lo) / 2;
        if (sorted[mid] < needle) lo = mid + 1 else hi = mid;
    }
    if (lo < sorted.len and sorted[lo] == needle) return lo;
    return null;
}

/// v2:129-156 PositionsToIds — position -> relative id for the whole reordered
/// sequence, from the ascending id table + used bitmap (rank space) + the order
/// file: the j-th smallest used order-value maps to the j-th smallest used
/// rank; unused ranks (ascending) fill the tail positions.
fn positionsToIds(
    arena: A,
    sorted_rel_ids: []const i64,
    used_bitmap: []const u8,
    n_articles: usize,
    order_lines: []const i64,
) !?[]i64 {
    var used: std.ArrayList(usize) = .empty;
    var unused: std.ArrayList(usize) = .empty;
    for (0..n_articles) |c1| {
        const bit = (used_bitmap[c1 >> 3] >> @intCast(c1 & 7)) & 1;
        if (bit != 0) try used.append(arena, c1) else try unused.append(arena, c1);
    }
    if (used.items.len != order_lines.len) {
        std.debug.print("r1 v2 positionsToIds: used ranks {d} != order lines {d}\n", .{ used.items.len, order_lines.len });
        return null;
    }

    // remap: j-th smallest used c2 -> used_ranks[j] (C++ std::map assignment
    // semantics: on duplicate c2 values the later j wins).
    const sorted_c2 = try arena.dupe(i64, order_lines);
    std.mem.sort(i64, sorted_c2, {}, std.sort.asc(i64));
    var remap = std.AutoHashMap(i64, usize).init(arena);
    try remap.ensureTotalCapacity(@intCast(sorted_c2.len));
    for (sorted_c2, 0..) |c2, j| remap.putAssumeCapacity(c2, used.items[j]);

    const pos_ids = try arena.alloc(i64, n_articles);
    var w: usize = 0;
    for (order_lines) |c2| {
        const rank = remap.get(c2) orelse return null;
        pos_ids[w] = sorted_rel_ids[rank];
        w += 1;
    }
    for (unused.items) |rank| {
        pos_ids[w] = sorted_rel_ids[rank];
        w += 1;
    }
    if (w != n_articles) return null;
    return pos_ids;
}

/// v2:158-196 ClassPlan, flattened: class c spans [offsets[c], offsets[c+1]) of
/// both index arrays. Canonical iteration order (the load-bearing spec): classes
/// by delta ASCENDING (C++ std::map key order); within a class, positions by
/// introduced-article id `pos_ids[k+1]` ascending with ascending-k ties (C++
/// ascending push + stable_sort), and blocks by D86a ascending with
/// ascending-index ties. Implemented as two single 3-key sorts — provably the
/// same total order.
const ClassPlan = struct {
    offsets: []usize,
    pos_by_id: []usize,
    blk_by_d86a: []usize,
};

fn buildClassPlan(arena: A, pos_ids: []const i64, fields: []const BlockFields) !?ClassPlan {
    const count = fields.len;
    if (pos_ids.len != count + 1) {
        std.debug.print("r1 v2 classPlan: pos_ids len {d} != count+1 {d}\n", .{ pos_ids.len, count + 1 });
        return null;
    }

    const pos_by_id = try arena.alloc(usize, count);
    for (pos_by_id, 0..) |*p, k| p.* = k;
    const PosCtx = struct { ids: []const i64 };
    std.sort.block(usize, pos_by_id, PosCtx{ .ids = pos_ids }, struct {
        fn lt(ctx: PosCtx, a: usize, b: usize) bool {
            const da = ctx.ids[a + 1] -% ctx.ids[a];
            const db = ctx.ids[b + 1] -% ctx.ids[b];
            if (da != db) return da < db;
            if (ctx.ids[a + 1] != ctx.ids[b + 1]) return ctx.ids[a + 1] < ctx.ids[b + 1];
            return a < b;
        }
    }.lt);

    const blk_by_d86a = try arena.alloc(usize, count);
    for (blk_by_d86a, 0..) |*p, i| p.* = i;
    const BlkCtx = struct { f: []const BlockFields };
    std.sort.block(usize, blk_by_d86a, BlkCtx{ .f = fields }, struct {
        fn lt(ctx: BlkCtx, a: usize, b: usize) bool {
            if (ctx.f[a].delta != ctx.f[b].delta) return ctx.f[a].delta < ctx.f[b].delta;
            if (ctx.f[a].d86a != ctx.f[b].d86a) return ctx.f[a].d86a < ctx.f[b].d86a;
            return a < b;
        }
    }.lt);

    // Class boundaries must AGREE between the two partitions: same ascending
    // delta sequence, same per-class sizes (C++ map-size + per-key checks,
    // v2:177-183).
    var offsets: std.ArrayList(usize) = .empty;
    try offsets.append(arena, 0);
    var i: usize = 0;
    var j: usize = 0;
    while (i < count or j < count) {
        if (i >= count or j >= count) {
            std.debug.print("r1 v2 classPlan: partitions exhaust unevenly (i {d} j {d} count {d})\n", .{ i, j, count });
            return null;
        }
        const pd = pos_ids[pos_by_id[i] + 1] -% pos_ids[pos_by_id[i]];
        const bd = fields[blk_by_d86a[j]].delta;
        if (pd != bd) {
            std.debug.print("r1 v2 classPlan: delta mismatch pos {d} vs blk {d} (i {d} j {d})\n", .{ pd, bd, i, j });
            return null;
        }
        var i_end = i + 1;
        while (i_end < count and (pos_ids[pos_by_id[i_end] + 1] -% pos_ids[pos_by_id[i_end]]) == pd) i_end += 1;
        var j_end = j + 1;
        while (j_end < count and fields[blk_by_d86a[j_end]].delta == bd) j_end += 1;
        if (i_end - i != j_end - j) {
            std.debug.print("r1 v2 classPlan: class size mismatch delta {d}: pos {d} vs blk {d}\n", .{ pd, i_end - i, j_end - j });
            return null;
        }
        try offsets.append(arena, i_end);
        i = i_end;
        j = j_end;
    }
    return ClassPlan{
        .offsets = offsets.items,
        .pos_by_id = pos_by_id,
        .blk_by_d86a = blk_by_d86a,
    };
}

/// v2:202-266 MakeSideV2. `fields` are in ORIGINAL block order; truth: position
/// k <-> block k. Returns the full v2 side blob (arena-owned), or null on any
/// assumption break (duplicate ids, order/article count mismatch, class
/// partition disagreement).
fn makeSideV2(
    arena: A,
    fields: []const BlockFields,
    region_start: usize,
    region_len: usize,
    order_lines: []const i64,
) !?[]u8 {
    const count = fields.len;
    const n_articles = count + 1;
    const n_order = order_lines.len;
    if (n_order >= n_articles) {
        std.debug.print("r1 v2 makeSideV2: n_order {d} >= n_articles {d}\n", .{ n_order, n_articles });
        return null;
    }

    // relative ids in original (= reordered-article) order (v2:212-214)
    const ids = try arena.alloc(i64, n_articles);
    ids[0] = 0;
    for (fields, 0..) |f, k| ids[k + 1] = ids[k] +% f.delta;

    // ranks: ids must be distinct (v2:217-223)
    const sorted_ids = try arena.dupe(i64, ids);
    std.mem.sort(i64, sorted_ids, {}, std.sort.asc(i64));
    for (1..n_articles) |i| {
        if (sorted_ids[i] == sorted_ids[i - 1]) {
            std.debug.print("r1 v2 makeSideV2: duplicate id {d} at sorted index {d}\n", .{ sorted_ids[i], i });
            return null;
        }
    }

    var side: std.ArrayList(u8) = .empty;
    try side.appendSlice(arena, &kSideMagicV2);
    try appendVarint(&side, arena, region_start);
    try appendVarint(&side, arena, region_len);
    try appendVarint(&side, arena, count);
    try appendVarint(&side, arena, n_articles);
    try appendVarint(&side, arena, n_order);

    // used bitmap over ranks / c1 space (v2:234-239): bit set iff the article
    // appears in the order-file part of the reordered sequence (positions
    // 0..n_order).
    const bitmap = try arena.alloc(u8, (n_articles + 7) / 8);
    @memset(bitmap, 0);
    for (0..n_order) |k| {
        const rank = indexOfSorted(sorted_ids, ids[k]) orelse return null;
        bitmap[rank >> 3] |= @as(u8, 1) << @intCast(rank & 7);
    }
    try side.appendSlice(arena, bitmap);

    // id-gap table (v2:243-245): gaps between consecutive ascending ids (the
    // minimum itself is implicit — everything downstream is shift-invariant).
    for (1..n_articles) |i| {
        try appendVarint(&side, arena, @bitCast(sorted_ids[i] -% sorted_ids[i - 1]));
    }

    // class residual (v2:248-264) — truth: position k owns block k. Per class
    // (canonical order), each position emits the Fenwick rank of its true
    // block within the class's remaining D86a-ordered members.
    const plan = (try buildClassPlan(arena, ids, fields)) orelse {
        std.debug.print("r1 v2 makeSideV2: buildClassPlan refused\n", .{});
        return null;
    };
    const n_classes = plan.offsets.len - 1;
    const rank_of_block = try arena.alloc(usize, count);
    const class_of_block = try arena.alloc(usize, count);
    for (0..n_classes) |c| {
        const off = plan.offsets[c];
        for (off..plan.offsets[c + 1]) |jj| {
            rank_of_block[plan.blk_by_d86a[jj]] = jj - off;
            class_of_block[plan.blk_by_d86a[jj]] = c;
        }
    }
    for (0..n_classes) |c| {
        const off = plan.offsets[c];
        const n = plan.offsets[c + 1] - off;
        var fen = try Fenwick.init(arena, n);
        for (0..n) |t| {
            const k = plan.pos_by_id[off + t]; // position; true block index == k
            if (class_of_block[k] != c) {
                std.debug.print("r1 v2 makeSideV2: truth violated at class {d} t {d} (position {d} in class {d})\n", .{ c, t, k, class_of_block[k] });
                return null;
            } // v2:259
            const r = rank_of_block[k];
            try appendVarint(&side, arena, fen.sumLessThan(r));
            fen.add(r, -1);
        }
    }
    return side.items;
}

/// v2:270-275
const SideV2 = struct {
    region_start: usize = 0,
    region_len: usize = 0,
    count: usize = 0,
    n_articles: usize = 0,
    n_order: usize = 0,
    bitmap: []const u8 = &.{}, // slice into the side blob (no copy)
    sorted_rel_ids: []i64 = &.{}, // arena-owned, ascending, [0]=0
    residual_pos: usize = 0, // offset of the residual section in the blob
};

/// v2:277-302 ParseSideV2Header.
fn parseSideV2Header(arena: A, side: []const u8, m: *SideV2) !bool {
    if (!hasBytesAt(side, 0, &kSideMagicV2)) return false;
    var pos: usize = kSideMagicV2.len;
    var rs: u64 = 0;
    var rl: u64 = 0;
    var cnt: u64 = 0;
    var na: u64 = 0;
    var no: u64 = 0;
    if (!readVarint(side, &pos, &rs) or !readVarint(side, &pos, &rl) or
        !readVarint(side, &pos, &cnt) or !readVarint(side, &pos, &na) or
        !readVarint(side, &pos, &no))
    {
        return false;
    }
    m.region_start = std.math.cast(usize, rs) orelse return false;
    m.region_len = std.math.cast(usize, rl) orelse return false;
    m.count = std.math.cast(usize, cnt) orelse return false;
    m.n_articles = std.math.cast(usize, na) orelse return false;
    m.n_order = std.math.cast(usize, no) orelse return false;
    // (n_articles+7)/8 without the usize overflow on absurd headers.
    const bm = m.n_articles / 8 + @intFromBool(m.n_articles % 8 != 0);
    if (side.len - pos < bm) return false;
    m.bitmap = side[pos .. pos + bm];
    pos += bm;
    // n_articles-1 gap varints of >=1 byte each must fit: early-fail on a
    // corrupt header before allocating ids (the C++ fails at some ReadVarint).
    if (m.n_articles > 0 and m.n_articles - 1 > side.len - pos) return false;
    var ids: std.ArrayList(i64) = .empty;
    try ids.ensureTotalCapacity(arena, @max(m.n_articles, 1));
    try ids.append(arena, 0);
    var cur: i64 = 0;
    var i: usize = 1;
    while (i < m.n_articles) : (i += 1) {
        var g: u64 = 0;
        if (!readVarint(side, &pos, &g)) return false;
        cur = cur +% @as(i64, @bitCast(g));
        try ids.append(arena, cur);
    }
    m.sorted_rel_ids = ids.items;
    m.residual_pos = pos;
    return true;
}

/// v2:307-337 SolveAssignment — restore the ORIGINAL block order from the
/// side + the order file. `fields` are in SORTED-STREAM order. Returns, for
/// each original position k, the index (sorted-stream space) of its block.
fn solveAssignment(
    arena: A,
    side: []const u8,
    meta: *const SideV2,
    fields: []const BlockFields,
    order_lines: []const i64,
) !?[]usize {
    const pos_ids = (try positionsToIds(
        arena,
        meta.sorted_rel_ids,
        meta.bitmap,
        meta.n_articles,
        order_lines,
    )) orelse return null;
    const plan = (try buildClassPlan(arena, pos_ids, fields)) orelse return null;

    const block_at_position = try arena.alloc(usize, meta.count);
    @memset(block_at_position, std.math.maxInt(usize));
    var pos = meta.residual_pos;
    const n_classes = plan.offsets.len - 1;
    for (0..n_classes) |c| {
        const off = plan.offsets[c];
        const n = plan.offsets[c + 1] - off;
        var fen = try Fenwick.init(arena, n);
        for (0..n) |t| {
            var r64: u64 = 0;
            if (!readVarint(side, &pos, &r64) or r64 >= n - t) return null;
            const j = fen.findByOrder(@intCast(r64));
            fen.add(j, -1);
            const k = plan.pos_by_id[off + t];
            if (k >= block_at_position.len) return null;
            block_at_position[k] = plan.blk_by_d86a[off + j];
        }
    }
    if (pos != side.len) return null;
    for (block_at_position) |v| {
        if (v == std.math.maxInt(usize)) return null;
    }
    return block_at_position;
}

// ---------------------------------------------------------------------------
// Marker-based regime-1 detection (fork r1_reorder_transform.h:262-342, refs
// "mk:N"). The v2 scheme is defined AGAINST this region, NOT against v1's
// fixed-offset parseTailBlocks parse: the two block sets differ on the real
// stream (v1 includes the trailing D99 transition block that the marker run
// excludes — the fork-vs-zmix mains diverging at byte 557,242,677, reanchor
// memo), so the v2 sorted region is NOT byte-identical to v1's. The C++
// fixtures (db52476c… / a3632c91…) and the banked −170,309 pair are
// marker-derived; byte parity with them requires this exact detection.
// ---------------------------------------------------------------------------

/// mk:52 — the exact 4-byte D99 marker LINE (note: includes the '\n', unlike
/// v1's 3-byte body constant).
const kD99LineFull = [_]u8{ 0xDF, 0x99, 'N', '\n' };
// mk:56 — regular-block byte-length window (intrinsic regime-1 article gate).
const kMinRegularLen: usize = 20;
const kMaxRegularLen: usize = 400;

/// mk:266-278 FindD99Starts: all positions where an exact D99 line starts at a
/// line boundary (p == 0 or data[p-1] == '\n').
fn findD99Starts(arena: A, data: []const u8) ![]usize {
    var starts: std.ArrayList(usize) = .empty;
    if (data.len < kD99LineFull.len) return starts.items;
    var i: usize = 0;
    while (i + kD99LineFull.len <= data.len) : (i += 1) {
        if (data[i] == kD99LineFull[0] and data[i + 1] == kD99LineFull[1] and
            data[i + 2] == kD99LineFull[2] and data[i + 3] == kD99LineFull[3] and
            (i == 0 or data[i - 1] == '\n'))
        {
            try starts.append(arena, i);
        }
    }
    return starts.items;
}

/// mk:282-302 BlockIsRegular: [start,end) is a "regular" regime-1 article —
/// byte length in [20,400] and its 2nd line is a valid D86a line
/// {0xDF,0x86,'N'}<digits>.
fn blockIsRegular(data: []const u8, start: usize, end: usize) bool {
    if (end <= start) return false;
    const len = end - start;
    if (len < kMinRegularLen or len > kMaxRegularLen) return false;
    // Skip line 0 (the D99 line) -> start of line 1.
    var l1 = start;
    while (l1 < end and data[l1] != '\n') l1 += 1;
    l1 += 1; // past the '\n'
    if (l1 >= end) return false;
    var l1end = l1;
    while (l1end < end and data[l1end] != '\n') l1end += 1; // exclusive of '\n'
    if (l1end - l1 < kD86Prefix.len + 1) return false; // need >=1 digit
    if (!std.mem.eql(u8, data[l1 .. l1 + kD86Prefix.len], &kD86Prefix)) return false;
    var i = l1 + kD86Prefix.len;
    while (i < l1end) : (i += 1) {
        if (data[i] < '0' or data[i] > '9') return false;
    }
    return true;
}

const DetectedRegion = struct {
    start: usize,
    end: usize,
    blocks: []TailBlock, // file order; lines/sort_key arena-owned
};

/// mk:306-342 DetectRegion: the regime-1 region = the longest contiguous run
/// of regular blocks (ties: earliest), each block spanning one D99 start to
/// the next; blocks materialised in file order with sort_key = concat of the
/// full lines[2..]. Null == the C++ `return false` (no region).
fn detectRegionV2(arena: A, data: []const u8) !?DetectedRegion {
    const starts = try findD99Starts(arena, data);
    if (starts.len < 2) return null;
    const m = starts.len;
    var best_len: usize = 0;
    var best_first: usize = 0;
    var cur_len: usize = 0;
    var cur_first: usize = 0;
    var i: usize = 0;
    while (i + 1 < m) : (i += 1) {
        if (blockIsRegular(data, starts[i], starts[i + 1])) {
            if (cur_len == 0) cur_first = i;
            cur_len += 1;
            if (cur_len > best_len) { // strict '>' keeps the earliest longest run
                best_len = cur_len;
                best_first = cur_first;
            }
        } else {
            cur_len = 0;
        }
    }
    if (best_len == 0) return null;
    const run_first = best_first;
    const run_last = best_first + best_len - 1; // block index
    const blocks = try arena.alloc(TailBlock, best_len);
    for (blocks, 0..) |*b, bi| {
        const si = run_first + bi;
        const lines = try splitLines(arena, starts[si], starts[si + 1] - starts[si], data);
        var key: std.ArrayList(u8) = .empty;
        var k: usize = 2;
        while (k < lines.len) : (k += 1) {
            try key.appendSlice(arena, data[lines[k].start .. lines[k].start + lines[k].len]);
        }
        b.* = .{ .original_index = bi, .lines = lines, .sort_key = key.items };
    }
    return DetectedRegion{
        .start = starts[run_first],
        .end = starts[run_last + 1],
        .blocks = blocks,
    };
}

/// v2 restore core (RestoreFromTransformedData, v2:343-385): peel the R1ORDFT4
/// footer, re-DETECT the region from content (deterministic, identical on the
/// sorted stream), cross-check it against the side header, solve the
/// assignment, reassemble.
fn restoreEncodedTailV2(alloc: A, stream_with_footer: []const u8, order_bytes: []const u8) Error![]u8 {
    if (stream_with_footer.len < kFooterMagicV2.len + 8) return error.R1Corrupt;
    const footer_pos = stream_with_footer.len - kFooterMagicV2.len - 8;
    if (!hasBytesAt(stream_with_footer, footer_pos, &kFooterMagicV2)) return error.R1Corrupt;
    var side_len64: u64 = 0;
    if (!readU64LE(stream_with_footer, footer_pos + kFooterMagicV2.len, &side_len64) or side_len64 > footer_pos)
        return error.R1Corrupt;
    const side_pos = footer_pos - @as(usize, @intCast(side_len64));
    const side = stream_with_footer[side_pos..footer_pos];
    const stream = stream_with_footer[0..side_pos];
    // House engage-guard, same as v1: the transform only ever applies to the
    // exact canonical post-WRT stream.
    if (stream.len != kExpectedStreamLen) return error.R1Refused;

    var arena_state = std.heap.ArenaAllocator.init(alloc);
    defer arena_state.deinit();
    const arena = arena_state.allocator();

    var meta: SideV2 = .{};
    if (!try parseSideV2Header(arena, side, &meta)) return error.R1Corrupt;

    // Re-detect the region and cross-check the side header (v2:363-369).
    const region = (try detectRegionV2(arena, stream)) orelse return error.R1Corrupt;
    if (region.start != meta.region_start or
        region.end - region.start != meta.region_len or
        region.blocks.len != meta.count)
    {
        return error.R1Corrupt;
    }
    const region_end = region.end;
    const blocks = region.blocks; // SORTED-stream order

    const fields = (try parseAllFields(arena, stream, blocks)) orelse return error.R1Corrupt;
    const order_lines = (try readOrderLines(arena, order_bytes)) orelse return error.R1Corrupt;
    const block_at_position = (try solveAssignment(arena, side, &meta, fields, order_lines)) orelse
        return error.R1Corrupt;

    // Reassemble (v2:377-384): head ++ blocks in original order ++ tail. Each
    // block's lines are one contiguous byte span (splitLines tiling).
    const out = try alloc.alloc(u8, stream.len);
    errdefer alloc.free(out);
    @memcpy(out[0..meta.region_start], stream[0..meta.region_start]);
    var pos: usize = meta.region_start;
    for (block_at_position) |bi| {
        const b = blocks[bi];
        const first = b.lines[0].start;
        const last = b.lines[b.lines.len - 1];
        const blen = last.start + last.len - first;
        if (blen > region_end - pos) return error.R1Corrupt;
        @memcpy(out[pos..][0..blen], stream[first..][0..blen]);
        pos += blen;
    }
    if (pos != region_end) return error.R1Corrupt; // C++ restored.size()==stream.size()
    @memcpy(out[region_end..], stream[region_end..]);
    return out;
}

/// R1ORD4 v2 encode (ReorderEncodedTailFileV2, v2:391-446), in memory. The
/// regime-1 region comes from the MARKER-based DetectRegion (longest run of
/// regular blocks — the definition the v2 scheme, its fixtures and the banked
/// −170,309 pair are built on), NOT from v1's fixed-offset parse: the two
/// block sets differ on the real stream, so the v2 sorted region is NOT
/// byte-identical to v1's (divergence at byte 557,242,677 — a new substrate
/// either way, as the plan's tooling note says). The payload-lex sort itself
/// is the same (sort_key, original_index) ordering as v1. Appends the R1ORD4
/// derived side + R1ORDFT4 footer.
///
/// SELF-VERIFY: the assembled output is fully restored in memory and compared
/// against the input; ANY failure refuses to emit (error.R1SelfVerifyFailed,
/// printed loudly). An assumption break degrades to a hard abort in the first
/// minutes of -E — never a wrong stream (house R1LengthMismatch abort style).
pub fn reorderEncodedTailV2(alloc: A, input: []const u8, order_bytes: []const u8) Error!ReorderResult {
    if (input.len != kExpectedStreamLen) return error.R1Refused;

    var arena_state = std.heap.ArenaAllocator.init(alloc);
    defer arena_state.deinit();
    const arena = arena_state.allocator();

    // The C++ tool pass-throughs when no region is detected (the dict/order -c
    // compressions); zmix only ever calls this on the exact-length enwik9
    // stream, where a detection failure is an assumption break -> loud abort.
    const region = (try detectRegionV2(arena, input)) orelse {
        std.debug.print("r1 v2 encode: no regime-1 region detected\n", .{});
        return error.R1Corrupt;
    };
    const blocks = region.blocks; // ORIGINAL order
    const region_start = region.start;
    const region_end = region.end;
    const region_len = region_end - region_start;

    // sorted_indices: stable sort by (sort_key, original_index) — v1's exact sort.
    const sorted_indices = try arena.alloc(usize, blocks.len);
    for (sorted_indices, 0..) |*s, i| s.* = i;
    std.sort.block(usize, sorted_indices, blocks, struct {
        fn lt(bl: []TailBlock, a: usize, b: usize) bool {
            const ord = std.mem.order(u8, bl[a].sort_key, bl[b].sort_key);
            return if (ord != .eq) ord == .lt else bl[a].original_index < bl[b].original_index;
        }
    }.lt);

    const fields = (try parseAllFields(arena, input, blocks)) orelse {
        std.debug.print("r1 v2 encode: parseAllFields failed (a block lacks D86a or N-line)\n", .{});
        return error.R1Corrupt;
    };
    const order_lines = (try readOrderLines(arena, order_bytes)) orelse {
        std.debug.print("r1 v2 encode: readOrderLines failed\n", .{});
        return error.R1Corrupt;
    };
    const side_scratch = (try makeSideV2(arena, fields, region_start, region_len, order_lines)) orelse {
        std.debug.print("r1 v2 encode: makeSideV2 refused (assumption break; see tags above)\n", .{});
        return error.R1Corrupt;
    };

    // output = input[0..region_start] ++ blocks(sorted) ++ input[region_end..]
    //          ++ side ++ R1ORDFT4 ++ u64le(side.len)          (v2:422-431)
    const out = try alloc.alloc(u8, input.len + side_scratch.len + kFooterMagicV2.len + 8);
    errdefer alloc.free(out);
    @memcpy(out[0..region_start], input[0..region_start]);
    var pos: usize = region_start;
    for (sorted_indices) |idx| {
        const b = blocks[idx];
        const first = b.lines[0].start;
        const last = b.lines[b.lines.len - 1];
        const blen = last.start + last.len - first;
        if (blen > region_end - pos) return error.R1Corrupt;
        @memcpy(out[pos..][0..blen], input[first..][0..blen]);
        pos += blen;
    }
    if (pos != region_end) return error.R1Corrupt; // C++ output.size()==input.size()
    @memcpy(out[region_end..input.len], input[region_end..]);
    pos = input.len;
    @memcpy(out[pos..][0..side_scratch.len], side_scratch);
    pos += side_scratch.len;
    @memcpy(out[pos..][0..kFooterMagicV2.len], &kFooterMagicV2);
    pos += kFooterMagicV2.len;
    std.mem.writeInt(u64, out[pos..][0..8], side_scratch.len, .little);

    // SELF-VERIFY (v2:433-438): the full public restore path must reproduce
    // the input byte-for-byte from the bytes we are about to emit.
    const restored = restoreEncodedTailV2(alloc, out, order_bytes) catch |e| {
        std.debug.print(
            "FATAL: r1 v2 SELF-VERIFY could not restore its own output ({any}) — refusing to emit.\n",
            .{e},
        );
        return error.R1SelfVerifyFailed;
    };
    const same = std.mem.eql(u8, restored, input);
    alloc.free(restored);
    if (!same) {
        std.debug.print("FATAL: r1 v2 SELF-VERIFY mismatch — refusing to emit (assumption break).\n", .{});
        return error.R1SelfVerifyFailed;
    }

    const side = try alloc.dupe(u8, side_scratch);
    return .{ .stream = out, .side = side };
}

// ===========================================================================
// Tests: varint + Lehmer round-trip (the only self-contained pieces without the
// full 586 MB enwik9 stream).
// ===========================================================================
const testing = std.testing;

test "varint round-trip" {
    const alloc = testing.allocator;
    const vals = [_]u64{ 0, 1, 127, 128, 300, 16384, 1 << 40 };
    for (vals) |v| {
        var buf: std.ArrayList(u8) = .empty;
        defer buf.deinit(alloc);
        try appendVarint(&buf, alloc, v);
        var pos: usize = 0;
        var out: u64 = 0;
        try testing.expect(readVarint(buf.items, &pos, &out));
        try testing.expectEqual(v, out);
        try testing.expectEqual(buf.items.len, pos);
    }
}

test "Lehmer permutation round-trip" {
    const alloc = testing.allocator;
    const perm = [_]usize{ 3, 0, 4, 1, 2 };
    var buf: std.ArrayList(u8) = .empty;
    defer buf.deinit(alloc);
    try testing.expect(try appendLehmerPermutation(&buf, alloc, &perm));
    var pos: usize = 0;
    var got: []usize = &.{};
    try testing.expect(try readLehmerPermutation(alloc, buf.items, &pos, perm.len, &got));
    defer alloc.free(got);
    try testing.expectEqualSlices(usize, &perm, got);
    try testing.expectEqual(buf.items.len, pos);
}

// ===========================================================================
// v2 (R1ORD4) codec tests — the derived-side machinery is size-agnostic below
// the engage-guard, so it is exercised on synthetic fields/order data; the
// full-stream paths are covered by the canonical 586 MB fixture gates
// ===========================================================================

test "v2 readOrderLines matches ReadOrderFile semantics" {
    var arena_state = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena_state.deinit();
    const arena = arena_state.allocator();

    const lines = (try readOrderLines(arena, "12\n-3\n\n007\n42")).?;
    try testing.expectEqualSlices(i64, &[_]i64{ 12, -3, 7, 42 }, lines);
    const crlf = (try readOrderLines(arena, "5\r\n-6\r\n")).?;
    try testing.expectEqualSlices(i64, &[_]i64{ 5, -6 }, crlf);
    try testing.expect((try readOrderLines(arena, "")) == null);
    try testing.expect((try readOrderLines(arena, "\n\n")) == null);
    try testing.expect((try readOrderLines(arena, "12\nx3\n")) == null);
}

test "v2 parseDelta takes the LAST N[-]digits line" {
    var arena_state = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena_state.deinit();
    const arena = arena_state.allocator();

    const data = "\xDF\x99N\n\xDF\x86N5\nN12\nN-42\n";
    const lines = try splitLines(arena, 0, data.len, data);
    try testing.expectEqual(@as(usize, 4), lines.len);
    const block = TailBlock{ .lines = lines, .sort_key = &.{} };
    var d: i64 = 0;
    try testing.expect(parseDelta(data, block, &d));
    try testing.expectEqual(@as(i64, -42), d); // last match wins
    var v: u64 = 0;
    try testing.expect(parseFirstD86a(data, block, &v));
    try testing.expectEqual(@as(u64, 5), v);

    // no N-line at all -> false
    const data2 = "\xDF\x99N\n\xDF\x86N5\nWORD\n";
    const lines2 = try splitLines(arena, 0, data2.len, data2);
    const block2 = TailBlock{ .lines = lines2, .sort_key = &.{} };
    try testing.expect(!parseDelta(data2, block2, &d));
    // "N" / "N-" bodies are not deltas
    const data3 = "\xDF\x99N\nN\nN-\n";
    const lines3 = try splitLines(arena, 0, data3.len, data3);
    const block3 = TailBlock{ .lines = lines3, .sort_key = &.{} };
    try testing.expect(!parseDelta(data3, block3, &d));
}

test "v2 buildClassPlan canonical order (delta asc / pos-id asc / d86a asc)" {
    var arena_state = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena_state.deinit();
    const arena = arena_state.allocator();

    // pos deltas: k0:+5 k1:-2 k2:+5 k3:-2  -> classes -2 {1,3}, +5 {0,2}
    const pos_ids = [_]i64{ 0, 5, 3, 8, 6 };
    const fields = [_]BlockFields{
        .{ .delta = 5, .d86a = 50 },
        .{ .delta = -2, .d86a = 9 },
        .{ .delta = 5, .d86a = 7 },
        .{ .delta = -2, .d86a = 100 },
    };
    const plan = (try buildClassPlan(arena, &pos_ids, &fields)).?;
    try testing.expectEqualSlices(usize, &[_]usize{ 0, 2, 4 }, plan.offsets);
    // class -2: positions by pos_ids[k+1] (3@k1 < 6@k3); class +5: (5@k0 < 8@k2)
    try testing.expectEqualSlices(usize, &[_]usize{ 1, 3, 0, 2 }, plan.pos_by_id);
    // class -2: blocks by d86a (9@1 < 100@3); class +5: (7@2 < 50@0)
    try testing.expectEqualSlices(usize, &[_]usize{ 1, 3, 2, 0 }, plan.blk_by_d86a);

    // partition disagreement (a delta present on one side only) -> null
    const bad_fields = [_]BlockFields{
        .{ .delta = 5, .d86a = 50 },
        .{ .delta = -2, .d86a = 9 },
        .{ .delta = 5, .d86a = 7 },
        .{ .delta = 4, .d86a = 100 },
    };
    try testing.expect((try buildClassPlan(arena, &pos_ids, &bad_fields)) == null);
}

test "v2 side codec round-trips a synthetic mini-region" {
    var arena_state = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena_state.deinit();
    const arena = arena_state.allocator();

    // Truth (ORIGINAL order): ids = cumsum of deltas, all distinct; the tail
    // beyond the order-file part must be ascending (the unused-ranks rule).
    const ids = [_]i64{ 0, 7, 3, -2, 12, 20, 25, 30, 41 };
    const n_order = 5;
    var fields: [ids.len - 1]BlockFields = undefined;
    for (&fields, 0..) |*f, k| {
        f.* = .{ .delta = ids[k + 1] - ids[k], .d86a = 1000 + 7 * @as(u64, k) };
    }
    // order values: any sequence order-isomorphic to ids[0..n_order]
    var order_lines: [n_order]i64 = undefined;
    for (&order_lines, 0..) |*o, w| o.* = ids[w] * 10 + 7;

    const side = (try makeSideV2(arena, &fields, 1234, 56789, &order_lines)).?;

    var meta: SideV2 = .{};
    try testing.expect(try parseSideV2Header(arena, side, &meta));
    try testing.expectEqual(@as(usize, 1234), meta.region_start);
    try testing.expectEqual(@as(usize, 56789), meta.region_len);
    try testing.expectEqual(fields.len, meta.count);
    try testing.expectEqual(ids.len, meta.n_articles);
    try testing.expectEqual(@as(usize, n_order), meta.n_order);

    // "sorted stream": blocks permuted by P (sorted pos -> original index)
    const P = [_]usize{ 5, 2, 7, 0, 4, 1, 6, 3 };
    var sorted_fields: [fields.len]BlockFields = undefined;
    for (P, 0..) |orig, s| sorted_fields[s] = fields[orig];

    const bap = (try solveAssignment(arena, side, &meta, &sorted_fields, &order_lines)).?;
    // block_at_position[k] must be P^-1[k]: the sorted-space index holding block k
    for (bap, 0..) |s, k| try testing.expectEqual(k, P[s]);

    // wrong order file (not order-isomorphic) must fail loudly, not misassign:
    // swapping two order values breaks the position->id map => class partition
    // disagreement (different delta multiset) => null.
    var bad_order = order_lines;
    std.mem.swap(i64, &bad_order[0], &bad_order[1]);
    try testing.expect((try solveAssignment(arena, side, &meta, &sorted_fields, &bad_order)) == null);
}

test "v2 detectRegionV2 selects the longest regular run, excluding edge blocks" {
    var arena_state = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena_state.deinit();
    const arena = arena_state.allocator();

    // [junk][irregular D99 block (2nd line not D86a)][regular A][regular B]
    // [trailing D99 with no following D99 start] — the run must be A..B only.
    const junk = "junk\n";
    const irr = "\xDF\x99N\nnot-d86a-here\npad\n"; // len 22, 2nd line invalid
    const blk_a = "\xDF\x99N\n\xDF\x86N5\nkeyAAAA\nN3\n"; // len 20, regular
    const blk_b2 = "\xDF\x99N\n\xDF\x86N7\nkeyBBx\nN-2\n"; // len 20, regular
    const tail = "\xDF\x99N\ntrailing-no-next-marker\n";
    const data = junk ++ irr ++ blk_a ++ blk_b2 ++ tail;

    const region = (try detectRegionV2(arena, data)).?;
    const a_start = junk.len + irr.len;
    try testing.expectEqual(a_start, region.start);
    try testing.expectEqual(a_start + blk_a.len + blk_b2.len, region.end);
    try testing.expectEqual(@as(usize, 2), region.blocks.len);
    try testing.expectEqual(@as(usize, 0), region.blocks[0].original_index);
    try testing.expectEqual(@as(usize, 1), region.blocks[1].original_index);
    // sort_key = concat of full lines[2..]
    try testing.expectEqualSlices(u8, "keyAAAA\nN3\n", region.blocks[0].sort_key);
    try testing.expectEqualSlices(u8, "keyBBx\nN-2\n", region.blocks[1].sort_key);
    // block fields parse on the detected blocks
    var d: i64 = 0;
    var v: u64 = 0;
    try testing.expect(parseFirstD86a(data, region.blocks[0], &v));
    try testing.expectEqual(@as(u64, 5), v);
    try testing.expect(parseDelta(data, region.blocks[1], &d));
    try testing.expectEqual(@as(i64, -2), d);

    // no D99 markers at all -> null
    try testing.expect((try detectRegionV2(arena, "plain text\nonly\n")) == null);
    // markers but nothing regular -> null
    try testing.expect((try detectRegionV2(arena, "\xDF\x99N\nxx\n\xDF\x99N\nyy\n")) == null);
}

test "v2 makeSideV2 refuses broken assumptions" {
    var arena_state = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena_state.deinit();
    const arena = arena_state.allocator();

    // duplicate ids (a zero delta) -> null
    const dup = [_]BlockFields{ .{ .delta = 0, .d86a = 1 }, .{ .delta = 3, .d86a = 2 } };
    const order1 = [_]i64{7};
    try testing.expect((try makeSideV2(arena, &dup, 1, 2, &order1)) == null);
    // n_order >= n_articles -> null
    const ok = [_]BlockFields{ .{ .delta = 4, .d86a = 1 }, .{ .delta = 3, .d86a = 2 } };
    const order3 = [_]i64{ 7, 8, 9 };
    try testing.expect((try makeSideV2(arena, &ok, 1, 2, &order3)) == null);
}

test "restoreEncodedTail dispatches on the footer magic" {
    const alloc = testing.allocator;
    // A bare FT4 footer (empty stream, zero-length side): under -Dr1v2 it must
    // route to the v2 path (order REQUIRED, then the engage-guard refuses the
    // 0-byte stream); flags-off it falls to the v1 parser (no FTR magic ->
    // R1Corrupt) and the v2 code does not exist in the binary.
    var buf: [kFooterMagicV2.len + 8]u8 = undefined;
    @memcpy(buf[0..kFooterMagicV2.len], &kFooterMagicV2);
    std.mem.writeInt(u64, buf[kFooterMagicV2.len..][0..8], 0, .little);
    if (comptime build_options.r1v2) {
        try testing.expectError(error.R1OrderMissing, restoreEncodedTail(alloc, &buf, null));
        try testing.expectError(error.R1Refused, restoreEncodedTail(alloc, &buf, "1\n"));
    } else {
        try testing.expectError(error.R1Corrupt, restoreEncodedTail(alloc, &buf, null));
        try testing.expectError(error.R1Corrupt, restoreEncodedTail(alloc, &buf, "1\n"));
    }
    // a v1 FTR footer with a short/absent side must stay R1Corrupt either way
    var v1buf: [kFooterMagic.len + 8]u8 = undefined;
    @memcpy(v1buf[0..kFooterMagic.len], &kFooterMagic);
    std.mem.writeInt(u64, v1buf[kFooterMagic.len..][0..8], 0, .little);
    try testing.expectError(error.R1Refused, restoreEncodedTail(alloc, &v1buf, null));
}
