//! Port of cmix-lex's readalike article reorder
//! (`ref/cmix-lex/src/readalike_prepr/article_reorder.h` + `article_remap.cpp`).
//!
//! These are DETERMINISTIC byte transforms run inside the `-e` pipeline, upstream
//! of phda9/WRT/coder:
//!   COMPRESS:   reorder(.main, .new_article_order) -> .main_reordered
//!   DECOMPRESS: sort(.main_decomp_restored)        -> .main_decomp_restored_sorted
//! `sort` bubblesorts the restored articles back into ascending-id order, which
//! undoes `reorder`'s permutation because enwik9 articles are natively id-sorted.
//!
//! Public API:
//!   loadFile(alloc, buf)                       -> LoadResult   (article_reorder.h:51-91)
//!   reorder(alloc, main_buf, order_buf, N)     -> []u8         (article_reorder.h:92-164)
//!   sortMain(alloc, main_restored_buf)         -> []u8         (article_reorder.h:166-185)
//!   reorderFiles / sortFiles                                   (thin file wrappers)
//!
//! The article-order asset (1,094,862 B, 172,277 decimal lines) is not carried as
//! a loose file: it ships cmix-compressed, as comp_order inside the artifact.
//!
//! C++ line refs are into article_reorder.h unless noted.

const std = @import("std");

/// article_reorder.h:12 — total <page> count in enwik9's .main (incl. redirects).
/// Exposed as a parameter to `reorder` so tests can use a small synthetic corpus;
/// production callers pass this constant.
pub const NUM_OF_ARTICLES: i32 = 243425;

/// wfgets buffer size: `static char s[8192*8]` (article_reorder.h:32). wfgets reads
/// at most `count-1` bytes, so a physical line longer than this is split into
/// multiple chunk "lines" — this cap is load-bearing for line_count / spans.
const WF_COUNT: usize = 8192 * 8; // 65536

/// article_reorder.h:14-23 (DUMPARTICLE fields omitted — that path is compiled out).
pub const Accumulator = struct {
    id: i32 = 0,
    start: i32 = 0,
    end: i32 = 0,
};

// article_reorder.h:25-29
const ParserState = enum(u2) { expect_page = 0, expect_id, expect_pageend };
// patterns1 / transitions1 (article_reorder.h:43-44)
const patterns1 = [3][]const u8{ "<page>", "<id>", "</page>" };
const transitions1 = [3]ParserState{ .expect_id, .expect_pageend, .expect_page };

// Redirect prefixes (article_reorder.h:105-110). Exact strings — a mismatch shifts
// every remapped index.
const redirect_prefixes = [_][]const u8{
    "      <text xml:space=\"preserve\">#REDIRECT",
    "      <text xml:space=\"preserve\">#redirect",
    "      <text xml:space=\"preserve\">#Redirect",
    "      <text xml:space=\"preserve\">#REdirect",
    "      <text xml:space=\"preserve\">{{softredirect",
};

pub const LoadResult = struct {
    vec: []Accumulator,
    /// Each entry is a slice into the source buffer (a wfgets chunk, incl. its '\n').
    lines: [][]const u8,

    pub fn deinit(self: *LoadResult, alloc: std.mem.Allocator) void {
        alloc.free(self.vec);
        alloc.free(self.lines);
    }
};

// ---------------------------------------------------------------------------
// wfgets (phda9_preprocess.h:118-129): read up to WF_COUNT-1 bytes, stop *after*
// a '\n'. Returns the chunk slice, or null at EOF (wfgets would return 0, ending
// the `while(wfgets(...))` loop).
const WfReader = struct {
    buf: []const u8,
    pos: usize = 0,

    fn next(self: *WfReader) ?[]const u8 {
        if (self.pos >= self.buf.len) return null; // i == 0 -> loop ends
        const start = self.pos;
        var i: usize = 0;
        while (i < WF_COUNT - 1 and self.pos < self.buf.len) {
            const c = self.buf[self.pos];
            self.pos += 1;
            i += 1;
            if (c == '\n') break;
        }
        return self.buf[start..self.pos];
    }
};

// std::getline over a byte buffer: yields each line WITHOUT its trailing '\n'. A
// buffer ending in '\n' does not yield a trailing empty line (matches getline
// returning false at EOF).
const GetlineReader = struct {
    buf: []const u8,
    pos: usize = 0,

    fn next(self: *GetlineReader) ?[]const u8 {
        if (self.pos >= self.buf.len) return null;
        const start = self.pos;
        while (self.pos < self.buf.len and self.buf[self.pos] != '\n') self.pos += 1;
        const line = self.buf[start..self.pos];
        if (self.pos < self.buf.len) self.pos += 1; // consume the '\n'
        return line;
    }
};

/// C `atoi` on a slice: skip leading ASCII whitespace, optional sign, decimal digits.
fn atoiSlice(s: []const u8) i32 {
    var i: usize = 0;
    while (i < s.len and (s[i] == ' ' or s[i] == '\t' or s[i] == '\n' or
        s[i] == '\r' or s[i] == '\x0b' or s[i] == '\x0c')) : (i += 1)
    {}
    var neg = false;
    if (i < s.len and (s[i] == '+' or s[i] == '-')) {
        neg = s[i] == '-';
        i += 1;
    }
    var v: i64 = 0;
    while (i < s.len and s[i] >= '0' and s[i] <= '9') : (i += 1) {
        v = v * 10 + @as(i64, s[i] - '0');
    }
    if (neg) v = -v;
    return @truncate(v);
}

/// Write a stored line via wfputs (phda9_preprocess.h:131-133): emit bytes until a
/// NUL terminator (or the slice end — enwik9 main has no embedded NULs, so the
/// chunk slice and the C++ std::string are identical).
fn wfputs(out: *std.ArrayList(u8), alloc: std.mem.Allocator, line: []const u8) !void {
    var n = line.len;
    if (std.mem.indexOfScalar(u8, line, 0)) |z| n = z;
    try out.appendSlice(alloc, line[0..n]);
}

// ---------------------------------------------------------------------------
/// loadFile (article_reorder.h:51-91). FSM over wfgets chunks building Accumulator
/// spans; also records every chunk into `lines`.
pub fn loadFile(alloc: std.mem.Allocator, buf: []const u8) !LoadResult {
    var vec: std.ArrayList(Accumulator) = .empty;
    errdefer vec.deinit(alloc);
    var lines: std.ArrayList([]const u8) = .empty;
    errdefer lines.deinit(alloc);

    var state: ParserState = .expect_page;
    var acc: Accumulator = .{};
    var line_count: i32 = 0;

    var rd = WfReader{ .buf = buf };
    while (rd.next()) |chunk| {
        // strstr stops at NUL; match on the strlen-prefix of the chunk.
        var s = chunk;
        if (std.mem.indexOfScalar(u8, chunk, 0)) |z| s = chunk[0..z];

        if (std.mem.indexOf(u8, s, patterns1[@intFromEnum(state)])) |p| {
            switch (state) {
                .expect_page => acc.start = line_count,
                .expect_id => acc.id = atoiSlice(s[p + 4 ..]), // atoi(p+4), "<id>" is 4 chars
                .expect_pageend => acc.end = line_count,
            }
            state = transitions1[@intFromEnum(state)];
            if (state == .expect_page) try vec.append(alloc, acc);
        }
        line_count += 1;
        try lines.append(alloc, chunk);
    }

    return .{
        .vec = try vec.toOwnedSlice(alloc),
        .lines = try lines.toOwnedSlice(alloc),
    };
}

// ---------------------------------------------------------------------------
/// reorder (article_reorder.h:92-164). `num_articles` = NUM_OF_ARTICLES (parameter
/// for testability). Returns the `.main_reordered` byte stream.
pub fn reorder(
    alloc: std.mem.Allocator,
    main_buf: []const u8,
    order_buf: []const u8,
    num_articles: i32,
) ![]u8 {
    const num: usize = @intCast(num_articles);

    var lr = try loadFile(alloc, main_buf);
    defer lr.deinit(alloc);

    // Inverse remap: stream .main again with getline, building remap[count2]=count1
    // (compact-index -> original-page-index). Same skip logic as article_remap.cpp
    // but opposite direction. The redirect flag is set by a *previous* line and
    // consumed at the next "  <page>" (a deliberate one-line lag, mirrored exactly
    // by the generator, so the mapping is self-consistent). (article_reorder.h:100-126)
    var remap = std.AutoHashMap(i32, i32).init(alloc);
    defer remap.deinit();
    {
        var count1: i32 = -1;
        var count2: i32 = -1;
        var redirect = false;
        var gl = GetlineReader{ .buf = main_buf };
        while (gl.next()) |line| {
            for (redirect_prefixes) |pre| {
                if (line.len >= pre.len and std.mem.eql(u8, line[0..pre.len], pre)) {
                    redirect = true;
                    break;
                }
            }
            if (std.mem.eql(u8, line, "  <page>")) {
                try remap.put(count2, count1);
                if (!redirect) count2 += 1;
                count1 += 1;
                redirect = false;
            }
        }
    }

    // Read .new_article_order: each line stoi -> remap[...] (missing key => 0, the
    // std::unordered_map operator[] default). (article_reorder.h:128-133)
    var positions: std.ArrayList(i32) = .empty;
    defer positions.deinit(alloc);
    const used = try alloc.alloc(bool, num);
    defer alloc.free(used);
    @memset(used, false);
    {
        var gl = GetlineReader{ .buf = order_buf };
        while (gl.next()) |line| {
            const key = atoiSlice(line); // stoi
            const res = remap.get(key) orelse 0; // operator[] default-inserts 0
            try positions.append(alloc, res);
            if (res >= 0 and res < num_articles) used[@intCast(res)] = true;
        }
    }

    // Append any unused article index in natural order. (article_reorder.h:135-141)
    if (positions.items.len < num) {
        var i: i32 = 0;
        while (i < num_articles) : (i += 1) {
            if (!used[@intCast(i)]) try positions.append(alloc, i);
        }
    }

    // Emit .main_reordered: for each position, write lines[start..=end].
    // (article_reorder.h:143-155)
    var out: std.ArrayList(u8) = .empty;
    errdefer out.deinit(alloc);
    for (positions.items) |pos| {
        // C++ (article_reorder.h:150) does `vec[pos]` with NO bounds check. `pos`
        // (= remap[stoi(order_line)]) can exceed vec.len — same reason the `used[res]`
        // write above is guarded (line ~228): a page counted in count1 but absent from
        // vec (e.g. malformed/redirect page with `  <page>` but no `<id></page>`) makes
        // a remap value land past the article records. In C++ that OOB read hits the
        // vector's spare capacity (benign); Zig's exact-sized toOwnedSlice alloc faults
        // at a page boundary (enwik9 -e SIGSEGV, iter 308). A phantom index MUST emit
        // nothing — else the reorder stops being an invertible permutation (decompress
        // sorts by id, so any phantom line would be unrecoverable). Skipping is exactly
        // that, and matches the benign local runs whose ratios we measured.
        if (pos < 0 or @as(usize, @intCast(pos)) >= lr.vec.len) continue;
        const a = lr.vec[@intCast(pos)];
        var j: i32 = a.start;
        while (j <= a.end) : (j += 1) {
            try wfputs(&out, alloc, lr.lines[@intCast(j)]);
        }
    }
    return out.toOwnedSlice(alloc);
}

// ---------------------------------------------------------------------------
/// sort (article_reorder.h:166-185): the decompress-side inverse. loadFile the
/// restored main, bubblesort articles by ascending id, re-emit. Returns the
/// `.main_decomp_restored_sorted` byte stream.
pub fn sortMain(alloc: std.mem.Allocator, main_restored_buf: []const u8) ![]u8 {
    var lr = try loadFile(alloc, main_restored_buf);
    defer lr.deinit(alloc);

    // bubblesort by id ascending (article_reorder.h:34-42). ids are unique, so any
    // total order on id is equivalent; std.sort is deterministic here.
    std.sort.block(Accumulator, lr.vec, {}, struct {
        fn lt(_: void, a: Accumulator, b: Accumulator) bool {
            return a.id < b.id;
        }
    }.lt);

    var out: std.ArrayList(u8) = .empty;
    errdefer out.deinit(alloc);
    for (lr.vec) |a| {
        var j: i32 = a.start;
        while (j <= a.end) : (j += 1) {
            try wfputs(&out, alloc, lr.lines[@intCast(j)]);
        }
    }
    return out.toOwnedSlice(alloc);
}

// ---------------------------------------------------------------------------
// File wrappers (match the C++ fixed filenames when called from the -e driver).

fn readAll(alloc: std.mem.Allocator, path: []const u8) ![]u8 {
    const f = try std.fs.cwd().openFile(path, .{});
    defer f.close();
    const sz = (try f.stat()).size;
    const buf = try alloc.alloc(u8, sz);
    errdefer alloc.free(buf);
    const n = try f.readAll(buf);
    return buf[0..n];
}

fn writeAll(path: []const u8, bytes: []const u8) !void {
    const f = try std.fs.cwd().createFile(path, .{});
    defer f.close();
    try f.writeAll(bytes);
}

/// reorderover files: reads main_path + order_path, writes out_path.
pub fn reorderFiles(
    alloc: std.mem.Allocator,
    main_path: []const u8,
    order_path: []const u8,
    out_path: []const u8,
    num_articles: i32,
) !void {
    const main_buf = try readAll(alloc, main_path);
    defer alloc.free(main_buf);
    const order_buf = try readAll(alloc, order_path);
    defer alloc.free(order_buf);
    const out = try reorder(alloc, main_buf, order_buf, num_articles);
    defer alloc.free(out);
    try writeAll(out_path, out);
}

/// sortover files: reads restored_path, writes out_path.
pub fn sortFiles(
    alloc: std.mem.Allocator,
    restored_path: []const u8,
    out_path: []const u8,
) !void {
    const buf = try readAll(alloc, restored_path);
    defer alloc.free(buf);
    const out = try sortMain(alloc, buf);
    defer alloc.free(out);
    try writeAll(out_path, out);
}

// ===========================================================================
// Tests
// ===========================================================================
const testing = std.testing;

// Build asset sanity: the shipped new_article_order (later cmix-compressed as
// comp_order).
const order_asset = @embedFile("new_article_order_asset");

test "new_article_order asset: size + line count" {
    try testing.expectEqual(@as(usize, 1094862), order_asset.len);
    var lines: usize = 0;
    for (order_asset) |c| {
        if (c == '\n') lines += 1;
    }
    try testing.expectEqual(@as(usize, 172277), lines);
    // ends with a newline
    try testing.expectEqual(@as(u8, '\n'), order_asset[order_asset.len - 1]);
}

// A synthetic id-sorted .main with redirects, exercising loadFile FSM, the remap
// lag, order-file remap, natural-order fill, and the sortinverse.
fn synthMain() []const u8 {
    return "  <page>\n" ++ // page 0, id 10
        "    <title>A</title>\n" ++
        "    <id>10</id>\n" ++
        "      <text xml:space=\"preserve\">alpha</text>\n" ++
        "  </page>\n" ++
        "  <page>\n" ++ // page 1, id 20, REDIRECT
        "    <title>B</title>\n" ++
        "    <id>20</id>\n" ++
        "      <text xml:space=\"preserve\">#REDIRECT [[A]]</text>\n" ++
        "  </page>\n" ++
        "  <page>\n" ++ // page 2, id 30
        "    <title>C</title>\n" ++
        "    <id>30</id>\n" ++
        "      <text xml:space=\"preserve\">gamma</text>\n" ++
        "  </page>\n" ++
        "  <page>\n" ++ // page 3, id 40
        "    <title>D</title>\n" ++
        "    <id>40</id>\n" ++
        "      <text xml:space=\"preserve\">delta</text>\n" ++
        "  </page>\n";
}

test "reorder then sort round-trips id-sorted main" {
    const alloc = testing.allocator;
    const main_buf = synthMain();
    // 4 pages total -> NUM_OF_ARTICLES(test) = 4.
    const N: i32 = 4;
    // order file references compact indices; leave some unlisted for natural fill.
    const order_buf = "1\n0\n";

    const reordered = try reorder(alloc, main_buf, order_buf, N);
    defer alloc.free(reordered);

    // sortmust restore ascending-id order == original (which is id-sorted).
    const sorted = try sortMain(alloc, reordered);
    defer alloc.free(sorted);
    try testing.expectEqualSlices(u8, main_buf, sorted);
}

test "loadFile parses spans + ids" {
    const alloc = testing.allocator;
    var lr = try loadFile(alloc, synthMain());
    defer lr.deinit(alloc);
    try testing.expectEqual(@as(usize, 4), lr.vec.len);
    try testing.expectEqual(@as(i32, 10), lr.vec[0].id);
    try testing.expectEqual(@as(i32, 20), lr.vec[1].id);
    try testing.expectEqual(@as(i32, 40), lr.vec[3].id);
    // page 0 spans lines [0,4]
    try testing.expectEqual(@as(i32, 0), lr.vec[0].start);
    try testing.expectEqual(@as(i32, 4), lr.vec[0].end);
}
