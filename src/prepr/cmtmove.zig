//! cmtmove — the GATED comment-relocation lever (metadata-codecs family), as a
//! deployable temp-stream stage beside fieldcodec's tsswap.
//!
//! MEASURED: moving each
//! sub-768-byte page record's <comment> line from its grammar slot (directly before
//! the text-open line) to directly BEFORE the revision-close line reads
//! Δ = −4,432 B on the comment-dense e9 window [534.8M,554.8M) and −454 B on the
//! typical-body window [80M,100M) — sign-consistent through the real ship-line
//! coder after the UNGATED form measured regime-inconsistent (+2,235 body).
//! e9-naive band −7…−18 K (cold -n basis); net band −5…−16 K. Predictions
//!  /miss-low-favourable; bars frozen in the cmtswap2 preflight.
//!
//! Page grammar (post-WRT byte forms; dict codewords: page=DF98 title=DF9B
//! comment=DF93 text=DFA7 revision=DF99, ranks verified against english.dic):
//!
//!   page open:      4C DF 98 4E                       "L<page>N"
//!   page close:     4C 2F DF 98 4E                    "L/<page>N"
//!   comment line:   <indent spaces> 4C DF 93 4E <value> 4C 2F DF 93 4E 0A
//!   text open:      4C DF A7 ...                      "L<text ...>"  (attrs follow)
//!   revision close: <indent spaces> 4C 2F DF 99 4E 0A
//!
//! THE GATE (frozen a priori in the cmtswap2 preflight, BEFORE the arms were
//! built): a page participates ONLY IF its whole record length (page-open through
//! page-close inclusive) is < 768 bytes. The record length is INVARIANT under the
//! move, so encoder and decoder evaluate the gate identically on their respective
//! sides. W1 comment-pages median 128 B (98.3 % gated in) vs W2 median 2,339 B
//! (85.7 % gated out) — the gate isolates the earning regime (stub pages).
//!
//! ENCODE per gated page: locate the unique comment line; require the next line
//! (space-stripped) to open <text ...>; require a revision-close line after it
//! whose PREVIOUS line is not itself comment-like (decoder-ambiguity guard);
//! rotate the comment line (verbatim, indent included) to directly before the
//! revision-close line. DECODE: a comment line directly before revision close
//! moves back to directly before the (space-stripped) text-open line. Pages with
//! non-canonical shape are skipped by tests both sides evaluate identically.
//!
//! This scanner is the deterministic equivalent of the arm builder
//! (`experiments/metacodec-census/build_arms.py --gate 768`) and is verified
//! byte-exact against it on the FULL 587,138,826-byte e9 temp via the env-gated
//! test at the bottom (ZMIX_CMTMOVE_T0/T1).
//!
//! Like fieldcodec's swap this is NOT a universal-inverse pair (a source stream
//! that NATURALLY holds a comment line directly before a revision close inside a
//! sub-768 B page would be falsely un-moved); real WRT temps never do (the XML
//! grammar puts comments before <text>), and losslessness is owed to — and
//! settled by — the roundtrip gates on actual shipped content at every tier.
//! ⚠ Unlike tsswap (whose record grammar is phda9-minted, -e only), the comment
//! grammar EXISTS at -c tiers, so the engaged knob CHANGES -c archives; the
//! verify expectation there is roundtrip-losslessness, not size identity.
//!
//! Placement: composed AFTER fieldcodec's tsswap on encode and inverted BEFORE
//! it on decode (LIFO). The two transforms touch DISJOINT bytes (tsswap: tail
//! revision blocks; cmtmove: body page records), so the order is immaterial in
//! fact and fixed by convention.

const std = @import("std");

pub const GATE_T: usize = 768;

const PAGE_O = "\x4c\xdf\x98\x4e";
const PAGE_C = "\x4c\x2f\xdf\x98\x4e";
const CMT_O = "\x4c\xdf\x93\x4e";
const CMT_C = "\x4c\x2f\xdf\x93\x4e";
const TXT_O = "\x4c\xdf\xa7";
const REV_C = "\x4c\x2f\xdf\x99\x4e";

/// python: s = buf.rfind('\n', 0, pos) + 1  (line start at-or-before pos)
fn lineStart(p: []const u8, pos: usize) usize {
    if (pos == 0) return 0;
    if (std.mem.lastIndexOfScalar(u8, p[0..pos], 0x0a)) |n| return n + 1;
    return 0;
}

/// python: e = buf.find('\n', pos); e = len if e < 0 else e + 1  (line end incl \n)
fn lineEnd(p: []const u8, pos: usize) usize {
    if (std.mem.indexOfScalarPos(u8, p, pos, 0x0a)) |n| return n + 1;
    return p.len;
}

/// python: p[a:a+64].lstrip(b' ').startswith(pat)
fn stripStartsWith(p: []const u8, a: usize, cap: usize, pat: []const u8) bool {
    const hi = @min(p.len, a + cap);
    var i = a;
    while (i < hi and p[i] == ' ') i += 1;
    if (hi - i < pat.len) return false;
    return std.mem.eql(u8, p[i..][0..pat.len], pat);
}

/// Encode one gated page slice in place. Returns true if a comment was moved.
fn encodePage(page: []u8) bool {
    if (page.len >= GATE_T) return false;
    const ci = std.mem.indexOf(u8, page, CMT_O) orelse return false;
    if (std.mem.indexOfPos(u8, page, ci + 1, CMT_O) != null) return false; // multi
    const cls = lineStart(page, ci);
    const cle = lineEnd(page, ci);
    if (std.mem.indexOfPos(u8, page[0..cle], ci, CMT_C) == null) return false; // no close in line
    if (!stripStartsWith(page, cle, 64, TXT_O)) return false; // canonical slot
    const rc = std.mem.indexOfPos(u8, page, cle, REV_C) orelse return false;
    const s3 = lineStart(page, rc);
    // decoder-ambiguity guard: the line before revision close must not be comment-like
    const pp = lineStart(page, if (s3 > 0) s3 - 1 else 0);
    if (stripStartsWith(page, pp, s3 - pp, CMT_O)) return false;
    // rotate [cls,cle) to end at s3
    const clen = cle - cls;
    var tmp: [GATE_T]u8 = undefined;
    @memcpy(tmp[0..clen], page[cls..cle]);
    std.mem.copyForwards(u8, page[cls .. cls + (s3 - cle)], page[cle..s3]);
    @memcpy(page[s3 - clen .. s3], tmp[0..clen]);
    return true;
}

/// Decode one gated page slice in place. Returns true if a comment was moved back.
fn decodePage(page: []u8) bool {
    if (page.len >= GATE_T) return false;
    const rc = std.mem.indexOf(u8, page, REV_C) orelse return false;
    const s3 = lineStart(page, rc);
    const pp = lineStart(page, if (s3 > 0) s3 - 1 else 0);
    if (!stripStartsWith(page, pp, s3 - pp, CMT_O)) return false;
    const ci = std.mem.indexOfPos(u8, page, pp, CMT_O) orelse return false;
    if (ci >= s3) return false;
    if (std.mem.indexOfPos(u8, page[0..s3], ci, CMT_C) == null) return false;
    // original slot: directly before the (space-stripped) text-open line
    var to = std.mem.indexOf(u8, page, TXT_O);
    var s2: usize = 0;
    while (to) |tpos| {
        s2 = lineStart(page, tpos);
        var ok = true;
        for (page[s2..tpos]) |c| {
            if (c != ' ') {
                ok = false;
                break;
            }
        }
        if (ok) break;
        to = std.mem.indexOfPos(u8, page, tpos + 1, TXT_O);
    }
    if (to == null) return false; // dec_anom: leave untouched
    // rotate [pp,s3) up to start at s2
    const clen = s3 - pp;
    var tmp: [GATE_T]u8 = undefined;
    @memcpy(tmp[0..clen], page[pp..s3]);
    std.mem.copyBackwards(u8, page[s2 + clen .. s3], page[s2..pp]);
    @memcpy(page[s2 .. s2 + clen], tmp[0..clen]);
    return true;
}

fn transform(buf: []u8, comptime dir: enum { enc, dec }) usize {
    var moved: usize = 0;
    var i: usize = 0;
    while (std.mem.indexOfPos(u8, buf, i, PAGE_O)) |po| {
        const pc = std.mem.indexOfPos(u8, buf, po, PAGE_C) orelse break;
        const pe = pc + PAGE_C.len;
        const page = buf[po..pe];
        const did = if (dir == .enc) encodePage(page) else decodePage(page);
        if (did) moved += 1;
        i = pe;
    }
    return moved;
}

/// ENCODE: relocate every gated page's comment line to before revision close.
pub fn encodeInPlace(buf: []u8) usize {
    return transform(buf, .enc);
}

/// DECODE: relocate every gated page's post-text comment line back.
pub fn decodeInPlace(buf: []u8) usize {
    return transform(buf, .dec);
}

// ---------------------------------------------------------------------------

const t = std.testing;

const PG = "\x4c\xdf\x98\x4e\x0a"; // L<page>N \n
const PGC = "  \x4c\x2f\xdf\x98\x4e"; // L/<page>N
const TTL = "    \x4c\xdf\x9b\x4etitle\x4c\x2f\xdf\x9b\x4e\x0a";
const CMT = "      \x4c\xdf\x93\x4esome comment\x4c\x2f\xdf\x93\x4e\x0a";
const TXT = "      \x4c\xdf\xa7 attrs\x4etext body here\x0a";
const RVC = "    \x4c\x2f\xdf\x99\x4e\x0a";

fn rt(original: []const u8, expect_moves: usize) !void {
    var work: [1024]u8 = undefined;
    const w = work[0..original.len];
    @memcpy(w, original);
    try t.expectEqual(expect_moves, encodeInPlace(w));
    if (expect_moves == 0) try t.expectEqualSlices(u8, original, w);
    try t.expectEqual(expect_moves, decodeInPlace(w));
    try t.expectEqualSlices(u8, original, w);
}

test "cmtmove: canonical stub page moves and inverts" {
    const page = PG ++ TTL ++ CMT ++ TXT ++ RVC ++ PGC;
    const want = PG ++ TTL ++ TXT ++ CMT ++ RVC ++ PGC;
    var work: [page.len]u8 = page.*;
    try t.expectEqual(@as(usize, 1), encodeInPlace(&work));
    try t.expectEqualSlices(u8, want, &work);
    try t.expectEqual(@as(usize, 1), decodeInPlace(&work));
    try t.expectEqualSlices(u8, page, &work);
}

test "cmtmove: no comment, no move; multi-comment skipped; wrong slot skipped" {
    try rt(PG ++ TTL ++ TXT ++ RVC ++ PGC, 0); // no comment
    try rt(PG ++ TTL ++ CMT ++ CMT ++ TXT ++ RVC ++ PGC, 0); // two comments
    try rt(PG ++ CMT ++ TTL ++ TXT ++ RVC ++ PGC, 0); // comment not before text
}

test "cmtmove: gate excludes big pages" {
    const pad = "x" ** 800;
    try rt(PG ++ TTL ++ CMT ++ "      \x4c\xdf\xa7 a\x4e" ++ pad ++ "\x0a" ++ RVC ++ PGC, 0);
}

test "cmtmove: two pages, one gated out by a fat text" {
    const small = PG ++ TTL ++ CMT ++ TXT ++ RVC ++ PGC;
    const smallm = PG ++ TTL ++ TXT ++ CMT ++ RVC ++ PGC;
    const pad = "y" ** 780;
    const big = PG ++ TTL ++ CMT ++ "      \x4c\xdf\xa7 a\x4e" ++ pad ++ "\x0a" ++ RVC ++ PGC;
    var work: [(small ++ big).len]u8 = (small ++ big).*;
    try t.expectEqual(@as(usize, 1), encodeInPlace(&work));
    try t.expectEqualSlices(u8, smallm ++ big, &work);
    try t.expectEqual(@as(usize, 1), decodeInPlace(&work));
    try t.expectEqualSlices(u8, small ++ big, &work);
}

test "cmtmove: truncated page (no close) untouched; empty buffer" {
    try rt(PG ++ TTL ++ CMT ++ TXT, 0);
    var empty: [0]u8 = .{};
    try t.expectEqual(@as(usize, 0), encodeInPlace(&empty));
    try t.expectEqual(@as(usize, 0), decodeInPlace(&empty));
}

test "cmtmove: frozen e9 arm identity (env-gated)" {
    // Ground truth: encode(prep.temp T0) must equal the python arm builder's
    // full-temp output byte-for-byte and decode must invert:
    //   ZMIX_CMTMOVE_T0=<prep.temp> ZMIX_CMTMOVE_T1=<cmtmove_full.temp> zig build test
    const a = t.allocator;
    const t0_path = std.process.getEnvVarOwned(a, "ZMIX_CMTMOVE_T0") catch return error.SkipZigTest;
    defer a.free(t0_path);
    const t1_path = std.process.getEnvVarOwned(a, "ZMIX_CMTMOVE_T1") catch return error.SkipZigTest;
    defer a.free(t1_path);
    const t0 = try std.fs.cwd().readFileAlloc(a, t0_path, 1 << 31);
    defer a.free(t0);
    const t1 = try std.fs.cwd().readFileAlloc(a, t1_path, 1 << 31);
    defer a.free(t1);
    const work = try a.dupe(u8, t0);
    defer a.free(work);
    _ = encodeInPlace(work);
    try t.expectEqualSlices(u8, t1, work);
    _ = decodeInPlace(work);
    try t.expectEqualSlices(u8, t0, work);
}
