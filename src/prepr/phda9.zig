//! Port of cmix-lex's phda9 "WIT" text preprocessor
//! (`ref/cmix-lex/src/readalike_prepr/phda9_preprocess.h`, 30034 bytes).
//!
//! Rhatushnyak's reversible, streaming, line-oriented Wikipedia Info Transform.
//! Pure byte rewriting (no model state) so it round-trips bit-exactly.
//!
//! Public API:
//!   encodeTxtWit(alloc, in) -> []u8   (encode_txt_wit, C:747-900)
//!   decodeTxtWit(alloc, in) -> []u8   (decode_txt_wit, C:573-745; size = in.len)
//!
//! The C uses `*(int*)&s[i]` 4-byte little-endian tag comparisons everywhere; those
//! ASCII hex constants are preserved verbatim (x86-LE specific). C `char*` pointers are
//! modeled as Zig `[*]u8`; working buffers carry a leading PAD so the negative-index
//! reads (`in-2`, `in-5`, `p4[-8]`) land in a zero region instead of out of bounds.

const std = @import("std");
const prepr_options = @import("prepr_options");
const A = std.mem.Allocator;

/// -Dphda9-nullpred (default false ⇒ this whole lever is comptime-dead and the
/// stock stream is bit-identical). See the block comment in `HentTail.run`.
const NULLPRED: bool = prepr_options.nullpred;

/// nullpred SIBLINGS (cpp1-nullsib, aug-18) — decoder-free "never fire" forms
/// of the remaining encode-side entity rewrites. All default false ⇒ stock
/// bit-identical. Decode side (hent1/hent3/hent6/restoreamp) is UNTOUCHED by
/// design: it is shape-driven and inverts the un-compacted stream exactly —
/// see each guard's comment for the invariant it rests on.
/// ⚠ The &amp;→&& doubling + escape-3 marking in `hent` is NOT nullable:
/// the aug-18 collision census found 16 in-text sites (raw `&amp;` followed by
/// one of ! * ^ <) where decode hent3 would misfire without the marker.
const NULLQLG: bool = prepr_options.nullqlg;
const NULLH2: bool = prepr_options.nullh2;
const NULLH5: bool = prepr_options.nullh5;

pub const Phda9Error = error{Phda9Fail} || A.Error;

// ---------------------------------------------------------------------------
// C-string / memory primitives over `[*]u8`
// ---------------------------------------------------------------------------

inline fn rd32(p: [*]const u8) u32 {
    return std.mem.readInt(u32, p[0..4], .little);
}
inline fn wr32(p: [*]u8, v: u32) void {
    std.mem.writeInt(u32, p[0..4], v, .little);
}
inline fn pdiff(a: anytype, b: anytype) i64 {
    return @as(i64, @intCast(@intFromPtr(a))) - @as(i64, @intCast(@intFromPtr(b)));
}

fn strlen(p: [*]const u8) usize {
    var i: usize = 0;
    while (p[i] != 0) i += 1;
    return i;
}

fn cstrchr(p: [*]u8, ch: u8) ?[*]u8 {
    var i: usize = 0;
    while (true) : (i += 1) {
        const c = p[i];
        if (c == ch) return p + i;
        if (c == 0) return null;
    }
}

fn cstrstr(hay: [*]u8, needle: []const u8) ?[*]u8 {
    var i: usize = 0;
    while (hay[i] != 0) : (i += 1) {
        var m: usize = 0;
        while (m < needle.len and hay[i + m] == needle[m]) m += 1;
        if (m == needle.len) return hay + i;
    }
    return null;
}

/// memcmp(a, lit, lit.len) == 0
fn memeq(a: [*]const u8, lit: []const u8) bool {
    var i: usize = 0;
    while (i < lit.len) : (i += 1) {
        if (a[i] != lit[i]) return false;
    }
    return true;
}

fn amemcpy(dst: [*]u8, src: [*]const u8, n: i64) void {
    var i: usize = 0;
    const nn: usize = @intCast(n);
    while (i < nn) : (i += 1) dst[i] = src[i];
}

/// C atoi (skip leading ws, optional sign, digits).
fn atoi(p: [*]const u8) i64 {
    var i: usize = 0;
    while (p[i] == ' ' or p[i] == '\t' or p[i] == '\n' or p[i] == '\r' or p[i] == 0x0b or p[i] == 0x0c) i += 1;
    var sign: i64 = 1;
    if (p[i] == '+') {
        i += 1;
    } else if (p[i] == '-') {
        sign = -1;
        i += 1;
    }
    var v: i64 = 0;
    while (p[i] >= '0' and p[i] <= '9') : (i += 1) v = v * 10 + @as(i64, p[i] - '0');
    return sign * v;
}

/// numlen (C:188-195): length of digit run terminated by ';'; 0 if any non-digit hit first.
fn numlen(p: [*]const u8) i32 {
    var i: i32 = 0;
    var k: usize = 0;
    while (p[k] != ';') {
        if (p[k] < '0' or p[k] > '9') return 0;
        i += 1;
        k += 1;
    }
    return i;
}

/// Write C "%d" decimal at dst; returns byte count (no NUL).
fn putDec(dst: [*]u8, val: i64) usize {
    var buf: [24]u8 = undefined;
    var v = val;
    var neg = false;
    if (v < 0) {
        neg = true;
        v = -v;
    }
    var n: usize = 0;
    if (v == 0) {
        buf[n] = '0';
        n += 1;
    }
    while (v > 0) {
        buf[n] = @intCast('0' + @as(u8, @intCast(@mod(v, 10))));
        n += 1;
        v = @divTrunc(v, 10);
    }
    var o: usize = 0;
    if (neg) {
        dst[0] = '-';
        o = 1;
    }
    var k = n;
    while (k > 0) {
        k -= 1;
        dst[o] = buf[k];
        o += 1;
    }
    return o;
}

/// Write C "%02d" (min field width 2, zero padded; sign counts toward width).
fn putDec2(dst: [*]u8, val: i64) usize {
    var buf: [24]u8 = undefined;
    var v = val;
    var neg = false;
    if (v < 0) {
        neg = true;
        v = -v;
    }
    var n: usize = 0;
    if (v == 0) {
        buf[n] = '0';
        n += 1;
    }
    while (v > 0) {
        buf[n] = @intCast('0' + @as(u8, @intCast(@mod(v, 10))));
        n += 1;
        v = @divTrunc(v, 10);
    }
    var o: usize = 0;
    if (neg) {
        dst[0] = '-';
        o = 1;
    }
    var cur = o + n;
    while (cur < 2) {
        dst[o] = '0';
        o += 1;
        cur += 1;
    }
    var k = n;
    while (k > 0) {
        k -= 1;
        dst[o] = buf[k];
        o += 1;
    }
    return o;
}

// UTF8bytes table (C:94-111).
const UTF8bytes = [256]u8{
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
    2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
    3, 3, 3, 3, 3, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5,
};

fn utf8len(s: [*]const u8) i32 {
    return @as(i32, UTF8bytes[s[0]]) + 1;
}

fn utf8towc(dest: [*]const u8, ch: u32) i32 {
    if (ch == 1) return @as(i32, @as(i8, @bitCast(dest[0])));
    if (ch == 2) {
        var val: i32 = dest[1] & 0x3F;
        val |= (@as(i32, dest[0] & 0x1F)) << 6;
        return val;
    }
    if (ch == 3) {
        var val: i32 = 0;
        val |= (@as(i32, dest[0] & 0x1F)) << 12;
        val |= (@as(i32, dest[1] & 0x3F)) << 6;
        val |= (@as(i32, dest[2] & 0x3F));
        return val;
    }
    if (ch == 4) {
        var val: i32 = 0;
        val |= (@as(i32, dest[0] & 0x0F)) << 18;
        val |= (@as(i32, dest[1] & 0x3F)) << 12;
        val |= (@as(i32, dest[2] & 0x3F)) << 6;
        val |= (@as(i32, dest[3] & 0x3F));
        return val;
    }
    return 0;
}

fn wctoutf8(dest: [*]u8, ch: u32) i32 {
    if (ch < 0x80) {
        dest[0] = @intCast(ch);
        return 1;
    }
    if (ch < 0x800) {
        dest[0] = @intCast((ch >> 6) | 0xC0);
        dest[1] = @intCast((ch & 0x3F) | 0x80);
        return 2;
    }
    if (ch < 0x10000) {
        dest[0] = @intCast((ch >> 12) | 0xE0);
        dest[1] = @intCast(((ch >> 6) & 0x3F) | 0x80);
        dest[2] = @intCast((ch & 0x3F) | 0x80);
        return 3;
    }
    if (ch < 0x110000) {
        dest[0] = @intCast((ch >> 18) | 0xF0);
        dest[1] = @intCast(((ch >> 12) & 0x3F) | 0x80);
        dest[2] = @intCast(((ch >> 6) & 0x3F) | 0x80);
        dest[3] = @intCast((ch & 0x3F) | 0x80);
        return 4;
    }
    return 0;
}

// ---------------------------------------------------------------------------
// Buffers with leading PAD (for negative-index reads) + generous slack.
// ---------------------------------------------------------------------------

const PAD: usize = 16;
const CAP: usize = 1 << 20; // 1 MB working room per buffer

const Buf = struct {
    mem: []u8,
    ptr: [*]u8,
    fn init(alloc: A) !Buf {
        const mem = try alloc.alloc(u8, PAD + CAP + PAD);
        @memset(mem, 0);
        return .{ .mem = mem, .ptr = mem.ptr + PAD };
    }
    fn deinit(self: *Buf, alloc: A) void {
        alloc.free(self.mem);
    }
};

// ---------------------------------------------------------------------------
// Input reader (FILE* replacement over an in-memory slice).
// ---------------------------------------------------------------------------

const In = struct {
    data: []const u8,
    pos: usize = 0,
    eof: bool = false,

    fn getc(self: *In) i32 {
        if (self.pos >= self.data.len) {
            self.eof = true;
            return -1;
        }
        const c = self.data[self.pos];
        self.pos += 1;
        return c;
    }
    /// wfgets (C:118-127): reads up to count-1 chars or through a '\n'; NUL-terminates.
    fn wfgets(self: *In, str: [*]u8, count: i32) i32 {
        var i: i32 = 0;
        while (i < count - 1) {
            const c = self.getc();
            if (c < 0) break;
            str[@intCast(i)] = @intCast(c);
            i += 1;
            if (c == '\n') break;
        }
        str[@intCast(i)] = 0;
        return i;
    }
    fn setpos(self: *In, p: u64) void {
        self.pos = @intCast(p);
        self.eof = false;
    }
    fn curpos(self: *In) u64 {
        return self.pos;
    }
    fn blockread(self: *In, ptr: [*]u8, cnt: u64) u64 {
        var n: u64 = 0;
        while (n < cnt and self.pos < self.data.len) : (n += 1) {
            ptr[@intCast(n)] = self.data[self.pos];
            self.pos += 1;
        }
        return n;
    }
};

// Output helpers (FILE* replacement over ArrayList(u8)).
fn wfputs(out: *std.ArrayList(u8), alloc: A, str: [*]const u8) !void {
    var i: usize = 0;
    while (str[i] != 0) : (i += 1) try out.append(alloc, str[i]);
}

// ---------------------------------------------------------------------------
// Line rewriters (char* -> char*, NUL-terminated) — all ported literally.
// ---------------------------------------------------------------------------

/// skipline (C:385-391): copy in->out including the terminating NUL.
fn skipline(in0: [*]u8, out0: [*]u8) void {
    var in = in0;
    var out = out0;
    while (true) {
        const j = in[0];
        out[0] = j;
        in += 1;
        out += 1;
        if (j == 0) break;
    }
}

/// hent1 — re3 (C:198-212).
fn hent1(in0: [*]u8, out0: [*]u8) void {
    var in = in0;
    var out = out0;
    while (true) {
        const j = in[0];
        in += 1;
        out[0] = j;
        out += 1;
        if (j == '&') {
            const k = in[0];
            in += 1;
            if (k == '&') {
                wr32(out, 0x3B706D61);
                out += 4;
            } else if (k == '"') {
                wr32(out, 0x746F7571);
                out += 4;
                out[0] = ';';
                out += 1;
            } else if (k == '<') {
                wr32(out, 0x203B746C);
                out += 3;
            } else if (k == '>') {
                wr32(out, 0x203B7467);
                out += 3;
            } else {
                out[0] = k;
                out += 1;
            }
        }
        if (j == 0) break;
    }
}

/// hent6 — escape-5 -> &#NNN; (C:214-234).
fn hent6(in0: [*]u8, out0: [*]u8) void {
    var in = in0;
    var out = out0;
    while (true) {
        const j = in[0];
        in += 1;
        out[0] = j;
        out += 1;
        if (j == '&') {
            const k = in[0];
            in += 1;
            if (k == 5) {
                const cl = utf8len(in);
                const nu = utf8towc(in, @intCast(cl));
                out[0] = '&';
                (out + 1)[0] = '#';
                const dl = putDec(out + 2, nu);
                (out + 2 + dl)[0] = ';';
                (out + 2 + dl + 1)[0] = 0;
                const a = numlen(out + 2);
                out += @intCast(2 + a + 1);
                in += @intCast(cl);
            } else {
                out[0] = k;
                out += 1;
            }
        }
        if (j == 0) break;
    }
}

/// hent3 — re4 (C:236-262).
fn hent3(in0: [*]u8, out0: [*]u8) void {
    var in = in0;
    var out = out0;
    while (true) {
        const j = in[0];
        in += 1;
        out[0] = j;
        out += 1;
        if (j == ';' and rd32(in - 5) == 0x706D6126) {
            const k = in[0];
            in += 1;
            if (k == 3) {
                // empty statement (drop escape marker)
            } else if (k == '"') {
                wr32(out, 0x746F7571);
                out += 4;
                out[0] = ';';
                out += 1;
            } else if (k == '<') {
                wr32(out, 0x203B746C);
                out += 3;
            } else if (k == '>') {
                wr32(out, 0x203B7467);
                out += 3;
            } else if (k == '!') {
                wr32(out, 0x7073626E);
                out += 4;
                out[0] = ';';
                out += 1;
            } else if (k == '*') {
                wr32(out, 0x7361646E);
                out += 4;
                out[0] = 'h';
                out += 1;
                out[0] = ';';
                out += 1;
            } else if (k == '^') {
                wr32(out, 0x7361646D);
                out += 4;
                out[0] = 'h';
                out += 1;
                out[0] = ';';
                out += 1;
            } else if (k == 0xc2 and in[0] == 0xb0) {
                wr32(out, 0x3B676564);
                out += 4;
                in += 1;
            } else if (k == 0xc3 and in[0] == 0x97) {
                wr32(out, 0x656D6974);
                out += 4;
                out[0] = 's';
                out += 1;
                out[0] = ';';
                out += 1;
                in += 1;
            } else {
                out[0] = k;
                out += 1;
            }
        }
        if (j == 0) break;
    }
}

/// hent — pre3 (C:264-294).
fn hent(in0: [*]u8, out0: [*]u8) void {
    var in = in0;
    var out = out0;
    while (true) {
        const j = in[0];
        in += 1;
        out[0] = j;
        out += 1;
        if (j == '&') {
            var k = rd32(in);
            if (k == 0x3B706D61) {
                out[0] = '&';
                out += 1;
                in += 4;
                const c0 = in[0];
                const c1 = in[1];
                const c2 = in[2];
                if (c0 == '"' or c0 == '<' or c0 == '>' or c0 == '!' or c0 == '*' or c0 == '^' or
                    (c0 == 0xc2 and c1 == 0xb2) or (c0 == 0xc2 and c1 == 0xb3) or
                    (c0 == 0xc2 and c1 == 0xae) or (c0 == 0xc2 and c1 == 0xb0) or
                    (c0 == 0xe2 and c1 == 0x82 and c2 == 0xac) or (c0 == 0xc3 and c1 == 0x97) or
                    (c0 == 0xe2 and c1 == 0x88 and c2 == 0x92) or (c0 == 0xe2 and c1 == 0x88 and c2 == 0x88) or
                    (c0 == 0xe2 and c1 == 0x86 and c2 == 0x92))
                {
                    out[0] = 3;
                    out += 1;
                }
            } else if (!NULLQLG and k == 0x746F7571 and in[4] == ';') {
                // -Dphda9-nullqlg nulls this branch and the lt/gt one below:
                // &quot;/&lt;/&gt; stay spelled out. Decode-free because hent1
                // fires only on `&` + one of {&,",<,>} (entity-start letters
                // never match) and hent3 fires only at a raw `&amp;` boundary
                // with a fire-set char after it (census: the spelled forms all
                // continue with letters). removeamp then never sees a literal
                // " < > in text ⇒ it and restoreamp go inert by construction.
                out[0] = '"';
                out += 1;
                in += 5;
            } else if (!NULLQLG) {
                k = k *% 256 +% ' ';
                if (k == 0x3B746C20) {
                    out[0] = '<';
                    out += 1;
                    in += 3;
                } else if (k == 0x3B746720) {
                    out[0] = '>';
                    out += 1;
                    in += 3;
                }
            }
        }
        if (j == 0) break;
    }
}

/// hent2 — pre4 (C:296-326).
fn hent2(in0: [*]u8, out0: [*]u8) void {
    var in = in0;
    var out = out0;
    const start = in0;
    while (true) {
        const j = in[0];
        in += 1;
        out[0] = j;
        out += 1;
        // -Dphda9-nullh2: never fire ⇒ hent2 is a pure copy. Decoder-free:
        // hent1 restores the (still-doubled) && to &amp; and hent3 then sees a
        // letter after the boundary `;` for every second-level entity name,
        // so both invert the un-compacted stream exactly.
        if (!NULLH2 and j == '&' and (pdiff(in, start) > 1 and (in - 2)[0] == '&')) {
            var k = rd32(in);
            if (k == 0x746F7571 and in[4] == ';') {
                out[0] = '"';
                out += 1;
                in += 5;
            } else if (k == 0x7073626E and in[4] == ';') {
                out[0] = '!';
                out += 1;
                in += 5;
            } else if (k == 0x7361646E and in[4] == 'h' and in[5] == ';') {
                out[0] = '*';
                out += 1;
                in += 6;
            } else if (k == 0x7361646D and in[4] == 'h' and in[5] == ';') {
                out[0] = '^';
                out += 1;
                in += 6;
            } else if (k == 0x3B676564) {
                out[0] = 0xc2;
                out += 1;
                out[0] = 0xb0;
                out += 1;
                in += 4;
            } else if (k == 0x656D6974 and in[4] == 's' and in[5] == ';') {
                out[0] = 0xc3;
                out += 1;
                out[0] = 0x97;
                out += 1;
                in += 6;
            } else {
                k = k *% 256 +% ' ';
                if (k == 0x3B746C20) {
                    out[0] = '<';
                    out += 1;
                    in += 3;
                } else if (k == 0x3B746720) {
                    out[0] = '>';
                    out += 1;
                    in += 3;
                }
            }
        }
        if (j == 0) break;
    }
}

/// hent5 — &#NNN; -> escape-5 + UTF8 (C:329-351).
fn hent5(in0: [*]u8, out0: [*]u8) void {
    var in = in0;
    var out = out0;
    while (true) {
        const j = in[0];
        in += 1;
        out[0] = j;
        out += 1;
        // -Dphda9-nullh5: never fire ⇒ hent5 is a pure copy and the stream
        // carries no ESC5 byte at all, so hent6 (the only decoder that reads
        // `&`+0x05) never triggers. The &&#NNN; spelling round-trips through
        // hent1/hent3 as plain text ('#' is in neither fire set).
        if (!NULLH5 and j == '&' and in[0] == '#' and (in - 2)[0] == '&' and in[1] > '0' and in[1] <= '9') {
            const n = numlen(in + 1);
            const d = atoi(in + 1);
            if (d > 255 and n != 0) {
                in += 1;
                (out - 1)[0] = 5;
                const e = wctoutf8(out, @intCast(d));
                out += @intCast(e);
                in += @intCast(n + 1);
            }
        }
        if (j == 0) break;
    }
}

/// removeamp (C:393-406): strip '&' preceding " < > after `skip` prefix bytes.
fn removeamp(in0: [*]u8, out0: [*]u8, skip: i32) void {
    var in = in0;
    var out = out0;
    var i: i32 = 0;
    while (i < skip) : (i += 1) {
        const j = in[0];
        in += 1;
        out[0] = j;
        out += 1;
    }
    while (true) {
        const j = in[0];
        in += 1;
        if (j == '"' or j == '<' or j == '>') out -= 1;
        out[0] = j;
        out += 1;
        if (j == 0) break;
    }
}

/// restoreamp (C:407-417): re-insert '&' before " < >.
fn restoreamp(in0: [*]u8, out0: [*]u8, skip: i32) void {
    var in = in0;
    var out = out0;
    var i: i32 = 0;
    while (i < skip) : (i += 1) {
        const j = in[0];
        in += 1;
        out[0] = j;
        out += 1;
    }
    while (true) {
        const j = in[0];
        in += 1;
        if (j == '"' or j == '<' or j == '>') {
            out[0] = '&';
            out += 1;
        }
        out[0] = j;
        out += 1;
        if (j == 0) break;
    }
}

/// PROCESS macro (C:357-369): run-length double/collapse of `sym`.
fn process(src: [*]u8, dst: [*]u8, sym: u8, cond: bool) void {
    var p = src;
    const end = src + strlen(src);
    var q = dst;
    while (cstrchr(p, sym)) |t0| {
        var t = t0;
        const n = pdiff(t, p);
        amemcpy(q, p, n);
        q += @intCast(n);
        var count: i32 = 0;
        while (t[0] == sym) {
            t += 1;
            count += 1;
        }
        t += 1; // C post-increment consumes the non-sym char too
        if (cond and (count == 1 or count == 2)) count = 3 - count;
        var mm: usize = 0;
        while (mm < @as(usize, @intCast(count))) : (mm += 1) q[mm] = sym;
        q += @intCast(count);
        p = t - 1;
    }
    const rem = pdiff(end, p);
    amemcpy(q, p, rem);
    q[@intCast(rem)] = 0;
}

/// hent9 (C:353-384). NOTE: the trailing copy loop copies `in`->`out`, which discards
/// the `&` PROCESS pass; replicated verbatim (matches the reference bit-for-bit).
fn hent9(in: [*]u8, out: [*]u8) void {
    process(in, out, '{', true);
    process(out, in, '}', true);
    process(in, out, '[', true);
    process(out, in, ']', true);
    process(in, out, '&', true);
    var pin = in;
    var pout = out;
    while (true) {
        const j = pin[0];
        pout[0] = j;
        pin += 1;
        pout += 1;
        if (j == 0) break;
    }
}

// interwiki-link exception literals (henttail, C:445-490). The Wikipédia one carries a
// trailing NUL byte because the C memcmp length (20) exceeds the UTF-8 string length (19).
const ex_http = "http:";
const ex_user = "user:";
const ex_media = "media:";
const ex_mage = "mage:";
const ex_ategory = "ategory:";
const ex_wiki = "r:Wikip\xC3\xA9dia:Aide]]\x00";
const ex_boogie = "de:Boogie Down Produ";
const ex_hvordan = "da:Wikipedia:Hvordan";
const ex_indiska = "sv:Indiska musikinstrument";

// ---------------------------------------------------------------------------
// henttail (encode side, C:419-524) — statics captured in a struct.
// ---------------------------------------------------------------------------

const HentTail = struct {
    lnu: i32 = 0,
    f: i32 = 0,
    b1: i32 = 0,
    lc: i32 = 0,
    co: i32 = 0,

    fn run(self: *HentTail, in: [*]u8, out: [*]u8, lang: *std.ArrayList(u8), alloc: A) !void {
        // -Dphda9-nullpred: the NULL language predicate (CP-P1 arm P). Every
        // non-firing exit of this function is exactly `skipline(in, out)` with
        // nothing appended to `lang`, so "never fire" IS the whole arm — the
        // caller then runs its own hent9 pass over the line and emits it inline,
        // byte-for-byte the content the firing path would have parked in the
        // lang blob. The blob therefore stays empty (tsize = 0) and the decoder's
        // shape-driven re-insertion (HentTail1.run: `</revision>` while a <text>
        // is still open) never triggers, so NO decoder change is needed.
        if (comptime NULLPRED) {
            skipline(in, out);
            return;
        }
        self.lnu += 1;
        const j: i32 = @intCast(strlen(in));
        const jj: usize = @intCast(j);
        {
            var i: usize = 0;
            while (i < jj) : (i += 1) {
                if (rd32(in + i) == 0x7865743C) self.b1 = self.lnu; // "<tex"
            }
        }
        if (memeq(in + 6, "<comment>") and self.f == 0) self.co = 1;
        {
            var i: usize = 0;
            while (i < jj) : (i += 1) {
                if (rd32(in + i) == 0x6F632F3C and self.f == 0 and self.co == 1) { // "</co"
                    self.co = 0;
                    self.lnu = 0;
                    self.b1 = 0;
                    skipline(in, out);
                    return;
                }
            }
        }
        if (self.f == 0) {
            if (in[0] == '[' and in[1] == '[') {
                const ps: [*]u8 = in + @as(usize, if (in[2] == ':') 1 else 0);
                var c: i32 = 2;
                while (c < j) : (c += 1) {
                    if (in[@intCast(c)] < 'a' or in[@intCast(c)] > 'z') break;
                }
                if (c < j and in[@intCast(c)] == ':' and !(in[3] == ':' or in[2] == ':') and self.co == 0) {
                    if (memeq(ps + 2, ex_http) or memeq(ps + 2, ex_user) or memeq(ps + 2, ex_media) or
                        memeq(ps + 3, ex_mage) or memeq(ps + 3, ex_ategory) or
                        memeq(ps + 2, ex_wiki) or memeq(ps + 2, ex_boogie) or
                        memeq(ps + 2, ex_hvordan) or memeq(ps + 2, ex_indiska) or
                        (self.lnu - self.b1 < 4))
                    {
                        skipline(in, out);
                        return;
                    }
                    self.f = 1;
                    self.lc = 0;
                    self.lc += 1;
                    {
                        var i: usize = 0;
                        while (i < jj) : (i += 1) {
                            if (rd32(in + i) == 0x65742F3C and (in + 4 + i + 2)[0] == '>') {
                                self.f = 0;
                                self.lnu = 0;
                                self.b1 = 0;
                            }
                        }
                    }
                    hent9(in, out);
                    try wfputs(lang, alloc, in);
                    out[0] = 0;
                    return;
                }
            }
            skipline(in, out);
            return;
        } else if (self.f == 1) {
            self.lc += 1;
            {
                var i: usize = 0;
                while (i < jj) : (i += 1) {
                    if (rd32(in + i) == 0x65742F3C) {
                        self.f = 0;
                        self.lnu = 0;
                        self.b1 = 0;
                    }
                }
            }
            hent9(in, out);
            try wfputs(lang, alloc, in);
            out[0] = 0;
        }
    }
};

// ---------------------------------------------------------------------------
// encode_txt_wit (C:747-900)
// ---------------------------------------------------------------------------

pub fn encodeTxtWit(alloc: A, input: []const u8) Phda9Error![]u8 {
    var out: std.ArrayList(u8) = .empty;
    errdefer out.deinit(alloc);
    var out1: std.ArrayList(u8) = .empty; // lang
    defer out1.deinit(alloc);
    var out3: std.ArrayList(u8) = .empty; // header
    defer out3.deinit(alloc);

    var sbuf = try Buf.init(alloc);
    defer sbuf.deinit(alloc);
    var obuf = try Buf.init(alloc);
    defer obuf.deinit(alloc);
    const s = sbuf.ptr;
    const o = obuf.ptr;

    var in = In{ .data = input };
    var ht = HentTail{};

    // 21-space + '\n' tail-length placeholder (C:753-756).
    {
        var i: usize = 0;
        while (i < 21) : (i += 1) try out.append(alloc, 32);
    }
    try out.append(alloc, '\n');

    var f: i32 = 0;
    var lastID: i32 = 0;
    var tf: i32 = 0;

    while (true) {
        const j = in.wfgets(s, 65536);
        body: {
            if (f == 2) {
                if (rd32(s + 4) == 0x3E736E3C) { // "<ns>"
                    if (cstrchr(s, '>')) |p1| {
                        if (cstrchr(p1 + 1, '<')) |p2| {
                            p2[0] = 10;
                            p2[1] = 0;
                        }
                    }
                    if (s[0] != ' ' or s[1] != ' ' or s[2] != ' ' or s[3] != ' ') return error.Phda9Fail;
                    try wfputs(&out3, alloc, s + 5);
                    break :body;
                }
                const curID: i32 = @intCast(atoi(s + 8));
                if (rd32(s + 4) != 0x3E64693C) return error.Phda9Fail; // "<id>"
                o[0] = '>';
                const l = putDec(o + 1, curID - lastID);
                (o + 1 + l)[0] = '\n';
                (o + 1 + l + 1)[0] = 0;
                try wfputs(&out3, alloc, o);
                lastID = curID;
                f = 1;
                break :body;
            }
            if (f != 0) {
                if (rd32(s + 6) == 0x6D69743C) { // "<tim"
                    const year: i32 = @intCast(atoi(s + 17));
                    const month: i32 = @intCast(atoi(s + 22));
                    const day: i32 = @intCast(atoi(s + 25));
                    const hour: i32 = @intCast(atoi(s + 28));
                    const minute: i32 = @intCast(atoi(s + 31));
                    const second: i32 = @intCast(atoi(s + 34));
                    var q: usize = 0;
                    for ("timestamp>") |ch| {
                        o[q] = ch;
                        q += 1;
                    }
                    q += putDec2(o + q, year - 2001);
                    q += putDec(o + q, @as(i64, month) * 31 + day - 32);
                    o[q] = ':';
                    q += 1;
                    q += putDec(o + q, @as(i64, hour) * 3600 + @as(i64, minute) * 60 + second);
                    o[q] = '\n';
                    q += 1;
                    o[q] = 0;
                    try wfputs(&out3, alloc, o);
                    break :body;
                }
                if (cstrchr(s, '>')) |p1| {
                    if (cstrchr(p1 + 1, '<')) |p2| {
                        p2[0] = 10;
                        p2[1] = 0;
                    }
                }
                if (s[0] != ' ' or s[1] != ' ' or s[2] != ' ' or s[3] != ' ') return error.Phda9Fail;
                var s2: i32 = 0;
                if (f == 3) {
                    if (rd32(s + 6) == 0x6F632F3C) s2 = 7 else s2 = 9; // "</co"
                } else {
                    if (rd32(s + 4) == 0x7665723C or rd32(s + 4) == 0x7365723C or rd32(s + 4) == 0x6465723C) {
                        s2 = 5; // "<rev" "<res" "<red"
                    } else {
                        s2 = 7;
                    }
                    if (rd32(s + 6) == 0x6E6F633C) { // "<con"
                        f = 3;
                        if (rd32(s + 6 + 12) == 0x6C656420) f = 0; // " del"
                    }
                }
                if (s2 != 0) try wfputs(&out3, alloc, s + @as(usize, @intCast(s2)));
            } else {
                hent(s, o);
                hent2(o, s);
                hent5(s, o);
                var skip: i32 = 0;
                if (cstrstr(o, "<text ")) |p| {
                    tf = 1;
                    const pg = cstrchr(p, '>').?;
                    skip = @intCast(pdiff(pg + 1, o));
                    if ((pg - 1)[0] == '/') tf = 0;
                }
                if (cstrstr(o, "</text>") != null or cstrstr(o, "</revision>") != null or cstrstr(o, "</page>") != null) tf = 0;
                if (tf != 0) {
                    removeamp(o, s, skip);
                    skipline(s, o);
                }
                try ht.run(o, s, &out1, alloc);
                skipline(s, o);
                hent9(o, s);
                try wfputs(&out, alloc, o);
            }
            // trailing scans (C:852-857)
            if (tf == 0) {
                var i: usize = 0;
                const jjj: usize = @intCast(j);
                while (i < jjj) : (i += 1) {
                    if (rd32(s + i) == 0x69742F3C and rd32(s + i + 4) == 0x3E656C74 and rd32(s) == 0x20202020) f = 2;
                }
            }
            {
                var i: usize = 0;
                const jjj: usize = @intCast(j);
                while (i < jjj) : (i += 1) {
                    if (rd32(s + i) == 0x6F632F3C and rd32(s + i + 4) == 0x6972746E) f = 0;
                }
            }
        }
        if (in.eof) break;
    }

    // Tail assembly (C:861-899).
    const headersize: i32 = @truncate(@as(i64, @intCast(out3.items.len)));
    var tsize: i32 = @truncate(@as(i64, @intCast(out1.items.len)));

    var j2: i32 = 0;
    {
        const l = putDec(o, headersize);
        o[l] = '\n';
        o[l + 1] = 0;
        j2 = @intCast(strlen(o));
        try wfputs(&out, alloc, o);
    }
    {
        const l = putDec(o, tsize);
        o[l] = '\n';
        o[l + 1] = 0;
        j2 = j2 + @as(i32, @intCast(strlen(o)));
        try wfputs(&out, alloc, o);
    }
    try out.appendSlice(alloc, out3.items);
    try out.appendSlice(alloc, out1.items);

    tsize = tsize +% j2 +% headersize;

    // overwrite the placeholder with the tail length (digits only; C:892-899).
    {
        const l = putDec(o, tsize);
        o[l] = 20;
        var i: usize = 0;
        while (i < 20) : (i += 1) {
            const a = o[i];
            if (a >= '0' and a <= '9') {
                out.items[i] = a;
            } else break;
        }
    }

    return out.toOwnedSlice(alloc);
}

// ---------------------------------------------------------------------------
// id-blob header rebuild (C:18-66) — dead for streams produced by encode_txt_wit
// (id_blob_len is always 0), but ported for parity.
// ---------------------------------------------------------------------------

fn phda9LineLen(p: [*]const u8, end: [*]const u8) i32 {
    const total: usize = @intCast(pdiff(end, p));
    var i: usize = 0;
    while (i < total) : (i += 1) {
        if (p[i] == '\n') return @intCast(i + 1);
    }
    return @intCast(total);
}

fn phda9IsContributorEndLine(p: [*]const u8, len: i32) bool {
    return (len >= 13 and memeq(p, "/contributor>")) or
        (len >= 16 and memeq(p, "contributor dele"));
}

/// Returns rebuilt header (owned by alloc) and its length, or null on mismatch.
fn phda9RebuildHeaderWithIdBlob(
    alloc: A,
    header: [*]u8,
    header_len: i32,
    id_blob: [*]u8,
    id_blob_len: i32,
) !?struct { buf: []u8, len: i32 } {
    const cap: usize = @intCast(header_len + id_blob_len + 1);
    const rebuilt = try alloc.alloc(u8, cap);
    @memset(rebuilt, 0);
    var out: usize = 0;
    var hp = header;
    const hend = header + @as(usize, @intCast(header_len));
    var ip = id_blob;
    const iend = id_blob + @as(usize, @intCast(id_blob_len));
    var need_id = true;
    while (pdiff(hend, hp) > 0) {
        const hlen = phda9LineLen(hp, hend);
        if (need_id) {
            if (pdiff(iend, ip) <= 0) {
                alloc.free(rebuilt);
                return null;
            }
            const ilen = phda9LineLen(ip, iend);
            amemcpy(rebuilt.ptr + out, ip, ilen);
            out += @intCast(ilen);
            ip += @intCast(ilen);
            need_id = false;
        }
        amemcpy(rebuilt.ptr + out, hp, hlen);
        out += @intCast(hlen);
        if (phda9IsContributorEndLine(hp, hlen)) need_id = true;
        hp += @intCast(hlen);
    }
    if (pdiff(iend, ip) != 0) {
        alloc.free(rebuilt);
        return null;
    }
    rebuilt[out] = 0;
    return .{ .buf = rebuilt, .len = @intCast(out) };
}

// ---------------------------------------------------------------------------
// henttail1 (decode side, C:526-568) — statics + su/ou buffers in a struct.
// ---------------------------------------------------------------------------

const HentTail1 = struct {
    c: i32 = 0,
    lnu: i32 = 0,
    f: i32 = 0,
    p4: usize = 0, // index into the lang buffer (p1)
    su: Buf,
    ou: Buf,

    fn run(self: *HentTail1, in: [*]u8, out: [*]u8, output: *std.ArrayList(u8), alloc: A, p1base: [*]u8) !void {
        self.lnu += 1;
        const j: i32 = @intCast(strlen(in));
        const jj: usize = @intCast(j);
        {
            var i: usize = 0;
            while (i < jj) : (i += 1) {
                if (rd32(in + i) == 0x7865743C) { // "<tex"
                    self.c = 1;
                    self.f = self.lnu;
                }
            }
        }
        {
            var i: usize = 0;
            while (i < jj) : (i += 1) {
                if (rd32(in + i) == 0x65742F3C and (in + 4 + i + 2)[0] == '>') self.c = 0; // "</te"
            }
        }
        if (j >= 12 and memeq(in + @as(usize, @intCast(j - 12)), "</revision>") and self.c == 1 and (self.lnu - self.f >= 4)) {
            self.c = 0;
            const sp = self.su.ptr;
            const oup = self.ou.ptr;
            while (true) {
                const p4 = p1base + self.p4;
                const kk: i32 = @intCast(pdiff(cstrchr(p4, 10).? + 1, p4));
                var i: usize = 0;
                while (i < @as(usize, @intCast(kk))) : (i += 1) sp[i] = p4[i];
                self.p4 += @intCast(kk);
                sp[@intCast(kk)] = 0;
                hent9(sp, oup);
                if (!(cstrstr(sp, "</text>") != null or cstrstr(sp, "</revision>") != null or cstrstr(sp, "</page>") != null)) {
                    restoreamp(sp, oup, 0);
                    skipline(oup, sp);
                }
                hent6(sp, oup);
                hent1(oup, sp);
                hent3(sp, oup);
                try wfputs(output, alloc, oup);
                if (self.p4 >= 8 and memeq((p1base + self.p4) - 8, "</text>")) break;
            }
            try wfputs(output, alloc, in);
            out[0] = 0;
        } else {
            skipline(in, out);
        }
    }
};

// ---------------------------------------------------------------------------
// decode_txt_wit (C:573-745)
// ---------------------------------------------------------------------------

pub fn decodeTxtWit(alloc: A, input: []const u8) Phda9Error![]u8 {
    var out: std.ArrayList(u8) = .empty;
    errdefer out.deinit(alloc);

    var sbuf = try Buf.init(alloc);
    defer sbuf.deinit(alloc);
    var obuf = try Buf.init(alloc);
    defer obuf.deinit(alloc);
    const s = sbuf.ptr;
    const o = obuf.ptr;

    var ht1 = HentTail1{ .su = try Buf.init(alloc), .ou = try Buf.init(alloc) };
    defer ht1.su.deinit(alloc);
    defer ht1.ou.deinit(alloc);

    const size: u64 = input.len;
    var in = In{ .data = input };

    var lastID: i32 = 0;
    var tf: i32 = 0;

    _ = in.wfgets(s, 22);
    const winfo: u64 = @intCast(atoi(s));
    const insize: u64 = in.curpos();
    const tstart: u64 = size - winfo; // tail data pos
    in.setpos(tstart);

    _ = in.wfgets(s, 16);
    var headerlenght: i32 = @intCast(atoi(s));
    var h1_mem = try alloc.alloc(u8, @as(usize, @intCast(headerlenght)) + 1 + PAD);
    @memset(h1_mem, 0);
    var h1: [*]u8 = h1_mem.ptr + PAD; // leading pad for hent3-style negative reads
    defer alloc.free(h1_mem);

    _ = in.wfgets(s, 16);
    const langlenght: i32 = @intCast(atoi(s));
    const p1_mem = try alloc.alloc(u8, @as(usize, @intCast(langlenght)) + 1 + 2 * PAD);
    @memset(p1_mem, 0);
    const p1: [*]u8 = p1_mem.ptr + PAD; // leading pad for p4[-8]
    defer alloc.free(p1_mem);

    if (headerlenght != 0) _ = in.blockread(h1, @intCast(headerlenght));
    if (langlenght != 0) _ = in.blockread(p1, @intCast(langlenght));

    if (in.curpos() > size) return error.Phda9Fail;
    const id_blob_len64: u64 = size - in.curpos();
    if (id_blob_len64 != 0) {
        if (id_blob_len64 > 0x7fffffff) return error.Phda9Fail;
        const idb = try alloc.alloc(u8, @as(usize, @intCast(id_blob_len64)) + 1);
        defer alloc.free(idb);
        @memset(idb, 0);
        _ = in.blockread(idb.ptr, id_blob_len64);
        const rb = try phda9RebuildHeaderWithIdBlob(alloc, h1, headerlenght, idb.ptr, @intCast(id_blob_len64));
        if (rb == null) return error.Phda9Fail;
        // move rebuilt into a padded buffer so negative reads stay valid
        alloc.free(h1_mem);
        h1_mem = try alloc.alloc(u8, rb.?.buf.len + 1 + PAD);
        @memset(h1_mem, 0);
        h1 = h1_mem.ptr + PAD;
        amemcpy(h1, rb.?.buf.ptr, rb.?.len);
        headerlenght = rb.?.len;
        alloc.free(rb.?.buf);
    }

    in.setpos(insize + 1);
    var header: i32 = 0;
    var h1p = h1;

    while (true) {
        var j = in.wfgets(s, 65536);
        if (in.curpos() > tstart) {
            j = j - @as(i32, @intCast(in.curpos() - tstart));
            s[@intCast(j)] = 0;
        }

        if (header == 1) {
            var n: i32 = 0;
            var cont: i32 = 0;
            while (true) {
                wr32(o, 0x20202020);
                var jjv: i32 = 4;
                if (rd32(h1p) != 0x69646572 and rd32(h1p) != 0x69766572 and rd32(h1p) != 0x74736572 and n != 0) {
                    wr32(o + @as(usize, @intCast(jjv)), 0x20202020);
                    jjv += 2;
                }
                if (cont == 1) {
                    wr32(o + @as(usize, @intCast(jjv)), 0x20202020);
                    jjv += 2;
                }
                if (rd32(h1p) == 0x6E6F632F) cont = 0; // "/con"
                const kk: i32 = @intCast(pdiff(cstrchr(h1p, 10).? + 1, h1p));

                if (n == 0) {
                    if ((rd32(h1p) & 0xffffff) == 0x3E736E) { // "ns>"
                        o[@intCast(jjv)] = '<';
                        amemcpy(o + @as(usize, @intCast(jjv + 1)), h1p, kk);
                        const e: i32 = @as(i32, @intCast(pdiff(cstrchr(h1p, '>').?, h1p))) + 2;
                        if (e != kk or cont == 1) {
                            o[@intCast(kk + jjv)] = '<';
                            jjv += 1;
                            o[@intCast(kk + jjv)] = '/';
                            jjv += 1;
                            amemcpy(o + @as(usize, @intCast(jjv + kk)), h1p, kk);
                            jjv = jjv + @as(i32, @intCast(pdiff(cstrchr(h1p, '>').?, h1p))) + 1;
                        }
                        o[@intCast(kk + jjv)] = 10;
                        jjv += 1;
                        o[@intCast(kk + jjv)] = 0;
                        try wfputs(&out, alloc, o);
                    } else {
                        n += 1;
                        lastID = lastID + @as(i32, @intCast(atoi(h1p + 1)));
                        var q: usize = @intCast(jjv);
                        for ("<id>") |ch| {
                            o[q] = ch;
                            q += 1;
                        }
                        q += putDec(o + q, lastID);
                        for ("</id>") |ch| {
                            o[q] = ch;
                            q += 1;
                        }
                        o[q] = '\n';
                        q += 1;
                        o[q] = 0;
                        try wfputs(&out, alloc, o);
                    }
                } else if (rd32(h1p) == 0x656D6974) { // "time"
                    const p = cstrchr(h1p, ':').?;
                    const d: i32 = @intCast(atoi(h1p + 12));
                    const hms: i32 = @intCast(atoi(p + 1));
                    const h: i32 = @divTrunc(hms, 3600);
                    o[0] = h1p[10];
                    o[1] = h1p[11];
                    o[2] = ' ';
                    const y: i32 = @intCast(atoi(o));
                    var q: usize = 0;
                    for ("      <timestamp>") |ch| {
                        o[q] = ch;
                        q += 1;
                    }
                    q += putDec(o + q, y + 2001);
                    o[q] = '-';
                    q += 1;
                    q += putDec2(o + q, @divTrunc(d, 31) + 1);
                    o[q] = '-';
                    q += 1;
                    q += putDec2(o + q, @mod(d, 31) + 1);
                    o[q] = 'T';
                    q += 1;
                    q += putDec2(o + q, h);
                    o[q] = ':';
                    q += 1;
                    q += putDec2(o + q, @divTrunc(hms, 60) - h * 60);
                    o[q] = ':';
                    q += 1;
                    q += putDec2(o + q, @mod(hms, 60));
                    o[q] = 'Z';
                    q += 1;
                    for ("</timestamp>") |ch| {
                        o[q] = ch;
                        q += 1;
                    }
                    o[q] = '\n';
                    q += 1;
                    o[q] = 0;
                    try wfputs(&out, alloc, o);
                } else if (n != 0 or cont == 1) {
                    o[@intCast(jjv)] = '<';
                    amemcpy(o + @as(usize, @intCast(jjv + 1)), h1p, kk);
                    const e: i32 = @as(i32, @intCast(pdiff(cstrchr(h1p, '>').?, h1p))) + 2;
                    if (e != kk or cont == 1) {
                        o[@intCast(kk + jjv)] = '<';
                        jjv += 1;
                        o[@intCast(kk + jjv)] = '/';
                        jjv += 1;
                        amemcpy(o + @as(usize, @intCast(jjv + kk)), h1p, kk);
                        jjv = jjv + @as(i32, @intCast(pdiff(cstrchr(h1p, '>').?, h1p))) + 1;
                    }
                    o[@intCast(kk + jjv)] = 10;
                    jjv += 1;
                    o[@intCast(kk + jjv)] = 0;
                    try wfputs(&out, alloc, o);
                    if (rd32(h1p) == 0x746E6F63) cont = 1; // "cont"
                }
                h1p = h1p + @as(usize, @intCast(kk));
                if (memeq(h1p, "contributor dele")) break;
                if (memeq(h1p, "/contributor>")) break;
            }
            // after-loop reconstruction (C:698-708)
            wr32(o, 0x20202020);
            var jjv2: i32 = 4;
            wr32(o + @as(usize, @intCast(jjv2)), 0x20202020);
            jjv2 += 2;
            const kk2: i32 = @intCast(pdiff(cstrchr(h1p, 10).? + 1, h1p));
            o[@intCast(jjv2)] = '<';
            amemcpy(o + @as(usize, @intCast(jjv2 + 1)), h1p, kk2);
            o[@intCast(kk2 + jjv2)] = 10;
            jjv2 += 1;
            o[@intCast(kk2 + jjv2)] = 0;
            try wfputs(&out, alloc, o);
            h1p = h1p + @as(usize, @intCast(kk2));
            header = 0;
        }

        if (tf == 0 and j >= 9 and memeq(s + @as(usize, @intCast(j - 9)), "</title>") and rd32(s) == 0x20202020) {
            header = 1;
        }

        skipline(s, o);
        hent9(o, s);
        try ht1.run(o, s, &out, alloc, p1);
        if (s[0] == 0) {
            if (in.curpos() >= tstart) break;
            continue;
        }

        var skip: i32 = 0;
        if (cstrstr(s, "<text ")) |p| {
            tf = 1;
            const pg = cstrchr(p, '>').?;
            skip = @intCast(pdiff(pg + 1, s));
            if ((pg - 1)[0] == '/') tf = 0;
        }
        if (cstrstr(s, "</text>") != null or cstrstr(s, "</revision>") != null or cstrstr(s, "</page>") != null) tf = 0;
        if (tf != 0) {
            restoreamp(s, o, skip);
            skipline(o, s);
        }
        hent6(s, o);
        hent1(o, s);
        hent3(s, o);
        try wfputs(&out, alloc, o);

        if (in.curpos() >= tstart) break;
    }

    return out.toOwnedSlice(alloc);
}
