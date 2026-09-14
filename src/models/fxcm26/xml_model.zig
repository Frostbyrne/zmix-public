//! XMLModel1 (integration step B) — bit-exact Zig port of cmix-lex `fxcmv1.cpp`
//! XMLModel1 (3332-3533) + DetectContent (3288-3330) + the XMLTag cache structs
//! (3240-3286). Outputs per bit: `xmlS = p` plus the persisted globals
//! xlU1/xlU2/xlU3/xlU4 (xlU4 zeroed every call, set at bpos==0) and isXML
//! (`blpos - lastState < 64`), consumed by the updatecontext section (step C).
//!
//! Pointer semantics preserved: Tag/Attribute/Content are captured at bpos==0 ENTRY
//! (slot `Cache.Index&31`) and keep pointing at that slot even when Cache.Index is
//! incremented inside the switch; pTag is recomputed in the xReadTagName scan loop
//! and again after the switch.
//!
//! No-dict: lastCW==cwISBN==0 and lastCW==cwHTTP==0 hold, so the xISBN/xURL guards
//! reduce to their byte checks (faithful — the cw values are ctor params).
//!
//! Validated bit-exact vs v26-port-tools/xml_oracle.cpp over all 120056 bits:
//! XML_CS = 0x0b766563f23fae6c (xmlS + xlU1..4 + isXML per bit), per-bit columns
//! vs oracle-goldens/xml_dump_nodict.txt.
//!
//! Rust source: cmix-fxcm/src/xml_model_v26.rs (hash3 lives in parse_byte_v26.rs;
//! ported locally here since parse_byte is a later Wave-3 sibling).

const std = @import("std");

const CACHE_SIZE: usize = 1 << 5;
const MASK: u32 = @as(u32, CACHE_SIZE) - 1;

// post-WRT char constants (subset used here)
const SPACE: u8 = 32;
const FIRSTUPPER: u8 = 64;
const UPPER: u8 = 7;
const GREATERTHAN: u8 = 78;
const LESSTHAN: u8 = 76;
const COLON: u8 = 74;
const EQUALS: u8 = 77;

// ContentFlags
const X_TEXT: u32 = 0x001;
const X_NUMBER: u32 = 0x002;
const X_DATE: u32 = 0x004;
const X_TIME: u32 = 0x008;
const X_URL: u32 = 0x010;
const X_LINK: u32 = 0x020;
const X_COORDINATES: u32 = 0x040;
const X_TEMPERATURE: u32 = 0x080;
const X_ISBN: u32 = 0x100;

// XMLState
const X_NONE: u32 = 0;
const X_READ_TAG_NAME: u32 = 1;
const X_READ_TAG: u32 = 2;
const X_READ_ATTRIBUTE_NAME: u32 = 3;
const X_READ_ATTRIBUTE_VALUE: u32 = 4;
const X_READ_CONTENT: u32 = 5;
const X_READ_CDATA: u32 = 6;
const X_READ_COMMENT: u32 = 7;

/// fxcmv1.cpp `hash(U32 a,U32 b,U32 c=0xffffffff)`.
inline fn hash3(a: u32, b: u32, c: u32) u32 {
    const h = a *% 110002499 +% b *% 30005491 +% c *% 50004239;
    return h ^ (h >> 9) ^ (a >> 3) ^ (b >> 3) ^ (c >> 4);
}

inline fn is_ascii_lowercase(b: u8) bool {
    return b >= 'a' and b <= 'z';
}
inline fn is_ascii_digit(b: u8) bool {
    return b >= '0' and b <= '9';
}
/// 0x30..=0x39 inclusive over a u32
inline fn is_dig32(j: u32) bool {
    return j >= 0x30 and j <= 0x39;
}

const XmlAttribute = struct {
    name: u32 = 0,
    value: u32 = 0,
    length: u32 = 0,
};

const XmlContent = struct {
    data: u32 = 0,
    length: u32 = 0,
    type_: u32 = 0,
};

const XmlTag = struct {
    name: u32 = 0,
    length: u32 = 0,
    level: i32 = 0,
    end_tag: bool = false,
    empty: bool = false,
    content: XmlContent = .{},
    attributes: [4]XmlAttribute = [_]XmlAttribute{.{}} ** 4,
    attr_index: u32 = 0,
};

/// DetectContentmacro (fxcmv1.cpp 3288-3330). `hist[i]` = buf(i), i in 1..=9.
fn detect_content(
    content: *XmlContent,
    b: u8,
    c4: u32,
    c8: u32,
    hist: *const [10]u8,
    last_cw: u32,
    cw_isbn: u32,
) void {
    const buf = struct {
        h: *const [10]u8,
        inline fn get(self: @This(), i: usize) u32 {
            return @as(u32, self.h[i]);
        }
    }{ .h = hist };

    if ((c4 & 0xF0F0F0F0) == 0x30303030) {
        var i: u32 = 0;
        while (i < 4) {
            const j = (c4 >> @as(u5, @intCast(8 * i))) & 0xFF;
            if (!is_dig32(j)) break;
            i += 1;
        }
        if (i == 4 and
            (((c8 & 0xFDF0F0FD) == 0x2D30302D and is_dig32(buf.get(9))) or
                ((c8 & 0xF0FDF0FD) == 0x302D302D)))
        {
            content.type_ |= X_DATE;
        }
    } else if (((c8 & 0xF0F0FDF0) == 0x30302D30 or (c8 & 0xF0F0F0FD) == 0x3030302D) and
        is_dig32(buf.get(9)))
    {
        var i: u32 = 2;
        while (i < 4) {
            const j = (c8 >> @as(u5, @intCast(8 * i))) & 0xFF;
            if (!is_dig32(j)) break;
            i += 1;
        }
        if (i == 4 and (c4 & 0xF0FDF0F0) == 0x302D3030) {
            content.type_ |= X_DATE;
        }
    }
    if ((c4 & 0xF0FFF0F0) == (0x30003030 + @as(u32, COLON) * 256 * 256) and
        is_dig32(buf.get(5)) and
        (!is_dig32(buf.get(6)) or
            ((c8 & 0xF0F0FF00) == (0x30300000 + @as(u32, COLON) * 256) and !is_dig32(buf.get(9)))))
    {
        content.type_ |= X_TIME;
    }
    if (content.length >= 8 and (c8 & 0x80808080) == 0 and (c4 & 0x80808080) == 0) {
        content.type_ |= X_TEXT;
    }
    if ((c8 & 0xF0F0FF) == 0x3030C2 and (c4 & 0xFFF0F0FF) == 0xB0303027) {
        var i: usize = 2;
        while (i < 7 and is_dig32(buf.get(i))) {
            i += (i & 1) * 2 + 1;
        }
        if (i == 10) {
            content.type_ |= X_COORDINATES;
        }
    }
    if ((c4 & 0xFFFFFA) == 0xC2B042 and
        b != 0x47 and
        (is_dig32(c4 >> 24) or ((c4 >> 24) == 0x20 and is_dig32(buf.get(5)))))
    {
        content.type_ |= X_TEMPERATURE;
    }
    if (is_dig32(@as(u32, b))) {
        content.type_ |= X_NUMBER;
    }
    if (last_cw == cw_isbn and buf.get(4) == @as(u32, UPPER)) {
        content.type_ |= X_ISBN;
    }
}

pub const XmlModel1 = struct {
    tags: [CACHE_SIZE]XmlTag,
    cache_index: u32,
    state: u32,
    p_state: u32,
    c8: u32,
    white_space_run: u32,
    p_ws_run: u32,
    indent_tab: u32,
    indent_step: u32,
    line_ending: u32,
    last_state: u32,
    state_bh: [8]u32,
    // persisted outputs (globals xlU1..4 / isXML in the C++)
    xl_u1: u32,
    xl_u2: u32,
    xl_u3: u32,
    xl_u4: u32,
    is_xml: bool,
    // dict codeword consts (0 in no-dict)
    cw_isbn: u32,
    cw_http: u32,

    pub fn new() XmlModel1 {
        return XmlModel1{
            .tags = [_]XmlTag{.{}} ** CACHE_SIZE,
            .cache_index = 0,
            .state = X_NONE,
            .p_state = X_NONE,
            .c8 = 0,
            .white_space_run = 0,
            .p_ws_run = 0,
            .indent_tab = 0,
            .indent_step = 2,
            .line_ending = 2,
            .last_state = 0,
            .state_bh = [_]u32{0} ** 8,
            .xl_u1 = 0,
            .xl_u2 = 0,
            .xl_u3 = 0,
            .xl_u4 = 0,
            .is_xml = false,
            .cw_isbn = 0,
            .cw_http = 0,
        };
    }

    /// XMLModel1::p. `hist[i]` = buf(i) (raw byte history, post-parseByte), used
    /// only at bpos==0. `c4`/`blpos`/`last_cw` are the post-parseByte globals.
    pub fn p(self: *XmlModel1, bpos: u32, c4: u32, blpos: u32, hist: *const [10]u8, last_cw: u32) i32 {
        self.xl_u4 = 0;
        if (bpos == 0) {
            const b: u8 = @truncate(c4 & 0xff);
            // pointer captures (fixed for this call, except ptag_i)
            var ptag_i: usize = @intCast((self.cache_index -% 1) & MASK);
            const tag_i: usize = @intCast(self.cache_index & MASK);
            const attr_j: usize = @intCast(self.tags[tag_i].attr_index & 3);
            self.p_state = self.state;
            self.c8 = (self.c8 << 8) | @as(u32, hist[5]);
            // whitespace / indent tracking
            if ((b == 0x09 or b == 0x20) and (@as(u32, b) == ((c4 >> 8) & 0xff) or self.white_space_run == 0)) {
                self.white_space_run +%= 1;
                self.indent_tab = @intFromBool(b == 0x09);
            } else {
                if ((self.state == X_NONE or
                    (self.state == X_READ_CONTENT and
                        self.tags[tag_i].content.length <= self.line_ending +% self.white_space_run)) and
                    self.white_space_run > 1 +% self.indent_tab and
                    self.white_space_run != self.p_ws_run)
                {
                    const diff: i32 = @bitCast(self.white_space_run -% self.p_ws_run);
                    self.indent_step = @abs(diff);
                    self.p_ws_run = self.white_space_run;
                }
                self.white_space_run = 0;
            }
            if (b == 0x0A) {
                self.line_ending = 1 + @as(u32, @intFromBool(((c4 >> 8) & 0xff) == 0x0D));
            }
            if (self.state != X_NONE) {
                self.last_state = blpos;
            }
            switch (self.state) {
                X_NONE => {
                    if (b == LESSTHAN) {
                        self.state = X_READ_TAG_NAME;
                        const p_end = self.tags[ptag_i].end_tag;
                        const p_empty = self.tags[ptag_i].empty;
                        const p_level = self.tags[ptag_i].level;
                        self.tags[tag_i] = XmlTag{};
                        self.tags[tag_i].level = if (p_end or p_empty) p_level else p_level +% 1;
                    }
                    if (self.tags[tag_i].level > 1) {
                        detect_content(
                            &self.tags[tag_i].content,
                            b,
                            c4,
                            self.c8,
                            hist,
                            last_cw,
                            self.cw_isbn,
                        );
                    }
                    self.xl_u4 = hash3(
                        self.p_state,
                        self.state,
                        @as(u32, @bitCast(self.tags[ptag_i].level +% 1)) *% self.indent_step -% self.white_space_run,
                    );
                },
                X_READ_TAG_NAME => {
                    if (self.tags[tag_i].length > 0 and (b == 0x09 or b == 0x0A or b == 0x0D or b == 0x20)) {
                        self.state = X_READ_TAG;
                    } else if (b > 127 or
                        (b == COLON or b == FIRSTUPPER or b == UPPER or b == 0x5F or is_ascii_lowercase(b)) or
                        (self.tags[tag_i].length > 0 and (b == 0x2D or b == 0x2E or is_ascii_digit(b))))
                    {
                        self.tags[tag_i].length +%= 1;
                        self.tags[tag_i].name = self.tags[tag_i].name *% (263 * 32) +% @as(u32, b & 0xDF);
                    } else if (b == GREATERTHAN) {
                        if (self.tags[tag_i].end_tag) {
                            self.state = X_NONE;
                            self.cache_index +%= 1;
                        } else {
                            self.state = X_READ_CONTENT;
                        }
                    } else if (b != 0x21 and b != 0x2D and b != 0x2F and b != 0x5B) {
                        self.state = X_NONE;
                        self.cache_index +%= 1;
                    } else if (self.tags[tag_i].length == 0) {
                        if (b == 0x2F) {
                            self.tags[tag_i].end_tag = true;
                            self.tags[tag_i].level = @max(self.tags[tag_i].level -% 1, 0);
                        } else if (c4 == @as(u32, LESSTHAN) * 256 * 256 * 256 + 0x212D2D) {
                            self.state = X_READ_COMMENT;
                            self.tags[tag_i].level = @max(self.tags[tag_i].level -% 1, 0);
                        }
                    }
                    if (self.tags[tag_i].length == 1 and (c4 & 0xFFFF00) == @as(u32, LESSTHAN) * 256 * 256 + 0x2100) {
                        self.tags[tag_i] = XmlTag{};
                        self.state = X_NONE;
                    }
                    // pTag scan (do-while)
                    var i: u32 = 1;
                    while (true) {
                        ptag_i = @intCast((self.cache_index -% i) & MASK);
                        const prev_i: usize = @intCast((self.cache_index -% (i +% 1)) & MASK);
                        const inc: u32 = @intFromBool(self.tags[ptag_i].end_tag and
                            self.tags[prev_i].name == self.tags[ptag_i].name);
                        i +%= 1 +% inc;
                        if (!(i < @as(u32, CACHE_SIZE) and (self.tags[ptag_i].end_tag or self.tags[ptag_i].empty))) {
                            break;
                        }
                    }
                    self.xl_u4 = hash3(
                        self.p_state *% 8 +% self.state,
                        hash3(self.tags[tag_i].name, @as(u32, @bitCast(self.tags[tag_i].level)), 0xffffffff),
                        hash3(
                            self.tags[ptag_i].name,
                            @intFromBool(self.tags[ptag_i].level != self.tags[tag_i].level),
                            0xffffffff,
                        ),
                    );
                },
                X_READ_TAG => {
                    if (b == 0x2F) {
                        self.tags[tag_i].empty = true;
                    } else if (b == GREATERTHAN) {
                        if (self.tags[tag_i].empty) {
                            self.state = X_NONE;
                            self.cache_index +%= 1;
                        } else {
                            self.state = X_READ_CONTENT;
                        }
                    } else if (b != 0x09 and b != 0x0A and b != 0x0D and b != 0x20) {
                        self.state = X_READ_ATTRIBUTE_NAME;
                        self.tags[tag_i].attributes[attr_j].name = @as(u32, b);
                    }
                    self.xl_u4 = hash3(
                        self.p_state,
                        self.state,
                        hash3(self.tags[tag_i].name, @as(u32, b), self.tags[tag_i].attr_index),
                    );
                },
                X_READ_ATTRIBUTE_NAME => {
                    if ((c4 & 0xFFF0) == @as(u32, EQUALS) * 256 + @as(u32, SPACE) and (b == 0x22 or b == 0x27)) {
                        self.state = X_READ_ATTRIBUTE_VALUE;
                        if ((self.c8 & 0xDFDF) == 0x4852 and (c4 & 0xDFDF0000) == 0x45460000) {
                            self.tags[tag_i].content.type_ |= X_LINK;
                        }
                    } else if (b != 0x22 and b != 0x27 and b != EQUALS) {
                        self.tags[tag_i].attributes[attr_j].name =
                            self.tags[tag_i].attributes[attr_j].name *% (263 * 32) +% @as(u32, b & 0xDF);
                    }
                    self.xl_u4 = hash3(
                        self.p_state *% 8 +% self.state,
                        self.tags[tag_i].attributes[attr_j].name,
                        hash3(
                            self.tags[tag_i].attr_index,
                            self.tags[tag_i].name,
                            self.tags[tag_i].content.type_,
                        ),
                    );
                },
                X_READ_ATTRIBUTE_VALUE => {
                    if (b == 0x22 or b == 0x27) {
                        self.tags[tag_i].attr_index +%= 1;
                        self.state = X_READ_TAG;
                    } else {
                        self.tags[tag_i].attributes[attr_j].value =
                            self.tags[tag_i].attributes[attr_j].value *% (997 * 16) +% @as(u32, b);
                        self.tags[tag_i].attributes[attr_j].length +%= 1;
                        if (last_cw == self.cw_http and (c4 >> 8) == @as(u32, COLON) * 256 * 256 + 0x2F2F) {
                            self.tags[tag_i].content.type_ |= X_URL;
                        }
                    }
                    self.xl_u4 = hash3(
                        self.p_state,
                        self.state,
                        hash3(
                            self.tags[tag_i].attributes[attr_j].name,
                            self.tags[tag_i].content.type_,
                            0xffffffff,
                        ),
                    );
                },
                X_READ_CDATA, X_READ_CONTENT => {
                    if (b == LESSTHAN) {
                        self.state = X_READ_TAG_NAME;
                        self.cache_index +%= 1;
                        const new_i: usize = @intCast(self.cache_index & MASK);
                        const lvl = self.tags[tag_i].level;
                        self.tags[new_i] = XmlTag{};
                        self.tags[new_i].level = lvl +% 1;
                    } else {
                        self.tags[tag_i].content.length +%= 1;
                        self.tags[tag_i].content.data =
                            self.tags[tag_i].content.data *% (997 * 16) +% @as(u32, b);
                        detect_content(
                            &self.tags[tag_i].content,
                            b,
                            c4,
                            self.c8,
                            hist,
                            last_cw,
                            self.cw_isbn,
                        );
                    }
                    self.xl_u4 = hash3(
                        self.p_state,
                        self.state,
                        hash3(self.tags[tag_i].name, c4 & 0xC0FF, 0xffffffff),
                    );
                },
                X_READ_COMMENT => {
                    if ((c4 & 0xFFFFFF) == 0x2D2D00 + @as(u32, GREATERTHAN)) {
                        self.state = X_NONE;
                        self.cache_index +%= 1;
                    }
                    self.xl_u4 = hash3(self.p_state, self.state, 0xffffffff);
                },
                else => unreachable,
            }
            self.state_bh[@as(usize, self.p_state)] = (self.state_bh[@as(usize, self.p_state)] << 8) | @as(u32, b);
            ptag_i = @intCast((self.cache_index -% 1) & MASK);
            // set context if last state was less than 64 bytes ago
            self.is_xml = (blpos -% self.last_state) < 64;
            self.xl_u1 = hash3(
                self.state,
                @as(u32, @bitCast(self.tags[tag_i].level)),
                hash3(
                    self.p_state *% 2 +% @as(u32, @intFromBool(self.tags[tag_i].end_tag)),
                    self.tags[tag_i].name,
                    0xffffffff,
                ),
            );
            self.xl_u2 = hash3(
                self.tags[ptag_i].name,
                self.state *% 2 +% @as(u32, @intFromBool(self.tags[ptag_i].end_tag)),
                hash3(
                    self.tags[ptag_i].content.type_,
                    self.tags[tag_i].content.type_,
                    0xffffffff,
                ),
            );
            self.xl_u3 = hash3(
                self.state *% 2 +% @as(u32, @intFromBool(self.tags[tag_i].end_tag)),
                self.tags[tag_i].name,
                hash3(self.tags[tag_i].content.type_, c4 & 0xE0FF, 0xffffffff),
            );
        }
        const sbh = self.state_bh[@as(usize, self.state)];
        const s = ((sbh >> @as(u5, @intCast(28 - bpos))) & 0x08) |
            ((sbh >> @as(u5, @intCast(21 - bpos))) & 0x04) |
            ((sbh >> @as(u5, @intCast(14 - bpos))) & 0x02) |
            ((sbh >> @as(u5, @intCast(7 - bpos))) & 0x01) |
            (bpos << 4);
        return @intCast((s << 3) | self.state);
    }
};

// ===================================================================
// Test: xmlS + xlU1..4 + isXML per bit vs xml_oracle.cpp over all
// 120056 bits; XML_CS = 0x0b766563f23fae6c.
//
// XmlModel1 consumes ParseByte outputs (c4/blpos/hist[1..9]/last_cw) which are
// produced by the Wave-3 parse_byte port (not yet available). Until then the
// per-byte ParseByte snapshot is replayed from a recorded golden
// (goldens/xml_pb_inputs.bin): 21 bytes/byte = c4(LE u32) blpos(LE u32)
// last_cw(LE u32) hist[1..9]. Recorded from cmix-fxcm ParseByte over the same
// stream.bin; drives XmlModel1 identically to the Rust test.
// ===================================================================
test "xml_model matches cpp oracle" {
    const inputs = @embedFile("goldens/xml_pb_inputs.bin");
    const dump = @embedFile("goldens/xml_dump_nodict.txt");

    const nbytes: usize = 15007;
    try std.testing.expectEqual(nbytes * 21, inputs.len);

    var xml = XmlModel1.new();
    var cs: u64 = 0;
    var bit_idx: u32 = 0;

    // dump line iterator (skip '#' lines)
    var lines = std.mem.tokenizeScalar(u8, dump, '\n');

    var bi: usize = 0;
    while (bi < nbytes) : (bi += 1) {
        const off = bi * 21;
        const c4 = std.mem.readInt(u32, inputs[off .. off + 4][0..4], .little);
        const blpos = std.mem.readInt(u32, inputs[off + 4 .. off + 8][0..4], .little);
        const last_cw = std.mem.readInt(u32, inputs[off + 8 .. off + 12][0..4], .little);
        var hist = [_]u8{0} ** 10;
        var q: usize = 1;
        while (q <= 9) : (q += 1) {
            hist[q] = inputs[off + 12 + (q - 1)];
        }

        var k: u32 = 0;
        while (k < 8) : (k += 1) {
            const bpos = (k + 1) & 7;
            const xs = xml.p(bpos, c4, blpos, &hist, last_cw);

            cs = cs *% 1000003 +% @as(u64, @as(u32, @bitCast(xs)));
            cs = cs *% 1000003 +% @as(u64, xml.xl_u1);
            cs = cs *% 1000003 +% @as(u64, xml.xl_u2);
            cs = cs *% 1000003 +% @as(u64, xml.xl_u3);
            cs = cs *% 1000003 +% @as(u64, xml.xl_u4);
            cs = cs *% 1000003 +% @as(u64, @intFromBool(xml.is_xml));

            // next non-comment dump line
            var line = lines.next() orelse return error.DumpExhausted;
            while (line.len > 0 and line[0] == '#') {
                line = lines.next() orelse return error.DumpExhausted;
            }
            var toks = std.mem.tokenizeScalar(u8, line, ' ');
            _ = toks.next(); // bitidx
            const f_xs = try std.fmt.parseInt(i64, toks.next().?, 10);
            const f_u1 = try std.fmt.parseInt(i64, toks.next().?, 10);
            const f_u2 = try std.fmt.parseInt(i64, toks.next().?, 10);
            const f_u3 = try std.fmt.parseInt(i64, toks.next().?, 10);
            const f_u4 = try std.fmt.parseInt(i64, toks.next().?, 10);
            const f_isxml = try std.fmt.parseInt(i64, toks.next().?, 10);

            const got = [6]i64{
                @as(i64, xs),
                @as(i64, xml.xl_u1),
                @as(i64, xml.xl_u2),
                @as(i64, xml.xl_u3),
                @as(i64, xml.xl_u4),
                @as(i64, @intFromBool(xml.is_xml)),
            };
            const want = [6]i64{ f_xs, f_u1, f_u2, f_u3, f_u4, f_isxml };
            for (0..6) |c| {
                if (got[c] != want[c]) {
                    std.debug.print(
                        "FIRST DIVERGENCE bit {} (byte {} bpos {}): col {} got {} want {}\n",
                        .{ bit_idx, bit_idx / 8, bpos, c, got[c], want[c] },
                    );
                    return error.Divergence;
                }
            }
            bit_idx += 1;
        }
    }

    try std.testing.expectEqual(@as(u32, 120056), bit_idx);
    try std.testing.expectEqual(@as(u64, 0x0b766563f23fae6c), cs);
}
