//! bcs_cold — `byte_context_sets` (the bpos==0 per-BYTE cm.set section),
//! extracted VERBATIM from fxcm_v26.zig into the ReleaseSmall module: -O3
//! unrolled this 1-per-byte function to 28,668 B .text (+18,519 vs -Oz) for
//! negligible retired-instruction benefit (see round-3 report size table).
//! `anytype` (instead of *FxcmV26) keeps the type dependency acyclic; all
//! model .setcalls dispatch to the hot module's out-of-line RF functions,
//! only the per-byte hash/context glue compiles at -Oz here.
const parse_byte = @import("parse_byte.zig");
const hash3 = parse_byte.hash3;
// fxcm_v26.zig's file-scope constants used here (trivial, kept in sync there):
const VERB: u32 = 1 << 0;
const NOUN: u32 = 1 << 1;
const SPACE: i32 = 32;
const APOSTROPHE: i32 = 39;
const CURLYOPENING: i32 = 80;
const ESCAPE: i32 = 12;
const FIRSTUPPER: i32 = 64;
const HTLINK: i32 = 31;
const HTML: i32 = 30;
const LESSTHAN: i32 = 76;
const SQUAREOPEN: i32 = 91;
const WIKITABLE: u8 = 45;
const WRT_2B = parse_byte.WRT_2B;
const WRT_3B = parse_byte.WRT_3B;
const WordsContext = @import("cmfast").WordsContext;

/// VERBATIM copy of fxcm_v26.zig's shl helper (kept in sync there).
inline fn shl(x: u32, comptime k: u32) u32 {
    return x *% (@as(u32, 1) << @as(u5, k));
}

/// The bpos==0 cm.set section.
pub fn byteContextSets(self: anytype, c4: u32) void {
        const fc = self.pb.fc;
        const c1 = self.pb.c1;
        const c1u: u32 = @bitCast(c1);
        const c1b: u8 = @truncate(c1u);
        const fcu: u32 = @bitCast(fc);
        const skip_m1 = self.pb.skip_m1;
        const col = self.pb.col;
        const h = self.pb.h;
        const word0 = self.pb.word0;
        const word00 = self.pb.word00;
        const stream2b = self.pb.stream2b;
        const stream2b_r = self.pb.stream2b_r;
        const stream3b = self.pb.stream3b;
        const stream3b_r = self.pb.stream3b_r;
        const brfc = self.pb.brfc_idx;
        const fcidx = self.pb.fc_idx;
        const is_match = self.is_match_v;

        // 5058-5066
        if ((fc == SPACE and c1 == SPACE) or skip_m1) {
            self.cm_c2[0].sets();
            self.cm_c2[0].sets();
            self.cm_c2[0].sets();
        } else {
            for (3..6) |i| self.cm_c2[0].set(self.pb.t[i]);
        }
        self.cm_c2[1].set(self.pb.t[6]);
        self.cm_c2[2].set(self.pb.t[8]);
        self.cm_c2[3].set(self.pb.t[13]);
        // 5068
        self.cm_c[5].set((self.pb.fccxt.context & 0xff00) +% c1u +% (stream2b & 12) *% 256 +% shl(@as(u32, self.pb.brcxt.cxt) +% @as(u32, @bitCast(self.pb.brcxt.last())), 24));
        // 5070-5073
        if (self.pb.qocxt.context != 0) {
            self.cm_c[4].set(shl(self.pb.qocxt.context, 8) +% c1u);
        } else {
            self.cm_c[4].set(shl(self.pb.brcxt.context, 8) +% c1u);
        }
        // 5076: rcmA[0]
        self.rcm_a.set(self.pb.worcxt0.word0(3) +% c1u +% @as(u32, 193) *% (stream3b & 0xfff), c1b);
        // 5078-5094
        if (col < 2 or fc == SPACE) {
            self.cm_c2[4].sets();
            self.cm_c2[4].sets();
            self.cm_c2[17].sets();
        } else {
            self.cm_c2[4].set(word00 +% (self.pb.number0 *% 191 +% @as(u32, @bitCast(self.pb.numlen0))) +% self.pb.u8w);
            if (self.pb.colcxt.lastfc(0) == '&' or self.pb.u8w_left != 0) {
                self.cm_c2[4].sets();
            } else {
                self.cm_c2[4].set(h +% self.pb.worcxt0.word0(1));
            }
            if (@as(i32, self.pb.brcxt.cxt) == LESSTHAN or self.pb.skip_see_external or self.pb.colcxt.nl_char == WIKITABLE) {
                self.cm_c2[17].sets();
            } else {
                self.cm_c2[17].set(self.pb.worcxt1.word(1) *% 53 +% self.pb.worcxt1.word(2) *% 11 +% h +% (self.pb.last_wt & 0xf));
            }
        }
        // 5095-5101
        if (c1 == ESCAPE or col < 2 or self.pb.u8w_left != 0 or fc == SPACE) {
            self.cm_c2[5].sets();
        } else if (is_match > 61) {
            self.cm_c2[5].sets();
        } else {
            self.cm_c2[5].set(h +% self.pb.worcxt0.word0(2) *% 71);
        }
        // 5102-5126
        if (fc == SPACE or @as(i32, self.pb.brcxt.cxt) == LESSTHAN) {
            for (0..5) |_| self.cm_c2[5].sets();
            self.cm_c2[20].sets();
            self.cm_c2[20].sets();
            self.cm_c2[20].sets();
            self.cm_c4[8].sets();
            self.cm_c4[8].sets();
        } else {
            self.cm_c2[5].set(hash3(self.pb.worcxt.word(4), self.pb.worcxt1.word(1) +% h, stream3b & 511));
            self.cm_c2[5].set(hash3(self.pb.worcxt.last(4, self.pb.worcxt.type_at(4) ^ VERB), self.pb.s_verb +% h, stream3b_r & 63));
            self.cm_c2[5].set(hash3(self.pb.worcxt.fword, self.pb.worcxt1.word(1) +% h, stream3b & 63));
            self.cm_c2[5].set(hash3(self.pb.worcxt2.word(1), self.pb.worcxt2.word(2), word00 +% c1u));
            const last_par_verb = self.pb.worcxt2.last_if(1, self.pb.worcxt.type_at(1) & VERB);
            if (last_par_verb != 0) {
                self.cm_c2[5].set(hash3(last_par_verb, word00, c1u));
            } else {
                self.cm_c2[5].sets();
            }
            self.cm_c4[8].set(hash3(h, self.pb.worcxt0.word0(1), 0xffffffff));
            self.cm_c4[8].set(hash3(self.pb.worcxt1.word(1), self.pb.worcxt1.word(2), h));
            self.cm_c2[20].set(hash3(self.pb.wt3cxt, word0, stream3b & 63));
            self.cm_c2[20].set(hash3(self.pb.wt3cxt, self.pb.worcxt2.word(1), stream3b & 511));
            self.cm_c2[20].set(hash3(self.pb.wt3cxt_w, word0, stream3b & 63));
        }
        // 5128-5186: sentence contexts
        const lastfc0 = self.pb.colcxt.lastfc(0);
        const lastfc1 = self.pb.colcxt.lastfc(1);
        const wc = self.pb.worcxt.wordcount;
        var lastwor: *const WordsContext = undefined;
        var lastwor1: *const WordsContext = undefined;
        var lastwor3: *const WordsContext = undefined;
        if (lastfc0 == '*') {
            lastwor = self.pb.sencxt.sentence_at(1);
            lastwor1 = self.pb.sencxt_l.sentence_at(2);
            lastwor3 = self.pb.sencxt.sentence_at(1);
        } else if (self.pb.colcxt.nl_char == WIKITABLE) {
            lastwor = self.pb.sencxt_t.sentence_at(2);
            lastwor1 = self.pb.sencxt_t.sentence_at(4);
            lastwor3 = self.pb.sencxt_t.sentence_at(6);
        } else {
            lastwor = self.pb.sencxt.sentence_at(1);
            lastwor1 = self.pb.sencxt.sentence_at(2);
            lastwor3 = self.pb.sencxt.sentence_at(3);
        }
        const last_word_mt = lastwor.last_r(1, wc, VERB);
        var xword: u32 = undefined;
        if (self.pb.is_paragraph == 0) {
            if (lastfc0 == '*') {
                xword = self.pb.sencxt_l.sentence_at(1).word_r(wc);
            } else if (@as(i32, lastfc0) == SQUAREOPEN) {
                xword = self.pb.sencxt_cl.similar_sentence(&self.pb.worcxt, wc, 53).word_r(wc);
            } else {
                xword = lastwor1.word_r(wc);
            }
        } else {
            xword = lastwor.last_r(1, wc, NOUN);
        }
        const last_wr = lastwor.word_r(wc);
        const lw3_wr = lastwor3.word_r(wc);
        self.xword1 = lw3_wr;
        self.cm_c2[18].set(word00 *% 1471 +% last_wr +% last_word_mt +% (stream3b & 511) *% 191 +% brfc);
        self.cm_c2[18].set(word00 *% 1471 +% xword +% (stream3b_r & 511) *% 191 +% brfc);
        var sim_noun: u32 = 0;
        var sim_verb: u32 = 0;
        if (wc != 0) {
            const sim = self.pb.sencxt.similar_sentence(&self.pb.worcxt, wc, 53);
            self.xword1 = sim.word_r(wc);
            sim_noun = sim.last_r(1, wc, NOUN);
            sim_verb = sim.last_r(1, wc, VERB);
        }
        if (wc != 0 and (lastfc0 == '*' or lastfc1 == '#')) {
            self.xword1 = self.pb.sencxt_l.similar_sentence(&self.pb.worcxt, wc, 53).word_r(wc);
        } else if (wc != 0 and self.pb.colcxt.nl_char == WIKITABLE) {
            self.xword1 = self.pb.sencxt_t.similar_sentence(&self.pb.worcxt, wc, 53).word_r(wc);
        } else if (wc != 0 and @as(i32, lastfc0) == SQUAREOPEN) {
            self.xword1 = self.pb.sencxt_cl.similar_sentence(&self.pb.worcxt, wc, 53).word_r(wc);
        }
        self.cm_c2[18].set(word0 *% 83 +% self.pb.worcxt1.word(1) +% self.xword1 *% 191);
        self.cm_c2[18].set(h +% self.pb.worcxt.word(2) *% 83 +% self.xword1 *% 53 +% (stream3b & 511) +% fcidx +% (self.pb.indirect_br_byte & 0x7ff) *% 191);
        self.cm_c2[18].set(h +% sim_verb +% sim_noun +% self.pb.linkword +% @as(u32, self.pb.fccxt.cxt));
        // 5185-5186
        if (@as(i32, self.pb.brcxt.cxt) == LESSTHAN or (@as(i32, lastfc0) != FIRSTUPPER and lastfc0 != '*' and @as(i32, lastfc0) != APOSTROPHE)) {
            self.cm_c2[19].sets();
        } else {
            self.cm_c2[19].set(h +% self.pb.worcxt0.word0(3) +% self.pb.worcxt0.word0(4));
        }
        // 5189-5190
        self.cm_c4[6].set(h +% (self.pb.worcxt.type_at(1) & 0x1FF) +% self.pb.worcxt1.word(1));
        // 5192
        if (is_match > 61) {
            self.cm_c2[6].sets();
        } else {
            self.cm_c2[6].set(shl(stream2b & 15, 16) +% (self.pb.t[2] & 0xffff));
        }
        // 5194-5197
        if (c1 == ESCAPE or self.pb.u8w_left != 0 or @as(i32, self.pb.fccxt.cxt) == CURLYOPENING) {
            self.cm_c2[7].set(0);
        } else {
            self.cm_c2[7].set(self.pb.indirect_br_byte);
        }
        // 5199-5211
        if (skip_m1) {
            self.cm_c2[8].sets();
            self.cm_c2[8].sets();
            self.cm_c2[8].sets();
        } else {
            self.cm_c2[8].set((self.pb.indirect_br_byte & 0x7ff) *% 32 +% shl(self.pb.stream4b & 0xfff0, 16) +% brfc);
            self.cm_c2[8].set((stream3b_r & 0x3fffffff) *% 4 +% (stream2b & 3));
            self.cm_c2[8].set(@as(u32, self.pb.fccxt.cxt) *% 4 +% shl(stream3b_r & 0x3ffff, 9) +% brfc);
        }
        if (@as(i32, self.pb.fccxt.cxt) == HTLINK) {
            self.cm_c2[8].sets();
        } else {
            self.cm_c2[8].set((c4 & 0xffffff) +% (shl(stream2b, 18) & 0xff000000));
        }
        // 5213-5220
        if (skip_m1) {
            self.cm_c4[0].sets();
            self.cm_c4[0].sets();
        } else {
            self.cm_c4[0].set(@as(u32, lastfc0) | shl(@as(u32, self.pb.fccxt.cxt), 15) | shl(stream3b & 63, 7) | shl(@as(u32, self.pb.brcxt.cxt), 24));
            self.cm_c4[0].set(@as(u32, lastfc0) | shl(c4 & 0xffffff, 8));
        }
        // 5222-5224
        self.cm_c4[1].set((stream2b & 3) +% word00 *% 11);
        self.cm_c4[1].set(c4 & 0xffff);
        self.cm_c4[1].set((shl(fcu, 11) | c1u) +% shl(stream2b & 3, 18));
        // 5226-5234
        self.cm_c4[2].set((stream2b & 15) +% shl(stream3b & 7, 6));
        self.cm_c4[2].set(c1u | shl(@as(u32, @bitCast(col)) *% @as(u32, @intFromBool(c1 == SPACE)), 8) | shl(stream2b & 15, 16));
        if (self.pb.is_category) {
            self.cm_c4[2].sets();
        } else {
            self.cm_c4[2].set(self.pb.wt4cxt_w1 *% 191 +% word0);
        }
        if (c1 == ESCAPE or fc == SPACE or self.pb.u8w_left != 0) {
            self.cm_c4[2].sets();
        } else {
            self.cm_c4[2].set(@as(u32, 91) *% 83 *% self.pb.worcxt.word(1) +% @as(u32, 89) *% word0);
        }
        // 5236-5244
        if (fc == SPACE) {
            self.cm_c4[4].sets();
        } else {
            self.cm_c4[4].set(c1u +% shl(stream3b & 0xe38, 6));
        }
        self.cm_c4[4].set(self.pb.worcxt.fword *% 11 +% brfc);
        self.cm_c4[4].set(c1u +% word0 +% self.pb.number0 *% 191);
        self.cm_c4[4].set(shl(c4 & 0xffff, 16) | shl(@as(u32, self.pb.fccxt.cxt), 8) | fcu);
        self.cm_c4[4].set(shl(stream3b_r & 0xfff, 8) +% (stream2b & 0xfc));
        self.cm_c4[4].set(h +% self.pb.pstate_h);
        // 5246-5279: cmC[0]
        if (c1 == ESCAPE) {
            for (0..6) |_| self.cm_c[0].sets();
        } else {
            if (self.pb.is_paragraph == 1) {
                if (@as(i32, self.pb.fccxt.cxt) == SQUAREOPEN) {
                    self.cm_c[0].sets();
                } else {
                    self.cm_c[0].set(self.pb.worcxt.fword *% 3191 +% (stream2b & 3));
                }
                self.cm_c[0].set(h +% self.pb.first_word *% 89);
                self.cm_c[0].set(word0 *% 53 +% c1u +% brfc +% self.pb.pstate_h);
            } else {
                if (@as(i32, self.pb.fccxt.cxt) == SQUAREOPEN) {
                    self.cm_c[0].sets();
                } else {
                    self.cm_c[0].set(@as(u32, @bitCast(self.pb.above)) | shl(stream3b & 0x3f, 9) | shl(@as(u32, @bitCast(self.pb.colcxt.collen(0, 0))), 19) | shl(stream2b & 3, 16));
                }
                self.cm_c[0].set(h +% self.pb.first_word *% 89);
                if (@as(i32, self.pb.fccxt.cxt) == SQUAREOPEN) {
                    self.cm_c[0].sets();
                } else {
                    self.cm_c[0].set(@as(u32, @bitCast(self.pb.above)) | shl(c1u, 16) | shl(@as(u32, @bitCast(col +% self.pb.numlen0 +% @as(i32, @bitCast(brfc)))), 8) | shl(@as(u32, @bitCast(self.pb.above1)), 24));
                }
            }
            if (lastfc0 == '*') {
                self.cm_c[0].set((word0 +% shl(@as(u32, self.pb.fccxt.cxt), 8)) | shl(brfc, 16));
                self.cm_c[0].set(c1u);
                self.cm_c[0].set(word0 +% self.pb.pstate_h);
            } else {
                const above_cell = self.pb.raw_bufr(self.pb.colcxt.abovecellpos);
                self.cm_c[0].set(@as(u32, WRT_2B[above_cell]) | shl(@as(u32, self.pb.fccxt.cxt), 8) | shl(brfc, 16));
                self.cm_c[0].set(@as(u32, above_cell) | shl(c1u, 8));
                self.cm_c[0].set(word0 +% @as(u32, WRT_2B[above_cell]));
            }
        }
        // 5281-5291: xml
        if (self.xml.is_xml) {
            self.cm_c44.set(self.xml.xl_u1);
            self.cm_c44.set(self.xml.xl_u2);
            self.cm_c44.set(self.xml.xl_u3);
            self.cm_c44.set(self.xml.xl_u4);
        } else {
            for (0..4) |_| self.cm_c44.sets();
        }
        // 5293-5299
        if (fc == SPACE or skip_m1 or self.pb.skip_see_external or !self.pb.is_page_started) {
            self.cm_c[1].sets();
            self.cm_c[1].sets();
            self.cm_c[1].sets();
        } else {
            self.cm_c[1].set((stream3b & 0x7fff) *% word0 +% brfc);
            self.cm_c[1].set((self.pb.x4 & 0xff0000ff) | shl(stream3b & 0xe07, 8));
            self.cm_c[1].set((self.pb.indirect_br_byte & 0xffff) | shl(stream3b & 0x38, 16));
        }
        // 5301-5302
        if (self.pb.is_math) {
            self.cm_c[0].sets();
        } else {
            self.cm_c[0].set((self.pb.indirect_byte & 0xff00) +% @as(u32, 257) *% self.pb.worcxt.word(1) *% 53 +% c1u);
        }
        // 5304-5305
        self.cm_c[2].set(shl(c1u, 8) | (self.pb.indirect_byte >> 2) | shl(fcu, 16));
        self.cm_c[2].set((c4 & 0xffff) +% @as(u32, @intFromBool(self.pb.c2 == self.pb.c3)));
        // 5307-5309
        self.cm_c4[3].set(((stream3b & self.pb.stream3b_mask) *% 256) | (stream2b & self.pb.stream2b_mask & 255));
        if (self.pb.skip_see_external) {
            self.cm_c4[3].sets();
        } else {
            self.cm_c4[3].set(self.pb.x4);
        }
        // 5311-5313
        self.cm_c2[9].set(@as(u32, 257) *% self.pb.pword_hash() +% @as(u32, self.pb.fccxt.cxt) +% @as(u32, 193) *% (stream3b & self.pb.stream3b_mask));
        self.cm_c2[9].set(fcu | shl(stream2b_r & 0xfff, 9) | shl(c1u, 24));
        // 5315-5317
        self.cm_c2[16].set(self.pb.worcxt.fword *% 83 +% (stream2b & 15) *% 11 +% @as(u32, self.pb.brcxt.cxt));
        if (skip_m1 or self.pb.skip_see_external) {
            self.cm_c2[17].sets();
        } else {
            self.cm_c2[17].set(self.pb.worcxt.last(1, VERB) +% self.pb.worcxt.word(1) *% 83 +% h);
        }
        // 5319
        self.cm_c2[9].set((self.pb.x4 & 0xffff00) +% @as(u32, self.pb.brcxt.cxt) +% shl(@as(u32, self.pb.fccxt.cxt), 24));
        // 5320-5333
        if (self.pb.linkword != 0) {
            self.cm_c2[9].set(self.pb.linkword);
        } else if (self.pb.is_math) {
            self.cm_c2[9].sets();
        } else if (self.pb.senword != 0) {
            self.cm_c2[9].set(self.pb.senword *% 1471 +% c1u);
        } else if (fc == HTML or @as(i32, self.pb.brcxt.cxt) == LESSTHAN) {
            self.cm_c2[9].sets();
        } else {
            self.cm_c2[9].set(0);
        }
        // 5335-5342: cmC2[10]
        self.cm_c2[10].set(self.pb.indirect_byte);
        self.cm_c2[10].set(((self.pb.indirect_byte & 0xffff00) >> 4) | (stream2b & self.pb.stream2b_mask & 0xf) | shl(stream3b & 0xfff, 20));
        self.cm_c2[10].set((self.pb.x4 >> 16) | shl(stream2b & 255, 24));
        if (c1 > 127) {
            self.cm_c2[10].set(shl(((stream2b & 12) *% 256) +% c1u, 11) | ((self.pb.indirect_word & 0xffffff) >> 16));
        } else {
            self.cm_c2[10].set(shl(c1u, 11) | shl(brfc, 8) | ((self.pb.indirect_word & 0xffffff) >> 16));
        }
        if (self.pb.is_math) {
            self.cm_c2[10].sets();
        } else {
            self.cm_c2[10].set((@as(u32, self.pb.fccxt.cxt) *% 4 +% brfc) | shl(c4 & 0xffff, 9) | shl(stream2b & 0xff, 24));
        }
        self.cm_c2[10].set((self.pb.indirect_word >> 16) | shl(stream2b & 0x3c, 25) | shl(stream3b & 0x1ff, 16));
        // 5344-5356: cmC2[11]
        {
            const words2: u32 = @as(u32, self.pb.words);
            const spaces2: u32 = @as(u32, self.pb.spaces);
            self.cm_c2[11].set(words2 +% shl(spaces2, 8) +% shl(stream2b & 15, 16) +% shl((stream3b_r >> 3) & 511, 21) +% shl(@as(u32, @bitCast(self.pb.is_paragraph)), 30));
        }
        self.cm_c2[11].set(c1u +% (shl(stream3b, 5) & 0x1fffff00));
        self.cm_c2[11].set(stream2b_r *% 16 +% brfc);
        if (@as(i32, self.pb.brcxt.cxt) == LESSTHAN) {
            self.cm_c2[11].sets();
        } else {
            self.cm_c2[11].set(((self.pb.indirect_byte & 0xffff) >> 8) +% ((@as(u32, 64) *% stream2b_r) & 0x3ffff00) +% shl(@as(u32, self.pb.brcxt.cxt), 25));
        }
        if ((@as(i32, self.pb.fccxt.cxt) == FIRSTUPPER and @as(i32, self.pb.brcxt.cxt) == SQUAREOPEN) or @as(i32, self.pb.brcxt.cxt) == LESSTHAN) {
            self.cm_c2[11].sets();
        } else {
            self.cm_c2[11].set(self.pb.indirect_word0_pos | shl(self.pb.indirect_byte & 0xff00, 16));
        }
        // 5358-5380: cmC2[12]
        if (@as(i32, self.pb.brcxt.cxt) == LESSTHAN or self.pb.is_category) {
            self.cm_c2[12].sets();
        } else {
            self.cm_c2[12].set((self.pb.x4 & 0x80f00000) +% shl(self.pb.x4 & 0x0000f0ff, 12));
        }
        if (@as(i32, self.pb.brcxt.cxt) == LESSTHAN) {
            self.cm_c2[12].set(h +% self.pb.worcxt3.word(1) *% 53 *% 79 +% self.pb.worcxt3.word(2) *% 53 *% 47 *% 71);
        } else if (self.pb.is_paragraph == 1) {
            if (c1 == ESCAPE or @as(i32, self.pb.fccxt.cxt) == HTLINK or @as(i32, self.pb.fccxt.cxt) == CURLYOPENING or self.pb.is_math or self.pb.is_pre) {
                self.cm_c2[12].sets();
            } else {
                self.cm_c2[12].set(h +% self.pb.worcxt.word(1) *% 53 *% 79 +% self.pb.worcxt.word(3) *% 53 *% 47 *% 71);
            }
        } else if (@as(i32, self.pb.fccxt.cxt) == HTLINK or @as(i32, self.pb.brcxt.cxt) == LESSTHAN or self.pb.htcxt.cxt != 0 or self.pb.is_category) {
            self.cm_c2[12].sets();
        } else if (col == 31) {
            self.cm_c2[12].set(shl(c4, 16));
        } else {
            self.cm_c2[12].set(@as(u32, @bitCast(self.pb.above)) | shl(c4 & 0xffff, 16) | shl(@as(u32, @bitCast(self.pb.above1)), 8));
        }
        // 5382-5397: cmC2[13]
        if (c1 == ESCAPE or self.pb.u8w_left != 0 or @as(i32, self.pb.fccxt.cxt) == CURLYOPENING or @as(i32, self.pb.fccxt.cxt) == HTLINK or fc == HTML or self.pb.htcxt.cxt != 0 or self.pb.is_category or fc == SPACE or self.pb.is_pre or c1 == '&' or @as(i32, self.pb.brcxt.cxt) == LESSTHAN or self.pb.is_math or col < 2 or (self.pb.worcxt.s_bytes(0) >> 8) == @as(u16, '\\')) {
            self.cm_c2[13].sets();
            self.cm_c2[13].sets();
        } else {
            self.cm_c2[13].set(self.pb.worcxt.word(1) *% 83 *% 1471 -% word0 *% 53 +% self.pb.worcxt.word(2));
            self.cm_c2[13].set(h +% self.pb.worcxt.word(2) *% 53 *% 79 +% self.pb.worcxt.word(3) *% 53 *% 47 *% 71);
        }
        // 5400-5402: cmC[3]
        self.cm_c[3].set(shl(stream3b_r & 7, 10) +% (stream2b & 3) +% fcu *% 4 +% shl(brfc, 24));
        {
            const lw = if (self.pb.linkword != 0) self.pb.linkword else word0;
            self.cm_c[3].set(lw *% 3301 +% self.pb.number0 *% 3191);
        }
        // 5403-5414: cmC2[14]
        if (c1 == ESCAPE or skip_m1 or self.pb.u8w_left != 0 or @as(i32, self.pb.fccxt.cxt) == CURLYOPENING or @as(i32, self.pb.fccxt.cxt) == HTLINK or fc == SPACE or fc == HTML or self.pb.is_category or @as(i32, self.pb.brcxt.cxt) == LESSTHAN or col < 2 or self.pb.is_math or (self.pb.worcxt.s_bytes(0) >> 8) == @as(u16, '\\')) {
            self.cm_c2[14].sets();
        } else {
            self.cm_c2[14].set(brfc +% self.pb.worcxt.word(2) *% (stream3b_r & self.pb.stream3b_r_mask2) +% (self.pb.worcxt.type_at(1) & 0x1ff));
        }
        // 5416-5424: cmC4[7]
        if (c1 == ESCAPE or self.pb.u8w_left != 0 or fc == SPACE or self.pb.skip_see_external) {
            for (0..4) |_| self.cm_c4[7].sets();
        } else {
            self.cm_c4[7].set(self.pb.worcxt1.word(1) +% word00);
            self.cm_c4[7].set(self.pb.worcxt.word(2) +% word0 *% 191 +% (stream3b_r & 63));
            self.cm_c4[7].set(word0 *% 191 +% (stream3b_r & 63));
            self.cm_c4[7].set((self.pb.indirect_word0_pos & 0xffff) *% 191 +% word0 +% (stream3b_r & 63));
        }
        // 5426-5431: cmCR
        if (@as(i32, self.pb.brcxt.cxt) == SQUAREOPEN) {
            self.cm_cr[0].set(self.pb.worcxt.word(1) +% word0);
        } else if (@as(i32, self.pb.fccxt.cxt) == HTLINK) {
            self.cm_cr[0].sets();
        } else {
            self.cm_cr[0].set(0);
        }
        self.cm_cr[1].set(shl(@as(u32, self.pb.fccxt.cxt), 15) | shl(stream3b & 7, 3) | shl(@as(u32, self.pb.brcxt.cxt), 24));
        self.cm_cr[2].set(hash3(self.pb.worcxt.word(1), stream2b & 0xFC, c1u));
        // 5433-5435: scmA
        self.scm_a[0].set(c1u);
        self.scm_a[1].set(stream3b & 0x1ff);
        self.scm_a[2].set(@as(u32, self.pb.brcxt.cxt));
        // 5437-5445
        self.pb.apply_wshift_block();
        // 5447
        self.cm_c2[15].set(brfc *% 256 +% fcu +% shl(stream3b_r & 0xFFF, 16));
        // 5449-5463: stream5b + sparse indirect contexts
        const w4 = shl(c4, 8) & 0xff000000;
        self.stream5b = shl(self.stream5b, 3) | (if (c1b > 127) @as(u32, WRT_3B['a']) else @as(u32, WRT_3B[c1b]));
        const buf2: usize = @intCast((c4 >> 8) & 0xff);
        const buf3: usize = @intCast((c4 >> 16) & 0xff);
        {
            const g = self.indirect2[c1b];
            self.indirect2[buf2] = c1b;
            self.cmcr[0].set(hash3(self.pb.indirect_br_byte, w4 | @as(u32, g), 0xffffffff));
        }
        {
            const g = self.indirect3[(buf2 << 8) | @as(usize, c1b)];
            self.indirect3[(buf3 << 8) | buf2] = c1b;
            self.cmcr[0].set(hash3(self.pb.indirect_br_byte, w4 | @as(u32, g), @as(u32, lastfc0)));
        }
        self.s5b_byte[@intCast((self.stream5b >> 3) & 7)] = c4 & 0xffffff;
        for (0..8) |i| {
            const k = self.s5b_byte[i];
            self.cmcr[i + 1].set(hash3(brfc +% (if (!self.pb.nest_list) word0 *% 191 else 0), k, @as(u32, @bitCast(self.pb.deccode >> 2))));
            self.cmcr[i + 1].set(hash3(brfc, self.stream5b & 0x7fff, k & 0xffff));
            self.cmcr[i + 1].set(hash3(brfc, shl(self.stream5b & 7, 8) | (k & 0xff00ff), @as(u32, lastfc0)));
        }
        self.s2_word0[@intCast(brfc *% 16 +% ((stream2b >> 2) & 15))] = word0;
        for (0..4) |i| {
            const k = self.s2_word0[@intCast(brfc *% 16 +% ((stream2b >> @as(u5, @intCast(2 * i))) & 15))];
            self.cmcr2[i].set(k);
            self.cmcr2[i].set(k +% h);
            self.cmcr2[i].set(k +% self.pb.worcxt.word(1));
        }
        // 5478-5480: APM hashes
        self.ah1 = hash3(self.pb.x5 & 255, (self.pb.x5 >> 8) & 255, (self.pb.x5 >> 16) & 0x80ff);
        self.ah2 = hash3(19, self.pb.x5 & 0x80ffff, 0xffffffff);
        // 5482-5483
        self.maps1.set(word0 *% 191);
        self.maps2.set(@as(u32, @bitCast(self.pb.deccode >> 2)));
}
