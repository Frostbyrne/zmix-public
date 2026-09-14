//! parseByte word/punctuation byte-driven core — word0/words/numbers/spaces and the
//! c1/c2 overrides, ported from cmix-lex `fxcmv1.cpp` parseByte (4467-4810).
//!
//! These are byte-driven: word0 = word0*2104 + c1 (letter/escape), reset to 0 at word
//! end; words/spaces/numbers shift left each byte then get bits set by classification/
//! punctuation; the c1/c2 overrides (`c2='.'` on EQUALS, `c1=SPACE` on "&!") carry
//! forward via the c1->c2->c3 shift.
//! NOTE: `j` is u32 — the letter test `(j-'a')<=25` is c1 in [97,122] via unsigned
//! underflow. words/spaces/numbers are u8; word0 is u32.

// post-WRT char constants
const ESCAPE: u32 = 12;
const SPACE: u32 = 32;
const LF: u32 = 10;
const EQUALS: u32 = 77;
const LESSTHAN: u32 = 76;
const COLON: u32 = 74;
const SEMICOLON: u32 = 75;
const QUESTION: u32 = 79;
const CURLYOPENING: u32 = 80;
const CURLYCLOSE: u32 = 82;
const SQUARECLOSE: u32 = 93;

pub const ParseWords = struct {
    c1: u32,
    c2: u32,
    c3: u32,
    word0: u32,
    words: u8,
    spaces: u8,
    numbers: u8,

    pub fn new() ParseWords {
        return ParseWords{ .c1 = 0, .c2 = 0, .c3 = 0, .word0 = 0, .words = 0, .spaces = 0, .numbers = 0 };
    }

    pub fn parse_byte(self: *ParseWords, byte: u8) void {
        self.c3 = self.c2;
        self.c2 = self.c1;
        self.c1 = @as(u32, byte);
        self.words = self.words << 1;
        self.spaces = self.spaces << 1;
        self.numbers = self.numbers << 1;
        const j = self.c1;
        if (j -% @as(u32, 'a') <= 25 or (self.c1 > 127 and self.c2 != ESCAPE)) {
            // word char
            self.words |= 1;
            self.word0 = self.word0 *% 2104 +% j;
        } else if (((self.c1 == ESCAPE and self.c2 != ESCAPE) or (self.c1 > 127 and self.c2 == ESCAPE)) and (self.words & 4) == 4) {
            if (self.c1 != ESCAPE) {
                self.words |= 3;
                self.word0 = self.word0 *% 2104 +% j;
            }
        } else {
            // non-word byte
            if (self.c1 >= @as(u32, '0') and self.c1 <= @as(u32, '9')) {
                self.numbers = self.numbers +% 1;
            }
            self.word0 = 0;
            // punctuation chain (only the word0/words/c1/c2 effects)
            if (self.c1 == SPACE) {
                self.spaces = self.spaces +% 1;
            } else if (self.c1 == LF) {
                self.words = 0xfc;
            } else if (self.c1 == @as(u32, '.') or self.c1 == @as(u32, ')') or self.c1 == QUESTION) {
                self.words |= 0xfe;
            } else if (self.c1 == @as(u32, ',')) {
                self.words |= 0xfc;
            } else if (self.c1 == @as(u32, '(')) {} else if (self.c1 == SEMICOLON) {} else if (self.c1 == COLON) {} else if (self.c1 == CURLYCLOSE or self.c1 == CURLYOPENING) {
                self.words |= 0xfc;
            } else if (self.c1 == SQUARECLOSE) {} else if (self.c1 == LESSTHAN or self.c2 == @as(u32, '&')) {
                self.words |= 0xfc;
            }
            // list-to-paragraph ('-' && lastfc=='*') omitted: only sets isParagraph
            else if (self.c1 == EQUALS) {
                self.c2 = @as(u32, '.'); // override (carries to next byte's c3)
                self.words = self.words *% 2;
            }
            // separate: "&!" -> c1=SPACE override (carries to next byte's c2)
            if (self.c1 == @as(u32, '!') and self.c2 == @as(u32, '&')) {
                self.c1 = SPACE;
            }
            // the else (lastfc=='*' ...) branch omitted: only sets isParagraph
        }
    }
};

const std = @import("std");

test "parse_words_matches_ctxdump_nodict" {
    const stream: []const u8 = @embedFile("goldens/stream.bin");
    try std.testing.expectEqual(@as(usize, 15007), stream.len);
    var pw = ParseWords.new();
    var cs: u64 = 0;
    for (stream) |byte| {
        pw.parse_byte(byte);
        cs = cs *% 1000003 +% @as(u64, pw.word0);
        cs = cs *% 1000003 +% @as(u64, pw.words);
        cs = cs *% 1000003 +% @as(u64, pw.numbers);
    }
    try std.testing.expectEqual(@as(u64, 0x0f2a25e345fd7ade), cs);
}
