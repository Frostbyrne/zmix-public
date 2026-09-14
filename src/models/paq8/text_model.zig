//! Faithful Zig port of cmix PAQ8's TextModel (reference/cmix-src/models/paq8.cpp,
//! ~3070-3518).  Builds on core.zig, maps.zig and text_primitives.zig.
//!
//! The model tracks a running parse of the text (words, segments, sentences,
//! paragraphs, per-language stemming, numbers, punctuation/nesting masks) and
//! feeds a 33-context ContextMap2 plus 8 direct mixer.set contexts.
//!
//! Faithfulness notes:
//!  - C `unsigned` wrap -> `+%`/`-%`/`*%`; truncating stores -> `@truncate`/`@intCast`.
//!  - paq8 hashconverts each argument to U64; every value passed here is
//!    non-negative so sign- vs zero-extension is irrelevant and the Zig arg
//!    types (u8/u32/u64/i32) reproduce the same U64.
//!  - The C++ `U64 i = State<<6; ... hash(i++, ...)` counter is unrolled to
//!    `base+0 .. base+21` (the 22 i++ uses appear in straight-line order).
//!  - The switch(c) fall-through chains ('.'->'?'/'!'->',;:' and NEW_LINE->
//!    whitespace) are reproduced with sequential guarded blocks.
const std = @import("std");
const core = @import("core.zig");
const maps = @import("maps.zig");
const tp = @import("text_primitives.zig");

const Word = tp.Word;
const Language = tp.Language;

// paq8.cpp:3051 const U8 AsciiGroup[128]
const AsciiGroup = [128]u8{
    0,  5,  5,  5,  5,  5,  5,  5,
    5,  5,  4,  5,  5,  4,  5,  5,
    5,  5,  5,  5,  5,  5,  5,  5,
    5,  5,  5,  5,  5,  5,  5,  5,
    6,  7,  8,  17, 17, 9,  17, 10,
    11, 12, 17, 17, 13, 14, 15, 16,
    1,  1,  1,  1,  1,  1,  1,  1,
    1,  1,  18, 19, 20, 23, 21, 22,
    23, 2,  2,  2,  2,  2,  2,  2,
    2,  2,  2,  2,  2,  2,  2,  2,
    2,  2,  2,  2,  2,  2,  2,  2,
    2,  2,  2,  24, 27, 25, 27, 26,
    27, 3,  3,  3,  3,  3,  3,  3,
    3,  3,  3,  3,  3,  3,  3,  3,
    3,  3,  3,  3,  3,  3,  3,  3,
    3,  3,  3,  28, 30, 29, 30, 30,
};

const TAB: u8 = 0x09;
const NEW_LINE: u8 = 0x0A;
const CARRIAGE_RETURN: u8 = 0x0D;
const SPACE: u8 = 0x20;

// Parse states (paq8 enum Parse)
const Parse = struct {
    const Unknown: u32 = 0;
    const ReadingWord: u32 = 1;
    const PossibleHyphenation: u32 = 2;
    const WasAbbreviation: u32 = 3;
    const AfterComma: u32 = 4;
    const AfterQuote: u32 = 5;
    const AfterAbbreviation: u32 = 6;
    const ExpectDigit: u32 = 7;
};

const LangState = struct {
    Count: [3]u32 = .{ 0, 0, 0 }, // recognized words per language (last 64 words)
    Mask: [3]u64 = .{ 0, 0, 0 }, // recognition status mask (last 64 words)
    Id: i32 = 0, // current detected language
    pId: i32 = 0, // language of the previous word
};

const InfoT = struct {
    numbers: [2]u64 = .{ 0, 0 },
    numHashes: [2]u64 = .{ 0, 0 },
    numLength: [2]u8 = .{ 0, 0 },
    numMask: u32 = 0,
    numDiff: u32 = 0,
    lastUpper: u32 = 0,
    maskUpper: u32 = 0,
    lastLetter: u32 = 0,
    lastDigit: u32 = 0,
    lastPunct: u32 = 0,
    lastNewLine: u32 = 0,
    prevNewLine: u32 = 0,
    wordGap: u32 = 0,
    spaces: u32 = 0,
    spaceCount: u32 = 0,
    commas: u32 = 0,
    quoteLength: u32 = 0,
    maskPunct: u32 = 0,
    nestHash: u32 = 0,
    lastNest: u32 = 0,
    asciiMask: u64 = 0,
    masks: [5]u32 = .{ 0, 0, 0, 0, 0 },
    wordLength: [2]u32 = .{ 0, 0 },
    UTF8Remaining: i32 = 0,
    firstLetter: u8 = 0,
    firstChar: u8 = 0,
    expectedDigit: u8 = 0,
    prevPunct: u8 = 0,
    TopicDescriptor: Word = .{},
};

inline fn tolower(c: u8) u8 {
    return if (c >= 'A' and c <= 'Z') c + 32 else c;
}
inline fn b2u(x: bool) u32 {
    return @intFromBool(x);
}
// u32-typed min/max so results are not narrowed (Zig @min/@max narrow to the
// comptime bound, which breaks subsequent shifts).
inline fn umin(a: u32, b: u32) u32 {
    return @min(a, b);
}
inline fn umax(a: u32, b: u32) u32 {
    return @max(a, b);
}

// language dispatch (the stemmers are stateless empty structs)
fn stemFor(id: i32, w: *Word) bool {
    return switch (id) {
        @as(i32, @intCast(Language.English)) => blk: {
            var s = tp.EnglishStemmer{};
            break :blk s.stem(w);
        },
        @as(i32, @intCast(Language.French)) => blk: {
            var s = tp.FrenchStemmer{};
            break :blk s.stem(w);
        },
        @as(i32, @intCast(Language.German)) => blk: {
            var s = tp.GermanStemmer{};
            break :blk s.stem(w);
        },
        else => false,
    };
}
fn isVowelFor(id: i32, c: u8) bool {
    return switch (id) {
        @as(i32, @intCast(Language.English)) => (tp.EnglishStemmer{}).isVowel(c),
        @as(i32, @intCast(Language.French)) => (tp.FrenchStemmer{}).isVowel(c),
        @as(i32, @intCast(Language.German)) => (tp.GermanStemmer{}).isVowel(c),
        else => false,
    };
}
fn isAbbreviationFor(id: i32, w: *const Word) bool {
    return switch (id) {
        @as(i32, @intCast(Language.English)) => tp.English.isAbbreviation(w),
        @as(i32, @intCast(Language.French)) => tp.French.isAbbreviation(w),
        @as(i32, @intCast(Language.German)) => tp.German.isAbbreviation(w),
        else => false,
    };
}

pub const TextModel = struct {
    const MIN_RECOGNIZED_WORDS: u32 = 4;

    Map: maps.ContextMap2,
    Words: [4]tp.Cache(Word, 8), // [Language::Count]
    Segments: tp.Cache(tp.Segment, 4),
    Sentences: tp.Cache(tp.Sentence, 4),
    Paragraphs: tp.Cache(tp.Paragraph, 2),
    WordPos: core.Array(u32, 0),
    BytePos: [256]u32,
    cWord: *Word,
    pWord: *Word,
    cSegment: *tp.Segment,
    cSentence: *tp.Sentence,
    cParagraph: *tp.Paragraph,
    State: u32,
    pState: u32,
    Lang: LangState,
    Info: InfoT,
    ParseCtx: u64,
    alloc: std.mem.Allocator,

    pub fn init(a: std.mem.Allocator, size: u64) *TextModel {
        const self = a.create(TextModel) catch unreachable;
        self.* = .{
            .Map = maps.ContextMap2.init(a, size, 33),
            .Words = .{ .{}, .{}, .{}, .{} },
            .Segments = .{},
            .Sentences = .{},
            .Paragraphs = .{},
            .WordPos = core.Array(u32, 0).initSize(a, 0x10000),
            .BytePos = [_]u32{0} ** 256,
            .cWord = undefined,
            .pWord = undefined,
            .cSegment = undefined,
            .cSentence = undefined,
            .cParagraph = undefined,
            .State = Parse.Unknown,
            .pState = Parse.Unknown,
            .Lang = .{},
            .Info = .{},
            .ParseCtx = 0,
            .alloc = a,
        };
        self.cWord = self.Words[@intCast(self.Lang.Id)].at(0);
        self.pWord = self.Words[@intCast(self.Lang.Id)].at(1);
        self.cSegment = self.Segments.at(0);
        self.cSentence = self.Sentences.at(0);
        self.cParagraph = self.Paragraphs.at(0);
        return self;
    }

    pub fn deinit(self: *TextModel) void {
        self.Map.deinit();
        self.WordPos.deinit();
        self.alloc.destroy(self);
    }

    fn update(self: *TextModel, stats: ?*core.ModelStats) void {
        const info = &self.Info;
        info.lastUpper = umin(0xFF, info.lastUpper + 1);
        info.maskUpper <<= 1;
        info.lastLetter = umin(0x1F, info.lastLetter + 1);
        info.lastDigit = umin(0xFF, info.lastDigit + 1);
        info.lastPunct = umin(0x3F, info.lastPunct + 1);
        info.lastNewLine +%= 1;
        info.prevNewLine +%= 1;
        info.lastNest +%= 1;
        info.spaceCount -%= (info.spaces >> 31);
        info.spaces <<= 1;
        info.masks[0] <<= 2;
        info.masks[1] <<= 2;
        info.masks[2] <<= 4;
        info.masks[3] <<= 3;
        self.pState = self.State;

        var c: u8 = @intCast(core.buf.get(1));
        var pC: u8 = tolower(c);
        const g: u8 = if (c < 0x80) AsciiGroup[c] else 31;
        if (!((g <= 4) and g == @as(u8, @intCast(info.asciiMask & 0x1f)))) {
            info.asciiMask = ((info.asciiMask << 5) | g) & ((@as(u64, 1) << 60) - 1);
        }
        info.masks[4] = @intCast(info.asciiMask & ((1 << 30) - 1));
        self.BytePos[c] = @bitCast(core.pos);
        if (c != pC) {
            c = pC;
            info.lastUpper = 0;
            info.maskUpper |= 1;
        }
        pC = @intCast(core.buf.get(2));
        self.State = Parse.Unknown;
        self.ParseCtx = maps.hash(.{
            self.State,
            self.pWord.Hash[1],
            c,
            (maps.ilog2(info.lastNewLine) + 1) * b2u(info.lastNewLine *% 3 > info.prevNewLine),
            info.masks[1] & 0xFC,
        });

        if ((c >= 'a' and c <= 'z') or c == '\'' or c == '-' or c > 0x7F) {
            if (info.wordLength[0] == 0) {
                // check for hyphenation with "+"
                if (pC == NEW_LINE and ((info.lastLetter == 3 and core.buf.get(3) == '+') or
                    (info.lastLetter == 4 and core.buf.get(3) == CARRIAGE_RETURN and core.buf.get(4) == '+')))
                {
                    info.wordLength[0] = info.wordLength[1];
                    var i: usize = @intCast(Language.Unknown);
                    while (i < Language.Count) : (i += 1) self.Words[i].dec();
                    self.cWord = self.pWord;
                    self.pWord = self.Words[@intCast(self.Lang.pId)].at(1);
                    self.cWord.* = Word{};
                    var k: u32 = 0;
                    while (k < info.wordLength[0]) : (k += 1) {
                        self.cWord.plusEq(@intCast(core.buf.get(info.wordLength[0] - k + info.lastLetter)));
                    }
                    info.wordLength[1] = @intCast(self.pWord.length());
                    self.cSegment.WordCount -%= 1;
                    self.cSentence.WordCount -%= 1;
                } else {
                    info.wordGap = info.lastLetter;
                    info.firstLetter = c;
                }
            }
            info.lastLetter = 0;
            info.wordLength[0] += 1;
            info.masks[0] +%= if (self.Lang.Id != @as(i32, @intCast(Language.Unknown))) 1 + b2u(isVowelFor(self.Lang.Id, c)) else 1;
            info.masks[1] +%= 1;
            info.masks[3] +%= info.masks[0] & 3;
            if (c == '\'') {
                info.masks[2] +%= 12;
                if (info.wordLength[0] == 1) {
                    if (info.quoteLength == 0 and pC == SPACE) {
                        info.quoteLength = 1;
                    } else if (info.quoteLength > 0 and info.lastPunct == 1) {
                        info.quoteLength = 0;
                        self.State = Parse.AfterQuote;
                        self.ParseCtx = maps.hash(.{ self.State, pC });
                    }
                }
            }
            self.cWord.plusEq(c);
            self.cWord.getHashes();
            self.State = Parse.ReadingWord;
            self.ParseCtx = maps.hash(.{ self.State, self.cWord.Hash[1] });
        } else {
            if (self.cWord.length() > 0) {
                if (self.Lang.Id != @as(i32, @intCast(Language.Unknown)))
                    self.Words[@intCast(Language.Unknown)].at(0).* = self.cWord.*;

                var i: i32 = @as(i32, @intCast(Language.Count)) - 1;
                while (i > @as(i32, @intCast(Language.Unknown))) : (i -= 1) {
                    const li: usize = @intCast(i - 1);
                    self.Lang.Count[li] -%= @intCast(self.Lang.Mask[li] >> 63);
                    self.Lang.Mask[li] <<= 1;
                    if (i != self.Lang.Id)
                        self.Words[@intCast(i)].at(0).* = self.cWord.*;
                    if (stemFor(i, self.Words[@intCast(i)].at(0))) {
                        self.Lang.Count[li] +%= 1;
                        self.Lang.Mask[li] |= 1;
                    }
                }
                self.Lang.Id = @intCast(Language.Unknown);
                var best: u32 = MIN_RECOGNIZED_WORDS;
                i = @as(i32, @intCast(Language.Count)) - 1;
                while (i > @as(i32, @intCast(Language.Unknown))) : (i -= 1) {
                    const li: usize = @intCast(i - 1);
                    if (self.Lang.Count[li] >= best) {
                        best = self.Lang.Count[li] + b2u(i == self.Lang.pId); // bias to previous language
                        self.Lang.Id = i;
                    }
                    self.Words[@intCast(i)].inc();
                }
                self.Words[@intCast(Language.Unknown)].inc();
                self.Lang.pId = self.Lang.Id;
                self.pWord = self.Words[@intCast(self.Lang.Id)].at(1);
                self.cWord = self.Words[@intCast(self.Lang.Id)].at(0);
                self.cWord.* = Word{};
                self.WordPos.at(@intCast(self.pWord.Hash[1] & (self.WordPos.size() - 1))).* = @bitCast(core.pos);
                if (self.cSegment.WordCount == 0)
                    self.cSegment.FirstWord = self.pWord.*;
                self.cSegment.WordCount +%= 1;
                if (self.cSentence.WordCount == 0)
                    self.cSentence.FirstWord = self.pWord.*;
                self.cSentence.WordCount +%= 1;
                info.wordLength[1] = info.wordLength[0];
                info.wordLength[0] = 0;
                info.quoteLength +%= b2u(info.quoteLength > 0);
                if (info.quoteLength > 0x1F) info.quoteLength = 0;
                self.cSentence.VerbIndex +%= 1;
                self.cSentence.NounIndex +%= 1;
                self.cSentence.CapitalIndex +%= 1;
                if ((self.pWord.Type & Language.Verb) != 0) {
                    self.cSentence.VerbIndex = 0;
                    self.cSentence.lastVerb = self.pWord.*;
                }
                if ((self.pWord.Type & Language.Noun) != 0) {
                    self.cSentence.NounIndex = 0;
                    self.cSentence.lastNoun = self.pWord.*;
                }
                if (self.cSentence.WordCount > 1 and info.lastUpper < info.wordLength[1]) {
                    self.cSentence.CapitalIndex = 0;
                    self.cSentence.lastCapital = self.pWord.*;
                }
            }
            var skip = false;
            // switch(c) with fall-through chains emulated by guarded sequential blocks
            sw: {
                // case '.'
                if (c == '.') {
                    if (self.Lang.Id != @as(i32, @intCast(Language.Unknown)) and info.lastUpper == info.wordLength[1] and
                        isAbbreviationFor(self.Lang.Id, self.pWord))
                    {
                        self.State = Parse.WasAbbreviation;
                        self.ParseCtx = maps.hash(.{ self.State, self.pWord.Hash[1] });
                        break :sw;
                    }
                    // else fall through to '?'/'!'
                }
                // case '?' case '!'
                if (c == '.' or c == '?' or c == '!') {
                    self.cSentence.Type = if (c == '.') .Declarative else if (c == '?') .Interrogative else .Exclamative;
                    self.cSentence.SegmentCount +%= 1;
                    self.cParagraph.SentenceCount +%= 1;
                    self.cParagraph.TypeCount[@intFromEnum(self.cSentence.Type)] +%= 1;
                    self.cParagraph.TypeMask <<= 2;
                    self.cParagraph.TypeMask |= @intFromEnum(self.cSentence.Type);
                    self.cSentence = self.Sentences.next();
                    info.masks[3] +%= 3;
                    skip = true;
                    // fall through to ',',';',':'
                }
                // case ',' case ';' case ':'
                if (c == '.' or c == '?' or c == '!' or c == ',' or c == ';' or c == ':') {
                    if (c == ',') {
                        info.commas +%= 1;
                        self.State = Parse.AfterComma;
                        self.ParseCtx = maps.hash(.{
                            self.State,
                            maps.ilog2(info.quoteLength + 1),
                            maps.ilog2(info.lastNewLine),
                            b2u(info.lastUpper < info.lastLetter +% info.wordLength[1]),
                        });
                    } else if (c == ':') {
                        info.TopicDescriptor = self.pWord.*;
                    }
                    if (!skip) {
                        self.cSentence.SegmentCount +%= 1;
                        info.masks[3] +%= 4;
                    }
                    info.lastPunct = 0;
                    info.prevPunct = c;
                    info.masks[0] +%= 3;
                    info.masks[1] +%= 2;
                    info.masks[2] +%= 15;
                    self.cSegment = self.Segments.next();
                    break :sw;
                }
                // case NEW_LINE
                if (c == NEW_LINE) {
                    info.prevNewLine = info.lastNewLine;
                    info.lastNewLine = 0;
                    info.commas = 0;
                    if (info.prevNewLine == 1 or (info.prevNewLine == 2 and pC == CARRIAGE_RETURN))
                        self.cParagraph = self.Paragraphs.next()
                    else if ((info.lastLetter == 2 and pC == '+') or (info.lastLetter == 3 and pC == CARRIAGE_RETURN and core.buf.get(3) == '+')) {
                        self.ParseCtx = maps.hash(.{ Parse.ReadingWord, self.pWord.Hash[1] });
                        self.State = Parse.PossibleHyphenation;
                    }
                    // fall through to whitespace
                }
                // case TAB case CARRIAGE_RETURN case SPACE
                if (c == NEW_LINE or c == TAB or c == CARRIAGE_RETURN or c == SPACE) {
                    info.spaceCount +%= 1;
                    info.spaces |= 1;
                    info.masks[1] +%= 3;
                    info.masks[3] +%= 5;
                    if (c == SPACE and self.pState == Parse.WasAbbreviation) {
                        self.State = Parse.AfterAbbreviation;
                        self.ParseCtx = maps.hash(.{ self.State, self.pWord.Hash[1] });
                    }
                    break :sw;
                }
                // remaining single-character cases
                switch (c) {
                    '(' => {
                        info.masks[2] +%= 1;
                        info.masks[3] +%= 6;
                        info.nestHash +%= 31;
                        info.lastNest = 0;
                    },
                    '[' => {
                        info.masks[2] +%= 2;
                        info.nestHash +%= 11;
                        info.lastNest = 0;
                    },
                    '{' => {
                        info.masks[2] +%= 3;
                        info.nestHash +%= 17;
                        info.lastNest = 0;
                    },
                    '<' => {
                        info.masks[2] +%= 4;
                        info.nestHash +%= 23;
                        info.lastNest = 0;
                    },
                    0xAB => info.masks[2] +%= 5,
                    ')' => {
                        info.masks[2] +%= 6;
                        info.nestHash -%= 31;
                        info.lastNest = 0;
                    },
                    ']' => {
                        info.masks[2] +%= 7;
                        info.nestHash -%= 11;
                        info.lastNest = 0;
                    },
                    '}' => {
                        info.masks[2] +%= 8;
                        info.nestHash -%= 17;
                        info.lastNest = 0;
                    },
                    '>' => {
                        info.masks[2] +%= 9;
                        info.nestHash -%= 23;
                        info.lastNest = 0;
                    },
                    0xBB => info.masks[2] +%= 10,
                    '"' => {
                        info.masks[2] +%= 11;
                        // start/stop counting
                        if (info.quoteLength == 0) {
                            info.quoteLength = 1;
                        } else {
                            info.quoteLength = 0;
                            self.State = Parse.AfterQuote;
                            self.ParseCtx = maps.hash(.{ self.State, @as(u32, 0x100) | @as(u32, pC) });
                        }
                    },
                    '/', '-', '+', '*', '=', '%' => info.masks[2] +%= 13,
                    '\\', '|', '_', '@', '&', '^' => info.masks[2] +%= 14,
                    else => {},
                }
            }

            if (c >= '0' and c <= '9') {
                info.numbers[0] = info.numbers[0] *% 10 +% (c & 0xF);
                info.numLength[0] = @intCast(umin(19, @as(u32, info.numLength[0]) + 1));
                info.numHashes[0] = maps.combine64(info.numHashes[0], c);
                info.expectedDigit = 0xFF; // -1
                if (info.numLength[0] < info.numLength[1] and
                    (self.pState == Parse.ExpectDigit or ((info.numDiff & 3) == 0 and info.numLength[0] <= 1)))
                {
                    const ExpectedNum: u64 = info.numbers[1] +% (info.numMask & 3) -% 2;
                    var PlaceDivisor: u64 = 1;
                    var k: i32 = 0;
                    while (k < @as(i32, info.numLength[1]) - @as(i32, info.numLength[0])) : (k += 1) PlaceDivisor *%= 10;
                    if (ExpectedNum / PlaceDivisor == info.numbers[0]) {
                        PlaceDivisor /= 10;
                        info.expectedDigit = @intCast((ExpectedNum / PlaceDivisor) % 10);
                        self.State = Parse.ExpectDigit;
                    }
                } else {
                    const d: u8 = @intCast(core.buf.get(@as(u32, info.numLength[0]) + 2));
                    if (info.numLength[0] < 3 and core.buf.get(@as(u32, info.numLength[0]) + 1) == ',' and d >= '0' and d <= '9')
                        self.State = Parse.ExpectDigit;
                }
                info.lastDigit = 0;
                info.masks[3] +%= 7;
            } else if (info.numbers[0] > 0) {
                info.numMask <<= 2;
                info.numMask |= 1 + b2u(info.numbers[0] >= info.numbers[1]) + b2u(info.numbers[0] > info.numbers[1]);
                info.numDiff <<= 2;
                info.numDiff |= umin(3, maps.ilog2(@abs(@as(i32, @bitCast(@as(u32, @truncate(info.numbers[0] -% info.numbers[1])))))));
                info.numbers[1] = info.numbers[0];
                info.numbers[0] = 0;
                info.numHashes[1] = info.numHashes[0];
                info.numHashes[0] = 0;
                info.numLength[1] = info.numLength[0];
                info.numLength[0] = 0;
                self.cSegment.NumCount +%= 1;
                self.cSentence.NumCount +%= 1;
            }
        }
        if (info.lastNewLine == 1)
            info.firstChar = if (self.Lang.Id != @as(i32, @intCast(Language.Unknown))) c else @min(c, 96);
        if (info.lastNest > 512)
            info.nestHash = 0;
        var leadingBitsSet: i32 = 0;
        while (((@as(i32, c) >> @intCast((7 - leadingBitsSet) & 31)) & 1) != 0) leadingBitsSet += 1;

        if (info.UTF8Remaining > 0 and leadingBitsSet == 1)
            info.UTF8Remaining -= 1
        else
            info.UTF8Remaining = if (leadingBitsSet != 1)
                (if (c != 0xC0 and c != 0xC1 and c < 0xF5) (leadingBitsSet - b2u_i(leadingBitsSet > 0)) else -1)
            else
                0;
        info.maskPunct = b2u(self.BytePos[','] > self.BytePos['.']) |
            (b2u(self.BytePos[','] > self.BytePos['!']) << 1) |
            (b2u(self.BytePos[','] > self.BytePos['?']) << 2) |
            (b2u(self.BytePos[','] > self.BytePos[':']) << 3) |
            (b2u(self.BytePos[','] > self.BytePos[';']) << 4);
        if (stats) |s| {
            s.Text.state = @intCast(self.State);
            s.Text.lastPunct = @intCast(umin(0x1F, info.lastPunct));
            s.Text.wordLength = @intCast(umin(0xF, info.wordLength[0]));
            s.Text.boolmask = @intCast((b2u(info.lastDigit < info.wordLength[0] +% info.wordGap)) |
                (b2u(info.lastUpper < info.lastLetter +% info.wordLength[1]) << 1) |
                (b2u(info.lastPunct < info.wordLength[0] +% info.wordGap) << 2) |
                (b2u(info.lastUpper < info.wordLength[0]) << 3));
            s.Text.firstLetter = info.firstLetter;
            s.Text.mask = @intCast(info.masks[1] & 0xFF);
        }
    }

    fn setContexts(self: *TextModel) void {
        const info = &self.Info;
        const c: u8 = @intCast(core.buf.get(1));
        const lc: u8 = tolower(c);
        const m2: u32 = info.masks[2] & 0xF;
        const column: u32 = umin(0xFF, info.lastNewLine);
        const w: u16 = @truncate((if (self.State == Parse.ReadingWord) self.cWord.Hash[1] else self.pWord.Hash[1]) & 0xFFFF);
        const h: u32 = @truncate((if (self.State == Parse.ReadingWord) self.cWord.Hash[1] else self.pWord.Hash[2]) *% 271 +% c);
        const base: u64 = @as(u64, self.State) << 6;
        const pid: usize = @intCast(self.Lang.pId);

        self.Map.set(self.ParseCtx);
        self.Map.set(maps.hash(.{ base + 0, self.cWord.Hash[0], self.pWord.Hash[0], b2u(info.lastUpper < info.wordLength[0]) | (b2u(info.lastDigit < info.wordLength[0] +% info.wordGap) << 1) }));
        self.Map.set(maps.hash(.{ base + 1, self.cWord.Hash[1], self.Words[pid].at(2).Hash[1], umin(10, maps.ilog2(@as(u32, @truncate(info.numbers[0])))), b2u(info.lastUpper < info.lastLetter +% info.wordLength[1]) | (b2u(info.lastLetter > 3) << 1) | (b2u(info.lastLetter > 0 and info.wordLength[1] < 3) << 2) }));
        self.Map.set(maps.hash(.{ base + 2, self.cWord.Hash[1] & 0xFFF, info.masks[1] & 0x3FF, self.Words[pid].at(3).Hash[2], b2u(info.lastDigit < info.wordLength[0] +% info.wordGap) | (b2u(info.lastUpper < info.lastLetter +% info.wordLength[1]) << 1) | ((info.spaces & 0x7F) << 2) }));
        self.Map.set(maps.hash(.{ base + 3, self.cWord.Hash[1], self.pWord.Hash[3], self.Words[pid].at(2).Hash[3] }));
        self.Map.set(maps.hash(.{ base + 4, @as(u32, h) & 0x7FFF, self.Words[pid].at(2).Hash[1] & 0xFFF, self.Words[pid].at(3).Hash[1] & 0xFFF }));
        self.Map.set(maps.hash(.{ base + 5, self.cWord.Hash[1], c, if (self.cSentence.VerbIndex < self.cSentence.WordCount) self.cSentence.lastVerb.Hash[1] else @as(u64, 0) }));
        self.Map.set(maps.hash(.{ base + 6, self.pWord.Hash[2], info.masks[1] & 0xFC, lc, info.wordGap }));
        self.Map.set(maps.hash(.{ base + 7, if (info.lastLetter == 0) self.cWord.Hash[1] else self.pWord.Hash[1], c, self.cSegment.FirstWord.Hash[2], umin(3, maps.ilog2(self.cSegment.WordCount + 1)) }));
        self.Map.set(maps.hash(.{ base + 8, self.cWord.Hash[1], c, self.Segments.at(1).FirstWord.Hash[3] }));
        self.Map.set(maps.hash(.{ base + 9, umax(31, @as(u32, lc)), info.masks[1] & 0xFFC, (info.spaces & 0xFE) | b2u(info.lastPunct < info.lastLetter), (info.maskUpper & 0xFF) | (((@as(u32, 0x100) | @as(u32, info.firstLetter)) * b2u(info.wordLength[0] > 1)) << 8) }));
        self.Map.set(maps.hash(.{ base + 10, column, umin(7, maps.ilog2(info.lastUpper + 1)), maps.ilog2(info.lastPunct + 1) }));
        self.Map.set((column & 0xF8) | (info.masks[1] & 3) | (b2u(info.prevNewLine -% info.lastNewLine > 63) << 2) |
            (umin(3, info.lastLetter) << 8) |
            (@as(u32, info.firstChar) << 10) |
            (b2u(info.commas > 4) << 18) |
            (b2u(m2 >= 1 and m2 <= 5) << 19) |
            (b2u(m2 >= 6 and m2 <= 10) << 20) |
            (b2u(m2 == 11 or m2 == 12) << 21) |
            (b2u(info.lastUpper < column) << 22) |
            (b2u(info.lastDigit < column) << 23) |
            (b2u(column < info.prevNewLine -% info.lastNewLine) << 24));
        self.Map.set(maps.hash(.{
            (2 * column) / 3,
            umin(13, info.lastPunct) + b2u(info.lastPunct > 16) + b2u(info.lastPunct > 32) + info.maskPunct * 16,
            maps.ilog2(info.lastUpper + 1),
            maps.ilog2(info.prevNewLine -% info.lastNewLine),
            b2u((info.masks[1] & 3) == 0) | (b2u(m2 < 6) << 1) | (b2u(m2 < 11) << 2),
        }));
        self.Map.set(maps.hash(.{ base + 11, column >> 1, info.spaces & 0xF }));
        self.Map.set(maps.hash(.{
            info.masks[3] & 0x3F,
            @min((umax(info.wordLength[0], 3) - 2) * b2u(info.wordLength[0] < 8), 3),
            @as(u32, info.firstLetter) * b2u(info.wordLength[0] < 5),
            @as(u32, w) & 0x3FF,
            b2u(@as(i32, c) == core.buf.get(2)) | (b2u(info.masks[2] > 0) << 1) | (b2u(info.lastPunct < info.wordLength[0] +% info.wordGap) << 2) | (b2u(info.lastUpper < info.wordLength[0]) << 3) | (b2u(info.lastDigit < info.wordLength[0] +% info.wordGap) << 4) | (b2u(info.lastPunct < 2 +% info.wordLength[0] +% info.wordGap +% info.wordLength[1]) << 5),
        }));
        self.Map.set(maps.hash(.{ base + 12, w, c, info.numHashes[1] }));
        self.Map.set(maps.hash(.{ base + 13, w, c, core.llog(@as(u32, @bitCast(core.pos)) -% self.WordPos.get(w)) >> 1 }));
        self.Map.set(maps.hash(.{ base + 14, w, c, info.TopicDescriptor.Hash[1] & 0x7FFF }));
        self.Map.set(maps.hash(.{ base + 15, info.numLength[0], c, info.TopicDescriptor.Hash[1] & 0x7FFF }));
        self.Map.set(maps.hash(.{ base + 16, if (info.lastLetter > 0) @as(u32, c) else @as(u32, 0x100), info.masks[1] & 0xFFC, info.nestHash & 0x7FF }));
        self.Map.set(maps.hash(.{
            base + 17,
            @as(u32, w) *% 17 +% c,
            info.masks[3] & 0x1FF,
            (b2u(self.cSentence.VerbIndex == 0 and self.cSentence.lastVerb.length() > 0) << 6) |
                (b2u(info.wordLength[1] > 3) << 5) |
                (b2u(self.cSegment.WordCount == 0) << 4) |
                (b2u(self.cSentence.SegmentCount == 0 and self.cSentence.WordCount < 2) << 3) |
                (b2u(info.lastPunct >= info.lastLetter +% info.wordLength[1] +% info.wordGap) << 2) |
                (b2u(info.lastUpper < info.lastLetter +% info.wordLength[1]) << 1) |
                b2u(info.lastUpper < info.wordLength[0] +% info.wordGap +% info.wordLength[1]),
        }));
        self.Map.set(maps.hash(.{
            base + 18,
            c,
            self.pWord.Hash[2],
            @as(u32, info.firstLetter) * b2u(info.wordLength[0] < 6),
            (b2u(info.lastPunct < info.wordLength[0] +% info.wordGap) << 1) | b2u(info.lastPunct >= info.lastLetter +% info.wordLength[1] +% info.wordGap),
        }));
        {
            const wp = self.Words[pid].at(1 + b2u(info.wordLength[0] == 0));
            self.Map.set(maps.hash(.{ base + 19, @as(u32, w) *% 23 +% c, wp.Letters[wp.Start], @as(u32, info.firstLetter) * b2u(info.wordLength[0] < 7) }));
        }
        self.Map.set(maps.hash(.{ base + 20, column, info.spaces & 7, info.nestHash & 0x7FF }));
        self.Map.set(maps.hash(.{ base + 21, self.cWord.Hash[1], b2u(info.lastUpper < column) | (b2u(info.lastUpper < info.wordLength[0]) << 1), umin(5, info.wordLength[0]) }));
        self.Map.set(info.masks[4]); // last 6 groups
        self.Map.set(maps.hash(.{ @as(u32, @truncate(info.asciiMask)), @as(u32, @truncate(info.asciiMask >> 32)) })); // last 12 groups
        self.Map.set(info.asciiMask & ((1 << 20) - 1)); // last 4 groups
        self.Map.set(info.asciiMask & ((1 << 10) - 1)); // last 2 groups
        self.Map.set(maps.hash(.{ (info.asciiMask >> 5) & ((1 << 30) - 1), core.buf.get(1) }));
        self.Map.set(maps.hash(.{ (info.asciiMask >> 10) & ((1 << 30) - 1), core.buf.get(1), core.buf.get(2) }));
        self.Map.set(maps.hash(.{ (info.asciiMask >> 15) & ((1 << 30) - 1), core.buf.get(1), core.buf.get(2), core.buf.get(3) }));
    }

    pub fn predict(self: *TextModel, mixer: *core.Mixer, stats: ?*core.ModelStats) void {
        const info = &self.Info;
        if (core.bpos == 0) {
            self.update(stats);
            self.setContexts();
        }
        _ = self.Map.mix(mixer);
        mixer.set(@intCast(maps.finalize64(maps.hash(.{
            if (self.Lang.Id != @as(i32, @intCast(Language.Unknown))) 1 + b2u(isVowelFor(self.Lang.Id, @intCast(core.buf.get(1)))) else @as(u32, 0),
            info.masks[1] & 0xFF,
            core.c0,
        }), 11)), 2048);
        mixer.set(@intCast(maps.finalize64(maps.hash(.{
            maps.ilog2(info.wordLength[0] + 1),
            core.c0,
            b2u(info.lastDigit < info.wordLength[0] +% info.wordGap) |
                (b2u(info.lastUpper < info.lastLetter +% info.wordLength[1]) << 1) |
                (b2u(info.lastPunct < info.wordLength[0] +% info.wordGap) << 2) |
                (b2u(info.lastUpper < info.wordLength[0]) << 3),
        }), 11)), 2048);
        mixer.set(@intCast(maps.finalize64(maps.hash(.{
            info.masks[1] & 0x3FF,
            core.grp0,
            b2u(info.lastUpper < info.wordLength[0]),
            b2u(info.lastUpper < info.lastLetter +% info.wordLength[1]),
        }), 12)), 4096);
        mixer.set(@intCast(maps.finalize64(maps.hash(.{
            info.spaces & 0x1FF,
            core.grp0,
            b2u(info.lastUpper < info.wordLength[0]) |
                (b2u(info.lastUpper < info.lastLetter +% info.wordLength[1]) << 1) |
                (b2u(info.lastPunct < info.lastLetter) << 2) |
                (b2u(info.lastPunct < info.wordLength[0] +% info.wordGap) << 3) |
                (b2u(info.lastPunct < info.lastLetter +% info.wordLength[1] +% info.wordGap) << 4),
        }), 12)), 4096);
        mixer.set(@intCast(maps.finalize64(maps.hash(.{
            @as(u32, info.firstLetter) * b2u(info.wordLength[0] < 4),
            umin(6, info.wordLength[0]),
            core.c0,
        }), 11)), 2048);
        mixer.set(@intCast(maps.finalize64(maps.hash(.{
            self.pWord.at(0),
            self.pWord.atEnd(0),
            umin(4, info.wordLength[0]),
            b2u(info.lastPunct < info.lastLetter),
        }), 11)), 2048);
        mixer.set(@intCast(maps.finalize64(maps.hash(.{
            umin(4, info.wordLength[0]),
            core.grp0,
            b2u(info.lastUpper < info.wordLength[0]),
            if (info.nestHash > 0) info.nestHash & 0xFF else @as(u32, 0x100) | (@as(u32, info.firstLetter) * b2u(info.wordLength[0] > 0 and info.wordLength[0] < 4)),
        }), 12)), 4096);
        mixer.set(@intCast(maps.finalize64(maps.hash(.{
            core.grp0,
            info.masks[4] & 0x1F,
            (info.masks[4] >> 5) & 0x1F,
        }), 13)), 8192);
    }
};

inline fn b2u_i(x: bool) i32 {
    return @intFromBool(x);
}

// ================================== tests ==================================
// Structural smoke test. Bit-exact behaviour is verified by a full differential
// test against the compiled C++ reference (see porter report).
const testing = std.testing;

test "TextModel runs; 8 sets/bit; nx in {0,231}; finite" {
    core.level = 6;
    core.init();
    core.resetState();
    const a = testing.allocator;
    core.buf.setsize(a, @intCast(core.MEM() * 8));
    defer core.buf.deinit();
    const model = TextModel.init(a, 1 << 20);
    defer model.deinit();
    const m = core.Mixer.init(a, 512, 0, 32, 0);
    defer m.deinit();
    var stats = core.ModelStats{};
    const data = "Mr. Smith said \"hello\" to Dr. Jones. Numbers 1 2 3, 100 200.\nNew line here.";
    for (data) |byte| {
        var k: i32 = 0;
        while (k < 8) : (k += 1) {
            core.y = (@as(i32, byte) >> @intCast(7 - k)) & 1;
            core.c0 += core.c0 + core.y;
            if (core.c0 >= 256) {
                core.buf.at(@bitCast(core.pos)).* = @truncate(@as(u32, @intCast(core.c0)));
                core.pos += 1;
                core.c0 -= 256;
                core.c4 = (core.c4 << 8) +% @as(u32, @intCast(core.c0));
                core.c0 = 1;
            }
            core.bpos = (core.bpos + 1) & 7;
            if (core.bpos > 0) {
                const idx: usize = @intCast((@as(i32, 1) << @intCast(core.bpos)) - 2 + (core.c0 & ((@as(i32, 1) << @intCast(core.bpos)) - 1)));
                core.grp0 = core.AsciiGroupC0[idx];
            } else core.grp0 = 0;
            core.resetPredictions();
            model.predict(m, &stats);
            try testing.expectEqual(@as(usize, 8), m.ncxt);
            try testing.expect(m.nx == 0 or m.nx == 231);
            for (core.model_predictions[0..core.prediction_index]) |p| try testing.expect(std.math.isFinite(p));
            while (m.nx & 7 != 0) {
                m.tx[m.nx] = 0;
                m.nx += 1;
            }
            m.update();
        }
    }
}
