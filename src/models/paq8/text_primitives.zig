//! Faithful Zig port of the self-contained TEXT PRIMITIVES of cmix's PAQ8 model.
//!
//! Source: reference/cmix-src/models/paq8.cpp, lines ~1537-3069:
//!   Word, Segment, Sentence, Paragraph, Language(+English/French/German),
//!   Stemmer(+EnglishStemmer/FrenchStemmer/GermanStemmer), Cache.
//!
//! These are pure word/string-processing classes with NO dependency on the
//! mixer/context-map infrastructure. TextModel (paq8.cpp:3070+) is NOT ported
//! here; it is done separately by the TextModel porter.
//!
//! Faithfulness notes:
//!  - C `unsigned` wraparound is reproduced with Zig's `*%` / `+%` / `-%`.
//!  - C char/byte equality is sign-agnostic, so `u8` is used throughout.
//!  - `tolower`/`toupper` are ASCII-only (matches glibc C locale for bytes
//!    0x00-0x7F; bytes 0x80-0xFF are returned unchanged, as in C locale).
//!  - All hash multipliers, region logic, suffix tables and exception lists
//!    are copied verbatim from the C++ and cross-checked against a compiled
//!    C++ reference harness (see tests below for exact expected outputs).

const std = @import("std");

pub const U8 = u8;
pub const U16 = u16;
pub const U32 = u32;
pub const U64 = u64;

pub const MAX_WORD_SIZE: usize = 64;

// ---------------------------------------------------------------------------
// hash— the multiply-add hash used by Word::GetHashes (paq8.cpp:714-770).
// Only the 2- and 3-argument overloads are used by the text primitives.
// ---------------------------------------------------------------------------
const PHI64: u64 = 0x9E3779B97F4A7C15;
const MUL64_1: u64 = 0x993DDEFFB1462949;
const MUL64_2: u64 = 0xE9C91DC159AB0D2D;

inline fn hash2(x0: u64, x1: u64) u64 {
    return (x0 +% 1) *% PHI64 +% (x1 +% 1) *% MUL64_1;
}
inline fn hash3(x0: u64, x1: u64, x2: u64) u64 {
    return (x0 +% 1) *% PHI64 +% (x1 +% 1) *% MUL64_1 +% (x2 +% 1) *% MUL64_2;
}

// ---------------------------------------------------------------------------
// Small char helpers (paq8.cpp:1537 CharInArray, ctype tolower/toupper).
// ---------------------------------------------------------------------------
inline fn charInArray(c: u8, a: []const u8) bool {
    for (a) |x| {
        if (c == x) return true;
    }
    return false;
}
inline fn tolower(c: u8) u8 {
    return if (c >= 'A' and c <= 'Z') c + 32 else c;
}
inline fn toupper(c: u8) u8 {
    return if (c >= 'a' and c <= 'z') c - 32 else c;
}
inline fn boolByte(x: bool) u8 {
    return @intFromBool(x);
}

// ---------------------------------------------------------------------------
// Language ids and per-language flag bits (paq8.cpp:1652-1725).
// ---------------------------------------------------------------------------
pub const Language = struct {
    pub const Unknown: u64 = 0;
    pub const English: u64 = 1;
    pub const French: u64 = 2;
    pub const German: u64 = 3;
    pub const Count: u64 = 4;

    pub const Verb: u64 = 1 << 0;
    pub const Noun: u64 = 1 << 1;
};

/// English::Flags (paq8.cpp:1674-1698)
pub const English = struct {
    pub const Verb: u64 = 1 << 0;
    pub const Noun: u64 = 1 << 1;
    pub const Adjective: u64 = 1 << 2;
    pub const Plural: u64 = 1 << 3;
    pub const Male: u64 = 1 << 4;
    pub const Female: u64 = 1 << 5;
    pub const Negation: u64 = 1 << 6;
    pub const PastTense: u64 = (1 << 7) | Verb;
    pub const PresentParticiple: u64 = (1 << 8) | Verb;
    pub const AdjectiveSuperlative: u64 = (1 << 9) | Adjective;
    pub const AdjectiveWithout: u64 = (1 << 10) | Adjective;
    pub const AdjectiveFull: u64 = (1 << 11) | Adjective;
    pub const AdverbOfManner: u64 = 1 << 12;
    pub const SuffixNESS: u64 = 1 << 13;
    pub const SuffixITY: u64 = (1 << 14) | Noun;
    pub const SuffixCapable: u64 = 1 << 15;
    pub const SuffixNCE: u64 = 1 << 16;
    pub const SuffixNT: u64 = 1 << 17;
    pub const SuffixION: u64 = 1 << 18;
    pub const SuffixAL: u64 = (1 << 19) | Adjective;
    pub const SuffixIC: u64 = (1 << 20) | Adjective;
    pub const SuffixIVE: u64 = 1 << 21;
    pub const SuffixOUS: u64 = (1 << 22) | Adjective;
    pub const PrefixOver: u64 = 1 << 23;
    pub const PrefixUnder: u64 = 1 << 24;

    const abbreviations = [_][]const u8{ "mr", "mrs", "ms", "dr", "st", "jr" };
    pub fn isAbbreviation(w: *const Word) bool {
        return w.matchesAny(&abbreviations);
    }
};

/// French::Flags (paq8.cpp:1707-1710)
pub const French = struct {
    pub const Verb: u64 = 1 << 0;
    pub const Noun: u64 = 1 << 1;
    pub const Adjective: u64 = 1 << 2;
    pub const Plural: u64 = 1 << 3;

    const abbreviations = [_][]const u8{ "m", "mm" };
    pub fn isAbbreviation(w: *const Word) bool {
        return w.matchesAny(&abbreviations);
    }
};

/// German::Flags (paq8.cpp:1719-1723)
pub const German = struct {
    pub const Verb: u64 = 1 << 0;
    pub const Noun: u64 = 1 << 1;
    pub const Adjective: u64 = 1 << 2;
    pub const Plural: u64 = 1 << 3;
    pub const Female: u64 = 1 << 4;

    const abbreviations = [_][]const u8{ "fr", "hr", "hrn" };
    pub fn isAbbreviation(w: *const Word) bool {
        return w.matchesAny(&abbreviations);
    }
};

// ---------------------------------------------------------------------------
// Word (paq8.cpp:1547-1622)
// ---------------------------------------------------------------------------
pub const Word = struct {
    Letters: [MAX_WORD_SIZE]u8 = [_]u8{0} ** MAX_WORD_SIZE,
    Start: u8 = 0,
    End: u8 = 0,
    Hash: [4]u64 = .{ 0, 0, 0, 0 },
    Type: u64 = 0,
    Language: u64 = 0,

    /// operator==(const char*) : compares the active [Start..End] letters.
    pub fn eq(self: *const Word, s: []const u8) bool {
        const extra: usize = if (self.Letters[self.Start] != 0) 1 else 0;
        const cnt = @as(usize, self.End) - @as(usize, self.Start) + extra;
        if (cnt != s.len) return false;
        return std.mem.eql(u8, self.Letters[self.Start .. self.Start + s.len], s);
    }
    pub fn neq(self: *const Word, s: []const u8) bool {
        return !self.eq(s);
    }

    /// operator+=(char) : append one lowercased letter.
    pub fn plusEq(self: *Word, c: u8) void {
        if (self.End < MAX_WORD_SIZE - 1) {
            self.End +%= boolByte(self.Letters[self.End] > 0);
            self.Letters[self.End] = tolower(c);
        }
    }

    /// operator[](i) : i-th letter counted from Start (0-based), else 0.
    pub fn at(self: *const Word, i: u8) u8 {
        return if (@as(i32, self.End) - @as(i32, self.Start) >= @as(i32, i))
            self.Letters[self.Start + i]
        else
            0;
    }

    /// operator(i) : i-th letter counted from End (0-based), else 0.
    pub fn atEnd(self: *const Word, i: u8) u8 {
        return if (@as(i32, self.End) - @as(i32, self.Start) >= @as(i32, i))
            self.Letters[self.End - i]
        else
            0;
    }

    /// Length: number of active letters (0 if empty).
    pub fn length(self: *const Word) usize {
        if (self.Letters[self.Start] != 0)
            return @as(usize, self.End) - @as(usize, self.Start) + 1;
        return 0;
    }

    /// GetHashes: the generic four-way rolling hash.
    pub fn getHashes(self: *Word) void {
        self.Hash[0] = 0xc01df;
        self.Hash[1] = ~self.Hash[0];
        var i: usize = self.Start;
        while (i <= @as(usize, self.End)) : (i += 1) {
            const l = self.Letters[i];
            self.Hash[0] ^= hash3(self.Hash[0], @as(u64, l), @as(u64, @intCast(i)));
            const masked: u8 = if ((l & 0x80) == 0)
                l & 0x5F
            else if ((l & 0xC0) == 0x80)
                l & 0x3F
            else if ((l & 0xE0) == 0xC0)
                l & 0x1F
            else if ((l & 0xF0) == 0xE0)
                l & 0xF
            else
                l & 0x7;
            self.Hash[1] ^= hash2(self.Hash[1], @as(u64, masked));
        }
        self.Hash[2] = (~self.Hash[0]) ^ self.Hash[1];
        self.Hash[3] = (~self.Hash[1]) ^ self.Hash[0];
    }

    pub fn changeSuffix(self: *Word, old: []const u8, new: []const u8) bool {
        // (Length>len && memcmp(...)==0) == EndsWith(old)
        if (!self.endsWith(old)) return false;
        const end_i: i32 = self.End;
        const old_len: i32 = @intCast(old.len);
        const new_len: i32 = @intCast(new.len);
        if (new_len > 0) {
            const count = @min(@as(i32, MAX_WORD_SIZE - 1), end_i + new_len) - end_i;
            const dst: i32 = end_i - old_len + 1;
            var k: i32 = 0;
            while (k < count) : (k += 1) {
                self.Letters[@intCast(dst + k)] = new[@intCast(k)];
            }
            self.End = @intCast(@min(@as(i32, MAX_WORD_SIZE - 1), end_i - old_len + new_len));
        } else {
            self.End -%= @as(u8, @intCast(old.len));
        }
        return true;
    }

    pub fn matchesAny(self: *const Word, list: []const []const u8) bool {
        const len: usize = self.length();
        for (list) |s| {
            if (len == s.len and std.mem.eql(u8, self.Letters[self.Start .. self.Start + len], s))
                return true;
        }
        return false;
    }

    pub fn endsWith(self: *const Word, suffix: []const u8) bool {
        if (self.length() <= suffix.len) return false;
        const start: usize = @as(usize, self.End) - suffix.len + 1;
        return std.mem.eql(u8, self.Letters[start .. start + suffix.len], suffix);
    }

    pub fn startsWith(self: *const Word, prefix: []const u8) bool {
        if (self.length() <= prefix.len) return false;
        return std.mem.eql(u8, self.Letters[self.Start .. self.Start + prefix.len], prefix);
    }

    /// memcmp(&Letters[idx], s, s.len) == 0, bounds-guarded (out of range -> false).
    fn cmpAt(self: *const Word, idx: i32, s: []const u8) bool {
        var k: i32 = 0;
        while (k < @as(i32, @intCast(s.len))) : (k += 1) {
            const j = idx + k;
            if (j < 0 or j >= @as(i32, MAX_WORD_SIZE)) return false;
            if (self.Letters[@intCast(j)] != s[@intCast(k)]) return false;
        }
        return true;
    }

    /// Read the active letters into `buf`, returning the slice (helper for callers/tests).
    pub fn slice(self: *const Word, buf: []u8) []const u8 {
        if (self.length() == 0) return buf[0..0];
        const n = @as(usize, self.End) - @as(usize, self.Start) + 1;
        @memcpy(buf[0..n], self.Letters[self.Start .. self.Start + n]);
        return buf[0..n];
    }
};

// ---------------------------------------------------------------------------
// Segment / Sentence / Paragraph (paq8.cpp:1624-1650)
// ---------------------------------------------------------------------------
pub const Segment = struct {
    FirstWord: Word = .{},
    WordCount: u32 = 0,
    NumCount: u32 = 0,
};

pub const Sentence = struct {
    pub const Types = enum(u32) { Declarative = 0, Interrogative = 1, Exclamative = 2 };
    pub const TypesCount: usize = 3;

    // Segment base fields
    FirstWord: Word = .{},
    WordCount: u32 = 0,
    NumCount: u32 = 0,

    Type: Types = .Declarative,
    SegmentCount: u32 = 0,
    VerbIndex: u32 = 0,
    NounIndex: u32 = 0,
    CapitalIndex: u32 = 0,
    lastVerb: Word = .{},
    lastNoun: Word = .{},
    lastCapital: Word = .{},
};

pub const Paragraph = struct {
    SentenceCount: u32 = 0,
    TypeCount: [Sentence.TypesCount]u32 = .{ 0, 0, 0 },
    TypeMask: u32 = 0,
};

// ---------------------------------------------------------------------------
// Stemmer base helpers (paq8.cpp:1729-1751)
// ---------------------------------------------------------------------------
fn getRegion(w: *const Word, from: u32, vowels: []const u8) u32 {
    var has_vowel = false;
    var i: i32 = @as(i32, w.Start) + @as(i32, @intCast(from));
    while (i <= @as(i32, w.End)) : (i += 1) {
        const c = w.Letters[@intCast(i)];
        if (charInArray(c, vowels)) {
            has_vowel = true;
            continue;
        } else if (has_vowel) {
            return @intCast(i - @as(i32, w.Start) + 1);
        }
    }
    return @as(u32, w.Start) + @as(u32, @intCast(w.length()));
}

fn suffixInRn(w: *const Word, rn: u32, suffix_len: usize) bool {
    if (w.Start == w.End) return false;
    const l = w.length();
    if (l >= suffix_len) return rn <= @as(u32, @intCast(l - suffix_len));
    // C computes Length-strlen in unsigned size_t: underflow -> always true
    return true;
}

// ===========================================================================
// EnglishStemmer (paq8.cpp:1764-2422) — modified Porter2
// ===========================================================================
const eng_vowels = [_]u8{ 'a', 'e', 'i', 'o', 'u', 'y' };
const eng_doubles = [_]u8{ 'b', 'd', 'f', 'g', 'm', 'n', 'p', 'r', 't' };
const eng_li_endings = [_]u8{ 'c', 'd', 'e', 'g', 'h', 'k', 'm', 'n', 'r', 't' };
const eng_non_short_consonants = [_]u8{ 'w', 'x', 'Y' };
const eng_male = [_][]const u8{ "he", "him", "his", "himself", "man", "men", "boy", "husband", "actor" };
const eng_female = [_][]const u8{ "she", "her", "herself", "woman", "women", "girl", "wife", "actress" };
const eng_common = [_][]const u8{ "the", "be", "to", "of", "and", "in", "that", "you", "have", "with", "from", "but" };
const eng_step0 = [_][]const u8{ "'s'", "'s", "'" };

const SufType = struct { s: []const u8, t: u64 };
const Pair3 = struct { o: []const u8, n: []const u8, t: u64 };
const Exc = struct { o: []const u8, n: []const u8, t: u64 };

const eng_step1b = [_]SufType{
    .{ .s = "eedly", .t = English.AdverbOfManner },
    .{ .s = "eed", .t = 0 },
    .{ .s = "ed", .t = English.PastTense },
    .{ .s = "edly", .t = English.AdverbOfManner | English.PastTense },
    .{ .s = "ing", .t = English.PresentParticiple },
    .{ .s = "ingly", .t = English.AdverbOfManner | English.PresentParticiple },
};

const eng_step2 = [_]Pair3{
    .{ .o = "ization", .n = "ize", .t = English.SuffixION },
    .{ .o = "ational", .n = "ate", .t = English.SuffixION | English.SuffixAL },
    .{ .o = "ousness", .n = "ous", .t = English.SuffixNESS },
    .{ .o = "iveness", .n = "ive", .t = English.SuffixNESS },
    .{ .o = "fulness", .n = "ful", .t = English.SuffixNESS },
    .{ .o = "tional", .n = "tion", .t = English.SuffixION | English.SuffixAL },
    .{ .o = "lessli", .n = "less", .t = English.AdverbOfManner },
    .{ .o = "biliti", .n = "ble", .t = English.AdverbOfManner | English.SuffixITY },
    .{ .o = "entli", .n = "ent", .t = English.AdverbOfManner },
    .{ .o = "ation", .n = "ate", .t = English.SuffixION },
    .{ .o = "alism", .n = "al", .t = 0 },
    .{ .o = "aliti", .n = "al", .t = English.SuffixITY },
    .{ .o = "fulli", .n = "ful", .t = English.AdverbOfManner },
    .{ .o = "ousli", .n = "ous", .t = English.AdverbOfManner },
    .{ .o = "iviti", .n = "ive", .t = English.SuffixITY },
    .{ .o = "enci", .n = "ence", .t = 0 },
    .{ .o = "anci", .n = "ance", .t = 0 },
    .{ .o = "abli", .n = "able", .t = English.AdverbOfManner },
    .{ .o = "izer", .n = "ize", .t = 0 },
    .{ .o = "ator", .n = "ate", .t = 0 },
    .{ .o = "alli", .n = "al", .t = English.AdverbOfManner },
    .{ .o = "bli", .n = "ble", .t = English.AdverbOfManner },
};

const eng_step3 = [_]Pair3{
    .{ .o = "ational", .n = "ate", .t = English.SuffixION | English.SuffixAL },
    .{ .o = "tional", .n = "tion", .t = English.SuffixION | English.SuffixAL },
    .{ .o = "alize", .n = "al", .t = 0 },
    .{ .o = "icate", .n = "ic", .t = 0 },
    .{ .o = "iciti", .n = "ic", .t = English.SuffixITY },
    .{ .o = "ical", .n = "ic", .t = English.SuffixAL },
    .{ .o = "ful", .n = "", .t = English.AdjectiveFull },
    .{ .o = "ness", .n = "", .t = English.SuffixNESS },
};

const eng_step4 = [_]SufType{
    .{ .s = "al", .t = English.SuffixAL },
    .{ .s = "ance", .t = English.SuffixNCE },
    .{ .s = "ence", .t = English.SuffixNCE },
    .{ .s = "er", .t = 0 },
    .{ .s = "ic", .t = English.SuffixIC },
    .{ .s = "able", .t = English.SuffixCapable },
    .{ .s = "ible", .t = English.SuffixCapable },
    .{ .s = "ant", .t = English.SuffixNT },
    .{ .s = "ement", .t = 0 },
    .{ .s = "ment", .t = 0 },
    .{ .s = "ent", .t = English.SuffixNT },
    .{ .s = "ou", .t = 0 },
    .{ .s = "ism", .t = 0 },
    .{ .s = "ate", .t = 0 },
    .{ .s = "iti", .t = English.SuffixITY },
    .{ .s = "ous", .t = English.SuffixOUS },
    .{ .s = "ive", .t = English.SuffixIVE },
    .{ .s = "ize", .t = 0 },
    .{ .s = "sion", .t = English.SuffixION },
    .{ .s = "tion", .t = English.SuffixION },
};

const eng_exception_region1 = [_][]const u8{ "gener", "arsen", "commun" };

const eng_exceptions1 = [_]Exc{
    .{ .o = "skis", .n = "ski", .t = English.Noun | English.Plural },
    .{ .o = "skies", .n = "sky", .t = English.Plural },
    .{ .o = "dying", .n = "die", .t = English.PresentParticiple },
    .{ .o = "lying", .n = "lie", .t = English.PresentParticiple },
    .{ .o = "tying", .n = "tie", .t = English.PresentParticiple },
    .{ .o = "idly", .n = "idle", .t = English.AdverbOfManner },
    .{ .o = "gently", .n = "gentle", .t = English.AdverbOfManner },
    .{ .o = "ugly", .n = "ugli", .t = English.Adjective },
    .{ .o = "early", .n = "earli", .t = English.Adjective | English.AdverbOfManner },
    .{ .o = "only", .n = "onli", .t = 0 },
    .{ .o = "singly", .n = "singl", .t = English.AdverbOfManner },
    .{ .o = "sky", .n = "sky", .t = English.Noun },
    .{ .o = "news", .n = "news", .t = English.Noun },
    .{ .o = "howe", .n = "howe", .t = 0 },
    .{ .o = "atlas", .n = "atlas", .t = English.Noun },
    .{ .o = "cosmos", .n = "cosmos", .t = English.Noun },
    .{ .o = "bias", .n = "bias", .t = English.Noun },
    .{ .o = "andes", .n = "andes", .t = 0 },
};

const eng_exceptions2 = [_]SufType{
    .{ .s = "inning", .t = English.Noun },
    .{ .s = "outing", .t = English.Noun },
    .{ .s = "canning", .t = English.Noun },
    .{ .s = "herring", .t = English.Noun },
    .{ .s = "earring", .t = English.Noun },
    .{ .s = "proceed", .t = English.Verb },
    .{ .s = "exceed", .t = English.Verb },
    .{ .s = "succeed", .t = English.Verb },
};

inline fn engVowel(c: u8) bool {
    return charInArray(c, &eng_vowels);
}
inline fn engConsonant(c: u8) bool {
    return !engVowel(c);
}
inline fn engShortConsonant(c: u8) bool {
    return !charInArray(c, &eng_non_short_consonants);
}
inline fn engDouble(c: u8) bool {
    return charInArray(c, &eng_doubles);
}
inline fn engLiEnding(c: u8) bool {
    return charInArray(c, &eng_li_endings);
}

fn engGetRegion1(w: *const Word) u32 {
    for (eng_exception_region1) |e| {
        if (w.startsWith(e)) return @intCast(e.len);
    }
    return getRegion(w, 0, &eng_vowels);
}

fn engEndsInShortSyllable(w: *const Word) bool {
    if (w.End == w.Start) return false;
    if (w.End == w.Start + 1) return engVowel(w.atEnd(1)) and engConsonant(w.atEnd(0));
    return engConsonant(w.atEnd(2)) and engVowel(w.atEnd(1)) and
        engConsonant(w.atEnd(0)) and engShortConsonant(w.atEnd(0));
}

fn engIsShortWord(w: *const Word) bool {
    return engEndsInShortSyllable(w) and engGetRegion1(w) == @as(u32, @intCast(w.length()));
}

fn engHasVowels(w: *const Word) bool {
    var i: usize = w.Start;
    while (i <= @as(usize, w.End)) : (i += 1) {
        if (engVowel(w.Letters[i])) return true;
    }
    return false;
}

fn engTrimStartingApostrophe(w: *Word) bool {
    const r = (w.Start != w.End and w.at(0) == '\'');
    w.Start +%= boolByte(r);
    return r;
}

fn engMarkYsAsConsonants(w: *Word) void {
    if (w.at(0) == 'y') w.Letters[w.Start] = 'Y';
    var i: usize = @as(usize, w.Start) + 1;
    while (i <= @as(usize, w.End)) : (i += 1) {
        if (engVowel(w.Letters[i - 1]) and w.Letters[i] == 'y') w.Letters[i] = 'Y';
    }
}

fn engProcessPrefixes(w: *Word) bool {
    if (w.startsWith("irr") and w.length() > 5 and (w.at(3) == 'a' or w.at(3) == 'e')) {
        w.Start +%= 2;
        w.Type |= English.Negation;
    } else if (w.startsWith("over") and w.length() > 5) {
        w.Start +%= 4;
        w.Type |= English.PrefixOver;
    } else if (w.startsWith("under") and w.length() > 6) {
        w.Start +%= 5;
        w.Type |= English.PrefixUnder;
    } else if (w.startsWith("unn") and w.length() > 5) {
        w.Start +%= 2;
        w.Type |= English.Negation;
    } else if (w.startsWith("non") and w.length() > @as(usize, 5 + boolByte(w.at(3) == '-'))) {
        w.Start +%= 2 + boolByte(w.at(3) == '-');
        w.Type |= English.Negation;
    } else {
        return false;
    }
    return true;
}

fn engProcessSuperlatives(w: *Word) bool {
    if (w.endsWith("est") and w.length() > 4) {
        const i: u8 = w.End;
        w.End -%= 3;
        w.Type |= English.AdjectiveSuperlative;

        if (w.atEnd(0) == w.atEnd(1) and w.atEnd(0) != 'r' and
            !(w.length() >= 4 and w.cmpAt(@as(i32, w.End) - 3, "sugg")))
        {
            w.End -%= boolByte(
                (((w.atEnd(0) != 'f' and w.atEnd(0) != 'l' and w.atEnd(0) != 's')) or
                    (w.length() > 4 and w.atEnd(1) == 'l' and (w.atEnd(2) == 'u' or w.atEnd(3) == 'u' or w.atEnd(3) == 'v'))) and
                    (!(w.length() == 3 and w.atEnd(1) == 'd' and w.atEnd(2) == 'o')),
            );
            if (w.length() == 2 and (w.at(0) != 'i' or w.at(1) != 'n')) {
                w.End = i;
                w.Type &= ~English.AdjectiveSuperlative;
            }
        } else {
            switch (w.atEnd(0)) {
                'd', 'k', 'm', 'y' => {},
                'g' => {
                    if (!(w.length() > 3 and (w.atEnd(1) == 'n' or w.atEnd(1) == 'r') and !w.cmpAt(@as(i32, w.End) - 3, "cong"))) {
                        w.End = i;
                        w.Type &= ~English.AdjectiveSuperlative;
                    } else {
                        w.End +%= boolByte(w.atEnd(2) == 'a');
                    }
                },
                'i' => {
                    w.Letters[w.End] = 'y';
                },
                'l' => {
                    if (w.End == w.Start + 1 or w.cmpAt(@as(i32, w.End) - 2, "mo")) {
                        w.End = i;
                        w.Type &= ~English.AdjectiveSuperlative;
                    } else {
                        w.End +%= boolByte(engConsonant(w.atEnd(1)));
                    }
                },
                'n' => {
                    if (w.length() < 3 or engConsonant(w.atEnd(1)) or engConsonant(w.atEnd(2))) {
                        w.End = i;
                        w.Type &= ~English.AdjectiveSuperlative;
                    }
                },
                'r' => {
                    if (w.length() > 3 and engVowel(w.atEnd(1)) and engVowel(w.atEnd(2))) {
                        w.End +%= boolByte(w.atEnd(2) == 'u' and (w.atEnd(1) == 'a' or w.atEnd(1) == 'i'));
                    } else {
                        w.End = i;
                        w.Type &= ~English.AdjectiveSuperlative;
                    }
                },
                's' => {
                    w.End +%= 1;
                },
                'w' => {
                    if (!(w.length() > 2 and engVowel(w.atEnd(1)))) {
                        w.End = i;
                        w.Type &= ~English.AdjectiveSuperlative;
                    }
                },
                'h' => {
                    if (!(w.length() > 2 and engConsonant(w.atEnd(1)))) {
                        w.End = i;
                        w.Type &= ~English.AdjectiveSuperlative;
                    }
                },
                else => {
                    w.End +%= 3;
                    w.Type &= ~English.AdjectiveSuperlative;
                },
            }
        }
    }
    return (w.Type & English.AdjectiveSuperlative) > 0;
}

fn engStep0(w: *Word) bool {
    for (eng_step0) |suf| {
        if (w.endsWith(suf)) {
            w.End -%= @as(u8, @intCast(suf.len));
            w.Type |= English.Plural;
            return true;
        }
    }
    return false;
}

fn engStep1a(w: *Word) bool {
    if (w.endsWith("sses")) {
        w.End -%= 2;
        w.Type |= English.Plural;
        return true;
    }
    if (w.endsWith("ied") or w.endsWith("ies")) {
        w.Type |= if (w.atEnd(0) == 'd') English.PastTense else English.Plural;
        w.End -%= 1 + boolByte(w.length() > 4);
        return true;
    }
    if (w.endsWith("us") or w.endsWith("ss")) return false;
    if (w.atEnd(0) == 's' and w.length() > 2) {
        var i: i32 = w.Start;
        while (i <= @as(i32, w.End) - 2) : (i += 1) {
            if (engVowel(w.Letters[@intCast(i)])) {
                w.End -%= 1;
                w.Type |= English.Plural;
                return true;
            }
        }
    }
    if (w.endsWith("n't") and w.length() > 4) {
        switch (w.atEnd(3)) {
            'a' => {
                if (w.atEnd(4) == 'c') w.End -%= 2 else _ = w.changeSuffix("n't", "ll");
            },
            'i' => {
                _ = w.changeSuffix("in't", "m");
            },
            'o' => {
                if (w.atEnd(4) == 'w') _ = w.changeSuffix("on't", "ill") else w.End -%= 3;
            },
            else => w.End -%= 3,
        }
        w.Type |= English.Negation;
        return true;
    }
    if (w.endsWith("hood") and w.length() > 7) {
        w.End -%= 4;
        return true;
    }
    return false;
}

fn engStep1b(w: *Word, r1: u32) bool {
    for (eng_step1b, 0..) |entry, i| {
        if (w.endsWith(entry.s)) {
            switch (i) {
                0, 1 => {
                    if (suffixInRn(w, r1, entry.s.len))
                        w.End -%= @as(u8, @intCast(1 + i * 2));
                },
                else => {
                    const j: u8 = w.End;
                    w.End -%= @as(u8, @intCast(entry.s.len));
                    if (engHasVowels(w)) {
                        if (w.endsWith("at") or w.endsWith("bl") or w.endsWith("iz") or engIsShortWord(w)) {
                            w.plusEq('e');
                        } else if (w.length() > 2) {
                            if (w.atEnd(0) == w.atEnd(1) and engDouble(w.atEnd(0))) {
                                w.End -%= 1;
                            } else if (i == 2 or i == 3) {
                                switch (w.atEnd(0)) {
                                    'c', 's', 'v' => {
                                        w.End +%= boolByte(!(w.endsWith("ss") or w.endsWith("ias")));
                                    },
                                    'd' => {
                                        const nAllowed = [_]u8{ 'a', 'e', 'i', 'o' };
                                        w.End +%= boolByte(engVowel(w.atEnd(1)) and !charInArray(w.atEnd(2), &nAllowed));
                                    },
                                    'k' => {
                                        w.End +%= boolByte(w.endsWith("uak"));
                                    },
                                    'l' => {
                                        const Allowed1 = [_]u8{ 'b', 'c', 'd', 'f', 'g', 'k', 'p', 't', 'y', 'z' };
                                        const Allowed2 = [_]u8{ 'a', 'i', 'o', 'u' };
                                        w.End +%= boolByte(charInArray(w.atEnd(1), &Allowed1) or
                                            (charInArray(w.atEnd(1), &Allowed2) and engConsonant(w.atEnd(2))));
                                    },
                                    else => {},
                                }
                            } else if (i >= 4) {
                                switch (w.atEnd(0)) {
                                    'd' => {
                                        if (engVowel(w.atEnd(1)) and w.atEnd(2) != 'a' and w.atEnd(2) != 'e' and w.atEnd(2) != 'o')
                                            w.plusEq('e');
                                    },
                                    'g' => {
                                        const Allowed = [_]u8{ 'a', 'd', 'e', 'i', 'l', 'r', 'u' };
                                        if (charInArray(w.atEnd(1), &Allowed) or
                                            (w.atEnd(1) == 'n' and
                                                (w.atEnd(2) == 'e' or
                                                    (w.atEnd(2) == 'u' and w.atEnd(3) != 'b' and w.atEnd(3) != 'd') or
                                                    (w.atEnd(2) == 'a' and (w.atEnd(3) == 'r' or (w.atEnd(3) == 'h' and w.atEnd(4) == 'c'))) or
                                                    (w.endsWith("ring") and (w.atEnd(4) == 'c' or w.atEnd(4) == 'f')))))
                                            w.plusEq('e');
                                    },
                                    'l' => {
                                        if (!(w.atEnd(1) == 'l' or w.atEnd(1) == 'r' or w.atEnd(1) == 'w' or
                                            (engVowel(w.atEnd(1)) and engVowel(w.atEnd(2)))))
                                            w.plusEq('e');
                                        if (w.endsWith("uell") and w.length() > 4 and w.atEnd(4) != 'q')
                                            w.End -%= 1;
                                    },
                                    'r' => {
                                        if ((((w.atEnd(1) == 'i' and w.atEnd(2) != 'a' and w.atEnd(2) != 'e' and w.atEnd(2) != 'o') or
                                            (w.atEnd(1) == 'a' and (!(w.atEnd(2) == 'e' or w.atEnd(2) == 'o' or (w.atEnd(2) == 'l' and w.atEnd(3) == 'l')))) or
                                            (w.atEnd(1) == 'o' and (!(w.atEnd(2) == 'o' or (w.atEnd(2) == 't' and w.atEnd(3) != 's')))) or
                                            w.atEnd(1) == 'c' or w.atEnd(1) == 't')) and (!w.endsWith("str")))
                                            w.plusEq('e');
                                    },
                                    't' => {
                                        if (w.atEnd(1) == 'o' and w.atEnd(2) != 'g' and w.atEnd(2) != 'l' and w.atEnd(2) != 'i' and w.atEnd(2) != 'o')
                                            w.plusEq('e');
                                    },
                                    'u' => {
                                        if (!(w.length() > 3 and engVowel(w.atEnd(1)) and engVowel(w.atEnd(2))))
                                            w.plusEq('e');
                                    },
                                    'z' => {
                                        if (w.endsWith("izz") and w.length() > 3 and (w.atEnd(3) == 'h' or w.atEnd(3) == 'u'))
                                            w.End -%= 1
                                        else if (w.atEnd(1) != 't' and w.atEnd(1) != 'z')
                                            w.plusEq('e');
                                    },
                                    'k' => {
                                        if (w.endsWith("uak")) w.plusEq('e');
                                    },
                                    'b', 'c', 's', 'v' => {
                                        if (!((w.atEnd(0) == 'b' and (w.atEnd(1) == 'm' or w.atEnd(1) == 'r')) or
                                            w.endsWith("ss") or w.endsWith("ias") or w.eq("zinc")))
                                            w.plusEq('e');
                                    },
                                    else => {},
                                }
                            }
                        }
                    } else {
                        w.End = j;
                        return false;
                    }
                },
            }
            w.Type |= entry.t;
            return true;
        }
    }
    return false;
}

fn engStep1c(w: *Word) bool {
    if (w.length() > 2 and tolower(w.atEnd(0)) == 'y' and engConsonant(w.atEnd(1))) {
        w.Letters[w.End] = 'i';
        return true;
    }
    return false;
}

fn engStep2(w: *Word, r1: u32) bool {
    for (eng_step2) |e| {
        if (w.endsWith(e.o) and suffixInRn(w, r1, e.o.len)) {
            _ = w.changeSuffix(e.o, e.n);
            w.Type |= e.t;
            return true;
        }
    }
    if (w.endsWith("logi") and suffixInRn(w, r1, 3)) { // "ogi"
        w.End -%= 1;
        return true;
    } else if (w.endsWith("li")) {
        if (suffixInRn(w, r1, 2) and engLiEnding(w.atEnd(2))) {
            w.End -%= 2;
            w.Type |= English.AdverbOfManner;
            return true;
        } else if (w.length() > 3) {
            switch (w.atEnd(2)) {
                'b' => {
                    w.Letters[w.End] = 'e';
                    w.Type |= English.AdverbOfManner;
                    return true;
                },
                'i' => {
                    if (w.length() > 4) {
                        w.End -%= 2;
                        w.Type |= English.AdverbOfManner;
                        return true;
                    }
                },
                'l' => {
                    if (w.length() > 5 and (w.atEnd(3) == 'a' or w.atEnd(3) == 'u')) {
                        w.End -%= 2;
                        w.Type |= English.AdverbOfManner;
                        return true;
                    }
                },
                's' => {
                    w.End -%= 2;
                    w.Type |= English.AdverbOfManner;
                    return true;
                },
                'e', 'g', 'm', 'n', 'r', 'w' => {
                    if (w.length() > @as(usize, 4 + boolByte(w.atEnd(2) == 'r'))) {
                        w.End -%= 2;
                        w.Type |= English.AdverbOfManner;
                        return true;
                    }
                },
                else => {},
            }
        }
    }
    return false;
}

fn engStep3(w: *Word, r1: u32, r2: u32) bool {
    var res = false;
    for (eng_step3) |e| {
        if (w.endsWith(e.o) and suffixInRn(w, r1, e.o.len)) {
            _ = w.changeSuffix(e.o, e.n);
            w.Type |= e.t;
            res = true;
            break;
        }
    }
    if (w.endsWith("ative") and suffixInRn(w, r2, 5)) {
        w.End -%= 5;
        w.Type |= English.SuffixIVE;
        return true;
    }
    if (w.length() > 5 and w.endsWith("less")) {
        w.End -%= 4;
        w.Type |= English.AdjectiveWithout;
        return true;
    }
    return res;
}

fn engStep4(w: *Word, r2: u32) bool {
    var res = false;
    for (eng_step4, 0..) |e, i| {
        if (w.endsWith(e.s) and suffixInRn(w, r2, e.s.len)) {
            w.End -%= @as(u8, @intCast(e.s.len - @as(usize, boolByte(i > 17))));
            if (i != 10 or w.atEnd(0) != 'm')
                w.Type |= e.t;
            if (i == 0 and w.endsWith("nti")) {
                w.End -%= 1;
                res = true;
                continue;
            }
            return true;
        }
    }
    return res;
}

fn engStep5(w: *Word, r1: u32, r2: u32) bool {
    if (w.atEnd(0) == 'e' and w.neq("here")) {
        if (suffixInRn(w, r2, 1)) {
            w.End -%= 1;
        } else if (suffixInRn(w, r1, 1)) {
            w.End -%= 1;
            w.End +%= boolByte(engEndsInShortSyllable(w));
        } else return false;
        return true;
    } else if (w.length() > 1 and w.atEnd(0) == 'l' and suffixInRn(w, r2, 1) and w.atEnd(1) == 'l') {
        w.End -%= 1;
        return true;
    }
    return false;
}

fn engHash(w: *Word) void {
    w.Hash[2] = 0xb0a710ad;
    w.Hash[3] = 0xb0a710ad;
    var i: usize = w.Start;
    while (i <= @as(usize, w.End)) : (i += 1) {
        const l = w.Letters[i];
        w.Hash[2] = w.Hash[2] *% 263 *% 32 +% @as(u64, l);
        if (engVowel(l)) {
            w.Hash[3] = w.Hash[3] *% 997 *% 8 +% (@as(u64, l) / 4 -% 22);
        } else if (l >= 'b' and l <= 'z') {
            w.Hash[3] = w.Hash[3] *% 271 *% 32 +% (@as(u64, l) -% 97);
        } else {
            w.Hash[3] = w.Hash[3] *% 11 *% 32 +% @as(u64, l);
        }
    }
}

fn engStem(w: *Word) bool {
    if (w.length() < 2) {
        engHash(w);
        return false;
    }
    var res = engTrimStartingApostrophe(w);
    if (engProcessPrefixes(w)) res = true;
    if (engProcessSuperlatives(w)) res = true;
    for (eng_exceptions1, 0..) |e, idx| {
        if (w.eq(e.o)) {
            if (idx < 11) {
                @memcpy(w.Letters[w.Start .. w.Start + e.n.len], e.n);
                w.End = w.Start + @as(u8, @intCast(e.n.len - 1));
            }
            engHash(w);
            w.Type |= e.t;
            w.Language = Language.English;
            return idx < 11;
        }
    }

    engMarkYsAsConsonants(w);
    const r1 = engGetRegion1(w);
    const r2 = getRegion(w, r1, &eng_vowels);
    if (engStep0(w)) res = true;
    if (engStep1a(w)) res = true;
    for (eng_exceptions2) |e| {
        if (w.eq(e.s)) {
            engHash(w);
            w.Type |= e.t;
            w.Language = Language.English;
            return res;
        }
    }
    if (engStep1b(w, r1)) res = true;
    if (engStep1c(w)) res = true;
    if (engStep2(w, r1)) res = true;
    if (engStep3(w, r1, r2)) res = true;
    if (engStep4(w, r2)) res = true;
    if (engStep5(w, r1, r2)) res = true;

    var i: usize = w.Start;
    while (i <= @as(usize, w.End)) : (i += 1) {
        if (w.Letters[i] == 'Y') w.Letters[i] = 'y';
    }
    if (w.Type == 0 or w.Type == English.Plural) {
        if (w.matchesAny(&eng_male)) {
            res = true;
            w.Type |= English.Male;
        } else if (w.matchesAny(&eng_female)) {
            res = true;
            w.Type |= English.Female;
        }
    }
    if (!res) res = w.matchesAny(&eng_common);
    engHash(w);
    if (res) w.Language = Language.English;
    return res;
}

pub const EnglishStemmer = struct {
    pub fn isVowel(_: *const EnglishStemmer, c: u8) bool {
        return engVowel(c);
    }
    pub fn hash(_: *const EnglishStemmer, w: *Word) void {
        engHash(w);
    }
    pub fn stem(_: *EnglishStemmer, w: *Word) bool {
        return engStem(w);
    }
};

// ===========================================================================
// FrenchStemmer (paq8.cpp:2433-2822) — modified Porter
// ===========================================================================
const fr_vowels = [_]u8{ 'a', 'e', 'i', 'o', 'u', 'y', 0xE2, 0xE0, 0xEB, 0xE9, 0xEA, 0xE8, 0xEF, 0xEE, 0xF4, 0xFB, 0xF9 };
const fr_common = [_][]const u8{ "de", "la", "le", "et", "en", "un", "une", "du", "que", "pas" };
const fr_exceptions = [_]Exc{
    .{ .o = "monument", .n = "monument", .t = French.Noun },
    .{ .o = "yeux", .n = "oeil", .t = French.Noun | French.Plural },
    .{ .o = "travaux", .n = "travail", .t = French.Noun | French.Plural },
};
const fr_step1 = [_][]const u8{
    "ance",     "iqUe",   "isme",  "able",    "iste",    "eux",  "ances", "iqUes",   "ismes", "ables",   "istes", // 11
    "atrice",   "ateur",  "ation", "atrices", "ateurs",  "ations", // 6
    "logie",    "logies", // 2
    "usion",    "ution",  "usions", "utions", // 4
    "ence",     "ences", // 2
    "issement", "issements", // 2
    "ement",    "ements", // 2
    "it\xE9",   "it\xE9s", // 2
    "if",       "ive",    "ifs",   "ives", // 4
    "euse",     "euses", // 2
    "ment",     "ments", // 2
};
const fr_step2a = [_][]const u8{
    "issaIent", "issantes", "iraIent", "issante",
    "issants",  "issions",  "irions",  "issais",
    "issait",   "issant",   "issent",  "issiez", "issons",
    "irais",    "irait",    "irent",   "iriez",  "irons",
    "iront",    "isses",    "issez",   "\xEEmes",
    "\xEEtes",  "irai",     "iras",    "irez",   "isse",
    "ies",      "ira",      "\xEEt",   "ie",     "ir",   "is",
    "it",       "i",
};
const fr_step2b = [_][]const u8{
    "eraIent",  "assions", "erions", "assent",
    "assiez",   "\xE8rent", "erais", "erait",
    "eriez",    "erons",   "eront",  "aIent", "antes",
    "asses",    "ions",    "erai",   "eras",  "erez",
    "\xE2mes",  "\xE2tes", "ante",   "ants",
    "asse",     "\xE9es",  "era",    "iez",   "ais",
    "ait",      "ant",     "\xE9e",  "\xE9s", "er",
    "ez",       "\xE2t",   "ai",     "as",    "\xE9", "a",
};
const fr_set_step4 = [_]u8{ 'a', 'i', 'o', 'u', 0xE8, 's' };
const fr_step4 = [_][]const u8{ "i\xE8re", "I\xE8re", "ion", "ier", "Ier", "e", "\xEB" };
const fr_step5 = [_][]const u8{ "enn", "onn", "ett", "ell", "eill" };

inline fn frVowel(c: u8) bool {
    return charInArray(c, &fr_vowels);
}
inline fn frConsonant(c: u8) bool {
    return !frVowel(c);
}

fn frConvertUTF8(w: *Word) void {
    var i: i32 = w.Start;
    while (i < @as(i32, w.End)) : (i += 1) {
        const ii: usize = @intCast(i);
        const nxt = w.Letters[ii + 1];
        const c: u8 = nxt +% (if (nxt < 0xA0) @as(u8, 0x60) else @as(u8, 0x40));
        if (w.Letters[ii] == 0xC3 and (frVowel(c) or (nxt & 0xDF) == 0x87)) {
            w.Letters[ii] = c;
            if (i + 1 < @as(i32, w.End)) {
                const count: usize = @as(usize, w.End) - ii - 1;
                std.mem.copyForwards(u8, w.Letters[ii + 1 .. ii + 1 + count], w.Letters[ii + 2 .. ii + 2 + count]);
            }
            w.End -%= 1;
        }
    }
}

fn frMarkVowelsAsConsonants(w: *Word) void {
    var i: usize = w.Start;
    while (i <= @as(usize, w.End)) : (i += 1) {
        switch (w.Letters[i]) {
            'i', 'u' => {
                if (i > @as(usize, w.Start) and i < @as(usize, w.End) and
                    (frVowel(w.Letters[i - 1]) or (w.Letters[i - 1] == 'q' and w.Letters[i] == 'u')) and
                    frVowel(w.Letters[i + 1]))
                    w.Letters[i] = toupper(w.Letters[i]);
            },
            'y' => {
                if ((i > @as(usize, w.Start) and frVowel(w.Letters[i - 1])) or
                    (i < @as(usize, w.End) and frVowel(w.Letters[i + 1])))
                    w.Letters[i] = toupper(w.Letters[i]);
            },
            else => {},
        }
    }
}

fn frGetRV(w: *Word) u32 {
    const len = w.length();
    const res: u32 = @as(u32, w.Start) + @as(u32, @intCast(len));
    if (len >= 3 and ((frVowel(w.Letters[w.Start]) and frVowel(w.Letters[w.Start + 1])) or
        w.startsWith("par") or w.startsWith("col") or w.startsWith("tap")))
        return @as(u32, w.Start) + 3;
    var i: i32 = @as(i32, w.Start) + 1;
    while (i <= @as(i32, w.End)) : (i += 1) {
        if (frVowel(w.Letters[@intCast(i)])) return @intCast(i + 1);
    }
    return res;
}

fn frStep1(w: *Word, rv: u32, r1: u32, r2: u32, force_step2a: *bool) bool {
    var i: usize = 0;
    while (i < 11) : (i += 1) {
        const suf = fr_step1[i];
        if (w.endsWith(suf) and suffixInRn(w, r2, suf.len)) {
            w.End -%= @as(u8, @intCast(suf.len));
            if (i == 3) w.Type |= French.Adjective; // "able"
            return true;
        }
    }
    while (i < 17) : (i += 1) {
        const suf = fr_step1[i];
        if (w.endsWith(suf) and suffixInRn(w, r2, suf.len)) {
            w.End -%= @as(u8, @intCast(suf.len));
            if (w.endsWith("ic")) _ = w.changeSuffix("c", "qU");
            return true;
        }
    }
    while (i < 25) : (i += 1) {
        const suf = fr_step1[i];
        if (w.endsWith(suf) and suffixInRn(w, r2, suf.len)) {
            w.End -%= @as(u8, @intCast(suf.len - 1 - @as(usize, boolByte(i < 19)) * 2));
            if (i > 22) {
                w.End +%= 2;
                w.Letters[w.End] = 't';
            }
            return true;
        }
    }
    while (i < 27) : (i += 1) {
        const suf = fr_step1[i];
        if (w.endsWith(suf) and suffixInRn(w, r1, suf.len) and frConsonant(w.atEnd(@as(u8, @intCast(suf.len))))) {
            w.End -%= @as(u8, @intCast(suf.len));
            return true;
        }
    }
    while (i < 29) : (i += 1) {
        const suf = fr_step1[i];
        if (w.endsWith(suf) and suffixInRn(w, rv, suf.len)) {
            w.End -%= @as(u8, @intCast(suf.len));
            if (w.endsWith("iv") and suffixInRn(w, r2, 2)) {
                w.End -%= 2;
                if (w.endsWith("at") and suffixInRn(w, r2, 2)) w.End -%= 2;
            } else if (w.endsWith("eus")) {
                if (suffixInRn(w, r2, 3)) w.End -%= 3 else if (suffixInRn(w, r1, 3)) w.Letters[w.End] = 'x';
            } else if ((w.endsWith("abl") and suffixInRn(w, r2, 3)) or (w.endsWith("iqU") and suffixInRn(w, r2, 3))) {
                w.End -%= 3;
            } else if ((w.endsWith("i\xE8r") and suffixInRn(w, rv, 3)) or (w.endsWith("I\xE8r") and suffixInRn(w, rv, 3))) {
                w.End -%= 2;
                w.Letters[w.End] = 'i';
            }
            return true;
        }
    }
    while (i < 31) : (i += 1) {
        const suf = fr_step1[i];
        if (w.endsWith(suf) and suffixInRn(w, r2, suf.len)) {
            w.End -%= @as(u8, @intCast(suf.len));
            if (w.endsWith("abil")) {
                if (suffixInRn(w, r2, 4)) {
                    w.End -%= 4;
                } else {
                    w.End -%= 1;
                    w.Letters[w.End] = 'l';
                }
            } else if (w.endsWith("ic")) {
                if (suffixInRn(w, r2, 2)) w.End -%= 2 else _ = w.changeSuffix("c", "qU");
            } else if (w.endsWith("iv") and suffixInRn(w, r2, 2)) {
                w.End -%= 2;
            }
            return true;
        }
    }
    while (i < 35) : (i += 1) {
        const suf = fr_step1[i];
        if (w.endsWith(suf) and suffixInRn(w, r2, suf.len)) {
            w.End -%= @as(u8, @intCast(suf.len));
            if (w.endsWith("at") and suffixInRn(w, r2, 2)) {
                w.End -%= 2;
                if (w.endsWith("ic")) {
                    if (suffixInRn(w, r2, 2)) w.End -%= 2 else _ = w.changeSuffix("c", "qU");
                }
            }
            return true;
        }
    }
    while (i < 37) : (i += 1) {
        const suf = fr_step1[i];
        if (w.endsWith(suf)) {
            if (suffixInRn(w, r2, suf.len)) {
                w.End -%= @as(u8, @intCast(suf.len));
                return true;
            } else if (suffixInRn(w, r1, suf.len)) {
                _ = w.changeSuffix(suf, "eux");
                return true;
            }
        }
    }
    while (i < fr_step1.len) : (i += 1) {
        const suf = fr_step1[i];
        if (w.endsWith(suf) and suffixInRn(w, rv + 1, suf.len) and frVowel(w.atEnd(@as(u8, @intCast(suf.len))))) {
            w.End -%= @as(u8, @intCast(suf.len));
            force_step2a.* = true;
            return true;
        }
    }
    if (w.endsWith("eaux") or w.eq("eaux")) {
        w.End -%= 1;
        w.Type |= French.Plural;
        return true;
    } else if (w.endsWith("aux") and suffixInRn(w, r1, 3)) {
        w.End -%= 1;
        w.Letters[w.End] = 'l';
        w.Type |= French.Plural;
        return true;
    } else if (w.endsWith("amment") and suffixInRn(w, rv, 6)) {
        _ = w.changeSuffix("amment", "ant");
        force_step2a.* = true;
        return true;
    } else if (w.endsWith("emment") and suffixInRn(w, rv, 6)) {
        _ = w.changeSuffix("emment", "ent");
        force_step2a.* = true;
        return true;
    }
    return false;
}

fn frStep2a(w: *Word, rv: u32) bool {
    for (fr_step2a, 0..) |suf, i| {
        if (w.endsWith(suf) and suffixInRn(w, rv + 1, suf.len) and frConsonant(w.atEnd(@as(u8, @intCast(suf.len))))) {
            w.End -%= @as(u8, @intCast(suf.len));
            if (i == 31) w.Type |= French.Verb; // "ir"
            return true;
        }
    }
    return false;
}

fn frStep2b(w: *Word, rv: u32, r2: u32) bool {
    for (fr_step2b, 0..) |suf, i| {
        if (w.endsWith(suf) and suffixInRn(w, rv, suf.len)) {
            switch (suf[0]) {
                'a', 0xE2 => {
                    w.End -%= @as(u8, @intCast(suf.len));
                    if (w.endsWith("e") and suffixInRn(w, rv, 1)) w.End -%= 1;
                    return true;
                },
                else => {
                    if (i != 14 or suffixInRn(w, r2, suf.len)) {
                        w.End -%= @as(u8, @intCast(suf.len));
                        return true;
                    }
                },
            }
        }
    }
    return false;
}

fn frStep3(w: *Word) void {
    const f = w.Letters[w.End];
    if (f == 'Y') w.Letters[w.End] = 'i' else if (f == 0xE7) w.Letters[w.End] = 'c';
}

fn frStep4(w: *Word, rv: u32, r2: u32) bool {
    var res = false;
    if (w.length() >= 2 and w.Letters[w.End] == 's' and !charInArray(w.atEnd(1), &fr_set_step4)) {
        w.End -%= 1;
        res = true;
    }
    for (fr_step4, 0..) |suf, i| {
        if (w.endsWith(suf) and suffixInRn(w, rv, suf.len)) {
            switch (i) {
                2 => { // ion
                    const prec = w.atEnd(3);
                    if (suffixInRn(w, r2, suf.len) and suffixInRn(w, rv + 1, suf.len) and (prec == 's' or prec == 't')) {
                        w.End -%= 3;
                        return true;
                    }
                },
                5 => { // e
                    w.End -%= 1;
                    return true;
                },
                6 => { // ë
                    if (w.endsWith("gu\xEB")) {
                        w.End -%= 1;
                        return true;
                    }
                },
                else => {
                    _ = w.changeSuffix(suf, "i");
                    return true;
                },
            }
        }
    }
    return res;
}

fn frStep5(w: *Word) bool {
    for (fr_step5) |suf| {
        if (w.endsWith(suf)) {
            w.End -%= 1;
            return true;
        }
    }
    return false;
}

fn frStep6(w: *Word) bool {
    var i: i32 = w.End;
    while (i >= @as(i32, w.Start)) : (i -= 1) {
        const c = w.Letters[@intCast(i)];
        if (frVowel(c)) {
            if (i < @as(i32, w.End) and (c & 0xFE) == 0xE8) {
                w.Letters[@intCast(i)] = 'e';
                return true;
            }
            return false;
        }
    }
    return false;
}

fn frHash(w: *Word) void {
    w.Hash[2] = 0x100e3531; // ~0xeff1cace (32-bit)
    w.Hash[3] = 0x100e3531;
    var i: usize = w.Start;
    while (i <= @as(usize, w.End)) : (i += 1) {
        const l = w.Letters[i];
        w.Hash[2] = w.Hash[2] *% 251 *% 32 +% @as(u64, l);
        if (frVowel(l)) {
            w.Hash[3] = w.Hash[3] *% 997 *% 16 +% @as(u64, l);
        } else if (l >= 'b' and l <= 'z') {
            w.Hash[3] = w.Hash[3] *% 271 *% 32 +% (@as(u64, l) -% 97);
        } else {
            w.Hash[3] = w.Hash[3] *% 11 *% 32 +% @as(u64, l);
        }
    }
}

fn frStem(w: *Word) bool {
    frConvertUTF8(w);
    if (w.length() < 2) {
        frHash(w);
        return false;
    }
    for (fr_exceptions) |e| {
        if (w.eq(e.o)) {
            @memcpy(w.Letters[w.Start .. w.Start + e.n.len], e.n);
            w.End = w.Start + @as(u8, @intCast(e.n.len - 1));
            frHash(w);
            w.Type |= e.t;
            w.Language = Language.French;
            return true;
        }
    }
    frMarkVowelsAsConsonants(w);
    const rv = frGetRV(w);
    const r1 = getRegion(w, 0, &fr_vowels);
    const r2 = getRegion(w, r1, &fr_vowels);
    var do_next: bool = false;
    var res = frStep1(w, rv, r1, r2, &do_next);
    if (!res) do_next = true;
    if (do_next) {
        do_next = !frStep2a(w, rv);
        if (!do_next) res = true;
        if (do_next) {
            if (frStep2b(w, rv, r2)) res = true;
        }
    }
    if (res) frStep3(w) else {
        if (frStep4(w, rv, r2)) res = true;
    }
    if (frStep5(w)) res = true;
    if (frStep6(w)) res = true;
    var i: usize = w.Start;
    while (i <= @as(usize, w.End)) : (i += 1) {
        w.Letters[i] = tolower(w.Letters[i]);
    }
    if (!res) res = w.matchesAny(&fr_common);
    frHash(w);
    if (res) w.Language = Language.French;
    return res;
}

pub const FrenchStemmer = struct {
    pub fn isVowel(_: *const FrenchStemmer, c: u8) bool {
        return frVowel(c);
    }
    pub fn hash(_: *const FrenchStemmer, w: *Word) void {
        frHash(w);
    }
    pub fn stem(_: *FrenchStemmer, w: *Word) bool {
        return frStem(w);
    }
};

// ===========================================================================
// GermanStemmer (paq8.cpp:2831-2997) — modified Porter
// ===========================================================================
const de_vowels = [_]u8{ 'a', 'e', 'i', 'o', 'u', 'y', 0xE4, 0xF6, 0xFC };
const de_common = [_][]const u8{ "der", "die", "das", "und", "sie", "ich", "mit", "sich", "auf", "nicht" };
const de_endings = [_]u8{ 'b', 'd', 'f', 'g', 'h', 'k', 'l', 'm', 'n', 't' }; // plus 'r' for words ending in 's'
const de_step1 = [_][]const u8{ "em", "ern", "er", "e", "en", "es" };
const de_step2 = [_][]const u8{ "en", "er", "est" };
const de_step3 = [_][]const u8{ "end", "ung", "ik", "ig", "isch", "lich", "heit" };

inline fn deVowel(c: u8) bool {
    return charInArray(c, &de_vowels);
}
inline fn deValidEnding(c: u8, include_r: bool) bool {
    return charInArray(c, &de_endings) or (include_r and c == 'r');
}

fn deConvertUTF8(w: *Word) void {
    var i: i32 = w.Start;
    while (i < @as(i32, w.End)) : (i += 1) {
        const ii: usize = @intCast(i);
        const nxt = w.Letters[ii + 1];
        const c: u8 = nxt +% (if (nxt < 0x9F) @as(u8, 0x60) else @as(u8, 0x40));
        if (w.Letters[ii] == 0xC3 and (deVowel(c) or c == 0xDF)) {
            w.Letters[ii] = c;
            if (i + 1 < @as(i32, w.End)) {
                const count: usize = @as(usize, w.End) - ii - 1;
                // DELIBERATE, DOCUMENTED DIVERGENCE FROM LITERAL C++:
                // paq8.cpp:2851 uses memcpyon OVERLAPPING regions (dest=i+1,
                // src=i+2), which is undefined behaviour and produces corrupted
                // output in cmix for umlaut-containing German words. The sibling
                // French ConvertUTF8 (paq8.cpp:2501) correctly uses memmove. We
                // use the well-defined forward copy here (== memmove for dest<src
                // == the obviously-intended behaviour). Verified: patching the C++
                // memcpy->memmove makes a 40-word accented differential test match
                // this port byte-for-byte on every field.
                std.mem.copyForwards(u8, w.Letters[ii + 1 .. ii + 1 + count], w.Letters[ii + 2 .. ii + 2 + count]);
            }
            w.End -%= 1;
        }
    }
}

fn deReplaceSharpS(w: *Word) void {
    var i: usize = w.Start;
    while (i <= @as(usize, w.End)) : (i += 1) {
        if (w.Letters[i] == 0xDF) {
            w.Letters[i] = 's';
            if (i + 1 < MAX_WORD_SIZE) {
                const count: usize = MAX_WORD_SIZE - i - 2;
                std.mem.copyBackwards(u8, w.Letters[i + 2 .. i + 2 + count], w.Letters[i + 1 .. i + 1 + count]);
                w.Letters[i + 1] = 's';
                w.End +%= boolByte(w.End < MAX_WORD_SIZE - 1);
            }
        }
    }
}

fn deMarkVowelsAsConsonants(w: *Word) void {
    var i: usize = @as(usize, w.Start) + 1;
    while (i < @as(usize, w.End)) : (i += 1) {
        const c = w.Letters[i];
        if ((c == 'u' or c == 'y') and deVowel(w.Letters[i - 1]) and deVowel(w.Letters[i + 1]))
            w.Letters[i] = toupper(c);
    }
}

fn deStep1(w: *Word, r1: u32) bool {
    var i: usize = 0;
    while (i < 3) : (i += 1) {
        const suf = de_step1[i];
        if (w.endsWith(suf) and suffixInRn(w, r1, suf.len)) {
            w.End -%= @as(u8, @intCast(suf.len));
            return true;
        }
    }
    while (i < de_step1.len) : (i += 1) {
        const suf = de_step1[i];
        if (w.endsWith(suf) and suffixInRn(w, r1, suf.len)) {
            w.End -%= @as(u8, @intCast(suf.len));
            w.End -%= boolByte(w.endsWith("niss"));
            return true;
        }
    }
    if (w.endsWith("s") and suffixInRn(w, r1, 1) and deValidEnding(w.atEnd(1), true)) {
        w.End -%= 1;
        return true;
    }
    return false;
}

fn deStep2(w: *Word, r1: u32) bool {
    for (de_step2) |suf| {
        if (w.endsWith(suf) and suffixInRn(w, r1, suf.len)) {
            w.End -%= @as(u8, @intCast(suf.len));
            return true;
        }
    }
    if (w.endsWith("st") and suffixInRn(w, r1, 2) and w.length() > 5 and deValidEnding(w.atEnd(2), false)) {
        w.End -%= 2;
        return true;
    }
    return false;
}

fn deStep3(w: *Word, r1: u32, r2: u32) bool {
    var i: usize = 0;
    while (i < 2) : (i += 1) {
        const suf = de_step3[i];
        if (w.endsWith(suf) and suffixInRn(w, r2, suf.len)) {
            w.End -%= @as(u8, @intCast(suf.len));
            if (w.endsWith("ig") and w.atEnd(2) != 'e' and suffixInRn(w, r2, 2)) w.End -%= 2;
            if (i != 0) w.Type |= German.Noun;
            return true;
        }
    }
    while (i < 5) : (i += 1) {
        const suf = de_step3[i];
        if (w.endsWith(suf) and suffixInRn(w, r2, suf.len) and w.atEnd(@as(u8, @intCast(suf.len))) != 'e') {
            w.End -%= @as(u8, @intCast(suf.len));
            if (i > 2) w.Type |= German.Adjective;
            return true;
        }
    }
    while (i < de_step3.len) : (i += 1) {
        const suf = de_step3[i];
        if (w.endsWith(suf) and suffixInRn(w, r2, suf.len)) {
            w.End -%= @as(u8, @intCast(suf.len));
            if ((w.endsWith("er") or w.endsWith("en")) and suffixInRn(w, r1, 2)) w.End -%= 2;
            if (i > 5) w.Type |= German.Noun | German.Female;
            return true;
        }
    }
    if (w.endsWith("keit") and suffixInRn(w, r2, 4)) {
        w.End -%= 4;
        if (w.endsWith("lich") and suffixInRn(w, r2, 4)) {
            w.End -%= 4;
        } else if (w.endsWith("ig") and suffixInRn(w, r2, 2)) {
            w.End -%= 2;
        }
        w.Type |= German.Noun | German.Female;
        return true;
    }
    return false;
}

fn deHash(w: *Word) void {
    w.Hash[2] = 0x415854e1; // ~0xbea7ab1e (32-bit)
    w.Hash[3] = 0x415854e1;
    var i: usize = w.Start;
    while (i <= @as(usize, w.End)) : (i += 1) {
        const l = w.Letters[i];
        w.Hash[2] = w.Hash[2] *% 263 *% 32 +% @as(u64, l);
        if (deVowel(l)) {
            w.Hash[3] = w.Hash[3] *% 997 *% 16 +% @as(u64, l);
        } else if (l >= 'b' and l <= 'z') {
            w.Hash[3] = w.Hash[3] *% 251 *% 32 +% (@as(u64, l) -% 97);
        } else {
            w.Hash[3] = w.Hash[3] *% 11 *% 32 +% @as(u64, l);
        }
    }
}

fn deStem(w: *Word) bool {
    deConvertUTF8(w);
    if (w.length() < 2) {
        deHash(w);
        return false;
    }
    deReplaceSharpS(w);
    deMarkVowelsAsConsonants(w);
    var r1 = getRegion(w, 0, &de_vowels);
    const r2 = getRegion(w, r1, &de_vowels);
    r1 = @min(3, r1);
    var res = deStep1(w, r1);
    if (deStep2(w, r1)) res = true;
    if (deStep3(w, r1, r2)) res = true;
    var i: usize = w.Start;
    while (i <= @as(usize, w.End)) : (i += 1) {
        switch (w.Letters[i]) {
            0xE4 => w.Letters[i] = 'a',
            0xF6, 0xFC => w.Letters[i] -%= 0x87,
            else => w.Letters[i] = tolower(w.Letters[i]),
        }
    }
    if (!res) res = w.matchesAny(&de_common);
    deHash(w);
    if (res) w.Language = Language.German;
    return res;
}

pub const GermanStemmer = struct {
    pub fn isVowel(_: *const GermanStemmer, c: u8) bool {
        return deVowel(c);
    }
    pub fn hash(_: *const GermanStemmer, w: *Word) void {
        deHash(w);
    }
    pub fn stem(_: *GermanStemmer, w: *Word) bool {
        return deStem(w);
    }
};

// ===========================================================================
// Cache (paq8.cpp:3006-3027) — small fixed-size ring cache.
// ===========================================================================
pub fn Cache(comptime T: type, comptime Size: u32) type {
    comptime {
        if (!(Size > 1 and (Size & (Size - 1)) == 0))
            @compileError("Cache size must be a power of 2 bigger than 1");
    }
    return struct {
        const Self = @This();
        Data: [Size]T = std.mem.zeroes([Size]T),
        Index: u32 = 0,

        /// operator(i)
        pub fn at(self: *Self, i: u32) *T {
            return &self.Data[(self.Index -% i) & (Size - 1)];
        }
        /// operator++
        pub fn inc(self: *Self) void {
            self.Index +%= 1;
        }
        /// operator--
        pub fn dec(self: *Self) void {
            self.Index -%= 1;
        }
        /// Next
        pub fn next(self: *Self) *T {
            self.Index +%= 1;
            const idx = self.Index & (Size - 1);
            self.Data[idx] = std.mem.zeroes(T);
            return &self.Data[idx];
        }
    };
}

// ===========================================================================
// TESTS
// Expected outputs were produced by compiling the exact C++ classes from
// paq8.cpp into a reference harness (g++ -std=c++17) and recording their
// output. See the scratchpad ref.cpp harness used during development.
// ===========================================================================
const testing = std.testing;

fn buildWord(s: []const u8) Word {
    var w = Word{};
    for (s) |c| w.plusEq(c);
    w.getHashes();
    return w;
}

test "Word accumulation, comparison, indexing" {
    var w = buildWord("Hello");
    try testing.expectEqual(@as(usize, 5), w.length());
    try testing.expectEqual(@as(u8, 0), w.Start);
    try testing.expectEqual(@as(u8, 4), w.End);
    try testing.expect(w.eq("hello")); // stored lowercased
    try testing.expect(!w.eq("hell"));
    try testing.expect(w.startsWith("he"));
    try testing.expect(!w.startsWith("hello")); // strict: Length()>len
    try testing.expect(w.endsWith("lo"));
    try testing.expect(!w.endsWith("hello"));
    try testing.expectEqual(@as(u8, 'h'), w.at(0));
    try testing.expectEqual(@as(u8, 'o'), w.atEnd(0));
    try testing.expectEqual(@as(u8, 'e'), w.at(1));
    try testing.expectEqual(@as(u8, 'l'), w.atEnd(1));

    var e = Word{};
    try testing.expectEqual(@as(usize, 0), e.length());

    var a = buildWord("a");
    try testing.expectEqual(@as(usize, 1), a.length());
    try testing.expect(a.eq("a"));
}

test "Word.getHashes exact values (vs C++ reference)" {
    const cases = [_]struct { s: []const u8, h0: u64, h1: u64, h2: u64, h3: u64 }{
        .{ .s = "a", .h0 = 0xf02eaeefb32783a0, .h1 = 0x3c8b3ba7494f87a7, .h2 = 0x335a6ab70597fbf8, .h3 = 0x335a6ab70597fbf8 },
        .{ .s = "ab", .h0 = 0x5cb82608b0265a6a, .h1 = 0x1960657f3467c944, .h2 = 0xba27bc887bbe6cd1, .h3 = 0xba27bc887bbe6cd1 },
        .{ .s = "hello", .h0 = 0x24d2fe7539a17ac2, .h1 = 0xff5135e73523fa5, .h2 = 0xd4d812d4b50cba98, .h3 = 0xd4d812d4b50cba98 },
        .{ .s = "the", .h0 = 0x4e94802cdf7342a2, .h1 = 0x1b8fc58e97868717, .h2 = 0xaae4ba5db70a3a4a, .h3 = 0xaae4ba5db70a3a4a },
        .{ .s = "running", .h0 = 0x9bc767332ecb4640, .h1 = 0x6fc0370b22547b4d, .h2 = 0xbf8afc7f360c2f2, .h3 = 0xbf8afc7f360c2f2 },
    };
    for (cases) |c| {
        const w = buildWord(c.s);
        try testing.expectEqual(c.h0, w.Hash[0]);
        try testing.expectEqual(c.h1, w.Hash[1]);
        try testing.expectEqual(c.h2, w.Hash[2]);
        try testing.expectEqual(c.h3, w.Hash[3]);
    }
}

test "EnglishStemmer.stem exact stems (vs C++ reference)" {
    var en = EnglishStemmer{};
    const cases = [_]struct { in: []const u8, out: []const u8 }{
        .{ .in = "running", .out = "run" },
        .{ .in = "happiness", .out = "happi" },
        .{ .in = "national", .out = "nation" },
        .{ .in = "generalization", .out = "general" },
        .{ .in = "flies", .out = "fli" },
        .{ .in = "agreed", .out = "agr" }, // NOTE: modified stemmer removes "eed" wholesale
        .{ .in = "die", .out = "die" },
        .{ .in = "dying", .out = "die" },
        .{ .in = "ties", .out = "tie" },
        .{ .in = "tying", .out = "tie" },
        .{ .in = "skis", .out = "ski" },
        .{ .in = "skies", .out = "sky" },
        .{ .in = "early", .out = "earli" },
        .{ .in = "only", .out = "onli" },
        .{ .in = "singly", .out = "singl" },
        .{ .in = "consign", .out = "consign" },
        .{ .in = "fairly", .out = "fair" },
        .{ .in = "cats", .out = "cat" },
        .{ .in = "ponies", .out = "poni" },
        .{ .in = "caresses", .out = "caress" },
        .{ .in = "feed", .out = "feed" },
        .{ .in = "succeed", .out = "succeed" },
        .{ .in = "proceed", .out = "proceed" },
        .{ .in = "inning", .out = "inning" },
        .{ .in = "housing", .out = "hous" },
        .{ .in = "hopefulness", .out = "hope" },
        .{ .in = "relational", .out = "relat" },
        .{ .in = "conditional", .out = "condition" },
        .{ .in = "rational", .out = "ration" },
        .{ .in = "goodness", .out = "good" },
        .{ .in = "analogously", .out = "analog" },
        .{ .in = "differently", .out = "differ" },
        .{ .in = "sensational", .out = "sensat" },
        .{ .in = "fluffiness", .out = "fluffi" },
        .{ .in = "controllable", .out = "control" },
        .{ .in = "survival", .out = "surviv" },
        .{ .in = "commonness", .out = "common" },
        .{ .in = "unhappy", .out = "unhappi" },
        .{ .in = "overlook", .out = "look" },
        .{ .in = "undervalue", .out = "valu" },
        .{ .in = "biggest", .out = "big" },
        .{ .in = "luckiest", .out = "lucki" },
        .{ .in = "he", .out = "he" },
        .{ .in = "she", .out = "she" },
        .{ .in = "the", .out = "the" },
        .{ .in = "fully", .out = "fulli" },
        .{ .in = "goodbye", .out = "goodby" },
        .{ .in = "neighborhood", .out = "neighbor" },
    };
    var buf: [64]u8 = undefined;
    for (cases) |c| {
        var w = buildWord(c.in);
        _ = en.stem(&w);
        try testing.expectEqualStrings(c.out, w.slice(&buf));
    }
}

test "EnglishStemmer.stem Type and Language flags" {
    var en = EnglishStemmer{};
    const cases = [_]struct { in: []const u8, ty: u64, lang: u64 }{
        .{ .in = "running", .ty = 0x101, .lang = 1 }, // PresentParticiple|Verb
        .{ .in = "happiness", .ty = 0x2000, .lang = 1 }, // SuffixNESS
        .{ .in = "national", .ty = 0x80004, .lang = 1 }, // SuffixAL|Adjective
        .{ .in = "flies", .ty = 0x8, .lang = 1 }, // Plural
        .{ .in = "he", .ty = 0x10, .lang = 1 }, // Male
        .{ .in = "she", .ty = 0x20, .lang = 1 }, // Female
        .{ .in = "succeed", .ty = 0x1, .lang = 1 }, // Verb (exception2)
        .{ .in = "inning", .ty = 0x2, .lang = 1 }, // Noun (exception2)
        .{ .in = "overlook", .ty = 0x800000, .lang = 1 }, // PrefixOver
        .{ .in = "undervalue", .ty = 0x1000000, .lang = 1 }, // PrefixUnder
    };
    for (cases) |c| {
        var w = buildWord(c.in);
        _ = en.stem(&w);
        try testing.expectEqual(c.ty, w.Type);
        try testing.expectEqual(c.lang, w.Language);
    }
}

test "EnglishStemmer.stem hash exact (vs C++ reference)" {
    var en = EnglishStemmer{};
    var w = buildWord("running");
    _ = en.stem(&w);
    try testing.expectEqual(@as(u64, 0xc590d4cfb8bbcece), w.Hash[2]);
    try testing.expectEqual(@as(u64, 0x5ed5ce45e92b282d), w.Hash[3]);
}

test "FrenchStemmer.stem exact stems (vs C++ reference)" {
    var fr = FrenchStemmer{};
    const cases = [_]struct { in: []const u8, out: []const u8, lang: u64 }{
        .{ .in = "parler", .out = "parl", .lang = 2 },
        .{ .in = "finissons", .out = "fin", .lang = 2 },
        .{ .in = "rapidement", .out = "rapid", .lang = 2 },
        .{ .in = "de", .out = "de", .lang = 2 },
        .{ .in = "que", .out = "que", .lang = 2 },
        .{ .in = "monument", .out = "monument", .lang = 2 }, // exception
        .{ .in = "yeux", .out = "oeil", .lang = 2 }, // exception
        .{ .in = "travaux", .out = "travail", .lang = 2 }, // exception
        .{ .in = "continu", .out = "continu", .lang = 0 }, // not recognized
    };
    var buf: [64]u8 = undefined;
    for (cases) |c| {
        var w = buildWord(c.in);
        _ = fr.stem(&w);
        try testing.expectEqualStrings(c.out, w.slice(&buf));
        try testing.expectEqual(c.lang, w.Language);
    }
}

test "GermanStemmer.stem exact stems (vs C++ reference)" {
    var de = GermanStemmer{};
    const cases = [_]struct { in: []const u8, out: []const u8, lang: u64 }{
        .{ .in = "heiten", .out = "heit", .lang = 3 },
        .{ .in = "kleinen", .out = "klein", .lang = 3 },
        .{ .in = "kleiner", .out = "klein", .lang = 3 },
        .{ .in = "der", .out = "der", .lang = 3 },
        .{ .in = "nicht", .out = "nicht", .lang = 3 },
        .{ .in = "lichkeit", .out = "lichkeit", .lang = 0 }, // not recognized
    };
    var buf: [64]u8 = undefined;
    for (cases) |c| {
        var w = buildWord(c.in);
        _ = de.stem(&w);
        try testing.expectEqualStrings(c.out, w.slice(&buf));
        try testing.expectEqual(c.lang, w.Language);
    }
}

// UTF-8 umlaut inputs exercise deConvertUTF8 / deReplaceSharpS and the accented
// vowel tables. These outputs are the WELL-DEFINED result (see the divergence
// note in deConvertUTF8); the literal cmix C++ produces corrupted output here
// due to memcpy-on-overlap UB, but matches this port once its memcpy is fixed
// to memmove (as the French sibling already is).
test "GermanStemmer.stem UTF-8 umlaut inputs (well-defined behavior)" {
    var de = GermanStemmer{};
    const cases = [_]struct { in: []const u8, out: []const u8, lang: u64 }{
        .{ .in = "müssen", .out = "muss", .lang = 3 },
        .{ .in = "grüßen", .out = "gruss", .lang = 3 }, // ü -> u, ß -> ss
        .{ .in = "straße", .out = "strass", .lang = 3 }, // ß -> ss
        .{ .in = "über", .out = "ub", .lang = 3 },
        .{ .in = "schönheit", .out = "schonheit", .lang = 0 },
    };
    var buf: [64]u8 = undefined;
    for (cases) |c| {
        var w = buildWord(c.in);
        _ = de.stem(&w);
        try testing.expectEqualStrings(c.out, w.slice(&buf));
        try testing.expectEqual(c.lang, w.Language);
    }
}

// French UTF-8 (accented) inputs: deConvertUTF8's sibling frConvertUTF8 uses
// memmove, so these match the literal C++ exactly (verified over 40 words).
test "FrenchStemmer.stem UTF-8 accented inputs (matches C++ exactly)" {
    var fr = FrenchStemmer{};
    const cases = [_]struct { in: []const u8, out: []const u8 }{
        .{ .in = "publié", .out = "publi" },
        .{ .in = "français", .out = "franc" },
        // ConvertUTF8 folds UTF-8 (é = C3 A9) to single-byte accented chars
        // (é = 0xE9), so the retained accent is a single 0xE9 byte, not UTF-8.
        .{ .in = "égalité", .out = "\xe9gal" },
    };
    var buf: [64]u8 = undefined;
    for (cases) |c| {
        var w = buildWord(c.in);
        _ = fr.stem(&w);
        try testing.expectEqualStrings(c.out, w.slice(&buf));
    }
}

test "Language abbreviation detection" {
    const abbr = [_][]const u8{ "mr", "mrs", "ms", "dr", "st", "jr" };
    for (abbr) |s| {
        var w = buildWord(s);
        try testing.expect(English.isAbbreviation(&w));
    }
    const non = [_][]const u8{ "xyz", "the", "cat" };
    for (non) |s| {
        var w = buildWord(s);
        try testing.expect(!English.isAbbreviation(&w));
    }
    var m = buildWord("m");
    try testing.expect(French.isAbbreviation(&m));
    var hrn = buildWord("hrn");
    try testing.expect(German.isAbbreviation(&hrn));
}

test "Cache ring semantics" {
    var c = Cache(u32, 4){};
    // Fill via next
    c.next().* = 10;
    c.next().* = 20;
    c.next().* = 30;
    try testing.expectEqual(@as(u32, 30), c.at(0).*);
    try testing.expectEqual(@as(u32, 20), c.at(1).*);
    try testing.expectEqual(@as(u32, 10), c.at(2).*);
    c.inc();
    try testing.expectEqual(@as(u32, 0), c.at(0).*); // freshly zeroed slot not yet written; wrap
}
