//! EnglishStemmer — bit-exact Zig port of `stemmer_v26.rs`, itself a 1:1 port of
//! cmix-lex `fxcmv1.cpp` Word class (2297-2680) + EnglishStemmer (2680-3240, Porter2).
//! Build a Word via `add(c)` over lowercase letters, then `stem(&W, blpos)` sets
//! W.type_/hash/suffix/preffix and the stemmed Letters. Type is consumed downstream.
//!
//! Validated bit-exact vs v26-port-tools/stemmer_oracle.cpp: TYPE_CS over all 44515
//! english.dic words. Self-contained except the `x.blpos<451531986` Verb-word gate
//! (always true at our sizes => `stem` takes blpos as a param, default 0).

const std = @import("std");

const APOSTROPHE: u8 = 39;
const MAX_WORD_SIZE: usize = 64;

// EngWordTypeFlags
const VERB: u32 = 1 << 0;
const NOUN: u32 = 1 << 1;
const ADJECTIVE: u32 = 1 << 2;
const PLURAL: u32 = 1 << 3;
const PAST_TENSE: u32 = (1 << 5) | VERB;
const PRESENT_PARTICIPLE: u32 = (1 << 4) | VERB;
const ADJECTIVE_SUPERLATIVE: u32 = (1 << 5) | ADJECTIVE;
const ADJECTIVE_WITHOUT: u32 = (1 << 6) | ADJECTIVE;
const ADJECTIVE_FULL: u32 = (1 << 7) | ADJECTIVE;
const ADVERB_OF_MANNER: u32 = 1 << 8;
const SUFFIX: u32 = 1 << 9;
const PREFIX: u32 = 1 << 10;
const MALE: u32 = 1 << 11;
const FEMALE: u32 = 1 << 13;
const ARTICLE: u32 = 1 << 14;
const CONJUNCTION: u32 = 1 << 15;
const ADPOSITION: u32 = 1 << 16;
const NUMBER: u32 = 1 << 17;
const CONJUNCTIVE_ADVERB: u32 = 1 << 19;
const PRONOUN: u32 = 1 << 20;
// Negation
const NEGATION: u32 = 1 << 0;
const PREFIX_IRR: u32 = (1 << 1) | NEGATION;
const PREFIX_OVER: u32 = 1 << 2;
const PREFIX_UNDER: u32 = 1 << 3;
const PREFIX_UNN: u32 = (1 << 4) | NEGATION;
const PREFIX_NON: u32 = (1 << 5) | NEGATION;
const PREFIX_ANTI: u32 = (1 << 6) | NEGATION;
const PREFIX_DIS: u32 = (1 << 7) | NEGATION;
// Suffix flags
const SUFFIX_NESS: u32 = 1 << 0;
const SUFFIX_ITY: u32 = (1 << 1) | NOUN;
const SUFFIX_CAPABLE: u32 = 1 << 2;
const SUFFIX_NCE: u32 = 1 << 3;
const SUFFIX_NT: u32 = 1 << 4;
const SUFFIX_ION: u32 = 1 << 5;
const SUFFIX_AL: u32 = (1 << 6) | ADJECTIVE;
const SUFFIX_IC: u32 = (1 << 7) | ADJECTIVE;
const SUFFIX_IVE: u32 = 1 << 8;
const SUFFIX_OUS: u32 = (1 << 9) | ADJECTIVE;

const VOWELS: []const u8 = "aeiouy";
const DOUBLES: []const u8 = "bdfgmnprt";
const LI_ENDINGS: []const u8 = "cdeghkmnrt";
const NON_SHORT_CONSONANTS: []const u8 = "wxY";

const PRONOUNS = [14][5][]const u8{
    .{ " ", "me", "myself", "mine", "my" },
    .{ "we", "us", "ourselves", "ours", "our" },
    .{ "you", "you", "yourself", "yours", "your" },
    .{ "thou", "thee", "thyself", "thine", "thy" },
    .{ "you", "you", "yourselves", "yours", "your" },
    .{ "he", "him", "himself", "his", "his" },
    .{ "she", "her", "herself", "hers", "her" },
    .{ "it", "it", "itself", " ", "its" },
    .{ "they", "them", "themself", "theirs", "their" },
    .{ "they", "them", "themselves", "theirs", "their" },
    .{ "one", "one", "oneself", "one's", "one's" },
    .{ "who", "whom", " ", "whose", "whose" },
    .{ "what", "what", " ", " ", " " },
    .{ "which", "which", " ", " ", " " },
};
const VERB_WORDS1 = [_][]const u8{
    "has", "had",   "have", "was",   "were", "may", "might", "must",
    "shall", "should", "can", "could", "will", "would", "is", "am",
    "are", "be", "being", "been", "do", "does", "did",
};
const NUMBERS = [_][]const u8{
    "one",    "two",   "three", "four",  "five",  "six",     "seven",
    "eight",  "nine",  "ten",   "twenty", "thirty", "forty", "fifty",
    "sixty",  "seventy", "eighty", "ninety", "hundred", "thousand", "million",
};
const CONJ_WORDS = [_][]const u8{
    "for",   "and",   "nor",     "but",   "or",       "yet",   "so",
    "than",  "as",    "that",    "if",    "when",     "because", "while",
    "where", "after", "though",  "whether", "before", "although", "like",
    "once",  "unless", "now",    "except",
};
const APO_WORDS = [_][]const u8{
    "in",     "during", "at",      "on",     "since",  "until",   "above",
    "across", "against", "along",  "among",  "around", "behind",  "below",
    "beneath", "beside", "between", "by",    "down",   "from",    "into",
    "near",   "of",     "off",     "to",     "toward", "under",   "upon",
    "with",   "within",
};
const CON_AD_VER_PREP_WORDS = [_][]const u8{ "also", "thus" };
const MALE_WORDS = [_][]const u8{ "he", "him", "his", "himself", "man", "men", "boy", "husband", "actor" };
const FEMALE_WORDS = [_][]const u8{ "she", "her", "herself", "woman", "women", "girl", "wife", "actress" };
const ARTICLE_WORDS = [_][]const u8{ "a", "an", "the" };
const SUFFIXES_STEP0 = [_][]const u8{ "'s'", "'s", "'" };
const SUFFIXES_STEP1B = [_][]const u8{ "eedly", "eed", "ed", "edly", "ing", "ingly" };
const TYPES_STEP1B = [6]u32{
    ADVERB_OF_MANNER,
    0,
    PAST_TENSE,
    ADVERB_OF_MANNER | PAST_TENSE,
    PRESENT_PARTICIPLE,
    ADVERB_OF_MANNER | PRESENT_PARTICIPLE,
};
const SUFFIXES_STEP2 = [22][2][]const u8{
    .{ "ization", "ize" }, .{ "ational", "ate" }, .{ "ousness", "ous" }, .{ "iveness", "ive" },
    .{ "fulness", "ful" }, .{ "tional", "tion" }, .{ "lessli", "less" }, .{ "biliti", "ble" },
    .{ "entli", "ent" },   .{ "ation", "ate" },   .{ "alism", "al" },    .{ "aliti", "al" },
    .{ "fulli", "ful" },   .{ "ousli", "ous" },   .{ "iviti", "ive" },   .{ "enci", "ence" },
    .{ "anci", "ance" },   .{ "abli", "able" },   .{ "izer", "ize" },    .{ "ator", "ate" },
    .{ "alli", "al" },     .{ "bli", "ble" },
};
const TYPES_STEP2 = [22]u32{
    SUFFIX,           SUFFIX | ADJECTIVE, SUFFIX, SUFFIX, SUFFIX, SUFFIX | ADJECTIVE, ADVERB_OF_MANNER,
    ADVERB_OF_MANNER | NOUN | SUFFIX, ADVERB_OF_MANNER, SUFFIX, 0, NOUN | SUFFIX, ADVERB_OF_MANNER,
    ADVERB_OF_MANNER, NOUN | SUFFIX, 0, 0, ADVERB_OF_MANNER, 0, 0, ADVERB_OF_MANNER, ADVERB_OF_MANNER,
};
const TYPES_STEP2_SUFFIX = [22]u32{
    SUFFIX_ION, SUFFIX_ION | SUFFIX_AL, SUFFIX_NESS, SUFFIX_NESS, SUFFIX_NESS, SUFFIX_ION | SUFFIX_AL,
    0, SUFFIX_ITY, 0, SUFFIX_ION, 0, SUFFIX_ITY, 0, 0, SUFFIX_ITY, 0, 0, 0, 0, 0, 0, 0,
};
const SUFFIXES_STEP3 = [8][2][]const u8{
    .{ "ational", "ate" }, .{ "tional", "tion" }, .{ "alize", "al" }, .{ "icate", "ic" },
    .{ "iciti", "ic" },    .{ "ical", "ic" },     .{ "ful", "" },     .{ "ness", "" },
};
const TYPES_STEP3 = [8]u32{
    SUFFIX | ADJECTIVE, SUFFIX | ADJECTIVE, 0, 0, NOUN | SUFFIX, SUFFIX | ADJECTIVE, ADJECTIVE_FULL,
    SUFFIX,
};
const TYPES_STEP3_SUFFIX = [8]u32{ SUFFIX_ION | SUFFIX_AL, SUFFIX_ION | SUFFIX_AL, 0, 0, SUFFIX_ITY, SUFFIX_AL, 0, SUFFIX_NESS };
const SUFFIXES_STEP4 = [_][]const u8{
    "al",  "ance", "ence", "er",  "ic",   "able", "ible", "ant", "ement", "ment",
    "ent", "ou",   "ism",  "ate", "iti",  "ous",  "ive",  "ize", "sion",  "tion",
};
const TYPES_STEP4 = [20]u32{
    SUFFIX | ADJECTIVE, SUFFIX, SUFFIX, 0, SUFFIX | ADJECTIVE, SUFFIX, SUFFIX, SUFFIX, 0, 0, SUFFIX,
    0, 0, 0, SUFFIX | NOUN, SUFFIX | ADJECTIVE, SUFFIX, 0, SUFFIX, SUFFIX,
};
const TYPES_STEP4_SUFFIX = [20]u32{
    SUFFIX_AL, SUFFIX_NCE, SUFFIX_NCE, 0, SUFFIX_IC, SUFFIX_CAPABLE, SUFFIX_CAPABLE, SUFFIX_NT, 0, 0,
    SUFFIX_NT, 0, 0, 0, SUFFIX_ITY, SUFFIX_OUS, SUFFIX_IVE, 0, SUFFIX_ION, SUFFIX_ION,
};
const EXCEPTIONS_REGION1 = [_][]const u8{ "gener", "arsen", "commun" };
const EXCEPTIONS1 = [19][2][]const u8{
    .{ "skis", "ski" },   .{ "skies", "sky" },   .{ "dying", "die" },   .{ "lying", "lie" },
    .{ "tying", "tie" },  .{ "idly", "idle" },   .{ "gently", "gentle" }, .{ "ugly", "ugli" },
    .{ "early", "earli" }, .{ "only", "onli" },  .{ "singly", "singl" }, .{ "sky", "sky" },
    .{ "news", "news" },  .{ "howe", "howe" },   .{ "atlas", "atlas" }, .{ "cosmos", "cosmos" },
    .{ "bias", "bias" },  .{ "andes", "andes" }, .{ "texas", "texas" },
};
const TYPES_EXCEPTIONS1 = [19]u32{
    NOUN | PLURAL, NOUN | PLURAL, PRESENT_PARTICIPLE, PRESENT_PARTICIPLE, PRESENT_PARTICIPLE,
    ADVERB_OF_MANNER, ADVERB_OF_MANNER, ADJECTIVE, ADJECTIVE | ADVERB_OF_MANNER, 0, ADVERB_OF_MANNER,
    NOUN, NOUN, 0, NOUN, NOUN, NOUN, NOUN | PLURAL, NOUN,
};
const EXCEPTIONS2 = [_][]const u8{ "inning", "outing", "canning", "herring", "earring", "proceed", "exceed", "succeed" };
const TYPES_EXCEPTIONS2 = [8]u32{ NOUN, NOUN, NOUN, NOUN, NOUN, VERB, VERB, VERB };

inline fn char_in_array(c: u8, a: []const u8) bool {
    return std.mem.indexOfScalar(u8, a, c) != null;
}

inline fn b(x: bool) u8 {
    return @intFromBool(x);
}

pub const Word = struct {
    letters: [MAX_WORD_SIZE]u8,
    start: u8,
    end: u8,
    hash: u32,
    type_: u32,
    suffix: u32,
    preffix: u32,
    next_w: u32,

    pub fn new() Word {
        return Word{
            .letters = [_]u8{0} ** MAX_WORD_SIZE,
            .start = 0,
            .end = 0,
            .hash = 0,
            .type_ = 0,
            .suffix = 0,
            .preffix = 0,
            .next_w = 0,
        };
    }

    /// operator+=(c)
    pub fn add(self: *Word, c: u8) void {
        if (c > 0 and @as(usize, self.end) < MAX_WORD_SIZE - 1) {
            self.end += b(self.letters[self.end] > 0);
            self.letters[self.end] = c;
        }
    }

    /// operator[](i): i-th from start
    inline fn at(self: *const Word, i: u8) u8 {
        if (@as(i32, self.end) - @as(i32, self.start) >= @as(i32, i)) {
            return self.letters[@as(usize, self.start) + @as(usize, i)];
        } else {
            return 0;
        }
    }

    /// operator(i): i-th from end (0 = last)
    inline fn rat(self: *const Word, i: u8) u8 {
        if (@as(i32, self.end) - @as(i32, self.start) >= @as(i32, i)) {
            return self.letters[@as(usize, self.end) - @as(usize, i)];
        } else {
            return 0;
        }
    }

    pub inline fn length(self: *const Word) u32 {
        if (self.letters[self.start] != 0) {
            return @as(u32, self.end -% self.start +% 1);
        } else {
            return 0;
        }
    }

    /// bytes letters[start .. start+n] (bounds-safe; returns null if out of range)
    inline fn span_from_start(self: *const Word, n: usize) ?[]const u8 {
        const s: usize = self.start;
        if (s + n <= MAX_WORD_SIZE) {
            return self.letters[s .. s + n];
        } else {
            return null;
        }
    }

    fn eq_str(self: *const Word, s: []const u8) bool {
        const len = s.len;
        const sz: i32 = (@as(i32, self.end) - @as(i32, self.start)) + @as(i32, b(self.letters[self.start] != 0));
        const sz_usize: usize = @bitCast(@as(isize, sz));
        return sz_usize == len and optEq(self.span_from_start(len), s);
    }

    fn change_suffix(self: *Word, old: []const u8, newv: []const u8) bool {
        const len = old.len;
        if (@as(usize, self.length()) > len and @as(usize, self.end) + 1 >= len and
            std.mem.eql(u8, self.letters[@as(usize, self.end) + 1 - len .. @as(usize, self.end) + 1], old))
        {
            const n = newv.len;
            if (n > 0) {
                const dst = @as(usize, self.end) + 1 - len;
                const copy_len = @min(MAX_WORD_SIZE - 1, @as(usize, self.end) + n) - @as(usize, self.end);
                var k: usize = 0;
                while (k < copy_len) : (k += 1) {
                    self.letters[dst + k] = newv[k];
                }
                self.end = @intCast(@min(@as(i32, MAX_WORD_SIZE - 1), @as(i32, self.end) - @as(i32, @intCast(len)) + @as(i32, @intCast(n))));
            } else {
                self.end -= @as(u8, @intCast(len));
            }
            return true;
        } else {
            return false;
        }
    }

    fn matches_any(self: *const Word, a: []const []const u8) bool {
        const len: usize = self.length();
        for (a) |s| {
            if (len == s.len and optEq(self.span_from_start(len), s)) {
                return true;
            }
        }
        return false;
    }

    fn matches_any_p(self: *Word, a: []const []const u8) bool {
        self.next_w = 0xff;
        const len: usize = self.length();
        for (a, 0..) |s, i| {
            if (len == s.len and optEq(self.span_from_start(len), s)) {
                self.next_w = @intCast(i);
                return true;
            }
        }
        return false;
    }

    fn ends_with(self: *const Word, suffix: []const u8) bool {
        const len = suffix.len;
        return @as(usize, self.length()) > len and @as(usize, self.end) + 1 >= len and
            std.mem.eql(u8, self.letters[@as(usize, self.end) + 1 - len .. @as(usize, self.end) + 1], suffix);
    }

    fn starts_with(self: *const Word, prefix: []const u8) bool {
        const len = prefix.len;
        return @as(usize, self.length()) > len and optEq(self.span_from_start(len), prefix);
    }

    /// memcmp(s, &Letters[End-off], len)==0  (bounds-safe)
    inline fn mem_at_end(self: *const Word, off: usize, s: []const u8) bool {
        if (@as(usize, self.end) >= off and @as(usize, self.end) - off + s.len <= MAX_WORD_SIZE) {
            return std.mem.eql(u8, self.letters[@as(usize, self.end) - off .. @as(usize, self.end) - off + s.len], s);
        } else {
            return false;
        }
    }
};

inline fn optEq(a: ?[]const u8, s: []const u8) bool {
    if (a) |x| return std.mem.eql(u8, x, s);
    return false;
}

inline fn is_vowel(c: u8) bool {
    return char_in_array(c, VOWELS);
}
inline fn is_consonant(c: u8) bool {
    return !is_vowel(c);
}
inline fn is_short_consonant(c: u8) bool {
    return !char_in_array(c, NON_SHORT_CONSONANTS);
}
inline fn is_double(c: u8) bool {
    return char_in_array(c, DOUBLES);
}
inline fn is_li_ending(c: u8) bool {
    return char_in_array(c, LI_ENDINGS);
}

fn hash(w: *Word) void {
    var h: u32 = 0xb0a710ad;
    var i: usize = w.start;
    while (i <= @as(usize, w.end)) : (i += 1) {
        h = h *% 263 *% 32 +% @as(u32, w.letters[i]);
    }
    w.hash = h;
}

fn get_region(w: *const Word, from: u32) u32 {
    var has_vowel = false;
    var i: i32 = @as(i32, w.start) + @as(i32, @intCast(from));
    while (i <= @as(i32, w.end)) : (i += 1) {
        const c = w.letters[@as(usize, @intCast(i))];
        if (is_vowel(c)) {
            has_vowel = true;
        } else if (has_vowel) {
            return @as(u32, @intCast(i - @as(i32, w.start) + 1));
        }
    }
    return w.length();
}

fn get_region1(w: *const Word) u32 {
    for (EXCEPTIONS_REGION1) |e| {
        if (w.starts_with(e)) {
            return @intCast(e.len);
        }
    }
    return get_region(w, 0);
}

inline fn suffix_in_rn(w: *const Word, rn: u32, suffix: []const u8) bool {
    return w.start != w.end and @as(u64, rn) <= @as(u64, w.length()) -% @as(u64, suffix.len);
}

fn ends_in_short_syllable(w: *const Word) bool {
    if (w.end == w.start) {
        return false;
    } else if (w.end == w.start + 1) {
        return is_vowel(w.rat(1)) and is_consonant(w.rat(0));
    } else {
        return is_consonant(w.rat(2)) and is_vowel(w.rat(1)) and is_consonant(w.rat(0)) and is_short_consonant(w.rat(0));
    }
}

fn is_short_word(w: *const Word) bool {
    return ends_in_short_syllable(w) and get_region1(w) == w.length();
}

fn has_vowels(w: *const Word) bool {
    var i: usize = w.start;
    while (i <= @as(usize, w.end)) : (i += 1) {
        if (is_vowel(w.letters[i])) {
            return true;
        }
    }
    return false;
}

fn trim_starting_apostrophe(w: *Word) bool {
    var result = false;
    var cnt: i32 = 0;
    while (w.start != w.end and w.at(0) == APOSTROPHE) {
        result = true;
        w.start += 1;
        cnt += 1;
    }
    while (w.start != w.end and w.rat(0) == APOSTROPHE) {
        if (cnt == 0) {
            break;
        }
        w.end -= 1;
        cnt -= 1;
    }
    if (w.start != w.end and w.rat(0) == '-') {
        w.end -= 1;
    }
    return result;
}

fn mark_ys_as_consonants(w: *Word) void {
    if (w.at(0) == 'y') {
        w.letters[w.start] = 'Y';
    }
    var i: usize = @as(usize, w.start) + 1;
    while (i <= @as(usize, w.end)) : (i += 1) {
        if (is_vowel(w.letters[i - 1]) and w.letters[i] == 'y') {
            w.letters[i] = 'Y';
        }
    }
}

fn process_prefixes(w: *Word) bool {
    if (w.starts_with("irr") and w.length() > 5 and (w.at(3) == 'a' or w.at(3) == 'e')) {
        w.start += 2;
        w.type_ |= PREFIX;
        w.preffix |= PREFIX_IRR;
    } else if (w.starts_with("over") and w.length() > 5) {
        w.start += 4;
        w.type_ |= PREFIX;
        w.preffix |= PREFIX_OVER;
    } else if (w.starts_with("under") and w.length() > 6) {
        w.start += 5;
        w.type_ |= PREFIX;
        w.preffix |= PREFIX_UNDER;
    } else if (w.starts_with("unn") and w.length() > 5) {
        w.start += 2;
        w.type_ |= PREFIX;
        w.preffix |= PREFIX_UNN;
    } else if (w.starts_with("non") and w.length() > (5 + @as(u32, b(w.at(3) == '-')))) {
        w.start += 2 + b(w.at(3) == '-');
        w.type_ |= PREFIX;
        w.preffix |= PREFIX_NON;
    } else if (w.starts_with("anti") and w.length() > 6 and (w.at(4) == '-')) {
        w.start += 4 + b(w.at(4) == '-');
        w.type_ |= PREFIX;
        w.preffix |= PREFIX_ANTI;
    } else if (w.starts_with("dis") and w.length() > 5 and (w.at(3) == '-')) {
        w.start += 2 + b(w.at(3) == '-');
        w.type_ |= PREFIX;
        w.preffix |= PREFIX_DIS;
    } else {
        return false;
    }
    return true;
}

fn process_superlatives(w: *Word) bool {
    if (w.ends_with("est") and w.length() > 4) {
        const i = w.end;
        w.end -= 3;
        w.type_ |= ADJECTIVE_SUPERLATIVE;
        if (w.rat(0) == w.rat(1) and w.rat(0) != 'r' and !(w.length() >= 4 and w.mem_at_end(3, "sugg"))) {
            const dec = ((w.rat(0) != 'f' and w.rat(0) != 'l' and w.rat(0) != 's') or
                (w.length() > 4 and w.rat(1) == 'l' and (w.rat(2) == 'u' or w.rat(3) == 'u' or w.rat(3) == 'v'))) and
                !(w.length() == 3 and w.rat(1) == 'd' and w.rat(2) == 'o');
            w.end -= b(dec);
            if (w.length() == 2 and (w.at(0) != 'i' or w.at(1) != 'n')) {
                w.end = i;
                w.type_ &= ~ADJECTIVE_SUPERLATIVE;
            }
        } else {
            switch (w.rat(0)) {
                'd', 'k', 'm', 'y' => {},
                'g' => {
                    if (!(w.length() > 3 and (w.rat(1) == 'n' or w.rat(1) == 'r') and !w.mem_at_end(3, "cong"))) {
                        w.end = i;
                        w.type_ &= ~ADJECTIVE_SUPERLATIVE;
                    } else {
                        w.end += b(w.rat(2) == 'a');
                    }
                },
                'i' => {
                    w.letters[w.end] = 'y';
                },
                'l' => {
                    if (w.end == w.start + 1 or w.mem_at_end(2, "mo")) {
                        w.end = i;
                        w.type_ &= ~ADJECTIVE_SUPERLATIVE;
                    } else {
                        w.end += b(is_consonant(w.rat(1)));
                    }
                },
                'n' => {
                    if (w.length() < 3 or is_consonant(w.rat(1)) or is_consonant(w.rat(2))) {
                        w.end = i;
                        w.type_ &= ~ADJECTIVE_SUPERLATIVE;
                    }
                },
                'r' => {
                    if (w.length() > 3 and is_vowel(w.rat(1)) and is_vowel(w.rat(2))) {
                        w.end += b((w.rat(2) == 'u') and (w.rat(1) == 'a' or w.rat(1) == 'i'));
                    } else {
                        w.end = i;
                        w.type_ &= ~ADJECTIVE_SUPERLATIVE;
                    }
                },
                's' => {
                    w.end += 1;
                },
                'w' => {
                    if (!(w.length() > 2 and is_vowel(w.rat(1)))) {
                        w.end = i;
                        w.type_ &= ~ADJECTIVE_SUPERLATIVE;
                    }
                },
                'h' => {
                    if (!(w.length() > 2 and is_consonant(w.rat(1)))) {
                        w.end = i;
                        w.type_ &= ~ADJECTIVE_SUPERLATIVE;
                    }
                },
                else => {
                    w.end += 3;
                    w.type_ &= ~ADJECTIVE_SUPERLATIVE;
                },
            }
        }
    }
    return (w.type_ & ADJECTIVE_SUPERLATIVE) > 0;
}

fn step0(w: *Word) bool {
    for (SUFFIXES_STEP0) |s| {
        if (w.ends_with(s)) {
            w.end -= @as(u8, @intCast(s.len));
            w.type_ |= PLURAL;
            return true;
        }
    }
    return false;
}

fn step1a(w: *Word) bool {
    if (w.ends_with("sses")) {
        w.end -= 2;
        w.type_ |= PLURAL;
        return true;
    }
    if (w.ends_with("ied") or w.ends_with("ies")) {
        w.type_ |= if (w.rat(0) == 'd') PAST_TENSE else PLURAL;
        w.end -= 1 + b(w.length() > 4);
        return true;
    }
    if (w.ends_with("us") or w.ends_with("ss")) {
        return false;
    }
    if (w.rat(0) == 's' and w.length() > 2) {
        var i: i32 = w.start;
        while (i <= @as(i32, w.end) - 2) : (i += 1) {
            if (is_vowel(w.letters[@as(usize, @intCast(i))])) {
                w.end -= 1;
                w.type_ |= PLURAL;
                return true;
            }
        }
    }
    if (w.ends_with("n't") and w.length() > 4) {
        switch (w.rat(3)) {
            'a' => {
                if (w.rat(4) == 'c') {
                    w.end -= 2;
                } else {
                    _ = w.change_suffix("n't", "ll");
                }
            },
            'i' => {
                _ = w.change_suffix("in't", "m");
            },
            'o' => {
                if (w.rat(4) == 'w') {
                    _ = w.change_suffix("on't", "ill");
                } else {
                    w.end -= 3;
                }
            },
            else => {
                w.end -= 3;
            },
        }
        w.type_ |= PREFIX;
        w.preffix |= NEGATION;
        return true;
    }
    if (w.ends_with("hood") and w.length() > 7) {
        w.end -= 4;
        return true;
    }
    return false;
}

fn step1b(w: *Word, r1: u32) bool {
    for (SUFFIXES_STEP1B, 0..) |suf, i| {
        if (w.ends_with(suf)) {
            switch (i) {
                0, 1 => {
                    if (suffix_in_rn(w, r1, suf)) {
                        w.end -= 1 + @as(u8, @intCast(i * 2));
                    }
                },
                else => {
                    const j = w.end;
                    w.end -= @as(u8, @intCast(suf.len));
                    if (has_vowels(w)) {
                        if (w.ends_with("at") or w.ends_with("bl") or w.ends_with("iz") or is_short_word(w)) {
                            w.add('e');
                        } else if (w.length() > 2) {
                            if (w.rat(0) == w.rat(1) and is_double(w.rat(0))) {
                                w.end -= 1;
                            } else if (i == 2 or i == 3) {
                                switch (w.rat(0)) {
                                    'c', 's', 'v' => {
                                        w.end += b(!(w.ends_with("ss") or w.ends_with("ias")));
                                    },
                                    'd' => {
                                        w.end += b(is_vowel(w.rat(1)) and !char_in_array(w.rat(2), "aeio"));
                                    },
                                    'k' => {
                                        w.end += b(w.ends_with("uak"));
                                    },
                                    'l' => {
                                        w.end += b(char_in_array(w.rat(1), "bcdfgkptyz") or
                                            (char_in_array(w.rat(1), "aiou") and is_consonant(w.rat(2))));
                                    },
                                    else => {},
                                }
                            } else if (i >= 4) {
                                switch (w.rat(0)) {
                                    'd' => {
                                        if (is_vowel(w.rat(1)) and w.rat(2) != 'a' and w.rat(2) != 'e' and w.rat(2) != 'o') {
                                            w.add('e');
                                        }
                                    },
                                    'g' => {
                                        if (char_in_array(w.rat(1), "adeilru") or
                                            (w.rat(1) == 'n' and
                                                (w.rat(2) == 'e' or
                                                    (w.rat(2) == 'u' and w.rat(3) != 'b' and w.rat(3) != 'd') or
                                                    (w.rat(2) == 'a' and (w.rat(3) == 'r' or (w.rat(3) == 'h' and w.rat(4) == 'c'))) or
                                                    (w.ends_with("ring") and (w.rat(4) == 'c' or w.rat(4) == 'f')))))
                                        {
                                            w.add('e');
                                        }
                                    },
                                    'l' => {
                                        if (!(w.rat(1) == 'l' or w.rat(1) == 'r' or w.rat(1) == 'w' or (is_vowel(w.rat(1)) and is_vowel(w.rat(2))))) {
                                            w.add('e');
                                        }
                                        if (w.ends_with("uell") and w.length() > 4 and w.rat(4) != 'q') {
                                            w.end -= 1;
                                        }
                                    },
                                    'r' => {
                                        if (((w.rat(1) == 'i' and w.rat(2) != 'a' and w.rat(2) != 'e' and w.rat(2) != 'o') or
                                            (w.rat(1) == 'a' and !(w.rat(2) == 'e' or w.rat(2) == 'o' or (w.rat(2) == 'l' and w.rat(3) == 'l'))) or
                                            (w.rat(1) == 'o' and !(w.rat(2) == 'o' or (w.rat(2) == 't' and w.rat(3) != 's'))) or
                                            w.rat(1) == 'c' or
                                            w.rat(1) == 't') and
                                            !w.ends_with("str"))
                                        {
                                            w.add('e');
                                        }
                                    },
                                    't' => {
                                        if (w.rat(1) == 'o' and w.rat(2) != 'g' and w.rat(2) != 'l' and w.rat(2) != 'i' and w.rat(2) != 'o') {
                                            w.add('e');
                                        }
                                    },
                                    'u' => {
                                        if (!(w.length() > 3 and is_vowel(w.rat(1)) and is_vowel(w.rat(2)))) {
                                            w.add('e');
                                        }
                                    },
                                    'z' => {
                                        if (w.ends_with("izz") and w.length() > 3 and (w.rat(3) == 'h' or w.rat(3) == 'u')) {
                                            w.end -= 1;
                                        } else if (w.rat(1) != 't' and w.rat(1) != 'z') {
                                            w.add('e');
                                        }
                                    },
                                    'k' => {
                                        if (w.ends_with("uak")) {
                                            w.add('e');
                                        }
                                    },
                                    'b', 'c', 's', 'v' => {
                                        if (!((w.rat(0) == 'b' and (w.rat(1) == 'm' or w.rat(1) == 'r')) or
                                            w.ends_with("ss") or
                                            w.ends_with("ias") or
                                            w.eq_str("zinc")))
                                        {
                                            w.add('e');
                                        }
                                    },
                                    else => {},
                                }
                            }
                        }
                    } else {
                        w.end = j;
                        return false;
                    }
                },
            }
            w.type_ |= TYPES_STEP1B[i];
            return true;
        }
    }
    return false;
}

fn step1c(w: *Word) bool {
    if (w.length() > 2 and w.rat(0) == 'y' and is_consonant(w.rat(1))) {
        w.letters[w.end] = 'i';
        return true;
    }
    return false;
}

fn step2(w: *Word, r1: u32) bool {
    for (SUFFIXES_STEP2, 0..) |pair, i| {
        if (w.ends_with(pair[0]) and suffix_in_rn(w, r1, pair[0])) {
            _ = w.change_suffix(pair[0], pair[1]);
            w.type_ |= TYPES_STEP2[i];
            w.suffix |= TYPES_STEP2_SUFFIX[i];
            return true;
        }
    }
    if (w.ends_with("logi") and suffix_in_rn(w, r1, "ogi")) {
        w.end -= 1;
        return true;
    } else if (w.ends_with("li")) {
        if (suffix_in_rn(w, r1, "li") and is_li_ending(w.rat(2))) {
            w.end -= 2;
            w.type_ |= ADVERB_OF_MANNER;
            return true;
        } else if (w.length() > 3) {
            switch (w.rat(2)) {
                'b' => {
                    w.letters[w.end] = 'e';
                    w.type_ |= ADVERB_OF_MANNER;
                    return true;
                },
                'i' => {
                    if (w.length() > 4) {
                        w.end -= 2;
                        w.type_ |= ADVERB_OF_MANNER;
                        return true;
                    }
                },
                'l' => {
                    if (w.length() > 5 and (w.rat(3) == 'a' or w.rat(3) == 'u')) {
                        w.end -= 2;
                        w.type_ |= ADVERB_OF_MANNER;
                        return true;
                    }
                },
                's' => {
                    w.end -= 2;
                    w.type_ |= ADVERB_OF_MANNER;
                    return true;
                },
                'e', 'g', 'm', 'n', 'r', 'w' => {
                    if (w.length() > (4 + @as(u32, b(w.rat(2) == 'r')))) {
                        w.end -= 2;
                        w.type_ |= ADVERB_OF_MANNER;
                        return true;
                    }
                },
                else => {},
            }
        }
    }
    return false;
}

fn step3(w: *Word, r1: u32, r2: u32) bool {
    var res = false;
    for (SUFFIXES_STEP3, 0..) |pair, i| {
        if (w.ends_with(pair[0]) and suffix_in_rn(w, r1, pair[0])) {
            _ = w.change_suffix(pair[0], pair[1]);
            w.type_ |= TYPES_STEP3[i];
            w.suffix |= TYPES_STEP3_SUFFIX[i];
            res = true;
            break;
        }
    }
    if (w.ends_with("ative") and suffix_in_rn(w, r2, "ative")) {
        w.end -= 5;
        w.type_ |= SUFFIX;
        w.suffix |= SUFFIX_IVE;
        return true;
    }
    if (w.length() > 5 and w.ends_with("less")) {
        w.end -= 4;
        w.type_ |= ADJECTIVE_WITHOUT;
        return true;
    }
    return res;
}

fn step4(w: *Word, r2: u32) bool {
    var res = false;
    for (SUFFIXES_STEP4, 0..) |suf, i| {
        if (w.ends_with(suf) and suffix_in_rn(w, r2, suf)) {
            w.end -= @as(u8, @intCast(suf.len - @as(usize, b(i > 17))));
            if (i != 10 or w.rat(0) != 'm') {
                w.type_ |= TYPES_STEP4[i];
                w.suffix |= TYPES_STEP4_SUFFIX[i];
            }
            if (i == 0 and w.ends_with("nti")) {
                w.end -= 1;
                res = true;
                continue;
            }
            return true;
        }
    }
    return res;
}

fn step5(w: *Word, r1: u32, r2: u32) bool {
    if (w.rat(0) == 'e' and !w.eq_str("here")) {
        if (suffix_in_rn(w, r2, "e")) {
            w.end -= 1;
        } else if (suffix_in_rn(w, r1, "e")) {
            w.end -= 1;
            w.end += b(ends_in_short_syllable(w));
        } else {
            return false;
        }
        return true;
    } else if (w.length() > 1 and w.rat(0) == 'l' and suffix_in_rn(w, r2, "l") and w.rat(1) == 'l') {
        w.end -= 1;
        return true;
    }
    return false;
}

/// EnglishStemmer::Stem. `blpos` is x.blpos (gate for the Verb-word list; <451531986
/// always true at our sizes).
pub fn stem(w: *Word, blpos: u32) bool {
    var res = trim_starting_apostrophe(w);
    if (process_prefixes(w)) {
        res = true;
    }
    if (process_superlatives(w)) {
        res = true;
    }
    for (EXCEPTIONS1, 0..) |pair, i| {
        if (w.eq_str(pair[0])) {
            if (i < 11) {
                const new = pair[1];
                const len = new.len;
                var k: usize = 0;
                while (k < len) : (k += 1) {
                    w.letters[@as(usize, w.start) + k] = new[k];
                }
                w.end = w.start + @as(u8, @intCast(len - 1));
            }
            hash(w);
            w.type_ |= TYPES_EXCEPTIONS1[i];
            return i < 11;
        }
    }
    mark_ys_as_consonants(w);
    const r1 = get_region1(w);
    const r2 = get_region(w, r1);
    if (step0(w)) {
        res = true;
    }
    if (step1a(w)) {
        res = true;
    }
    for (EXCEPTIONS2, 0..) |s, i| {
        if (w.eq_str(s)) {
            hash(w);
            w.type_ |= TYPES_EXCEPTIONS2[i];
            return res;
        }
    }
    if (step1b(w, r1)) {
        res = true;
    }
    if (step1c(w)) {
        res = true;
    }
    if (step2(w, r1)) {
        res = true;
    }
    if (step3(w, r1, r2)) {
        res = true;
    }
    if (step4(w, r2)) {
        res = true;
    }
    if (step5(w, r1, r2)) {
        res = true;
    }
    var i: usize = w.start;
    while (i <= @as(usize, w.end)) : (i += 1) {
        if (w.letters[i] == 'Y') {
            w.letters[i] = 'y';
        }
    }
    if (w.type_ == 0 or w.type_ == PLURAL) {
        if (w.matches_any(&MALE_WORDS)) {
            res = true;
            w.type_ |= MALE;
        } else if (w.matches_any(&FEMALE_WORDS)) {
            res = true;
            w.type_ |= FEMALE;
        } else if (w.matches_any(&ARTICLE_WORDS)) {
            res = true;
            w.type_ |= ARTICLE;
        } else if (w.matches_any(&CONJ_WORDS)) {
            res = true;
            w.type_ |= CONJUNCTION;
        } else if (w.matches_any(&APO_WORDS)) {
            res = true;
            w.type_ |= ADPOSITION;
        } else if (w.matches_any(&CON_AD_VER_PREP_WORDS)) {
            res = true;
            w.type_ |= CONJUNCTIVE_ADVERB;
        } else if (blpos < 451531986 and w.matches_any(&VERB_WORDS1)) {
            res = true;
            w.type_ |= VERB;
        } else if (w.matches_any(&NUMBERS)) {
            res = true;
            w.type_ |= NUMBER;
        } else {
            var pi: usize = 0;
            while (pi < 14) : (pi += 1) {
                if (w.matches_any_p(&PRONOUNS[pi])) {
                    res = true;
                    w.type_ |= PRONOUN;
                    break;
                }
            }
        }
    }
    hash(w);
    return res;
}

test "stemmer_matches_cpp_type_oracle" {
    const dict: []const u8 = @embedFile("goldens/english.dic");
    var cs: u64 = 0;
    var nwords: u32 = 0;
    var it = std.mem.splitScalar(u8, dict, '\n');
    while (it.next()) |raw| {
        // strip trailing \r if present
        var line = raw;
        if (line.len > 0 and line[line.len - 1] == '\r') {
            line = line[0 .. line.len - 1];
        }
        if (line.len == 0) {
            continue;
        }
        var w = Word.new();
        for (line) |c| {
            w.add(c);
        }
        _ = stem(&w, 0);
        cs = cs *% 1000003 +% @as(u64, w.type_);
        cs = cs *% 1000003 +% @as(u64, w.hash);
        cs = cs *% 1000003 +% @as(u64, w.suffix);
        cs = cs *% 1000003 +% @as(u64, w.preffix);
        cs = cs *% 1000003 +% @as(u64, @as(u32, w.start) * 256 + @as(u32, w.end));
        var i: usize = w.start;
        while (i <= @as(usize, w.end) and i < MAX_WORD_SIZE) : (i += 1) {
            cs = cs *% 1000003 +% @as(u64, w.letters[i]);
        }
        nwords += 1;
    }
    try std.testing.expectEqual(@as(u32, 44515), nwords);
    try std.testing.expectEqual(@as(u64, 0x6318477ab324400e), cs);
}
