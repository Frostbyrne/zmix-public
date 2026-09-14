//! ParseByte — bit-exact Zig port of cmix-lex `fxcmv1.cpp` parseByte (4389-5041) +
//! setbufstem/setbuf/updateSen/procWord + the post-parseByte 5443 sVerb-LF reset
//! (apply_wshift_block). Ported 1:1 from parse_byte_v26.rs (the golden Rust port).
//!
//! It owns/updates the sub-context models: bracket_context (brcxt/qocxt/fccxt),
//! bracket_context_w (htcxt), column_context (colcxt), words_context (worcxt0..3),
//! sentence_context (sencxt family), stemmer (stem_words), decoded_buffer (db).
//!
//! Validated bit-exact vs the three oracle dumps over all 15007 stream bytes:
//!   WORCXT_CS = 0xca6fcbfd442eac3e  (word machinery + lastWT + sVerb)
//!   PRE2_CS   = 0x3b1dd0ab214a83f4  (worcxt0/1/2/3, htcxt, PState, indirect, sencxt, ...)
//!   ctxdump full-column (c1/c2/c3/stream*/x4/word0/words/numbers/deccode/t[1..13]).

const std = @import("std");
const zero_alloc = @import("cmf_zero_alloc.zig"); // in cmfast: module-private copy

const bc_mod = @import("bracket_context.zig");
const BracketContext = bc_mod.BracketContext;
const BracketContextW = bc_mod.BracketContextW;
const ColumnContext = @import("column_context.zig").ColumnContext;
const SentenceContext = @import("cmfast").SentenceContext;
const db_mod = @import("decoded_buffer.zig");
const DecodedBuffer = db_mod.DecodedBuffer;
const char_swap = db_mod.char_swap;
const stemmer = @import("stemmer.zig");
const Word = stemmer.Word;
const stem = stemmer.stem;
const WordsContext = @import("cmfast").WordsContext;

const BMASK: u32 = 0xffffff;
// post-WRT char constants (untyped so they coerce to i32/u8/u32/usize at use site)
const ESCAPE = 12;
const SPACE = 32;
const LF = 10;
const FIRSTUPPER = 64;
const UPPER = 7;
const APOSTROPHE = 39;
const SQUAREOPEN = 91;
const SQUARECLOSE = 93;
const HTLINK = 31;
const HTML = 30;
const TEXTDATA = 96;
const GREATERTHAN = 78;
const LESSTHAN = 76;
const COLON = 74;
const EQUALS = 77;
const CURLYOPENING = 80;
const CURLYCLOSE = 82;
const VERTICALBAR = 81;
const QUESTION = 79;
const SEMICOLON = 75;
const WIKITABLE = 45;
const WIKIHEADER = 78;
// EngWordTypeFlags (fxcmv1.cpp 2368-2390)
const VERB = 1 << 0;
const NOUN = 1 << 1;
const ADJECTIVE = 1 << 2;
const PLURAL = 1 << 3;
const PRESENT_PARTICIPLE = (1 << 4) | VERB;
const PAST_TENSE = (1 << 5) | VERB;
const ADJECTIVE_SUPERLATIVE = (1 << 5) | ADJECTIVE;
const ADJECTIVE_WITHOUT = (1 << 6) | ADJECTIVE;
const ADJECTIVE_FULL = (1 << 7) | ADJECTIVE;
const ADVERB_OF_MANNER = 1 << 8;
const SUFFIX_F = 1 << 9;
const PREFIX_F = 1 << 10;
const MALE = 1 << 11;
const FEMALE = 1 << 13;
const ARTICLE = 1 << 14;
const CONJUNCTION = 1 << 15;
const ADPOSITION = 1 << 16;
const NUMBER = 1 << 17;
const PREPOSITION = 1 << 18;
const CONJUNCTIVE_ADVERB = 1 << 19;
const PRONOUN = 1 << 20;

// fxcmv1.cpp wrt_2b/wrt_3b/wrt_4b (1639-1695) — the WRT byte-class tables.
pub const WRT_2B = [256]u8{ 2,3,1,3,3,0,1,2,3,3,0,0,1,3,3,3,3,3,3,3,3,3,3,3,3,3,3,0,3,3,3,3,3,2,0,2,1,3,2,1,3,3,3,3,2,3,0,2,1,1,1,1,1,1,1,1,1,1,3,2,2,3,2,2,2,2,0,0,2,3,1,2,1,2,2,2,2,2,0,0,2,2,2,2,2,2,2,2,3,0,2,3,2,0,2,3,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0 };
pub const WRT_3B = [256]u8{ 0,0,2,0,5,6,0,6,0,2,0,4,3,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,2,4,1,4,4,7,4,7,3,7,2,2,3,5,3,1,1,1,1,1,1,1,1,1,1,1,0,5,3,3,5,5,0,5,5,7,5,0,1,5,4,5,0,0,6,0,7,1,3,3,7,4,5,5,7,0,2,2,5,4,4,7,4,6,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,6,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7 };
const WRT_4B = [256]u8{ 6,0,12,15,12,15,14,14,5,3,14,0,15,13,8,13,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,13,5,15,11,10,12,6,12,0,11,14,1,1,10,9,8,7,7,7,7,7,7,7,7,7,7,9,11,6,1,0,4,9,10,10,4,5,1,4,2,11,8,4,1,0,10,10,5,4,7,15,4,5,13,0,1,4,12,0,1,3,3,3,11,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,3,8,0,11,7,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2 };

/// fxcmv1.cpp primes[] (order-hash multipliers), shared with byte_core.
const PRIMES = [14]u32{ 0, 257, 251, 241, 239, 233, 229, 227, 223, 211, 199, 197, 193, 191 };

// PageState (fxcmv1.cpp 3703-3712)
const P_NONE = 0;
const P_TEMPLATE = 1;
const P_TEXT = 2;
const P_TOPIC = 4;
const P_CATEGORY = 8;

// fxcmv1.cpp fcy/fcq (3599-3621): bracket/quote and first-char index maps
const FCY = [128]u8{
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 5, 0, 0, 0, 0, 6, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 4, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
};
const FCQ = [128]u8{
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 6, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3, 0, 4, 5, 0, 0, 2, 7, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0,
    2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
};
/// htcxt element pairs (fxcmv1.cpp 2022): {'&'*256+'L', '&'*256+'N'}
const HTML_ELE = [2]u16{ @as(u16, '&') * 256 + @as(u16, 'L'), @as(u16, '&') * 256 + @as(u16, 'N') };

/// fxcmv1.cpp `hash(U32 a,U32 b,U32 c=0xffffffff)`.
pub inline fn hash3(a: u32, b: u32, c: u32) u32 {
    const h = a *% 110002499 +% (b *% 30005491) +% (c *% 50004239);
    return h ^ (h >> 9) ^ (a >> 3) ^ (b >> 3) ^ (c >> 4);
}

/// fxcmv1.cpp getWT (mixer wt nibble; feeds lastWT).
fn getwt(t: u32) u32 {
    if (t & VERB != 0) return 1;
    if (t & NOUN != 0) return 2;
    if (t & ADJECTIVE != 0) return 3;
    if (t & MALE != 0) return 4;
    if (t & FEMALE != 0) return 5;
    if (t & ARTICLE != 0) return 6;
    if (t & CONJUNCTION != 0) return 7;
    if (t & ADPOSITION != 0) return 8;
    if (t & CONJUNCTIVE_ADVERB != 0) return 9;
    if (t & ADVERB_OF_MANNER != 0) return 11;
    if (t & SUFFIX_F != 0) return 12;
    if (t & PREFIX_F != 0) return 13;
    if (t & PLURAL != 0) return 10;
    if (t & PRONOUN != 0) return 2;
    if (t != 0) return 14;
    return 15;
}

/// fxcmv1.cpp getWT3 (sentence-type ladder; feeds wt3cxt).
pub fn getwt3(t: u32) u32 {
    if (t & VERB != 0) return 1;
    if (t & NOUN != 0) return 2;
    if (t & ADJECTIVE != 0) return 3;
    if (t & PLURAL != 0) return 4;
    if (t & PAST_TENSE != 0) return 5;
    if (t & PRESENT_PARTICIPLE != 0) return 6;
    if (t & ADJECTIVE_SUPERLATIVE != 0) return 7;
    if (t & ADJECTIVE_WITHOUT != 0) return 8;
    if (t & ADJECTIVE_FULL != 0) return 9;
    if (t & ADVERB_OF_MANNER != 0) return 10;
    if (t & SUFFIX_F != 0) return 11;
    if (t & PREFIX_F != 0) return 12;
    if (t & MALE != 0) return 13;
    if (t & FEMALE != 0) return 14;
    if (t & ARTICLE != 0) return 15;
    if (t & CONJUNCTION != 0) return 16;
    if (t & ADPOSITION != 0) return 17;
    if (t & NUMBER != 0) return 18;
    if (t & PREPOSITION != 0) return 19;
    if (t & CONJUNCTIVE_ADVERB != 0) return 20;
    if (t & PRONOUN != 0) return 21;
    return 0;
}

inline fn u8t(x: i32) u8 {
    return @truncate(@as(u32, @bitCast(x)));
}
inline fn csb(c: i32) u8 {
    return u8t(char_swap(c));
}

pub const ParseByte = struct {
    alloc: std.mem.Allocator,
    // raw byte history + registers
    c1: i32,
    c2: i32,
    c3: i32,
    c4: u32,
    blpos: u32,
    buffer: []u8,
    pos: u32,
    // wrt bit-stream states (fxcmv1.cpp 3654-3659)
    n2b: u32,
    n3b: u32,
    n4b: u32,
    o2b: u32,
    o3b: u32,
    stream2b: u32,
    stream2b_r: u32,
    stream3b: u32,
    stream3b_r: u32,
    stream4b: u32,
    stream2b_mask: u32,
    stream3b_mask: u32,
    stream3b_mask1: u32,
    stream3b_r_mask1: u32,
    stream3b_r_mask2: u32,
    // order-hash core
    x4: u32,
    x5: u32,
    t: [16]u32,
    // word core
    word0: u32,
    word00: u32,
    words: u8,
    spaces: u8,
    numbers: u8,
    h: u32,
    number_a: u32,
    number0: u32,
    number1: u32,
    numlen0: i32,
    numlen1: i32,
    mybenum: i32,
    u8w: u32,
    u8w_left: i32, // C++ utf8left
    linkword: u32,
    senword: u32,
    wp: []u32, // wp[0x10000]: last pos of word0&0xffff
    np: []u32, // np[0x10000]: last pos of numberA&0xffff
    // dict codeword machinery (no-dict degenerate: lastCW always decodes to 0)
    dcw: i32,
    dcwl: i32,
    last_cw: u32,
    cw_str: u32, // init 0x10000
    cw_colon: u32, // init 0x10000
    deccode: i32,
    // sentence-type hashes (step C context inputs)
    wt3cxt: u32,
    wt3cxt_w: u32,
    wt3cxt_w1: u32,
    wt4cxt_w: u32,
    wt4cxt_w1: u32,
    // decoded buffer + stemmer
    db: DecodedBuffer,
    stem_words: [4]Word,
    cword: usize,
    pword: usize,
    stem_index: usize,
    // satellites
    worcxt: WordsContext,
    worcxt0: WordsContext, // undecoded
    worcxt1: WordsContext, // paragraph
    worcxt2: WordsContext, // stream (typed words)
    worcxt3: WordsContext, // tag
    sencxt: SentenceContext,
    sencxt_l: SentenceContext, // lists '*'
    sencxt_t: SentenceContext, // table
    sencxt_cl: SentenceContext, // wikilinks
    htcxt: BracketContextW,
    brcxt: BracketContext,
    fccxt: BracketContext,
    qocxt: BracketContext,
    colcxt: ColumnContext,
    // parse state
    is_paragraph: i32,
    fc: i32,
    last_art: bool,
    last_wt: u32,
    s_verb: u32,
    in_br: bool,
    is_http_tag: bool,
    was_tag: bool,
    first_word: u32,
    is_category: bool,
    is_long_top: bool,
    nl: u32,
    nl1: u32,
    col: i32,
    above: i32,
    above1: i32,
    wshift: u32,
    pstate: u32,
    pstate_h: u32,
    brfc_idx: u32,
    fc_idx: u32,
    oldwt1: u32,
    // indirect tables (fxcmv1.cpp 3645-3650)
    t1: []u32, // [0x100]
    t2: []u32, // [0x10000]
    ind3: []u16, // [0x2000000]
    context1_ind3: u32,
    cxtind3: u32,
    indirect_word: u32,
    indirect_byte: u32,
    indirect_br_byte: u32,
    indirect_word0_pos: u32,
    indirect_numberd0_pos: u32,
    is_match: u32, // set by the caller (MatchModel) before parse_byte; 0 standalone
    skip_m1: bool,
    is_text: bool,
    is_nowiki: bool,
    is_pre: bool,
    is_math: bool,
    is_page_started: bool,
    skip_see_external: bool,
    nest_list: bool,
    was_verb: bool,
    was_noun: bool,
    was_verb_h: u32,
    was_noun_h: u32,
    page_parag: u32,
    page_sent: u32,
    last_ptop: u32, // C++ U32 lastPTOP=-1
    /// page-tag reset fired this byte: caller must reset cmC4[1,2,4,7,8,6] + cmC[0].
    page_reset: bool,
    /// with page_reset: (lastPTOP&63)==63 held -> caller must also reset cmC[1].
    page_reset_cm1: bool,
    // dict codeword tag constants (0 in no-dict; loaddict sets the real line indices)
    cw_text: u32,
    cw_nowiki: u32,
    cw_math: u32,
    cw_pre: u32,
    cw_page: u32,
    cw_category: u32,
    cw_user: u32,
    cw_wikipedia: u32,
    cw_image: u32,
    cw_external: u32,
    cw_links: u32,
    cw_see: u32,
    cw_also: u32,
    cw_references: u32,
    cw_bibliography: u32,
    cw_http: u32,
    cw_isbn: u32,
    // dictionary (dosym/loaddict, fxcmv1.cpp 441-542)
    is_dict_loaded: bool,
    dict_words: [][]u8,
    codeword2sym: [256]i32,

    pub fn new(alloc: std.mem.Allocator, brackets: []const u8, quotes: []const u8, fchar: []const u8) !*ParseByte {
        const self = try alloc.create(ParseByte);
        self.alloc = alloc;
        // calloc-parity (C++ alloc): the 16 MB ring and 64 MB ind3 get
        // kernel-zeroed lazy pages instead of an up-front commit.
        self.buffer = try zero_alloc.alloc(alloc, u8, BMASK + 1);
        self.wp = try zero_alloc.alloc(alloc, u32, 0x10000);
        self.np = try zero_alloc.alloc(alloc, u32, 0x10000);
        self.t1 = try zero_alloc.alloc(alloc, u32, 0x100);
        self.t2 = try zero_alloc.alloc(alloc, u32, 0x10000);
        self.ind3 = try zero_alloc.alloc(alloc, u16, 0x2000000);

        self.brcxt = BracketContext.new();
        self.brcxt.init(brackets, 8, false, 256);
        self.qocxt = BracketContext.new();
        self.qocxt.init(quotes, 4, true, 256);
        self.fccxt = BracketContext.new();
        self.fccxt.init(fchar, 20, false, 1 << 8);
        self.colcxt = try ColumnContext.new(alloc);
        self.colcxt.init(31);
        self.htcxt = BracketContextW.new();
        self.htcxt.init(&HTML_ELE, 2, false, 0xfff);
        self.db = DecodedBuffer.new();
        self.worcxt = WordsContext.new();
        self.worcxt0 = WordsContext.new();
        self.worcxt1 = WordsContext.new();
        self.worcxt2 = WordsContext.new();
        self.worcxt3 = WordsContext.new();
        self.sencxt = SentenceContext.new();
        self.sencxt_l = SentenceContext.new();
        self.sencxt_t = SentenceContext.new();
        self.sencxt_cl = SentenceContext.new();
        self.stem_words = .{ Word.new(), Word.new(), Word.new(), Word.new() };

        self.c1 = 0;
        self.c2 = 0;
        self.c3 = 0;
        self.c4 = 0;
        self.blpos = 0;
        self.pos = 0;
        self.n2b = 0;
        self.n3b = 0;
        self.n4b = 0;
        self.o2b = 0;
        self.o3b = 0;
        self.stream2b = 0;
        self.stream2b_r = 0;
        self.stream3b = 0;
        self.stream3b_r = 0;
        self.stream4b = 0;
        self.stream2b_mask = 0;
        self.stream3b_mask = 0;
        self.stream3b_mask1 = 0;
        self.stream3b_r_mask1 = 0;
        self.stream3b_r_mask2 = 0;
        self.x4 = 0;
        self.x5 = 0;
        self.t = [_]u32{0} ** 16;
        self.word0 = 0;
        self.word00 = 0;
        self.words = 0;
        self.spaces = 0;
        self.numbers = 0;
        self.h = 0;
        self.number_a = 0;
        self.number0 = 0;
        self.number1 = 0;
        self.numlen0 = 0;
        self.numlen1 = 0;
        self.mybenum = 0;
        self.u8w = 0;
        self.u8w_left = 0;
        self.linkword = 0;
        self.senword = 0;
        self.dcw = 0;
        self.dcwl = 0;
        self.last_cw = 0;
        self.cw_str = 0x10000;
        self.cw_colon = 0x10000;
        self.deccode = 0;
        self.wt3cxt = 0;
        self.wt3cxt_w = 0;
        self.wt3cxt_w1 = 0;
        self.wt4cxt_w = 0;
        self.wt4cxt_w1 = 0;
        self.cword = 0;
        self.pword = 3;
        self.stem_index = 0;
        self.is_paragraph = 0;
        self.fc = 0;
        self.last_art = false;
        self.last_wt = 0;
        self.s_verb = 0;
        self.in_br = false;
        self.is_http_tag = false;
        self.was_tag = false;
        self.first_word = 0;
        self.is_category = false;
        self.is_long_top = false;
        self.nl = 0;
        self.nl1 = 0;
        self.col = 0;
        self.above = 0;
        self.above1 = 0;
        self.wshift = 0;
        self.pstate = P_NONE;
        self.pstate_h = 0;
        self.brfc_idx = 0;
        self.fc_idx = 0;
        self.oldwt1 = 0;
        self.context1_ind3 = 0;
        self.cxtind3 = 0;
        self.indirect_word = 0;
        self.indirect_byte = 0;
        self.indirect_br_byte = 0;
        self.indirect_word0_pos = 0;
        self.indirect_numberd0_pos = 0;
        self.is_match = 0;
        self.skip_m1 = false;
        self.is_text = false;
        self.is_nowiki = false;
        self.is_pre = false;
        self.is_math = false;
        self.is_page_started = false;
        self.skip_see_external = false;
        self.nest_list = false;
        self.was_verb = false;
        self.was_noun = false;
        self.was_verb_h = 0;
        self.was_noun_h = 0;
        self.page_parag = 0;
        self.page_sent = 0;
        self.last_ptop = 0xffffffff;
        self.page_reset = false;
        self.page_reset_cm1 = false;
        self.cw_text = 0;
        self.cw_nowiki = 0;
        self.cw_math = 0;
        self.cw_pre = 0;
        self.cw_page = 0;
        self.cw_category = 0;
        self.cw_user = 0;
        self.cw_wikipedia = 0;
        self.cw_image = 0;
        self.cw_external = 0;
        self.cw_links = 0;
        self.cw_see = 0;
        self.cw_also = 0;
        self.cw_references = 0;
        self.cw_bibliography = 0;
        self.cw_http = 0;
        self.cw_isbn = 0;
        self.is_dict_loaded = false;
        self.dict_words = &[_][]u8{};
        self.codeword2sym = [_]i32{0} ** 256;
        return self;
    }

    /// Frees the directly-owned heap allocations, including colcxt's internal
    /// GVecs. dict_words (if loaded) is freed too.
    pub fn deinit(self: *ParseByte) void {
        const a = self.alloc;
        self.colcxt.deinit(a);
        zero_alloc.free(a, self.buffer);
        zero_alloc.free(a, self.wp);
        zero_alloc.free(a, self.np);
        zero_alloc.free(a, self.t1);
        zero_alloc.free(a, self.t2);
        zero_alloc.free(a, self.ind3);
        if (self.dict_words.len != 0) {
            for (self.dict_words) |w| a.free(w);
            a.free(self.dict_words);
        }
        a.destroy(self);
    }

    /// dosym/loaddict: newline-separated word list; sets the cw* tag indices
    /// (first match wins, index 0 rejected like the C++ `cwX==0 &&` guards) and the
    /// codeword2sym map (bytes 128..255 -> 0..127).
    pub fn load_dict(self: *ParseByte, data: []const u8) !void {
        var list = std.ArrayListUnmanaged([]u8){};
        errdefer {
            for (list.items) |w| self.alloc.free(w);
            list.deinit(self.alloc);
        }
        var it = std.mem.splitScalar(u8, data, '\n');
        while (it.next()) |w| {
            if (w.len == 0) continue;
            const copy = try self.alloc.dupe(u8, w);
            try list.append(self.alloc, copy);
        }
        self.dict_words = try list.toOwnedSlice(self.alloc);
        for (self.dict_words, 0..) |w, ii| {
            const i: u32 = @intCast(ii);
            if (self.cw_text == 0 and std.mem.eql(u8, w, "text")) {
                self.cw_text = i;
            } else if (self.cw_nowiki == 0 and std.mem.eql(u8, w, "nowiki")) {
                self.cw_nowiki = i;
            } else if (self.cw_math == 0 and std.mem.eql(u8, w, "math")) {
                self.cw_math = i;
            } else if (self.cw_pre == 0 and std.mem.eql(u8, w, "pre")) {
                self.cw_pre = i;
            } else if (self.cw_page == 0 and std.mem.eql(u8, w, "page")) {
                self.cw_page = i;
            } else if (self.cw_image == 0 and std.mem.eql(u8, w, "image")) {
                self.cw_image = i;
            } else if (self.cw_category == 0 and std.mem.eql(u8, w, "category")) {
                self.cw_category = i;
            } else if (self.cw_user == 0 and std.mem.eql(u8, w, "user")) {
                self.cw_user = i;
            } else if (self.cw_wikipedia == 0 and std.mem.eql(u8, w, "wikipedia")) {
                self.cw_wikipedia = i;
            } else if (self.cw_external == 0 and std.mem.eql(u8, w, "external")) {
                self.cw_external = i;
            } else if (self.cw_links == 0 and std.mem.eql(u8, w, "links")) {
                self.cw_links = i;
            } else if (self.cw_see == 0 and std.mem.eql(u8, w, "see")) {
                self.cw_see = i;
            } else if (self.cw_also == 0 and std.mem.eql(u8, w, "also")) {
                self.cw_also = i;
            } else if (self.cw_references == 0 and std.mem.eql(u8, w, "references")) {
                self.cw_references = i;
            } else if (self.cw_bibliography == 0 and std.mem.eql(u8, w, "bibliography")) {
                self.cw_bibliography = i;
            } else if (self.cw_http == 0 and std.mem.eql(u8, w, "http")) {
                self.cw_http = i;
            } else if (self.cw_isbn == 0 and std.mem.eql(u8, w, "isbn")) {
                self.cw_isbn = i;
            }
        }
        for (0..256) |c| self.codeword2sym[c] = 0;
        var chars_used: i32 = 0;
        for (128..256) |c| {
            self.codeword2sym[c] = chars_used;
            chars_used += 1;
        }
        self.is_dict_loaded = true;
    }

    /// fxcmv1.cpp decodeCodeWord (494-515).
    fn decode_code_word(self: *const ParseByte, cw: i32) i32 {
        const D1: i32 = 80;
        const D2: i32 = 32;
        const D12: i32 = D1 * D2;
        var c: usize = @intCast(cw & 255);
        if (self.codeword2sym[c] < D1) {
            return self.codeword2sym[c];
        }
        var i: i32 = D1 * (self.codeword2sym[c] - D1);
        c = @intCast((cw >> 8) & 255);
        if (self.codeword2sym[c] < D1) {
            i += self.codeword2sym[c];
            return i + D1;
        }
        i = (i - D12) * D2;
        i += D1 * (self.codeword2sym[c] - D1);
        c = @intCast((cw >> 16) & 255);
        i += self.codeword2sym[c];
        return i + 80 * 49;
    }

    /// decodeWord(c) (536-541).
    fn decode_word(self: *ParseByte, c: i32) void {
        if (!self.is_dict_loaded) {
            self.last_cw = 0;
            return;
        }
        self.last_cw = @bitCast(self.decode_code_word(c));
        if (self.last_cw >= 44515) {
            self.last_cw = 0;
        }
    }

    inline fn buf(self: *const ParseByte, i: u32) i32 {
        return @as(i32, self.buffer[@intCast((self.pos -% i) & BMASK)]);
    }

    /// Raw byte history (the C++ global buf(i)) for satellite models (XMLModel1 etc).
    pub inline fn raw_buf(self: *const ParseByte, i: u32) u8 {
        return self.buffer[@intCast((self.pos -% i) & BMASK)];
    }

    /// The C++ global bufr(i) — absolute-indexed raw buffer read.
    pub inline fn raw_bufr(self: *const ParseByte, i: i32) u8 {
        return self.buffer[@intCast(@as(u32, @bitCast(i)) & BMASK)];
    }

    /// `(*pWord).Hash` — the last stemmed word's hash (cm context input).
    pub inline fn pword_hash(self: *const ParseByte) u32 {
        return self.stem_words[self.pword].hash;
    }

    pub inline fn fccxt_cxt(self: *const ParseByte) u8 {
        return self.fccxt.cxt;
    }

    /// fxcmv1.cpp updateSen(int i=0) — worcxt-affecting parts.
    fn update_sen(self: *ParseByte, i: i32) void {
        if (self.is_paragraph != 0) {
            self.worcxt.paragraph = true; // read (copied) by sencxt.Update before the Reset
        }
        if (self.colcxt.lastfc(1) == SQUAREOPEN and self.c2 == SQUARECLOSE) {
            self.sencxt.update(&self.worcxt);
            self.sencxt_cl.update(&self.worcxt);
        } else if (self.colcxt.nl_char == WIKITABLE) {
            self.sencxt_t.update(&self.worcxt);
        } else if (self.colcxt.lastfc(1) != '*') {
            self.sencxt.update(&self.worcxt);
        } else if (self.colcxt.lastfc(1) == '*' or self.colcxt.lastfc(1) == '#') {
            self.sencxt_l.update(&self.worcxt);
        }
        self.worcxt.reset();
        self.wt3cxt = 0;
        if (i == 0 and !self.colcxt.is_temp) {
            self.worcxt1.reset();
        }
    }

    /// fxcmv1.cpp procWord(4360-4374).
    fn proc_word(self: *ParseByte) void {
        if (self.dcwl > 0) {
            if (self.dcwl == 2) {
                self.dcw = @divTrunc(self.dcw, 256) +% (self.dcw & 255) *% 256;
            }
            if (self.dcwl == 3) {
                self.dcw = @divTrunc(@divTrunc(self.dcw, 256), 256) +% (self.dcw & 0xff00) +% (self.dcw & 255) *% 256 *% 256;
            }
            const dcw = self.dcw;
            self.decode_word(dcw);
            self.cw_str = self.last_cw;
            self.dcw = 0;
            self.dcwl = 0;
            if (!self.is_dict_loaded) {
                return;
            }
            const w = self.dict_words[@intCast(self.last_cw)];
            for (w) |ch| {
                self.setbuf(@as(i32, ch));
            }
        }
    }

    /// setbufstem(c): build cWord from decoded chars, Stem at word end, feed worcxt.
    fn setbufstem(self: *ParseByte, c: i32) void {
        const cl = self.stem_words[self.cword].length();
        if ((c >= 'a' and c <= 'z') or (c == APOSTROPHE and self.c2 != APOSTROPHE) or (c == '-' and cl > 0)) {
            self.stem_words[self.cword].add(u8t(c));
        } else if (cl > 0 and c == SQUARECLOSE and @as(i32, self.fccxt.cxt) != HTLINK and self.is_paragraph != 0) {
            self.in_br = true;
        } else if (cl > 0) {
            _ = stem(&self.stem_words[self.cword], self.blpos);
            self.stem_index = (self.stem_index + 1) & 3;
            self.pword = self.cword;
            self.cword = self.stem_index;
            self.stem_words[self.cword] = Word.new();
            var p_type = self.stem_words[self.pword].type_;
            const p_hash = self.stem_words[self.pword].hash;
            if (p_type & VERB != 0) {
                self.s_verb = p_hash;
            }
            if (self.last_art) {
                self.stem_words[self.pword].type_ |= NOUN;
            }
            p_type = self.stem_words[self.pword].type_;
            if (p_type == ARTICLE and self.db.buffer1(5) == SPACE and self.db.buffer1(4) == 't' and self.db.buffer1(3) == 'h' and self.db.buffer1(2) == 'e') {
                self.last_art = true;
            } else {
                self.last_art = false;
            }
            const whash = if (self.is_math) self.word0 else p_hash;
            self.last_wt = self.last_wt *% 16 +% getwt(p_type);
            // isHTTAG: worcxt.tpbyte==LESSTHAN && !isHTTAG && qocxt.cxt==0
            if (self.is_http_tag) {} else if (self.worcxt.tpbyteVal() == LESSTHAN and !self.is_http_tag and self.qocxt.cxt == 0) {
                self.is_http_tag = true;
                self.was_tag = true;
            }
            if (self.is_http_tag) {
                self.worcxt3.update(self.word0, u8t(self.c1), 0, whash, self.last_cw, 0);
            }
            const wor_in: i32 = if (@as(i32, self.brcxt.cxt) == SQUAREOPEN or self.c1 == SQUARECLOSE or self.c2 == SQUARECLOSE or self.in_br) 1 else 0;
            self.worcxt.update(self.word0, u8t(self.c1), p_type, whash, self.last_cw, wor_in);
            self.in_br = false;
            // Paragraph (worcxt1): most words, exclude Conjunction etc.
            if ((p_type & (CONJUNCTION | ARTICLE | MALE | FEMALE | NUMBER | CONJUNCTIVE_ADVERB)) == 0 and !self.is_http_tag) {
                self.worcxt1.update(self.word0, u8t(self.c1), p_type, whash, 0, 0);
            }
            // Stream (worcxt2): typed words, exclude Conjunction etc.
            if ((p_type & (CONJUNCTION | ARTICLE | MALE | FEMALE | ADPOSITION | NUMBER | ADVERB_OF_MANNER | CONJUNCTIVE_ADVERB)) == 0 and !self.is_http_tag and p_type != 0) {
                self.worcxt2.update(self.word0, u8t(self.c1), p_type, whash, self.last_cw, 0);
            }
        }
        if (c == LF and self.was_tag) {
            self.was_tag = false;
            self.is_http_tag = false;
        }
        if (c == '>') {
            self.is_http_tag = false;
        }
    }

    inline fn setbuf(self: *ParseByte, c: i32) void {
        self.db.setbuf(u8t(c));
        self.setbufstem(c);
    }

    /// One full parseByte (fxcmv1.cpp 4389-5041) + the 5443 sVerb LF reset.
    pub fn parse_byte(self: *ParseByte, raw: u8) void {
        self.page_reset = false;
        self.page_reset_cm1 = false;
        // --- updatepreamble: c4 packs raw bytes; blpos++ per byte ---
        self.c4 = (self.c4 << 8) | @as(u32, raw);
        const c4 = self.c4;
        self.blpos = self.blpos +% 1;

        // 4393-4399: skipM1 (reads PRE-update satellite state; isMatch set externally)
        self.skip_m1 = self.is_match > 61;
        if (self.colcxt.is_temp or self.colcxt.nl_char == WIKITABLE) {
            self.skip_m1 = self.is_match > 15;
        }
        if (self.colcxt.lastfc(0) == SQUAREOPEN or self.worcxt.wordcount > 1) {
            self.skip_m1 = self.is_match > 15;
        }
        if (self.colcxt.lastfc(0) == '*' or self.colcxt.lastfc(0) == '#') {
            self.skip_m1 = self.is_match > 15;
        }
        if (self.colcxt.lastfc(0) == EQUALS) {
            self.skip_m1 = self.is_match > 5;
        }
        if (self.qocxt.context == APOSTROPHE) {
            self.skip_m1 = self.is_match > 5;
        }
        if (self.is_paragraph != 0 and self.worcxt.wordcount > 3) {
            self.skip_m1 = self.is_match > 13;
        }

        // 4400-4402: byte history
        self.c3 = self.c2;
        self.c2 = self.c1;
        self.c1 = @intCast(c4 & 0xff);
        var c1 = self.c1;
        // 4403-4409: wrt states + serial 2b/4b streams + buffer
        self.n2b = @as(u32, WRT_2B[@intCast(c1)]);
        self.n3b = @as(u32, WRT_3B[@intCast(c1)]);
        self.n4b = @as(u32, WRT_4B[@intCast(c1)]);
        self.stream2b = self.stream2b *% 4 +% self.n2b;
        self.stream4b = self.stream4b *% 16 +% self.n4b;
        self.buffer[@intCast(self.pos & BMASK)] = @intCast(c1);
        self.pos = self.pos +% 1;

        // 4412-4423: 'text' tag end forces an LF reset.
        if (self.c2 == GREATERTHAN and self.is_text) {
            self.is_page_started = true;
            self.is_text = false;
            self.pstate = P_TEXT;
            self.pstate_h = hash3(self.pstate_h, self.pstate, 0);
            if (c1 != LF) {
                self.colcxt.update(LF, 0, self.blpos);
                self.update_sen(0);
                self.fc = 0;
                self.is_paragraph = 0;
                self.first_word = 0;
                self.nl1 = self.nl;
                self.nl = self.pos -% 2;
            }
        }
        // 4424: isLongTOP ("===")
        if ((c4 & 0xffffff) == (((EQUALS * 256) + EQUALS) * 256 + EQUALS)) {
            self.is_long_top = true;
        }
        // 4425-4427
        if (c1 == CURLYCLOSE and self.c2 == VERTICALBAR and self.colcxt.nl_char == WIKITABLE) {
            self.update_sen(0);
        }
        // 4428
        if (self.worcxt.wordcount < 6 and self.colcxt.is_temp and c1 == CURLYCLOSE) {
            self.worcxt.remove_words_l(8, CURLYOPENING, CURLYCLOSE, true);
        }
        // 4430: column context
        self.colcxt.update(c1, c4 & 0xffffff, self.blpos);
        // 4432-4434: bracket context
        if (c1 < 'a' or c1 == SQUAREOPEN) {
            self.brcxt.update(c1);
        }
        if (c1 == SPACE and self.c2 == LESSTHAN) {
            self.brcxt.update(GREATERTHAN);
        }
        // 4437: quote context
        self.qocxt.update(c1);
        // 4439-4442: html-tag context
        if (self.htcxt.cxt != 0 and self.c2 == 'L' and (c1 == SPACE or c1 == '!' or c1 < 128)) {
            self.htcxt.update('&' * 256 + 'N'); // not an html tag
        }
        self.htcxt.update(@intCast(c4 & 0xffff));

        // 4444-4456: end-marker duplicate ('$' ']' '|' ')' '[')
        if (c1 == '$' or c1 == SQUARECLOSE or c1 == VERTICALBAR or c1 == ')' or c1 == SQUAREOPEN) {
            if (c1 != self.c2) {
                var i: usize = 13;
                while (i > 0) : (i -= 1) {
                    self.t[i] = self.t[i - 1] *% PRIMES[i];
                }
            }
            self.x4 = (self.x4 *% 256) +% @as(u32, @bitCast(self.c2));
            self.stream2b = self.stream2b *% 4 +% self.n2b;
            self.stream2b_r = self.stream2b_r *% 4 +% self.n2b;
            self.stream3b_r = self.stream3b_r *% 8 +% self.n3b;
        }
        // 4458-4461: x4 + order-X hashes
        self.x4 = (self.x4 *% 256) +% @as(u32, @intCast(c1));
        {
            var i: usize = 13;
            while (i > 0) : (i -= 1) {
                self.t[i] = self.t[i - 1] *% PRIMES[i] +% @as(u32, @intCast(c1)) +% (@as(u32, @intCast(i)) *% 256);
            }
        }

        // 4463-4466: shift word/space/number bit-histories
        self.words = self.words *% 2;
        self.spaces = self.spaces *% 2;
        self.numbers = self.numbers *% 2;
        const j: u32 = @intCast(c1);

        if ((j -% 'a') <= 25 or (c1 > 127 and self.c2 != ESCAPE)) {
            // ===== word char (4467-4520) =====
            if (self.word0 == 0) {
                if (self.is_math and self.c2 == '/' and self.c3 == LESSTHAN) {
                    self.is_math = false;
                }
                var re_char = self.c2;
                if (self.c2 == FIRSTUPPER or self.c2 == UPPER) {
                    if (self.c3 != APOSTROPHE) {
                        re_char = self.c3;
                    } else if (self.buf(4) != APOSTROPHE) {
                        re_char = self.buf(4);
                    } else if (self.buf(5) != APOSTROPHE) {
                        re_char = self.buf(5);
                    } else if (self.buf(6) != APOSTROPHE) {
                        re_char = self.buf(6);
                    } else {
                        re_char = self.c3;
                    }
                } else if (self.c2 == '/' and self.c3 == LESSTHAN) {
                    re_char = self.c3;
                }
                const g: i32 = if (self.c2 == LESSTHAN) self.c2 else if (self.c3 == LESSTHAN and self.c2 == '/') self.c3 else 0;
                self.worcxt.set(u8t(re_char), if (self.c2 == FIRSTUPPER) @as(i32, 1) else @as(i32, 0), u8t(g));
                self.worcxt0.set(u8t(re_char), if (self.c2 == FIRSTUPPER) @as(i32, 1) else @as(i32, 0), 0);
                self.worcxt1.set(u8t(re_char), 0, 0);
            }
            self.words |= 1;
            self.word0 = self.word0 *% 2104 +% j;
            self.word00 = self.word0;
            self.h = self.word0 *% 271;
            self.u8w = 0;
            // 4490-4491: linkword/senword
            if (@as(i32, self.brcxt.cxt) == SQUAREOPEN and @as(i32, self.fccxt.cxt) != HTLINK and self.fc != HTML) {
                self.linkword = self.linkword *% 2104 +% j;
            }
            if (self.is_paragraph != 0 and @as(i32, self.fccxt.cxt) != HTLINK and !self.colcxt.is_temp) {
                self.senword = self.senword *% 2104 +% j;
            }
            // 4494-4498: quote-context removal
            const word3bit: i32 = @as(i32, self.words & 7);
            if ((word3bit == 5 and self.c2 == APOSTROPHE) or (word3bit == 1 and self.c3 == SQUARECLOSE and self.c2 == APOSTROPHE) or (word3bit == 1 and (self.numbers & 4) != 0 and self.c2 == APOSTROPHE)) {
                const q: i32 = @as(i32, self.qocxt.cxt);
                self.qocxt.update(q);
            }
            // 4500-4516: dict codeword accumulation (no-dict: decodeCodeWord skipped)
            if (c1 > 127 and self.dcwl < 3) {
                self.dcw = self.dcw *% 256 +% c1;
                self.dcwl += 1;
                if (self.blpos > 6) {
                    // partial decode: dcw2 ignores the first byte
                    var dcw2: i32 = 0;
                    if (self.dcwl == 2) {
                        dcw2 = @divTrunc(self.dcw, 256) +% (self.dcw & 255) *% 256;
                    } else if (self.dcwl == 3) {
                        dcw2 = @divTrunc(@divTrunc(self.dcw, 256), 256) +% (self.dcw & 0xff00) +% (self.dcw & 255) *% 256 *% 256;
                    }
                    const i: i32 = if (self.is_dict_loaded) self.decode_code_word(dcw2) else 0;
                    if (i > 0) {
                        self.deccode = i;
                    }
                }
            } else if (self.dcw != 0) {
                self.proc_word();
                if (self.blpos < 448131719) {
                    self.deccode = @bitCast(self.last_cw);
                }
            }
            // 4518-4520: decoded char to buffer
            if (c1 == 10 or c1 == 9 or (c1 > 31 and c1 < 128)) {
                self.setbuf(char_swap(c1));
            }
        } else if (((c1 == ESCAPE and self.c2 != ESCAPE) or (c1 > 127 and self.c2 == ESCAPE)) and (self.words & 4) == 4) {
            // ===== escape branch (4522-4527) =====
            if (c1 != ESCAPE) {
                self.words |= 3;
                self.word0 = self.word0 *% 2104 +% j;
                self.word00 = self.word0;
                self.h = self.word0 *% 271;
            }
        } else {
            // ===== non-word byte (4528-4801) =====
            // 4529-4541: procWord/deccode
            if (self.word0 != 0) {
                self.proc_word();
                if (self.blpos < 448131719) {
                    self.deccode = @bitCast(self.last_cw);
                }
            } else {
                self.deccode = 0x10000 + @as(i32, @intCast(self.stream2b & 0xffff));
                self.last_cw = 0;
            }
            // 4543-4545: decoded char to buffer
            if (c1 == 10 or c1 == 9 or (c1 > 31 and c1 < 128)) {
                self.setbuf(char_swap(c1));
            }
            // 4546-4564: numbers
            if (c1 >= '0' and c1 <= '9') {
                self.numbers = self.numbers +% 1;
                self.np[@intCast(self.number_a & 0xffff)] = self.pos;
                self.number_a = self.number_a *% 2104 +% j;
                if (self.mybenum != 0 and self.numlen1 <= 2) {
                    self.number0 = self.number1;
                    self.number1 = 0;
                    self.numlen0 = self.numlen1;
                    self.numlen1 = 0;
                }
                self.number0 = self.number0 *% 10 +% @as(u32, @intCast(c1 & 0x0f));
                self.numlen0 = @min(self.numlen0 +% 1, 19);
                self.mybenum = 0;
            } else {
                if (self.number_a != 0) {
                    self.worcxt0.update(self.number_a, u8t(c1), NUMBER, self.number_a, 0, 0);
                    const wor: i32 = if (@as(i32, self.brcxt.cxt) == SQUAREOPEN or c1 == SQUARECLOSE) 1 else 0;
                    self.worcxt.update(self.number_a, u8t(c1), NUMBER, self.number_a, 0, wor);
                }
                self.number_a = 0;
                if (self.numlen0 != 0 or (self.numbers & 0xf) == 0) {
                    self.number1 = self.number0;
                    self.numlen1 = self.numlen0;
                    self.number0 = 0;
                    self.numlen0 = 0;
                }
                if (self.numlen1 <= 2 and self.numlen1 != 0 and (self.numbers & 5) == 5 and self.numlen0 == 0 and self.c2 == '.') {
                    self.mybenum = 2;
                } else if (self.numlen1 <= 2 and self.numlen1 != 0 and (self.numbers & 2) != 0 and self.numlen0 == 0 and c1 == '.') {
                    self.mybenum = 1;
                } else if (self.mybenum == 1 and c1 != '.') {
                    self.mybenum = 0;
                }
            }
            // 4566-4572: quote-context removal
            const word3bit: i32 = @as(i32, self.words & 7);
            if ((word3bit == 4 and c1 == SPACE and self.c2 == APOSTROPHE) or (c1 == FIRSTUPPER and (self.numbers & 4) != 0 and self.c2 == APOSTROPHE) or (word3bit == 4 and c1 == FIRSTUPPER and self.c2 == APOSTROPHE) or (word3bit == 4 and (self.numbers & 1) != 0 and self.c2 == APOSTROPHE)) {
                const q: i32 = @as(i32, self.qocxt.cxt);
                self.qocxt.update(q);
            }
            // 4574
            if (self.word00 != 0 and !(@as(i32, self.fccxt.cxt) == SQUAREOPEN)) {
                self.word00 = 0;
            }
            // 4576-4637: word-end block (exact C++ order)
            if (self.word0 != 0) {
                const t1v = self.worcxt.type_at(1);
                if (self.wt3cxt == 0 or (t1v & ARTICLE) == ARTICLE) {
                    // wt3cxt_w1 store dropped: field is written-only (read nowhere;
                    // dead in the reference too). wt3cxt_w=0 is the live effect.
                    self.wt3cxt_w = 0;
                }
                self.wt3cxt = hash3(self.wt3cxt, getwt3(t1v), @as(u32, @bitCast(self.worcxt.wordcount)));
                if (t1v != 0) {
                    self.wt4cxt_w = hash3(self.wt4cxt_w, self.worcxt.code(1), t1v);
                }
                if ((t1v & NOUN) == 0 and t1v != 0) {
                    self.wt3cxt_w = hash3(self.wt3cxt_w, self.worcxt.code(1), t1v);
                }
                if ((self.stem_words[self.pword].type_ & (CONJUNCTIVE_ADVERB | CONJUNCTION)) == 0) {
                    const b0: i32 = if (c1 != LF) c1 else 0;
                    self.worcxt0.update(self.word0, u8t(b0), 0, self.word0, 0, 0);
                }
                if (self.worcxt.type_at(1) == NUMBER) {
                    self.stream3b_r = self.stream3b_r *% 128 +% 1;
                    self.stream3b = self.stream3b *% 128 +% 1;
                }
                if (self.first_word == 0 and @as(i32, self.fccxt.cxt) != SQUAREOPEN) {
                    self.first_word = self.word0;
                }
                if (self.worcxt.type_at(1) & CONJUNCTION != 0) {
                    self.stream3b_r *%= 128;
                    self.stream3b *%= 128;
                    if (self.is_paragraph != 0) {
                        self.senword = 0;
                    }
                }
                if (self.worcxt.type_at(1) & ARTICLE != 0) {
                    self.stream3b_r = self.stream3b_r *% 128 +% 2;
                    self.stream3b = self.stream3b *% 128 +% 2;
                }
                if (self.worcxt.type_at(1) & ADPOSITION != 0 or (self.is_paragraph != 0 and self.worcxt.type_at(1) & PRESENT_PARTICIPLE != 0)) {
                    self.stream2b_r = self.stream2b_r *% 4 +% (self.stream2b_r & 3);
                    self.stream2b = self.stream2b *% 4 +% (self.stream2b & 3);
                }
                if (self.worcxt.type_at(1) & ADVERB_OF_MANNER != 0) {
                    if (self.is_paragraph != 0) {
                        self.worcxt.remove();
                    }
                }
                if (self.worcxt.type_at(1) & NOUN != 0 and self.worcxt.type_at(2) & ARTICLE != 0) {
                    self.stream3b_r = self.stream3b_r *% 64 +% 1;
                    self.stream3b = self.stream3b *% 64 +% 1;
                    const sb = self.worcxt.s_bytes(1);
                    const w = self.worcxt.word(1);
                    const t = self.worcxt.type_at(1);
                    const ca = self.worcxt.capital(1);
                    const co = self.worcxt.code(1);
                    self.worcxt.remove();
                    self.worcxt.remove();
                    self.worcxt.set(@truncate(sb >> 8), @as(i32, ca), 0);
                    self.worcxt.update(w, u8t(c1), t, w, co, 2);
                }
                // reset stream masks after a word
                self.stream3b_r_mask2 = self.stream3b_r_mask1;
                self.stream3b_mask1 = self.stream3b_mask;
                self.stream3b_mask = 0;
                self.stream2b_mask = 0;
                self.stream3b_r_mask1 = 0;
            }

            // 4639-4655: tag detection
            if (self.db.buffer1(6) == csb(LESSTHAN) and self.db.buffer1(5) == 't' and !self.is_text and c1 == SPACE and self.cw_str == self.cw_text) {
                self.is_text = true;
                self.cw_str = 0x10000;
            }
            if (self.db.buffer1(8) == csb(LESSTHAN) and !self.is_nowiki and self.cw_str == self.cw_nowiki) {
                self.is_nowiki = true;
            } else if (self.db.buffer1(9) == '/' and c1 == GREATERTHAN and self.is_nowiki and self.cw_str == self.cw_nowiki) {
                self.is_nowiki = false;
                self.is_pre = false;
                self.cw_str = 0x10000;
            }
            if (self.is_math and ((c1 == SPACE and self.colcxt.lastfc(0) != COLON) or c1 == ',') and self.c2 == GREATERTHAN and self.cw_str == self.cw_math) {
                self.is_math = false;
                self.cw_str = 0x10000;
            }
            if (self.is_math and c1 == '/' and self.c2 == LESSTHAN and self.c3 == GREATERTHAN and self.db.buffer1(4) == 'h') {
                self.is_math = false;
                self.cw_str = 0x10000;
            }
            if (!self.is_nowiki and self.db.buffer1(6) == csb(LESSTHAN) and self.db.buffer1(5) == 'm' and !self.is_math and c1 != '.' and self.db.buffer1(7) != '&' and self.db.buffer1(8) != '&' and self.cw_str == self.cw_math) {
                self.is_math = true;
            } else if (self.db.buffer1(6) == '/' and (c1 == GREATERTHAN or c1 == '&') and self.is_math and self.cw_str == self.cw_math) {
                self.is_math = false;
                self.cw_str = 0x10000;
            }
            if (self.db.buffer1(5) == csb(LESSTHAN) and c1 == GREATERTHAN and self.db.buffer1(4) == 'p' and !self.is_pre and self.cw_str == self.cw_pre) {
                self.is_pre = true;
                self.cw_str = 0x10000;
            } else if (self.db.buffer1(5) == '/' and c1 == GREATERTHAN and self.db.buffer1(4) == 'p' and self.cw_str == self.cw_pre) {
                self.is_pre = false;
                self.cw_str = 0x10000;
            }
            // 4656-4684: wikipedia page tag ends -> big reset.
            if (self.db.buffer1(6) == '/' and c1 == GREATERTHAN and self.db.buffer1(5) == 'p' and self.cw_str == self.cw_page) {
                self.is_pre = false;
                self.is_math = false;
                self.is_nowiki = false;
                self.colcxt.nl_char = LF;
                self.colcxt.reset_cells();
                self.fccxt.reset();
                self.brcxt.reset();
                self.qocxt.reset();
                self.htcxt.reset();
                self.worcxt.reset();
                self.wt3cxt = 0;
                self.worcxt0.reset();
                self.worcxt1.reset();
                self.worcxt2.reset();
                self.skip_see_external = false;
                self.is_category = false;
                self.is_page_started = false;
                self.pstate = P_NONE;
                self.pstate_h = 0;
                self.last_ptop = self.last_ptop *% 2;
                if (self.page_parag < 2 and self.page_sent < 5) {
                    self.last_ptop = self.last_ptop +% 1;
                }
                self.page_reset = true;
                if ((self.last_ptop & 63) == 63) {
                    self.page_reset_cm1 = true;
                    self.last_ptop = self.last_ptop +% 1;
                }
                self.page_parag = 0;
                self.page_sent = 0;
            }

            // 4686-4687: word0 pos + reset
            self.wp[@intCast(self.word0 & 0xffff)] = self.pos;
            self.word0 = 0;
            self.h = 0;
            // 4689-4691
            if (self.linkword != 0 and c1 == COLON) {
                self.linkword = 0;
            }
            if (c1 == '-' and self.c2 == SPACE) {
                self.worcxt1.reset();
                self.s_verb = 0;
            }
            if (c1 == '-' and self.worcxt.wordcount == 1 and self.worcxt.type_at(1) == NUMBER and self.colcxt.nl_char != WIKITABLE) {
                self.worcxt.reset();
                self.wt3cxt = 0;
            }
            // 4693-4791: punctuation chain
            if (c1 == SPACE) {
                self.spaces = self.spaces +% 1;
            } else if (c1 == LF) {
                if (self.wt4cxt_w != 0) {
                    self.wt4cxt_w1 = self.wt4cxt_w;
                    self.wt4cxt_w = 0;
                }
                self.nl1 = self.nl;
                self.nl = self.pos -% 1;
                self.stream3b_r *%= 128;
                self.stream2b |= 0x3fc;
                self.words = 0xfc;
                self.update_sen(0);
                self.stream2b_r *%= 4;
                self.stream4b |= 0xfff0;
                if (self.colcxt.lastfc(1) == FIRSTUPPER) {
                    self.page_parag = self.page_parag +% 1;
                }
                self.page_sent = self.page_sent +% @as(u32, @bitCast(self.is_paragraph));
                self.fc = 0;
                self.is_paragraph = 0;
                self.first_word = 0;
                self.last_wt = 0;
                if (self.c2 == LF) {
                    self.is_nowiki = false;
                }
                if (!(self.colcxt.is_temp or self.colcxt.nl_char == WIKITABLE)) {
                    self.brcxt.reset();
                    self.qocxt.reset();
                }
                self.wt3cxt = 0;
                self.was_verb = false;
                self.was_noun = false;
                self.nest_list = false;
                self.was_verb_h = 0;
                self.was_noun_h = 0;
            } else if (c1 == '.' or c1 == ')' or c1 == QUESTION) {
                self.last_wt = self.last_wt *% 16;
                self.stream3b_r *%= 128;
                self.stream3b *%= 128;
                self.words |= 0xfe;
                self.x5 = (self.x5 *% 256) +% (c4 & 0xff);
                self.stream2b |= 204;
                self.stream4b = ((self.stream4b & 0xffff0) *% 256) +% (self.stream4b & 0xf);
                self.stream2b_r &= 0xffffffc0;
                if (c1 == '.') {
                    if (@as(i32, self.fccxt.cxt) != SQUAREOPEN) {
                        self.wshift = 1;
                    }
                    if (!(@as(i32, self.fccxt.cxt) == SQUAREOPEN or @as(i32, self.fccxt.cxt) == '(' or self.colcxt.nl_char == WIKITABLE or self.colcxt.lastfc(0) == '*')) {
                        // inline sentence end (4731-4739)
                        if (self.is_paragraph != 0) {
                            self.worcxt.paragraph = true;
                        }
                        self.sencxt.update(&self.worcxt);
                        self.was_verb = false;
                        self.was_noun = false;
                        self.was_verb_h = 0;
                        self.was_noun_h = 0;
                        self.worcxt.reset();
                        self.wt3cxt = 0;
                        if (self.wt4cxt_w != 0) {
                            self.wt4cxt_w1 = self.wt4cxt_w;
                            self.wt4cxt_w = 0;
                        }
                    }
                    self.senword = 0;
                }
                if (c1 == ')') {
                    self.senword = 0;
                }
            } else if (c1 == ',') {
                self.was_verb = false;
                self.was_noun = false;
                self.was_verb_h = 0;
                self.was_noun_h = 0;
                if (self.wt4cxt_w != 0) {
                    self.wt4cxt_w1 = self.wt4cxt_w;
                    self.wt4cxt_w = 0;
                }
                self.words |= 0xfc;
                self.senword = 0;
            } else if (c1 == '(') {
                self.senword = 0;
            } else if (c1 == SEMICOLON and self.colcxt.nl_char != WIKITABLE) {
                self.update_sen(1);
            } else if (c1 == COLON) {
                self.stream3b = (self.stream3b & 0xfffffff8) +% 4;
                self.stream2b |= 12;
                self.x5 = (self.x5 *% 256) +% (c4 & 0xff);
                self.senword = 0;
            } else if (c1 == CURLYCLOSE or c1 == CURLYOPENING) {
                self.words |= 0xfc;
                self.stream3b_r &= 0xffffffc0;
                self.x5 = (self.x5 *% 256) +% (c4 & 0xff);
                self.stream3b = (self.stream3b & 0xfffffff8) +% 3;
            } else if (c1 == SQUARECLOSE) {
                self.stream3b = (self.stream3b & 0xfffffff8) +% 3;
                self.linkword = 0;
            } else if (c1 == LESSTHAN or self.c2 == '&') {
                self.words |= 0xfc;
            } else if (c1 == '-' and self.colcxt.lastfc(0) == '*' and @as(i32, self.brcxt.cxt) != SQUAREOPEN and self.is_paragraph == 0) {
                self.is_paragraph = 1;
                self.fc = FIRSTUPPER;
            } else if (c1 == EQUALS) {
                self.stream3b = (self.stream3b & 0xfffffff8) +% 4;
                self.c2 = '.';
                self.words = self.words *% 2;
            }
            // 4793-4799
            if (c1 == '!' and self.c2 == '&') {
                self.c1 = SPACE;
                self.stream2b = (self.stream2b & 0xfffffffc) +% @as(u32, WRT_2B[@as(usize, SPACE)]);
                self.stream3b = (self.stream3b & 0xfffffff8) +% @as(u32, WRT_3B[@as(usize, SPACE)]);
            } else if (self.colcxt.lastfc(0) == '*' and (c1 == ',' or c1 == SPACE) and self.c2 == SQUARECLOSE and self.is_paragraph == 0) {
                self.is_paragraph = 1;
                self.fc = FIRSTUPPER;
            }
            // 4801
            if (!self.nest_list and self.colcxt.lastfc(0) == '*' and (c4 & 0xffff) == 0x2a2a) {
                self.nest_list = true;
            }
        }

        // 4803: x5 (unconditional)
        self.x5 = (self.x5 *% 256) +% (c4 & 0xff);
        // 4805-4813: non-repeating stream switches + masks
        if (self.o2b != self.n2b) {
            self.stream2b_r = self.stream2b_r *% 4 +% self.n2b;
            self.o2b = self.n2b;
        }
        self.stream2b_mask = self.stream2b_mask *% 4 +% 3;
        if (self.o3b != self.n3b) {
            self.stream3b_r = self.stream3b_r *% 8 +% self.n3b;
            self.stream3b_r_mask1 = self.stream3b_r_mask1 *% 8 +% 7;
            self.stream3b_r_mask2 = self.stream3b_r_mask2 *% 8 +% 7;
            self.o3b = self.n3b;
        }
        self.stream3b = self.stream3b *% 8 +% self.n3b;
        self.stream3b_mask = self.stream3b_mask *% 8 +% 7;
        self.stream3b_mask1 = self.stream3b_mask1 *% 8 +% 7;
        // 4815-4820: brcontext capture + BrFcIdx from bracket/quote (fcy)
        const brcontext = self.brcxt.cxt;
        self.brfc_idx = 0;
        if (self.brcxt.context != 0) {
            self.brfc_idx = @as(u32, FCY[@intCast(brcontext)]);
        }
        if (self.brcxt.context == 0 and self.qocxt.context != 0) {
            self.brfc_idx = @as(u32, FCY[@intCast(self.qocxt.context >> 8)]);
        }

        // ===== column/first-char tail (4830-5041) =====
        self.col = self.colcxt.collen(0, 0);
        self.above = @as(i32, self.buffer[@intCast((self.nl1 +% @as(u32, @bitCast(self.col))) & BMASK)]);
        self.above1 = @as(i32, self.buffer[@intCast((self.nl1 +% @as(u32, @bitCast(self.col)) -% 1) & BMASK)]);
        if (self.colcxt.nl_char == WIKIHEADER) {
            self.above = @as(i32, self.colcxt.colb(1, 0, 0));
            self.above1 = @as(i32, self.colcxt.colb(1, 1, 0));
        }
        if (self.colcxt.is_new_line()) {
            if ((@as(i64, self.colcxt.nlpos(0)) + 2 - @as(i64, self.colcxt.nlpos(1))) < 4) {
                self.fccxt.reset();
                self.htcxt.reset();
                self.worcxt0.reset();
                self.is_long_top = false;
            }
            self.fc = @as(i32, self.colcxt.lastfc(0));
            if (self.fc == WIKIHEADER) {
                self.fccxt.reset();
            }
            if (self.fc == FIRSTUPPER) {
                self.is_paragraph = 1;
            } else {
                self.is_paragraph = 0;
            }
            if (self.fc != SQUAREOPEN) {
                self.is_category = false;
            }
            self.fccxt.update(self.fc);
            if (self.fc == EQUALS) {
                self.pstate = P_TOPIC;
            } else if (self.is_paragraph != 0) {
                self.pstate = P_TEXT;
            } else if (self.colcxt.is_temp) {
                self.pstate = P_TEMPLATE;
            }
            self.pstate_h = hash3(self.pstate_h, self.pstate, 0);
        }
        c1 = self.c1; // NB: may have been overridden to SPACE above
        // 4865-4881
        if (self.col > 2 and c1 > FIRSTUPPER and !self.is_math) {
            if ((@as(i32, self.fccxt.cxt) == EQUALS or @as(i32, self.fccxt.cxt) == VERTICALBAR or @as(i32, self.fccxt.cxt) == COLON or @as(i32, self.fccxt.cxt) == FIRSTUPPER or @as(i32, self.fccxt.cxt) == CURLYOPENING) and c1 == CURLYCLOSE) {
                while (@as(i32, self.fccxt.cxt) == EQUALS or @as(i32, self.fccxt.cxt) == VERTICALBAR or @as(i32, self.fccxt.cxt) == COLON or @as(i32, self.fccxt.cxt) == FIRSTUPPER) {
                    self.fccxt.update(LF);
                }
                self.fccxt.update(c1);
            }
            if (@as(i32, self.fccxt.cxt) == VERTICALBAR and (c1 == SQUARECLOSE or c1 == CURLYCLOSE)) {
                while (@as(i32, self.fccxt.cxt) == VERTICALBAR) {
                    self.fccxt.update(LF);
                }
            }
            if ((@as(i32, self.fccxt.cxt) == COLON or @as(i32, self.fccxt.cxt) == HTLINK) and c1 == SQUARECLOSE) {
                while (@as(i32, self.fccxt.cxt) == COLON or @as(i32, self.fccxt.cxt) == HTLINK) {
                    self.fccxt.update(LF);
                }
            }
            if (c1 < 128) {
                self.fccxt.update(c1);
            }
        }
        // 4883: copy word codeword at colon
        if (c1 == COLON and (self.words & 2) == 2) {
            self.cw_colon = self.cw_str;
        }
        // 4884-4889
        if (c1 == SPACE and @as(i32, self.fccxt.cxt) == COLON and self.colcxt.lastfc(0) != COLON and self.colcxt.nl_char != WIKITABLE) {
            if (self.cw_colon != self.cw_image) {
                while (@as(i32, self.fccxt.cxt) == COLON) {
                    self.fccxt.update(LF);
                }
            }
        }
        // 4892-4900: [category:/user:/wikipedia: link
        if (c1 == COLON and (self.cw_colon == self.cw_category or self.cw_colon == self.cw_user or self.cw_colon == self.cw_wikipedia)) {
            if (self.cw_colon == self.cw_category) {
                self.pstate = P_CATEGORY;
                self.is_category = true;
            }
            self.pstate_h = hash3(self.pstate_h, self.pstate, 0);
            self.fccxt.update(LF);
            self.worcxt.remove();
            self.worcxt0.remove();
        }
        // 4902
        if (c1 == SPACE and self.c2 == LESSTHAN) {
            self.fccxt.update(GREATERTHAN);
        }
        // 4904-4907: [word:// -> http link
        if (@as(i32, self.fccxt.cxt) == COLON and self.c2 == '/' and c1 == '/') {
            self.fccxt.update(LF);
            self.fccxt.update(HTLINK);
        }
        // 4909-4919: wiki link in the beginning of line
        if (self.colcxt.lastfc(0) == SQUAREOPEN and c1 == SPACE and self.is_paragraph == 0) {
            if (self.c2 == SQUARECLOSE or self.c3 == SQUARECLOSE) {
                self.fc = FIRSTUPPER;
                self.is_paragraph = 1;
                self.fccxt.reset();
                self.fccxt.update(self.fc);
            }
        }
        // 4921-4928: first char was space, look for another non-space char
        if (self.fc == SPACE and c1 != SPACE) {
            self.fc = @min(c1, TEXTDATA);
            if (self.fc == FIRSTUPPER) {
                self.is_paragraph = 1;
            } else {
                self.is_paragraph = 0;
            }
            self.fccxt.update(self.fc);
        }
        // 4929-4931: BrFcIdx fallback from fccxt + FcIdx
        const fccontext = self.fccxt.cxt;
        if (self.brfc_idx == 0 and self.fccxt.context != 0) {
            self.brfc_idx = @as(u32, FCY[@intCast(fccontext)]);
        }
        self.fc_idx = @as(u32, FCQ[@intCast(fccontext)]);
        // 4934-4936: list
        if (self.fc == '*' and c1 != SPACE) {
            self.fc = @min(c1, TEXTDATA);
        }
        // 4937
        if (self.fc == '&' and c1 == LESSTHAN) {
            self.fc = HTML;
        }
        // 4939
        if (self.c2 == GREATERTHAN and self.fc == LESSTHAN and c1 == APOSTROPHE) {
            self.fc = APOSTROPHE;
        }
        // 4943-4949: bold/italic words end -> paragraph
        if ((self.colcxt.lastfc(0) == APOSTROPHE or (self.fc == APOSTROPHE and self.colcxt.lastfc(0) != '*')) and c1 == SPACE) {
            if (self.c2 == APOSTROPHE or self.c3 == APOSTROPHE) {
                self.fc = FIRSTUPPER;
                self.is_paragraph = 1;
                self.fccxt.reset();
                self.fccxt.update(self.fc);
            }
        }
        // 4951-4953: http link (c4&0xffffff == COLON '/' '/')
        if (self.fc != FIRSTUPPER and (self.c4 & 0xffffff) == 0x4a2f2f) {
            self.fc = HTLINK;
        }
        // 4955-4968: removeWords chains (exact C++ order)
        self.worcxt.remove_words_l(8, '(', ')', true);
        self.worcxt1.remove_words_l(8, '(', ')', true);
        self.worcxt0.remove_words_l(8, SQUAREOPEN, VERTICALBAR, true);
        self.worcxt.remove_words_l(8, SQUAREOPEN, VERTICALBAR, true);
        self.worcxt1.remove_words_l(8, SQUAREOPEN, VERTICALBAR, true);
        self.worcxt.remove_words_l(8, LESSTHAN, COLON, true);
        if (self.colcxt.is_temp) {
            self.worcxt.remove_words_r(10, EQUALS, VERTICALBAR, true);
        }
        self.worcxt0.remove_words_l(8, LESSTHAN, GREATERTHAN, true);
        self.worcxt.remove_words_l(8, LESSTHAN, GREATERTHAN, true);
        self.worcxt1.remove_words_l(8, LESSTHAN, GREATERTHAN, true);
        // 4971-4999: indirect t1/t2/ind3 + word0/number positions
        self.indirect_word = (c4 >> 8) & 0xffff;
        self.t2[@intCast(self.indirect_word)] = (self.t2[@intCast(self.indirect_word)] *% 256) | @as(u32, @intCast(c1));
        self.indirect_word = c4 & 0xffff;
        self.indirect_word |= self.t2[@intCast(self.indirect_word)] *% 65536;
        self.indirect_byte = (c4 >> 8) & 0xff;
        self.t1[@intCast(self.indirect_byte)] = (self.t1[@intCast(self.indirect_byte)] *% 256) | @as(u32, @intCast(c1));
        self.indirect_byte = @as(u32, @intCast(c1)) | (self.t1[@intCast(c1)] *% 256);
        self.t1[@intCast(brcontext)] = (self.t1[@intCast(brcontext)] *% 4) | (self.stream2b & 3);
        self.indirect_br_byte = (self.stream3b & 7) | (self.t1[@intCast(brcontext)] *% 8);
        self.indirect_word0_pos = self.pos -% self.wp[@intCast(self.word0 & 0xffff)];
        if (self.indirect_word0_pos > 255) {
            self.indirect_word0_pos = 256 +% (@as(u32, @intCast(c1)) *% 65536);
        } else {
            self.indirect_word0_pos = self.indirect_word0_pos +% (@as(u32, @intCast(self.buf(self.indirect_word0_pos))) *% 256) +% (@as(u32, @intCast(c1)) *% 65536);
        }
        self.indirect_numberd0_pos = self.pos -% self.np[@intCast(self.number_a & 0xffff)];
        if (self.indirect_numberd0_pos > 1024) {
            self.indirect_numberd0_pos = 0;
        } else {
            self.indirect_numberd0_pos = (@as(u32, @intCast(self.buf(self.indirect_numberd0_pos))) *% 256) +% (@as(u32, @intCast(c1)) *% 65536);
        }
        if (self.indirect_numberd0_pos != 0 and self.number_a != 0) {
            self.indirect_word0_pos = self.indirect_numberd0_pos;
        }
        self.ind3[@intCast(self.context1_ind3)] = @truncate((self.cxtind3 *% 32 +% @as(u32, @intCast(c1))) & 0x1ffffff);
        self.context1_ind3 = (self.context1_ind3 *% 32 +% @as(u32, @intCast(c1))) & 0x1ffffff;
        self.cxtind3 = @as(u32, self.ind3[@intCast(self.context1_ind3)]);
        // 5001-5012: utf8 tracking
        if (self.c2 == 12) {
            if (self.u8w_left == 0) {
                if ((c1 >> 5) == 6) {
                    self.u8w_left = 1;
                    self.u8w = self.u8w *% 191 +% @as(u32, @intCast(c1));
                } else if ((c1 >> 4) == 0xE) {
                    self.u8w_left = 2;
                    self.u8w = self.u8w *% 191 +% @as(u32, @intCast(c1));
                } else if ((c1 >> 3) == 0x1E) {
                    self.u8w_left = 3;
                    self.u8w = self.u8w *% 191 +% @as(u32, @intCast(c1));
                } else {
                    self.u8w_left = 0;
                }
            } else {
                self.u8w_left -= 1;
                if ((c1 >> 6) != 2) {
                    self.u8w_left = 0;
                }
            }
        }
        // 5014-5017
        if (self.is_paragraph != 0) {
            if (!self.was_verb and (self.worcxt.type_at(1) & VERB) == VERB) {
                self.was_verb = true;
                self.was_verb_h = self.worcxt.word(1);
            } else if (!self.was_noun and (self.worcxt.type_at(1) & NOUN) == NOUN) {
                self.was_noun = true;
                self.was_noun_h = self.worcxt.word(1);
            }
        }
        // 5019
        self.h = self.h +% self.number_a +% @as(u32, @intCast(c1));
        // 5020
        self.oldwt1 = getwt3(self.worcxt2.type_at(1));
        // 5022-5036: skipSeeExternal
        if (!self.skip_see_external and self.colcxt.lastfc(0) == EQUALS) {
            if (self.worcxt.wordcount == 2 and c1 == EQUALS) {
                if ((self.worcxt.code(2) == self.cw_external and self.worcxt.code(1) == self.cw_links) or (self.worcxt.code(2) == self.cw_see and self.worcxt.code(1) == self.cw_also)) {
                    self.skip_see_external = true;
                }
            } else if (self.worcxt.wordcount == 1 and c1 == EQUALS) {
                if (self.worcxt.code(1) == self.cw_references or self.worcxt.code(1) == self.cw_bibliography) {
                    self.skip_see_external = true;
                }
            }
        }
        // 5037-5041
        if (!self.is_category and self.colcxt.lastfc(0) == SQUAREOPEN) {
            if (self.last_cw == self.cw_category and c1 == COLON) {
                self.is_category = true;
            }
        }
    }

    /// The wshift/LF worcxt0-rotation block from modelPrediction's bpos==0 section
    /// (fxcmv1.cpp 5436-5444) — the only parse-state mutation after parseByte.
    pub fn apply_wshift_block(self: *ParseByte) void {
        if (self.wshift != 0 or self.c1 == LF) {
            const sb = self.worcxt0.s_bytes(1);
            const w = self.worcxt0.word(1);
            self.worcxt0.remove();
            self.worcxt0.set(@truncate(sb >> 8), 0, 0);
            self.worcxt0.update(w, LF, 0, w, 0, 0);
            self.wshift = 0;
            if (self.c1 == LF) {
                self.s_verb = 0;
            }
        }
    }
};

// ===================== tests =====================

const BRACKETS = [8]u8{ 40, 41, 80, 82, 91, 93, 76, 78 };
const QUOTES = [4]u8{ 39, 39, 34, 34 };
const FCHAR = [20]u8{ 64, 10, 96, 10, 74, 10, 76, 78, 77, 10, 91, 93, 80, 82, 42, 10, 81, 10, 31, 10 };

/// Yields successive non-comment, non-empty lines of a dump.
const LineIter = struct {
    it: std.mem.SplitIterator(u8, .scalar),
    fn init(data: []const u8) LineIter {
        return .{ .it = std.mem.splitScalar(u8, data, '\n') };
    }
    fn next(self: *LineIter) ?[]const u8 {
        while (self.it.next()) |l| {
            if (l.len == 0) continue;
            if (l[0] == '#') continue;
            return l;
        }
        return null;
    }
};

fn parseFields(line: []const u8, out: []i64) usize {
    var it = std.mem.tokenizeAny(u8, line, " \t\r");
    var n: usize = 0;
    while (it.next()) |tok| {
        if (n >= out.len) break;
        out[n] = std.fmt.parseInt(i64, tok, 10) catch 0;
        n += 1;
    }
    return n;
}

test "parse_byte_worcxt_matches" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();

    const stream: []const u8 = @embedFile("goldens/stream.bin");
    const dump: []const u8 = @embedFile("goldens/worcxt_dump_nodict.txt");
    var lines = LineIter.init(dump);

    const pb = try ParseByte.new(a, &BRACKETS, &QUOTES, &FCHAR);
    var cs: u64 = 0;
    var f: [32]i64 = undefined;
    for (stream, 0..) |byte, bi| {
        pb.parse_byte(byte);
        pb.apply_wshift_block();
        // WORCXT_CS fold
        cs = cs *% 1000003 +% @as(u64, @as(u32, @bitCast(pb.worcxt.wordcount)));
        var q: i32 = 1;
        while (q <= 4) : (q += 1) {
            cs = cs *% 1000003 +% @as(u64, pb.worcxt.word(q));
            cs = cs *% 1000003 +% @as(u64, pb.worcxt.type_at(q));
            cs = cs *% 1000003 +% @as(u64, pb.worcxt.code(q));
        }
        cs = cs *% 1000003 +% @as(u64, pb.last_wt);
        cs = cs *% 1000003 +% @as(u64, pb.s_verb);
        // per-byte localization vs dump
        const line = lines.next().?;
        _ = parseFields(line, &f);
        const got = [_]i64{
            @as(i64, pb.worcxt.wordcount),
            @as(i64, pb.worcxt.word(1)),
            @as(i64, pb.worcxt.type_at(1)),
            @as(i64, pb.last_wt),
            @as(i64, pb.s_verb),
            @as(i64, pb.is_paragraph),
            @as(i64, pb.fc),
            @as(i64, pb.fccxt_cxt()),
            if (pb.is_math) @as(i64, 1) else 0,
            if (pb.is_text) @as(i64, 1) else 0,
            if (pb.is_pre) @as(i64, 1) else 0,
        };
        const want = [_]i64{ f[2], f[3], f[4], f[6], f[7], f[9], f[10], f[11], f[8], f[12], f[13] };
        for (got, want, 0..) |g, wv, k| {
            if (g != wv) {
                std.debug.print("WORCXT DIVERGENCE byte {} c1={}: col {} got {} want {}\n", .{ bi, byte, k, g, wv });
                return error.Divergence;
            }
        }
    }
    try std.testing.expectEqual(@as(u64, 0xca6fcbfd442eac3e), cs);
}

test "parse_byte_preamble2_matches_oracle" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();

    const stream: []const u8 = @embedFile("goldens/stream.bin");
    const dump: []const u8 = @embedFile("goldens/pre2_dump_nodict.txt");
    var lines = LineIter.init(dump);

    const pb = try ParseByte.new(a, &BRACKETS, &QUOTES, &FCHAR);
    var cs: u64 = 0;
    var f: [40]i64 = undefined;
    for (stream, 0..) |byte, bi| {
        pb.parse_byte(byte);
        pb.apply_wshift_block();
        const got = [33]u32{
            pb.worcxt0.word0(1), pb.worcxt0.word0(2), pb.worcxt0.word0(3), pb.worcxt0.word0(4),
            pb.worcxt0.word(1),
            pb.worcxt1.word(1), pb.worcxt1.word(2), pb.worcxt1.word(3), pb.worcxt1.word(4),
            pb.worcxt2.word(1), pb.worcxt2.type_at(1),
            pb.worcxt3.word(1), pb.worcxt3.word(2),
            @as(u32, pb.htcxt.cxt), pb.htcxt.context,
            pb.pstate, pb.pstate_h, pb.brfc_idx, pb.fc_idx,
            @as(u32, @bitCast(pb.above)), @as(u32, @bitCast(pb.above1)), pb.oldwt1,
            pb.indirect_byte, pb.indirect_word, pb.indirect_br_byte, pb.indirect_word0_pos,
            pb.cxtind3,
            @as(u32, @bitCast(pb.sencxt.sentence_at(1).wordcount)), pb.sencxt.sentence_at(1).word(1),
            pb.sencxt.total, pb.sencxt_l.total, pb.sencxt_t.total, pb.sencxt_cl.total,
        };
        for (got) |v| {
            cs = cs *% 1000003 +% @as(u64, v);
        }
        const line = lines.next().?;
        _ = parseFields(line, &f);
        for (got, 0..) |v, k| {
            if (@as(i64, v) != f[k + 1]) {
                std.debug.print("PRE2 DIVERGENCE byte {} raw={}: col {} got {} want {}\n", .{ bi, byte, k, v, f[k + 1] });
                return error.Divergence;
            }
        }
    }
    try std.testing.expectEqual(@as(u64, 0x3b1dd0ab214a83f4), cs);
}

test "parse_byte_full_matches_ctxdump" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();

    const stream: []const u8 = @embedFile("goldens/stream.bin");
    const dump: []const u8 = @embedFile("goldens/ctxdump_nodict.txt");
    var lines = LineIter.init(dump);

    const pb = try ParseByte.new(a, &BRACKETS, &QUOTES, &FCHAR);
    var f: [40]i64 = undefined;
    for (stream, 0..) |byte, bi| {
        pb.parse_byte(byte);
        pb.apply_wshift_block();
        const line = lines.next().?;
        _ = parseFields(line, &f);
        const got = [13]i64{
            @as(i64, pb.c1),
            @as(i64, pb.c2),
            @as(i64, pb.c3),
            @as(i64, pb.stream2b),
            @as(i64, pb.stream2b_r),
            @as(i64, pb.stream3b),
            @as(i64, pb.stream3b_r),
            @as(i64, pb.stream4b),
            @as(i64, pb.x4),
            @as(i64, pb.word0),
            @as(i64, pb.words),
            @as(i64, pb.numbers),
            @as(i64, pb.deccode),
        };
        for (got, 0..) |g, k| {
            if (g != f[k + 1]) {
                std.debug.print("CTXDUMP DIVERGENCE byte {} raw={}: col {} got {} want {}\n", .{ bi, byte, k, g, f[k + 1] });
                return error.Divergence;
            }
        }
        var k: usize = 1;
        while (k <= 13) : (k += 1) {
            if (@as(i64, pb.t[k]) != f[13 + k]) {
                std.debug.print("CTXDUMP DIVERGENCE byte {} raw={}: t[{}] got {} want {}\n", .{ bi, byte, k, pb.t[k], f[13 + k] });
                return error.Divergence;
            }
        }
    }
}
