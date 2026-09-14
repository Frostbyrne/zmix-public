//! fxcm_v26 predictor, ported from reference/fxcmv1_v26.cpp.
//!
//! This is the fxcm model's own predictor: the history buffer, all model
//! instances (state maps, context maps, mixers, APMs, run map, text contexts),
//! the match model, PredictorInit (model instantiation), modelPrediction (the
//! per-byte orchestration / parseByte), update1, and the FXCM Model wrapper.
//!
//! COMPLETE and wired: globals, history buffer, match model, PredictorInit,
//! modelPrediction (the ~700-line per-bit parseByte), update1, and the FXCM
//! Model wrapper are all ported and added to the main predictor as an auxiliary
//! text model (predictor.zig AddFXCM). TEXTMODE is OFF (VERSION 16 — the
//! cmix/dictionary path), so #ifdef TEXTMODE branches are dropped.
//!
//! fxcm's file-scope globals are kept module-level (as in the reference) and
//! FULLY reset in PredictorInit at each FXCM construction (zmix builds a fresh
//! FXCM per compress/decompress run, so every mutable global must return to its
//! initial value or encoder/decoder diverge); only one FXCM is ever live.
//! Large context-map memory is capped below the reference's 512 MB-per-map
//! (cmix targets 32 GB); encoder/decoder use identical sizes so round-trips stay
//! exact (parity size in ROADMAP Phase 5).
const std = @import("std");
const tab = @import("tables.zig");
const prim = @import("primitives.zig");
const core = @import("core.zig");
const st = @import("state.zig");
const cmaps = @import("context_maps.zig");
const txt = @import("text_contexts.zig");
const model_mod = @import("../../model.zig");
const Model = model_mod.Model;

const stretch = prim.stretch;
const squash = prim.squash;
const x = &st.x;

// Per-map context-map memory cap (reference uses up to 32*4096*4096 = 512 MB).
const CM_CAP: u32 = 1 << 24;
inline fn cap(m: u32) u32 {
    return @min(m, CM_CAP);
}

// ---- history buffer ----
pub const BMASK: u32 = 0xffffff; // 16 MB
var buffer: []u8 = &.{};
pub var pos: i32 = 0;

pub inline fn buf(i: i32) i32 {
    return buffer[@intCast((pos - i) & @as(i32, @bitCast(BMASK)))];
}
pub inline fn bufr(i: i32) i32 {
    return buffer[@intCast(i & @as(i32, @bitCast(BMASK)))];
}

// ---- model instances ----
var smA: [3]core.StateMap = undefined;
var scmA: [8]cmaps.SmallStationaryContextMap = undefined;
var mxA: [10]core.Mixer1 = undefined;
var cmC: [27]cmaps.ContextMap = undefined;
var apmA: [6]cmaps.APM = undefined;
var rcmA: [1]cmaps.RunContextMap = undefined;
var brcxt: txt.BracketContext = undefined;
var qocxt: txt.BracketContext = undefined;
var fccxt: txt.BracketContext = undefined;
var colcxt: txt.ColumnContext = undefined;
var worcxt: txt.WordsContext = undefined;

// ---- scalar globals ----
var t: [14]u32 = undefined; // ref `t[14]`: order-X context hashes, also read by the match model (t[LEN1/2/3])
var c1: i32 = 0;
var c2: i32 = 0;
var c3: i32 = 0;
// U8 bit-histories in the reference; kept as u32 with explicit &0xff masking on
// updates (they feed wider arithmetic like 4*words / (numbers|words)<<bpos).
var words: u32 = 0;
var spaces: u32 = 0;
var numbers: u32 = 0;
var word0: u32 = 0;
var word1: u32 = 0;
var word2: u32 = 0;
var word3: u32 = 0;
var wshift: u32 = 0;
var w4: u32 = 0;
var w4r: u32 = 0;
var w4br: u32 = 0;
var x4: u32 = 0;
var x5: u32 = 0;
var ismatch: u32 = 0;
var firstWord: u32 = 0;
var number0: u32 = 0;
var number1: u32 = 0;
var numlen0: u32 = 0;
var numlen1: u32 = 0;
var mybenum: u32 = 0;
var fqcxt: u32 = 0;
var AH1: u32 = 0;
var AH2: u32 = 0x765BA55C;
var fails: u32 = 0;
var failz: u32 = 0;
var failcount: u32 = 0;
var nl: i32 = 0;
var col: i32 = 0;
var fc: i32 = 0;
var fc1: i32 = 0;
var nl1: i32 = 0;
var t1: [0x100]u32 = undefined;
var t2: [0x10000]u32 = undefined;
var wp: [0x10000]i32 = undefined;
var oState: u32 = 0;
var wtype: u32 = 0;
var ttype: u32 = 0;
var nState: u32 = 0;
var oStatew4: u32 = 0;
var nStatew4: u32 = 0;
var ord: i32 = 0;
var ord2: i32 = 0;
var w41: u32 = 0;
var w42: u32 = 0;
var wtype1: u32 = 0;
var ttype1: u32 = 0;
var pr: i32 = 2048; // fxcm's own last prediction (12-bit), exported via FXCM.p()
var rate: i32 = 6; // APM update rate, grows with file position

// =================== Match model 2 (from paq8px v208) ===================
const mHashN = 3;
const HashElementForMatchPositions = struct {
    matchPositions: [mHashN]u32 = .{0} ** mHashN,
    fn add(self: *HashElementForMatchPositions, p: u32) void {
        if (mHashN > 1) {
            var k: usize = mHashN - 1;
            while (k >= 1) : (k -= 1) self.matchPositions[k] = self.matchPositions[k - 1];
        }
        self.matchPositions[0] = p;
    }
};

const MINLEN_RM = 3;
const LEN1 = 5;
const LEN2 = 7;
const LEN3 = 9;

// IDR = ±1 indel recovery (VELLUM cmix `-DINDEL_RECOVERY`). At the pre-recovery
// probe site, if the exact realignment fails, also probe ±1 to recover from a
// single-byte insertion/deletion in the text. Confirmed ratio win on enwik
// (~17K full-stream); banked source: submission_record/src_edits/
// fxcmv1 INDEL_RECOVERY variant (hutter). On by default.
const INDEL_RECOVERY = true;

const MatchInfo = struct {
    length: u32 = 0,
    index: u32 = 0,
    lengthBak: u32 = 0,
    indexBak: u32 = 0,
    expectedByte: u8 = 0,
    delta: bool = false,

    fn isInNoMatchMode(self: *const MatchInfo) bool {
        return self.length == 0 and !self.delta and self.lengthBak == 0;
    }
    fn isInPreRecoveryMode(self: *const MatchInfo) bool {
        return self.length == 0 and !self.delta and self.lengthBak != 0;
    }
    fn isInRecoveryMode(self: *const MatchInfo) bool {
        return self.length != 0 and self.lengthBak != 0;
    }
    fn recoveryModePos(self: *const MatchInfo) u32 {
        // ref: U32 length-lengthBak; relies on unsigned wrap (NDEBUG, assert off).
        return self.length -% self.lengthBak;
    }
    fn prio(self: *MatchInfo) u32 {
        return (@as(u32, @intFromBool(self.length != 0)) << 31) |
            (@as(u32, @intFromBool(self.delta)) << 30) |
            ((if (self.delta) (self.lengthBak >> 1) else (self.length >> 1)) << 24) |
            (self.index & 0x00ffffff);
    }
    fn isBetterThan(self: *MatchInfo, other: *MatchInfo) bool {
        return self.prio() > other.prio();
    }

    fn update(self: *MatchInfo) void {
        if (self.length != 0) {
            const expectedBit: i32 = (@as(i32, self.expectedByte) >> @intCast((8 - x.bpos) & 7)) & 1;
            if (x.y != expectedBit) {
                if (self.isInRecoveryMode()) {
                    self.lengthBak = 0;
                    self.indexBak = 0;
                } else {
                    self.lengthBak = self.length;
                    self.indexBak = self.index;
                    self.delta = true;
                }
                self.length = 0;
            }
        }
        if (x.bpos == 0) {
            if (self.isInPreRecoveryMode()) {
                self.indexBak += 1;
                if (self.lengthBak < txt.MAXLEN) self.lengthBak += 1;
                if (bufr(@intCast(self.indexBak)) == c1) { // match continues -> recover
                    self.length = self.lengthBak;
                    self.index = self.indexBak;
                } else if (INDEL_RECOVERY and bufr(@intCast(self.indexBak +% 1)) == c1) {
                    // IDR: deletion in text — reference ran one byte ahead.
                    self.length = self.lengthBak;
                    self.index = self.indexBak +% 1;
                } else if (INDEL_RECOVERY and self.indexBak >= 1 and bufr(@intCast(self.indexBak -% 1)) == c1) {
                    // IDR: insertion in text — reference is one byte behind.
                    self.length = self.lengthBak;
                    self.index = self.indexBak -% 1;
                } else {
                    self.lengthBak = 0;
                    self.indexBak = 0;
                }
            }
            if (self.length != 0) {
                self.index += 1;
                if (self.length < txt.MAXLEN) self.length += 1;
                if (self.isInRecoveryMode() and self.recoveryModePos() >= MINLEN_RM) {
                    self.lengthBak = 0;
                    self.indexBak = 0;
                }
            }
            self.delta = false;
        }
    }

    fn registerMatch(self: *MatchInfo, p: u32, LEN: u32) void {
        self.length = LEN - LEN1 + 1;
        self.index = p;
        self.lengthBak = 0;
        self.indexBak = 0;
        self.expectedByte = 0;
        self.delta = false;
    }
};

const matchN = 4;
var matchCandidates: [matchN]MatchInfo = undefined;
var numberOfActiveCandidates: u32 = 0;
var mhashtable: []HashElementForMatchPositions = &.{};
var mhashtablemask: u32 = 0;
const nST = 3;
var ctx: [nST]u32 = undefined;

fn isMatch(p: u32, MINLEN: i32) bool {
    var length: i32 = 1;
    while (length <= MINLEN) : (length += 1) {
        if (buf(length) != bufr(@as(i32, @intCast(p)) - length)) return false;
    }
    return true;
}

fn addCandidates(matches: *HashElementForMatchPositions, LEN: u32) void {
    var i: usize = 0;
    while (numberOfActiveCandidates < matchN and i < mHashN) : (i += 1) {
        const matchpos = matches.matchPositions[i];
        if (matchpos == 0) break;
        if (isMatch(matchpos, @intCast(LEN))) {
            var isSame = false;
            var j: u32 = 0;
            while (j < numberOfActiveCandidates) : (j += 1) {
                isSame = (matchCandidates[j].index == matchpos);
                if (isSame) break;
            }
            if (!isSame) {
                matchCandidates[numberOfActiveCandidates].registerMatch(matchpos, LEN);
                numberOfActiveCandidates += 1;
            }
        }
    }
}

fn matchModel2update() void {
    var n = @max(numberOfActiveCandidates, 1);
    var i: u32 = 0;
    // ref: `i--` then loop `i++` relies on unsigned wrap (i=0 -> 0xffffffff -> 0).
    while (i < n) : (i +%= 1) {
        matchCandidates[i].update();
        if (numberOfActiveCandidates != 0 and matchCandidates[i].isInNoMatchMode()) {
            numberOfActiveCandidates -= 1;
            if (numberOfActiveCandidates == i) break;
            var k: u32 = i;
            while (k < numberOfActiveCandidates) : (k += 1) matchCandidates[k] = matchCandidates[k + 1];
            i -%= 1;
            n = @max(numberOfActiveCandidates, 1);
        }
    }
    if (x.bpos == 0) {
        var hash: u32 = t[LEN3];
        var matches = &mhashtable[hash & mhashtablemask];
        if (numberOfActiveCandidates < matchN) addCandidates(matches, LEN3);
        matches.add(@intCast(pos));

        hash = t[LEN2];
        matches = &mhashtable[hash & mhashtablemask];
        if (numberOfActiveCandidates < matchN) addCandidates(matches, LEN2);
        matches.add(@intCast(pos));

        hash = t[LEN1];
        matches = &mhashtable[hash & mhashtablemask];
        if (numberOfActiveCandidates < matchN) addCandidates(matches, LEN1);
        matches.add(@intCast(pos));

        var k: u32 = 0;
        while (k < numberOfActiveCandidates) : (k += 1) {
            matchCandidates[k].expectedByte = @intCast(bufr(@intCast(matchCandidates[k].index)));
        }
    }
}

fn matchModel2mix(m: usize) i32 {
    matchModel2update();
    for (0..nST) |i| ctx[i] = 0;
    var bestCandidateIdx: usize = 0;
    var i: u32 = 1;
    while (i < numberOfActiveCandidates) : (i += 1) {
        if (matchCandidates[i].isBetterThan(&matchCandidates[bestCandidateIdx])) bestCandidateIdx = i;
    }
    const length = matchCandidates[bestCandidateIdx].length;
    const expectedByte = matchCandidates[bestCandidateIdx].expectedByte;
    const isInDeltaMode = matchCandidates[bestCandidateIdx].delta;
    const expectedBit: i32 = if (length != 0) (@as(i32, expectedByte) >> @intCast(7 - x.bpos)) & 1 else 0;

    var denselength: u32 = 0;
    if (length != 0) {
        if (length <= 16) denselength = length - 1 else denselength = 12 + (length >> 2);
        ctx[0] = (denselength << 4) | (@as(u32, @intCast(expectedBit)) << 3) | @as(u32, @intCast(x.bpos));
        ctx[1] = (@as(u32, expectedByte) << 11) | (@as(u32, @intCast(x.bpos)) << 8) | @as(u32, @intCast(c1));
        const sign: i32 = 2 * expectedBit - 1;
        x.mxInputs[m].add(sign * @as(i32, @intCast(length << 5)));
    } else {
        x.mxInputs[m].add(0);
    }
    if (isInDeltaMode) ctx[2] = (@as(u32, expectedByte) << 8) | @as(u32, @intCast(x.c0));

    for (0..nST) |k| {
        const c = ctx[k];
        if (c != 0) {
            smA[k].set(c, x.y);
            const p1 = smA[k].pr;
            const s = stretch(p1);
            x.mxInputs[m].add(s >> 2);
            x.mxInputs[m].add((p1 - 2048) >> 3);
        } else {
            x.mxInputs[m].add(0);
            x.mxInputs[m].add(0);
        }
    }
    return @intCast(length);
}

// =================== PredictorInit ===================
pub fn predictorInit(a: std.mem.Allocator) void {
    st.reset(a, 416, 16); // mxInputs sizes: 410->416, 8->16 (rounded to mult of 16)
    // Always allocate from the current arena: these module-level buffers must
    // not outlive the arena they came from (compress's arena is freed before
    // decompress's FXCM is built), so never reuse a stale pointer.
    buffer = a.alloc(u8, BMASK + 1) catch unreachable;
    @memset(buffer, 0);
    pos = 0;
    nState = 0xffffffff;
    nStatew4 = 0xffffffff;
    prim.init();

    smA[0].init(a, 1 << 9, 1023, &tab.STA1);
    smA[1].init(a, 1 << 19, 1023, &tab.STA1);
    smA[2].init(a, 1 << 16, 1023, &tab.STA1);
    for (0..8) |i| scmA[i].init(a, 8, 8);

    mxA[1].init(a, 2 * 256, @intCast(txt.m_s[1]), @intCast(txt.m_e[1]), @intCast(txt.m_m[1]));
    mxA[2].init(a, 6 * 256, @intCast(txt.m_s[2]), @intCast(txt.m_e[2]), @intCast(txt.m_m[2]));
    mxA[3].init(a, 6 * 256, @intCast(txt.m_s[3]), @intCast(txt.m_e[3]), @intCast(txt.m_m[3]));
    mxA[4].init(a, 8 * 256, @intCast(txt.m_s[4]), @intCast(txt.m_e[4]), @intCast(txt.m_m[4]));
    mxA[5].init(a, 6 * 256, @intCast(txt.m_s[5]), @intCast(txt.m_e[5]), @intCast(txt.m_m[5]));
    mxA[6].init(a, 7 * 256 * 4, @intCast(txt.m_s[6]), @intCast(txt.m_e[6]), @intCast(txt.m_m[6]));
    mxA[7].init(a, 8 * 256, @intCast(txt.m_s[7]), @intCast(txt.m_e[7]), @intCast(txt.m_m[7]));
    mxA[8].init(a, 8 * 256, @intCast(txt.m_s[8]), @intCast(txt.m_e[8]), @intCast(txt.m_m[8]));
    mxA[9].init(a, 8 * 7 * 2, @intCast(txt.m_s[9]), @intCast(txt.m_e[9]), @intCast(txt.m_m[9]));

    apmA[0].init(a, 256);
    apmA[1].init(a, 0x8000 * 2);
    apmA[2].init(a, 0x8000 * 2);
    apmA[3].init(a, 0x20000 * 2);
    apmA[4].init(a, 0x10000 * 2);
    apmA[5].init(a, 0x10000 * 2);
    rcmA[0].init(a, cap(1 * 4096 * 4096), 6);

    // mixers bind to mxInputs[0] (layers 1..8) and mxInputs[1] (final layer 9)
    for (1..9) |i| mxA[i].setTxWx(x.mxInputs[0].n.len, x.mxInputs[0].n.ptr);
    mxA[9].setTxWx(x.mxInputs[1].n.len, x.mxInputs[1].n.ptr);

    const staTabs = [_]*const [1024]u8{ &tab.STA1, &tab.STA2, &tab.STA3, &tab.STA4, &tab.STA5 };
    // cmC config: (mem, order|(c_r<<8)|(c_s<<16)|(c_s2<<24), c_s3, STAtable, c_s4, kep, skip2)
    const CmCfg = struct { mem: u32, order: u32, sta: usize, kep: i32, skip2: i32 };
    const cfg = [27]CmCfg{
        .{ .mem = 32 * 4096 * 4096, .order = 3, .sta = 4, .kep = 0xf0, .skip2 = 1 },
        .{ .mem = 32 * 4096 * 4096, .order = 1, .sta = 0, .kep = 0xf0, .skip2 = 1 },
        .{ .mem = 32 * 4096 * 4096, .order = 1, .sta = 0, .kep = 0xf0, .skip2 = 1 },
        .{ .mem = 32 * 4096 * 4096, .order = 1, .sta = 0, .kep = 0xf0, .skip2 = 1 },
        .{ .mem = 32 * 4096 * 4096, .order = 2, .sta = 0, .kep = 0xf0, .skip2 = 1 },
        .{ .mem = 32 * 4096 * 4096, .order = 1, .sta = 2, .kep = 0xf0, .skip2 = 1 },
        .{ .mem = 1 * 4096 * 4096, .order = 1, .sta = 0, .kep = 0xf0, .skip2 = 1 },
        .{ .mem = 2 * 4096 * 4096, .order = 1, .sta = 4, .kep = 0xf0, .skip2 = 1 },
        .{ .mem = 32 * 4096 * 4096, .order = 3, .sta = 3, .kep = 0, .skip2 = 1 },
        .{ .mem = 32 * 4096, .order = 2, .sta = 0, .kep = 0xf0, .skip2 = 0 },
        .{ .mem = 32 * 4096, .order = 3, .sta = 1, .kep = 0, .skip2 = 1 },
        .{ .mem = 32 * 4096, .order = 4, .sta = 1, .kep = 0, .skip2 = 1 },
        .{ .mem = 16 * 4096, .order = 5, .sta = 1, .kep = 0, .skip2 = 1 },
        .{ .mem = 16 * 4096, .order = 8, .sta = 1, .kep = 0, .skip2 = 1 },
        .{ .mem = 64 * 2 * 4096, .order = 3, .sta = 4, .kep = 0xf0, .skip2 = 0 },
        .{ .mem = 2 * 4096, .order = 2, .sta = 1, .kep = 0xf0, .skip2 = 0 },
        .{ .mem = 128 * 4096, .order = 2, .sta = 0, .kep = 0, .skip2 = 0 },
        .{ .mem = 4 * 4096 * 4096, .order = 3, .sta = 0, .kep = 0xf0, .skip2 = 1 },
        .{ .mem = 32 * 4096 * 4096, .order = 6, .sta = 4, .kep = 0xf0, .skip2 = 1 },
        .{ .mem = 32 * 4096 * 4096, .order = 5, .sta = 4, .kep = 0xf0, .skip2 = 1 },
        .{ .mem = 32 * 4096 * 4096, .order = 2, .sta = 0, .kep = 0xf0, .skip2 = 1 },
        .{ .mem = 32 * 4096 * 4096, .order = 2, .sta = 2, .kep = 0xf0, .skip2 = 1 },
        .{ .mem = 32 * 4096, .order = 2, .sta = 1, .kep = 0x00, .skip2 = 1 },
        .{ .mem = 16 * 4096 * 4096 / 2, .order = 1, .sta = 2, .kep = 0xf0, .skip2 = 1 },
        .{ .mem = 8 * 64 * 4096, .order = 1, .sta = 0, .kep = 0, .skip2 = 0 },
        .{ .mem = 512 * 4096, .order = 1, .sta = 0, .kep = 0xf0, .skip2 = 1 },
        .{ .mem = 512 * 4096, .order = 1, .sta = 0, .kep = 0xf0, .skip2 = 1 },
    };
    for (0..27) |i| {
        const cc = cfg[i];
        const packed_c: i32 = @bitCast(cc.order | (txt.c_r[i] << 8) | (txt.c_s[i] << 16) | (txt.c_s2[i] << 24));
        cmC[i].init(a, cap(cc.mem), packed_c, @intCast(txt.c_s3[i]), staTabs[cc.sta], @intCast(txt.c_s4[i]), cc.kep, cc.skip2);
    }

    brcxt.init(a, &txt.brackets, false, 255);
    qocxt.init(a, &txt.quotes, true, 255);
    fccxt.init(a, &txt.fchar, false, 255);
    colcxt.init(a, 31);
    worcxt.init(a);

    // reset match model + remaining globals
    for (&matchCandidates) |*mc| mc.* = .{};
    numberOfActiveCandidates = 0;
    // ref Predictor ctor: 0x200000 entries, mask 0x1FFFFF (line 3023-3024).
    // Alloc fresh from the current arena (see buffer note above).
    mhashtablemask = 0x200000 - 1;
    mhashtable = a.alloc(HashElementForMatchPositions, 0x200000) catch unreachable;
    @memset(mhashtable, .{});
    @memset(&t, 0);
    @memset(&t1, 0);
    @memset(&t2, 0);
    @memset(&wp, 0);
    // Full scalar reset. The reference relies on file-scope zero-init done once
    // at process start; zmix reconstructs FXCM per run (compress then decompress,
    // and per test), so EVERY mutable global must return to its start value or
    // encoder/decoder diverge and the round-trip breaks.
    c1 = 0;
    c2 = 0;
    c3 = 0;
    words = 0;
    spaces = 0;
    numbers = 0;
    word0 = 0;
    word1 = 0;
    word2 = 0;
    word3 = 0;
    wshift = 0;
    w4 = 0;
    w4r = 0;
    w4br = 0;
    x4 = 0;
    x5 = 0;
    ismatch = 0;
    firstWord = 0;
    number0 = 0;
    number1 = 0;
    numlen0 = 0;
    numlen1 = 0;
    mybenum = 0;
    fqcxt = 0;
    AH1 = 0;
    AH2 = 0x765BA55C;
    fails = 0;
    failz = 0;
    failcount = 0;
    nl = 0;
    col = 0;
    fc = 0;
    fc1 = 0;
    nl1 = 0;
    oState = 0;
    wtype = 0;
    ttype = 0;
    oStatew4 = 0;
    ord = 0;
    ord2 = 0;
    w41 = 0;
    w42 = 0;
    wtype1 = 0;
    ttype1 = 0;
    pr = 2048; // ref PredictorInit sets pr=2048
    rate = 6;
    prim.init();
}

// =================== modelPrediction (ref 2217-2917) ===================
// TEXTMODE is OFF (VERSION 16, the cmix/dictionary path): all #ifdef TEXTMODE
// branches are dropped, only #else / #ifndef TEXTMODE branches are kept.
// All U32 arithmetic wraps mod 2^32 — Zig `<<`/`>>` truncate to width, and
// `*%`/`+%`/`-%` give wrapping mul/add/sub to match the reference exactly.
fn u(v: i32) u32 {
    return @bitCast(v);
}
fn ci(v: anytype) i32 {
    return @intCast(v);
}
fn cu(v: anytype) u32 {
    return @intCast(v);
}

fn modelPrediction(c0: i32, bpos: i32, c4_in: u32) i32 {
    var c4 = c4_in;
    const bm: i32 = @intCast(BMASK);
    var i: i32 = undefined;
    var c: i32 = undefined;
    var h: u32 = undefined;
    var j: u32 = undefined;

    if (bpos == 0) {
        wshift = 0;
        c3 = c2;
        c2 = c1;
        c1 = @intCast(c4 & 0xff);

        i = tab.WRT_W[@intCast(c1)]; // TEXTMODE off: no charSwap
        nStatew4 = @intCast(i);
        w4 = w4 *% 4 +% u(i);
        buffer[@intCast(pos & bm)] = @intCast(c1);
        pos += 1;

        // Column content update (non-TEXTMODE)
        if (colcxt.lastfc(0) == txt.GREATERTHAN) colcxt.nlChar = txt.GREATERTHAN;
        if (colcxt.lastfc(0) == txt.SQUAREOPEN and colcxt.nlChar == txt.GREATERTHAN) colcxt.nlChar = 10;
        colcxt.update(c1, c4 & 0xffffff);

        // Bracket content update (non-TEXTMODE): advance only if not a letter
        if (c1 < 'a') brcxt.update(c1);
        cmC[25].set((@as(u32, brcxt.context) << 8) +% u(c1));
        qocxt.update(c1);

        // order-X context end marker
        if (c1 != txt.SPACE) {
            if (c1 == '$' or c1 == txt.SQUARECLOSE or c1 == txt.VERTICALBAR or c1 == ')' or c1 == txt.SQUAREOPEN) {
                if (c1 != c2) {
                    i = 13;
                    while (i > 0) : (i -= 1) t[@intCast(i)] = t[@intCast(i - 1)] *% txt.primes[@intCast(i)];
                }
                x4 = (x4 << 8) +% u(c2);
            }
        }
        x4 = (x4 << 8) +% u(c1);
        i = 13;
        while (i > 0) : (i -= 1) t[@intCast(i)] = t[@intCast(i - 1)] *% txt.primes[@intCast(i)] +% u(c1) +% u(i) *% 256;

        i = 3;
        while (i < 6) : (i += 1) cmC[0].set(t[@intCast(i)]);
        cmC[1].set(t[6]);
        cmC[2].set(t[8]);
        cmC[3].set(t[13]);

        nState = tab.WRT_T[@intCast(c1)];
        words = (words << 1) & 0xff;
        spaces = (spaces << 1) & 0xff;
        numbers = (numbers << 1) & 0xff;
        j = @intCast(c1);

        if ((j -% 'a') <= ('z' - 'a') or (c1 > 127 and c2 != 12)) {
            // letter
            if (word0 == 0) {
                if (c2 == txt.ATSIGN or c2 == 7) worcxt.set(@intCast(c3)) else worcxt.set(@intCast(c2));
            }
            words = words | 1;
            word0 = word0 *% 2104 +% j; // 263*8
            const word3bit = words & 7;
            if (((word3bit == 5) and (c2 == txt.APOSTROPHE)) or
                ((word3bit == 1) and (c3 == txt.SQUARECLOSE) and (c2 == txt.APOSTROPHE)) or
                ((word3bit == 1) and (numbers & 4) != 0 and (c2 == txt.APOSTROPHE)))
                qocxt.update(qocxt.cxt);
        } else {
            const word3bit = words & 7;
            if (((word3bit == 4) and (c1 == txt.SPACE) and (c2 == txt.APOSTROPHE)) or
                ((c1 == txt.ATSIGN) and (numbers & 4) != 0 and (c2 == txt.APOSTROPHE)) or
                ((word3bit == 4) and (c1 == txt.ATSIGN) and (c2 == txt.APOSTROPHE)))
                qocxt.update(qocxt.cxt);
            if (word0 != 0) {
                word3 = word2 *% 47;
                word2 = word1 *% 53;
                word1 = word0 *% 83;
                worcxt.update(word0, @intCast(c1));
                if (firstWord == 0) firstWord = word0;
                w42 = w41;
                w41 = 0;
                wtype1 = 0;
                ttype1 = 0;
            }
            wp[word0 & 0xffff] = pos;
            word0 = 0;
            if (c1 >= '0' and c1 <= '9') {
                numbers = (numbers + 1) & 0xff;
                if ((numbers & 4) != 0 and c2 == ',') {
                    number0 = number1;
                    number1 = 0;
                    numlen0 = numlen1;
                    numlen1 = 0;
                }
                if (mybenum != 0 and numlen1 <= 2) {
                    number0 = number1;
                    number1 = 0;
                    numlen0 = numlen1;
                    numlen1 = 0;
                }
                number0 = number0 *% 10 +% u(c1 & 0x0f);
                numlen0 = @min(19, numlen0 + 1);
                mybenum = 0;
            } else {
                if (numlen0 != 0 or (numbers & 0xf) == 0) {
                    number1 = number0;
                    numlen1 = numlen0;
                    number0 = 0;
                    numlen0 = 0;
                }
                if (numlen1 <= 2 and numlen1 != 0 and (numbers & 5) == 5 and numlen0 == 0 and c2 == '.') mybenum = 2 else if (numlen1 <= 2 and numlen1 != 0 and (numbers & 2) != 0 and numlen0 == 0 and c1 == '.') mybenum = 1 else if (mybenum == 1 and c1 != '.') mybenum = 0;
            }

            if (c1 == txt.SPACE) {
                spaces = (spaces + 1) & 0xff;
            } else if (c1 == 10) {
                fc = 0;
                fc1 = 0;
                firstWord = 0;
                nl1 = nl;
                nl = pos - 1;
                wtype = wtype << 7;
                w4 = w4 | 0x3fc;
                words = 0xfc;
                worcxt.reset();
                w4r = w4r << 2;
            } else if (c1 == '.' or c1 == ')' or c1 == txt.QUESTION) {
                wtype = wtype << 7;
                ttype = ttype << 7;
                words = words | 0xfe;
                x5 = (x5 << 8) +% (c4 & 0xff);
                w4 = w4 | 204;
                w4r = w4r & 0xffffffc0;
            } else if (c1 == ',') {
                words = words | 0xfc;
            } else if (c1 == txt.COLON) {
                ttype = (ttype & 0xfffffff8) + 4;
                w4 = w4 | 12;
                x5 = (x5 << 8) +% (c4 & 0xff);
            } else if (c1 == txt.CURLYCLOSE or c1 == txt.CURLYOPENING) {
                words = words | 0xfc;
                wtype = wtype & 0xffffffc0;
                x5 = (x5 << 8) +% (c4 & 0xff);
                ttype = (ttype & 0xfffffff8) + 3;
            } else if (c1 == txt.SQUARECLOSE) {
                ttype = (ttype & 0xfffffff8) + 3;
            } else if (c1 == txt.LESSTHAN or c2 == '&') {
                words = words | 0xfc;
            } else if (c1 == txt.EQUALS) {
                ttype = (ttype & 0xfffffff8) + 4;
                c2 = '.';
            } else if (c1 == txt.SEMICOLON) {
                worcxt.reset();
            }
            if (c1 == '!' and c2 == '&') { // '&nbsp;' -> '&!' -> ' '
                c1 = txt.SPACE;
                c4 = (c4 & 0xffffff00) + txt.SPACE;
                w4 = (w4 & 0xfffffffc) + tab.WRT_W[txt.SPACE];
                ttype = (ttype & 0xfffffff8) + tab.WRT_T[txt.SPACE];
            }
            if (c1 == '.') {
                wshift = 1;
                worcxt.reset();
            }
        }

        x5 = (x5 << 8) +% (c4 & 0xff);
        if (oStatew4 != nStatew4) {
            w4r = (w4r << 2) +% nStatew4;
            oStatew4 = nStatew4;
        }
        if (oState != nState) {
            wtype = (wtype << 3) +% nState;
            w41 = (w41 << 3) +% 7;
            w42 = (w42 << 3) +% 7;
            oState = nState;
        }
        wtype1 = (wtype1 << 2) +% 3;
        ttype = (ttype << 3) +% nState;
        ttype1 = (ttype1 << 3) +% 7;
        const brcontext: u32 = brcxt.cxt;

        rcmA[0].set(word3 *% 53 +% u(c1) +% 193 *% (ttype & 0x7fff), @intCast(c1));
        w4br = 0;
        if (brcxt.context != 0) w4br = tab.FCY[brcontext];
        if (brcxt.context == 0 and qocxt.context != 0) w4br = tab.FCY[qocxt.context >> 8];
        col = colcxt.collen(0, 0);
        var above: i32 = buffer[@intCast((nl1 + col) & bm)];
        var above1: i32 = buffer[@intCast((nl1 + col - 1) & bm)];
        if (colcxt.nlChar == txt.GREATERTHAN) {
            above = colcxt.colb(1, 0, 0);
            above1 = colcxt.colb(1, 1, 0);
        }
        if (colcxt.isNewLine()) {
            if ((colcxt.nlpos(0) +% 2 -% colcxt.nlpos(1)) < 4) {
                fccxt.reset();
                brcxt.reset();
                qocxt.reset();
            }
            fc = colcxt.lastfc(0);
            if (fc == txt.GREATERTHAN) fccxt.reset();
            if (fc == txt.ATSIGN) fc1 = 1 else fc1 = 0;
            fccxt.update(fc);
        }

        if (col > 2 and c1 > txt.ATSIGN) {
            if (fccxt.cxt == txt.VERTICALBAR and (c1 == txt.SQUARECLOSE or c1 == txt.CURLYCLOSE)) {
                while (fccxt.cxt == txt.VERTICALBAR) fccxt.update(10);
            }
            if ((fccxt.cxt == txt.COLON or fccxt.cxt == 31) and (c1 == txt.SPACE or c1 == txt.SQUARECLOSE)) {
                while (fccxt.cxt == txt.COLON or fccxt.cxt == 31) fccxt.update(10);
            }
            if (c1 < 128) fccxt.update(c1);
        }
        if (fccxt.cxt == txt.COLON and c2 == '/' and c1 == '/') {
            fccxt.update(10);
            fccxt.update(31);
        }
        if (colcxt.lastfc(0) == txt.SQUAREOPEN and c1 == txt.SPACE) {
            if (c2 == txt.SQUARECLOSE or c3 == txt.SQUARECLOSE) {
                fc = txt.ATSIGN;
                fc1 = 1;
                fccxt.reset();
                fccxt.update(fc);
            }
        }
        if (fc == txt.SPACE and c1 != txt.SPACE) {
            fc = @min(c1, 96);
            if (fc == txt.ATSIGN) fc1 = 1 else fc1 = 0;
            fccxt.update(fc);
        }
        const fccontext: u32 = fccxt.cxt;
        if (w4br == 0 and fccxt.context != 0) w4br = tab.FCY[fccontext];
        fqcxt = tab.FCQ[fccontext];

        cmC[26].set((fccxt.context & 0xff00) +% u(c1) +% (w4 & 12) *% 256 +% ((brcontext +% u(brcxt.last())) << 24));
        if (fc == '*' and c1 != txt.SPACE) fc = @min(c1, 96);
        if (fc == '&' and c1 == txt.LESSTHAN) fc = 30;
        if (colcxt.lastfc(0) == txt.APOSTROPHE and c1 == txt.SPACE) {
            if (c2 == txt.APOSTROPHE or c3 == txt.APOSTROPHE) {
                fc = txt.ATSIGN;
                fc1 = 1;
                fccxt.reset();
                fccxt.update(fc);
            }
        }
        if (fc != txt.ATSIGN and (c4 & 0xffffff) == 0x4a2f2f) fc = 31; // http link

        // Words surrounded by
        if ((worcxt.sBytes(1) & 0xff) == ')') {
            var isC = false;
            var k: i32 = 1;
            while (k < 8) : (k += 1) if ((worcxt.sBytes(k) >> 8) == '(') {
                isC = true;
            };
            if (isC) {
                while ((worcxt.sBytes(1) >> 8) != '(') worcxt.remove();
                worcxt.remove();
            }
        }
        // Words surrounded by [|
        if ((worcxt.sBytes(1) & 0xff) == txt.VERTICALBAR) {
            var isC = false;
            var k: i32 = 1;
            while (k < 8) : (k += 1) if ((worcxt.sBytes(k) >> 8) == txt.SQUAREOPEN) {
                isC = true;
            };
            if (isC) {
                while ((worcxt.sBytes(1) >> 8) != txt.SQUAREOPEN) worcxt.remove();
                worcxt.remove();
            }
        }
        // Template
        if (fccontext == txt.VERTICALBAR and colcxt.isTemp == true) {
            var isC = false;
            var k: i32 = 1;
            while (k < 10) : (k += 1) if ((worcxt.sBytes(k) & 0xff) == txt.EQUALS) {
                isC = true;
            };
            if (isC) {
                while ((worcxt.sBytes(1) & 0xff) != txt.EQUALS) worcxt.remove();
                worcxt.remove();
            }
        }

        if (word0 != 0) {
            h = word0 *% 271 +% (c4 & 0xff);
        } else {
            h = word0 *% 271 +% u(c1);
        }
        // Word stream cm(4-5)
        if (c1 == 12) cmC[4].set(0) else cmC[4].set(word0 +% (number0 *% 191 +% numlen0));
        if (c1 == 12) cmC[4].set(0) else cmC[4].set(h +% word1);
        cmC[5].set(h +% word2 *% 71);
        cmC[6].set(((ttype & 0x3f) << 16) +% (c4 & 0xffff));
        cmC[8].set((c4 & 0xffffff) +% ((w4 << 18) & 0xff000000));
        cmC[8].set((wtype & 0x3fffffff) *% 4 +% (w4 & 3));
        cmC[8].set((fccontext *% 4) +% ((wtype & 0x3ffff) << 9) +% w4br);

        cmC[9].set(@as(u32, colcxt.lastfc(0)) | (fccontext << 15) | ((ttype & 63) << 7) | (brcontext << 24));
        cmC[9].set(@as(u32, colcxt.lastfc(0)) | ((c4 & 0xffffff) << 8));

        cmC[10].set((w4 & 3) +% word0 *% 11);
        cmC[10].set(c4 & 0xffff);
        cmC[10].set(((u(fc) << 11) | u(c1)) +% ((w4 & 3) << 18));

        cmC[11].set((w4 & 15) +% ((ttype & 7) << 6));
        cmC[11].set(u(c1) | (u(col * @intFromBool(c1 == txt.SPACE)) << 8) | ((w4 & 15) << 16));
        cmC[11].set(if (fc1 != 0) firstWord else (u(fc) << 11));
        if (c1 == 12) cmC[11].set(0) else cmC[11].set(91 *% 83 *% worcxt.word(1) +% 89 *% word0);

        cmC[12].set(u(c1) +% ((ttype & 0x38) << 6));
        cmC[12].set(worcxt.fword *% 11 +% w4br);
        cmC[12].set(u(c1) +% word0 +% number0 *% 191);
        cmC[12].set(((c4 & 0xffff) << 16) | (fccontext << 8) | u(fc));
        cmC[12].set(((wtype & 0xfff) << 8) +% (w4 & 0xfc));

        if (fc1 == 1) {
            cmC[13].set(worcxt.fword *% 3191 +% (w4 & 3));
            cmC[13].set(h +% firstWord *% 89);
            cmC[13].set(word0 *% 53 +% u(c1) +% w4br);
        } else {
            cmC[13].set(u(above) | ((ttype & 0x3f) << 9) | (u(colcxt.collen(0, 0)) << 19) | ((w4 & 3) << 16));
            cmC[13].set(h +% firstWord *% 89);
            cmC[13].set(u(above) | (u(c1) << 16) | ((u(col) +% numlen0 +% w4br) << 8) | (u(above1) << 24));
        }
        if (colcxt.lastfc(0) == '*') {
            cmC[13].set((fccontext << 8) | ((w4br & 0xfff) << 16));
            cmC[13].set(u(c1));
            cmC[13].set(word0);
        } else {
            cmC[13].set(@as(u32, tab.WRT_W[@intCast(bufr(colcxt.abovecellpos))]) | (fccontext << 8) | ((w4br & 0xff) << 16));
            cmC[13].set(u(bufr(colcxt.abovecellpos)) | (u(c1) << 8));
            cmC[13].set(word0 +% tab.WRT_W[@intCast(bufr(colcxt.abovecellpos))]);
        }

        cmC[14].set(x4 & 0xff00ff);
        cmC[14].set((x4 & 0xff0000ff) | ((ttype & 0xe07) << 8));

        // Indirect
        var f: u32 = (c4 >> 8) & 0xffff;
        t2[f] = (t2[f] << 8) | u(c1);
        f = c4 & 0xffff;
        f = f | (t2[f] << 16);
        var d: u32 = (c4 >> 8) & 0xff;
        t1[d] = (t1[d] << 8) | u(c1);
        d = u(c1) | (t1[@intCast(c1)] << 8);

        t1[brcontext] = (t1[brcontext] << 2) | (w4 & 3);
        const d4: u32 = (ttype & 7) | (t1[brcontext] << 3);
        if (c1 == 12 or fccontext == txt.CURLYOPENING) cmC[7].set(0) else cmC[7].set(d4);

        cmC[14].set((d4 & 0xffff) | ((ttype & 0x38) << 16));
        cmC[13].set(f & 0xffffff);

        cmC[15].set((u(c1) << 8) | (d >> 2) | (u(fc) << 16));
        cmC[15].set((c4 & 0xffff) +% @as(u32, @intFromBool(c2 == c3)));

        cmC[16].set((ttype & ttype1) *% 256 | ((w4 & wtype1 & 255) << 0));
        cmC[16].set(x4);
        cmC[17].set(257 *% word1 +% fccontext +% 193 *% (ttype & ttype1));
        cmC[17].set(u(fc) | ((w4r & 0xfff) << 9) | (u(c1) << 24));
        if (colcxt.lastfc(0) == txt.SQUAREOPEN and fccontext == txt.COLON)
            cmC[17].set(worcxt.fword *% 83 +% (w4 & 3) *% 11 +% brcontext)
        else
            cmC[17].set((x4 & 0xffff00) +% brcontext +% (fccontext << 24));

        cmC[18].set(d);
        cmC[18].set(((d & 0xffff00) >> 4) | (w4 & 0xf) | ((ttype & 0xfff) << 20));
        cmC[18].set((x4 >> 16) | ((w4 & 255) << 24));
        if (c1 > 127) cmC[18].set(((((w4 & 12) *% 256) +% u(c1)) << 11) | ((f & 0xffffff) >> 16)) else cmC[18].set((u(c1) << 11) | (w4br << 8) | ((f & 0xffffff) >> 16));
        cmC[18].set((fccontext *% 4 +% w4br) | ((c4 & 0xffff) << 9) | ((w4 & 0xff) << 24));
        cmC[18].set((f >> 16) | ((w4 & 0x3c) << 25) | ((ttype & 0x1ff) << 16));

        cmC[19].set(words +% (spaces << 8) +% ((w4 & 15) << 16) +% (((wtype >> 3) & 511) << 21) +% (u(fc1) << 30));
        cmC[19].set(u(c1) +% ((ttype << 5) & 0x1fffff00));
        cmC[19].set(w4r *% 16 +% w4br);
        cmC[19].set(((d & 0xffff) >> 8) +% ((64 *% w4r) & 0x3ffff00) +% (brcontext << 25));
        var dd: i32 = pos - wp[word0 & 0xffff];
        if (dd > 255) dd = 256 + (c1 << 16) else dd = dd + (buf(dd) << 8) + (c1 << 16);
        if (fccontext == 64 and brcontext == txt.SQUAREOPEN) cmC[19].set(0) else cmC[19].set(u(dd) | (u(fc) << 24));

        cmC[20].set((x4 & 0x80f00000) +% ((x4 & 0x0000f0ff) << 12));
        if (fc1 == 1) {
            if (c1 == 12 or fccontext == 31 or fccontext == txt.CURLYOPENING) cmC[20].set(0) else cmC[20].set(h +% worcxt.word(1) *% 53 *% 79 +% worcxt.word(3) *% 53 *% 47 *% 71);
        } else {
            if (col == 31) cmC[20].set(c4 << 16) else cmC[20].set(u(above) | ((c4 & 0xffff) << 16) | (u(above1) << 8));
        }

        if (c1 == 12 or fccontext == txt.CURLYOPENING or fccontext == 31 or fc == 30 or brcontext == txt.LESSTHAN) {
            cmC[21].set(0);
            cmC[21].set(0);
        } else {
            cmC[21].set(worcxt.word(1) *% 83 *% 1471 -% word0 *% 53 +% worcxt.word(2));
            cmC[21].set(h +% worcxt.word(2) *% 53 *% 79 +% worcxt.word(3) *% 53 *% 47 *% 71);
        }
        cmC[22].set(((wtype & 7) << 10) +% (w4 & 3) +% u(fc) *% 4 +% (w4br << 24));
        cmC[22].set(word0 *% 3301 +% number0 *% 3191);
        if (c1 == 12 or fccontext == txt.CURLYOPENING or fccontext == 31 or fc == 30 or brcontext == txt.LESSTHAN) cmC[23].set(0) else cmC[23].set(w4br +% worcxt.word(2) *% (wtype & w42));

        scmA[0].set(u(c1));
        scmA[1].set(u(c2 * fc1));
        scmA[2].set((f & 0xffffff) >> 16);
        scmA[3].set(ttype & 0x3f);
        scmA[4].set(w4 & 0xff);
        scmA[5].set(brcontext);
        scmA[6].set(u(fc1) + 2 *% (wtype & 0x3f));
        scmA[7].set(u(fc));

        if (wshift != 0 or c1 == 10) {
            word3 = word3 *% 47;
            word2 = word2 *% 53;
            word1 = word1 *% 83;
        }

        cmC[24].set((w4br *% 256) +% u(fc) +% (((wtype >> 0) & 0xFFF) << 16));
        AH1 = txt.hash((x5 >> 0) & 255, (x5 >> 8) & 255, (x5 >> 16) & 0x80ff);
        AH2 = txt.hash(19, x5 & 0x80ffff, 0xffffffff);
    }

    const c0b: i32 = c0 << @as(u5, @intCast(8 - bpos));
    ismatch = @bitCast(matchModel2mix(0));

    scmA[0].mix(0);
    scmA[1].mix(0);
    scmA[2].mix(0);
    scmA[3].mix(0);
    scmA[4].mix(0);
    scmA[5].mix(0);
    scmA[6].mix(0);
    if (fccxt.cxt == txt.COLON and fccxt.last() == txt.SQUAREOPEN) {
        x.mxInputs[0].add(0);
        x.mxInputs[0].add(0);
    } else {
        scmA[7].mix(0);
    }

    // order X
    ord = cmC[0].mix(0);
    if (ord == 3) ord = 2;
    ord = ord + cmC[1].mix(0);
    ord = ord + cmC[2].mix(0);
    ord = ord + cmC[3].mix(0);
    if (c1 == 12) {
        cmC[4].mix4(0);
        cmC[4].mix4(0);
        cmC[4].cn = 0;
        ord2 = 0;
    } else {
        ord2 = cmC[4].mix(0);
    }
    if (c1 == 12) {
        cmC[5].mix4(0);
        cmC[5].cn = 0;
    } else {
        ord2 = ord2 + cmC[5].mix(0);
    }
    _ = cmC[6].mix(0);
    _ = cmC[7].mix(0);
    _ = cmC[8].mix(0);
    _ = cmC[9].mix(0);
    _ = cmC[10].mix(0);
    _ = cmC[11].mix(0);
    _ = cmC[12].mix(0);
    if (c1 == 12) {
        cmC[13].mix4(0);
        cmC[13].mix4(0);
        cmC[13].mix4(0);
        cmC[13].mix4(0);
        cmC[13].mix4(0);
        cmC[13].mix4(0);
        cmC[13].mix4(0);
        cmC[13].cn = 0;
    } else {
        _ = cmC[13].mix(0);
    }
    _ = cmC[14].mix(0);
    _ = cmC[15].mix(0);
    _ = cmC[16].mix(0);
    _ = cmC[17].mix(0);
    _ = cmC[18].mix(0);
    _ = cmC[19].mix(0);
    if (fc1 == 1 and (c1 == 12 or fccxt.cxt == 31 or fccxt.cxt == txt.CURLYOPENING)) {
        cmC[20].cn = 1;
        _ = cmC[20].mix(0);
        cmC[20].mix4(0);
    } else {
        _ = cmC[20].mix(0);
    }
    // order Word
    if (c1 == 12 or fccxt.cxt == txt.CURLYOPENING or fccxt.cxt == 31 or fc == 30 or brcxt.cxt == txt.LESSTHAN) {
        cmC[21].mix4(0);
        cmC[21].mix4(0);
        cmC[21].cn = 0;
    } else {
        ord2 = ord2 + cmC[21].mix(0);
    }
    _ = cmC[22].mix(0);
    if (c1 == 12 or fccxt.cxt == txt.CURLYOPENING or fccxt.cxt == 31 or fc == 30 or brcxt.cxt == txt.LESSTHAN) {
        cmC[23].mix4(0);
        cmC[23].cn = 0;
    } else {
        ord2 = ord2 + cmC[23].mix(0);
    }
    _ = cmC[24].mix(0);
    _ = cmC[25].mix(0);
    _ = cmC[26].mix(0);
    _ = rcmA[0].mix(0);

    // ---- Mixer contexts ----
    if (bpos == 0) {
        mxA[1].cxt = @intCast((w4 & 63) *% 8 +% (ttype & 7));
    } else if (bpos > 3) {
        c = tab.WRT_W[@intCast(c0b & 255)];
        mxA[1].cxt = @intCast((ci((w4 << 2) & 63) + c) * 8 + ci(w4br));
    } else {
        mxA[1].cxt = @intCast((w4 & 63) *% 8 +% w4br);
    }

    if (bpos != 0) {
        c = c0b;
        if (bpos == 1) c = c + 16 * ci(words * 2 & 4) else if (bpos > 3) c = @as(i32, tab.WRT_W[@intCast(c0b & 255)]) * 64;
        c = (@min(bpos, 5)) * 256 + ci(wtype & 7) + ci(fqcxt) * 8 + (c & 192);
    } else {
        c = ci((words & 12) * 16) + ci(wtype & 7) + ci(w4br) * 8;
    }
    mxA[2].cxt = @intCast(c);

    mxA[3].cxt = @intCast((4 * ci(words) & 0xf0) + ord * 256 + ci(w4 & 15));
    mxA[7].cxt = @intCast(ci(wtype & 0x1f8) * 4 + (2 * ci(words) & 0x1c) + ci(w4 & 3));
    c = c0b;
    mxA[4].cxt = @intCast(bpos * 256 + ((@as(i32, @intCast(((numbers | words) << @as(u5, @intCast(bpos))) & 255)) >> @as(u5, @intCast(bpos))) | (c & 255)));
    mxA[9].cxt = @intCast(ord * 8 + (if (w4br != 0) @as(i32, 1) else 0) * 4 + ci(w4 & 3));

    if (bpos != 0) {
        if (bpos == 1) c = c + 16 * ci(ttype & 7) else if (bpos == 2) c = c + 16 * ci(w4 & 3) else if (bpos == 3) c = c + 16 * ci(words & 1) else c = bpos + (c & 0xf0);
        if (bpos < 5) c = bpos + (c & 0xf0);
    } else c = 16 * ci(w4 & 0xf);
    ord = ord - 1;
    if (ord < 0) ord = 0;
    if (ismatch != 0) ord = ord + 1;
    mxA[5].cxt = @intCast(c + ord * 256 + 8 * fc1);

    mxA[6].cxt = @intCast((ord2 * 256 + ci(w4 & 0xf0) + ci((ttype & 0x38) >> 2)) * 4 + ci(fqcxt));

    if (bpos > 2)
        mxA[8].cxt = @intCast(@as(i32, tab.WRT_T[@intCast(c0b & 255)]) * 256 + ci(w4br) * 32 + ci(words & 7) * 4 + fc1 + (if (ismatch != 0) @as(i32, 2) else 0))
    else
        mxA[8].cxt = @intCast(ci(ttype & 3) * 256 + ci(w4br) * 16 + ci(words & 7) * 2 + fc1 + (if (ismatch != 0) @as(i32, 128) else 0));

    x.mxInputs[1].add(mxA[1].p1());
    x.mxInputs[1].add(mxA[2].p1());
    x.mxInputs[1].add(mxA[3].p1());
    x.mxInputs[1].add(mxA[4].p1());
    x.mxInputs[1].add(mxA[5].p1());
    x.mxInputs[1].add(mxA[6].p1());
    x.mxInputs[1].add(mxA[7].p1());
    x.mxInputs[1].add(mxA[8].p1());
    return mxA[9].p();
}

// =================== update1 (ref 2920-2990) ===================
fn update1() void {
    x.c0 += x.c0 + x.y;
    if (x.c0 >= 256) {
        x.c4 = (x.c4 << 8) +% @as(u32, @intCast(x.c0 & 0xff));
        x.c0 = 1;
        x.blpos += 1;
        // Adapt each mixer's error-limit based on recent failure rate.
        if ((fails & 255) == 0) {
            var k: usize = 1;
            while (k < 9) : (k += 1) if (k != 7) {
                mxA[k].elim = @max(256, mxA[k].elim + 1);
            };
        } else {
            var k: usize = 1;
            while (k < 9) : (k += 1) if (k != 7) {
                mxA[k].elim = @min(16, mxA[k].elim - 1);
            };
        }
        rate = 6 + @as(i32, @intFromBool(x.blpos > 14 * 256 * 1024)) + @as(i32, @intFromBool(x.blpos > 28 * 512 * 1024));
    }
    x.bpos = (x.bpos + 1) & 7;
    x.bposshift = 7 - x.bpos;
    x.c0shift_bpos = (x.c0 << 1) ^ (@as(i32, 256) >> @as(u5, @intCast(x.bposshift)));
    mxA[1].update(x.y);
    mxA[2].update(x.y);
    mxA[3].update(x.y);
    mxA[4].update(x.y);
    mxA[5].update(x.y);
    mxA[6].update(x.y);
    mxA[7].update(x.y);
    mxA[8].update(x.y);
    mxA[9].update(x.y);
    x.mxInputs[0].ncount = 0;
    x.mxInputs[1].ncount = 0;
    // paq8hp12 fail tracking
    if ((fails & 0x00000080) != 0) failcount -%= 1;
    fails = fails *% 2;
    failz = failz *% 2;

    if (x.y != 0) pr = 4095 - pr;
    if (pr >= tab.E_L[@intCast(x.bpos)]) {
        fails +%= 1;
        failcount +%= 1;
    }
    if (pr >= 848) failz +%= 1;

    pr = modelPrediction(x.c0, x.bpos, x.c4);
    st.addPrediction(pr);

    var pu: i32 = (apmA[0].p(pr, @intCast(x.c0), 3, x.y) + 7 * pr + 4) >> 3;
    var pz: i32 = @as(i32, @bitCast(failcount +% 1));
    pz += @as(i32, @intCast(txt.tri[@intCast((fails >> 5) & 3)]));
    pz += @as(i32, @intCast(txt.trj[@intCast((fails >> 3) & 3)]));
    pz += @as(i32, @intCast(txt.trj[@intCast((fails >> 1) & 3)]));
    if ((fails & 1) != 0) pz += 8;
    pz = @divTrunc(pz, 2);

    pu = apmA[3].p(pu, @intCast((u(x.c0 * 2) ^ AH1) & 0x3ffff), @intCast(rate), x.y);
    st.addPrediction(pu);
    var pv: i32 = apmA[1].p(pr, @intCast((u(x.c0 * 8) ^ txt.hash(29, failz & 2047, 0xffffffff)) & 0xffff), @intCast(rate + 1), x.y);
    st.addPrediction(pv);
    if ((fails & 255) != 0)
        pv = apmA[4].p(pv, @intCast(txt.hash(u(x.c0), w4 & 0xfffc, wtype & 0x1ff) & 0x1ffff), @intCast(rate), x.y)
    else
        pv = apmA[4].p(pv, @intCast(txt.hash(u(x.c0), (w4r & 0xfffc) + 0x10000, wtype & 0x1ff) & 0x1ffff), @intCast(rate), x.y);
    st.addPrediction(pv);
    const pt: i32 = apmA[2].p(pr, @intCast((u(x.c0 * 32) ^ AH2) & 0xffff), @intCast(rate), x.y);
    st.addPrediction(pt);
    pz = apmA[5].p(pu, @intCast((u(x.c0 * 4) ^ txt.hash(u(@min(9, pz)), x5 & 0x80ff, 0xffffffff)) & 0x1ffff), @intCast(rate), x.y);
    st.addPrediction(pz);
    if ((fails & 255) != 0)
        pr = (pt * 6 + pu + pv * 11 + pz * 14 + 31) >> 5
    else
        pr = (pt * 4 + pu * 5 + pv * 12 + pz * 11 + 31) >> 5;
    st.addPrediction(pr);
}

// =================== FXCM Model wrapper (ref FXCM::*) ===================
// Auxiliary text model. Lazy-eval like paq8: Predictreturns the predictions
// computed by the previous Perceive->update1->modelPrediction.
pub const FXCM = struct {
    pub fn create(a: std.mem.Allocator) *FXCM {
        const self = a.create(FXCM) catch unreachable;
        // fxcmv1::Predictor: precompute tables, x.Init, alloc buffers, PredictorInit.
        // predictorInitalready does st.reset (x.Init + mxInput/prediction alloc,
        // prim.init) and allocates the history buffer + match hashtable.
        predictorInit(a);
        return self;
    }

    pub fn predict(self: *FXCM) []const f32 {
        _ = self;
        return st.model_predictions;
    }

    pub fn numOutputs(self: *FXCM) usize {
        _ = self;
        return st.model_predictions.len;
    }

    pub fn perceive(self: *FXCM, bit: i32) void {
        _ = self;
        st.x.y = bit;
        update1();
        st.resetPredictions();
    }

    pub fn byteUpdate(self: *FXCM) void {
        _ = self; // fxcm has no byte-level hook
    }

    pub fn model(self: *FXCM) Model {
        return .{ .ptr = self, .vtable = model_mod.vtableFor(FXCM) };
    }
};
