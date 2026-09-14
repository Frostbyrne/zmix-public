//! FxcmV26 — the full fxcm_v26 predictor assembly (integration steps C+D), a 1:1
//! Zig port of cmix-fxcm/src/fxcm_v26.rs (the golden-verified Rust port).
//!
//! Owns all sub-models and drives them per bit. `add1`/`add2`/`add4` push a
//! clipped value into BOTH the mixer-input array and the 560-slot raw-prediction
//! array (slots / model_predictions1); `add1_internal` pushes only the mixer
//! input; `add1_tail` pushes only the slot.
//!
//! Gate (step E, no-dict): per-slot folds vs goldens/pcs_nodict.raw, total RAW
//! 0x91521e61b9ea0014.

const std = @import("std");

// -Dfxcm-final-blend (C7): weight on pr in the FINAL output blend. Ships 3 (3:1),
// i.e. only 1/4 of the coded probability comes from the whole APM/mmmO cascade.
// Exact-shift blends only, so the default path is bit-identical to the shipped shift.
const kFinalBlendW: i32 = @intCast(@as(u32, @import("build_options").fxcm_final_blend));
const kFinalBlendS: u5 = if (kFinalBlendW == 1) 1 else if (kFinalBlendW == 3) 2 else if (kFinalBlendW == 7) 3 else @compileError("-Dfxcm-final-blend must be 1, 3 or 7");
const Allocator = std.mem.Allocator;
const build_options = @import("build_options");
const emit_sink = @import("emit_sink.zig");
const Sink = emit_sink.Sink;
const cm3_mod = @import("cm3.zig");
const slotdel_mod = @import("slotdel.zig");
// -Dleafgrad-fxcm (reopen trigger). fxcm owns the two per-slot side
// arrays because it is the one object that spans "predictor writes a gradient
// per mixer input" and "a ContextMap3 instance emitted that input".
const LGF: bool = cm3_mod.LGF;
const LGF_DIAG: bool = cm3_mod.LGF_DIAG;

// Hermetic-guarded env float (same pattern as predictor_lex): a root declaring
// `pub const zmix_hermetic = true` (ship.zig) compiles the env read away.
const root_mod = @import("root");
const FXCM_HERMETIC: bool = @hasDecl(root_mod, "zmix_hermetic") and root_mod.zmix_hermetic;
fn envF32Fxcm(name: []const u8, default: f32) f32 {
    // Also compiled out on Windows: std.posix.getenv is a WTF-16 compile error.
    if (comptime (FXCM_HERMETIC or @import("builtin").os.tag == .windows)) return default;
    const v = std.posix.getenv(name) orelse return default;
    return std.fmt.parseFloat(f32, v) catch default;
}

const cmcold = @import("cmcold"); // per-byte parse layer + shared base (RS module)
const byte_core = cmcold.byte_core;
const X = byte_core.X;
// This file (and the whole fxcm26 integer predictor) lives in the `cmfast`
// module (see cm_fast.zig): pure-integer hot compute gets ReleaseFast -O3
// codegen bit-identically at RS binary size. tables.zig stays in the root
// module (it owns the float table generation, pinned-math); the root caller
// generates STRT/SQT + the Tables instance and passes them into new.
const Tables = cmcold.Tables;
const mixer1 = @import("mixer1.zig");
const Mixer1 = mixer1.Mixer1;
const Mix = mixer1.Mix;
const ContextMap3 = cm3_mod.ContextMap3;
const Spill = @import("cm3.zig").Spill;
const ContextMap4 = @import("cm4.zig").ContextMap4;
const DirectStateMap = @import("direct_state_map.zig").DirectStateMap;
const StationaryMap = @import("leaf_maps.zig").StationaryMap;
const SmallStationaryContextMap = @import("sscm.zig").SmallStationaryContextMap;
const RunContextMap = @import("run_context_map.zig").RunContextMap;
const MatchModel2 = @import("match_model.zig").MatchModel2;
const XmlModel1 = @import("xml_model.zig").XmlModel1;
const pb_mod = cmcold.parse_byte;
const ParseByte = pb_mod.ParseByte;
const hash3 = pb_mod.hash3;
const WRT_2B = pb_mod.WRT_2B;
const WRT_3B = pb_mod.WRT_3B;
const WordsContext = @import("words_context.zig").WordsContext;

const VERB: u32 = 1 << 0;
const NOUN: u32 = 1 << 1;

// -Dfxcm-slotdelete: the `slots` bank (and the 4 `mx_a2` mixers over it) shrinks
// from the stock 560 by the deleted emissions, rounded up to a multiple of 16
// (mixer1's kernels read `round16(n)` lanes of the bank). == 560 at the default.
const NUM_MODELS: usize = slotdel_mod.SLOTS_W;
/// The `in1` bank (and the 18 `mx_a` mixers over it), stock 544. Same rule.
const IN1_W: usize = slotdel_mod.IN1_W;
// -Dfxcm-slotmask engagement witness: printed ONCE per process.
var slotmask_announced: bool = false;
const E_L = [8]i32{ 1830, 1997, 1973, 1851, 1897, 1690, 1998, 1842 };
const TRI = [4]i32{ 0, 4, 3, 7 };
const TRJ = [4]i32{ 0, 6, 6, 12 };
const C_S = [27]i32{ 28, 32, 32, 32, 34, 31, 33, 33, 35, 35, 29, 32, 33, 34, 30, 36, 31, 32, 32, 32, 32, 32, 33, 32, 32, 32, 32 };
const C_S3 = [27]i32{ 43, 32, 32, 32, 34, 29, 32, 33, 37, 35, 33, 28, 31, 35, 28, 30, 33, 34, 32, 32, 32, 32, 32, 32, 32, 32, 32 };
const C_S4 = [27]i32{ 9, 8, 12, 12, 8, 12, 15, 8, 8, 12, 10, 7, 7, 8, 8, 13, 13, 14, 8, 8, 12, 12, 12, 12, 12, 12, 12 };

const ESCAPE: i32 = 12;
const SPACE: i32 = 32;
const LF: i32 = 10;
const FIRSTUPPER: i32 = 64;
const HTLINK: i32 = 31;
const HTML: i32 = 30;
const GREATERTHAN: i32 = 78;
const LESSTHAN: i32 = 76;
const CURLYOPENING: i32 = 80;
const SQUAREOPEN: i32 = 91;
const APOSTROPHE: i32 = 39;
const WIKITABLE: u8 = 45;

const M4K: u32 = 4096;
// C-GROW factor for the high-value cmC2 families (byte-order cm_c2[0-3] +
// word/sentence cm_c2[4,5,13,17,18,19,20]). Default 1 = byte-identical; ship
// recipe sets -Dcmc2-grow=2 to recover collision quality (the located award
// gap) funded by the ppmd-mmap valve + fix14-16 freed RAM. Applied ONLY to the
// value-carrying families so the RAM cost stays ~+1.75GB, not +4.7GB.
const HV_GROW: u32 = build_options.cmc2_grow;
// C-GROW-HOT: targeted growth of the four EXTREME-churn tables from the cm3-diag
// 100m established-kill census — cm_c2[18] (28.7M estk, the largest single churn
// pool, +134MB@x2), cm_c2[5] (13.7M estk, +537MB), cm_c2[16] (23.5% estk rate,
// +17MB), cm_c2[6] (26.1% rate, +0.5MB). Together ~94% of the established-kill
// volume among the grown families => captures most of grow2's -18,147@100m at
// ~1/5 the RAM (+689MB@x2 vs +3.4GB). Composes with HV_GROW (both default 1).
const HOT_GROW: u32 = build_options.cmc2_grow_hot;
// C-GROW-HOT v2 (-Dcmc2-hot2=0..6, default 0 = byte-identical): census-driven
// escalation of the hot set, per-table total multipliers vs stock. Ranking
// metric: score = replacements + established_kills per added MB (the pure-estk
// model FAILS the v1/full-grow retrodiction — 107% predicted vs 53% measured;
// repl+estk retrodicts 53.4% vs 53.3% measured, kappa ~ 18 B/M-score at 100m).
// Both tiers carry the v1 hot set ([5]x2, [18]x2/x4, [16]x2/x4, [6]x4) and add
// the cmcr/cmcr2 recovery family — the census' top growable candidates
// (3.3-8.0M score/MB, 100% occupancy, 15-24% estk) AND the cmix-lex author's
// own largest RAM bet (fx2->lex added the whole ~352MB cmcr complex while
// shrinking other tables to fit the gate) — plus the tiny saturated cm_c[0]
// (page-reset; 497M score on 0.25MB), cm_c[2] (233M on 16KB), cm_c[3] (48M on
// 0.25MB) at x4 for ~1.6MB total. Tier 1 (conservative) +271MB over v1; tier 2
// (stretch) +1085MB over v1 (adds cmcr x4, cmcr2[all] x2, cm_c2[7,15,19] x2,
// cm_c[4] x2, cm_c2[16,18] x4). Excluded on census evidence: cm_c2[4],[8],[13]
// (63-68% occupancy at 100m — upstream lex grew [4]/[8], but that pressure only
// appears at enwik9 scale; a 100m race cannot confirm it), cm_c44 (5.7% occ).
const HOT2: u32 = build_options.cmc2_hot2;
comptime {
    if (HOT2 > 8) @compileError("-Dcmc2-hot2 takes 0 (off), 1 (conservative), 2 (stretch), 3 (budget), 4 (light), 5 (t1+[16]x4, +33.55MB), 6 (t1+[16]x4+[18]x4, +302.0MB), 7 (t6+[17]x2, +64MiB) or 8 (t6+[20]x2, +256MiB)");
    if (HOT2 != 0 and HOT_GROW != 1)
        @compileError("-Dcmc2-grow-hot (v1) and -Dcmc2-hot2 (v2) are alternative hot-set levers; set only one");
}
const H2_ONES_21: [21]u32 = @splat(1);
/// Per-instance grow-hot-v2 total multiplier vs stock, cm_c2 family.
const H2_C2: [21]u32 = switch (HOT2) {
    0 => H2_ONES_21,
    1 => .{ 1, 1, 1, 1, 1, 2, 4, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 1, 2, 1, 1 },
    2 => .{ 1, 1, 1, 1, 1, 2, 4, 2, 1, 1, 1, 1, 1, 1, 1, 2, 4, 1, 4, 2, 1 },
    // tier 3 BUDGET = tier 1 minus cm_c2[5] (537MB for the WORST score/MB of the
    // hot set — census ~200M score = 0.4M/MB vs cmcr's 3.3-8.0M/MB). Fits the
    // sm30-measured RAM reality: +422.5MB over stock vs tier 1's +959.5MB.
    3 => .{ 1, 1, 1, 1, 1, 1, 4, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 1, 2, 1, 1 },
    // tier 4 LIGHT = cm_c2 [18,16,6]-only (+152MB over stock): the c1_wall race
    // measured full-C1 walls 1.0999c/1.1573d vs the 1.040 gate with tier 3's
    // +422MB in the stack — the cmcr/cm_c growth is the prime wall suspect.
    // Tier 4 keeps the three cheapest-per-MB cm_c2 grows and drops the rest.
    4 => .{ 1, 1, 1, 1, 1, 1, 4, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 1, 2, 1, 1 },
    // ===== INTERMEDIATE TIERS  =====
    // The t1->t2 step is -14,046 B @100m but costs +814MB, ~4x the fourth-E's
    // ~197MB headroom, so it has never been shippable. The audit's own note:
    // "intermediate tiers between t1 and t2 never built (pow2-locked hand-picked
    // sets; e.g. t1+[16]x4)". These build that ladder, ordered by score/MB from
    // the 100m established-kill census: cm_c2[16] is 23.5%% estk RATE for only
    // +17MB@x2, and cm_c2[18] is the LARGEST single churn pool (28.7M estk) at
    // +134MB@x2. Both are already at x2 in t1; t2 takes both to x4.
    // tier 5 = t1 + cm_c2[16] x4 (+33.55MB over t1; x2->x4 costs the
    // same as x1->x2). Whole-span 100MB result: -81 B, neutral.
    5 => .{ 1, 1, 1, 1, 1, 2, 4, 1, 1, 1, 1, 1, 1, 1, 1, 1, 4, 1, 2, 1, 1 },
    // tier 6 = tier 5 + cm_c2[18] x4 (+302.0MB over t1). Whole-span 100MB
    // result: -3,445 B; corrected anon accounting says it fits outright,
    // with a composed gate-relevant RAM measurement still owed.
    6 => .{ 1, 1, 1, 1, 1, 2, 4, 1, 1, 1, 1, 1, 1, 1, 1, 1, 4, 1, 4, 1, 1 },
    // ===== TIERS 7 / 8  — spending the RAM the 08-20 operator ruling freed.
    // Both are "tier 6 + ONE further UNGROWN cm_c2 bank at x2", never a sum of the two:
    // fxcm-INTERNAL calibration levers SUBSTITUTE at 0-7% retention (plan 8.140), so they
    // are measured as alternative REPLACEMENTS for tier 6, not as a stack.
    //
    // Selected by the tier-6 precedent's own rule — LARGEST ABSOLUTE CHURN POOL, not best
    // score/MB ([18] with 28.7M estk paid where the best-rate [16] did not, tier5 = -81 B).
    // Applying that rule to the context-map churn censuses showed its two readings
    // DISAGREE, so both tiers are built:
    //
    //   metric                          winner among UNGROWN banks
    //   absolute estk, 300MB prefix     [17] 11,090,393  vs [20]  4,248,109
    //   absolute estk, 100MB            [17]  1,519,345  vs [20]     16,939
    //   absolute estk, MATURE @0.4GB    [20]  2,555,810  vs [17]  2,184,133
    //   absolute repl+estk, every tier  [20] (120.5M @100MB, 346.3M @300MB) — #1 by 1.6x
    //
    // Excluded from both tiers: [4]/[8]/[13] (banked closure cmc2-grow-4-8-13: full but
    // NOT evicting — estk% 0.008/0.364/0.215) and [1]/[2]/[3] (the coldest maps by
    // measured LOO value, and the very maps -Dcmc2-cut proposes to SHRINK 8x).
    //
    // tier 7 = tier 6 + cm_c2[17] x2 (64 -> 128 MiB, +64 MiB). [17] is the largest
    // absolute ESTK pool among ungrown banks at both census tiers (5.487% of 668.6M probes
    // at 300MB), is 100.000% occupied with 100.0000% full buckets, replaces 30% of its
    // probes — and appears in NO existing tier of the ladder.
    7 => .{ 1, 1, 1, 1, 1, 2, 4, 1, 1, 1, 1, 1, 1, 1, 1, 1, 4, 2, 4, 1, 1 },
    // tier 8 = tier 6 + cm_c2[20] x2 (256 -> 512 MiB, +256 MiB). [20] is the largest
    // absolute repl+estk pool among ungrown banks at every tier, and the largest absolute
    // ESTK pool at MATURE depth. It is a LATE SATURATOR: estk% 0.014 @100MB -> 1.242
    // @300MB, i.e. its damage is mostly INVISIBLE at the 100m screening tier.
    8 => .{ 1, 1, 1, 1, 1, 2, 4, 1, 1, 1, 1, 1, 1, 1, 1, 1, 4, 1, 4, 1, 2 },
    else => unreachable,
};
/// cm_c family ([0] is page_reset-wiped: resetmemsets the WHOLE table, so x4
/// costs a 1MB memset per </page> — accepted, its 497M score tops the census;
/// [1] is init-only in enwik9 (census probes=0), left alone).
const H2_C: [6]u32 = switch (HOT2) {
    0 => @splat(1),
    1 => .{ 4, 1, 4, 4, 1, 1 },
    2 => .{ 4, 1, 4, 4, 2, 1 },
    3 => .{ 4, 1, 4, 4, 1, 1 },
    4 => @splat(1),
    // tiers 5..8 = t1 + extra cm_c2 growth only => inherit t1 here
    5, 6, 7, 8 => .{ 4, 1, 4, 4, 1, 1 },
    else => unreachable,
};
// -Dcmc-grow (default 0 = all x1 = byte-identical): EXTRA per-instance growth of
// the cm_c family, multiplied ON TOP of H2_C. Lives in cmfast_opts, not the
// shared model_opts (the own-module rule), so it costs decomp_bin zero when off.
//
// THE CENSUS THAT SELECTS THE BANKS (tier-6 -Dcm3-diag, e8_20m WITH english.dic).
// estk% is the grow-value driver, NOT occ%:
//   cm_c[3] estk% 18.544 / occ 99.98 / fullbkt 99.78 -- most starved in the engine
//   cm_c[2] estk% 14.305 / occ 99.99 / fullbkt 99.80 -- 2.97M established kills,
//                                                       the largest ABSOLUTE pool
//                                                       anywhere, out of 64 KiB
//   cm_c[1] estk%  8.416 / occ 94.49 / fullbkt 73.07 -- never grown at ANY tier;
//                                                       the H2_C comment's
//                                                       "init-only in enwik9,
//                                                       census probes=0" is a
//                                                       NO-DICT artifact (80.2M
//                                                       probes here, dict mode)
//   cm_c[0] estk%  0.091 / occ 31.08 / fullbkt  0.01 -- NOT capacity-bound: it is
//                                                       reset-wiped every
//                                                       </page>, so its working
//                                                       set is one ARTICLE and it
//                                                       already fits. Its 21%
//                                                       miss rate is RESET misses,
//                                                       which capacity cannot fix.
// Tier 1 grows ONLY cm_c[0] and exists as the falsifier for that last reading.
// Tier 5 leaves cm_c alone and doubles the `cmcr` recovery complex instead
// (tier 6 ships it at x2; this takes it to x4, +320 MB). Same census, same rule:
// all nine cmcr banks read occ 99.97-100.00 %, fullbkt 99.78-100.00 % and
// estk% 0.033-5.027 (~2.1M established kills across the complex) at e8_20m dict
// -- i.e. FULL *and* EVICTING, the signature the `cmc2-grow-4-8-13` closure named
// as the grow-value driver. cmcr x4 is a COMPONENT of hot2 tier 2, which measured
// -14,046 B @100m over tier 1 but was killed on RAM at +814 MB; +320 MB alone now
// fits the 414.7 MiB of measured clean currency (121 MiB gate headroom +
// 294 MiB from -Dsse-ffl6=4), and this component has never been measured alone.
const CMC_GROW: u32 = build_options.cmc_grow;
comptime {
    if (CMC_GROW > 5) @compileError("-Dcmc-grow takes 0 (off), 1 ([0]x4 falsifier), 2 ([1,2,3]x4), 3 ([1]x16 [2]x16 [3]x16), 4 ([2]x16 alone) or 5 (cmcr x2 extra = x4 total)");
}
/// Extra per-instance multiplier for cm_c, applied on top of H2_C.
const CG_C: [6]u32 = switch (CMC_GROW) {
    0, 5 => @splat(1),
    1 => .{ 4, 1, 1, 1, 1, 1 },
    2 => .{ 1, 4, 4, 4, 1, 1 },
    3 => .{ 1, 16, 16, 16, 1, 1 },
    4 => .{ 1, 1, 16, 1, 1, 1 },
    else => unreachable,
};
/// Extra per-instance multiplier for cmcr, applied on top of H2_CR.
const CG_CR: u32 = if (CMC_GROW == 5) 2 else 1;

/// cmcr recovery family (all 9 members 100% occupied, 14-24% estk at 100m).
const H2_CR: [9]u32 = switch (HOT2) {
    0 => @splat(1),
    1 => @splat(2),
    2 => @splat(4),
    3 => @splat(2),
    4 => @splat(1),
    // tiers 5..8 = t1 + extra cm_c2 growth only => inherit t1 here
    5, 6, 7, 8 => @splat(2),
    else => unreachable,
};
/// cmcr2 recovery family (tier 1 skips [1], the weakest churn of the four).
const H2_CR2: [4]u32 = switch (HOT2) {
    0 => @splat(1),
    1 => .{ 2, 1, 2, 2 },
    2 => @splat(2),
    3 => .{ 2, 1, 2, 2 },
    4 => @splat(1),
    // tiers 5..8 = t1 + extra cm_c2 growth only => inherit t1 here
    5, 6, 7, 8 => .{ 2, 1, 2, 2 },
    else => unreachable,
};

// ------- ParseByte ctor tables (cm3_tables.rs BRACKETS/QUOTES/FCHAR) -------
const BRACKETS = [8]u8{ 40, 41, 80, 82, 91, 93, 76, 78 };
const QUOTES = [4]u8{ 39, 39, 34, 34 };
const FCHAR = [20]u8{ 64, 10, 96, 10, 74, 10, 76, 78, 77, 10, 91, 93, 80, 82, 42, 10, 81, 10, 31, 10 };

// ------- precomputed tables (regenerated at startup, like the C++ Predictor
// ctor) -------
// These used to be @embedFile'd golden .bin dumps (~31KB of .rodata, ~6-9KB
// after packing). They are now generated once in `new` from the same
// integer/pinned-float math as fxcmv1.cpp, and verified byte-identical to the
// goldens by the test at the bottom of this file (plus per-table tests in
// tables.zig and state_table.zig). Only PRE1 stays embedded: the C++ hardcodes
// `pre1[256]` as a literal array (fxcmv1.cpp:3514), so there is nothing to
// regenerate it from.
const state_table = cmcold.state_table;
const table_gen = cmcold.tables; // pure-int gens (St2P1/Rcpr)

fn loadI16(comptime path: []const u8, comptime n: usize) [n]i16 {
    @setEvalBranchQuota(n * 8 + 1000);
    const bytes = @embedFile(path);
    var arr: [n]i16 = undefined;
    for (0..n) |i| arr[i] = @bitCast(@as(u16, bytes[i * 2]) | (@as(u16, bytes[i * 2 + 1]) << 8));
    return arr;
}
var STA1: [1024]u8 = undefined;
var STA2: [1024]u8 = undefined;
var STA4: [1024]u8 = undefined;
var STA5: [1024]u8 = undefined;
var STA6: [1024]u8 = undefined;
var STA7: [1024]u8 = undefined;
var STRT_T: [4096]i16 = undefined;
var SQT_T: [4096]i16 = undefined;
var ST2_P1: [4096]i16 = undefined;
const ST2_P0 = [_]i16{0} ** 4096;
var RCPR_T: [512]i16 = undefined;
const PRE1_T = loadI16("goldens/dsm_PRE1.bin", 256);

// STA6 param row with the -Dsta6-mdc override (index 6 = mdc). At the default
// (22 = stock) this row is byte-equal to STA_PARAMS[4], so the generated table
// — and every archive byte — is identical to stock.
const STA6_ROW: [7]i32 = blk: {
    var r = state_table.STA_PARAMS[4];
    r[6] = build_options.sta6_mdc;
    break :blk r;
};

// C-PACK (`-Dcm3pack`): a NARROWED copy of the six state tables, used by the
// ContextMap3 banks ONLY. Narrowing must NOT leak into the DirectStateMap /
// ContextMap4 models: those are direct-mapped and untagged, and the findings'
// §6.4 law says narrowing REVERSES SIGN there (+0.59 % at w = 6, +1.35 % at
// w = 5 on `dmap`) because capacity is the cheapest claimant on an untagged
// map. Same intervention, opposite sign, inside one engine — so the two table
// sets are kept physically separate.
const CM3_NARROW: bool = cm3_mod.PACK and cm3_mod.W < 8;
var STA1P: [1024]u8 = undefined;
var STA2P: [1024]u8 = undefined;
var STA4P: [1024]u8 = undefined;
var STA5P: [1024]u8 = undefined;
var STA6P: [1024]u8 = undefined;
var STA7P: [1024]u8 = undefined;

var table_gen_once = std.once(generateTables);
fn generateTables() void {
    STA1 = state_table.generate(&state_table.STA_PARAMS[0]);
    STA2 = state_table.generate(&state_table.STA_PARAMS[1]);
    STA4 = state_table.generate(&state_table.STA_PARAMS[2]);
    STA5 = state_table.generate(&state_table.STA_PARAMS[3]);
    STA6 = state_table.generate(&STA6_ROW);
    STA7 = state_table.generate(&state_table.STA_PARAMS[5]);
    if (comptime CM3_NARROW) {
        const M = cm3_mod.NSTATES;
        STA1P = state_table.generateFitted(&state_table.STA_PARAMS[0], M);
        STA2P = state_table.generateFitted(&state_table.STA_PARAMS[1], M);
        STA4P = state_table.generateFitted(&state_table.STA_PARAMS[2], M);
        STA5P = state_table.generateFitted(&state_table.STA_PARAMS[3], M);
        STA6P = state_table.generateFitted(&STA6_ROW, M);
        STA7P = state_table.generateFitted(&state_table.STA_PARAMS[5], M);
    }
    ST2_P1 = table_gen.genSt2P1();
    RCPR_T = table_gen.genRcpr();
}

/// The table pointers the ContextMap3 banks are initialised with: stock unless
/// the decoupled bucket narrowed the state field.
const T1: *const [1024]u8 = if (CM3_NARROW) &STA1P else &STA1;
const T2: *const [1024]u8 = if (CM3_NARROW) &STA2P else &STA2;
const T4: *const [1024]u8 = if (CM3_NARROW) &STA4P else &STA4;
const T5: *const [1024]u8 = if (CM3_NARROW) &STA5P else &STA5;
const T6: *const [1024]u8 = if (CM3_NARROW) &STA6P else &STA6;
const T7: *const [1024]u8 = if (CM3_NARROW) &STA7P else &STA7;

// ---------------------------------------------------------------- helpers
inline fn clp(z: i32) i32 {
    if (z < -2047) return -2047;
    if (z > 2047) return 2047;
    return z;
}
inline fn clp1(z: i32) i32 {
    if (z < 0) return 0;
    if (z > 4095) return 4095;
    return z;
}
inline fn sar_pow2(v: i64, shift: i32) i32 {
    const sh: u6 = @intCast(shift);
    if (v >= 0) {
        return @intCast(v >> sh);
    } else {
        const t = ((-v) + ((@as(i64, 1) << sh) - 1)) >> sh;
        return -@as(i32, @intCast(t));
    }
}
inline fn gsbl(bpos: i32, c0: i32) i32 {
    const smask: i32 = @intCast((@as(u32, 0x31031010) >> @as(u5, @intCast(bpos << 2))) & 0x0F);
    return smask + (c0 & smask);
}
/// Wrapping left shift on u32 (Rust release `<<` semantics) — Zig `<<` panics on
/// overflow, so use `*%` by the power of two.
inline fn shl(x: u32, comptime k: u32) u32 {
    return x *% (@as(u32, 1) << @as(u5, k));
}

/// -Dapm-wupd — the APM interpolated-map UPDATE RULE (see build.zig).
///
/// PAQ7 (2005) refines BOTH bracket points of the 33-point map with the same full
/// step `(g-t)>>rate`, regardless of the interpolation weight `w` that actually
/// produced the prediction. Two consequences, both un-tuned since:
///   (1) point j is driven by every visit to bucket j-1 AND bucket j with equal
///       authority ⇒ the map is an implicit 2-cell UNWEIGHTED boxcar smoother of
///       the calibration curve, width fixed at 256 stretch units (= 1 nat);
///   (2) each joint update multiplies the inter-point gap by (1 - 2^-rate) ⇒ the
///       local SLOPE of the transfer curve is contracted at every touch.
/// Shelwien's SSE in the SAME binary (`src/sse.zig:sseUpdate`) does neither: it
/// moves the INTERPOLATED value and redistributes preserving the gap.
const APM_WUPD: u32 = build_options.apm_wupd;

/// -Dapm-diag — census of the APM's own access pattern, to settle by MEASUREMENT
/// whether the incumbent's 2-cell boxcar is first-order biased. It is unbiased iff
/// the visit density is flat across each adjacent cell pair; the bias grows with
/// the density RATIO between neighbours. Pure side-channel counters, never read by
/// the model, so ON stays bit-identical on the output.
const APM_DIAG: bool = build_options.apm_diag;

/// fxcmv1.cpp APM<B> (1565-1587): SarPow2 refine + internal cxt mask + clp1.
const ApmV26 = struct {
    index: usize,
    t: []u16,
    mask: u32,
    alloc: Allocator,
    /// -Dapm-wupd only: the interpolation weight that produced the prediction the
    /// NEXT `p` call will learn from. `void` (zero-size, no layout change at all)
    /// when the knob is off, so the stock path is byte-identical by construction.
    wprev: if (APM_WUPD != 0) i32 else void = if (APM_WUPD != 0) 0 else {},
    /// -Dapm-wupd=4 only: the in-row cell (0..31) of the last prediction, so the
    /// wider kernel can be clamped to its own 33-point row.
    cprev: if (APM_WUPD == 4) usize else void = if (APM_WUPD == 4) 0 else {},
    /// -Dapm-diag only: per-cell visit counts (33 grid points, cell j = lower point j)
    /// and the summed interpolation weight, so the mean w per cell is recoverable.
    hist: if (APM_DIAG) [33]u64 else void = if (APM_DIAG) .{0} ** 33 else {},
    wsum: if (APM_DIAG) [33]u64 else void = if (APM_DIAG) .{0} ** 33 else {},
    /// -Dapm-diag only: [total, stretch==-2047, stretch==+2047, pr_in==0, pr_in==4095].
    /// The last four are the fraction of traffic arriving at the secondary-estimation
    /// stage already SATURATED in the 12-bit probability domain — bits on which the
    /// whole APM chain is blind by construction.
    sat: if (APM_DIAG) [5]u64 else void = if (APM_DIAG) .{0} ** 5 else {},

    fn new(a: Allocator, tables: *const Tables, bits: u32) !ApmV26 {
        const s: usize = @as(usize, 1) << @as(u6, @intCast(bits));
        const t = try a.alloc(u16, s * 33);
        @memset(t, 0);
        var j: i32 = 0;
        while (j < 33) : (j += 1) {
            t[@intCast(j)] = @intCast(tables.squash((j - 16) * 128) * 16);
        }
        var i: usize = 33;
        while (i < s * 33) : (i += 1) t[i] = t[i - 33];
        return ApmV26{ .index = 0, .t = t, .mask = (@as(u32, 1) << @as(u5, @intCast(bits))) - 1, .alloc = a };
    }

    fn deinit(self: *ApmV26) void {
        self.alloc.free(self.t);
        self.t = &.{};
    }

    fn p(self: *ApmV26, tables: *const Tables, pr_in: i32, cxt: u32, rate: i32, y: i32) i32 {
        const pr: i32 = @as(i32, tables.stretch(pr_in));
        const g: i32 = (y << 16) + (y << @as(u5, @intCast(rate))) - y * 2;
        const lo = self.index;
        const t0: i32 = @as(i32, self.t[lo]);
        const t1: i32 = @as(i32, self.t[lo + 1]);
        if (comptime APM_WUPD == 0) {
            self.t[lo] = @truncate(@as(u32, @bitCast(t0 + sar_pow2(@as(i64, g - t0), rate))));
            self.t[lo + 1] = @truncate(@as(u32, @bitCast(t1 + sar_pow2(@as(i64, g - t1), rate))));
        } else {
            const wp = self.wprev; // 0..127, weight of t[lo+1] in that prediction
            if (comptime APM_WUPD == 1) {
                // Proportional SGD split, rate-normalised: total movement over the
                // two points is 2 plain steps, exactly as stock. (128-wp)/64 and
                // wp/64 ⇒ multiply then shift by rate+6. wp==64 reproduces stock.
                const d0 = sar_pow2(@as(i64, g - t0) * @as(i64, 128 - wp), rate + 6);
                const d1 = sar_pow2(@as(i64, g - t1) * @as(i64, wp), rate + 6);
                self.t[lo] = @truncate(@as(u32, @bitCast(t0 + d0)));
                self.t[lo + 1] = @truncate(@as(u32, @bitCast(t1 + d1)));
            } else if (comptime APM_WUPD == 2) {
                // Hard assignment: only the nearer point learns, at 2x so the total
                // movement still matches stock.
                if (wp < 64) {
                    self.t[lo] = @truncate(@as(u32, @bitCast(t0 + sar_pow2(@as(i64, g - t0) * 128, rate + 6))));
                } else {
                    self.t[lo + 1] = @truncate(@as(u32, @bitCast(t1 + sar_pow2(@as(i64, g - t1) * 128, rate + 6))));
                }
            } else if (comptime APM_WUPD == 4) {
                // WIDER kernel — the arm the 1 MB dose-response points at. Arms 1-3 all
                // REMOVE cross-cell smoothing and all read adverse, monotone in how much
                // they remove; so the informative direction is to ADD it. Four points,
                // total movement held at stock's 2 steps: inner 96/128 each, outer 32/128
                // each. Clamped to this context's own 33-point row.
                const c = self.cprev;
                const d0 = sar_pow2(@as(i64, g - t0) * 96, rate + 7);
                const d1 = sar_pow2(@as(i64, g - t1) * 96, rate + 7);
                self.t[lo] = @truncate(@as(u32, @bitCast(t0 + d0)));
                self.t[lo + 1] = @truncate(@as(u32, @bitCast(t1 + d1)));
                if (c >= 1) {
                    const tm: i32 = @as(i32, self.t[lo - 1]);
                    self.t[lo - 1] = @truncate(@as(u32, @bitCast(tm + sar_pow2(@as(i64, g - tm) * 32, rate + 7))));
                }
                if (c + 2 <= 32) {
                    const tp: i32 = @as(i32, self.t[lo + 2]);
                    self.t[lo + 2] = @truncate(@as(u32, @bitCast(tp + sar_pow2(@as(i64, g - tp) * 32, rate + 7))));
                }
            } else {
                // Shelwien form (src/sse.zig:sseUpdate): move the INTERPOLATED value
                // toward g by one plain step and redistribute preserving the gap, so
                // the curve translates instead of contracting.
                const dC: i32 = t0 - t1;
                const v: i32 = (t0 * (128 - wp) + t1 * wp) >> 7;
                const vn: i32 = v + sar_pow2(@as(i64, g - v), rate);
                const n0 = vn + ((dC * wp) >> 7);
                const n1 = vn - ((dC * (128 - wp)) >> 7);
                self.t[lo] = @truncate(@as(u32, @bitCast(n0)));
                self.t[lo + 1] = @truncate(@as(u32, @bitCast(n1)));
            }
        }
        const w: i32 = pr & 127;
        if (comptime APM_WUPD != 0) self.wprev = w;
        if (comptime APM_WUPD == 4) self.cprev = @intCast((pr + 2048) >> 7);
        if (comptime APM_DIAG) {
            self.sat[0] += 1;
            if (pr <= -2047) self.sat[1] += 1;
            if (pr >= 2047) self.sat[2] += 1;
            if (pr_in == 0) self.sat[3] += 1;
            if (pr_in == 4095) self.sat[4] += 1;
            const cell: usize = @intCast((pr + 2048) >> 7);
            self.hist[cell] += 1;
            self.wsum[cell] += @intCast(w);
        }
        self.index = @as(usize, @intCast((pr + 2048) >> 7)) + @as(usize, cxt & self.mask) * 33;
        return clp1((@as(i32, self.t[self.index]) * (128 - w) + @as(i32, self.t[self.index + 1]) * w) >> 11);
    }

    /// sm34: prefetch the 33-entry row `cxt` selects. `p` reads t[(cxt&mask)*33
    /// + off] where the intra-row `off` (0..32) comes from the serial pr input but
    /// the CACHE LINE is fixed by cxt alone — so issuing this for all 5 big APMs
    /// (17/8.6/8.6/4.3/4.3 MB, read per bit, unprefetched) before the serial chain
    /// overlaps ~5 DRAM misses into one. Pure cache hint, output-neutral.
    fn prefetchRow(self: *const ApmV26, cxt: u32) void {
        const base = @as(usize, cxt & self.mask) * 33;
        @prefetch(&self.t[base], .{});
        @prefetch(&self.t[base + 32], .{});
    }
};

pub const FxcmV26 = struct {
    alloc: Allocator,
    tables: Tables,
    pb: *ParseByte,
    xml: XmlModel1,
    mm: MatchModel2,
    maps1: StationaryMap,
    maps2: StationaryMap,
    scm_a: [3]SmallStationaryContextMap,
    rcm_a: RunContextMap,
    dcsm: DirectStateMap,
    dcsm0: DirectStateMap,
    dcsm1: DirectStateMap,
    dcsm2: DirectStateMap,
    dcsm_n: DirectStateMap,
    cm_c2: [21]ContextMap3,
    cm_c: [6]ContextMap3,
    cm_c44: ContextMap3,
    cmcr: [9]ContextMap3,
    cmcr2: [4]ContextMap3,
    // C-SPILL (`-Dcmc2-spill`): ONE shared cold-tail side-table every ContextMap3
    // above points at (each with a distinct salt). Empty `.{}` unless the flag is
    // on; never touched when SPILL is comptime-false.
    spill: Spill,
    cm_c4: [9]ContextMap4,
    cm_cr: [3]ContextMap4,
    mx_a: [18]Mixer1,
    mx_a1: [6]Mixer1,
    mx_a2: [4]Mixer1,
    mmm_o: [4]Mix,
    apm_a0: ApmV26,
    apm_a1: ApmV26,
    apm_a2: ApmV26,
    apm_a3: ApmV26,
    apm_a4: ApmV26,
    apm_a5: ApmV26,
    // shared bit state
    x: X,
    blpos: u32,
    pr: i32,
    fails: u32,
    failz: u32,
    failcount: u32,
    sscmrate: i32,
    rate: i32,
    mstate: u8,
    is_match_v: u32,
    xml_s: i32,
    xword1: u32,
    wrtcxt: i32,
    // modelPrediction-local persistent state
    indirect2: [256]u8,
    indirect3: []u8, // 65536
    stream5b: u32,
    s5b_byte: [8]u32,
    s2_word0: [128]u32,
    ah1: u32,
    ah2: u32,
    // slots (model_predictions1) + mixer inputs
    slots: [NUM_MODELS]i16,
    slot_idx: usize,
    active: usize,
    // -Dleafgrad-fxcm: dL/draw per slot -- g_final * Gamma_slot * dv/draw --
    // written by predictor_lex.perceive BEFORE fxcm.perceive, read by every
    // ContextMap3 in the lgApply pass at the head of its next mix. Zero-sized
    // `void` when the knob is off, so the stock layout is untouched.
    lg_slot_grad: if (LGF) [NUM_MODELS]f32 else void =
        if (LGF) [_]f32{0} ** NUM_MODELS else {},
    // LAB (`-Dleafgrad-diag`): which slots this round were emitted by a
    // ContextMap3 through an ADAPTIVE cell. Rebuilt every perceive, read by the
    // coverage census on the next bit -- the achieved reach, measured not assumed.
    lg_reach: if (LGF_DIAG) [NUM_MODELS]u8 else void =
        if (LGF_DIAG) [_]u8{0} ** NUM_MODELS else {},
    // Cached DCSM context bases (per-byte-invariant; see computeDcsmBases /
    // model_prediction). Seeded in newfor byte 0's pre-first-bpos0 bits.
    dcsm_bases: [19]u32 = undefined,
    in1: []i16, // IN1_W (stock 544)
    in1_n: usize,
    in2: []i16, // 32
    in2_n: usize,
    in4: []i16, // 16
    in4_n: usize,

    /// `strt`/`sqt`/`tbl` are generated by the ROOT module (tables.zig owns the
    /// pinned-math float generators, which cannot live in this module) and are
    /// value-copied here; `tbl`'s slices stay root-allocated (never freed — the
    /// predictor is one-shot, matching the old in-place Tables.new behavior).
    pub fn new(a: Allocator, dict: ?[]const u8, strt: *const [4096]i16, sqt: *const [4096]i16, tbl: Tables) !*FxcmV26 {
        STRT_T = strt.*;
        SQT_T = sqt.*;
        table_gen_once.call(); // fill STA*/ST2_P1/RCPR_T (idempotent)
        const self = try a.create(FxcmV26);
        self.alloc = a;
        self.tables = tbl;
        self.pb = try ParseByte.new(a, &BRACKETS, &QUOTES, &FCHAR);
        if (dict) |d| {
            try self.pb.load_dict(d);
        }
        self.xml = XmlModel1.new();
        if (dict != null) {
            self.xml.cw_isbn = self.pb.cw_isbn;
            self.xml.cw_http = self.pb.cw_http;
        }

        self.mm = MatchModel2.new(a);
        try self.mm.init(0x200000 * 2 - 1, .{ 1 << 9, 1 << 19, 1 << 16 }, &STRT_T);

        self.maps1 = StationaryMap.new();
        try self.maps1.init(a, 16, 8, 8, 0, &STRT_T);
        self.maps2 = StationaryMap.new();
        try self.maps2.init(a, 16, 8, 8, 0, &STRT_T);
        self.scm_a = .{
            try SmallStationaryContextMap.new(a, 8, 8),
            try SmallStationaryContextMap.new(a, 9, 8),
            try SmallStationaryContextMap.new(a, 8, 8),
        };
        self.rcm_a = try RunContextMap.new(a, &self.tables, 4096 * 4096, 6);

        self.dcsm = DirectStateMap.new();
        try self.dcsm.init(a, 28, 5, &STA7, &STRT_T, &SQT_T, &PRE1_T);
        self.dcsm0 = DirectStateMap.new();
        try self.dcsm0.init(a, 28, 6, &STA7, &STRT_T, &SQT_T, &PRE1_T);
        self.dcsm1 = DirectStateMap.new();
        try self.dcsm1.init(a, 20, 2, &STA7, &STRT_T, &SQT_T, &PRE1_T);
        self.dcsm2 = DirectStateMap.new();
        try self.dcsm2.init(a, 26, 3, &STA7, &STRT_T, &SQT_T, &PRE1_T);
        self.dcsm_n = DirectStateMap.new();
        try self.dcsm_n.init(a, 25, 3, &STA7, &STRT_T, &SQT_T, &PRE1_T);

        for (&self.cm_c2) |*cm| cm.* = ContextMap3.new();
        self.cm_c2[0].init(a, H2_C2[0] * HV_GROW * 8 * M4K * M4K, 3, C_S[0], C_S3[0], C_S4[0], T6, 0xf0, 1, &ST2_P1, &RCPR_T, &STRT_T);
        self.cm_c2[1].init(a, H2_C2[1] * HV_GROW * 16 * M4K * M4K, 1, C_S[1], C_S3[1], C_S4[1], T6, 0xf0, 0, &ST2_P1, &RCPR_T, &STRT_T);
        self.cm_c2[2].init(a, H2_C2[2] * HV_GROW * 8 * M4K * M4K, 1, C_S[2], C_S3[2], C_S4[2], T6, 0xf0, 0, &ST2_P1, &RCPR_T, &STRT_T);
        self.cm_c2[3].init(a, H2_C2[3] * HV_GROW * 8 * M4K * M4K, 1, C_S[3], C_S3[3], C_S4[3], T6, 0xf0, 0, &ST2_P1, &RCPR_T, &STRT_T);
        self.cm_c2[4].init(a, H2_C2[4] * HV_GROW * 16 * M4K * M4K, 2, C_S[4], C_S3[4], C_S4[4], T6, 0xf0, 1, &ST2_P1, &RCPR_T, &STRT_T);
        self.cm_c2[5].init(a, H2_C2[5] * HOT_GROW * HV_GROW * 16 * M4K * M4K, 6, C_S[5], C_S3[5], C_S4[5], T6, 0xf0, 1, &ST2_P1, &RCPR_T, &STRT_T);
        self.cm_c2[6].init(a, H2_C2[6] * HOT_GROW * 64 * M4K, 1, C_S[6], C_S3[6], C_S4[6], T1, 0x00, 1, &ST2_P1, &RCPR_T, &STRT_T);
        self.cm_c2[7].init(a, H2_C2[7] * 2 * M4K * M4K, 1, C_S[7], C_S3[7], C_S4[7], T5, 0xf0, 1, &ST2_P1, &RCPR_T, &STRT_T);
        self.cm_c2[8].init(a, H2_C2[8] * 8 * M4K * M4K, 4, C_S[8], C_S3[8], C_S4[8], T4, 0x00, 1, &ST2_P1, &RCPR_T, &STRT_T);
        self.cm_c2[9].init(a, H2_C2[9] * 8 * M4K * M4K, 4, C_S[17], C_S3[17], C_S4[17], T6, 0xf0, 1, &ST2_P1, &RCPR_T, &STRT_T);
        self.cm_c2[10].init(a, H2_C2[10] * 8 * M4K * M4K, 6, C_S[18], C_S3[18], C_S4[18], T5, 0xf0, 1, &ST2_P1, &RCPR_T, &STRT_T);
        self.cm_c2[11].init(a, H2_C2[11] * 8 * M4K * M4K, 5, C_S[19], C_S3[19], C_S4[19], T5, 0xf0, 1, &ST2_P1, &RCPR_T, &STRT_T);
        self.cm_c2[12].init(a, H2_C2[12] * 8 * M4K * M4K, 2, C_S[20], C_S3[20], C_S4[20], T6, 0xf0, 1, &ST2_P1, &RCPR_T, &STRT_T);
        self.cm_c2[13].init(a, H2_C2[13] * HV_GROW * 16 * M4K * M4K, 2, C_S[21], C_S3[21], C_S4[21], T6, 0xf0, 1, &ST2_P1, &RCPR_T, &STRT_T);
        self.cm_c2[14].init(a, H2_C2[14] * 2 * M4K * M4K, 1, C_S[23], C_S3[23], C_S4[23], T6, 0xf0, 1, &ST2_P1, &RCPR_T, &STRT_T);
        self.cm_c2[15].init(a, H2_C2[15] * 8 * 64 * M4K, 1, C_S[24], C_S3[24], C_S4[24], T1, 0x00, 0, &ST2_P0, &RCPR_T, &STRT_T);
        self.cm_c2[16].init(a, H2_C2[16] * HOT_GROW * 2048 * M4K, 1, C_S[17], C_S3[17], C_S4[17], T6, 0xf0, 1, &ST2_P1, &RCPR_T, &STRT_T);
        self.cm_c2[17].init(a, H2_C2[17] * HV_GROW * 2 * M4K * M4K, 2, C_S[17], C_S3[17], C_S4[17], T6, 0xf0, 1, &ST2_P1, &RCPR_T, &STRT_T);
        self.cm_c2[18].init(a, H2_C2[18] * HOT_GROW * HV_GROW * 4 * M4K * M4K, 5, C_S[5], C_S3[5], C_S4[5], T6, 0xf0, 1, &ST2_P1, &RCPR_T, &STRT_T);
        self.cm_c2[19].init(a, H2_C2[19] * HV_GROW * 2 * M4K * M4K, 1, C_S[17], C_S3[17], C_S4[17], T6, 0xf0, 1, &ST2_P1, &RCPR_T, &STRT_T);
        self.cm_c2[20].init(a, H2_C2[20] * HV_GROW * 8 * M4K * M4K, 3, C_S[5], C_S3[5], C_S4[5], T6, 0xf0, 1, &ST2_P1, &RCPR_T, &STRT_T);

        for (&self.cm_c) |*cm| cm.* = ContextMap3.new();
        self.cm_c[0].init(a, CG_C[0] * H2_C[0] * 2 * 16 * M4K, 7, C_S[13], C_S3[13], C_S4[13], T2, 0x00, 1, &ST2_P1, &RCPR_T, &STRT_T);
        self.cm_c[1].init(a, CG_C[1] * H2_C[1] * 64 * 2 * M4K, 3, C_S[14], C_S3[14], C_S4[14], T5, 0xf0, 0, &ST2_P0, &RCPR_T, &STRT_T);
        self.cm_c[2].init(a, CG_C[2] * H2_C[2] * 2 * M4K, 2, C_S[15], C_S3[15], C_S4[15], T2, 0xf0, 0, &ST2_P0, &RCPR_T, &STRT_T);
        self.cm_c[3].init(a, CG_C[3] * H2_C[3] * 32 * M4K, 2, C_S[22], C_S3[22], C_S4[22], T2, 0x00, 1, &ST2_P1, &RCPR_T, &STRT_T);
        self.cm_c[4].init(a, CG_C[4] * H2_C[4] * 512 * M4K, 1, C_S[25], C_S3[25], C_S4[25], T1, 0xf0, 1, &ST2_P1, &RCPR_T, &STRT_T);
        self.cm_c[5].init(a, CG_C[5] * H2_C[5] * 512 * M4K, 1, C_S[26], C_S3[26], C_S4[26], T1, 0xf0, 1, &ST2_P1, &RCPR_T, &STRT_T);

        self.cm_c44 = ContextMap3.new();
        self.cm_c44.init(a, M4K * M4K, 4, C_S[13], C_S3[13], C_S4[13], T2, 0xf0, 1, &ST2_P1, &RCPR_T, &STRT_T);

        for (&self.cmcr) |*cm| cm.* = ContextMap3.new();
        self.cmcr[0].init(a, CG_CR * H2_CR[0] * M4K * M4K, 2, C_S[13], C_S3[13], C_S4[13], T2, 0xf0, 1, &ST2_P1, &RCPR_T, &STRT_T);
        for (1..9) |i| self.cmcr[i].init(a, CG_CR * H2_CR[i] * 2048 * M4K, 3, C_S[13], C_S3[13], C_S4[13], T6, 0x00, 0, &ST2_P1, &RCPR_T, &STRT_T);

        for (&self.cmcr2) |*cm| cm.* = ContextMap3.new();
        for (0..4) |i| self.cmcr2[i].init(a, H2_CR2[i] * M4K * M4K, 3, C_S[13], C_S3[13], C_S4[13], T6, 0x00, 0, &ST2_P1, &RCPR_T, &STRT_T);

        // C-SPILL: allocate the ONE shared cold-tail table and opt every ContextMap3
        // in with a distinct, init-order-stable salt (deterministic ⇒ same on encode
        // and decode). Default flag OFF ⇒ this block is comptime-gone (byte-identical).
        self.spill = .{};
        if (comptime build_options.cmc2_spill != 0) {
            self.spill = try Spill.init(a, build_options.cmc2_spill);
            var salt: u32 = 1;
            for (&self.cm_c2) |*cm| {
                cm.set_spill(&self.spill, salt);
                salt += 1;
            }
            for (&self.cm_c) |*cm| {
                cm.set_spill(&self.spill, salt);
                salt += 1;
            }
            self.cm_c44.set_spill(&self.spill, salt);
            salt += 1;
            for (&self.cmcr) |*cm| {
                cm.set_spill(&self.spill, salt);
                salt += 1;
            }
            for (&self.cmcr2) |*cm| {
                cm.set_spill(&self.spill, salt);
                salt += 1;
            }
        }

        // -Dleafgrad-fxcm: point every ContextMap3 at the per-slot gradient array
        // (and, under -Dleafgrad-diag, the reach mask). ContextMap4 has no
        // statemap at all (cm4.zig) so it is not attachable and is out of reach
        // by construction; the DirectStateMaps, the frozen st32/rcpr LUTs and
        // fxcm's own mixers/APM are likewise out of scope for this arm.
        if (comptime LGF) {
            const gsrc: [*]const f32 = &self.lg_slot_grad;
            const rmask: ?[*]u8 = if (comptime LGF_DIAG) &self.lg_reach else null;
            for (&self.cm_c2) |*cm| cm.lgAttach(gsrc, rmask);
            for (&self.cm_c) |*cm| cm.lgAttach(gsrc, rmask);
            self.cm_c44.lgAttach(gsrc, rmask);
            for (&self.cmcr) |*cm| cm.lgAttach(gsrc, rmask);
            for (&self.cmcr2) |*cm| cm.lgAttach(gsrc, rmask);
        }

        for (&self.cm_c4) |*cm| cm.* = ContextMap4.new();
        try self.cm_c4[0].init(a, 32 * M4K, 2, C_S[9], C_S3[9], C_S4[9], &STA6, 0x00, 0, &RCPR_T, &STRT_T);
        try self.cm_c4[1].init(a, 8 * 32 * M4K, 3, C_S[10], C_S3[10], C_S4[10], &STA7, 0x00, 1, &RCPR_T, &STRT_T);
        try self.cm_c4[2].init(a, 8 * 32 * M4K, 4, C_S[11], C_S3[11], C_S4[11], &STA2, 0x00, 1, &RCPR_T, &STRT_T);
        try self.cm_c4[3].init(a, 128 * M4K, 2, C_S[16], C_S3[16], C_S4[16], &STA1, 0x00, 0, &RCPR_T, &STRT_T);
        try self.cm_c4[4].init(a, 32 * M4K, 6, C_S[12], C_S3[12], C_S4[12], &STA7, 0x00, 1, &RCPR_T, &STRT_T);
        try self.cm_c4[6].init(a, 2 * 16 * M4K, 1, C_S[5], C_S3[5], C_S4[5], &STA6, 0x00, 0, &RCPR_T, &STRT_T);
        try self.cm_c4[7].init(a, 16 * M4K, 4, C_S[12], C_S3[12], C_S4[12], &STA2, 0x00, 1, &RCPR_T, &STRT_T);
        try self.cm_c4[8].init(a, 16 * M4K, 2, C_S[12], C_S3[12], C_S4[12], &STA2, 0x00, 1, &RCPR_T, &STRT_T);
        try self.cm_c4[5].init(a, 2 * M4K, 1, C_S[0], C_S3[0], C_S4[0], &STA6, 0x00, 0, &RCPR_T, &STRT_T);

        for (&self.cm_cr) |*cm| cm.* = ContextMap4.new();
        try self.cm_cr[0].init(a, 8 * 32 * M4K, 1, C_S[10], C_S3[10], C_S4[10], &STA7, 0x00, 1, &RCPR_T, &STRT_T);
        try self.cm_cr[1].init(a, 32 * M4K, 1, C_S[10], C_S3[10], C_S4[10], &STA2, 0x00, 1, &RCPR_T, &STRT_T);
        try self.cm_cr[2].init(a, 16 * 32 * M4K, 1, C_S[10], C_S3[10], C_S4[10], &STA6, 0x00, 0, &RCPR_T, &STRT_T);

        const mxa_cfg = [18][4]i32{
            .{ 0x8000, 75, 8, 14 }, .{ 6 * 256, 75, 8, 14 }, .{ 6 * 256, 30, 1, 38 }, .{ 8 * 256, 31, 1, 34 },
            .{ 6 * 256, 53, 1, 23 }, .{ 32 * 256, 79, 1, 24 }, .{ 0x4000, 75, 1, 20 }, .{ 0x4000, 55, 1, 24 },
            .{ 0x20000, 55, 1, 24 }, .{ 0x20000, 55, 1, 24 }, .{ 0x10000, 55, 1, 24 }, .{ 0x4000, 55, 1, 24 },
            .{ 32 * 256, 6, 1, 4 }, .{ 32 * 256, 6, 1, 4 }, .{ 32 * 256, 55, 1, 24 }, .{ 16 * 256, 55, 1, 24 },
            .{ 1024, 6, 1, 4 }, .{ 2048, 6, 1, 4 },
        };
        // TUNING KNOB (task #21, group 2): scale the fxcm internal-mixer error
        // gains (uperr, mxa_cfg[i][3]) — a separate constant surface from the 23
        // outer-mixer LRs (whose 100m minimum was 0.7 = -1,838). Comptime default
        // (-Dmxa-uperr-scale, ship "0.7" = -1,841@100m) BAKED by the hermetic ship;
        // ZMIX_MXA_UPERR_SCALE env-overrides in runner_lex. Rounds to i32 like the
        // C++ would. Applied to mx_a only (the 18-strong main bank; mx_a1/mx_a2
        // gains stay stock until this group shows a direction).
        const uperr_scale = envF32Fxcm("ZMIX_MXA_UPERR_SCALE", build_options.mxa_uperr_scale);
        for (&self.mx_a, 0..) |*m, i| {
            m.* = Mixer1.new(a);
            const up: i32 = @intFromFloat(@round(@as(f32, @floatFromInt(mxa_cfg[i][3])) * uperr_scale));
            m.init(@intCast(mxa_cfg[i][0]), mxa_cfg[i][1], mxa_cfg[i][2], @max(1, up));
            m.set_tx_wx(IN1_W);
        }
        const mxa1_cfg = [6][4]i32{
            .{ 8 * 7 * 4, 6, 0, 4 }, .{ 1, 6, 0, 4 }, .{ 2048, 6, 1, 4 }, .{ 2048, 6, 1, 4 }, .{ 2048, 6, 1, 4 }, .{ 2048, 6, 1, 4 },
        };
        const u1s = build_options.mxa1_uperr_scale;
        for (&self.mx_a1, 0..) |*m, i| {
            m.* = Mixer1.new(a);
            const up1: i32 = @max(1, @as(i32, @intFromFloat(@round(@as(f32, @floatFromInt(mxa1_cfg[i][3])) * u1s))));
            m.init(@intCast(mxa1_cfg[i][0]), mxa1_cfg[i][1], mxa1_cfg[i][2], up1);
            m.set_tx_wx(if (i < 5) 32 else 16);
        }
        const mxa2_cfg = [4][4]i32{ .{ 0x100, 30, 1, 14 }, .{ 0x100, 30, 1, 14 }, .{ 0x8000, 30, 1, 14 }, .{ 0x10000, 30, 1, 14 } };
        const u2s = build_options.mxa2_uperr_scale;
        for (&self.mx_a2, 0..) |*m, i| {
            m.* = Mixer1.new(a);
            const up2: i32 = @max(1, @as(i32, @intFromFloat(@round(@as(f32, @floatFromInt(mxa2_cfg[i][3])) * u2s))));
            m.init(@intCast(mxa2_cfg[i][0]), mxa2_cfg[i][1], mxa2_cfg[i][2], up2);
            m.set_tx_wx(NUM_MODELS);
        }
        for (&self.mmm_o) |*mx| mx.* = Mix.new(a, 256);

        self.apm_a0 = try ApmV26.new(a, &self.tables, 8);
        self.apm_a1 = try ApmV26.new(a, &self.tables, 16);
        self.apm_a2 = try ApmV26.new(a, &self.tables, 16);
        self.apm_a3 = try ApmV26.new(a, &self.tables, 18);
        self.apm_a4 = try ApmV26.new(a, &self.tables, 17);
        self.apm_a5 = try ApmV26.new(a, &self.tables, 17);

        self.x = X{ .c0 = 1 };
        self.blpos = 0;
        self.pr = 2048;
        self.fails = 0;
        self.failz = 0;
        self.failcount = 0;
        self.sscmrate = 0;
        self.rate = 6;
        self.mstate = 0;
        self.is_match_v = 0;
        self.xml_s = 0;
        self.xword1 = 0;
        self.wrtcxt = 0;
        self.indirect2 = [_]u8{0} ** 256;
        self.indirect3 = try a.alloc(u8, 65536);
        @memset(self.indirect3, 0);
        self.stream5b = 0;
        self.s5b_byte = [_]u32{0} ** 8;
        self.s2_word0 = [_]u32{0} ** 128;
        self.ah1 = 0;
        self.ah2 = 0x765BA55C;
        self.slots = [_]i16{0} ** NUM_MODELS;
        self.slot_idx = 0;
        self.active = 0;
        // -Dleafgrad-fxcm: allocator-created struct -> field defaults do not
        // apply; zero explicitly. All-zero gradients are a hard no-op, which is
        // what makes the pretrain pass (which never runs the mixer cascade, so
        // never writes this array) inert rather than stale.
        if (comptime LGF) self.lg_slot_grad = [_]f32{0} ** NUM_MODELS;
        if (comptime LGF_DIAG) self.lg_reach = [_]u8{0} ** NUM_MODELS;
        // Seed the DCSM bases from the freshly-constructed parse state: byte 0's
        // bits 1..7 run BEFORE the first bpos==0 (which comes at bit 8), and the
        // parse state is unchanged until then, so this matches the old per-bit
        // recompute exactly.
        self.computeDcsmBases();
        self.in1 = try a.alloc(i16, IN1_W);
        @memset(self.in1, 0);
        self.in1_n = 0;
        self.in2 = try a.alloc(i16, 32);
        @memset(self.in2, 0);
        self.in2_n = 0;
        self.in4 = try a.alloc(i16, 16);
        @memset(self.in4, 0);
        self.in4_n = 0;
        self.applySlotMask();
        return self;
    }

    /// Full teardown — the Zig equivalent of the C++ Predictor::FreeMemory
    /// (fxcmv1.cpp:5883, cmC2 tables + match hash) EXTENDED to every owned
    /// allocation, since Zig has no destructors to free the rest on scope exit.
    /// zero_alloc-backed tables munmap immediately (RSS returns to the OS).
    pub fn deinit(self: *FxcmV26) void {
        if (comptime build_options.cm3_diag) self.cm3_diag_dump();
        if (comptime APM_DIAG) self.apm_diag_dump();
        const a = self.alloc;
        for (&self.cm_c2) |*cmx| cmx.deinit();
        for (&self.cm_c) |*cmx| cmx.deinit();
        self.cm_c44.deinit();
        for (&self.cmcr) |*cmx| cmx.deinit();
        for (&self.cmcr2) |*cmx| cmx.deinit();
        if (comptime build_options.cmc2_spill != 0) self.spill.deinit();
        for (&self.cm_c4) |*cmx| cmx.deinit();
        for (&self.cm_cr) |*cmx| cmx.deinit();
        if (comptime mixer1.DIAG) {
            for (&self.mx_a, 0..) |*mx, i| mx.diagDump("mx_a", i);
            for (&self.mx_a1, 0..) |*mx, i| mx.diagDump("mx_a1", i);
            for (&self.mx_a2, 0..) |*mx, i| mx.diagDump("mx_a2", i);
        }
        for (&self.mx_a) |*mx| mx.deinit();
        for (&self.mx_a1) |*mx| mx.deinit();
        for (&self.mx_a2) |*mx| mx.deinit();
        for (&self.mmm_o) |*mx| mx.deinit();
        self.apm_a0.deinit();
        self.apm_a1.deinit();
        self.apm_a2.deinit();
        self.apm_a3.deinit();
        self.apm_a4.deinit();
        self.apm_a5.deinit();
        self.dcsm.deinit();
        self.dcsm0.deinit();
        self.dcsm1.deinit();
        self.dcsm2.deinit();
        self.dcsm_n.deinit();
        self.rcm_a.deinit(a);
        for (&self.scm_a) |*s| s.deinit(a);
        self.maps1.deinit(a);
        self.maps2.deinit(a);
        self.mm.deinit();
        self.pb.deinit();
        self.tables.deinit(a);
        a.free(self.indirect3);
        a.free(self.in1);
        a.free(self.in2);
        a.free(self.in4);
        a.destroy(self);
    }

    /// Track-2 (`-Dcm3-diag`): dump one TSV row per ContextMap3 instance to
    /// stderr. Names mirror the init layout (cm_c2[0..20]/cm_c[0..5]/cm_c44/
    /// cmcr[0..8]/cmcr2[0..3]). Called from deinit BEFORE the tables are freed.
    /// -Dapm-diag: one TSV row per (APM, cell). `ratio` is the visit-count ratio to
    /// the neighbouring cell that shares grid point `cell` — the quantity that decides
    /// whether the incumbent's unweighted 2-cell average is first-order biased.
    fn apm_diag_dump(self: *FxcmV26) void {
        if (comptime !APM_DIAG) return;
        const names = [_][]const u8{ "a0", "a1", "a2", "a3", "a4", "a5" };
        const aps = [_]*const ApmV26{ &self.apm_a0, &self.apm_a1, &self.apm_a2, &self.apm_a3, &self.apm_a4, &self.apm_a5 };
        std.debug.print("APMDIAG\tapm\tcell\tvisits\tmean_w\tratio_next\n", .{});
        for (names, aps) |nm, ap| {
            const t = @as(f64, @floatFromInt(ap.sat[0]));
            std.debug.print("APMSAT\t{s}\t{d}\t{d:.4}\t{d:.4}\t{d:.4}\t{d:.4}\n", .{
                nm, ap.sat[0],
                100.0 * @as(f64, @floatFromInt(ap.sat[1])) / t,
                100.0 * @as(f64, @floatFromInt(ap.sat[2])) / t,
                100.0 * @as(f64, @floatFromInt(ap.sat[3])) / t,
                100.0 * @as(f64, @floatFromInt(ap.sat[4])) / t,
            });
        }
        for (names, aps) |nm, ap| {
            for (0..33) |c| {
                const v = ap.hist[c];
                if (v == 0) continue;
                const mw = @as(f64, @floatFromInt(ap.wsum[c])) / @as(f64, @floatFromInt(v));
                const nxt = if (c + 1 < 33) ap.hist[c + 1] else 0;
                const hi = @max(v, nxt);
                const lo = @min(v, nxt);
                const r = if (lo == 0) -1.0 else @as(f64, @floatFromInt(hi)) / @as(f64, @floatFromInt(lo));
                std.debug.print("APMDIAG\t{s}\t{d}\t{d}\t{d:.2}\t{d:.3}\n", .{ nm, c, v, mw, r });
            }
        }
    }

    fn cm3_diag_dump(self: *FxcmV26) void {
        if (comptime !build_options.cm3_diag) return;
        std.debug.print(
            "CM3DIAG\tname\ttable_mb\tprobes\trecent0\tscan\thit%\trepl\testk\testk%\tbackfill\tsamp\tsampfalse\tfmerge%\tocc%\tfullbkt%\tocc_slots\tev1_5\tev6_31\tev32+\n",
            .{},
        );
        var buf: [24]u8 = undefined;
        for (&self.cm_c2, 0..) |*cm, i| cm.diag_dump(std.fmt.bufPrint(&buf, "cm_c2[{d}]", .{i}) catch unreachable);
        for (&self.cm_c, 0..) |*cm, i| cm.diag_dump(std.fmt.bufPrint(&buf, "cm_c[{d}]", .{i}) catch unreachable);
        self.cm_c44.diag_dump("cm_c44");
        for (&self.cmcr, 0..) |*cm, i| cm.diag_dump(std.fmt.bufPrint(&buf, "cmcr[{d}]", .{i}) catch unreachable);
        for (&self.cmcr2, 0..) |*cm, i| cm.diag_dump(std.fmt.bufPrint(&buf, "cmcr2[{d}]", .{i}) catch unreachable);
    }

    // ---------------- slot / mixer-input plumbing ----------------
    inline fn slot_push(self: *FxcmV26, v: i16) void {
        if (self.slot_idx < NUM_MODELS) {
            self.slots[self.slot_idx] = v;
            self.slot_idx += 1;
        }
    }
    inline fn add1(self: *FxcmV26, p: i32) void {
        const v: i16 = @intCast(clp(p));
        if (self.in1_n < IN1_W) {
            self.in1[self.in1_n] = v;
            self.in1_n += 1;
        }
        self.slot_push(v);
    }
    inline fn add1_internal(self: *FxcmV26, p: i32) void {
        const v: i16 = @intCast(clp(p));
        if (self.in1_n < IN1_W) {
            self.in1[self.in1_n] = v;
            self.in1_n += 1;
        }
    }
    inline fn add2(self: *FxcmV26, p: i32) void {
        const v: i16 = @intCast(clp(p));
        if (self.in2_n < 32) {
            self.in2[self.in2_n] = v;
            self.in2_n += 1;
        }
        self.slot_push(v);
    }
    inline fn add4(self: *FxcmV26, p: i32) void {
        const v: i16 = @intCast(clp(p));
        if (self.in4_n < 16) {
            self.in4[self.in4_n] = v;
            self.in4_n += 1;
        }
        self.slot_push(v);
    }
    inline fn add1_tail(self: *FxcmV26, p: i32) void {
        self.slot_push(@intCast(clp(p)));
    }
    /// Push one ALREADY-clp'd i16 into in1 + slots (the drain body for a single
    /// value): models that produce one value per mix return it and skip the
    /// emitted-buffer round-trip entirely.
    inline fn emit1(self: *FxcmV26, v: i16) void {
        if (self.in1_n < IN1_W) {
            self.in1[self.in1_n] = v;
            self.in1_n += 1;
        }
        self.slot_push(v);
    }


    // ===== -Dfxcm-slotmask (cm-wall-census, transformer-era route A) ==================
    /// Slot count contributed by each of the 52 maskable CM instances, in
    /// MIX-CALL order. cm3: cn*(3+skip2); cm4: cn*(2+skip2). Sums to 474 of the
    /// 494 layer-0 slots (the other 20: match 7, dcsm.mix 5, maps1/2 4, sscm 3,
    /// rcm 1). Derived from the init args and cross-checked against the total:
    /// 474 + 20 = 494 == the index of the bias slot.
    pub const MASK_SLOTS = slotdel_mod.MASK_SLOTS;
    pub const MASK_NAMES = slotdel_mod.MASK_NAMES;
    inline fn bitOn(comptime i: u6) bool {
        return (cm3_mod.SLOTMASK >> i) & 1 != 0;
    }
    /// Apply the drop mask. Comptime-dead (and therefore bit-identical) at the
    /// default mask of 0.
    fn applySlotMask(self: *FxcmV26) void {
        if (comptime !cm3_mod.MASK_ANY) return;
        self.cm_c2[0].masked = bitOn(0);   // 12 slots
        self.cm_c2[1].masked = bitOn(1);   // 3 slots
        self.cm_c2[2].masked = bitOn(2);   // 3 slots
        self.cm_c2[3].masked = bitOn(3);   // 3 slots
        self.cm_c2[4].masked = bitOn(4);   // 8 slots
        self.cm_c2[5].masked = bitOn(5);   // 24 slots
        self.cm_c2[6].masked = bitOn(6);   // 4 slots
        self.cm_c2[7].masked = bitOn(7);   // 4 slots
        self.cm_c2[8].masked = bitOn(8);   // 16 slots
        self.cm_c4[0].masked = bitOn(9);   // 4 slots
        self.cm_c4[1].masked = bitOn(10);   // 9 slots
        self.cm_c4[2].masked = bitOn(11);   // 12 slots
        self.cm_c4[4].masked = bitOn(12);   // 18 slots
        self.cm_c[0].masked = bitOn(13);   // 28 slots
        self.cm_c[1].masked = bitOn(14);   // 9 slots
        self.cm_c[2].masked = bitOn(15);   // 6 slots
        self.cm_c4[3].masked = bitOn(16);   // 4 slots
        self.cm_c2[9].masked = bitOn(17);   // 16 slots
        self.cm_c2[10].masked = bitOn(18);   // 24 slots
        self.cm_c2[11].masked = bitOn(19);   // 20 slots
        self.cm_c2[12].masked = bitOn(20);   // 8 slots
        self.cm_c44.masked = bitOn(21);   // 16 slots
        self.cm_c2[13].masked = bitOn(22);   // 8 slots
        self.cm_c[3].masked = bitOn(23);   // 8 slots
        self.cm_c2[14].masked = bitOn(24);   // 4 slots
        self.cm_c2[15].masked = bitOn(25);   // 3 slots
        self.cm_c[4].masked = bitOn(26);   // 4 slots
        self.cm_c[5].masked = bitOn(27);   // 4 slots
        self.cm_c2[16].masked = bitOn(28);   // 4 slots
        self.cm_c2[17].masked = bitOn(29);   // 8 slots
        self.cm_c2[19].masked = bitOn(30);   // 4 slots
        self.cm_c4[6].masked = bitOn(31);   // 2 slots
        self.cm_c4[7].masked = bitOn(32);   // 12 slots
        self.cm_c4[8].masked = bitOn(33);   // 6 slots
        self.cm_c2[18].masked = bitOn(34);   // 20 slots
        self.cm_c2[20].masked = bitOn(35);   // 12 slots
        self.cm_cr[0].masked = bitOn(36);   // 3 slots
        self.cm_cr[1].masked = bitOn(37);   // 3 slots
        self.cm_cr[2].masked = bitOn(38);   // 2 slots
        self.cmcr[0].masked = bitOn(39);   // 8 slots
        self.cmcr[1].masked = bitOn(40);   // 9 slots
        self.cmcr[2].masked = bitOn(41);   // 9 slots
        self.cmcr[3].masked = bitOn(42);   // 9 slots
        self.cmcr[4].masked = bitOn(43);   // 9 slots
        self.cmcr[5].masked = bitOn(44);   // 9 slots
        self.cmcr[6].masked = bitOn(45);   // 9 slots
        self.cmcr[7].masked = bitOn(46);   // 9 slots
        self.cmcr[8].masked = bitOn(47);   // 9 slots
        self.cmcr2[0].masked = bitOn(48);   // 9 slots
        self.cmcr2[1].masked = bitOn(49);   // 9 slots
        self.cmcr2[2].masked = bitOn(50);   // 9 slots
        self.cmcr2[3].masked = bitOn(51);   // 9 slots
        if (!slotmask_announced) {
            slotmask_announced = true;
            comptime var ns: u32 = 0;
            comptime var ni: u32 = 0;
            comptime {
                for (0..52) |i| if ((cm3_mod.SLOTMASK >> @intCast(i)) & 1 != 0) {
                    ns += MASK_SLOTS[i];
                    ni += 1;
                };
            }
            if (comptime slotdel_mod.SLOTDEL) {
                std.debug.print("fxcm-slotdelete: ENGAGED mask=0x{x} -> {d} of 52 CM instances TRULY DELETED, {d} of 474 CM slots emit NOTHING; set()/sets()/mix() return at once (no cn/cxt_mask bookkeeping), in1 544->{d}, slots 560->{d}, layer-0 590->{d}\n", .{ cm3_mod.SLOTMASK, ni, ns, IN1_W, NUM_MODELS, 590 - slotdel_mod.DEL_COLS });
            } else {
                std.debug.print("fxcm-slotmask: ENGAGED mask=0x{x} -> {d} of 52 CM instances dropped, {d} of 474 CM slots (of 494 layer-0) emit exact neutral 0; set() keeps cn/cxt_mask bookkeeping only, prefetch_c0 no-op, table never touched\n", .{ cm3_mod.SLOTMASK, ni, ns });
            }
        }
    }

    inline fn md3(self: *FxcmV26, cm: *ContextMap3, x: *const X, sink: *const Sink) i32 {
        _ = self;
        return cm.mix(x, sink);
    }
    inline fn md4(self: *FxcmV26, cm: *ContextMap4, x: *const X, sink: *const Sink) i32 {
        _ = self;
        return cm.mix(x, sink);
    }

    /// One perceived bit == fxcmv1::update+ ResetPredictions.
    pub fn perceive(self: *FxcmV26, bit: i32) void {
        // LAB: the reach mask describes the slots THIS perceive is about to emit
        // (which the next predictconsumes), so it is cleared here, before the
        // model pass, and read by the census on the following bit.
        if (comptime LGF_DIAG) self.lg_reach = [_]u8{0} ** NUM_MODELS;
        self.x.y = bit;
        // ===== updatehead =====
        self.x.c0 += self.x.c0 + bit;
        var byte: u8 = 0;
        if (self.x.c0 >= 256) {
            byte = @intCast(self.x.c0 & 0xff);
            self.x.c4 = (self.x.c4 << 8) | @as(u32, byte);
            self.x.c0 = 1;
            self.blpos += 1;
            if ((self.fails & 255) == 0) {
                for (&self.mx_a) |*m| m.elim = @max(256 + 63, m.elim + 1);
            } else {
                for (&self.mx_a) |*m| m.elim = @max(0, @min(16, m.elim - 1));
            }
            self.sscmrate = @intFromBool(self.blpos > 14 * 256 * 1024);
            self.rate = 6 + @as(i32, @intFromBool(self.blpos > 14 * 256 * 1024)) + @as(i32, @intFromBool(self.blpos > 28 * 512 * 1024));
        }
        self.x.bpos = (self.x.bpos + 1) & 7;
        self.x.bposshift = 7 - self.x.bpos;
        self.x.c0shift_bpos = (self.x.c0 << 1) ^ (@as(i32, 256) >> @as(u5, @intCast(self.x.bposshift)));
        self.x.cm_bit_state = gsbl(self.x.bpos, self.x.c0);
        // mixer updates (train on the previous bit's inputs)
        for (0..16) |i| self.mx_a[i].update(bit, self.in1);
        if (self.xml.is_xml) self.mx_a[16].update(bit, self.in1);
        self.mx_a[17].update(bit, self.in1);
        for (0..5) |i| self.mx_a1[i].update(bit, self.in2);
        self.mx_a1[5].update(bit, self.in4);
        for (0..4) |i| self.mx_a2[i].update(bit, self.slots[0..]);
        // clear mixer inputs
        @memset(self.in1, 0);
        self.in1_n = 0;
        @memset(self.in2, 0);
        self.in2_n = 0;
        @memset(self.in4, 0);
        self.in4_n = 0;
        // paq8hp12 fails bookkeeping
        if (self.fails & 0x00000080 != 0) self.failcount = self.failcount -% 1;
        self.fails = self.fails *% 2;
        self.failz = self.failz *% 2;
        var pr = self.pr;
        if (bit != 0) pr = 4095 - pr;
        if (pr >= E_L[@intCast(self.x.bpos)]) {
            self.fails = self.fails +% 1;
            self.failcount = self.failcount +% 1;
        }
        if (pr >= 848) self.failz = self.failz +% 1;
        // ===== pr = modelPrediction=====
        pr = self.model_prediction(byte);
        self.add1_tail(@as(i32, self.tables.stretch(pr)));
        // ===== APM chain — slots 519..524 =====
        const y = bit;
        const c0 = self.x.c0;
        const c0u: u32 = @bitCast(c0);
        var pu = (self.apm_a0.p(&self.tables, pr, c0u, 3, y) + 7 * pr + 4) >> 3;
        // sm34: hoist the 5 big-APM contexts and prefetch their rows before the
        // serial APM chain (each read per bit from a 4-17 MB table, unprefetched).
        // The `else`/if of apm_a4 differs only in its context, captured by acx4.
        const acx3 = shl(c0u, 1) ^ self.ah1;
        const acx1 = shl(c0u, 3) ^ hash3(29, self.failz & 2047, 0xffffffff);
        const acx4 = if (self.fails & 255 != 0)
            hash3(c0u, self.pb.stream2b & 0xfffc, self.pb.stream3b_r & 0x1ff)
        else
            hash3(c0u, (self.pb.stream2b_r & 0xfffc) +% 0x10000, self.pb.stream3b_r & 0x1ff);
        const acx2 = shl(c0u, 5) ^ self.ah2;
        self.apm_a3.prefetchRow(acx3);
        self.apm_a1.prefetchRow(acx1);
        self.apm_a4.prefetchRow(acx4);
        self.apm_a2.prefetchRow(acx2);
        var pz: i32 = @as(i32, @bitCast(self.failcount)) + 1;
        pz += TRI[@intCast((self.fails >> 5) & 3)];
        pz += TRJ[@intCast((self.fails >> 3) & 3)];
        pz += TRJ[@intCast((self.fails >> 1) & 3)];
        if (self.fails & 1 != 0) pz += 8;
        pz = @divTrunc(pz, 2);
        const acx5 = shl(c0u, 2) ^ hash3(@as(u32, @bitCast(@min(@as(i32, 9), pz))), self.pb.x5 & 0x80ff, 0xffffffff);
        self.apm_a5.prefetchRow(acx5);
        pu = self.apm_a3.p(&self.tables, pu, acx3, self.rate, y);
        self.add1_tail(@as(i32, self.tables.stretch(pu)));
        var pv = self.apm_a1.p(&self.tables, pr, acx1, self.rate + 1, y);
        self.add1_tail(@as(i32, self.tables.stretch(pv)));
        pv = self.apm_a4.p(&self.tables, pv, acx4, self.rate, y);
        self.add1_tail(@as(i32, self.tables.stretch(pv)));
        const pt = self.apm_a2.p(&self.tables, pr, acx2, self.rate, y);
        self.add1_tail(@as(i32, self.tables.stretch(pt)));
        const pz2 = self.apm_a5.p(&self.tables, pu, acx5, self.rate, y);
        self.add1_tail(@as(i32, self.tables.stretch(pz2)));
        if (self.fails & 255 != 0) {
            pr = (pt * 6 + pu + pv * 11 + pz2 * 14 + 31) >> 5;
        } else {
            pr = (pt * 4 + pu * 7 + pv * 12 + pz2 * 9 + 31) >> 5;
        }
        self.add1_tail(@as(i32, self.tables.stretch(pr)));
        // ===== mxA2 / mmmO tail — affects only the returned p =====
        self.mx_a2[0].cxt = @as(usize, @as(u32, @bitCast(self.pb.c1))) & 0xff;
        self.mx_a2[1].cxt = @as(usize, @as(u32, @bitCast(self.pb.c2))) & 0xff;
        self.mx_a2[2].cxt = @intCast(self.stream5b & 0x7fff);
        self.mx_a2[3].cxt = @intCast(self.pb.stream2b & 0xffff);
        // sm31 (perceive side): prefetch the 4 mx_a2 rows (n=560) before their p1s.
        inline for (0..4) |i| self.mx_a2[i].prefetchRow();
        const mp0 = self.mx_a2[0].p1(&self.tables, self.slots[0..]);
        const mp1 = self.mx_a2[1].p1(&self.tables, self.slots[0..]);
        const mp2 = self.mx_a2[2].p1(&self.tables, self.slots[0..]);
        const mp3 = self.mx_a2[3].p1(&self.tables, self.slots[0..]);
        var pu2: i32 = @as(i32, self.tables.stretch(pr));
        const ms: usize = @as(usize, self.mstate);
        self.mmm_o[0].update(y, &self.tables);
        pu2 = clp(self.mmm_o[0].pp(0, mp0, ms));
        self.mmm_o[1].update(y, &self.tables);
        pu2 = clp(self.mmm_o[1].pp(pu2, mp1, ms));
        self.mmm_o[2].update(y, &self.tables);
        pu2 = clp(self.mmm_o[2].pp(pu2, mp2, ms));
        self.mmm_o[3].update(y, &self.tables);
        pu2 = clp(self.mmm_o[3].pp(pu2, mp3, ms));
        pr = (self.tables.squash(clp(pu2)) + pr * kFinalBlendW) >> kFinalBlendS;
        self.pr = pr;
        // ResetPredictions
        self.active = self.slot_idx;
        self.slot_idx = 0;
    }

    /// Recompute the 19 DCSM context bases from the current parse state. Called
    /// once per byte (bpos==0, after parse_byte) and once in newto seed byte 0.
    /// Each base is the per-byte-invariant part of a dcsm context; model_prediction
    /// adds the per-bit `c0` to form the final context. Order matches the old
    /// dcsm/dcsm2/dcsm1/dcsm0/dcsm_n layout.
    fn computeDcsmBases(self: *FxcmV26) void {
        const w0 = self.pb.word0 *% 191;
        const w00 = self.pb.word00 *% 191;
        const ibb = self.pb.indirect_br_byte;
        const h = self.pb.h;
        const fcc: u32 = @as(u32, self.pb.fccxt.cxt);
        self.dcsm_bases = .{
            w0 *% 256,
            (w0 +% self.pb.worcxt.word(1)) *% 256,
            (w0 +% self.pb.worcxt.word(2)) *% 256,
            (w0 +% self.pb.worcxt.word(3)) *% 256,
            (w0 +% self.pb.worcxt.word(4)) *% 256,
            (w00 +% ibb) *% 256,
            (w00 +% self.pb.worcxt0.word(1) +% ibb) *% 256,
            (w00 +% self.pb.worcxt0.word(2) +% ibb) *% 256,
            ibb *% 256,
            self.pb.cxtind3 *% 191,
            (h +% self.pb.worcxt1.word(1)) *% 256,
            (h +% self.pb.worcxt1.word(2)) *% 256,
            (h +% self.pb.worcxt1.word(3)) *% 256,
            (h +% self.pb.worcxt1.word(4)) *% 256,
            (h +% self.pb.worcxt1.word(5)) *% 256,
            (h +% self.pb.worcxt1.word(6)) *% 256,
            (h +% self.pb.worcxt.code(1) +% fcc) *% 256,
            (h +% self.pb.worcxt.code(2) +% fcc) *% 256,
            (h +% self.pb.worcxt.code(3) +% fcc) *% 256,
        };
    }

    /// fxcmv1::modelPrediction.
    fn model_prediction(self: *FxcmV26, byte: u8) i32 {
        // Direct mixer-input sink (r3o surgery): models push each prediction
        // straight into in1 + slots where they used to buffer + drain.
        comptime std.debug.assert(emit_sink.SLOT_CAP == NUM_MODELS);
        const sink = Sink{ .in1 = self.in1.ptr, .in1_n = &self.in1_n, .slots = &self.slots, .slot_n = &self.slot_idx };
        const xc = self.x;
        const bpos = xc.bpos;
        const c4 = xc.c4;
        const c0 = xc.c0;
        if (bpos == 0) {
            self.pb.is_match = self.is_match_v;
            self.pb.parse_byte(byte);
            if (self.pb.page_reset) {
                // fxcmv1.cpp 4673-4681: `</page>` wipes the per-article maps
                // (fires per article in dict mode; not observed in no-dict).
                self.cm_c4[1].reset();
                self.cm_c4[2].reset();
                self.cm_c4[4].reset();
                self.cm_c4[7].reset();
                self.cm_c4[8].reset();
                self.cm_c4[6].reset();
                self.cm_c[0].reset();
                if (self.pb.page_reset_cm1) self.cm_c[1].reset();
            }
        }
        if (bpos == 2 or bpos == 5) {
            const c0u: u32 = @bitCast(c0);
            for (&self.cm_c2) |*cm| cm.prefetch_c0(c0u);
            for (&self.cm_c) |*cm| cm.prefetch_c0(c0u);
            self.cm_c44.prefetch_c0(c0u);
            for (&self.cmcr) |*cm| cm.prefetch_c0(c0u);
            for (&self.cmcr2) |*cm| cm.prefetch_c0(c0u);
            for (&self.cm_c4) |*cm| cm.prefetch_c0(c0u);
            for (&self.cm_cr) |*cm| cm.prefetch_c0(c0u);
        }
        // xmlS = xml.p
        var hist = [_]u8{0} ** 10;
        if (bpos == 0) {
            var q: u32 = 1;
            while (q < 10) : (q += 1) hist[q] = self.pb.raw_buf(q);
        }
        self.xml_s = self.xml.p(@intCast(bpos), self.pb.c4, self.pb.blpos, &hist, self.pb.last_cw);

        if (bpos == 0) cmcold.bcs.byteContextSets(self, c4);

        // ===== per-bit dcsm sets =====
        // The 19 DCSM context BASES depend only on per-byte parse state
        // (word0/word00/h/cxtind3/indirect_br_byte/worcxt*/fccxt — all mutated
        // ONLY by parse_byte, which runs at bpos==0), so they are invariant
        // across the 8 bits of a byte. Cache them once per byte (computeDcsmBases,
        // seeded in newfor byte 0's pre-bpos0 bits) and add the per-bit c0
        // offset. `base +% c0u` is a wrapping add == the old inline expression:
        // bit-identical, but drops ~15 WordsContext accessor calls (each a size
        // load + bounds branch + data load) and ~38 muls for 7 of every 8 bits.
        if (bpos == 0) self.computeDcsmBases();
        const c0u: u32 = @bitCast(c0);
        const db = &self.dcsm_bases;
        const dcsm_cx = [5]u32{ db[0] +% c0u, db[1] +% c0u, db[2] +% c0u, db[3] +% c0u, db[4] +% c0u };
        const dcsm2_cx = [3]u32{ db[5] +% c0u, db[6] +% c0u, db[7] +% c0u };
        const dcsm1_cx = [2]u32{ db[8] +% c0u, db[9] +% c0u };
        var dcsm0_cx = [_]u32{0} ** 6;
        for (0..6) |q| dcsm0_cx[q] = db[10 + q] +% c0u;
        var dcsm_n_cx = [_]u32{0} ** 3;
        for (0..3) |q| dcsm_n_cx[q] = db[16 + q] +% c0u;
        // sm33: prefetch each dcsm slot's carried-over (previous-bit) context state
        // before its setreads it — issued first for max lead time (these depend
        // only on prior-bit state, unlike the NEW-context prefetches below).
        self.dcsm.prefetchPrev();
        self.dcsm2.prefetchPrev();
        self.dcsm1.prefetchPrev();
        self.dcsm0.prefetchPrev();
        self.dcsm_n.prefetchPrev();
        for (dcsm_cx) |v| self.dcsm.prefetch(v);
        for (dcsm2_cx) |v| self.dcsm2.prefetch(v);
        for (dcsm1_cx) |v| self.dcsm1.prefetch(v);
        for (dcsm0_cx) |v| self.dcsm0.prefetch(v);
        for (dcsm_n_cx) |v| self.dcsm_n.prefetch(v);

        for (dcsm_cx) |cx| self.add1_internal(@as(i32, self.dcsm.set(cx, &xc)));
        for (dcsm2_cx) |cx| self.add1_internal(@as(i32, self.dcsm2.set(cx, &xc)));
        for (dcsm1_cx) |cx| self.add1_internal(@as(i32, self.dcsm1.set(cx, &xc)));
        for (dcsm0_cx) |cx| self.add1_internal(@as(i32, self.dcsm0.set(cx, &xc)));
        for (dcsm_n_cx) |cx| self.add1_internal(@as(i32, self.dcsm_n.set(cx, &xc)));

        // ===== mixes =====
        const m1 = self.maps1.mix(&xc);
        self.emit1(m1[0]);
        self.emit1(m1[1]);
        const m2 = self.maps2.mix(&xc);
        self.emit1(m2[0]);
        self.emit1(m2[1]);
        for (0..3) |i| {
            const r = self.scm_a[i].mix(&self.tables, xc.y, self.sscmrate);
            self.add1(r[0]);
            self.add1_internal(r[1]);
        }
        if (bpos == 0) {
            self.mm.parse_byte(@intCast(c4 & 0xff));
            self.mm.c1 = @truncate(@as(u32, @bitCast(self.pb.c1)));
            self.mm.set_order_hashes(self.pb.t[5], self.pb.t[7], self.pb.t[9]);
        }
        self.is_match_v = self.mm.mix(&xc, self.pb.worcxt.word(1), &sink);

        // Order X
        var ord_x: i32 = 0;
        if (self.cm_c2[0].cxt_mask_val() != 0) ord_x = 2;
        ord_x += self.md3(&self.cm_c2[0], &xc, &sink);
        if (ord_x == 3) ord_x = 2;
        ord_x += self.md3(&self.cm_c2[1], &xc, &sink);
        ord_x += self.md3(&self.cm_c2[2], &xc, &sink);
        ord_x += self.md3(&self.cm_c2[3], &xc, &sink);
        var ord_w = self.md3(&self.cm_c2[4], &xc, &sink);
        ord_w += self.md3(&self.cm_c2[5], &xc, &sink);
        if (ord_w > 3) ord_w = 3;
        _ = self.md3(&self.cm_c2[6], &xc, &sink);
        _ = self.md3(&self.cm_c2[7], &xc, &sink);
        _ = self.md3(&self.cm_c2[8], &xc, &sink);
        _ = self.md4(&self.cm_c4[0], &xc, &sink);
        _ = self.md4(&self.cm_c4[1], &xc, &sink);
        _ = self.md4(&self.cm_c4[2], &xc, &sink);
        _ = self.md4(&self.cm_c4[4], &xc, &sink);
        _ = self.md3(&self.cm_c[0], &xc, &sink);
        _ = self.md3(&self.cm_c[1], &xc, &sink);
        _ = self.md3(&self.cm_c[2], &xc, &sink);
        _ = self.md4(&self.cm_c4[3], &xc, &sink);
        _ = self.md3(&self.cm_c2[9], &xc, &sink);
        _ = self.md3(&self.cm_c2[10], &xc, &sink);
        _ = self.md3(&self.cm_c2[11], &xc, &sink);
        _ = self.md3(&self.cm_c2[12], &xc, &sink);
        _ = self.md3(&self.cm_c44, &xc, &sink);
        // Order Word
        ord_w += self.md3(&self.cm_c2[13], &xc, &sink);
        _ = self.md3(&self.cm_c[3], &xc, &sink);
        ord_w += self.md3(&self.cm_c2[14], &xc, &sink);
        _ = self.md3(&self.cm_c2[15], &xc, &sink);
        _ = self.md3(&self.cm_c[4], &xc, &sink);
        _ = self.md3(&self.cm_c[5], &xc, &sink);
        _ = self.md3(&self.cm_c2[16], &xc, &sink);
        _ = self.md3(&self.cm_c2[17], &xc, &sink);
        _ = self.md3(&self.cm_c2[19], &xc, &sink);
        _ = self.md4(&self.cm_c4[6], &xc, &sink);
        _ = self.md4(&self.cm_c4[7], &xc, &sink);
        _ = self.md4(&self.cm_c4[8], &xc, &sink);
        _ = self.md3(&self.cm_c2[18], &xc, &sink);
        _ = self.md3(&self.cm_c2[20], &xc, &sink);
        _ = self.md4(&self.cm_cr[0], &xc, &sink);
        _ = self.md4(&self.cm_cr[1], &xc, &sink);
        _ = self.md4(&self.cm_cr[2], &xc, &sink);
        for (0..9) |i| _ = self.md3(&self.cmcr[i], &xc, &sink);
        for (0..4) |i| _ = self.md3(&self.cmcr2[i], &xc, &sink);
        self.add1(self.rcm_a.predict(xc.c0shift_bpos, xc.bposshift));
        self.emit1(self.dcsm.mix());
        self.emit1(self.dcsm0.mix());
        self.emit1(self.dcsm1.mix());
        self.emit1(self.dcsm2.mix());
        self.emit1(self.dcsm_n.mix());

        self.mstate = STA2[@as(usize, self.mstate) * 4 + @as(usize, @intCast(self.x.y))];

        // bias (slot 494)
        self.add1_tail(64);

        // ===== mixer contexts =====
        const c0b = c0 << @as(u5, @intCast(8 - bpos));
        const stream2b = self.pb.stream2b;
        const stream2b_r = self.pb.stream2b_r;
        const stream3b = self.pb.stream3b;
        const stream3b_r = self.pb.stream3b_r;
        const words: u32 = @as(u32, self.pb.words);
        const numbers: u32 = @as(u32, self.pb.numbers);
        const brfc = self.pb.brfc_idx;
        const fcidx = self.pb.fc_idx;

        {
            const vv = ((@as(u32, @intCast(ord_x)) *% 8 +% @as(u32, @intFromBool(brfc != 0)) *% 4 +% (stream2b & 3)) *% 2) +% (words & 1);
            self.mx_a1[0].cxt = @min(@as(usize, vv), 8 * 7 * 4 - 1);
        }
        self.mx_a1[2].cxt = @intCast(stream2b & 0x3f);
        self.mx_a1[3].cxt = @intCast((stream2b & 3) *% 4 +% @as(u32, WRT_2B[@intCast(c0b & 255)]));
        self.mx_a1[4].cxt = @intCast(@as(u32, @intCast(ord_x)) *% 4 +% @as(u32, WRT_2B[@intCast(c0b & 255)]));
        self.mx_a[17].cxt = @intCast(self.pb.pstate *% 16 +% (stream2b & 15));
        // mixer 0
        var c: i32 = 0;
        if (bpos == 0) {
            self.mx_a[0].cxt = @intCast((stream2b_r & 4095) *% 8 +% (stream3b & 7));
        } else if (bpos > 3) {
            c = @as(i32, WRT_2B[@intCast(c0b & 255)]);
            self.mx_a[0].cxt = @intCast(((shl(stream2b, 2) & 4095) +% @as(u32, @intCast(c))) *% 8 +% brfc);
        } else {
            self.mx_a[0].cxt = @intCast((stream2b & 4095) *% 8 +% brfc);
        }
        // mixer 1
        if (bpos != 0) {
            c = c0b;
            if (bpos == 1) {
                c += 16 * @as(i32, @intCast((words *% 2) & 4));
            } else if (bpos > 3) {
                c = @as(i32, WRT_2B[@intCast(c0b & 255)]) * 64;
            }
            c = @min(bpos, 5) * 256 + @as(i32, @intCast(stream3b_r & 7)) + @as(i32, @intCast(fcidx)) * 8 + (c & 192);
        } else {
            c = @intCast((words & 12) *% 16 +% (stream3b_r & 7) +% brfc *% 8);
        }
        self.mx_a[1].cxt = @intCast(c);
        // mixer 2
        self.mx_a[2].cxt = @intCast(@as(i32, @intCast((4 *% words) & 0xf0)) + ord_x * 256 + @as(i32, @intCast(stream2b & 15)));
        // mixer 6
        if (bpos > 3) {
            c = @as(i32, WRT_2B[@intCast(c0b & 255)]);
            self.mx_a[6].cxt = @as(usize, @intCast((stream3b_r & 0xff8) *% 4 +% ((stream2b & 3) *% 4))) + @as(usize, @intCast(c));
        } else if (bpos == 0) {
            self.mx_a[6].cxt = @intCast((stream3b_r & 0xff8) *% 4 +% (4 *% fcidx) +% (stream2b & 3));
        } else {
            self.mx_a[6].cxt = @intCast((stream3b_r & 0xff8) *% 4 +% ((2 *% words) & 0x1c) +% (stream2b & 3));
        }
        c = c0b;
        // mixer 3
        self.mx_a[3].cxt = @intCast(bpos * 256 + @as(i32, @intCast(((((numbers | words) << @as(u5, @intCast(bpos))) & 255) >> @as(u5, @intCast(bpos))) | (@as(u32, @bitCast(c)) & 255))));
        // mixer 4
        if (bpos != 0) {
            if (bpos == 1) {
                c += 16 * @as(i32, @intCast(stream3b & 7));
            } else if (bpos == 2) {
                c += 16 * @as(i32, @intCast(stream2b & 3));
            } else if (bpos == 3) {
                c += 16 * @as(i32, @intCast(words & 1));
            } else {
                c = bpos + (c & 0xf0);
            }
            if (bpos < 5) {
                c = bpos + (c & 0xf0);
            }
        } else {
            c = 16 * @as(i32, @intCast(stream2b & 0xf));
        }
        var ord_x2 = ord_x - 1;
        if (ord_x2 < 0) ord_x2 = 0;
        if (self.is_match_v != 0) ord_x2 += 1;
        if (ord_x2 > 5) ord_x2 = 5;
        self.mx_a[4].cxt = @intCast(c + ord_x2 * 256 + 8 * self.pb.is_paragraph);
        // mixer 5
        self.mx_a[5].cxt = @intCast((ord_w * 256 + @as(i32, @intCast(stream2b & 0xf0)) + @as(i32, @intCast((stream3b & 0x38) >> 2))) * 4 + @as(i32, @intCast(fcidx)));
        // mixer 7
        if (bpos > 2) {
            self.mx_a[7].cxt = @intCast(((stream3b & 7) *% 8 +% @as(u32, WRT_3B[@intCast(c0b & 255)])) *% 256 +% brfc *% 32 +% (words & 7) *% 4 +% @as(u32, @bitCast(self.pb.is_paragraph)) +% (if (self.is_match_v != 0) @as(u32, 2) else 0));
        } else {
            self.mx_a[7].cxt = @intCast(((stream3b & 63) *% 256 +% brfc *% 16 +% (words & 7) *% 2 +% @as(u32, @bitCast(self.pb.is_paragraph))) | (if (self.is_match_v != 0) @as(u32, 128) else 0));
        }
        self.wrtcxt = self.pb.deccode;
        self.mx_a[8].cxt = @intCast(self.pb.deccode);
        self.mx_a[9].cxt = @intCast((stream3b_r & 511) *% 16 *% 16 +% fcidx *% 32 +% @as(u32, @bitCast(self.pb.is_paragraph)) *% 16 +% (self.pb.last_wt & 15));
        self.mx_a[10].cxt = @intCast(c4 & 0xffff);
        self.mx_a[11].cxt = @as(usize, @intCast((stream3b & 0x3f) *% 256)) + @as(usize, @intCast(c0));
        self.mx_a[12].cxt = @intCast(self.pb.oldwt1 +% (stream2b_r & 255) *% 32);
        self.mx_a[13].cxt = @intCast(stream3b_r & 511);
        self.mx_a[14].cxt = @intCast((brfc *% 8 +% fcidx) *% 8 +% @as(u32, WRT_3B[@intCast(c0b & 255)]));
        self.mx_a[15].cxt = @intCast((numbers | words) *% 16 +% (stream2b_r & 15));
        self.mx_a[16].cxt = @intCast(self.xml_s & 1023);

        // sm31: prefetch all 18 mx_a weight rows (contexts fully resolved above)
        // before the p1 cascade — each is a 544-wide i16 row in a multi-hundred-MB
        // bank that misses to DRAM, and the outer sm26 prefetch never covered the
        // fxcm-internal mixers. Latency-bound workload → the miss hides behind the
        // earlier mixers' dot compute. Pure cache hint, bit-identical.
        inline for (0..18) |i| self.mx_a[i].prefetchRow();

        // ===== p1 chain — slots 495..517 =====
        for (0..16) |i| {
            self.add2(self.mx_a[i].p1(&self.tables, self.in1));
        }
        if (self.xml.is_xml) {
            self.add2(self.mx_a[16].p1(&self.tables, self.in1));
        } else {
            self.add2(0);
        }
        self.add4(self.mx_a[17].p1(&self.tables, self.in1));
        for (0..5) |i| {
            self.add4(self.mx_a1[i].p1(&self.tables, self.in2));
        }
        self.mx_a1[5].cxt = 0;
        return self.mx_a1[5].p(&self.tables, self.in4);
    }


    /// Construct with the WRT dictionary loaded.
    pub fn new_with_dict(a: Allocator, dict: []const u8, strt: *const [4096]i16, sqt: *const [4096]i16, tbl: Tables) !*FxcmV26 {
        return FxcmV26.new(a, dict, strt, sqt, tbl);
    }

    pub fn wrtcxt_val(self: *const FxcmV26) u64 {
        return @bitCast(@as(i64, self.wrtcxt));
    }
};

fn parse_raw(a: Allocator, text: []const u8) ![]u64 {
    var list = std.ArrayListUnmanaged(u64){};
    var lines = std.mem.tokenizeScalar(u8, text, '\n');
    while (lines.next()) |line| {
        var toks = std.mem.tokenizeAny(u8, line, " \t\r");
        _ = toks.next() orelse continue;
        const hx = toks.next() orelse continue;
        const v = try std.fmt.parseInt(u64, hx, 16);
        try list.append(a, v);
    }
    return list.toOwnedSlice(a);
}

test "fxcm_v26_nodict_slot_gate" {
    var arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const stream: []const u8 = @embedFile("goldens/stream.bin");
    const want = try parse_raw(a, @embedFile("goldens/pcs_nodict.raw"));
    try std.testing.expectEqual(NUM_MODELS, want.len);
    const t_strt = loadI16("goldens/dsm_STRT.bin", 4096);
    const t_sqt = loadI16("goldens/dsm_SQT.bin", 4096);
    var f = try FxcmV26.new(a, null, &t_strt, &t_sqt, try table_gen.newFromGoldensForTests(a));
    var cs = [_]u64{0} ** NUM_MODELS;
    var total: u64 = 0;
    for (stream) |byte| {
        var b: i32 = 7;
        while (b >= 0) : (b -= 1) {
            f.perceive(@as(i32, (byte >> @as(u3, @intCast(b))) & 1));
            for (0..NUM_MODELS) |i| {
                const v: u32 = if (i < f.active) @as(u32, @as(u16, @bitCast(f.slots[i]))) else 0;
                cs[i] = cs[i] *% 1000003 +% @as(u64, v);
                total = total *% 1000003 +% @as(u64, v);
            }
        }
    }
    var bad: usize = 0;
    var first_bad: usize = NUM_MODELS;
    for (0..NUM_MODELS) |i| {
        if (cs[i] != want[i]) {
            bad += 1;
            if (first_bad == NUM_MODELS) first_bad = i;
        }
    }
    if (bad != 0) {
        std.debug.print("{d} slots diverge; first bad slot {d} (see MODULE_CHECKSUM_MAP.txt)\n", .{ bad, first_bad });
        return error.SlotsDiverge;
    }
    try std.testing.expectEqual(@as(u64, 0x91521e61b9ea0014), total);
}

test "fxcm_v26_dict_slot_gate" {
    var arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const stream: []const u8 = @embedFile("goldens/stream.bin");
    const dict: []const u8 = @embedFile("goldens/english.dic");
    const want = try parse_raw(a, @embedFile("goldens/pcs_dict_v2.raw"));
    try std.testing.expectEqual(NUM_MODELS, want.len);
    const t_strt = loadI16("goldens/dsm_STRT.bin", 4096);
    const t_sqt = loadI16("goldens/dsm_SQT.bin", 4096);
    var f = try FxcmV26.new_with_dict(a, dict, &t_strt, &t_sqt, try table_gen.newFromGoldensForTests(a));
    var cs = [_]u64{0} ** NUM_MODELS;
    var total: u64 = 0;
    for (stream) |byte| {
        var b: i32 = 7;
        while (b >= 0) : (b -= 1) {
            f.perceive(@as(i32, (byte >> @as(u3, @intCast(b))) & 1));
            for (0..NUM_MODELS) |i| {
                const v: u32 = if (i < f.active) @as(u32, @as(u16, @bitCast(f.slots[i]))) else 0;
                cs[i] = cs[i] *% 1000003 +% @as(u64, v);
                total = total *% 1000003 +% @as(u64, v);
            }
        }
    }
    var bad: usize = 0;
    var first_bad: usize = NUM_MODELS;
    for (0..NUM_MODELS) |i| {
        if (cs[i] != want[i]) {
            bad += 1;
            if (first_bad == NUM_MODELS) first_bad = i;
        }
    }
    if (bad != 0) {
        std.debug.print("{d} slots diverge; first bad slot {d} (see MODULE_CHECKSUM_MAP.txt)\n", .{ bad, first_bad });
        return error.SlotsDiverge;
    }
    try std.testing.expectEqual(@as(u64, 0x6f3bcb21f6239d0f), total);
}

// Golden parity for the startup-regenerated tables: `generateTables` output
// must be byte-identical to the captured C++ oracle dumps. The goldens are
// referenced only from this test-only decl scope, so they stay OUT of the
// shipping binary's .rodata.
test "fxcm_v26_regenerated_tables_match_goldens" {
    table_gen_once.call();
    // STRT_T/SQT_T are no longer generated here (the float generators live in
    // the root module's tables.zig, whose own test gates generator==golden);
    // fill them the way newdoes, from the golden dumps, so the comparisons
    // below still pin the copy path.
    STRT_T = loadI16("goldens/dsm_STRT.bin", 4096);
    SQT_T = loadI16("goldens/dsm_SQT.bin", 4096);
    try std.testing.expectEqualSlices(u8, @embedFile("goldens/cm3_STA1.bin"), &STA1);
    try std.testing.expectEqualSlices(u8, @embedFile("goldens/cm3_STA2.bin"), &STA2);
    try std.testing.expectEqualSlices(u8, @embedFile("goldens/cm3_STA4.bin"), &STA4);
    try std.testing.expectEqualSlices(u8, @embedFile("goldens/cm3_STA5.bin"), &STA5);
    try std.testing.expectEqualSlices(u8, @embedFile("goldens/cm3_STA6.bin"), &STA6);
    try std.testing.expectEqualSlices(u8, @embedFile("goldens/cm3_STA7.bin"), &STA7);
    try std.testing.expectEqualSlices(u8, @embedFile("goldens/dsm_STA7.bin"), &STA7);
    const g_strt = loadI16("goldens/cm3_STRT.bin", 4096);
    const g_strt2 = loadI16("goldens/dsm_STRT.bin", 4096);
    const g_sqt = loadI16("goldens/dsm_SQT.bin", 4096);
    const g_st2 = loadI16("goldens/cm3_ST2_P1.bin", 4096);
    const g_rcpr = loadI16("goldens/cm3_RCPR.bin", 512);
    const g_rcpr2 = loadI16("goldens/cm4_RCPR.bin", 512);
    try std.testing.expectEqualSlices(i16, &g_strt, &STRT_T);
    try std.testing.expectEqualSlices(i16, &g_strt2, &STRT_T);
    try std.testing.expectEqualSlices(i16, &g_sqt, &SQT_T);
    try std.testing.expectEqualSlices(i16, &g_st2, &ST2_P1);
    try std.testing.expectEqualSlices(i16, &g_rcpr, &RCPR_T);
    try std.testing.expectEqualSlices(i16, &g_rcpr2, &RCPR_T);
}
