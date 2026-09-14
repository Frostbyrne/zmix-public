//! cm_fast — root of the OPTIONAL ReleaseFast module holding the pure-integer
//! context-map compute (ContextMap3/ContextMap4). Built at ReleaseFast when
//! -Dcmfast=true so the hot cm3.mix/cm4.mix (13%+ of RS self-time, branchy integer
//! bit-history gather that -Oz schedules poorly) gets -O3 codegen WITHOUT the whole
//! binary paying the RF fee. cm3/cm4 are integer-only (no @setFloatMode), so RF vs
//! RS codegen is BIT-IDENTICAL — this lands on the current archive.
//!
//! The boundary is fxcm_v26 → cm3.mix, which -Oz already keeps out-of-line (big fn,
//! never inlined), so nothing is lost — unlike the rejected whole-predictor split
//! (build.zig:9-13) which broke perceive/predictinlining into the coder loop.
//!
//! X crosses the boundary only in mix/updargs; the caller @ptrCasts its own
//! (layout-identical, same byte_core.zig source + target) X pointer.
// -Dfxcm-slotmask / -Dfxcm-slotdelete comptime arithmetic. Re-exported so the ROOT
// module (predictor_lex) can size its layer-0 vector from the same single source
// that sized fxcm's in1/slots banks — the two must not be able to disagree.
pub const slotdel = @import("slotdel.zig");
pub const ContextMap3 = @import("cm3.zig").ContextMap3;
pub const Spill = @import("cm3.zig").Spill;
// -Dleafgrad-fxcm LAB: the cm3-side step-magnitude census, re-exported so the
// root module can print it at predictor deinit without pulling cm3.zig into two
// modules at once (files must belong to exactly one Zig module).
pub const LG_STAT = struct {
    pub inline fn sum() f64 { return @import("cm3.zig").LG_STAT_SUM; }
    pub inline fn n() u64 { return @import("cm3.zig").LG_STAT_N; }
    pub inline fn live() u64 { return @import("cm3.zig").LG_STAT_LIVE; }
    pub inline fn sat() u64 { return @import("cm3.zig").LG_STAT_SAT; }
};
pub const ContextMap4 = @import("cm4.zig").ContextMap4;
// dcsm: 19 set/bit + 5 mix/bit of branchy integer state-machine work — the
// same shape as cm3.mix (out-of-line under -Oz, pure integer → bit-identical).
pub const DirectStateMap = @import("direct_state_map.zig").DirectStateMap;
pub const STRT_PTR = @import("direct_state_map.zig").STRT_PTR;
// match_model: 1 branchy integer mix/bit + per-byte candidate scan; brings
// state_map1.zig (std-only) with it. Reads build_options.idr (cmfast_opts).
pub const MatchModel2 = @import("match_model.zig").MatchModel2;
// rcm/sscm: per-bit integer predict/mix; Tables crosses as a @ptrCast'd
// pointer to the layout copy in cmf_tables.zig (root module owns the instance).
pub const RunContextMap = @import("run_context_map.zig").RunContextMap;
pub const SmallStationaryContextMap = @import("sscm.zig").SmallStationaryContextMap;
pub const Tables = @import("cmcold").Tables;
pub const StationaryMap = @import("leaf_maps.zig").StationaryMap;
// The whole fxcm26 integer predictor lives here (model_prediction/perceive are
// -Oz-out-of-line below PredictorLex.predict/perceive, so the coder-loop
// inlining that build.zig:9-13 protects is untouched). Root-side float stays
// out: tables.zig generates STRT/SQT + the Tables instance and passes them in.
pub const FxcmV26 = @import("fxcm_v26.zig").FxcmV26;
pub const parse_byte = @import("cmcold").parse_byte;
pub const X = @import("cmcold").X;
pub const WordsContext = @import("words_context.zig").WordsContext;
pub const SentenceContext = @import("sentence_context.zig").SentenceContext;

test {
    // Collect the module's in-file tests (`zig build test` runs this module as
    // its own test root — root-module test compilations cannot see them).
    // sscm.zig/run_context_map.zig tests stay dark: they need the ROOT module's
    // Tables.new(float table construction), which this module cannot import;
    // leaf_maps' equivalent test lives in src/tests.zig for the same reason.
    _ = @import("slotdel.zig");
    _ = @import("cm3.zig");
    _ = @import("cm4.zig");
    _ = @import("direct_state_map.zig");
    _ = @import("match_model.zig");
    _ = @import("mixer1.zig");
    _ = @import("words_context.zig");
    _ = @import("sentence_context.zig");
    _ = @import("run_context_map.zig");
    _ = @import("sscm.zig");
    _ = @import("xml_model.zig");
    _ = @import("fxcm_v26.zig");
    // parse layer + base files: collected by cm_cold.zig's test block.
}
