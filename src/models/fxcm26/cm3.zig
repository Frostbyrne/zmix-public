//! ContextMap3 — bit-exact Zig port of Rust `cm3.rs` (itself a golden-verified
//! port of cmix-lex `fxcmv1.cpp` `struct ContextMap3`, the highest-slot-count v26
//! model). Large context map: double-size element E1<14,128>, inline `ts[256]`
//! statemap per context, built-in run model, set/sets/mix API; returns a result
//! count.
//!
//! Representation: the `E1<14,128>` bucket array `t` is a flat `[]u8`; each element
//! is 128 bytes: chk[14] u16-LE at [0..28), `last` at [28], bh[14][7] at [29..127).
//! `cp/cp0` are byte offsets into `t` (`?usize`, null == C++ null pointer); `runp`
//! is a byte offset (always valid). `&bh[i][0]` == elem_base + 29 + i*7.
//!
//! `add` pushes `clp(p)` straight into the predictor's Sink (in1 + slots) —
//! the C++ shape; the per-mix `emitted` buffer round-trip is gone (r3o surgery).
//!
//! Determinism: integer-only at runtime (wrapping ops for all u32 hashing/state);
//! the only float dependency is `strt[]` (stretch), supplied by the caller.

const std = @import("std");
const Sink = @import("emit_sink.zig").Sink;
const SLOT_CAP: usize = @import("emit_sink.zig").SLOT_CAP;
const X = @import("cmcold").X;
const zero_alloc = @import("cmcold").zero_alloc;
const build_options = @import("build_options");
const leafgrad_options = @import("leafgrad_options");
// -Dfxcm-slotmask / -Dfxcm-slotdelete: see slotdel.zig. Parsed at COMPTIME so
// `MASK_ANY == false` (the default) makes every guard below comptime-dead =>
// bit-identical stock build, zero added branches. Re-exported here because
// cm4.zig / fxcm_v26.zig already reach the mask through this file.
const slotdel = @import("slotdel.zig");
pub const SLOTMASK: u64 = slotdel.SLOTMASK;
pub const MASK_ANY: bool = slotdel.MASK_ANY;
/// -Dfxcm-slotdelete: the masked instances emit NOTHING and keep no bookkeeping.
pub const SLOTDEL: bool = slotdel.SLOTDEL;

const MAXCXT: usize = 8;

// -Dcm3-ts-rate (default 14 = shipped, bit-identical). `ts[256]` is the per-bank
// adaptive decoder that turns a bit-history STATE into a probability; its output
// feeds `st1[]`/`st2[]` and hence ContextMap3's 399 of `predictor_lex`'s 590 layer-0 inputs (cm3-staterep census: 41 instances = 67.6% of layer 0; 2 of every 3 cm3 slots are functions of ts, the third st32[s] is not). It is a
// fixed-rate EMA — no count, no warm-up — whose rate is pinned by the shift PAIR:
// the equilibrium is t = 2^32·p only when the two shifts sum to 32, which is why
// stock's `>>14` / `<<18` must move together. Shipped since cmix; NEVER SWEPT (the
// `postnorm-mixer-hyperparam` prune deliberately scoped to mixer-output consumers,
// which this is not). Time constant 2^14 = 16,384 bit-observations per state.
const TS_RATE: u32 = build_options.cm3_ts_rate;
// -Dcm3-ts-runidx (default false = shipped, bit-identical). THE PLACEMENT
// ARBITRAGE (readout-placement-arbitrage): the per-context run
// model below (`runp`/`runByte`/`runCnt`, node 3/4 of the bpos-0 slot, i.e. the
// SAME 128-byte bucket the bit-history state was just read from) is already
// computed and already spends ONE MIXER INPUT per context (`rcpr[...]` at the
// bottom of `mix`). A linear mixer cannot express "trust input j more when
// input j's own run still agrees" -- that is a product term. `ts[s][k]` IS the
// product term. This knob adds the run bucket to the leaf readout index,
// keeping the mixer input. No new memory traffic, no new per-slot state; ts
// grows 1 kB -> 32 kB per bank.
// Value is the CARDINALITY: 0 = off (bit-identical), 4 = agree-class only
// (unseen / agrees-predicts-0 / agrees-predicts-1 / disagrees), 32 = that class
// crossed with 8 run-count buckets. The offline census puts Q=4 at 88% of Q=32's
// value at 1/8 the dilution, and `ts` is a fixed-rate EMA over 2^cm3-ts-rate
// VISITS PER CELL, so the split has to be paid for in evidence.
const TS_Q_RUN: usize = build_options.cm3_ts_runidx;
const TS_RUNIDX: bool = TS_Q_RUN > 1;
// -Dcm3-ts-runidx-ctrl: MATCHED-CARDINALITY CONTROL. Same table split, same
// number of cells, index replaced by a random-but-stable function of the
// context id. Isolates the DILUTION cost from the INFORMATION term: the arm's
// value is (arm - control), not (arm - stock).
const TS_CTRL: bool = build_options.cm3_ts_runidx_ctrl;
// -Dcm3-ts-visidx=N: THE SLOT-OCCUPANCY READOUT. `readout-placement-arbitrage`
//  measured a NODE/SLOT dissociation: the per-NODE visit count at
// the readout is exactly null (the state byte already carries it) while the
// per-CONTEXT-SLOT visit count is worth -2.02 % -- the state byte says nothing
// about how busy the CONTEXT is. cm3 has no per-slot counter, but it has FREE
// SPACE FOR ONE: a bpos-0 slot uses nodes 0-2 as bit-history states and 3-4 as
// the run model; node 5 is never read and node 6 is explicitly written to 0 as
// vestigial (cm3.zig's own comment). So the counter costs ZERO RAM, ZERO extra
// cache lines and ZERO extra probes -- the slot base `off` is already in a
// register when it is written.
const TS_VQ: usize = build_options.cm3_ts_visidx;
const TS_VISIDX: bool = TS_VQ > 1;
const TS_VCTRL: bool = build_options.cm3_ts_visidx_ctrl;
// -Dcm3-ts-recidx=N: the SECOND free byte. Node 6 of a bpos-0 slot is written
// to 0 by the backfill and never read ("vestigial", cm3.zig's own comment), so
// it can hold a 1-byte truncated recency stamp at 64-byte ticks -- the
// `estimator-time-axis` §7 construction, but at ZERO RAM instead of the
// associativity 14 -> 12 trade that report priced it at.
const TS_RQ: usize = build_options.cm3_ts_recidx;
const TS_RECIDX: bool = TS_RQ > 1;
const TS_RCTRL: bool = build_options.cm3_ts_recidx_ctrl;
// -Dcm3-ts-dividx=N: node 6 as an 8-bit DISTINCT-SUCCESSOR SKETCH -- one hash
// bit set per byte that has followed this context; the readout is indexed by
// its popcount. This is PPM's escape statistic, which no ContextMap has ever
// computed, at zero RAM. Instrument: crossing it into the visit index is worth
// +31 % of that index's own marginal (-0.388 % -> -0.510 % @32 MB/300 MB).
const TS_DQ: usize = build_options.cm3_ts_dividx;
const TS_DIVIDX: bool = TS_DQ > 1;
const TS_DCTRL: bool = build_options.cm3_ts_dividx_ctrl;
// -Dcm3-ts-div2idx=N: the SAME sketch widened across BOTH free bytes (nodes 5
// and 6), 16 hash bits instead of 8, popcount 0..16. The 8-bit form's own
// cardinality ladder saturates at Q=9 because a popcount of 8 bits cannot
// exceed 8 -- i.e. the channel is RESOLUTION-limited, not information-limited.
const TS_D2Q: usize = build_options.cm3_ts_div2idx;
const TS_DIV2IDX: bool = TS_D2Q > 1;
const TS_D2CTRL: bool = build_options.cm3_ts_div2idx_ctrl;
// -Dcm3-ts-escidx: node 6 as an 8-bit mask over (byte>>5) of the bytes that
// have followed this context; at each BIT the readout is indexed by how many of
// the eight top-3-bit values still CONSISTENT with the partial byte are set.
// That is PPM's escape decision evaluated on the CURRENT PATH rather than as a
// whole-context diversity summary. One 64 kB lookup table, one load per model
// per bit. Q = 5 (0, 1, 2, >=3, mask-empty).
const TS_EQ: usize = build_options.cm3_ts_escidx;
const TS_ESCIDX: bool = TS_EQ > 1;
const TS_ECTRL: bool = build_options.cm3_ts_escidx_ctrl;
const TS_RSHIFT: u6 = 6; // 64-byte ticks -> an 8-bit stamp resolves 64 B .. 16 KB
comptime {
    if (!(TS_VQ == 1 or TS_VQ == 4 or TS_VQ == 8 or TS_VQ == 16))
        @compileError("-Dcm3-ts-visidx must be 1 (off), 4, 8 or 16");
    if (TS_VISIDX and TS_RUNIDX) @compileError("-Dcm3-ts-visidx and -Dcm3-ts-runidx are exclusive for now");
}
/// Effective readout cardinality and master switch.
const TS_Q: usize = if (TS_VISIDX or TS_RECIDX or TS_DIVIDX or TS_ESCIDX or TS_DIV2IDX)
    (if (TS_VISIDX) TS_VQ else 1) * (if (TS_RECIDX) TS_RQ else 1) *
        (if (TS_DIVIDX) TS_DQ else 1) * (if (TS_ESCIDX) TS_EQ else 1) *
        (if (TS_DIV2IDX) TS_D2Q else 1)
else
    TS_Q_RUN;
const TS_IDX: bool = TS_RUNIDX or TS_VISIDX or TS_RECIDX or TS_DIVIDX or TS_ESCIDX or TS_DIV2IDX;
comptime {
    if (TS_DIV2IDX and (TS_VISIDX or TS_DIVIDX or TS_RECIDX or TS_ESCIDX))
        @compileError("-Dcm3-ts-div2idx uses BOTH free nodes (5 and 6); it is exclusive with the single-byte channels");
    if (TS_DIV2IDX and PACK)
        @compileError("-Dcm3-ts-div2idx assumes the unpacked 1-byte node stride; not supported with -Dcm3pack");
}
comptime {
    var n6: u32 = 0;
    if (TS_DIVIDX) n6 += 1;
    if (TS_RECIDX) n6 += 1;
    if (TS_ESCIDX) n6 += 1;
    if (n6 > 1) @compileError("-Dcm3-ts-dividx / -recidx / -escidx all use node 6; pick one");
}
/// CONSIST[c0][mask] = how many top-3-bit values consistent with the partial
/// byte `c0` are set in `mask`, capped at 3. Built once at comptime.
const CONSIST: [256][256]u8 = blk: {
    @setEvalBranchQuota(1 << 22);
    var t: [256][256]u8 = undefined;
    for (0..256) |c0| {
        // nb = number of already-known bits = floor(log2(c0)); c0 == 0 is unused.
        var nb: usize = 0;
        var v: usize = c0;
        while (v > 1) : (v >>= 1) nb += 1;
        const known: usize = if (c0 == 0) 0 else c0 - (@as(usize, 1) << @intCast(nb));
        for (0..256) |mask| {
            var cnt: u8 = 0;
            for (0..8) |t3| {
                var ok = true;
                var u: usize = 0;
                while (u < 3 and u < nb) : (u += 1) {
                    const want = (t3 >> @intCast(2 - u)) & 1;
                    const have = (known >> @intCast(nb - 1 - u)) & 1;
                    if (want != have) {
                        ok = false;
                        break;
                    }
                }
                if (ok and ((mask >> @intCast(t3)) & 1) != 0) cnt += 1;
            }
            t[c0][mask] = if (cnt >= 3) 3 else cnt;
        }
    }
    break :blk t;
};
comptime {
    if (!(TS_Q_RUN == 1 or TS_Q_RUN == 4 or TS_Q_RUN == 8 or TS_Q_RUN == 16 or TS_Q_RUN == 32))
        @compileError("-Dcm3-ts-runidx must be 1 (off), 4, 8, 16 or 32");
}
const TS_SHR: u5 = @intCast(TS_RATE);
const TS_SHL: u5 = @intCast(32 - TS_RATE);
comptime {
    if (TS_RATE < 10 or TS_RATE > 20) @compileError("-Dcm3-ts-rate must be in 10..20");
}
// ===========================================================================
// -Dleafgrad-fxcm — CODING-LOSS LEAF TRAINING ON ContextMap3's `ts[256]` CELLS
// ===========================================================================
//
// The reopen trigger, verbatim: "a slot->cell gradient path into fxcm's
// ContextMap3 `ts` cells that (a) handles LUT saturation, (b) sums many-to-one,
// and (c) passes the bit-identical Z=0 null". This is that path.
//
// WHAT THE CELL IS. `ts[c]` is a u32 per STATE BYTE c (not per context), one
// [256]u32 array per instance. It is the only free parameter ContextMap3 has:
// `st1`/`st2`/`st32`/`rcpr` are all built once at init and never written. The
// incumbent local rule is `upd`: ts += (y<<18) - (ts>>14), i.e. an EMA of the
// bit toward y*2^32 at rate 2^-14 -- and, exactly as the offline finding says,
// it moves the cell just as hard when the ENSEMBLE was already right and just as
// hard when the mixture puts no weight on the slots that read it.
//
// WHAT THIS ADDS (never replaces -- replacing measured far worse offline):
//     tau_cell -= Z * dL/dtau_cell
// with the chain rule carried EXACTLY from the top-level mixer input back to the
// cell. Per emitting slot j the caller supplies
//     lg.src[j] = g_final * Gamma_j * dv_j/draw_j        (= dL/draw_j)
// -- Gamma_j = dz/dv_j from the same backward recursion -Dmixer-e2e runs, and
// dv/draw from the predictor's own fxcm_stretched table, both folded in by
// predictor_lex because only it knows them. Here we carry the last two factors:
//     draw/dp1   -- the slope of the st1/st2 LUT at p1 = ts[c]>>20
//     dp1/dtau   -- 4096*p*(1-p), p = ts[c]/2^32
//
// (a) SATURATION. st1[i] = clp(sc(cms*strt[i])) saturates at +-2047 over wide
//     state ranges, so its true local slope is EXACTLY ZERO there and a naive
//     chain rule would push a cell that cannot move the output at all. draw/dp1
//     is therefore read off the real table as a windowed difference: a window
//     entirely inside a saturated plateau differences to 0 and the step is
//     skipped. The window (LG_W) is wide because `sc` is an integer >>7: a
//     1-step difference on st2 (slope 13/128) quantises to 0 nine times out of
//     ten, which would silently zero 90% of a perfectly live gradient. LG_W=64
//     is quantisation-robust and still resolves the plateau.
// (b) MANY-TO-ONE. `sm_set` dedups states within one bit, so several contexts
//     (hence several SLOTS) can read one cell and emit the identical value. Each
//     slot is recorded separately at emit time and pass 1 SUMS them into
//     `lg.acc[c]`; pass 2 applies ONE step per distinct cell. Applying per slot
//     would compound the steps and read a cell this bit already moved.
// (c) STATE-INDEXED. `c` is the state byte; the accumulator is [256] per
//     instance and `ts` is per-instance, so nothing couples across the 41
//     instances (only the frozen st2/rcpr pointers are shared).
//
// TIMING. `mix` emits the slots the NEXT bit is predicted from, and `upd` at
// the head of the following `mix` is the local rule for exactly those slots.
// The gradient step therefore lands immediately before `upd`, on the cells
// recorded during the previous `mix`, using the gradient predictor_lex wrote
// from that bit's own mixer weights. Encode and decode see identical inputs.
//
// Z=0 => `d` is identically zero => `continue` before any write => bit-identical.
pub const LGF: bool = leafgrad_options.leafgrad_fxcm;
pub const LGF_Z: f32 = leafgrad_options.leafgrad_z;
pub const LGF_DIAG: bool = leafgrad_options.leafgrad_fxcm and leafgrad_options.leafgrad_diag;
/// Half-width of the LUT slope window (see (a) above).
const LG_W: usize = 32;

/// LAB (`-Dleafgrad-fxcm -Dleafgrad-diag`): running mean of the UNSCALED step
/// |dL/dtau_cell|, so the Z bracket is set from a measurement rather than a
/// guess (the Indirect arm's equivalent was 2.138e-3). Accumulated even at Z=0,
/// which is why the stat is taken BEFORE the `d == 0` early-out.
pub var LG_STAT_SUM: f64 = 0;
pub var LG_STAT_N: u64 = 0;
/// LAB: hazard (a) made quantitative -- how much of the reached slot mass the
/// SATURATING st1/st2 LUT actually kills. `LG_STAT_LIVE` counts pass-1 entries
/// with a non-zero upstream gradient; `LG_STAT_SAT` counts the subset whose LUT
/// window is a flat plateau, i.e. cells that provably cannot move the output.
pub var LG_STAT_LIVE: u64 = 0;
pub var LG_STAT_SAT: u64 = 0;

/// Per-instance leaf-gradient state. Zero-sized (`void`) when the knob is off,
/// so the stock ContextMap3 layout is untouched.
const LgState = struct {
    /// dL/draw per GLOBAL fxcm slot index, written by predictor_lex before
    /// fxcm.perceive. Null in the in-file oracle tests (which build a local
    /// Sink and never run the gradient), so a null pointer is a hard no-op.
    src: ?[*]const f32 = null,
    /// LAB: per-slot "this slot has a reachable adaptive cell" mask, so the
    /// coverage census measures the ACHIEVED reach instead of assuming it.
    reach: ?[*]u8 = null,
    /// One record per emitted adaptive slot: <=MAXCXT contexts x {st1, st2}.
    slot: [MAXCXT * 2]u16 = .{0} ** (MAXCXT * 2),
    cell: [MAXCXT * 2]u8 = .{0} ** (MAXCXT * 2),
    /// 0 = the value came from st1, 1 = from st2.
    tbl: [MAXCXT * 2]u8 = .{0} ** (MAXCXT * 2),
    n: usize = 0,
    /// Pass-1 per-cell gradient accumulator (hazard (b)). Always fully zero
    /// between bits: pass 2 zeroes every cell it visits, unconditionally.
    acc: [256]f32 = .{0} ** 256,
};

// C-HASH64 (-Dcmc2-hash64): the stock checksum `(cxt[i] >> 16) ^ i` is a bit-slice
// of the SAME hash the index masks from the low bits — on the big (2^22-bucket)
// tables idx eats bits 0-21 and chk is bits 16-31, so only ~10 chk bits are
// independent of the index ⇒ ~1.3% of absent-context probes false-merge into a
// stranger's bit-history (cm3-diag measures the rate). Deriving chk from an
// INDEPENDENT multiplicative mix of the full context makes all 16 bits disjoint
// from the index — zero RAM, zero time, attacks the located memory-quality gap.
// NUMERICS VARIANT (different chk ⇒ different eviction decisions), NOT
// bit-identical: needs the ratio ladder. Default off ⇒ stock chk.
const HASH64 = build_options.cmc2_hash64;
inline fn checksum(cxt: u32, i: u32) u16 {
    if (comptime HASH64) {
        // Fresh multiplicative permutation of the full 32-bit context; the top 16
        // bits of a distinct golden-ratio mix are ~independent of the low index bits.
        return @truncate(((cxt ^ (cxt >> 15)) *% 2654435761 +% i) >> 16);
    }
    return @truncate((cxt >> 16) ^ i);
}

// --------------------------------------------------------------------------
// Track-2 telemetry (`-Dcm3-diag`, comptime-gated). DIAG==false ⇒ `Diag` is a
// zero-size empty struct, every counter site is behind `if (comptime DIAG)`,
// and the ContextMap3 layout/behaviour is byte-identical to stock. ON adds the
// per-instance counters below plus a 1/256-sampled false-merge detector.
//
// Measures the two suspected cmC2 defects before we build a fix:
//   (1) checksum/index bit-overlap — the sampled detector counts probes where
//       the 16-bit chk MATCHED but the full 32-bit context differs (a stranger's
//       bit-history got read). Rate ≥~0.3% on the big tables ⇒ C-HASH64 (u64
//       hash pipeline) is worth building.
//   (2) cold-tail churn — `established_kills` (LFU evicted a victim whose state
//       byte ≥6, i.e. a bucket saturated with established entries) + `replacements`
//       vs occupancy tell us which tables are overfilled ⇒ C-GROW/C-SPILL targets.
const DIAG = build_options.cm3_diag;

const DiagCounters = struct {
    probes: u64 = 0, // bucket_get calls
    recent0_hits: u64 = 0, // matched via the `last & 15` fast slot
    scan_hits: u64 = 0, // matched via the 14-slot chk_find scan
    replacements: u64 = 0, // no chk match ⇒ LFU evict + refill
    established_kills: u64 = 0, // replacement whose evicted victim had state ≥6
    backfill_execs: u64 = 0, // pending 2nd-occurrence bit-history rescues realized
    sample_true: u64 = 0, // sampled chk match, full cxt agreed (real revisit)
    sample_false: u64 = 0, // sampled chk match, full cxt differed (FALSE MERGE)
    // Parallel full-cxt store for sampled buckets ((idx & 0xFF)==0), keyed by
    // (idx>>8)*14 + slot. 0 == "no replace recorded here yet" (cxt is ~never 0),
    // which cleanly excludes cold initial-state (chk==0) matches from the rate.
    sample_owner: []u32 = &.{},
};

/// `.{}` inits either the real counters or a zero-size placeholder.
const Diag = if (DIAG) DiagCounters else struct {};

/// Integer count → percentage (0 when the denominator is 0).
fn pct(num: u64, den: u64) f64 {
    if (den == 0) return 0;
    return 100.0 * @as(f64, @floatFromInt(num)) / @as(f64, @floatFromInt(den));
}

// --------------------------------------------------------------------------
// C-PACK (`-Dcm3pack=WP`) — THE DECOUPLED BUCKET.
// Family `state-width-exchange`.
//
// THE FINDING. `bucket_get` below evicts on `argmin` of the RAW ROOT STATE BYTE
// (the `pri` line, ~line 313 pre-change). A bit-history state is therefore TWO
// objects sharing one byte: a predictor state AND the context map's cache
// priority. That is why the obvious lever — narrow the state, spend the freed
// bits on slots — measures **+1.86 % ADVERSE**: a 56-state byte spans 0..55
// instead of 0..255, so the argmin ties constantly and the LFU degrades into
// near-arbitrary choice. Index-monotonicity is necessary and NOT sufficient;
// the RANGE matters too.
//
// THE FIX THE LAW NAMES. Decouple the roles: give the bucket an EXPLICIT
// per-slot eviction priority, charged against the same 128 bytes, and the state
// is then free to shrink:
//
//     S(w, p) = floor(1016 / (16 + 7w + p))        (16-bit chk, 7 w-bit states,
//                                                   p-bit priority; 1016 bits =
//                                                   127 B, 1 B reserved for the
//                                                   bucket's LRU metadata)
//
//     S(8,0) = 14  ==  the shipped `E1<14,128>`, BY CONSTRUCTION (the report's
//                      own G1 gate — reproduced as a comptime assert + a unit
//                      test + the behavioural check that the C++ oracle goldens
//                      still pass under `-Dcm3pack=80`)
//     S(5,4) = 18  ==  5-bit state + 4-bit priority, +28.6 % slots at ZERO RAM
//     S(6,4) = 16 ; S(7,4) = 14 ; S(4,4) = 21
//
// LAYOUT (bit offsets from the bucket's base bit = idx*1024; the bucket is still
// exactly 128 B, so the allocation is byte-for-byte unchanged ⇒ ZERO RAM):
//
//     [0, 16S)                    chk[S] u16              (byte-aligned)
//     [16S, 16S+16)               meta u16: r0 = bits 0..5, r1 = bits 5..10
//     [16S+16, 16S+16+Sp)         pri[S], p bits each
//     [16S+16+Sp, ... + 7SW)      ONE contiguous state bit-array; node (i,j)
//                                 lives at (i*7 + j)*w bits from its base
//
// ⚠ ARRAY-PACK, NOT SLOT-PACK (the report's own §5.2 warning): if each slot's
// seven states were byte-aligned, w = 6 would cost 6 B/slot and give S = 15
// instead of 17 — a 2.8x under-read of the lever. The state region above is one
// contiguous bit array exactly as the formula assumes.
//
// ★ THE RUN MODEL RIDES INSIDE THE STATE REGION, exactly as it does in stock.
// A slot fetched at bpos 0 uses nodes 0,1,2 as states and nodes 3,4 as the run
// model (an 8-bit run count + the 8-bit run byte); nodes 5,6 of such a slot are
// never read. A slot fetched at bpos 2/5 uses all seven nodes as states and
// never touches the run model. So the run field is 16 bits at bit offset 3w
// inside the slot's 7w-bit region — which is bytes 3..4 at w = 8 (stock) and
// requires 4w >= 16, i.e. **w >= 4**. This is why the seven nodes can ALL be
// narrowed even though two of them carry 8-bit non-state payloads: the two roles
// are disjoint per slot-occupancy, in stock and here alike.
// The single exception is the vestigial `t[off+6] = 0` write in the backfill
// block: node 6 of a bpos-0 slot is provably dead, but at w <= 5 its bits ALIAS
// the run byte's MSB, so that write is comptime-dropped exactly when it would
// alias (6w < 3w+16). At w >= 6 it is retained, keeping w = 8 bit-identical.
const PACK_RAW: u32 = build_options.cm3pack;
// ★ THE COUPLING CONTROL (`-Dcm3pack=1WP`, i.e. add 100). Same packed layout,
// same slot count, same automaton, same priority field allocated AND maintained
// (bumped on hit, reset on replace, so the write traffic is identical) — but
// eviction reads the ROOT STATE BYTE, i.e. stock's coupling. This isolates the
// EVICTION RULE at identical geometry, which the naive P54-vs-P50 differential
// cannot: those differ by one slot as well as by the rule.
const PRI_COUPLED: bool = PACK_RAW >= 100;
const PACK_CODE: u32 = PACK_RAW % 100;
/// Master switch. false ⇒ every accessor below is the stock byte expression and
/// the compiled code is the pre-change code (proved by the stock bit-identity gate).
pub const PACK: bool = PACK_CODE != 0;
/// Bit-history state width in bits (8 = stock).
pub const W: usize = if (PACK) @as(usize, PACK_CODE / 10) else 8;
/// Explicit eviction-priority width in bits (0 = stock: the priority IS the root state byte).
pub const PB: usize = if (PACK) @as(usize, PACK_CODE % 10) else 0;
/// Bits charged per slot: 16-bit chk + seven w-bit states + a p-bit priority.
pub const SLOT_BITS: usize = 16 + 7 * W + PB;
/// THE SLOT FORMULA. S(8,0) = 14 = the shipped E1<14,128>.
pub const SLOTS: usize = 1016 / SLOT_BITS;
/// Number of automaton states the packed state field can name.
pub const NSTATES: usize = @as(usize, 1) << @intCast(W);
const PRI_CAP: u32 = if (PB == 0) 0 else (@as(u32, 1) << @intCast(PB)) - 1;

// Bit-offset region bases inside a bucket (packed layout only).
const CHK0: usize = 0;
const META0: usize = 16 * SLOTS;
const PRI0: usize = META0 + 16;
const ST0: usize = PRI0 + SLOTS * PB;
/// Bucket stride: BITS when packed (all addresses are bit offsets), BYTES when stock.
const EU: usize = if (PACK) 1024 else 128;
/// Node-0 address of slot 0 in bucket 0 — the `cp/cp0` init value (stock: 29).
const SLOT0_N0: usize = if (PACK) ST0 else 29;
/// Node-3 address of slot 0 in bucket 0 — the `runp` init value (stock: 32).
const SLOT0_N3: usize = if (PACK) ST0 + 3 * W else 32;
/// An out-of-range slot index meaning "no protected slot" (stock writes 15 via `kep`).
const NO_SLOT: usize = 31;

comptime {
    if (PACK) {
        // w >= 4 so the 16-bit run model fits in nodes 3..6 (4w >= 16); w <= 8.
        if (W < 4 or W > 8) @compileError("cm3pack: state width must be 4..8");
        if (PB > 5) @compileError("cm3pack: priority width must be 0..5");
        if (SLOTS > NO_SLOT) @compileError("cm3pack: slot index must fit the 5-bit meta field");
        // Everything must fit the 128-byte bucket, metadata included.
        if (ST0 + SLOTS * 7 * W > 1024) @compileError("cm3pack: bucket overflow");
    }
    // ★ G1, the report's own by-construction check, asserted at compile time in
    // EVERY build (stock included): the slot formula must reproduce the shipped
    // geometry at w = 8, p = 0.
    if (1016 / (16 + 7 * 8 + 0) != 14) @compileError("cm3pack: S(8,0) != 14");
}

/// Placeholder target for the shared-table pointers before `init` (never read).
const NN_INIT: [1024]u8 = .{0} ** 1024;

inline fn clp(z: i32) i16 {
    if (z < -2047) return -2047;
    if (z > 2047) return 2047;
    return @intCast(z);
}

inline fn sc(p: i32) i32 {
    if (p > 0) return p >> 7;
    return (p + 127) >> 7;
}

/// Software-prefetch hint (speed only; never changes behavior). Warms the hash
/// bucket lines the structure-preserved call sites already compute — the cm3
/// tables are the largest random-access surface after PPMd.
inline fn prefetch_t(t: []const u8, off: usize) void {
    if (off < t.len) @prefetch(&t[off], .{ .rw = .read, .locality = 3, .cache = .data });
}

// --------------------------------------------------------------------------
// C-SPILL (`-Dcmc2-spill=N`, N = MB, default 0 = OFF = byte-identical). Catches
// cold-tail singletons the ContextMap3 buckets evict before their 2nd occurrence
// (cm3-diag: the located award gap = cmC2 collision/churn) and resurrects them on
// a later miss, so the 2nd-occurrence machinery gets warm state instead of cold.
//
// Layout: ONE shared open-addressed `[]u32` table (all ContextMap3 instances point
// at it, disambiguated by a per-instance `spill_salt`), sized to N MB → a
// power-of-two slot count. 4 bytes/entry = 25x denser than a 128-byte bucket:
//   entry u32:  bits [0..24) = 24-bit fingerprint,  bits [24..32) = root state byte.
//   empty  ⟺  state byte == 0  (we only ever store victims with state ≥ 1).
//
// Key = (bucket index, victim checksum). That pair is the bucket's OWN notion of a
// context's identity (chk == (cxt>>16)^i recovers cxt's high 16 bits exactly; idx
// recovers the low bits), so two contexts that alias it are exactly the 16-bit
// false-merge event the bucket already can't tell apart — a spill alias is thus at
// most a harmless mis-seed. The fingerprint is hash bits DISJOINT from the slot
// index bits (index = low bits of the avalanche, fp = high bits), giving spill-
// internal collision resistance (~2^-24 false-resurrection among slot-colliding
// keys). Deterministic in the processed bytes ⇒ encoder and decoder run identical
// logic ⇒ the coded stream stays valid regardless of any mis-seed (codec-invertible).
const SPILL_MB: usize = build_options.cmc2_spill;
const SPILL: bool = SPILL_MB != 0;
// Only spill victims whose root state is this low (seen ~once/twice): the true
// cold tail. Established victims (state ≥ 6, the cm3-diag `established_kills`) are a
// capacity problem for C-GROW, not a resurrection target. Tunable.
const SPILL_EV_MAX: i32 = 2;

/// Shared cmC2 cold-tail spill side-table. Open-addressed, linear-probe, no delete
/// (put updates-or-inserts; a full probe run drops gracefully). Zeroed via the
/// calloc-parity page path so an N-MB reservation stays lazily committed (RAM gate).
pub const Spill = struct {
    slots: []u32 = &.{},
    mask: u64 = 0,
    alloc: std.mem.Allocator = undefined,

    const PROBE_LIMIT: usize = 8;

    pub fn init(a: std.mem.Allocator, mb: usize) !Spill {
        const nslots = std.math.floorPowerOfTwo(usize, (mb * 1024 * 1024) / @sizeOf(u32));
        const slots = try zero_alloc.alloc(a, u32, nslots);
        return .{ .slots = slots, .mask = nslots - 1, .alloc = a };
    }

    pub fn deinit(self: *Spill) void {
        if (self.slots.len != 0) zero_alloc.free(self.alloc, self.slots);
        self.* = undefined;
    }

    /// Avalanche (idx, chk) under a per-instance salt into a 64-bit hash; the low
    /// bits index the array, the high bits are the fingerprint (disjoint ranges).
    inline fn hash(salt: u32, idx: usize, chk: u16) u64 {
        var h: u64 = ((@as(u64, idx) << 16) | @as(u64, chk)) ^ (@as(u64, salt) *% 0x9E3779B97F4A7C15);
        h *%= 0xFF51AFD7ED558CCD;
        h ^= h >> 33;
        h *%= 0xC4CEB9FE1A85EC53;
        h ^= h >> 33;
        return h;
    }

    inline fn fp_of(h: u64) u32 {
        return @intCast((h >> 40) & 0xFFFFFF);
    }

    fn put(self: *Spill, salt: u32, idx: usize, chk: u16, state: u8) void {
        const h = hash(salt, idx, chk);
        const fp = fp_of(h);
        var pos: usize = @intCast(h & self.mask);
        var n: usize = 0;
        while (n < PROBE_LIMIT) : (n += 1) {
            const e = self.slots[pos];
            if ((e >> 24) == 0 or (e & 0xFFFFFF) == fp) { // empty OR same key
                self.slots[pos] = fp | (@as(u32, state) << 24);
                return;
            }
            pos = (pos + 1) & @as(usize, @intCast(self.mask));
        }
    }

    fn get(self: *Spill, salt: u32, idx: usize, chk: u16) u8 {
        const h = hash(salt, idx, chk);
        const fp = fp_of(h);
        var pos: usize = @intCast(h & self.mask);
        var n: usize = 0;
        while (n < PROBE_LIMIT) : (n += 1) {
            const e = self.slots[pos];
            if ((e >> 24) == 0) return 0; // empty (no-delete) ⇒ key absent
            if ((e & 0xFFFFFF) == fp) return @intCast(e >> 24); // hit
            pos = (pos + 1) & @as(usize, @intCast(self.mask));
        }
        return 0;
    }
};

pub const ContextMap3 = struct {
    c: usize = 0,
    cn: usize = 0,
    result: i32 = 0,
    sti: usize = 0,
    cxt_mask: u16 = 0,
    skip2: i32 = 0,
    /// -Dfxcm-slotmask: this instance is DROPPED. `set` keeps only the cn /
    /// cxt_mask bookkeeping (so the slot layout is unchanged), `mix` emits the
    /// exact neutral 0 for each live context, `prefetch_c0` is a no-op, and the
    /// `t` table is never touched (zero_alloc => its pages never become resident).
    masked: bool = false,
    kep: u8 = 0,
    tmask: u32 = 0,
    cp: [MAXCXT]?usize = .{null} ** MAXCXT,
    cp0: [MAXCXT]?usize = .{null} ** MAXCXT,
    runp: [MAXCXT]usize = .{0} ** MAXCXT,
    /// -Dcm3-ts-visidx: this byte's per-slot visit bucket, captured at bpos 0
    /// and reused for all 8 bits (the slot is only resolved once per byte).
    visb: [MAXCXT]u8 = .{0} ** MAXCXT,
    /// -Dcm3-ts-recidx: this byte's per-slot recency bucket, same discipline.
    recb: [MAXCXT]u8 = .{0} ** MAXCXT,
    /// -Dcm3-ts-dividx: this byte's distinct-successor popcount bucket, and the
    /// PREVIOUS occurrence's node-6 address (the sketch is updated exactly where
    /// the run model updates its own byte: on the slot the last occurrence used).
    divb: [MAXCXT]u8 = .{0} ** MAXCXT,
    divp: [MAXCXT]usize = .{0} ** MAXCXT,
    /// -Dcm3-ts-escidx: this byte's cached node-6 mask, per context.
    escm: [MAXCXT]u8 = .{0} ** MAXCXT,
    /// Byte counter for the recency stamp (advanced once per byte at bpos 7).
    nbytes: u64 = 0,
    cxt: [MAXCXT]u32 = .{0} ** MAXCXT,
    cxtn: [MAXCXT]i32 = .{0} ** MAXCXT,
    t: []u8 = &.{},
    ts: [256 * TS_Q]u32 = .{0} ** (256 * TS_Q),
    st1: [4096]i16 = .{0} ** 4096,
    st32: [256]i16 = .{0} ** 256,
    // Shared read-only lookup tables (the C++ points every instance at ONE
    // global st2_p0/st2_p1/rcpr and a shared STA*, fxcmv1.cpp:1084-1086; 41
    // instances x ~10KB of per-instance copies evict each other from cache).
    nn: *const [1024]u8 = &NN_INIT,
    st2: []const i16 = &.{},
    rcpr: []const i16 = &.{},
    alloc: std.mem.Allocator = undefined,
    /// Track-2 telemetry (zero-size unless `-Dcm3-diag`); pure side-channel,
    /// never read by the model, so ON stays bit-identical on the output.
    diag: Diag = .{},
    /// C-SPILL (`-Dcmc2-spill`): shared cold-tail side-table + this instance's salt.
    /// Null unless the owner opts in via `set_spill`; the whole spill code path is
    /// `if (comptime SPILL) if (self.spill) |sp| ...`, so a null pointer (the oracle
    /// tests never set one) keeps behaviour byte-identical even with the flag ON.
    spill: ?*Spill = null,
    spill_salt: u32 = 0,
    /// `-Dleafgrad-fxcm` state; zero-sized `void` when the knob is off.
    lg: if (LGF) LgState else void = if (LGF) .{} else {},

    /// `-Dleafgrad-fxcm`: point this instance at the predictor's per-slot
    /// gradient array (and, under `-Dleafgrad-diag`, the reach mask). Called
    /// once by FxcmV26 after construction; until then `src` is null and the
    /// whole path is a no-op (this is what keeps the in-file oracle tests,
    /// which build their own Sink, working unchanged).
    pub fn lgAttach(self: *ContextMap3, src: [*]const f32, reach: ?[*]u8) void {
        if (comptime !LGF) return;
        self.lg.src = src;
        self.lg.reach = reach;
    }

    pub fn new() ContextMap3 {
        return ContextMap3{};
    }

    /// Opt this instance into the shared C-SPILL table with a distinct salt.
    pub fn set_spill(self: *ContextMap3, sp: *Spill, salt: u32) void {
        self.spill = sp;
        self.spill_salt = salt;
    }

    pub fn deinit(self: *ContextMap3) void {
        if (comptime DIAG) {
            if (self.diag.sample_owner.len != 0) zero_alloc.free(self.alloc, self.diag.sample_owner);
        }
        if (self.t.len != 0) zero_alloc.free(self.alloc, self.t);
        self.* = undefined;
    }

    inline fn next(self: *const ContextMap3, i: i32, y: i32) u8 {
        // C++: nn[ y + i*4 ]
        return self.nn[@intCast(y + i * 4)];
    }

    inline fn pre(self: *const ContextMap3, state: i32) i32 {
        const n0: u32 = @as(u32, self.next(state, 2)) * 3 + 1;
        const n1: u32 = @as(u32, self.next(state, 3)) * 3 + 1;
        return @intCast((n1 << 12) / (n0 + n1));
    }

    // ---- C-PACK accessors -------------------------------------------------
    // Every one of these is the STOCK byte expression when PACK == false, so the
    // default build compiles to the pre-change code (proved by the bit-identity
    // gate, not asserted here). When PACK == true they address a bit-packed
    // bucket; all `elem_base` / node addresses are then BIT offsets.
    //
    // The u32 window read/write is safe past the last bucket: `init` allocates
    // `(m>>7) + 128` elements while only `tmask+1 = m>>7` are addressable, i.e.
    // 16 KB of slack beyond any legal bucket.

    inline fn rdBits(self: *const ContextMap3, bit: usize, comptime n: usize) u32 {
        const by = bit >> 3;
        const sh: u5 = @intCast(bit & 7);
        const v = std.mem.readInt(u32, self.t[by..][0..4], .little);
        return (v >> sh) & ((@as(u32, 1) << @intCast(n)) - 1);
    }

    inline fn wrBits(self: *ContextMap3, bit: usize, comptime n: usize, val: u32) void {
        const by = bit >> 3;
        const sh: u5 = @intCast(bit & 7);
        const mask: u32 = ((@as(u32, 1) << @intCast(n)) - 1) << sh;
        var v = std.mem.readInt(u32, self.t[by..][0..4], .little);
        v = (v & ~mask) | ((val << sh) & mask);
        std.mem.writeInt(u32, self.t[by..][0..4], v, .little);
    }

    /// Bucket index -> its base address (bit offset packed, byte offset stock).
    inline fn elemBase(idx: usize) usize {
        return idx * EU;
    }
    /// Inverse of `elemBase` (the spill/diag key).
    inline fn elemIdx(elem_base: usize) usize {
        return elem_base / EU;
    }
    /// `&bh[i][0]` — the node-0 address of slot `i`.
    inline fn slotBase(elem_base: usize, i: usize) usize {
        return if (comptime PACK) elem_base + ST0 + i * 7 * W else elem_base + 29 + i * 7;
    }
    /// Address of node `j` given the address of node 0.
    /// Byte distance between consecutive nodes of a slot (unpacked layout only;
    /// -Dcm3-ts-div2idx is rejected at comptime under PACK below).
    const W_BYTES: usize = 1;
    inline fn nodeAt(base: usize, j: usize) usize {
        return if (comptime PACK) base + j * W else base + j;
    }
    /// Byte address for a prefetch hint.
    inline fn byteOf(addr: usize) usize {
        return if (comptime PACK) addr >> 3 else addr;
    }

    /// Bit-history state at a node address.
    inline fn stGet(self: *const ContextMap3, addr: usize) u32 {
        return if (comptime PACK) self.rdBits(addr, W) else self.t[addr];
    }
    inline fn stSet(self: *ContextMap3, addr: usize, v: u32) void {
        if (comptime PACK) self.wrBits(addr, W, v) else {
            self.t[addr] = @truncate(v);
        }
    }
    /// The built-in run model, which lives at node 3 of a bpos-0 slot: an 8-bit
    /// run count and the 8-bit run byte (stock: the bytes at node 3 and node 4).
    inline fn runCnt(self: *const ContextMap3, rp: usize) u32 {
        return if (comptime PACK) self.rdBits(rp, 8) else self.t[rp];
    }
    inline fn runCntSet(self: *ContextMap3, rp: usize, v: u32) void {
        if (comptime PACK) self.wrBits(rp, 8, v) else {
            self.t[rp] = @truncate(v);
        }
    }
    inline fn runByte(self: *const ContextMap3, rp: usize) u32 {
        return if (comptime PACK) self.rdBits(rp + 8, 8) else self.t[rp + 1];
    }
    inline fn runByteSet(self: *ContextMap3, rp: usize, v: u32) void {
        if (comptime PACK) self.wrBits(rp + 8, 8, v) else {
            self.t[rp + 1] = @truncate(v);
        }
    }

    /// ★ THE EVICTION PRIORITY. Stock reads the raw root state byte — the coupling
    /// this family exists to break. When p > 0 it is an explicit per-slot counter,
    /// bumped (saturating) on every probe that lands on the slot and reset on
    /// replacement, i.e. an LFU over PROBES rather than over the automaton's
    /// numbering — so the state may be narrowed without collapsing the cache score.
    inline fn priGet(self: *const ContextMap3, elem_base: usize, i: usize) i32 {
        if (comptime PB != 0 and !PRI_COUPLED) return @intCast(self.rdBits(elem_base + PRI0 + i * PB, PB));
        return @intCast(self.stGet(slotBase(elem_base, i)));
    }
    inline fn priBump(self: *ContextMap3, elem_base: usize, i: usize) void {
        if (comptime PB != 0) {
            const a = elem_base + PRI0 + i * PB;
            const v = self.rdBits(a, PB);
            if (v < PRI_CAP) self.wrBits(a, PB, v + 1);
        }
    }
    inline fn priReset(self: *ContextMap3, elem_base: usize, i: usize) void {
        if (comptime PB != 0) self.wrBits(elem_base + PRI0 + i * PB, PB, 0);
    }

    /// The bucket's 2-slot LRU. Stock packs (recent0, recent1) as two nibbles of
    /// the `last` byte; the packed layout needs 5 bits each (up to 21 slots) and
    /// uses a u16. `kep != 0` means "do NOT protect the previous slot" — stock
    /// achieves that by OR-ing 0xf0 so recent1 becomes 15, an invalid slot index;
    /// here it becomes NO_SLOT. Identical behaviour, both are >= SLOTS.
    inline fn metaRaw(self: *const ContextMap3, elem_base: usize) u32 {
        if (comptime PACK) {
            const o = (elem_base + META0) >> 3;
            return @as(u32, self.t[o]) | (@as(u32, self.t[o + 1]) << 8);
        }
        return self.t[elem_base + 28];
    }
    inline fn metaR0(m: u32) usize {
        return if (comptime PACK) m & 31 else m & 15;
    }
    inline fn metaR1(m: u32) usize {
        return if (comptime PACK) (m >> 5) & 31 else m >> 4;
    }
    /// Record `bi` as the most-recent slot (stock: `last = (last<<4) | bi | keep`).
    inline fn metaTouch(self: *ContextMap3, elem_base: usize, m: u32, bi: usize, keep: u8) void {
        if (comptime PACK) {
            const r1: u32 = if (keep != 0) @as(u32, NO_SLOT) else @as(u32, @intCast(metaR0(m)));
            const v: u32 = @as(u32, @intCast(bi)) | (r1 << 5);
            const o = (elem_base + META0) >> 3;
            self.t[o] = @truncate(v & 0xff);
            self.t[o + 1] = @truncate(v >> 8);
        } else {
            self.t[elem_base + 28] = @as(u8, @truncate(m << 4)) | @as(u8, @intCast(bi)) | keep;
        }
    }

    inline fn chk_get(self: *const ContextMap3, elem_base: usize, i: usize) u16 {
        const o = if (comptime PACK) (elem_base + CHK0 + i * 16) >> 3 else elem_base + i * 2;
        return @as(u16, self.t[o]) | (@as(u16, self.t[o + 1]) << 8);
    }

    inline fn chk_set(self: *ContextMap3, elem_base: usize, i: usize, v: u16) void {
        const o = if (comptime PACK) (elem_base + CHK0 + i * 16) >> 3 else elem_base + i * 2;
        self.t[o] = @truncate(v & 0xff);
        self.t[o + 1] = @truncate(v >> 8);
    }

    /// First index `i in 0..SLOTS` with `chk[i] == ch` (the scalar loop's hit order).
    fn chk_find(self: *const ContextMap3, elem_base: usize, ch: u16) ?usize {
        var i: usize = 0;
        while (i < SLOTS) : (i += 1) {
            if (self.chk_get(elem_base, i) == ch) return i;
        }
        return null;
    }

    /// E1::get — checksum match or LFU replace with 2-slot LRU. Returns the byte
    /// offset of `&bh[bi][0]` (elem_base + 29 + bi*7). `full_cxt` is the caller's
    /// 32-bit context hash — only read under `-Dcm3-diag` (unused otherwise) to
    /// distinguish a real revisit from a 16-bit-checksum false merge.
    fn bucket_get(self: *ContextMap3, elem_base: usize, ch: u16, keep: u8, full_cxt: u32) usize {
        if (comptime DIAG) self.diag.probes += 1;
        const last = self.metaRaw(elem_base);
        const recent0: usize = metaR0(last);
        const recent1: usize = metaR1(last);
        if (recent0 < SLOTS and self.chk_get(elem_base, recent0) == ch) {
            if (comptime DIAG) {
                self.diag.recent0_hits += 1;
                self.diag_match(elem_base, recent0, full_cxt);
            }
            self.priBump(elem_base, recent0);
            return slotBase(elem_base, recent0);
        }
        if (self.chk_find(elem_base, ch)) |i| {
            if (comptime DIAG) {
                self.diag.scan_hits += 1;
                self.diag_match(elem_base, i, full_cxt);
            }
            self.priBump(elem_base, i);
            self.metaTouch(elem_base, last, i, 0);
            return slotBase(elem_base, i);
        }
        var b: i32 = 0xffff;
        var bi: usize = 0;
        var i: usize = 0;
        while (i < SLOTS) : (i += 1) {
            if (i != recent0 and i != recent1) {
                const pri: i32 = self.priGet(elem_base, i); // stock: bh[i][0]
                if (pri < b) {
                    b = pri;
                    bi = i;
                }
            }
        }
        if (comptime DIAG) {
            self.diag.replacements += 1;
            // Every candidate's state byte ≥6 ⇒ we just evicted an established
            // entry (the LFU min itself is ≥6): a real damage event.
            if (self.stGet(slotBase(elem_base, bi)) >= 6) self.diag.established_kills += 1;
            const idx = elemIdx(elem_base);
            if (idx & 0xFF == 0) self.diag.sample_owner[(idx >> 8) * SLOTS + bi] = full_cxt;
        }
        // C-SPILL eviction hook: `b` IS the victim's root state (bh[bi][0], the LFU
        // min). If it is a low-evidence cold-tail singleton, park it in the shared
        // side-table keyed by (this bucket idx, the victim's OWN checksum) — read
        // that checksum BEFORE chk_set overwrites the slot below.
        if (comptime SPILL) {
            if (self.spill) |sp| {
                // The victim's ROOT STATE (== `b` only when the priority is the
                // state byte; under an explicit priority it must be read out).
                const vs: i32 = @intCast(self.stGet(slotBase(elem_base, bi)));
                if (vs >= 1 and vs <= SPILL_EV_MAX)
                    sp.put(self.spill_salt, elemIdx(elem_base), self.chk_get(elem_base, bi), @intCast(vs));
            }
        }
        self.metaTouch(elem_base, last, bi, keep);
        self.chk_set(elem_base, bi, ch);
        self.priReset(elem_base, bi);
        const sb = slotBase(elem_base, bi);
        var j: usize = 0;
        while (j < 7) : (j += 1) {
            self.stSet(nodeAt(sb, j), 0);
        }
        // C-SPILL resurrection hook: if THIS context (idx, ch) was spilled on a past
        // eviction, seed the freshly-zeroed slot's root (node 0) with the saved
        // state instead of cold 0. Only node 0 is touched — nodes 3/4/6 (the run
        // model + backfill trigger `t[off+3]==2`) stay 0, so the 2nd-occurrence
        // machinery is fed warm, never mis-fired.
        if (comptime SPILL) {
            if (self.spill) |sp| {
                const rs = sp.get(self.spill_salt, elemIdx(elem_base), ch);
                if (rs != 0) self.stSet(sb, @min(@as(u32, rs), @as(u32, @intCast(NSTATES - 1))));
            }
        }
        return sb;
    }

    /// A checksum match landed on slot `slot` of the bucket at `elem_base`. On a
    /// sampled bucket, compare the recorded owner cxt against the current one:
    /// agree ⇒ a real revisit, differ ⇒ a 16-bit-checksum false merge (defect 1).
    /// Only ever called under `-Dcm3-diag`.
    inline fn diag_match(self: *ContextMap3, elem_base: usize, slot: usize, full_cxt: u32) void {
        const idx = elemIdx(elem_base);
        if (idx & 0xFF != 0) return;
        const owner = self.diag.sample_owner[(idx >> 8) * SLOTS + slot];
        if (owner == 0) return; // no replace recorded here yet (cold / initial chk==0)
        if (owner == full_cxt) {
            self.diag.sample_true += 1;
        } else {
            self.diag.sample_false += 1;
        }
    }

    /// fxcmv1.cpp ContextMap3::Init.
    pub fn init(
        self: *ContextMap3,
        a: std.mem.Allocator,
        m1: u32,
        c1: usize,
        cms: i32,
        cms3: i32,
        cms4: i32,
        nn: *const [1024]u8,
        kep: u8,
        skip2: i32,
        st2: []const i16,
        rcpr: []const i16,
        strt: *const [4096]i16,
    ) void {
        self.alloc = a;
        // Preallocate the per-mix sink once: MAXCXT(8) contexts x <=4 slots << 64.
        // The hot addbelow then never touches the allocator (C++ writes into a
        // fixed array; the checked append was ~2% of runtime).
        self.c = @min(c1, MAXCXT);
        const m = m1 *% 2;
        self.tmask = (m >> 7) -% 1;
        self.cn = 0;
        self.result = 0;
        self.cxt_mask = if (self.c > 1) 0 else 0xfffe;
        self.kep = kep;
        self.skip2 = skip2;
        const nelem: usize = @as(usize, (m >> 7)) + 128;
        // calloc-parity (C++ alloc1): kernel-zeroed lazy pages — the cm banks
        // total ~5 GB and must not be committed up front (10 GB Hutter gate).
        self.t = zero_alloc.alloc(a, u8, nelem * 128) catch unreachable;
        if (comptime DIAG) {
            // One u32 per (sampled-bucket, slot). Sampled buckets are the 1/256
            // with (idx & 0xFF)==0, so length = ((tmask>>8)+1)*14. Kernel-zeroed
            // (0 == unseen), sparsely touched — cheap even on the 537MB tables.
            const slen: usize = ((@as(usize, self.tmask) >> 8) + 1) * SLOTS;
            self.diag.sample_owner = zero_alloc.alloc(a, u32, slen) catch unreachable;
        }
        self.nn = nn;
        self.st2 = st2;
        self.rcpr = rcpr;
        for (0..256) |i| {
            const si: i32 = @intCast(i);
            const n0: u32 = @as(u32, self.next(si, 2)) * 3 + 1;
            const n1: u32 = @as(u32, self.next(si, 3)) * 3 + 1;
            // ((n1 << 20) / (n0 + n1)) wrapping_shl 12  ==  ... *% 4096
            const seed: u32 = ((n1 << 20) / (n0 + n1)) *% 4096;
            if (comptime TS_IDX) {
                for (0..TS_Q) |q| self.ts[i * TS_Q + q] = seed;
            } else {
                self.ts[i] = seed;
            }
        }
        for (0..self.c) |i| {
            self.cp0[i] = SLOT0_N0;
            self.cp[i] = SLOT0_N0;
            self.runp[i] = SLOT0_N3;
        }
        for (0..4096) |i| {
            self.st1[i] = clp(sc(cms * @as(i32, strt[i])));
        }
        for (0..256) |s| {
            const si: i32 = @intCast(s);
            const n0: i32 = -@as(i32, @intFromBool(self.next(si, 2) == 0));
            const n1: i32 = -@as(i32, @intFromBool(self.next(si, 3) == 0));
            var r: i32 = 0;
            var sp0: i32 = 0;
            if (n1 - n0 == 1) {
                sp0 = 0;
                r = 1;
            }
            if (n1 - n0 == -1) {
                sp0 = 4095;
                r = 1;
            }
            if (r == 1) {
                const pre_s = self.pre(si);
                const st8 = clp(sc(cms4 * (pre_s - sp0)));
                const pre_str: i32 = strt[@intCast(std.math.clamp(pre_s, 0, 4095))];
                const st32v = clp(sc(cms3 * pre_str));
                if (s < 8) {
                    self.st32[s] = st8;
                } else {
                    self.st32[s] = @intCast((@as(i32, st8) + @as(i32, st32v)) >> 1);
                }
            } else {
                self.st32[s] = 0;
            }
        }
    }

    /// fxcmv1.cpp ContextMap3::reset — wipe the table and re-arm at element 0.
    /// Fired at `</page>` (parse_byte.page_reset) on cmC[0] (+cmC[1] conditional).
    pub fn reset(self: *ContextMap3) void {
        @memset(self.t[0 .. (@as(usize, self.tmask) + 1) * 128], 0);
        if (comptime DIAG) {
            // The table just got wiped; drop the parallel owners so a post-reset
            // chk match can't count against a pre-reset stranger (false positive).
            if (self.diag.sample_owner.len != 0) @memset(self.diag.sample_owner, 0);
        }
        for (0..self.c) |i| {
            self.cp0[i] = SLOT0_N0;
            self.cp[i] = SLOT0_N0;
            self.runp[i] = SLOT0_N3;
        }
        self.cn = 0;
        self.result = 0;
        self.sti = 0;
        self.cxt_mask = if (self.c > 1) 0 else 0xfffe;
    }

    /// C++ reads the raw cxtMask (e.g. `if (cmC2[0].cxtMask) ordX=2`).
    pub fn cxt_mask_val(self: *const ContextMap3) u16 {
        return self.cxt_mask;
    }

    /// Track-2: scan this instance's live table for occupancy/evidence and print
    /// one TSV row (see fxcm_v26.cm3_diag_dump for the header). Pure reads —
    /// touches only `self.t` (read) and `self.diag` (read). No-op unless
    /// `-Dcm3-diag`. Called once at predictor teardown.
    pub fn diag_dump(self: *const ContextMap3, name: []const u8) void {
        if (comptime !DIAG) return;
        const nbkt: usize = @as(usize, self.tmask) + 1;
        var occupied: u64 = 0; // slots with a nonzero state byte
        var full_buckets: u64 = 0; // all 14 slots occupied
        var ev_lo: u64 = 0; // state 1..5   (fresh / cold-tail)
        var ev_mid: u64 = 0; // state 6..31  (established)
        var ev_hi: u64 = 0; // state 32..255 (deep history)
        var bk: usize = 0;
        while (bk < nbkt) : (bk += 1) {
            const eb = elemBase(bk);
            var full = true;
            var s: usize = 0;
            while (s < SLOTS) : (s += 1) {
                const ev = self.stGet(slotBase(eb, s)); // bh[s][0]
                if (ev == 0) {
                    full = false;
                    continue;
                }
                occupied += 1;
                if (ev < 6) ev_lo += 1 else if (ev < 32) ev_mid += 1 else ev_hi += 1;
            }
            if (full) full_buckets += 1;
        }
        const total_slots = nbkt * SLOTS;
        const d = &self.diag;
        const samp = d.sample_true + d.sample_false;
        std.debug.print(
            "CM3DIAG\t{s}\t{d}\t{d}\t{d}\t{d}\t{d:.3}\t{d}\t{d}\t{d:.3}\t{d}\t{d}\t{d}\t{d:.4}\t{d:.3}\t{d:.4}\t{d}\t{d}\t{d}\t{d}\n",
            .{
                name,
                (nbkt * 128) / (1024 * 1024), // table MB
                d.probes,
                d.recent0_hits,
                d.scan_hits,
                pct(d.recent0_hits + d.scan_hits, d.probes), // hit%
                d.replacements,
                d.established_kills,
                pct(d.established_kills, d.replacements), // estk%
                d.backfill_execs,
                samp,
                d.sample_false,
                pct(d.sample_false, samp), // fmerge%
                pct(occupied, total_slots), // occ%
                pct(full_buckets, nbkt), // fullbkt%
                occupied,
                ev_lo,
                ev_mid,
                ev_hi,
            },
        );
    }

    pub fn set(self: *ContextMap3, cx_in: u32) void {
        // -Dfxcm-slotdelete: a deleted instance keeps NO state at all — `cn` stays
        // 0 forever, so `mix` emits nothing, `prefetch_c0` loops zero times and
        // the `cxt`/`cxt_mask` bookkeeping the mask arm still paid for is gone.
        if (comptime SLOTDEL) {
            if (self.masked) return;
        }
        if (self.cn >= self.c) return;
        if (comptime MASK_ANY) {
            if (self.masked) {
                // Layout-preserving: cn and cxt_mask advance exactly as stock, so
                // this instance still occupies its cn*(3+skip2) slots. Skipped:
                // the cxt[] hash write and the two 128-B bucket prefetches.
                self.cn += 1;
                self.cxt_mask = self.cxt_mask *% 2;
                return;
            }
        }
        const i = self.cn;
        self.cn += 1;
        var cx = cx_in *% 987654323 +% @as(u32, @intCast(i));
        cx = (cx << 16) | (cx >> 16);
        self.cxt[i] = cx *% 123456791 +% @as(u32, @intCast(i));
        self.cxt_mask = self.cxt_mask *% 2;
        const idx: usize = @intCast((self.cxt[i] +% 1) & self.tmask);
        prefetch_t(self.t, idx * 128);
        prefetch_t(self.t, idx * 128 + 64);
    }

    /// Prefetch the bpos-2/5 buckets for every live context. Purely a cache hint.
    pub fn prefetch_c0(self: *const ContextMap3, c0: u32) void {
        if (comptime MASK_ANY) {
            if (self.masked) return;
        }
        var i: usize = 0;
        while (i < self.cn) : (i += 1) {
            if ((self.cxt_mask >> @as(u4, @intCast(self.cn - i))) & 1 != 0) continue;
            if (self.runCnt(self.runp[i]) == 0) continue;
            const idx: usize = @intCast((self.cxt[i] +% c0) & self.tmask);
            prefetch_t(self.t, idx * 128);
            prefetch_t(self.t, idx * 128 + 64);
        }
    }

    pub fn sets(self: *ContextMap3) void {
        if (comptime SLOTDEL) {
            if (self.masked) return;
        }
        if (self.cn >= self.c) return;
        self.cn += 1;
        self.cxt_mask = self.cxt_mask +% 1;
        self.cxt_mask = self.cxt_mask *% 2;
    }

    inline fn add(self: *ContextMap3, sink: *const Sink, p: i32) void {
        _ = self;
        sink.push(clp(p));
    }

    /// Internal statemap set (returns the prediction). Mirrors `int set(c,i)`.
    inline fn sm_set(self: *ContextMap3, c: i32, i: usize) i32 {
        const pr: i32 = @intCast(self.ts[@intCast(c)] >> 20);
        if (self.sti >= MAXCXT) return pr;
        if (i == 0) {
            self.cxtn[self.sti] = c;
            self.sti += 1;
            return pr;
        }
        var j: usize = 0;
        while (j < self.sti) : (j += 1) {
            if (self.cxtn[j] == c) return pr;
        }
        self.cxtn[self.sti] = c;
        self.sti += 1;
        return pr;
    }

    /// `-Dleafgrad-fxcm`: the fused coding-loss step on `ts`, applied to the
    /// cells the PREVIOUS `mix` read, immediately BEFORE the incumbent local
    /// rule `upd` (the local running average stays the estimator; the gradient
    /// is a correction -- replacing it measured far worse offline, §3.3 of the
    /// offline findings). Comptime-dead when the knob is off.
    fn lgApply(self: *ContextMap3) void {
        if (comptime !LGF) return;
        const src = self.lg.src orelse return;
        if (self.lg.n == 0) return;

        // ---- pass 1: SUM into the cell (hazard (b): sm_set dedup is many-to-one,
        // so several slots can read one `ts` cell and each owes it a term). ----
        var k: usize = 0;
        while (k < self.lg.n) : (k += 1) {
            const g: f32 = src[self.lg.slot[k]];
            if (g == 0) continue; // e.g. an -Dadaptive-depth skip bit
            const c: usize = self.lg.cell[k];
            const p1: usize = @intCast(self.ts[c] >> 20);
            // (a) LUT slope off the REAL table over a +-LG_W window: a window
            // wholly inside a clpplateau differences to exactly 0, and the
            // width keeps the integer scquantisation from faking a plateau.
            const lo: usize = if (p1 > LG_W) p1 - LG_W else 0;
            const hi: usize = @min(p1 + LG_W, 4095);
            if (hi == lo) continue;
            const tab: []const i16 = if (self.lg.tbl[k] == 0) self.st1[0..] else self.st2;
            const slope: f32 = @as(f32, @floatFromInt(@as(i32, tab[hi]) - @as(i32, tab[lo]))) /
                @as(f32, @floatFromInt(@as(i32, @intCast(hi - lo))));
            if (comptime LGF_DIAG) LG_STAT_LIVE += 1;
            if (slope == 0) {
                if (comptime LGF_DIAG) LG_STAT_SAT += 1;
                continue; // saturated: this cell cannot move the output
            }
            self.lg.acc[c] += g * slope;
        }

        // ---- pass 2: ONE step per distinct cell. Every visited cell is zeroed
        // unconditionally, so `acc` is all-zero between bits without needing the
        // (capped, hence unreliable) cxtn list to enumerate the touched set. ----
        k = 0;
        while (k < self.lg.n) : (k += 1) {
            const c: usize = self.lg.cell[k];
            const a: f32 = self.lg.acc[c];
            self.lg.acc[c] = 0;
            if (a == 0) continue;
            var p: f64 = @as(f64, @floatFromInt(self.ts[c])) / 4294967296.0;
            if (p < 1.0e-7) p = 1.0e-7 else if (p > 1.0 - 1.0e-7) p = 1.0 - 1.0e-7;
            // (c) the cell is STATE-indexed; p is its own probability, p1 = 4096p,
            // so dL/dtau = dL/dp1 * dp1/dtau with dp1/dtau = 4096*p*(1-p).
            const dl_dtau: f64 = @as(f64, a) * 4096.0 * p * (1.0 - p);
            if (comptime LGF_DIAG) {
                LG_STAT_SUM += @abs(dl_dtau);
                LG_STAT_N += 1;
            }
            var d: f64 = @as(f64, LGF_Z) * dl_dtau;
            if (d == 0) continue; // Z == 0 -> nothing is written -> bit-identical null
            if (d > 12.0) d = 12.0 else if (d < -12.0) d = -12.0;
            // sigma(logit(p) - d) with no log: p / (p + (1-p)*exp(d)).
            var q: f64 = p / (p + (1.0 - p) * @exp(d));
            if (q < 1.0e-7) q = 1.0e-7 else if (q > 1.0 - 1.0e-7) q = 1.0 - 1.0e-7;
            var nv: f64 = @round(q * 4294967296.0);
            if (nv > 4294967295.0) nv = 4294967295.0 else if (nv < 0.0) nv = 0.0;
            self.ts[c] = @intFromFloat(nv);
        }
        self.lg.n = 0;
    }

    inline fn upd(self: *ContextMap3, x: *const X) void {
        var j: usize = 0;
        while (j < self.sti) : (j += 1) {
            const idx: usize = @intCast(self.cxtn[j]);
            const p0 = self.ts[idx];
            const pr1: i32 = @intCast(p0 >> TS_SHR);
            self.ts[idx] = p0 +% @as(u32, @bitCast((x.y << TS_SHL) - pr1));
        }
    }

    /// `-Dleafgrad-fxcm`: record that the slot about to be pushed reads cell `s`
    /// through table `tbl` (0=st1, 1=st2). The global slot index is `slot_n.*`
    /// -- the position `push` is about to write -- so no static slot->producer
    /// map is needed at all (the plumbing estimate assumed one). Slots
    /// past SLOT_CAP are dropped by the sink and never become mixer inputs, so
    /// they are not recorded.
    inline fn lgRecord(self: *ContextMap3, sink: *const Sink, s: i32, tbl: u8) void {
        if (comptime !LGF) return;
        if (comptime TS_IDX) @compileError("-Dleafgrad-fxcm and -Dcm3-ts-* are exclusive: lg.cell is a u8 and cannot hold the 8192-cell composite index");
        if (self.lg.src == null) return;
        const j = sink.slot_n.*;
        if (j >= SLOT_CAP) return;
        if (self.lg.n >= MAXCXT * 2) return;
        self.lg.slot[self.lg.n] = @intCast(j);
        self.lg.cell[self.lg.n] = @intCast(s);
        self.lg.tbl[self.lg.n] = tbl;
        self.lg.n += 1;
        if (comptime LGF_DIAG) if (self.lg.reach) |r| {
            r[j] = 1;
        };
    }

    inline fn mix3(self: *ContextMap3, sink: *const Sink, s: i32, i: usize, k: usize) i32 {
        if (s == 0) {
            self.add(sink, 0);
            if (self.skip2 == 1) self.add(sink, 0);
            self.add(sink, 0);
            return 0;
        } else {
            // The ONLY change under -Dcm3-ts-runidx: which ts cell this state
            // reads. `st32[s]` below stays on the raw state (it is not a ts read).
            const cc: i32 = if (comptime TS_IDX)
                @intCast(@as(usize, @intCast(s)) * TS_Q + k)
            else
                s;
            const p1 = self.sm_set(cc, i);
            self.lgRecord(sink, cc, 0);
            self.add(sink, self.st1[@intCast(p1)]);
            if (self.skip2 == 1) {
                self.lgRecord(sink, cc, 1);
                self.add(sink, self.st2[@intCast(p1)]);
            }
            self.add(sink, self.st32[@intCast(s)]);
            return 1;
        }
    }

    inline fn mix4(self: *ContextMap3, sink: *const Sink) void {
        self.add(sink, 0);
        if (self.skip2 == 1) self.add(sink, 0);
        self.add(sink, 0);
        self.add(sink, 0);
    }

    /// fxcmv1.cpp ContextMap3::mix — update bit histories and predict.
    pub fn mix(self: *ContextMap3, x: *const X, sink: *const Sink) i32 {
        self.result = 0;
        // -Dfxcm-slotdelete: TRUE DELETION — no emission at all, so every
        // downstream bank (in1, slots, layer-0) is `DEL_SLOTS` columns narrower.
        // `cn` is pinned at 0 by `set`/`sets` above, so the stock bpos-7 reset
        // would be a no-op and is skipped with everything else.
        if (comptime SLOTDEL) {
            if (self.masked) return 0;
        }
        if (comptime MASK_ANY) {
            if (self.masked) {
                // mix4is the stock NEUTRAL emitter (3 + skip2 zeros), so the
                // slot count per live context is identical to the live path.
                var q: usize = 0;
                while (q < self.cn) : (q += 1) self.mix4(sink);
                self.sti = 0;
                if (x.bpos == 7) {
                    self.cn = 0;
                    self.cxt_mask = 0;
                }
                return 0;
            }
        }
        // -Dleafgrad-fxcm: the coding-loss step lands on the cells the previous
        // mixread, BEFORE the incumbent local rule (fused form).
        if (comptime LGF) self.lgApply();
        self.upd(x);
        self.sti = 0;
        // sm32: prefetch each live context's carried-over bucket before the loop
        // reads t[cp[j]] / t[runp[j]] (the two hottest cm3 loads) — the bucket was
        // last touched a full predict/perceive ago and is likely L2-evicted. One
        // prefetch of runp[j] covers both: cp[j] = off+cm_bit_state and
        // runp[j] = off+3 live in the same bh[bi] (≤15 B apart → same cache line),
        // so prefetching runp[j] alone (always valid, unlike optional cp[j])
        // halves the prefetch overhead. Pure cache hint, bit-identical.
        {
            var j: usize = 0;
            while (j < self.cn) : (j += 1) prefetch_t(self.t, byteOf(self.runp[j]));
        }
        var i: usize = 0;
        while (i < self.cn) : (i += 1) {
            if ((self.cxt_mask >> @as(u4, @intCast(self.cn - i))) & 1 != 0) {
                self.mix4(sink);
            } else {
                if (self.cp[i]) |cpi| {
                    const st: i32 = @intCast(self.stGet(cpi));
                    self.stSet(cpi, self.next(st, x.y));
                }
                var s: i32 = 0;
                if (x.bpos > 1 and self.runCnt(self.runp[i]) == 0) {
                    self.cp[i] = null;
                } else {
                    const chksum: u16 = checksum(self.cxt[i], @intCast(i));
                    if (x.bpos != 0) {
                        if (x.bpos == 2 or x.bpos == 5) {
                            const idx: usize = @intCast((self.cxt[i] +% @as(u32, @intCast(x.c0))) & self.tmask);
                            const off = self.bucket_get(elemBase(idx), chksum, self.kep, self.cxt[i]);
                            self.cp0[i] = off;
                            self.cp[i] = off;
                        } else {
                            self.cp[i] = nodeAt(self.cp0[i].?, @as(usize, @intCast(x.cm_bit_state)));
                        }
                    } else {
                        const idx: usize = @intCast((self.cxt[i] +% @as(u32, @intCast(x.c0))) & self.tmask);
                        const off = self.bucket_get(elemBase(idx), chksum, self.kep, self.cxt[i]);
                        self.cp0[i] = off;
                        self.cp[i] = off;
                        // pending bit histories for bits 2-7. `off+3` is the RUN
                        // COUNT of this (bpos-0) slot, not a bit-history state:
                        // == 2 means "seen exactly once before" (see the run-count
                        // update below), which is the backfill trigger.
                        const rp0 = nodeAt(off, 3);
                        if (self.runCnt(rp0) == 2) {
                            if (comptime DIAG) self.diag.backfill_execs += 1;
                            const c: i32 = @as(i32, @intCast(self.runByte(rp0))) + 256;
                            const idx1: usize = @intCast((self.cxt[i] +% @as(u32, @intCast(c >> 6))) & self.tmask);
                            const idx2p: usize = @intCast((self.cxt[i] +% @as(u32, @intCast(c >> 3))) & self.tmask);
                            prefetch_t(self.t, idx2p * 128);
                            prefetch_t(self.t, idx2p * 128 + 64);
                            const p = self.bucket_get(elemBase(idx1), chksum, self.kep, self.cxt[i]);
                            self.stSet(p, @intCast(1 + ((c >> 5) & 1)));
                            self.stSet(nodeAt(p, 1 + @as(usize, @intCast((c >> 5) & 1))), @intCast(1 + ((c >> 4) & 1)));
                            self.stSet(nodeAt(p, 3 + @as(usize, @intCast((c >> 4) & 3))), @intCast(1 + ((c >> 3) & 1)));
                            const idx2: usize = @intCast((self.cxt[i] +% @as(u32, @intCast(c >> 3))) & self.tmask);
                            const p2 = self.bucket_get(elemBase(idx2), chksum, self.kep, self.cxt[i]);
                            self.stSet(p2, @intCast(1 + ((c >> 2) & 1)));
                            self.stSet(nodeAt(p2, 1 + @as(usize, @intCast((c >> 2) & 1))), @intCast(1 + ((c >> 1) & 1)));
                            self.stSet(nodeAt(p2, 3 + @as(usize, @intCast((c >> 1) & 3))), @intCast(1 + (c & 1)));
                            // Vestigial: node 6 of a bpos-0 slot is never read (that
                            // slot only uses nodes 0-2 as states and 3-4 as the run
                            // model). Kept where it cannot alias the run field —
                            // node 6 starts at 6w, the run field ends at 3w+16 — so
                            // it survives at w >= 6 (including stock w = 8, keeping
                            // that arm bit-identical) and is dropped at w <= 5,
                            // where writing it would clear the run byte's MSB.
                            if (comptime !TS_RECIDX and !TS_DIVIDX and !TS_ESCIDX and !TS_DIV2IDX) {
                                if (comptime 6 * W >= 3 * W + 16) self.stSet(nodeAt(off, 6), 0);
                            }
                        }
                        const c1: u32 = x.c4 & 0xff;
                        // run-count update applies to the PREVIOUS runp[i] location.
                        const rp = self.runp[i];
                        const rc = self.runCnt(rp);
                        if (rc == 0) {
                            self.runCntSet(rp, 2);
                            self.runByteSet(rp, c1);
                        } else if (self.runByte(rp) != c1) {
                            self.runCntSet(rp, 1);
                            self.runByteSet(rp, c1);
                        } else if (rc < 254) {
                            self.runCntSet(rp, rc + 2);
                        }
                        self.runp[i] = rp0;
                        if (comptime TS_DIV2IDX) {
                            const c1w: u32 = x.c4 & 0xff;
                            const h16: u32 = (c1w *% 0x9E3779B1) >> 28; // 4 bits -> 16 buckets
                            const oldp = self.divp[i];
                            if (oldp != 0) {
                                if (h16 < 8)
                                    self.stSet(oldp, self.stGet(oldp) | (@as(u32, 1) << @as(u5, @intCast(h16))))
                                else
                                    self.stSet(oldp + W_BYTES, self.stGet(oldp + W_BYTES) |
                                        (@as(u32, 1) << @as(u5, @intCast(h16 - 8))));
                            }
                            const dp = nodeAt(off, 5);
                            if (comptime TS_D2CTRL) {
                                const h: u32 = (self.cxt[i] *% 0x85EBCA6B) ^ (self.cxt[i] >> 15);
                                self.divb[i] = @intCast(@as(usize, h) % TS_D2Q);
                            } else {
                                const pc: u32 = @popCount(@as(u8, @truncate(self.stGet(dp)))) +
                                    @popCount(@as(u8, @truncate(self.stGet(dp + W_BYTES))));
                                self.divb[i] = @intCast(@min(@as(usize, pc), TS_D2Q - 1));
                            }
                            self.divp[i] = dp;
                        }
                        if (comptime TS_ESCIDX) {
                            const c1e: u32 = x.c4 & 0xff;
                            const hb: u32 = @as(u32, 1) << @as(u5, @intCast(c1e >> 5));
                            const oldp = self.divp[i];
                            if (oldp != 0) self.stSet(oldp, self.stGet(oldp) | hb);
                            const dp = nodeAt(off, 6);
                            self.escm[i] = @truncate(self.stGet(dp));
                            self.divp[i] = dp;
                        }
                        if (comptime TS_DIVIDX) {
                            // Fold the byte that followed the PREVIOUS occurrence
                            // into that slot's sketch (exactly the run model's own
                            // update discipline, two lines above).
                            const c1d: u32 = x.c4 & 0xff;
                            const hb: u32 = @as(u32, 1) << @as(u5, @intCast((c1d *% 0x9E3779B1) >> 29));
                            const oldp = self.divp[i];
                            if (oldp != 0) self.stSet(oldp, self.stGet(oldp) | hb);
                            const dp = nodeAt(off, 6);
                            if (comptime TS_DCTRL) {
                                const h: u32 = (self.cxt[i] *% 0x85EBCA6B) ^ (self.cxt[i] >> 15);
                                self.divb[i] = @intCast(@as(usize, h) % TS_DQ);
                            } else {
                                const pc: u32 = @popCount(@as(u8, @truncate(self.stGet(dp))));
                                self.divb[i] = @intCast(@min(@as(usize, pc), TS_DQ - 1));
                            }
                            self.divp[i] = dp;
                        }
                        if (comptime TS_RECIDX) {
                            const rp6 = nodeAt(off, 6);
                            const st6: u32 = self.stGet(rp6);
                            const tick: u32 = @intCast((self.nbytes >> TS_RSHIFT) & 0xFF);
                            if (comptime TS_RCTRL) {
                                const h: u32 = (self.cxt[i] *% 0xC2B2AE35) ^ (self.cxt[i] >> 11);
                                self.recb[i] = @intCast(@as(usize, h) % TS_RQ);
                            } else if (st6 == 0) {
                                self.recb[i] = 0; // never stamped
                            } else {
                                const d: u32 = (tick -% (st6 - 1)) & 0xFF;
                                var bkt: usize = 0;
                                var dd: u32 = d;
                                while (dd != 0 and bkt + 2 < TS_RQ) : (bkt += 1) dd >>= 1;
                                self.recb[i] = @intCast(bkt + 1);
                            }
                            self.stSet(rp6, (tick +% 1) & 0xFF);
                        }
                        if (comptime TS_VISIDX) {
                            // Node 5 of a bpos-0 slot: allocated, in this very
                            // cache line, and never read by the lineage.
                            const vp = nodeAt(off, 5);
                            const v: u32 = self.stGet(vp);
                            if (comptime TS_VCTRL) {
                                const h: u32 = (self.cxt[i] *% 0x9E3779B1) ^ (self.cxt[i] >> 13);
                                self.visb[i] = @intCast(@as(usize, h) % TS_VQ);
                            } else {
                                var bkt: usize = 0;
                                var vv: u32 = v;
                                while (vv != 0 and bkt + 1 < TS_VQ) : (bkt += 1) vv >>= 1;
                                self.visb[i] = @intCast(bkt);
                            }
                            if (v < 255) self.stSet(vp, v + 1);
                        }
                    }
                    s = @intCast(self.stGet(self.cp[i].?));
                }
                // -Dcm3-ts-runidx: the readout index. Both quantities are already
                // loaded -- the `b`/`runCnt` pair below reads the very same words.
                var kq: usize = 0;
                if (comptime TS_VISIDX) kq = self.visb[i];
                if (comptime TS_RECIDX) kq = kq * TS_RQ + self.recb[i];
                if (comptime TS_DIVIDX) kq = kq * TS_DQ + self.divb[i];
                if (comptime TS_DIV2IDX) kq = kq * TS_D2Q + self.divb[i];
                if (comptime TS_ESCIDX) {
                    const mk = self.escm[i];
                    const e: usize = if (comptime TS_ECTRL)
                        (@as(usize, (self.cxt[i] *% 0xD6E8FEB8) >> 28) % TS_EQ)
                    else if (mk == 0) 4 else CONSIST[@intCast(x.c0 & 0xff)][mk];
                    kq = kq * TS_EQ + e;
                }
                if (comptime TS_RUNIDX) {
                    const rb: i32 = x.c0shift_bpos ^
                        (@as(i32, @intCast(self.runByte(self.runp[i]))) >> @as(u5, @intCast(x.bposshift)));
                    const rc: u32 = self.runCnt(self.runp[i]);
                    if (comptime TS_CTRL) {
                        const h: u32 = (self.cxt[i] *% 0x9E3779B1) ^ (self.cxt[i] >> 13);
                        kq = @as(usize, h) % TS_Q;
                    } else {
                        const cls: usize = if (rc == 0) 0 else if (rb == 0) 1 else if (rb == 1) 2 else 3;
                        const nrun: usize = TS_Q / 4; // 1, 2, 4 or 8 run buckets
                        kq = cls * nrun + @min(@as(usize, rc >> 1), nrun - 1);
                    }
                }
                self.result += self.mix3(sink, s, i, kq);
                const b: i32 = x.c0shift_bpos ^ (@as(i32, @intCast(self.runByte(self.runp[i]))) >> @as(u5, @intCast(x.bposshift)));
                if (b <= 1) {
                    const bb = b * 256;
                    self.add(sink, self.rcpr[@intCast(@as(i32, @intCast(self.runCnt(self.runp[i]))) + bb)]);
                } else {
                    self.add(sink, 0);
                }
            }
        }
        if (x.bpos == 7) {
            if (comptime TS_RECIDX) self.nbytes +%= 1;
            self.cn = 0;
            self.cxt_mask = 0;
        }
        return self.result;
    }
};

// ---------------------------------------------------------------------------
// Tests (ported from cm3.rs `mod tests`; fixtures from cm3_*.bin goldens, which
// are the exact tables dumped in cm3_tables.rs).
// ---------------------------------------------------------------------------

const testing = std.testing;

fn loadI16(comptime path: []const u8, comptime n: usize) [n]i16 {
    @setEvalBranchQuota(n * 8 + 1000);
    const bytes = @embedFile(path);
    var arr: [n]i16 = undefined;
    for (0..n) |i| {
        arr[i] = @bitCast(@as(u16, bytes[i * 2]) | (@as(u16, bytes[i * 2 + 1]) << 8));
    }
    return arr;
}

fn loadU8(comptime path: []const u8) [1024]u8 {
    const bytes = @embedFile(path);
    var arr: [1024]u8 = undefined;
    @memcpy(&arr, bytes[0..1024]);
    return arr;
}

const STRT = loadI16("goldens/cm3_STRT.bin", 4096);
const ST2_P1 = loadI16("goldens/cm3_ST2_P1.bin", 4096);
const RCPR = loadI16("goldens/cm3_RCPR.bin", 512);
const STA1 = loadU8("goldens/cm3_STA1.bin");
const STA2 = loadU8("goldens/cm3_STA2.bin");
const STA4 = loadU8("goldens/cm3_STA4.bin");
const STA5 = loadU8("goldens/cm3_STA5.bin");
const STA6 = loadU8("goldens/cm3_STA6.bin");
const STA7 = loadU8("goldens/cm3_STA7.bin");

fn sta(s: i32) *const [1024]u8 {
    return switch (s) {
        1 => &STA1,
        2 => &STA2,
        4 => &STA4,
        5 => &STA5,
        6 => &STA6,
        else => &STA7,
    };
}

inline fn get_state_byte_location(bpos: i32, c0: i32) i32 {
    const smask: i32 = @intCast((@as(u32, 0x31031010) >> @as(u5, @intCast(bpos << 2))) & 0x0F);
    return smask + (c0 & smask);
}

/// Identical synthetic drive to scratchpad/cm3_oracle.cpp::run_cfg.
fn run_cfg(a: std.mem.Allocator, mem: u32, c: usize, cms: i32, cms3: i32, sta_id: i32, cms4: i32, kep: u8, skip2: i32) u64 {
    var cm = ContextMap3.new();
    cm.init(a, mem, c, cms, cms3, cms4, sta(sta_id), kep, skip2, &ST2_P1, &RCPR, &STRT);
    defer cm.deinit();
    var x = X{ .c0 = 1 };
    var cs: u64 = 0;
    var r: u32 = 0x12345678;
    var blpos: u32 = 0;
    // Test-local sink stands in for the predictor's in1/slots; reset per mix so
    // in1_buf[0..in1_n] is exactly the old per-mix emitted sequence.
    var in1_buf: [544]i16 = undefined;
    var slot_buf: [560]i16 = undefined;
    var in1_n: usize = 0;
    var slot_n: usize = 0;
    const sink = Sink{ .in1 = &in1_buf, .in1_n = &in1_n, .slots = &slot_buf, .slot_n = &slot_n };
    var byte_i: usize = 0;
    while (byte_i < 4000) : (byte_i += 1) {
        r = r *% 1664525 +% 1013904223;
        const by: i32 = @intCast((r >> 16) & 0xff);
        var b: i32 = 7;
        while (b >= 0) : (b -= 1) {
            const bit = (by >> @as(u5, @intCast(b))) & 1;
            x.y = bit;
            x.c0 += x.c0 + x.y;
            if (x.c0 >= 256) {
                x.c4 = (x.c4 << 8) + (@as(u32, @intCast(x.c0)) & 0xff);
                x.c0 = 1;
                blpos += 1;
            }
            x.bpos = (x.bpos + 1) & 7;
            x.bposshift = 7 - x.bpos;
            x.c0shift_bpos = (x.c0 << 1) ^ (@as(i32, 256) >> @as(u5, @intCast(x.bposshift)));
            x.cm_bit_state = get_state_byte_location(x.bpos, x.c0);
            if (x.bpos == 0) {
                var j: usize = 0;
                while (j < c) : (j += 1) {
                    const salt = (@as(u32, @intCast(j)) *% 0x9e3779b1) ^ x.c4;
                    if (salt & 7 == 0) {
                        cm.sets();
                    } else {
                        cm.set(((x.c4 & 0x3ff) *% 1000003) +% (@as(u32, @intCast(j)) *% 40503));
                    }
                }
            }
            _ = &blpos;
            in1_n = 0;
            slot_n = 0;
            _ = cm.mix(&x, &sink);
            for (in1_buf[0..in1_n]) |v| {
                cs = cs *% 1000003 +% @as(u64, @as(u16, @bitCast(v)));
            }
        }
    }
    return cs;
}

// ★ C-PACK G1 — THE BY-CONSTRUCTION CHECK, as a test rather than a claim.
//
// The findings report's own instrument gate G1 is "slots/bucket S(w) =
// floor(1016/(16+7w)) gives S = 14 at w = 8 — exactly the shipped E1<14,128>".
// The decoupled design generalises it to S(w,p) = floor(1016/(16+7w+p)) with an
// explicit p-bit priority. This asserts the whole row set the design is priced
// on, INCLUDING the geometry actually compiled into this build, and that every
// configuration fits a 128-byte bucket with its metadata.
//
// It also pins the two dimensions the report says BRACKET the optimum: (5,4) is
// the primary arm at 18 slots, and both (4,4) — narrower state — and (5,3) —
// narrower priority — were measured adverse offline, so a future edit that
// silently moves those geometries would invalidate the comparison.
test "cm3pack: the slot formula reproduces the shipped E1<14,128> at w=8" {
    const S = struct {
        fn f(w: usize, p: usize) usize {
            return 1016 / (16 + 7 * w + p);
        }
    }.f;
    // THE reproduction: stock geometry, from the formula alone.
    try testing.expectEqual(@as(usize, 14), S(8, 0));
    // ...and the shipped bucket really is that: chk[14] u16 = 28 B, `last` = 1 B,
    // bh[14][7] = 98 B  ==  127 B inside a 128-byte element (cm3.zig:7-8).
    try testing.expectEqual(@as(usize, 127), 14 * 2 + 1 + 14 * 7);
    try testing.expect(14 * 2 + 1 + 14 * 7 <= 128);

    // The design's own table (findings §6.8): 5-bit state + 4-bit priority = 18.
    try testing.expectEqual(@as(usize, 18), S(5, 4));
    try testing.expectEqual(@as(usize, 16), S(6, 4));
    try testing.expectEqual(@as(usize, 21), S(4, 4)); // adverse offline (+0.375 %)
    try testing.expectEqual(@as(usize, 18), S(5, 3)); // adverse offline (+0.03..+0.27 %)
    try testing.expectEqual(@as(usize, 14), S(7, 4));
    // Narrowing WITHOUT the priority — the known-adverse configuration (+1.86 %).
    try testing.expectEqual(@as(usize, 17), S(6, 0));
    try testing.expectEqual(@as(usize, 19), S(5, 0));

    // The compiled build agrees with the formula, and its layout closes.
    try testing.expectEqual(S(W, PB), SLOTS);
    try testing.expect(SLOTS >= 14);
    if (comptime PACK) {
        try testing.expect(ST0 + SLOTS * 7 * W <= 1024); // fits the 128 B bucket
        try testing.expect(4 * W >= 16); // the 16-bit run model fits nodes 3..6
        try testing.expect(SLOTS <= NO_SLOT); // slot index fits the 5-bit meta field
    } else {
        try testing.expectEqual(@as(usize, 14), SLOTS); // default build is stock
    }
}

// The packed bit-array accessors: every state/priority/run field must survive a
// write→read at every slot and node of a real bucket, INCLUDING the ones that
// straddle byte boundaries (at w=5 only 1 node in 8 is byte-aligned). A silent
// off-by-one here would not crash — it would quietly corrupt one model's history
// and show up only as a worse ratio, which is exactly the failure this family
// must not make.
test "cm3pack: packed accessors round-trip at every slot and node" {
    if (comptime PACK) {
        const a = testing.allocator;
        var cm = ContextMap3.new();
        cm.init(a, 1 << 16, 1, 28, 32, 8, sta(6), 0, 0, &ST2_P1, &RCPR, &STRT);
        defer cm.deinit();
        const eb = ContextMap3.elemBase(3); // an arbitrary interior bucket
        // states
        var i: usize = 0;
        while (i < SLOTS) : (i += 1) {
            var j: usize = 0;
            while (j < 7) : (j += 1) {
                const v: u32 = @intCast((i * 7 + j) % NSTATES);
                cm.stSet(ContextMap3.nodeAt(ContextMap3.slotBase(eb, i), j), v);
            }
        }
        i = 0;
        while (i < SLOTS) : (i += 1) {
            var j: usize = 0;
            while (j < 7) : (j += 1) {
                const want: u32 = @intCast((i * 7 + j) % NSTATES);
                try testing.expectEqual(want, cm.stGet(ContextMap3.nodeAt(ContextMap3.slotBase(eb, i), j)));
            }
        }
        // the run model (16 bits at node 3 of a slot), and chk/meta/priority
        i = 0;
        while (i < SLOTS) : (i += 1) {
            const rp = ContextMap3.nodeAt(ContextMap3.slotBase(eb, i), 3);
            cm.runCntSet(rp, 254);
            cm.runByteSet(rp, 0xA5);
            try testing.expectEqual(@as(u32, 254), cm.runCnt(rp));
            try testing.expectEqual(@as(u32, 0xA5), cm.runByte(rp));
            cm.chk_set(eb, i, @intCast(0x1234 +% i));
            try testing.expectEqual(@as(u16, @intCast(0x1234 +% i)), cm.chk_get(eb, i));
            if (comptime PB != 0) {
                var k: u32 = 0;
                while (k < PRI_CAP + 3) : (k += 1) cm.priBump(eb, i);
                try testing.expectEqual(@as(i32, @intCast(PRI_CAP)), cm.priGet(eb, i)); // saturates
                cm.priReset(eb, i);
                try testing.expectEqual(@as(i32, 0), cm.priGet(eb, i));
            }
            cm.metaTouch(eb, 0, i, 0);
            try testing.expectEqual(i, ContextMap3.metaR0(cm.metaRaw(eb)));
        }
        // the run field must NOT have been disturbed by the neighbouring writes
        i = 0;
        while (i < SLOTS) : (i += 1) {
            const rp = ContextMap3.nodeAt(ContextMap3.slotBase(eb, i), 3);
            try testing.expectEqual(@as(u32, 254), cm.runCnt(rp));
            try testing.expectEqual(@as(u32, 0xA5), cm.runByte(rp));
        }
        // ...and no write may have escaped into the neighbouring buckets.
        for (cm.t[0..128]) |b| try testing.expectEqual(@as(u8, 0), b);
        for (cm.t[512..640]) |b| try testing.expectEqual(@as(u8, 0), b);
    }
}

test "cm3_matches_cpp_oracle" {
    const Case = struct {
        name: []const u8,
        mem: u32,
        c: usize,
        cms: i32,
        cms3: i32,
        sta_id: i32,
        cms4: i32,
        kep: u8,
        skip2: i32,
        golden: u64,
    };
    const cases = [_]Case{
        .{ .name = "cmC2[0]", .mem = 1 << 16, .c = 3, .cms = 28, .cms3 = 43, .sta_id = 6, .cms4 = 9, .kep = 0xf0, .skip2 = 1, .golden = 0xb565f18432398d0d },
        .{ .name = "cmC2[1]", .mem = 1 << 16, .c = 1, .cms = 32, .cms3 = 32, .sta_id = 6, .cms4 = 8, .kep = 0xf0, .skip2 = 0, .golden = 0x4bcdd54366ee8f03 },
        .{ .name = "cmC2[6]", .mem = 1 << 16, .c = 1, .cms = 33, .cms3 = 32, .sta_id = 1, .cms4 = 15, .kep = 0x00, .skip2 = 1, .golden = 0xf98bc83ceff17c5b },
        .{ .name = "cmC2[7]", .mem = 1 << 16, .c = 1, .cms = 33, .cms3 = 33, .sta_id = 5, .cms4 = 8, .kep = 0xf0, .skip2 = 1, .golden = 0x131bdf9121e361ef },
        .{ .name = "cmC2[8]", .mem = 1 << 16, .c = 4, .cms = 35, .cms3 = 37, .sta_id = 4, .cms4 = 8, .kep = 0x00, .skip2 = 1, .golden = 0x19a627fccbabcc8e },
        .{ .name = "cmC[0]", .mem = 1 << 16, .c = 7, .cms = 34, .cms3 = 35, .sta_id = 2, .cms4 = 8, .kep = 0x00, .skip2 = 1, .golden = 0xda6b0b4a8281a7ba },
        .{ .name = "cmC[1]", .mem = 1 << 16, .c = 3, .cms = 30, .cms3 = 28, .sta_id = 5, .cms4 = 8, .kep = 0xf0, .skip2 = 0, .golden = 0xe00033b1d7b69333 },
        .{ .name = "cmcr1", .mem = 1 << 16, .c = 3, .cms = 34, .cms3 = 35, .sta_id = 6, .cms4 = 8, .kep = 0x00, .skip2 = 0, .golden = 0xc30231a0ac78831c },
        .{ .name = "big-mem", .mem = 1 << 22, .c = 3, .cms = 28, .cms3 = 43, .sta_id = 6, .cms4 = 9, .kep = 0xf0, .skip2 = 1, .golden = 0x9fd81e419cb56825 },
        .{ .name = "cmC2[5]-C6", .mem = 1 << 16, .c = 6, .cms = 31, .cms3 = 29, .sta_id = 6, .cms4 = 12, .kep = 0xf0, .skip2 = 1, .golden = 0x6cbeb70b84f580a3 },
        .{ .name = "cmC2[11]-C5", .mem = 1 << 16, .c = 5, .cms = 32, .cms3 = 32, .sta_id = 5, .cms4 = 8, .kep = 0xf0, .skip2 = 1, .golden = 0x629c17881774482e },
        .{ .name = "cmC2[2]-cms4_12", .mem = 1 << 16, .c = 1, .cms = 32, .cms3 = 32, .sta_id = 6, .cms4 = 12, .kep = 0xf0, .skip2 = 0, .golden = 0xe429d02459eafb17 },
        .{ .name = "cmC[2]-cms4_13", .mem = 1 << 16, .c = 2, .cms = 36, .cms3 = 30, .sta_id = 2, .cms4 = 13, .kep = 0xf0, .skip2 = 0, .golden = 0xa33c9608879519bf },
    };
    for (cases) |cse| {
        const got = run_cfg(testing.allocator, cse.mem, cse.c, cse.cms, cse.cms3, cse.sta_id, cse.cms4, cse.kep, cse.skip2);
        testing.expectEqual(cse.golden, got) catch |e| {
            std.debug.print("ContextMap3 config {s} diverged: got {x:0>16} want {x:0>16}\n", .{ cse.name, got, cse.golden });
            return e;
        };
    }
}

// Proves the -Dcm3-diag false-merge detector actually fires (independent of
// scale — at 50k the big tables are too sparse to collide, so this is the unit
// evidence that a 0 there is real emptiness, not a stuck counter). Runs only
// under `zig build test -Dcm3-diag=true`; a no-op otherwise.
test "cm3_diag detects a 16-bit-checksum false merge" {
    // Wrapped in `if (comptime DIAG)` (not an early return) so the body — which
    // touches self.diag fields that only exist under the flag — is not even
    // analyzed in a stock `zig build test`. No-op there; real assertions under
    // `zig build test -Dcm3-diag=true`.
    if (comptime DIAG) {
        const a = testing.allocator;
        var cm = ContextMap3.new();
        // mem=64 ⇒ tmask=0 ⇒ a single bucket at idx 0, which is always sampled
        // ((idx & 0xFF)==0). All probes below hit that one bucket.
        cm.init(a, 64, 1, 28, 32, 8, sta(6), 0, 0, &ST2_P1, &RCPR, &STRT);
        defer cm.deinit();

        const chk: u16 = 0x1234; // one fixed checksum ⇒ all probes land in one slot
        const cxt_a: u32 = 0xAAAAAAAA;
        const cxt_b: u32 = 0xBBBBBBBB;

        // (1) context A claims a slot (a replacement records owner=A).
        _ = cm.bucket_get(0, chk, 0, cxt_a);
        try testing.expectEqual(@as(u64, 1), cm.diag.replacements);
        try testing.expectEqual(@as(u64, 0), cm.diag.sample_true);
        try testing.expectEqual(@as(u64, 0), cm.diag.sample_false);

        // (2) context B probes the same bucket with the SAME checksum but a
        //     DIFFERENT full context ⇒ chk matches A's slot ⇒ a false merge.
        _ = cm.bucket_get(0, chk, 0, cxt_b);
        try testing.expectEqual(@as(u64, 1), cm.diag.sample_false);
        try testing.expectEqual(@as(u64, 0), cm.diag.sample_true);

        // (3) context A revisits ⇒ chk matches its OWN slot ⇒ a true match, and
        //     the false count is unchanged (the detector distinguishes the two).
        _ = cm.bucket_get(0, chk, 0, cxt_a);
        try testing.expectEqual(@as(u64, 1), cm.diag.sample_true);
        try testing.expectEqual(@as(u64, 1), cm.diag.sample_false);
        // No spurious replacement on the matches.
        try testing.expectEqual(@as(u64, 1), cm.diag.replacements);
    }
}
// ===========================================================================
// hash64 attribution fixture (aug-06). Three gaps this closes, in order:
//
//  (1) NO fixture ever covered `checksum` itself. The pre-existing detector
//      test above INJECTS a chk into `bucket_get`, so it proves the detector
//      fires but says nothing about which checksum the build compiled. That is
//      why "-Dcmc2-hash64 may not be engaged" could not be excluded from the
//      desk. These tests assert KNOWN VALUES for both variants and are written
//      so that the WRONG variant fails — the wiring proof `zig build test
//      -Dcmc2-hash64=true|false` was missing.
//  (2) The generating mechanism of the measured false merges: the bucket index
//      is `(cxt + c0) & tmask` while chk is `f(cxt)` alone ⇒ **chk is BLIND to
//      c0**, so a context at partial byte k shares a bucket with `cxt-d` at
//      byte k+d for c0 in [1,256). Under STOCK chk = (cxt>>16)^i those two
//      collide whenever their top 16 bits agree; hash64 decorrelates them.
//  (3) The selftest rule: known-value fixtures AT REAL SCALE plus a
//      NEGATIVE CONTROL that must not fire.
//
// Real scale here means the real big-table geometry: tmask = 2^22-1 (a 512 MB
// bank, the size of cm_c2[1]/[4]/[5]/[13]), where the stock chk's bits 16..21
// overlap the index — the exact configuration the C-HASH64 comment describes.
const H64_M1_512MB: u32 = 0x1000_0000; // (m1*2)>>7 - 1 == 0x3FFFFF

test "cm3 checksum: known values pin the compiled variant (wiring proof)" {
    // Hand-computed from the two formulas; the compiled build must reproduce
    // its OWN column exactly and must NOT reproduce the other one.
    const cases = [_]struct { cxt: u32, i: u32, stock: u16, h64: u16 }{
        .{ .cxt = 0x12345678, .i = 0, .stock = 0x1234, .h64 = 0xC19C },
        .{ .cxt = 0x12345678, .i = 3, .stock = 0x1237, .h64 = 0xC19C },
        .{ .cxt = 0xFFFFFFFF, .i = 7, .stock = 0xFFF8, .h64 = 0x0C9E },
        .{ .cxt = 0x00000000, .i = 0, .stock = 0x0000, .h64 = 0x0000 },
        .{ .cxt = 0xDEADBEEF, .i = 5, .stock = 0xDEA8, .h64 = 0x265A },
        .{ .cxt = 0x0000FFFF, .i = 1, .stock = 0x0001, .h64 = 0x3D42 },
    };
    var differ: usize = 0;
    for (cases) |c| {
        const want: u16 = if (comptime HASH64) c.h64 else c.stock;
        const other: u16 = if (comptime HASH64) c.stock else c.h64;
        try testing.expectEqual(want, checksum(c.cxt, c.i));
        if (want != other) differ += 1;
    }
    // At least one case separates the two formulas, so a build that silently
    // ignored -Dcmc2-hash64 cannot pass this test.
    try testing.expect(differ >= 4);
    // Banked side-fingerprint: under hash64 the slot index `i` is added BEFORE
    // the >>16, so chk is very nearly i-INDEPENDENT (row 1 vs row 2 above);
    // under stock, `^i` lands in the low bits and always separates slots.
    if (comptime HASH64) {
        try testing.expectEqual(checksum(0x12345678, 0), checksum(0x12345678, 3));
    } else {
        try testing.expect(checksum(0x12345678, 0) != checksum(0x12345678, 3));
    }
}

test "cm3 checksum is c0-blind: cross-c0 same-bucket collision, at real geometry" {
    // Behavioural, through the real `bucket_get` + the real `-Dcm3-diag`
    // detector, with the chk taken from the real `checksum` — no injection.
    if (comptime DIAG) {
        const a = testing.allocator;
        var cm = ContextMap3.new();
        cm.init(a, H64_M1_512MB, 1, 28, 32, 8, sta(6), 0, 0, &ST2_P1, &RCPR, &STRT);
        defer cm.deinit();
        try testing.expectEqual(@as(u32, 0x3FFFFF), cm.tmask);

        // The cross-c0 pair: A at c0=4 and B = A+3 at c0=1 land in ONE bucket
        // because the index is (cxt + c0) & tmask. The detector only watches
        // buckets with (idx & 0xFF) == 0, so A is chosen with (A + 4) & 0xFF == 0.
        const A: u32 = 0x123456FC;
        const B: u32 = A +% 3; // 0x123456FF, probed at c0 = 1
        const bkt: usize = @intCast((A +% 4) & cm.tmask);
        try testing.expectEqual(bkt, @as(usize, @intCast((B +% 1) & cm.tmask)));
        try testing.expectEqual(@as(usize, 0), bkt & 0xFF); // sampled bucket
        const eb: usize = bkt * 128;

        const ch_a = checksum(A, 0);
        const ch_b = checksum(B, 0);
        // The whole mechanism in one assertion: STOCK cannot tell A from B
        // (identical top 16 bits), hash64 can.
        if (comptime HASH64) {
            try testing.expect(ch_a != ch_b);
        } else {
            try testing.expectEqual(ch_a, ch_b);
        }

        // A claims a slot; B then probes the SAME bucket with ITS OWN chk.
        _ = cm.bucket_get(eb, ch_a, 0, A);
        try testing.expectEqual(@as(u64, 1), cm.diag.replacements);
        _ = cm.bucket_get(eb, ch_b, 0, B);
        if (comptime HASH64) {
            // POSITIVE-CONTROL-BY-ABSENCE: hash64 must MISS and allocate.
            try testing.expectEqual(@as(u64, 0), cm.diag.sample_false);
            try testing.expectEqual(@as(u64, 2), cm.diag.replacements);
        } else {
            // Stock: B reads A's bit history — a false merge, and it is
            // top-16-equal AND a near-neighbour (the signature the aug-06
            // fingerprint counters look for on the branch
            // claude/hash64-diag-fingerprint-aug06).
            try testing.expectEqual(@as(u64, 1), cm.diag.sample_false);
            try testing.expectEqual(@as(u64, 1), cm.diag.replacements);
        }

        // NEGATIVE CONTROL: C = A + 2^22 shares the bucket at the SAME c0 (the
        // index masks bit 22 away) but differs in the top 16 bits, so NEITHER
        // checksum may merge it. If this ever fires, the detector is broken.
        const C: u32 = A +% (1 << 22);
        try testing.expectEqual(bkt, @as(usize, @intCast((C +% 4) & cm.tmask)));
        try testing.expect(checksum(C, 0) != ch_a);
        const false_before = cm.diag.sample_false;
        _ = cm.bucket_get(eb, checksum(C, 0), 0, C);
        try testing.expectEqual(false_before, cm.diag.sample_false);

        // TRUE-MATCH CONTROL: A revisits its own slot ⇒ sample_true, not false.
        _ = cm.bucket_get(eb, ch_a, 0, A);
        try testing.expectEqual(@as(u64, 1), cm.diag.sample_true);
        try testing.expectEqual(false_before, cm.diag.sample_false);
    }
}

test "cm3 checksum: cross-c0 population collision rate (the 1.6% vs 0.02% split)" {
    // The population that generates false merges, enumerated: two contexts
    // share a bucket iff cxt2 = cxt1 + (c01 - c02) + k*(tmask+1). Draw it and
    // count chk collisions with the COMPILED checksum. Known-value bands:
    //   stock  — collide iff the top 16 bits survive, i.e. iff k == 0 and no
    //            carry crosses bit 16 => ~1/1024 = 0.0977 % per pair.
    //   hash64 — decorrelated => 2^-16 = 0.00153 % per pair.
    // x14 slots per probe: stock ~1.37 %/probe, hash64 ~0.021 %/probe. The
    // second number is the "aliasing floor" the aug-06 C1 leg measured at
    // 0.0209 % on a real hash64 binary.
    const N: u32 = 1 << 21;
    const TMASK: u32 = 0x3FFFFF; // 512 MB bank
    var rng = std.Random.DefaultPrng.init(0x5eed_1234);
    const r = rng.random();
    var hit: u32 = 0;
    var n: u32 = 0;
    while (n < N) : (n += 1) {
        const cxt1 = r.int(u32);
        const c01 = 1 + r.uintLessThan(u32, 255);
        var c02 = 1 + r.uintLessThan(u32, 255);
        if (c02 == c01) c02 = 1 + ((c01) % 255);
        const k = r.uintLessThan(u32, 1024); // the free bits above tmask
        const cxt2 = cxt1 +% (c01 -% c02) +% (k *% (TMASK + 1));
        std.debug.assert(((cxt1 +% c01) & TMASK) == ((cxt2 +% c02) & TMASK));
        if (checksum(cxt1, 0) == checksum(cxt2, 0)) hit += 1;
    }
    const rate = 100.0 * @as(f64, @floatFromInt(hit)) / @as(f64, @floatFromInt(N));
    std.debug.print("\n[h64fix] cross-c0 same-bucket chk collision rate = {d:.5} % per pair, {d:.4} % per 14-slot probe (hash64={})\n", .{ rate, rate * 14.0, HASH64 });
    if (comptime HASH64) {
        try testing.expect(rate < 0.01); // floor: 0.00153 %
    } else {
        try testing.expect(rate > 0.05 and rate < 0.20); // ~0.0977 %
    }

    // And the part hash64 does NOT fix: the SAME context probed at two
    // different c0 lands in two different buckets carrying ONE chk. That is
    // 100 % identical under both variants — the c1assoc leg's pairing-L
    // finding, and the reason a wider/better chk cannot separate those.
    try testing.expectEqual(checksum(0x0BADF00D, 2), checksum(0x0BADF00D, 2));
}
