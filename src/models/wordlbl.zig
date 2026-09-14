// ===========================================================================
// -Dwordlbl: the WORD-IDENTITY DISTRIBUTED-REPRESENTATION expert.
// ===========================================================================
// Every predictor in the PAQ → lpaq → zpaq → cmix → zmix lineage estimates word
// identity with hashed-context LOOKUP TABLES: all 44,515 dictionary words are
// learned independently, and nothing learned about one word transfers to another
// that behaves alike. This module adds the one parameterisation that DOES share:
// an online LOG-BILINEAR context model (Mnih & Hinton 2007) read out through a
// HIERARCHICAL SOFTMAX over the WRT codeword's own mixed-radix digit tree
// (Morin & Bengio 2005) — cost O(d) per coded bit, never O(V).
//
// Evidence chain (measure-before-build, in order):
//  (offline, −1.7%)
//  (σ vs the REAL
//     shipped ensemble: −0.1714 % cumulative / plateau ~−0.19 % on the 29-flag
//     recipe at 13.1 M tokens; φ = 0.6617 ⇒ e9 −123…−136 KB; §11)
// This file is a PORT OF THE MEASURED INSTRUMENT — wordlbl.c's LBL expert with
// the swept optimum hyperparameters (η_N=0.060, η_E=0.030; D from -Dwordlbl-d,
// default 96 = 97.7 % of D=192's value at half the RAM). Any divergence from the
// instrument is a bug unless annotated.
//
// Integration: ONE layer-0 input, pad slot 559 — the fxcm_neutral pad
// [1+active..560] is already MAC'd and weight-updated by all 23 mixers every bit
// (the pad is the only free real estate, zero
// extra wall and zero extra weight-row RAM). Slot 560 is avoided because mixer1
// consumes it separately (fxcm_model_index). -Dwordlbl-attn-head adds TWO more
// pad inputs at 557/558 (still zero weight-row RAM).
//
// Determinism: no libm — squash goes through vendored exp exactly like
// lstm.zig's expF; the attention head's stretch goes through the vendored musl
// logf (src/vendor/crt/log.zig), the SAME implementation detm_math.zig exports
// as the binary's libm symbol; embedding init is a fixed-seed xorshift
// (bit-exact, no entropy); everything else is f32 arithmetic. Nothing outside
// the binary: no files, no env vars, no OS facility, no absolute paths.
//
// The whole type is comptime-void unless -Dwordlbl (own options module
// "wordlbl_opts" ⇒ S cost of the dead knob is EXACTLY ZERO, the own-module rule).
//
// V2 (-Dwordlbl-ctx=N -Dwordlbl-lambda=L -Dwordlbl-eta0=E): the offline-
// validated tied-decay config (
// addendum §2/§3): ONE shared embedding table, cvec = 0.25·topic +
// Σ_{s=0}^{N-1} λ^s·E[w_{t-1-s}], token-end update λ^s-weighted per row, and a
// separate η for byte-0 trie nodes. Port of scurve.c agg=2 + SC_ETA0.
// MEASURED IN-ENGINE: −40,754 B @e8_100m over shipped v1 on the 38-flag
//
// V2 aggregation cost note (why cvec is REBUILT per token from the ring rather
// than maintained as an O(D)/word incremental EMA): the offline optimum is a
// TRUNCATED L-word window over CURRENT embedding values — an incremental EMA
// computes a different model (infinite tail past word N, λ^N/(1−λ) ≈ 0.49
// relative weight at N=16/λ=.85, and STALE embedding values, material when 77 %
// of the measured capture is online tracking). The exact form cannot be O(D)
// anyway: the token-end training update must touch all N rows (λ^s-weighted
// clamped gradients), so the rebuild rides the same O(N·D) memory traffic —
// ~3.1 K madds/word at D=192/N=16 ≈ 1e-4 of the engine's per-byte budget.
// Any divergence from the instrument here would be a divergence from the
// number that authorised the build.
//
// ===========================================================================
// V2-ATTN (-Dwordlbl-attn): the CONTENT-ADDRESSED INDUCTION MEMORY.
// ===========================================================================
// Port of the offline champion. The instrument is `experiments/attn-word-expert/
// attn.c`, aggregation mode 7 at the finals config:
//     AT_N=128 AT_DK=64 AT_ETAW=0.003 AT_QW1=1 AT_KW1=1 AT_HEAD=1
//     AT_TOPK=16 AT_HTEMP=2   (config `f_champ192`, σ_all −0.4666 %)
// on the v2 tied-decay state (λ=.85, L=16, η .045/.0225, η0 .0225, D=192),
// = +23.7 % relative over tied-decay alone (−0.3886 %).
//
// Mechanism, in one paragraph. The tied-decay state s can only see ~7 words
// back at its interior λ*. The memory is a ring of N slots; slot j was written
// at token j and holds (key_j = Wk^T s_j + Wk1^T E[w_j], state_j = s_j,
// value_j = w_j) — the state that PRECEDED word w_j, paired with w_j itself.
// Reading it with query q = Wq^T s + Wq1^T E[w_1] is INDUCTION retrieval: "the
// last time the context looked like this, which word followed?". The learned
// recency-bucket bias brec[log2 age] measured POSITIVE again at ages 32–256 —
// the memory retrieves from far beyond the decay horizon, which is exactly the
// information the decay state cannot carry. The retrieved value mixture enters
// the context as γ·attout with γ learned (it climbs 0.3 → 0.89–1.09 offline).
//
// The candidate head (-Dwordlbl-attn-head) is a SECOND READOUT of the same
// memory with ZERO new state: the top-12 attention slots are a distribution
// over WORD IDS; map each id back to its codeword bytes (rank2bytes, 5 integer
// ops) and, per coded bit, sum the mass of candidates whose codeword prefix
// matches what has been coded so far. That sharp identity mass is something the
// D-dimensional value mixture cannot express. HTEMP=2 squares the masses before
// renormalising — the two readouts want different temperatures.
//
// DIVERGENCES FROM THE INSTRUMENT (declared, as the v2 port declared its own):
//  A. `expf` → `expF` (vendored f64 exp cast to f32) in the softmax, matching
//     this module's existing squashF. ≤1 ulp.
//  B. `logf` → `crt_log.logf` (musl port) in the head's stretch. This is the
//     SAME symbol detm_math.zig exports, so it is what a glibc-ABI ship build
//     would have called anyway; only a non-detm host libm would differ.
//  C. `powf(x, 2.0f)` → `x*x` for HTEMP=2. Both are the correctly-rounded
//     square of an f32, so this is an identity, not an approximation; HTEMP is
//     comptime-restricted to {1.0, 2.0} precisely so it stays one.
//  D. The projection matrices are stored TRANSPOSED ([dk][D] instead of the
//     instrument's [D][dk]). This is a pure layout change: every accumulation
//     keeps its k-ascending / d2-ascending order, so every value is
//     bit-identical to the instrument's. It exists because it turns the four
//     hot loops (query build, key insertion, Wk/Wk1 top-K update) from
//     stride-dk gathers into contiguous scans — the top-K Wk/Wk1 update alone
//     is 396 K of the 560 K MACs/token, and it is the only thing that decides
//     whether the arm is affordable at ReleaseSmall.
//  E. Ring value ids are the rank-CLAMPED ri = min(r, VMAX−1) (v2 divergence 4;
//     max real rank 44,879 < 45,056, so identical in practice). Candidate
//     codeword bytes are kept as u32, not u8, because rank2bytes is only
//     surjective up to rank 44,879 and the instrument stores them in `int`; a
//     u8 would trap in a safe build on an out-of-range rank that can never
//     match a real prefix byte anyway.
//  F. Mixer-input SCALE. The instrument multiplies EVERY one of its inputs by
//     0.30 (its single mixer row is calibrated on that). The engine's layer-0
//     convention is a raw logit — v1/v2 already drop the 0.30 from slot 559 —
//     so the head's logit input drops it too: z_head = stretch(ph)·wsh. The
//     coverage input keeps the literal 0.30·wsh of the port plan (§7.4). Scale
//     on a mixer input is an adaptation-rate choice, not a model choice: the
//     engine's 23 context-gated mixers relearn the weight either way.
// ===========================================================================
const std = @import("std");
const build_options = @import("build_options");
const wordlbl_opts = @import("wordlbl_opts");

pub const WL_ON: bool = wordlbl_opts.wordlbl;
pub const WL_D: usize = wordlbl_opts.wordlbl_d;
pub const WL_ETA_N: f32 = wordlbl_opts.wordlbl_etan;
pub const WL_ETA_E: f32 = wordlbl_opts.wordlbl_etae;
// ---- V2 (offline-validated config, +
// addendum; port of scurve.c agg=2 tieddecay + SC_ETA0 per-depth eta) ----
//   WL_CTX > 0 : context = previous WL_CTX words from ONE shared embedding
//                table (e1), tied-decay weights lambda^s, s=0 = most recent
//                (instrument scurve.c:273-279). e2/e3 are then not allocated.
//   WL_ETA0    : eta_N for byte-0 trie nodes (nid 1..127) only — the addendum
//                §2 by-catch (scurve.c:328 `eta_here=(bp==0)?eta0:etaN`).
// Both comptime-dead at defaults: WL_CTX=0 and WL_ETA0==WL_ETA_N reproduce v1
// EXACTLY (the base-leg bit-identity gate proves it).
pub const WL_CTX: usize = wordlbl_opts.wordlbl_ctx;
pub const WL_LAMBDA: f32 = wordlbl_opts.wordlbl_lambda;
pub const WL_ETA0: f32 = wordlbl_opts.wordlbl_eta0;
const V2: bool = WL_CTX > 0;

// ---- V2-ATTN (attn.c agg=7 at the `f_champ192` config) --------------------
const ATTN: bool = wordlbl_opts.wordlbl_attn;
/// True iff the candidate head is live — predictor_lex reads this to decide
/// whether pad slots 557/558 are claimed.
pub const WL_HEAD: bool = ATTN and wordlbl_opts.wordlbl_attn_head;
const AN: usize = if (ATTN) wordlbl_opts.wordlbl_attn_n else 0; // AT_N   ring slots
const ADK: usize = if (ATTN) wordlbl_opts.wordlbl_attn_dk else 0; // AT_DK  key/query width
const ATOPK: usize = if (ATTN) wordlbl_opts.wordlbl_attn_topk else 0; // AT_TOPK
const AETAW: f32 = wordlbl_opts.wordlbl_attn_etaw; // AT_ETAW  projection lr
const AHTEMP: f32 = wordlbl_opts.wordlbl_attn_htemp; // AT_HTEMP head sharpening
const ASTATS: bool = ATTN and wordlbl_opts.wordlbl_attn_stats;
// attn.c's own defaults for the axes the finals never moved, frozen here as
// comptime constants (port plan §7.5: "N/dk/ηW/γ/brec/HTEMP comptime
// constants"). Each cites the instrument line that sets it.
const AG0: f32 = 0.3; // AT_G   — gamma init            (attn.c:137)
const AETAG: f32 = 0.005; // AT_ETAG — gamma lr            (attn.c:137)
const AETAB: f32 = 0.01; // AT_ETAB — recency-bias lr     (attn.c:137)
const AETAV: f32 = WL_ETA_E; // AT_ETAV<0 ⇒ etaE              (attn.c:268)
const ACAND: usize = 12; // head candidate slots          (attn.c:480)
const AAGEB: usize = 9; // recency buckets 1,2,4..256    (attn.c:111)
// 1/sqrt(dk) — comptime, so no runtime sqrt and no libm (attn.c:421).
const ISQ: f32 = if (ATTN) 1.0 / @sqrt(@as(f32, @floatFromInt(ADK))) else 0.0;

comptime {
    if (V2 and !(WL_LAMBDA > 0.0 and WL_LAMBDA < 1.0))
        @compileError("-Dwordlbl-ctx>0 requires -Dwordlbl-lambda in (0,1)");
    if (!V2 and wordlbl_opts.wordlbl_lambda_set)
        @compileError("-Dwordlbl-lambda has no effect without -Dwordlbl-ctx>0");
    if (WL_CTX > 64) @compileError("-Dwordlbl-ctx > 64 unsupported");
    if (ATTN and !V2)
        @compileError("-Dwordlbl-attn requires -Dwordlbl-ctx>0 (it extends the v2 tied-decay state)");
    if (ATTN and (AN < 1 or ADK < 1))
        @compileError("-Dwordlbl-attn-n and -Dwordlbl-attn-dk must be >= 1");
    if (ATTN and (ATOPK < 1 or ATOPK > 64))
        @compileError("-Dwordlbl-attn-topk must be in 1..64 (attn.c clamps at 64)");
    if (ATTN and !(AHTEMP == 1.0 or AHTEMP == 2.0))
        @compileError("-Dwordlbl-attn-htemp: only 1.0 and 2.0 are supported (2.0 == x*x exactly; " ++
            "any other exponent would need a runtime powf, which is a determinism exposure)");
}
// lambda^s by iterated f32 multiply (comptime IEEE f32 — deterministic, no
// libm; the instrument's runtime powf may differ in the last ulp, which is a
// declared instrument divergence, not a determinism exposure).
const LAMPOW: [WL_CTX]f32 = blk: {
    var t: [WL_CTX]f32 = undefined;
    var w: f32 = 1.0;
    for (&t) |*p| {
        p.* = w;
        w *= WL_LAMBDA;
    }
    break :blk t;
};
const NNODEBITS: u5 = 17;
const NNODEMASK: u32 = (@as(u32, 1) << NNODEBITS) - 1;
const VMAX: usize = 45056; // >= 44,880 codeword ranks (instrument line 78)

const use_vendored_libm = build_options.vendored_libm;
const vendored_math = @import("../vendored_math.zig");
// Pinned musl logf — the SAME implementation detm_math.zig exports as the
// binary's `logf`, called directly so table-free stretch is bit-identical in
// every build (the fxcm26/tables.zig idiom).
const crt_log = @import("../vendor/crt/log.zig");
inline fn expF(x: f32) f32 {
    return if (comptime use_vendored_libm)
        @floatCast(vendored_math.exp(x))
    else
        @exp(x);
}
inline fn squashF(x_in: f32) f32 {
    var x = x_in;
    if (x < -30.0) x = -30.0;
    if (x > 30.0) x = 30.0;
    return 1.0 / (1.0 + expF(-x));
}
/// attn.c:50-52 `stretchf` — logit with the instrument's own guard rails.
inline fn stretchF(p_in: f32) f32 {
    var p = p_in;
    if (p < 1e-6) p = 1e-6;
    if (p > 1.0 - 1e-6) p = 1.0 - 1e-6;
    return crt_log.logf(p / (1.0 - p));
}
// instrument hsh: splitmix-style avalanche then mask (an AND, not a modulo —
// the bytepos-2 call site masks with a NON-power-of-2 value and that is faithful)
inline fn hsh(x_in: u64, mask: u32) u32 {
    var x = x_in;
    x *%= 0x9E3779B97F4A7C15;
    x ^= x >> 29;
    x *%= 0xBF58476D1CE4E5B9;
    x ^= x >> 32;
    return @as(u32, @truncate(x)) & mask;
}
// node id: exact for byte 0 and byte 1, hashed for byte 2 (instrument :167-172)
inline fn nodeid(bytepos: u2, b0: u32, b1: u32, nodeix: u32) u32 {
    if (bytepos == 0) return nodeix; // 1..127
    if (bytepos == 1) return 128 + ((b0 - 128) * 128 + nodeix); // <= 16512
    return 16640 + hsh((@as(u64, b0) << 20) ^ (@as(u64, b1) << 8) ^ nodeix, NNODEMASK - 16640);
}
/// rank -> codeword bytes, the exact inverse of byteDone's parser (attn.c:106).
/// Surjective only up to rank 44,879; beyond that b[0] exceeds 0xFF and simply
/// never matches a coded prefix byte, exactly as in the instrument's `int` array.
inline fn rank2bytes(r: u32, b3: *[3]u32) u8 {
    if (r < 80) {
        b3[0] = 0x80 + r;
        return 1;
    }
    if (r < 3920) {
        const x = r - 80;
        b3[0] = 0xD0 + x / 80;
        b3[1] = 0x80 + x % 80;
        return 2;
    }
    const x = r - 3920;
    const q = x / 80;
    b3[0] = 0xF0 + q / 32;
    b3[1] = 0xD0 + q % 32;
    b3[2] = 0x80 + x % 80;
    return 3;
}
/// attn.c:111 — 1→0, 2→1, 3..4→2, ..., 129..256→8 (clamped at 8).
inline fn agebkt(age: usize) usize {
    var b: usize = 0;
    while ((@as(usize, 1) << @intCast(b)) < age and b < AAGEB - 1) b += 1;
    return b;
}
/// dst[i] -= src[i] * c, elementwise. Bit-identical to the scalar form (no FMA
/// contraction: Zig emits strict IEEE ops), written as an explicit vector so
/// ReleaseSmall's optsize attribute cannot decline to vectorise the loop that
/// carries 78 % of the arm's MACs.
inline fn axpySub(dst: []f32, src: []const f32, c: f32) void {
    const VL = 8;
    const cs: @Vector(VL, f32) = @splat(c);
    var i: usize = 0;
    while (i + VL <= dst.len) : (i += VL) {
        const d: @Vector(VL, f32) = dst[i..][0..VL].*;
        const s: @Vector(VL, f32) = src[i..][0..VL].*;
        dst[i..][0..VL].* = d - s * cs;
    }
    while (i < dst.len) : (i += 1) dst[i] -= src[i] * c;
}

pub const WordLbl = struct {
    const D = WL_D;

    // ---- learned state (the instrument's Nd/Nb/E1..E3/topic) ----
    nd: []f32, // [nodes][D]
    nb: []f32, // [nodes]
    e1: []f32, // [VMAX][D] — v2's ONE shared table; v1's position-1 table
    e2: if (V2) void else []f32, // v1 only: untied position-2/3 tables
    e3: if (V2) void else []f32,
    topic: [D]f32,

    // ---- per-token working state ----
    cvec: [D]f32,
    gacc: [D]f32,
    cvec_valid: bool = false,

    // ---- word history ----
    // v1: w1..w3 (three untied positions). v2: ring[0..WL_CTX), ring[0] = most
    // recent (the instrument's hist[]; shift-down at token end, scurve.c:437-438).
    w1: u32 = 0,
    w2: u32 = 0,
    w3: u32 = 0,
    ring: [WL_CTX]u32 = [_]u32{0} ** WL_CTX,

    // ---- V2-ATTN: the content-addressed memory (attn.c agg=7) --------------
    // Projections, stored TRANSPOSED as [ADK][D] (divergence D above).
    wqt: if (ATTN) []f32 else void, // Wq^T   query  <- decay state s
    wq1t: if (ATTN) []f32 else void, // Wq1^T  query  <- E[w_1]        (AT_QW1)
    wkt: if (ATTN) []f32 else void, // Wk^T   key    <- decay state s
    wk1t: if (ATTN) []f32 else void, // Wk1^T  key    <- E[own word]   (AT_KW1)
    // The ring itself.
    ring_k: if (ATTN) []f32 else void, // [AN][ADK] keys
    ring_s: if (ATTN) []f32 else void, // [AN][D]   state snapshots
    ring_v: if (ATTN) []u32 else void, // [AN]      values (word ids)
    ringn: usize = 0, // slots filled
    ringpos: usize = 0, // next write slot
    gam: f32 = AG0, // learned output gain on the retrieved mixture
    brec: if (ATTN) [AAGEB]f32 else void, // learned log2-age recency bias
    // Per-token scratch (all rebuilt from scratch each token; no hidden state).
    sq: if (ATTN) [D]f32 else void, // the tied-decay state WITHOUT topic
    qv: if (ATTN) [ADK]f32 else void, // query
    dqv: if (ATTN) [ADK]f32 else void, // dL/dquery
    tbuf: if (ATTN) [ADK]f32 else void, // (ds*q[d2])*isq, per selected slot
    att: if (ATTN) [AN]f32 else void, // softmax weights
    attsc: if (ATTN) [AN]f32 else void, // pre-softmax scores
    datt: if (ATTN) [AN]f32 else void, // dL/d att_j
    dsc: if (ATTN) [AN]f32 else void, // dL/d score_j
    ao: if (ATTN) [D]f32 else void, // the retrieved value mixture
    gbuf: if (ATTN) [D]f32 else void, // frozen copy of gacc (see tokenDone)
    gqb: if (ATTN) [D]f32 else void, // query-path gradient back into the state
    sbuf: if (ATTN) [D]f32 else void, // AETAW-scaled left factor for a W update
    // Candidate head (AT_HEAD): a second readout, ZERO extra model state.
    cbid: if (WL_HEAD) [ACAND]u32 else void, // candidate word ids
    cby: if (WL_HEAD) [ACAND][3]u32 else void, // their codeword bytes
    cln: if (WL_HEAD) [ACAND]u8 else void, // their codeword lengths
    cpb: if (WL_HEAD) [ACAND]f32 else void, // their masses
    ncand: usize = 0,
    /// Layer-0 pad input 558: stretch(candidate-mass bit prob) * mass.
    z_head: f32 = 0,
    /// Layer-0 pad input 557: 0.30 * mass (the head's own coverage signal).
    c_head: f32 = 0,
    // -Dwordlbl-attn-stats diagnostics (comptime-dead by default).
    st_toks: if (ASTATS) u64 else void,
    st_amax: if (ASTATS) f64 else void,
    st_cov: if (ASTATS) u64 else void,
    st_hits: if (ASTATS) u64 else void,
    st_hbits: if (ASTATS) u64 else void,

    // ---- causal parse of the coded byte stream ----
    // The instrument reads the stream with lookahead; the engine must parse
    // CAUSALLY. Codeword geometry (verified, 0 parse failures over 64 MB):
    //   1 byte : b0 in [0x80,0xCF]
    //   2 bytes: b0 in [0xD0,0xFF], b1 in [0x80,0xCF]
    //   3 bytes: b0 in [0xF0,0xFF], b1 in [0xD0,0xFF], b2 in [0x80,0xCF]
    // (tiers 2 and 3 SHARE the b0 range [0xF0,0xFF]; b1 disambiguates.)
    // A byte's class is known after its FIRST coded bit (MSB): 1 ⇒ codeword
    // byte, 0 ⇒ literal — so bits 6..0 of every codeword byte are causally
    // modellable, exactly the bits the σ instrument scored.
    // 0x0C is the WRT escape: the byte AFTER a completed 0x0C is a literal even
    // if its MSB is 1 (instrument line 262 consumes the pair).
    breg: u32 = 1, // partial-byte register, 1 = byte boundary (SSE M_j convention)
    nbits_in: u5 = 0, // bits received for the current byte
    first_bit: u1 = 0,
    expecting: u2 = 0, // 0 = at token boundary; 1/2 = awaiting continuation byte
    pfx0: u32 = 0,
    pfx1: u32 = 0,
    escape_next: bool = false,

    // ---- per-bit state carried from forwardto perceive----
    nodeix: u32 = 1,
    live: bool = false, // did forward() produce a payload prediction this bit?
    nid: u32 = 0,
    plbl: f32 = 0.5,

    pub fn init(a: std.mem.Allocator) !*WordLbl {
        const self = try a.create(WordLbl);
        errdefer a.destroy(self);
        const nodes: usize = @as(usize, NNODEMASK) + 1;
        self.* = .{
            .nd = try a.alloc(f32, nodes * D),
            .nb = try a.alloc(f32, nodes),
            .e1 = try a.alloc(f32, VMAX * D),
            .e2 = undefined,
            .e3 = undefined,
            .topic = [_]f32{0} ** D,
            .cvec = [_]f32{0} ** D,
            .gacc = [_]f32{0} ** D,
            .wqt = undefined,
            .wq1t = undefined,
            .wkt = undefined,
            .wk1t = undefined,
            .ring_k = undefined,
            .ring_s = undefined,
            .ring_v = undefined,
            .brec = if (comptime ATTN) [_]f32{0} ** AAGEB else {},
            .sq = if (comptime ATTN) [_]f32{0} ** D else {},
            .qv = if (comptime ATTN) [_]f32{0} ** ADK else {},
            .dqv = if (comptime ATTN) [_]f32{0} ** ADK else {},
            .tbuf = if (comptime ATTN) [_]f32{0} ** ADK else {},
            .att = if (comptime ATTN) [_]f32{0} ** AN else {},
            .attsc = if (comptime ATTN) [_]f32{0} ** AN else {},
            .datt = if (comptime ATTN) [_]f32{0} ** AN else {},
            .dsc = if (comptime ATTN) [_]f32{0} ** AN else {},
            .ao = if (comptime ATTN) [_]f32{0} ** D else {},
            .gbuf = if (comptime ATTN) [_]f32{0} ** D else {},
            .gqb = if (comptime ATTN) [_]f32{0} ** D else {},
            .sbuf = if (comptime ATTN) [_]f32{0} ** D else {},
            .cbid = if (comptime WL_HEAD) [_]u32{0} ** ACAND else {},
            .cby = if (comptime WL_HEAD) [_][3]u32{[_]u32{0} ** 3} ** ACAND else {},
            .cln = if (comptime WL_HEAD) [_]u8{0} ** ACAND else {},
            .cpb = if (comptime WL_HEAD) [_]f32{0} ** ACAND else {},
            .st_toks = if (comptime ASTATS) 0 else {},
            .st_amax = if (comptime ASTATS) 0.0 else {},
            .st_cov = if (comptime ASTATS) 0 else {},
            .st_hits = if (comptime ASTATS) 0 else {},
            .st_hbits = if (comptime ASTATS) 0 else {},
        };
        if (comptime !V2) {
            self.e2 = try a.alloc(f32, VMAX * D);
            self.e3 = try a.alloc(f32, VMAX * D);
        }
        @memset(self.nd, 0); // instrument: calloc
        @memset(self.nb, 0);
        // fixed-seed xorshift, EXACTLY the instrument's rnd(s is the static
        // inside rnd; the file-scope xs is dead there and dead here)
        var s: u64 = 0x243F6A8885A308D3;
        const rnd = struct {
            fn next(st: *u64) f64 {
                st.* ^= st.* << 13;
                st.* ^= st.* >> 7;
                st.* ^= st.* << 17;
                return @as(f64, @floatFromInt(st.* >> 11)) / 9007199254740992.0;
            }
        };
        const sc: f64 = 0.05; // mode-1 init scale (mode 4's 0.30 is the frozen-random control)
        if (comptime V2) {
            // instrument agg=2: ONE table, filled sequentially from the same
            // seed with NO burn (scurve.c:180-183, ntab=1) — bit-exact draws.
            for (self.e1) |*v| v.* = @floatCast((rnd.next(&s) - 0.5) * 2.0 * sc);
        } else {
            for ([_][]f32{ self.e1, self.e2, self.e3 }) |e| {
                for (e) |*v| v.* = @floatCast((rnd.next(&s) - 0.5) * 2.0 * sc);
            }
        }
        if (comptime ATTN) {
            const wsz = D * ADK;
            self.wqt = try a.alloc(f32, wsz);
            self.wq1t = try a.alloc(f32, wsz);
            self.wkt = try a.alloc(f32, wsz);
            self.wk1t = try a.alloc(f32, wsz);
            self.ring_k = try a.alloc(f32, AN * ADK);
            self.ring_s = try a.alloc(f32, AN * D);
            self.ring_v = try a.alloc(u32, AN);
            @memset(self.ring_k, 0); // instrument: calloc
            @memset(self.ring_s, 0);
            @memset(self.ring_v, 0);
            // Draw order is the instrument's EXACTLY (attn.c:259-264): Wq and
            // Wk INTERLEAVED over one linear index, then all of Wq1, then all
            // of Wk1 — continuing the same rndstream the E table left off.
            // Linear index p means logical (k = p/ADK, d2 = p%ADK), which the
            // transposed layout stores at [d2*D + k].
            var p: usize = 0;
            while (p < wsz) : (p += 1) {
                const t = (p % ADK) * D + (p / ADK);
                self.wqt[t] = @floatCast((rnd.next(&s) - 0.5) * 2.0 * sc);
                self.wkt[t] = @floatCast((rnd.next(&s) - 0.5) * 2.0 * sc);
            }
            p = 0;
            while (p < wsz) : (p += 1)
                self.wq1t[(p % ADK) * D + (p / ADK)] = @floatCast((rnd.next(&s) - 0.5) * 2.0 * sc);
            p = 0;
            while (p < wsz) : (p += 1)
                self.wk1t[(p % ADK) * D + (p / ADK)] = @floatCast((rnd.next(&s) - 0.5) * 2.0 * sc);
        }
        return self;
    }

    pub fn deinit(self: *WordLbl, a: std.mem.Allocator) void {
        if (comptime ASTATS) self.dumpStats();
        a.free(self.nd);
        a.free(self.nb);
        a.free(self.e1);
        if (comptime !V2) {
            a.free(self.e2);
            a.free(self.e3);
        }
        if (comptime ATTN) {
            a.free(self.wqt);
            a.free(self.wq1t);
            a.free(self.wkt);
            a.free(self.wk1t);
            a.free(self.ring_k);
            a.free(self.ring_s);
            a.free(self.ring_v);
        }
        a.destroy(self);
    }

    /// -Dwordlbl-attn-stats: the instrument's ATT/CHEAD diagnostic lines, so an
    /// engine run can be checked against the offline mechanism (gamma should
    /// climb 0.3 → ~0.9-1.1; brec should go positive again at ages 32-256).
    /// Comptime-dead by default; no env var, no file.
    fn dumpStats(self: *WordLbl) void {
        if (comptime !ASTATS) return;
        const n: f64 = @floatFromInt(@max(self.st_toks, 1));
        std.debug.print(
            "WLBL-ATTN toks {d} amax {d:.4} gamma {d:.4} headcov {d:.4} hitrate {d:.4} brec",
            .{
                self.st_toks,
                self.st_amax / n,
                self.gam,
                @as(f64, @floatFromInt(self.st_cov)) / n,
                @as(f64, @floatFromInt(self.st_hits)) / @as(f64, @floatFromInt(@max(self.st_hbits, 1))),
            },
        );
        for (self.brec) |b| std.debug.print(" {d:.3}", .{b});
        std.debug.print("\n", .{});
    }

    fn buildCvec(self: *WordLbl) void {
        if (comptime ATTN) {
            self.buildCvecAttn();
            self.cvec_valid = true;
            return;
        }
        if (comptime V2) {
            // instrument agg=2 (scurve.c:273-279): topic term first, then the
            // tied-decay sum over the ring, s ascending — same f32 order.
            // Rebuilt ONCE per token (lazily, exactly like v1): the exact
            // aggregate over CURRENT embedding values, as measured offline.
            // A pure O(D)/word incremental EMA was NOT used — see the module
            // header note on the v2 aggregation cost.
            for (0..D) |k| self.cvec[k] = 0.25 * self.topic[k];
            for (0..WL_CTX) |sx| {
                const w = LAMPOW[sx];
                const e = self.e1[@as(usize, self.ring[sx]) * D ..][0..D];
                for (0..D) |k| self.cvec[k] += w * e[k];
            }
        } else {
            const p1 = self.e1[@as(usize, self.w1) * D ..][0..D];
            const p2 = self.e2[@as(usize, self.w2) * D ..][0..D];
            const p3 = self.e3[@as(usize, self.w3) * D ..][0..D];
            for (0..D) |k| self.cvec[k] = p1[k] + p2[k] + p3[k] + 0.25 * self.topic[k];
        }
        self.cvec_valid = true;
    }

    /// attn.c:414-500 (agg 7 forward), once per token at the token boundary.
    fn buildCvecAttn(self: *WordLbl) void {
        if (comptime !ATTN) return;
        // (1) the tied-decay state s — NOTE: attn.c keeps `topic` OUT of s, so
        // the query/key see the pure decay state (attn.c:415-420).
        for (0..D) |k| self.sq[k] = 0.0;
        for (0..WL_CTX) |sx| {
            const w = LAMPOW[sx];
            const e = self.e1[@as(usize, self.ring[sx]) * D ..][0..D];
            for (0..D) |k| self.sq[k] += w * e[k];
        }
        const m = self.ringn;
        const e1r = self.e1[@as(usize, self.ring[0]) * D ..][0..D];
        // (2) query q = Wq^T s + Wq1^T E[w_1]   (attn.c:431-439)
        for (0..ADK) |d2| {
            const wrow = self.wqt[d2 * D ..][0..D];
            var acc: f32 = 0;
            for (0..D) |k| acc += wrow[k] * self.sq[k];
            const w1row = self.wq1t[d2 * D ..][0..D];
            for (0..D) |k| acc += w1row[k] * e1r[k];
            self.qv[d2] = acc;
        }
        // (3) scores = q·k/sqrt(dk) + brec[log2 age]   (attn.c:440-449)
        var mx: f32 = -1e30;
        for (0..m) |j| {
            const kj = self.ring_k[j * ADK ..][0..ADK];
            var sc: f32 = 0;
            for (0..ADK) |d2| sc += self.qv[d2] * kj[d2];
            const age = ((self.ringpos + AN - 1 - j) % AN) + 1;
            sc = sc * ISQ + self.brec[agebkt(age)];
            self.attsc[j] = sc;
            if (sc > mx) mx = sc;
        }
        // (4) softmax, max-subtracted (deterministic)   (attn.c:450-455)
        var ssum: f32 = 0;
        for (0..m) |j| {
            const ex = expF(self.attsc[j] - mx);
            self.att[j] = ex;
            ssum += ex;
        }
        if (m > 0) {
            const inv: f32 = 1.0 / ssum;
            for (0..m) |j| self.att[j] *= inv;
        }
        // (5) retrieved value mixture, sparse over a >= 1e-4  (attn.c:456-462)
        for (0..D) |k| self.ao[k] = 0.0;
        for (0..m) |j| {
            const a = self.att[j];
            if (a < 1e-4) continue;
            const ev = self.e1[@as(usize, self.ring_v[j]) * D ..][0..D];
            for (0..D) |k| self.ao[k] += a * ev[k];
        }
        // (6) cvec = s + 0.25*topic + gamma*attout   (attn.c:463-471; AT_H=1,
        // AT_VG=0, AT_CAT=0, AT_PURE=0 all hold at the champion config, so
        // attsum == attv == gamma*ao and the two zero-init temporaries fold out)
        const g = self.gam;
        for (0..D) |k| self.cvec[k] = (self.sq[k] + 0.25 * self.topic[k]) + g * self.ao[k];
        if (comptime ASTATS) {
            var amax: f32 = 0;
            for (0..m) |j| {
                if (self.att[j] > amax) amax = self.att[j];
            }
            self.st_toks += 1;
            self.st_amax += amax;
        }
        // (7) the candidate head: a SECOND readout of the same distribution
        if (comptime WL_HEAD) self.buildHead(m);
    }

    /// attn.c:475-500 — top-12 attention slots → per-word candidate masses.
    fn buildHead(self: *WordLbl, m: usize) void {
        if (comptime !WL_HEAD) return;
        self.ncand = 0;
        var cconf: f32 = 0;
        // top-12 by attention weight. The instrument's selection is
        // ORDER-DEPENDENT (fill the first 12 in ring order, then replace the
        // FIRST argmin); replicated exactly, ties included.
        var sj: [ACAND]usize = undefined;
        var sw: [ACAND]f32 = undefined;
        var ns: usize = 0;
        for (0..m) |j| {
            const w = self.att[j]; // AT_H == 1 ⇒ (0 + att[j]) / 1.0f
            if (ns < ACAND) {
                sj[ns] = j;
                sw[ns] = w;
                ns += 1;
            } else {
                var mi: usize = 0;
                for (1..ns) |t2| {
                    if (sw[t2] < sw[mi]) mi = t2;
                }
                if (w > sw[mi]) {
                    sj[mi] = j;
                    sw[mi] = w;
                }
            }
        }
        for (0..ns) |t2| {
            if (self.ncand >= ACAND) break;
            const id = self.ring_v[sj[t2]];
            const w = sw[t2];
            var dup: ?usize = null;
            for (0..self.ncand) |q2| {
                if (self.cbid[q2] == id) dup = q2;
            }
            if (dup) |q| {
                self.cpb[q] += w; // the same word can occupy several slots
                cconf += w;
                continue;
            }
            self.cbid[self.ncand] = id;
            self.cln[self.ncand] = rank2bytes(id, &self.cby[self.ncand]);
            self.cpb[self.ncand] = w;
            cconf += w;
            self.ncand += 1;
        }
        // HTEMP: sharpen the head relative to the value path, mass-preserving.
        if (comptime AHTEMP != 1.0) {
            if (self.ncand > 0 and cconf > 1e-9) {
                var tsum: f32 = 0;
                for (0..self.ncand) |q2| {
                    self.cpb[q2] = self.cpb[q2] * self.cpb[q2]; // powf(x,2) == x*x
                    tsum += self.cpb[q2];
                }
                if (tsum > 1e-12) {
                    const sc2: f32 = cconf / tsum;
                    for (0..self.ncand) |q2| self.cpb[q2] *= sc2;
                }
            }
        }
        if (comptime ASTATS) {
            if (self.ncand > 0) self.st_cov += 1;
        }
    }

    /// The layer-0 input for THIS coded bit (call once per bit, before mixing).
    /// Returns the LBL logit on codeword payload bits, exactly 0.0 elsewhere —
    /// so with the expert quiescent the pad slot holds the same 0.0 it held
    /// stock, and the null arm is bit-identical by construction. With
    /// -Dwordlbl-attn-head it ALSO sets z_head/c_head (pad slots 558/557), which
    /// are likewise exactly 0.0 on every quiescent bit.
    pub fn forward(self: *WordLbl) f32 {
        self.live = false;
        if (comptime WL_HEAD) {
            self.z_head = 0.0;
            self.c_head = 0.0;
        }
        if (self.nbits_in == 0) return 0.0; // the byte's MSB: marker/literal-msb, never payload
        if (self.escape_next) return 0.0;
        const bp: u2 = self.expecting;
        if (bp == 0 and self.first_bit == 0) return 0.0; // literal byte
        // payload bit. cvec is per-token; (re)build lazily.
        if (!self.cvec_valid) self.buildCvec();
        const nid = nodeid(bp, self.pfx0, self.pfx1, self.nodeix);
        var zl: f32 = self.nb[nid];
        const nv = self.nd[@as(usize, nid) * D ..][0..D];
        for (0..D) |k| zl += self.cvec[k] * nv[k];
        if (zl > 20.0) zl = 20.0;
        if (zl < -20.0) zl = -20.0;
        self.nid = nid;
        self.plbl = squashF(zl);
        self.live = true;
        if (comptime WL_HEAD) self.headInputs(bp);
        return zl;
    }

    /// attn.c:572-596 — the candidate head's two mixer inputs for this bit.
    /// The instrument's `bit` (6..0) is `7 - nbits_in` here; its `nodeix` is
    /// ours verbatim. Divergence F (module header): the uniform 0.30 the
    /// instrument puts on EVERY mixer input is dropped from the logit path,
    /// matching slot 559's engine convention.
    fn headInputs(self: *WordLbl, bp: u2) void {
        if (comptime !WL_HEAD) return;
        if (self.ncand == 0) return;
        const bitpos: u5 = 7 - self.nbits_in; // 6..0
        const bpi: usize = bp;
        var m1: f32 = 0;
        var m0: f32 = 0;
        for (0..self.ncand) |q2| {
            if (self.cln[q2] <= bpi) continue; // candidate is shorter than this byte
            if (bpi >= 1 and self.cby[q2][0] != self.pfx0) continue;
            if (bpi >= 2 and self.cby[q2][1] != self.pfx1) continue;
            const cb: u32 = self.cby[q2][bpi];
            if ((cb >> (bitpos + 1)) != self.nodeix) continue; // prefix mismatch
            if ((cb >> bitpos) & 1 != 0) m1 += self.cpb[q2] else m0 += self.cpb[q2];
        }
        const mm = m0 + m1;
        if (mm > 1e-6) {
            var ph: f32 = m1 / mm;
            if (ph < 1e-4) ph = 1e-4;
            if (ph > 1.0 - 1e-4) ph = 1.0 - 1e-4;
            self.z_head = stretchF(ph) * mm; // wsh == mm (agg 7: no cconf factor)
            self.c_head = 0.30 * mm;
            if (comptime ASTATS) self.st_hbits += 1;
        }
    }

    /// Train on the observed bit and advance the parse. Call once per coded bit,
    /// after forward. Runs on EVERY coded bit (including adaptive-depth skips
    /// — the σ measurement trained on those too; the mixer merely ignores the
    /// input there).
    pub fn perceive(self: *WordLbl, bit: i32) void {
        const y: u1 = @intCast(bit & 1);
        if (self.live) {
            if (comptime ASTATS) {
                if (self.z_head != 0.0 and ((self.z_head > 0) == (y == 1))) self.st_hits += 1;
            }
            const g: f32 = self.plbl - @as(f32, @floatFromInt(y));
            const nv = self.nd[@as(usize, self.nid) * D ..][0..D];
            // -Dwordlbl-eta0: per-depth eta_N. Byte-0 trie nodes are EXACTLY
            // nid 1..127 (nodeid: bp0 -> nodeix; bp1 -> >=128; bp2 -> >=16640),
            // so nid<128 is the instrument's `(bp==0)?eta0:etaN` (scurve.c:328).
            // Comptime-dead when WL_ETA0 == WL_ETA_N (the default).
            const eta_n: f32 = if (comptime WL_ETA0 != WL_ETA_N)
                (if (self.nid < 128) WL_ETA0 else WL_ETA_N)
            else
                WL_ETA_N;
            // instrument order: gacc from the OLD node vector, then descend it
            for (0..D) |k| {
                self.gacc[k] += g * nv[k];
                nv[k] -= eta_n * g * self.cvec[k];
            }
            self.nb[self.nid] -= eta_n * g;
            self.nodeix = self.nodeix * 2 + y;
            self.live = false;
        }
        // ---- advance the byte register ----
        if (self.nbits_in == 0) self.first_bit = y;
        self.breg = (self.breg << 1) | y;
        self.nbits_in += 1;
        if (self.nbits_in < 8) return;
        // ---- byte complete ----
        const byte: u32 = self.breg & 0xFF;
        self.breg = 1;
        self.nbits_in = 0;
        self.nodeix = 1;
        self.byteDone(byte);
    }

    fn byteDone(self: *WordLbl, byte: u32) void {
        if (self.escape_next) {
            // the escaped literal: consumed, no parse effect
            self.escape_next = false;
            return;
        }
        switch (self.expecting) {
            0 => {
                if (byte == 0x0C) {
                    self.escape_next = true;
                } else if (byte < 0x80) {
                    // literal: nothing (the LBL uses only word history)
                } else if (byte <= 0xCF) {
                    self.tokenDone(byte - 0x80); // 1-byte codeword, rank 0..79
                } else {
                    self.pfx0 = byte; // 0xD0..0xFF: continuation follows
                    self.expecting = 1;
                }
            },
            1 => {
                if (byte >= 0x80 and byte <= 0xCF) {
                    // 2-byte codeword complete
                    self.tokenDone(80 + (self.pfx0 - 0xD0) * 80 + (byte - 0x80));
                } else if (self.pfx0 >= 0xF0 and byte >= 0xD0) {
                    self.pfx1 = byte;
                    self.expecting = 2;
                } else {
                    self.parseAbort();
                }
            },
            2 => {
                if (byte >= 0x80 and byte <= 0xCF) {
                    const q = 80 * (32 * (self.pfx0 - 0xF0) + (self.pfx1 - 0xD0));
                    self.tokenDone(3920 + q + (byte - 0x80));
                } else {
                    self.parseAbort();
                }
            },
            else => unreachable,
        }
    }

    fn parseAbort(self: *WordLbl) void {
        // invalid continuation: the instrument's r<0 path treats the bytes as
        // literals and never reaches token-end. Discard the aborted token's
        // gradient and reset. (Measured 0 parse failures over 64 MB, so this is
        // armour, not a hot path.) The attention memory is NOT pushed and the
        // history is NOT shifted — exactly the instrument's `continue`.
        self.expecting = 0;
        self.cvec_valid = false;
        if (comptime WL_HEAD) self.ncand = 0;
        @memset(&self.gacc, 0);
    }

    fn tokenDone(self: *WordLbl, r: u32) void {
        self.expecting = 0;
        const ri: usize = @min(@as(usize, r), VMAX - 1);
        if (comptime ATTN) {
            // attn.c:721-830 — the attention backprop runs BEFORE the decay
            // window update, because it folds the query-path gradient INTO
            // gacc, which the decay update then consumes.
            self.attnBackward();
        }
        if (comptime V2) {
            // v2 token-end embedding update (instrument agg=2, scurve.c:396-409
            // minus the posbag-only alpha branch): each ring word's SHARED-table
            // row descends the lambda^s-WEIGHTED gradient, clamped per element
            // AFTER weighting; duplicates in the ring update sequentially,
            // exactly as the instrument's per-s row walk does.
            for (0..WL_CTX) |sx| {
                const w = LAMPOW[sx];
                const e = self.e1[@as(usize, self.ring[sx]) * D ..][0..D];
                for (0..D) |k| {
                    var gk: f32 = WL_ETA_E * w * self.gacc[k];
                    if (gk > 0.5) gk = 0.5;
                    if (gk < -0.5) gk = -0.5;
                    e[k] -= gk;
                }
            }
            // topic EMA from the completed token's row, read AFTER the updates
            // (instrument order, scurve.c:432-433)
            const er = self.e1[ri * D ..][0..D];
            for (0..D) |k| self.topic[k] += 0.004 * (er[k] - self.topic[k]);
            // attention memory push: (the state that PREDICTED r) -> r. Comes
            // after the topic EMA and before the history shift (attn.c:894-913):
            // key and value both read the POST-update embedding table.
            if (comptime ATTN) self.attnPush(ri);
            // history shift-down (instrument scurve.c:437-438)
            var sx: usize = WL_CTX - 1;
            while (sx > 0) : (sx -= 1) self.ring[sx] = self.ring[sx - 1];
            self.ring[0] = @intCast(ri);
        } else {
            // v1 token-end embedding update (instrument :409-420), from the
            // token's accumulated hierarchical-softmax gradient
            const p1 = self.e1[@as(usize, self.w1) * D ..][0..D];
            const p2 = self.e2[@as(usize, self.w2) * D ..][0..D];
            const p3 = self.e3[@as(usize, self.w3) * D ..][0..D];
            for (0..D) |k| {
                var gk: f32 = WL_ETA_E * self.gacc[k];
                if (gk > 0.5) gk = 0.5;
                if (gk < -0.5) gk = -0.5;
                p1[k] -= gk;
                p2[k] -= gk;
                p3[k] -= gk;
            }
            const er = self.e1[ri * D ..][0..D];
            for (0..D) |k| self.topic[k] += 0.004 * (er[k] - self.topic[k]);
            self.w3 = self.w2;
            self.w2 = self.w1;
            self.w1 = @intCast(ri);
        }
        @memset(&self.gacc, 0);
        self.cvec_valid = false;
    }

    /// attn.c:721-830 — one head, no BPTT: the softmax jacobian, the recency
    /// bias, the top-K value/Wk/Wk1 grads, Wq/Wq1, and the query-path gradient
    /// folded back into the decay-state gradient.
    fn attnBackward(self: *WordLbl) void {
        if (comptime !ATTN) return;
        const m = self.ringn;
        // gbuf is a FROZEN copy: the Wq update below writes into gacc, and the
        // decay-window update must see the modified gacc while every attention
        // gradient must see the unmodified one (attn.c:733).
        for (0..D) |k| self.gbuf[k] = self.gacc[k];
        // gamma: the readout gain. `gmh` is the PRE-update value — the harness
        // rev-2 fix (report §1), and the true gradient.
        const gmh = self.gam;
        var dgm: f32 = 0;
        for (0..D) |k| dgm += self.gbuf[k] * self.ao[k];
        self.gam -= AETAG * dgm;
        if (self.gam > 4.0) self.gam = 4.0;
        if (self.gam < -4.0) self.gam = -4.0;
        // dL/d att_j and the softmax normaliser S
        var s_norm: f32 = 0;
        for (0..m) |j| {
            const a = self.att[j];
            if (a < 1e-5) {
                self.datt[j] = 0;
                self.dsc[j] = 0;
                continue;
            }
            const ev = self.e1[@as(usize, self.ring_v[j]) * D ..][0..D];
            var acc: f32 = 0;
            for (0..D) |k| acc += self.gbuf[k] * ev[k];
            self.datt[j] = gmh * acc;
            s_norm += a * self.datt[j];
        }
        // dL/d score_j via the softmax jacobian; recency bias; dL/d query
        for (0..ADK) |d2| self.dqv[d2] = 0;
        for (0..m) |j| {
            const a = self.att[j];
            if (a < 1e-5) continue;
            const ds = a * (self.datt[j] - s_norm);
            self.dsc[j] = ds;
            const age = ((self.ringpos + AN - 1 - j) % AN) + 1;
            self.brec[agebkt(age)] -= AETAB * ds;
            const kj = self.ring_k[j * ADK ..][0..ADK];
            for (0..ADK) |d2| self.dqv[d2] += ds * kj[d2] * ISQ;
        }
        // Value-embedding + Wk/Wk1 grads for the TOP-K slots by |dscore|.
        // Selection is the instrument's order-dependent fill-then-replace-first-
        // argmin, replicated exactly.
        {
            var selj: [ATOPK]usize = undefined;
            var selm: [ATOPK]f32 = undefined;
            var ns: usize = 0;
            for (0..m) |j| {
                const mag = @abs(self.dsc[j]) + self.att[j] * 1e-3;
                if (mag <= 0.0) continue;
                if (ns < ATOPK) {
                    selj[ns] = j;
                    selm[ns] = mag;
                    ns += 1;
                } else {
                    var mi: usize = 0;
                    for (1..ns) |t2| {
                        if (selm[t2] < selm[mi]) mi = t2;
                    }
                    if (mag > selm[mi]) {
                        selj[mi] = j;
                        selm[mi] = mag;
                    }
                }
            }
            for (0..ns) |t2| {
                const j = selj[t2];
                const a = self.att[j];
                const ds = self.dsc[j];
                const ev = self.e1[@as(usize, self.ring_v[j]) * D ..][0..D];
                // value embedding (clamped exactly like every other emb update)
                const cva: f32 = (AETAV * gmh) * a;
                for (0..D) |k| {
                    var gk: f32 = cva * self.gbuf[k];
                    if (gk > 0.5) gk = 0.5;
                    if (gk < -0.5) gk = -0.5;
                    ev[k] -= gk;
                }
                // t[d2] = (ds * q[d2]) * isq — loop-invariant in k
                for (0..ADK) |d2| self.tbuf[d2] = (ds * self.qv[d2]) * ISQ;
                // Wk: left factor is the slot's STATE SNAPSHOT
                const sjrow = self.ring_s[j * D ..][0..D];
                for (0..D) |k| self.sbuf[k] = AETAW * sjrow[k];
                for (0..ADK) |d2| axpySub(self.wkt[d2 * D ..][0..D], &self.sbuf, self.tbuf[d2]);
                // Wk1: left factor is the slot's own word embedding, read AFTER
                // the value update above (attn.c:789 reads ev[k] post-descent)
                for (0..D) |k| self.sbuf[k] = AETAW * ev[k];
                for (0..ADK) |d2| axpySub(self.wk1t[d2 * D ..][0..D], &self.sbuf, self.tbuf[d2]);
            }
        }
        // Wq update + the query-path gradient back into the decay-state grad.
        // Per k the accumulation runs d2-ascending, and every read is
        // pre-update, exactly as the instrument's interleaved loop does.
        for (0..D) |k| self.gqb[k] = 0;
        for (0..D) |k| self.sbuf[k] = AETAW * self.sq[k];
        for (0..ADK) |d2| {
            const dq = self.dqv[d2];
            const wrow = self.wqt[d2 * D ..][0..D];
            for (0..D) |k| self.gqb[k] += wrow[k] * dq;
            axpySub(wrow, &self.sbuf, dq);
        }
        for (0..D) |k| self.gacc[k] += self.gqb[k]; // rmsir == 1 (AT_NORM off)
        // Wq1 (query <- E[w_1]); e1r is re-read here because a top-K value
        // update may have moved that very row (attn.c:805 re-derives the pointer)
        {
            const e1r = self.e1[@as(usize, self.ring[0]) * D ..][0..D];
            for (0..D) |k| self.sbuf[k] = AETAW * e1r[k];
            for (0..ADK) |d2| axpySub(self.wq1t[d2 * D ..][0..D], &self.sbuf, self.dqv[d2]);
        }
    }

    /// attn.c:894-913 — write (state that predicted ri) -> ri into the ring.
    fn attnPush(self: *WordLbl, ri: usize) void {
        if (comptime !ATTN) return;
        const erv = self.e1[ri * D ..][0..D];
        const pos = self.ringpos;
        for (0..ADK) |d2| {
            const wrow = self.wkt[d2 * D ..][0..D];
            var acc: f32 = 0;
            for (0..D) |k| acc += wrow[k] * self.sq[k];
            const w1row = self.wk1t[d2 * D ..][0..D];
            for (0..D) |k| acc += w1row[k] * erv[k];
            self.ring_k[pos * ADK + d2] = acc;
        }
        @memcpy(self.ring_s[pos * D ..][0..D], self.sq[0..D]);
        self.ring_v[pos] = @intCast(ri);
        self.ringpos = (pos + 1) % AN;
        if (self.ringn < AN) self.ringn += 1;
    }
};
