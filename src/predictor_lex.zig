//! `PredictorLex` — Zig port of cmix-lex's top-level `Predictor`
//! (ref/cmix-lex/src/predictor.{h,cpp}); the 590-input, 2-layer predictor of the
//! real Hutter-prize submission. This file is the INTEGRATION of all the ported
//! sub-pieces + the verification harness against the real-cmix-lex oracle.
//! (The oracle goldens are derived verification data and are not distributed
//! with this source package; regenerate them from the cmix-lex reference.)
//!
//! GROUND TRUTH: predictor.cpp (+ predictor.h). The Rust predictor_lex.rs is a
//! secondary cross-reference; C++ wins on any disagreement.
//!
//! Layer-0 input fill order (predictor.cpp Predict, 0..589):
//!   [0]        bracket byte model              (SetInput -> Logit)
//!   [1..560]   fxcm 560 slots                  (SetStretchedInputUnchecked)
//!   [561]      direct on bracketCtx            (SetInput)
//!   [562..571] 10 match (5 word, 5 hash)       (SetInput)
//!   [572..586] 15 indirect_ns                  (SetInput)
//!   [587]      indirect_r (run-map)            (SetInput)
//!   [588]      PPMDLex bit-tree                 (SetInput)
//!   [589]      byte_mixer_output (state var)    (SetInput)
//!
//! RNG parity: the real C++ uses glibc `rand` (default seed 1) for the 16
//! Indirect map_offsets (in construction order) and then the LSTM Glorot weight
//! init (the "16-draw-then-weights order"). We reproduce glibc randexactly
//! (GlibcRand) so the oracle's offsets/LSTM weights match bit-for-bit — required
//! to hit the ±512 / 99.9% final-prob gate (the byte-mixer feeds the final prob).

const std = @import("std");
const strict_fp = @import("strict_fp");
const builtin = @import("builtin");
const build_options = @import("build_options");

// Cross-platform self-pid for the instrument/scratch file names below. Only
// exists so `zig build -Dwindows=true` compiles (`std.os.linux.getpid` emits
// a raw Linux `syscall` on any x86_64 target, which would fault on Windows).
// The Linux branch is textually the previous expression, so Linux codegen —
// and therefore the archive — is unchanged. See the design notes.
inline fn selfPid() u32 {
    if (comptime builtin.os.tag == .windows) return std.os.windows.GetCurrentProcessId();
    return @intCast(std.os.linux.getpid());
}
// the own-module rule: the mixer-e2e/mixshadow knobs live in the dedicated "mixer_options"
// module (relocated from shared model_opts at merge; every model_opts entry
// lands plumbing in decomp_bin, charged twice under Form-1).
const mixer_options = @import("mixer_options");
// the own-module rule again: -Dleafgrad's consumers are this file + src/models/indirect.zig,
// so its knobs live in their own "leafgrad_options" module.
const leafgrad_options = @import("leafgrad_options");
// -Dextshed (own "extshed_options" module, the own-module rule): the wave-3 EXTERNAL-STACK
// SHED deletion arm. Sheds the 28 external layer-0 READOUT inputs (bracket
// in[0], direct in[561], match x10 in[562..571], indirect x16 in[572..587])
// to the exact neutral 0.0 and skips their model readout/update work.
// RETAINED under the shed (the wave-2 hand-off's design constraint):
//   * `mgr.longest_match` — mixer #8's GATING context. It is produced by the
//     matchers' own machinery (Match.perceive advances match_length per bit +
//     writes the position map; Match.byteUpdate re-anchors and writes
//     longest_match), so ALL TEN matchers keep perceive+byteUpdate
//     VERBATIM and only their predictreadout is shed ⇒ the gating key is
//     bit-exact vs stock BY CONSTRUCTION.
//   * every manager-maintained gating stream (mx*, b2/b3stream, line_break,
//     wordscxt, recent_bytes, auxiliary_context) — the manager is untouched
//     except the double-indirect hashes_ind* derivation, whose ONLY consumers
//     are the 4 shed double-indirect models (see context_manager_lex.zig).
//   * PPMd in[588], LSTM in[589], wordlbl in[559] — NOT in scope.
// Model construction stays STOCK (identical glibc randdraw order ⇒ the
// LSTM Glorot weights are bit-identical); the RAM shed realizes through
// zero_alloc laziness — the shared map + hash-index arrays are simply never
// touched. Comptime-dead at default = bit-identical stock.
const extshed_options = @import("extshed_options");
const EXTSHED: bool = extshed_options.extshed;
const coldram = @import("coldram.zig");

const cml = @import("context_manager_lex.zig");
const mixerlex = @import("mixer/mixer_lex.zig");
const shadow_mix = @import("mixer/shadow_mix.zig");
const Sigmoid = @import("sigmoid.zig").Sigmoid;
const SSE = @import("sse.zig").SSE;

/// -Dmix-sparse-diag — census of exact zeros among the 590 layer-0 mixer inputs.
/// A zero input contributes nothing to the dot product AND nothing to its own weight
/// update, so its count is the exact size of the "sparse mixer" wall opportunity.
const MIXSPARSE: bool = @import("build_options").mix_sparse_diag;

/// -Dfxcm-zero-neutral — the 20-year-old off-by-one that hides HALF the mixer's
/// sparsity.
///
/// cmix's `RawPredictionProbability(raw) = squash(raw)/4095`. `squash(0) = 2048`, so
/// a ContextMap's "no opinion" emission (`clp(0)`) becomes p = 2048/4095 = 0.500122,
/// i.e. mixer input `logit(p) = +0.000488` — **not** the exact neutral 0.0 that the
/// 35-slot pad uses. MEASURED (`-Dmix-sparse-diag`): **295.29 of the 525 active fxcm
/// slots carry raw 0 on a typical bit = 50.05 % of all 590 layer-0 inputs.** Because
/// they are +0.000488 rather than 0.0, neither the forward MAC `w·x` nor the update
/// `w += lr·err·x` can be skipped, and the exact-zero count sees only the 35-slot pad.
///
///   1 = snap ONLY the raw-0 entry to the exact neutral (every other slot bit-identical)
///   2 = normalise by 4096 throughout, which is what makes squash(0) land on 0.5
///
/// Default 0 = stock. This is a RATIO screen first: if it is byte-neutral, it unlocks a
/// sparse forward+update path over ~50 % of the layer-0 work.
const ZERO_NEUTRAL: u32 = @import("build_options").fxcm_zero_neutral;
var ms_bits: u64 = 0;
var ms_zero: u64 = 0;
var ms_max: u32 = 0;
var ms_min: u32 = 1 << 30;
var ms_hist: [10]u64 = .{0} ** 10;
var ms_rawzero: u64 = 0;
var ms_runs: u64 = 0;
var ms_b8: u64 = 0;
var ms_b16: u64 = 0;
pub fn mixSparseDump(n_inputs: usize) void {
    if (comptime !MIXSPARSE) return;
    if (ms_bits == 0) return;
    const mean = @as(f64, @floatFromInt(ms_zero)) / @as(f64, @floatFromInt(ms_bits));
    const rawmean = @as(f64, @floatFromInt(ms_rawzero)) / @as(f64, @floatFromInt(ms_bits));
    std.debug.print("MIXSPARSE\tbits\t{d}\tn_inputs\t{d}\tmean_zero\t{d:.2}\tmean_pct\t{d:.2}\tmin\t{d}\tmax\t{d}\tmean_rawzero_slots\t{d:.2}\trawzero_pct_of_590\t{d:.2}\n",
        .{ ms_bits, n_inputs, mean, 100.0 * mean / @as(f64, @floatFromInt(n_inputs)), ms_min, ms_max,
           rawmean, 100.0 * rawmean / @as(f64, @floatFromInt(n_inputs)) });
    const runs = @as(f64, @floatFromInt(ms_runs)) / @as(f64, @floatFromInt(ms_bits));
    std.debug.print("MIXSPARSE\truns_per_bit\t{d:.2}\tmean_run_len\t{d:.2}\n",
        .{ runs, if (runs > 0) rawmean / runs else 0 });
    const fb = @as(f64, @floatFromInt(ms_bits));
    std.debug.print("MIXSPARSE\tallzero_vec8_per_bit\t{d:.2}\tof\t{d}\tpct\t{d:.2}\tallzero_line16_per_bit\t{d:.2}\tof\t{d}\tpct\t{d:.2}\n",
        .{ @as(f64, @floatFromInt(ms_b8)) / fb, n_inputs / 8, 100.0 * (@as(f64, @floatFromInt(ms_b8)) / fb) / @as(f64, @floatFromInt(n_inputs / 8)),
           @as(f64, @floatFromInt(ms_b16)) / fb, n_inputs / 16, 100.0 * (@as(f64, @floatFromInt(ms_b16)) / fb) / @as(f64, @floatFromInt(n_inputs / 16)) });
    for (ms_hist, 0..) |h, i| {
        std.debug.print("MIXSPARSE\tdecile\t{d}\t{d}\t{d:.3}\n", .{ i * 10, h, 100.0 * @as(f64, @floatFromInt(h)) / @as(f64, @floatFromInt(ms_bits)) });
    }
}

const Bracket = @import("models/bracket.zig").Bracket;
const Direct = @import("models/direct.zig").Direct;
const Indirect = @import("models/indirect.zig").Indirect;
const Match = @import("models/match.zig").Match;
const PPMDLex = @import("models/ppmd.zig").PPMDLex;
const states = @import("states.zig");
const Nonstationary = states.Nonstationary;
const RunMap = states.RunMap;

const ByteMixer = @import("mixer/byte_mixer.zig").ByteMixer;
const Lstm = @import("lstmfast").Lstm; // ReleaseFast module (-Dlstmfast)
// -Dtransformer (own options module per the own-module rule — never the shared model_opts).
const tfmod = @import("mixer/transformer.zig");
const build_options_tf = @import("transformer_options");

const cmfast_m = @import("cmfast");
// -Dfxcm-slotdelete: the TRUE-DELETION variant of
// -Dfxcm-slotmask. The masked CM instances emit NOTHING, so fxcm's layer-0 BLOCK
// shrinks from the stock 560 columns by `DEL_COLS` and every index after it moves
// down in lockstep. `DEL_COLS == 0` at the default mask (and with the deletion
// off), so every constant below recovers its stock literal exactly and the whole
// remap is comptime-dead. Single source of truth: models/fxcm26/slotdel.zig,
// which also sized fxcm's own in1/slots banks — the two cannot disagree.
const SD = cmfast_m.slotdel;
/// Width of the fxcm block in `mi0.inputs`: [1 .. 1+FX_N). Stock 560.
const FX_N: usize = 560 - SD.DEL_COLS;
const IDX_DIRECT: usize = FX_N + 1; // stock 561
const IDX_MATCH0: usize = FX_N + 2; // stock 562
const IDX_IND0: usize = FX_N + 12; // stock 572
const IDX_INDR: usize = FX_N + 27; // stock 587
const IDX_PPMD: usize = FX_N + 28; // stock 588
const IDX_BMIX: usize = FX_N + 29; // stock 589
/// Total layer-0 input count. Stock 590.
const NIN: usize = FX_N + 30;
const FxcmV26 = cmfast_m.FxcmV26;
// Root-side float table generation for fxcm (pinned math; see fxcm_v26.new).
const fx_table_gen = @import("models/fxcm26/tables.zig");

const MixerInputLex = mixerlex.MixerInputLex;
const MixerLex = mixerlex.MixerLex;

// PPMD sub-allocator heap (MB). The oracle used 14000; for a 64k stream the PPMD
// never hits memory pressure, so its predictions are heap-size-invariant (pure
// integer tree arithmetic, pText never reaches UnitsStart). We use a safe,
// large-enough heap for the gate. Production would pass 14000.
const PPMD_MEMORY_MB: u32 = @import("build_options").ppmd_arena_mb;

// Env-var hyperparameter overrides (for tuning the LSTM to zmix's @Vector numerics).
//
// HERMETIC SHIP GUARD (footgun review P1): the shipping self-extractor must not
// read tuning env vars — a stray ZMIX_LSTM_* on the encode box or the judge's
// machine changes the model numerics and silently desyncs the coder at decode.
// A root that declares `pub const zmix_hermetic = true;` (ship.zig) compiles
// every override away; experiment roots (runner_lex) keep them.
const root = @import("root");
const HERMETIC: bool = @hasDecl(root, "zmix_hermetic") and root.zmix_hermetic;
/// transformer-era weights-source guard, same root-decl pattern as `zmix_hermetic` above.
/// A root that declares `pub const weights_embedded_only = true` promises it
/// ALWAYS parks the blob from its own artifact tail (no CLI ⇒ no explicit path
/// argument, and `-Dtransformer-weights` is a dev/bench affordance). Declaring it
/// compiles the path-loading fallback in `init` away for that root only.
///
/// ⚠ `zmix_hermetic` is NOT the right predicate here and must not be reused:
/// `src/ship.zig` declares it too, and ship's `-c`/`-d` asset-compression ops run
/// on the bare packed engine BEFORE comp9 exists, so they have no tail and
/// legitimately need the explicit-path arm (it is what a committee rebuild runs).
const WEIGHTS_EMBEDDED_ONLY: bool =
    @hasDecl(root, "weights_embedded_only") and root.weights_embedded_only;
// (selfPid lives above with the other cross-platform helpers — the merge of
// claude/fp-tap-fix and claude/windows-target briefly carried two copies.)
// Env reads are experiment-only knobs: hermetic ship roots must read NO env
// var, and std.posix.getenv is a compile error on Windows (WTF-16),
// so both are comptime-excluded — the defaults apply there.
fn envF32(name: []const u8, default: f32) f32 {
    if (comptime (HERMETIC or builtin.os.tag == .windows)) return default;
    const v = std.posix.getenv(name) orelse return default;
    return std.fmt.parseFloat(f32, v) catch default;
}
fn envUsize(name: []const u8, default: usize) usize {
    if (comptime (HERMETIC or builtin.os.tag == .windows)) return default;
    const v = std.posix.getenv(name) orelse return default;
    return std.fmt.parseInt(usize, v, 10) catch default;
}

// ===========================================================================
// GlibcRand — reproduces glibc rand(TYPE_3 additive feedback, DEG=31, SEP=3),
// default seed 1 (no srand). Verified against gcc rand: 1804289383, 846930886,
// 1681692777, ...  Used for the Indirect map_offsets and the LSTM Glorot init.
// ===========================================================================
pub const GlibcRand = struct {
    r: [31]i32 = undefined,
    fptr: usize = 0,
    rptr: usize = 0,

    pub fn init(seed: i32) GlibcRand {
        var self = GlibcRand{};
        self.r[0] = seed;
        var i: usize = 1;
        while (i < 31) : (i += 1) {
            const prev: i64 = self.r[i - 1];
            const hi = @divTrunc(prev, 127773);
            const lo = @rem(prev, 127773);
            var word: i64 = 16807 * lo - 2836 * hi;
            if (word < 0) word += 2147483647;
            self.r[i] = @intCast(word);
        }
        self.fptr = 3;
        self.rptr = 0;
        var k: usize = 0;
        while (k < 310) : (k += 1) _ = self.next();
        return self;
    }

    pub fn next(self: *GlibcRand) i32 {
        const f: u32 = @bitCast(self.r[self.fptr]);
        const rr: u32 = @bitCast(self.r[self.rptr]);
        const sum: u32 = f +% rr;
        self.r[self.fptr] = @bitCast(sum);
        const result: u32 = sum >> 1;
        self.fptr = (self.fptr + 1) % 31;
        self.rptr = (self.rptr + 1) % 31;
        return @bitCast(result);
    }

    // rand% modulus (glibc `%` on non-negative int).
    pub fn nextMod(self: *GlibcRand, modulus: u64) u64 {
        const v: i64 = self.next();
        return @intCast(@rem(v, @as(i64, @intCast(modulus))));
    }

    // static_cast<float>(rand) / static_cast<float>(RAND_MAX).
    pub fn nextFloat(self: *GlibcRand) f32 {
        return @as(f32, @floatFromInt(self.next())) / @as(f32, 2147483647.0);
    }
};

// ===========================================================================
// GateDumper — LAB-ONLY instrument for family `mixer-gating-contexts`
//
// Comptime-gated by -Dgate-dump=<prefix>. Empty (default) ⇒ the type is `void`,
// every call site is `if (comptime GATE_DUMP_ON)`, and the build is byte-identical
// to stock. This is NOT a ship knob and must never appear on a recipe line.
//
// Per coded bit-step it captures
//   * 32 f16 columns: a slim view of the 590 mixer inputs (leaf votes, fxcm's
//     exported APM chain) PLUS five aggregates over the fxcm slot block that the
//     linear mixer provably cannot compute for itself (mean/sd/min/max/|·|-mean);
//   * the 23 layer-0 gating context values, TRUNCATED TO u32 — which is exactly
//     the row key `mixer_lex.getContextData` uses (`@truncate(self.context.*)`),
//     so an offline replay can reproduce row identity bit-for-bit;
//   * z = the pre-SSE layer-1 logit (the shipped mixing stack's own output),
//   * the final coded probability the arithmetic coder consumes, and the bit.
//
// Record 164 B, header 32 B:
//   hdr: "ZMGATE01" u32 ver=1 u32 ncol=32 u32 nctx=23 u32 stride u32 reclen u32 rsv
//   rec: f16[32] | u32[23] | f32 z | u16 P | u8 y | u8 flags   (bit0 = LSTM
//        byte-range override fired ⇒ the coder ignored the mixer for this bit)
// ===========================================================================
pub const GATE_DUMP_PATH: []const u8 = build_options.gate_dump;
pub const GATE_DUMP_ON: bool = GATE_DUMP_PATH.len > 0;
pub const GATE_STRIDE: u64 = build_options.gate_dump_stride;
pub const GATE_MAX: u64 = build_options.gate_dump_max;
// ---------------------------------------------------------------------------
// WIDE RECORDS: `-Dgate-dump=wide:<prefix>` emits the FULL 590-input layer-0
// vector per staged bit — the instrument `mixer-gating-contexts`' closure named
// as its own reopen trigger ("an instrument carrying the 525 INDIVIDUAL fxcm
// slots; this dump carries aggregates"). Selected by a PREFIX ON THE EXISTING
// PATH STRING, deliberately NOT a new `-D` option (same reasoning as the slim
// variant on `claude/gate-dump-slim` @19af0e3: every option in the shared
// `model_opts` costs ~40 B of decomp_bin even dead, charged TWICE in S).
//
// Record (ZMGATE03), 1330 B, little-endian, tail == the other layouts:
//   f16[590]  mi0.inputs (post-clamp, exactly what every layer-0 mixer reads)
//   f16[23]   layer-0 cascade outputs (mi0.extra_inputs, post-clamp) — zeros
//             on an adaptive-depth SKIP record (flags bit1): never computed
//   u32[23]   gating context values (@truncate to the row key mixer_lex uses)
//   u32       byte index (ctr/8 — REQUIRED: wide strides by BYTE, so the byte
//             stream is NOT reconstructible from the y column alone)
//   f32 z | u16 P | u8 y | u8 flags
//     flags: bit0 = LSTM byte-range override fired (coder ignored the mixer)
//            bit1 = adaptive-depth SKIP (z is the ADEPTH HEAD's logit, and the
//                   23 output columns are zeros)
//            bits2..4 = bit position within the byte (MSB-first, 0..7)
//
// STRIDE SEMANTICS DIFFER: wide strides by coded BYTE — all 8 bits of every
// GATE_STRIDE-th byte are recorded (within-byte structure intact), because at
// 1330 B/bit a stride-1 dump of even a 20 m tier would cost 132 GB.
pub const GATE_WIDE: bool = GATE_DUMP_ON and std.mem.startsWith(u8, GATE_DUMP_PATH, "wide:");
pub const GATE_PATH_EFF: []const u8 = if (GATE_WIDE) GATE_DUMP_PATH["wide:".len..] else GATE_DUMP_PATH;
pub const GATE_NIN: usize = NIN; // wide only: full layer-0 input vector (stock 590)
pub const GATE_NCOL: usize = 32;
pub const GATE_NCTX: usize = 23;
pub const GATE_REC: usize = if (GATE_WIDE)
    GATE_NIN * 2 + 23 * 2 + 23 * 4 + 4 + 4 + 2 + 1 + 1 // 1330
else
    GATE_NCOL * 2 + GATE_NCTX * 4 + 4 + 2 + 1 + 1; // 164

var gate_instances = std.atomic.Value(u32).init(0);

// -Dextshed engagement witness: printed ONCE per process (first predictor
// constructed), so a leg's log proves the arm ran ENGAGED, per the screen
// discipline (a silent arm is indistinguishable from a never-built one).
var extshed_announced: bool = false;

pub const GateDumper = struct {
    file: std.fs.File,
    buf: []u8,
    pos: usize,
    ctr: u64,
    nrec: u64,
    stage: [GATE_REC]u8,
    stage_active: bool,
    alloc: std.mem.Allocator,

    const BUF_BYTES: usize = 16 << 20;

    fn init(a: std.mem.Allocator) !*GateDumper {
        const self = try a.create(GateDumper);
        errdefer a.destroy(self);
        var name_buf: [512]u8 = undefined;
        const path = try std.fmt.bufPrint(&name_buf, "{s}.{d}.bin", .{ GATE_PATH_EFF, selfPid() });
        self.file = try std.fs.cwd().createFile(path, .{});
        self.buf = try a.alloc(u8, BUF_BYTES);
        self.pos = 0;
        self.ctr = 0;
        self.nrec = 0;
        self.stage_active = false;
        self.alloc = a;
        var hdr: [32]u8 = undefined;
        // ZMGATE03 = the wide 1330 B layout. A distinct magic rather than
        // "ZMGATE01 with ncol=590" so a reader cannot silently mis-parse.
        @memcpy(hdr[0..8], if (GATE_WIDE) "ZMGATE03" else "ZMGATE01");
        std.mem.writeInt(u32, hdr[8..12], 1, .little);
        std.mem.writeInt(u32, hdr[12..16], @intCast(if (GATE_WIDE) GATE_NIN else GATE_NCOL), .little);
        std.mem.writeInt(u32, hdr[16..20], @intCast(GATE_NCTX), .little);
        std.mem.writeInt(u32, hdr[20..24], @intCast(GATE_STRIDE), .little);
        std.mem.writeInt(u32, hdr[24..28], @intCast(GATE_REC), .little);
        std.mem.writeInt(u32, hdr[28..32], 0, .little);
        try self.file.writeAll(&hdr);
        std.debug.print("gate-dump: {s} stride={d} rec={d}B ncol={d} nctx={d}\n", .{ path, GATE_STRIDE, GATE_REC, GATE_NCOL, GATE_NCTX });
        return self;
    }

    inline fn putF16(b: []u8, i: usize, v: f32) void {
        const h: f16 = @floatCast(v);
        std.mem.writeInt(u16, b[i * 2 ..][0..2], @bitCast(h), .little);
    }

    fn flush(self: *GateDumper) void {
        if (self.pos == 0) return;
        self.file.writeAll(self.buf[0..self.pos]) catch |e| {
            std.debug.print("gate-dump: write error {s}\n", .{@errorName(e)});
        };
        self.pos = 0;
    }

    fn deinit(self: *GateDumper) void {
        self.flush();
        self.file.close();
        std.debug.print("gate-dump: closed after {d} records ({d} bit-steps)\n", .{ self.nrec, self.ctr });
        self.alloc.free(self.buf);
        self.alloc.destroy(self);
    }
};

// ===========================================================================
// -Dfree-prior-b: coding-time prior recorder (variant b)
// ===========================================================================
// Streams the post-×2-scale ByteMixer aux vector (== the exact prior PRL reads)
// to a /home scratch file during forward coding, one vs-float row per byte, 1 MB
// buffered so RSS stays flat (impl-preflight gate-4). freePriorReplayChunk
// preads it back per row at replay time. The file is written AND read by the
// same op (encoder writes+replays its own; decoder likewise) — it is never a
// cross-op hand-over, so no side data. Path: ZMIX_FP_PRIOR_FILE (per-op unique) or
// a pid-stamped /home scratch default. Row width (= vocab_size) is captured on
// first record; a 6-byte header (u32 row_floats + u8 tap-tag + u8 K) lets replay
// validate it. The whole type is compiled away unless -Dfree-prior-b.
// Tap fidelity (comptime): exact full-f32 rows, or the e9-shippable lossy
// top-K+f16 compaction (-Dfree-prior-tap=lossy). Both encode a fixed number of
// bytes per row so the buffered write/pread streaming below stays format-blind.
const FP_LOSSY = build_options.free_prior_tap_lossy;
const FP_TOPK: usize = build_options.free_prior_topk;
// Lossy row = K × (u8 index + f16 prob) + f16 tail-mass. Fixed width.
const FP_LOSSY_ROW_BYTES: usize = FP_TOPK * (1 + 2) + 2;
// Header tag byte: distinguishes exact(0) from lossy(1) so a replay against a
// mismatched recording panics rather than silently mis-parsing.
const FP_TAP_TAG: u8 = if (FP_LOSSY) 1 else 0;

const FpPriorRec = struct {
    file: ?std.fs.File = null,
    path_buf: [512]u8 = undefined,
    path_len: usize = 0,
    row_floats: usize = 0, // vocab_size, set on first record
    // write side
    wbuf: [1 << 20]u8 = undefined,
    wlen: usize = 0,
    wpos: u64 = 0, // bytes appended (past the header)
    // read side (replay)
    rbuf: [1 << 20]u8 = undefined,
    rlen: usize = 0, // valid bytes in rbuf
    roff: usize = 0, // cursor within rbuf
    rfile_off: u64 = 0, // next file offset to pread (relative to data start)
    header_len: u64 = 6, // u32 row_floats + u8 tap-tag + u8 topk
    // Lossy scratch: one packed row buffer (comptime-erased under exact, so the
    // struct is unchanged for the exact path — preserves exact bit-identity).
    row_pack: if (FP_LOSSY) [FP_LOSSY_ROW_BYTES]u8 else void = if (FP_LOSSY) undefined else {},

    fn ensureOpen(self: *FpPriorRec, vs: usize) void {
        if (self.file != null) return;
        self.row_floats = vs;
        // Resolve the path. SUBMISSION-VALIDITY FIX (task #45):
        //  (1) the default is CWD-RELATIVE + PID-STAMPED — `fp_prior.{pid}.bin`,
        //      exactly the pattern ppmd.zig uses for `ppm.{pid}.temp` (rule 8
        //      grants 100 GB of temp files, and cmix-lex itself does this). The
        //      old default was an ABSOLUTE build-machine path that
        //      PANICS on any machine without it — a judge machine, most likely.
        //  (2) the env override is EXPERIMENT-ONLY: hermetic ship roots never
        //      read it (the ruling classes an env-var-selected path
        //      with sysfs dependence), and the read is also compiled out on
        //      Windows, where `std.posix.getenv` is a compile error (WTF-16).
        var have_env = false;
        if (comptime (!HERMETIC and @import("builtin").os.tag != .windows)) {
            if (std.posix.getenv("ZMIX_FP_PRIOR_FILE")) |e| {
                @memcpy(self.path_buf[0..e.len], e);
                self.path_len = e.len;
                have_env = true;
            }
        }
        if (!have_env) {
            const s = std.fmt.bufPrint(&self.path_buf, "fp_prior.{d}.bin", .{selfPid()}) catch unreachable;
            self.path_len = s.len;
        }
        const p = self.path_buf[0..self.path_len];
        const f = std.fs.cwd().createFile(p, .{ .read = true, .truncate = true }) catch |err| {
            std.debug.panic("free-prior-b: cannot open prior file '{s}': {}", .{ p, err });
        };
        // Header: row width + tap tag + K (validated on replay).
        var hdr: [6]u8 = undefined;
        std.mem.writeInt(u32, hdr[0..4], @intCast(vs), .little);
        hdr[4] = FP_TAP_TAG;
        hdr[5] = @intCast(@min(FP_TOPK, 255));
        f.writeAll(&hdr) catch |err| std.debug.panic("free-prior-b: header write: {}", .{err});
        // Derive the replay read origin from the header ACTUALLY written, so the
        // writer and the reader can never disagree about the header width.
        self.header_len = hdr.len;
        self.file = f;
    }

    /// Append `bytes` to the 1MB write buffer, flushing on overflow. Streaming
    /// primitive: format-blind (exact rows and lossy packed rows both go through
    /// here), so RSS stays flat (impl-preflight gate-4).
    fn appendBytes(self: *FpPriorRec, bytes: []const u8) void {
        if (self.wlen + bytes.len > self.wbuf.len) {
            self.file.?.writeAll(self.wbuf[0..self.wlen]) catch |err| std.debug.panic("free-prior-b: flush: {}", .{err});
            self.wlen = 0;
        }
        @memcpy(self.wbuf[self.wlen..][0..bytes.len], bytes);
        self.wlen += bytes.len;
        self.wpos += bytes.len;
    }

    /// Append one prior row (post-×2-scale aux, vs floats). Exact tap writes the
    /// full f32 vector; the lossy tap writes the top-K+f16 compaction. Buffered.
    fn recordRow(self: *FpPriorRec, prior: []const f32) void {
        self.ensureOpen(prior.len);
        std.debug.assert(prior.len == self.row_floats);
        if (comptime FP_LOSSY) {
            self.packLossyRow(prior, &self.row_pack);
            self.appendBytes(&self.row_pack);
        } else {
            self.appendBytes(std.mem.sliceAsBytes(prior));
        }
    }

    /// Pack one prior row into the lossy on-disk form (CUDA_LSTM_SIM_SPEC §3):
    /// the K largest entries as [u8 vocab-relative index][f16 prob] pairs, then a
    /// single f16 = the total mass of the remaining (vs−K) entries. Selection is
    /// a total order — (prob descending, index ascending) — so BOTH ops, seeing
    /// the identical exact prior, pick the identical top-K set and emit a
    /// byte-identical row (replay symmetry preserved; weights-hash gate). When
    /// vs ≤ K every entry is a top entry and the tail is 0.
    fn packLossyRow(self: *FpPriorRec, prior: []const f32, out: *[FP_LOSSY_ROW_BYTES]u8) void {
        const vs = self.row_floats;
        const k = @min(FP_TOPK, vs);
        // Partial selection sort of the top-k indices by (value desc, index asc).
        // vs ≤ 256 and k ≤ 64, so k·vs ≤ ~13k comparisons/row — cheap vs the LSTM
        // forward pass this row feeds, and it avoids any heap allocation.
        var idx: [256]u8 = undefined;
        for (0..vs) |i| idx[i] = @intCast(i);
        for (0..k) |a| {
            var best = a;
            for (a + 1..vs) |c| {
                const pc = prior[idx[c]];
                const pb = prior[idx[best]];
                if (pc > pb or (pc == pb and idx[c] < idx[best])) best = c;
            }
            const tmp = idx[a];
            idx[a] = idx[best];
            idx[best] = tmp;
        }
        // Write K pairs (pad with index 0 / prob 0 when vs < K so the row width is
        // fixed; those pads reconstruct to dst[0]+=0, harmless — but vs<K only
        // happens in tiny-vocab tests, never at enwik scale).
        var o: usize = 0;
        var tail: f32 = 0;
        // Tail = exact-f32 sum of the non-top entries (computed before f16 rounding
        // so the compaction error is confined to the stored quantities).
        for (k..vs) |r| tail += prior[idx[r]];
        for (0..FP_TOPK) |p| {
            if (p < k) {
                const vi = idx[p];
                out[o] = vi;
                const h: f16 = @floatCast(prior[vi]);
                std.mem.writeInt(u16, out[o + 1 ..][0..2], @bitCast(h), .little);
            } else {
                out[o] = 0;
                std.mem.writeInt(u16, out[o + 1 ..][0..2], @bitCast(@as(f16, 0)), .little);
            }
            o += 3;
        }
        const th: f16 = @floatCast(tail);
        std.mem.writeInt(u16, out[o..][0..2], @bitCast(th), .little);
    }

    /// Reconstruct a prior row from the lossy on-disk form into `dst` (vs floats):
    /// dst[idx] = f16→f32(prob) for each stored top-K entry, and
    /// tail_mass/(vs−K) uniformly for every other vocab index (spec §3). The
    /// reconstruction is a pure function of the recorded bytes ⇒ identical on both
    /// ops. Feeds the LSTM the SAME (approximate) coding-time prior the shipped
    /// PRL softmax blends, minus the compaction error under test in the screen.
    fn unpackLossyRow(self: *FpPriorRec, src: *const [FP_LOSSY_ROW_BYTES]u8, dst: []f32) void {
        const vs = self.row_floats;
        const k = @min(FP_TOPK, vs);
        const n_tail = vs - k; // vs > k at enwik scale; guarded below
        // Tail first (uniform fill), then overwrite the top-K slots — this way a
        // padded top slot (vs<K) that maps to index 0 does not clobber a tail
        // fill, and top indices always take precedence.
        var o: usize = 0;
        // read tail (last 2 bytes)
        const tbits = std.mem.readInt(u16, src[FP_TOPK * 3 ..][0..2], .little);
        const tail: f32 = @floatCast(@as(f16, @bitCast(tbits)));
        const fill: f32 = if (n_tail == 0) 0 else tail / @as(f32, @floatFromInt(n_tail));
        for (0..vs) |i| dst[i] = fill;
        for (0..FP_TOPK) |p| {
            if (p < k) {
                const vi = src[o];
                const bits = std.mem.readInt(u16, src[o + 1 ..][0..2], .little);
                dst[vi] = @floatCast(@as(f16, @bitCast(bits)));
            }
            o += 3;
        }
    }

    /// erased trampoline installed into ByteMixer.fp_prior_record.
    fn recordTrampoline(ctx: *anyopaque, prior: []const f32) void {
        const self: *FpPriorRec = @ptrCast(@alignCast(ctx));
        self.recordRow(prior);
    }

    /// Flush the write buffer and reset the read cursor to the first data row.
    /// Called once per epoch (fpPriorBeginReplay).
    fn beginReplay(self: *FpPriorRec) void {
        if (self.file) |f| {
            if (self.wlen != 0) {
                f.writeAll(self.wbuf[0..self.wlen]) catch |err| std.debug.panic("free-prior-b: pre-replay flush: {}", .{err});
                self.wlen = 0;
            }
        }
        self.rlen = 0;
        self.roff = 0;
        self.rfile_off = 0;
    }

    /// Fill `dst` with the next `dst.len` on-disk bytes via buffered pread. The
    /// read offset is independent of the append offset (pread), and each epoch
    /// restarts from row 0 (beginReplay). Streaming primitive: format-blind.
    fn readBytes(self: *FpPriorRec, dst: []u8) void {
        var got: usize = 0;
        while (got < dst.len) {
            if (self.roff == self.rlen) {
                const n = self.file.?.pread(&self.rbuf, self.header_len + self.rfile_off) catch |err| std.debug.panic("free-prior-b: pread: {}", .{err});
                if (n == 0) std.debug.panic("free-prior-b: prior stream underrun (recorded {d} bytes, replay needs more)", .{self.wpos});
                self.rlen = n;
                self.roff = 0;
                self.rfile_off += n;
            }
            const take = @min(dst.len - got, self.rlen - self.roff);
            @memcpy(dst[got..][0..take], self.rbuf[self.roff..][0..take]);
            self.roff += take;
            got += take;
        }
    }

    /// Read the next prior row into `dst` (vs floats). Exact tap reads the full
    /// f32 vector directly; the lossy tap reads FP_LOSSY_ROW_BYTES and
    /// reconstructs (top-K exact + uniform tail).
    fn readRow(self: *FpPriorRec, dst: []f32) void {
        std.debug.assert(dst.len == self.row_floats);
        if (comptime FP_LOSSY) {
            self.readBytes(&self.row_pack);
            self.unpackLossyRow(&self.row_pack, dst);
        } else {
            self.readBytes(std.mem.sliceAsBytes(dst));
        }
    }

    fn deinit(self: *FpPriorRec) void {
        if (self.file) |f| {
            f.close();
            self.file = null;
            // Best-effort remove the scratch file (regenerable; not an artifact).
            std.fs.cwd().deleteFile(self.path_buf[0..self.path_len]) catch {};
        }
    }

    /// Field-by-field init of the SMALL fields only. NEVER init this struct
    /// with an aggregate literal (`.{}`): the two 1 MB stream buffers make the
    /// literal a ~2.1 MB template blob that Zig embeds in .text — measured
    /// +2,103 KB on the composed zmix_ship, a Form-1 S disaster (binary bytes
    /// count TWICE). The big arrays (path_buf/wbuf/rbuf/row_pack) are left
    /// truly undefined; every one is written before it is read (ensureOpen /
    /// appendBytes / beginReplay+readBytes / packLossyRow respectively).
    fn initInPlace(self: *FpPriorRec) void {
        self.file = null;
        self.path_len = 0;
        self.row_floats = 0;
        self.wlen = 0;
        self.wpos = 0;
        self.rlen = 0;
        self.roff = 0;
        self.rfile_off = 0;
        self.header_len = 6;
    }
};

// ===========================================================================
// -Dadaptive-depth: ensemble-saturation ADAPTIVE-DEPTH cascade skip.
// Frozen mechanism (zero retuning latitude):
//   (sha256 91ffad6c0e3fd8dde1aee8a81dba9208559e984f8c9cd70bfb81f1fb9395e1fd),
//   reproducing the banked depth4.c screen of family ensemble-saturation
//   (FINDINGS.md 42.08% of bits skip the cascade for +0.012% h1 at
//   tau=8 — the exact tau table was regenerated from the cov20m dump on 08-09).
//
// Head: single online logistic over the TIER-D pre-cascade values — everything
// already computed before the outer 23-mixer cascade runs. Feature vector, in
// depth4's exact order: f[0]=1 (bias), then v[idx] for
//   idx = [0, 519..525, 561..571, 572..587, 588, 589, 0]
// (v = mi0.inputs; v[519..525] = fxcm slots 518..524 = fxcm layer-1 output +
// its APM chain; the trailing 0 is v[0] DUPLICATED — depth4's ND_=38 vs a
// 37-entry SET_D read zeroed alignment padding as idx[37], objdump-verified,
// so the measured head double-counts bracket. Reproduced as measured.)
//
// Gate: skip iff |clamp(z,±30)| > tau. On skip the coder receives p_head and
// the cascade (23 layer-0 mixers incl. their context_map row resolution),
// mixer1, SSE (learned tables; parse counters still advance) and HERO are
// neither evaluated nor updated. The head itself trains on EVERY coded bit
// (AdaGrad, lr 0.01, eps 1e-8 — depth4's constants), never during pretrain.
// Decoder-symmetric: the gate reads only values both coder sides compute
// before the bit is resolved.
// ===========================================================================
// the own-module rule: the knob lives in the consuming module's OWN options module
// ("adepth_options"), not in the shared model_opts (whose every entry lands
// plumbing in decomp_bin — charged twice under Form-1).
const adepth_options = @import("adepth_options");
const AD_ON: bool = adepth_options.adaptive_depth;
// -Dmixer1-rls (`rls8`; own options module "rls_options", the own-module rule): the layer-1
// head's LEARNING RULE swaps from plain SGD to a full-matrix online IRLS /
// extended-Kalman recursion. d = 25 with one weight row is the only place in
// the engine where exact curvature is affordable. Comptime-dead at 0 — a stock
// build never instantiates the state and never emits the recursion.
const rls_options = @import("rls_options");
const rls_mod = @import("mixer/mixer1_rls.zig");
const RLS_MODE: u32 = rls_options.mixer1_rls;
const RLS_ON: bool = RLS_MODE != 0;
// See build.zig: route the -Dmixer-e2e/-Dleafgrad gradient through the RLS head
// (self-consistent, and MEASURED at +12.86 % = catastrophic) or leave it on the
// still-running SGD row (DEFAULT, measured -0.0945 % on the 29-flag line).
const RLS_E2E_HEAD: bool = rls_options.mixer1_rls_e2e_head;
// -Dwordlbl (own options module; the own-module rule): the word-identity distributed-
// representation expert. Comptime-void by default — the null arm is bit-identical
// stock BY CONSTRUCTION because pad slot 559 then keeps its fxcm_neutral 0.0.
const wordlbl_mod = @import("models/wordlbl.zig");
const WL_ON: bool = wordlbl_mod.WL_ON;
// The LOWEST layer-0 pad index the wordlbl family claims: 559 alone, or 557
// when -Dwordlbl-attn-head adds the candidate head's two inputs. fxcm slot j
// lands at mixer index 1+j, so the neutral pad starts at 1+active — the assert
// below is `1 + active <= WL_PAD_LO`.
const WL_PAD_LO: usize = if (wordlbl_mod.WL_HEAD) FX_N - 3 else FX_N - 1;
const AD_TAU: f32 = adepth_options.adaptive_depth_tau;
// depth4 SET_D + the padding-read duplicate (see block comment above).
const AD_IDX = blk: {
    var a = [38]u16{
        0,   519, 520, 521, 522, 523, 524, 525,
        561, 562, 563, 564, 565, 566, 567, 568,
        569, 570, 571, 572, 573, 574, 575, 576,
        577, 578, 579, 580, 581, 582, 583, 584,
        585, 586, 587, 588, 589, 0,
    };
    // -Dfxcm-slotdelete: two DIFFERENT shifts, and conflating them would silently
    // re-point the head at the wrong features. Entries >= 561 sit AFTER the fxcm
    // block, so they move by the BLOCK's shrink (DEL_COLS). Entries 519..525 are
    // fxcm slots 518..524 — fxcm's own layer-1 output and its APM chain, emitted
    // LAST, after every CM instance — so they move by the EMISSION shrink
    // (DEL_SLOTS). Both are 0 at the default mask; index 0 (v[0], twice) never
    // moves. Verified against the engagement witness (`in1 544->N`).
    if (SD.SLOTDEL) {
        for (&a, 0..) |*v, i| {
            if (i == 0 or i == a.len - 1) continue;
            v.* -= @intCast(if (v.* >= 561) SD.DEL_COLS else SD.DEL_SLOTS);
        }
    }
    break :blk a;
};
const AD_NW: usize = 1 + AD_IDX.len; // bias + 38 features = 39 weights
const AdepthState = struct {
    w: [AD_NW]f32 = [_]f32{0} ** AD_NW,
    g: [AD_NW]f32 = [_]f32{0} ** AD_NW,
    skip: bool = false, // this bit's gate decision (predict -> perceive/runner)
    p: f32 = 0.5, // this bit's head probability
    // Coverage instrumentation (reported via ZMIX_ADEPTH_STATS at deinit):
    // bit skips drive the MAC win; ALL-8-skip bytes drive the row-traffic win
    // (the 23-row ~54,280 B fetch is per-byte — preflight §4).
    bits: u64 = 0,
    bits_skipped: u64 = 0,
    bytes: u64 = 0,
    bytes_allskip: u64 = 0,
    cur_byte_all: bool = true,
};

// ---- -Dleafgrad: coding-loss leaf training ------------------
// See build.zig for the derivation. Here we only compute, per bit,
//     Gamma_m = dz/dv_m = sum_{i=0..22} gamma_i * W0_i[m]
// for the 16 Indirect models (layer-0 inputs 572..587) and hand each model
// dL/dtau_m = (sigma(z) - y) * Gamma_m. gamma_i is the SAME backward recursion
// -Dmixer-e2e runs; it is recomputed here rather than shared so that a build
// with this knob off is textually and arithmetically the stock file (253 MAC/bit
// of deliberate redundancy, ~0.6% of the 13,570 MAC/bit layer-0 mixer).
const LG_ON: bool = leafgrad_options.leafgrad;
const LG_Z: f32 = leafgrad_options.leafgrad_z;
const LG_DIAG: bool = leafgrad_options.leafgrad_diag;
// -Dleafgrad-fxcm: the REACH-RESPONSE arm (reopen trigger). The
// Indirect arm above reaches 2.42% of Sum|Gamma| @10 MB and measured NULL-to-
// ADVERSE; this one pushes the same exact gradient into fxcm's ContextMap3
// `ts[256]` cells, the ~36% of fxcm slots that have an adaptive cell at all.
//
// The division of labour: only THIS file knows Gamma_j and dv_j/draw_j (the
// mixer backward recursion and the fxcm_stretched table are both here), and only
// cm3.zig knows draw/dp1 and dp1/dtau. So we hand fxcm dL/draw per slot and cm3
// carries the last two factors. Note the fxcm slot input is NOT a logit -- it is
// `fxcm_stretched[raw+2047]` -- so unlike the Indirect arm dv/draw is NOT 1 and
// must be carried; it is read off the very table that produced the input.
const LG_FX: bool = leafgrad_options.leafgrad_fxcm;
const LG_ANY: bool = LG_ON or LG_FX;
// fxcm slots occupy layer-0 inputs [1..561).
const LG_FX_BASE: usize = 1;
const LG_FX_N: usize = FX_N;
// The 16 Indirect models occupy layer-0 inputs [572..587] (15 ns + 1 run-map).
const LG_BASE: usize = IDX_IND0;
const LG_N: usize = 16;
// LAB census groups over the 590 layer-0 inputs.
const LgDiag = struct {
    bits: u64 = 0,
    // |Gamma| summed per group: bracket, fxcm, direct, match, indirect, ppmd, bytemix
    g_bracket: f64 = 0,
    g_fxcm: f64 = 0,
    g_direct: f64 = 0,
    g_match: f64 = 0,
    g_ind: f64 = 0,
    g_ppmd: f64 = 0,
    g_bmix: f64 = 0,
    // |dL/dtau| summed over the 16 indirect models (the actual step magnitude
    // before Z), so the Z grid can be set from a measurement, not a guess.
    step_abs: f64 = 0,
    // ACHIEVED REACH of -Dleafgrad-fxcm: |Gamma| summed over exactly the slots a
    // ContextMap3 emitted through an adaptive `ts` cell on the bit being scored.
    // This is the coverage number the reach-response measurement turns on, and it
    // is measured (from fxcm's own reach mask), never assumed from a slot census.
    g_cm3_adapt: f64 = 0,
};

// ===========================================================================
// PredictorLex
// ===========================================================================
pub const PredictorLex = struct {
    alloc: std.mem.Allocator,
    vocab: [256]bool,
    vocab_size: u32,

    // shared sigmoid (logit_size 100001, exactly matching predictor.cpp sigmoid_).
    sig: Sigmoid,

    mgr: *cml.ContextManagerLex,
    fxcm: *FxcmV26,

    // state machines (stable pointers for the Indirect State union).
    nonstationary: Nonstationary = .{},
    run_map: RunMap = undefined,

    // fxcm stretched-input lookup (predictor.cpp:9-16).
    fxcm_stretched: [4096]f32 = undefined,
    fxcm_neutral: f32 = 0,
    // -Dleafgrad-fxcm: d(mixer input)/d(raw slot value), i.e. the local slope of
    // `fxcm_stretched` itself. The fxcm slot input is NOT a logit (that is what
    // made dv/dtau == 1 for the Indirect arm), so this factor is real and is read
    // off the exact table that produced the input. It goes to ~0 where the
    // squash/logit clamp at [1e-4, 1-1e-4] flattens the curve -- the second
    // saturation the chain rule has to respect, alongside cm3's st1 clp.
    fxcm_dvdraw: if (LG_FX) [4096]f32 else void = if (LG_FX) undefined else {},

    // models
    bracket: *Bracket,
    direct: *Direct,
    ppmd: *PPMDLex,
    match_models: [10]*Match,
    indirect_ns: [15]*Indirect,
    indirect_r: *Indirect,

    // byte-mixer (LSTM)
    lstm: *Lstm,
    byte_mixer: *ByteMixer,
    byte_mixer_output: f32 = 0.0,

    // -Dfree-prior-b prior recorder (variant b). `void`-shaped & comptime-dead
    // when the flag is off. Records the coding-time PPMd prior to a /home scratch
    // file during forward coding; replayed by freePriorReplayChunk.
    // NO default value on purpose: a `.{}` default materializes the 2 MB
    // buffer template into .text (see FpPriorRec.initInPlace). initSeedInner
    // is the only constructor and inits it field-by-field.
    fp_prior: if (build_options.free_prior_b) FpPriorRec else void,

    // mixers
    mi0: MixerInputLex,
    mi1: MixerInputLex,
    mixers0: [23]MixerLex = undefined,
    mixer1: MixerLex = undefined,
    shadow: if (shadow_mix.enabled) *shadow_mix.Shadow else void = undefined,
    shadow_pre: f32 = 0.5,
    shadow_post: f32 = 0.5,
    shadow_free: bool = false,
    sse: SSE,

    // rng (glibc-parity)
    grand: GlibcRand,

    // LAB-ONLY gating-context dump (comptime-gated; `void` in a stock build).
    gdump: if (GATE_DUMP_ON) ?*GateDumper else void = if (GATE_DUMP_ON) null else {},

    // -Dmixer1-rls state: one 25x25 f64 P plus w/g/k/x (~5,600 B, L1-resident,
    // no per-context state). Zero-sized void when the knob is off.
    rls: if (RLS_ON) rls_mod.Rls else void = if (RLS_ON) undefined else {},
    // pre-SSE p produced by the RLS head this bit — the curvature h = p(1-p)
    // and the residual (y - p) of its own Newton step.
    rls_p: if (RLS_ON) f32 else void = if (RLS_ON) 0.5 else {},

    // -Dadaptive-depth head state (312 B; zero-sized void when the knob is off).
    adepth: if (AD_ON) AdepthState else void,
    wlbl: if (WL_ON) *wordlbl_mod.WordLbl else void = undefined,

    // -Dleafgrad-diag LAB census: running sum of |Gamma_m| per input GROUP, i.e.
    // where the coded-length gradient actually has leverage across the 590
    // layer-0 inputs. `void` (zero-sized) when the diag knob is off.
    lg_diag: if (LG_DIAG) LgDiag else void = if (LG_DIAG) .{} else {},

    // Golden-gate / default entry point: glibc default seed (== srand(1)), matching
    // the oracle driver (oracle_main.cpp does NOT call srand). Use `initSeed` for the
    // real cmix-lex binary, which is built `-DSEED=923` and calls `srand(923)`
    // (runner.cpp:350) before constructing the Predictor.
    pub fn init(a: std.mem.Allocator, vocab: [256]bool) !*PredictorLex {
        return initSeed(a, vocab, 1);
    }

    pub fn initSeed(a: std.mem.Allocator, vocab: [256]bool, seed: i32) !*PredictorLex {
        return initSeedInner(a, vocab, seed, null);
    }

    // Same as `initSeed`, but loads the fxcm WRT dictionary (fxcmv1.cpp dosym—
    // Predictor construction reads the command-line `dictionary_path`). Encoder and
    // decoder MUST pass byte-identical `dict` or the codec desyncs. Callers on the
    // WITH-DICTIONARY paths (runner_lex `-c/-d/-e/-D`) use this; no-dict paths keep
    // `initSeed` (null dict), leaving the nodict goldens untouched.
    pub fn initSeedWithDict(a: std.mem.Allocator, vocab: [256]bool, seed: i32, dict: []const u8) !*PredictorLex {
        return initSeedInner(a, vocab, seed, dict);
    }

    fn initSeedInner(a: std.mem.Allocator, vocab: [256]bool, seed: i32, dict: ?[]const u8) !*PredictorLex {
        const self = try a.create(PredictorLex);
        self.alloc = a;
        // `a.create` returns UNINITIALIZED memory: Zig does NOT apply a struct's
        // default field values to it, so every field must be assigned explicitly
        // here. `fp_prior` is the one *aggregate* field whose defaults matter
        // (header_len, the null file handle, the buffer cursors); without this
        // reset it holds whatever the allocation happened to contain and replay
        // preads from a bogus offset. Comptime-dead unless -Dfree-prior-b.
        if (comptime build_options.free_prior_b) self.fp_prior.initInPlace();
        self.vocab = vocab;
        var vs: u32 = 0;
        for (vocab) |present| {
            if (present) vs += 1;
        }
        self.vocab_size = vs;
        self.byte_mixer_output = 0.0;
        self.nonstationary = .{};
        self.run_map = RunMap.init();
        self.grand = GlibcRand.init(seed);
        if (comptime EXTSHED) {
            if (!extshed_announced) {
                extshed_announced = true;
                std.debug.print("extshed: ENGAGED — 28 external layer-0 readout inputs shed (bracket 1 + direct 1 + match 10 + indirect 16 -> exact neutral 0.0); longest_match gating RETAINED (match perceive+byteUpdate verbatim); manager streams untouched; hashes_ind1/2/3/5 derivation skipped; ppmd/lstm/wordlbl untouched\n", .{});
            }
        }
        if (comptime AD_ON) self.adepth = .{};
        if (comptime WL_ON) self.wlbl = try wordlbl_mod.WordLbl.init(a);

        // LAB dump: only the FIRST constructed predictor (the -c/-e main coding
        // predictor) captures; the -e dict/order self-compression predictors skip.
        if (comptime GATE_DUMP_ON) {
            const inst = gate_instances.fetchAdd(1, .monotonic);
            self.gdump = if (inst == 0) try GateDumper.init(a) else null;
        }

        // sigmoid + sse
        self.sig = try Sigmoid.init(a, 100001);
        self.sse = try SSE.init(a);

        // context manager
        self.mgr = try cml.ContextManagerLex.init(a);

        // fxcm. `dict==null` -> no-dict path (matches the oracle's nodict goldens);
        // otherwise fxcm loads the WRT dict (fxcmv1.cpp dosym).
        const fx_strt = fx_table_gen.genStrt();
        const fx_sqt = fx_table_gen.genSqt();
        var fx_tbl_root = try fx_table_gen.Tables.new(a);
        const fx_tbl: cmfast_m.Tables = @as(*const cmfast_m.Tables, @ptrCast(&fx_tbl_root)).*;
        self.fxcm = try FxcmV26.new(a, dict, &fx_strt, &fx_sqt, fx_tbl);

        // fxcm_stretched table: p = RawPredictionProbability(raw) = squash(raw)/4095,
        // clamped to [1e-4,1-1e-4], then Logit via the 100001-sigmoid.
        self.fxcm_neutral = self.sig.logit(0.5); // == 0
        // -Dfxcm-zero-neutral — see ZERO_NEUTRAL above. conv = 1/4095 is cmix's
        // RawPredictionProbability; mode 2 uses 1/4096, which is what makes
        // squash(0) = 2048 map to exactly 0.5.
        const conv: f32 = if (ZERO_NEUTRAL == 2) 1.0 / 4096.0 else 1.0 / 4095.0;
        var raw: i32 = -2047;
        while (raw <= 2047) : (raw += 1) {
            const sqv: i32 = self.fxcm.tables.squash(raw);
            var p: f32 = @as(f32, @floatFromInt(sqv)) * conv;
            if (p < 1.0e-4) p = 1.0e-4 else if (p > 1.0 - 1.0e-4) p = 1.0 - 1.0e-4;
            self.fxcm_stretched[@intCast(raw + 2047)] = self.sig.logit(p);
        }
        // Mode 1: surgical — only the "no opinion" point is snapped to the exact
        // neutral, so every other slot value is bit-identical to stock.
        if (comptime ZERO_NEUTRAL == 1) self.fxcm_stretched[2047] = self.fxcm_neutral;
        self.fxcm_stretched[4095] = self.fxcm_neutral;
        if (comptime LG_FX) {
            // Central difference over the live range [0..4094] (raw in
            // [-2047, 2047]); index 4095 is an unreachable pad (clpbounds raw
            // at +-2047), so its slope is set to 0.
            for (0..4095) |i| {
                const lo: usize = if (i == 0) 0 else i - 1;
                const hi: usize = @min(i + 1, 4094);
                self.fxcm_dvdraw[i] = (self.fxcm_stretched[hi] - self.fxcm_stretched[lo]) /
                    @as(f32, @floatFromInt(hi - lo));
            }
            self.fxcm_dvdraw[4095] = 0;
        }

        // ---- model construction (predictor.cpp AddBracket/AddPPMD/AddWord/
        //      AddMatch/AddDoubleIndirect), in the exact order that drives the
        //      glibc randoffset sequence. ----
        const shared_map = self.mgr.shared_map;

        // AddBracket
        const bracketCtx = self.mgr.addBracketContext(256, 15);
        self.bracket = Bracket.createLex(a, &self.mgr.bit_context, 200, 10, 100000, &self.vocab);
        self.direct = Direct.create(a, bracketCtx.value_ptr(), &self.mgr.bit_context, 30, 0, @intCast(bracketCtx.size));
        var ns_i: usize = 0;
        self.indirect_ns[ns_i] = self.newIndirectNs(bracketCtx.value_ptr(), 300, shared_map);
        ns_i += 1;

        // AddPPMD. Order comes from -Dppm-order (default 25 = stock cmix-lex,
        // keeping the inputs590/final-prob goldens; the banked ship recipe bakes
        // 20 — −3,949 @100 MB and faster, BUILD_LOG iters 95-100/124).
        self.ppmd = PPMDLex.create(a, &self.mgr.bit_context, @import("build_options").ppm_order, PPMD_MEMORY_MB, &self.vocab);

        // AddWord — 10 sparse indirect_ns (delta 200)
        const word_ns = [_][]const u32{
            &.{0},      &.{ 0, 1 }, &.{1},         &.{ 1, 2 },    &.{ 1, 3 },
            &.{ 2, 3 }, &.{ 3, 4 }, &.{ 1, 2, 4 }, &.{ 2, 3, 4 }, &.{2},
        };
        for (word_ns) |orders| {
            const sp = self.mgr.addSparse(orders);
            self.indirect_ns[ns_i] = self.newIndirectNs(sp.value_ptr(), 200, shared_map);
            ns_i += 1;
        }

        // AddWord — 5 word-sparse match models; indirect_r on the `{1}` run-map.
        var mm_i: usize = 0;
        const word_match = [_][]const u32{ &.{0}, &.{1}, &.{ 1, 3 }, &.{ 1, 2, 3 }, &.{ 7, 2 } };
        for (word_match) |orders| {
            const sp = self.mgr.addSparse(orders);
            self.match_models[mm_i] = Match.create(a, self.mgr.history, sp.value_ptr(), &self.mgr.bit_context, 200, 0.5, 2_000_000, &self.mgr.longest_match);
            mm_i += 1;
            if (orders.len == 1 and orders[0] == 1) {
                self.indirect_r = self.newIndirectR(sp.value_ptr(), 200, shared_map);
            }
        }

        // AddMatch — 5 hash match models (delta 0.5, limit 200, min(2e6,size)).
        const hash_params = [_][2]u32{ .{ 0, 8 }, .{ 1, 8 }, .{ 7, 4 }, .{ 11, 3 }, .{ 13, 2 } };
        for (hash_params) |hp| {
            const ch = self.mgr.addContextHash(hp[0], hp[1]);
            const msize: u64 = @min(@as(u64, 2_000_000), ch.size);
            self.match_models[mm_i] = Match.create(a, self.mgr.history, ch.value_ptr(), &self.mgr.bit_context, 200, 0.5, @intCast(msize), &self.mgr.longest_match);
            mm_i += 1;
        }

        // AddDoubleIndirect — ind1,ind2,ind3,ind5 (delta 400). ind4 is DEAD.
        self.indirect_ns[ns_i] = self.newIndirectNs(&self.mgr.ind1, 400, shared_map);
        ns_i += 1;
        self.indirect_ns[ns_i] = self.newIndirectNs(&self.mgr.ind2, 400, shared_map);
        ns_i += 1;
        self.indirect_ns[ns_i] = self.newIndirectNs(&self.mgr.ind3, 400, shared_map);
        ns_i += 1;
        self.indirect_ns[ns_i] = self.newIndirectNs(&self.mgr.ind5, 400, shared_map);
        ns_i += 1;
        std.debug.assert(ns_i == 15);
        std.debug.assert(mm_i == 10);

        // ---- AddMixers ----
        // byte-mixer LSTM. Construct with a throwaway rng, then overwrite the
        // Glorot weights with the glibc randstream (continuing from the 16
        // offset draws) so they match the oracle bit-for-bit.
        var seed_prng = std.Random.DefaultPrng.init(0);
        var tmp_rng = seed_prng.random();
        // LSTM hyperparameters. Build-option defaults (-Dlstm-lr/-Dlstm-cells/
        // -Dlstm-layers; stock cmix-lex = 0.03 / 170 / 1) overridable via env
        // (ZMIX_LSTM_LR/CLIP/CELLS/HORIZON/LAYERS) for experiments. The @Vector
        // reduction changes the training landscape, so re-tuning for zmix's own
        // numerics happens through the env knobs; the SHIP recipe shape is baked
        // via the build options — the hermetic ship (root.zmix_hermetic) compiles
        // the env reads away entirely and always runs the baked values.
        const lr = envF32("ZMIX_LSTM_LR", build_options.lstm_lr);
        const clip = envF32("ZMIX_LSTM_CLIP", 10.0);
        const cells = envUsize("ZMIX_LSTM_CELLS", build_options.lstm_cells);
        const horizon = envUsize("ZMIX_LSTM_HORIZON", 128);
        const layers = envUsize("ZMIX_LSTM_LAYERS", build_options.lstm_layers);
        self.lstm = Lstm.init(a, vs, vs, cells, layers, horizon, lr, clip, &tmp_rng);
        self.initLstmWeightsGlibc();
        self.byte_mixer = ByteMixer.create(a, 1, &self.mgr.bit_context, &self.vocab, vs, self.lstm);
        // -Dfree-prior-b: wire the record trampoline, but leave the sink NULL so
        // recording is OFF until a coder loop with a live schedule enables it
        // (fpPriorEnableRecording). This keeps -Dfree-prior-b inert when no
        // schedule is set (no stray file, no wasted writes). Comptime-dead off.
        if (comptime build_options.free_prior_b) {
            self.byte_mixer.fp_prior_record = FpPriorRec.recordTrampoline;
        }
        // -Dtransformer: load the frozen 6M transformer and install it in the
        // byte-mixer slot. Comptime-dead at default.
        //
        // ⚠ The LSTM above is STILL constructed under -Dtransformer, and that is
        // deliberate for now: `initLstmWeightsGlibc` draws from the shared
        // glibc-rand stream, and fx2-cmix-transformer's own integration comment
        // (predictor.cpp, `byte_mixer_.emplace(..., transformer_ ? nullptr :
        // new Lstm(...))`) records that they could skip it only because
        // "nothing after this point draws from rand". zmix has NOT verified
        // that, and skipping the draws would shift every other model's random
        // state and change every archive for a reason unrelated to the lever.
        // Removing the allocation is a measured ~7 MB of RSS and ~16-30 KB of S
        // (design memo §4.4, §5) and is OWED work, gated on that verification.
        if (comptime tfmod.enabled) {
            // ★ HERO's tap has no transformer analogue and FAILS SILENTLY, so it
            // is refused at compile time rather than degraded quietly.
            // `cacheProbeHidden` (1341) returns `lstm.hiddenState`, the
            // layer-0 recurrent h. Under -Dtransformer the LSTM is allocated but
            // never `perceive`d, so h stays frozen at its init value for the
            // whole run and HERO collapses to an intercept-only head that still
            // costs its full wall — a green build producing a silently different
            // (and strictly worse) model. Exactly the failure class the
            // wrong-pid and green-gate-on-the-wrong-machine rules exist to stop.
            //
            // The natural substitute EXISTS — the transformer's residual stream
            // is also 192-wide (d_model = 192) — but HERO's value was measured
            // against LSTM hidden semantics and owes a re-measurement, and their
            // kernels do not currently export the stream. Design memo §3.4.4.
            comptime {
                if (@import("build_options").hero)
                    @compileError("-Dtransformer with -Dhero: HERO reads lstm.hiddenState(), which is frozen at its init value when the transformer replaces the LSTM. Build with -Dhero=false, or implement the d_model=192 residual-stream tap and re-measure HERO against it (design memo §3.4.4).");
            }
            // SHIP PATH FIRST. `ship.zig` slices `tfweights_comp` out of the running
            // artifact's own tail and parks it in `tfmod.Transformer.embedded_blob`.
            // Preferring it means a shipped binary NEVER resolves a build-time path
            // — the defect that made an earlier build unsubmittable (an absolute
            // an absolute build-machine path baked into `decomp_bin`, which panicked
            // on a judge box after passing every ratio, RAM and wall gate).
            if (tfmod.Transformer.embedded_blob) |blob| {
                self.byte_mixer.tf = tfmod.Transformer.createFromBlob(a, blob, build_options_tf.transformer_attn) catch
                    std.debug.panic("-Dtransformer: failed to load the {d}-byte embedded weights blob", .{blob.len});
            } else if (comptime WEIGHTS_EMBEDDED_ONLY) {
                // A root that declares `pub const weights_embedded_only = true`
                // (`src/form1.zig`) has NO CLI at all, so neither remaining source
                // below is reachable there: `weights_path_override` is set only by
                // `ship.zig`'s `-c`/`-d` argv, and `-Dtransformer-weights` is a
                // build-time path a judge machine need not have. Refusing here is
                // the whole point — a shipped op that reaches a path source is a
                // submission bug, and this makes it a compile-time impossibility
                // rather than a run-time hope.
                std.debug.panic(
                    "-Dtransformer: this root loads weights ONLY from its own artifact tail, and none was set (weights_embedded_only)",
                    .{},
                );
            } else {
                // No tail to read. Two remaining sources, in order:
                //   (a) `Transformer.weights_path_override` — the EXPLICIT path
                //       `zmix_ship -c/-d <in> <out> <tfweights>` was given. This is
                //       the construct-time path (`construct_ship.sh` self-compresses
                //       english.dic + the order asset with the bare engine, whose tail
                //       is empty), and it is what a COMMITTEE REBUILD runs.
                //   (b) `-Dtransformer-weights`, now a CWD-RELATIVE default
                //       (`assets/6m-q4-fp32.tfwc2`, shipped with the source) — dev
                //       builds and the bench CLI only.
                // Until (b) was the only source here AND was an ABSOLUTE
                // dev-box path, so a rebuild on any other machine died at the first
                // `-c` with `weights_io: cannot open an absolute build-machine path. Reaching (b)
                // inside a shipped op is still a submission bug — but it can no longer
                // bake one developer's filesystem into `decomp_bin`, which is charged 2x.
                const wp = tfmod.Transformer.weights_path_override orelse
                    build_options_tf.transformer_weights;
                var pathbuf: [512]u8 = undefined;
                if (wp.len + 1 > pathbuf.len) @panic("-Dtransformer-weights: path too long");
                @memcpy(pathbuf[0..wp.len], wp);
                pathbuf[wp.len] = 0;
                const pathz: [:0]const u8 = pathbuf[0..wp.len :0];
                self.byte_mixer.tf = tfmod.Transformer.create(a, pathz, build_options_tf.transformer_attn) catch
                    std.debug.panic("-Dtransformer: failed to load weights from '{s}'", .{wp});
            }
        }

        // mixer inputs
        self.mi0 = MixerInputLex.init(a, &self.sig, 1.0e-4);
        self.mi0.setNumModels(NIN);
        self.mi1 = MixerInputLex.init(a, &self.sig, 1.0e-4);
        self.mi1.setNumModels(25);

        // 23 layer-0 mixers (context, learning_rate, extra_input_size = index).
        const Ctx = struct { ptr: *const u64, lr: f32 };
        const m = self.mgr;
        const table = [23]Ctx{
            .{ .ptr = &m.mx9, .lr = 0.005 },
            .{ .ptr = &m.mx10, .lr = 0.0005 },
            .{ .ptr = &m.mx11, .lr = 0.005 },
            .{ .ptr = &m.mx12, .lr = 0.0005 },
            .{ .ptr = &m.mx13, .lr = 0.005 },
            .{ .ptr = &m.mxx, .lr = 0.001 },
            .{ .ptr = &m.recent_bytes[2], .lr = 0.002 },
            .{ .ptr = &m.line_break, .lr = 0.0007 },
            .{ .ptr = &m.longest_match, .lr = 0.0005 },
            .{ .ptr = &m.mx19cxt, .lr = 0.002 },
            .{ .ptr = &m.auxiliary_context, .lr = 0.0005 },
            .{ .ptr = &m.mx18, .lr = 0.001 },
            .{ .ptr = &m.mx7, .lr = 0.001 },
            .{ .ptr = &m.wordscxt, .lr = 0.005 },
            .{ .ptr = &m.b2streamcxt, .lr = 0.001 },
            .{ .ptr = &m.mx5, .lr = 0.001 },
            .{ .ptr = &m.mx6, .lr = 0.005 },
            .{ .ptr = &m.b3streamcxt, .lr = 0.001 },
            .{ .ptr = &m.mx8, .lr = 0.001 },
            .{ .ptr = &m.mx17, .lr = 0.005 },
            .{ .ptr = &m.mx16, .lr = 0.005 },
            .{ .ptr = &m.mx14, .lr = 0.005 },
            .{ .ptr = &m.mx15, .lr = 0.005 },
        };
        // TUNING KNOB (task #21): global scale on all mixer learning rates
        // (hand-tuned on enwik8-era data, re-optimized for enwik9
        // unimodal 100m curve, minimum at 0.7 = -1,838). The comptime default
        // (-Dmix-lr-scale, ship recipe "0.7") is what the hermetic ship bakes;
        // ZMIX_MIX_LR_SCALE env-overrides it in runner_lex for experiments.
        const mix_lr_scale = envF32("ZMIX_MIX_LR_SCALE", @import("build_options").mix_lr_scale);
        // coldram: rows go to a single registered bump pool (first-touch
        // order preserved); identical values, page-manageable home. With the
        // knob off this IS `a`.
        const mixer_a = coldram.rowPoolAllocator(a);
        for (0..23) |i| {
            self.mixers0[i] = MixerLex.init(mixer_a, &self.mi0, table[i].ptr, table[i].lr * mix_lr_scale, i);
        }
        // -Dm0lr-comp8: measured mixer0 per-input LR composite (m0lr depth
        // ladder a–g + h-batch keeper evals, retention 0.886/0.895 at
        // 268/403MB). ABSOLUTE post-scale values — the keeper arms replaced
        // the final learning_rate with exactly these, so the override sits
        // AFTER the mix_lr_scale loop. Default false = loop untouched,
        // bit-identical stock.
        if (comptime @import("build_options").m0lr_comp8) {
            const comp8 = [_]struct { i: usize, lr: f32 }{
                .{ .i = 1, .lr = 0.000245 }, // m1_mx10_dn
                .{ .i = 8, .lr = 0.000455 }, // m8_lgm_up (depth-growing)
                .{ .i = 12, .lr = 0.00049 }, // m12_mx7_dn
                .{ .i = 15, .lr = 0.00049 }, // m15_mx5_dn
                .{ .i = 16, .lr = 0.00245 }, // m16_mx6_dn
                .{ .i = 18, .lr = 0.00049 }, // m18_mx8_dn
                .{ .i = 19, .lr = 0.00245 }, // m19_mx17_dn
                .{ .i = 20, .lr = 0.00245 }, // m20_mx16_dn
                // comp10 extension (batch-j survivors; keeper-eval composite
                // retention 0.828/0.816 @268/403MB vs the naive sum —):
                .{ .i = 0, .lr = 0.00049 }, // m0_j_dn
                .{ .i = 13, .lr = 0.00049 }, // m13_j_dn (−2,443 B/win @268MB solo)
            };
            for (comp8) |ov| self.mixers0[ov.i].learning_rate = ov.lr;
        }
        self.mi0.setExtraInputSize(23);

        // layer-1 mixer: 25 inputs, zero context, lr 0.0003, no extra inputs.
        // -Dmix-l1-rate overrides the stock 0.0003 literal (default = 0.0003 =
        // bit-identical). Post-e2e this is a META-LR: W1 is the leading term of
        // gamma, so it sets how fast the layer-0 LR allocation adapts.
        self.mixer1 = MixerLex.init(mixer_a, &self.mi1, &self.mgr.zero_context, mixer_options.mix_l1_rate * mix_lr_scale, 0);

        if (comptime mixer_options.mixer_e2e) {
            self.mixer1.seedHead(23, 1.0 / 23.0);
        }

        if (comptime RLS_ON) {
            // NOTHING OUTSIDE THE BINARY : every parameter is a
            // comptime -D option. The lab tree read ZMIX_RLS_* env vars; that
            // path is deliberately not ported.
            self.rls = rls_mod.Rls.init(.{
                .lambda = rls_options.mixer1_rls_lambda,
                .p0 = rls_options.mixer1_rls_p0,
                .h_min = rls_options.mixer1_rls_hmin,
                .step_cap = rls_options.mixer1_rls_stepcap,
                .guard = rls_options.mixer1_rls_guard,
                .mode = @enumFromInt(RLS_MODE),
                .sgd_lr = @as(f64, mixer_options.mix_l1_rate) * @as(f64, mix_lr_scale),
                .sgd_floor = @as(f64, build_options.mixer_decay_floor),
            });
        }

        if (comptime shadow_mix.enabled) {
            var sctx: [23]*const u64 = undefined;
            var slr: [23]f32 = undefined;
            for (0..23) |i| {
                sctx[i] = table[i].ptr;
                slr[i] = self.mixers0[i].learning_rate; // post-scale, post-comp8
            }
            self.shadow = shadow_mix.Shadow.create(a, &self.sig, &sctx, &slr, self.mixer1.learning_rate, &self.mgr.long_bit_context);
        }

        if (comptime coldram.enabled) {
            // cold-eligible slabs from the census (exp-coldram-census
            // preflight Q3): mx_a Mixer1 banks + lex indirect-hash tables +
            // shared_map. Deliberately NOT registered: cm banks / dcsm
            // (page-hot by hash spreading), mgr.history (read-heavy ring),
            // LSTM (hot every byte).
            for (&self.fxcm.mx_a) |*mxa| coldram.registerRange("fx.mx_a", mxa.wx);
            coldram.registerRange("lex.shared_map", self.mgr.shared_map);
            coldram.registerRange("lex.ind1", self.mgr.hashes_ind1);
            coldram.registerRange("lex.ind2", self.mgr.hashes_ind2);
            coldram.registerRange("lex.ind3", self.mgr.hashes_ind3);
            coldram.registerRange("lex.ind5", self.mgr.hashes_ind5);
        }

        return self;
    }

    /// The layer-1 probability that the DOWNSTREAM GRADIENT should differentiate
    /// through (-Dmixer-e2e, -Dleafgrad). ⚠ NOT necessarily the head that coded
    /// the bit: routing the gradient through the RLS head is the self-consistent
    /// choice and is MEASURED at +12.86 % on the 29-flag ship line, because the
    /// RLS row is Newton-scaled and gamma propagates that down 23 cascade layers.
    /// The default keeps layer-0 training through the still-running SGD row.
    /// Comptime-folds to the stock expression when RLS is off.
    inline fn headLogistic(self: *PredictorLex) f32 {
        if (comptime RLS_ON and RLS_E2E_HEAD) return Sigmoid.logistic(@as(f32, @floatCast(self.rls.z)));
        return Sigmoid.logistic(self.mixer1.p);
    }

    /// The 25 layer-1 weights of that same head, as f32. `scratch` is only
    /// written on the RLS path; the stock path returns the resolved row's own
    /// storage, so a stock build does no extra work.
    inline fn headRow(self: *PredictorLex, scratch: *[rls_mod.D]f32) []const f32 {
        if (comptime RLS_ON and RLS_E2E_HEAD) {
            self.rls.rowF32(scratch);
            return scratch[0..];
        }
        return self.mixer1.resolvedRow().weights[0..rls_mod.D];
    }

    /// Full teardown, ordered leaf-first. The C++ scopes the Predictor and calls
    /// FreeFxcmMemory+ malloc_trim(0) before the un-transform chain
    /// (cmix-lex runner.cpp:308-321); here EVERY owned allocation is freed —
    /// zero_alloc big tables munmap (RSS returns immediately), the rest goes back
    /// to the caller's allocator. Callers on the -e/-D paths MUST run this before
    /// the un-transform stages or the 10 GB Hutter RSS gate is breached.
    /// ★ THE CANONICAL-205 SUBMISSION GATE, at its call site.
    ///
    /// The submission gate requires a hard assertion that their 15-token
    /// `kArticleSeparator` decodes to exactly 243,426 occurrences on the e9
    /// post-WRT stream. The map itself is asserted at COMPILE time
    /// (`mixer/transformer.zig`'s comptime block: both anchor ranks plus the
    /// full 15-token constant, zero bytes); this is the runtime half — the
    /// running count, checked at end-of-stream.
    ///
    /// ⚠ It was written on and had ZERO CALLERS until
    /// including on the live e9 leg (
    /// §6). A gate nothing calls is a comment. Comptime-dead without
    /// `-Dtransformer`.
    pub fn checkTransformerSeparators(self: *PredictorLex, total_bytes: u64) void {
        if (comptime !tfmod.enabled) return;
        if (self.byte_mixer.tf) |tf| tf.checkSeparatorCount(total_bytes);
    }

    pub fn deinit(self: *PredictorLex) void {
        coldram.dumpStats(); // end-of-run counters (ZMIX_COLDRAM_STATS only)
        if (comptime shadow_mix.enabled) {
            self.shadow.report();
            self.shadow.destroy();
        }
        if (comptime AD_ON) self.adepthDumpStats(); // ZMIX_ADEPTH_STATS only
        const a = self.alloc;
        if (comptime GATE_DUMP_ON) {
            if (self.gdump) |d| d.deinit();
        }
        if (comptime @import("models/indirect.zig").CENSUS) {
            const map = self.mgr.shared_map;
            var touched: u64 = 0;
            for (map) |b| { if (b != 0) touched += 1; }
            std.debug.print("ind-census: shared_map {d} B, touched {d} ({d:.2}%)\n",
                .{ map.len, touched, 100.0 * @as(f64, @floatFromInt(touched)) / @as(f64, @floatFromInt(map.len)) });
            var tot_d: u64 = 0; var tot_o: u64 = 0;
            for (0..16) |i| {
                const ind = if (i < 15) self.indirect_ns[i] else self.indirect_r;
                std.debug.print("ind-census: model {d:>2} distinct_ctx {d:>10} occurrences {d:>11} occ/ctx {d:.2} window_bytes {d}\n",
                    .{ i, ind.census.distinct, ind.census.occ,
                       @as(f64, @floatFromInt(ind.census.occ)) / @as(f64, @floatFromInt(@max(ind.census.distinct, 1))),
                       ind.census.distinct * 257 });
                tot_d += ind.census.distinct; tot_o += ind.census.occ;
            }
            std.debug.print("ind-census: TOTAL distinct {d} -> claimed {d} B vs map {d} B = OVERSUBSCRIPTION {d:.1}x\n",
                .{ tot_d, tot_d * 257, map.len,
                   @as(f64, @floatFromInt(tot_d * 257)) / @as(f64, @floatFromInt(map.len)) });
        }
        if (comptime LG_DIAG) {
            const d = &self.lg_diag;
            const n: f64 = @floatFromInt(@max(d.bits, 1));
            const tot = d.g_bracket + d.g_fxcm + d.g_direct + d.g_match + d.g_ind + d.g_ppmd + d.g_bmix;
            std.debug.print("leafgrad-diag: bits {d} sum|Gamma| {e}\n", .{ d.bits, tot / n });
            const pct = struct {
                fn f(v: f64, t: f64) f64 {
                    return if (t > 0) 100.0 * v / t else 0.0;
                }
            }.f;
            std.debug.print("leafgrad-diag: group  bracket {d:.4}%  fxcm[1..560] {d:.4}%  direct {d:.4}%  match {d:.4}%  INDIRECT[572..587] {d:.4}%  ppmd {d:.4}%  bytemix {d:.4}%\n", .{
                pct(d.g_bracket, tot), pct(d.g_fxcm, tot),  pct(d.g_direct, tot),
                pct(d.g_match, tot),   pct(d.g_ind, tot),   pct(d.g_ppmd, tot),
                pct(d.g_bmix, tot),
            });
            std.debug.print("leafgrad-diag: mean |dL/dtau| over the 16 indirect leaves = {e}  (offline optimum step was Z*w = 0.3/15 = 2.0e-2)\n", .{d.step_abs / n / 16.0});
            if (comptime LG_FX) {
                // ★ THE REACH-RESPONSE COVERAGE NUMBER: |Gamma| share of exactly
                // the slots -Dleafgrad-fxcm can move, measured from fxcm's own
                // per-bit reach mask (not inferred from a static slot census).
                std.debug.print("leafgrad-diag: REACH(-Dleafgrad-fxcm, CM3 adaptive ts cells) = {d:.4}% of sum|Gamma|   [fxcm total {d:.4}%, of which frozen/unreached {d:.4}%]\n", .{
                    pct(d.g_cm3_adapt, tot), pct(d.g_fxcm, tot), pct(d.g_fxcm - d.g_cm3_adapt, tot),
                });
                const lgn = cmfast_m.LG_STAT.n();
                const sn: f64 = @floatFromInt(@max(lgn, 1));
                std.debug.print("leafgrad-diag: cm3 cell steps {d}  mean |dL/dtau_cell| = {e}  (incumbent local rule's own step at p=0.5 is 1.22e-4)\n", .{ lgn, cmfast_m.LG_STAT.sum() / sn });
                const lv = cmfast_m.LG_STAT.live();
                std.debug.print("leafgrad-diag: hazard(a) LUT saturation: {d} of {d} live slot-gradients hit a flat st1/st2 plateau = {d:.4}% killed by clp()\n", .{ cmfast_m.LG_STAT.sat(), lv, pct(@floatFromInt(cmfast_m.LG_STAT.sat()), @floatFromInt(@max(lv, 1))) });
            }
        }
        if (comptime build_options.free_prior_b) self.fp_prior.deinit();
        self.mixer1.deinit();
        for (&self.mixers0) |*mx| mx.deinit();
        self.mi1.deinit();
        self.mi0.deinit();
        if (comptime tfmod.enabled) if (self.byte_mixer.tf) |tf| tf.destroy(self.alloc);
        self.byte_mixer.destroy();
        self.lstm.deinit();
        self.indirect_r.destroy();
        for (self.indirect_ns) |ind| ind.destroy();
        for (self.match_models) |mm| mm.destroy();
        self.ppmd.destroy();
        self.direct.destroy();
        self.bracket.destroy();
        self.fxcm.deinit();
        self.mgr.deinit();
        self.sse.deinit(a);
        if (comptime WL_ON) self.wlbl.deinit(a);
        self.sig.deinit();
        a.destroy(self);
    }

    fn newIndirectNs(self: *PredictorLex, byte_context: *const u64, delta: f32, map: []u8) *Indirect {
        var tmp_prng = std.Random.DefaultPrng.init(0);
        var tmp_rng = tmp_prng.random();
        const ind = Indirect.create(self.alloc, self.nonstationary.state(), byte_context, &self.mgr.bit_context, delta, map, &tmp_rng);
        ind.map_offset = self.grand.nextMod(map.len - 257);
        return ind;
    }

    fn newIndirectR(self: *PredictorLex, byte_context: *const u64, delta: f32, map: []u8) *Indirect {
        var tmp_prng = std.Random.DefaultPrng.init(0);
        var tmp_rng = tmp_prng.random();
        const ind = Indirect.create(self.alloc, self.run_map.state(), byte_context, &self.mgr.bit_context, delta, map, &tmp_rng);
        ind.map_offset = self.grand.nextMod(map.len - 257);
        return ind;
    }

    // LstmLayer Glorot init (lstm-layer.hpp:75-85), glibc randorder.
    // Only layers_[0] gets the glibc-parity Glorot; with ZMIX_LSTM_LAYERS>1 the
    // extra layers keep their DefaultPrng(0) init — deterministic and identical
    // on encode/decode, so the codec stays consistent.
    fn initLstmWeightsGlibc(self: *PredictorLex) void {
        const layer = &self.lstm.layers_[0];
        const aux_plus_out: usize = @as(usize, self.vocab_size) + @as(usize, self.vocab_size);
        const val: f32 = std.math.sqrt(@as(f32, 6.0) / @as(f32, @floatFromInt(aux_plus_out)));
        const low: f32 = -val;
        const range: f32 = 2.0 * val;
        const ncell: usize = self.lstm.num_cells_;
        const rowlen: usize = layer.forget_gate_.weights_[0].len;
        for (0..ncell) |i| {
            for (0..rowlen) |j| {
                layer.forget_gate_.weights_[i][j] = low + self.grand.nextFloat() * range;
                layer.input_node_.weights_[i][j] = low + self.grand.nextFloat() * range;
                layer.output_gate_.weights_[i][j] = low + self.grand.nextFloat() * range;
            }
            layer.forget_gate_.weights_[i][rowlen - 1] = 1;
        }
    }

    /// Read-only accessor for the hidden-state levers (HERO head / episodic
    /// cache): the live layer-0 h[t-1]. Same name as the pinned trace
    /// producer's tap (48a7ede9) to keep the provenance link explicit.
    pub inline fn cacheProbeHidden(self: *const PredictorLex) []const f32 {
        return self.lstm.hiddenState();
    }

    /// -Dadaptive-depth: did the just-predicted bit skip the cascade? Runner
    /// coder loops consult this to bypass hero.mix/observe on skipped bits (the
    /// coder receives p_head directly — preflight §1 step 7). Comptime-folds to
    /// `false` when the knob is off, so every call site keeps stock codegen.
    pub inline fn adepthSkipped(self: *const PredictorLex) bool {
        if (comptime !AD_ON) return false;
        return self.adepth.skip;
    }

    /// -Dadaptive-depth coverage counters, printed at deinit when
    /// ZMIX_ADEPTH_STATS is set (non-hermetic roots only — the ship binary
    /// reads no env). bits_skipped/bits is the MAC-side win; bytes_allskip/bytes
    /// is the per-byte row-traffic win (preflight §4 live-set narrative).
    fn adepthDumpStats(self: *const PredictorLex) void {
        if (comptime !AD_ON) return;
        if (comptime (HERMETIC or builtin.os.tag == .windows)) return;
        if (std.posix.getenv("ZMIX_ADEPTH_STATS") == null) return;
        const ad = &self.adepth;
        const fb = if (ad.bits == 0) 0.0 else 100.0 * @as(f64, @floatFromInt(ad.bits_skipped)) / @as(f64, @floatFromInt(ad.bits));
        const fby = if (ad.bytes == 0) 0.0 else 100.0 * @as(f64, @floatFromInt(ad.bytes_allskip)) / @as(f64, @floatFromInt(ad.bytes));
        std.debug.print(
            "adepth-stats: tau={d} bits={d} skipped={d} ({d:.2}%)  bytes={d} allskip={d} ({d:.2}%)\n",
            .{ AD_TAU, ad.bits, ad.bits_skipped, fb, ad.bytes, ad.bytes_allskip, fby },
        );
    }

    // ---- Predict (predictor.cpp:190-272) — leaves the 590 inputs in mi0.inputs. ----
    pub fn predict(self: *PredictorLex) f32 {
        @setFloatMode(.optimized);
        const mi0 = &self.mi0;

        // sm35: prefetch the 16 indirect models' shared-map bytes up front — each
        // is a cold miss into the 100 MB shared map, and issuing them all here
        // hides the latency behind the bracket/fxcm/direct/match predicts that run
        // before the indirects predict at [572..587]. Pure cache hint.
        // -Dextshed: the indirects never read the map, so the prefetches go too.
        if (comptime !EXTSHED) {
            for (self.indirect_ns) |ind| ind.prefetch();
            self.indirect_r.prefetch();
        }

        // [0] bracket. -Dextshed: readout shed — exact neutral, model not called.
        if (comptime EXTSHED)
            mi0.setStretchedInputUnchecked(0, self.fxcm_neutral)
        else
            mi0.setInput(0, self.bracket.predict()[0]);

        // [1..560] fxcm slots (dense from 0; rest neutral)
        const active = @min(self.fxcm.active, FX_N);
        for (0..active) |j| {
            const raw: i32 = self.fxcm.slots[j];
            mi0.setStretchedInputUnchecked(1 + j, self.fxcm_stretched[@intCast(raw + 2047)]);
        }
        for (active..FX_N) |j| {
            mi0.setStretchedInputUnchecked(1 + j, self.fxcm_neutral);
        }
        // -Dwordlbl: ONE pad slot (559) carries the word-identity expert's logit.
        // Slot 560 is avoided (mixer1 consumes it as fxcm_model_index); the write
        // happens AFTER the pad fill so the neutral never overwrites it. When the
        // expert is quiescent (literal bytes, byte MSBs) forwardreturns exactly
        // 0.0 == fxcm_neutral, so quiescent bits are stock-identical.
        // -Dwordlbl-attn-head claims TWO more pad slots (558/557) for the
        // attention memory's second readout: the candidate-mass bit logit and
        // its coverage. Same argument — quiescent bits emit exactly 0.0, and the
        // pad costs zero weight-row RAM and zero wall (exp-layer0-slot-price).
        if (comptime WL_ON) {
            std.debug.assert(active <= WL_PAD_LO - 1);
            mi0.setStretchedInput(FX_N - 1, self.wlbl.forward());
            if (comptime wordlbl_mod.WL_HEAD) {
                mi0.setStretchedInput(FX_N - 2, self.wlbl.z_head);
                mi0.setStretchedInput(FX_N - 3, self.wlbl.c_head);
            }
        }
        // Stock reads index 560 — the LAST PAD, not the fxcm's prediction. The fxcm
        // emits `active` (525 shipped) slots into [1..active]; [1+active..560] hold
        // fxcm_neutral == sig.logit(0.5) == exactly 0.0. cmix-lex predictor.cpp:213
        // derives the same pad via `input_index - 1`, so the PORT is faithful and the
        // defect is upstream's. -Dfxcm-auxctx-fix repoints the auxiliary_context
        // consumer at the last ACTIVE slot, restoring its selector from 9 live buckets
        // to 16 (MEASURED: -707 B @e8_20m nodict, -3,294 @100MB, wall +0.078%, RSS
        // +384 kB).
        // NOTE the sibling -Dfxcm-skip-fix (mixer1 input 23) is KILLED (+20 B @1m) and
        // is deliberately NOT ported here; bundling the two cancelled auxctx's -25 to -3.
        const fxcm_model_index: usize = FX_N;
        const fxcm_aux_index: usize = if (comptime build_options.fxcm_auxctx_fix)
            (if (active > 0) active else FX_N)
        else
            FX_N;

        // [561] direct · [562..571] match ×10 · [572..586] indirect_ns ×15 ·
        // [587] indirect_r. -Dextshed: all 27 readouts shed to the exact
        // neutral 0.0 (the pad convention — w·0 contributes nothing forward and
        // the SGD update adds lr·err·0 = 0, so the columns are dead exactly as
        // in the wave-2 DROP construction). The matchers still PERCEIVE and
        // BYTE-UPDATE (gating; see perceive); Indirect.predict's map_index
        // advance is paired with its perceive's retreat, both skipped together.
        if (comptime EXTSHED) {
            mi0.setStretchedInputUnchecked(IDX_DIRECT, self.fxcm_neutral);
            for (0..10) |k| mi0.setStretchedInputUnchecked(IDX_MATCH0 + k, self.fxcm_neutral);
            for (0..15) |k| mi0.setStretchedInputUnchecked(IDX_IND0 + k, self.fxcm_neutral);
            mi0.setStretchedInputUnchecked(IDX_INDR, self.fxcm_neutral);
        } else {
            // [561] direct
            mi0.setInput(IDX_DIRECT, self.direct.predict()[0]);

            // [562..571] match ×10
            for (0..10) |k| mi0.setInput(IDX_MATCH0 + k, self.match_models[k].predict()[0]);

            // [572..586] indirect_ns ×15
            for (0..15) |k| mi0.setInput(IDX_IND0 + k, self.indirect_ns[k].predict()[0]);

            // [587] indirect_r
            mi0.setInput(IDX_INDR, self.indirect_r.predict()[0]);
        }

        // [588] ppmd
        mi0.setInput(IDX_PPMD, self.ppmd.predict());

        // [589] byte_mixer_output (override captured here)
        const byte_mixer_override: f32 = if (self.byte_mixer_output == 0 or self.byte_mixer_output == 1) self.byte_mixer_output else -1;
        mi0.setInput(IDX_BMIX, self.byte_mixer_output);
        const byte_mixer_index: usize = IDX_BMIX;

        if (comptime MIXSPARSE) {
            // -Dmix-sparse-diag: how many of the 590 layer-0 inputs are EXACTLY 0
            // (logit(0.5) = "no opinion")? For such an input both the forward term
            // w*x and the SGD update w += lr*err*x are no-ops, so this is the exact
            // size of the sparse-mixer wall opportunity. Side-channel only.
            var nz: u32 = 0;
            for (mi0.inputs) |v| { if (v == 0.0) nz += 1; }
            // LATENT sparsity: how many fxcm slots carry the RAW value 0 (cm3's
            // "no opinion" / st32-inert emission)? Those become logit(squash(0)/4095)
            // = +0.0005, NOT 0.0, so they are invisible to the exact-zero count above
            // and the dot product cannot skip them. If this number is large, an exactly
            // -zero neutral would UNLOCK a sparse path; if it is small, the wall axis
            // has nothing here at all.
            var rz: u32 = 0;
            var runs: u32 = 0;
            var inrun = false;
            for (0..active) |j| {
                if (self.fxcm.slots[j] == 0) {
                    rz += 1;
                    if (!inrun) { runs += 1; inrun = true; }
                } else inrun = false;
            }
            ms_rawzero += rz;
            ms_runs += runs;
            // Block-level sparsity — the quantity that decides whether a sparse path
            // can pay. A scalar gather loop LOSES to a dense SIMD one, so the only
            // profitable form is skipping whole ALIGNED blocks: 8 floats = one AVX2
            // vector, 16 floats = one 64 B cache line. Counted over the 590 mixer
            // inputs as they will actually be laid out (index 1+j for slot j).
            var b8: u32 = 0;
            var b16: u32 = 0;
            var blk: usize = 0;
            while (blk + 8 <= mi0.inputs.len) : (blk += 8) {
                var all = true;
                for (mi0.inputs[blk .. blk + 8]) |v| { if (v != 0.0) { all = false; break; } }
                if (all) b8 += 1;
            }
            blk = 0;
            while (blk + 16 <= mi0.inputs.len) : (blk += 16) {
                var all = true;
                for (mi0.inputs[blk .. blk + 16]) |v| { if (v != 0.0) { all = false; break; } }
                if (all) b16 += 1;
            }
            ms_b8 += b8;
            ms_b16 += b16;
            ms_bits += 1;
            ms_zero += nz;
            if (nz > ms_max) ms_max = nz;
            if (nz < ms_min) ms_min = nz;
            const b = @min(@as(usize, 9), @as(usize, nz) * 10 / mi0.inputs.len);
            ms_hist[b] += 1;
        }
        // auxiliary context = ((Logistic(in[560]) + Logistic(in[589])) / 2) * 15
        const aux_avg: f32 = (Sigmoid.logistic(mi0.inputs[fxcm_aux_index]) + Sigmoid.logistic(mi0.inputs[byte_mixer_index])) / 2.0;
        self.mgr.auxiliary_context = @intFromFloat(aux_avg * 15.0);

        // -Dadaptive-depth gate (preflight §1 steps 1-3). Runs AFTER the model
        // fills + aux-context update (side effects every bit) and BEFORE any
        // cascade work. depth4's exact arithmetic: sequential w·f accumulation,
        // clamp z to ±30, p = 1/(1+exp(−z)), gate on |z_clamped| > tau. On skip
        // the 23 rows are never resolved (no context_map mutation), nothing
        // downstream of this point runs, and the coder gets p_head.
        if (comptime AD_ON) {
            const ad = &self.adepth;
            var z: f32 = ad.w[0];
            inline for (AD_IDX, 0..) |vi, i| {
                z += ad.w[1 + i] * mi0.inputs[vi];
            }
            if (z > 30) z = 30;
            if (z < -30) z = -30;
            const ph: f32 = 1.0 / (1.0 + std.math.exp(-z));
            ad.p = ph;
            ad.skip = @abs(z) > AD_TAU;
            if (ad.skip) {
                // A skipped bit is still CODED — the coder gets p_head — so it
                // must still get a dump record, or the record stream stops being
                // one per coded bit (same defect the slim variant found on the
                // 29-flag line: ~33 % of bits skipped ⇒ nrec % 8 != 0). Flag
                // bit1 marks these records: their z is the ADEPTH HEAD's logit,
                // not the layer-1 pre-SSE logit, and in wide mode the 23 output
                // columns are zeros (the cascade never ran; mi0.extra_inputs
                // still hold the PREVIOUS bit's outputs and must not leak).
                if (comptime GATE_DUMP_ON) {
                    if (self.gdump) |d| self.gateStage(d, z, ph, false, true);
                }
                return ph;
            }
        }

        // -Dmixer-blockskip: compute the all-zero input block mask ONCE, here, after
        // every layer-0 input is set and before any mixer touches them. All 23 mixers
        // then reuse it on both the mix and the update pass. Comptime no-op when off.
        mi0.computeBlockMask();

        // cascade — resolve+prefetch all 23 context rows first (latency hiding;
        // rows are ~2.4KB each and the resolve order matches the mix order, so
        // map state and output bits are identical).
        for (0..23) |i| self.mixers0[i].prefetchRow();
        for (0..23) |i| {
            const p = self.mixers0[i].mix();
            self.mi0.setExtraInput(i, p);
            self.mi1.setStretchedInput(i, p);
        }
        self.mi1.setStretchedInput(23, mi0.inputs[fxcm_model_index]);
        self.mi1.setStretchedInput(24, mi0.inputs[byte_mixer_index]);

        // -Dmixer1-rls: the STOCK mixer1 still mixes and still perceives, so the
        // A/B isolates the LEARNING RULE rather than the presence of a head, and
        // `mixer1.resolvedRow` stays live for any consumer that wants it. Only
        // the coded logit is taken from the RLS head.
        const z_pre: f32 = blk: {
            const z_sgd: f32 = self.mixer1.mix();
            if (comptime RLS_ON) break :blk @floatCast(self.rls.mix(self.mi1.inputs[0..rls_mod.D]));
            break :blk z_sgd;
        };
        var p: f32 = Sigmoid.logistic(z_pre);
        if (comptime RLS_ON) self.rls_p = p;
        const pre_sse: f32 = p;
        p = self.sse.predict(p);
        if (comptime GATE_DUMP_ON) {
            if (self.gdump) |d| self.gateStage(d, z_pre, if (byte_mixer_override >= 0) byte_mixer_override else p, byte_mixer_override >= 0, false);
        }
        if (comptime shadow_mix.enabled) {
            self.shadow.predict(mi0.inputs);
            self.shadow_pre = pre_sse;
            self.shadow_post = p;
            self.shadow_free = byte_mixer_override >= 0;
        }
        if (byte_mixer_override >= 0) return byte_mixer_override;
        return p;
    }

    // ---- LAB gating-context dump: stage this bit's record (comptime-dead). ----
    fn gateStage(self: *PredictorLex, d: *GateDumper, z_pre: f32, p_coded: f32, override: bool, adepth_skipped: bool) void {
        if (comptime GATE_WIDE) {
            // BYTE stride: all 8 bits of every GATE_STRIDE-th coded byte.
            if ((d.ctr / 8) % GATE_STRIDE != 0) return;
        } else {
            if (d.ctr % GATE_STRIDE != 0) return;
        }
        if (GATE_MAX != 0 and d.nrec >= GATE_MAX) return;
        var b: [GATE_REC]u8 = undefined;
        if (comptime GATE_WIDE) {
            const in = self.mi0.inputs;
            for (0..GATE_NIN) |k| GateDumper.putF16(&b, k, in[k]);
            // 23 cascade outputs — mi0.extra_inputs are NEVER cleared (valarray
            // semantics), so on an adaptive-depth skip they hold the PREVIOUS
            // bit's outputs. Write zeros there instead; flags bit1 says so.
            for (0..23) |k| GateDumper.putF16(&b, GATE_NIN + k, if (adepth_skipped) 0.0 else self.mi0.extra_inputs[k]);
            const cbase = (GATE_NIN + 23) * 2;
            for (0..23) |i| {
                const c: u32 = @truncate(self.mixers0[i].context.*);
                std.mem.writeInt(u32, b[cbase + i * 4 ..][0..4], c, .little);
            }
            std.mem.writeInt(u32, b[cbase + 23 * 4 ..][0..4], @intCast((d.ctr / 8) & 0xFFFF_FFFF), .little);
            const zoff = GATE_REC - 8;
            std.mem.writeInt(u32, b[zoff..][0..4], @bitCast(z_pre), .little);
            var dq: i64 = @intFromFloat(1.0 + 65534.0 * p_coded);
            if (dq < 1) dq = 1;
            if (dq > 65535) dq = 65535;
            std.mem.writeInt(u16, b[zoff + 4 ..][0..2], @intCast(dq), .little);
            const bitpos: u8 = @intCast(d.ctr % 8);
            b[zoff + 7] = (if (override) @as(u8, 1) else 0) | (if (adepth_skipped) @as(u8, 2) else 0) | (bitpos << 2);
            d.stage = b;
            d.stage_active = true;
            return;
        }
        const in = self.mi0.inputs;
        // --- 32 slim columns -------------------------------------------------
        GateDumper.putF16(&b, 0, in[0]); // bracket
        GateDumper.putF16(&b, 1, in[IDX_DIRECT]); // direct
        GateDumper.putF16(&b, 2, in[IDX_PPMD]); // ppmd
        GateDumper.putF16(&b, 3, in[IDX_BMIX]); // lstm byte-mixer q
        for (0..10) |k| GateDumper.putF16(&b, 4 + k, in[IDX_MATCH0 + k]); // match[0..9]
        GateDumper.putF16(&b, 14, in[IDX_INDR]); // ind_r
        GateDumper.putF16(&b, 15, in[IDX_IND0]); // ind_ns[0]
        GateDumper.putF16(&b, 16, in[IDX_IND0 + 4]); // ind_ns[4]
        GateDumper.putF16(&b, 17, in[IDX_IND0 + 8]); // ind_ns[8]
        GateDumper.putF16(&b, 18, in[IDX_IND0 + 12]); // ind_ns[12]
        for (0..7) |k| GateDumper.putF16(&b, 19 + k, in[518 + k]); // fxcm exports 518..524
        // Aggregates over the live fxcm slot block — NOT expressible by a linear
        // head over the same inputs (mean is; sd/min/max/|·| are not).
        const active = @min(self.fxcm.active, FX_N);
        var sum: f32 = 0;
        var sumsq: f32 = 0;
        var sumabs: f32 = 0;
        var vmin: f32 = 0;
        var vmax: f32 = 0;
        if (active > 0) {
            vmin = in[1];
            vmax = in[1];
            for (0..active) |j| {
                const x = in[1 + j];
                sum += x;
                sumsq += x * x;
                sumabs += @abs(x);
                if (x < vmin) vmin = x;
                if (x > vmax) vmax = x;
            }
        }
        const n: f32 = @floatFromInt(@max(active, 1));
        const mean = sum / n;
        const varr = @max(sumsq / n - mean * mean, 0.0);
        GateDumper.putF16(&b, 26, mean);
        GateDumper.putF16(&b, 27, @sqrt(varr));
        GateDumper.putF16(&b, 28, vmin);
        GateDumper.putF16(&b, 29, vmax);
        GateDumper.putF16(&b, 30, sumabs / n);
        GateDumper.putF16(&b, 31, in[FX_N]); // pad control — must be exactly 0
        // --- 23 gating contexts (u32 = the row key mixer_lex uses) ------------
        const cbase = GATE_NCOL * 2;
        for (0..GATE_NCTX) |i| {
            const c: u32 = @truncate(self.mixers0[i].context.*);
            std.mem.writeInt(u32, b[cbase + i * 4 ..][0..4], c, .little);
        }
        // --- z, coded prob, flags --------------------------------------------
        const zoff = cbase + GATE_NCTX * 4;
        std.mem.writeInt(u32, b[zoff..][0..4], @bitCast(z_pre), .little);
        var dq: i64 = @intFromFloat(1.0 + 65534.0 * p_coded);
        if (dq < 1) dq = 1;
        if (dq > 65535) dq = 65535;
        std.mem.writeInt(u16, b[zoff + 4 ..][0..2], @intCast(dq), .little);
        // bit0 = LSTM byte-range override fired; bit1 = adaptive-depth skipped
        // (fat-mode skip records: the 23 context columns above are the current
        // context VALUES but no row was resolved this bit).
        b[zoff + 7] = (if (override) @as(u8, 1) else 0) | (if (adepth_skipped) @as(u8, 2) else 0);
        d.stage = b;
        d.stage_active = true;
    }

    // ---- -Dleafgrad-diag LAB census (comptime-dead) ----
    // Where does the coded-length gradient have leverage? Gamma_m = dz/dv_m for
    // ALL 590 layer-0 inputs, accumulated |.| per group. 13,570 extra MAC/bit —
    // never in a measured build.
    fn lgDiagBit(self: *PredictorLex, gam: *const [23]f32, g_final: f32, acc: *const [LG_N]f32) void {
        if (comptime !LG_DIAG) return;
        const d = &self.lg_diag;
        d.bits += 1;
        var full: [NIN]f32 = [_]f32{0} ** NIN;
        for (0..23) |ii| {
            const gv = gam[ii];
            if (gv == 0) continue;
            const wr = self.mixers0[ii].resolvedRow().weights;
            for (0..NIN) |m| full[m] += gv * wr[m];
        }
        d.g_bracket += @abs(full[0]);
        for (1..IDX_DIRECT) |m| d.g_fxcm += @abs(full[m]);
        d.g_direct += @abs(full[IDX_DIRECT]);
        for (IDX_MATCH0..IDX_IND0) |m| d.g_match += @abs(full[m]);
        for (IDX_IND0..IDX_PPMD) |m| d.g_ind += @abs(full[m]);
        d.g_ppmd += @abs(full[IDX_PPMD]);
        d.g_bmix += @abs(full[IDX_BMIX]);
        for (0..LG_N) |k| d.step_abs += @abs(g_final * acc[k]);
    }

    // ---- Perceive (predictor.cpp:274-337). ----
    pub fn perceive(self: *PredictorLex, bit: i32) void {
        @setFloatMode(.optimized);
        if (comptime GATE_DUMP_ON) {
            if (self.gdump) |d| {
                if (d.stage_active) {
                    d.stage[GATE_REC - 2] = @intCast(bit & 0xff);
                    if (d.pos + GATE_REC > d.buf.len) d.flush();
                    @memcpy(d.buf[d.pos..][0..GATE_REC], &d.stage);
                    d.pos += GATE_REC;
                    d.nrec += 1;
                    d.stage_active = false;
                }
                d.ctr += 1;
            }
        }
        // -Dwordlbl: train on EVERY coded bit and advance the causal parse (the
        // σ measurement trained on adaptive-depth-skipped bits too — the mixer
        // merely ignores the input on those).
        if (comptime WL_ON) self.wlbl.perceive(bit);
        coldram.tickBit();

        // -Dadaptive-depth (preflight §1 steps 5-6): the head trains on EVERY
        // coded bit, skipped or not — depth4's exact AdaGrad (lr 0.01, eps 1e-8,
        // f32, per-weight order f[0]=bias then the 38 features). mi0.inputs are
        // untouched between predict and perceive, so the features are re-read in
        // place. Also the coverage counters (byte boundary read the same way the
        // stock byte_update flag reads it, BEFORE updateContexts advances it).
        if (comptime AD_ON) {
            const ad = &self.adepth;
            const er: f32 = ad.p - @as(f32, @floatFromInt(bit));
            var gr: f32 = er; // j=0: bias feature == 1
            ad.g[0] += gr * gr;
            ad.w[0] -= 0.01 * gr / (@sqrt(ad.g[0]) + 1e-8);
            inline for (AD_IDX, 0..) |vi, i| {
                gr = er * self.mi0.inputs[vi];
                ad.g[1 + i] += gr * gr;
                ad.w[1 + i] -= 0.01 * gr / (@sqrt(ad.g[1 + i]) + 1e-8);
            }
            ad.bits += 1;
            if (ad.skip) ad.bits_skipped += 1 else ad.cur_byte_all = false;
            if (self.mgr.bit_context >= 128) {
                ad.bytes += 1;
                if (ad.cur_byte_all) ad.bytes_allskip += 1;
                ad.cur_byte_all = true;
            }
        }

        // -Dleafgrad: hand each Indirect its dL/dtau BEFORE it perceives. Must run
        // here, not after the cascade update: the models perceive first, and the
        // gradient has to be the one of the row/weights that actually produced this
        // bit's prediction. Nothing between predictand this point touches
        // mixers0/mixer1 weights or `p`, and resolvedRowonly reads the row mix
        // already resolved (no context_map mutation), so this is side-effect-free.
        if (comptime LG_ANY) {
            // On an -Dadaptive-depth skip the cascade never ran: `resolved` is null
            // (calling resolvedRowwould INSERT a context row and desync the
            // decoder) and mixer1.p is stale. Zero the gradient on those bits.
            const lg_live: bool = if (comptime AD_ON) !self.adepth.skip else true;
            if (lg_live) {
                const g_final: f32 = self.headLogistic() - @as(f32, @floatFromInt(bit));
                var w1_scratch: [rls_mod.D]f32 = undefined;
                const w1w = self.headRow(&w1_scratch);
                const lo: f32 = self.mi0.stretched_min;
                const hi: f32 = self.mi0.stretched_max;
                var gam: [23]f32 = undefined;
                var gi: usize = 23;
                while (gi > 0) {
                    gi -= 1;
                    var s: f32 = w1w[gi];
                    var k: usize = gi + 1;
                    while (k < 23) : (k += 1) {
                        s += gam[k] * self.mixers0[k].resolvedRow().extra_weights[gi];
                    }
                    const h = self.mixers0[gi].p;
                    gam[gi] = if (h > hi or h < lo) 0.0 else s;
                }
                var acc: [LG_N]f32 = [_]f32{0} ** LG_N;
                if (comptime LG_ON) {
                    for (0..23) |ii| {
                        const gv = gam[ii];
                        if (gv == 0) continue;
                        const wr = self.mixers0[ii].resolvedRow().weights;
                        for (0..LG_N) |k| acc[k] += gv * wr[LG_BASE + k];
                    }
                    for (0..15) |k| self.indirect_ns[k].lg_grad = g_final * acc[k];
                    self.indirect_r.lg_grad = g_final * acc[15];
                }
                // -Dleafgrad-fxcm: the same Gamma, for the 560 fxcm slots, with
                // dv/draw folded in so cm3 receives dL/draw directly. `active`
                // bounds it to the slots that really became mixer inputs; the
                // rest were fed fxcm_neutral and own no cell. fxcm.slots still
                // holds THIS bit's raws -- fxcm.perceive runs after this block.
                if (comptime LG_FX) {
                    var fg: [LG_FX_N]f32 = [_]f32{0} ** LG_FX_N;
                    for (0..23) |ii| {
                        const gv = gam[ii];
                        if (gv == 0) continue;
                        const wr = self.mixers0[ii].resolvedRow().weights;
                        for (0..LG_FX_N) |k| fg[k] += gv * wr[LG_FX_BASE + k];
                    }
                    const nact = @min(self.fxcm.active, LG_FX_N);
                    for (0..nact) |k| {
                        const raw: i32 = self.fxcm.slots[k];
                        self.fxcm.lg_slot_grad[k] = g_final * fg[k] *
                            self.fxcm_dvdraw[@intCast(raw + 2047)];
                    }
                    for (nact..LG_FX_N) |k| self.fxcm.lg_slot_grad[k] = 0;
                    if (comptime LG_DIAG) {
                        const d = &self.lg_diag;
                        for (0..nact) |k| {
                            if (self.fxcm.lg_reach[k] != 0) d.g_cm3_adapt += @abs(fg[k]);
                        }
                    }
                }
                if (comptime LG_DIAG) self.lgDiagBit(&gam, g_final, &acc);
            } else {
                if (comptime LG_ON) {
                    for (self.indirect_ns) |ind| ind.lg_grad = 0;
                    self.indirect_r.lg_grad = 0;
                }
                if (comptime LG_FX) self.fxcm.lg_slot_grad = [_]f32{0} ** LG_FX_N;
            }
        }

        // -Dextshed: bracket/direct/indirect model updates are shed with their
        // readouts (fully self-contained state — nothing else reads it). The
        // MATCHERS keep perceiving VERBATIM: Match.perceive advances
        // match_length (bit-level match tracking) and writes the position map
        // on byte end — the machinery that byteUpdate turns into
        // mgr.longest_match, mixer #8's gating context. Retaining the whole
        // stock perceive (including its predictions[]/counts[] update, which
        // nothing reads once predictis shed) keeps the gating key bit-exact
        // by construction rather than by argument.
        if (comptime !EXTSHED) {
            self.bracket.perceive(bit);
            self.direct.perceive(bit);
        }
        for (self.match_models) |mm| mm.perceive(bit);
        if (comptime !EXTSHED) {
            for (self.indirect_ns) |ind| ind.perceive(bit);
            self.indirect_r.perceive(bit);
        }
        self.ppmd.perceive(bit);
        self.byte_mixer.perceive(bit);
        if (comptime shadow_mix.enabled) {
            self.shadow.perceive(self.mi0.inputs, bit, self.shadow_pre, self.shadow_post, self.shadow_free);
        }
        // -Dadaptive-depth: on skipped bits the cascade (23 layer-0 mixers),
        // mixer1 and the SSE learned tables are FROZEN — their mixnever ran
        // this bit (stale staging) and skipping their update is the wall
        // mechanism itself. SSE's stream-parse counters still advance
        // (sse.advanceParse) or its context selection desyncs from the byte
        // stream for every later full-path bit (preflight §1 step 6).
        // Comptime-dead when the knob is off (do_full folds to true).
        const do_full: bool = if (comptime AD_ON) !self.adepth.skip else true;
        if (do_full) {
            if (comptime mixer_options.mixer_e2e) {
                // EXACT end-to-end gradient. The stack is affine in v, so
                //   gamma_i = dz/dh_i = W1[i] + sum_{k>i} gamma_k * u_k[i]   (gated by the clamp)
                // and dL/dw_i = (sigma(z) - y) * gamma_i * v. 253 MAC/bit.
                const g_final: f32 = self.headLogistic() - @as(f32, @floatFromInt(bit));
                var w1_scratch: [rls_mod.D]f32 = undefined;
                const w1w = self.headRow(&w1_scratch);
                const lo: f32 = self.mi0.stretched_min;
                const hi: f32 = self.mi0.stretched_max;
                var gamma: [23]f32 = undefined;
                var i: usize = 23;
                while (i > 0) {
                    i -= 1;
                    var s: f32 = w1w[i];
                    var k: usize = i + 1;
                    while (k < 23) : (k += 1) {
                        s += gamma[k] * self.mixers0[k].resolvedRow().extra_weights[i];
                    }
                    const h = self.mixers0[i].p;
                    gamma[i] = if (h > hi or h < lo) 0.0 else s;
                }
                var scale: f32 = 1.0;
                if (comptime mixer_options.mixer_e2e_norm) {
                    var ss: f32 = 0;
                    for (gamma) |gv| ss += gv * gv;
                    // P0 `1.0 / @sqrt(x)` in an `.optimized` scope was
                    // fused by LLVM into VRSQRTSS (a vendor-defined estimate), and
                    // `scale` multiplies the training error of ALL 23 layer-0
                    // mixers. See src/strict_fp.zig.
                    const rms: f32 = strict_fp.sqrt(ss / 23.0);
                    scale = if (rms > 1e-20) strict_fp.div(@as(f32, 1.0), rms) else 0.0;
                }
                for (&self.mixers0, 0..) |*mx, j| mx.perceiveErr(g_final * gamma[j] * scale);
            } else {
                for (&self.mixers0) |*mx| mx.perceive(bit);
            }
            self.mixer1.perceive(bit);
            // -Dmixer1-rls: INSIDE `do_full`, so an -Dadaptive-depth skip freezes
            // the RLS recursion in lockstep with mixer1 and the SSE. On a skipped
            // bit `rls.mix` never ran, so `rls.x` is stale and updating from it
            // would train on a prediction that was never coded — and would desync
            // the decoder, which skips the same bits.
            if (comptime RLS_ON) self.rls.update(bit, @floatCast(self.rls_p));
            self.sse.perceive(bit);
        } else {
            self.sse.advanceParse(bit);
        }

        const byte_update = self.mgr.bit_context >= 128;

        // UpdateContexts BEFORE fxcm.perceive: mx19cxt sees the previous bit's wrtcxt.
        const wrtcxt = self.fxcm.wrtcxt_val();
        self.mgr.updateContexts(@intCast(bit), wrtcxt);

        if (byte_update) {
            if (comptime !EXTSHED) {
                self.bracket.byteUpdate();
                self.direct.byteUpdate();
            }
            for (self.match_models) |mm| mm.byteUpdate(); // writes mgr.longest_match — RETAINED under -Dextshed (gating)
            if (comptime !EXTSHED) {
                for (self.indirect_ns) |ind| ind.byteUpdate();
                self.indirect_r.byteUpdate();
            }
            self.ppmd.byteUpdate();
            const bp = self.ppmd.bytePredict();
            for (0..256) |j| self.byte_mixer.setInput(j, bp[j]);
            self.byte_mixer.byteUpdate();
        }
        self.byte_mixer_output = self.byte_mixer.predict()[0];
        self.fxcm.perceive(bit);
        if (byte_update) self.mgr.bit_context = 1;
    }

    // ---- Pretrain (predictor.cpp:339-392) — the `-c` dictionary warm-up drive. ----
    // A STRIPPED Perceive that warms ONLY bracket + fxcm + direct[] + match[] +
    // indirect_ns[] + indirect_r[]. It does NOT touch ppmd, byte_mixer/lstm, the
    // layer-0/1 mixers, or the sse (predictor.cpp Pretrain omits them entirely).
    //
    // ORDERING GOTCHA (predictor.cpp:357 vs 374, the OPPOSITE of `perceive` above):
    // in Pretrain `fxcm.perceive(bit)` runs in the Perceive pass BEFORE
    // `mgr.updateContexts`, so the wrtcxt captured after it is the CURRENT bit's
    // value (in normal Perceive fxcm perceives LAST, so updateContexts sees the
    // PREVIOUS bit's wrtcxt). This is a deliberate divergence, not a bug.
    pub fn pretrain(self: *PredictorLex, bit: i32) void {
        @setFloatMode(.optimized);
        // Predict pass (predictor.cpp:340-354): advances the Indirect map indices
        // (predict `+= bit_context`, perceive `-= bit_context`) and the other models'
        // per-bit predict state. fxcm.Predictis a cached read (no state change),
        // so there is nothing to call for fxcm here.
        // -Dextshed: the shed models' pretrain warms state that is never read
        // (bracket/direct/indirect) or a pure readout (Match.predict writes only
        // its own outputs[] scalar) — skipped. The matchers' perceive stays so
        // the match maps + match_length carry the SAME warm state into coding
        // that stock carries ⇒ longest_match gating parity across the pretrain
        // boundary. Indirect.predict/perceive's paired map_index walk is
        // skipped as a pair, exactly as in the coding path.
        if (comptime !EXTSHED) {
            _ = self.bracket.predict();
            _ = self.direct.predict();
            for (self.match_models) |mm| _ = mm.predict();
            for (self.indirect_ns) |ind| _ = ind.predict();
            _ = self.indirect_r.predict();
        }

        // Perceive pass (predictor.cpp:356-370): fxcm right after bracket, BEFORE
        // updateContexts.
        if (comptime !EXTSHED) self.bracket.perceive(bit);
        self.fxcm.perceive(bit);
        if (comptime !EXTSHED) self.direct.perceive(bit);
        for (self.match_models) |mm| mm.perceive(bit);
        if (comptime !EXTSHED) {
            for (self.indirect_ns) |ind| ind.perceive(bit);
            self.indirect_r.perceive(bit);
        }

        const byte_update = self.mgr.bit_context >= 128;
        // fxcm ALREADY perceived this bit -> wrtcxt is CURRENT (predictor.cpp:357,374).
        const wrtcxt = self.fxcm.wrtcxt_val();
        self.mgr.updateContexts(@intCast(bit), wrtcxt);

        if (byte_update) {
            if (comptime !EXTSHED) {
                self.bracket.byteUpdate();
                self.direct.byteUpdate();
            }
            for (self.match_models) |mm| mm.byteUpdate(); // RETAINED under -Dextshed (longest_match gating)
            if (comptime !EXTSHED) {
                for (self.indirect_ns) |ind| ind.byteUpdate();
                self.indirect_r.byteUpdate();
            }
            self.mgr.bit_context = 1;
        }
    }

    // ---- -Dfree-prior (variant a/b): deterministic mid-stream LSTM self-retrain
    // Replays already-coded TEMP
    // prefix bytes through the LSTM training path ONLY — per byte the exact
    // ByteMixer.byteUpdate LSTM calls (setInput + perceive: forward, truncated
    // BPTT, Adam) with the same byte_map symbol mapping. No coder bits, no
    // CM/PPMd/mixer/SSE/manager state is touched, and ByteMixer.base.probs (the
    // live per-bit distribution) is bypassed, so coding resumes exactly where it
    // paused.
    //
    // Variant (a) (-Dfree-prior alone): aux = UNIFORM 2/V per vocab symbol. With
    // -Dlstm-prior-rho>0 the blend rho*(2/V)+(1-rho)*(2/V)=2/V cancels in the
    // softmax normalization ⇒ PRL neutralized during replay (DAMAGES the PRL
    // surface,+1,645).
    //
    // Variant (b) (+ -Dfree-prior-b): aux = the ACTUAL coding-time prior, read
    // back per position from the recorder's scratch file (fp_prior). This is the
    // post-×2-scale ByteMixer aux (== the exact vector PRL reads from
    // layer_input_[epoch][0]) that the recorder streamed during forward coding.
    // PRL stays ACTIVE and CONSISTENT with the shipped softmax. The prior stream
    // is regenerated identically by both ops (same bytes → same PPMd → same
    // prior) so it is not side data; replay symmetry is by construction, proven
    // by -Dfree-prior-hash matching between -c and -d.
    //
    // Callers (runner_lex coder loops) fire this at identical TEMP positions with
    // identical bytes on both ops. Unreferenced in stock builds (lazy analysis ⇒
    // zero cost, the -D pattern).
    pub fn freePriorReplayChunk(self: *PredictorLex, bytes: []const u8) void {
        const vs: usize = self.vocab_size;
        if (comptime build_options.free_prior_b) {
            // Variant (b): feed the recorded coding-time prior, one row per byte.
            // -Dlstm-aux-sparse: replay must feed the LSTM the same KIND of
            // input the coding path feeds it (byte_mixer.byteUpdate): the
            // recorded DENSE row goes through the SAME strict top-K selection
            // for the gathered aux, and PRL reads the dense row out-of-band via
            // setAuxPrior — so the fold sees exactly the recorded prior and the
            // aux keeps the coding path's sparse layout. Both ops replay the
            // same recorded stream, so symmetry is preserved by construction.
            var prior: [256]f32 = undefined;
            if (comptime build_options.lstm_aux_sparse > 0) {
                const K = build_options.lstm_aux_sparse;
                var ids: [K]u32 = undefined;
                var vals: [K]f32 = undefined;
                for (bytes) |b| {
                    self.fp_prior.readRow(prior[0..vs]);
                    const n = @import("mixer/byte_mixer.zig").selectTopK(K, prior[0..vs], &ids, &vals);
                    self.lstm.setSparseInput(ids[0..n], vals[0..n]);
                    self.lstm.setAuxPrior(prior[0..vs]);
                    _ = self.lstm.perceive(@intCast(self.byte_mixer.byte_map[b]));
                }
            } else {
                for (bytes) |b| {
                    self.fp_prior.readRow(prior[0..vs]);
                    self.lstm.setInput(prior[0..vs]);
                    _ = self.lstm.perceive(@intCast(self.byte_mixer.byte_map[b]));
                }
            }
        } else {
            // Variant (a): uniform prior.
            var uni: [256]f32 = undefined;
            @memset(uni[0..vs], 2.0 / @as(f32, @floatFromInt(self.vocab_size)));
            // -Dlstm-aux-sparse: route the uniform vector through the same
            // selection. Every symbol ties at 2/V, so the strict (mass DESC,
            // id ASC) order picks ids 0..K-1 and the scatter puts 2/V at exactly
            // those positions. PRL still reads the DENSE uniform vector
            // out-of-band, so the fold stays exactly neutral (rho*(2/V) +
            // (1-rho)*(2/V) = 2/V, constant across symbols, cancels in the
            // softmax normalization) and variant (a)'s premise — no PPMd
            // snapshot needed — is preserved.
            if (comptime build_options.lstm_aux_sparse > 0) {
                const K = build_options.lstm_aux_sparse;
                var ids: [K]u32 = undefined;
                var vals: [K]f32 = undefined;
                const n = @min(K, vs);
                for (0..n) |i| {
                    ids[i] = @intCast(i);
                    vals[i] = 2.0 / @as(f32, @floatFromInt(self.vocab_size));
                }
                for (bytes) |b| {
                    self.lstm.setSparseInput(ids[0..n], vals[0..n]);
                    self.lstm.setAuxPrior(uni[0..vs]);
                    _ = self.lstm.perceive(@intCast(self.byte_mixer.byte_map[b]));
                }
                return;
            }
            for (bytes) |b| {
                self.lstm.setInput(uni[0..vs]);
                _ = self.lstm.perceive(@intCast(self.byte_mixer.byte_map[b]));
            }
        }
    }

    /// -Dfree-prior-b: turn ON coding-time prior recording by installing the
    /// sink into the byte-mixer. Called once per coder loop by runner_lex ONLY
    /// under `comptime FP_ON` (a live schedule), so -Dfree-prior-b alone stays
    /// inert. `&self.fp_prior` is heap-stable.
    pub fn fpPriorEnableRecording(self: *PredictorLex) void {
        if (comptime build_options.free_prior_b) self.byte_mixer.fp_prior_sink = @ptrCast(&self.fp_prior);
    }

    /// -Dfree-prior-b: reset the prior-file read cursor to 0 (each epoch replays
    /// the prefix from the start, so priors must too). Flushes the write buffer
    /// first so the file holds every recorded row. Called once per epoch by the
    /// runner_lex fpEpoch* wrappers before the chunked replay. No-op / absent
    /// unless variant (b) is compiled in.
    pub fn fpPriorBeginReplay(self: *PredictorLex) void {
        if (comptime build_options.free_prior_b) self.fp_prior.beginReplay();
    }

    /// -Dfree-prior-b: turn OFF prior recording once no further scheduled epoch
    /// can fire (runner_lex gate ii). Rows past the last fireable epoch are
    /// never read, so recording them is pure waste — at e9 the ungated tail is
    /// ~550 MB of temp: a top-K pack per byte (~2-3% of pipeline wall) plus
    /// ~54 GB of scratch, the dominant record-side term of the fpb17 e8 wall
    /// kill. Uninstalls the sink and deletes the scratch file. The trigger is a
    /// pure function of the comptime schedule + stream length + position —
    /// identical on both ops, so replay symmetry is untouched; byte-neutrality
    /// vs the ungated build is verified (gated/ungated archives cmp-identical).
    /// FpPriorRec.deinit is idempotent (predictor deinit calls it again), and a
    /// later coder loop on the same predictor would simply re-open fresh.
    pub fn fpPriorEndRecording(self: *PredictorLex) void {
        if (comptime build_options.free_prior_b) {
            self.byte_mixer.fp_prior_sink = null;
            self.fp_prior.deinit();
        }
    }

    /// sha256 over the full dynamic LSTM state: gate weights + Adam moments +
    /// layer-norm gamma/beta (+ their moments) + cell states + update counters +
    /// output-layer slab + output_error_ sufficient-statistic ring (compact
    /// mode) + hidden/input-history rings. Gate-2 evidence: after a
    /// free-prior epoch the -c and -d hashes must match bit-for-bit (f32 memory
    /// images; both ops run the same binary on the same arch).
    pub fn freePriorLstmStateHash(self: *PredictorLex) [32]u8 {
        var h = std.crypto.hash.sha2.Sha256.init(.{});
        for (self.lstm.layers_) |*layer| {
            inline for (.{ &layer.forget_gate_, &layer.input_node_, &layer.output_gate_ }) |gate| {
                for (gate.weights_) |row| h.update(std.mem.sliceAsBytes(row));
                for (gate.m_) |row| h.update(std.mem.sliceAsBytes(row));
                for (gate.v_) |row| h.update(std.mem.sliceAsBytes(row));
                h.update(std.mem.sliceAsBytes(gate.gamma_));
                h.update(std.mem.sliceAsBytes(gate.gamma_m_));
                h.update(std.mem.sliceAsBytes(gate.gamma_v_));
                h.update(std.mem.sliceAsBytes(gate.beta_));
                h.update(std.mem.sliceAsBytes(gate.beta_m_));
                h.update(std.mem.sliceAsBytes(gate.beta_v_));
            }
            h.update(std.mem.sliceAsBytes(layer.state_));
            h.update(std.mem.asBytes(&layer.update_steps_));
            h.update(std.mem.asBytes(&layer.epoch_));
        }
        for (self.lstm.output_layer_) |epoch_rows| {
            for (epoch_rows) |row| h.update(std.mem.sliceAsBytes(row));
        }
        // -Dlstm-head-rank: the factorised head IS the output weights when the
        // knob is on (output_layer_ is then allocated but never read or written).
        // Empty slices at rank 0 => these loops are no-ops and the stock hash is
        // byte-for-byte unchanged.
        for (self.lstm.head_proj_) |epoch_rows| {
            for (epoch_rows) |row| h.update(std.mem.sliceAsBytes(row));
        }
        for (self.lstm.head_out_) |epoch_rows| {
            for (epoch_rows) |row| h.update(std.mem.sliceAsBytes(row));
        }
        h.update(std.mem.sliceAsBytes(self.lstm.hp_));
        // Compact history (-Dlstm-compact-hist): output_error_ is the BPTT
        // sufficient-statistic ring that REPLACES the horizon-deep weight
        // ring as future-affecting state — without it two states differing
        // only there would falsely hash as matching. Empty (len 0) in stock
        // mode, where this loop is a no-op and the hash is unchanged.
        for (self.lstm.output_error_) |row| h.update(std.mem.sliceAsBytes(row));
        h.update(std.mem.sliceAsBytes(self.lstm.hidden_));
        h.update(std.mem.sliceAsBytes(self.lstm.input_history_));
        h.update(std.mem.asBytes(&self.lstm.epoch_));
        var out: [32]u8 = undefined;
        h.final(&out);
        return out;
    }
};

// ===========================================================================
// Verification harness (gate vs the real-cmix-lex oracle in lex-goldens/)
// ===========================================================================
// CWD-relative. The oracle goldens are derived verification data, not a build
// input, and are not distributed with the source package; regenerate them from
// the cmix-lex reference before running these tests.
const GOLDEN_DIR = "reference/lex-goldens";

fn loadVocab(path: []const u8) ![256]bool {
    var buf: [32]u8 = undefined;
    const f = try std.fs.cwd().openFile(path, .{});
    defer f.close();
    _ = try f.readAll(&buf);
    var vocab: [256]bool = .{false} ** 256;
    for (0..32) |i| {
        for (0..8) |j| {
            if ((buf[i] >> @intCast(j)) & 1 != 0) vocab[i * 8 + j] = true;
        }
    }
    return vocab;
}

// -------- deinit completeness gate: every allocator-tracked byte returns ------
// std.testing.allocator FAILS the test on any leak, so this is the oracle that
// PredictorLex.deinit tears down the full model graph (the -e/-D RSS-gate fix).
// A few hundred bits are run first so runtime-lazy allocations (MixerLex context
// rows) exist at teardown. zero_alloc big tables bypass the tracking allocator
// (page_allocator mmaps) but are freed by the same symmetric calls.
test "predictor_lex init+run+deinit is leak-free" {
    const a = std.testing.allocator;
    const vocab: [256]bool = .{true} ** 256;
    const pred = try PredictorLex.initSeed(a, vocab, 923);
    const sample = "The quick brown fox jumps over the lazy dog. <page> </page>\n";
    for (sample) |byte| {
        var j: i32 = 7;
        while (j >= 0) : (j -= 1) {
            const bit: i32 = @intCast((byte >> @intCast(j)) & 1);
            _ = pred.predict();
            pred.perceive(bit);
        }
    }
    pred.deinit();
}

// -------- INPUTS gate (localization): first N bits, per-index diff --------
test "predictor_lex INPUTS gate vs oracle inputs590" {
    // These gates compare against goldens captured from the cmix-lex ORACLE,
    // whose byte-mixer slot is the online LSTM. Under -Dtransformer the frozen
    // transformer occupies that slot by design, so the recorded values cannot
    // hold and a failure here would be the knob working, not a regression.
    // Skipped rather than left red: a suite that is "known to fail" trains
    // people to ignore it, and then it cannot report a real break.
    if (comptime tfmod.enabled) return error.SkipZigTest;
    var arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer arena.deinit();
    const a = arena.allocator();

    const stream = try std.fs.cwd().readFileAlloc(a, GOLDEN_DIR ++ "/stream_64k.bin", 1 << 20);
    const vocab = try loadVocab(GOLDEN_DIR ++ "/vocab_nodict.bin");

    const pred = try PredictorLex.init(a, vocab);
    // Arena frees the RAM either way, but deinit also tears down the non-arena
    // resources (zero_alloc page_allocator tables; -Dppmd-mmap: munmap +
    // delete ppm.temp) so a flag-ON test run leaves no stale mapping/file.
    defer pred.deinit();

    const inf = try std.fs.cwd().openFile(GOLDEN_DIR ++ "/inputs590_nodict.bin", .{});
    defer inf.close();
    var reader_buf: [590 * 4]u8 = undefined;

    const N_BITS: usize = 40000; // 5000 bytes
    var max_abs = [_]f32{0} ** 590;
    var exceed_cnt = [_]u32{0} ** 590; // bits with |Δ| > 1e-2 per index
    var first_bad_idx: ?usize = null;
    var first_bad_bit: usize = 0;
    var first_bad_want: f32 = 0;
    var first_bad_got: f32 = 0;

    var bit_no: usize = 0;
    outer: for (stream) |byte| {
        var j: i32 = 7;
        while (j >= 0) : (j -= 1) {
            if (bit_no >= N_BITS) break :outer;
            const bit: i32 = @intCast((byte >> @intCast(j)) & 1);
            _ = pred.predict();

            // read the golden record for this bit
            _ = try inf.readAll(&reader_buf);
            for (0..@min(@as(usize, 590), NIN)) |k| {
                const want: f32 = @bitCast(std.mem.readInt(u32, reader_buf[k * 4 ..][0..4], .little));
                const got: f32 = pred.mi0.inputs[k];
                const d = @abs(want - got);
                if (d > max_abs[k]) max_abs[k] = d;
                if (d > 1.0e-2) {
                    exceed_cnt[k] += 1;
                    if (first_bad_idx == null and k <= 561) {
                        // localize wiring bugs on the RNG-independent indices
                        first_bad_idx = k;
                        first_bad_bit = bit_no;
                        first_bad_want = want;
                        first_bad_got = got;
                    }
                }
            }
            pred.perceive(bit);
            bit_no += 1;
        }
    }

    // ---- report ----
    std.debug.print("\n=== INPUTS GATE (first {d} bits) ===\n", .{bit_no});
    const groups = [_]struct { name: []const u8, lo: usize, hi: usize }{
        .{ .name = "bracket[0]", .lo = 0, .hi = 1 },
        .{ .name = "fxcm[1..560]", .lo = 1, .hi = 561 },
        .{ .name = "direct[561]", .lo = 561, .hi = 562 },
        .{ .name = "match[562..571]", .lo = 562, .hi = 572 },
        .{ .name = "ind_ns[572..586]", .lo = 572, .hi = 587 },
        .{ .name = "ind_r[587]", .lo = 587, .hi = 588 },
        .{ .name = "ppmd[588]", .lo = 588, .hi = 589 },
        .{ .name = "byte_mixer[589]", .lo = 589, .hi = 590 },
    };
    for (groups) |g| {
        var gmax: f32 = 0;
        var worst: usize = g.lo;
        var gexc: u32 = 0;
        for (g.lo..g.hi) |k| {
            if (max_abs[k] > gmax) {
                gmax = max_abs[k];
                worst = k;
            }
            gexc += exceed_cnt[k];
        }
        std.debug.print("  {s:<18} max|Δ|={d:.6} (idx {d})  bits>1e-2={d}\n", .{ g.name, gmax, worst, gexc });
    }
    if (first_bad_idx) |bi| {
        std.debug.print("  FIRST wiring divergence (idx<=561): idx {d} @bit {d} want={d:.5} got={d:.5}\n", .{ bi, first_bad_bit, first_bad_want, first_bad_got });
    } else {
        std.debug.print("  no divergence >1e-2 on RNG-independent indices [0..561]\n", .{});
    }

    // RNG-independent wiring must match tightly.
    try std.testing.expect(first_bad_idx == null);
}

// -------- FINAL-PROB gate (primary): full disc + checksum + XENT --------
test "predictor_lex FINAL-PROB gate vs oracle disc" {
    // These gates compare against goldens captured from the cmix-lex ORACLE,
    // whose byte-mixer slot is the online LSTM. Under -Dtransformer the frozen
    // transformer occupies that slot by design, so the recorded values cannot
    // hold and a failure here would be the knob working, not a regression.
    // Skipped rather than left red: a suite that is "known to fail" trains
    // people to ignore it, and then it cannot report a real break.
    if (comptime tfmod.enabled) return error.SkipZigTest;
    var arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer arena.deinit();
    const a = arena.allocator();

    const stream = try std.fs.cwd().readFileAlloc(a, GOLDEN_DIR ++ "/stream_64k.bin", 1 << 20);
    const vocab = try loadVocab(GOLDEN_DIR ++ "/vocab_nodict.bin");
    const disc = try std.fs.cwd().readFileAlloc(a, GOLDEN_DIR ++ "/disc_nodict.u16", 4 << 20);

    const pred = try PredictorLex.init(a, vocab);
    defer pred.deinit(); // see INPUTS gate note: releases ppm.temp under -Dppmd-mmap
    const total_bits: usize = stream.len * 8;
    var cs: u64 = 0;
    var within: usize = 0;
    var max_dev: u32 = 0;
    var xent: f64 = 0;

    var bit_no: usize = 0;
    for (stream) |byte| {
        var j: i32 = 7;
        while (j >= 0) : (j -= 1) {
            const bit: i32 = @intCast((byte >> @intCast(j)) & 1);
            const p = pred.predict();
            var d: u32 = 1 + @as(u32, @intFromFloat(65534.0 * p));
            if (d < 1) d = 1;
            if (d > 65535) d = 65535;
            cs = cs *% 1000003 +% @as(u64, d);

            const d_cpp: u32 = std.mem.readInt(u16, disc[bit_no * 2 ..][0..2], .little);
            const dev: u32 = if (d > d_cpp) d - d_cpp else d_cpp - d;
            if (dev <= 512) within += 1;
            if (dev > max_dev) max_dev = dev;

            var pe: f64 = @as(f64, @floatFromInt(d)) / 65536.0;
            if (pe < 1.0 / 65536.0) pe = 1.0 / 65536.0;
            if (pe > 65535.0 / 65536.0) pe = 65535.0 / 65536.0;
            const pc = if (bit != 0) pe else 1.0 - pe;
            xent += -std.math.log2(pc);

            pred.perceive(bit);
            bit_no += 1;
        }
    }

    const pass_rate = @as(f64, @floatFromInt(within)) / @as(f64, @floatFromInt(total_bits));
    const bpc = xent / @as(f64, @floatFromInt(stream.len));
    std.debug.print("\n=== FINAL-PROB GATE ({d} bits) ===\n", .{total_bits});
    std.debug.print("  within +/-512 : {d}/{d} = {d:.5}%\n", .{ within, total_bits, pass_rate * 100.0 });
    std.debug.print("  max deviation : {d} ADU\n", .{max_dev});
    std.debug.print("  disc checksum : {d}   (oracle 15432318940569149700)\n", .{cs});
    std.debug.print("  XENT bpc      : {d:.6}   (oracle 2.071565)\n", .{bpc});

    try std.testing.expect(pass_rate >= 0.999);
}
