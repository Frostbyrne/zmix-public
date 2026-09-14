//! LSTM layer, ported from cmix `mixer/lstm-layer.{h,cpp}`.
//!
//! A single LSTM layer with layer normalization on each gate and an Adam
//! optimizer, trained by truncated BPTT over `horizon` steps. Faithful port of
//! the reference valarray math using explicit f32 loops.
const std = @import("std");
const strict_fp = @import("strict_fp");
const ram_census = @import("ramcensus");
const Sigmoid = @import("lf_sigmoid.zig").Sigmoid;

// Numerics mode for the hot tanh (comptime three-way):
//   -Dvendored-libm  -> vendored_math.tanhf: build-time-fixed ARM
//     optimized-routines port (expm1f-based, <=2 ulp vs glibc 2.43) — the
//     judged run cannot drift when the judge's glibc changes.
//   -Dglibc (extern) -> glibc tanhf resolved on the RUNNING machine: faster
//     than Zig's portable tanh and matches what the record binary calls.
//   default (static) -> Zig's deterministic std.math.tanh.
// Comptime-dead branches never reference the extern, so no undefined symbol.
const use_vendored_libm = @import("build_options").vendored_libm;
// Avenue #1: when true, the backward/forward element-wise loops use fast-math
// @Vector maps (NOT bit-identical to scalar). Set by -Dlstm-bwvec; for fresh-archive
// (self-consistent encode+decode) builds only. Default false = bit-identical ship.
const lstm_bwvec = @import("build_options").lstm_bwvec;
// -Dlstm-aux-sparse=K: THE WALL LEVER. The auxiliary input keeps its shipped
// width and its shipped index-by-symbol layout, but only K of its ~vocab_size
// entries are non-zero (byte_mixer.selectTopK scatters them). Every gate row is
//     [ output_size one-hot cols | AUX (input_size_) | recurrent | bias ]
// and forwardNeuron/backwardNeuron walk the AUX span densely — the two biggest
// LSTM hot spots. Under this knob they walk a K-long GATHER over the live
// symbol indices instead, and the recurrent+bias TAIL keeps its dense
// vectorised pass. 0 = stock (every branch below comptime-dead; identical
// codegen). See dotGather/axpyScatter for the exactness argument.
const aux_sparse: usize = @import("build_options").lstm_aux_sparse;
// -Dlstm-aux-sparse-dense: falsification control — keep the truncation, walk
// the aux span densely anyway (no wall, isolates truncation from the gathered
// dot's reassociation). `gather` is the single predicate both hot loops read.
const aux_sparse_dense: bool = @import("build_options").lstm_aux_sparse_dense;
const gather: bool = aux_sparse > 0 and !aux_sparse_dense;
const use_glibc_libm = @import("builtin").target.abi.isGnu();
const vendored_math = @import("lf_vendored_math.zig");
extern fn tanhf(x: f32) f32;
inline fn tanhF(x: f32) f32 {
    return if (comptime use_vendored_libm)
        vendored_math.tanhf(x)
    else if (comptime use_glibc_libm)
        tanhf(x)
    else
        std.math.tanh(x);
}

// AVX2-width SIMD kernels for the LSTM hot loops. @Vector(8,f32) with per-lane ops +
// @mulAdd FMA contraction (single rounding — the same vfmadd codegen the record's
// `-ffp-model=fast` build emits; NOT an approximation). Only the summation ORDER of
// the dot-products changes (8 lanes + tree reduce). Full fast-math was rejected: it
// also approximated the layer-norm 1/sqrt and degraded the model (+28% on 50k).
const VL = 8;
const Vf = @Vector(VL, f32);

/// dot(a, b) over a.len elements — 4×8-lane FMA accumulators + horizontal reduce.
/// Four independent accumulators break the loop-carried FMA latency chain (~4-5cy),
/// the same unroll clang's fast-math autovectorizer gives the record build. Changes
/// summation order only (in-class; converges at scale — iter 289).
inline fn dotV(a: []const f32, b: []const f32) f32 {
    var acc0: Vf = @splat(0.0);
    var acc1: Vf = @splat(0.0);
    var acc2: Vf = @splat(0.0);
    var acc3: Vf = @splat(0.0);
    var j: usize = 0;
    while (j + 4 * VL <= a.len) : (j += 4 * VL) {
        acc0 = @mulAdd(Vf, @as(Vf, a[j..][0..VL].*), @as(Vf, b[j..][0..VL].*), acc0);
        acc1 = @mulAdd(Vf, @as(Vf, a[j + VL ..][0..VL].*), @as(Vf, b[j + VL ..][0..VL].*), acc1);
        acc2 = @mulAdd(Vf, @as(Vf, a[j + 2 * VL ..][0..VL].*), @as(Vf, b[j + 2 * VL ..][0..VL].*), acc2);
        acc3 = @mulAdd(Vf, @as(Vf, a[j + 3 * VL ..][0..VL].*), @as(Vf, b[j + 3 * VL ..][0..VL].*), acc3);
    }
    while (j + VL <= a.len) : (j += VL) {
        acc0 = @mulAdd(Vf, @as(Vf, a[j..][0..VL].*), @as(Vf, b[j..][0..VL].*), acc0);
    }
    var s: f32 = @reduce(.Add, (acc0 + acc2) + (acc1 + acc3));
    while (j < a.len) : (j += 1) s = @mulAdd(f32, a[j], b[j], s);
    return s;
}

/// -Dlstm-aux-sparse: sum over the K LIVE entries only — `s = SUM_t vals[t] *
/// w[ids[t]]`, four independent accumulators reduced in dotV's own tree shape
/// `(a0+a2)+(a1+a3)`.
///
/// EXACTNESS. Against a dense pass over the same sparsified vector this changes
/// the summation ORDER (the zeros are skipped instead of accumulated, and K
/// scattered terms land in different accumulators than they would at their own
/// lane positions), so it is not bit-identical to that reference. It is
/// deterministic, which is the property that matters: both ops run this code
/// over a bit-identically computed (ids, vals) — the selection is a strict
/// total order over `inputs`, which every existing roundtrip already proves
/// symmetric — and the accumulator assignment is a function of the SLOT index
/// t alone, not of the value or the address.
///
/// The four accumulators are not decoration: a single chain of K dependent
/// FMAs would cost K x ~4 cycles of latency per row, and this runs
/// num_cells x 3 gates times per byte. They mirror dotV so the two halves of
/// the split dot have the same numeric character.
inline fn dotGather(vals: []const f32, ids: []const u32, w: []const f32) f32 {
    var a0: f32 = 0;
    var a1: f32 = 0;
    var a2: f32 = 0;
    var a3: f32 = 0;
    var t: usize = 0;
    while (t + 4 <= vals.len) : (t += 4) {
        a0 = @mulAdd(f32, vals[t], w[ids[t]], a0);
        a1 = @mulAdd(f32, vals[t + 1], w[ids[t + 1]], a1);
        a2 = @mulAdd(f32, vals[t + 2], w[ids[t + 2]], a2);
        a3 = @mulAdd(f32, vals[t + 3], w[ids[t + 3]], a3);
    }
    while (t < vals.len) : (t += 1) a0 = @mulAdd(f32, vals[t], w[ids[t]], a0);
    return (a0 + a2) + (a1 + a3);
}

/// -Dlstm-aux-sparse: `y[ids[t]] += x * vals[t]` for the K live entries.
///
/// ★ Unlike the gathered dot this IS bit-identical to the dense axpyV over the
/// sparsified vector: axpyV's update is ELEMENT-WISE (`y[j] = fma(x, a[j],
/// y[j])`, one independent rounding per element, vector and scalar paths alike),
/// so skipping an entry whose `a[j]` is 0 skips `fma(x, 0, y[j]) == y[j]`. Only
/// the sign of an exact zero can differ, which no downstream operation can
/// observe. The operand order below is axpyV's, deliberately.
inline fn axpyScatter(y: []f32, x: f32, vals: []const f32, ids: []const u32) void {
    for (vals, ids) |v, ix| y[ix] = @mulAdd(f32, x, v, y[ix]);
}

/// y[j] += x * a[j] for all j — no reordering; FMA-contracted (single rounding,
/// same contraction the record's -ffp-model=fast build emits).
inline fn axpyV(y: []f32, x: f32, a: []const f32) void {
    const xv: Vf = @splat(x);
    var j: usize = 0;
    while (j + VL <= a.len) : (j += VL) {
        var yv: Vf = y[j..][0..VL].*;
        yv = @mulAdd(Vf, xv, @as(Vf, a[j..][0..VL].*), yv);
        y[j..][0..VL].* = yv;
    }
    while (j < a.len) : (j += 1) y[j] = @mulAdd(f32, x, a[j], y[j]);
}


/// norm[j] *= iv ; state[j] = fma(norm[j], gamma[j], beta[j]) — bit-exact @Vector
/// map (per-lane FMA single rounding == the scalar codegen under .optimized).
inline fn normScaleV(norm: []f32, iv: f32, gamma: []const f32, beta: []const f32, state: []f32) void {
    @setFloatMode(.optimized);
    const ivv: Vf = @splat(iv);
    var j: usize = 0;
    while (j + VL <= norm.len) : (j += VL) {
        const nv: Vf = @as(Vf, norm[j..][0..VL].*) * ivv;
        norm[j..][0..VL].* = nv;
        // Match the scalar source expression `norm*gamma + beta` exactly (let
        // @setFloatMode(.optimized) make the same fuse choice per lane it makes
        // per scalar) — bit-identity is verified by sha256, not assumed.
        state[j..][0..VL].* = nv * @as(Vf, gamma[j..][0..VL].*) + @as(Vf, beta[j..][0..VL].*);
    }
    while (j < norm.len) : (j += 1) {
        norm[j] *= iv;
        state[j] = norm[j] * gamma[j] + beta[j];
    }
}

// ---- Avenue #1: fast-math @Vector maps for the LSTM backward element-wise loops.
// These are NOT bit-identical to the scalar codegen (per-lane reassociation under
// @setFloatMode(.optimized) — the same fast-math freedom that diverges RF at byte
// 407). They are used ONLY when a fresh self-consistent archive is minted (encode
// and decode share this binary → roundtrip is exact regardless of rounding). Guard:
// ZMIX_LSTM_BWVEC comptime (build_options.lstm_bwvec) so the default ship stays
// bit-identical and the change is measured behind a flag. ----

/// beta_u[j] += err[j] ; gamma_u[j] += err[j]*norm[j]  (fused accumulate)
inline fn betaGammaAccV(beta_u: []f32, gamma_u: []f32, err: []const f32, norm: []const f32) void {
    @setFloatMode(.optimized);
    var j: usize = 0;
    while (j + VL <= err.len) : (j += VL) {
        const e: Vf = err[j..][0..VL].*;
        beta_u[j..][0..VL].* = @as(Vf, beta_u[j..][0..VL].*) + e;
        gamma_u[j..][0..VL].* = @mulAdd(Vf, e, @as(Vf, norm[j..][0..VL].*), @as(Vf, gamma_u[j..][0..VL].*));
    }
    while (j < err.len) : (j += 1) {
        beta_u[j] += err[j];
        gamma_u[j] += err[j] * norm[j];
    }
}

/// err[j] *= gamma[j] * ivar
inline fn errScaleV(err: []f32, gamma: []const f32, ivar: f32) void {
    @setFloatMode(.optimized);
    const iv: Vf = @splat(ivar);
    var j: usize = 0;
    while (j + VL <= err.len) : (j += VL) {
        err[j..][0..VL].* = @as(Vf, err[j..][0..VL].*) * @as(Vf, gamma[j..][0..VL].*) * iv;
    }
    while (j < err.len) : (j += 1) err[j] *= gamma[j] * ivar;
}

/// err[j] -= mean * norm[j]
inline fn errSubMeanV(err: []f32, mean: f32, norm: []const f32) void {
    @setFloatMode(.optimized);
    const mv: Vf = @splat(mean);
    var j: usize = 0;
    while (j + VL <= err.len) : (j += VL) {
        err[j..][0..VL].* = @as(Vf, err[j..][0..VL].*) - mv * @as(Vf, norm[j..][0..VL].*);
    }
    while (j < err.len) : (j += 1) err[j] -= mean * norm[j];
}

/// dst[j] += src[j]
inline fn addAccV(dst: []f32, src: []const f32) void {
    @setFloatMode(.optimized);
    var j: usize = 0;
    while (j + VL <= dst.len) : (j += VL) {
        dst[j..][0..VL].* = @as(Vf, dst[j..][0..VL].*) + @as(Vf, src[j..][0..VL].*);
    }
    while (j < dst.len) : (j += 1) dst[j] += src[j];
}

/// dst[j] *= src[j]
inline fn mulInPlaceV(dst: []f32, src: []const f32) void {
    @setFloatMode(.optimized);
    var j: usize = 0;
    while (j + VL <= dst.len) : (j += VL) {
        dst[j..][0..VL].* = @as(Vf, dst[j..][0..VL].*) * @as(Vf, src[j..][0..VL].*);
    }
    while (j < dst.len) : (j += 1) dst[j] *= src[j];
}

/// The LSTM backward gate-gradient map (backwardPass inner loop) as a fast-math
/// @Vector sweep. og=output_gate.state, ts=tanh_state, in=input_node.state,
/// igs=input_gate_state, ls=last_state, fgs=forget_gate.state (all [epoch], len nc).
/// Writes og_err/in_err/fg_err and accumulates state_error in place.
inline fn gateGradV(
    og_err: []f32,
    in_err: []f32,
    fg_err: []f32,
    state_error: []f32,
    stored_error: []const f32,
    og: []const f32,
    ts: []const f32,
    in: []const f32,
    igs: []const f32,
    ls: []const f32,
    fgs: []const f32,
    nc: usize,
) void {
    @setFloatMode(.optimized);
    const one: Vf = @splat(1.0);
    var i: usize = 0;
    while (i + VL <= nc) : (i += VL) {
        const ogv: Vf = og[i..][0..VL].*;
        const tsv: Vf = ts[i..][0..VL].*;
        const st: Vf = stored_error[i..][0..VL].*;
        og_err[i..][0..VL].* = tsv * st * ogv * (one - ogv);
        var se: Vf = state_error[i..][0..VL].*;
        se = se + st * ogv * (one - tsv * tsv);
        const inv: Vf = in[i..][0..VL].*;
        const igsv: Vf = igs[i..][0..VL].*;
        in_err[i..][0..VL].* = se * igsv * (one - inv * inv);
        const lsv: Vf = ls[i..][0..VL].*;
        const fgsv: Vf = fgs[i..][0..VL].*;
        fg_err[i..][0..VL].* = (lsv - inv) * se * fgsv * igsv;
        state_error[i..][0..VL].* = se;
    }
    while (i < nc) : (i += 1) {
        const ogs = og[i];
        const tss = ts[i];
        og_err[i] = tss * stored_error[i] * ogs * (1.0 - ogs);
        state_error[i] += stored_error[i] * ogs * (1.0 - tss * tss);
        const ins = in[i];
        in_err[i] = state_error[i] * igs[i] * (1.0 - ins * ins);
        fg_err[i] = (ls[i] - ins) * state_error[i] * fgs[i] * igs[i];
    }
}

/// Inline small-row copy/zero: these rows are ~170-680 bytes, where glibc's
/// memcpy PLT call + ifunc dispatch costs as much as the copy (measured 9.6%
/// of runtime). clang inlines the valarray twins; do the same.
inline fn copyV(dst: []f32, src: []const f32) void {
    var j: usize = 0;
    while (j + VL <= src.len) : (j += VL) {
        dst[j..][0..VL].* = @as(Vf, src[j..][0..VL].*);
    }
    while (j < src.len) : (j += 1) dst[j] = src[j];
}
inline fn zeroV(dst: []f32) void {
    const z: Vf = @splat(0.0);
    var j: usize = 0;
    while (j + VL <= dst.len) : (j += VL) dst[j..][0..VL].* = z;
    while (j < dst.len) : (j += 1) dst[j] = 0;
}

inline fn auxSparseK() usize { return if (comptime gather) aux_sparse else 1; }
fn alloc1(a: std.mem.Allocator, n: usize) []f32 {
    ram_census.note(n * @sizeOf(f32), @returnAddress());
    const s = a.alloc(f32, n) catch unreachable;
    @memset(s, 0);
    return s;
}
fn alloc2(a: std.mem.Allocator, rows: usize, cols: usize) [][]f32 {
    // One contiguous slab with a row-slice table into it (NOT per-row allocations):
    // rows that are iterated together stay dense in cache, and the hardware
    // prefetcher sees a linear stream. Values/iteration order are unchanged —
    // bit-identical; this is purely a locality fix (jagged rows were the largest
    // source of the measured 1.6x L1d-miss excess vs cmix-lex's valarrays).
    const slab = alloc1(a, rows * cols);
    const r = a.alloc([]f32, rows) catch unreachable;
    for (r, 0..) |*x, i| x.* = slab[i * cols .. (i + 1) * cols];
    return r;
}

/// Free an alloc2 table: reconstruct the slab slice (rows are equal-length views
/// into one contiguous allocation starting at row 0) and free slab + row table.
pub fn free2(a: std.mem.Allocator, r: [][]f32) void {
    if (r.len != 0 and r[0].len != 0) a.free(r[0].ptr[0 .. r.len * r[0].len]);
    a.free(r);
}

// Memoized Adam bias-correction terms: te is capped at update_limit, so after the
// first update_limit steps these are literally constant, yet adamis called per
// weight-row (hundreds of times per byte). Same inputs -> same pow results, so the
// memo is output-neutral. Single-threaded (like the rest of the predictor).
var adam_te_memo: f64 = -1.0;
var adam_b1t: f32 = undefined;
var adam_b2t: f32 = undefined;

fn adam(g: []f32, m: []f32, v: []f32, w: []f32, learning_rate: f32, t: f64, update_limit: u64) void {
    @setFloatMode(.optimized); // record parity: clang -ffp-model=fast semantics (FMA, reassoc, rcp/rsqrt approx)
    const beta1: f32 = @import("build_options").lstm_adam_beta1; // -Dlstm-adam-beta1 (C5)
    const beta2: f32 = 0.9999;
    const eps: f32 = 1e-6;
    const ul: f64 = @floatFromInt(update_limit);
    const te = if (t < ul) t else ul;
    const alpha: f32 = @floatCast(@as(f64, learning_rate) * 0.1 / std.math.sqrt(5e-5 * te + 1.0));
    if (te != adam_te_memo) {
        adam_te_memo = te;
        adam_b1t = @floatCast(1.0 - std.math.pow(f64, beta1, te));
        adam_b2t = @floatCast(1.0 - std.math.pow(f64, beta2, te));
    }
    const b1t = adam_b1t;
    const b2t = adam_b2t;
    // 8-lane vectorization of the weight loop. Every op is element-wise IEEE
    // (mul/add/div/sqrt, correctly rounded per lane; deliberately NO @mulAdd here)
    // so lanes are bit-identical to the scalar loop below.
    const vb1: Vf = @splat(beta1);
    const vb1c: Vf = @splat(1.0 - beta1);
    const vb2: Vf = @splat(beta2);
    const vb2c: Vf = @splat(1.0 - beta2);
    const veps: Vf = @splat(eps);
    const valpha: Vf = @splat(alpha);
    const vb1t: Vf = @splat(b1t);
    const vb2t: Vf = @splat(b2t);
    var j: usize = 0;
    while (j + VL <= g.len) : (j += VL) {
        const gv: Vf = g[j..][0..VL].*;
        var mv: Vf = m[j..][0..VL].*;
        var vv: Vf = v[j..][0..VL].*;
        mv = mv * vb1 + vb1c * gv;
        vv = vv * vb2 + vb2c * (gv * gv);
        m[j..][0..VL].* = mv;
        v[j..][0..VL].* = vv;
        var wv: Vf = w[j..][0..VL].*;
        // P0 every division here goes through strict_fp. Under
        // `.optimized` LLVM lowered these three divides to VRCPPS/VRSQRTPS
        // *estimates* (vendor-defined tables) — see the header of src/strict_fp.zig.
        wv -= valpha * strict_fp.divSqrt(strict_fp.div(mv, vb1t), strict_fp.div(vv, vb2t) + veps);
        w[j..][0..VL].* = wv;
    }
    while (j < g.len) : (j += 1) {
        m[j] = m[j] * beta1 + (1.0 - beta1) * g[j];
        v[j] = v[j] * beta2 + (1.0 - beta2) * g[j] * g[j];
        w[j] -= alpha * strict_fp.divSqrt(strict_fp.div(m[j], b1t), strict_fp.div(v[j], b2t) + eps);
    }
}

const NeuronLayer = struct {
    error_: []f32,
    ivar_: []f32,
    gamma_: []f32,
    gamma_u_: []f32,
    gamma_m_: []f32,
    gamma_v_: []f32,
    beta_: []f32,
    beta_u_: []f32,
    beta_m_: []f32,
    beta_v_: []f32,
    weights_: [][]f32,
    state_: [][]f32,
    update_: [][]f32,
    m_: [][]f32,
    v_: [][]f32,
    transpose_: [][]f32,
    norm_: [][]f32,
    err_hist_: [][]f32,

    fn init(a: std.mem.Allocator, input_size: usize, num_cells: usize, horizon: usize, offset: usize) NeuronLayer {
        const gamma = alloc1(a, num_cells);
        for (gamma) |*g| g.* = 1.0;
        return .{
            .error_ = alloc1(a, num_cells),
            .ivar_ = alloc1(a, horizon),
            .gamma_ = gamma,
            .gamma_u_ = alloc1(a, num_cells),
            .gamma_m_ = alloc1(a, num_cells),
            .gamma_v_ = alloc1(a, num_cells),
            .beta_ = alloc1(a, num_cells),
            .beta_u_ = alloc1(a, num_cells),
            .beta_m_ = alloc1(a, num_cells),
            .beta_v_ = alloc1(a, num_cells),
            .weights_ = alloc2(a, num_cells, input_size),
            .state_ = alloc2(a, horizon, num_cells),
            .update_ = alloc2(a, num_cells, input_size),
            .m_ = alloc2(a, num_cells, input_size),
            .v_ = alloc2(a, num_cells, input_size),
            .transpose_ = alloc2(a, input_size - offset, num_cells),
            .norm_ = alloc2(a, horizon, num_cells),
            .err_hist_ = alloc2(a, horizon, num_cells),
        };
    }

    fn deinit(self: *NeuronLayer, a: std.mem.Allocator) void {
        a.free(self.error_);
        free2(a, self.err_hist_);
        a.free(self.ivar_);
        a.free(self.gamma_);
        a.free(self.gamma_u_);
        a.free(self.gamma_m_);
        a.free(self.gamma_v_);
        a.free(self.beta_);
        a.free(self.beta_u_);
        a.free(self.beta_m_);
        a.free(self.beta_v_);
        free2(a, self.weights_);
        free2(a, self.state_);
        free2(a, self.update_);
        free2(a, self.m_);
        free2(a, self.v_);
        free2(a, self.transpose_);
        free2(a, self.norm_);
    }
};

pub const LstmLayer = struct {
    state_: []f32,
    state_error_: []f32,
    stored_error_: []f32,
    tanh_state_: [][]f32,
    input_gate_state_: [][]f32,
    last_state_: [][]f32,
    gradient_clip_: f32,
    learning_rate_: f32,
    num_cells_: usize,
    epoch_: usize = 0,
    horizon_: usize,
    input_hist_: [][]f32 = &.{},
    aux_idx_hist_: []u32 = &.{},
    aux_val_hist_: []f32 = &.{},
    sym_hist_: []u32 = &.{},
    input_size_: usize, // auxiliary_input_size
    output_size_: usize,
    update_steps_: u64 = 0,
    update_limit_: u64 = if (@import("build_options").lstm_update_limit == 0)
        std.math.maxInt(u64)
    else
        @import("build_options").lstm_update_limit,
    forget_gate_: NeuronLayer,
    input_node_: NeuronLayer,
    output_gate_: NeuronLayer,
    // -Dlstm-aux-sparse: the live (symbol index, mass) pairs of the aux input
    // ROW CURRENTLY BEING PROCESSED. Borrowed slices into Lstm's per-epoch ring,
    // re-pointed by Lstm.predict before each forwardPass and by Lstm.perceive
    // before each BPTT backwardPass — BPTT visits PAST epochs, whose live sets
    // differ from the current byte's. Never written or read in stock builds
    // (aux_sparse == 0 makes every use site comptime-dead).
    aux_live_idx_: []const u32 = &.{},
    aux_live_val_: []const f32 = &.{},

    pub fn init(a: std.mem.Allocator, input_size: usize, auxiliary_input_size: usize, output_size: usize, num_cells: usize, horizon: usize, gradient_clip: f32, learning_rate: f32, rng: *std.Random) LstmLayer {
        const offset = output_size + auxiliary_input_size;
        var self: LstmLayer = .{
            .state_ = alloc1(a, num_cells),
            .state_error_ = alloc1(a, num_cells),
            .stored_error_ = alloc1(a, num_cells),
            .tanh_state_ = alloc2(a, horizon, num_cells),
            .input_gate_state_ = alloc2(a, horizon, num_cells),
            .last_state_ = alloc2(a, horizon, num_cells),
            .input_hist_ = alloc2(a, horizon, input_size - output_size),
            .aux_idx_hist_ = a.alloc(u32, horizon * @max(1, auxSparseK())) catch unreachable,
            .aux_val_hist_ = alloc1(a, horizon * @max(1, auxSparseK())),
            .sym_hist_ = a.alloc(u32, horizon) catch unreachable,
            .gradient_clip_ = gradient_clip,
            .learning_rate_ = learning_rate,
            .num_cells_ = num_cells,
            .horizon_ = horizon,
            .input_size_ = auxiliary_input_size,
            .output_size_ = output_size,
            .forget_gate_ = NeuronLayer.init(a, input_size, num_cells, horizon, offset),
            .input_node_ = NeuronLayer.init(a, input_size, num_cells, horizon, offset),
            .output_gate_ = NeuronLayer.init(a, input_size, num_cells, horizon, offset),
        };
        const val: f32 = std.math.sqrt(6.0 / @as(f32, @floatFromInt(auxiliary_input_size + output_size)));
        const low = -val;
        const range = 2 * val;
        for (0..num_cells) |i| {
            for (0..self.forget_gate_.weights_[i].len) |j| {
                self.forget_gate_.weights_[i][j] = low + rng.float(f32) * range;
                self.input_node_.weights_[i][j] = low + rng.float(f32) * range;
                self.output_gate_.weights_[i][j] = low + rng.float(f32) * range;
            }
            self.forget_gate_.weights_[i][self.forget_gate_.weights_[i].len - 1] = 1;
        }
        return self;
    }

    pub fn deinit(self: *LstmLayer, a: std.mem.Allocator) void {
        a.free(self.state_);
        a.free(self.state_error_);
        a.free(self.stored_error_);
        free2(a, self.tanh_state_);
        free2(a, self.input_gate_state_);
        free2(a, self.last_state_);
        free2(a, self.input_hist_);
        a.free(self.aux_idx_hist_);
        a.free(self.aux_val_hist_);
        a.free(self.sym_hist_);
        self.forget_gate_.deinit(a);
        self.input_node_.deinit(a);
        self.output_gate_.deinit(a);
    }

    fn forwardNeuron(self: *LstmLayer, neurons: *NeuronLayer, input: []const f32, input_symbol: usize) void {
        @setFloatMode(.optimized);
        // Strict FP (the layer-norm 1/sqrt below must stay exact); only the O(nc*input)
        // neuron dot-product is @Vector-reduced for AVX2 (see dotV).
        const nc = self.num_cells_;
        if (comptime gather) {
            // Split the dot at the aux/recurrent boundary. `input` is
            //   [ aux (input_size_) | recurrent (+ below-layer h) | bias ]
            // for EVERY layer (Lstm.setInput/setSparseInput write [0..aux) of
            // each layer's row; the recurrent copies start at input_size_), so
            // the split is layer-independent. The aux half is gathered over the
            // K live symbols; the tail keeps the dense vectorised pass.
            const aw = self.input_size_;
            const li = self.aux_live_idx_;
            const lv = self.aux_live_val_;
            for (0..nc) |i| {
                const w = neurons.weights_[i][self.output_size_ .. self.output_size_ + input.len];
                const s = dotGather(lv, li, w[0..aw]) + dotV(input[aw..], w[aw..]);
                neurons.norm_[self.epoch_][i] = neurons.weights_[i][input_symbol] + s;
            }
        } else {
            for (0..nc) |i| {
                const w = neurons.weights_[i][self.output_size_ .. self.output_size_ + input.len];
                neurons.norm_[self.epoch_][i] = neurons.weights_[i][input_symbol] + dotV(input, w);
            }
        }
        var ss: f32 = 0;
        for (0..nc) |i| ss += neurons.norm_[self.epoch_][i] * neurons.norm_[self.epoch_][i];
        // P0 strict_fp.rsqrt, not `1.0 / sqrt(..)`. The comment two
        // lines up has ALWAYS said this 1/sqrt "must stay exact" and line 54 records
        // that approximating it costs +28% on 50k — but the enclosing scope is
        // `@setFloatMode(.optimized)`, so LLVM was emitting VRSQRTSS (a vendor-defined
        // estimate) here. See src/strict_fp.zig.
        neurons.ivar_[self.epoch_] = strict_fp.rsqrt(ss / @as(f32, @floatFromInt(nc)) + 1e-5);
        // Bit-exact @Vector map: norm*=ivar; state = fma(norm,gamma,beta). Per-lane
        // ops with @mulAdd single-rounding == the scalar FMA -Oz emits under
        // @setFloatMode(.optimized) — no reduction, no reassociation.
        normScaleV(neurons.norm_[self.epoch_][0..nc], neurons.ivar_[self.epoch_], neurons.gamma_[0..nc], neurons.beta_[0..nc], neurons.state_[self.epoch_][0..nc]);
    }

    pub fn forwardPass(self: *LstmLayer, input: []const f32, input_symbol: usize, hidden: []f32, hidden_start: usize) void {
        @setFloatMode(.optimized);
        const nc = self.num_cells_;
        const e = self.epoch_;
        copyV(self.last_state_[e], self.state_);
        self.forwardNeuron(&self.forget_gate_, input, input_symbol);
        self.forwardNeuron(&self.input_node_, input, input_symbol);
        self.forwardNeuron(&self.output_gate_, input, input_symbol);
        for (0..nc) |i| {
            self.forget_gate_.state_[e][i] = Sigmoid.logistic(self.forget_gate_.state_[e][i]);
            self.input_node_.state_[e][i] = tanhF(self.input_node_.state_[e][i]);
            self.output_gate_.state_[e][i] = Sigmoid.logistic(self.output_gate_.state_[e][i]);
        }
        if (comptime lstm_bwvec) {
            // Split the state-update around the scalar tanh so the arithmetic
            // vectorizes (tanh stays a per-element libm call). Fast-math @Vector.
            const fg = self.forget_gate_.state_[e];
            const innode = self.input_node_.state_[e];
            const igs = self.input_gate_state_[e];
            const one: Vf = @splat(1.0);
            var i: usize = 0;
            while (i + VL <= nc) : (i += VL) {
                const fgv: Vf = fg[i..][0..VL].*;
                const igv: Vf = one - fgv;
                igs[i..][0..VL].* = igv;
                self.state_[i..][0..VL].* = @as(Vf, self.state_[i..][0..VL].*) * fgv + @as(Vf, innode[i..][0..VL].*) * igv;
            }
            while (i < nc) : (i += 1) {
                igs[i] = 1.0 - fg[i];
                self.state_[i] = self.state_[i] * fg[i] + innode[i] * igs[i];
            }
            for (0..nc) |k| self.tanh_state_[e][k] = tanhF(self.state_[k]);
            const og = self.output_gate_.state_[e];
            const tss = self.tanh_state_[e];
            i = 0;
            while (i + VL <= nc) : (i += VL) {
                hidden[hidden_start + i ..][0..VL].* = @as(Vf, og[i..][0..VL].*) * @as(Vf, tss[i..][0..VL].*);
            }
            while (i < nc) : (i += 1) hidden[hidden_start + i] = og[i] * tss[i];
        } else {
            for (0..nc) |i| {
                self.input_gate_state_[e][i] = 1.0 - self.forget_gate_.state_[e][i];
                self.state_[i] = self.state_[i] * self.forget_gate_.state_[e][i] + self.input_node_.state_[e][i] * self.input_gate_state_[e][i];
                self.tanh_state_[e][i] = tanhF(self.state_[i]);
                hidden[hidden_start + i] = self.output_gate_.state_[e][i] * self.tanh_state_[e][i];
            }
        }
        self.epoch_ += 1;
        if (self.epoch_ == self.horizon_) self.epoch_ = 0;
    }

    fn clip(self: *LstmLayer, arr: []f32) void {
        @setFloatMode(.optimized);
        if (comptime lstm_bwvec) {
            const lo: Vf = @splat(-self.gradient_clip_);
            const hi: Vf = @splat(self.gradient_clip_);
            var j: usize = 0;
            while (j + VL <= arr.len) : (j += VL) {
                arr[j..][0..VL].* = @min(@max(@as(Vf, arr[j..][0..VL].*), lo), hi);
            }
            while (j < arr.len) : (j += 1) {
                if (arr[j] < -self.gradient_clip_) arr[j] = -self.gradient_clip_ else if (arr[j] > self.gradient_clip_) arr[j] = self.gradient_clip_;
            }
        } else {
            for (arr) |*x| {
                if (x.* < -self.gradient_clip_) x.* = -self.gradient_clip_ else if (x.* > self.gradient_clip_) x.* = self.gradient_clip_;
            }
        }
    }

    pub fn backwardPass(self: *LstmLayer, input: []const f32, epoch: usize, layer: usize, input_symbol: usize, hidden_error: []f32) void {
        @setFloatMode(.optimized);
        const nc = self.num_cells_;
        if (epoch == self.horizon_ - 1) {
            copyV(self.stored_error_, hidden_error);
            zeroV(self.state_error_);
        } else {
            if (comptime lstm_bwvec) {
                addAccV(self.stored_error_[0..nc], hidden_error[0..nc]);
            } else {
                for (0..nc) |i| self.stored_error_[i] += hidden_error[i];
            }
        }
        if (comptime lstm_bwvec) {
            gateGradV(
                self.output_gate_.error_[0..nc],
                self.input_node_.error_[0..nc],
                self.forget_gate_.error_[0..nc],
                self.state_error_[0..nc],
                self.stored_error_[0..nc],
                self.output_gate_.state_[epoch][0..nc],
                self.tanh_state_[epoch][0..nc],
                self.input_node_.state_[epoch][0..nc],
                self.input_gate_state_[epoch][0..nc],
                self.last_state_[epoch][0..nc],
                self.forget_gate_.state_[epoch][0..nc],
                nc,
            );
        } else {
            for (0..nc) |i| {
                const og = self.output_gate_.state_[epoch][i];
                const ts = self.tanh_state_[epoch][i];
                self.output_gate_.error_[i] = ts * self.stored_error_[i] * og * (1.0 - og);
                self.state_error_[i] += self.stored_error_[i] * og * (1.0 - ts * ts);
                const in = self.input_node_.state_[epoch][i];
                self.input_node_.error_[i] = self.state_error_[i] * self.input_gate_state_[epoch][i] * (1.0 - in * in);
                self.forget_gate_.error_[i] = (self.last_state_[epoch][i] - in) * self.state_error_[i] * self.forget_gate_.state_[epoch][i] * self.input_gate_state_[epoch][i];
            }
        }
        zeroV(hidden_error);
        if (epoch > 0) {
            if (comptime lstm_bwvec) {
                mulInPlaceV(self.state_error_[0..nc], self.forget_gate_.state_[epoch][0..nc]);
            } else {
                for (0..nc) |i| self.state_error_[i] *= self.forget_gate_.state_[epoch][i];
            }
            zeroV(self.stored_error_);
        } else {
            if (self.update_steps_ < self.update_limit_) self.update_steps_ += 1;
        }
        self.backwardNeuron(&self.forget_gate_, input, epoch, layer, input_symbol, hidden_error);
        self.backwardNeuron(&self.input_node_, input, epoch, layer, input_symbol, hidden_error);
        self.backwardNeuron(&self.output_gate_, input, epoch, layer, input_symbol, hidden_error);
        self.clip(self.state_error_);
        self.clip(self.stored_error_);
        self.clip(hidden_error);
    }

    fn backwardNeuron(self: *LstmLayer, neurons: *NeuronLayer, input: []const f32, epoch: usize, layer: usize, input_symbol: usize, hidden_error: []f32) void {
        @setFloatMode(.optimized);
        // Strict FP; the O(nc*nc)/O(nc*input) dot-products + weight-update AXPY below are
        // @Vector'd (dotV/axpyV) — the biggest hot spot (~23% of total runtime).
        const nc = self.num_cells_;
        if (epoch == self.horizon_ - 1) {
            zeroV(neurons.gamma_u_);
            zeroV(neurons.beta_u_);
            const offset = self.output_size_ + self.input_size_;
            for (0..nc) |i| {
                zeroV(neurons.update_[i]);
                for (0..neurons.transpose_.len) |j| {
                    neurons.transpose_[j][i] = neurons.weights_[i][j + offset];
                }
            }
        }
        if (comptime lstm_bwvec) {
            betaGammaAccV(neurons.beta_u_[0..nc], neurons.gamma_u_[0..nc], neurons.error_[0..nc], neurons.norm_[epoch][0..nc]);
            errScaleV(neurons.error_[0..nc], neurons.gamma_[0..nc], neurons.ivar_[epoch]);
            const mean = dotV(neurons.error_[0..nc], neurons.norm_[epoch][0..nc]) / @as(f32, @floatFromInt(nc));
            errSubMeanV(neurons.error_[0..nc], mean, neurons.norm_[epoch][0..nc]);
        } else {
            for (0..nc) |i| {
                neurons.beta_u_[i] += neurons.error_[i];
                neurons.gamma_u_[i] += neurons.error_[i] * neurons.norm_[epoch][i];
            }
            for (0..nc) |i| neurons.error_[i] *= neurons.gamma_[i] * neurons.ivar_[epoch];
            var dot: f32 = 0;
            for (0..nc) |i| dot += neurons.error_[i] * neurons.norm_[epoch][i];
            const mean = dot / @as(f32, @floatFromInt(nc));
            for (0..nc) |i| neurons.error_[i] -= mean * neurons.norm_[epoch][i];
        }
        if (layer > 0) {
            for (0..nc) |i| hidden_error[i] += dotV(neurons.error_[0..nc], neurons.transpose_[nc + i][0..nc]);
        }
        if (epoch > 0) {
            for (0..nc) |i| self.stored_error_[i] += dotV(neurons.error_[0..nc], neurons.transpose_[i][0..nc]);
        }
        // update_[i][output_size .. output_size+input.len] += error_[i]*input ; update_[i][input_symbol] += error_[i]
        if (comptime gather) {
            // Same split as forwardNeuron, and here it is EXACT: the aux
            // entries this skips would each contribute fma(err, 0, u) == u.
            const aw = self.input_size_;
            const li = self.aux_live_idx_;
            const lv = self.aux_live_val_;
            // DEFERRED-ACCUM: record e_t, x_t and the live aux set; the whole
            // rank-1 outer product (scatter half AND dense half) is applied once
            // per horizon by the blocked kernels below, in the same per-element
            // summation order, so the result is unchanged bit-for-bit.
            self.sym_hist_[epoch] = @intCast(input_symbol);
            @memcpy(neurons.err_hist_[epoch][0..nc], neurons.error_[0..nc]);
            @memcpy(self.input_hist_[epoch][aw..input.len], input[aw..input.len]);
            {
                const K = comptime auxSparseK();
                const b0 = epoch * K;
                for (0..K) |t| {
                    self.aux_idx_hist_[b0 + t] = if (t < li.len) li[t] else 0;
                    self.aux_val_hist_[b0 + t] = if (t < lv.len) lv[t] else 0.0;
                }
            }
        } else {
            self.sym_hist_[epoch] = @intCast(input_symbol);
            @memcpy(neurons.err_hist_[epoch][0..nc], neurons.error_[0..nc]);
            @memcpy(self.input_hist_[epoch][0..input.len], input[0..input.len]);
        }
        if (epoch == 0) {
            const off = self.output_size_;
            const H = self.horizon_;
            const B: usize = 24;
            const gbeg: usize = if (comptime gather) self.input_size_ else 0;
            const gend: usize = input.len;
            {
                // deferred ONE-HOT COLUMN half: update_[i][sym_t] += e_t[i].
                // sym_t varies per epoch, so this was 3*nc scattered row touches
                // PER STEP; blocked over cells the rows are resident for all H
                // steps. Per element the sum over t is still descending.
                var ib0: usize = 0;
                while (ib0 < nc) : (ib0 += B) {
                    const iend0 = @min(ib0 + B, nc);
                    var t0: usize = H;
                    while (t0 > 0) {
                        t0 -= 1;
                        const sym = self.sym_hist_[t0];
                        const eh0 = neurons.err_hist_[t0];
                        for (ib0..iend0) |i| neurons.update_[i][sym] += eh0[i];
                    }
                }
            }
            if (comptime gather) {
                // deferred SCATTER half, blocked over cells
                const K = comptime auxSparseK();
                var ib: usize = 0;
                while (ib < nc) : (ib += B) {
                    const iend = @min(ib + B, nc);
                    var t: usize = H;
                    while (t > 0) {
                        t -= 1;
                        const eh = neurons.err_hist_[t];
                        const b0 = t * K;
                        for (ib..iend) |i| {
                            const u = neurons.update_[i][off .. off + gbeg];
                            const e = eh[i];
                            for (0..K) |q| u[self.aux_idx_hist_[b0 + q]] += e * self.aux_val_hist_[b0 + q];
                        }
                    }
                }
            }
            if (gend > gbeg) {
                // deferred DENSE half, blocked over cells
                var ib2: usize = 0;
                while (ib2 < nc) : (ib2 += B) {
                    const iend2 = @min(ib2 + B, nc);
                    var t2: usize = H;
                    while (t2 > 0) {
                        t2 -= 1;
                        const x = self.input_hist_[t2][gbeg..gend];
                        const eh2 = neurons.err_hist_[t2];
                        for (ib2..iend2) |i| axpyV(neurons.update_[i][off + gbeg .. off + gend], eh2[i], x);
                    }
                }
            }
            for (0..nc) |i| {
                adam(neurons.update_[i], neurons.m_[i], neurons.v_[i], neurons.weights_[i], self.learning_rate_, @floatFromInt(self.update_steps_), self.update_limit_);
            }
            adam(neurons.gamma_u_, neurons.gamma_m_, neurons.gamma_v_, neurons.gamma_, self.learning_rate_, @floatFromInt(self.update_steps_), self.update_limit_);
            adam(neurons.beta_u_, neurons.beta_m_, neurons.beta_v_, neurons.beta_, self.learning_rate_, @floatFromInt(self.update_steps_), self.update_limit_);
        }
    }
};

// ---- -Dlstm-aux-sparse kernel gates ----------------------------------------
// Comptime-parameterised on K, so they run in EVERY build including stock ones
// where the kernels themselves are comptime-dead. They prove the two exactness
// claims stated on dotGather/axpyScatter, which is what licenses replacing the
// dense aux pass with the sparse one.

test "axpyScatter: BIT-IDENTICAL to the dense axpyV over the sparsified vector" {
    // ★ The strong claim, and the one that makes the backward half free: the
    // dense pass's contribution at a zeroed position is fma(x, 0, y) == y, and
    // axpyV rounds every element independently, so skipping those positions
    // cannot change a single bit of the surviving ones.
    var prng = std.Random.DefaultPrng.init(0xB33F);
    const r = prng.random();
    inline for (.{ 4, 8, 16 }) |K| {
        var trial: usize = 0;
        while (trial < 100) : (trial += 1) {
            const V = 195; // the measured post-WRT vocab, deliberately not a multiple of 8
            var ids: [K]u32 = undefined;
            var vals: [K]f32 = undefined;
            var dense = [_]f32{0} ** V;
            var used = [_]bool{false} ** V;
            for (0..K) |t| {
                var ix = r.uintLessThan(u32, V);
                while (used[ix]) ix = r.uintLessThan(u32, V);
                used[ix] = true;
                ids[t] = ix;
                vals[t] = (r.float(f32) - 0.5) * 3.0;
                dense[ix] = vals[t];
            }
            const x = (r.float(f32) - 0.5) * 0.25;
            var y_dense = [_]f32{0} ** V;
            var y_sparse = [_]f32{0} ** V;
            for (&y_dense, &y_sparse) |*a, *b| {
                const v = (r.float(f32) - 0.5) * 7.0;
                a.* = v;
                b.* = v;
            }
            axpyV(&y_dense, x, &dense);
            axpyScatter(&y_sparse, x, &vals, &ids);
            try std.testing.expectEqualSlices(f32, &y_dense, &y_sparse);
        }
    }
}

test "dotGather: sums exactly the live terms (exact-arithmetic oracle)" {
    // The gathered dot reassociates, so it is compared against an oracle on
    // inputs where every partial sum is exactly representable (small integers
    // scaled by 2^-4): under those the reassociation is provably inert and any
    // MISSED or DOUBLED term shows up as a hard mismatch.
    var prng = std.Random.DefaultPrng.init(0xD07E);
    const r = prng.random();
    inline for (.{ 1, 2, 4, 8, 16, 32 }) |K| {
        var trial: usize = 0;
        while (trial < 100) : (trial += 1) {
            const V = 195;
            var ids: [K]u32 = undefined;
            var vals: [K]f32 = undefined;
            var w = [_]f32{0} ** V;
            var used = [_]bool{false} ** V;
            var want: f32 = 0;
            for (0..K) |t| {
                var ix = r.uintLessThan(u32, V);
                while (used[ix]) ix = r.uintLessThan(u32, V);
                used[ix] = true;
                ids[t] = ix;
                vals[t] = @as(f32, @floatFromInt(r.intRangeAtMost(i32, -8, 8))) / 16.0;
                w[ix] = @as(f32, @floatFromInt(r.intRangeAtMost(i32, -8, 8))) / 16.0;
                want += vals[t] * w[ix];
            }
            // Poison every non-live weight: a gather that walked the dense span
            // (or that used a stale index) would pick these up.
            for (&w, 0..) |*x, s| if (!used[s]) {
                x.* = 1024.0;
            };
            try std.testing.expectEqual(want, dotGather(&vals, &ids, &w));
        }
    }
}

test "dotGather: (0, 0.0) pad slots are an exact no-op" {
    // setSparseInput pads short live sets with (index 0, mass 0). The pad must
    // not perturb the sum even when w[0] is large.
    var w = [_]f32{0} ** 32;
    w[0] = 1e30;
    w[5] = 0.5;
    w[9] = -0.25;
    const ids_full = [_]u32{ 5, 9, 0, 0, 0, 0, 0, 0 };
    const vals_full = [_]f32{ 2.0, 4.0, 0, 0, 0, 0, 0, 0 };
    const ids_live = [_]u32{ 5, 9 };
    const vals_live = [_]f32{ 2.0, 4.0 };
    try std.testing.expectEqual(dotGather(&vals_live, &ids_live, &w), dotGather(&vals_full, &ids_full, &w));
    try std.testing.expectEqual(@as(f32, 0.0), dotGather(&vals_full, &ids_full, &w));
}
