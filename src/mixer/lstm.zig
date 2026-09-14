//! LSTM network, ported from cmix `mixer/lstm.{h,cpp}`.
//!
//! Stacks `num_layers` LstmLayers and a softmax output layer, trained by
//! truncated BPTT over `horizon` steps. Predicts a probability distribution
//! over the vocabulary for the next byte.
const std = @import("std");
const ram_census = @import("ramcensus");
const LstmLayer = @import("lstm_layer.zig").LstmLayer;

// Numerics mode for the hot softmax exp (the third hot transcendental beside
// sigmoid's exp and the gate tanh; ~vocab-size calls per byte):
//   -Dvendored-libm  -> vendored_math.exp through f64, rounded back to f32.
//     Build-time-fixed (no runtime libm), and the f64 detour is if anything
//     MORE accurate than a direct expf. Measured faster than compiler-rt expf
//     at ReleaseSmall (the ship mode) on the dev box.
//   -Dglibc          -> @exp lowers to the RUNNING machine's glibc expf
//     (record-parity behavior, but numerics float with the judge's glibc).
//   default (static) -> @exp lowers to compiler-rt expf (deterministic).
const use_vendored_libm = @import("build_options").vendored_libm;
// PRL (-Dlstm-prior-rho): blend the output softmax with the PPMd prior the
// LSTM already receives as input — softmax(z + log A), A = rho*P + (1-rho)*2/V.
// 0.0 = stock (the blend loop is comptime-dead).
const lstm_prior_rho: f32 = @import("build_options").lstm_prior_rho;
const lstm_prior_form: u32 = @import("build_options").lstm_prior_form;
const lstm_prior_clamp: u32 = @import("build_options").lstm_prior_clamp;
const auxparam_instr: bool = @import("build_options").auxparam_instr;
// -Dlstm-aux-sparse=K: the auxiliary input keeps its SHIPPED WIDTH and its
// SHIPPED index-by-symbol LAYOUT, but only the K largest masses are written and
// the rest are zero (byte_mixer.selectTopK). Two consequences live in this file:
//   (1) PRL folds the aux distribution into the output softmax and indexes it
//       BY SYMBOL. The index is in bounds either way (input_size_ is unchanged
//       = output_size_ = vocab_size, so the -Dlstm-aux-topk out-of-bounds trap
//       is structurally impossible here), but the CONTENT would be a truncated
//       prior, which is a second variable. The caller hands the dense vector
//       over separately via setAuxPrior so the fold stays exactly as shipped
//       and the arm is a clean single-variable change: the LSTM's INPUT channel
//       is sparsified, its OUTPUT prior is not.
//   (2) truncated BPTT re-reads the aux of PAST epochs, so the live index set
//       must be remembered per epoch — the ring below, written in lockstep with
//       layer_input_ and handed to the layer before each forward/backward pass.
// 0 = stock: every branch here is comptime-dead, the fold reads layer_input_
// exactly as before and nothing is allocated.
const lstm_aux_sparse: usize = @import("build_options").lstm_aux_sparse;
// -Dlstm-head-rank=M: factorise the 256-way output head as Head[256][M+1] x
// Ph[M][hidden]. See build.zig for the mechanism and the claim under test.
// 0 = stock: every branch below is comptime-dead and archives are
// bit-identical (no rng draws, no allocations, no extra fields touched).
const lstm_head_rank: usize = @import("build_options").lstm_head_rank;
// -Dlstm-head-lr-scale: see build.zig. Read ONLY inside the rank>0 branches.
const lstm_head_lr_scale: f32 = @import("build_options").lstm_head_lr_scale;

/// A = x^tau for dyadic tau, using IEEE sqrt only (exactly rounded => bit-exact
/// and libm-free, which the determinism contract requires). tau is comptime.
inline fn powDyadic(x: f32) f32 {
    return switch (comptime @as(u32, @intFromFloat(lstm_prior_rho * 100.0 + 0.5))) {
        25 => @sqrt(@sqrt(x)),
        50 => @sqrt(x),
        75 => @sqrt(x) * @sqrt(@sqrt(x)),
        100 => x,
        else => @compileError("-Dlstm-prior-form=1 requires -Dlstm-prior-rho in {0.25,0.5,0.75,1.0}"),
    };
}
// -Dlstm-compact-hist (Candidate A adoption guard): comptime knob for the
// one-layer compact output history. Default OFF ⇒ compactHistoryfolds to
// `false` and every compact branch below is comptime-dead — the stock pre-A
// horizon-ring code is what compiles (the -D pattern; dead code costs 0).
const lstm_compact_hist = @import("build_options").lstm_compact_hist;
const vendored_math = @import("lf_vendored_math.zig");
inline fn expF(x: f32) f32 {
    return if (comptime use_vendored_libm)
        @floatCast(vendored_math.exp(x))
    else
        @exp(x);
}

// AVX2 AXPY (y[j] += x*a[j], no reorder, FMA-contracted) for the output-layer
// update/backprop.
const VL = 8;
const Vf = @Vector(VL, f32);
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

/// out[j] = last[j] + x*a[j] — the output-layer copy+AXPY fused into one pass.
/// Bit-identical to `@memcpy(out, last)` followed by `axpyV(out, x, a)` (same
/// @mulAdd per element; the copied intermediate was redundant), but halves the
/// memory traffic of the hottest per-byte loop (256 rows × hidden floats).
inline fn copyAxpyV(out: []f32, last: []const f32, x: f32, a: []const f32) void {
    const xv: Vf = @splat(x);
    var j: usize = 0;
    while (j + VL <= a.len) : (j += VL) {
        out[j..][0..VL].* = @mulAdd(Vf, xv, @as(Vf, a[j..][0..VL].*), @as(Vf, last[j..][0..VL].*));
    }
    while (j < a.len) : (j += 1) out[j] = @mulAdd(f32, x, a[j], last[j]);
    if (a.len < out.len) copyV(out[a.len..], last[a.len..out.len]);
}

/// dot(a,b) — 4×8-lane FMA accumulators + tree reduce (same as lstm_layer.dotV:
/// independent accumulators break the FMA latency chain).
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

fn alloc1(a: std.mem.Allocator, n: usize) []f32 {
    ram_census.note(n * @sizeOf(f32), @returnAddress());
    const s = a.alloc(f32, n) catch unreachable;
    @memset(s, 0);
    return s;
}

/// THE compact-output-history predicate — the ONLY place compact mode is
/// decided; every call site (init / perceive / predict) calls this, no
/// per-site re-derivation (fifth-e-compiler-layout-findings-jul-23 §Candidate
/// A adoption requirements). Compact iff ALL of:
///   * -Dlstm-compact-hist is ON — comptime: OFF folds this to `false`, so
///     the stock horizon-deep ring is what compiles;
///   * exactly one layer — multi-layer output contributions are accumulated
///     on top of the upper layer's recurrent error, which the `output_error_`
///     sufficient statistic does not cover;
///   * EVEN horizon — the ping-pong `epoch_ & 1` scheme needs the source and
///     destination W versions to alternate every step including across the
///     ring wrap, which holds iff horizon is even (the two-version parity
///     proof covers even horizons only; ship H=128, tests H=8).
/// Anything else falls back to the stock ring AT INIT (the horizon-deep
/// `output_layer_` is allocated and `output_error_` stays empty), and the
/// same predicate keeps perceive/predict on the stock path — the arguments
/// (layers_.len, horizon_) are immutable after init, so the decision cannot
/// drift between sites.
pub inline fn compactHistory(num_layers: usize, horizon: usize) bool {
    if (comptime !lstm_compact_hist) return false;
    return num_layers == 1 and horizon % 2 == 0;
}

pub const Lstm = struct {
    a: std.mem.Allocator,
    layers_: []LstmLayer,
    input_history_: []u32,
    hidden_: []f32,
    hidden_error_: []f32,
    layer_input_: [][][]f32, // [horizon][num_layers][layer-specific size]
    // Compact history (-Dlstm-compact-hist, gated by compactHistory): the
    // output weights are needed later only through W_epoch^T * error_epoch.
    // Keep two ping-pong matrices so the existing out-of-place copyAxpyV
    // operation is unchanged, and retain that sufficient statistic in
    // output_error_.  Knob-off / multi-layer / odd-horizon builds keep the
    // original horizon-deep weight ring (multi-layer because its output
    // contribution is accumulated on top of the upper layer's recurrent
    // error; odd horizon because the ping-pong parity proof needs even H).
    output_layer_: [][][]f32, // [2 or horizon][output_size][num_cells*num_layers+1]
    output_error_: [][]f32, // one-layer only: [horizon][num_cells]
    // -Dlstm-head-rank=M (comptime; all four are empty slices at M=0 and every
    // branch that reads them is comptime-dead). The flat `output_layer_` slab is
    // replaced by a shared projection `head_proj_[versions][M][hidden_size]` and
    // a per-symbol row set `head_out_[versions][output_size][M+1]` (the trailing
    // column is the bias, paired with `hp_[M] == 1`, exactly as the flat row's
    // last column pairs with `hidden_[hidden_size-1] == 1`).
    // `hp_` is the projected hidden the LAST predictproduced -- the head's
    // twin of `hidden_`, and the vector the per-step head update reads.
    // `head_u_` is that step's gradient wrt hp, u = Head^T . err: it is the
    // sufficient statistic BOTH the Ph update and the hidden-error contraction
    // need, so it is computed once per step.
    head_proj_: [][][]f32,
    head_out_: [][][]f32,
    hp_: []f32,
    head_u_: []f32,
    output_: [][]f32, // [horizon][output_size]
    learning_rate_: f32,
    num_cells_: usize,
    epoch_: usize = 0,
    horizon_: usize,
    input_size_: usize,
    output_size_: usize,

    // ---- auxparam instrument (branch-local): per-instance output prior fold.
    // form 255 = stock (the comptime -Dlstm-prior-rho path, untouched).
    // form 0 = none; 1 = mixture  A_i = par*p_i + (1-par)/V   (== shipped PRL);
    // form 2 = power A_i = p_i^par   (log-domain / temperature fold).
    // The fold is applied BEFORE normalization and the folded distribution is
    // what output_ carries, so the LSTM trains THROUGH the fold exactly as the
    // shipped PRL does. prior_p_/prior_log2_ are supplied by the caller so the
    // fold is decoupled from the aux input transform.
    prior_form_: u8 = 255,
    prior_par_: f32 = 0,
    prior_clamp_: f32 = 32.0,
    prior_p_: ?[]const f32 = null,
    prior_log2_: ?[]const f32 = null,

    // ---- -Dlstm-aux-sparse: the DENSE prior, decoupled from the aux input.
    // Borrowed (not owned): ByteMixer.byteUpdate points this at its own
    // `inputs` buffer for the duration of the perceivecall, before it is
    // cleared for the next byte. Empty in stock builds, where the PRL fold
    // reads layer_input_ and this field is never touched.
    aux_prior_: []const f32 = &.{},
    // ---- -Dlstm-aux-sparse: the live symbol set, one K-slot record per epoch.
    // Truncated BPTT walks epochs horizon-1..0 and re-reads each epoch's aux
    // input, so the gathered dot needs that epoch's index set — a scalar field
    // would silently reuse the CURRENT byte's indices against a past epoch's
    // weights. Written by setSparseInput at layer_input_'s epoch, in lockstep
    // with it, so ring[e] describes layer_input_[e] at every instant (including
    // the wrap-time aliasing where BPTT's epoch-0 pass reads the row setInput
    // has just overwritten — stock behaviour, reproduced exactly).
    aux_live_idx_: []u32 = &.{},
    aux_live_val_: []f32 = &.{},

    /// -Dlstm-aux-sparse only. Supply the dense (vocab_size-wide) auxiliary
    /// distribution that PRL folds into the output softmax. MUST be called
    /// before perceive, with a buffer that stays live and unmodified across
    /// the perceive call.
    pub fn setAuxPrior(self: *Lstm, dense: []const f32) void {
        self.aux_prior_ = dense;
    }

    /// -Dlstm-aux-sparse only. The sparse analogue of setInput: zero the aux
    /// span of every layer's input row for this epoch and scatter the K live
    /// masses back to THEIR OWN SYMBOL POSITIONS, then record the live set in
    /// the ring for BPTT. The dense span is kept written (rather than left
    /// stale) so that any read of layer_input_[e][l][0..input_size_] — the PRL
    /// fold's index-by-symbol read included — sees exactly the vector the model
    /// is trained on. That costs one vocab_size memset per byte against the
    /// 3-gate x num_cells dot it guards, and it is what makes the layout claim
    /// checkable at any point in the program rather than only at the gather.
    /// `ids.len == vals.len <= K`. Slots the caller did not fill are PADDED in
    /// the ring with (index 0, mass 0), which both hot kernels treat as an
    /// exact no-op (`fma(0, w[0], acc) == acc`, `y[0] = fma(e, 0, y[0]) ==
    /// y[0]`) — the pad is never scattered into the dense row, so it cannot
    /// clobber symbol 0's real mass.
    pub fn setSparseInput(self: *Lstm, ids: []const u32, vals: []const f32) void {
        const K = comptime lstm_aux_sparse;
        for (self.layers_, 0..) |_, i| {
            const row = self.layer_input_[self.epoch_][i][0..self.input_size_];
            @memset(row, 0);
            for (ids, vals) |ix, v| row[ix] = v;
        }
        const base = self.epoch_ * K;
        for (0..K) |t| {
            self.aux_live_idx_[base + t] = if (t < ids.len) ids[t] else 0;
            self.aux_live_val_[base + t] = if (t < vals.len) vals[t] else 0.0;
        }
    }

    pub fn setPriorFold(self: *Lstm, form: u8, par: f32) void {
        self.prior_form_ = form;
        self.prior_par_ = par;
    }
    pub fn setPriorClamp(self: *Lstm, c: f32) void {
        self.prior_clamp_ = c;
    }

    /// The dense per-symbol prior the PRL fold multiplies into the softmax.
    /// Stock: layer 0's auxiliary input slice, which IS that distribution.
    /// -Dlstm-aux-sparse: layer_input_ holds the same slice, at the same width,
    /// with the same symbol indexing — but with the tail masses ZEROED, which
    /// would silently make this a truncated-prior fold as well as a sparsified
    /// input. The dense vector arrives out-of-band instead so the fold is
    /// bit-for-bit the shipped one. The comptime branch means stock codegen is
    /// unchanged.
    inline fn priorVec(self: *Lstm) []const f32 {
        return if (comptime lstm_aux_sparse > 0)
            self.aux_prior_
        else
            self.layer_input_[self.epoch_][0][0..self.input_size_];
    }
    pub fn setPriorVecs(self: *Lstm, p: []const f32, l2: []const f32) void {
        self.prior_p_ = p;
        self.prior_log2_ = l2;
    }

    pub fn init(a: std.mem.Allocator, input_size: usize, output_size: usize, num_cells: usize, num_layers: usize, horizon: usize, learning_rate: f32, gradient_clip: f32, rng: *std.Random) *Lstm {
        const self = a.create(Lstm) catch unreachable;
        const hidden_size = num_cells * num_layers + 1;
        // Contiguous slabs behind the existing row-slice tables (bit-identical:
        // same values and iteration order; purely a locality fix). output_layer_
        // is the hot one — 256 rows x hidden floats touched twice per byte; as
        // per-row heap allocations those lines were scattered all over the heap.
        var lisz: usize = 0;
        for (0..num_layers) |i| {
            lisz += if (i == 0) 1 + num_cells + input_size else input_size + 1 + num_cells * 2;
        }
        const li_slab = alloc1(a, horizon * lisz);
        const layer_input = a.alloc([][]f32, horizon) catch unreachable;
        for (layer_input, 0..) |*epoch_arr, h| {
            epoch_arr.* = a.alloc([]f32, num_layers) catch unreachable;
            var off = h * lisz;
            for (epoch_arr.*, 0..) |*li, i| {
                const sz = if (i == 0) 1 + num_cells + input_size else input_size + 1 + num_cells * 2;
                li.* = li_slab[off .. off + sz];
                li.*[sz - 1] = 1; // bias
                off += sz;
            }
        }
        const compact_output_history = compactHistory(num_layers, horizon);
        const output_versions: usize = if (compact_output_history) 2 else horizon;
        const ol_slab = alloc1(a, output_versions * output_size * hidden_size);
        const output_layer = a.alloc([][]f32, output_versions) catch unreachable;
        for (output_layer, 0..) |*epoch_arr, h| {
            epoch_arr.* = a.alloc([]f32, output_size) catch unreachable;
            for (epoch_arr.*, 0..) |*ol, i| {
                const off = (h * output_size + i) * hidden_size;
                ol.* = ol_slab[off .. off + hidden_size];
            }
        }
        const output_error: [][]f32 = if (compact_output_history) blk: {
            const oe_slab = alloc1(a, horizon * num_cells);
            const oe = a.alloc([]f32, horizon) catch unreachable;
            for (oe, 0..) |*row, h| {
                row.* = oe_slab[h * num_cells .. (h + 1) * num_cells];
            }
            break :blk oe;
        } else @constCast(&.{});
        const out_slab = a.alloc(f32, horizon * output_size) catch unreachable;
        for (out_slab) |*x| x.* = 1.0 / @as(f32, @floatFromInt(output_size));
        const output = a.alloc([]f32, horizon) catch unreachable;
        for (output, 0..) |*o, h| o.* = out_slab[h * output_size .. (h + 1) * output_size];
        const hidden = alloc1(a, hidden_size);
        hidden[hidden_size - 1] = 1;

        const layers = a.alloc(LstmLayer, num_layers) catch unreachable;
        for (layers, 0..) |*layer, i| {
            layer.* = LstmLayer.init(a, layer_input[0][i].len + output_size, input_size, output_size, num_cells, horizon, gradient_clip, learning_rate, rng);
        }

        // -Dlstm-head-rank: allocate the factorised head and draw Ph. The draw
        // happens AFTER the layers so the rng stream every LstmLayer consumed is
        // untouched -- base and arm therefore start from BIT-IDENTICAL layer
        // weights and the only difference is the head's parameterisation.
        var head_proj: [][][]f32 = &.{};
        var head_out: [][][]f32 = &.{};
        var hp: []f32 = &.{};
        var head_u: []f32 = &.{};
        if (comptime lstm_head_rank > 0) {
            const M = lstm_head_rank;
            const hp_slab = alloc1(a, output_versions * M * hidden_size);
            head_proj = a.alloc([][]f32, output_versions) catch unreachable;
            for (head_proj, 0..) |*ver, h| {
                ver.* = a.alloc([]f32, M) catch unreachable;
                for (ver.*, 0..) |*row, k| {
                    const off = (h * M + k) * hidden_size;
                    row.* = hp_slab[off .. off + hidden_size];
                }
            }
            const ho_slab = alloc1(a, output_versions * output_size * (M + 1));
            head_out = a.alloc([][]f32, output_versions) catch unreachable;
            for (head_out, 0..) |*ver, h| {
                ver.* = a.alloc([]f32, output_size) catch unreachable;
                for (ver.*, 0..) |*row, i| {
                    const off = (h * output_size + i) * (M + 1);
                    row.* = ho_slab[off .. off + M + 1];
                }
            }
            hp = alloc1(a, M + 1);
            hp[M] = 1; // bias companion of hidden_[hidden_size-1] == 1
            head_u = alloc1(a, M);
            // Uniform +/- sqrt(3/hidden_size) => unit expected row norm => Ph^T Ph
            // is a near-projection, so the composite step at t=0 is EXACTLY the
            // stock SGD step restricted to a random M-dim subspace. The init is
            // therefore not a disguised learning-rate knob. Head starts at zero,
            // as the stock slab does. All versions get the SAME draw: the
            // ping-pong scheme requires version parity to hold at init (the stock
            // slab is all-zero in both versions for the same reason).
            const scale: f32 = @sqrt(3.0 / @as(f32, @floatFromInt(hidden_size)));
            const low: f32 = -scale;
            const range: f32 = 2.0 * scale;
            for (0..M) |k| {
                for (0..hidden_size) |j| {
                    const w = low + rng.float(f32) * range;
                    for (head_proj) |ver| ver[k][j] = w;
                }
            }
        }

        self.* = .{
            .a = a,
            .layers_ = layers,
            .input_history_ = a.alloc(u32, horizon) catch unreachable,
            .hidden_ = hidden,
            .hidden_error_ = alloc1(a, num_cells),
            .layer_input_ = layer_input,
            .output_layer_ = output_layer,
            .output_error_ = output_error,
            .head_proj_ = head_proj,
            .head_out_ = head_out,
            .hp_ = hp,
            .head_u_ = head_u,
            .output_ = output,
            .learning_rate_ = learning_rate,
            .num_cells_ = num_cells,
            .horizon_ = horizon,
            .input_size_ = input_size,
            .output_size_ = output_size,
        };
        @memset(self.input_history_, 0);
        // -Dlstm-aux-sparse: the per-epoch live-index ring (horizon x K, ~4 kB
        // at the shipped horizon 128 and K 8). Allocated only when engaged, so
        // the stock allocation sequence is untouched.
        if (comptime lstm_aux_sparse > 0) {
            self.aux_live_idx_ = a.alloc(u32, horizon * lstm_aux_sparse) catch unreachable;
            self.aux_live_val_ = a.alloc(f32, horizon * lstm_aux_sparse) catch unreachable;
            @memset(self.aux_live_idx_, 0);
            @memset(self.aux_live_val_, 0);
        }
        return self;
    }

    pub fn deinit(self: *Lstm) void {
        const a = self.a;
        const num_layers = self.layers_.len;
        const hidden_size = self.num_cells_ * num_layers + 1;
        // layer_input_: one slab of unequal-size rows (see init) + per-epoch row
        // tables + the outer table. Reconstruct the slab from row [0][0].
        var lisz: usize = 0;
        for (0..num_layers) |i| {
            lisz += if (i == 0) 1 + self.num_cells_ + self.input_size_ else self.input_size_ + 1 + self.num_cells_ * 2;
        }
        a.free(self.layer_input_[0][0].ptr[0 .. self.horizon_ * lisz]);
        for (self.layer_input_) |epoch_arr| a.free(epoch_arr);
        a.free(self.layer_input_);
        // output_layer_: equal rows of hidden_size over one slab. Frees are
        // SHAPE-driven (output_layer_.len / output_error_.len) rather than
        // re-deriving compactHistory: the shapes are exactly what init's
        // predicate decision allocated, so teardown matches by construction
        // in either mode.
        a.free(self.output_layer_[0][0].ptr[0 .. self.output_layer_.len * self.output_size_ * hidden_size]);
        for (self.output_layer_) |epoch_arr| a.free(epoch_arr);
        a.free(self.output_layer_);
        if (self.output_error_.len != 0) {
            a.free(self.output_error_[0].ptr[0 .. self.horizon_ * self.num_cells_]);
            a.free(self.output_error_);
        }
        // output_: equal rows of output_size over one slab.
        a.free(self.output_[0].ptr[0 .. self.horizon_ * self.output_size_]);
        a.free(self.output_);
        for (self.layers_) |*layer| layer.deinit(a);
        a.free(self.layers_);
        a.free(self.input_history_);
        if (comptime lstm_head_rank > 0) {
            const M = lstm_head_rank;
            a.free(self.head_proj_[0][0].ptr[0 .. self.head_proj_.len * M * hidden_size]);
            for (self.head_proj_) |ver| a.free(ver);
            a.free(self.head_proj_);
            a.free(self.head_out_[0][0].ptr[0 .. self.head_out_.len * self.output_size_ * (M + 1)]);
            for (self.head_out_) |ver| a.free(ver);
            a.free(self.head_out_);
            a.free(self.hp_);
            a.free(self.head_u_);
        }
        a.free(self.hidden_);
        a.free(self.hidden_error_);
        if (comptime lstm_aux_sparse > 0) {
            a.free(self.aux_live_idx_);
            a.free(self.aux_live_val_);
        }
        a.destroy(self);
    }

    pub fn setInput(self: *Lstm, input: []const f32) void {
        for (self.layers_, 0..) |_, i| {
            copyV(self.layer_input_[self.epoch_][i][0..self.input_size_], input[0..self.input_size_]);
        }
    }

    /// Read-only tap for the HERO hidden-state residual head (same tap the
    /// episodic-cache runtime and the pinned trace producer used). This is the
    /// layer-zero recurrent hidden output h (NOT LstmLayer.state_, the cell
    /// state c), and excludes hidden_'s constant final bias. Reading it cannot
    /// alter numerics.
    pub inline fn hiddenState(self: *const Lstm) []const f32 {
        return self.hidden_[0..self.num_cells_];
    }

    pub fn perceive(self: *Lstm, input: u32) []f32 {
        @setFloatMode(.optimized);
        const nc = self.num_cells_;
        var last_epoch: usize = if (self.epoch_ == 0) self.horizon_ - 1 else self.epoch_ - 1;
        const old_input = self.input_history_[last_epoch];
        self.input_history_[last_epoch] = input;
        const compact_output_history = compactHistory(self.layers_.len, self.horizon_);
        if (compact_output_history) {
            // Capture the exact contraction the wrap-time BPTT loop would have
            // computed from this historical matrix.  The target is now known,
            // W_last is still current, and the i/AXPY order is unchanged.
            //
            // FUSED with the weight update (the old post-BPTT copyAxpyV loop):
            // both loops walked 0..output_size_, computed the SAME `err`, and
            // read the SAME source matrix W_last (w == output_layer_[src_epoch]
            // since src_epoch == last_epoch & 1). One pass over W instead of
            // two. Exactness: `err` is the identical expression; `g` still
            // accumulates by increasing i with the same per-element @mulAdd, so
            // no FP operand or order changes; `dst` is the other ping-pong
            // buffer — compactHistoryadmits even horizons only, so
            // `epoch_ & 1 != last_epoch & 1` at every step and src == dst
            // aliasing is impossible. Legality of the move across the BPTT block: in
            // the compact path the BPTT block reads output_error_ and writes
            // hidden_error_/layer state only — it never touches output_layer_,
            // output_, or hidden_ (LstmLayer.backwardPass is not given them).
            const g = self.output_error_[last_epoch];
            zeroV(g);
            if (comptime lstm_head_rank > 0) {
                // Factorised twin of the fused loop below. The flat loop computes
                //   g = W_src^T . err   and   W_dst = W_src - lr * err (x) hidden.
                // With W = Head . Ph that is exactly
                //   u    = Head_src^T . err          (rank-M sufficient statistic)
                //   g    = Ph_src^T . u
                //   Head_dst = Head_src - lr * err (x) hp
                //   Ph_dst   = Ph_src   - lr * u  (x) hidden
                // Both source matrices are read BEFORE either is written (src and
                // dst are the two ping-pong versions and cannot alias -- same
                // even-horizon parity argument as the flat path), so the pair of
                // updates is a simultaneous SGD step, not a sequential one.
                const M = lstm_head_rank;
                const hlr = self.learning_rate_ * lstm_head_lr_scale;
                const u = self.head_u_;
                zeroV(u);
                const hsrc = self.head_out_[last_epoch & 1];
                const hdst = self.head_out_[self.epoch_ & 1];
                for (0..self.output_size_) |i| {
                    const err = if (i == input) self.output_[last_epoch][i] - 1 else self.output_[last_epoch][i];
                    axpyV(u, err, hsrc[i][0..M]);
                    copyAxpyV(hdst[i], hsrc[i], -(hlr * err), self.hp_);
                }
                const psrc = self.head_proj_[last_epoch & 1];
                const pdst = self.head_proj_[self.epoch_ & 1];
                for (0..M) |k| {
                    axpyV(g, u[k], psrc[k][0..self.num_cells_]);
                    copyAxpyV(pdst[k], psrc[k], -(hlr * u[k]), self.hidden_);
                }
            } else {
                const src = self.output_layer_[last_epoch & 1];
                const dst = self.output_layer_[self.epoch_ & 1];
                for (0..self.output_size_) |i| {
                    const err = if (i == input) self.output_[last_epoch][i] - 1 else self.output_[last_epoch][i];
                    axpyV(g, err, src[i][0..self.num_cells_]);
                    copyAxpyV(dst[i], src[i], -(self.learning_rate_ * err), self.hidden_);
                }
            }
        }
        if (self.epoch_ == 0) {
            var epoch: i64 = @as(i64, @intCast(self.horizon_)) - 1;
            while (epoch >= 0) : (epoch -= 1) {
                const ep: usize = @intCast(epoch);
                var layer: i64 = @as(i64, @intCast(self.layers_.len)) - 1;
                while (layer >= 0) : (layer -= 1) {
                    const ly: usize = @intCast(layer);
                    const offset = ly * nc;
                    if (compact_output_history) {
                        copyV(self.hidden_error_, self.output_error_[ep]);
                    } else if (comptime lstm_head_rank > 0) {
                        // (Head_ep . Ph_ep)^T . err, sliced to this layer's span:
                        // contract over the 256 rows FIRST (u, length M), then
                        // over the M projection rows. Accumulates onto
                        // hidden_error_ exactly as the flat AXPY chain does.
                        const M = lstm_head_rank;
                        const u = self.head_u_;
                        zeroV(u);
                        const hrow = self.head_out_[ep];
                        for (0..self.output_size_) |i| {
                            const err = if (i == self.input_history_[ep]) self.output_[ep][i] - 1 else self.output_[ep][i];
                            axpyV(u, err, hrow[i][0..M]);
                        }
                        const prow = self.head_proj_[ep];
                        for (0..M) |k| {
                            axpyV(self.hidden_error_, u[k], prow[k][offset .. offset + self.hidden_error_.len]);
                        }
                    } else {
                        for (0..self.output_size_) |i| {
                            const err = if (i == self.input_history_[ep]) self.output_[ep][i] - 1 else self.output_[ep][i];
                            axpyV(self.hidden_error_, err, self.output_layer_[ep][i][offset .. offset + self.hidden_error_.len]);
                        }
                    }
                    const prev_epoch: usize = if (ep == 0) self.horizon_ - 1 else ep - 1;
                    var input_symbol = self.input_history_[prev_epoch];
                    if (ep == 0) input_symbol = old_input;
                    // -Dlstm-aux-sparse: hand the layer THIS EPOCH's live set —
                    // BPTT is re-reading layer_input_[ep], and ring[ep] is its
                    // index/value companion by construction (setSparseInput
                    // writes both together).
                    if (comptime lstm_aux_sparse > 0) {
                        const base = ep * lstm_aux_sparse;
                        self.layers_[ly].aux_live_idx_ = self.aux_live_idx_[base..][0..lstm_aux_sparse];
                        self.layers_[ly].aux_live_val_ = self.aux_live_val_[base..][0..lstm_aux_sparse];
                    }
                    self.layers_[ly].backwardPass(self.layer_input_[ep][ly], ep, ly, input_symbol, self.hidden_error_);
                }
            }
        }
        if (!compact_output_history) {
            // Stock-ring path (knob off / multi-layer / odd horizon): the
            // horizon-deep weight ring's BPTT above READS output_layer_[ep]
            // for every epoch — so this update must stay after the BPTT
            // block. Untouched by the compact-path fusion; with the knob off
            // this is the only path that compiles (pre-A stock code).
            if (comptime lstm_head_rank > 0) {
                const M = lstm_head_rank;
                const hlr = self.learning_rate_ * lstm_head_lr_scale;
                const u = self.head_u_;
                zeroV(u);
                const hsrc = self.head_out_[last_epoch];
                const hdst = self.head_out_[self.epoch_];
                for (0..self.output_size_) |i| {
                    const err = if (i == input) self.output_[last_epoch][i] - 1 else self.output_[last_epoch][i];
                    axpyV(u, err, hsrc[i][0..M]);
                    copyAxpyV(hdst[i], hsrc[i], -(hlr * err), self.hp_);
                }
                const psrc = self.head_proj_[last_epoch];
                const pdst = self.head_proj_[self.epoch_];
                for (0..M) |k| {
                    copyAxpyV(pdst[k], psrc[k], -(hlr * u[k]), self.hidden_);
                }
            } else {
                for (0..self.output_size_) |i| {
                    const err = if (i == input) self.output_[last_epoch][i] - 1 else self.output_[last_epoch][i];
                    copyAxpyV(self.output_layer_[self.epoch_][i], self.output_layer_[last_epoch][i], -(self.learning_rate_ * err), self.hidden_);
                }
            }
        }
        _ = &last_epoch;
        return self.predict(input);
    }

    pub fn predict(self: *Lstm, input: u32) []f32 {
        @setFloatMode(.optimized);
        const nc = self.num_cells_;
        const output_epoch = if (compactHistory(self.layers_.len, self.horizon_)) self.epoch_ & 1 else self.epoch_;
        for (self.layers_, 0..) |*layer, i| {
            copyV(self.layer_input_[self.epoch_][i][self.input_size_ .. self.input_size_ + nc], self.hidden_[i * nc .. i * nc + nc]);
            // -Dlstm-aux-sparse: the current epoch's live set (see perceive's
            // BPTT twin). setSparseInput wrote ring[epoch_] and
            // layer_input_[epoch_] in the same call, so they agree here.
            if (comptime lstm_aux_sparse > 0) {
                const base = self.epoch_ * lstm_aux_sparse;
                layer.aux_live_idx_ = self.aux_live_idx_[base..][0..lstm_aux_sparse];
                layer.aux_live_val_ = self.aux_live_val_[base..][0..lstm_aux_sparse];
            }
            layer.forwardPass(self.layer_input_[self.epoch_][i], input, self.hidden_, i * nc);
            if (i < self.layers_.len - 1) {
                copyV(self.layer_input_[self.epoch_][i + 1][nc + self.input_size_ .. nc + self.input_size_ + nc], self.hidden_[i * nc .. i * nc + nc]);
            }
        }
        var max_out: f32 = 0;
        if (comptime lstm_head_rank > 0) {
            // hp = Ph . hidden ONCE (M dots of hidden_size), then the 256 output
            // logits are dots of length M+1 instead of hidden_size. hp_[M] stays
            // 1 and carries Head's bias column.
            const P = self.head_proj_[output_epoch];
            for (0..lstm_head_rank) |k| self.hp_[k] = dotV(P[k], self.hidden_);
            const H = self.head_out_[output_epoch];
            for (0..self.output_size_) |i| {
                const sum = dotV(H[i], self.hp_);
                self.output_[self.epoch_][i] = sum;
                if (sum > max_out) max_out = sum;
            }
        } else {
            for (0..self.output_size_) |i| {
                const sum = dotV(self.hidden_, self.output_layer_[output_epoch][i][0..self.hidden_.len]);
                self.output_[self.epoch_][i] = sum;
                if (sum > max_out) max_out = sum;
            }
        }
        var total: f32 = 0;
        var folded = false;
        if (comptime auxparam_instr) if (self.prior_form_ != 255) {
            // auxparam instrument path (see setPriorFold above).
            folded = true;
            const V: f32 = @floatFromInt(self.output_size_);
            switch (self.prior_form_) {
                0 => {
                    for (0..self.output_size_) |i| {
                        const e = expF(self.output_[self.epoch_][i] - max_out);
                        self.output_[self.epoch_][i] = e;
                        total += e;
                    }
                },
                1 => {
                    const pp = self.prior_p_.?;
                    const fl: f32 = (1.0 - self.prior_par_) / V;
                    for (0..self.output_size_) |i| {
                        const A = self.prior_par_ * pp[i] + fl;
                        const e = expF(self.output_[self.epoch_][i] - max_out) * A;
                        self.output_[self.epoch_][i] = e;
                        total += e;
                    }
                },
                else => {
                    const l2 = self.prior_log2_.?;
                    const cl = -self.prior_clamp_;
                    for (0..self.output_size_) |i| {
                        const A = @exp2(self.prior_par_ * @max(l2[i], cl));
                        const e = expF(self.output_[self.epoch_][i] - max_out) * A;
                        self.output_[self.epoch_][i] = e;
                        total += e;
                    }
                },
            }
        };
        if (folded) {
            // instrument already folded + accumulated `total`; fall through.
        } else if (comptime lstm_prior_rho != 0.0 and lstm_prior_form == 1) {
            // form 1: clamped power fold. A = max(P, 2^-C)^tau. Decouples the
            // prior's STRENGTH (tau) from its tail FLOOR (C); the stock form 0
            // uses rho for both. See build.zig -Dlstm-prior-form.
            const prior = self.priorVec();
            const fl: f32 = comptime 2.0 / @as(f32, @floatFromInt(@as(u64, 1) << @as(u6, @intCast(lstm_prior_clamp))));
            for (0..self.output_size_) |i| {
                const A = powDyadic(@max(prior[i], fl));
                const e = expF(self.output_[self.epoch_][i] - max_out) * A;
                self.output_[self.epoch_][i] = e;
                total += e;
            }
        } else if (comptime lstm_prior_rho != 0.0) {
            // PRL form 0 (stock): fold the prior multiplicatively — exp(z-max)*A is softmax(z+log A)
            // after the /total below, no log needed. The prior is this step's
            // ByteMixer feed (layer 0 input), 2x the compacted PPMd distribution
            // (ByteMixer scales by 2/num_models with num_models=1), hence the
            // 2/V floor; the common factor 2 cancels in the normalization. The
            // floor keeps A > 0 (rho < 1) where PPMd assigns zero mass, and the
            // cross-entropy gradient at the output logits stays Q_i - 1[i=y], so
            // the existing update/BPTT paths remain correct unchanged.
            const prior = self.priorVec();
            const floor: f32 = (1.0 - lstm_prior_rho) * 2.0 / @as(f32, @floatFromInt(self.output_size_));
            for (0..self.output_size_) |i| {
                const e = expF(self.output_[self.epoch_][i] - max_out) * (lstm_prior_rho * prior[i] + floor);
                self.output_[self.epoch_][i] = e;
                total += e;
            }
        } else {
            for (0..self.output_size_) |i| {
                self.output_[self.epoch_][i] = expF(self.output_[self.epoch_][i] - max_out);
                total += self.output_[self.epoch_][i];
            }
        }
        for (0..self.output_size_) |i| self.output_[self.epoch_][i] /= total;
        const epoch = self.epoch_;
        self.epoch_ += 1;
        if (self.epoch_ == self.horizon_) self.epoch_ = 0;
        return self.output_[epoch];
    }
};
