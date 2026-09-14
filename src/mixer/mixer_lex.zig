//! Lex Mixer + MixerInput, ported from the REAL cmix-lex C++:
//!   mixer/mixer.{h,cpp}  and  mixer/mixer-input.{h,cpp}
//!  (an absolute build-machine path)
//!
//! This is the PredictorLex (590-input, 2-layer) mixer and is DIFFERENT from the
//! cmix-v21 mixer in ../mixer.zig (which predictor.zig still uses). Do not mix
//! them up. Kept in a parallel file so the v21 path is untouched.
//!
//! Ground-truth C++ lines are cited inline. Where the Rust re-impl disagreed,
//! the C++ wins (there were no disagreements in this piece; see the report).
//!
//! Key differences vs the v21 mixer.zig:
//!   * Update rule (mixer.cpp:70-108): a GLOBAL per-mixer step count only, with a
//!     step-thresholded decay (1.0 / 0.7 / 0.3 / 0.2 at 1e6 / 5e6 / 25e6 steps);
//!     an EARLY-OUT when |update| < 5e-12 && extra_input_size > 0 (taken BEFORE
//!     applying decay AND before touching any weight row); NO per-context step
//!     count and NO periodic weight decay.
//!   * Context overflow (mixer.cpp:19-42): after 10000 distinct u32 contexts, an
//!     unseen context uses a SHARED `context_base_` weight row (NOT a 0xDEADBEEF
//!     bucket like v21). The base row IS updated in place — matching C++.
//!   * MixerInputLex (mixer-input.{h,cpp}): a FIXED extra-input buffer set once by
//!     setExtraInputSize (zero-init, NEVER cleared) with indexed setExtraInput,
//!     plus setStretchedInputUnchecked. This is the C++ fixed-buffer cascade, not
//!     the v21 growing-ArrayList + snapshot + clear.

const std = @import("std");
const ram_census = @import("ramcensus");
const Sigmoid = @import("../sigmoid.zig").Sigmoid;

// AVX2-width SIMD kernels (same as lstm_layer), FMA-contracted via @mulAdd (single
// rounding, not an approximation). The AXPY (weight update) is element-wise — no
// reordering. The dot-product reduces 8 lanes + tree, which changes summation ORDER
// only (the online mixer adapts to its own numerics).
const VL = 8;
const Vf = @Vector(VL, f32);

/// dot(a,b) over a.len — 4×8-lane FMA accumulators + horizontal reduce (independent
/// accumulators break the FMA latency chain; same unroll the record's fast-math
/// autovectorizer emits).
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

/// -Dmixer-blockskip variants. Identical arithmetic on every block the mask does not
/// mark, and provably identity on the ones it does.
inline fn dotVMask(a: []const f32, b: []const f32, m0: u64, m1: u64) f32 {
    // Mirrors dotV EXACTLY — same four accumulators, same block-to-accumulator
    // assignment, same final reduction order — so that skipping an all-zero block
    // (fma(0,w,acc) == acc in IEEE-754) is bit-identical, not merely equivalent.
    var acc0: Vf = @splat(0.0);
    var acc1: Vf = @splat(0.0);
    var acc2: Vf = @splat(0.0);
    var acc3: Vf = @splat(0.0);
    var j: usize = 0;
    var blk: usize = 0;
    while (j + 4 * VL <= a.len) : ({
        j += 4 * VL;
        blk += 4;
    }) {
        const w: u64 = if (blk < 64) m0 else m1;
        const sh: u6 = @intCast(blk & 63);
        const nib = (w >> sh) & 0xF;
        if (nib & 1 == 0) acc0 = @mulAdd(Vf, @as(Vf, a[j..][0..VL].*), @as(Vf, b[j..][0..VL].*), acc0);
        if (nib & 2 == 0) acc1 = @mulAdd(Vf, @as(Vf, a[j + VL ..][0..VL].*), @as(Vf, b[j + VL ..][0..VL].*), acc1);
        if (nib & 4 == 0) acc2 = @mulAdd(Vf, @as(Vf, a[j + 2 * VL ..][0..VL].*), @as(Vf, b[j + 2 * VL ..][0..VL].*), acc2);
        if (nib & 8 == 0) acc3 = @mulAdd(Vf, @as(Vf, a[j + 3 * VL ..][0..VL].*), @as(Vf, b[j + 3 * VL ..][0..VL].*), acc3);
    }
    while (j + VL <= a.len) : ({
        j += VL;
        blk += 1;
    }) {
        const w: u64 = if (blk < 64) m0 else m1;
        if ((w >> @intCast(blk & 63)) & 1 == 0)
            acc0 = @mulAdd(Vf, @as(Vf, a[j..][0..VL].*), @as(Vf, b[j..][0..VL].*), acc0);
    }
    var s: f32 = @reduce(.Add, (acc0 + acc2) + (acc1 + acc3));
    while (j < a.len) : (j += 1) s = @mulAdd(f32, a[j], b[j], s);
    return s;
}

inline fn axpyVMask(y: []f32, x: f32, a: []const f32, m0: u64, m1: u64) void {
    const xv: Vf = @splat(x);
    var j: usize = 0;
    var blk: usize = 0;
    while (j + VL <= a.len) : ({
        j += VL;
        blk += 1;
    }) {
        const w: u64 = if (blk < 64) m0 else m1;
        if ((w >> @intCast(blk & 63)) & 1 != 0) continue;
        var yv: Vf = y[j..][0..VL].*;
        yv = @mulAdd(Vf, xv, @as(Vf, a[j..][0..VL].*), yv);
        y[j..][0..VL].* = yv;
    }
    while (j < a.len) : (j += 1) y[j] = @mulAdd(f32, x, a[j], y[j]);
}

/// y[j] += x * a[j] for all j — no reorder; FMA-contracted (single rounding,
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

// ---------------------------------------------------------------------------
// MixerInputLex  (mirror of MixerInput in mixer-input.{h,cpp})
// ---------------------------------------------------------------------------
// C++ backs both `inputs_` and `extra_inputs_` with std::valarray<float>.
//   - inputs_ : SetNumModels(n) resizes to n filled 0.5.
//   - extra_inputs_ : SetExtraInputSize(n) resizes to n (valarray::resize
//     zero-inits) and is NEVER cleared. SetExtraInput(index,p) writes one slot
//     after clamping to the stretched range.
// Both Mixers hold `const valarray<float>&` references to these two arrays and
// read them directly in Mix/Perceive— no snapshot.
/// -Dmixer-blockskip — skip whole ALIGNED all-zero input blocks in the layer-0
/// dot product and weight update.
///
/// A zero mixer input contributes nothing to `w·x` and nothing to `w += lr·err·x`.
/// MEASURED (`-Dmix-sparse-diag`, with `-Dfxcm-zero-neutral=1`): **25.74 % of aligned
/// 8-float vectors and 16.60 % of 64 B cache lines are entirely zero**, because half
/// the fxcm slots are inert whole-context emissions (mean run 5.22). Element-level
/// sparsity is 56 % but a scalar gather loop loses to `dotV`/`axpyV` by ~3× per
/// element, so block skipping is the only profitable form.
///
/// The block mask is computed ONCE per bit over the 590 inputs and reused by all 23
/// layer-0 mixers on BOTH passes, so its 73 tests amortise over 1,679 vector ops
/// (≈4.3 % overhead for a ≈25 % saving).
///
/// ★ **BIT-IDENTICAL by construction**: for an all-zero block, `fma(0, w, acc) == acc`
/// and `y += a·0 == y` exactly in IEEE-754, so skipping changes no rounding and no
/// accumulation order. ⇒ this knob is a PURE instruction/wall lever with a provably
/// unchanged archive, measurable against `-Dfxcm-zero-neutral=1` alone. (With the
/// neutral NOT exact, only the 35-slot pad is ever all-zero and the knob does almost
/// nothing — the two levers are designed to be measured in that order.)
const BLOCKSKIP: bool = @import("build_options").mixer_blockskip;

/// -Dmixskip-probe — INSTRUMENT ONLY. 1 = compute
/// the mask exactly as normal, then throw it away and mark NOTHING skippable, so every
/// per-block test still executes and no block is ever skipped. Isolates the lever's
/// bookkeeping cost from its gross saving. Bit-identical to the dense base by
/// construction. Lives in mixer_options (the own-module rule) ⇒ zero decomp_bin cost.
const MIXSKIP_PROBE: u32 = @import("mixer_options").mixskip_probe;

pub const MixerInputLex = struct {
    inputs: []f32 = &.{},
    extra_inputs: []f32 = &.{},
    /// -Dmixer-blockskip: bit b set ⇒ input block [b*VL, b*VL+VL) is entirely zero.
    /// Computed once per bit by `computeBlockMask`, consumed by every layer-0 mixer.
    zmask: if (BLOCKSKIP) [2]u64 else void = if (BLOCKSKIP) .{ 0, 0 } else {},
    sigmoid: *const Sigmoid,
    min: f32,
    max: f32,
    stretched_min: f32,
    stretched_max: f32,
    alloc: std.mem.Allocator,

    // mixer-input.cpp:3-5 — inputs_(0.5,1), min_=eps, max_=1-eps,
    // stretched_min_=Logit(0), stretched_max_=Logit(1).
    pub fn init(a: std.mem.Allocator, sigmoid: *const Sigmoid, eps: f32) MixerInputLex {
        return .{
            .sigmoid = sigmoid,
            .min = eps,
            .max = 1 - eps,
            .stretched_min = sigmoid.logit(0),
            .stretched_max = sigmoid.logit(1),
            .alloc = a,
        };
    }

    pub fn deinit(self: *MixerInputLex) void {
        if (self.inputs.len != 0) self.alloc.free(self.inputs);
        if (self.extra_inputs.len != 0) self.alloc.free(self.extra_inputs);
    }

    // mixer-input.cpp:7-9 — inputs_.resize(num_models, 0.5).
    pub fn setNumModels(self: *MixerInputLex, num_models: usize) void {
        if (self.inputs.len != 0) self.alloc.free(self.inputs);
        self.inputs = self.alloc.alloc(f32, num_models) catch unreachable;
        for (self.inputs) |*v| v.* = 0.5;
    }

    /// -Dmixer-blockskip: recompute the all-zero block mask. Call ONCE per bit, after
    /// every input is set and before the mixer cascade.
    pub fn computeBlockMask(self: *MixerInputLex) void {
        if (comptime !BLOCKSKIP) return;
        self.zmask = .{ 0, 0 };
        const n = self.inputs.len;
        var blk: usize = 0;
        var b: u7 = 0;
        while (blk + VL <= n) : ({
            blk += VL;
            b += 1;
        }) {
            const v: Vf = self.inputs[blk..][0..VL].*;
            // "every lane is +0.0 or -0.0" — compare against a zero vector, which is
            // true for both signed zeros and false for anything else (NaN included).
            const isz = v == @as(Vf, @splat(@as(f32, 0.0)));
            if (@reduce(.And, isz)) {
                self.zmask[b >> 6] |= @as(u64, 1) << @intCast(b & 63);
            }
        }
        // -Dmixskip-probe=1: the mask was computed at full cost and is now discarded, so
        // the per-block tests below all run and all decide "do the work". Isolates
        // bookkeeping from saving.
        if (comptime MIXSKIP_PROBE == 1) self.zmask = .{ 0, 0 };
    }

    /// True iff block `b` is entirely zero.
    pub inline fn blockZero(self: *const MixerInputLex, b: usize) bool {
        if (comptime !BLOCKSKIP) return false;
        return (self.zmask[b >> 6] >> @intCast(b & 63)) & 1 != 0;
    }

    // mixer-input.cpp:11-15 — clamp to [min_,max_] then store Logit(p).
    pub fn setInput(self: *MixerInputLex, index: usize, p_in: f32) void {
        var p = p_in;
        if (p < self.min) p = self.min else if (p > self.max) p = self.max;
        self.inputs[index] = self.sigmoid.logit(p);
    }

    // mixer-input.cpp:17-21 — clamp to stretched range, store as-is.
    pub fn setStretchedInput(self: *MixerInputLex, index: usize, p_in: f32) void {
        var p = p_in;
        if (p > self.stretched_max) p = self.stretched_max else if (p < self.stretched_min) p = self.stretched_min;
        self.inputs[index] = p;
    }

    // mixer-input.h:15 — SetStretchedInputUnchecked: no clamp.
    pub fn setStretchedInputUnchecked(self: *MixerInputLex, index: usize, p: f32) void {
        self.inputs[index] = p;
    }

    // mixer-input.cpp:22-25 — SetZero.
    pub fn setZero(self: *MixerInputLex, index: usize) void {
        self.inputs[index] = 0.0;
    }

    // mixer-input.h:18 — SetExtraInputSize: extra_inputs_.resize(size).
    // std::valarray::resize zero-inits every slot. NEVER cleared afterwards.
    pub fn setExtraInputSize(self: *MixerInputLex, size: usize) void {
        if (self.extra_inputs.len != 0) self.alloc.free(self.extra_inputs);
        self.extra_inputs = self.alloc.alloc(f32, size) catch unreachable;
        @memset(self.extra_inputs, 0);
    }

    // mixer-input.cpp:27-31 — clamp to stretched range, store at index.
    pub fn setExtraInput(self: *MixerInputLex, index: usize, p_in: f32) void {
        var p = p_in;
        if (p > self.stretched_max) p = self.stretched_max else if (p < self.stretched_min) p = self.stretched_min;
        self.extra_inputs[index] = p;
    }
};

// ---------------------------------------------------------------------------
// ContextData  (mixer.h:9-18)
// ---------------------------------------------------------------------------
// weights(input_size), extra_weights(extra_input_size); both valarray -> 0-init.
// The `steps` member is commented out in the lex C++ (no per-context steps).
const ContextData = struct {
    weights: []f32,
    extra_weights: []f32,
};

// ---------------------------------------------------------------------------
// MixerLex  (mixer.{h,cpp})
// ---------------------------------------------------------------------------
pub const MixerLex = struct {
    resolved: ?*ContextData = null,
    mi: *MixerInputLex,
    // inputs_size_ / extra_inputs_size_ are captured at construction (mixer.cpp:13).
    // NOTE: extra_inputs_size_ comes from the `extra_input_size` CTOR arg (the
    // mixer's own index in layer 0), NOT from mi.extra_inputs.len. Each layer-0
    // mixer i therefore reads only extra slots [0, i).
    inputs_size: usize,
    extra_inputs_size: usize,
    p: f32 = 0.5,
    learning_rate: f32,
    context: *const u64,
    steps: u64 = 0,
    context_map: std.AutoHashMap(u32, *ContextData),
    context_base: *ContextData, // shared overflow row (mixer.cpp:24)
    alloc: std.mem.Allocator,

    // mixer.cpp:9-17 — CTOR. inputs_size_ = inputs.size(captured now),
    // extra_inputs_size_ = extra_input_size arg, context_base_(inputs.size,
    // extra_inputs_size_).
    pub fn init(
        a: std.mem.Allocator,
        mi: *MixerInputLex,
        context: *const u64,
        learning_rate: f32,
        extra_input_size: usize,
    ) MixerLex {
        const inputs_size = mi.inputs.len;
        const base = a.create(ContextData) catch unreachable;
        base.* = .{
            .weights = a.alloc(f32, inputs_size) catch unreachable,
            .extra_weights = a.alloc(f32, extra_input_size) catch unreachable,
        };
        @memset(base.weights, 0);
        @memset(base.extra_weights, 0);
        return .{
            .mi = mi,
            .inputs_size = inputs_size,
            .extra_inputs_size = extra_input_size,
            .learning_rate = learning_rate,
            .context = context,
            .context_map = std.AutoHashMap(u32, *ContextData).init(a),
            .context_base = base,
            .alloc = a,
        };
    }

    pub fn deinit(self: *MixerLex) void {
        var it = self.context_map.valueIterator();
        while (it.next()) |v| {
            self.alloc.free(v.*.weights);
            self.alloc.free(v.*.extra_weights);
            self.alloc.destroy(v.*);
        }
        self.context_map.deinit();
        self.alloc.free(self.context_base.weights);
        self.alloc.free(self.context_base.extra_weights);
        self.alloc.destroy(self.context_base);
    }

    fn newContextData(self: *MixerLex) *ContextData {
        ram_census.note((self.inputs_size + self.extra_inputs_size) * @sizeOf(f32), @returnAddress());
        const d = self.alloc.create(ContextData) catch unreachable;
        d.* = .{
            .weights = self.alloc.alloc(f32, self.inputs_size) catch unreachable,
            .extra_weights = self.alloc.alloc(f32, self.extra_inputs_size) catch unreachable,
        };
        @memset(d.weights, 0);
        @memset(d.extra_weights, 0);
        return d;
    }

    // mixer.cpp:19-42 — GetContextData.
    // C++ keys the map on `unsigned int` (u32), truncating the 64-bit context.
    //   size>=10000 && absent -> shared context_base_ (NOT inserted).
    //   present               -> that row.
    //   size<10000 && absent -> insert a fresh zero row.
    //
    // Hit path is a SINGLE probe: the old form did `count>=limit &&
    // !contains(ctx)` then `getOrPut(ctx)`, so every steady-state hit (the common
    // case once the 10000-context cap is reached — which the low-order mixers hit
    // almost immediately) cost TWO hash probes. `getPtr` returns on the first.
    // Bit-identical: same map mutations, same row (verified by the oracle disc
    // checksum gate).
    fn getContextData(self: *MixerLex) *ContextData {
        const ctx: u32 = @truncate(self.context.*);
        if (self.context_map.getPtr(ctx)) |v| return v.*;
        // absent:
        if (self.context_map.count() >= 10000) return self.context_base;
        const gop = self.context_map.getOrPut(ctx) catch unreachable;
        if (!gop.found_existing) gop.value_ptr.* = self.newContextData();
        return gop.value_ptr.*;
    }

    // mixer.cpp:44-68 — Mix. Scalar dot product (ratio parity, not byte-exact).
    // Reads mi.inputs / mi.extra_inputs directly; no snapshot.
    /// Resolve this bit's context row and prefetch its weight lines. Called for
    /// all mixers BEFORE the mix cascade: 23 rows x ~2.4KB resolved up front puts
    /// the row loads in flight while earlier mixers compute (the workload is
    /// latency-bound: ~2GB/s DRAM traffic, far from bandwidth-saturated).
    /// getContextData is idempotent within a bit (context unchanged), so the
    /// early resolve performs the identical map mutation mixwould.
    pub fn prefetchRow(self: *MixerLex) void {
        const data = self.getContextData();
        self.resolved = data;
        var off: usize = 0;
        while (off < self.inputs_size * 4) : (off += 64) {
            @prefetch(@as([*]const u8, @ptrCast(data.weights.ptr)) + off, .{ .rw = .read, .locality = 3, .cache = .data });
        }
        if (self.extra_inputs_size != 0) {
            off = 0;
            while (off < self.extra_inputs_size * 4) : (off += 64) {
                @prefetch(@as([*]const u8, @ptrCast(data.extra_weights.ptr)) + off, .{ .rw = .read, .locality = 3, .cache = .data });
            }
        }
    }

    pub fn mix(self: *MixerLex) f32 {
        @setFloatMode(.optimized);
        // Resolve (prefetchRow already did this for layer-0 mixers) and KEEP the
        // row in `resolved` for perceiveto reuse: the mixer context is
        // unchanged between predict's mix and perceive's update (mixers
        // perceive BEFORE updateContexts), so re-looking-up the same row was a
        // redundant hash probe. perceiveconsumes and clears it.
        const data = self.resolved orelse self.getContextData();
        self.resolved = data;
        self.p = if (comptime BLOCKSKIP)
            dotVMask(self.mi.inputs[0..self.inputs_size], data.weights[0..self.inputs_size], self.mi.zmask[0], self.mi.zmask[1])
        else
            dotV(self.mi.inputs[0..self.inputs_size], data.weights[0..self.inputs_size]);
        if (self.extra_inputs_size != 0) {
            self.p += dotV(self.mi.extra_inputs[0..self.extra_inputs_size], data.extra_weights[0..self.extra_inputs_size]);
        }
        return self.p;
    }

    /// -Dmixer-e2e ONLY. Same AXPY as `perceive`, but the error scalar is supplied
    /// by the caller (the exact end-to-end gradient dL/dh_i = (sigma(z)-y)*gamma_i,
    /// computed once in predictor_lex over all 23 mixers for 253 MAC/bit).
    /// Comptime-dead when the knob is off: nothing calls it.
    pub fn perceiveErr(self: *MixerLex, err_scalar: f32) void {
        @setFloatMode(.optimized);
        var decay: f32 = @import("build_options").mixer_decay_floor;
        if (self.steps < 25000000) {
            decay = 0.3;
            if (self.steps < 5000000) {
                decay = 0.7;
                if (self.steps < 1000000) decay = 1.0;
            }
        }
        self.steps += 1;
        var update: f32 = self.learning_rate * err_scalar;
        if (@abs(update) < 0.000000000005 and self.extra_inputs_size > 0) {
            self.resolved = null;
            return;
        }
        update = decay * update;
        const data = self.resolved orelse self.getContextData();
        self.resolved = null;
        if (comptime BLOCKSKIP)
            axpyVMask(data.weights[0..self.inputs_size], -update, self.mi.inputs[0..self.inputs_size], self.mi.zmask[0], self.mi.zmask[1])
        else
            axpyV(data.weights[0..self.inputs_size], -update, self.mi.inputs[0..self.inputs_size]);
        if (self.extra_inputs_size != 0) {
            axpyV(data.extra_weights[0..self.extra_inputs_size], -update, self.mi.extra_inputs[0..self.extra_inputs_size]);
        }
    }

    /// -Dmixer-e2e ONLY. Read-only access to this bit's resolved row, for the
    /// gamma backward pass. Never called with the knob off.
    pub fn resolvedRow(self: *MixerLex) *ContextData {
        return self.resolved orelse self.getContextData();
    }

    /// -Dmixer-e2e ONLY. Seed the layer-1 weight row to 1/23 on its mixer inputs
    /// ("start as the plain average"). Zero-init is a fixed point of exact
    /// gradient descent on this network — see the affine-gradient preflight.
    pub fn seedHead(self: *MixerLex, head: usize, val: f32) void {
        const d = self.getContextData();
        self.resolved = null;
        for (d.weights[0..head]) |*w| w.* = val;
    }

    // mixer.cpp:70-108 — Perceive.
    pub fn perceive(self: *MixerLex, bit: i32) void {
        @setFloatMode(.optimized);
        // Step-thresholded decay on the GLOBAL per-mixer step count.
        var decay: f32 = @import("build_options").mixer_decay_floor; // -Dmixer-decay-floor (C11)
        if (self.steps < 25000000) {
            decay = 0.3;
            if (self.steps < 5000000) {
                decay = 0.7;
                if (self.steps < 1000000) decay = 1.0;
            }
        }
        self.steps += 1; // ++steps_ happens BEFORE the early-out (mixer.cpp:81)

        var update: f32 = self.learning_rate * (Sigmoid.logistic(self.p) - @as(f32, @floatFromInt(bit)));
        // Early-out: before decay, before touching any weight row (mixer.cpp:84-86).
        if (@abs(update) < 0.000000000005 and self.extra_inputs_size > 0) return;

        update = decay * update;
        // Reuse the row mixalready resolved this bit (context is unchanged);
        // clear it so a layer-1 mixer's next mixre-resolves (it has no
        // prefetchRow). Falls back to a fresh lookup if predictdidn't run.
        const data = self.resolved orelse self.getContextData();
        self.resolved = null;

        // Weight update is an EXACT AXPY (no reorder) — pure speedup.
        if (comptime BLOCKSKIP)
            axpyVMask(data.weights[0..self.inputs_size], -update, self.mi.inputs[0..self.inputs_size], self.mi.zmask[0], self.mi.zmask[1])
        else
            axpyV(data.weights[0..self.inputs_size], -update, self.mi.inputs[0..self.inputs_size]);
        if (self.extra_inputs_size != 0) {
            axpyV(data.extra_weights[0..self.extra_inputs_size], -update, self.mi.extra_inputs[0..self.extra_inputs_size]);
        }
    }
};

// ---------------------------------------------------------------------------
// Compile-time smoke test (zig test -fllvm src/mixer/mixer_lex.zig)
// ---------------------------------------------------------------------------
test "mixer_lex basic wiring" {
    const a = std.testing.allocator;
    var sig = try Sigmoid.init(a, 4096);
    defer sig.deinit();

    var mi = MixerInputLex.init(a, &sig, 1.0e-4);
    defer mi.deinit();
    mi.setNumModels(590);
    mi.setExtraInputSize(23);

    // fill a couple inputs
    mi.setInput(0, 0.9);
    mi.setStretchedInput(1, 3.0);
    mi.setStretchedInputUnchecked(2, 12345.0); // unchecked: no clamp
    try std.testing.expect(mi.inputs[2] == 12345.0);
    mi.setExtraInput(5, 100.0); // should clamp to stretched_max
    try std.testing.expect(mi.extra_inputs[5] == mi.stretched_max);
    // untouched extra slots stay zero (never cleared)
    try std.testing.expect(mi.extra_inputs[0] == 0.0);

    var ctx: u64 = 42;
    // layer-0 mixer with extra_input_size = 3
    var m0 = MixerLex.init(a, &mi, &ctx, 0.005, 3);
    defer m0.deinit();
    const p0 = m0.mix();
    try std.testing.expect(std.math.isFinite(p0));
    m0.perceive(1);

    // extra_input_size == 0 path (layer-1 style has extra=0? here just test 0)
    var m1 = MixerLex.init(a, &mi, &ctx, 0.0003, 0);
    defer m1.deinit();
    _ = m1.mix();
    m1.perceive(0);

    // early-out only fires when extra_inputs_size > 0; drive many steps to make
    // sure nothing panics and weights stay finite.
    var s: usize = 0;
    while (s < 1000) : (s += 1) {
        _ = m0.mix();
        m0.perceive(@intCast(s & 1));
    }
    for (0..m0.inputs_size) |i| try std.testing.expect(std.math.isFinite(m0.context_base.weights[i]));
}

test "mixer_lex context overflow uses shared base row" {
    const a = std.testing.allocator;
    var sig = try Sigmoid.init(a, 4096);
    defer sig.deinit();
    var mi = MixerInputLex.init(a, &sig, 1.0e-4);
    defer mi.deinit();
    mi.setNumModels(4);
    mi.setExtraInputSize(0);
    for (0..4) |i| mi.setStretchedInput(i, 1.0);

    var ctx: u64 = 0;
    var m = MixerLex.init(a, &mi, &ctx, 0.01, 0);
    defer m.deinit();

    // Insert exactly 10000 distinct contexts.
    var c: u64 = 0;
    while (c < 10000) : (c += 1) {
        ctx = c;
        _ = m.mix();
        m.perceive(1);
    }
    try std.testing.expect(m.context_map.count() == 10000);
    // A brand-new context must NOT be inserted; it hits context_base.
    ctx = 999999;
    const d = m.getContextData();
    try std.testing.expect(d == m.context_base);
    try std.testing.expect(m.context_map.count() == 10000);
}
