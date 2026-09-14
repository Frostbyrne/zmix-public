//! `Mixer1` — the fxcm per-context logistic mixer (fxcmv1.cpp VERSION 26, 472-660),
//! plus the integer `dot_product`/`train` kernels (mixer.rs) and the tiny 2-input
//! `Mix` chain mixer (fxcm_v26.rs Mix, fxcmv1.cpp 543-576).
//!
//! Port of src/mixer1.rs + src/mixer.rs (dot_product/train) + the `Mix` struct.
//! The C++ AVX2 dot_product/train are expressed here with 16-lane @Vector ops with
//! identical (bit-for-bit) semantics: dot_product `madd`→`srai 8`→accumulate;
//! train `adds(t,t)`→`mulhi(*,err)`→`adds(+1)`→`srai 1`→saturating add into `w`.

const std = @import("std");
const zero_alloc = @import("cmcold").zero_alloc;
const build_options = @import("build_options");
const Allocator = std.mem.Allocator;

/// -Dmxa-radix=K — fixed-point radix shift of the Mixer1 weight word.
/// K=0 is stock and every expression below collapses to the original constants.
/// Weights are stored K bits finer: the C++ init state 129 becomes 129<<K, the
/// update rounds at 2^(16-K) instead of 2^16, and `p`/`p1` shift the dot
/// product down by 11+K instead of 11, so the OUTPUT SCALE IS UNCHANGED.
/// Net: identical nominal learning rate, 2^K finer update quantum, 2^K less
/// weight headroom before the (already present) i16 saturation.
pub const RADIX: u4 = @intCast(build_options.mxa_radix);
pub const WBIAS: i16 = @as(i16, 129) << RADIX;
const TRAIN_SHIFT: u5 = 16 - @as(u5, RADIX);
const OUT_SHIFT: u6 = 11 + @as(u6, RADIX);

/// -Dmxa-diag — update-quantisation telemetry; comptime-dead at default.
pub const DIAG: bool = build_options.mxa_diag;
/// Sampling stride for the (expensive) scalar re-derivation in `diagScan`.
/// Every figure it produces is a RATIO, so a fixed stride is unbiased.
const DIAG_STRIDE: u64 = 64;

const V16i16 = @Vector(16, i16);
const V16i32 = @Vector(16, i32);
const V8i32 = @Vector(8, i32);

/// Round a width up to a multiple of 16, matching the C++ `(n+15)&-16`.
pub inline fn round16(n: usize) usize {
    return (n + 15) & ~@as(usize, 15);
}

/// Kernel body shared by the plain and bias-offset variants: weights are read
/// as `stored +% bias` and written back as `true -% bias` (an exact i16
/// bijection for any bias). `bias = 0` compiles to the original kernel.
inline fn dotProductBias(comptime bias: i16, t: []const i16, w: []const i16, n_in: usize) i32 {
    const n = round16(n_in);
    const bv: V16i16 = @splat(bias);
    var sum: V8i32 = @splat(0);
    var i: usize = 0;
    while (i < n) : (i += 16) {
        const tv: V16i16 = t[i..][0..16].*;
        const wv: V16i16 = @as(V16i16, w[i..][0..16].*) +% bv;
        const tw: V16i32 = @as(V16i32, tv) *% @as(V16i32, wv);
        var pair: V8i32 = undefined;
        inline for (0..8) |k| pair[k] = (tw[2 * k] +% tw[2 * k + 1]) >> 8;
        sum +%= pair;
    }
    var acc: i32 = 0;
    inline for (0..8) |k| acc +%= sum[k];
    return acc;
}

/// -Dmxa-dither — DITHERED (stochastic) rounding of the weight update.
///
/// The stock kernel applies `Δw = floor((2·t·err + 2^17/2) / 2^17)`, i.e.
/// round-half-up on a 1-LSB grid. That is a THRESHOLD, not a scaling: for every
/// weight whose exact step |t·err/2^16| < 1/2 the applied step is identically 0
/// — so its EXPECTED update is zero no matter how strong its true gradient is.
/// `-Dmxa-diag` measures that regime at 83.4 % of all fxcm mixer weight updates
/// (mean exact step 1.31 LSB: the mixer runs AT its own quantum).
///
/// Dithering replaces the fixed +1/2 offset with u ~ U[0, 2^17), making the
/// rounding UNBIASED: E[Δw] = t·err/2^16 exactly, at identical step size,
/// identical i16 storage, identical weight range and identical `uperr`.
/// The dither stream is a per-mixer 16-lane xorshift32 advanced in lockstep by
/// the (identical) traincall sequence on both ops ⇒ decoder-symmetric with
/// zero side data.
pub const DITHER_MODE: u32 = build_options.mxa_dither;
pub const DITHER: bool = DITHER_MODE != 0;
const V16u32 = @Vector(16, u32);
/// Per-mixer dither state; 64 B, distinct per mixer, never transmitted.
pub const DitherState = if (DITHER) V16u32 else void;
const SHIFT_D: u5 = TRAIN_SHIFT + 1;
const DITHER_HI: u5 = @intCast(32 - @as(u32, TRAIN_SHIFT) - 1);

inline fn trainBiasDither(comptime bias: i16, t: []const i16, w: []i16, n_in: usize, e: i32, st: *V16u32) void {
    if (e == 0) return;
    const n = round16(n_in);
    const bv: V16i16 = @splat(bias);
    const errv: V16i32 = @splat(e);
    var s = st.*;
    var i: usize = 0;
    while (i < n) : (i += 16) {
        // xorshift32, 16 lanes at a time (one PRNG step per weight vector).
        s ^= s << @splat(13);
        s ^= s >> @splat(17);
        s ^= s << @splat(5);
        const tv: V16i16 = t[i..][0..16].*;
        const tt: V16i16 = tv +| tv; // 2·t, saturating — same as stock
        const prod: V16i32 = @as(V16i32, tt) *% errv;
        // Mode 1 (the arm): u ~ U[0, 2^SHIFT_D) from the lane's PRNG word.
        // Mode 2 (null control): u pinned to 2^SHIFT_D/2 — identical code path,
        //   identical PRNG advance, so any delta vs stock would be an artefact.
        // Mode 3 (pessimum control): u biased by the sign of the pending step,
        //   i.e. a deliberate away-from-zero ratchet.
        const uni: V16i32 = @intCast(s >> @as(@Vector(16, u5), @splat(DITHER_HI)));
        const u: V16i32 = switch (DITHER_MODE) {
            2 => @splat(@as(i32, 1) << (SHIFT_D - 1)),
            3 => blk: {
                const half: V16i32 = @splat(@as(i32, 1) << (SHIFT_D - 1));
                const neg = @as(V16i32, @splat(0)) > prod;
                break :blk @select(i32, neg, @as(V16i32, @splat(0)), half + half - @as(V16i32, @splat(1)));
            },
            else => uni,
        };
        // Arithmetic shift == floor, so this is exactly stochastic rounding of
        // prod / 2^SHIFT_D. |prod| ≤ 4094·32767 ⇒ |q| ≤ 1024, fits i16.
        const q: V16i32 = (prod +% u) >> @as(@Vector(16, u5), @splat(SHIFT_D));
        var tmp: V16i16 = @intCast(q);
        const wv: V16i16 = @as(V16i16, w[i..][0..16].*) +% bv;
        tmp +|= wv;
        w[i..][0..16].* = tmp -% bv;
    }
    st.* = s;
}

inline fn trainBias(comptime bias: i16, t: []const i16, w: []i16, n_in: usize, e: i32) void {
    if (e == 0) return;
    const n = round16(n_in);
    const bv: V16i16 = @splat(bias);
    const err: V16i16 = @splat(@intCast(e));
    const one: V16i16 = @splat(1);
    var i: usize = 0;
    while (i < n) : (i += 16) {
        const tv: V16i16 = t[i..][0..16].*;
        var tmp: V16i16 = tv +| tv;
        // mulhi_epi16: (i16*i16 -> i32) >> 16, back to i16.
        const prod: V16i32 = @as(V16i32, tmp) *% @as(V16i32, err);
        tmp = @intCast(prod >> @as(@Vector(16, u5), @splat(TRAIN_SHIFT)));
        tmp +|= one;
        tmp = tmp >> @as(@Vector(16, u4), @splat(1));
        // The saturating accumulate must happen in the true-weight domain
        // (saturation bounds don't commute with the offset).
        const wv: V16i16 = @as(V16i16, w[i..][0..16].*) +% bv;
        tmp +|= wv;
        w[i..][0..16].* = tmp -% bv;
    }
}

/// `dot_product(t, w, n)` — `n` is rounded UP to a multiple of 16.
/// Bit-identical to the C++ AVX2 `_mm256_madd_epi16`→`srai 8`→accumulate loop.
pub inline fn dot_product(t: []const i16, w: []const i16, n_in: usize) i32 {
    return dotProductBias(0, t, w, n_in);
}

/// `train(t, w, n, err)` — `w[i] += round(t[i]*err/2^16 / 2)`, saturating to i16.
/// Skips when `err == 0` (C++ `if (e)`; the delta is identically zero).
pub inline fn train(t: []const i16, w: []i16, n_in: usize, e: i32) void {
    trainBias(0, t, w, n_in, e);
}

/// Bias-offset kernels for `Mixer1`'s weight banks, which store `w -% 129` so
/// the initial all-129 bank is all-zeros (lazy kernel zero pages, F1).
pub inline fn dot_product129(t: []const i16, w: []const i16, n_in: usize) i32 {
    return dotProductBias(WBIAS, t, w, n_in);
}

pub inline fn train129(t: []const i16, w: []i16, n_in: usize, e: i32, st: *DitherState) void {
    if (comptime DITHER) {
        trainBiasDither(WBIAS, t, w, n_in, e, st);
    } else {
        trainBias(WBIAS, t, w, n_in, e);
    }
}

/// -Dmxa-w32=K — MECHANISM ARM. The weight word becomes i32 with K extra bits
/// under the radix point, which removes the resolution floor (quantum 2^-K of
/// stock) AND the i16 saturation ceiling in one move. It costs 2x the mixer
/// banks, so it is a RESEARCH arm that isolates whether the quantised update
/// costs bytes — not a shippable design. K=0 disables it entirely.
pub const W32: u32 = build_options.mxa_w32;
pub const WIDE: bool = W32 != 0;
pub const WT = if (WIDE) i32 else i16;
const W32_BIAS: i32 = @as(i32, 129) << @as(u5, @intCast(if (WIDE) W32 else 0));
const W32_TRAIN: u6 = 16 - @as(u6, @intCast(if (WIDE) W32 else 0));
const W32_OUT: u6 = 11 + @as(u6, @intCast(if (WIDE) W32 else 0));
const V16i64 = @Vector(16, i64);

/// i32-weight dot product. `(t*w)` pairs are summed in 64-bit and the pairwise
/// `>>8` of the stock kernel is preserved so the output scale is unchanged.
inline fn dotProduct32(t: []const i16, w: []const i32, n_in: usize) i64 {
    const n = round16(n_in);
    const bv: @Vector(16, i32) = @splat(W32_BIAS);
    var sum: i64 = 0;
    var i: usize = 0;
    while (i < n) : (i += 16) {
        const tv: @Vector(16, i32) = @as(V16i16, t[i..][0..16].*);
        const wv: @Vector(16, i32) = @as(@Vector(16, i32), w[i..][0..16].*) +% bv;
        const tw: V16i64 = @as(V16i64, tv) *% @as(V16i64, wv);
        var k: usize = 0;
        while (k < 16) : (k += 2) sum +%= (tw[k] +% tw[k + 1]) >> 8;
    }
    return sum;
}

/// i32-weight train. Same nominal step, quantum 2^-K of stock, no saturation.
inline fn train32(t: []const i16, w: []i32, n_in: usize, e: i32) void {
    if (e == 0) return;
    const n = round16(n_in);
    const half: i64 = @as(i64, 1) << (W32_TRAIN - 1);
    var i: usize = 0;
    while (i < n) : (i += 1) {
        const p: i64 = @as(i64, t[i]) *% @as(i64, e);
        w[i] +%= @intCast((p + half) >> W32_TRAIN);
    }
}

/// -Dmxa-diag accumulators. Zero-sized (and every write dead) at default.
const DiagCounters = if (DIAG) struct {
    calls: u64 = 0, // update() calls
    err0: u64 = 0, // calls short-circuited by err==0 (incl. `elim`)
    w: u64 = 0, // weights visited by a live train()
    dead: u64 = 0, // ...whose ROUNDED delta is 0
    sat: u64 = 0, // ...whose |true weight| is within 1024 of i16 saturation
    exact: f64 = 0, // Σ |exact real-valued delta|
    appl: f64 = 0, // Σ |applied integer delta|
    lost: f64 = 0, // Σ |exact − applied|
} else struct {};

/// Per-context logistic mixer. `tables` (squash) is threaded in at call sites.
pub const Mixer1 = struct {
    /// inputs per context (`N`)
    n: usize = 0,
    /// number of contexts (`M`)
    m: usize = 0,
    /// `(N*M)+32` weights, context `c` occupying `wx[c*N .. c*N+N]`.
    /// Stored bias-offset: `wx[i] = w[i] -% 129`, so the C++ init state
    /// (every weight 129, fxcmv1.cpp:752) is all-zeros and the ~579 MiB of
    /// banks stay on lazy kernel zero pages until a context first trains.
    /// Only `dot_product129`/`train129` may touch this array.
    wx: []WT = &.{},
    /// active context
    cxt: usize = 0,
    /// last prediction (scaled 12 bits)
    pr: i32 = 2048,
    /// output scale (`>>11` after multiply)
    shift1: i32 = 0,
    /// error-elimination threshold
    elim: i32 = 0,
    /// error gain (`uperr`)
    uperr: i32 = 0,
    /// last error (set by `update`)
    err: i32 = 0,
    alloc: Allocator,
    /// -Dmxa-dither PRNG state (zero-sized at default)
    ds: DitherState = if (DITHER) @splat(0) else {},
    /// -Dmxa-diag telemetry (zero-sized at default)
    d: DiagCounters = .{},

    /// Bare, uninitialised mixer (`init` + `set_tx_wx` finish setup).
    pub fn new(a: Allocator) Mixer1 {
        return .{ .alloc = a };
    }

    pub fn deinit(self: *Mixer1) void {
        if (self.wx.len != 0) zero_alloc.free(self.alloc, self.wx);
        self.wx = &.{};
    }

    /// `Init(m, s, e, ue)` — set context count and the scale/elim/gain constants.
    pub fn init(self: *Mixer1, m: usize, shift1: i32, elim: i32, uperr: i32) void {
        self.m = m;
        self.cxt = 0;
        self.shift1 = shift1;
        self.elim = elim;
        self.uperr = uperr;
        self.err = 0;
        self.pr = 2048; // initial p = 0.5
        if (comptime DITHER) {
            // Deterministic, distinct-per-mixer, distinct-per-lane seeding from
            // the mixer's OWN configuration (no clock, no rand, no side data).
            var seed: u32 = @bitCast(@as(i32, @truncate(@as(i64, @intCast(m)) *% 0x9E3779B1)));
            seed ^= @as(u32, @bitCast(shift1)) *% 0x85EBCA77;
            seed ^= @as(u32, @bitCast(uperr)) *% 0xC2B2AE3D;
            seed ^= @as(u32, @bitCast(elim)) *% 0x27D4EB2F;
            var lanes: V16u32 = @splat(0);
            inline for (0..16) |k| {
                var x: u32 = seed +% (@as(u32, k) +% 1) *% 0x9E3779B9;
                x ^= x >> 16;
                x *%= 0x7FEB352D;
                x ^= x >> 15;
                x *%= 0x846CA68B;
                x ^= x >> 16;
                lanes[k] = if (x == 0) 0x1234567 else x; // xorshift32 must not be 0
            }
            self.ds = lanes;
        }
    }

    /// `setTxWx(n, mn)` — set the input width and allocate the weight bank.
    /// C++ allocates `(N*M)+32` shorts and eagerly sets the first `M*N` to
    /// bias `129` (committing every page); here the bank is bias-offset
    /// storage, so the init state is all-zeros — a lazy zero_alloc mapping.
    /// The `+32` tail is C++ overhang slack the kernels can never reach:
    /// every `n` in fxcm is a multiple of 16, so `round16(n) == n` and reads
    /// stop exactly at `m*n`.
    pub fn set_tx_wx(self: *Mixer1, n: usize) void {
        std.debug.assert(n % 16 == 0); // keeps the +32 tail unreachable
        self.n = n;
        self.wx = zero_alloc.alloc(self.alloc, WT, self.n * self.m + 32) catch unreachable;
    }

    /// Software-prefetch this context's weight row into cache. The row
    /// `wx[cxt*n .. +n]` is a 544-/560-wide i16 slice inside a multi-hundred-MB
    /// bank that misses to DRAM per bit; issuing the prefetch for ALL of a
    /// cascade's mixers before their `p1`/`p` dot-products hides the miss
    /// latency behind earlier mixers' compute (the fxcm-internal analog of the
    /// outer mixer_lex sm26 cascade prefetch — the workload is latency-bound).
    /// Pure cache hint: never changes `cxt`/`wx`/output.
    pub inline fn prefetchRow(self: *const Mixer1) void {
        const base = self.cxt * self.n;
        if (base + self.n > self.wx.len) return;
        const p8: [*]const u8 = @ptrCast(self.wx.ptr + base);
        // Prefetch the WHOLE row: measured (300k A/B) the full ~17-line prefetch
        // hides −17% of LLC-load-misses (−2.16% cycles) vs only −4% for a
        // first-3-line prefetch — the HW prefetcher does NOT reliably stream these
        // rows (18 interleaved 17-line rows defeat its stream detection), so every
        // line must be prefetched explicitly. Pure cache hint, bit-identical.
        var off: usize = 0;
        while (off < self.n * 2) : (off += 64) {
            @prefetch(p8 + off, .{ .rw = .read, .locality = 3, .cache = .data });
        }
    }

    /// `p` — predict from the shared inputs `tx`, returning `pr` (12-bit).
    /// Mutates only `pr`. `tables` is duck-typed (`.squash(i32) i32`).
    pub fn p(self: *Mixer1, tables: anytype, tx: []const i16) i32 {
        const base = self.cxt * self.n;
        if (comptime WIDE) {
            const raw64 = dotProduct32(tx, self.wx[base..], self.n);
            const dp32: i32 = @intCast((raw64 * self.shift1) >> W32_OUT);
            self.pr = tables.squash(dp32);
            return self.pr;
        }
        const raw = dot_product129(tx, self.wx[base..], self.n);
        // raw*shift1 needs 64 bits (|raw| up to ~2^31, shift1 up to 79); after
        // >>11 the result always fits i32 again (≤ ~8.3e7). Ref C++ :732.
        const dp: i32 = @intCast((@as(i64, raw) * self.shift1) >> OUT_SHIFT);
        self.pr = tables.squash(dp);
        return self.pr;
    }

    /// `p1` — predict, returning the clamped pre-squash logit `dp ∈ [-2047, 2047]`
    /// and setting `pr`. Mutates only `pr`.
    pub inline fn p1(self: *Mixer1, tables: anytype, tx: []const i16) i32 {
        const base = self.cxt * self.n;
        var dp: i32 = undefined;
        if (comptime WIDE) {
            const raw64 = dotProduct32(tx, self.wx[base..], self.n);
            dp = @intCast((raw64 * self.shift1) >> W32_OUT);
        } else {
            const raw = dot_product129(tx, self.wx[base..], self.n);
            // 64-bit product as in pabove.
            dp = @intCast((@as(i64, raw) * self.shift1) >> OUT_SHIFT);
        }
        if (dp < -2047) {
            dp = -2047;
        } else if (dp > 2047) {
            dp = 2047;
        }
        self.pr = tables.squash(dp);
        return dp;
    }

    /// `update(y)` — scale the prediction error and train the active context on the
    /// shared inputs `tx` (the same buffer the last `p`/`p1` predicted from).
    pub inline fn update(self: *Mixer1, y: i32, tx: []const i16) void {
        // err = ((y<<12) - pr) * uperr / 4  (signed division truncates toward 0)
        self.err = @divTrunc(((y << 12) - self.pr) *% self.uperr, 4);
        if (self.err > 32767) self.err = 32767;
        if (self.err < -32768) self.err = -32768;
        if (self.err >= -self.elim and self.err <= self.elim) self.err = 0;
        const base = self.cxt * self.n;
        if (comptime DIAG and !WIDE) self.diagScan(tx, base);
        if (comptime WIDE) {
            train32(tx, self.wx[base..], self.n, self.err);
        } else {
            train129(tx, self.wx[base..], self.n, self.err, &self.ds);
        }
    }

    /// -Dmxa-diag: re-derive, in scalar, the delta the SIMD kernel is about to
    /// apply and compare it to the exact real-valued gradient step. Runs BEFORE
    /// train129 so `wx` still holds the pre-update row. Dead at default.
    fn diagScan(self: *Mixer1, tx: []const i16, base: usize) void {
        self.d.calls += 1;
        if (self.err == 0) {
            self.d.err0 += 1;
            return;
        }
        if (self.d.calls % DIAG_STRIDE != 0) return;
        const scale: f64 = @floatFromInt(@as(i64, 1) << TRAIN_SHIFT);
        var i: usize = 0;
        while (i < self.n) : (i += 1) {
            const t: i32 = tx[i];
            // `tv +| tv` (saturating i16 double)
            var tt: i32 = t + t;
            if (tt > 32767) tt = 32767;
            if (tt < -32768) tt = -32768;
            // mulhi-with-shift, truncated back to i16 exactly as the kernel does
            const shifted: i32 = (tt * self.err) >> TRAIN_SHIFT;
            const as16: i16 = @truncate(shifted);
            var acc: i32 = @as(i32, as16) + 1;
            if (acc > 32767) acc = 32767;
            const applied: i32 = acc >> 1; // arithmetic
            const exact: f64 = (@as(f64, @floatFromInt(t)) * @as(f64, @floatFromInt(self.err))) / scale;
            self.d.w += 1;
            if (applied == 0) self.d.dead += 1;
            self.d.exact += @abs(exact);
            self.d.appl += @abs(@as(f64, @floatFromInt(applied)));
            self.d.lost += @abs(exact - @as(f64, @floatFromInt(applied)));
            const trueW: i32 = @as(i32, @intCast(self.wx[base + i])) + @as(i32, WBIAS);
            if (trueW > 31743 or trueW < -31744) self.d.sat += 1;
        }
    }

    /// -Dmxa-diag: one TSV line per mixer instance at teardown. Also scans the
    /// bank for the live |true weight| distribution (the headroom question).
    pub fn diagDump(self: *const Mixer1, name: []const u8, idx: usize) void {
        if (comptime !DIAG or WIDE) return;
        var nz: u64 = 0;
        var maxw: i32 = 0;
        var hist = [_]u64{0} ** 17; // |true w − WBIAS| by power of two
        const live = @min(self.wx.len, self.n * self.m);
        for (self.wx[0..live]) |sw| {
            if (sw == 0) continue; // untouched row (stored-domain zero == init)
            nz += 1;
            const tw: i32 = @as(i32, sw) + @as(i32, WBIAS);
            const aw: i32 = if (tw < 0) -tw else tw;
            if (aw > maxw) maxw = aw;
            const dev: i32 = if (sw < 0) -@as(i32, sw) else @as(i32, sw);
            var b: usize = 0;
            var v: i32 = dev;
            while (v > 1 and b < 16) : (b += 1) v >>= 1;
            hist[b] += 1;
        }
        const wf: f64 = if (self.d.w == 0) 1.0 else @as(f64, @floatFromInt(self.d.w));
        std.debug.print("MXADIAG\t{s}{d}\tn={d}\tm={d}\tshift1={d}\telim={d}\tuperr={d}\t", .{ name, idx, self.n, self.m, self.shift1, self.elim, self.uperr });
        std.debug.print("calls={d}\terr0={d}\tw={d}\tdead={d}\tdeadfrac={d:.5}\t", .{ self.d.calls, self.d.err0, self.d.w, self.d.dead, @as(f64, @floatFromInt(self.d.dead)) / wf });
        std.debug.print("exact={e:.4}\tappl={e:.4}\tlost={e:.4}\tlostfrac={d:.5}\tmeanexact={d:.5}\t", .{ self.d.exact, self.d.appl, self.d.lost, if (self.d.exact == 0) 0.0 else self.d.lost / self.d.exact, self.d.exact / wf });
        std.debug.print("nz={d}\tmaxw={d}\tsat={d}\thist=", .{ nz, maxw, self.d.sat });
        for (hist) |h| std.debug.print("{d},", .{h});
        std.debug.print("\n", .{});
    }
};

/// fxcmv1.cpp Mix (543-576) — the mmmO 2-input chain mixer (weights scaled 24 bits).
pub const Mix = struct {
    wt: []i32 = &.{},
    x1: i32 = 0,
    x2: i32 = 0,
    cxt: usize = 0,
    pr: i32 = 0,
    alloc: Allocator,

    pub fn new(a: Allocator, n: usize) Mix {
        const wt = a.alloc(i32, n * 2) catch unreachable;
        @memset(wt, 1 << 23);
        return .{ .wt = wt, .alloc = a };
    }

    pub fn deinit(self: *Mix) void {
        if (self.wt.len != 0) self.alloc.free(self.wt);
        self.wt = &.{};
    }

    pub fn pp(self: *Mix, p1: i32, p2: i32, cx: usize) i32 {
        self.cxt = cx * 2;
        self.x1 = p1;
        self.x2 = p2;
        self.pr = (p1 *% (self.wt[self.cxt] >> 16) +%
            p2 *% (self.wt[self.cxt + 1] >> 16) +% 128) >> 8;
        return self.pr;
    }

    /// `tables` is duck-typed (`.squash(i32) i32`).
    pub fn update(self: *Mix, y: i32, tables: anytype) void {
        var err = (y << 12) - tables.squash(self.pr);
        if ((self.wt[self.cxt] & 3) < 3) {
            self.wt[self.cxt] +%= 1;
            err *%= (4 - (self.wt[self.cxt] & 3));
        }
        err = (err + 8) >> 4;
        self.wt[self.cxt] = self.wt[self.cxt] +% ((self.x1 *% err) & @as(i32, -4));
        self.wt[self.cxt + 1] = self.wt[self.cxt + 1] +% (self.x2 *% err);
    }
};

// ===================================================================== tests

const testing = std.testing;

const Xs = struct {
    s: u32,
    fn next(self: *Xs) u32 {
        self.s ^= @as(u32, @truncate(@as(u64, self.s) << 13));
        self.s ^= self.s >> 17;
        self.s ^= @as(u32, @truncate(@as(u64, self.s) << 5));
        return self.s;
    }
};

// ---- mixer_ref.rs goldens (dot_product/train against compiled C++) ----------
const DP_REF = [_]i32{ 7754, -7994, 13540, -11472, -45024, -2045, -28580, -32840, 27067, -29190, 8875, 33627, 41234, -17410, -21328, -18525, -48102, 47677, -26809, -109110, 14420, 22137, 22123, -42150, 54033, 41089, -71707, 17522, -15161, 128855, -4741, -55122, 96019, -28313, 4881, -191176, 214110, -172652 };
const W_REF = [_]i16{ 132, 74, 203, 18, 120, -89, 137, 593, -547, -591, -402, 319, 408, -281, -735, -855, 328, 474, -155, 122, -489, 207, 749, -862, 95, -523, 65, 793, -548, -233, -729, 988, 970, -813, 249, -716, 970, 856, -897, 248, -789, 949, 896, -955, 227, -402, 591, 587, -1276, 206, -382, 636, 609, -1221, 234, -755, 991, 226, -1285, 788, -722, 212, -369, -1001, 124, -118, 310, -168, -793, 93, -592, 569, -83, -1145, -4, -470, 734, 243, -1532, 397, -23, 1163, 457, -1975, 834, -23, 1158, 458, -1967, 831, -48, 1587, -466, -2731, 1697, -269, 1632, -721, -2840, 1938, 141, 2162, -804, -3038, 2204, -549, 1814, -726, -2801, 2860, -258, 2404, -674, -1859, 2477, -227, 2410, -654, -1852, 2476, 1, 2470, -788, -1795, 2620, -204, 3107, -352, -869, 2322, -598, 3432, -475, -650, 1869, -645, 3151, -1207, -198, 1172, 13, 3679, -1002, 487, 1468, 499, 4431, -1668, -226, 1734, -65, 3408, -1894, 261, 3067, -1268, 3523, -1551, 1904, 4143, -2463, 3684, -1055, 1909, 5979, -538, 153, 1708, -1174, 7520, 1150, -718, 2194, -2032, 7303, 4834, -24, 2166, 187, 8231, 2842, -131, 586, 437, 7483, 3194, 236, -452, 878, 4322 };

test "dot_product and train match cpp" {
    if (RADIX != 0) return error.SkipZigTest; // goldens are the stock radix
    const N: usize = 48;
    var t = [_]i16{0} ** 64;
    var w = [_]i16{129} ** 64;
    var x = Xs{ .s = 0xBEEF };
    var dp_got = std.ArrayList(i32){};
    defer dp_got.deinit(testing.allocator);
    var w_got = std.ArrayList(i16){};
    defer w_got.deinit(testing.allocator);
    var step: usize = 0;
    while (step < 200) : (step += 1) {
        for (0..N) |k| {
            t[k] = @intCast(@as(i32, @intCast(x.next() % 4094)) - 2047);
        }
        const dp = dot_product(&t, &w, N);
        const err: i32 = @as(i32, @intCast(x.next() % 65535)) - 32768;
        train(&t, &w, N, err);
        if (step < 30 or step % 20 == 0) {
            try dp_got.append(testing.allocator, dp);
            for (0..5) |k| try w_got.append(testing.allocator, w[k]);
        }
    }
    try testing.expectEqualSlices(i32, &DP_REF, dp_got.items);
    try testing.expectEqualSlices(i16, &W_REF, w_got.items);
}

test "bias-offset kernels are an exact bijection of the plain kernels" {
    const N: usize = 48;
    const B: i16 = WBIAS;
    var t = [_]i16{0} ** 64;
    var w = [_]i16{WBIAS} ** 64; // true domain
    var ws = [_]i16{0} ** 64; // stored domain: w -% WBIAS
    var x = Xs{ .s = 0xBEEF };
    var step: usize = 0;
    while (step < 200) : (step += 1) {
        for (0..N) |k| {
            t[k] = @intCast(@as(i32, @intCast(x.next() % 4094)) - 2047);
        }
        try testing.expectEqual(dot_product(&t, &w, N), dot_product129(&t, &ws, N));
        const err: i32 = @as(i32, @intCast(x.next() % 65535)) - 32768;
        train(&t, &w, N, err);
        var dummy_ds: DitherState = if (DITHER) @splat(1) else {};
        train129(&t, &ws, N, err, &dummy_ds);
        for (w, ws) |wv, sv| try testing.expectEqual(wv, sv +% B);
    }
}

// ---- mixer1_ref.rs goldens (Mixer1 against compiled C++ oracle) --------------
const M1_N: usize = 32;
const M1_M: usize = 4;
const SHIFT1: i32 = 237;
const ELIM: i32 = 8;
const UPERR: i32 = 69;
const NSTEPS: usize = 500;
const SAMP_IDX = [_]usize{ 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 50, 75, 100, 125, 150, 175, 200, 225, 250, 275, 300, 325, 350, 375, 400, 425, 450, 475 };
const PR_REF = [_]i32{ 3285, 1829, 75, 47, 1001, 106, 78, 2884, 4095, 1, 809, 4024, 4095, 2164, 1, 224, 1, 4095, 1, 4095, 229, 1, 4095, 2, 1, 4087, 4095, 1, 1, 935, 1, 1, 4095, 143, 1, 4081, 4095, 4095, 1, 3914, 2813, 4095, 4056, 1821, 4095, 3, 4095, 3878, 3370, 1, 4095, 1, 4095, 3705, 3901, 1, 4095, 1 };
const DP1_REF = [_]i32{ 358, -55, -1021, -1140, -289, -930, -1009, 222, 2047, -2047, -359, 1031, 2047, 29, -2047, -729, -2047, 2047, -2047, 2047, -723, -2047, 2047, -1957, -2047, 1564, 2047, -2047, -2047, -312, -2047, -2047, 2047, -849, -2047, 1436, 2047, 2047, -2047, 785, 201, 2047, 1180, -57, 2047, -1845, 2047, 737, 393, -2047, 2047, -2047, 2047, 576, 767, -2047, 2047, -2047 };
const ERR_REF = [_]i32{ 13989, -31550, 32767, 32767, -17267, 32767, 32767, -32768, -32768, -17, 32767, -32768, -32768, 32767, 32767, 32767, -17, -32768, -17, -32768, -3950, 32767, -32768, -34, -17, 155, 17, -17, 32767, -16128, -17, -17, 17, -2466, -17, 258, -32768, 17, -17, -32768, 22131, 17, 690, 32767, -32768, 32767, 17, -32768, 12523, -17, 17, -17, 17, -32768, -32768, 32767, 17, 32767 };
const CXT_REF = [_]usize{ 2, 0, 2, 2, 1, 1, 2, 3, 2, 1, 3, 3, 2, 0, 3, 2, 0, 3, 2, 1, 1, 2, 2, 2, 2, 3, 2, 3, 2, 2, 0, 3, 3, 3, 1, 3, 1, 1, 3, 3, 3, 1, 3, 0, 0, 3, 3, 1, 0, 3, 0, 2, 0, 0, 0, 3, 0, 3 };
const WX_CS: u64 = 11621237012126820838;
const WX_CNT: usize = 160;
const WX_SAMP = [_]i16{ -970, 1521, -1648, 3868, -220, 1433, 1644, -1762, -3220, 1984, -1597, -289, 2541, 326, 2472, 65, 1492, -1360, 2950, 101, 833, 610, 1208, -1552, 1281, 4671, 1310, -1291, 572, 335, 4457, -2892 };

// squash from tables.rs (fxcmv1.cpp squashc), used to verify p/p1 goldens.
fn squashc(d: i32) i16 {
    if (d < -2047) return 1;
    if (d > 2047) return 4095;
    const p1: f32 = @floatCast(1.0 / (1.0 + @exp(-@as(f64, @floatFromInt(d)) / 256.0)));
    const p2: f32 = @floatCast(@as(f64, p1) * 4096.0);
    var pi: i64 = @intFromFloat(@round(p2));
    if (pi > 4095) pi = 4095;
    if (pi < 1) pi = 1;
    return @intCast(pi);
}

const TestTables = struct {
    sqt: [4095]i16,
    fn make() TestTables {
        var tt: TestTables = .{ .sqt = undefined };
        var d: i32 = -2047;
        while (d <= 2047) : (d += 1) tt.sqt[@intCast(d + 2047)] = squashc(d);
        return tt;
    }
    fn squash(self: *const TestTables, d: i32) i32 {
        if (d < -2047) return 1;
        if (d > 2047) return 4095;
        return self.sqt[@intCast(d + 2047)];
    }
};

test "mixer1 matches cpp" {
    if (RADIX != 0) return error.SkipZigTest; // goldens are the stock radix
    const tables = TestTables.make();
    var m = Mixer1.new(testing.allocator);
    defer m.deinit();
    m.init(M1_M, SHIFT1, ELIM, UPERR);
    m.set_tx_wx(M1_N);
    var tx = [_]i16{0} ** round16(M1_N);

    var x = Xs{ .s = 0x1234567 };

    var samp: usize = 0;
    var pr_got = std.ArrayList(i32){};
    defer pr_got.deinit(testing.allocator);
    var dp1_got = std.ArrayList(i32){};
    defer dp1_got.deinit(testing.allocator);
    var err_got = std.ArrayList(i32){};
    defer err_got.deinit(testing.allocator);
    var cxt_got = std.ArrayList(usize){};
    defer cxt_got.deinit(testing.allocator);

    var step: usize = 0;
    while (step < NSTEPS) : (step += 1) {
        for (0..M1_N) |k| {
            tx[k] = @intCast(@as(i32, @intCast(x.next() % 4095)) - 2047);
        }
        m.cxt = @intCast(x.next() % @as(u32, M1_M));
        const dp1 = m.p1(&tables, &tx);
        const pr = m.p(&tables, &tx);
        const y: i32 = @intCast(x.next() & 1);
        m.update(y, &tx);
        if (step < 40 or step % 25 == 0) {
            try testing.expectEqual(SAMP_IDX[samp], step);
            try pr_got.append(testing.allocator, pr);
            try dp1_got.append(testing.allocator, dp1);
            try err_got.append(testing.allocator, m.err);
            try cxt_got.append(testing.allocator, m.cxt);
            samp += 1;
        }
    }

    try testing.expectEqualSlices(i32, &PR_REF, pr_got.items);
    try testing.expectEqualSlices(i32, &DP1_REF, dp1_got.items);
    try testing.expectEqualSlices(i32, &ERR_REF, err_got.items);
    try testing.expectEqualSlices(usize, &CXT_REF, cxt_got.items);

    // whole-bank weight checksum (cs = cs*1000003 + (u16)w), incl. 32-short tail.
    // wx is stored bias-offset (`w -% 129`); the golden is in the true domain.
    // The C++ tail is true-0 and unreachable by the kernels — ours stays
    // stored-0 (asserted), and the checksum feeds the golden's literal 0.
    try testing.expectEqual(WX_CNT, m.wx.len);
    var cs: u64 = 0;
    for (m.wx, 0..) |sw, idx| {
        var tw: i16 = undefined;
        if (idx < M1_M * M1_N) {
            tw = sw +% 129;
        } else {
            try testing.expectEqual(@as(i16, 0), sw);
            tw = 0;
        }
        cs = cs *% 1000003 +% @as(u64, @as(u16, @bitCast(tw)));
    }
    try testing.expectEqual(WX_CS, cs);

    // sampled first-8 weights of each context after the full run.
    var wx_samp = std.ArrayList(i16){};
    defer wx_samp.deinit(testing.allocator);
    for (0..M1_M) |c| {
        for (0..8) |k| try wx_samp.append(testing.allocator, m.wx[c * M1_N + k] +% 129);
    }
    try testing.expectEqualSlices(i16, &WX_SAMP, wx_samp.items);
}
