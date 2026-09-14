//! HERO hidden-state ensemble-residual head — RUNTIME implementation.
//!
//! FROZEN mechanism (zero retuning latitude):
//!   (+ its result doc; scorer experiments/hero_hidden_residual_head.cpp,
//!   sha256 ea9b169f898f837d6e00f43c38d333cb8a0d7466a41f27d0c4d92c553f22644d).
//! Re-priced at the filler tier by
//! Rate 0.0003869 b/pb on the 60-80MB tranche = 28,398 B e9 gross,
//! decaying ~4%/tranche. The HERO reopen clause was formally opened by
//!  (cache KILLED on encode wall;
//! HERO is the memory-light sibling on the same h[t-1] pool: one 255-row
//! table, ~dim MAC/bit, no retrieval, no window, no LSH).
//!
//! Mechanism, verbatim from the scorer:
//!   - 256-row table (row 0 allocated, never used); rows 1..255 are the
//!     nonterminal MSB-first binary-prefix nodes of the target byte:
//!     node=1 at byte start, node = 2*node + bit after each coded bit, so at
//!     plane b the node is (1<<b) | already-decoded-prefix.
//!   - Row = intercept + one f64 weight per live layer-0 LSTM hidden coord
//!     h[t-1] (the SAME tap the episodic cache keyed on; cell state EXCLUDED).
//!   - Per bit:  raw = a_c + dot(w_c, h);  r = clamp(raw, -8, 8);
//!               p = sigmoid(logit(p0) + r)   with p0 = coder numerator/65536.
//!   - Row-normalized AdaGrad after each scored bit (accumulator update
//!     precedes the step; saturated bits have e=0 but keep the L2 decay):
//!         e      = (p - y) if raw strictly inside (-8, 8) else 0
//!         G_c   += e*e*(1 + ||h||^2)
//!         step   = eta / sqrt(G_c)
//!         a_c   -= step*(e + 1e-7*a_c)
//!         w_c[j]-= step*(e*h[j] + 1e-7*w_c[j])
//!     eta = 0.03 (train-only selected at 20m, 40m AND 80m — stable, so the
//!     runtime carries the single selected rate; the {0.003,0.01,0.03} grid
//!     and the intercept-only control were shadow diagnostics, not shipped).
//!   - All state zero-init (accumulators 1e-3); no model file, no checkpoint.
//!
//! Integration: a final-logit residual on the incumbent final SSE numerator,
//! structurally the same tap the hidden-cache runtime used (its plumbing is
//! reused verbatim: beginByte / mix / observe around the five coder loops).
//! The head NEVER feeds back into the predictor — pred.predict/perceive
//! are untouched, so h[t-1] and the incumbent numerators evolve bit-identically
//! stock vs engaged, and the runtime head sees exactly the shadow scorer's
//! inputs (p0 = the stock coder numerator, h = the stock hidden trajectory).
//!
//! Decoder symmetry: h[t-1] is a pure function of already-decoded history and
//! the row update for a bit runs only after that bit is resolved (the runtime
//! analog of the cache preflight's query-before-insert ordering), so encoder
//! and decoder run the identical table trajectory — a mirror-state model, no
//! side channel.
//!
//! Numerics: f64 state + strict (non-fast-math) f64 ops in the scorer's exact
//! operation order — the shadow that produced the 28,398 B evidence is f64;
//! the preflight's "runtime feasibility ceiling" sketched f32 state (181,560
//! B) but the f64 table is ~0.35-0.38 MB, immaterial vs the RSS slack, and
//! keeps the runtime on the measured trajectory. sigmoid/logit are the
//! two-branch/clamped forms proven cross-box deterministic by the cache
//! runtime (logit clamp 1e-12 per this scorer; it never binds on u16-derived
//! p0).
//!
//! Frozen evidence surface = cells-176 (the traced h[t-1] is 176-dim). The
//! head itself is dimension-generic — the table is sized from the live hidden
//! width at create— because unlike the cache there is no pinned-key
//! identity gate (rows learn online from zero on whatever surface arms it).
//! A non-176 ship surface engages mechanically but carries an
//! evidence-transfer caveat its preflight must own.

const std = @import("std");
const build_options = @import("build_options");

pub const ON: bool = build_options.hero;

const kRows: usize = 256; // row 0 unused; binary-prefix nodes 1..255 (scorer kRows)
// kEta: 80m held-forward re-selected.
// Default 0.03 via build_options = the shipped constant, bit-identical; the
// promote value is 0.5 (-Dhero-eta=0.5). Train-only selection is structurally
// biased low here — never re-tune this from a train fold.
const kEta: f64 = @import("build_options").hero_eta;
const kAccumInit: f64 = 1e-3;
const kL2: f64 = 1e-7;
const kClamp: f64 = 8.0;

// Scorer sigmoid (two-branch numerically-stable form, f64).
fn sigmoid(x: f64) f64 {
    if (x >= 0) {
        const z = @exp(-x);
        return 1.0 / (1.0 + z);
    }
    const z = @exp(x);
    return z / (1.0 + z);
}

// Scorer logit: clamp [1e-12, 1-1e-12] then log(p/(1-p)). The clamp never
// binds for p0 = numerator/65536 with numerator in [1, 65535].
fn logitClamp(p_in: f64) f64 {
    const p = std.math.clamp(p_in, 1e-12, 1.0 - 1e-12);
    return @log(p / (1.0 - p));
}

pub const HeroHead = struct {
    alloc: std.mem.Allocator,
    dim: usize, // live h[t-1] width (frozen evidence surface: 176)

    // Adaptive state (zero-init; accumulators 1e-3). rows[c*(1+dim)] = a_c,
    // rows[c*(1+dim)+1 ..] = w_c — the scorer's row layout.
    rows: []f64, // kRows * (1 + dim)
    accum: []f64, // kRows

    // Per-byte scratch: f64 copy of h[t-1] + its norm-squared, both computed
    // once at byte start (frozen optimizer: H2 computed once per byte; h is
    // captured before any bit of the byte so mid-byte LSTM updates cannot
    // touch it — matches the trace producer's once-per-byte record).
    h: []f64,
    h2: f64,

    // Per-bit staging (mix -> observe). p is the CONTINUOUS head output — the
    // scorer's update uses it, not the discretized coder numerator.
    last_node: usize,
    last_p: f64,
    last_live: bool, // raw strictly inside (-kClamp, kClamp)

    pub fn create(alloc: std.mem.Allocator, dim: usize) !*HeroHead {
        std.debug.assert(dim > 0);
        const self = try alloc.create(HeroHead);
        errdefer alloc.destroy(self);
        self.alloc = alloc;
        self.dim = dim;
        self.rows = try alloc.alloc(f64, kRows * (1 + dim));
        errdefer alloc.free(self.rows);
        self.accum = try alloc.alloc(f64, kRows);
        errdefer alloc.free(self.accum);
        self.h = try alloc.alloc(f64, dim);
        @memset(self.rows, 0);
        for (self.accum) |*g| g.* = kAccumInit;
        @memset(self.h, 0);
        self.h2 = 0;
        self.last_node = 0;
        self.last_p = 0.5;
        self.last_live = false;
        return self;
    }

    pub fn destroy(self: *HeroHead) void {
        self.alloc.free(self.h);
        self.alloc.free(self.accum);
        self.alloc.free(self.rows);
        self.alloc.destroy(self);
    }

    /// Adaptive-state bytes (RAM accounting; excludes the struct header).
    pub fn stateBytes(self: *const HeroHead) usize {
        return (self.rows.len + self.accum.len + self.h.len) * @sizeOf(f64);
    }

    inline fn row(self: *HeroHead, node: usize) []f64 {
        const base = node * (1 + self.dim);
        return self.rows[base .. base + 1 + self.dim];
    }

    /// Capture h[t-1] and compute ||h||^2 once. Call at the top of every
    /// target byte, before any of its bits (both coder directions).
    pub fn beginByte(self: *HeroHead, hidden: []const f32) void {
        std.debug.assert(hidden.len == self.dim);
        var h2: f64 = 0;
        var j: usize = 0;
        while (j < self.dim) : (j += 1) {
            const v: f64 = hidden[j];
            self.h[j] = v;
            h2 += v * v; // scorer accumulates alongside the copy, index order
        }
        self.h2 = h2;
    }

    /// Apply the HERO residual to the incumbent final numerator (base_num =
    /// discretize(pred.predict), in [1, 65535]) for the bit at `plane`
    /// (0..7, MSB first) given the already-decoded `prefix` bits of the byte.
    /// Returns the final coder numerator in [1, 65535]; stages the update.
    pub fn mix(self: *HeroHead, base_num: u32, plane: usize, prefix: u32) u32 {
        const node: usize = (@as(usize, 1) << @intCast(plane)) | @as(usize, prefix);
        std.debug.assert(node >= 1 and node < kRows);
        const r_ = self.row(node);
        // scorer rawResidual: r = row[0]; for j: r += row[j+1]*h[j]
        var raw: f64 = r_[0];
        var j: usize = 0;
        while (j < self.dim) : (j += 1) raw += r_[1 + j] * self.h[j];
        const live = raw > -kClamp and raw < kClamp;
        const resid = std.math.clamp(raw, -kClamp, kClamp);
        const base_p = @as(f64, @floatFromInt(base_num)) / 65536.0;
        const p_final = sigmoid(logitClamp(base_p) + resid);
        self.last_node = node;
        self.last_p = p_final;
        self.last_live = live;
        // Re-discretization exactly as the hidden-cache runtime (Discretize
        // form 1 + 65534*p, f32, truncating) — proven lossless + cross-box
        // deterministic there.
        const d: f32 = 1.0 + 65534.0 * @as(f32, @floatCast(p_final));
        var num: u32 = @intFromFloat(d);
        if (num < 1) num = 1;
        if (num > 65535) num = 65535;
        return num;
    }

    /// Frozen row-normalized AdaGrad update for the just-coded bit. Call
    /// after the bit is resolved (encode: known input bit; decode: the
    /// arithmetic-decoded bit), BEFORE pred.perceive — prediction always
    /// precedes the target-driven mutation, on both directions identically.
    pub fn observe(self: *HeroHead, bit: i32) void {
        const y: f64 = @floatFromInt(bit);
        const e: f64 = if (self.last_live) self.last_p - y else 0.0;
        self.accum[self.last_node] += e * e * (1.0 + self.h2);
        const step: f64 = kEta / @sqrt(self.accum[self.last_node]);
        const r_ = self.row(self.last_node);
        r_[0] -= step * (e + kL2 * r_[0]);
        var j: usize = 0;
        while (j < self.dim) : (j += 1) {
            r_[1 + j] -= step * (e * self.h[j] + kL2 * r_[1 + j]);
        }
    }
};

// ---------------------------------------------------------------------------
// Focused unit tests (run under `zig build test`). These pin the pure head
// math — node mapping, the frozen update equations, saturation semantics,
// numerator range, determinism — independently of the LSTM/coder. The
// codec-level validation (stock bit-identity, engaged lossless roundtrip)
// lives in the verify_change ritual, not here.
// ---------------------------------------------------------------------------
const testing = std.testing;

test "hero: encode-side (plane,prefix) node equals the decode-side tree walk" {
    var b: u32 = 0;
    while (b < 256) : (b += 1) {
        var node: u32 = 1;
        var j: i32 = 7;
        while (j >= 0) : (j -= 1) {
            const bit: u32 = (b >> @intCast(j)) & 1;
            const plane: u5 = @intCast(7 - j);
            const prefix: u32 = b >> @intCast(j + 1);
            const enc_node: u32 = (@as(u32, 1) << plane) | prefix;
            try testing.expectEqual(node, enc_node);
            try testing.expect(node >= 1 and node < 256); // scorer row bounds
            node = 2 * node + bit;
        }
        try testing.expectEqual(256 + b, node);
    }
}

test "hero: fresh head is residual-zero; one bit reproduces the frozen update exactly" {
    const a = testing.allocator;
    const dim: usize = 3;
    var head = try HeroHead.create(a, dim);
    defer head.destroy();

    const hvals = [dim]f32{ 0.5, -0.25, 1.0 };
    head.beginByte(hvals[0..]);

    // plane 1, prefix 1 -> node 3. Fresh head: raw=0 -> p = p0 exactly; at
    // p0 = 0.5 the re-discretization is exact (1 + 65534*0.5 = 32768).
    const num = head.mix(32768, 1, 1);
    try testing.expectEqual(@as(u32, 32768), num);

    head.observe(1);

    // Replicate the frozen equations in the scorer's operation order.
    var h: [dim]f64 = undefined;
    var h2: f64 = 0;
    for (hvals, 0..) |v, j| {
        h[j] = v;
        h2 += h[j] * h[j];
    }
    const p: f64 = 0.5; // sigmoid(logit(0.5) + 0) is exact
    const e: f64 = p - 1.0;
    var g: f64 = kAccumInit;
    g += e * e * (1.0 + h2);
    const step: f64 = kEta / @sqrt(g);
    try testing.expectEqual(g, head.accum[3]);
    try testing.expectEqual(0.0 - step * (e + kL2 * 0.0), head.row(3)[0]);
    for (0..dim) |j| {
        try testing.expectEqual(0.0 - step * (e * h[j] + kL2 * 0.0), head.row(3)[1 + j]);
    }
    // Same node, same byte: the moved intercept/weights now push p above p0.
    const num2 = head.mix(32768, 1, 1);
    try testing.expect(num2 > 32768);
}

test "hero: saturated residual has e=0, keeps L2 decay, accumulator untouched" {
    const a = testing.allocator;
    const dim: usize = 2;
    var head = try HeroHead.create(a, dim);
    defer head.destroy();

    head.row(1)[0] = 9.0; // force |raw| >= kClamp at node 1
    const zeros = [dim]f32{ 0, 0 };
    head.beginByte(zeros[0..]);
    const num = head.mix(32768, 0, 0); // node 1, raw = 9 -> r = +8 clamped
    try testing.expect(num >= 1 and num <= 65535);
    try testing.expect(num > 60000); // sigmoid(0 + 8) ~ 0.99966

    head.observe(0);
    // e = 0: G unchanged; intercept decays by step*kL2*a; weights stay 0.
    try testing.expectEqual(kAccumInit, head.accum[1]);
    const step: f64 = kEta / @sqrt(kAccumInit);
    try testing.expectEqual(9.0 - step * (0.0 + kL2 * 9.0), head.row(1)[0]);
    try testing.expectEqual(@as(f64, 0), head.row(1)[1]);
    try testing.expectEqual(@as(f64, 0), head.row(1)[2]);

    // Extreme numerators stay in coder range on a hot row (both clamp sides).
    head.row(1)[0] = 9.0;
    try testing.expect(head.mix(65535, 0, 0) <= 65535);
    head.row(1)[0] = -9.0;
    try testing.expect(head.mix(1, 0, 0) >= 1);
}

test "hero: deterministic — identical streams give identical numerators and state" {
    const a = testing.allocator;
    const dim: usize = 8;
    var h1 = try HeroHead.create(a, dim);
    defer h1.destroy();
    var h2_ = try HeroHead.create(a, dim);
    defer h2_.destroy();

    var prng = std.Random.DefaultPrng.init(923);
    const rnd = prng.random();
    var moved = false;
    var t: usize = 0;
    while (t < 500) : (t += 1) {
        var hv: [dim]f32 = undefined;
        for (&hv) |*v| v.* = (rnd.float(f32) - 0.5) * 2.0;
        const byte: u32 = rnd.int(u8);
        h1.beginByte(hv[0..]);
        h2_.beginByte(hv[0..]);
        var j: i32 = 7;
        while (j >= 0) : (j -= 1) {
            const bit: i32 = @intCast((byte >> @intCast(j)) & 1);
            const base: u32 = 1 + rnd.uintLessThan(u32, 65535);
            const plane: usize = @intCast(7 - j);
            const prefix: u32 = byte >> @intCast(j + 1);
            const n1 = h1.mix(base, plane, prefix);
            const n2 = h2_.mix(base, plane, prefix);
            try testing.expectEqual(n1, n2);
            if (n1 != base) moved = true;
            h1.observe(bit);
            h2_.observe(bit);
        }
    }
    try testing.expect(moved); // the head engaged (residual moved numerators)
    try testing.expectEqualSlices(f64, h1.rows, h2_.rows);
    try testing.expectEqualSlices(f64, h1.accum, h2_.accum);
}
