//! mixer1_rls.zig — second-order online learning for the LAYER-1 mixer (`rls8`).
//!
//! The PAQ/cmix lineage trains every mixer by plain SGD because a second-order
//! update costs O(d^2) and the outer mixer is d = 590. The engine's FINAL mixing
//! stage is not 590 wide: `predictor_lex.zig` builds `mixer1` as **d = 25 with a
//! SINGLE weight row** (`zero_context`). At d = 25 the exact curvature of the
//! coding loss is computable online — ~2,300 flops/bit against a per-bit budget
//! of ~120,600 cycles at e9.
//!
//! Method: online IRLS / extended-Kalman recursion with exponential forgetting.
//! Maintain  P_t ~= ( sum_s lambda^{t-s} h_s x_s x_s^T + (1/p0) I )^{-1},
//! h_s = p_s (1 - p_s) = the Bernoulli curvature. Rank-1 (Sherman-Morrison):
//!     g     = P x
//!     u     = x . g
//!     k     = g / (lambda + h*u)          (= P_t x)
//!     w    += k * (y - p)                 (exact Newton step on the coding loss)
//!     P     = (P - h * k g^T) / lambda    (symmetrised: upper triangle mirrored)
//!
//! ARITHMETIC IS STRICT f64 IN A FIXED ORDER (`@setFloatMode(.strict)`, no
//! `@mulAdd`) so the recursion is bit-reproducible across compilers/ISAs by
//! IEEE-754 alone — which matters because the judging committee REBUILDS comp9
//! from source. d = 25 is what makes strict f64 affordable; the same recursion
//! at d = 590 would be ~1e6 f64 ops/bit.
//!
//! Modes exist so the claim can be attributed. FULL keeps the off-diagonal
//! curvature (the correlations between experts); DIAG keeps only the diagonal
//! (the class family `c6-mixer-preconditioner` already KILLED at layer-0);
//! SCALAR keeps only the trace (an automatic step-size schedule, no anisotropy);
//! SHUF is the pessimum control (correct step-size distribution, destroyed
//! input<->curvature correspondence); SGD64 is the precision control (the
//! engine's own SGD rule executed in strict f64 through this scaffolding).
//! On the real d = 25 object 100 % of the gain is OFF-DIAGONAL: diag and scalar
//! both LOSE (+0.169 % / +0.167 %) where the full matrix wins -0.1385 % pre-SSE.
//!
//! Ported from a standalone RLS implementation together with its tranche
//! collection.
//!
//! NOTHING OUTSIDE THE BINARY: every parameter arrives as a comptime `-D`
//! option. The lab tree drove these from `ZMIX_RLS_*` environment variables;
//! that is the defect class the operator ruling names, so the env
//! path is deliberately NOT ported.

const std = @import("std");

pub const D: usize = 25;

pub const Mode = enum(u32) {
    off = 0,
    full = 1,
    diag = 2,
    scalar = 3,
    shuf = 4,
    /// PRECISION CONTROL: the engine's own SGD rule, but executed in strict f64
    /// through this same scaffolding. Isolates "f64 instead of f32" from
    /// "second-order curvature" in an in-engine A/B.
    sgd64 = 5,
};

pub const Params = struct {
    lambda: f64 = 0.999999,
    p0: f64 = 0.1,
    h_min: f64 = 1.0e-4,
    /// trust region on the weight step (in stretched-logit units); <=0 disables.
    step_cap: f64 = 0.0,
    /// sgd64 control: base learning rate (engine mixer1 = mix_l1_rate * mix_lr_scale).
    sgd_lr: f64 = 0.00021,
    /// sgd64 control: terminal decay rung (engine -Dmixer-decay-floor).
    sgd_floor: f64 = 0.14,
    mode: Mode = .full,
    /// PSD/finiteness guard. Default FALSE so the engaged arithmetic reproduces
    /// the lab binary exactly. lambda = 0.9999 is measured to NaN on
    /// the dense e8_1m stream, so a SHIPPED configuration owes this guard on
    /// (or a UD/Bierman square-root form) plus a proof it never fires at e9.
    guard: bool = false,
};

pub const Rls = struct {
    w: [D]f64 = [_]f64{0.0} ** D,
    p: [D * D]f64 = [_]f64{0.0} ** (D * D),
    g: [D]f64 = [_]f64{0.0} ** D,
    k: [D]f64 = [_]f64{0.0} ** D,
    x: [D]f64 = [_]f64{0.0} ** D,
    z: f64 = 0.0,
    prm: Params = .{},
    steps: u64 = 0,
    /// diagnostic: number of times the PSD guard fired (0 expected).
    resets: u64 = 0,
    perm: [D]u8 = undefined,

    pub fn init(prm: Params) Rls {
        var s = Rls{ .prm = prm };
        for (0..D) |i| s.p[i * D + i] = prm.p0;
        // deterministic permutation for the SHUF pessimum control
        for (0..D) |i| s.perm[i] = @intCast((i * 7 + 3) % D);
        return s;
    }

    fn reset(self: *Rls) void {
        @memset(self.p[0..], 0.0);
        for (0..D) |i| self.p[i * D + i] = self.prm.p0;
        self.resets += 1;
    }

    /// z = w . x, strict order, no FMA contraction.
    pub fn mix(self: *Rls, inputs: []const f32) f64 {
        @setFloatMode(.strict);
        for (0..D) |i| self.x[i] = @floatCast(inputs[i]);
        var acc: f64 = 0.0;
        for (0..D) |i| acc += self.w[i] * self.x[i];
        self.z = acc;
        return acc;
    }

    /// The weight row that produced the last `mix`, as f32 — the layer-0
    /// end-to-end gradient (-Dmixer-e2e) must differentiate through the head
    /// that actually coded the bit, not through the dormant SGD row.
    pub fn rowF32(self: *const Rls, out: *[D]f32) void {
        for (0..D) |i| out[i] = @floatCast(self.w[i]);
    }

    pub fn update(self: *Rls, bit: i32, p_pred: f64) void {
        @setFloatMode(.strict);
        const lam = self.prm.lambda;
        var h = p_pred * (1.0 - p_pred);
        if (h < self.prm.h_min) h = self.prm.h_min;
        const err = @as(f64, @floatFromInt(bit)) - p_pred;

        switch (self.prm.mode) {
            .off => return,
            .sgd64 => {
                // mixer_lex.zig step-thresholded decay on the global step count.
                var dec: f64 = self.prm.sgd_floor;
                if (self.steps < 25000000) {
                    dec = 0.3;
                    if (self.steps < 5000000) {
                        dec = 0.7;
                        if (self.steps < 1000000) dec = 1.0;
                    }
                }
                const upd = self.prm.sgd_lr * dec * err;
                for (0..D) |i| self.w[i] += upd * self.x[i];
            },
            .scalar => {
                // P = s I; the same recursion with x x^T replaced by (|x|^2/D) I.
                var xx: f64 = 0.0;
                for (0..D) |i| xx += self.x[i] * self.x[i];
                const s = self.p[0];
                const u = s * xx;
                const den = lam + h * u;
                const sc = s / den;
                for (0..D) |i| self.w[i] += sc * self.x[i] * err;
                const snew = (s - h * sc * s * xx / D) / lam;
                for (0..D) |i| self.p[i * D + i] = snew;
            },
            .diag => {
                // P = diag(d); rank-1 update restricted to the diagonal.
                var u: f64 = 0.0;
                for (0..D) |i| u += self.p[i * D + i] * self.x[i] * self.x[i];
                const den = lam + h * u;
                for (0..D) |i| {
                    const gi = self.p[i * D + i] * self.x[i];
                    const ki = gi / den;
                    self.w[i] += ki * err;
                    self.p[i * D + i] = (self.p[i * D + i] - h * ki * gi) / lam;
                }
            },
            .full, .shuf => {
                // g = P x   (symmetric P: row-major read is fine)
                for (0..D) |i| {
                    var acc: f64 = 0.0;
                    const row = self.p[i * D ..][0..D];
                    for (0..D) |j| acc += row[j] * self.x[j];
                    self.g[i] = acc;
                }
                var u: f64 = 0.0;
                for (0..D) |i| u += self.x[i] * self.g[i];
                const den = lam + h * u;
                if (self.prm.guard) {
                    // Riccati loses positive-definiteness => den <= 0 or non-finite.
                    // Restart from p0*I rather than propagate NaN into the coder.
                    if (!(den > 0.0) or !std.math.isFinite(den)) {
                        self.reset();
                        self.steps += 1;
                        return;
                    }
                }
                for (0..D) |i| self.k[i] = self.g[i] / den;

                var step: [D]f64 = undefined;
                if (self.prm.mode == .shuf) {
                    // pessimum control: same step magnitudes, permuted assignment
                    for (0..D) |i| step[i] = self.k[self.perm[i]] * err;
                } else {
                    for (0..D) |i| step[i] = self.k[i] * err;
                }
                if (self.prm.step_cap > 0.0) {
                    var mx: f64 = 0.0;
                    for (0..D) |i| {
                        const a = @abs(step[i]);
                        if (a > mx) mx = a;
                    }
                    if (mx > self.prm.step_cap) {
                        const sc = self.prm.step_cap / mx;
                        for (0..D) |i| step[i] *= sc;
                    }
                }
                for (0..D) |i| self.w[i] += step[i];

                // P = (P - h k g^T)/lambda, symmetrised via the upper triangle.
                const inv_lam = 1.0 / lam;
                for (0..D) |i| {
                    const hk = h * self.k[i];
                    for (i..D) |j| {
                        const v = (self.p[i * D + j] - hk * self.g[j]) * inv_lam;
                        self.p[i * D + j] = v;
                        self.p[j * D + i] = v;
                    }
                }
                if (self.prm.guard) {
                    var t: f64 = 0.0;
                    for (0..D) |i| t += self.p[i * D + i];
                    if (!std.math.isFinite(t) or t <= 0.0) self.reset();
                }
            },
        }
        self.steps += 1;
    }

    /// diagnostic: trace(P) — grows without bound if forgetting outruns information.
    pub fn trace(self: *const Rls) f64 {
        var t: f64 = 0.0;
        for (0..D) |i| t += self.p[i * D + i];
        return t;
    }
};

test "rls: off mode is inert" {
    var r = Rls.init(.{ .mode = .off });
    const x = [_]f32{0.5} ** D;
    _ = r.mix(&x);
    r.update(1, 0.5);
    for (r.w) |wi| try std.testing.expectEqual(@as(f64, 0.0), wi);
}

test "rls: full mode moves weights toward the observed bit" {
    var r = Rls.init(.{ .mode = .full, .lambda = 0.999999, .p0 = 0.1 });
    const x = [_]f32{1.0} ** D;
    var i: usize = 0;
    while (i < 64) : (i += 1) {
        const z = r.mix(&x);
        const p = 1.0 / (1.0 + @exp(-z));
        r.update(1, p);
    }
    try std.testing.expect(r.z > 0.0);
    try std.testing.expect(std.math.isFinite(r.trace()));
    try std.testing.expectEqual(@as(u64, 0), r.resets);
}

test "rls: guard resets instead of propagating NaN" {
    var r = Rls.init(.{ .mode = .full, .lambda = 0.999999, .p0 = 0.1, .guard = true });
    const x = [_]f32{1.0} ** D;
    _ = r.mix(&x);
    // poison P so the Sherman-Morrison denominator is non-finite
    r.p[0] = std.math.nan(f64);
    _ = r.mix(&x);
    r.update(1, 0.5);
    try std.testing.expect(r.resets >= 1);
    try std.testing.expect(std.math.isFinite(r.trace()));
}
