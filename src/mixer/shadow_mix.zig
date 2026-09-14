//! Shadow-mixer instrument — comptime-dead unless `-Dmixshadow`.
//!
//! WHY THIS IS SOUND (the load-bearing property):
//!   The 590-input vector `mi0.inputs` (v) is computed entirely from leaf models
//!   (bracket / fxcm / direct / match / indirect / ppmd / byte-mixer), and NOTHING
//!   downstream of the mixers ever writes back into any of them:
//!     * `mgr.auxiliary_context` reads mi0.inputs[560] and [589] — leaf slots, not
//!       mixer outputs;
//!     * the SSE consumes the mixer-1 probability but feeds no leaf;
//!     * the arithmetic coder consumes p and feeds nothing back (y is the data).
//!   => v_t is a pure function of the input stream and is INVARIANT to the mixing
//!   stage.  Therefore a shadow mixer fed the same v, updated by a different rule,
//!   measures EXACTLY the loss that rule would have produced in a real run (up to
//!   the SSE stage, which is not shadowed here).  Zero divergence, perfect pairing,
//!   one run per N rules.
//!
//! The shadow never touches the shipped path: with `-Dmixshadow` the produced
//! archive must be BYTE-IDENTICAL to stock.  That is the instrument's own gate.
//!
//! RULES
//!   stock     update_i = lr_i * decay * (sigma(h_i) - y)          <- cmix/PAQ family
//!   finalerr  update_i = lr_i * decay * (sigma(z)   - y)          <- ensemble residual
//!   e2e       update_i = lr_i * decay * (sigma(z)-y) * beta_i     <- TRUE gradient
//!   e2enorm   ... * beta_i / rms(beta)                            <- true direction, stock scale
//!   blend     0.5*stock + 0.5*e2enorm
//!
//! beta_i = dz/dh_i is exact because the layer-0 cascade + layer-1 mixer are AFFINE
//! in v (raw logits are passed between layers; the only nonlinearities are the
//! +-logit(1e-4) clamps and the final sigmoid).  Backward recursion over 23 mixers
//! costs 253 MACs/bit — 1.9% of one forward cascade.
const std = @import("std");
const Sigmoid = @import("../sigmoid.zig").Sigmoid;
const build_options = @import("build_options");
// the own-module rule: the -Dmixshadow knob lives in the dedicated "mixer_options" module
// (not shared model_opts); build_options stays imported for mixer_decay_floor.
const mixer_options = @import("mixer_options");

pub const enabled: bool = mixer_options.mixshadow;

const VL = 8;
const Vf = @Vector(VL, f32);

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

pub const Rule = enum(u8) {
    stock = 0,
    finalerr = 1,
    e2e = 2,
    e2enorm = 3,
    blend = 4,
    // `*i` variants initialise the layer-1 weight row to 1/23 on its 23 mixer
    // inputs ("start as the plain average") instead of 0. Zero-init makes the
    // end-to-end gradient identically zero (beta_i = W1[i] = 0 and h_i = 0), so
    // the origin is a fixed point; local-error training is what breaks that
    // symmetry in stock. `stocki` is the init-only control.
    stocki = 5,
    e2ei = 6,
    e2enormi = 7,
    finalerri = 8,
    // CONTROLS separating 'true gradient' from 'smaller effective step':
    //   stockgamma = LOCAL error x gamma_i  (same per-mixer scaling as e2ei, wrong error)
    //   stockscl   = LOCAL error x 0.12     (flat scale = mean gamma; pure LR-scale control)
    stockgamma = 9,
    stockscl = 10,
    // e2eada: exact gradient + per-INPUT normalisation. Having the true gradient
    // makes a real optimiser legitimate. Each input j is divided by its running
    // RMS (soft-floored at 1 logit) before the AXPY, so every expert contributes
    // a unit-scale direction and the inherited eta_i keeps its meaning. Tuning-free.
    // stockada = the same normalisation on the LOCAL error (isolation control).
    e2eada = 11,
    stockada = 12,
    // *c1 variants key the LAYER-1 mixer by c0 (long_bit_context, 255 live values)
    // instead of the shipped constant-0 context. The affine view says gamma IS the
    // ensemble's credit vector; shipped, it is ONE global row for the whole 1 GB
    // file. These make it per-bit-position.
    stockc1 = 13,
    e2eic1 = 14,
    e2enormc1 = 15,

    pub fn name(r: Rule) []const u8 {
        return switch (r) {
            .stock => "stock",
            .finalerr => "finalerr",
            .e2e => "e2e",
            .e2enorm => "e2enorm",
            .blend => "blend",
            .stocki => "stocki",
            .e2ei => "e2ei",
            .e2enormi => "e2enormi",
            .finalerri => "finalerri",
            .stockgamma => "stockgamma",
            .stockscl => "stockscl",
            .e2eada => "e2eada",
            .stockada => "stockada",
            .stockc1 => "stockc1",
            .e2eic1 => "e2eic1",
            .e2enormc1 => "e2enormc1",
        };
    }

    /// does this rule need beta (the exact chain factor)?
    pub fn needsBeta(r: Rule) bool {
        return switch (r) {
            .stock, .finalerr, .stocki, .finalerri, .stockscl, .stockada, .stockc1 => false,
            else => true,
        };
    }

    /// layer-1 weight-row init value on the 23 mixer inputs
    pub fn w1Init(r: Rule) f32 {
        return switch (r) {
            .stocki, .e2ei, .e2enormi, .finalerri, .stockgamma, .stockscl, .e2eada, .e2eic1, .e2enormc1 => 1.0 / 23.0,
            else => 0.0,
        };
    }

    /// which context does this rule's LAYER-1 mixer key on?  false = the shipped
    /// constant 0 (one global row for the whole file); true = c0.
    pub fn ctxLayer1(r: Rule) bool {
        return switch (r) {
            .stockc1, .e2eic1, .e2enormc1 => true,
            else => false,
        };
    }

    /// does this rule use the LOCAL error term?
    pub fn usesLocal(r: Rule) bool {
        return switch (r) {
            .stock, .stocki, .blend, .stockgamma, .stockscl, .stockada, .stockc1 => true,
            else => false,
        };
    }
};

const Row = struct {
    w: []f32,
    u: []f32,
};

/// One shadow mixer — a faithful re-implementation of MixerLex with a pluggable
/// update scalar.  Deliberately a copy, not a refactor of MixerLex: the shipped
/// mixer must not gain a branch.
const SMixer = struct {
    map: std.AutoHashMap(u32, *Row),
    base: *Row,
    ctx: *const u64,
    lr: f32,
    n_in: usize,
    n_extra: usize,
    steps: u64 = 0,
    p: f32 = 0,
    resolved: ?*Row = null,
    alloc: std.mem.Allocator,
    // census
    base_hits: u64 = 0,
    beta_sum: f64 = 0,
    beta_abs_sum: f64 = 0,
    beta_neg: u64 = 0,
    init_head: usize = 0,
    init_val: f32 = 0,

    fn init(a: std.mem.Allocator, ctx: *const u64, lr: f32, n_in: usize, n_extra: usize) SMixer {
        var s = SMixer{
            .map = std.AutoHashMap(u32, *Row).init(a),
            .base = undefined,
            .ctx = ctx,
            .lr = lr,
            .n_in = n_in,
            .n_extra = n_extra,
            .alloc = a,
        };
        s.base = s.newRow();
        return s;
    }

    fn initHead(a: std.mem.Allocator, ctx: *const u64, lr: f32, n_in: usize, n_extra: usize, head: usize, val: f32) SMixer {
        var s = SMixer{
            .map = std.AutoHashMap(u32, *Row).init(a),
            .base = undefined,
            .ctx = ctx,
            .lr = lr,
            .n_in = n_in,
            .n_extra = n_extra,
            .alloc = a,
            .init_head = head,
            .init_val = val,
        };
        s.base = s.newRow();
        return s;
    }

    fn newRow(self: *SMixer) *Row {
        const r = self.alloc.create(Row) catch unreachable;
        r.* = .{
            .w = self.alloc.alloc(f32, self.n_in) catch unreachable,
            .u = self.alloc.alloc(f32, self.n_extra) catch unreachable,
        };
        @memset(r.w, 0);
        @memset(r.u, 0);
        if (self.init_head != 0) @memset(r.w[0..self.init_head], self.init_val);
        return r;
    }

    fn deinit(self: *SMixer) void {
        var it = self.map.valueIterator();
        while (it.next()) |v| {
            self.alloc.free(v.*.w);
            self.alloc.free(v.*.u);
            self.alloc.destroy(v.*);
        }
        self.map.deinit();
        self.alloc.free(self.base.w);
        self.alloc.free(self.base.u);
        self.alloc.destroy(self.base);
    }

    fn row(self: *SMixer) *Row {
        const c: u32 = @truncate(self.ctx.*);
        if (self.map.getPtr(c)) |v| return v.*;
        if (self.map.count() >= 10000) {
            self.base_hits += 1;
            return self.base;
        }
        const gop = self.map.getOrPut(c) catch unreachable;
        if (!gop.found_existing) gop.value_ptr.* = self.newRow();
        return gop.value_ptr.*;
    }

    fn mix(self: *SMixer, inputs: []const f32, extra: []const f32) f32 {
        @setFloatMode(.optimized);
        const d = self.row();
        self.resolved = d;
        self.p = dotV(inputs[0..self.n_in], d.w[0..self.n_in]);
        if (self.n_extra != 0) self.p += dotV(extra[0..self.n_extra], d.u[0..self.n_extra]);
        return self.p;
    }

    /// `err_scalar` is the rule's (dL/dh_i)-like term BEFORE lr and decay.
    fn update(self: *SMixer, err_scalar: f32, inputs: []const f32, extra: []const f32, decay: f32) void {
        @setFloatMode(.optimized);
        var upd: f32 = self.lr * err_scalar;
        self.steps += 1;
        if (@abs(upd) < 0.000000000005 and self.n_extra > 0) {
            self.resolved = null;
            return;
        }
        upd = decay * upd;
        const d = self.resolved orelse self.row();
        self.resolved = null;
        axpyV(d.w[0..self.n_in], -upd, inputs[0..self.n_in]);
        if (self.n_extra != 0) axpyV(d.u[0..self.n_extra], -upd, extra[0..self.n_extra]);
    }
};

const N_IN = 590;
const N_M0 = 23;
const N_IN1 = 25;

/// One complete shadow predictor stack (23 layer-0 + 1 layer-1) under one rule.
const Variant = struct {
    rule: Rule,
    m0: [N_M0]SMixer,
    m1: SMixer,
    extra0: [N_M0]f32 = [_]f32{0} ** N_M0, // mi0.extra_inputs equivalent
    in1: [N_IN1]f32 = [_]f32{0} ** N_IN1, // mi1.inputs equivalent
    h: [N_M0]f32 = [_]f32{0} ** N_M0, // raw (unclamped) layer-0 outputs
    z: f32 = 0,
    bits: f64 = 0, // accumulated coded cost, in bits (coder-quantised)
    bits_tranche: f64 = 0,
};

pub const Shadow = struct {
    alloc: std.mem.Allocator,
    sig: *const Sigmoid,
    variants: []Variant,
    stretched_min: f32,
    stretched_max: f32,
    decay_floor: f32,
    // reference channels measured on the REAL stack
    real_pre_sse_bits: f64 = 0,
    real_post_sse_bits: f64 = 0,
    real_pre_sse_tranche: f64 = 0,
    real_post_sse_tranche: f64 = 0,
    nbits: u64 = 0,
    ms: [N_IN]f64 = [_]f64{0} ** N_IN, // running sum of v_j^2 (shared: v is shared)
    vn: [N_IN]f32 = [_]f32{0} ** N_IN, // this bit's normalised input vector
    tranche_bits: u64 = 0,
    tranche_every: u64 = 8 * 1024 * 1024, // 1 MiB of stream

    pub fn create(a: std.mem.Allocator, sig: *const Sigmoid, ctxs: []const *const u64, lrs: []const f32, lr1: f32, ctx1: *const u64) *Shadow {
        const self = a.create(Shadow) catch unreachable;
        var want_buf: [8]Rule = undefined;
        const want = rulesFromEnv(&want_buf);
        const vs = a.alloc(Variant, want.len) catch unreachable;
        for (want, 0..) |r, k| {
            vs[k] = .{ .rule = r, .m0 = undefined, .m1 = SMixer.initHead(a, if (r.ctxLayer1()) ctx1 else &ZERO_CTX, lr1, N_IN1, 0, if (r.w1Init() != 0) 23 else 0, r.w1Init()) };
            for (0..N_M0) |i| vs[k].m0[i] = SMixer.init(a, ctxs[i], lrs[i], N_IN, i);
        }
        self.* = .{
            .alloc = a,
            .sig = sig,
            .variants = vs,
            .stretched_min = sig.logit(0),
            .stretched_max = sig.logit(1),
            .decay_floor = build_options.mixer_decay_floor,
        };
        std.debug.print("MIXSHADOW_HDR\tcoded_bits\treal_pre_sse\treal_post_sse", .{});
        for (self.variants) |v| std.debug.print("\t{s}", .{v.rule.name()});
        std.debug.print("\n", .{});
        return self;
    }

    var ZERO_CTX: u64 = 0;

    fn rulesFromEnv(buf: []Rule) []Rule {
        const all = [_]Rule{ .stock, .finalerr, .e2e, .e2enorm, .blend, .stocki, .e2ei, .e2enormi, .finalerri, .stockgamma, .stockscl, .e2eada, .stockada, .stockc1, .e2eic1, .e2enormc1 };
        if (comptime @import("builtin").os.tag == .windows) {
            @memcpy(buf[0..all.len], &all);
            return buf[0..all.len];
        }
        const spec = std.posix.getenv("ZMIX_MIXSHADOW") orelse {
            @memcpy(buf[0..all.len], &all);
            return buf[0..all.len];
        };
        var n: usize = 0;
        var it = std.mem.splitScalar(u8, spec, ',');
        while (it.next()) |tok| {
            const t = std.mem.trim(u8, tok, " ");
            for (all) |r| {
                if (std.mem.eql(u8, t, r.name()) and n < buf.len) {
                    buf[n] = r;
                    n += 1;
                }
            }
        }
        return buf[0..n];
    }

    pub fn destroy(self: *Shadow) void {
        const a = self.alloc;
        for (self.variants) |*v| {
            for (0..N_M0) |i| v.m0[i].deinit();
            v.m1.deinit();
        }
        a.free(self.variants);
        a.destroy(self);
    }

    inline fn clampS(self: *const Shadow, x: f32) f32 {
        if (x > self.stretched_max) return self.stretched_max;
        if (x < self.stretched_min) return self.stretched_min;
        return x;
    }

    /// Coder-exact cost of coding `bit` at probability-of-1 `p` (matches coder.zig's
    /// P = 1 + floor(65534*p) over a 16-bit interval).
    inline fn codedBits(p_in: f32, bit: i32) f64 {
        var p: f64 = @floatCast(p_in);
        if (!(p >= 0)) p = 0;
        if (p > 1) p = 1;
        const P: f64 = 1.0 + @floor(65534.0 * p);
        const q1: f64 = P / 65536.0;
        return if (bit == 1) -std.math.log2(q1) else -std.math.log2(1.0 - q1);
    }

    /// Run every shadow's forward cascade.  `v` is the shared 590-input vector.
    pub fn predict(self: *Shadow, v: []const f32) void {
        for (self.variants) |*var_| {
            for (0..N_M0) |i| {
                const hi = var_.m0[i].mix(v, var_.extra0[0..i]);
                var_.h[i] = hi;
                const c = self.clampS(hi);
                var_.extra0[i] = c;
                var_.in1[i] = c;
            }
            var_.in1[23] = self.clampS(v[560]);
            var_.in1[24] = self.clampS(v[589]);
            var_.z = var_.m1.mix(var_.in1[0..N_IN1], &.{});
        }
    }

    /// Accumulate loss and apply each rule's update.  `real_pre` is the real
    /// mixer-1 probability (post-logistic, pre-SSE); `real_post` is post-SSE.
    /// `free_bit` = the byte-mixer's [bot,top] collapsed, so PREDICT's return value
    /// is overridden to a certainty and the bit costs 0 REGARDLESS of the mixer.
    /// Those bits are excluded from every cost channel (real and shadow) so the
    /// deltas stay directly comparable to archive bytes.
    pub fn perceive(self: *Shadow, v: []const f32, bit: i32, real_pre: f32, real_post: f32, free_bit: bool) void {
        const y: f32 = @floatFromInt(bit);
        const cb_pre = if (free_bit) 0 else codedBits(real_pre, bit);
        const cb_post = if (free_bit) 0 else codedBits(real_post, bit);
        self.real_pre_sse_bits += cb_pre;
        self.real_post_sse_bits += cb_post;
        self.real_pre_sse_tranche += cb_pre;
        self.real_post_sse_tranche += cb_post;
        self.nbits += 1;
        self.tranche_bits += 1;

        // Per-input normalisation, computed ONCE (v is shared across variants).
        // rms_j = sqrt(E[v_j^2] + 1): the +1 soft-floor (one logit) stops a
        // near-dead input from acquiring an unbounded effective learning rate.
        var need_norm = false;
        for (self.variants) |*vv| {
            if (vv.rule == .e2eada or vv.rule == .stockada) need_norm = true;
        }
        if (need_norm) {
            const n: f64 = @floatFromInt(self.nbits);
            for (0..N_IN) |j| {
                const vj: f64 = @floatCast(v[j]);
                self.ms[j] += vj * vj;
                const rms: f64 = @sqrt(self.ms[j] / n + 1.0);
                self.vn[j] = @floatCast(vj / rms);
            }
        }

        for (self.variants) |*var_| {
            const pz = Sigmoid.logistic(var_.z);
            const cb = if (free_bit) 0 else codedBits(pz, bit);
            var_.bits += cb;
            var_.bits_tranche += cb;

            // --- decay: identical schedule to MixerLex, on the per-mixer global step
            // count.  All shadow mixers step every bit, so one value serves all 23.
            const st = var_.m0[0].steps;
            var decay: f32 = self.decay_floor;
            if (st < 25000000) {
                decay = 0.3;
                if (st < 5000000) {
                    decay = 0.7;
                    if (st < 1000000) decay = 1.0;
                }
            }

            const g_final: f32 = pz - y;

            // --- backward pass for beta_i = dz/dh_i  (exact; the stack is affine).
            var beta: [N_M0]f32 = undefined;
            if (var_.rule.needsBeta()) {
                const w1 = var_.m1.resolved orelse var_.m1.row();
                var i: usize = N_M0;
                while (i > 0) {
                    i -= 1;
                    var s: f32 = w1.w[i]; // direct path into layer 1
                    var k: usize = i + 1;
                    while (k < N_M0) : (k += 1) {
                        // mixer k's extra-weight on h_i  (u_k has length k)
                        const rk = var_.m0[k].resolved orelse var_.m0[k].row();
                        s += beta[k] * rk.u[i];
                    }
                    // clamp gate: d clamp(h)/dh
                    const hi = var_.h[i];
                    const gate: f32 = if (hi > self.stretched_max or hi < self.stretched_min) 0.0 else 1.0;
                    beta[i] = s * gate;
                    var_.m0[i].beta_sum += beta[i];
                    var_.m0[i].beta_abs_sum = var_.m0[i].beta_abs_sum + @abs(beta[i]);
                    if (beta[i] < 0) var_.m0[i].beta_neg += 1;
                }
            }

            var scale: f32 = 1.0;
            if (var_.rule == .e2enorm or var_.rule == .blend or var_.rule == .e2enormi or var_.rule == .e2enormc1) {
                var ss: f32 = 0;
                for (0..N_M0) |i| ss += beta[i] * beta[i];
                const rms: f32 = @sqrt(ss / @as(f32, N_M0));
                scale = if (rms > 1e-20) 1.0 / rms else 0.0;
            }

            for (0..N_M0) |i| {
                const local: f32 = if (var_.rule.usesLocal()) Sigmoid.logistic(var_.h[i]) - y else 0;
                const e: f32 = switch (var_.rule) {
                    .stock, .stocki => local,
                    .stockgamma => local * beta[i],
                    .e2eada => g_final * beta[i],
                    .stockada => local,
                    .stockc1 => local,
                    .e2eic1 => g_final * beta[i],
                    .stockscl => local * 0.12,
                    .finalerr, .finalerri => g_final,
                    .e2e, .e2ei => g_final * beta[i],
                    .e2enorm, .e2enormi, .e2enormc1 => g_final * beta[i] * scale,
                    .blend => 0.5 * local + 0.5 * g_final * beta[i] * scale,
                };
                const vv: []const f32 = if (var_.rule == .e2eada or var_.rule == .stockada) self.vn[0..N_IN] else v;
                var_.m0[i].update(e, vv, var_.extra0[0..i], decay);
            }
            var_.m1.update(g_final, var_.in1[0..N_IN1], &.{}, decay);
        }

        if (self.tranche_bits >= self.tranche_every) self.flushTranche();
    }

    fn flushTranche(self: *Shadow) void {
        std.debug.print("MIXSHADOW_T\t{d}\t{d:.1}\t{d:.1}", .{
            self.nbits, self.real_pre_sse_tranche, self.real_post_sse_tranche,
        });
        for (self.variants) |*v| {
            std.debug.print("\t{d:.1}", .{v.bits_tranche});
            v.bits_tranche = 0;
        }
        std.debug.print("\n", .{});
        self.real_pre_sse_tranche = 0;
        self.real_post_sse_tranche = 0;
        self.tranche_bits = 0;
    }

    pub fn report(self: *Shadow) void {
        self.flushTranche();
        std.debug.print("\n[mixshadow] bits={d}  real_pre_sse={d:.0} B  real_post_sse={d:.0} B\n", .{
            self.nbits, self.real_pre_sse_bits / 8.0, self.real_post_sse_bits / 8.0,
        });
        var base: f64 = 0;
        for (self.variants) |*v| {
            if (v.rule == .stock) base = v.bits;
        }
        if (base == 0 and self.variants.len > 0) base = self.variants[0].bits;
        for (self.variants) |*v| {
            std.debug.print("[mixshadow] {s}\t{d:.0} B\tdelta_vs_stock {d:.1} B\t{d:.4} pct\n", .{
                v.rule.name(), v.bits / 8.0, (v.bits - base) / 8.0, 100.0 * (v.bits - base) / base,
            });
        }
        // Per-mixer census, taken from the FIRST variant (row occupancy is a
        // function of the contexts alone, so it is identical across rules; the
        // beta census needs a variant that computes beta, hence the search).
        var cv: ?*Variant = null;
        for (self.variants) |*v| {
            if (v.rule.needsBeta()) {
                cv = v;
                break;
            }
        }
        const rv = cv orelse &self.variants[0];
        const nb: f64 = @floatFromInt(if (self.nbits == 0) 1 else self.nbits);
        std.debug.print("MIXCENSUS\tmixer\trows\tbase_hit_pct\tmean_beta\tmean_abs_beta\tneg_beta_pct\n", .{});
        for (0..N_M0) |i| {
            const m = &rv.m0[i];
            std.debug.print("MIXCENSUS\t{d}\t{d}\t{d:.3}\t{d:.5}\t{d:.5}\t{d:.2}\n", .{
                i,
                m.map.count(),
                100.0 * @as(f64, @floatFromInt(m.base_hits)) / nb,
                m.beta_sum / nb,
                m.beta_abs_sum / nb,
                100.0 * @as(f64, @floatFromInt(m.beta_neg)) / nb,
            });
        }
    }
};
