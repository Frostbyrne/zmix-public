//! SSE (secondary symbol estimation), ported from cmix `mixer/sse.{h,cpp}`.
//! Original SSE code by Eugene Shelwien (mod_ppmd). This is the final
//! calibration stage applied to the mixer output in the predictor.
//!
//! Faithful port: two interpolating SSE stages (s6/s7) each feeding a small
//! per-context mixer (x1/x2), indexed by quantised probability, a few flag bits,
//! the previous byte, and the partial byte. Flat arrays replace the templated
//! SSEi<7>/Mixer arrays to keep memory tight.
const std = @import("std");
const masks = @import("sse_masks.zig");
const zero_alloc = @import("zero_alloc.zig");

const SCALElog: u5 = 15;
const SCALE: i32 = 1 << 15;
const hSCALE: i32 = SCALE / 2;
const mSCALE: i32 = SCALE - 1;
const SSEQuant: i32 = 7;

// Tuning constants (verbatim from cmix sse.cpp).
const M_f0C = 10240;
const M_f1C = 7935;
const M_f2C = 9592;
// -Dsse-wr6 / -Dsse-wr7 / -Dsse-xw1 / -Dsse-xw7: the FOUR rate constants of
// Shelwien's SSE. `M_sm6wrB`/`M_sm7wrB` are the SSE table update rates
// (`sseUpdate`: P moves toward the bit by wr0/32768, i.e. 1/309 and 1/258);
// `M_x1wr`/`M_x2wr` scale the two per-context mixer learning rates
// (`mixerUpdate`). All four were tuned by Shelwien against mod_ppmd's byte-model
// output distribution (~2008) and carried verbatim through cmix into zmix, which
// feeds this stage the output of a three-layer 590-input mixer plus an LSTM.
// NEVER re-derived. Defaults = shipped, bit-identical. Same shape as
// `postnorm-mixer-hyperparam`, one stage further out.
const M_sm6wrB = @import("build_options").sse_wr6;
const M_sm6mw = 0;
const M_sm6C1 = 8092;
const M_x1W0 = 7649;
const M_x1wr = @import("build_options").sse_xw1;
const M_f3C = 8200;
const M_f4C = 7677;
const M_sm7wrB = @import("build_options").sse_wr7;
const M_sm7mw = 8192;
const M_sm7C1 = 8202;
const M_x2W0 = 2561;
const M_x2wr = @import("build_options").sse_xw2;

// -Dsse-ffl6 / -Dsse-ffl7 — the SSE flag-history context WIDTH.
//
// `s6` is 352 MiB and `s7` 84 MiB: together 4.9 % of the whole anon budget, on the
// third-largest PAQ-lineage structure in the engine. Their volumes are Shelwien's
// `mod_ppmd` sizing (≈2008), carried verbatim through cmix, and NEVER re-derived for
// zmix — which feeds them the output of a 590-input three-layer mixer plus an LSTM,
// a completely different input law from PPMd's. The `ffl` dimension (a run-length
// history of "previous byte >= 0x40") is the most speculative of the three context
// components and the one that multiplies the volume: s6 spends 2^7 = 128 on it.
// Each bit removed HALVES the bank. Defaults 7/5 = shipped, bit-identical.
const FFL6: u5 = @import("build_options").sse_ffl6;
const FFL7: u5 = @import("build_options").sse_ffl7;
comptime {
    if (FFL6 > 7 or FFL7 > 5) @compileError("-Dsse-ffl6 <= 7 and -Dsse-ffl7 <= 5 (shipped values)");
}
const FFL6_MASK: u32 = (@as(u32, 1) << FFL6) - 1;
const FFL7_MASK: u32 = (@as(u32, 1) << FFL7) - 1;

const M_mix1_Volume: usize = 1 * 4 * (1 << 8) * (1 << 3) * 79;
const M_mix2_Volume: usize = 1 * 3 * (1 << 1) * (1 << 8) * 256;
const M_sm6x_Volume: usize = 1 * 3 * (@as(usize, 1) << FFL6) * (1 << 8) * 256;
const M_sm7x_Volume: usize = 1 * 3 * (@as(usize, 1) << FFL7) * (1 << 8) * 255;

/// The 7-word row seed of SSEi.Init (`wi/2 + 8192 + i*(SCALE-wi)/6`), kept as
/// a comptime ramp: s6/s7 store `true -% ramp[i%7]` so the freshly-allocated
/// state is all-zeros and the ~420 MiB sits on lazy kernel zero pages until a
/// context is first updated. Reads reconstruct the exact u16 (wrapping add is
/// a bijection) — bit-identical to the eager fill.
fn sseiRamp(wi: i32) [7]u16 {
    const scw = @divTrunc(SCALE - wi, SSEQuant - 1);
    const inc = @divTrunc(wi, 2) + 8192;
    var r: [7]u16 = undefined;
    for (&r, 0..) |*v, i| v.* = @truncate(@as(u32, @bitCast(inc + @as(i32, @intCast(i)) * scw)));
    return r;
}
const RAMP6 = sseiRamp(M_sm6mw);
const RAMP7 = sseiRamp(M_sm7mw);
/// x1/x2 mixer weight seeds — same trick with a constant offset; the update
/// path (`w +%= d`) is offset-invariant, so only the read site adds it back.
const X1_INIT: i32 = M_x1W0 + hSCALE;
const X2_INIT: i32 = M_x2W0 + hSCALE;

// Shared st/sq tables, computed once.
var t_st: [1 << 15]u16 = undefined;
var t_sq: [1 << 15]u16 = undefined;
var tables_ready = false;

fn log2d(a: f64) f64 {
    return std.math.log2(a);
}
fn exp2d(a: f64) f64 {
    return std.math.exp2(a);
}
fn stf(p: f64) f64 {
    return log2d((1 - p) / p);
}
fn sqf(p: f64) f64 {
    return 1.0 / (1.0 + exp2d(p));
}

fn initTables() void {
    if (tables_ready) return;
    const st_coef: f64 = @as(f64, hSCALE - 1) / log2d(@as(f64, SCALE - 1));
    const sq_coef: f64 = 1.0 / st_coef;
    const st_i = struct {
        fn f(coef: f64, p: u32) u32 {
            const v = stf(@as(f64, @floatFromInt(p)) / @as(f64, SCALE)) * coef + @as(f64, hSCALE);
            return @intFromFloat(v);
        }
    }.f;
    const sq_i = struct {
        fn f(coef: f64, p: u32) u32 {
            const v = sqf((@as(f64, @floatFromInt(@as(i32, @intCast(p)) - hSCALE))) * coef) * @as(f64, SCALE);
            return @intFromFloat(v);
        }
    }.f;

    t_sq[0] = 0;
    var i: u32 = 1;
    while (i < SCALE) : (i += 1) t_sq[i] = @truncate(sq_i(sq_coef, i));

    var x: u32 = 0;
    t_st[0] = 0;
    i = 1;
    while (i < SCALE) : (i += 1) {
        const s: u32 = st_i(st_coef, i);
        t_st[i] = @truncate(s);
        if (t_st[i] != t_st[x]) {
            const y = i - 1;
            t_sq[t_st[x]] = @truncate((x + y + 1) / 2);
            x = i;
        }
    }
    tables_ready = true;
}

fn extrap(p1_in: i32, c: i32) i32 {
    var p1 = (((p1_in - hSCALE) *% c) >> 13) + hSCALE;
    if (p1 < 1) p1 = 1;
    if (p1 > mSCALE) p1 = mSCALE;
    return p1;
}

fn rdiv(x: i32, a: i32, d: u5) i32 {
    return if (x >= 0) (x + a) >> d else -((-x + a) >> d);
}

pub const SSE = struct {
    // s6/s7: flat SSEi<7> P arrays (Volume * 7 words).
    s6: []u16,
    s7: []u16,
    // x1/x2: per-context mixer weights.
    x1: []i32,
    x2: []i32,

    // per-step state carried Predict -> Perceive
    sm6x: usize = 0,
    sm7x: usize = 0,
    mix1: usize = 0,
    mix2: usize = 0,
    su6_C1: usize = 0,
    su6_sw: i32 = 0,
    su6_P: i32 = 0,
    su7_C1: usize = 0,
    su7_sw: i32 = 0,
    su7_P: i32 = 0,
    mix1_s0: i32 = 0,
    mix1_s1: i32 = 0,
    mix1_p: i32 = 0,
    mix2_s0: i32 = 0,
    mix2_s1: i32 = 0,
    mix2_p: i32 = 0,
    M_j: u32 = 1,
    M_pc: u32 = 0,
    M_ffl: u32 = 0,

    pub fn init(a: std.mem.Allocator) !SSE {
        initTables();
        // Stored delta-from-init (see RAMP6/RAMP7/X1_INIT above): no eager
        // fill; zero_alloc's big path is a lazy kernel-zeroed mapping and
        // advises THP itself.
        return .{
            .s6 = try zero_alloc.alloc(a, u16, M_sm6x_Volume * 7),
            .s7 = try zero_alloc.alloc(a, u16, M_sm7x_Volume * 7),
            .x1 = try zero_alloc.alloc(a, i32, M_mix1_Volume),
            .x2 = try zero_alloc.alloc(a, i32, M_mix2_Volume),
        };
    }

    pub fn deinit(self: *SSE, a: std.mem.Allocator) void {
        zero_alloc.free(a, self.s6);
        zero_alloc.free(a, self.s7);
        zero_alloc.free(a, self.x1);
        zero_alloc.free(a, self.x2);
    }

    // SSEi.SSE_Pred on flat array `arr` at context `ctx` (base = ctx*7).
    // `arr` is ramp-offset storage; `+% ramp[...]` reconstructs the exact u16.
    fn ssePred(comptime ramp: [7]u16, arr: []u16, ctx: usize, iP: i32, C1_out: *usize, sw_out: *i32, P_out: *i32) i32 {
        const sseFreq: usize = @intCast(((SSEQuant - 1) * iP) >> SCALElog);
        sw_out.* = ((SSEQuant - 1) * iP) & mSCALE;
        const c1 = ctx * 7 + sseFreq;
        C1_out.* = c1;
        var f = (((SCALE - sw_out.*) * @as(i32, arr[c1] +% ramp[sseFreq]) + sw_out.* * @as(i32, arr[c1 + 1] +% ramp[sseFreq + 1])) >> SCALElog) - 8192;
        if (f <= 0) f = 1;
        if (f >= SCALE) f = mSCALE;
        P_out.* = f;
        return f;
    }

    fn sseUpdate(comptime ramp: [7]u16, arr: []u16, c1: usize, sw: i32, P_in: i32, bit: i32, wr0: i32) void {
        // c1 = ctx*7 + freq with freq in [0,5], so c1 % 7 recovers the row slot.
        const fr = c1 % 7;
        var P = P_in * (SCALE - wr0) >> SCALElog;
        if (bit == 0) P += wr0;
        const dC = @as(i32, arr[c1] +% ramp[fr]) - @as(i32, arr[c1 + 1] +% ramp[fr + 1]);
        const sw_dC = (sw *% dC + mSCALE) >> SCALElog;
        arr[c1] = @as(u16, @truncate(@as(u32, @bitCast(P + sw_dC + 8192)))) -% ramp[fr];
        arr[c1 + 1] = @as(u16, @truncate(@as(u32, @bitCast(P - (dC - sw_dC) + 8192)))) -% ramp[fr + 1];
    }

    fn mixup(w: i32, s1: i32, s0: i32) i32 {
        var x = s1 + rdiv((w -% hSCALE) *% (s0 -% s1), 1 << (SCALElog - 1), SCALElog);
        x = if (x > 0) (if (x < SCALE) x else SCALE - 1) else 1;
        return x;
    }

    fn mixerUpdate(w: *i32, y: i32, p0: i32, p1: i32, wq: i32, pm: i32) void {
        const py = SCALE - (y << SCALElog);
        const e = py - pm;
        var d = rdiv(e *% (p0 -% p1), 1 << (SCALElog - 1), SCALElog);
        d = rdiv(d *% wq, 1 << (SCALElog - 1), SCALElog);
        w.* +%= d;
    }

    fn estimate(self: *SSE, p: i32) i32 {
        // Index the shared tables through many-item pointers: naming the module
        // `var` arrays directly here made LLVM materialize a 64KB stack COPY of
        // each table (memcpy per call, 2x per bit — ~10% of total runtime, present
        // since the first build) to serve single data-dependent lookups. Pointer
        // indexing compiles to plain loads.
        const sq: [*]const u16 = &t_sq;
        const st: [*]const u16 = &t_st;
        const prq: i32 = p >> 11;
        const j = self.M_j;
        const ffl = self.M_ffl;
        const pc = self.M_pc;
        const b0: i32 = @intFromBool(prq > 0);
        const b7: i32 = @intFromBool(prq > 7);
        const b14: i32 = @intFromBool(prq > 14);

        var sm7x: i32 = (b0 + b14);
        sm7x = (sm7x << FFL7) + @as(i32, @intCast(ffl & FFL7_MASK));
        sm7x = (sm7x << 8) + @as(i32, @intCast(pc & 255));
        sm7x = (sm7x * 255) + masks.SM7MASK0[j];
        self.sm7x = @intCast(sm7x);

        var mix2: i32 = (b0 + b14);
        mix2 = (mix2 << 1) + @as(i32, @intCast(ffl & 1));
        mix2 = (mix2 << 8) + @as(i32, @intCast(pc & 255));
        mix2 = (mix2 * 256) + @as(i32, @intCast(j));
        self.mix2 = @intCast(mix2);

        var sm6x: i32 = (b0 + b14);
        sm6x = (sm6x << FFL6) + @as(i32, @intCast(ffl & FFL6_MASK));
        sm6x = (sm6x << 8) + @as(i32, @intCast(pc & 255));
        sm6x = (sm6x * 256) + @as(i32, @intCast(j));
        self.sm6x = @intCast(sm6x);

        var mix1: i32 = (b0 + b7 + b14);
        mix1 = (mix1 << 8) + @as(i32, @intCast(ffl & 255));
        mix1 = (mix1 << 3) + @as(i32, @intCast((pc >> 5) & 7));
        mix1 = (mix1 * 79) + masks.MX1MASK0[j];
        self.mix1 = @intCast(mix1);

        const p0 = p;
        const p1 = ssePred(RAMP6, self.s6, self.sm6x, sq[@intCast(extrap(st[@intCast(p0)], M_f0C))], &self.su6_C1, &self.su6_sw, &self.su6_P);
        const s0 = extrap(st[@intCast(p0)], M_f1C);
        const s1 = extrap(st[@intCast(p1)], M_f2C);
        self.mix1_s0 = s0;
        self.mix1_s1 = s1;
        var s2 = mixup(self.x1[self.mix1] +% X1_INIT, self.mix1_s0, self.mix1_s1);
        s2 = extrap(s2, M_sm6C1);
        self.mix1_p = sq[@intCast(s2)];

        const p2 = ssePred(RAMP7, self.s7, self.sm7x, sq[@intCast(extrap(st[@intCast(p0)], M_f3C))], &self.su7_C1, &self.su7_sw, &self.su7_P);
        const s4 = extrap(st[@intCast(p2)], M_f4C);
        self.mix2_s0 = s2;
        self.mix2_s1 = s4;
        var s5 = mixup(self.x2[self.mix2] +% X2_INIT, self.mix2_s0, self.mix2_s1);
        s5 = extrap(s5, M_sm7C1);
        self.mix2_p = sq[@intCast(s5)];
        // (mix2_s0/mix2_s1 already hold s2/s4 from above; the C++ re-assigns them
        // here but the values are unchanged — dead stores, dropped.)
        return self.mix2_p;
    }

    fn update(self: *SSE, bit: i32) void {
        sseUpdate(RAMP6, self.s6, self.su6_C1, self.su6_sw, self.su6_P, bit, M_sm6wrB);
        mixerUpdate(&self.x1[self.mix1], bit, self.mix1_s0, self.mix1_s1, M_x1wr, self.mix1_p);
        sseUpdate(RAMP7, self.s7, self.su7_C1, self.su7_sw, self.su7_P, bit, M_sm7wrB);
        mixerUpdate(&self.x2[self.mix2], bit, self.mix2_s0, self.mix2_s1, M_x2wr, self.mix2_p);
        self.advanceParse(bit);
    }

    /// -Dadaptive-depth support: advance ONLY the stream-parsing state (partial
    /// byte M_j, flag history M_ffl, previous byte M_pc — the tail block of
    /// `update` above, kept in sync with it) WITHOUT touching the learned s6/s7/
    /// x1/x2 tables or requiring a preceding `predict`. On skipped bits the
    /// cascade+SSE are frozen, but this state tracks the BYTE STREAM, not the
    /// model — freezing it would desync SSE's context selection for every later
    /// full-path bit. Dead code (zero cost) when the knob is off.
    pub fn advanceParse(self: *SSE, bit: i32) void {
        self.M_j += self.M_j + @as(u32, @intCast(bit));
        if (self.M_j >= 256) {
            self.M_ffl = (self.M_ffl * 2 + @intFromBool(self.M_pc >= 0x40)) & 255;
            self.M_pc = self.M_j & 255;
            self.M_j = 1;
        }
    }

    /// Predict: calibrate a probability (float in [0,1]).
    pub fn predict(self: *SSE, input: f32) f32 {
        @setFloatMode(.optimized);
        const discrete: i32 = @intFromFloat(1 + (1 - input) * 32766);
        const est = self.estimate(discrete);
        return 1 - (@as(f32, @floatFromInt(est - 1)) / 32766.0);
    }

    pub fn perceive(self: *SSE, bit: i32) void {
        @setFloatMode(.optimized);
        self.update(bit);
    }
};
