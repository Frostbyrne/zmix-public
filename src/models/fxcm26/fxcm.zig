//! zmix Model wrapper for FxcmV26 (the fxcm_v26 auxiliary text model).
//!
//! Mirrors the old V16 `models/fxcm/predictor.zig` FXCM wrapper contract so the
//! predictor consumes it identically:
//!   - `predict` returns a fixed-length slice of PROBABILITIES in (0,1);
//!     `MixerInput.setInput` logit-transforms each into layer-0.
//!   - `numOutputs` is CONSTANT (the predictor sizes layer-0 once at build).
//!   - the LAST output is fxcm's single best final prediction (`pr` as a
//!     probability), which feeds the auxiliary average (auxiliary_context).
//!
//! Lazy-eval like paq8/V16: `predict` returns the predictions computed by the
//! previous `perceive` (FxcmV26.perceive runs update+modelPrediction).
//!
//! Slot -> probability: fxcm_v26 stores every slot as a stretched logit clamped to
//! [-2047, 2047] (add1/add2/add4/add1_tail all push clipped stretched values), so
//! the per-slot probability is cmix's RawPredictionProbability: squash(raw)/4095.
//! Inactive slots (i >= active) map to raw 0 -> squash(0)/4095 == neutral ~0.5,
//! matching the slot-gate's `i < active ? raw : 0` convention. The final `pr` is
//! already a 12-bit probability (0..4095), converted directly as pr/4095.
//!
//! Instance-based: `create` builds a fresh FxcmV26, so every predictor gets a
//! reset-clean model (unlike the V16 module-global fxcm) — the key determinism
//! property for zmix's sequential (compress/decompress/test) predictors.

const std = @import("std");
const model_mod = @import("../../model.zig");
const Model = model_mod.Model;
const cmfast_m = @import("cmfast");
const FxcmV26 = cmfast_m.FxcmV26;
const fx_table_gen = @import("tables.zig");

const NUM_SLOTS: usize = 560; // FxcmV26.NUM_MODELS
const NUM_OUTPUTS: usize = NUM_SLOTS + 1; // 561: 560 slot-probs ++ final pr-prob
const INV_4095: f32 = 1.0 / 4095.0;

pub const FXCM = struct {
    fx: *FxcmV26,
    outputs: [NUM_OUTPUTS]f32,

    pub fn create(a: std.mem.Allocator) *FXCM {
        const self = a.create(FXCM) catch unreachable;
        const fx_strt = fx_table_gen.genStrt();
        const fx_sqt = fx_table_gen.genSqt();
        var fx_tbl_root = fx_table_gen.Tables.new(a) catch unreachable;
        const fx_tbl: cmfast_m.Tables = @as(*const cmfast_m.Tables, @ptrCast(&fx_tbl_root)).*;
        self.fx = FxcmV26.new(a, null, &fx_strt, &fx_sqt, fx_tbl) catch unreachable;
        for (&self.outputs) |*o| o.* = 0.5;
        return self;
    }

    /// Probabilities for the previous perceive's predictions: 560 slot-probs
    /// (RawPredictionProbability) followed by fxcm's final `pr` as a probability.
    pub fn predict(self: *FXCM) []const f32 {
        const fx = self.fx;
        var i: usize = 0;
        while (i < NUM_SLOTS) : (i += 1) {
            const raw: i32 = if (i < fx.active) @as(i32, fx.slots[i]) else 0;
            self.outputs[i] = @as(f32, @floatFromInt(fx.tables.squash(raw))) * INV_4095;
        }
        self.outputs[NUM_SLOTS] = @as(f32, @floatFromInt(fx.pr)) * INV_4095;
        return self.outputs[0..];
    }

    pub fn numOutputs(self: *FXCM) usize {
        _ = self;
        return NUM_OUTPUTS;
    }

    pub fn perceive(self: *FXCM, bit: i32) void {
        self.fx.perceive(bit);
    }

    pub fn byteUpdate(self: *FXCM) void {
        _ = self; // fxcm has no byte-level hook
    }

    pub fn model(self: *FXCM) Model {
        return .{ .ptr = self, .vtable = model_mod.vtableFor(FXCM) };
    }
};
