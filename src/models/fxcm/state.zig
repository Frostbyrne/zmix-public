//! fxcm_v26 internal shared state, ported from reference/fxcmv1_v26.cpp globals.
//!
//! The reference uses file-scope globals (`BlockData x`, `model_predictions`,
//! prediction export). cmix only ever has one FXCM instance live, but zmix runs
//! predictors sequentially (compress then decompress, and multiple in tests), so
//! we keep these module-level but reset the scalar state in `reset` at each
//! FXCM construction. All large tables are per-instance heap allocations owned
//! by the model structs. Only one FXCM is ever active at a time.
const std = @import("std");
const prim = @import("primitives.zig");

pub const EXPORTED_MODELS: usize = 410 + 8 + 7; // 425
pub const NUM_EXTRA_PREDICTIONS: usize = 8 + 6; // 14
pub const NUM_PREDICTIONS: usize = EXPORTED_MODELS + NUM_EXTRA_PREDICTIONS; // 439

const conversion_factor: f32 = 1.0 / 4095.0;

pub var model_predictions: []f32 = &.{};
pub var prediction_index: usize = 0;

pub inline fn addPrediction(v: i32) void {
    model_predictions[prediction_index] = @as(f32, @floatFromInt(v)) * conversion_factor;
    prediction_index += 1;
}

pub inline fn resetPredictions() void {
    prediction_index = 0;
}

/// A mixer-input buffer. `add(p)` stores the stretched prediction `p` and
/// exports squash(p) as a probability to `model_predictions`.
pub const Inputs = struct {
    n: []i16 = &.{},
    ncount: usize = 0,

    pub inline fn add(self: *Inputs, p: i32) void {
        self.n[self.ncount] = @intCast(p);
        self.ncount += 1;
        addPrediction(prim.squash(p));
    }
};

pub const BlockData = struct {
    y: i32 = 0,
    c0: i32 = 1,
    c4: u32 = 0,
    bpos: i32 = 0,
    blpos: i32 = 0,
    bposshift: i32 = 0,
    c0shift_bpos: i32 = 0,
    mxInputs: [2]Inputs = .{ .{}, .{} },

    pub fn Init(self: *BlockData) void {
        self.y = 0;
        self.c0 = 1;
        self.c4 = 0;
        self.bpos = 0;
        self.blpos = 0;
        self.bposshift = 0;
        self.c0shift_bpos = 0;
    }
};

pub var x: BlockData = .{};

/// Allocate the prediction-export buffer and the two mixer-input buffers, and
/// reset scalar state. Call once per FXCM construction.
pub fn reset(a: std.mem.Allocator, mx0_size: usize, mx1_size: usize) void {
    prim.init();
    model_predictions = a.alloc(f32, NUM_PREDICTIONS) catch unreachable;
    for (model_predictions) |*v| v.* = 0.5;
    prediction_index = 0;
    x = .{};
    x.mxInputs[0] = .{ .n = a.alloc(i16, mx0_size) catch unreachable, .ncount = 0 };
    x.mxInputs[1] = .{ .n = a.alloc(i16, mx1_size) catch unreachable, .ncount = 0 };
    @memset(x.mxInputs[0].n, 0);
    @memset(x.mxInputs[1].n, 0);
}
