//! Logistic mixer + mixer-input, ported from cmix `mixer/mixer.{h,cpp}` and
//! `mixer/mixer-input.{h,cpp}`.
//!
//! Each Mixer keeps per-context weight vectors in a hash map and does online
//! gradient descent with a decaying learning rate, exactly as cmix does.
const std = @import("std");
const Sigmoid = @import("sigmoid.zig").Sigmoid;

pub const MixerInput = struct {
    inputs: []f32 = &.{},
    extra_inputs: std.ArrayList(f32) = .empty,
    sigmoid: *const Sigmoid,
    min: f32,
    max: f32,
    stretched_min: f32,
    stretched_max: f32,
    alloc: std.mem.Allocator,

    pub fn init(a: std.mem.Allocator, sigmoid: *const Sigmoid, eps: f32) MixerInput {
        return .{
            .sigmoid = sigmoid,
            .min = eps,
            .max = 1 - eps,
            .stretched_min = sigmoid.logit(0),
            .stretched_max = sigmoid.logit(1),
            .alloc = a,
        };
    }

    pub fn deinit(self: *MixerInput) void {
        if (self.inputs.len != 0) self.alloc.free(self.inputs);
        self.extra_inputs.deinit(self.alloc);
    }

    pub fn setNumModels(self: *MixerInput, num_models: usize) void {
        if (self.inputs.len != 0) self.alloc.free(self.inputs);
        self.inputs = self.alloc.alloc(f32, num_models) catch unreachable;
        for (self.inputs) |*v| v.* = 0.5;
    }

    pub fn setInput(self: *MixerInput, index: usize, p_in: f32) void {
        var p = p_in;
        if (p < self.min) p = self.min else if (p > self.max) p = self.max;
        self.inputs[index] = self.sigmoid.logit(p);
    }

    pub fn setStretchedInput(self: *MixerInput, index: usize, p_in: f32) void {
        var p = p_in;
        if (p > self.stretched_max) p = self.stretched_max else if (p < self.stretched_min) p = self.stretched_min;
        self.inputs[index] = p;
    }

    pub fn setExtraInput(self: *MixerInput, p_in: f32) void {
        var p = p_in;
        if (p > self.stretched_max) p = self.stretched_max else if (p < self.stretched_min) p = self.stretched_min;
        self.extra_inputs.append(self.alloc, p) catch unreachable;
    }

    pub fn clearExtraInputs(self: *MixerInput) void {
        self.extra_inputs.clearRetainingCapacity();
    }
};

const ContextData = struct {
    steps: u64 = 0,
    weights: []f32,
    extra_weights: []f32,
};

pub const Mixer = struct {
    mi: *MixerInput,
    extra_inputs: []f32, // this mixer's snapshot of extra inputs (size extra_input_size)
    p: f32 = 0.5,
    learning_rate: f32,
    context: *const u64,
    max_steps: u64 = 1,
    steps: u64 = 0,
    context_map: std.AutoHashMap(u32, *ContextData),
    alloc: std.mem.Allocator,

    pub fn init(a: std.mem.Allocator, mi: *MixerInput, context: *const u64, learning_rate: f32, extra_input_size: usize) Mixer {
        return .{
            .mi = mi,
            .extra_inputs = a.alloc(f32, extra_input_size) catch unreachable,
            .learning_rate = learning_rate,
            .context = context,
            .context_map = std.AutoHashMap(u32, *ContextData).init(a),
            .alloc = a,
        };
    }

    pub fn deinit(self: *Mixer) void {
        var it = self.context_map.valueIterator();
        while (it.next()) |v| {
            self.alloc.free(v.*.weights);
            self.alloc.free(v.*.extra_weights);
            self.alloc.destroy(v.*);
        }
        self.context_map.deinit();
        self.alloc.free(self.extra_inputs);
    }

    fn newContextData(self: *Mixer) *ContextData {
        const d = self.alloc.create(ContextData) catch unreachable;
        d.* = .{
            .weights = self.alloc.alloc(f32, self.mi.inputs.len) catch unreachable,
            .extra_weights = self.alloc.alloc(f32, self.extra_inputs.len) catch unreachable,
        };
        @memset(d.weights, 0);
        @memset(d.extra_weights, 0);
        return d;
    }

    fn getContextData(self: *Mixer) *ContextData {
        const limit: u64 = 10000;
        // cmix keys the weight map on `unsigned int` (u32), truncating the 64-bit
        // context (mixer.h). Match that so weight selection is faithful.
        const ctx: u32 = @truncate(self.context.*);
        const key: u32 = if (self.context_map.count() >= limit and !self.context_map.contains(ctx))
            0xDEADBEEF
        else
            ctx;
        const gop = self.context_map.getOrPut(key) catch unreachable;
        if (!gop.found_existing) gop.value_ptr.* = self.newContextData();
        return gop.value_ptr.*;
    }

    pub fn mix(self: *Mixer) f32 {
        const data = self.getContextData();
        var p: f32 = 0;
        const inputs = self.mi.inputs;
        for (inputs, data.weights) |inp, w| p += inp * w;
        self.p = p;
        const ev = self.mi.extra_inputs.items;
        var e: f32 = 0;
        for (0..self.extra_inputs.len) |i| {
            self.extra_inputs[i] = ev[i];
            e += self.extra_inputs[i] * data.extra_weights[i];
        }
        self.p += e;
        return self.p;
    }

    pub fn perceive(self: *Mixer, bit: i32) void {
        const data = self.getContextData();
        var decay: f32 = @floatCast(0.9 / std.math.pow(f64, 0.0000001 * @as(f64, @floatFromInt(self.steps)) + 0.8, 0.8));
        decay *= @floatCast(1.5 - (1.0 * @as(f64, @floatFromInt(data.steps))) / @as(f64, @floatFromInt(self.max_steps)));
        const update: f32 = decay * self.learning_rate * (Sigmoid.logistic(self.p) - @as(f32, @floatFromInt(bit)));
        self.steps += 1;
        data.steps += 1;
        if (data.steps > self.max_steps) self.max_steps = data.steps;
        const inputs = self.mi.inputs;
        for (data.weights, inputs) |*w, inp| w.* -= update * inp;
        for (data.extra_weights, self.extra_inputs) |*w, inp| w.* -= update * inp;
        if ((data.steps & 1023) == 0) {
            for (data.weights) |*w| w.* *= 1.0 - 3.0e-6;
            for (data.extra_weights) |*w| w.* *= 1.0 - 3.0e-6;
        }
    }
};
