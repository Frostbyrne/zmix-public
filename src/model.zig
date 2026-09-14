//! Model interface, ported from cmix `models/model.h`.
//!
//! cmix uses a C++ virtual base with a `valarray<float>` of outputs. In Zig we
//! use an explicit vtable so any concrete model can be stored uniformly and
//! dispatched through `Predict`/`Perceive`/`ByteUpdate`.
const std = @import("std");

pub const Model = struct {
    ptr: *anyopaque,
    vtable: *const VTable,

    pub const VTable = struct {
        predict: *const fn (*anyopaque) []const f32,
        perceive: *const fn (*anyopaque, bit: i32) void,
        byte_update: *const fn (*anyopaque) void,
        num_outputs: *const fn (*anyopaque) usize,
    };

    pub fn predict(self: Model) []const f32 {
        return self.vtable.predict(self.ptr);
    }
    pub fn perceive(self: Model, bit: i32) void {
        self.vtable.perceive(self.ptr, bit);
    }
    pub fn byteUpdate(self: Model) void {
        self.vtable.byte_update(self.ptr);
    }
    pub fn numOutputs(self: Model) usize {
        return self.vtable.num_outputs(self.ptr);
    }
};

/// Helper to build a vtable for a concrete model type `T` whose methods are
/// `predict(*T) []const f32`, `perceive(*T, i32) void`, `byteUpdate(*T) void`,
/// `numOutputs(*T) usize`.
pub fn vtableFor(comptime T: type) *const Model.VTable {
    const gen = struct {
        fn predict(p: *anyopaque) []const f32 {
            return T.predict(@ptrCast(@alignCast(p)));
        }
        fn perceive(p: *anyopaque, bit: i32) void {
            T.perceive(@ptrCast(@alignCast(p)), bit);
        }
        fn byteUpdate(p: *anyopaque) void {
            T.byteUpdate(@ptrCast(@alignCast(p)));
        }
        fn numOutputs(p: *anyopaque) usize {
            return T.numOutputs(@ptrCast(@alignCast(p)));
        }
        const vt = Model.VTable{
            .predict = predict,
            .perceive = perceive,
            .byte_update = byteUpdate,
            .num_outputs = numOutputs,
        };
    };
    return &gen.vt;
}
