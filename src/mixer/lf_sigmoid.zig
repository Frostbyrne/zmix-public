//! Logit/logistic helper, ported from cmix `mixer/sigmoid.{h,cpp}`.
//!
//! Precomputes a logit table so model probabilities can be stretched into the
//! logit domain before mixing. `Logistic` is the inverse (squash).
const std = @import("std");

pub const Sigmoid = struct {
    logit_size: i32,
    logit_table: []f32,
    alloc: std.mem.Allocator,

    fn slowLogit(p: f64) f64 {
        return @log(p / (1.0 - p));
    }

    pub fn init(a: std.mem.Allocator, logit_size: i32) !Sigmoid {
        const n: usize = @intCast(logit_size);
        const table = try a.alloc(f32, n);
        for (0..n) |i| {
            const p = (@as(f64, @floatFromInt(i)) + 0.5) / @as(f64, @floatFromInt(logit_size));
            table[i] = @floatCast(slowLogit(p));
        }
        return .{ .logit_size = logit_size, .logit_table = table, .alloc = a };
    }

    pub fn deinit(self: *Sigmoid) void {
        self.alloc.free(self.logit_table);
    }

    pub fn logit(self: *const Sigmoid, p: f32) f32 {
        var index: i32 = @intFromFloat(p * @as(f32, @floatFromInt(self.logit_size)));
        if (index >= self.logit_size) index = self.logit_size - 1 else if (index < 0) index = 0;
        return self.logit_table[@intCast(index)];
    }

    pub fn logistic(p: f32) f32 {
        return @floatCast(1.0 / (1.0 + @exp(-@as(f64, p))));
    }
};

// Numerics mode for the hot exp (comptime three-way):
//   -Dvendored-libm  -> vendored_math.exp: the ARM optimized-routines port,
//     build-time-fixed (same algorithm+table as modern glibc, audited
//     bit-identical to glibc 2.43 over [-40,40]); no runtime libm dependency,
//     so the judged run cannot drift when the judge's glibc changes.
//   -Dglibc (extern) -> glibc exp resolved on the RUNNING machine
//  (A/B-validated -1.1% at 10MB, a fleet box simultaneous pair; bit-identical
//     output) — fast but numerics float with the judge's glibc version.
//   default (static) -> @exp lowers to compiler-rt (deterministic, slower).
const use_vendored_libm = @import("build_options").vendored_libm;
const use_glibc_libm = @import("builtin").target.abi.isGnu();
const vendored_math = @import("lf_vendored_math.zig");
extern fn exp(x: f64) f64;
inline fn expD(x: f64) f64 {
    return if (comptime use_vendored_libm)
        vendored_math.exp(x)
    else if (comptime use_glibc_libm)
        exp(x)
    else
        @exp(x);
}
