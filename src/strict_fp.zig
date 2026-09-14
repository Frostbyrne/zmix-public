//! strict_fp — IEEE-exact division / square-root helpers for scopes that are
//! otherwise `@setFloatMode(.optimized)`.
//!
//! WHY THIS FILE EXISTS (P0, —
//!
//! Zig's `.optimized` float mode sets LLVM's full fast-math flag set, including
//! `arcp` (allow reciprocal) and `afn` (approximate functions). On x86-64 that
//! licenses the backend to lower
//!
//!     a / b          ->  VRCPPS/VRCPSS   (reciprocal ESTIMATE) + Newton refine
//!     1 / @sqrt(x)   ->  VRSQRTPS/VRSQRTSS (rsqrt ESTIMATE)    + Newton refine
//!
//! `RCPPS`/`RSQRTPS`/`RSQRTSS` are *approximations*. Intel specifies them only to
//! a relative error bound (<= 1.5 * 2^-12); the exact low mantissa bits come from
//! an on-die lookup table that is **implementation-defined** and differs between
//! vendors. Newton refinement reduces the error but is a fixed-point iteration on
//! the seed, so a different seed still yields a different final f32.
//!
//! In an arithmetic coder that is fatal: encoder and decoder must compute
//! bit-identical probabilities, so ONE differing mantissa bit desynchronises the
//! stream and corrupts the output. The Hutter judging machines are known to
//! include AMD parts (the committee's Ryzen 9 5900X, LTCB note 108), and the
//! `fx2-cmix-transformer` lineage records this exact failure destroying its July
//! 2026 submission on the committee's hardware.
//!
//! USE THESE HELPERS FOR EVERY DIVISION OR RECIPROCAL-SQRT WHOSE RESULT CAN REACH
//! A CODED PROBABILITY. They are `inline`, and `@setFloatMode` applies to the
//! scope in which an operation is *written*, so the strict flags survive inlining
//! into an `.optimized` caller — verified on emitted machine code, not assumed
//! (`vsqrtps`+`vdivps` replace `vrsqrtps`+FMA chain).
//!
//! Cost: a true `VDIVPS`/`VSQRTPS` instead of an estimate + 2-3 refinement FMAs.
//! Correctness is not negotiable here; see the report for the measured wall fee.
//!
//! ⚠ Do NOT "optimise" these back into the caller's scope. The whole point is the
//! float-mode boundary. `tools/recip_gate.sh` fails the build if a
//! reciprocal-estimate opcode reappears in a shipped artifact.

const std = @import("std");

/// `1.0` in `T`, for scalar or `@Vector` float types.
inline fn one(comptime T: type) T {
    return switch (@typeInfo(T)) {
        .vector => @splat(1.0),
        else => 1.0,
    };
}

/// IEEE-exact `a / b` (correctly rounded). Scalar or vector.
pub inline fn div(a: anytype, b: @TypeOf(a)) @TypeOf(a) {
    @setFloatMode(.strict);
    return a / b;
}

/// IEEE-exact `@sqrt(x)` (correctly rounded). Scalar or vector.
pub inline fn sqrt(x: anytype) @TypeOf(x) {
    @setFloatMode(.strict);
    return @sqrt(x);
}

/// IEEE-exact `1.0 / @sqrt(x)` — a correctly-rounded sqrt followed by a
/// correctly-rounded divide. NOT an rsqrt approximation.
pub inline fn rsqrt(x: anytype) @TypeOf(x) {
    @setFloatMode(.strict);
    const T = @TypeOf(x);
    return one(T) / @sqrt(x);
}

/// IEEE-exact `a / @sqrt(b)`.
pub inline fn divSqrt(a: anytype, b: @TypeOf(a)) @TypeOf(a) {
    @setFloatMode(.strict);
    return a / @sqrt(b);
}
