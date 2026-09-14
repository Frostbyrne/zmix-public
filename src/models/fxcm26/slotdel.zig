//! `-Dfxcm-slotmask` / `-Dfxcm-slotdelete` — the shared comptime arithmetic for the
//! CM-instance drop mask and its TRUE-DELETION arm.
//!
//! Lives in its own file (imported by `emit_sink.zig`, `cm3.zig`, `cm4.zig`,
//! `fxcm_v26.zig` and re-exported through `cm_fast.zig` for `predictor_lex.zig`)
//! purely to break the `cm3 -> emit_sink` import cycle: the sink's capacities are
//! themselves functions of the mask, so they cannot be derived inside cm3.
//!
//! MASK vs DELETE, in one line each:
//!   * MASK   — the masked instance still EMITS its `cn*(3+skip2)` exact-neutral
//!     zeros, so every consuming vector keeps full width. It is a FLOOR on the
//!     wall a removal buys (census phase-1 §6).
//!   * DELETE — the masked instance emits NOTHING and skips its cn/cxt_mask
//!     bookkeeping, and every consuming vector shrinks by `DEL_SLOTS`.
//!
//! ⛔ DELETE is NOT byte-identical to MASK, and cannot be made so: `mixer1`'s
//! `dotProductBias` truncates PAIRWISE (`(t0*w0 + t1*w1) >> 8`) and `mixer_lex`'s
//! `dotV` reduces into four independent FMA accumulators, so compacting the
//! survivors changes the integer truncation grouping and the f32 accumulation
//! grouping even though every removed term is exactly zero.

const std = @import("std");
const fxcmmask_options = @import("fxcmmask_options");

/// 52-bit CM-instance drop mask, parsed at COMPTIME so `MASK_ANY == false`
/// (the default) makes every guard comptime-dead => bit-identical stock build.
pub const SLOTMASK: u64 = blk: {
    const t = fxcmmask_options.fxcm_slotmask;
    break :blk std.fmt.parseInt(u64, t, 0) catch @compileError("-Dfxcm-slotmask must be a u64 literal (decimal or 0x hex)");
};
pub const MASK_ANY: bool = SLOTMASK != 0;

/// -Dfxcm-slotdelete: the true-deletion arm. Requires a non-zero mask; a build
/// that asks for the deletion with nothing masked is a recipe error.
pub const SLOTDEL: bool = blk: {
    const on = fxcmmask_options.fxcm_slotdelete;
    if (on and !MASK_ANY) @compileError("-Dfxcm-slotdelete=true requires a non-zero -Dfxcm-slotmask");
    break :blk on;
};

/// Slot count contributed by each of the 52 maskable CM instances, in MIX-CALL
/// order. cm3: cn*(3+skip2); cm4: cn*(2+skip2). Sums to 474 of the 494 layer-0
/// slots accounted by the census (the other 20: match 7, dcsm.mix 5, maps1/2 4,
/// sscm 3, rcm 1). Derived from the init args.
pub const MASK_SLOTS = [52]u16{ 12, 3, 3, 3, 8, 24, 4, 4, 16, 4, 9, 12, 18, 28, 9, 6, 4, 16, 24, 20, 8, 16, 8, 8, 4, 3, 4, 4, 4, 8, 4, 2, 12, 6, 20, 12, 3, 3, 2, 8, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9 };
pub const MASK_NAMES = [52][]const u8{ "cm_c2[0]", "cm_c2[1]", "cm_c2[2]", "cm_c2[3]", "cm_c2[4]", "cm_c2[5]", "cm_c2[6]", "cm_c2[7]", "cm_c2[8]", "cm_c4[0]", "cm_c4[1]", "cm_c4[2]", "cm_c4[4]", "cm_c[0]", "cm_c[1]", "cm_c[2]", "cm_c4[3]", "cm_c2[9]", "cm_c2[10]", "cm_c2[11]", "cm_c2[12]", "cm_c44", "cm_c2[13]", "cm_c[3]", "cm_c2[14]", "cm_c2[15]", "cm_c[4]", "cm_c[5]", "cm_c2[16]", "cm_c2[17]", "cm_c2[19]", "cm_c4[6]", "cm_c4[7]", "cm_c4[8]", "cm_c2[18]", "cm_c2[20]", "cm_cr[0]", "cm_cr[1]", "cm_cr[2]", "cmcr[0]", "cmcr[1]", "cmcr[2]", "cmcr[3]", "cmcr[4]", "cmcr[5]", "cmcr[6]", "cmcr[7]", "cmcr[8]", "cmcr2[0]", "cmcr2[1]", "cmcr2[2]", "cmcr2[3]" };

/// Number of MASKED CM instances.
pub const DEL_INSTANCES: usize = blk: {
    var n: usize = 0;
    for (0..52) |i| if ((SLOTMASK >> @intCast(i)) & 1 != 0) {
        n += 1;
    };
    break :blk n;
};

/// Emitted values removed per bit by the deletion (0 when the deletion is off).
/// Every removed emission leaves BOTH `in1` and `slots` (cm3/cm4 push through
/// `Sink.push`, which writes both banks), so one number serves both.
pub const DEL_SLOTS: usize = blk: {
    if (!SLOTDEL) break :blk 0;
    var n: usize = 0;
    for (0..52) |i| if ((SLOTMASK >> @intCast(i)) & 1 != 0) {
        n += MASK_SLOTS[i];
    };
    break :blk n;
};

/// Round a width up to a multiple of 16 — `mixer1.dotProductBias`/`trainBias`
/// process `round16(n)` lanes and read the input bank at that width, so EVERY
/// bank handed to a Mixer1 must itself be a multiple of 16 or the kernel reads
/// past the slice. Stock's 544 and 560 are both multiples of 16 for this reason.
inline fn r16(n: usize) usize {
    return (n + 15) & ~@as(usize, 15);
}

/// fxcm's `in1` bank (18 `mx_a` mixers), stock 544.
pub const IN1_W: usize = r16(544 - DEL_SLOTS);
/// fxcm's `slots` bank == `FxcmV26.NUM_MODELS` (4 `mx_a2` mixers), stock 560.
pub const SLOTS_W: usize = r16(560 - DEL_SLOTS);
/// Columns removed from predictor_lex's layer-0 fxcm block (== the `slots`
/// bank's shrink, so the two stay in lockstep). 0 when the deletion is off.
pub const DEL_COLS: usize = 560 - SLOTS_W;

comptime {
    // The deletion must not shrink a bank below what it still has to hold.
    std.debug.assert(IN1_W >= 544 - DEL_SLOTS);
    std.debug.assert(SLOTS_W >= 560 - DEL_SLOTS);
    std.debug.assert(IN1_W % 16 == 0 and SLOTS_W % 16 == 0);
    if (!SLOTDEL) {
        std.debug.assert(IN1_W == 544 and SLOTS_W == 560 and DEL_COLS == 0);
    }
}
