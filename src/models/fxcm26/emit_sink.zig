//! Direct mixer-input sink (PATH-2 instruction-bloat surgery, r3o/r3q).
//!
//! C++ fxcmv1 models write each prediction straight into the mixer-input
//! array inside `add`. The port buffered them per-model (`emitted_buf` +
//! `emitted_len`) and copied via `FxcmV26.drain` after each `mix` — a
//! store+load+loop-bookkeeping round-trip on ~470 values per bit. Models now
//! push through this sink exactly where they used to buffer: same values,
//! same order, same bounds semantics as add1/slot_push → bit-identical.
//!
//! Kept as explicit pointer fields (not *FxcmV26) so cm3/cm4/match_model
//! stay predictor-agnostic and their in-file oracle tests can drive them
//! with local arrays.

const slotdel = @import("slotdel.zig");
// -Dfxcm-slotdelete: both banks shrink by the deleted emissions (rounded up to a
// multiple of 16 — mixer1's kernels read `round16(n)` lanes of the bank). At the
// default mask these are exactly the stock 544 / 560.
pub const IN1_CAP: usize = slotdel.IN1_W; // FxcmV26.in1 capacity (mixer inputs 1)
pub const SLOT_CAP: usize = slotdel.SLOTS_W; // FxcmV26.NUM_MODELS (model slots)

pub const Sink = struct {
    in1: [*]i16,
    in1_n: *usize,
    slots: [*]i16,
    slot_n: *usize,

    /// Exact drain-body semantics for one ALREADY-clp'd value.
    pub inline fn push(s: *const Sink, v: i16) void {
        if (s.in1_n.* < IN1_CAP) {
            s.in1[s.in1_n.*] = v;
            s.in1_n.* += 1;
        }
        if (s.slot_n.* < SLOT_CAP) {
            s.slots[s.slot_n.*] = v;
            s.slot_n.* += 1;
        }
    }
};
