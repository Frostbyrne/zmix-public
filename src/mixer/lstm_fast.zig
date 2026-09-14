//! lstm_fast — root of the OPTIONAL ReleaseFast module holding the LSTM byte-mixer
//! compute (lstm.zig + lstm_layer.zig). Built at ReleaseFast when -Dlstmfast=true.
//!
//! Unlike the cmfast integer module (bit-identical, ~0 fleet wall because that path
//! is memory-bound), the LSTM is the COMPUTE-bound float path — ~47% of encode
//! wall (BPTT dependency chains). RF's -O3 scheduler/unroller on those FMA chains
//! converts to real wall (the "accumulator chains" gotcha), on top of what the
//! hand-@Vector bwvec pass already captured. FRESH-ARCHIVE lane: @setFloatMode(.optimized)
//! makes RF vs RS diverge (fast-math reassociation) — legal for a from-scratch archive.
//!
//! byte_mixer <-> Lstm crosses only []f32 slices (layout-trivial, no ptr bridges).
//! Module-private sigmoid/vendored_math copies (a file can't be in two modules).
pub const Lstm = @import("lstm.zig").Lstm;
// THE compact-output-history predicate (-Dlstm-compact-hist guard) — exported
// so tests can assert the fallback shapes against the module's own decision.
pub const compactHistory = @import("lstm.zig").compactHistory;
