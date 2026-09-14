//! cm_cold — root of the ReleaseSmall companion module under `cmfast`: the
//! per-byte/per-word PARSING layer + the shared base files. These are the
//! measured size-heavy / instruction-light files of the fxcm26 predictor
//! (-O3 on the parse layer cost ~+9K raw .text for negligible retired-
//! instruction benefit at 1-per-byte call rates; see
//!  size-attribution table). The hot
//! per-bit compute stays in `cmfast` (ReleaseFast), which imports this
//! module — acyclic: nothing here calls back up into the predictor.
//!
//! Shared base files (byte_core/cmf_tables/cmf_zero_alloc) live HERE so both
//! modules can use them via these re-exports (a file cannot belong to two
//! modules).

pub const byte_core = @import("byte_core.zig");
pub const X = byte_core.X;
pub const tables = @import("cmf_tables.zig");
pub const Tables = tables.Tables;
pub const zero_alloc = @import("cmf_zero_alloc.zig");
pub const state_table = @import("state_table.zig");
pub const parse_byte = @import("parse_byte.zig");
pub const ParseByte = parse_byte.ParseByte;
pub const bcs = @import("bcs_cold.zig");
// words/sentence are HOT (auto-vectorized O(n^2) scans, ~size-free at -O3):
pub const WordsContext = @import("cmfast").WordsContext;
pub const SentenceContext = @import("cmfast").SentenceContext;

test {
    // Collect this module's in-file tests (`zig build test` runs this module
    // as its own test root; see build.zig).
    _ = @import("byte_core.zig");
    _ = @import("parse_byte.zig");
    _ = @import("state_table.zig");
    _ = @import("stemmer.zig");
    _ = @import("bracket_context.zig");
    _ = @import("column_context.zig");
    _ = @import("decoded_buffer.zig");
}
