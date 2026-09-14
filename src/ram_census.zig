//! ram_census — an EXACT, output-neutral census of the engine's large allocations.
//!
//! Motivation: a RAM census DERIVED from constructor arithmetic re-read by hand
//! covers only about 88.6 % of anonymous memory, and a derived census cannot
//! see growth-on-demand structures (mixer rows),
//! cannot see what a flag line actually built, and cannot be reconciled against
//! a live `VmHWM` without a second act of arithmetic. This module measures it.
//!
//! Mechanism: `zero_alloc.alloc`'s big path (>= 1 MiB, the path every model
//! table takes) and a handful of explicit `note` calls at the growth sites
//! that bypass it (mixer rows, LSTM slabs) accumulate `bytes` into a table
//! keyed by `@returnAddress`. `dump` writes one line per call site. The
//! addresses resolve to file:line with `addr2line -e <binary>`.
//!
//! ⚠ COMPTIME-DEAD AT DEFAULT. `on` is `false` unless `-Dram-census=true`, so
//! every `note` body and the whole table DCE away; the knob lives in its own
//! options module (`ramcensus_opts`), never `model_opts` — the own-module rule:
//! a shared-`model_opts` entry costs ~40 B of `decomp_bin`, charged TWICE
//! under Form-1, even when dead. An own-module knob costs S exactly zero.
//!
//! ⚠ LAB-ONLY. This must not be relied on by any shipped code path.

const std = @import("std");
const opts = @import("ramcensus_options");

pub const on: bool = opts.ram_census;

/// Record `bytes` allocated at call site `ra`. No-op unless `-Dram-census`.
///
/// ⚠ EMITS ONE LINE PER EVENT rather than accumulating into a shared table.
/// The first version accumulated, and under-read the census by 8.4x: `zig build`
/// instantiates an imported module ONCE PER (target, optimize) PAIR, and the
/// engine compiles `root` at ReleaseFast while `cmfast`/`cmcold`/`lstmfast` are
/// ReleaseSmall — so each module graph got its OWN copy of the counter array and
/// `dump` from root printed only root's share (854 MiB of a 7,785 MiB run;
/// every ContextMap3 bank was invisible). Per-event emission has no shared state
/// and is therefore instance-count-proof. Aggregate offline; the volume is ~1e5
/// lines (mixer rows dominate) which is nothing next to a multi-hour leg.
pub inline fn note(bytes: usize, ra: usize) void {
    if (comptime !on) return;
    noteSlow(bytes, ra);
}

fn noteSlow(bytes: usize, ra: usize) void {
    std.debug.print("RAMALLOC\t{d}\t0x{x}\n", .{ bytes, ra });
}

/// End-of-run marker. Aggregation happens offline (see `note`). No-op unless
/// `-Dram-census`.
pub fn dump(tag: []const u8) void {
    if (comptime !on) return;
    std.debug.print("RAMALLOC-END\t{s}\n", .{tag});
}
