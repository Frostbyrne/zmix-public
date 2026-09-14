//! zero_alloc — calloc-parity zeroed allocation for the big model tables.
//!
//! The C++ twin allocates every large table with calloc (`alloc`/`alloc1`,
//! fxcmv1_v26_cmixlex.cpp:139-151): the kernel hands back zero pages that are
//! only committed to RSS when first written. The straight port did
//! `a.alloc + @memset(0)`, which faults in every page up front — ~8 GB of RSS
//! at t=0 that the reference never pays, against the 10 GB Hutter RAM gate.
//!
//! `alloc` restores calloc semantics: big requests go straight to
//! `std.heap.page_allocator` (a fresh anonymous mmap — kernel-zeroed, lazily
//! committed, page-aligned); small ones use the caller's allocator plus an
//! explicit @memset (arena/test allocators recycle memory, so the zero must
//! be written there). Reads of never-written slots return 0 on both paths, so
//! output is bit-identical either way. Release with `free` so the big/small
//! routing matches the allocation.

const std = @import("std");
const ram_census = @import("ramcensus");

/// At or above this many bytes an allocation gets its own kernel-zeroed
/// mapping. Below it the page-granularity savings don't matter.
const BIG: usize = 1 << 20;

pub fn alloc(a: std.mem.Allocator, comptime T: type, n: usize) std.mem.Allocator.Error![]T {
    if (n * @sizeOf(T) >= BIG) {
        // LAB census hook — comptime-dead unless -Dram-census (see ram_census.zig).
        ram_census.note(n * @sizeOf(T), @returnAddress());
        const s = try std.heap.page_allocator.alloc(T, n);
        // THP is comptime-OFF (see thp_enabled below): with `const false` this
        // whole branch DCEs away — the record's 4KB-page regime, RAM-gate-safe.
        if (thp_enabled) {
            const bytes = std.mem.sliceAsBytes(s);
            std.posix.madvise(@alignCast(bytes.ptr), bytes.len, std.posix.MADV.HUGEPAGE) catch {};
        }
        return s;
    }
    const s = try a.alloc(T, n);
    @memset(s, std.mem.zeroes(T));
    return s;
}

pub fn free(a: std.mem.Allocator, s: anytype) void {
    const T = @typeInfo(@TypeOf(s)).pointer.child;
    if (s.len * @sizeOf(T) >= BIG) {
        std.heap.page_allocator.free(s);
    } else {
        a.free(s);
    }
}

test "zero_alloc small and big paths return zeroed memory" {
    const a = std.testing.allocator;
    const small = try alloc(a, u32, 16);
    defer free(a, small);
    for (small) |v| try std.testing.expectEqual(@as(u32, 0), v);

    const big = try alloc(a, u8, (1 << 20) + 123);
    defer free(a, big);
    try std.testing.expectEqual(@as(u8, 0), big[0]);
    try std.testing.expectEqual(@as(u8, 0), big[big.len - 1]);
}

/// Ask the kernel for transparent huge pages on an existing big allocation.
/// The fleet (and most servers) run THP=madvise: without this, GB-scale
/// random-access tables sit on 4KB pages and pay constant dTLB misses — a tax
/// the C++ record binary always pays (it never madvises; measured ~1.7% on the
/// zero_alloc subset alone). Content-neutral: bit-identical output either way.
///
/// PERMANENTLY OFF (comptime const) — the "encode-only THP" lever (c87e9f9/sm27)
/// was never shippable and is now closed. THP's 2MB at-fault commit inflates the
/// sparse hash banks' RSS ~3GB past the 4KB footprint (cm banks go 100% resident;
/// PPMd gets 0% THP either way). The Hutter 10GB RAM gate (rule 7) binds EVERY
/// program — the encoder as well as the decoder — so the ~3GB inflation blows the
/// gate on the encode run too, not just decode. There is therefore no path on
/// which THP can be enabled, and the intended `-e/-c` setter was never wired
/// (nowhere safe to wire it). `const false` makes the off state comptime-enforced
/// (can't be flipped on at runtime = footgun-proof) and lets the compiler DCE the
/// `if (thp_enabled)` madvise branch + the adviseHuge body out of the shipped
/// binary. To re-measure on a non-gated box, flip to `var` and set it at the
/// -e/-c entrypoint (experiments only — never in a submission build).
pub const thp_enabled: bool = false;

pub fn adviseHuge(mem: anytype) void {
    if (!thp_enabled) return;
    const T = @typeInfo(@TypeOf(mem)).pointer.child;
    const bytes = mem.len * @sizeOf(T);
    if (bytes < BIG) return;
    const addr = @intFromPtr(mem.ptr);
    const page = std.heap.page_size_min;
    const lo = std.mem.alignForward(usize, addr, page);
    const hi = std.mem.alignBackward(usize, addr + bytes, page);
    if (hi <= lo) return;
    const p: [*]align(std.heap.page_size_min) u8 = @ptrFromInt(lo);
    std.posix.madvise(p, hi - lo, std.posix.MADV.HUGEPAGE) catch {};
}
