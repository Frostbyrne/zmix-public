//! StateMap — maps a context/state to a 12-bit probability with an online update.
//!
//! Faithful, bit-exact port of `struct StateMap` (`src/cmix/src/models/fxcmv1.cpp`
//! lines 672-705): a non-counted map whose slot table `t[]` is seeded from a
//! next-state table `nn[]` (bit-history zero/one counts) and updated with a
//! fixed-rate (1/8192) leak toward the observed bit.
//!
//! (The counted/`dt[]`-decayed sibling `StateMap1` — fxcmv1.cpp:707-736 — lives in
//! the separate `state_map1` module.)
//!
//! In C++ `update` reads the global bit `x.y`; here the observed bit is threaded
//! in as `y`.
//!
//! Determinism notes:
//!   - `t[]` is `u32`; the additive leak wraps mod 2^32 -> `+%`, and the
//!     `int` delta `(y<<19)-pr1` reinterprets to `u32` (2's complement) via `@bitCast`.
//!   - `>>` reads from a `u32`, so all shifts are logical.
//!   - The `Init` expression `((n1<<20)/(n0+n1))<<12` is unsigned throughout; the
//!     final `<<12` truncates mod 2^32 exactly as the C++ `U32` does.

const std = @import("std");

/// `StateMap` (fxcmv1.cpp:672-705). Initialised from a next-state table.
pub const StateMap = struct {
    /// Number of contexts (power of two).
    n: usize,
    /// Context of the last prediction.
    cxt: usize,
    /// `cxt -> prediction in high 22 bits` (low bits unused by this variant).
    t: []u32,
    /// Last prediction (0..4095).
    pr: i32,

    /// `Init(n, nn)`: seed each `t[i]` from the bit-history zero/one counts read
    /// out of the next-state table `nn` (`next(i,2)`/`next(i,3)`, where
    /// `next(i,k) = nn[k + i*4]`). `nn` is only consulted here, so it is not kept.
    pub fn new(a: std.mem.Allocator, n: usize, nn: []const u8) !StateMap {
        std.debug.assert(n & (n - 1) == 0); // n must be a power of two
        std.debug.assert(nn.len >= n * 4); // nn must have >= n*4 entries
        const t = try a.alloc(u32, n);
        var i: usize = 0;
        while (i < n) : (i += 1) {
            const n0 = @as(u32, nn[2 + i * 4]) *% 3 +% 1;
            const n1 = @as(u32, nn[3 + i * 4]) *% 3 +% 1;
            t[i] = ((n1 << 20) / (n0 +% n1)) << 12;
        }
        return StateMap{ .n = n, .cxt = 0, .t = t, .pr = 2048 };
    }

    pub fn deinit(self: *StateMap, a: std.mem.Allocator) void {
        a.free(self.t);
    }

    /// Number of contexts.
    pub fn len(self: *const StateMap) usize {
        return self.n;
    }

    /// True if the map holds no contexts.
    pub fn is_empty(self: *const StateMap) bool {
        return self.n == 0;
    }

    // Note: Rust's `pr`/`cxt` getters are omitted — Zig disallows a method
    // sharing a name with a field, and the `pr`/`cxt` fields are public here.

    fn update(self: *StateMap, y: i32) void {
        std.debug.assert(y == 0 or y == 1);
        const p0 = self.t[self.cxt];
        const pr1: i32 = @intCast(p0 >> 13);
        const delta: u32 = @bitCast((y << 19) - pr1);
        self.t[self.cxt] = p0 +% delta;
    }

    /// `set(c)`: update the previous context with the observed bit `y`, switch to
    /// context `c`, and return the new prediction (0..4095).
    pub fn set(self: *StateMap, c: usize, y: i32) i32 {
        std.debug.assert(self.cxt < self.n);
        self.update(y);
        self.cxt = c;
        self.pr = @intCast(self.t[c] >> 20);
        return self.pr;
    }
};

// ==== Golden values from state_map_ref.rs (compiled C++ oracle) ====
const R_N: usize = 256;
const R_STEPS: usize = 200000;
const R_SM_PR_CS: u64 = 13687776665848295843;
const R_SM_T_CS: u64 = 11295245757259041237;
const R_SM_FINAL_PR: i32 = 655;
const R_SM_FINAL_CXT: usize = 182;
const R_SM_PR_S = [48]i32{
    2113, 1579, 2041, 3373, 962,  2181, 2533, 1945, 412,  3988, 1452, 3373, 728,  2352, 1444, 1387,
    1736, 547,  3646, 1579, 2329, 1595, 3123, 3103, 647,  2243, 3610, 2025, 1649, 913,  2380, 727,
    1834, 2795, 1786, 1473, 2304, 1598, 1887, 547,  3815, 1321, 2343, 3076, 2381, 3260, 2352, 2306,
};
const R_SM_T0 = [4]u32{ 2111651840, 1995776000, 2147483648, 1910505472 };

fn xs(s: *u32) u32 {
    s.* ^= s.* << 13;
    s.* ^= s.* >> 17;
    s.* ^= s.* << 5;
    return s.*;
}

fn fold(cs: u64, v: u32) u64 {
    return cs *% 1000003 +% @as(u64, v);
}

test "state_map matches oracle" {
    const a = std.testing.allocator;

    // deterministic next-state table (N*4 bytes)
    var nn = [_]u8{0} ** (R_N * 4);
    {
        var s: u32 = 0x1234_5678;
        for (&nn) |*slot| {
            slot.* = @truncate(xs(&s) >> 3);
        }
    }

    var sm = try StateMap.new(a, R_N, &nn);
    defer sm.deinit(a);

    // initial t[] cross-check
    try std.testing.expectEqual(R_SM_T0, [4]u32{ sm.t[0], sm.t[1], sm.t[2], sm.t[3] });

    var pr_cs: u64 = 0;
    var pr_s = [_]i32{0} ** 48;

    var s: u32 = 0x9e37_79b9;
    var k: usize = 0;
    while (k < R_STEPS) : (k += 1) {
        const rr = xs(&s);
        const y: i32 = @intCast(rr & 1);
        const c: usize = @intCast((rr >> 1) & (@as(u32, R_N) - 1));
        const p = sm.set(c, y);
        pr_cs = fold(pr_cs, @intCast(p));
        if (k < 48) {
            pr_s[k] = p;
        }
    }

    try std.testing.expectEqual(R_SM_PR_S, pr_s);
    try std.testing.expectEqual(R_SM_PR_CS, pr_cs);
    try std.testing.expectEqual(R_SM_FINAL_PR, sm.pr);
    try std.testing.expectEqual(R_SM_FINAL_CXT, sm.cxt);

    var t_cs: u64 = 0;
    for (sm.t) |v| {
        t_cs = fold(t_cs, v);
    }
    try std.testing.expectEqual(R_SM_T_CS, t_cs);
}
