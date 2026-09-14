//! state_table — bit-exact port of fxcm's StateTable (fxcmv1.cpp lines 241-352).
//!
//! Generates the 1024-byte `ns` next-state table from 7 integer params
//! (`b[0..5]` and `mdc`). Layout: `ns[state*4 + {0,1,2,3}]` = `{next-if-0, next-if-1,
//! n0 (zeros), n1 (ones)}`. Pure integer arithmetic, so it is bit-exact by
//! construction. The 6 real fxcm tables (STA1/2/4/5/6/7) are validated against a
//! compiled C++ oracle (see state_table_ref.rs / state_table_tables.bin golden).

const std = @import("std");

/// `N` in the C++: dimension of the `t` scratch table (and the x/y count bound).
const N: usize = 64;
/// `B` in the C++: number of usable `b[]` slots in num_states (`y` ranges 0..B).
const B: i32 = 5;

/// `t[x][y][k]`: k=0 -> base state index for bucket (x,y); k=1 -> count of states.
const TTable = [N][N][2]u8;

/// Truncate a (non-negative) i32 to u8 by keeping the low 8 bits — matches the
/// C++ `unsigned char` store and Rust `x as u8`.
inline fn u8trunc(v: i32) u8 {
    return @truncate(@as(u32, @bitCast(v)));
}

/// `num_states(x, y)`: how many distinct states the (x,y) count pair maps to (0, 1 or 2).
fn num_states(b: *const [6]i32, x: i32, y: i32) i32 {
    if (x < y) {
        return num_states(b, y, x);
    }
    // Short-circuit order matches C++: `b[y]` is only read once 0 <= y < B.
    if (x < 0 or y < 0 or x >= @as(i32, N) or y >= @as(i32, N) or y >= B or x >= b[@intCast(y)]) {
        return 0;
    }
    return 1 + @as(i32, @intFromBool(y > 0 and x + y < b[5]));
}

/// `discount(&x)`: new value of count `x` once the opposite bit is observed.
fn discount(mdc: i32, x: *i32) void {
    var y: i32 = 0;
    if (x.* > 2) {
        var i: i32 = 1;
        while (i < mdc) : (i += 1) {
            y += @intFromBool(x.* >= i);
        }
        x.* = y;
    }
}

/// `next_state(&x, &y, b)`: advance count pair (x,y) given observed bit `b` (0/1).
fn next_state(t: *const TTable, mdc: i32, x: *i32, y: *i32, b: i32) void {
    if (x.* < y.*) {
        // C++ swaps the references: next_state(y, x, 1-b).
        next_state(t, mdc, y, x, 1 - b);
    } else {
        if (b != 0) {
            y.* += 1;
            discount(mdc, x);
        } else {
            x.* += 1;
            discount(mdc, y);
        }
        while (t[@intCast(x.*)][@intCast(y.*)][1] == 0) {
            if (y.* < 2) {
                x.* -= 1;
            } else {
                // Integer division, truncating toward zero (operands are non-negative).
                x.* = @divTrunc(x.* *% (y.* - 1) + @divTrunc(y.*, 2), y.*);
                y.* -= 1;
            }
        }
    }
}

/// How many states the generator actually produced. Unused rows are all-zero,
/// and only state 0 legitimately has `x = y = 0` (with nonzero successors), so
/// the last nonzero row is the last state.
pub fn stateCount(ns: *const [1024]u8) usize {
    var i: usize = 255;
    while (i > 0) : (i -= 1) {
        if ((ns[i * 4] | ns[i * 4 + 1] | ns[i * 4 + 2] | ns[i * 4 + 3]) != 0) return i + 1;
    }
    return 1;
}

/// Scale the count-grid caps `b[0..4]` by `pct/100` — the fxcm generator's own
/// knob for how deep a count the automaton may represent, and exactly how the
/// reduced automata in the `state-width-exchange` offline study were produced.
/// `pct = 100` is stock.
fn scaleParams(p: *const [7]i32, pct: i32) [7]i32 {
    if (pct == 100) return p.*;
    var q = p.*;
    for (0..5) |i| {
        const v = @divTrunc(p[i] * pct + 50, 100);
        q[i] = if (v < 1) 1 else v;
    }
    return q;
}

/// C-PACK (`-Dcm3pack`): the WIDEST automaton from fxcm's own generator that fits
/// `max_states`. Searching the cap scale downward (rather than hand-picking one
/// PCT for all six tables) matters because the six param sets have different
/// grids — one PCT does NOT give one state count (PCT = 18 lands STA2/STA7 on 34
/// states, which needs 6 bits, not 5).
///
/// ★ Automata from this generator are SAFE BY CONSTRUCTION for cm3's eviction:
/// phase 1 assigns state indices in ascending `x + y`, so the index is monotone
/// in observation count — the property `cm3.bucket_get`'s argmin silently depends
/// on (findings §6.5). Asserted by the selftest below, together with closure.
pub fn generateFitted(params: *const [7]i32, max_states: usize) [1024]u8 {
    if (max_states >= 256) return generate(params);
    var pct: i32 = 100;
    while (pct > 1) : (pct -= 1) {
        const ns = generateScaled(params, pct);
        if (stateCount(&ns) <= max_states) return ns;
    }
    return generateScaled(params, 1);
}

/// Stock entry point (`pct = 100` ⇒ byte-identical to the C++ oracle).
pub fn generate(params: *const [7]i32) [1024]u8 {
    return generateScaled(params, 100);
}

/// Generate the 1024-byte `ns` table for one param set.
///
/// `params` = `[b0, b1, b2, b3, b4, b5, mdc]` (the `Init(s0..s6)` arguments).
pub fn generateScaled(params_in: *const [7]i32, pct: i32) [1024]u8 {
    const scaled = scaleParams(params_in, pct);
    const params: *const [7]i32 = &scaled;
    const b: [6]i32 = .{
        params[0], params[1], params[2], params[3], params[4], params[5],
    };
    const mdc = params[6];

    var ns = [_]u8{0} ** 1024;
    var t: TTable = std.mem.zeroes(TTable);

    // Phase 1: assign a contiguous state range to each reachable (x,y) bucket.
    var state: i32 = 0;
    var i: i32 = 0;
    while (i < 256) : (i += 1) {
        var y: i32 = 0;
        while (y <= i) : (y += 1) {
            const x = i - y;
            const n = num_states(&b, x, y);
            if (n != 0) {
                // (x,y) here are < N (else num_states returned 0); u8 truncation
                // of `state` matches the C++ `unsigned char` store exactly.
                t[@intCast(x)][@intCast(y)][0] = u8trunc(state);
                t[@intCast(x)][@intCast(y)][1] = @intCast(n);
                state += n;
            }
        }
    }

    // Phase 2: emit next-state transitions per state.
    state = 0;
    i = 0;
    while (i < @as(i32, N)) : (i += 1) {
        var y: i32 = 0;
        while (y <= i) : (y += 1) {
            const x = i - y;
            const count: i32 = t[@intCast(x)][@intCast(y)][1];
            var k: i32 = 0;
            while (k < count) : (k += 1) {
                var x0 = x;
                var y0 = y;
                var x1 = x;
                var y1 = y;
                next_state(&t, mdc, &x0, &y0, 0);
                next_state(&t, mdc, &x1, &y1, 1);

                const ns0: i32 = t[@intCast(x0)][@intCast(y0)][0];
                const ns1: i32 = @as(i32, t[@intCast(x1)][@intCast(y1)][0]) +
                    @as(i32, @intFromBool(t[@intCast(x1)][@intCast(y1)][1] > 1));

                ns[@intCast(state * 4)] = u8trunc(ns0);
                ns[@intCast(state * 4 + 1)] = u8trunc(ns1);
                ns[@intCast(state * 4 + 2)] = u8trunc(x);
                ns[@intCast(state * 4 + 3)] = u8trunc(y);

                if (state > 0xff or
                    t[@intCast(x)][@intCast(y)][1] == 0 or
                    t[@intCast(x0)][@intCast(y0)][1] == 0 or
                    t[@intCast(x1)][@intCast(y1)][1] == 0)
                {
                    return ns;
                }
                state += 1;
                if (state > 0xff) {
                    return ns;
                }
            }
        }
    }

    return ns;
}

/// Checksum used by the C++ oracle: `cs = cs*1000003 + byte`, u64 wrapping, init 0.
pub fn checksum(table: []const u8) u64 {
    var cs: u64 = 0;
    for (table) |byte| {
        cs = cs *% 1_000_003 +% @as(u64, byte);
    }
    return cs;
}

/// Init params (s0..s5 -> b[0..5], s6 -> mdc), order: STA1, STA2, STA4, STA5,
/// STA6, STA7 — the exact `statetable.Init(...)` calls in the C++ Predictor
/// ctor (fxcmv1.cpp:5865-5870). fxcm_v26.zig regenerates its STA tables from
/// these at startup instead of embedding the 6KB of golden .bin dumps.
pub const STA_PARAMS = [6][7]i32{
    .{ 28, 28, 31, 29, 23, 4, 17 },
    .{ 32, 28, 31, 28, 21, 5, 6 },
    .{ 31, 27, 30, 27, 24, 4, 27 },
    .{ 33, 31, 31, 24, 20, 4, 33 },
    .{ 28, 29, 30, 30, 23, 3, 22 },
    .{ 28, 29, 33, 23, 23, 6, 14 },
};

// ---------------------------------------------------------------------------
// Tests (ported from state_table.rs `mod tests`; goldens from the C++ oracle).
// ---------------------------------------------------------------------------

const testing = std.testing;

const PARAMS = STA_PARAMS;

const CHECKSUMS = [6]u64{
    8851997607252249485,
    16973730588641806077,
    5763221967892164071,
    14551327963078775525,
    13494823847770978599,
    9093215923013781898,
};

/// Golden 6*1024-byte ns tables (concatenated), from the C++ oracle.
const GOLDEN_TABLES = @embedFile("goldens/state_table_tables.bin");

test "full_tables_match_oracle" {
    for (0..6) |p| {
        const ns = generate(&PARAMS[p]);
        try testing.expectEqualSlices(u8, GOLDEN_TABLES[p * 1024 .. (p + 1) * 1024], &ns);
    }
}

test "checksums_match_oracle" {
    for (0..6) |p| {
        const ns = generate(&PARAMS[p]);
        try testing.expectEqual(CHECKSUMS[p], checksum(&ns));
    }
}

test "checksums_match_known_good" {
    const known = [6]u64{
        8851997607252249485, // STA1
        16973730588641806077, // STA2
        5763221967892164071, // STA4
        14551327963078775525, // STA5
        13494823847770978599, // STA6
        9093215923013781898, // STA7
    };
    for (0..6) |p| {
        const ns = generate(&PARAMS[p]);
        try testing.expectEqual(known[p], checksum(&ns));
    }
}

const Sample = struct { p: usize, idx: usize, val: u8 };

/// xorshift32 index-walk samples: (table_index, byte_index, value).
const SAMPLES = [_]Sample{
    .{ .p = 0, .idx = 677, .val = 140 },
    .{ .p = 0, .idx = 163, .val = 1 },
    .{ .p = 0, .idx = 196, .val = 59 },
    .{ .p = 0, .idx = 152, .val = 46 },
    .{ .p = 0, .idx = 904, .val = 145 },
    .{ .p = 0, .idx = 589, .val = 157 },
    .{ .p = 0, .idx = 797, .val = 140 },
    .{ .p = 0, .idx = 553, .val = 129 },
    .{ .p = 0, .idx = 679, .val = 1 },
    .{ .p = 0, .idx = 785, .val = 206 },
    .{ .p = 0, .idx = 504, .val = 135 },
    .{ .p = 0, .idx = 504, .val = 135 },
    .{ .p = 0, .idx = 160, .val = 49 },
    .{ .p = 0, .idx = 533, .val = 143 },
    .{ .p = 0, .idx = 710, .val = 0 },
    .{ .p = 0, .idx = 361, .val = 101 },
    .{ .p = 0, .idx = 658, .val = 3 },
    .{ .p = 0, .idx = 925, .val = 162 },
    .{ .p = 0, .idx = 969, .val = 151 },
    .{ .p = 0, .idx = 660, .val = 154 },
    .{ .p = 0, .idx = 703, .val = 19 },
    .{ .p = 0, .idx = 62, .val = 2 },
    .{ .p = 0, .idx = 780, .val = 154 },
    .{ .p = 0, .idx = 289, .val = 62 },
    .{ .p = 0, .idx = 726, .val = 19 },
    .{ .p = 0, .idx = 593, .val = 129 },
    .{ .p = 0, .idx = 616, .val = 163 },
    .{ .p = 0, .idx = 1017, .val = 151 },
    .{ .p = 0, .idx = 388, .val = 106 },
    .{ .p = 0, .idx = 891, .val = 4 },
    .{ .p = 0, .idx = 1018, .val = 30 },
    .{ .p = 0, .idx = 172, .val = 52 },
    .{ .p = 1, .idx = 734, .val = 20 },
    .{ .p = 1, .idx = 973, .val = 28 },
    .{ .p = 1, .idx = 285, .val = 28 },
    .{ .p = 1, .idx = 85, .val = 28 },
    .{ .p = 1, .idx = 727, .val = 0 },
    .{ .p = 1, .idx = 16, .val = 8 },
    .{ .p = 1, .idx = 773, .val = 45 },
    .{ .p = 1, .idx = 515, .val = 14 },
    .{ .p = 1, .idx = 780, .val = 205 },
    .{ .p = 1, .idx = 514, .val = 2 },
    .{ .p = 1, .idx = 166, .val = 0 },
    .{ .p = 1, .idx = 270, .val = 3 },
    .{ .p = 1, .idx = 496, .val = 134 },
    .{ .p = 1, .idx = 458, .val = 12 },
    .{ .p = 1, .idx = 327, .val = 0 },
    .{ .p = 1, .idx = 943, .val = 0 },
    .{ .p = 1, .idx = 22, .val = 1 },
    .{ .p = 1, .idx = 225, .val = 66 },
    .{ .p = 1, .idx = 542, .val = 13 },
    .{ .p = 1, .idx = 599, .val = 17 },
    .{ .p = 1, .idx = 459, .val = 3 },
    .{ .p = 1, .idx = 754, .val = 2 },
    .{ .p = 1, .idx = 834, .val = 2 },
    .{ .p = 1, .idx = 984, .val = 56 },
    .{ .p = 1, .idx = 232, .val = 47 },
    .{ .p = 1, .idx = 511, .val = 13 },
    .{ .p = 1, .idx = 940, .val = 243 },
    .{ .p = 1, .idx = 56, .val = 22 },
    .{ .p = 1, .idx = 864, .val = 47 },
    .{ .p = 1, .idx = 982, .val = 26 },
    .{ .p = 1, .idx = 924, .val = 56 },
    .{ .p = 1, .idx = 717, .val = 189 },
    .{ .p = 2, .idx = 214, .val = 4 },
    .{ .p = 2, .idx = 470, .val = 0 },
    .{ .p = 2, .idx = 695, .val = 17 },
    .{ .p = 2, .idx = 672, .val = 178 },
    .{ .p = 2, .idx = 719, .val = 1 },
    .{ .p = 2, .idx = 633, .val = 169 },
    .{ .p = 2, .idx = 912, .val = 238 },
    .{ .p = 2, .idx = 79, .val = 1 },
    .{ .p = 2, .idx = 869, .val = 227 },
    .{ .p = 2, .idx = 241, .val = 71 },
    .{ .p = 2, .idx = 1009, .val = 255 },
    .{ .p = 2, .idx = 513, .val = 139 },
    .{ .p = 2, .idx = 560, .val = 150 },
    .{ .p = 2, .idx = 82, .val = 3 },
    .{ .p = 2, .idx = 461, .val = 125 },
    .{ .p = 2, .idx = 187, .val = 7 },
    .{ .p = 2, .idx = 329, .val = 62 },
    .{ .p = 2, .idx = 518, .val = 16 },
    .{ .p = 2, .idx = 747, .val = 21 },
    .{ .p = 2, .idx = 847, .val = 3 },
    .{ .p = 2, .idx = 994, .val = 2 },
    .{ .p = 2, .idx = 191, .val = 8 },
    .{ .p = 2, .idx = 114, .val = 2 },
    .{ .p = 2, .idx = 505, .val = 136 },
    .{ .p = 2, .idx = 754, .val = 23 },
    .{ .p = 2, .idx = 764, .val = 201 },
    .{ .p = 2, .idx = 809, .val = 162 },
    .{ .p = 2, .idx = 1001, .val = 229 },
    .{ .p = 2, .idx = 403, .val = 2 },
    .{ .p = 2, .idx = 493, .val = 133 },
    .{ .p = 2, .idx = 376, .val = 103 },
    .{ .p = 2, .idx = 80, .val = 26 },
    .{ .p = 3, .idx = 542, .val = 2 },
    .{ .p = 3, .idx = 117, .val = 37 },
    .{ .p = 3, .idx = 12, .val = 7 },
    .{ .p = 3, .idx = 439, .val = 1 },
    .{ .p = 3, .idx = 474, .val = 16 },
    .{ .p = 3, .idx = 856, .val = 222 },
    .{ .p = 3, .idx = 618, .val = 3 },
    .{ .p = 3, .idx = 488, .val = 132 },
    .{ .p = 3, .idx = 11, .val = 1 },
    .{ .p = 3, .idx = 754, .val = 23 },
    .{ .p = 3, .idx = 701, .val = 185 },
    .{ .p = 3, .idx = 258, .val = 3 },
    .{ .p = 3, .idx = 726, .val = 19 },
    .{ .p = 3, .idx = 965, .val = 248 },
    .{ .p = 3, .idx = 205, .val = 62 },
    .{ .p = 3, .idx = 785, .val = 204 },
    .{ .p = 3, .idx = 10, .val = 0 },
    .{ .p = 3, .idx = 79, .val = 1 },
    .{ .p = 3, .idx = 481, .val = 131 },
    .{ .p = 3, .idx = 698, .val = 3 },
    .{ .p = 3, .idx = 537, .val = 144 },
    .{ .p = 3, .idx = 255, .val = 6 },
    .{ .p = 3, .idx = 434, .val = 15 },
    .{ .p = 3, .idx = 871, .val = 3 },
    .{ .p = 3, .idx = 41, .val = 16 },
    .{ .p = 3, .idx = 633, .val = 169 },
    .{ .p = 3, .idx = 186, .val = 1 },
    .{ .p = 3, .idx = 637, .val = 170 },
    .{ .p = 3, .idx = 361, .val = 101 },
    .{ .p = 3, .idx = 419, .val = 11 },
    .{ .p = 3, .idx = 569, .val = 112 },
    .{ .p = 3, .idx = 727, .val = 3 },
    .{ .p = 4, .idx = 934, .val = 0 },
    .{ .p = 4, .idx = 699, .val = 20 },
    .{ .p = 4, .idx = 526, .val = 4 },
    .{ .p = 4, .idx = 329, .val = 92 },
    .{ .p = 4, .idx = 466, .val = 16 },
    .{ .p = 4, .idx = 252, .val = 72 },
    .{ .p = 4, .idx = 439, .val = 3 },
    .{ .p = 4, .idx = 855, .val = 23 },
    .{ .p = 4, .idx = 566, .val = 4 },
    .{ .p = 4, .idx = 817, .val = 214 },
    .{ .p = 4, .idx = 454, .val = 2 },
    .{ .p = 4, .idx = 530, .val = 3 },
    .{ .p = 4, .idx = 671, .val = 1 },
    .{ .p = 4, .idx = 989, .val = 210 },
    .{ .p = 4, .idx = 99, .val = 2 },
    .{ .p = 4, .idx = 81, .val = 27 },
    .{ .p = 4, .idx = 780, .val = 184 },
    .{ .p = 4, .idx = 295, .val = 9 },
    .{ .p = 4, .idx = 181, .val = 55 },
    .{ .p = 4, .idx = 815, .val = 22 },
    .{ .p = 4, .idx = 587, .val = 0 },
    .{ .p = 4, .idx = 748, .val = 197 },
    .{ .p = 4, .idx = 173, .val = 53 },
    .{ .p = 4, .idx = 106, .val = 2 },
    .{ .p = 4, .idx = 720, .val = 190 },
    .{ .p = 4, .idx = 734, .val = 2 },
    .{ .p = 4, .idx = 950, .val = 3 },
    .{ .p = 4, .idx = 263, .val = 10 },
    .{ .p = 4, .idx = 878, .val = 23 },
    .{ .p = 4, .idx = 342, .val = 0 },
    .{ .p = 4, .idx = 740, .val = 184 },
    .{ .p = 4, .idx = 946, .val = 25 },
    .{ .p = 5, .idx = 455, .val = 13 },
    .{ .p = 5, .idx = 656, .val = 113 },
    .{ .p = 5, .idx = 1011, .val = 30 },
    .{ .p = 5, .idx = 389, .val = 108 },
    .{ .p = 5, .idx = 311, .val = 2 },
    .{ .p = 5, .idx = 257, .val = 74 },
    .{ .p = 5, .idx = 844, .val = 140 },
    .{ .p = 5, .idx = 753, .val = 139 },
    .{ .p = 5, .idx = 752, .val = 198 },
    .{ .p = 5, .idx = 927, .val = 25 },
    .{ .p = 5, .idx = 452, .val = 122 },
    .{ .p = 5, .idx = 636, .val = 169 },
    .{ .p = 5, .idx = 310, .val = 9 },
    .{ .p = 5, .idx = 682, .val = 4 },
    .{ .p = 5, .idx = 109, .val = 35 },
    .{ .p = 5, .idx = 54, .val = 4 },
    .{ .p = 5, .idx = 249, .val = 72 },
    .{ .p = 5, .idx = 510, .val = 14 },
    .{ .p = 5, .idx = 26, .val = 0 },
    .{ .p = 5, .idx = 358, .val = 8 },
    .{ .p = 5, .idx = 4, .val = 3 },
    .{ .p = 5, .idx = 176, .val = 52 },
    .{ .p = 5, .idx = 1011, .val = 30 },
    .{ .p = 5, .idx = 185, .val = 56 },
    .{ .p = 5, .idx = 557, .val = 109 },
    .{ .p = 5, .idx = 570, .val = 2 },
    .{ .p = 5, .idx = 46, .val = 1 },
    .{ .p = 5, .idx = 603, .val = 14 },
    .{ .p = 5, .idx = 706, .val = 20 },
    .{ .p = 5, .idx = 813, .val = 213 },
    .{ .p = 5, .idx = 987, .val = 28 },
    .{ .p = 5, .idx = 932, .val = 233 },
};

// C-PACK selftest. Two properties the decoupled bucket depends on, for every
// width the knob can select (16/32/64/128 states = 4/5/6/7 bits) and all six
// tables: (1) CLOSURE — every transition of a fitted table lands inside
// `[0, K)`, so a K-state machine really is representable in ceil(log2 K) bits;
// (2) INDEX-MONOTONICITY IN EVIDENCE — the state index is non-decreasing in
// `n0 + n1`, the undocumented property `cm3.bucket_get`'s argmin depends on
// (findings §6.5: a hill-climbed table that broke it was a BETTER predictor and
// a +1.61 % WORSE compressor). Negative control: the check has teeth — a
// reversed or shuffled numbering fails it, see the assertion on `prev` below.
test "cm3pack: fitted reduced automata are closed and index-monotone" {
    for (0..6) |p| {
        for ([_]usize{ 16, 32, 64, 128, 256 }) |maxs| {
            const ns = generateFitted(&PARAMS[p], maxs);
            const k = stateCount(&ns);
            try testing.expect(k <= maxs);
            try testing.expect(k >= 5); // at least the start + two levels
            var prev: i32 = -1;
            var s: usize = 0;
            while (s < k) : (s += 1) {
                try testing.expect(ns[s * 4] < k); // next-if-0 closed
                try testing.expect(ns[s * 4 + 1] < k); // next-if-1 closed
                const tot: i32 = @as(i32, ns[s * 4 + 2]) + @as(i32, ns[s * 4 + 3]);
                try testing.expect(tot >= prev); // monotone in evidence
                prev = tot;
            }
        }
    }
}

test "xorshift_samples_match_oracle" {
    // Regenerate all 6 tables, then verify the sampled (table, idx, value) golden.
    var tables: [6][1024]u8 = undefined;
    for (0..6) |p| {
        tables[p] = generate(&PARAMS[p]);
    }

    var sample_i: usize = 0;
    var p: u32 = 0;
    while (p < 6) : (p += 1) {
        var s: u32 = 0x12345678 +% (p *% 0x9e3779b1);
        var step: u32 = 0;
        while (step < 32) : (step += 1) {
            s ^= s << 13;
            s ^= s >> 17;
            s ^= s << 5;
            const idx: usize = @intCast(s % 1024);
            const g = SAMPLES[sample_i];
            sample_i += 1;
            try testing.expectEqual(@as(usize, p), g.p);
            try testing.expectEqual(idx, g.idx);
            try testing.expectEqual(tables[p][idx], g.val);
        }
    }
    try testing.expectEqual(SAMPLES.len, sample_i);
}
