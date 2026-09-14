//! Port of `struct SmallStationaryContextMap` from
//! `src/cmix/src/models/fxcmv1.cpp` (lines ~831-863).
//!
//! Map for modelling contexts of (nearly-)stationary data. The context is
//! looked up directly. For each bit modelled, a 16-bit prediction is stored
//! and adapted with an SSE-style update whose rate is controlled by the caller.
//!
//! - `bits_of_context`: how many bits of the context to use (higher bits discarded).
//! - `input_bits` (1..=8): how many bits of input are modelled per context.
//!
//! Uses `(2^bits_of_context) * (2^input_bits - 1)` `u16` slots of memory.
//!
//! The C++ struct read the global bit `x.y` and pushed two values into the
//! global mixer. Here `mix` takes the bit `y` explicitly and *returns* the two
//! mixer inputs `(out1, out2)`. `stretch` is threaded in via the shared `Tables`.

const std = @import("std");
// In the `cmfast` module: layout-private Tables copy (see cmf_tables.zig);
// the root module's instance crosses as a @ptrCast'd pointer.
const Tables = @import("cmcold").Tables;

/// `SmallStationaryContextMap` — direct-lookup stationary bit predictor.
pub const SmallStationaryContextMap = struct {
    data: []u16,
    context: i32,
    mask: i32,
    stride: i32,
    b_count: i32,
    b_total: i32,
    b: i32,
    n: i32,
    /// Index into `data` of the slot that produced the last prediction
    /// (the C++ `cp` pointer). Persists across `set` calls, matching C++.
    cp: usize,

    /// `Init(BitsOfContext, InputBits = 8)`.
    pub fn new(a: std.mem.Allocator, bits_of_context: i32, input_bits: i32) !SmallStationaryContextMap {
        std.debug.assert(input_bits > 0 and input_bits <= 8);
        const mask = (@as(i32, 1) << @intCast(bits_of_context)) - 1;
        const stride = (@as(i32, 1) << @intCast(input_bits)) - 1;
        // N = (1<<BitsOfContext) * ((1<<InputBits)-1)
        const nval: u64 = (@as(u64, 1) << @intCast(bits_of_context)) * ((@as(u64, 1) << @intCast(input_bits)) - 1);
        const n: i32 = @bitCast(@as(u32, @truncate(nval)));

        const data = try a.alloc(u16, @intCast(n));
        @memset(data, 0x7FFF);

        return SmallStationaryContextMap{
            .data = data,
            .context = 0,
            .mask = mask,
            .stride = stride,
            .b_count = 0,
            .b_total = input_bits,
            .b = 0,
            .n = n,
            .cp = 0,
        };
    }

    pub fn deinit(self: *SmallStationaryContextMap, a: std.mem.Allocator) void {
        a.free(self.data);
    }

    /// `set(ctx)` — select the context block and reset the bit path.
    pub fn set(self: *SmallStationaryContextMap, ctx: u32) void {
        // Context = (ctx & Mask) * Stride;  (computed in u32, then to int)
        self.context = @bitCast((ctx & @as(u32, @bitCast(self.mask))) *% @as(u32, @bitCast(self.stride)));
        self.b_count = 0;
        self.b = 0;
    }

    /// `mix(r)` — adapt the previous slot toward bit `y`, then select the next
    /// slot and return the two mixer inputs `(out1, out2)`.
    ///
    /// `y` is the resolving bit (0 or 1) — the global `x.y` in the C++.
    pub fn mix(self: *SmallStationaryContextMap, tables: *const Tables, y: i32, r: i32) struct { i32, i32 } {
        const rate: i32 = r + @import("build_options").sscm_base_rate; // -Dsscm-base-rate (C9)
        const multiplier: i32 = 1;
        const divisor: i32 = 4;

        // *cp += ((y<<16) - (*cp) + (1<<(rate-1))) >> rate;
        const cur: i32 = @intCast(self.data[self.cp]);
        const delta: i32 = (((y << 16) -% cur) +% (@as(i32, 1) << @intCast(rate - 1))) >> @intCast(rate);
        self.data[self.cp] = @truncate(@as(u32, @bitCast(cur +% delta)));

        // B += (y && B>0);
        self.b +%= @intFromBool(y != 0 and self.b > 0);

        // cp = &Data[Context+B];
        self.cp = @intCast(self.context + self.b);

        const prediction: i32 = @as(i32, self.data[self.cp] >> 4);
        const out1: i32 = @divTrunc(@as(i32, tables.stretch(prediction)) * multiplier, divisor);
        const out2: i32 = @divTrunc((prediction - 2048) * multiplier, divisor * 2);

        // bCount++; B+=B+1;
        self.b_count += 1;
        self.b +%= self.b +% 1;
        if (self.b_count == self.b_total) {
            self.b_count = 0;
            self.b = 0;
        }

        return .{ out1, out2 };
    }

    /// Number of `u16` slots (`N`).
    pub fn len(self: *const SmallStationaryContextMap) usize {
        return @intCast(self.n);
    }

    /// Whether the map is empty (always false in practice).
    pub fn is_empty(self: *const SmallStationaryContextMap) bool {
        return self.n == 0;
    }
};

// ===========================================================================
// Golden values captured from the compiled C++ oracle
// (`fxport/small_stationary_context_map/oracle.cpp`, `g++ -std=c++14 -O2`).
// Driver: xorshift32 seed 0x12345678, single continuous stream. Configs in
// order (bits_of_context, input_bits, nbytes):
//   (2,8,2048) sampled, (8,8,4096), (9,8,4096), (7,8,4096), (5,4,4096).
// ===========================================================================

/// Per-config checksum over the final `Data` table (i64, wrapping).
const DATA_CS = [5]i64{
    -5406050387947743742, 1419663051340017936, -1526657626494611439, -5257918236074178091, 2138264317395037620,
};

/// Total number of (out1,out2) values folded into `REC_CS`.
const REC_N: u64 = 262144;

/// Rolling checksum over every (out1,out2) across all configs.
const REC_CS: i64 = -7638766385932795397;

/// First 64 `out1` values from the sampled config (2,8,2048).
const SAMPLE1 = [64]i32{
    0, 0, 0, 0, 0, 0, 0, 0, -1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, -1, 1, 0, 0,
};

/// First 64 `out2` values from the sampled config (2,8,2048).
const SAMPLE2 = [64]i32{
    0, 0, 0, 0, 0, 0, 0, 0, -2, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, -2, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1, 2, 0, 1, -2, 1, 0, 0,
};

// Deterministic xorshift32 driver (identical to the C++ oracle).
fn xs(s: *u32) u32 {
    var v = s.*;
    v ^= v << 13;
    v ^= v >> 17;
    v ^= v << 5;
    s.* = v;
    return v;
}

fn data_checksum(m: *const SmallStationaryContextMap) i64 {
    var cs: i64 = 0;
    for (m.data) |v| {
        cs = cs *% 1000003 +% @as(i64, v);
    }
    return cs;
}

test "sscm matches cpp oracle" {
    const a = std.testing.allocator;
    var tables = try @import("cmcold").tables.newFromGoldensForTests(a);
    defer tables.deinit(a);

    var s: u32 = 0x1234_5678;

    var rec_cs: i64 = 0;
    var rec_n: u64 = 0;
    var sample1 = [_]i32{0} ** 64;
    var sample2 = [_]i32{0} ** 64;
    var sample_n: usize = 0;

    // (bits_of_context, input_bits, nbytes, sampled)
    const Config = struct { bits: i32, inbits: i32, nbytes: i32, sampled: bool };
    const configs = [5]Config{
        .{ .bits = 2, .inbits = 8, .nbytes = 2048, .sampled = true },
        .{ .bits = 8, .inbits = 8, .nbytes = 4096, .sampled = false },
        .{ .bits = 9, .inbits = 8, .nbytes = 4096, .sampled = false },
        .{ .bits = 7, .inbits = 8, .nbytes = 4096, .sampled = false },
        .{ .bits = 5, .inbits = 4, .nbytes = 4096, .sampled = false },
    };

    for (configs, 0..) |cfg, ci| {
        var m = try SmallStationaryContextMap.new(a, cfg.bits, cfg.inbits);
        defer m.deinit(a);
        var byte: i32 = 0;
        while (byte < cfg.nbytes) : (byte += 1) {
            const ctx = xs(&s);
            m.set(ctx);
            var bit: i32 = 0;
            while (bit < cfg.inbits) : (bit += 1) {
                const y: i32 = @intCast(xs(&s) & 1);
                const r: i32 = @intCast(xs(&s) & 1);
                const out = m.mix(&tables, y, r);
                const out1 = out[0];
                const out2 = out[1];
                if (cfg.sampled and sample_n < 64) {
                    sample1[sample_n] = out1;
                    sample2[sample_n] = out2;
                    sample_n += 1;
                }
                rec_cs = rec_cs *% 1000003 +% @as(i64, out1);
                rec_cs = rec_cs *% 1000003 +% @as(i64, out2);
                rec_n += 2;
            }
        }
        const dcs = data_checksum(&m);
        try std.testing.expectEqual(DATA_CS[ci], dcs);
    }

    try std.testing.expectEqual(REC_N, rec_n);
    try std.testing.expectEqual(REC_CS, rec_cs);
    try std.testing.expectEqual(@as(usize, 64), sample_n);
    try std.testing.expectEqual(SAMPLE1, sample1);
    try std.testing.expectEqual(SAMPLE2, sample2);
}
