//! Unit tests for the zmix (cmix) port.
const std = @import("std");

// Pin the exp/log family exactly like the ship roots (detm_math strong-symbol
// exports): without this, `zig build test` on a -Dglibc build binds @exp/expf
// to the HOST libm and the FP-tolerance oracle gates fail on boxes whose glibc
// differs from the goldens' (3 gates fail on a fleet box glibc
// while runner outputs stay byte-identical). With it, the test gate is valid
// on every fleet box.
comptime {
    _ = @import("detm_math.zig");
}
comptime {
    _ = @import("hero_head.zig"); // pull in the HERO head unit tests
}
comptime {
    // pull in ByteMixer.selectTopK's determinism + layout gates
    // (-Dlstm-aux-sparse). They are comptime-parameterised on K, so they run in
    // EVERY build, including stock ones where the selector is comptime-dead.
    // (lstm_layer.zig's dotGather/axpyScatter gates cannot be pulled in here —
    // that file belongs to the `lstmfast` module and a file may live in only one
    // module. build.zig gives lstmfast its own test compilation, exactly as it
    // already does for cmfast/cmcold.)
    _ = @import("mixer/byte_mixer.zig");
}
comptime {
    // -Dmixer1-rls: the layer-1 second-order recursion. Its tests are NOT
    // comptime-parameterised on the knob — they construct Rls directly — so they
    // run in EVERY build, including stock ones where the lever is comptime-dead.
    // Without this pull the file is only reachable through predictor_lex.zig,
    // which is not in the test root's graph: the three tests existed and `zig
    // build test` never ran them (ref and arm both reported 187 passed / 1
    // skipped, identical, which is how the omission was caught).
    _ = @import("mixer/mixer1_rls.zig");
}
const Predictor = @import("predictor.zig").Predictor;
const coder = @import("coder.zig");
const states = @import("states.zig");
const Sigmoid = @import("sigmoid.zig").Sigmoid;
const Lstm = @import("lstmfast").Lstm; // ReleaseFast module (-Dlstmfast)
const fxcm_prim = @import("models/fxcm/primitives.zig");
const fxcm_tables = @import("models/fxcm/tables.zig");

test "fxcm tables extracted correctly (checksums)" {
    var sum: u64 = 0;
    for (fxcm_tables.STA1) |v| sum += v;
    try std.testing.expectEqual(@as(u64, 65911), sum);
    sum = 0;
    for (fxcm_tables.STA5) |v| sum += v;
    try std.testing.expectEqual(@as(u64, 70747), sum);
    sum = 0;
    for (fxcm_tables.WRT_T) |v| sum += v;
    try std.testing.expectEqual(@as(u64, 1213), sum);
}

test "fxcm Mixer1 converges toward the target bit" {
    fxcm_prim.init();
    const Mixer1 = @import("models/fxcm/core.zig").Mixer1;
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const tx = a.alignedAlloc(i16, .@"32", 16) catch unreachable;
    for (tx) |*v| v.* = 512; // fixed positive inputs
    var mx: Mixer1 = .{};
    mx.init(a, 1, 256, 0, 6);
    mx.setTxWx(16, tx.ptr);
    mx.cxt = 0;
    var pr: i32 = 2048;
    for (0..400) |_| {
        pr = mx.p();
        mx.update(1); // target bit is always 1
    }
    // predicting 1 -> pr should move well above the 2048 midpoint
    try std.testing.expect(pr > 3000);
}

test "fxcm squash/stretch monotonic" {
    fxcm_prim.init();
    try std.testing.expectEqual(@as(i32, 2048), fxcm_prim.squash(0));
    var prev: i32 = -1;
    var d: i32 = -2047;
    while (d <= 2047) : (d += 1) {
        const p = fxcm_prim.squash(d);
        try std.testing.expect(p >= prev);
        prev = p;
    }
    // stretch(squash(x)) recovers x reasonably mid-range
    try std.testing.expect(@abs(@as(i32, fxcm_prim.stretch(fxcm_prim.squash(600))) - 600) < 64);
}

test "lstm learns a periodic pattern" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var prng = std.Random.DefaultPrng.init(0xDEADBEEF);
    var rng = prng.random();
    const K = 4;
    const lstm = Lstm.init(a, K, K, 30, 1, 8, 0.03, 10, &rng);
    const zeros = [_]f32{0} ** K;
    const pattern = [_]u32{ 0, 1, 2, 3 };
    var correct: usize = 0;
    var total: usize = 0;
    var step: usize = 0;
    while (step < 400) : (step += 1) {
        lstm.setInput(&zeros);
        const dist = lstm.perceive(pattern[step % K]);
        if (step > 300) {
            const expected = pattern[(step + 1) % K];
            var argmax: usize = 0;
            for (dist, 0..) |p, i| {
                if (p > dist[argmax]) argmax = i;
            }
            if (argmax == expected) correct += 1;
            total += 1;
        }
    }
    try std.testing.expect(correct * 2 >= total); // well above 25% chance
}

// -Dlstm-compact-hist adoption guard (fifth-e-compiler-layout-findings-jul-23
// §Candidate A): compact mode iff knob ON AND one layer AND even horizon —
// decided ONCE by lstmfast.compactHistory. This test runs under BOTH knob
// settings: shapes for the supported case follow the module's own predicate;
// odd-horizon and multi-layer instances must ALWAYS get the stock
// horizon-deep ring (output_layer_.len == horizon, empty output_error_), and
// the odd-horizon instance must run perceive/BPTT/predict across many ring
// wraps on that stock path (it would index the absent ping-pong pair — an
// out-of-bounds — if compact mode engaged half-way).
test "lstm compact-history guard: odd horizon / multi-layer fall back to the stock ring" {
    const compactHistory = @import("lstmfast").compactHistory;
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var prng = std.Random.DefaultPrng.init(0xDEADBEEF);
    var rng = prng.random();
    const K = 4;

    // Even horizon, one layer: the supported compact case — ping-pong pair +
    // horizon-deep output_error_ iff the knob is on, stock ring otherwise.
    const l8 = Lstm.init(a, K, K, 30, 1, 8, 0.03, 10, &rng);
    try std.testing.expectEqual(@as(usize, if (compactHistory(1, 8)) 2 else 8), l8.output_layer_.len);
    try std.testing.expectEqual(@as(usize, if (compactHistory(1, 8)) 8 else 0), l8.output_error_.len);

    // ODD horizon, one layer: never compact, regardless of the knob.
    try std.testing.expect(!compactHistory(1, 7));
    const l7 = Lstm.init(a, K, K, 30, 1, 7, 0.03, 10, &rng);
    try std.testing.expectEqual(@as(usize, 7), l7.output_layer_.len);
    try std.testing.expectEqual(@as(usize, 0), l7.output_error_.len);

    // Two layers: never compact, regardless of the knob.
    try std.testing.expect(!compactHistory(2, 8));
    const l2 = Lstm.init(a, K, K, 30, 2, 8, 0.03, 10, &rng);
    try std.testing.expectEqual(@as(usize, 8), l2.output_layer_.len);
    try std.testing.expectEqual(@as(usize, 0), l2.output_error_.len);

    // Functional: drive the odd-horizon instance through 400 steps (57 full
    // horizon-7 wraps) of the stock path and require it still learns the
    // periodic pattern — same oracle as "lstm learns a periodic pattern".
    const zeros = [_]f32{0} ** K;
    const pattern = [_]u32{ 0, 1, 2, 3 };
    var correct: usize = 0;
    var total: usize = 0;
    var step: usize = 0;
    while (step < 400) : (step += 1) {
        l7.setInput(&zeros);
        const dist = l7.perceive(pattern[step % K]);
        if (step > 300) {
            const expected = pattern[(step + 1) % K];
            var argmax: usize = 0;
            for (dist, 0..) |p, i| {
                if (p > dist[argmax]) argmax = i;
            }
            if (argmax == expected) correct += 1;
            total += 1;
        }
    }
    try std.testing.expect(correct * 2 >= total); // well above 25% chance
}

test "sigmoid logit/logistic round trip" {
    var s = try Sigmoid.init(std.testing.allocator, 100001);
    defer s.deinit();
    try std.testing.expectApproxEqAbs(@as(f32, 0.5), Sigmoid.logistic(0), 1e-6);
    // logistic(logit(p)) ~= p for mid-range p
    var p: f32 = 0.1;
    while (p <= 0.9) : (p += 0.1) {
        const back = Sigmoid.logistic(s.logit(p));
        try std.testing.expect(@abs(back - p) < 0.02);
    }
}

test "run map init probability symmetric" {
    const rm = states.RunMap.init();
    // state 0 and near-128 boundaries produce valid probabilities.
    try std.testing.expect(rm.initProbability(0) > 0 and rm.initProbability(0) <= 0.5);
    try std.testing.expect(rm.initProbability(200) > 0.5);
}

fn roundTrip(gpa: std.mem.Allocator, data: []const u8) !void {
    var vocab: [256]bool = .{true} ** 256;
    if (data.len >= 10000) {
        vocab = .{false} ** 256;
        for (data) |c| vocab[c] = true;
    }
    var comp: std.ArrayList(u8) = .empty;
    defer comp.deinit(gpa);
    {
        const pred = try Predictor.init(gpa, &vocab);
        defer pred.deinit();
        var enc = coder.Encoder.init(gpa, &comp, pred);
        for (data) |c| {
            var b: i32 = 7;
            while (b >= 0) : (b -= 1) try enc.encode((@as(i32, c) >> @intCast(b)) & 1);
        }
        try enc.flush();
    }
    var decomp: std.ArrayList(u8) = .empty;
    defer decomp.deinit(gpa);
    {
        const pred = try Predictor.init(gpa, &vocab);
        defer pred.deinit();
        var dec = coder.Decoder.init(comp.items, pred);
        for (0..data.len) |_| {
            var c: i32 = 0;
            for (0..8) |_| c += c + dec.decode();
            try decomp.append(gpa, @intCast(c & 0xff));
        }
    }
    try std.testing.expectEqualSlices(u8, data, decomp.items);
}

test "round trip: empty" {
    try roundTrip(std.testing.allocator, "");
}

test "round trip: short text" {
    try roundTrip(std.testing.allocator, "hello, world! the quick brown fox.");
}

test "round trip: repetitive text" {
    const data = "the quick brown fox jumps over the lazy dog. " ** 40;
    try roundTrip(std.testing.allocator, data);
}

test "round trip: binary bytes" {
    var buf: [512]u8 = undefined;
    var rng = std.Random.DefaultPrng.init(999);
    rng.random().bytes(&buf);
    try roundTrip(std.testing.allocator, &buf);
}

// Regression: multi-line, mixed-case, punctuated prose of a few KB. paq8's
// filetype detector used to mis-fire on this class (text mis-read as a 24-bit
// image block) and overflow the pixel accumulators under safety checks; the
// short newline-free corpus above never triggered it. Keep this varied + >2KB.
test "round trip: varied prose (detector-misfire regression)" {
    const para =
        \\The XMLSTARLET USER'S GUIDE (v1.6.1) explains how one might parse,
        \\transform, and query structured documents from the command line.
        \\Consider: nested lists, tables, and 'quoted' phrases -- plus numbers
        \\like 3.14159, 42, and 0xFF. Context-mixing models predict bytes:
        \\order-0, order-1, PPM! Wikipedia's enwik8 mixes <ref> markup & prose.
        \\
    ;
    const data = para ** 12;
    try roundTrip(std.testing.allocator, data);
}

fn compressOnce(gpa: std.mem.Allocator, data: []const u8, out: *std.ArrayList(u8)) !void {
    var vocab: [256]bool = .{true} ** 256;
    const pred = try Predictor.init(gpa, &vocab);
    defer pred.deinit();
    var enc = coder.Encoder.init(gpa, out, pred);
    for (data) |c| {
        var b: i32 = 7;
        while (b >= 0) : (b -= 1) try enc.encode((@as(i32, c) >> @intCast(b)) & 1);
    }
    try enc.flush();
}

// Regression: module-level fxcm state must be fully reset per predictor, so a
// second predictor in the same process compresses identically to the first
// (a leaked SmallStationaryContextMap field once broke this — and round-trips).
test "sequential predictors compress identically (no state leak)" {
    const gpa = std.testing.allocator;
    const text = "the quick brown fox jumps over the lazy dog. " ** 40;
    var o1: std.ArrayList(u8) = .empty;
    defer o1.deinit(gpa);
    var o2: std.ArrayList(u8) = .empty;
    defer o2.deinit(gpa);
    try compressOnce(gpa, text, &o1);
    try compressOnce(gpa, text, &o2);
    try std.testing.expectEqualSlices(u8, o1.items, o2.items);
}

// ===========================================================================
// Lex-pipeline coverage. tests.zig historically imported only the v21 pipeline,
// so the lex modules' tests never ran (proof: the suite passed while the order
// asset was absent — article_reorder.zig's order-asset test could not have
// executed). Pull the lex tree in so `zig build test` exercises it. `@import` is
// transitive: analyzing the two roots analyzes the whole predictor + mixer + LSTM
// + fxcm26 + prepr shell, so their `test` blocks are collected too. The explicit
// prepr imports below make the previously-dark modules unmistakably covered.
// ===========================================================================
const runner_lex = @import("runner_lex.zig");
test {
    _ = runner_lex; // shell: runCompression/runDecompression, split, phda9, r1
    _ = @import("predictor_lex.zig"); // predictor + mixer + LSTM + fxcm26 goldens
    _ = @import("prepr/article_reorder.zig"); // order-asset size/line-count test
    _ = @import("prepr/s7_order.zig"); // s7_runlabel asset container: real-asset roundtrip golden
    _ = @import("prepr/self_extract.zig"); // self-extractor slice + trailer tests
    _ = @import("prepr/preprocessor_lex.zig");
    _ = @import("prepr/dictionary_lex.zig");
    _ = @import("prepr/split.zig");
    _ = @import("prepr/r1_reorder.zig");
    _ = @import("prepr/fieldcodec.zig"); // ts-before-revid swap: grammar + roundtrip tests
    _ = @import("prepr/cmtmove.zig"); // gated comment relocation: grammar + roundtrip tests
    _ = @import("zero_alloc.zig"); // calloc-parity big-table allocation
}

// The single most important invariant: the lex codec's ENCODE output must DECODE
// back to the exact input, via the very functions the shipped binary calls
// (runCompression / runDecompression). A separate-process "compress with the
// codec, decode with the shipped decoder" check is run in packaging/CI; this
// in-process version pins the same encode->decode identity in the unit suite.
// NOTE: the lex predictor allocates ~8 GB of tables and is slow, so this uses a
// small no-dict input and an arena (the runner does not free the predictor — it
// is a one-shot process — so leak-checked testing.allocator would false-fail).
test "lex codec round-trips (encode -> decode == identity)" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const input = "The quick brown fox; <ref>enwik8</ref> 3.14159, order-0 PPM!\n";
    const comp = try runner_lex.runCompression(a, input, null, true, 923);
    const back = try runner_lex.runDecompression(a, comp, null, 923);
    try std.testing.expectEqualSlices(u8, input, back);
}

// ---------------------------------------------------------------------------
// StationaryMap oracle tests — moved from leaf_maps.zig when it joined the
// cmfast module (in-module test blocks are not collected by `zig test`, and
// its test-only tables.zig import would put that file in two modules).
// Uses the cmfast module's public API + X type.
// ---------------------------------------------------------------------------
const cmfast_m = @import("cmfast");
const TablesLM = @import("models/fxcm26/tables.zig").Tables;

inline fn gsblLM(bpos: i32, c0: i32) i32 {
    const smask: i32 = @intCast((@as(u32, 0x31031010) >> @intCast(bpos << 2)) & 0x0F);
    return smask + (c0 & smask);
}

fn runStationaryLM(
    a: std.mem.Allocator,
    strt: []const i16,
    bits: u32,
    inbits: u32,
    mul: i32,
    rate: i32,
) !u64 {
    var m = cmfast_m.StationaryMap.new();
    try m.init(a, bits, inbits, mul, rate, strt);
    defer m.deinit(a);

    var x = cmfast_m.X{ .c0 = 1 };
    var cs: u64 = 0;
    var r: u32 = 0x12345678;
    var iter: usize = 0;
    while (iter < 4000) : (iter += 1) {
        r = r *% 1664525 +% 1013904223;
        const by: i32 = @intCast((r >> 16) & 0xff);
        var b: i32 = 7;
        while (b >= 0) : (b -= 1) {
            x.y = (by >> @intCast(b)) & 1;
            x.c0 += x.c0 + x.y;
            if (x.c0 >= 256) {
                x.c4 = (x.c4 << 8) +% (@as(u32, @intCast(x.c0)) & 0xff);
                x.c0 = 1;
            }
            x.bpos = (x.bpos + 1) & 7;
            x.bposshift = 7 - x.bpos;
            x.c0shift_bpos = (x.c0 << 1) ^ (@as(i32, 256) >> @intCast(x.bposshift));
            x.cm_bit_state = gsblLM(x.bpos, x.c0);
            if (x.bpos == 0) {
                m.set((x.c4 & 0x3ff) *% 1000003);
            }
            const e = m.mix(&x);
            for (e) |v| {
                cs = cs *% 1000003 +% @as(u64, @as(u16, @bitCast(v)));
            }
        }
    }
    return cs;
}

test "leaf_maps StationaryMap matches cpp oracle" {
    const a = std.testing.allocator;
    var tables = try TablesLM.new(a);
    defer tables.deinit(a);
    const strt = tables.strt;

    try std.testing.expectEqual(
        @as(u64, 0x57684c7efa967fe3),
        try runStationaryLM(a, strt, 16, 8, 8, 0),
    ); // StationaryMap maps1(16,8,8,0)
    try std.testing.expectEqual(
        @as(u64, 0xb06c333c0a08896f),
        try runStationaryLM(a, strt, 8, 8, 8, 0),
    ); // StationaryMap small(8,8,8,0)
    try std.testing.expectEqual(
        @as(u64, 0xd385ab3151ce9898),
        try runStationaryLM(a, strt, 10, 8, 4, 0),
    ); // StationaryMap mul4(10,8,4,0)
}
