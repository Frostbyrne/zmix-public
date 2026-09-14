//! Predictor, ported from cmix `predictor.{h,cpp}`.
//!
//! Wires cmix's model set into its 3-layer logistic mixer, following the exact
//! Predict/Perceive control flow. This phase includes the full non-"heavy" model
//! set — the Bracket-context Direct/Indirect, the Word (Sparse) models, the
//! Direct, Match and DoubleIndirect models — plus cmix's complete mixer wiring
//! (all interval/combined/bit-context mixer contexts).
//!
//! The LSTM byte-mixer, SSE, the Bracket byte model, and the FXCM (fxcm_v26)
//! auxiliary text model are ported and wired (AddFXCM below). Still to come
//! (ROADMAP Phase 3+): PAQ8 and the PPMd byte model (the `byte_models_` path).
//!
//! All predictor-lifetime memory is arena-allocated, so teardown is a single
//! arena free. Buffer sizes are capped below cmix's 32 GB-target defaults;
//! encoder and decoder share the predictor, so round-trips stay exact.
const std = @import("std");
const Sigmoid = @import("sigmoid.zig").Sigmoid;
const mixer = @import("mixer.zig");
const cm = @import("context_manager.zig");
const Model = @import("model.zig").Model;
const Direct = @import("models/direct.zig").Direct;
const DirectHash = @import("models/direct_hash.zig").DirectHash;
const Indirect = @import("models/indirect.zig").Indirect;
const Match = @import("models/match.zig").Match;
const Bracket = @import("models/bracket.zig").Bracket;
const maps = @import("interval_maps.zig");
const SSE = @import("sse.zig").SSE;
const ByteMixer = @import("mixer/byte_mixer.zig").ByteMixer;
const Lstm = @import("lstmfast").Lstm; // ReleaseFast module (-Dlstmfast)
const FXCM = @import("models/fxcm26/fxcm.zig").FXCM;
const PPMD = @import("models/ppmd.zig").PPMD;
const PAQ8 = @import("models/paq8/paq8.zig").PAQ8;

// PAQ8 memory level (cmix: PAQ8(11) -> MEM=0x10000<<11, ~20 GB of maps). zmix
// defaults to a capped level to stay runnable; the parity build uses 11.
const PAQ8_LEVEL: i32 = 4;

// PPMD suballocator memory in MB. cmix uses 14000 (~14 GB); zmix defaults to a
// capped value that avoids the model's memory-pressure paths on typical inputs
// while keeping RAM sane (see ROADMAP Phase 5 for the full-memory parity build).
const PPMD_MEM: u32 = 64;

// Memory caps (cmix targets 32 GB; see ROADMAP Phase 5 for parity).
const MATCH_MAP_CAP: u64 = 1 << 20;
const DIRECTHASH_CAP: usize = 1 << 15;

fn computedMap(comptime f: fn (i32) i32) [256]i32 {
    @setEvalBranchQuota(20000);
    var m: [256]i32 = undefined;
    for (0..256) |i| m[i] = f(@intCast(i));
    return m;
}
fn map1f(i: i32) i32 {
    return b(i < 1) + b(i < 32) + b(i < 64) + b(i < 128) + b(i < 255) + b(i < 142) + b(i < 138) + b(i < 140) + b(i < 137) + b(i < 97);
}
fn map2f(i: i32) i32 {
    return b(i < 41) + b(i < 92) + b(i < 124) + b(i < 58) + b(i < 11) + b(i < 46) + b(i < 36) + b(i < 47) + b(i < 64) + b(i < 4) + b(i < 61) + b(i < 97) + b(i < 125) + b(i < 45) + b(i < 48);
}
fn map3f(i: i32) i32 {
    if ((i >= 'a' and i <= 'z') or (i >= 'A' and i <= 'Z') or (i >= '0' and i <= '9') or i >= 0x80) return 1;
    return 0;
}
inline fn b(x: bool) i32 {
    return @intFromBool(x);
}
const MAP1 = computedMap(map1f);
const MAP2 = computedMap(map2f);
const MAP3 = computedMap(map3f);

// Zeroing backing allocator: zero-fills every chunk the arena requests. cmix's
// paq8/cmix models calloc their state; a fresh predictor gets zeroed pages from
// the OS (== C++), but the SECOND predictor in a process reuses the first's
// freed (dirty) arena memory, so any model field left uninitialised would read
// garbage and diverge. Zeroing chunks restores calloc semantics deterministically.
const ZeroBacking = struct {
    child: std.mem.Allocator,
    fn alloc(ctx: *anyopaque, len: usize, a: std.mem.Alignment, ra: usize) ?[*]u8 {
        const self: *ZeroBacking = @ptrCast(@alignCast(ctx));
        const p = self.child.rawAlloc(len, a, ra) orelse return null;
        @memset(p[0..len], 0);
        return p;
    }
    fn resize(ctx: *anyopaque, buf: []u8, a: std.mem.Alignment, nl: usize, ra: usize) bool {
        const self: *ZeroBacking = @ptrCast(@alignCast(ctx));
        return self.child.rawResize(buf, a, nl, ra);
    }
    fn remap(ctx: *anyopaque, buf: []u8, a: std.mem.Alignment, nl: usize, ra: usize) ?[*]u8 {
        const self: *ZeroBacking = @ptrCast(@alignCast(ctx));
        return self.child.rawRemap(buf, a, nl, ra);
    }
    fn free(ctx: *anyopaque, buf: []u8, a: std.mem.Alignment, ra: usize) void {
        const self: *ZeroBacking = @ptrCast(@alignCast(ctx));
        self.child.rawFree(buf, a, ra);
    }
    fn allocator(self: *ZeroBacking) std.mem.Allocator {
        return .{ .ptr = self, .vtable = &.{ .alloc = alloc, .resize = resize, .remap = remap, .free = free } };
    }
};

pub const Predictor = struct {
    gpa: std.mem.Allocator,
    zb: *ZeroBacking,
    arena: *std.heap.ArenaAllocator,
    a: std.mem.Allocator,
    sigmoid: Sigmoid,
    manager: *cm.ContextManager,
    prng: std.Random.DefaultPrng,
    rng: std.Random = undefined,
    sse: SSE = undefined,
    vocab: [256]bool = .{true} ** 256,

    models: std.ArrayList(Model) = .empty,
    byte_models: std.ArrayList(*PPMD) = .empty,
    byte_mixers: std.ArrayList(*ByteMixer) = .empty,
    auxiliary: std.ArrayList(usize) = .empty,
    layers: [3]*mixer.MixerInput = undefined,
    mixers: [3]std.ArrayList(*mixer.Mixer) = .{ .empty, .empty, .empty },

    pub fn init(gpa: std.mem.Allocator, vocab: *const [256]bool) !*Predictor {
        const zb = try gpa.create(ZeroBacking);
        zb.* = .{ .child = gpa };
        const arena = try gpa.create(std.heap.ArenaAllocator);
        arena.* = std.heap.ArenaAllocator.init(zb.allocator());
        const a = arena.allocator();
        const self = try gpa.create(Predictor);
        self.* = .{
            .gpa = gpa,
            .zb = zb,
            .arena = arena,
            .a = a,
            .sigmoid = try Sigmoid.init(a, 100001),
            .manager = try cm.ContextManager.init(a),
            .prng = std.Random.DefaultPrng.init(0xDEADBEEF),
            .vocab = vocab.*,
        };
        self.rng = self.prng.random();
        self.sse = try SSE.init(a);
        for (0..3) |i| {
            self.layers[i] = try a.create(mixer.MixerInput);
            self.layers[i].* = mixer.MixerInput.init(a, &self.sigmoid, 1.0e-4);
        }
        try self.build();
        return self;
    }

    pub fn deinit(self: *Predictor) void {
        self.arena.deinit();
        self.gpa.destroy(self.arena);
        self.gpa.destroy(self.zb);
        self.gpa.destroy(self);
    }

    // --- context/model construction helpers (all arena-allocated) ---
    fn ctxHash(self: *Predictor, order: u32, hash_size: u32) *cm.ContextHash {
        const c = cm.ContextHash.create(self.a, &self.manager.bit_context, order, hash_size);
        self.manager.addContext(c.context());
        return c;
    }
    fn sparse(self: *Predictor, orders: []const u32) *cm.Sparse {
        const c = cm.Sparse.create(self.a, &self.manager.words, orders);
        self.manager.addContext(c.context());
        return c;
    }
    fn indirectHash(self: *Predictor, o1: u32, h1: u32, o2: u32, h2: u32) *cm.IndirectHash {
        const c = cm.IndirectHash.create(self.a, &self.manager.bit_context, o1, h1, o2, h2);
        self.manager.addContext(c.context());
        return c;
    }
    fn interval(self: *Predictor, map: *const [256]i32, num_bits: u32) *cm.Interval {
        const c = cm.Interval.create(self.a, &self.manager.bit_context, map, num_bits);
        self.manager.addContext(c.context());
        return c;
    }
    fn intervalHash(self: *Predictor, map: *const [256]i32, num_bits: u32, order: u32, hash_size: u32) *cm.IntervalHash {
        const c = cm.IntervalHash.create(self.a, &self.manager.bit_context, map, num_bits, order, hash_size);
        self.manager.addContext(c.context());
        return c;
    }
    fn combined(self: *Predictor, c1: *const u64, c2: *const u64, s1: u64, s2: u64) *cm.CombinedContext {
        const c = cm.CombinedContext.create(self.a, c1, c2, s1, s2);
        self.manager.addContext(c.context());
        return c;
    }
    fn bracketCtx(self: *Predictor, distance_limit: u32, stack_limit: u32) *cm.BracketContext {
        const c = cm.BracketContext.create(self.a, &self.manager.bit_context, distance_limit, stack_limit);
        self.manager.addContext(c.context());
        return c;
    }
    fn bitCtx(self: *Predictor, byte_ctx: *const u64, byte_ctx_size: u64) *cm.BitContext {
        const c = cm.BitContext.create(self.a, &self.manager.long_bit_context, byte_ctx, byte_ctx_size);
        self.manager.addBitContext(c.context());
        return c;
    }

    fn addDirect(self: *Predictor, byte_ctx: *const u64, limit: i32, delta: f32, size: usize) void {
        const m = Direct.create(self.a, byte_ctx, &self.manager.bit_context, limit, delta, size);
        self.models.append(self.a, m.model()) catch unreachable;
    }
    fn addDirectHash(self: *Predictor, byte_ctx: *const u64, limit: i32, delta: f32, size: usize) void {
        const m = DirectHash.create(self.a, byte_ctx, &self.manager.bit_context, limit, delta, @min(size, DIRECTHASH_CAP));
        self.models.append(self.a, m.model()) catch unreachable;
    }
    fn addIndirect(self: *Predictor, state: @import("states.zig").State, byte_ctx: *const u64, delta: f32) void {
        const m = Indirect.create(self.a, state, byte_ctx, &self.manager.bit_context, delta, self.manager.shared_map, &self.rng);
        self.models.append(self.a, m.model()) catch unreachable;
    }
    fn addMatch(self: *Predictor, byte_ctx: *const u64, map_size: u64) void {
        const m = Match.create(self.a, self.manager.history, byte_ctx, &self.manager.bit_context, 200, 0.5, @intCast(@min(map_size, MATCH_MAP_CAP)), &self.manager.longest_match);
        self.models.append(self.a, m.model()) catch unreachable;
    }
    fn addMixer(self: *Predictor, layer: usize, context: *const u64, learning_rate: f32) void {
        const m = self.a.create(mixer.Mixer) catch unreachable;
        m.* = mixer.Mixer.init(self.a, self.layers[layer], context, learning_rate, self.mixers[layer].items.len);
        self.mixers[layer].append(self.a, m) catch unreachable;
    }

    fn build(self: *Predictor) !void {
        const mgr = self.manager;
        const nonstat = mgr.nonstationary.state();

        // AddBracket
        {
            const br = Bracket.create(self.a, &mgr.bit_context, 200, 10, 100000, &self.vocab);
            try self.models.append(self.a, br.model());
            const c = self.bracketCtx(256, 15);
            self.addDirect(c.value_ptr(), 30, 0, @intCast(c.size));
            self.addIndirect(nonstat, c.value_ptr(), 300);
        }

        // AddFXCM — the fxcm_v26 auxiliary text model (instance-based, reset-clean).
        // Its 561 outputs (560 slot-probs ++ final pr-prob) feed layer-0; the LAST
        // output (the final pr) feeds the auxiliary average (auxiliary_context).
        // (v26 has no LSTM dependency, so no deferred-perceive is needed.)
        {
            const fx = FXCM.create(self.a);
            try self.models.append(self.a, fx.model());
            try self.auxiliary.append(self.a, self.getNumModels() - 1);
        }

        // AddPAQ8 — the full paq8-derived auxiliary model (all context/match/
        // record/sparse/word/nest/text/image/audio/jpeg/exe/dmc models + its own
        // mixer + APM chain). Its last output feeds the auxiliary average.
        {
            const pq = PAQ8.create(self.a, PAQ8_LEVEL);
            try self.models.append(self.a, pq.model());
            try self.auxiliary.append(self.a, self.getNumModels() - 1);
        }

        // AddPPMD — order-25 PPMd byte model. It feeds layer-0 per-bit (like a
        // regular model) AND its 256-symbol distribution feeds the LSTM byte-mixer
        // each byte boundary (the byte_models_ path). (cmix: PPMD(25, 14000, ...))
        {
            const pm = PPMD.create(self.a, &mgr.bit_context, 25, PPMD_MEM, &self.vocab);
            try self.byte_models.append(self.a, pm);
        }

        // AddWord
        {
            const params1 = [_][]const u32{
                &.{0}, &.{ 0, 1 }, &.{ 7, 2 }, &.{7}, &.{1}, &.{ 1, 2 }, &.{ 1, 2, 3 },
                &.{ 1, 3 }, &.{ 1, 4 }, &.{ 1, 5 }, &.{ 2, 3 }, &.{ 3, 4 }, &.{ 1, 2, 4 },
                &.{ 1, 2, 3, 4 }, &.{ 2, 3, 4 }, &.{2}, &.{ 1, 2, 3, 4, 5 }, &.{ 1, 2, 3, 4, 5, 6 },
            };
            for (params1) |p| {
                const c = self.sparse(p);
                self.addIndirect(nonstat, c.value_ptr(), 200);
            }
            const params2 = [_][]const u32{ &.{0}, &.{1}, &.{7}, &.{ 1, 3 }, &.{ 1, 2, 3 }, &.{ 7, 2 } };
            for (params2) |p| {
                const c = self.sparse(p);
                self.addMatch(c.value_ptr(), 10000000);
                if (p.len == 1 and p[0] == 1) {
                    self.addIndirect(mgr.run_map.state(), c.value_ptr(), 200);
                    self.addDirectHash(c.value_ptr(), 30, 0, 500000);
                }
            }
        }

        // AddDirect
        {
            const params = [_][2]u32{ .{ 0, 8 }, .{ 1, 8 }, .{ 2, 8 }, .{ 3, 8 } };
            for (params) |p| {
                const c = self.ctxHash(p[0], p[1]);
                if (p[0] < 3) {
                    self.addDirect(c.value_ptr(), 30, 0, @intCast(c.size));
                } else {
                    self.addDirectHash(c.value_ptr(), 30, 0, 100000);
                }
            }
        }

        // AddMatch
        {
            const params = [_][2]u32{ .{ 0, 8 }, .{ 1, 8 }, .{ 2, 8 }, .{ 7, 4 }, .{ 11, 3 }, .{ 13, 2 }, .{ 15, 2 }, .{ 17, 2 }, .{ 20, 1 }, .{ 25, 1 } };
            for (params) |p| {
                const c = self.ctxHash(p[0], p[1]);
                self.addMatch(c.value_ptr(), @min(@as(u64, 20000000), c.size));
            }
        }

        // AddDoubleIndirect
        {
            const params = [_][4]u32{
                .{ 1, 8, 1, 8 }, .{ 2, 8, 1, 8 }, .{ 1, 8, 2, 8 }, .{ 2, 8, 2, 8 }, .{ 1, 8, 3, 8 },
                .{ 3, 8, 1, 8 }, .{ 4, 6, 4, 8 }, .{ 5, 5, 5, 5 }, .{ 1, 8, 4, 8 }, .{ 1, 8, 5, 6 }, .{ 6, 4, 6, 4 },
            };
            for (params) |p| {
                const c = self.indirectHash(p[0], p[1], p[2], p[3]);
                self.addIndirect(nonstat, c.value_ptr(), 400);
            }
        }

        try self.addMixers();
    }

    fn getNumModels(self: *Predictor) usize {
        var n: usize = 0;
        for (self.models.items) |m| n += m.numOutputs();
        for (self.byte_models.items) |bm| n += bm.numOutputs();
        for (self.byte_mixers.items) |bm| n += bm.numOutputs();
        return n;
    }

    fn addMixers(self: *Predictor) !void {
        const mgr = self.manager;

        // Byte-mixer (LSTM) — an auxiliary byte model. cmix: ByteMixer over an
        // Lstm(vocab_size, vocab_size, 200, 2, 100, 0.03, 10).
        var vocab_size: u32 = 0;
        for (self.vocab) |v| {
            if (v) vocab_size += 1;
        }
        // -Dtransformer-lstm-dead asserts that every ByteMixer carries a
        // transformer. This one does NOT — the v21 predictor is the LSTM
        // control arm — so the flag is refused here rather than silently
        // producing a binary whose byte-mixer path is `unreachable`. The flag
        // is a SHIP-LINE flag (form1/ship never compile this file); a test or
        // `zmix` build must not carry it.
        comptime {
            const tfm = @import("mixer/transformer.zig");
            if (tfm.enabled and tfm.lstm_dead)
                @compileError("-Dtransformer-lstm-dead with the v21 predictor.zig: that flag asserts every ByteMixer carries a transformer, and predictor.zig:334 builds one that does not (it is the online-LSTM control arm). It is a ship-line flag: build.zig routes it to the `ship` and `form1` roots ONLY, via `transformer_opts_ship`. Reaching this message means a root that compiles predictor.zig was wired to that module — fix the wiring in build.zig, do not drop the flag from SHIP_RECIPE.env.");
        }
        const lstm = Lstm.init(self.a, vocab_size, vocab_size, 200, 2, 100, 0.03, 10, &self.rng);
        const bm = ByteMixer.create(self.a, @intCast(self.byte_models.items.len), &mgr.bit_context, &self.vocab, vocab_size, lstm);
        try self.byte_mixers.append(self.a, bm);
        try self.auxiliary.append(self.a, self.getNumModels() - 1);

        var input_size: usize = self.getNumModels();
        self.layers[0].setNumModels(input_size);

        // Layer 0
        const l0 = [_][3]f64{ .{ 0, 8, 0.005 }, .{ 0, 8, 0.0005 }, .{ 1, 8, 0.005 }, .{ 1, 8, 0.0005 }, .{ 2, 4, 0.005 }, .{ 3, 2, 0.002 } };
        for (l0) |p| {
            const c = self.ctxHash(@intFromFloat(p[0]), @intFromFloat(p[1]));
            const bc = self.bitCtx(c.value_ptr(), c.size);
            self.addMixer(0, bc.value_ptr(), @floatCast(p[2]));
        }
        self.addMixer(0, &mgr.recent_bytes[2], 0.002);
        self.addMixer(0, &mgr.recent_bytes[3], 0.005);
        self.addMixer(0, &mgr.zero_context, 0.00005);
        self.addMixer(0, &mgr.line_break, 0.0007);
        self.addMixer(0, &mgr.longest_match, 0.0005);
        self.addMixer(0, &mgr.wrt_context, 0.002);
        self.addMixer(0, &mgr.auxiliary_context, 0.0005);

        const interval1 = self.interval(&MAP1, 8);
        self.addMixer(0, interval1.value_ptr(), 0.001);
        const interval2 = self.interval(&MAP2, 8);
        self.addMixer(0, interval2.value_ptr(), 0.001);
        const interval3 = self.interval(&MAP3, 7);
        self.addMixer(0, interval3.value_ptr(), 0.001);
        const bit_context5 = self.bitCtx(interval3.value_ptr(), interval3.size);
        self.addMixer(0, bit_context5.value_ptr(), 0.005);
        const interval4 = self.interval(&maps.MAP_A, 10);
        self.addMixer(0, interval4.value_ptr(), 0.001);
        const interval5 = self.interval(&maps.MAP_A, 15);
        self.addMixer(0, interval5.value_ptr(), 0.001);
        const interval8 = self.interval(&maps.MAP_A, 7);
        const bit_context4 = self.bitCtx(interval8.value_ptr(), interval8.size);
        self.addMixer(0, bit_context4.value_ptr(), 0.005);
        const interval6 = self.interval(&maps.MAP_B, 9);
        self.addMixer(0, interval6.value_ptr(), 0.001);
        const interval7 = self.intervalHash(&maps.MAP_B, 8, 7, 2);
        self.addMixer(0, interval7.value_ptr(), 0.001);
        const interval9 = self.interval(&maps.MAP_B, 7);
        const bit_context6 = self.bitCtx(interval9.value_ptr(), interval9.size);
        self.addMixer(0, bit_context6.value_ptr(), 0.005);
        const bit_context1 = self.bitCtx(&mgr.recent_bytes[1], 256);
        self.addMixer(0, bit_context1.value_ptr(), 0.005);
        const combined1 = self.combined(&mgr.recent_bytes[1], &mgr.recent_bytes[0], 256, 256);
        self.addMixer(0, combined1.value_ptr(), 0.005);
        const combined2 = self.combined(&mgr.recent_bytes[2], &mgr.recent_bytes[1], 256, 256);
        self.addMixer(0, combined2.value_ptr(), 0.003);

        // Layer 1
        input_size = self.mixers[0].items.len + self.auxiliary.items.len;
        self.layers[1].setNumModels(input_size);
        self.addMixer(1, &mgr.zero_context, 0.005);
        self.addMixer(1, &mgr.zero_context, 0.0005);
        self.addMixer(1, &mgr.long_bit_context, 0.005);
        self.addMixer(1, &mgr.long_bit_context, 0.0005);
        self.addMixer(1, &mgr.long_bit_context, 0.00001);
        self.addMixer(1, &mgr.recent_bytes[0], 0.005);
        self.addMixer(1, &mgr.recent_bytes[1], 0.005);
        self.addMixer(1, &mgr.recent_bytes[2], 0.005);
        self.addMixer(1, &mgr.longest_match, 0.0005);
        self.addMixer(1, &mgr.wrt_context, 0.002);
        self.addMixer(1, interval1.value_ptr(), 0.001);
        self.addMixer(1, interval2.value_ptr(), 0.001);
        self.addMixer(1, interval3.value_ptr(), 0.001);
        self.addMixer(1, interval4.value_ptr(), 0.001);
        self.addMixer(1, interval5.value_ptr(), 0.001);
        self.addMixer(1, interval6.value_ptr(), 0.001);
        self.addMixer(1, interval7.value_ptr(), 0.001);
        self.addMixer(1, bit_context4.value_ptr(), 0.001);
        self.addMixer(1, bit_context5.value_ptr(), 0.001);
        self.addMixer(1, bit_context6.value_ptr(), 0.001);

        // Layer 2
        input_size = self.mixers[0].items.len + self.mixers[1].items.len + self.auxiliary.items.len;
        self.layers[2].setNumModels(input_size);
        self.addMixer(2, &mgr.zero_context, 0.0003);
    }

    pub fn predict(self: *Predictor) f32 {
        var input_index: usize = 0;
        for (self.models.items) |m| {
            const outputs = m.predict();
            for (outputs) |o| {
                self.layers[0].setInput(input_index, o);
                input_index += 1;
            }
        }
        for (self.byte_models.items) |bm| {
            const outputs = bm.predict();
            for (outputs) |o| {
                self.layers[0].setInput(input_index, o);
                input_index += 1;
            }
        }
        var byte_mixer_override: f32 = -1;
        for (self.byte_mixers.items) |bm| {
            const outputs = bm.predict();
            for (outputs) |o| {
                if (o == 0 or o == 1) byte_mixer_override = o;
                self.layers[0].setInput(input_index, o);
                input_index += 1;
            }
        }
        // auxiliary average -> auxiliary_context
        if (self.auxiliary.items.len > 0) {
            var aux_avg: f32 = 0;
            for (self.auxiliary.items) |idx| aux_avg += Sigmoid.logistic(self.layers[0].inputs[idx]);
            aux_avg /= @floatFromInt(self.auxiliary.items.len);
            self.manager.auxiliary_context = @intFromFloat(aux_avg * 15);
        }

        for (self.mixers[0].items, 0..) |m, i| {
            const p = m.mix();
            self.layers[0].setExtraInput(p);
            self.layers[1].setStretchedInput(i, p);
            self.layers[2].setStretchedInput(i, p);
        }
        self.layers[0].clearExtraInputs();
        for (self.auxiliary.items, 0..) |idx, i| {
            const p = self.layers[0].inputs[idx];
            self.layers[1].setStretchedInput(self.mixers[0].items.len + i, p);
            self.layers[2].setStretchedInput(self.mixers[0].items.len + self.mixers[1].items.len + i, p);
        }
        for (self.mixers[1].items, 0..) |m, i| {
            const p = m.mix();
            self.layers[1].setExtraInput(p);
            self.layers[2].setStretchedInput(self.mixers[0].items.len + i, p);
        }
        self.layers[1].clearExtraInputs();
        const p = self.sse.predict(Sigmoid.logistic(self.mixers[2].items[0].mix()));
        if (byte_mixer_override >= 0) return byte_mixer_override;
        return p;
    }

    pub fn perceive(self: *Predictor, bit: i32) void {
        for (self.models.items) |m| m.perceive(bit);
        for (self.byte_models.items) |bm| bm.perceive(bit);
        for (self.byte_mixers.items) |bm| bm.perceive(bit);
        for (0..3) |i| {
            for (self.mixers[i].items) |m| m.perceive(bit);
        }
        self.sse.perceive(bit);
        const byte_update = self.manager.bit_context >= 128;
        self.manager.updateContexts(bit);
        if (byte_update) {
            for (self.models.items) |m| m.byteUpdate();
            for (self.byte_models.items) |bm| bm.byteUpdate();
            // Feed each byte model's 256-symbol distribution into the byte-mixer.
            for (self.byte_models.items) |bm| {
                const p = bm.bytePredict();
                for (self.byte_mixers.items) |mx| {
                    for (0..256) |j| mx.setInput(j, p[j]);
                }
            }
            for (self.byte_mixers.items) |bm| bm.byteUpdate();
            self.manager.bit_context = 1;
        }
    }
};
