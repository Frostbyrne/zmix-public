//! Indirect model, ported from cmix `models/indirect.{h,cpp}`.
//!
//! Maps a byte context, via a shared byte-history-state map, to an adaptive
//! probability keyed on the bit-history state. The state transitions come from a
//! `State` machine (Nonstationary or RunMap).
const std = @import("std");
const Model = @import("../model.zig").Model;
const vtableFor = @import("../model.zig").vtableFor;
const State = @import("../states.zig").State;
const STATE_DEPTH = @import("../states.zig").STATE_DEPTH;
const build_options = @import("build_options");
const leafgrad_options = @import("leafgrad_options");

/// -Dleafgrad / -Dleafgrad-z=<Z> — CODING-LOSS LEAF TRAINING.
///
/// `predictions[state]` is this model's StateMap: the calibrated probability of
/// the bit-history state the shared map holds for the current context. Stock
/// trains it by a LOCAL rule only — `p += (y - p) * divisor` — which moves the
/// cell just as hard when the ENSEMBLE was already right, and just as hard when
/// the mixture puts no weight on this model at all.
///
/// This adds (never replaces — replacing measured far worse offline) the exact
/// gradient of the CODED LENGTH w.r.t. this cell's logit. The mixer input for
/// this model is `Logit(predictions[state])` (MixerInputLex.setInput), so the
/// leaf logit IS the mixer input and
///     dL/dtau = (sigma(z) - y) * dz/dv_m  =  `lg_grad`, supplied by the caller.
/// The step is applied in the logit domain exactly, without a log:
///     sigma(logit(p) - d) == p / (p + (1-p) * exp(d))
/// Z == 0 makes `d` identically zero and the cell is not written at all, so the
/// -Dleafgrad build with Z=0 is bit-identical to stock (the null control).
pub const LEAFGRAD: bool = leafgrad_options.leafgrad;
pub const LEAFGRAD_Z: f32 = leafgrad_options.leafgrad_z;

/// -Dind-census: LAB-ONLY census of the shared indirect map's real occupancy and
/// oversubscription. Comptime-dead at default. Each Indirect keeps a side hash set
/// of the distinct byte contexts it has keyed, plus an occurrence histogram; the
/// predictor prints the roll-up at deinit. No effect on any prediction.
pub const CENSUS: bool = build_options.ind_census;

/// -Dind-admit=K — ADMISSION CONTROL for the shared bit-history map.
///
/// The map is a direct-mapped, checksum-free table of 8-bit bit-history states at
/// load factor λ = (Σ_models distinct_ctx × nodes_touched) / |map|. Census (1 MB,
/// -Dind-census): 4,855,399 distinct contexts across the 16 models, mean 1.33–1.89
/// occurrences per context in the six highest-cardinality models ⇒ the MAJORITY of
/// contexts are singletons. A singleton reads state it never wrote and writes 8
/// nodes it never reads again: it cannot benefit, and its writes overwrite the
/// states of contexts that DO repeat. λ scales linearly with the stream, so at e9
/// every mid-frequency context's state is overwritten thousands of times.
///
/// Admission uses information the map ALREADY holds and costs zero extra memory:
/// the ROOT node of a context's 257-byte bit-history tree (c0 == 1) is touched on
/// EVERY occurrence of that context, so its state is that context's own occurrence
/// counter. If the root reads 0 at byte start, this is (probably) a first visit ⇒
/// write only the first K levels of the path, claiming the root so the SECOND visit
/// is admitted in full. Decoder-symmetric by construction (a function of decoded
/// bytes only), and byte-identical to stock at K = 0.
pub const ADMIT: u32 = build_options.ind_admit;

/// -Dind-tag — COLLISION DETECTION for the shared bit-history map.
///
/// The map is the only large table in the system with no checksum: a read returns
/// whatever state occupies the position, whoever wrote it. Every other table in the
/// lineage (ContextMap) tags its buckets. With `-Dind-tag` the allocation is split
/// in half — states in the low half, one tag byte per state in the high half — so at
/// the SAME total memory there are half as many positions but a collided read is
/// DETECTED and returns a fresh state instead of another context's history.
/// Decoder-symmetric (tags are a function of decoded bytes). Default off.
pub const TAG: bool = build_options.ind_tag != 0;
/// 1 = split halves (states | tags) — an extra cache line per access.
/// 2 = INTERLEAVED [state, tag] pairs — one 2-byte load, ONE cache line, i.e. the
///     same memory traffic as stock. This is the shippable form.
/// 3 = interleaved + PRIORITY CLAIMING: a foreign reader still gets an honest fresh
///     state, but it only STEALS the position if the incumbent is young
///     (STATE_DEPTH < ADMIT_DEPTH). An established owner keeps its history instead of
///     being clobbered on every foreign touch, so two contexts sharing a position no
///     longer thrash each other.
pub const TAG_MODE: u32 = build_options.ind_tag;
/// -Dind-tag-bits=B — tag width. 8 bits ⇒ a 1/256 undetected-collision rate at one
/// byte per state (100 % memory overhead on the map). Narrower tags cut the overhead
/// proportionally but raise the miss rate; this sweep prices that exchange.
/// Implemented by masking the tag to B bits (the byte is still the storage unit, so
/// this measures the RATE, not the packing — packing is a separate engineering step).
pub const TAG_BITS: u32 = build_options.ind_tag_bits;
inline fn tagMask(v: u8) u8 {
    if (TAG_BITS >= 8) return v | 1;
    const m: u8 = @intCast((@as(u16, 1) << @intCast(TAG_BITS)) - 1);
    return (v & m) | 1;
}
pub const ADMIT_DEPTH: u8 = @intCast(build_options.ind_admit_depth);

/// -Dind-delta-div — the adaptation rate of `predictions[256]`, the state->probability
/// decoder of the 16 outer `Indirect` models.
///
/// `perceive` does `predictions[st] += (bit - predictions[st]) / delta` with `delta`
/// hard-coded per call site to 200 / 300 / 400 (cmix's `AddWord`/`AddDoubleIndirect`
/// arguments). delta IS the time constant, and at e9 each of the 256 entries receives
/// 4.697e9/256 = 18.3 M updates ⇒ **45,000-92,000 time constants**. The estimator is
/// drowning in data: it has ~14x MORE unused statistical strength than `ContextMap3.ts[]`,
/// whose own ladder (`cm3-ts-rate`) says the shipped rate is too SLOW.
///
/// This knob divides every delta by N, i.e. makes the decoder N x faster. Default 1 =
/// shipped, bit-identical. Predicted by the S12y starvation sweep, not by a hunch.
pub const DELTA_DIV: f32 = @floatFromInt(build_options.ind_delta_div);
pub const CensusState = if (CENSUS) struct {
    keys: []u64 = &.{},
    cnt: []u32 = &.{},
    mask: u64 = 0,
    distinct: u64 = 0,
    occ: u64 = 0,
    pub fn init(a: std.mem.Allocator, bits: u6) CensusState {
        const n = @as(usize, 1) << bits;
        return .{ .keys = a.alloc(u64, n) catch unreachable,
                  .cnt = a.alloc(u32, n) catch unreachable, .mask = n - 1 };
    }
    pub fn note(self: *CensusState, c: u64) void {
        const key = c | 1;
        var i: u64 = (c *% 0x9E3779B97F4A7C15) & self.mask;
        while (self.keys[i] != 0 and self.keys[i] != key) i = (i + 1) & self.mask;
        if (self.keys[i] == 0) { self.keys[i] = key; self.distinct += 1; self.cnt[i] = 0; }
        self.cnt[i] += 1;
        self.occ += 1;
    }
} else void;

pub const Indirect = struct {
    byte_context: *const u64,
    bit_context: *const u32,
    map_index: u64 = 0,
    map_offset: u64,
    divisor: f32,
    state: State,
    map: []u8, // shared map
    predictions: [256]f32,
    outputs: [1]f32 = .{0.5},
    alloc: std.mem.Allocator,
    census: CensusState = if (CENSUS) undefined else {},
    fresh: bool = false,
    tag: u8 = 1,
    foreign: bool = false,
    /// -Dleafgrad: dL/dtau for THIS model on THIS bit, written by
    /// PredictorLex.perceive before the model perceives. Zero on any bit whose
    /// mixer cascade did not run (an -Dadaptive-depth skip), so the leaf step is
    /// exactly zero there too. `void` (zero-sized, no layout change) when the
    /// knob is off — the stock struct is untouched.
    lg_grad: if (LEAFGRAD) f32 else void = if (LEAFGRAD) 0 else {},

    pub fn create(a: std.mem.Allocator, state: State, byte_context: *const u64, bit_context: *const u32, delta: f32, map: []u8, rng: *std.Random) *Indirect {
        const self = a.create(Indirect) catch unreachable;
        self.* = .{
            .byte_context = byte_context,
            .bit_context = bit_context,
            .map_offset = rng.uintLessThan(u64, map.len - 257),
            .divisor = DELTA_DIV / delta,
            .state = state,
            .map = map,
            .predictions = undefined,
            .alloc = a,
        };
        for (0..256) |i| self.predictions[i] = state.initProbability(i);
        if (comptime CENSUS) { @memset(std.mem.sliceAsBytes(blk: {
            self.census = CensusState.init(a, 24); break :blk self.census.keys; }), 0);
            @memset(self.census.cnt, 0); }
        return self;
    }

    pub fn destroy(self: *Indirect) void {
        self.alloc.destroy(self);
    }

    /// sm35: prefetch the shared-map byte this bit's predictwill read. 16
    /// instances share a 100 MB map; the base (map_index, set per byte from a
    /// high-entropy word/sentence hash) is a cold DRAM miss, unprefetched. Issued
    /// for all 16 at the top of predictor_lex.predict (max lead time — byteUpdate
    /// is too early, the intervening fxcm.perceive evicts it). Pure cache hint.
    pub fn prefetch(self: *const Indirect) void {
        if (comptime TAG_MODE >= 2) {
            @prefetch(&self.map[(self.map_index + self.bit_context.*) * 2], .{});
            return;
        }
        @prefetch(&self.map[self.map_index + self.bit_context.*], .{});
    }

    pub fn predict(self: *Indirect) []const f32 {
        self.map_index += self.bit_context.*;
        if (comptime TAG) {
            const half = self.map.len / 2;
            const idx = if (comptime TAG_MODE >= 2) self.map_index * 2 else self.map_index % half;
            const tix = if (comptime TAG_MODE >= 2) idx + 1 else half + idx;
            if (comptime CENSUS) { self.census.occ += 1; if (self.map[tix] != self.tag) self.census.distinct += 1; }
            if (self.map[tix] != self.tag) {          // foreign owner
                if (comptime TAG_MODE == 3) {
                    // read honest-fresh; steal only if the incumbent is not established
                    if (STATE_DEPTH[self.map[idx]] < ADMIT_DEPTH) {
                        self.map[tix] = self.tag;
                        self.map[idx] = 0;
                    } else {
                        self.foreign = true;
                        self.outputs[0] = self.predictions[0];
                        return self.outputs[0..1];
                    }
                } else {
                    self.map[tix] = self.tag;
                    self.map[idx] = 0;
                }
            } else if (comptime TAG_MODE == 3) self.foreign = false;
            self.outputs[0] = self.predictions[self.map[idx]];
            return self.outputs[0..1];
        }
        self.outputs[0] = self.predictions[self.map[self.map_index]];
        return self.outputs[0..1];
    }

    /// -Dleafgrad: one exact coded-length gradient step on the StateMap cell,
    /// applied BEFORE the incumbent local rule (the fused form — the local
    /// `1/(n+2)`-class running average stays the estimator, the gradient is a
    /// correction). Comptime-dead when the knob is off; a no-op when Z or the
    /// supplied gradient is zero, so no cell is written and the null control is
    /// bit-identical.
    inline fn leafGradStep(self: *Indirect, st: u8) void {
        if (comptime !LEAFGRAD) return;
        var d: f64 = @as(f64, LEAFGRAD_Z) * @as(f64, self.lg_grad);
        if (d == 0) return;
        if (d > 12.0) d = 12.0 else if (d < -12.0) d = -12.0;
        var p: f64 = @as(f64, self.predictions[st]);
        if (p < 1.0e-6) p = 1.0e-6 else if (p > 1.0 - 1.0e-6) p = 1.0 - 1.0e-6;
        // sigma(logit(p) - d) with no log: p / (p + (1-p)*exp(d)).
        var q: f64 = p / (p + (1.0 - p) * @exp(d));
        if (q < 1.0e-6) q = 1.0e-6 else if (q > 1.0 - 1.0e-6) q = 1.0 - 1.0e-6;
        self.predictions[st] = @floatCast(q);
    }

    pub fn perceive(self: *Indirect, bit: i32) void {
        if (comptime TAG) {
            const half = self.map.len / 2;
            const idx = if (comptime TAG_MODE >= 2) (self.map_index % half) * 2 else self.map_index % half;
            if (comptime TAG_MODE == 3) {
                if (self.foreign) { self.map_index -= self.bit_context.*; return; }  // don't touch a protected owner
            }
            const st2 = self.map[idx];
            self.leafGradStep(st2);
            self.predictions[st2] += (@as(f32, @floatFromInt(bit)) - self.predictions[st2]) * self.divisor;
            self.map[idx] = self.state.next(st2, bit);
            self.map_index -= self.bit_context.*;
            return;
        }
        const st = self.map[self.map_index];
        self.leafGradStep(st);
        self.predictions[st] += (@as(f32, @floatFromInt(bit)) - self.predictions[st]) * self.divisor;
        if (comptime ADMIT == 0) {
            self.map[self.map_index] = self.state.next(st, bit);
        } else if (comptime ADMIT == 3) {
            // PRIORITY form: a first-visit context may not overwrite an ESTABLISHED
            // state (one that is >= ADMIT_DEPTH transitions deep in the state graph);
            // it may still claim empty or young positions.
            if (!self.fresh or STATE_DEPTH[st] < ADMIT_DEPTH or self.bit_context.* == 1)
                self.map[self.map_index] = self.state.next(st, bit);
        } else {
            // first visit ⇒ write only the first ADMIT levels (bit_context < 2^ADMIT)
            if (!self.fresh or self.bit_context.* < (@as(u32, 1) << @intCast(ADMIT)))
                self.map[self.map_index] = self.state.next(st, bit);
        }
        self.map_index -= self.bit_context.*;
    }

    pub fn byteUpdate(self: *Indirect) void {
        if (comptime CENSUS) self.census.note(self.byte_context.*);
        if (comptime TAG_MODE >= 2) {
            // Index in POSITIONS (not bytes) over the state array, so the per-bit path
            // needs no division at all — exactly stock's one-division-per-byte cost.
            const positions = self.map.len / 2;
            self.map_index = (257 *% self.byte_context.* +% self.map_offset) % (positions - 257);
        } else {
            self.map_index = (257 *% self.byte_context.* +% self.map_offset) % (self.map.len - 257);
        }
        if (comptime ADMIT != 0) self.fresh = (self.map[self.map_index + 1] == 0);
        if (comptime TAG) self.tag = tagMask(@truncate((self.byte_context.* >> 32) *% 0x9E37 ^ (self.byte_context.* >> 7)));
    }
    pub fn numOutputs(self: *Indirect) usize {
        _ = self;
        return 1;
    }

    pub fn model(self: *Indirect) Model {
        return .{ .ptr = self, .vtable = vtableFor(Indirect) };
    }
};
