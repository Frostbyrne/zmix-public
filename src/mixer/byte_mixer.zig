//! ByteMixer, ported from cmix `mixer/byte-mixer.{h,cpp}`.
//!
//! A ByteModel whose byte distribution comes from the LSTM. Byte-model inputs
//! (summed) feed the LSTM as auxiliary input; the LSTM's softmax over the vocab
//! becomes the byte distribution, decoded per-bit via the ByteModel base.
const std = @import("std");
const build_options = @import("build_options");
const ByteModel = @import("../models/byte_model.zig").ByteModel;
const Lstm = @import("lstmfast").Lstm; // ReleaseFast module (-Dlstmfast); float hot path
const Model = @import("../model.zig").Model;
const vtableFor = @import("../model.zig").vtableFor;
// -Dslot-loss (LAB): own-options module, comptime-dead at default. See src/slot_loss.zig.
const slot_loss = @import("slotloss");

// -Dfree-prior-b (variant b): a comptime-erased sink for the coding-time PPMd
// prior. When the flag is off both fields below are `void` (zero size,
// comptime-dead record call) — stock builds are bit-identical. When on, the
// PredictorLex owns an FpPriorRec (predictor_lex.zig) and installs a pointer to
// it here so byteUpdate can stream the post-×2-scale aux vector (== the exact
// prior PRL reads) to the replay file. Erased *anyopaque keeps this module free
// of a predictor_lex import cycle.
const FP_B = build_options.free_prior_b;
const FpSinkField = if (FP_B) ?*anyopaque else void;
const FpRecordField = if (FP_B) *const fn (*anyopaque, []const f32) void else void;

// -Dlstm-aux-sparse=K: keep only the K largest auxiliary masses, AT THEIR OWN
// SYMBOL POSITIONS, and zero the rest. 0 = stock (every branch below is
// comptime-dead and byteUpdate compiles to exactly the pre-knob code).
const aux_sparse: usize = @import("build_options").lstm_aux_sparse;

// -Dtransformer: swap the online LSTM for the frozen fx2 6M transformer in this
// exact slot. Comptime-dead at default — with the flag off `TF` is `false`,
// every branch below folds away, the `tf` field is `void` (zero size), and the
// C++ objects are not even compiled into the binary (build.zig
// `attachTransformer`). Stock archives are therefore bit-identical by
// construction, not by luck.
const tfmod = @import("transformer.zig");
const TF = tfmod.enabled;
// ⚠ OPTIONAL, and that is load-bearing. ByteMixer has TWO construction sites —
// `predictor_lex.zig` (the ship path) and `predictor.zig:334` (the older v21
// predictor, still live via main.zig and exercised by tests.zig's round-trip
// tests). Only the first installs a transformer. When this field was a bare
// `*Transformer` defaulting to `undefined`, the v21 path dereferenced garbage
// and SIGSEGV'd — caught by `zig build test -Dtransformer=true`, NOT by any
// roundtrip or bit-identity gate, because the ship path never touches it.
// Making it optional means a ByteMixer without a transformer keeps its LSTM
// behaviour by construction, so a THIRD construction site cannot reintroduce
// the crash. Same principle as the -Dhero / -Dfree-prior-b comptime refusals:
// make the bad state unrepresentable instead of fixing each site.
const TfField = if (TF) ?*tfmod.Transformer else void;

/// Top-K selection over `x`, emitted as PARALLEL (index, mass) arrays in
/// canonical slot order. Slot t holds the t-th largest mass and the symbol id
/// it belongs to; the caller scatters `vals[t]` back to position `ids[t]`, so
/// the LSTM's aux vector keeps its shipped width and its shipped
/// index-by-symbol layout — only the tail masses are dropped.
///
/// DETERMINISM CONTRACT (encode and decode MUST select the same K).
/// Selection is by the STRICT TOTAL ORDER  (mass DESC, index ASC):
///     (x_i, i) precedes (x_j, j)  iff  x_i > x_j, or (x_i == x_j and i < j).
/// No two distinct indices are equivalent under it, so "the K best under the
/// order" is a UNIQUE set in a unique slot order — there is no tie-broken
/// branch left for the two ops to disagree about. Concretely:
///   * the scan is `v = 0..x.len` ascending over a slice, never an unordered
///     container, so equal masses are OFFERED in ascending index order;
///   * admission uses strict `>` against the weakest held slot, so an equal
///     mass NEVER displaces an earlier (⇒ lower-index) equal mass;
///   * the insertion scan uses strict `>` too, so an equal mass is placed
///     AFTER the equal masses already held — i.e. lower index stays nearer
///     slot 0. Worked case, x = [5,5,5], K = 2: v=0 fills slot 0; v=1 is not
///     > 5 at the insert scan so it lands in slot 1; v=2 fails `> kv[K-1]`
///     (5 > 5 is false) and is rejected. Result ids [0,1] — the two lowest.
///     This is the KT-floor case (PPMd emits many exactly-equal tail masses)
///     and it is decided by comparison alone.
///   * comparisons are exact IEEE predicates: reassociation licences apply to
///     ARITHMETIC, and there is none here — only compares and copies. (This
///     file contains no `@setFloatMode` call at all, and the attribute is
///     per-scope and does not propagate from a caller, so this function sits
///     in Zig's default STRICT scope regardless.) NaN compares false
///     everywhere, so a NaN is simply never admitted (deterministically)
///     rather than poisoning the order.
/// Both ops run the SAME machine code over a bit-identically computed `x`
/// (`inputs` is a pure function of the PPMd byte distribution, already proven
/// symmetric by every existing roundtrip), so identical input ⇒ identical
/// output by construction.
///
/// Slot order is canonical (descending mass), not arrival order. For the
/// SPARSE design the slot order does not reach the model — the values are
/// scattered back to symbol positions, so any permutation of the slots gives
/// the same aux vector — but it does fix the summation order of the gathered
/// dot, so keeping it canonical keeps that order a function of the
/// distribution alone.
///
/// Returns the number of FILLED slots, `min(K, x.len)`. Slots beyond it would
/// carry the -inf sentinel; the caller passes only `ids[0..n]` so an over-large
/// K (K > vocab_size — not the shipped case) degrades to "fewer live symbols"
/// instead of scattering a spurious 0 over symbol 0's real mass.
pub inline fn selectTopK(comptime K: usize, x: []const f32, ids: *[K]u32, vals: *[K]f32) usize {
    var kv: [K]f32 = .{-std.math.inf(f32)} ** K;
    var ki: [K]u32 = .{0} ** K;
    for (x, 0..) |p, v| {
        if (!(p > kv[K - 1])) continue; // strict: ties keep the lower index
        var pos: usize = K - 1;
        while (pos > 0 and p > kv[pos - 1]) : (pos -= 1) {
            kv[pos] = kv[pos - 1];
            ki[pos] = ki[pos - 1];
        }
        kv[pos] = p;
        ki[pos] = @intCast(v);
    }
    for (0..K) |i| {
        vals[i] = if (std.math.isFinite(kv[i])) kv[i] else 0.0;
        ids[i] = ki[i];
    }
    return @min(K, x.len);
}

pub const ByteMixer = struct {
    base: ByteModel,
    lstm: *Lstm,
    byte: *const u32,
    byte_map: [256]i32 = .{0} ** 256,
    inputs: []f32,
    num_models: u32,
    vocab_size: u32,
    offset: u32 = 0,
    alloc: std.mem.Allocator,
    // -Dfree-prior-b prior recorder. `fp_prior_sink` null = not recording (set
    // by PredictorLex.fpPriorEnableRecording under a live schedule);
    // `fp_prior_record` is the append trampoline (installed at create). Both are
    // `void`-shaped when the flag is off.
    fp_prior_sink: FpSinkField = if (FP_B) null else {},
    fp_prior_record: FpRecordField = if (FP_B) undefined else {},
    /// -Dtransformer: installed by PredictorLex after create. `void` when off.
    tf: TfField = if (TF) null else {},

    pub fn create(a: std.mem.Allocator, num_models: u32, bit_context: *const u32, vocab: *const [256]bool, vocab_size: u32, lstm: *Lstm) *ByteMixer {
        const self = a.create(ByteMixer) catch unreachable;
        self.* = .{
            .base = ByteModel.init(vocab),
            .lstm = lstm,
            .byte = bit_context,
            .inputs = a.alloc(f32, vocab_size) catch unreachable,
            .num_models = num_models,
            .vocab_size = vocab_size,
            .alloc = a,
        };
        @memset(self.inputs, 0);
        var off: u32 = 0;
        for (0..256) |i| {
            self.byte_map[i] = @intCast(off);
            if (vocab[i]) off += 1;
        }
        return self;
    }

    /// Frees own allocations only — the Lstm is owned by the caller.
    pub fn destroy(self: *ByteMixer) void {
        // -Dslot-loss (LAB): the EXACT end-of-stream total. The periodic tick
        // alone strands the tail below the last multiple of `report_every`,
        // while the archive it is compared against covers the whole stream.
        if (comptime slot_loss.on) slot_loss.dump("final");
        self.alloc.free(self.inputs);
        self.alloc.destroy(self);
    }

    pub fn setInput(self: *ByteMixer, index: usize, val: f32) void {
        if (!self.base.vocab[index]) return;
        self.inputs[self.offset] += val;
        self.offset += 1;
        if (self.offset == self.vocab_size) self.offset = 0;
    }

    pub fn predict(self: *ByteMixer) []const f32 {
        return self.base.predict();
    }
    pub fn perceive(self: *ByteMixer, bit: i32) void {
        self.base.perceive(bit);
    }
    pub fn numOutputs(self: *ByteMixer) usize {
        _ = self;
        return 1;
    }

    pub fn byteUpdate(self: *ByteMixer) void {
        // ---- MERGE NOTE  -----------------------------------
        // Trunk's -Dslot-loss instrument and the -Dtransformer substitution
        // point both land at the top of byteUpdate. ORDER IS LOAD-BEARING:
        // slot_loss reads `base.probs[b]` for the byte that just finished
        // coding — i.e. the distribution the slot produced for THIS byte —
        // and the transformer block REWRITES `base.probs` and returns. So
        // slot_loss must run FIRST or it would read the wrong byte's row.
        // ★ Ordered this way the instrument keeps working under -Dtransformer
        // and measures whichever model occupies the slot, which is exactly
        // what a transformer slot measurement needs.
        // -Dslot-loss (LAB): the LSTM's own prequential slot loss for the byte
        // that just finished coding. `base.probs` still holds the distribution
        // the LSTM produced for THIS byte (predict/perceive only move top/bot;
        // probs is rewritten at the bottom of this function), and `inputs` still
        // holds the pre-scale auxiliary (PPMd) distribution for the same byte.
        // Comptime-dead at default.
        if (comptime slot_loss.on) {
            const b: usize = @intCast(self.byte.*);
            const pos: usize = @intCast(self.byte_map[b]);
            slot_loss.note(self.base.probs[b], if (pos < self.inputs.len) self.inputs[pos] else 0);
        }

        // ---- -Dtransformer: the substitution point ------------------------
        // Structurally the same object as the LSTM path below — same cadence
        // (once per byte, on byte completion), same input (the PPMd byte
        // distribution), same output (a distribution over the next byte written
        // into `base.probs`), same consumer (`base.byteUpdate`). Seven things
        // differ, and all seven are handled here or in transformer.zig:
        //
        //  1. NO x2 SCALE. `self.inputs` at this point is the raw accumulated
        //     PPMd row (one setInput per byte, predictor_lex.zig:1913-1914).
        //     The x2 exists only to condition the LSTM's aux INPUT channel;
        //     their weights were trained against the unscaled
        //     `byte_model_->BytePredict` row (predictor.cpp:547), so scaling
        //     it here would evaluate the frozen model off-distribution.
        //     ⇒ the scale loop is skipped, not applied-and-undone.
        //  2. f16 BOUNDARY: done verbatim inside the shim, quirk included.
        //  3. -Dlstm-aux-sparse's top-8 truncation is LSTM-specific and is NOT
        //     applied — they use the prior dense.
        //  4. -Dlstm-prior-form=1 has a transformer analogue after all; see
        //     transformer.zig `priorFold` and -Dtransformer-prior-form.
        //  5. No online training: `step` takes no label.
        //  6. Per-article context reset: inside the shim, from their own
        //     15-token separator window.
        //  7. V = 205 canonical, not our vocab_size: transformer.zig `TOK`.
        //
        // -Dfree-prior-b is NOT recorded on this path: the replay epochs exist
        // to retrain the online LSTM and a frozen model has nothing to retrain
        // (design memo §3.4.1). The knob is rejected at build time below.
        if (comptime TF) {
            comptime {
                if (FP_B) @compileError("-Dtransformer is incompatible with -Dfree-prior-b: the replay epochs retrain the online LSTM, and a frozen transformer has nothing to retrain (design memo §3.4.1)");
            }
            if (self.tf) |tf| {
                var out256: [256]f32 = undefined;
                _ = tf.byteUpdate(@intCast(self.byte.*), self.inputs, self.base.vocab, &self.byte_map, &out256);
                @memset(self.inputs, 0);
                for (0..256) |i| {
                    self.base.probs[i] = if (self.base.vocab[i]) out256[i] else 0;
                }
                self.base.byteUpdate();
                return;
            }
            // No transformer installed on this ByteMixer (the v21 predictor.zig
            // path) — fall through to the stock LSTM path below, unchanged.
            //
            // -Dtransformer-lstm-dead asserts that this cannot happen, which is
            // TRUE of the ship binaries (`form1`, `ship`): their only ByteMixer
            // comes from `predictor_lex`, which installs a transformer or
            // panics. Asserting it makes the entire LSTM compute path below
            // unreachable and it is eliminated — measured -4,112 B packed on the
            // transformer-era prefix, = -8,224 B of S. `predictor.zig` refuses the flag at
            // compile time, so this `unreachable` cannot be reached by any build
            // the compiler accepts.
            if (comptime tfmod.lstm_dead) unreachable;
        }
        // inputs_ *= 2 / num_models_  (integer division as in cmix; guard /0)
        const scale: f32 = if (self.num_models == 0) 0 else @floatFromInt(2 / self.num_models);
        for (self.inputs) |*x| x.* *= scale;
        // -Dfree-prior-b: record the post-scale prior (== the vector PRL reads
        // from layer_input_[epoch][0]) BEFORE the aux hand-over. Comptime-dead
        // when off, and inert unless PredictorLex has installed a sink
        // (forward-coding only — the replay path bypasses byteUpdate). One
        // prior per byte. Under -Dlstm-aux-sparse the recorded row is the
        // DENSE distribution (`inputs` stays live and unmodified below), which
        // is exactly what setAuxPrior hands PRL — so record and fold see the
        // same vector under every knob combination.
        if (comptime FP_B) {
            if (self.fp_prior_sink) |sink| self.fp_prior_record(sink, self.inputs);
        }
        if (comptime aux_sparse > 0) {
            // The LSTM's aux INPUT keeps its width and its symbol positions but
            // carries only the K largest masses; the PRL fold still needs the
            // DENSE distribution (it multiplies the output softmax by prior[i]
            // for every symbol i), so that is handed over out-of-band and the
            // fold stays EXACTLY as shipped. `inputs` therefore has to stay
            // live and unmodified across perceive — the clear moves below.
            var ids: [aux_sparse]u32 = undefined;
            var vals: [aux_sparse]f32 = undefined;
            const n = selectTopK(aux_sparse, self.inputs, &ids, &vals);
            self.lstm.setSparseInput(ids[0..n], vals[0..n]);
            self.lstm.setAuxPrior(self.inputs);
        } else {
            self.lstm.setInput(self.inputs);
            @memset(self.inputs, 0);
        }
        const output = self.lstm.perceive(@intCast(self.byte_map[@intCast(self.byte.*)]));
        if (comptime aux_sparse > 0) @memset(self.inputs, 0);
        var off: usize = 0;
        for (0..256) |i| {
            if (self.base.vocab[i]) {
                self.base.probs[i] = output[off];
                off += 1;
            } else {
                self.base.probs[i] = 0;
            }
        }
        self.base.byteUpdate();
    }

    pub fn model(self: *ByteMixer) Model {
        return .{ .ptr = self, .vtable = vtableFor(ByteMixer) };
    }
};

// ---- selectTopK determinism gates -------------------------------------------
// A desynchronising decoder is a submission-killer, so the tie behaviour is
// proven here rather than inferred from a roundtrip (a roundtrip that happened
// to hit no ties would pass while the selection was still ill-defined). The
// oracle below is the DEFINITION — the strict total order (mass DESC, id ASC)
// — brute-forced by exhaustive argmax, and the quantized inputs guarantee the
// tie case is exercised on essentially every draw.
//
// Ported from the `claude/lstm-aux-topk` branch's `encodeTopK` gates
// (the 400-trial x K oracle), retargeted
// from the killed 2K-wide [masses | ids/255] layout to the (index, mass) pairs
// the SPARSE design scatters back to symbol positions. The tie semantics under
// test are identical; the layout gate (#6) is new and is the sparse design's
// load-bearing invariant.

/// Reference selection: repeatedly take the (mass DESC, index ASC)-best
/// remaining element. O(K*V), obviously correct, never shipped.
fn refTopK(comptime K: usize, x: []const f32, ids: *[K]u32, masses: *[K]f32) void {
    var used = [_]bool{false} ** 256;
    for (0..K) |slot| {
        var best: ?usize = null;
        for (x, 0..) |p, v| {
            if (used[v]) continue;
            if (best == null or p > x[best.?]) best = v; // strict: ties keep lower v
        }
        used[best.?] = true;
        ids[slot] = @intCast(best.?);
        masses[slot] = x[best.?];
    }
}

test "selectTopK: exact ties keep the lowest indices (the KT-floor case)" {
    const K = 3;
    var x = [_]f32{0.125} ** 8; // every symbol at the same floor mass
    var ids: [K]u32 = undefined;
    var vals: [K]f32 = undefined;
    _ = selectTopK(K, &x, &ids, &vals);
    for (0..K) |i| {
        try std.testing.expectEqual(@as(f32, 0.125), vals[i]);
        try std.testing.expectEqual(@as(u32, @intCast(i)), ids[i]);
    }
}

test "selectTopK: partial ties order by (mass DESC, id ASC)" {
    const K = 3;
    var x = [_]f32{ 0.5, 0.9, 0.5, 0.9, 0.1 };
    var ids: [K]u32 = undefined;
    var vals: [K]f32 = undefined;
    _ = selectTopK(K, &x, &ids, &vals);
    const want_m = [K]f32{ 0.9, 0.9, 0.5 };
    const want_i = [K]u32{ 1, 3, 0 };
    for (0..K) |i| {
        try std.testing.expectEqual(want_m[i], vals[i]);
        try std.testing.expectEqual(want_i[i], ids[i]);
    }
}

test "selectTopK: slots are canonical (descending mass), not arrival order" {
    const K = 4;
    var x = [_]f32{ 0.1, 0.9, 0.5, 0.7, 0.3, 0.8 };
    var ids: [K]u32 = undefined;
    var vals: [K]f32 = undefined;
    _ = selectTopK(K, &x, &ids, &vals);
    for (1..K) |i| try std.testing.expect(vals[i] <= vals[i - 1]);
    try std.testing.expectEqual(@as(f32, 0.9), vals[0]);
    try std.testing.expectEqual(@as(u32, 1), ids[0]);
}

test "selectTopK: K > vocab reports n = vocab and never emits the -inf sentinel" {
    const K = 4;
    var x = [_]f32{ 0.5, 0.25 };
    var ids: [K]u32 = undefined;
    var vals: [K]f32 = undefined;
    const n = selectTopK(K, &x, &ids, &vals);
    // ★ n is what protects symbol 0: the caller passes ids[0..n], so the two
    // unfilled slots are never scattered and cannot write 0 over x[0]'s mass.
    try std.testing.expectEqual(@as(usize, 2), n);
    for (vals) |v| try std.testing.expect(std.math.isFinite(v));
    try std.testing.expectEqual(@as(f32, 0.5), vals[0]);
    try std.testing.expectEqual(@as(f32, 0.25), vals[1]);
    try std.testing.expectEqual(@as(f32, 0.0), vals[2]);
    try std.testing.expectEqual(@as(f32, 0.0), vals[3]);
    var got = [_]f32{0} ** 2;
    for (ids[0..n], vals[0..n]) |ix, v| got[ix] = v;
    try std.testing.expectEqualSlices(f32, &x, &got);
}

test "selectTopK: matches the brute-force oracle on tie-dense random draws" {
    var prng = std.Random.DefaultPrng.init(0xA47C);
    const r = prng.random();
    inline for (.{ 1, 2, 4, 8, 16, 32 }) |K| {
        var trial: usize = 0;
        while (trial < 400) : (trial += 1) {
            // Quantize to 6 levels over a 200-symbol vocab: ~33 exact ties per
            // level, so the tie path is hit on every single trial.
            var x: [200]f32 = undefined;
            for (&x) |*p| p.* = @as(f32, @floatFromInt(r.uintLessThan(u32, 6))) / 8.0;
            var ids: [K]u32 = undefined;
            var vals: [K]f32 = undefined;
            _ = selectTopK(K, &x, &ids, &vals);
            var rids: [K]u32 = undefined;
            var rmasses: [K]f32 = undefined;
            refTopK(K, &x, &rids, &rmasses);
            for (0..K) |i| {
                try std.testing.expectEqual(rmasses[i], vals[i]);
                try std.testing.expectEqual(rids[i], ids[i]);
            }
        }
    }
}

test "selectTopK: scatter reproduces the dense vector with the tail zeroed (the SPARSE layout invariant)" {
    // ★ The load-bearing invariant of the sparse design, and the one that makes
    // the PRL fold's index-by-symbol read safe: after scatter, position s holds
    // EITHER x[s] (s is live) OR 0 (s is not) — never another symbol's mass and
    // never a shifted/renumbered slot. Compared against an independent oracle:
    // "zero every element outside the reference top-K set".
    var prng = std.Random.DefaultPrng.init(0x5A17);
    const r = prng.random();
    inline for (.{ 4, 8, 16, 32 }) |K| {
        var trial: usize = 0;
        while (trial < 200) : (trial += 1) {
            var x: [200]f32 = undefined;
            for (&x) |*p| p.* = @as(f32, @floatFromInt(r.uintLessThan(u32, 6))) / 8.0;
            var ids: [K]u32 = undefined;
            var vals: [K]f32 = undefined;
            _ = selectTopK(K, &x, &ids, &vals);
            var got = [_]f32{0} ** 200;
            for (ids, vals) |ix, v| got[ix] = v;

            var rids: [K]u32 = undefined;
            var rmasses: [K]f32 = undefined;
            refTopK(K, &x, &rids, &rmasses);
            var want = [_]f32{0} ** 200;
            for (rids) |ix| want[ix] = x[ix]; // the ORIGINAL value, at its OWN position
            try std.testing.expectEqualSlices(f32, &want, &got);
            // the K ids are DISTINCT — otherwise a scatter would silently drop
            // a live symbol and the live set would be smaller than K.
            var seen = [_]bool{false} ** 200;
            for (ids) |ix| {
                try std.testing.expect(!seen[ix]);
                seen[ix] = true;
            }
        }
    }
}
