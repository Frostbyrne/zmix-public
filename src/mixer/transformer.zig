//! -Dtransformer: the fx2-cmix-transformer 6M frozen transformer as a drop-in
//! replacement for the online LSTM in the byte-mixer slot (`in[589]`).
//!
//! Spec:.
//! Implementation notes + the disabled-feature decisions:
//!
//! The heavy lifting is the vendored C++ in `third_party/fx2_transformer`,
//! compiled by `zig build`'s own bundled clang at the pinned `x86_64_v3`
//! baseline with `-mrecip=none` (see build.zig). This file owns only three
//! things, all of which are zmix-side and none of which exist in their tree:
//!
//!   1. **The canonical-205 token map.** Their model hard-Fails unless
//!      `vocab_size == 205`. Our e9 `prep.temp` is 256-valued because it
//!      carries the 679,505-byte r1 side footer (0.116 % of the stream), and
//!      our smaller tiers read 202. So the token map cannot be "our vocab" —
//!      it is the frozen canonical set of the 205 byte values that occur in
//!      the post-WRT main region, exactly the class of constant their own
//!      `kArticleSeparator` is.
//!
//!   2. **The prior/output marshalling** between our `vocab_size`-wide
//!      compacted vectors and their 205-wide canonical ones.
//!
//!   3. **`-Dtransformer-prior-form=1`: the PRL fold, carried across.**
//!      See `priorFold` below — this is the load-bearing design decision of
//!      the whole port.
const std = @import("std");
const topts = @import("transformer_options");

pub const V: usize = 205;

pub const enabled: bool = topts.transformer;

// `link_libcpp` drags libc++abi's exception machinery and the whole Itanium
// demangler into a binary compiled `-fno-exceptions -fno-rtti` that can never
// throw. `cxx_noexcept.zig` (the ABI-tagged `std::__1::__throw_*` hooks) and
// `src/cxx_local.cpp` (std::string, to_string, __next_prime, operator
// new/delete, ...) supply every libc++ symbol the vendored objects need, so no
// archive member is ever extracted and the whole chain leaves with them.
// MEASURED on the transformer-era arm-C prefix: packed 198,712 -> 167,844 B, i.e.
// -30,868 B packed = -61,736 B of S (Form-1 charges decomp_bin TWICE).
// Referenced ONLY under `enabled`, so a stock LSTM-era build is untouched by
// construction — verified: its prefix is byte-identical across this change.
comptime {
    if (enabled) _ = @import("../cxx_noexcept.zig");
}

/// -Dtransformer-lstm-dead. See build.zig for what it asserts and what it is
/// worth; `byte_mixer.zig` and `predictor.zig` are its only consumers.
pub const lstm_dead: bool = topts.transformer_lstm_dead;
pub const prior_form: u32 = topts.transformer_prior_form;
pub const prior_rho: f32 = topts.transformer_prior_rho;
pub const prior_clamp: u32 = topts.transformer_prior_clamp;

/// ⛔ **NEGATIVE CONTROL ONLY — `-Dtransformer-map-fault`, default 0.**
///
/// A gate that has never been observed to fire is a comment, not a gate. This
/// knob deliberately corrupts the canonical-205 map so the guard below can be
/// PROVEN to abort, and it is the only way to demonstrate that, because every
/// other gate this port owns is structurally blind to a wrong map (a lossless
/// roundtrip agrees with itself; bit-identity runs with the arm OFF).
///
///   0 — off. The whole mechanism is comptime-dead: `FAULT_SWAP` is `null`,
///       both map blocks fold to their base tables, and the emitted binary is
///       byte-identical to one built before this knob existed.
///   1 — swap ranks 5/6 (the α-map's own disagreement). `TOK[0x0a]` becomes 6,
///       so the FAST PRE-CHECK fires.
///   2 — swap ranks 172/173. `TOK[0x0a] == 5` and `TOK[0x20] == 8` still hold,
///       so the pre-check PASSES and only the SEPARATOR COUNT catches it. This
///       is the arm that proves the count is strictly stronger than any
///       single-rank check — the claim `EXPECTED_SEPARATORS` is written on.
///
/// ⛔ It must never appear in a ship recipe: `construct_ship.sh` refuses a
/// nonzero value by name, the same way the r1v2 Form-1 gate does.
pub const map_fault: u32 = topts.transformer_map_fault;

// ---- extern "C" boundary (third_party/fx2_transformer/fx2_shim.h) ----------
extern fn fx2tf_create(weights_path: [*:0]const u8, attn_kind: c_int) ?*anyopaque;
extern fn fx2tf_destroy(h: ?*anyopaque) void;
extern fn fx2tf_byte_update(h: ?*anyopaque, token: u8, prior205: [*]const f32, out205: [*]f32) c_int;
extern fn fx2tf_passthrough(h: ?*anyopaque, prior205: [*]const f32, out205: [*]f32) void;
extern fn fx2tf_floor(p205: [*]f32) void;
extern fn fx2tf_separator_hits(h: ?*anyopaque) u64;
extern fn fx2tf_floats_to_halves(src: [*]const f32, dst: [*]u16, n: u64) void;
extern fn fx2tf_halves_to_floats(src: [*]const u16, dst: [*]f32, n: u64) void;

// ---- The canonical 205-value alphabet --------------------------------------
// MEASURED, not assumed: the
// distinct byte values of our own post-WRT+r1 main region, 586,459,321 B of
// real e9. The 51 complementary values occur ONLY in the r1 side footer.
//
// This table is cross-checked against three independent anchors at comptime by
// `checkCanonicalMap` below, and by the unit tests at the bottom of this file:
//   idx[0x20] == 8, idx[0x0a] == 5, idx[0xdf] == 172 (== 0xac), and their
//   `kArticleSeparator` indices decode through it to the byte string
//   "  " ++ <page-code> ++ "\n" ++ "    " ++ <title-code>.
// A one-value error anywhere below 0xac would shift the map and break all
// three at once.
pub const ABSENT_205 = [51]u8{
    0x00, 0x01, 0x02, 0x04, 0x08, 0x0b, 0x0d, 0x0e, 0x0f, 0x10,
    0x11, 0x12, 0x14, 0x15, 0x16, 0x17, 0x18, 0x19, 0x1a, 0x1b,
    0x1c, 0x1d, 0x1e, 0x1f, 0x3a, 0x3b, 0x3c, 0x3d, 0x3e, 0x3f,
    0x41, 0x42, 0x43, 0x44, 0x45, 0x46, 0x47, 0x48, 0x49, 0x54,
    0x55, 0x56, 0x57, 0x59, 0x5a, 0x60, 0x7b, 0x7c, 0x7d, 0x7e,
    0x7f,
};

/// `tok[b]` = canonical-205 index of byte `b`, or -1 if `b` is one of the 51
/// values that occur only in the r1 side footer.
///
/// ⚠ THE BASE MAP. Every comptime assertion below is taken against THIS table,
/// unconditionally, so `-Dtransformer-map-fault` cannot switch the derivation
/// gate off — it only perturbs the tables the engine actually uses.
const TOK_BASE: [256]i16 = blk: {
    var absent = [_]bool{false} ** 256;
    for (ABSENT_205) |b| absent[b] = true;
    var t: [256]i16 = .{-1} ** 256;
    var n: i16 = 0;
    for (0..256) |b| {
        if (!absent[b]) {
            t[b] = n;
            n += 1;
        }
    }
    if (n != V) @compileError("canonical alphabet is not 205 values");
    break :blk t;
};

/// Inverse of `TOK_BASE`: canonical index -> byte value.
const BYTE_OF_BASE: [V]u8 = blk: {
    var inv: [V]u8 = .{0} ** V;
    for (0..256) |b| {
        if (TOK_BASE[b] >= 0) inv[@intCast(TOK_BASE[b])] = @intCast(b);
    }
    break :blk inv;
};

/// The rank pair `-Dtransformer-map-fault` swaps. `null` at the default, which
/// makes both tables below fold to the base ones with no code emitted.
const FAULT_SWAP: ?[2]usize = switch (map_fault) {
    0 => null,
    1 => [2]usize{ 5, 6 },
    2 => [2]usize{ 172, 173 },
    else => @compileError("-Dtransformer-map-fault: only 0 (off), 1 (rank 5/6 swap, trips the pre-check) and 2 (rank 172/173 swap, trips only the separator count) exist"),
};

/// `TOK[b]` = the index the ENGINE uses. Equals `TOK_BASE` unless the negative
/// control is armed.
pub const TOK: [256]i16 = blk: {
    var t = TOK_BASE;
    if (FAULT_SWAP) |sw| {
        t[BYTE_OF_BASE[sw[0]]] = @intCast(sw[1]);
        t[BYTE_OF_BASE[sw[1]]] = @intCast(sw[0]);
    }
    break :blk t;
};

/// Inverse of `TOK`.
pub const BYTE_OF: [V]u8 = blk: {
    var inv = BYTE_OF_BASE;
    if (FAULT_SWAP) |sw| {
        const tmp = inv[sw[0]];
        inv[sw[0]] = inv[sw[1]];
        inv[sw[1]] = tmp;
    }
    break :blk inv;
};

/// Their `kArticleSeparator` (fx2_shim.cpp:111, VERBATIM from their
/// predictor.cpp:103-119), expressed in vocabulary INDICES.
pub const SEPARATOR_IDX = [15]u8{ 0x08, 0x08, 0x25, 0xac, 0x65, 0x27, 0x05, 0x08, 0x08, 0x08, 0x08, 0x25, 0xac, 0x68, 0x27 };
/// What those 15 indices MUST decode to through our map: the WRT-encoded
/// "  <page>\n    <title>" article separator.
pub const SEPARATOR_BYTES = [15]u8{ 0x20, 0x20, 0x4c, 0xdf, 0x98, 0x4e, 0x0a, 0x20, 0x20, 0x20, 0x20, 0x4c, 0xdf, 0x9b, 0x4e };

comptime {
    // The three anchors from the design memo, checked at compile time against
    // the BASE map (see TOK_BASE) so the negative-control knob cannot disarm
    // them.
    if (TOK_BASE[0x20] != 8) @compileError("canonical map: idx[0x20] != 8");
    if (TOK_BASE[0x0a] != 5) @compileError("canonical map: idx[0x0a] != 5");
    if (TOK_BASE[0xdf] != 172) @compileError("canonical map: idx[0xdf] != 172");

    // ★ THE FULL 15-TOKEN CONSTANT, not just two ranks. The submission gate asks for
    // "the FULL 15-token separator" precisely because a systematic off-by-one
    // above rank 7 passes every single-rank check: the separator spans 0x20
    // (rank 8) and 0xdf (rank 172), both ABOVE rank 7, so it does not. This is
    // the strongest statement about the map that can be made without running
    // the coder, and it costs ZERO bytes — it exists only at compile time.
    for (SEPARATOR_IDX, 0..) |ix, k| {
        if (BYTE_OF_BASE[ix] != SEPARATOR_BYTES[k])
            @compileError("canonical map: their kArticleSeparator does NOT decode to the WRT article separator through this table. One rank is wrong; every prediction the frozen weights make would be made under the wrong token ids, and no roundtrip, bit-identity or ISA gate can see it.");
    }

    // ---- THE α-MAP DISCRIMINATOR (map dispute RESOLVED) ----------
    //
    // Two maps circulated and BOTH derivations were correct about their own
    // input. They differ at ranks 0..7 ONLY and agree on every rank >= 8:
    //
    //   THIS MAP (S1, correct for us): low bytes {03,05,06,07,09,0a,0c,13}
    //                                  => TOK[0x0a] = 5
    //   the α map:                     low bytes {00,03,05,06,07,09,0a,0c}
    //                                  => TOK[0x0a] = 6
    //
    // Under this map the SHIPPED kArticleSeparator decodes to
    //   20 20 4c df 98 4e 0a 20 20 20 20 4c df 9b 4e
    // Under the α map it decodes to the same string but with 0x09 (TAB) where
    // 0x0a (NEWLINE) belongs — which is exactly why α measured 0 occurrences,
    // and why changing index 6 from 0x05 to 0x06 repaired it *under α's map*.
    // Both observations were simultaneously true.
    //
    // α's vocabulary came from the full `-e` temp INCLUDING the 5-byte
    // [TEXT][len32] block header that preprocessor::EncodeSegment writes. For
    // .ready4cmix = 941,105,152 = 0x38182000 that header injects 0x00 (rank 0 —
    // precisely what pushes 0x0a from 5 to 6) and 0x18. OUR coder codes the
    // main region directly and never sees that header.
    //
    // Two independent grounds for this map, either sufficient:
    //  (1) the shipped binary matches the SHIPPED constant. If TOK[0x0a] were 6
    //      the separator would never fire, the transformer would never reset per
    //      article, and their published 0.9131822825 would be unreachable.
    //  (2) this decode occurs 243,426x — the exact enwik9 article count — in
    //      BOTH our stream and theirs.
    //
    // ⇒ These two assertions are the discriminator. 0x00 and 0x18 are the ONLY
    //   bytes that separate the two maps, so a future re-derivation that picks
    //   up the block header fails HERE, at compile time, instead of silently
    //   shifting every token id below 0x13 and making every prediction wrong.
    if (TOK_BASE[0x00] >= 0) @compileError("canonical map: 0x00 is PRESENT — this is the alpha map, derived from a temp that includes the 5-byte [TEXT][len32] block header. Our coder codes the main region directly and never sees it. Using this map shifts every rank below 0x13 and silently breaks every prediction.");
    if (TOK_BASE[0x18] >= 0) @compileError("canonical map: 0x18 is PRESENT — same block-header contamination as 0x00 above.");
}

// ---- The r1 side-footer detector -------------------------------------------
//
// ⚠ THE DESIGN MEMO'S RULE CANNOT BE IMPLEMENTED, and the reason is a
// decoder-symmetry argument rather than an engineering inconvenience.
//
// `r1_reorder.zig:466-476` builds
//     prep.temp = main(input.len) ++ side ++ kFooterMagic(8) ++ u64(side.len)
// and the memo (§2c item 2) says to pass PPMd through "for the entire footer
// region". But that split is recoverable only from the TRAILING footer: the
// ENCODER knows `input.len`, the DECODER does not — it decodes `prep.temp`
// sequentially and peels the footer only after the whole stream is out. A model
// that changes behaviour at `main_len` would use information the decoder does
// not have at that point, and THE ARITHMETIC CODER WOULD DESYNCHRONISE.
//
// ⇒ any rule here must be a function of the DECODED PREFIX ALONE.
//
// The side blob carries its OWN leading magic (`r1_reorder.zig:87`, written at
// `:513` as the first bytes of `side`, appended directly after `main` at `:474`),
// so the causal rule is:
//
//     once the last 7 decoded bytes equal "R1ORD3\n", the footer has begun;
//     from that point on, never step the transformer again.
//
//   * CAUSAL — a function of the decoded prefix, so both ops agree by
//     construction. This is exactly what the `main_len` rule lacks.
//   * EXACT up to a false positive in the main text: 7 bytes, ~2^-56 per
//     position, ~8e-9 over 586 M positions. And a false positive is HARMLESS —
//     both ops still agree, we merely stop the transformer early.
//   * FAILS SAFE — if `-Dfieldcodec-tsswap` ever perturbed those bytes the rule
//     simply never fires and the alphabet rule below catches the remainder.
//
// The window is over RAW BYTES, not tokens, because the magic is a byte string
// and 4 of its 7 bytes ('R','1','O','R','D','3') are ordinary vocabulary bytes.
pub const SIDE_MAGIC = [_]u8{ 'R', '1', 'O', 'R', 'D', '3', '\n' };

// ---- The PRL fold, carried across ------------------------------------------
//
// ★ THE DESIGN DECISION THIS PORT TURNS ON (operator re-scoping
//
// fx2-cmix's byte-mixer slot carries a BARE LSTM softmax that merely receives
// the PPMd row as an input. Ours does not: `-Dlstm-prior-form=1` multiplies the
// LSTM's output softmax by `A_i = max(P_i, 2^-13)^0.5` and renormalizes
// (`lstm.zig:735-748`, `powDyadic` `:53-62`), so `in[589]` carries an
// LSTM (x) PPMd PRODUCT. That fold is measured to be *why* our slot is stronger
// than theirs (slot-code-length / own-ensemble ratio 1.12-1.25 vs their 1.8864
// / 1.6404), and the S1 screen's provisional STOP is precisely the finding that
// swapping it away is what would make their transformer's advantage real.
//
// The design memo listed "PRL has no transformer analogue" as routine. It is
// not: the fold is a PURE POST-HOC REWEIGHTING of a normalized 205-way
// distribution by a function of the PPMd prior. Nothing in it is LSTM-specific
// — it never touches the hidden state, the gradient, or the training path. The
// transformer's output plays exactly the role `softmax(z)` plays in the LSTM,
// so the fold transfers UNCHANGED and the port does NOT have to trade our
// structural advantage away.
//
// `-Dtransformer-prior-form`:
//   0 (default) = their verbatim contract. The slot is the bare transformer
//       distribution. This reproduces fx2-cmix-transformer exactly and is the
//       correct control arm.
//   1 = the zmix PRL analogue. `R_i = Q_i * max(2*P_i, 2^-C)^rho`, renormalized.
//
// The factor 2 is not cosmetic. The LSTM's `priorVec` is the ByteMixer's
// post-x2-scale aux vector (`byte_mixer.zig:158-159`, num_models == 1), and
// under form 1 the x2 does NOT cancel, because `max` decides which entries
// clamp: it shifts the clamp threshold by one power of two. Feeding the raw
// row here would be a silently different fold, so the x2 is applied
// explicitly.
//
// The transformer itself is fed the UNSCALED row, because that is what their
// weights were trained against (`byte_model_->BytePredict`, predictor.cpp:547).
// The x2 exists only for the LSTM's aux input channel and is not part of their
// contract.

/// `x^tau` for dyadic tau using IEEE sqrt only — exactly rounded, so
/// bit-exact and libm-free, which the encoder/decoder determinism contract
/// requires. Byte-for-byte the same construction as `lstm.zig:powDyadic`.
inline fn powDyadic(x: f32) f32 {
    return switch (comptime @as(u32, @intFromFloat(prior_rho * 100.0 + 0.5))) {
        25 => @sqrt(@sqrt(x)),
        50 => @sqrt(x),
        75 => @sqrt(x) * @sqrt(@sqrt(x)),
        100 => x,
        else => @compileError("-Dtransformer-prior-form=1 requires -Dtransformer-prior-rho in {0.25,0.5,0.75,1.0}"),
    };
}

/// In-place `q_i *= max(2*p_i, 2^-C)^rho`, then renormalize. Mirrors the
/// `lstm_prior_form == 1` branch of `lstm.zig:735-748`.
pub fn priorFold(q: *[V]f32, p: *const [V]f32) void {
    const fl: f32 = comptime 2.0 / @as(f32, @floatFromInt(@as(u64, 1) << @as(u6, @intCast(prior_clamp))));
    var total: f32 = 0;
    for (0..V) |i| {
        const a = powDyadic(@max(2.0 * p[i], fl));
        q[i] *= a;
        total += q[i];
    }
    if (total > 0) {
        for (0..V) |i| q[i] /= total;
    }
}

// ---- The handle ------------------------------------------------------------

pub const Transformer = struct {
    h: *anyopaque,
    /// scratch: the PPMd row compacted to the canonical 205 set, unscaled
    prior: [V]f32 = .{0} ** V,
    /// scratch: the model's distribution over the next byte
    out: [V]f32 = .{0} ** V,
    /// Rolling window of the last 7 RAW bytes, for the r1 side-footer detector.
    magic_window: [SIDE_MAGIC.len]u8 = .{0} ** SIDE_MAGIC.len,
    /// Set once the side footer has begun; never cleared. See `SIDE_MAGIC`.
    in_footer: bool = false,
    /// diagnostics — the engagement witness (see `witness`)
    n_steps: u64 = 0,
    n_boundaries: u64 = 0,
    n_offvocab: u64 = 0,
    n_footer: u64 = 0,

    /// SHIP PATH slot: set by `ship.zig` to the `tfweights_comp` segment sliced
    /// out of the running artifact's own tail, BEFORE the predictor is built.
    ///
    /// A module-level slot rather than a constructor parameter on purpose:
    /// `PredictorLex` is constructed from several call sites (bench CLI, ship
    /// encode, ship decode) and only the ship paths have an artifact to slice.
    /// Threading an `?[]const u8` through all of them to serve two would be a
    /// wider blast radius than this, for no added safety — `enabled` already
    /// gates the whole block at comptime.
    ///
    /// Null => fall back to `-Dtransformer-weights`, which is DEV-ONLY. See
    /// `createFromBlob` for why a shipped binary must never take that path.
    pub var embedded_blob: ?[]const u8 = null;

    /// CONSTRUCT-TIME slot: an EXPLICIT weights path handed to `zmix_ship -c/-d`
    /// as a fourth argument, set by `ship.zig` before the predictor is built.
    ///
    /// Why a third source rather than reusing `embedded_blob`: at asset-compression
    /// time there IS no artifact to slice — `construct_ship.sh` runs `engine -c
    /// english.dic dict.comp` on the bare packed engine, whose tail is empty, and only
    /// AFTER that does it concatenate comp9. So the one path that cannot read its own
    /// tail is exactly the path a committee rebuild runs first. It used to fall back to
    /// the build-time `-Dtransformer-weights` string, which was an ABSOLUTE dev-box
    /// path: the rebuild died `weights_io: cannot open <path>`.
    ///
    /// Points at `argv`, which outlives the predictor. Null on every ship path.
    pub var weights_path_override: ?[]const u8 = null;

    /// SHIP PATH: build from the weights blob EMBEDDED in our own artifact,
    /// never from a path supplied at build time.
    ///
    /// ⛔ WHY THIS EXISTS. `create` below takes a path, and `-Dtransformer-weights`
    /// bakes an ABSOLUTE one into the binary. That is exactly the defect that made
    /// an earlier build unsubmittable: its `decomp_bin` carried an absolute build-machine path
    /// and panicked on a judge box, after passing every ratio, RAM and wall gate.
    /// A shipped decoder may depend on NOTHING outside its own bytes.
    ///
    /// The C boundary is `fx2tf_create(const char* path)` and the chain beneath it
    /// (`TransformerOpt` -> `OptModel::load` -> `WeightsFile::load_compressed`) is
    /// path-based four layers deep, so a from-memory entry point means changing all
    /// four. Instead materialise the blob to a **CWD-relative, PID-stamped** temp,
    /// load, and unlink — the pattern `ppmd.zig:55,358` already uses for
    /// `ppm.{pid}.temp`, which cmix-lex itself does and rule 7 explicitly permits
    /// (100 GB of temp files). The blob is ~1.9 MB and is read exactly once.
    ///
    /// ⚠ Deliberately NOT `/tmp`: the HPJA build container's only writable mount is
    /// `/work` and `/tmp` is a symlink into a directory created at the judge's umask
    /// (see `submission/hpja/make_package.sh`). CWD-relative is the portable choice.
    pub fn createFromBlob(a: std.mem.Allocator, blob: []const u8, attn_kind: u32) !*Transformer {
        if (blob.len == 0) return error.EmptyWeightsBlob;
        var name_buf: [64]u8 = undefined;
        const name = try std.fmt.bufPrintZ(
            &name_buf,
            "tfw.{d}.tmp",
            .{std.os.linux.getpid()},
        );
        {
            const f = try std.fs.cwd().createFile(name, .{ .truncate = true });
            defer f.close();
            try f.writeAll(blob);
        }
        // Unlink whatever happens next: a failed load must not leave the artifact
        // behind for a later run to pick up.
        defer std.fs.cwd().deleteFile(name) catch {};
        return create(a, name, attn_kind);
    }

    pub fn create(a: std.mem.Allocator, weights_path: [:0]const u8, attn_kind: u32) !*Transformer {
        const h = fx2tf_create(weights_path.ptr, @intCast(attn_kind)) orelse
            return error.TransformerLoadFailed;
        const self = try a.create(Transformer);
        self.* = .{ .h = h };
        return self;
    }

    pub fn destroy(self: *Transformer, a: std.mem.Allocator) void {
        fx2tf_destroy(self.h);
        a.destroy(self);
    }

    /// One byte-cadence update.
    ///   `byte`      the byte that just completed (0..255, raw)
    ///   `inputs`    the ByteMixer's compacted PPMd row, UNSCALED (no x2)
    ///   `vocab`     which byte values are live in this run's alphabet
    ///   `byte_map`  byte value -> index into `inputs` (ByteMixer.byte_map)
    ///   `out256`    receives the distribution over the next byte; entries for
    ///               bytes outside the canonical 205 set are set to 0.
    ///
    /// The gather handles both directions of the vocab mismatch the design memo
    /// §2c flags: at small tiers our alphabet is a SUBSET of the canonical 205
    /// (202 at e8_1m) and the 3 missing symbols simply carry prior 0; at e9 it
    /// is a SUPERSET (256, because of the r1 side footer) and the 51 extra
    /// values have no token at all.
    ///
    /// Returns true if the transformer actually stepped (false at a piece
    /// boundary or on an off-canonical byte, where the PPMd row is passed
    /// through — their `predictor.cpp:566-576` contract).
    pub fn byteUpdate(
        self: *Transformer,
        byte: u8,
        inputs: []const f32,
        vocab: *const [256]bool,
        byte_map: *const [256]i32,
        out256: *[256]f32,
    ) bool {
        for (0..V) |i| {
            const b = BYTE_OF[i];
            self.prior[i] = if (vocab[b]) inputs[@intCast(byte_map[b])] else 0;
        }

        const t = TOK[byte];
        var stepped: bool = false;
        if (self.in_footer or t < 0) {
            // Either the side footer has begun (magic rule, causal) or this byte
            // has no canonical-205 token at all (alphabet rule). Do not step and
            // do not advance the separator window — pass the PPMd row through,
            // the same degradation their piece-first-token path accepts.
            fx2tf_passthrough(self.h, &self.prior, &self.out);
            if (self.in_footer) self.n_footer += 1 else self.n_offvocab += 1;
        } else {
            stepped = fx2tf_byte_update(self.h, @intCast(t), &self.prior, &self.out) != 0;
            if (stepped) self.n_steps += 1 else self.n_boundaries += 1;
        }

        // The PRL fold applies only where the transformer actually produced a
        // distribution. At a passthrough the slot IS the PPMd row, and folding
        // PPMd by a power of itself is a different object from what the LSTM
        // does — it would sharpen the prior against itself. 243,426 boundaries
        // out of ~586 M tokens (0.04 %), so the choice is immaterial to bytes
        // and this is the conservative one.
        if (comptime prior_form == 1) {
            if (stepped) priorFold(&self.out, &self.prior);
        }

        fx2tf_floor(&self.out);

        @memset(out256, 0);
        for (0..V) |i| out256[BYTE_OF[i]] = self.out[i];

        // ★ THE 51-VALUE SCATTER FIX — a real byte lever, not just tidiness.
        // Without this, every byte value outside the canonical 205 is left at
        // probability ZERO. Those 51 values occur ONLY in the r1 side footer —
        // i.e. exactly where the passthrough above is active — so `ByteModel`'s
        // interval descent hits `denom == 0` and emits p = 0.5, coding such a
        // byte at a flat 8 bits/byte. It does not desync (deterministic on both
        // ops) and no tier without a footer can see it, but over the 679,505 B
        // footer the worst case is ~+679 KB of payload. Priced separately in
        //
        // Applied UNCONDITIONALLY rather than only on the passthrough path: in
        // the main region PPMd assigns these values ~no mass, so it costs
        // nothing there, and doing it always removes the coupling between this
        // fix and "which path am I on" — one less way to be subtly wrong.
        // The f16 round-trip stays on the canonical 205 (their contract); these
        // 51 pass raw, since they do not exist in their stream and no contract
        // covers them. Floored to match their >= 1e-6 guard.
        for (0..256) |b| {
            if (TOK[b] >= 0) continue;
            if (!vocab[b]) continue;
            const p = inputs[@intCast(byte_map[b])];
            out256[b] = if (p >= 1e-6) p else 1e-6;
        }

        // Advance the raw-byte magic window LAST, so the 7 magic bytes
        // themselves are still handled as main-region bytes and only the byte
        // AFTER the complete magic sees `in_footer`.
        if (!self.in_footer) {
            std.mem.copyForwards(u8, self.magic_window[0 .. SIDE_MAGIC.len - 1], self.magic_window[1..]);
            self.magic_window[SIDE_MAGIC.len - 1] = byte;
            if (std.mem.eql(u8, &self.magic_window, &SIDE_MAGIC)) self.in_footer = true;
        }
        return stepped;
    }

    /// ★★ THE MAP GUARD — the only thing standing between a one-rank token-map
    /// error and a multi-day run that produces confidently wrong predictions.
    ///
    /// ⚠⚠ **A LOSSLESS ROUNDTRIP CANNOT CATCH A WRONG TOKEN MAP.** Encoder and
    /// decoder use the SAME map, agree perfectly, and emit a valid archive that
    /// is merely much larger than it should be. Every gate this port has —
    /// bit-identity, roundtrip, the ISA gate, the fp16 quirk gate — is blind to
    /// it. So the map needs an EXTERNAL assertion against a known-true count.
    ///
    /// The invariant: their `kArticleSeparator`, decoded through the map we
    /// actually use, must occur exactly **243,426** times in the e9 post-WRT
    /// main region — the enwik9 article count, independently corroborated by
    /// `r1_reorder.zig:63-72`'s 243,425 exact `DF 99 'N'` blocks.
    ///
    /// ✅ **STATUS: THE MAP QUESTION IS RESOLVED IN FAVOUR
    /// OF THIS TABLE** (the canonical-205 token map; both maps were run
    /// through the shipped weights and only this one interpolates their
    /// published 0.9131822825). The paragraph below is the ORIGINAL status,
    /// kept because its reasoning is what the guard is built on. The guard is
    /// NOT relaxed by the resolution: it is now WIRED (`runner_lex.zig`'s
    /// `codeTempFromFile`, the `-e`/ship-`-c` choke point) and PROVEN TO FIRE
    /// by `-Dtransformer-map-fault`.
    ///
    /// ⚠ **ORIGINAL STATUS: THE MAP WAS NOT SETTLED, AND THIS GUARD IS WHY.**
    /// Two plausible derivations of the token map differ by exactly one rank.
    /// Under one the SHIPPED separator constant occurs 243,426x; under the
    /// other it occurs **0** times and needs one index changed (`0x05 -> 0x06`,
    /// position 6) to reach 243,426. Both cannot be right about the same map,
    /// and only reproducing the upstream **0.9131822825** nats/byte through the
    /// published weights arbitrates it, because only that asks which map the
    /// WEIGHTS were trained under. This guard fires, and it must not be relaxed
    /// to make a run start.
    /// ★★ WHY THIS COUNT, AND NOT A `TOK[0x0a]` CHECK — do not weaken it.
    ///
    /// The measured map tie-break (`rank(0x0a) ∈ {5,6}`) CANNOT catch a
    /// *systematic* shift, because `0x0a` sits BELOW rank 7. Their index space
    /// inserts `0x12` at rank 7, so in principle every payload symbol above it
    /// could sit one rank higher than in a header-free alphabet, and a
    /// `TOK[0x0a]` agreement would not imply `TOK[0x20]`'s.
    ///
    /// This count does catch it. The 15-token separator spans **`0x20`
    /// (rank 8)** and **`0xdf` (rank 172)** — BOTH ABOVE RANK 7. Under a
    /// uniform off-by-one those tokens decode to a different byte sequence,
    /// which would not occur 243,426 times. ⇒ **the separator-count assertion
    /// is strictly stronger than any single-rank check, and it is the
    /// load-bearing gate of the entire port, not a nicety.**
    ///
    /// That constant is the upstream tree's only end-to-end vocabulary gate,
    /// and a port that reimplements the front end loses it unless it carries it
    /// deliberately. We carry it here.
    ///
    /// ⚠ Context worth knowing but NOT relying on: their `vocab_size_ == 205`
    /// is **204 genuine symbols + `0x12`**, a byte that exists only because of
    /// the 5-byte `[TEXT][len32]` header's length encoding
    /// (`.ready4cmix` = 934,220,400 = 0x37AF1270) and occurs **once in 593 M
    /// tokens** — so the shipped model carries trained embedding and unembedding
    /// rows for an artifact of `sprintf`. Our header-free `prep.temp`
    /// independently contains **`0x13` exactly once, at rank 7** — a different
    /// byte at the same rank. **That coincidence is why every symbol above rank
    /// 7 aligns and why their frozen weights read our stream at all. Nothing
    /// enforces it.** This assertion is what would notice if it ever stopped
    /// being true.
    pub const EXPECTED_SEPARATORS: u64 = 243_426;
    /// Below this the stream is not a full e9 main region and the exact count
    /// is not expected to hold.
    pub const E9_MAIN_MIN: u64 = 580_000_000;
    /// e9 article density: 586,459,321 post-WRT bytes / 243,426 separators.
    /// One article per 2,409 bytes. Used for the SUB-e9 rate check below.
    pub const BYTES_PER_ARTICLE: u64 = 2409;
    /// Below this even a rate check is meaningless (a 50 KB window holds ~20
    /// articles and Poisson noise dominates).
    pub const RATE_CHECK_MIN: u64 = 1_000_000;

    /// ⛔⛔ **`n_boundaries` IS NOT A SEPARATOR COUNT — measured on their own
    /// shim, .** `fx2tf_byte_update` returns `stepped == 0` for BOTH
    /// a separator-window match AND the `kMaxArticleTokens = 2^17` piece cut
    /// (`fx2_shim.cpp`), so `n_boundaries` = separators + piece cuts and the
    /// two are indistinguishable from the Zig side. Asserting
    /// `n_boundaries == 243,426` would therefore ABORT A 40-HOUR ENCODE on a
    /// perfectly correct map the moment enwik9 contains one article longer than
    /// 131,072 post-WRT tokens — the classic gate that false-alarms and then
    /// gets switched off. The shim now counts the memcmp match SEPARATELY
    /// (`fx2tf_separator_hits`) and that is what is asserted here.
    pub fn separatorHits(self: *const Transformer) u64 {
        return fx2tf_separator_hits(self.h);
    }

    /// The guard's DECISION, as a pure function of the two numbers it sees.
    ///
    /// Split out of `checkSeparatorCount` so the decision can be gated by unit
    /// tests at every boundary — a full-scale run costs hours and can only ever
    /// exercise one point of this domain, and a gate nobody has watched fire is
    /// a comment rather than a check. Byte-neutral: the same
    /// comparisons, in the same order, inlined at the one call site.
    pub const Verdict = enum { too_small, rate_ok, rate_bad, count_ok, count_bad, map_bad };

    pub fn separatorVerdict(total_bytes: u64, hits: u64, tok_0a: i16, tok_20: i16) Verdict {
        if (tok_0a != 5 or tok_20 != 8) return .map_bad;
        if (total_bytes < E9_MAIN_MIN) {
            if (total_bytes < RATE_CHECK_MIN) return .too_small;
            const expect = total_bytes / BYTES_PER_ARTICLE;
            return if (hits < expect / 8 or hits > expect * 8) .rate_bad else .rate_ok;
        }
        return if (hits != EXPECTED_SEPARATORS) .count_bad else .count_ok;
    }

    /// Call at end-of-op with the number of bytes the model was stepped over.
    /// Hard-fails on a full-scale run whose separator count is wrong.
    pub fn checkSeparatorCount(self: *const Transformer, total_bytes: u64) void {
        const seen = self.n_steps + self.n_boundaries + self.n_offvocab + self.n_footer;
        const hits = self.separatorHits();
        // ★ ONE decision, taken by the function the unit tests exercise. The
        // pre-check (two ranks the separator spans, one below rank 7 and one
        // above) fires first with a clear message; the COUNT remains the
        // authoritative one — see EXPECTED_SEPARATORS on why only the count
        // catches a systematic shift.
        //
        // ⚠ Until this returned early for anything under e9, which
        // made it a NO-OP at exactly the tier the payload pair runs at. An exact
        // count cannot be asserted on a partial stream, but the DENSITY can, and
        // the failure being guarded is not a small drift — it is a wrong map
        // producing essentially ZERO separators. The deliberately wide 8x band
        // catches that while being structurally incapable of crying wolf on
        // content variation (article density does not vary 8x across enwik9).
        // Same principle as recip_gate.sh excluding the crypto opcodes: a gate
        // that false-alarms gets switched off, and then it cannot report the
        // real break.
        switch (separatorVerdict(total_bytes, hits, TOK[0x0a], TOK[0x20])) {
            .too_small, .rate_ok, .count_ok => return,
            .map_bad => std.debug.panic(
                "-Dtransformer MAP GUARD: TOK[0x0a]={d} want 5, TOK[0x20]={d} want 8. Wrong token map; do NOT relax. exp-transformer-port-impl-aug-31.md 3.",
                .{ TOK[0x0a], TOK[0x20] },
            ),
            .rate_bad => std.debug.panic(
                "-Dtransformer MAP GUARD: {d} separators in {d} B, want ~{d} (1/{d} B). Near-zero = wrong token map; do NOT widen.",
                .{ hits, total_bytes, total_bytes / BYTES_PER_ARTICLE, BYTES_PER_ARTICLE },
            ),
            .count_bad => std.debug.panic(
                "-Dtransformer MAP GUARD: separators={d} want {d} over {d} B ({d} steps). Wrong token map; do NOT relax. exp-transformer-port-impl-aug-31.md 3.",
                .{ hits, EXPECTED_SEPARATORS, seen, self.n_steps },
            ),
        }
    }

    /// Engagement witness (§ the "an arm must prove it ENGAGED" rule).
    pub fn witness(self: *const Transformer, w: anytype) void {
        w.print("transformer: steps={d} boundaries={d} separators={d} offvocab={d} footer={d} in_footer={} prior_form={d} map_fault={d}\n", .{
            self.n_steps, self.n_boundaries, self.separatorHits(), self.n_offvocab, self.n_footer, self.in_footer, prior_form, map_fault,
        }) catch {};
    }
};

// ---- gates -----------------------------------------------------------------

test "canonical-205 map: 205 live tokens, 51 dead, and the map is a bijection" {
    var live: usize = 0;
    for (0..256) |b| {
        if (TOK[b] >= 0) live += 1;
    }
    try std.testing.expectEqual(@as(usize, V), live);
    for (0..V) |i| try std.testing.expectEqual(@as(i16, @intCast(i)), TOK[BYTE_OF[i]]);
}

test "canonical-205 map: the three measured anchors" {
    // Measured on our real e9
    // prep.temp. A one-value error below 0xac breaks all three.
    try std.testing.expectEqual(@as(i16, 8), TOK[0x20]);
    try std.testing.expectEqual(@as(i16, 5), TOK[0x0a]);
    try std.testing.expectEqual(@as(i16, 172), TOK[0xdf]);
}

test "canonical-205 map: their kArticleSeparator decodes to \"  <page>\\n    <title>\"" {
    // Their constant is expressed in vocabulary INDICES; decoding it through
    // OUR map must give the WRT-encoded separator. This is the single check
    // that proves the two 205-alphabets are the same set.
    var got: [15]u8 = undefined;
    for (SEPARATOR_IDX, 0..) |ix, k| got[k] = BYTE_OF[ix];
    try std.testing.expectEqualSlices(u8, &SEPARATOR_BYTES, &got);
}

test "map guard: the verdict function, at every boundary of its domain" {
    const T = Transformer;
    const V_ = T.Verdict;
    // (1) the map pre-check dominates everything, at any size.
    try std.testing.expectEqual(V_.map_bad, T.separatorVerdict(1_000_000_000, 243_426, 6, 8));
    try std.testing.expectEqual(V_.map_bad, T.separatorVerdict(1_000_000_000, 243_426, 5, 9));
    // (2) below RATE_CHECK_MIN nothing is asserted — including hits == 0, which
    //     is what makes every 50k/1m gate in the ritual blind to a wrong map.
    try std.testing.expectEqual(V_.too_small, T.separatorVerdict(T.RATE_CHECK_MIN - 1, 0, 5, 8));
    // (3) the rate band, exactly at its two edges. 1,010,420 B is the synthetic
    //     control temp: expect = 419, band [52, 3352].
    try std.testing.expectEqual(V_.rate_ok, T.separatorVerdict(1_010_420, 348, 5, 8));
    try std.testing.expectEqual(V_.rate_ok, T.separatorVerdict(1_010_420, 52, 5, 8));
    try std.testing.expectEqual(V_.rate_bad, T.separatorVerdict(1_010_420, 51, 5, 8));
    try std.testing.expectEqual(V_.rate_ok, T.separatorVerdict(1_010_420, 3352, 5, 8));
    try std.testing.expectEqual(V_.rate_bad, T.separatorVerdict(1_010_420, 3353, 5, 8));
    // (4) THE FAILURE THE GUARD EXISTS FOR: a wrong map produces ~no separator
    //     matches at all, so it lands far outside the band rather than near it.
    try std.testing.expectEqual(V_.rate_bad, T.separatorVerdict(1_010_420, 0, 5, 8));
    // (5) at e9 the exact count governs, and it is exact — off by one FAILS.
    try std.testing.expectEqual(V_.count_ok, T.separatorVerdict(586_459_321, 243_426, 5, 8));
    try std.testing.expectEqual(V_.count_bad, T.separatorVerdict(586_459_321, 243_425, 5, 8));
    try std.testing.expectEqual(V_.count_bad, T.separatorVerdict(586_459_321, 243_427, 5, 8));
    // (6) the e9 branch starts at E9_MAIN_MIN, not one byte earlier.
    try std.testing.expectEqual(V_.count_bad, T.separatorVerdict(T.E9_MAIN_MIN, 0, 5, 8));
    try std.testing.expectEqual(V_.rate_bad, T.separatorVerdict(T.E9_MAIN_MIN - 1, 0, 5, 8));
}

test "canonical-205 map: the negative control is OFF by default and the tables are the base ones" {
    // -Dtransformer-map-fault is a NEGATIVE CONTROL. If a default `zig build`
    // ever carries it, every byte measurement taken on that tree is worthless
    // and the guard it exists to prove is itself disarmed.
    try std.testing.expectEqual(@as(u32, 0), map_fault);
    try std.testing.expectEqual(@as(?[2]usize, null), FAULT_SWAP);
    try std.testing.expectEqualSlices(u8, &BYTE_OF_BASE, &BYTE_OF);
    try std.testing.expectEqualSlices(i16, &TOK_BASE, &TOK);
}

test "side-footer magic: the detector is causal and latches" {
    // The magic is the FIRST 7 bytes of the r1 side blob (r1_reorder.zig:87,
    // written at :513, appended directly after main at :474). Detection must
    // depend only on the decoded prefix, or encode and decode desynchronise.
    var win = [_]u8{0} ** SIDE_MAGIC.len;
    var latched = false;
    const stream = "abc" ++ "R1ORD3\n" ++ "xy";
    var fired_at: ?usize = null;
    for (stream, 0..) |b, i| {
        if (!latched) {
            std.mem.copyForwards(u8, win[0 .. SIDE_MAGIC.len - 1], win[1..]);
            win[SIDE_MAGIC.len - 1] = b;
            if (std.mem.eql(u8, &win, &SIDE_MAGIC)) {
                latched = true;
                fired_at = i;
            }
        }
    }
    // "abc" is 3 bytes, then the 7 magic bytes occupy indices 3..9; the window
    // completes on the LAST magic byte, index 9.
    try std.testing.expectEqual(@as(?usize, 9), fired_at);
    try std.testing.expect(latched);
}

test "side-footer magic: it is exactly the r1 side blob's own constant" {
    // Guards against a silent drift if r1_reorder.zig ever changes kSideMagicD86.
    try std.testing.expectEqualSlices(u8, "R1ORD3\n", &SIDE_MAGIC);
    try std.testing.expectEqual(@as(usize, 7), SIDE_MAGIC.len);
}
