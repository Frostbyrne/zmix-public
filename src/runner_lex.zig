//! `runner_lex` — Zig port of the cmix-lex `-c` / `-d` driver
//! (ref/cmix-lex/src/runner.cpp `RunCompression` 210-274, `RunDecompression`
//! 276-342, `WriteHeader`/`ReadHeader` 59-111, `Compress`/`Decompress` 133-187),
//! plus `preprocessor::Pretrain` (preprocess/preprocessor.cpp:18-50).
//!
//! This wires the already-ported sub-pieces into a byte-compatible cmix-lex codec:
//!   - WRT/dict preprocessing   -> src/prepr/preprocessor_lex.zig (byte-exact vs `-s`)
//!   - PredictorLex + pretrain  -> src/predictor_lex.zig
//!   - arithmetic coder         -> replicated inline (coder.zig is v21-typed)
//!
//! CLI (mirrors the real binary):
//!   runner_lex -c <dict> <in> <out>   compress with dictionary (WRT + Pretrain)
//!   runner_lex -c <in> <out>          compress, no dictionary
//!   runner_lex -n <in> <out>          no-preprocess (DEFAULT-framed), no Pretrain
//!   runner_lex -d <dict> <in> <out>   decompress with dictionary
//!   runner_lex -d <in> <out>          decompress, no dictionary
//!
//! SEED: the real cmix-lex binary is built `-DSEED=923` and calls `srand(923)`
//! (runner.cpp:350) BEFORE constructing the Predictor, which drives the Indirect
//! map_offsets and the LSTM Glorot weights via glibc rand. The oracle goldens
//! were captured WITHOUT srand (== seed 1), so `PredictorLex.init` defaults to 1;
//! this driver defaults to 923 to match the shipped binary. Override with
//! `ZMIX_SEED=<n>`.

const std = @import("std");
const ram_census = @import("ramcensus");
const build_options = @import("build_options");

comptime {
    _ = @import("detm_math.zig");
}

const PredictorLex = @import("predictor_lex.zig").PredictorLex;
const pre = @import("prepr/preprocessor_lex.zig");
const split = @import("prepr/split.zig");
const reorder_mod = @import("prepr/article_reorder.zig");
const phda9 = @import("prepr/phda9.zig");
const r1 = @import("prepr/r1_reorder.zig");
const self_extract = @import("prepr/self_extract.zig");
const hh = @import("hero_head.zig");
const fieldcodec = @import("prepr/fieldcodec.zig");
// -Dfieldcodec-tsswap: ts-before-revid null-swap on the coded temp (see
// src/prepr/fieldcodec.zig). Read from its OWN options module, never from the
// shared model_opts — the own-module rule: an option in model_opts moves the packed prefix
// (charged ×2 under Form-1) even when the knob is off.
const FIELDCODEC_TSSWAP = @import("fieldcodec_options").ts_swap;
const cmtmove = @import("prepr/cmtmove.zig");
// -Dfieldcodec-cmtmove: gated comment relocation (see src/prepr/cmtmove.zig).
// Composed AFTER tsswap on encode, inverted BEFORE it on decode (LIFO); the two
// transforms touch disjoint bytes (tail blocks vs body page records).
const FIELDCODEC_CMTMOVE = @import("fieldcodec_options").cmt_move;

const kMinVocabFileSize: u64 = 10000; // runner.cpp:27
const DEFAULT_SEED: i32 = 923; // makefile -DSEED=923

/// -Dcoder-diag — the decisive precision census: how much of the coded stream is
/// spent on bits the coder is ALREADY maximally confident about? That sum is an
/// exact UPPER BOUND on everything any precision widening anywhere in the stack
/// (fxcm's 12-bit APM interface, the +-2047 stretch clamp, the 16-bit coder) could
/// ever recover. Side-channel f64 accumulators; output bit-identical.
const CODER_DIAG: bool = @import("build_options").coder_diag;

/// -Dwrt-markmap=<prefix> — LAB. Pairs the per-temp-byte frontend CLASS map built
/// by the preprocessor with the per-temp-byte CODED COST measured by this coder, so
/// the archive can be stratified by frontend construct EXACTLY (every byte, no
/// sampling). Writes `<prefix>.cls` (u8/byte) + `<prefix>.cost` (u32/byte, 1/4096
/// bit) and prints a per-class aggregate. Comptime-dead when the prefix is empty.
const MARKMAP_PATH: []const u8 = @import("prepr_options").wrt_markmap;
/// TEST ONLY (-Dtest-no-r1): see build.zig. Default false => zero change.
const TEST_NO_R1: bool = @import("prepr_options").test_no_r1;
const MARKMAP: bool = MARKMAP_PATH.len > 0;

fn markmapDump(temp: []const u8, costmap: []const u32) void {
    if (comptime !MARKMAP) return;
    const cls = pre.clsStream();
    const CLS_N = @import("prepr/dictionary_lex.zig").CLS_N;
    var n: [CLS_N]u64 = .{0} ** CLS_N;
    var bits: [CLS_N]f64 = .{0} ** CLS_N;
    var total_bits: f64 = 0;
    var unclassed: u64 = 0;
    for (costmap, 0..) |u, i| {
        const c: f64 = @as(f64, @floatFromInt(u)) / 4096.0;
        total_bits += c;
        if (i < cls.len and cls[i] < CLS_N) {
            n[cls[i]] += 1;
            bits[cls[i]] += c;
        } else unclassed += 1;
    }
    std.debug.print(
        "MARKMAP\ttemp_B\t{d}\tcls_B\t{d}\tunclassed\t{d}\ttotal_bits\t{d:.1}\ttotal_B\t{d:.1}\n",
        .{ temp.len, cls.len, unclassed, total_bits, total_bits / 8.0 },
    );
    for (0..CLS_N) |k| {
        const mean = if (n[k] == 0) 0.0 else bits[k] / @as(f64, @floatFromInt(n[k]));
        std.debug.print(
            "MARKMAP\tclass\t{d}\tn\t{d}\tbits\t{d:.1}\tB\t{d:.1}\tmean_bits_per_byte\t{d:.5}\tpct_of_archive\t{d:.4}\n",
            .{ k, n[k], bits[k], bits[k] / 8.0, mean, if (total_bits == 0) 0.0 else 100.0 * bits[k] / total_bits },
        );
    }
    const f = std.fs.cwd().createFile(MARKMAP_PATH ++ ".cls", .{}) catch return;
    defer f.close();
    f.writeAll(cls) catch {};
    const g = std.fs.cwd().createFile(MARKMAP_PATH ++ ".cost", .{}) catch return;
    defer g.close();
    g.writeAll(std.mem.sliceAsBytes(costmap)) catch {};
}

var cd_bits: u64 = 0;
var cd_rail: u64 = 0;
var cd_cost_total: f64 = 0;
var cd_cost_rail: f64 = 0;
var cd_cost_hi: f64 = 0;
var cd_hi: u64 = 0;
inline fn coderDiag(p: u32, bit: i32) void {
    if (comptime !CODER_DIAG) return;
    const pa: f64 = if (bit != 0)
        @as(f64, @floatFromInt(p)) / 65536.0
    else
        (65536.0 - @as(f64, @floatFromInt(p))) / 65536.0;
    const c = -@log2(pa);
    cd_bits += 1;
    cd_cost_total += c;
    // "at the rail": the coded probability of the ACTUAL bit is the most extreme
    // the 16-bit coder can express (p in {1,65535} => pa >= 65535/65536).
    if ((bit != 0 and p >= 65535) or (bit == 0 and p <= 1)) {
        cd_rail += 1;
        cd_cost_rail += c;
    }
    // A softer band: pa >= 4095/4096, i.e. beyond what fxcm's 12-bit APM domain
    // can represent at all.
    if (pa >= 4095.0 / 4096.0) {
        cd_hi += 1;
        cd_cost_hi += c;
    }
}
pub fn coderDiagDump() void {
    if (comptime !CODER_DIAG) return;
    std.debug.print(
        "CODERDIAG\tbits\t{d}\ttotal_bits_coded\t{d:.1}\ttotal_B\t{d:.1}\n" ++
        "CODERDIAG\trail_n\t{d}\trail_pct\t{d:.4}\trail_cost_bits\t{d:.1}\trail_cost_B\t{d:.1}\n" ++
        "CODERDIAG\thi12_n\t{d}\thi12_pct\t{d:.4}\thi12_cost_bits\t{d:.1}\thi12_cost_B\t{d:.1}\n",
        .{ cd_bits, cd_cost_total, cd_cost_total / 8.0,
           cd_rail, 100.0 * @as(f64, @floatFromInt(cd_rail)) / @as(f64, @floatFromInt(cd_bits)), cd_cost_rail, cd_cost_rail / 8.0,
           cd_hi, 100.0 * @as(f64, @floatFromInt(cd_hi)) / @as(f64, @floatFromInt(cd_bits)), cd_cost_hi, cd_cost_hi / 8.0 },
    );
}

fn discretize(p: f32) u32 {
    // Encoder::Discretize / Decoder::Discretize = 1 + 65534*p (encoder.cpp:10-12).
    const d: f32 = 1.0 + 65534.0 * p;
    return @intFromFloat(d);
}

// ---------------------------------------------------------------------------
// -Dfree-prior (variant a): scheduled
// deterministic mid-stream LSTM self-retrain. The schedule is a comptime list
// of TEMP byte positions; when the count of fully-coded temp bytes reaches a
// position P, the coder pauses and replays temp[0..P) through the LSTM
// training path only (PredictorLex.freePriorReplayChunk — uniform prior, no
// CM/PPMd/mixer state touched). Positions are TEMP coordinates, identical in
// both ops: the encoder replays from its temp source (slice or staged file),
// the decoder from its own already-decoded output (slice or staged file) —
// byte-identical by construction, so both ops walk identical weight
// trajectories. Default schedule "" ⇒ FP_ON = false ⇒ every hook below is
// comptime-dead (stock bit-identical, the -D pattern). Positions past the end
// of a given temp stream simply never fire (e.g. the e9 schedule inside the
// -e dict/order self-compression sub-codecs, whose temps are ~1 MB).
// ---------------------------------------------------------------------------
const free_prior_schedule: []const u64 = fpParseSchedule(build_options.free_prior);
const FP_ON = free_prior_schedule.len != 0;

/// Comptime parse of the -Dfree-prior spec: comma-separated u64s; zeros
/// dropped (empty prefix = no-op epoch); sorted ascending; deduped.
fn fpParseSchedule(comptime spec: []const u8) []const u64 {
    comptime {
        @setEvalBranchQuota(100_000); // comptime parseInt/sort over the spec
        if (spec.len == 0) return &.{};
        var vals: [spec.len]u64 = undefined;
        var n: usize = 0;
        var it = std.mem.splitScalar(u8, spec, ',');
        while (it.next()) |tok| {
            const t = std.mem.trim(u8, tok, " \t");
            if (t.len == 0) continue;
            const v = std.fmt.parseInt(u64, t, 10) catch
                @compileError("-Dfree-prior: unparseable position '" ++ t ++ "' (want comma-separated decimal temp byte positions)");
            if (v == 0) continue;
            vals[n] = v;
            n += 1;
        }
        var i: usize = 1;
        while (i < n) : (i += 1) { // insertion sort ascending
            const key = vals[i];
            var j = i;
            while (j > 0 and vals[j - 1] > key) : (j -= 1) vals[j] = vals[j - 1];
            vals[j] = key;
        }
        var out: [spec.len]u64 = undefined;
        var m: usize = 0;
        for (vals[0..n]) |v| {
            if (m == 0 or out[m - 1] != v) {
                out[m] = v;
                m += 1;
            }
        }
        const final = out[0..m].*;
        return &final;
    }
}

/// Per-coder-loop schedule cursor. `hit(processed)` returns the scheduled
/// position iff the count of fully-coded temp bytes just reached it.
const FpSched = struct {
    next: usize = 0,
    inline fn hit(self: *FpSched, processed: u64) ?u64 {
        if (self.next < free_prior_schedule.len and processed == free_prior_schedule[self.next]) {
            self.next += 1;
            return free_prior_schedule[self.next - 1];
        }
        return null;
    }
};

/// -Dfree-prior-b record gates (wall/disk): the prior tap only ever REPLAYS
/// rows [0 .. last-fireable-epoch), so recording outside that range is pure
/// waste — at e9 the ungated tail is ~550 MB of temp (top-K pack per byte +
/// ~54 GB scratch), the dominant record-side wall term of the fpb17 e8 kill.
/// Two gates, both pure functions of (comptime schedule, stream length, byte
/// position), which BOTH ops know identically (encoder: measured temp length;
/// decoder: hdr.length) ⇒ deterministic, symmetric, and byte-neutral (gated vs
/// ungated builds must produce identical archives — verified at 1m).
///   gate i  (fpEnableRecording): skip recording entirely when the stream can
///           never reach schedule[0] (the -e dict/order sub-codec temps).
///   gate ii (fpScheduleExhausted → fpPriorEndRecording): stop recording after
///           the last epoch this stream can fire has replayed.
inline fn fpEnableRecording(pred: *PredictorLex, total_len: u64) void {
    if (total_len >= free_prior_schedule[0]) pred.fpPriorEnableRecording();
}

/// True when no further scheduled epoch can fire on a stream of `total_len`
/// (schedule done, or the next position lies past the end of the stream).
inline fn fpScheduleExhausted(fp: *const FpSched, total_len: u64) bool {
    return fp.next == free_prior_schedule.len or free_prior_schedule[fp.next] > total_len;
}

/// Free-prior epoch over an in-memory temp prefix (codeTemp / decodeToTemp).
fn fpEpochMem(pred: *PredictorLex, fp: *const FpSched, total_len: u64, prefix: []const u8) void {
    var t = std.time.Timer.start() catch unreachable;
    pred.fpPriorBeginReplay(); // -Dfree-prior-b: rewind the recorded-prior cursor
    pred.freePriorReplayChunk(prefix);
    fpLogEpoch(pred, prefix.len, t.read());
    if (fpScheduleExhausted(fp, total_len)) pred.fpPriorEndRecording(); // gate ii
}

/// Free-prior epoch replaying a FILE's [0..pos) bytes in 1 MB pread chunks —
/// the staged-coder paths (-e temp file; -d/-D decoded-output file, caller
/// flushes its write buffer first). pread leaves the caller's fd offset
/// untouched; the 1 MB chunk keeps replay RSS flat (impl-preflight gate-4).
fn fpEpochFile(pred: *PredictorLex, fp: *const FpSched, total_len: u64, f: std.fs.File, pos: u64) !void {
    var t = std.time.Timer.start() catch unreachable;
    pred.fpPriorBeginReplay(); // -Dfree-prior-b: rewind the recorded-prior cursor
    var rbuf: [1 << 20]u8 = undefined;
    var off: u64 = 0;
    while (off < pos) {
        const want: usize = @intCast(@min(pos - off, rbuf.len));
        const n = try f.preadAll(rbuf[0..want], off);
        if (n == 0) return error.UnexpectedEndOfFile;
        pred.freePriorReplayChunk(rbuf[0..n]);
        off += n;
    }
    fpLogEpoch(pred, pos, t.read());
    if (fpScheduleExhausted(fp, total_len)) pred.fpPriorEndRecording(); // gate ii
}

/// One stderr line per epoch: position, replayed bytes/s (gate-1 wall input),
/// and — under -Dfree-prior-hash — the full-LSTM-state sha256 that must match
/// between the -c and -d ops of the same archive (gate-2 proof obligation).
fn fpLogEpoch(pred: *PredictorLex, pos: u64, ns: u64) void {
    const ms = ns / std.time.ns_per_ms;
    if (comptime build_options.free_prior_hash) {
        const hex = std.fmt.bytesToHex(pred.freePriorLstmStateHash(), .lower);
        std.debug.print("free-prior epoch: pos={d} replay_ms={d} lstm_sha256={s}\n", .{ pos, ms, &hex });
    } else {
        std.debug.print("free-prior epoch: pos={d} replay_ms={d}\n", .{ pos, ms });
    }
}

test "free-prior schedule parser" {
    try std.testing.expectEqualSlices(u64, &.{}, comptime fpParseSchedule(""));
    try std.testing.expectEqualSlices(u64, &[_]u64{ 20000, 300000 }, comptime fpParseSchedule("300000,20000"));
    try std.testing.expectEqualSlices(u64, &[_]u64{12500000}, comptime fpParseSchedule("0,12500000,12500000"));
    try std.testing.expectEqualSlices(u64, &[_]u64{ 12500000, 36960000 }, comptime fpParseSchedule("12500000, 36960000"));
}

// ---------------------------------------------------------------------------
// preprocessor::Pretrain (preprocess/preprocessor.cpp:18-50): warm the predictor
// over the RAW dictionary bytes. 5-byte DEFAULT header {0, len>>24..len} then each
// dict byte with '\n'(0x0A) remapped to ' '(0x20), all MSB-first (j = 7..0).
// ---------------------------------------------------------------------------
fn preprocessorPretrain(pred: *PredictorLex, dict_bytes: []const u8) void {
    const len: u32 = @intCast(dict_bytes.len);
    const header = [5]u8{
        pre.DEFAULT,
        @intCast((len >> 24) & 0xFF),
        @intCast((len >> 16) & 0xFF),
        @intCast((len >> 8) & 0xFF),
        @intCast(len & 0xFF),
    };
    for (header) |h| {
        var j: i32 = 7;
        while (j >= 0) : (j -= 1) pred.pretrain((@as(i32, h) >> @intCast(j)) & 1);
    }
    for (dict_bytes) |raw| {
        const c: u8 = if (raw == '\n') ' ' else raw;
        var j: i32 = 7;
        while (j >= 0) : (j -= 1) pred.pretrain((@as(i32, c) >> @intCast(j)) & 1);
    }
}

// WriteHeader (runner.cpp:59-77): 5-byte big-endian temp length (39-bit); high bit
// of the top byte = dictionary_used. Then 32 vocab bytes iff length >= 10000.
fn writeHeader(gpa: std.mem.Allocator, out: *std.ArrayList(u8), length: u64, vocab: [256]bool, dict_used: bool) !void {
    var i: i32 = 4;
    while (i >= 0) : (i -= 1) {
        var c: u8 = @intCast((length >> @intCast(8 * i)) & 0xFF);
        if (i == 4) {
            c &= 0x7F;
            if (dict_used) c |= 0x80;
        }
        try out.append(gpa, c);
    }
    if (length < kMinVocabFileSize) return;
    for (0..32) |bi| {
        var c: u8 = 0;
        for (0..8) |j| {
            if (vocab[bi * 8 + j]) c |= @as(u8, 1) << @intCast(j);
        }
        try out.append(gpa, c);
    }
}

const Header = struct { length: u64, dict_used: bool, vocab: [256]bool, body_off: usize };

// ReadHeader (runner.cpp:87-111).
fn readHeader(data: []const u8) !Header {
    if (data.len < 5) return error.Truncated;
    var length: u64 = 0;
    var dict_used = false;
    for (0..5) |i| {
        length <<= 8;
        var c: u8 = data[i];
        if (i == 0) {
            dict_used = (c & 0x80) != 0;
            c &= 0x7F;
        }
        length += c;
    }
    var vocab: [256]bool = undefined;
    var off: usize = 5;
    if (length == 0) {
        // stored / preprocessed-only (WriteStorageHeader) — not produced by -c.
        return .{ .length = 0, .dict_used = dict_used, .vocab = .{true} ** 256, .body_off = 5 };
    }
    if (length < kMinVocabFileSize) {
        vocab = .{true} ** 256;
    } else {
        vocab = .{false} ** 256;
        if (data.len < 37) return error.Truncated;
        for (0..32) |bi| {
            const c = data[5 + bi];
            for (0..8) |j| {
                if (c & (@as(u8, 1) << @intCast(j)) != 0) vocab[bi * 8 + j] = true;
            }
        }
        off = 37;
    }
    return .{ .length = length, .dict_used = dict_used, .vocab = vocab, .body_off = off };
}

fn extractVocab(temp: []const u8) [256]bool {
    // ExtractVocab (runner.cpp:113-126) + the <10000 all-true rule (runner.cpp:259-260).
    if (temp.len < kMinVocabFileSize) return .{true} ** 256;
    var vocab: [256]bool = .{false} ** 256;
    for (temp) |c| vocab[c] = true;
    return vocab;
}

// ---------------------------------------------------------------------------
// codeTemp: the shared coder tail of RunCompression (runner.cpp:254-273): build
// vocab over the temp stream, write the storage header, construct the predictor,
// Pretrain (only when preprocessing with a dict), then arith-code the temp stream
// MSB-first. Returns header ++ coded. Both `-c` and `-e` funnel through here.
// ---------------------------------------------------------------------------
fn codeTemp(
    gpa: std.mem.Allocator,
    temp: []const u8,
    dict_bytes: ?[]const u8,
    enable_preprocess: bool,
    seed: i32,
) ![]u8 {
    const dict_used = dict_bytes != null;
    const vocab = extractVocab(temp);

    var out: std.ArrayList(u8) = .empty;
    errdefer out.deinit(gpa);

    // WriteHeader over temp length + temp vocab.
    try writeHeader(gpa, &out, temp.len, vocab, dict_used);

    // Predictor(vocab); fxcm loads the dict iff one was supplied (fxcmv1.cpp dosym
    // reads the command-line dictionary_path — runner.cpp:290 sets it for any dict
    // argument, so the fxcm dict is on for `-c <dict>`/`-e`). Pretrain over the raw
    // dict only when preprocessing w/ dict.
    const pred = if (dict_used)
        try PredictorLex.initSeedWithDict(gpa, vocab, seed, dict_bytes.?)
    else
        try PredictorLex.initSeed(gpa, vocab, seed);
    // Free the ~GBs of model state as soon as the coder is done (C++ scopes the
    // Predictor + FreeFxcmMemory + malloc_trim, runner.cpp:308-321) — on the -e
    // path two more predictors (asset self-compression) follow this one.
    defer pred.deinit();
    if (enable_preprocess and dict_used) preprocessorPretrain(pred, dict_bytes.?);

    var hero = if (comptime hh.ON)
        try hh.HeroHead.create(gpa, pred.cacheProbeHidden().len)
    else {};
    defer if (comptime hh.ON) hero.destroy();

    // Compress: arith-code the TEMP stream MSB-first (encoder.cpp:14-30, Flush 70-80).
    var fp: FpSched = .{};
    if (comptime FP_ON) fpEnableRecording(pred, temp.len); // -Dfree-prior-b: record coding-time prior (gate i)
    // LAB -Dwrt-markmap: exact per-temp-byte coded cost, accumulated in f64 and
    // stored as u32 in 1/4096-bit units (max 8 x 16 bits = 524,288 units, no
    // overflow). f64 throughout: an f32 accumulator over ~1.9e6 near-unity
    // probabilities lands at eps = 0.125 and has flipped a sign with no symptom.
    var costmap: []u32 = if (comptime MARKMAP) try gpa.alloc(u32, temp.len) else &.{};
    defer if (comptime MARKMAP) gpa.free(costmap);
    var x1: u32 = 0;
    var x2: u32 = 0xffffffff;
    for (temp, 0..) |byte, ti| {
        if (comptime hh.ON) hero.beginByte(pred.cacheProbeHidden());
        var byte_cost: f64 = 0;
        var j: i32 = 7;
        while (j >= 0) : (j -= 1) {
            const bit: i32 = (@as(i32, byte) >> @intCast(j)) & 1;
            const p: u32 = if (comptime hh.ON) blk: {
                const base = discretize(pred.predict());
                // -Dadaptive-depth: skipped bits code p_head; HERO bypassed +
                // frozen (preflight §1 step 7). Comptime-folds away when off.
                if (pred.adepthSkipped()) break :blk base;
                break :blk hero.mix(base, @intCast(7 - j), @as(u32, byte) >> @intCast(j + 1));
            } else discretize(pred.predict());
            if (comptime CODER_DIAG) coderDiag(p, bit);
            if (comptime MARKMAP) {
                // Cost of the ACTUAL bit under the coder's own 16-bit discretised p.
                const pa: f64 = if (bit != 0)
                    @as(f64, @floatFromInt(p)) / 65536.0
                else
                    (65536.0 - @as(f64, @floatFromInt(p))) / 65536.0;
                byte_cost += -@log2(pa);
            }
            const span = x2 - x1;
            const xmid = x1 +% (span >> 16) *% p +% (((span & 0xffff) *% p) >> 16);
            if (bit != 0) x2 = xmid else x1 = xmid + 1;
            if (comptime hh.ON) {
                // -Dadaptive-depth: HERO frozen on skipped bits (its mix never
                // staged this bit). Folds to unconditional observe when off.
                if (!pred.adepthSkipped()) hero.observe(bit);
            }
            pred.perceive(bit);
            while (((x1 ^ x2) & 0xff000000) == 0) {
                try out.append(gpa, @intCast(x2 >> 24));
                x1 <<= 8;
                x2 = (x2 << 8) + 255;
            }
        }
        if (comptime MARKMAP) costmap[ti] = @intFromFloat(@round(byte_cost * 4096.0));
        if (comptime FP_ON) {
            // -Dfree-prior: encoder replay source = the in-memory temp slice.
            if (fp.hit(ti + 1)) |pos| fpEpochMem(pred, &fp, temp.len, temp[0..@intCast(pos)]);
        }
    }
    if (comptime MARKMAP) markmapDump(temp, costmap);
    // Flush.
    while (((x1 ^ x2) & 0xff000000) == 0) {
        try out.append(gpa, @intCast(x2 >> 24));
        x1 <<= 8;
        x2 = (x2 << 8) + 255;
    }
    try out.append(gpa, @intCast(x2 >> 24));

    if (comptime CODER_DIAG) coderDiagDump();
    @import("predictor_lex.zig").mixSparseDump(590);
    return out.toOwnedSlice(gpa);
}

// ---------------------------------------------------------------------------
// Disk-staged coder variants for the enwik9-scale -e/-D paths. The C++ stages
// the temp stream through a disk file (runner.cpp temp_path + the FX dot-files)
// so the ~9.5 GB predictor never coexists with the multi-GB transform buffers —
// the 10 GB Hutter RSS gate breaks otherwise. Byte-identical to codeTemp /
// decodeToTemp: same header, same coder loop, only the byte source/sink is a
// file. Equivalence is unit-gated below ("staged coder variants match").
// ---------------------------------------------------------------------------

/// codeTemp over a temp FILE: pass 1 streams length+vocab, pass 2 streams the
/// coder. Returns header ++ coded payload (RAM; ~108 MB at enwik9 scale).
fn codeTempFromFile(
    gpa: std.mem.Allocator,
    temp_path: []const u8,
    dict_bytes: ?[]const u8,
    enable_preprocess: bool,
    seed: i32,
) ![]u8 {
    const f = try std.fs.cwd().openFile(temp_path, .{});
    defer f.close();

    // Pass 1 — ExtractVocab (runner.cpp:113-126) + length, streaming.
    var vocab: [256]bool = .{false} ** 256;
    var len: u64 = 0;
    var buf: [1 << 20]u8 = undefined;
    while (true) {
        const n = try f.read(&buf);
        if (n == 0) break;
        for (buf[0..n]) |c| vocab[c] = true;
        len += n;
    }
    if (len < kMinVocabFileSize) vocab = .{true} ** 256;

    const dict_used = dict_bytes != null;
    var out: std.ArrayList(u8) = .empty;
    errdefer out.deinit(gpa);
    try writeHeader(gpa, &out, len, vocab, dict_used);

    const pred = if (dict_used)
        try PredictorLex.initSeedWithDict(gpa, vocab, seed, dict_bytes.?)
    else
        try PredictorLex.initSeed(gpa, vocab, seed);
    defer pred.deinit();
    if (enable_preprocess and dict_used) preprocessorPretrain(pred, dict_bytes.?);

    var hero = if (comptime hh.ON)
        try hh.HeroHead.create(gpa, pred.cacheProbeHidden().len)
    else {};
    defer if (comptime hh.ON) hero.destroy();

    // Pass 2 — the exact codeTemp coder loop, fed in 1 MB chunks.
    try f.seekTo(0);
    var fp: FpSched = .{};
    if (comptime FP_ON) fpEnableRecording(pred, len); // -Dfree-prior-b: record coding-time prior (gate i)
    var fp_processed: u64 = 0;
    var x1: u32 = 0;
    var x2: u32 = 0xffffffff;
    while (true) {
        const n = try f.read(&buf);
        if (n == 0) break;
        for (buf[0..n]) |byte| {
            if (comptime hh.ON) hero.beginByte(pred.cacheProbeHidden());
            var j: i32 = 7;
            while (j >= 0) : (j -= 1) {
                const bit: i32 = (@as(i32, byte) >> @intCast(j)) & 1;
                const p: u32 = if (comptime hh.ON) blk: {
                    const base = discretize(pred.predict());
                    // -Dadaptive-depth: skipped bits code p_head; HERO bypassed
                    // + frozen (preflight §1 step 7). Comptime-folds away when off.
                    if (pred.adepthSkipped()) break :blk base;
                    break :blk hero.mix(base, @intCast(7 - j), @as(u32, byte) >> @intCast(j + 1));
                } else discretize(pred.predict());
                const span = x2 - x1;
                const xmid = x1 +% (span >> 16) *% p +% (((span & 0xffff) *% p) >> 16);
                if (bit != 0) x2 = xmid else x1 = xmid + 1;
                if (comptime hh.ON) {
                    // -Dadaptive-depth: HERO frozen on skipped bits (its mix
                    // never staged this bit). Folds to unconditional when off.
                    if (!pred.adepthSkipped()) hero.observe(bit);
                }
                pred.perceive(bit);
                while (((x1 ^ x2) & 0xff000000) == 0) {
                    try out.append(gpa, @intCast(x2 >> 24));
                    x1 <<= 8;
                    x2 = (x2 << 8) + 255;
                }
            }
            if (comptime FP_ON) {
                // -Dfree-prior: encoder replay source = the staged temp file
                // (pread; leaves this loop's streaming fd offset untouched).
                fp_processed += 1;
                if (fp.hit(fp_processed)) |pos| try fpEpochFile(pred, &fp, len, f, pos);
            }
        }
    }
    while (((x1 ^ x2) & 0xff000000) == 0) {
        try out.append(gpa, @intCast(x2 >> 24));
        x1 <<= 8;
        x2 = (x2 << 8) + 255;
    }
    try out.append(gpa, @intCast(x2 >> 24));

    // ★ THE CANONICAL-205 MAP GUARD, at the one choke point every encode passes
    // through: `-e`, `zmix_ship -c` (both call runCompressionE -> here) and
    // `--codetemp`. Deliberately NOT in `runCompression`, which codes the dict
    // and order ASSETS — those are not post-WRT streams and an article-density
    // check on them would be a false alarm, and a gate that false-alarms gets
    // switched off. `len` is the post-WRT+r1 temp length. Comptime-dead without
    // -Dtransformer; a no-op below 1 MB.
    pred.checkTransformerSeparators(len);

    return out.toOwnedSlice(gpa);
}

/// decodeToTemp streaming the temp stream OUT to a file (the C++ Decompress
/// writes temp_path while decoding — the ~0.6 GB temp never joins the ~9.5 GB
/// predictor in RSS). Returns the decoded length.
fn decodeTempToFile(
    gpa: std.mem.Allocator,
    archive: []const u8,
    dict_bytes: ?[]const u8,
    seed: i32,
    temp_path: []const u8,
) !u64 {
    const hdr = try readHeader(archive);
    if (hdr.dict_used != (dict_bytes != null)) return error.DictFlagMismatch;
    if (hdr.length == 0) return error.StoredHeaderUnsupported;

    const pred = if (hdr.dict_used)
        try PredictorLex.initSeedWithDict(gpa, hdr.vocab, seed, dict_bytes.?)
    else
        try PredictorLex.initSeed(gpa, hdr.vocab, seed);
    defer pred.deinit();
    if (hdr.dict_used) preprocessorPretrain(pred, dict_bytes.?);

    var hero = if (comptime hh.ON)
        try hh.HeroHead.create(gpa, pred.cacheProbeHidden().len)
    else {};
    defer if (comptime hh.ON) hero.destroy();

    // .read: the -Dfree-prior replay preads this file back (decoder replay
    // source = its own decoded output); comptime-false in stock builds.
    const outf = try std.fs.cwd().createFile(temp_path, .{ .read = FP_ON });
    defer outf.close();
    var wbuf: [1 << 20]u8 = undefined;
    var wn: usize = 0;

    const body = archive[hdr.body_off..];
    var pos: usize = 0;
    var fp: FpSched = .{};
    if (comptime FP_ON) fpEnableRecording(pred, hdr.length); // -Dfree-prior-b: record coding-time prior (gate i)
    var x1: u32 = 0;
    var x2: u32 = 0xffffffff;
    var x: u32 = 0;
    for (0..4) |_| {
        const b: u32 = if (pos < body.len) body[pos] else 0;
        pos += 1;
        x = (x << 8) + b;
    }

    var produced: u64 = 0;
    while (produced < hdr.length) : (produced += 1) {
        if (comptime hh.ON) hero.beginByte(pred.cacheProbeHidden());
        var byte: i32 = 1;
        while (byte < 256) {
            const p: u32 = if (comptime hh.ON) blk: {
                const base = discretize(pred.predict());
                // -Dadaptive-depth: on skipped bits the coder receives p_head
                // directly; HERO is bypassed + frozen (preflight §1 step 7).
                // Comptime-folds away when the knob is off.
                if (pred.adepthSkipped()) break :blk base;
                const plane: usize = @intCast(31 - @clz(@as(u32, @intCast(byte))));
                const prefix: u32 = @as(u32, @intCast(byte)) - (@as(u32, 1) << @intCast(plane));
                break :blk hero.mix(base, plane, prefix);
            } else discretize(pred.predict());
            const span = x2 - x1;
            const xmid = x1 +% (span >> 16) *% p +% (((span & 0xffff) *% p) >> 16);
            var bit: i32 = 0;
            if (x <= xmid) {
                bit = 1;
                x2 = xmid;
            } else {
                x1 = xmid + 1;
            }
            if (comptime hh.ON) {
                // -Dadaptive-depth: HERO frozen on skipped bits (its mix never
                // staged this bit). Folds to unconditional observe when off.
                if (!pred.adepthSkipped()) hero.observe(bit);
            }
            pred.perceive(bit);
            while (((x1 ^ x2) & 0xff000000) == 0) {
                x1 <<= 8;
                x2 = (x2 << 8) + 255;
                const b: u32 = if (pos < body.len) body[pos] else 0;
                pos += 1;
                x = (x << 8) + b;
            }
            byte += byte + bit;
        }
        wbuf[wn] = @intCast(byte & 0xff);
        wn += 1;
        if (wn == wbuf.len) {
            try outf.writeAll(wbuf[0..wn]);
            wn = 0;
        }
        if (comptime FP_ON) {
            // -Dfree-prior: decoder replay source = its own decoded output,
            // staged in temp_path. Flush the pending tail first so the file
            // holds all `fp_pos` bytes, then pread it back.
            if (fp.hit(produced + 1)) |fp_pos| {
                try outf.writeAll(wbuf[0..wn]);
                wn = 0;
                try fpEpochFile(pred, &fp, hdr.length, outf, fp_pos);
            }
        }
    }
    try outf.writeAll(wbuf[0..wn]);
    return hdr.length;
}

/// preprocessToTemp: the WRT/dict Encode (or `-n` DEFAULT framing) that produces
/// the temp stream fed to `codeTemp` (runner.cpp:224-231). Caller owns the result.
fn preprocessToTemp(
    gpa: std.mem.Allocator,
    input: []const u8,
    dict_bytes: ?[]const u8,
    enable_preprocess: bool,
) ![]u8 {
    if (!enable_preprocess) return pre.noPreprocess(gpa, input);
    if (dict_bytes) |db| {
        var dict = try pre.Dictionary.load(gpa, db);
        defer dict.deinit();
        return pre.encode(gpa, input, &dict);
    }
    return pre.encode(gpa, input, null);
}

// ---------------------------------------------------------------------------
// RunCompression (runner.cpp:210-274). Returns the full archive (header ++ coded).
// ---------------------------------------------------------------------------
pub fn runCompression(
    gpa: std.mem.Allocator,
    input: []const u8,
    dict_bytes: ?[]const u8,
    enable_preprocess: bool,
    seed: i32,
) ![]u8 {
    const temp = try preprocessToTemp(gpa, input, dict_bytes, enable_preprocess);
    defer gpa.free(temp);
    if (comptime FIELDCODEC_TSSWAP) _ = fieldcodec.encodeInPlace(temp);
    if (comptime FIELDCODEC_CMTMOVE) _ = cmtmove.encodeInPlace(temp);
    // LAB gate-dump: also persist the exact stream the predictor codes (post-WRT,
    // post-r1 framing, post-fieldcodec swap) — the wide dump strides by BYTE, so
    // token/novelty features are NOT reconstructible from the y column alone.
    // Comptime-dead at default; never on a recipe line.
    if (comptime @import("predictor_lex.zig").GATE_DUMP_ON) {
        var nb: [512]u8 = undefined;
        const tp = std.fmt.bufPrint(&nb, "{s}.temp", .{@import("predictor_lex.zig").GATE_PATH_EFF}) catch unreachable;
        std.fs.cwd().writeFile(.{ .sub_path = tp, .data = temp }) catch |e| {
            std.debug.print("gate-dump: temp side-file write failed {s}\n", .{@errorName(e)});
        };
    }
    return codeTemp(gpa, temp, dict_bytes, enable_preprocess, seed);
}

// ---------------------------------------------------------------------------
// decodeToTemp: read the storage header, rebuild the predictor + Pretrain, and
// arith-decode back to the temp stream (runner.cpp:284-320). Returns the temp
// bytes; caller inverts the WRT/dict framing (and, for `-e`, r1_reorder first).
// ---------------------------------------------------------------------------
fn decodeToTemp(
    gpa: std.mem.Allocator,
    archive: []const u8,
    dict_bytes: ?[]const u8,
    seed: i32,
) ![]u8 {
    const hdr = try readHeader(archive);
    if (hdr.dict_used != (dict_bytes != null)) return error.DictFlagMismatch;
    if (hdr.length == 0) return error.StoredHeaderUnsupported;

    // fxcm loads the dict iff the header says one was used — MUST mirror the encode
    // side (codeTemp) exactly, using the same dict bytes, or the coder desyncs.
    const pred = if (hdr.dict_used)
        try PredictorLex.initSeedWithDict(gpa, hdr.vocab, seed, dict_bytes.?)
    else
        try PredictorLex.initSeed(gpa, hdr.vocab, seed);
    // Freed when the temp stream is decoded: the -D un-transform chain (r1 +
    // WRT decode + reassembly, ~4 GB of buffers) runs AFTER this returns and
    // must not coexist with the ~9.5 GB predictor (10 GB Hutter RSS gate).
    defer pred.deinit();
    if (hdr.dict_used) preprocessorPretrain(pred, dict_bytes.?);

    var hero = if (comptime hh.ON)
        try hh.HeroHead.create(gpa, pred.cacheProbeHidden().len)
    else {};
    defer if (comptime hh.ON) hero.destroy();

    // Decompress the temp stream (Decompress runner.cpp:170-187; Decoder decoder.cpp).
    const body = archive[hdr.body_off..];
    var pos: usize = 0;
    var x1: u32 = 0;
    var x2: u32 = 0xffffffff;
    var x: u32 = 0;
    for (0..4) |_| {
        const b: u32 = if (pos < body.len) body[pos] else 0;
        pos += 1;
        x = (x << 8) + b;
    }

    var temp: std.ArrayList(u8) = .empty;
    errdefer temp.deinit(gpa);
    try temp.ensureTotalCapacity(gpa, @intCast(hdr.length));

    var fp: FpSched = .{};
    if (comptime FP_ON) fpEnableRecording(pred, hdr.length); // -Dfree-prior-b: record coding-time prior (gate i)
    var produced: u64 = 0;
    while (produced < hdr.length) : (produced += 1) {
        if (comptime hh.ON) hero.beginByte(pred.cacheProbeHidden());
        var byte: i32 = 1;
        while (byte < 256) {
            const p: u32 = if (comptime hh.ON) blk: {
                const base = discretize(pred.predict());
                // -Dadaptive-depth: on skipped bits the coder receives p_head
                // directly; HERO is bypassed + frozen (preflight §1 step 7).
                // Comptime-folds away when the knob is off.
                if (pred.adepthSkipped()) break :blk base;
                const plane: usize = @intCast(31 - @clz(@as(u32, @intCast(byte))));
                const prefix: u32 = @as(u32, @intCast(byte)) - (@as(u32, 1) << @intCast(plane));
                break :blk hero.mix(base, plane, prefix);
            } else discretize(pred.predict());
            const span = x2 - x1;
            const xmid = x1 +% (span >> 16) *% p +% (((span & 0xffff) *% p) >> 16);
            var bit: i32 = 0;
            if (x <= xmid) {
                bit = 1;
                x2 = xmid;
            } else {
                x1 = xmid + 1;
            }
            if (comptime hh.ON) {
                // -Dadaptive-depth: HERO frozen on skipped bits (its mix never
                // staged this bit). Folds to unconditional observe when off.
                if (!pred.adepthSkipped()) hero.observe(bit);
            }
            pred.perceive(bit);
            while (((x1 ^ x2) & 0xff000000) == 0) {
                x1 <<= 8;
                x2 = (x2 << 8) + 255;
                const b: u32 = if (pos < body.len) body[pos] else 0;
                pos += 1;
                x = (x << 8) + b;
            }
            byte += byte + bit;
        }
        temp.appendAssumeCapacity(@intCast(byte & 0xff));
        if (comptime FP_ON) {
            // -Dfree-prior: decoder replay source = its own decoded output.
            if (fp.hit(produced + 1)) |fp_pos| fpEpochMem(pred, &fp, hdr.length, temp.items[0..@intCast(fp_pos)]);
        }
    }
    return temp.toOwnedSlice(gpa);
}

// ---------------------------------------------------------------------------
// RunDecompression (runner.cpp:276-342). Returns the reconstructed original bytes.
// ---------------------------------------------------------------------------
pub fn runDecompression(
    gpa: std.mem.Allocator,
    archive: []const u8,
    dict_bytes: ?[]const u8,
    seed: i32,
) ![]u8 {
    const temp = try decodeToTemp(gpa, archive, dict_bytes, seed);
    defer gpa.free(temp);
    if (comptime FIELDCODEC_CMTMOVE) _ = cmtmove.decodeInPlace(temp);
    if (comptime FIELDCODEC_TSSWAP) _ = fieldcodec.decodeInPlace(temp);

    // Invert the WRT/dict framing back to the original bytes.
    if (dict_bytes) |db| {
        var dict = try pre.Dictionary.load(gpa, db);
        defer dict.deinit();
        return pre.decode(gpa, temp, &dict);
    }
    return pre.decode(gpa, temp, null);
}

// ============================================================================
// Full enwik9 `-e` pipeline (runner.cpp:422-472) and `-D` inverse (377-408).
//
//   COMPRESS (-e): read enwik9
//     -> split4Comp -> {intro, main, coda}
//     -> reorder(main, new_article_order)            = .main_reordered
//     -> phda9.encodeTxtWit(main_reordered)          = .main_phda9prepr
//     -> cat(main_phda9prepr, intro, coda)           = .ready4cmix
//     -> WRT/dict Encode                             = post-WRT temp
//     -> r1_reorder::ReorderEncodedTailFile          = temp (+ side footer)
//     -> codeTemp (storage header + Pretrain + arith)= payload
//     -> archive9 = decomp_bin ++ dict.comp ++ payload ++ HeaderInfo(12)
//
//   DECOMPRESS (-D): payload
//     -> arith-decode (decodeToTemp)                 = temp (+ side footer)
//     -> r1_reorder restore                          = post-WRT stream
//     -> WRT/dict Decode                             = .input_decomp (== ready4cmix)
//     -> split4Decomp -> {main, intro, coda}
//     -> phda9.decodeTxtWit(main)                    = .main_decomp_restored
//     -> sortMain                                    = .main_decomp_restored_sorted
//     -> cat(intro, main_sorted, coda)               = enwik9
// ============================================================================

pub const NUM_OF_ARTICLES: i32 = reorder_mod.NUM_OF_ARTICLES;

/// buildReady4Cmix (runner.cpp:433-443). The cat order is EXACTLY
/// [main_phda9prepr][intro][coda]. `enwik9`/`order_bytes` are borrowed.
pub fn buildReady4Cmix(gpa: std.mem.Allocator, enwik9: []const u8, order_bytes: []const u8) ![]u8 {
    const sp = split.split4Comp(enwik9);
    const main_reordered = try reorder_mod.reorder(gpa, sp.main, order_bytes, NUM_OF_ARTICLES);
    defer gpa.free(main_reordered);
    const main_phda9 = try phda9.encodeTxtWit(gpa, main_reordered);
    defer gpa.free(main_phda9);

    const total = main_phda9.len + sp.intro.len + sp.coda.len;
    var out = try gpa.alloc(u8, total);
    var i: usize = 0;
    @memcpy(out[i..][0..main_phda9.len], main_phda9);
    i += main_phda9.len;
    @memcpy(out[i..][0..sp.intro.len], sp.intro);
    i += sp.intro.len;
    @memcpy(out[i..][0..sp.coda.len], sp.coda);
    return out;
}

/// The `-e` preprocessing up to (but not including) the arith coder. `temp` is the
/// exact stream `FX_PREPARE_ONLY` dumps and that the coder codes.
pub const PrepareE = struct {
    ready4cmix: []u8,
    temp: []u8, // post-WRT + post-r1 (== out.cmix.temp)
    side: ?[]u8, // == .r1_payload_lex_side (null iff r1_reorder did not apply)
    r1_applied: bool,

    pub fn deinit(self: *PrepareE, gpa: std.mem.Allocator) void {
        gpa.free(self.ready4cmix);
        gpa.free(self.temp);
        if (self.side) |s| gpa.free(s);
    }
};

pub fn prepareE(
    gpa: std.mem.Allocator,
    enwik9: []const u8,
    dict_bytes: []const u8,
    order_bytes: []const u8,
) !PrepareE {
    if (comptime TEST_NO_R1) {
        // ⛔ TEST ONLY (-Dtest-no-r1). Bypass EVERY enwik9-only stage — split4Comp
        // (fixed e9 line numbers), the article reorder (243,425-line e9 order
        // asset; every missing key maps to article 0, so a 1 MB input became a
        // 56.8 MB stream with article 1 repeated 172,233x), phda9 and r1 — on BOTH
        // sides, so the Form-1 CONTAINER (archive9 assembly, trailer/blob slicing,
        // embedded weights, no-arg self-extraction) can be exercised below e9.
        // The first version of this knob skipped only r1; the decode then died
        // Phda9Fail in split4Decomp/phda9 with the coder entirely innocent
        // ready4cmix == input verbatim;
        // untransformTempFile/roundtripE mirror this. Not the shipping codec.
        std.debug.print(
            "TEST-NO-R1: bypassing split4/reorder/phda9/r1 on a {d}-byte input (THIS IS NOT A SHIPPABLE ARCHIVE)\n",
            .{enwik9.len},
        );
        const r4c = try gpa.dupe(u8, enwik9);
        errdefer gpa.free(r4c);
        var dict_t = try pre.Dictionary.load(gpa, dict_bytes);
        defer dict_t.deinit();
        const wrt_t = try pre.encode(gpa, r4c, &dict_t);
        if (comptime FIELDCODEC_TSSWAP) _ = fieldcodec.encodeInPlace(wrt_t);
        if (comptime FIELDCODEC_CMTMOVE) _ = cmtmove.encodeInPlace(wrt_t);
        return .{ .ready4cmix = r4c, .temp = wrt_t, .side = null, .r1_applied = false };
    }
    const ready4cmix = try buildReady4Cmix(gpa, enwik9, order_bytes);
    errdefer gpa.free(ready4cmix);

    var dict = try pre.Dictionary.load(gpa, dict_bytes);
    defer dict.deinit();
    const wrt = try pre.encode(gpa, ready4cmix, &dict);

    // r1_reorder (payload_lex) applies only when the post-WRT stream is the exact
    // enwik9 length (541,126,651 + 45,332,670 = 586,459,321). cmix-lex HARD-ABORTS on a
    // size mismatch (r1_reorder_transform.cpp:423-428) rather than shipping a non-r1
    // (worse) archive. We match that: a mismatch means an upstream split/reorder/phda9/
    // WRT byte-count drift and MUST NOT be shipped silently — abort with the exact
    // actual-vs-expected length so it can be localized and fixed.
    if (wrt.len != r1.kExpectedStreamLen) {
        std.debug.print(
            "FATAL: post-WRT length {d} != expected {d} (Δ={d}); phda9/WRT drift — refusing to ship a non-payload_lex archive (cmix-lex aborts here).\n",
            .{ wrt.len, r1.kExpectedStreamLen, @as(i64, @intCast(wrt.len)) - @as(i64, @intCast(r1.kExpectedStreamLen)) },
        );
        gpa.free(wrt);
        return error.R1LengthMismatch;
    }
    defer gpa.free(wrt);
    // Under -Dr1v2 the R1ORD4 derived side is emitted instead (same sorted main
    // region, R1ORDFT4 footer, in-memory self-verify hard-abort). order_bytes
    // is already a parameter here, so the encode side needs no new plumbing.
    const rr = if (comptime build_options.r1v2)
        try r1.reorderEncodedTailV2(gpa, wrt, order_bytes)
    else
        try r1.reorderEncodedTail(gpa, wrt);
    // fieldcodec is the LAST transform before the coder: applied to the whole
    // post-r1 temp (data + side footer), exactly as the measured T1 arm was
    // built from prep.temp. The r1 side blob above is minted on the
    // un-swapped stream; decode un-swaps BEFORE the r1 restore reads it.
    if (comptime FIELDCODEC_TSSWAP) _ = fieldcodec.encodeInPlace(rr.stream);
    if (comptime FIELDCODEC_CMTMOVE) _ = cmtmove.encodeInPlace(rr.stream);
    return .{ .ready4cmix = ready4cmix, .temp = rr.stream, .side = rr.side, .r1_applied = true };
}

/// runCompressionE (runner.cpp:445-451): full `-e` -> coded payload. NOTE: this
/// invokes the ~40 h arithmetic coder; it exists for completeness and is NOT run
/// during preprocessing verification.
///
/// RSS discipline (10 GB gate): TAKES OWNERSHIP of `enwik9` and frees it — with
/// every other transform buffer — before the coder stage builds its ~9.5 GB
/// predictor. The post-WRT temp is staged through `.zmix_e.temp` in CWD (the
/// C++ runs the same flow through its temp/dot files; 100 GB HDD is allowed).
pub fn runCompressionE(
    gpa: std.mem.Allocator,
    enwik9: []u8,
    dict_bytes: []const u8,
    order_bytes: []const u8,
    seed: i32,
) ![]u8 {
    var tp_buf: [64]u8 = undefined;
    const temp_path = tempName(&tp_buf, "e");
    var enwik9_owned: ?[]u8 = enwik9;
    defer if (enwik9_owned) |e9| gpa.free(e9);
    {
        const prep = try prepareE(gpa, enwik9_owned.?, dict_bytes, order_bytes);
        gpa.free(enwik9_owned.?);
        enwik9_owned = null;
        gpa.free(prep.ready4cmix);
        if (prep.side) |s| gpa.free(s);
        const werr: ?anyerror = blk: {
            writeFileBytes(temp_path, prep.temp) catch |e| break :blk e;
            break :blk null;
        };
        gpa.free(prep.temp);
        if (werr) |e| return e;
    }
    const payload = try codeTempFromFile(gpa, temp_path, dict_bytes, true, seed);
    std.fs.cwd().deleteFile(temp_path) catch {};
    return payload;
}

/// reassemble enwik9 from the WRT-decoded ready4cmix stream (`.input_decomp`)
/// (runner.cpp:396-406): split4Decomp -> phda9 decode -> sort -> intro++main++coda.
fn reassembleFromInputDecomp(gpa: std.mem.Allocator, input_decomp: []const u8) ![]u8 {
    const sp = split.split4Decomp(input_decomp); // {main, intro, coda}
    const main_restored = try phda9.decodeTxtWit(gpa, sp.main);
    // Freed EAGERLY after the sort (not deferred): main_restored + main_sorted +
    // the output concat would otherwise stack ~2.8 GB simultaneously.
    const main_sorted = reorder_mod.sortMain(gpa, main_restored) catch |e| {
        gpa.free(main_restored);
        return e;
    };
    gpa.free(main_restored);
    defer gpa.free(main_sorted);

    const total = sp.intro.len + main_sorted.len + sp.coda.len;
    var out = try gpa.alloc(u8, total);
    var i: usize = 0;
    @memcpy(out[i..][0..sp.intro.len], sp.intro);
    i += sp.intro.len;
    @memcpy(out[i..][0..main_sorted.len], main_sorted);
    i += main_sorted.len;
    @memcpy(out[i..][0..sp.coda.len], sp.coda);
    return out;
}

/// runDecompressionD (runner.cpp:377-408): coded payload -> enwik9. `order_bytes`
/// is unused on a stock decode (sortundoes reorder purely by article id) and
/// may be null; under -Dr1v2 the r1 v2 restore REQUIRES the raw order asset to
/// derive the block permutation (v1 archives still decode with it unused).
pub fn runDecompressionD(
    gpa: std.mem.Allocator,
    archive: []const u8,
    dict_bytes: []const u8,
    order_bytes: ?[]const u8,
    seed: i32,
) ![]u8 {
    // Stage 1 — arith-decode with the temp streamed to disk. The predictor is
    // constructed AND torn down inside (C++ scopes it + FreeFxcmMemory +
    // malloc_trim before the un-transform chain, runner.cpp:308-321).
    var tp_buf: [64]u8 = undefined;
    const temp_path = tempName(&tp_buf, "d");
    _ = try decodeTempToFile(gpa, archive, dict_bytes, seed, temp_path);
    return untransformTempFile(gpa, temp_path, dict_bytes, order_bytes);
}

/// Process-unique temp name (".zmix_<tag>.<pid>.temp"): a fixed CWD name lets a
/// second concurrent run O_TRUNC/consume the first's live stage file (fleet
/// A/Bs from one checkout; parallel test steps) — silent corruption, exit 0.
fn tempName(buf: *[64]u8, comptime tag: []const u8) []const u8 {
    return std.fmt.bufPrint(buf, ".zmix_" ++ tag ++ ".{d}.temp", .{selfPid()}) catch unreachable;
}

/// Cross-platform self-pid for `tempName` above. `std.os.linux.getpid`
/// COMPILES for any x86_64 target -- it emits a raw Linux `syscall` -- so on
/// Windows it would link fine and then execute an arbitrary NT service call at
/// the first `-e`/`-d` staging step. The Linux branch is textually the previous
/// expression, so Linux codegen (and every archive golden) is unchanged.
/// See 8.177b.
inline fn selfPid() u32 {
    if (comptime @import("builtin").os.tag == .windows) return std.os.windows.GetCurrentProcessId();
    return @intCast(std.os.linux.getpid());
}

/// Stage 2 of -D — the un-transform chain off the decoded temp FILE, each
/// buffer freed as soon as it is consumed (shared by the in-RAM-archive and
/// streamed-archive stage-1 variants; peak here is ~2.9 GB, after the ~9.5 GB
/// predictor is gone).
fn untransformTempFile(gpa: std.mem.Allocator, temp_path: []const u8, dict_bytes: []const u8, order_bytes: ?[]const u8) ![]u8 {
    const temp_with_footer = try readFile(gpa, temp_path);
    std.fs.cwd().deleteFile(temp_path) catch {};
    if (comptime FIELDCODEC_CMTMOVE) _ = cmtmove.decodeInPlace(temp_with_footer);
    if (comptime FIELDCODEC_TSSWAP) _ = fieldcodec.decodeInPlace(temp_with_footer);
    if (comptime TEST_NO_R1) {
        // ⛔ TEST ONLY: mirror prepareE's bypass — no r1 footer to restore and no
        // split4Decomp/phda9/sortMain to run; the WRT-decoded stream IS the input.
        defer gpa.free(temp_with_footer);
        var dict_t = try pre.Dictionary.load(gpa, dict_bytes);
        defer dict_t.deinit();
        return pre.decode(gpa, temp_with_footer, &dict_t);
    }
    const wrt = r1.restoreEncodedTail(gpa, temp_with_footer, order_bytes) catch |e| {
        gpa.free(temp_with_footer);
        return e;
    };
    gpa.free(temp_with_footer);
    var dict = try pre.Dictionary.load(gpa, dict_bytes);
    const input_decomp = pre.decode(gpa, wrt, &dict) catch |e| {
        dict.deinit();
        gpa.free(wrt);
        return e;
    };
    dict.deinit();
    gpa.free(wrt);
    defer gpa.free(input_decomp);
    return reassembleFromInputDecomp(gpa, input_decomp);
}

/// runDecompressionD with the coded payload STREAMED from a file range instead
/// of held in RAM: the ~109 MB archive image would otherwise sit under the
/// ~9.5 GB coder peak — ~45% of the whole RSS margin under the 1e10-byte gate
/// (the C++ record streams its payload from disk too). Byte-equal by
/// construction: same header parse, same coder loop, the byte source reads
/// sequential 1 MB chunks via pread and yields 0 past the end exactly like the
/// in-memory variant.
pub fn runDecompressionDStreamed(
    gpa: std.mem.Allocator,
    f: std.fs.File,
    off: u64,
    len: u64,
    dict_bytes: []const u8,
    order_bytes: ?[]const u8,
    seed: i32,
) ![]u8 {
    var tp_buf: [64]u8 = undefined;
    const temp_path = tempName(&tp_buf, "d");
    _ = try decodeTempRangeToFile(gpa, f, off, len, dict_bytes, seed, temp_path);
    return untransformTempFile(gpa, temp_path, dict_bytes, order_bytes);
}

/// Sequential chunked byte source over a file range; returns 0 past the end
/// (the coder reads a few bytes of virtual zero tail, matching the slice
/// variant's `if (pos < body.len) body[pos] else 0`). I/O errors latch a flag
/// checked after the decode loop.
const FileByteSource = struct {
    f: std.fs.File,
    buf: []u8,
    fpos: u64,
    fend: u64,
    rlen: usize = 0,
    rpos: usize = 0,
    io_err: bool = false,

    fn next(s: *FileByteSource) u32 {
        if (s.rpos == s.rlen) {
            if (s.fpos >= s.fend) return 0;
            const want: usize = @intCast(@min(@as(u64, s.buf.len), s.fend - s.fpos));
            const n = s.f.preadAll(s.buf[0..want], s.fpos) catch {
                s.io_err = true;
                s.fpos = s.fend;
                return 0;
            };
            if (n == 0) {
                s.fpos = s.fend;
                return 0;
            }
            s.fpos += n;
            s.rlen = n;
            s.rpos = 0;
        }
        const b = s.buf[s.rpos];
        s.rpos += 1;
        return b;
    }
};

/// decodeTempToFile over a file RANGE (see runDecompressionDStreamed).
fn decodeTempRangeToFile(
    gpa: std.mem.Allocator,
    f: std.fs.File,
    off: u64,
    len: u64,
    dict_bytes: ?[]const u8,
    seed: i32,
    temp_path: []const u8,
) !u64 {
    var hbuf: [37]u8 = undefined;
    const hn: usize = @intCast(@min(len, 37));
    const got = try f.preadAll(hbuf[0..hn], off);
    const hdr = try readHeader(hbuf[0..got]);
    if (hdr.dict_used != (dict_bytes != null)) return error.DictFlagMismatch;
    if (hdr.length == 0) return error.StoredHeaderUnsupported;

    const pred = if (hdr.dict_used)
        try PredictorLex.initSeedWithDict(gpa, hdr.vocab, seed, dict_bytes.?)
    else
        try PredictorLex.initSeed(gpa, hdr.vocab, seed);
    defer pred.deinit();
    if (hdr.dict_used) preprocessorPretrain(pred, dict_bytes.?);

    var hero = if (comptime hh.ON)
        try hh.HeroHead.create(gpa, pred.cacheProbeHidden().len)
    else {};
    defer if (comptime hh.ON) hero.destroy();

    // .read: the -Dfree-prior replay preads this file back (decoder replay
    // source = its own decoded output); comptime-false in stock builds.
    const outf = try std.fs.cwd().createFile(temp_path, .{ .read = FP_ON });
    defer outf.close();
    var wbuf: [1 << 20]u8 = undefined;
    var wn: usize = 0;

    var rbuf: [1 << 20]u8 = undefined;
    var src = FileByteSource{
        .f = f,
        .buf = &rbuf,
        .fpos = off + hdr.body_off,
        .fend = off + len,
    };

    var fp: FpSched = .{};
    if (comptime FP_ON) fpEnableRecording(pred, hdr.length); // -Dfree-prior-b: record coding-time prior (gate i)
    var x1: u32 = 0;
    var x2: u32 = 0xffffffff;
    var x: u32 = 0;
    for (0..4) |_| x = (x << 8) + src.next();

    var produced: u64 = 0;
    while (produced < hdr.length) : (produced += 1) {
        if (comptime hh.ON) hero.beginByte(pred.cacheProbeHidden());
        var byte: i32 = 1;
        while (byte < 256) {
            const p: u32 = if (comptime hh.ON) blk: {
                const base = discretize(pred.predict());
                // -Dadaptive-depth: on skipped bits the coder receives p_head
                // directly; HERO is bypassed + frozen (preflight §1 step 7).
                // Comptime-folds away when the knob is off.
                if (pred.adepthSkipped()) break :blk base;
                const plane: usize = @intCast(31 - @clz(@as(u32, @intCast(byte))));
                const prefix: u32 = @as(u32, @intCast(byte)) - (@as(u32, 1) << @intCast(plane));
                break :blk hero.mix(base, plane, prefix);
            } else discretize(pred.predict());
            const span = x2 - x1;
            const xmid = x1 +% (span >> 16) *% p +% (((span & 0xffff) *% p) >> 16);
            var bit: i32 = 0;
            if (x <= xmid) {
                bit = 1;
                x2 = xmid;
            } else {
                x1 = xmid + 1;
            }
            if (comptime hh.ON) {
                // -Dadaptive-depth: HERO frozen on skipped bits (its mix never
                // staged this bit). Folds to unconditional observe when off.
                if (!pred.adepthSkipped()) hero.observe(bit);
            }
            pred.perceive(bit);
            while (((x1 ^ x2) & 0xff000000) == 0) {
                x1 <<= 8;
                x2 = (x2 << 8) + 255;
                x = (x << 8) + src.next();
            }
            byte += byte + bit;
        }
        wbuf[wn] = @intCast(byte & 0xff);
        wn += 1;
        if (wn == wbuf.len) {
            try outf.writeAll(wbuf[0..wn]);
            wn = 0;
        }
        if (comptime FP_ON) {
            // -Dfree-prior: decoder replay source = its own decoded output,
            // staged in temp_path (flush pending tail, then pread back).
            if (fp.hit(produced + 1)) |fp_pos| {
                try outf.writeAll(wbuf[0..wn]);
                wn = 0;
                try fpEpochFile(pred, &fp, hdr.length, outf, fp_pos);
            }
        }
    }
    try outf.writeAll(wbuf[0..wn]);
    if (src.io_err) return error.InputOutput;
    return hdr.length;
}

/// roundtripE: prove the full `-e` transform chain is invertible WITHOUT the coder
/// (enwik9 -> ready4cmix -> WRT -> r1 -> [identity] -> r1 restore -> WRT decode ->
/// split4Decomp -> phda9 decode -> sort -> reassemble). Returns reconstructed enwik9.
pub fn roundtripE(
    gpa: std.mem.Allocator,
    enwik9: []const u8,
    dict_bytes: []const u8,
    order_bytes: []const u8,
) ![]u8 {
    var prep = try prepareE(gpa, enwik9, dict_bytes, order_bytes);
    defer prep.deinit(gpa);
    // prepareE returns the coder-facing temp, i.e. WITH the swap applied;
    // invert it before the r1 restore, mirroring untransformTempFile.
    if (comptime FIELDCODEC_CMTMOVE) _ = cmtmove.decodeInPlace(prep.temp);
    if (comptime FIELDCODEC_TSSWAP) _ = fieldcodec.decodeInPlace(prep.temp);
    if (comptime TEST_NO_R1) {
        // ⛔ TEST ONLY: same bypass as untransformTempFile.
        var dict_t = try pre.Dictionary.load(gpa, dict_bytes);
        defer dict_t.deinit();
        return pre.decode(gpa, prep.temp, &dict_t);
    }

    var wrt: []u8 = prep.temp;
    var wrt_owned = false;
    if (prep.r1_applied) {
        wrt = try r1.restoreEncodedTail(gpa, prep.temp, order_bytes);
        wrt_owned = true;
    }
    defer if (wrt_owned) gpa.free(wrt);

    var dict = try pre.Dictionary.load(gpa, dict_bytes);
    defer dict.deinit();
    const input_decomp = try pre.decode(gpa, wrt, &dict);
    defer gpa.free(input_decomp);
    return reassembleFromInputDecomp(gpa, input_decomp);
}

// ---------------------------------------------------------------------------

// Equivalence gate for the disk-staged coder twins: same bytes as the in-memory
// coder, both directions, and the file-staged decode inverts the file-staged
// encode. Runs on a 12 KB stream so the >=10000 vocab-scan/header path is hit.
test "staged coder variants match the in-memory coder" {
    const a = std.testing.allocator;
    var input: [12000]u8 = undefined;
    var prng = std.Random.DefaultPrng.init(42);
    const rnd = prng.random();
    const pat = "the quick brown fox <page> jumps </page> 0123456789 ";
    for (&input, 0..) |*c, i| {
        c.* = pat[i % pat.len];
        if (rnd.uintLessThan(u8, 16) == 0) c.* = rnd.int(u8);
    }

    const mem_payload = try codeTemp(a, &input, null, false, 923);
    defer a.free(mem_payload);

    const tmp_path = ".zmix_test.temp";
    try writeFileBytes(tmp_path, &input);
    defer std.fs.cwd().deleteFile(tmp_path) catch {};
    const file_payload = try codeTempFromFile(a, tmp_path, null, false, 923);
    defer a.free(file_payload);
    try std.testing.expectEqualSlices(u8, mem_payload, file_payload);

    const mem_temp = try decodeToTemp(a, mem_payload, null, 923);
    defer a.free(mem_temp);
    try std.testing.expectEqualSlices(u8, &input, mem_temp);

    const out_path = ".zmix_test.out";
    _ = try decodeTempToFile(a, file_payload, null, 923, out_path);
    defer std.fs.cwd().deleteFile(out_path) catch {};
    const file_temp = try std.fs.cwd().readFileAlloc(a, out_path, 1 << 20);
    defer a.free(file_temp);
    try std.testing.expectEqualSlices(u8, &input, file_temp);
}

fn usage() void {
    std.debug.print(
        \\runner_lex — cmix-lex -c/-d codec + full -e/-D enwik9 pipeline (zmix)
        \\
        \\  -c <dict> <in> <out>          compress with dictionary
        \\  -c <in> <out>                 compress, no dictionary
        \\  -n <in> <out>                 no-preprocess body, no pretrain
        \\  -d <dict> <in> <out>          decompress with dictionary
        \\  -d <in> <out>                 decompress, no dictionary
        \\
        \\  -e <dict> <order> <enwik9> <out>       full enwik9 compress (~40 h coder!)
        \\  -D <dict> <order> <payload> <enwik9>   full enwik9 decompress
        \\  --prepare <dict> <order> <enwik9> <ready4cmix_out> <temp_out>
        \\                                          -e preprocessing only (no coder):
        \\                                          dumps .ready4cmix and post-WRT+r1 temp
        \\  --roundtrip <dict> <order> <enwik9>     lossless -e/-D transform check (no coder)
        \\  --codetemp <dict> <temp> <out>          code a prepared temp (coder stage only)
        \\  --ready4 <order> <enwik9> <out>         dump ready4cmix (exact -e WRT input)
        \\  --phda9rt <order> <enwik9>              phda9 stage encode+decode losslessness
        \\  --r1encode <order> <post_wrt> <out>     r1 transform only (v2 under -Dr1v2)
        \\  --r1restore <order> <temp> <out>        r1 restore only (footer-dispatched)
        \\
        \\  Env: ZMIX_SEED=<n>  (default 923, the shipped binary's srand seed)
        \\
    , .{});
}

fn readFile(gpa: std.mem.Allocator, path: []const u8) ![]u8 {
    return std.fs.cwd().readFileAlloc(gpa, path, std.math.maxInt(usize));
}

fn writeFileBytes(path: []const u8, bytes: []const u8) !void {
    const f = try std.fs.cwd().createFile(path, .{});
    defer f.close();
    try f.writeAll(bytes);
}

fn readSeed(gpa: std.mem.Allocator) i32 {
    if (std.process.getEnvVarOwned(gpa, "ZMIX_SEED")) |sv| {
        defer gpa.free(sv);
        return std.fmt.parseInt(i32, sv, 10) catch DEFAULT_SEED;
    } else |_| {}
    return DEFAULT_SEED;
}

// FTZ/DAZ, exactly as ship.zig sets it: the shipped binary flushes denormals
// (record -ffp-model=fast parity), so the experiment/benchmark binary must run
// the same numerics for its bitstreams to be ship-representative at scale
// (iter 288/289 measured outputs identical with/without at 10 MB — this pins
// it structurally). Gate on landing: 50k outputs must stay 13,520 / 10,524.
fn enableFtzDaz() void {
    if (@import("builtin").cpu.arch == .x86_64) {
        var csr: u32 = undefined;
        asm volatile ("stmxcsr (%[p])"
            :
            : [p] "r" (&csr),
            : .{ .memory = true });
        csr |= 0x8040; // FTZ (1<<15) | DAZ (1<<6)
        asm volatile ("ldmxcsr (%[p])"
            :
            : [p] "r" (&csr),
            : .{ .memory = true });
    }
}

pub fn main() !void {
    enableFtzDaz();
    var gpa_state = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa_state.deinit();
    const gpa = gpa_state.allocator();

    const args = try std.process.argsAlloc(gpa);
    defer std.process.argsFree(gpa, args);

    if (args.len < 2) {
        usage();
        return;
    }
    const m = args[1];
    var timer = try std.time.Timer.start();

    // ---- WRT-only (pre-r1) dump: WRT/dict Encode of an arbitrary file ------
    if (std.mem.eql(u8, m, "--wrt")) {
        // --wrt <dict> <in> <out>   (dumps the segment-framed WRT/dict stream)
        if (args.len != 5) return usage();
        const dict_bytes = try readFile(gpa, args[2]);
        defer gpa.free(dict_bytes);
        const in = try readFile(gpa, args[3]);
        defer gpa.free(in);
        var dict = try pre.Dictionary.load(gpa, dict_bytes);
        defer dict.deinit();
        const wrt = try pre.encode(gpa, in, &dict);
        defer gpa.free(wrt);
        try writeFileBytes(args[4], wrt);
        const ms = timer.read() / std.time.ns_per_ms;
        std.debug.print("--wrt: in={d} -> wrt={d} ({d} ms)\n", .{ in.len, wrt.len, ms });
        return;
    }

    // ---- ready4cmix dump (pre-WRT, exact -e cat order) ---------------------
    // cpp1-nullsib screens: mint the exact -e WRT INPUT (main_phda9+intro+coda)
    // so `--wrt` on it reproduces the post-WRT pre-r1 stream class the cpp1
    // fixtures were cut from, WITHOUT running r1 (whose length gate refuses
    // length-changing prepr knobs until their offsets are measured).
    if (std.mem.eql(u8, m, "--ready4")) {
        // --ready4 <order> <enwik9> <out>
        if (args.len != 5) return usage();
        const order_bytes = try readFile(gpa, args[2]);
        defer gpa.free(order_bytes);
        const enwik9 = try readFile(gpa, args[3]);
        defer gpa.free(enwik9);
        const r4c = try buildReady4Cmix(gpa, enwik9, order_bytes);
        defer gpa.free(r4c);
        try writeFileBytes(args[4], r4c);
        const ms = timer.read() / std.time.ns_per_ms;
        std.debug.print("--ready4: enwik9={d} -> ready4cmix={d} ({d} ms)\n", .{ enwik9.len, r4c.len, ms });
        return;
    }

    // ---- phda9 stage-level roundtrip (the decoder-free losslessness gate) --
    // Runs split4+reorder, then phda9 encode AND the UNMODIFIED decoder, and
    // compares. This is the e9-content losslessness proof for encode-side-only
    // (decoder-free) phda9 knobs, independent of the r1 length gate.
    if (std.mem.eql(u8, m, "--phda9rt")) {
        // --phda9rt <order> <enwik9>
        if (args.len != 4) return usage();
        const order_bytes = try readFile(gpa, args[2]);
        defer gpa.free(order_bytes);
        const enwik9 = try readFile(gpa, args[3]);
        defer gpa.free(enwik9);
        const sp = split.split4Comp(enwik9);
        const main_reordered = try reorder_mod.reorder(gpa, sp.main, order_bytes, NUM_OF_ARTICLES);
        defer gpa.free(main_reordered);
        const main_phda9 = try phda9.encodeTxtWit(gpa, main_reordered);
        defer gpa.free(main_phda9);
        const restored = try phda9.decodeTxtWit(gpa, main_phda9);
        defer gpa.free(restored);
        const ok = std.mem.eql(u8, restored, main_reordered);
        const ms = timer.read() / std.time.ns_per_ms;
        std.debug.print(
            "--phda9rt: main_reordered={d} phda9={d} restored={d} LOSSLESS={s} ({d} ms)\n",
            .{ main_reordered.len, main_phda9.len, restored.len, if (ok) "OK" else "FAIL", ms },
        );
        if (!ok) return error.Phda9RoundtripFail;
        return;
    }

    // ---- LAB: phda9 stage applied to an ARBITRARY file --------------------
    // The frontend earnings census needs
    // the phda9 stage as a standalone pessimum control at a sub-e9 tier. The
    // shipped `-e` path only reaches phda9 after split4 + the e9 article
    // reorder, both of which require the full enwik9 article set, so phda9 has
    // no OFF configuration inside `-c`. These two commands let the SAME 100 MB
    // input be coded with and without the stage, holding everything else.
    // `--phda9enc` also verifies its own inverse before writing.
    if (std.mem.eql(u8, m, "--phda9enc")) {
        // --phda9enc <in> <out>
        if (args.len != 4) return usage();
        const in = try readFile(gpa, args[2]);
        defer gpa.free(in);
        const enc = try phda9.encodeTxtWit(gpa, in);
        defer gpa.free(enc);
        const back = try phda9.decodeTxtWit(gpa, enc);
        defer gpa.free(back);
        const ok = std.mem.eql(u8, back, in);
        try writeFileBytes(args[3], enc);
        const ms = timer.read() / std.time.ns_per_ms;
        std.debug.print(
            "--phda9enc: in={d} phda9={d} delta={d} LOSSLESS={s} ({d} ms)\n",
            .{ in.len, enc.len, @as(i64, @intCast(enc.len)) - @as(i64, @intCast(in.len)), if (ok) "OK" else "FAIL", ms },
        );
        if (!ok) return error.Phda9RoundtripFail;
        return;
    }

    // ---- Stage dump (localize a preprocessing divergence) -----------------
    if (std.mem.eql(u8, m, "--stages")) {
        // --stages <order> <enwik9> <main_out> <main_reordered_out> <main_phda9_out>
        if (args.len != 7) return usage();
        const order_bytes = try readFile(gpa, args[2]);
        defer gpa.free(order_bytes);
        const enwik9 = try readFile(gpa, args[3]);
        defer gpa.free(enwik9);

        const sp = split.split4Comp(enwik9);
        try writeFileBytes(args[4], sp.main);
        const main_reordered = try reorder_mod.reorder(gpa, sp.main, order_bytes, NUM_OF_ARTICLES);
        defer gpa.free(main_reordered);
        try writeFileBytes(args[5], main_reordered);
        const main_phda9 = try phda9.encodeTxtWit(gpa, main_reordered);
        defer gpa.free(main_phda9);
        try writeFileBytes(args[6], main_phda9);

        const ms = timer.read() / std.time.ns_per_ms;
        std.debug.print(
            "--stages: intro={d} main={d} coda={d} main_reordered={d} main_phda9={d} ({d} ms)\n",
            .{ sp.intro.len, sp.main.len, sp.coda.len, main_reordered.len, main_phda9.len, ms },
        );
        return;
    }

    // ---- Full enwik9 pipeline modes ---------------------------------------
    if (std.mem.eql(u8, m, "--prepare")) {
        if (args.len != 7) return usage();
        const dict_bytes = try readFile(gpa, args[2]);
        defer gpa.free(dict_bytes);
        const order_bytes = try readFile(gpa, args[3]);
        defer gpa.free(order_bytes);
        const enwik9 = try readFile(gpa, args[4]);
        defer gpa.free(enwik9);

        var prep = try prepareE(gpa, enwik9, dict_bytes, order_bytes);
        defer prep.deinit(gpa);
        try writeFileBytes(args[5], prep.ready4cmix);
        try writeFileBytes(args[6], prep.temp);
        const ms = timer.read() / std.time.ns_per_ms;
        std.debug.print(
            "--prepare: enwik9={d} ready4cmix={d} temp={d} r1_applied={} side={d} ({d} ms)\n",
            .{ enwik9.len, prep.ready4cmix.len, prep.temp.len, prep.r1_applied, if (prep.side) |s| s.len else 0, ms },
        );
        return;
    }
    if (std.mem.eql(u8, m, "--roundtrip")) {
        if (args.len != 5) return usage();
        const dict_bytes = try readFile(gpa, args[2]);
        defer gpa.free(dict_bytes);
        const order_bytes = try readFile(gpa, args[3]);
        defer gpa.free(order_bytes);
        const enwik9 = try readFile(gpa, args[4]);
        defer gpa.free(enwik9);

        const back = try roundtripE(gpa, enwik9, dict_bytes, order_bytes);
        defer gpa.free(back);
        const ok = std.mem.eql(u8, back, enwik9);
        const ms = timer.read() / std.time.ns_per_ms;
        if (ok) {
            std.debug.print("--roundtrip: LOSSLESS OK ({d} bytes, {d} ms)\n", .{ enwik9.len, ms });
        } else {
            var first: usize = 0;
            const n = @min(back.len, enwik9.len);
            while (first < n and back[first] == enwik9[first]) first += 1;
            std.debug.print(
                "--roundtrip: MISMATCH len {d} vs {d}, first diff @ {d} ({d} ms)\n",
                .{ back.len, enwik9.len, first, ms },
            );
            return error.RoundtripMismatch;
        }
        return;
    }
    if (std.mem.eql(u8, m, "-e")) {
        if (args.len != 6) return usage();
        const seed = readSeed(gpa);
        const dict_bytes = try readFile(gpa, args[2]);
        defer gpa.free(dict_bytes);
        const order_bytes = try readFile(gpa, args[3]);
        defer gpa.free(order_bytes);
        // Ownership of enwik9 passes to runCompressionE (freed there pre-coder).
        const enwik9 = try readFile(gpa, args[4]);

        const payload = try runCompressionE(gpa, enwik9, dict_bytes, order_bytes, seed);
        defer gpa.free(payload);
        try writeFileBytes(args[5], payload);

        // Real self-extractor assembly — PORT of runner.cpp:428/455/465/466 +
        // build_and_construct_comp.sh:67-80. The dict/order are self-compressed with
        // our OWN no-dict codec (`cmix -c`, NOT raw), and decomp_bin is the stripped+UPX'd
        // ship binary supplied via ZMIX_DECOMP_BIN (construct_ship.sh sets it). Absent a
        // decomp_bin the archive9 is layout-only (empty prefix) but still slices/parses.
        // archive9 = decomp_bin ++ dict.comp ++ payload ++ HeaderInfo(12).
        const dict_comp = try runCompression(gpa, dict_bytes, null, true, seed);
        defer gpa.free(dict_comp);
        const order_comp = try runCompression(gpa, order_bytes, null, true, seed);
        defer gpa.free(order_comp);
        var decomp_bin: []u8 = &.{};
        var decomp_bin_owned = false;
        if (std.process.getEnvVarOwned(gpa, "ZMIX_DECOMP_BIN")) |p| {
            defer gpa.free(p);
            decomp_bin = try readFile(gpa, p);
            decomp_bin_owned = true;
        } else |_| {}
        defer if (decomp_bin_owned) gpa.free(decomp_bin);
        // DEV/bench path: no weights blob is available here (the ship path gets it
        // from comp9's own tail). Empty => trailer tfweights_size=0, unchanged on LSTM-era.
        const archive9 = try self_extract.assembleArchive9(gpa, decomp_bin, dict_comp, payload, @intCast(order_comp.len), "");
        defer gpa.free(archive9);
        const a9_path = try std.fmt.allocPrint(gpa, "{s}.archive9", .{args[5]});
        defer gpa.free(a9_path);
        try writeFileBytes(a9_path, archive9);

        const ms = timer.read() / std.time.ns_per_ms;
        std.debug.print("-e seed={d}: enwik9={d} -> payload={d} dict.comp={d} archive9={d} ({d} ms)\n", .{ seed, enwik9.len, payload.len, dict_comp.len, archive9.len, ms });
        return;
    }
    if (std.mem.eql(u8, m, "-D")) {
        if (args.len != 6) return usage();
        const seed = readSeed(gpa);
        const dict_bytes = try readFile(gpa, args[2]);
        defer gpa.free(dict_bytes);
        // args[3] (order): unused on a stock decode (kept for CLI symmetry; the
        // file is not even read) — under -Dr1v2 the r1 v2 restore derives the
        // block permutation FROM it, so it is read and threaded through.
        const order_bytes: ?[]u8 = if (comptime build_options.r1v2) try readFile(gpa, args[3]) else null;
        defer if (order_bytes) |ob| gpa.free(ob);
        const payload = try readFile(gpa, args[4]);
        defer gpa.free(payload);
        const enwik9 = try runDecompressionD(gpa, payload, dict_bytes, order_bytes, seed);
        defer gpa.free(enwik9);
        try writeFileBytes(args[5], enwik9);
        const ms = timer.read() / std.time.ns_per_ms;
        std.debug.print("-D seed={d}: payload={d} -> enwik9={d} ({d} ms)\n", .{ seed, payload.len, enwik9.len, ms });
        return;
    }

    // ---- r1-only fixture harnesses (verification ladder §6.4, not ship ops) ----
    // --r1encode <order> <pure_post_wrt> <out>: apply the r1 transform (v1, or
    // v2 under -Dr1v2) directly to a post-WRT stream — isolates the r1 codec
    // from the phda9/WRT pipeline for byte-parity gates against the orderpred
    // fixtures. --r1restore <order> <transformed> <out>: the dispatch inverse.
    if (std.mem.eql(u8, m, "--r1encode")) {
        if (args.len != 5) return usage();
        const order_bytes = try readFile(gpa, args[2]);
        defer gpa.free(order_bytes);
        const pure = try readFile(gpa, args[3]);
        defer gpa.free(pure);
        const rr = if (comptime build_options.r1v2)
            try r1.reorderEncodedTailV2(gpa, pure, order_bytes)
        else
            try r1.reorderEncodedTail(gpa, pure);
        defer {
            gpa.free(rr.stream);
            gpa.free(rr.side);
        }
        try writeFileBytes(args[4], rr.stream);
        const ms = timer.read() / std.time.ns_per_ms;
        std.debug.print("--r1encode: {d} -> {d} bytes (side={d}) ({d} ms)\n", .{ pure.len, rr.stream.len, rr.side.len, ms });
        return;
    }
    // --codetemp <dict> <temp> <out>: code an ALREADY-PREPARED temp stream
    // (the exact bytes `--prepare` dumps) with dict pretraining — i.e. the `-e`
    // pipeline's coder stage on a substrate someone else minted. Same fixture-
    // harness class as --r1encode/--r1restore above: it exists so an arrangement
    // A/B can be coded on the SHIP FLAG LINE without re-running the ~40 h `-e`,
    // and it is the engine equivalent of the lab runner's `-t-caravan`.
    // NOT a ship op — `zmix_ship`/`zmix_form1` never reference this main.
    if (std.mem.eql(u8, m, "--codetemp")) {
        if (args.len != 5) return usage();
        const dict_bytes = try readFile(gpa, args[2]);
        defer gpa.free(dict_bytes);
        const seed = readSeed(gpa);
        const payload = try codeTempFromFile(gpa, args[3], dict_bytes, true, seed);
        defer gpa.free(payload);
        try writeFileBytes(args[4], payload);
        const ms = timer.read() / std.time.ns_per_ms;
        std.debug.print("--codetemp seed={d}: temp={s} -> {d} bytes ({d} ms)\n", .{ seed, args[3], payload.len, ms });
        return;
    }
    if (std.mem.eql(u8, m, "--r1restore")) {
        if (args.len != 5) return usage();
        const order_bytes = try readFile(gpa, args[2]);
        defer gpa.free(order_bytes);
        const temp = try readFile(gpa, args[3]);
        defer gpa.free(temp);
        const wrt = try r1.restoreEncodedTail(gpa, temp, order_bytes);
        defer gpa.free(wrt);
        try writeFileBytes(args[4], wrt);
        const ms = timer.read() / std.time.ns_per_ms;
        std.debug.print("--r1restore: {d} -> {d} bytes ({d} ms)\n", .{ temp.len, wrt.len, ms });
        return;
    }

    // ---- Legacy -c/-d/-n codec (unchanged) --------------------------------
    if (args.len < 4 or args.len > 5 or m.len != 2 or m[0] != '-') {
        usage();
        return;
    }
    const mode = m[1];
    if (mode != 'c' and mode != 'd' and mode != 'n') {
        usage();
        return;
    }
    const has_dict = args.len == 5;
    if (mode == 'n' and has_dict) {
        usage();
        return;
    }

    const seed = readSeed(gpa);
    const dict_path: ?[]const u8 = if (has_dict) args[2] else null;
    const in_path = if (has_dict) args[3] else args[2];
    const out_path = if (has_dict) args[4] else args[3];

    const input = try readFile(gpa, in_path);
    defer gpa.free(input);

    var dict_bytes: ?[]u8 = null;
    if (dict_path) |dp| dict_bytes = try readFile(gpa, dp);
    defer if (dict_bytes) |db| gpa.free(db);

    const result = switch (mode) {
        'c' => try runCompression(gpa, input, dict_bytes, true, seed),
        'n' => try runCompression(gpa, input, null, false, seed),
        'd' => try runDecompression(gpa, input, dict_bytes, seed),
        else => unreachable,
    };
    defer gpa.free(result);
    const elapsed_ms = timer.read() / std.time.ns_per_ms;

    try writeFileBytes(out_path, result);
    std.debug.print("-{c} seed={d}: {d} -> {d} bytes ({d} ms)\n", .{ mode, seed, input.len, result.len, elapsed_ms });
    // LAB: comptime-dead unless -Dram-census (ram_census.zig). Printed AFTER the
    // result line so a census run's stdout/stderr still parses like a stock run.
    ram_census.dump("runner_lex");
}
