const std = @import("std");

// zmix — a Zig port of cmix-lex (fxcm_v26 predictor), a PAQ-derived text
// compressor built for the Hutter Prize.
//
// Optimization notes:
//   * The general `zmix` exe and the shipping/decode binaries default to
//     ReleaseSmall (the small-binary goal of the port).
//   * A per-module split (ReleaseFast predictor inside a ReleaseSmall shell) was
//     measured and REJECTED: it is ~35% slower than plain ReleaseSmall because
//     splitting the predictor into its own module breaks cross-module inlining of
//     the hot predict/perceivecalls into the arithmetic-coder loop. If decode
//     SPEED is ever the binding constraint, build the whole binary ReleaseFast
//     (single module) — that is ~1.6x faster than ReleaseSmall but ~4 MB of code
//     vs ~0.3 MB. Use `-Dship-optimize=ReleaseFast` for that variant.
//
// Target strategy (reproducible + portable release):
//   * Default target is a pinned x86_64 baseline (x86_64_v2: SSE4.2 + POPCNT,
//     supported by essentially every x86-64 CPU since ~2010). This keeps codegen
//     — and therefore the produced bitstream — reproducible across build hosts and
//     avoids emitting ISA the judges' machine may lack.
//   * Pass `-Dnative` to build for the host CPU (with -Dtarget=/-Dcpu= available)
//     instead.
//
// Release build command (produces the shipping self-extractor):
//   zig build ship
pub fn build(b: *std.Build) void {
    // ---- Target: pinned x86_64_v3 baseline by default, native opt-in --------
    // x86_64_v3 == Haswell-level (AVX2 + BMI1/2 + FMA), universal on x86 since ~2015
    // and present on the Hutter judge's Tiger Lake. The ported mixer dot_product/train
    // kernels are 16-lane @Vector(i16); at v2 (SSE, 8 lanes) they compile to 2 ops per
    // 16, at v3 (AVX2, 16 lanes) to 1 — matching the record's `_mm256_madd_epi16`. We
    // deliberately do NOT use AVX-512/native (Tiger Lake downclocks 512-bit). Fixed
    // target = byte-identical across build hosts (reproducibility).
    const use_native = b.option(
        bool,
        "native",
        "Build for the native/host CPU instead of the pinned x86_64_v3 baseline",
    ) orelse false;
    // Optionally link glibc dynamically, like the record binary (cmix-lex ships
    // dynamic). Buys glibc's AVX2/ERMS memcpy (~12% of runtime is memcpy in the
    // static build) and glibc libm (expf). Default stays fully static.
    const use_glibc = b.option(
        bool,
        "glibc",
        "Link glibc dynamically (record-style; faster memcpy/libm)",
    ) orelse false;

    // Optional explicit CPU model for the pinned target (e.g. "sapphirerapids" to
    // match a -march=native C++ baseline on the fleet Xeons).
    const cpu_model = b.option(
        []const u8,
        "cpu-model",
        "Explicit CPU model for the pinned x86_64 target (e.g. sapphirerapids)",
    );

    // -Dwindows: cross-compile the pinned x86_64_v3 baseline for
    // x86_64-windows-gnu instead of Linux. OPT-IN ONLY — the default stays
    // `.os_tag = .linux` so every existing golden/oracle and the shipped
    // recipe are untouched (this option changes no model behaviour and enters
    // no options module, so a default build is bit-identical to one built
    // before the option existed).
    //
    // Why it exists:
    // operator ruling says ONE JUDGE MACHINE MAY BE WINDOWS, and a
    // Windows archive that is byte-identical to the Linux one is the strongest
    // reproducibility evidence available (the committee rebuilds comp9 from
    // source) as well as the decisive test of's determinism P0 — Windows
    // links a completely different libm (UCRT/mingw) than glibc, and
    // `slowLogit`'s `@log` is the one FP site `-Dvendored-libm` does not cover.
    //
    // Constraints of the Windows target, measured:
    //   * `-Dppmd-mmap` is INCOMPATIBLE (std.posix.MAP is `void` on Windows).
    //     Leave it off; it is output-neutral (bit-identical), so the archive
    //     bytes are unaffected — only the arena's residency is, which needs a
    //     large-RAM box (the i9 test box has 63.9 GB).
    //   * `-Dcoldram` is INCOMPATIBLE for the same class of reason
    //     (sigaction/pagemap/madvise). It is also output-neutral.
    //   * `-Dglibc` is meaningless here; the Windows build links the mingw-w64
    //     CRT unconditionally (Zig ships it), which is what supplies libm.
    const use_windows = b.option(
        bool,
        "windows",
        "Cross-compile for x86_64-windows-gnu instead of Linux (opt-in; -Dppmd-mmap is supported via the CreateFileMappingW arena; -Dcoldram remains POSIX-only)",
    ) orelse false;

    const target = if (use_native)
        b.standardTargetOptions(.{})
    else if (use_windows)
        b.resolveTargetQuery(.{
            .cpu_arch = .x86_64,
            .os_tag = .windows,
            .abi = .gnu,
            .cpu_model = .{ .explicit = &std.Target.x86.cpu.x86_64_v3 },
        })
    else if (cpu_model) |cm|
        b.resolveTargetQuery(std.Target.Query.parse(.{
            .arch_os_abi = if (use_glibc) "x86_64-linux-gnu" else "x86_64-linux",
            .cpu_features = cm,
        }) catch @panic("bad -Dcpu-model"))
    else
        b.resolveTargetQuery(.{
            .cpu_arch = .x86_64,
            .os_tag = .linux,
            .abi = if (use_glibc) .gnu else null,
            .cpu_model = .{ .explicit = &std.Target.x86.cpu.x86_64_v3 },
        });
    // -Dglibc must link libc on EVERY artifact: the gnu-abi extern exp/tanhf
    // in sigmoid.zig/lstm_layer.zig otherwise fail to resolve at link time
    // (before this, only `ship` set link_libc, so `zig build -Dglibc=true`
    // could not build runner_lex/zmix/tests at all).
    const link_libc: ?bool = if (use_glibc or use_windows) true else null;

    // Default to ReleaseSmall to keep the binary small (the stated goal of the
    // port). The user can still override with -Doptimize=.
    const optimize = b.option(
        std.builtin.OptimizeMode,
        "optimize",
        "Optimization mode (default: ReleaseSmall for a small binary)",
    ) orelse .ReleaseSmall;

    // ---- Model-lever build options (comptime-baked; the ship recipe sets them,
    // defaults are STOCK cmix-lex so every oracle/golden gate stays green) ----
    // -Didr: INDEL_RECOVERY + IDR_DEPTH=2 + IDR_PATIENCE=3 match-model stack
    //   (fleet-banked: −11,137 @100 MB post-WRT ≈ −62 K enwik9, RAM/time-neutral;
    //   submission_record/{IDR_FINDING,PATIENCE_FINDING}.md).
    // -Dppm-order: PPMd order (banked recipe: 20 — −3,949 @100 MB and faster;
    //   default 25 == stock).
    const use_idr = b.option(
        bool,
        "idr",
        "Enable the banked IDR match-model stack (indel recovery, depth 2, patience 3)",
    ) orelse false;
    const ppm_order = b.option(
        u32,
        "ppm-order",
        "PPMd model order (default 25 = stock cmix-lex; banked ship recipe uses 20)",
    ) orelse 25;
    // -Dvendored-libm: build-time-fixed exp/tanhf ports (src/vendored_math.zig,
    //   ARM optimized-routines algorithms — the same ones modern glibc uses).
    //   Removes the judged-run dependency on the judge's glibc libm (glibc
    //   changed exp at 2.28; 1 ulp of drift desyncs the arithmetic coder) while
    //   keeping glibc-class speed. Applies to BOTH static and -Dglibc builds.
    const vendored_libm = b.option(
        bool,
        "vendored-libm",
        "Use vendored deterministic exp/tanhf (ARM optimized-routines ports) instead of libc/compiler-rt",
    ) orelse false;
    // -Dppmd-mmap: port of cmix-lex `mmap_to_disk` (ppmd.cpp:42-59): back the
    //   PPMd sub-allocator heap with a file-backed MAP_SHARED mapping of
    //   ./ppm.temp and drop its residency on a MADV_DONTNEED cadence, so the
    //   touched PPMd heap stops counting against the Hutter RSS gate.
    //   Output-neutral (bit-identical); costs disk traffic + refault I/O. The
    //   judged cmix-lex ran with this ON; default OFF keeps experiments/tests
    //   on the RAM arena (same as the C++ A/B arms with -DPPMD_MMAP_TO_DISK=0).
    const ppmd_mmap = b.option(
        bool,
        "ppmd-mmap",
        "Back the PPMd heap with ./ppm.temp (cmix-lex mmap_to_disk): bounded RSS, more disk I/O",
    ) orelse false;
    // -Dlstm-cells / -Dlstm-layers / -Dlstm-lr: byte-mixer LSTM shape (stock
    //   cmix-lex: 170 cells x 1 layer, lr 0.03). These are the comptime DEFAULTS
    //   behind predictor_lex.zig's ZMIX_LSTM_* env overrides: the hermetic ship
    //   (which compiles env reads away) BAKES them — the ONLY way the recipe
    //   shape reaches the shipped binary — while runner_lex keeps env override
    //   for experiments. Ship recipe lives in SHIP_RECIPE.env (candidate: 2x96).
    //   -Dlstm-lr is accepted as a STRING and parsed here with the exact same
    //   std.fmt.parseFloat the ZMIX_LSTM_LR env path uses (f32 round-trips
    //   bit-exactly through the options file — verified "0.03" -> 0x3cf5c28f).
    const lstm_cells = b.option(
        u32,
        "lstm-cells",
        "LSTM cells per layer (default 170 = stock cmix-lex; ship recipe: 96)",
    ) orelse 170;
    const lstm_layers = b.option(
        u32,
        "lstm-layers",
        "LSTM layer count (default 1 = stock cmix-lex; ship recipe: 2)",
    ) orelse 1;
    const lstm_lr_str = b.option(
        []const u8,
        "lstm-lr",
        "LSTM learning rate as a decimal string (default \"0.03\" = stock cmix-lex)",
    ) orelse "0.03";
    const lstm_lr: f32 = std.fmt.parseFloat(f32, lstm_lr_str) catch
        @panic("-Dlstm-lr: not a parseable float (want e.g. \"0.03\")");
    // -Dlstm-aux-sparse=K: SPARSIFY the byte-mixer LSTM's AUXILIARY INPUT
    //   *in place* — the opposite design to the killed -Dlstm-aux-topk.
    //   The aux keeps its SHIPPED WIDTH (vocab_size) and its SHIPPED LAYOUT
    //   (mass of symbol s at position s); only the K largest masses are written
    //   and every other position is ZERO. The zero structure is then exploited
    //   arithmetically: layer 0's gate dot and weight-update AXPY walk a K-long
    //   GATHER over the live symbol indices instead of a vocab_size-long dense
    //   span (lstm_layer.zig forwardNeuron/backwardNeuron).
    //   Bytes: MEASURED FREE — the auxparam channel ladder reads
    //   +0.00008 b/B at K=8 and
    //   -0.00334 at K=16, i.e. top-16 sparse BEATS the full dense aux.
    //   ★ Because width, weight-row layout, Glorot fan-in and the glibc-rand
    //   DRAW COUNT are all unchanged, initial weights are bit-identical to
    //   stock and banked LSTM state packs stay comparable — the
    //   no-statepack-reuse trap that applied to -Dlstm-aux-topk does NOT apply
    //   here. Ties: selection is by (mass DESC, id ASC), a STRICT total order,
    //   so the live set AND its slot order are UNIQUE — byte_mixer.selectTopK.
    //   0 = stock (every branch below comptime-dead).
    const lstm_aux_sparse = b.option(
        usize,
        "lstm-aux-sparse",
        "byte-mixer LSTM auxiliary input: 0 = stock dense, K>0 = keep only the K largest masses AT THEIR OWN SYMBOL POSITIONS (gathered dot)",
    ) orelse 0;
    // -Dlstm-head-rank=M: RANK-LIMIT the byte-mixer LSTM's 256-way output head.
    //   Stock is a flat `output_layer_[versions][256][hidden]` slab: 256x193 =
    //   49,408 online-SGD parameters, every row learning independently.
    //   M>0 replaces it with `Ph[versions][M][hidden]` + `Head[versions][256][M+1]`;
    //   `predict` computes `hp = Ph.hidden` once (M dots) and the existing
    //   softmax + PRL fold runs over `Head[i].hp`. The BPTT sufficient statistic
    //   W^T.err becomes Ph^T(Head^T.err) and Ph's own gradient is (Head^T.err) (x) h.
    //  Claim under test: the shared
    //   projection pools statistical strength across all 256 output rows, so in a
    //   single-pass online regime the rank bottleneck buys accuracy AND costs less
    //   work. Offline instrument at d=192: 911,848 -> 842,904 work/byte (-7.6%),
    //   -0.62% half-bpb. This knob exists to price that IN-ENGINE.
    //   Ph init: uniform +/- sqrt(3/hidden) (unit row norm) drawn from the SAME
    //   deterministic rng AFTER the layers, so (a) the layer weights are
    //   bit-identical to stock and (b) at step 0 the composite step is exactly
    //   the stock SGD step PROJECTED onto a random M-dim subspace (Ph^T Ph is a
    //   near-projection for unit-norm rows) -- i.e. the init is not a hidden
    //   learning-rate knob. Head starts at zero, as the stock slab does.
    //   0 = stock (every branch comptime-dead; archives bit-identical).
    const lstm_head_rank = b.option(
        usize,
        "lstm-head-rank",
        "byte-mixer LSTM output head: 0 = stock flat 256xhidden slab, M>0 = rank-M factorisation Ph[M][hidden] + Head[256][M+1]",
    ) orelse 0;
    // -Dlstm-head-lr-scale: scale applied ONLY to the factorised head's own SGD
    //   step (both Head and Ph), and ONLY when -Dlstm-head-rank > 0. The gates,
    //   the BPTT path and -Dlstm-lr are untouched.
    //   WHY IT EXISTS: the offline instrument gave every arm its OWN grid-optimal
    //   learning rate (flat 0.003, rank-64 0.002 at d=176) and the parent report
    //   pre-registers that "any in-engine screen that keeps the shipped lr will
    //   under-read it". Without this knob an adverse in-engine rank reading is
    //   confounded with an lr mismatch and the KILL would be scoped away.
    //   String-parsed like -Dlstm-lr so the value is exact; default "1.0" and
    //   comptime-dead at -Dlstm-head-rank=0.
    const lstm_head_lr_scale_str = b.option(
        []const u8,
        "lstm-head-lr-scale",
        "scale on the factorised head's SGD step only (needs -Dlstm-head-rank>0); default 1.0",
    ) orelse "1.0";
    const lstm_head_lr_scale: f32 = std.fmt.parseFloat(f32, lstm_head_lr_scale_str) catch
        @panic("-Dlstm-head-lr-scale: not a parseable float (want e.g. \"0.5\")");
    // -Dlstm-aux-sparse-dense: the FALSIFICATION CONTROL for the knob above.
    //   Keeps the truncation (aux tail zeroed, same K, same selection) but
    //   walks the aux span DENSELY, exactly as stock does. It therefore buys
    //   NO wall — its only purpose is to separate the two things the sparse
    //   arm does at once:
    //     * dropping the tail masses  (what the auxparam channel ladder
    //       measured as free-or-better), and
    //     * reassociating the gate dot (gathered K-term sum + dense tail
    //       instead of one dense pass), which the ladder did NOT measure.
    //   If the sparse arm's bytes disagree with the ladder, this control says
    //   which half is responsible. Ignored when -Dlstm-aux-sparse=0.
    const lstm_aux_sparse_dense = b.option(
        bool,
        "lstm-aux-sparse-dense",
        "control for -Dlstm-aux-sparse: keep the truncation but walk the aux span densely (isolates truncation from FP reassociation; buys no wall)",
    ) orelse false;
    // -Dmix-lr-scale: global scale on the 23 layer-0 + layer-1 mixer learning
    //   rates (comptime default behind the ZMIX_MIX_LR_SCALE env override; the
    //   hermetic ship BAKES it). 100m curve (unimodal, bracketed):
    //   0.7 = -1,838 minimum vs 1.0 stock. Ship recipe: "0.7".
    const mix_lr_scale_str = b.option(
        []const u8,
        "mix-lr-scale",
        "Global mixer learning-rate scale as a decimal string (default \"1.0\" = stock; ship: \"0.7\")",
    ) orelse "1.0";
    const mix_lr_scale: f32 = std.fmt.parseFloat(f32, mix_lr_scale_str) catch
        @panic("-Dmix-lr-scale: not a parseable float (want e.g. \"0.7\")");
    // -Dmxa-uperr-scale: scale on the 18 fxcm internal-mixer error gains (uperr),
    //   the tuning group-2 surface. Comptime default behind ZMIX_MXA_UPERR_SCALE;
    //   hermetic ship BAKES it. 100m: 0.7 = -1,841 (DOWN wins, like group-1);
    //   1.4 = +4,043. Ship recipe: "0.7".
    const mxa_uperr_scale_str = b.option(
        []const u8,
        "mxa-uperr-scale",
        "fxcm internal-mixer uperr scale as a decimal string (default \"1.0\" = stock; ship: \"0.7\")",
    ) orelse "1.0";
    const mxa_uperr_scale: f32 = std.fmt.parseFloat(f32, mxa_uperr_scale_str) catch
        @panic("-Dmxa-uperr-scale: not a parseable float (want e.g. \"0.7\")");
    // -Dmxa-diag: comptime-gated telemetry on the fxcm internal-mixer (Mixer1)
    //   WEIGHT UPDATE QUANTISATION. Default OFF => every counter sits behind
    //   `if (comptime build_options.mxa_diag)` and vanishes (byte-identical).
    //   ON: per-instance counts of weight updates that round to zero, the
    //   aggregate exact-vs-applied gradient mass, and a |w-129| magnitude
    //   summary dumped at teardown. Measures whether the i16
    //   `w += round(t*err/2^16)` kernel operates at or below its own quantum.
    const mxa_diag = b.option(
        bool,
        "mxa-diag",
        "fxcm Mixer1 update-quantisation telemetry to stderr at teardown (default off = byte-identical)",
    ) orelse false;
    // -Dmxa-radix=K: shift the fxcm Mixer1 fixed-point radix by K bits — weights
    //   stored K bits finer (init bias 129<<K), the update rounds at 2^(16-K)
    //   instead of 2^16, and the output shift absorbs the K back (`>> (11+K)`).
    //   Same learning rate, same prediction scale, 2^K finer update quantum and
    //   2^K smaller weight range. Default 0 = stock, bit-identical.
    const mxa_radix = b.option(
        u32,
        "mxa-radix",
        "fxcm Mixer1 weight radix shift in bits (default 0 = stock/bit-identical; 1..3 legal)",
    ) orelse 0;
    if (mxa_radix > 3) @panic("-Dmxa-radix: only 0..3 keeps the i16 train intermediate in range");
    // -Dmxa-dither: replace the fxcm Mixer1 weight update's round-half-up with
    //   DITHERED (stochastic) rounding on the SAME 1-LSB grid.
    //   Stock:  w += floor((2*t*err + 2^16) / 2^17)   — E[Δw] = 0 whenever
    //           |t*err/2^16| < 1/2, which the -Dmxa-diag census measures at
    //           83.4% of ALL weight updates. The expected gradient of those
    //           updates is ANNIHILATED, not merely rounded.
    //   Armed:  w += floor((2*t*err + u) / 2^17), u ~ U[0, 2^17) from a per-mixer
    //           16-lane xorshift32 => E[Δw] = t*err/2^16 EXACTLY, at the same
    //           step size, the same i16 storage and the same weight range.
    //   Deterministic and decoder-symmetric: the PRNG advances identically on
    //   both ops (same traincall sequence). Default off = bit-identical.
    //   Modes: 0 = off (stock, bit-identical) · 1 = UNIFORM dither (the arm) ·
    //          2 = CONSTANT-half offset (NULL CONTROL: same code path, same extra
    //              arithmetic, same PRNG advance, offset pinned to 2^16 ⇒ must
    //              reproduce stock BIT-FOR-BIT) ·
    //          3 = SIGN-BIASED offset (PESSIMUM CONTROL: a deliberate ratchet).
    const mxa_dither = b.option(
        u32,
        "mxa-dither",
        "fxcm Mixer1 update rounding: 0=stock 1=uniform-dither 2=null-control 3=pessimum-control",
    ) orelse 0;
    if (mxa_dither > 3) @panic("-Dmxa-dither: modes are 0..3");
    // -Dmxa-w32=K: MECHANISM ARM — widen the fxcm Mixer1 weight word to i32 and
    //   put K extra fixed-point bits under the radix point. i32 removes BOTH the
    //   resolution floor and the saturation ceiling at once, so K=8 makes the
    //   update quantum 256x finer with no clipping. Costs 2x the mixer banks
    //   (~+607 MB at full fill) — this is a RESEARCH arm that isolates whether
    //   the quantised update costs bytes, not a shippable design.
    //   0 = off (i16, stock).
    const mxa_w32 = b.option(
        u32,
        "mxa-w32",
        "fxcm Mixer1 weights as i32 with K extra radix bits (default 0 = off/stock i16)",
    ) orelse 0;
    if (mxa_w32 > 12) @panic("-Dmxa-w32: 0..12");
    // -Dmxa1-uperr-scale / -Dmxa2-uperr-scale: the `mxa-uperr-scale` knob reaches
    //   only the 18 `mx_a` mixers. The `-Dmxa-diag` census shows the DEEPEST
    //   dead-zone mixers are exactly the ones it cannot reach — `mx_a1[0..5]`
    //   (the layer-1 chain, 0.53-0.82 LSB) and `mx_a2[0..3]` (the 560-wide tail,
    //   0.63-0.67 LSB, 91-92 % dead) — so a best-vs-best comparison of the update
    //   resolution is incomplete without them. Default "1.0" = stock.
    const mxa1_uperr_scale_str = b.option([]const u8, "mxa1-uperr-scale",
        "fxcm mx_a1 (layer-1) uperr scale as a decimal string (default \"1.0\" = stock)") orelse "1.0";
    const mxa1_uperr_scale: f32 = std.fmt.parseFloat(f32, mxa1_uperr_scale_str) catch
        @panic("-Dmxa1-uperr-scale: not a parseable float");
    const mxa2_uperr_scale_str = b.option([]const u8, "mxa2-uperr-scale",
        "fxcm mx_a2 (tail) uperr scale as a decimal string (default \"1.0\" = stock)") orelse "1.0";
    const mxa2_uperr_scale: f32 = std.fmt.parseFloat(f32, mxa2_uperr_scale_str) catch
        @panic("-Dmxa2-uperr-scale: not a parseable float");
    // -Dlstm-prior-rho: PRL — the byte-LSTM softmax becomes softmax(z + log A),
    //   A = rho*P + (1-rho)*2/V, where P is the compacted PPMd byte distribution
    //   the LSTM already receives as input (the LSTM is trained as a residual on
    //   the PPMd prior instead of predicting from scratch). Default "0.0" =
    //   stock, comptime-dead. rho=1 is legal but unsafe (PPMd zero-mass symbols
    //   lose the uniform floor); bracket at 0.25/0.5/0.75/0.9.
    const lstm_prior_rho_str = b.option(
        []const u8,
        "lstm-prior-rho",
        "byte-LSTM PPMd-prior blend rho as a decimal string (default \"0.0\" = stock)",
    ) orelse "0.0";
    const lstm_prior_rho: f32 = std.fmt.parseFloat(f32, lstm_prior_rho_str) catch
        @panic("-Dlstm-prior-rho: not a parseable float (want e.g. \"0.5\")");
    // -Dlstm-prior-form: the FORM of the PRL fold.
    //   0 (default, stock) = mixture in probability space, A = rho*P + (1-rho)*2/V.
    //   1                  = clamped power (log-domain / temperature) fold,
    //                        A = max(P, 2^-clamp)^tau, tau = -Dlstm-prior-rho.
    //   Form 0 uses rho for BOTH the prior's strength and its tail floor, so the
    //   two cannot be set independently: raising strength removes the floor and
    //   lowering it crushes the log-prior's tail. Form 1 decouples them (tau =
    //   strength, clamp = floor) and delivers a CONSTANT fraction tau of the
    //   centered log-prior at every surprise level. tau is restricted to dyadic
    //   values so A is computed with IEEE sqrt only -- no libm, bit-exact.
    const lstm_prior_form = b.option(
        u32,
        "lstm-prior-form",
        "PRL fold form: 0 = stock probability-space mixture, 1 = clamped power (default 0)",
    ) orelse 0;
    const lstm_prior_clamp = b.option(
        u32,
        "lstm-prior-clamp",
        "form-1 tail floor exponent C: A = max(P, 2^-C)^tau (default 20)",
    ) orelse 20;
    // -Dauxparam-instr: compile the branch-local per-instance runtime fold used
    //   by the `auxparam` research instrument. Default OFF => the shipped hot
    //   loop carries no extra branch.
    const auxparam_instr = b.option(
        bool,
        "auxparam-instr",
        "compile the auxparam instrument's runtime prior fold (default off)",
    ) orelse false;
    // -Dppmd-arena-mb: PPMd sub-allocator heap in MB (default 14000 = ship).
    //   Predictions are heap-size-invariant until pText reaches UnitsStart
    //   (predictor_lex.zig:55-58), so a smaller arena is BYTE-IDENTICAL on the
    //   small tiers and lets a 1m/20m paired screen run without a 14 GB mapping.
    const ppmd_arena_mb = b.option(u32, "ppmd-arena-mb", "PPMd arena MB (default 14000 = ship)") orelse 14000;
    // -Dcm3-diag: comptime-gated ContextMap3 collision/occupancy telemetry
    //   (Track-2 measurement layer). Default OFF ⇒ every counter/scan is behind
    //   `if (comptime build_options.cm3_diag)` and vanishes — byte-identical to
    //   stock. ON adds per-instance probe/hit/replacement/established-kill counters,
    //   a sampled 16-bit-checksum false-merge detector (measures the cmC2 chk/idx
    //   bit-overlap defect that C-HASH64 would fix), and an end-of-run occupancy
    //   scan dumped as one TSV line per ContextMap3 instance to stderr at predictor
    //   teardown. Measure-before-build: tells us if C-HASH64 (≥~0.3% false-merge on
    //   the big tables) is worth building vs pointing C-GROW at the churny tables.
    const cm3_diag = b.option(
        bool,
        "cm3-diag",
        "ContextMap3 collision/occupancy telemetry to stderr at teardown (default off = byte-identical)",
    ) orelse false;
    // -Dcmc2-grow=N: multiply the HIGH-VALUE cmC2 family table sizes by N (default 1).
    //   The award gap (iter 213) is memory quality = cmC2 collision rate vs cmix-v21's
    //   31GB. cm3-diag confirmed these families saturate/churn; the ratio arms (M-B
    //   byte-order cm_c2[0-3] +8,378/halved, M-C word/sentence cm_c2[4,5,13,17,18,19,20]
    //   +14,368/halved @100MB) confirmed they carry value. Growing them ×2 (~+1.75GB,
    //   funded by the ppmd-mmap valve + fix14-16 lazy-fill freed RAM) recovers collision
    //   quality. N must be a power of 2 (tmask stays pow2). Default 1 = byte-identical.
    const cmc2_grow = b.option(
        u32,
        "cmc2-grow",
        "Multiply high-value cmC2 family table sizes by N (default 1; ship C-GROW: 2)",
    ) orelse 1;
    // -Dcmc2-hash64: derive the ContextMap3 checksum from an INDEPENDENT mix of the
    // context hash instead of the index-overlapping bit-slice (fixes ~1.3% false
    // merges on the big tables). Zero RAM/time; NUMERICS variant (ratio ladder).
    const cmc2_hash64 = b.option(
        bool,
        "cmc2-hash64",
        "Independent (non-index-overlapping) ContextMap3 checksum (default off = stock)",
    ) orelse false;
    // -Dcmc2-spill=N (N = MB, default 0 = OFF = byte-identical): C-SPILL, the
    //   highest-value-per-RAM-byte memory-quality lever. cm3-diag located the award
    //   gap in cmC2 cold-tail churn — low-evidence singletons evicted before their
    //   2nd occurrence, their bit-history discarded and relearned cold. C-SPILL
    //   catches each evicted low-evidence victim in ONE shared open-addressed side
    //   table (4 bytes/entry: 24-bit fingerprint + 8-bit root state) keyed by
    //   (bucket index, victim checksum). On a later bucket miss for the same context
    //   the saved state is resurrected into the fresh slot instead of starting cold,
    //   feeding the existing 2nd-occurrence machinery warm. 4 B/entry vs the
    //   ContextMap's 128 B/bucket = ~25x denser use of the same capped RAM. Default 0
    //   ⇒ the whole path is `if (comptime cmc2_spill != 0)` and vanishes
    //   (byte-identical). Deterministic ⇒ codec-invertible (encoder/decoder run the
    //   same logic on the same past bytes). Ship/fleet: -Dcmc2-spill=128.
    const cmc2_spill = b.option(
        u32,
        "cmc2-spill",
        "Shared cmC2 cold-tail spill side-table size in MB (default 0 = off = byte-identical; fleet: 128)",
    ) orelse 0;
    // -Dcm3pack=WP — family `state-width-exchange`, THE DECOUPLED BUCKET.
    //
    //   The discovery: `cm3.zig:313` evicts on argmin of the RAW ROOT STATE BYTE, so
    //   a bit-history state is TWO objects sharing one byte — a predictor state AND
    //   the cache priority. Narrowing the state alone therefore collapses the
    //   priority's dynamic range and measures +1.86 % ADVERSE. The fix the law names
    //   is to DECOUPLE: give the bucket an EXPLICIT per-slot eviction priority,
    //   charged against the same 128 bytes, and the state is then free to shrink:
    //       S(w, p) = floor(1016 / (16 + 7w + p))
    //   16-bit chk + seven w-bit states + a p-bit priority. S(8,0) = 14 = the shipped
    //   `E1<14,128>` BY CONSTRUCTION (the report's own G1 gate), and (w,p) = (5,4)
    //   gives 18 slots/bucket at IDENTICAL RAM.
    //
    //   Encoding: 0 = OFF (stock byte layout, bit-identical). Otherwise WP = 10*w + p,
    //   e.g. 80 = the packed NULL-TRANSFORM CONTROL (w=8, p=0, 14 slots — must
    //   reproduce stock bit-for-bit, including the C++ oracle goldens), 54 = the
    //   primary arm (5-bit state + 4-bit priority, 18 slots).
    //
    //  ⚠ ONE option, and it lives in `cmfast_opts` ONLY — NOT `model_opts` (the own-module rule:
    //   model_opts is imported by every module, and every option string plus its
    //   plumbing lands in decomp_bin, which Form-1 charges TWICE). The pre-existing
    //   cm3 knobs (cm3-diag, cmc2-spill, cmc2-hash64) are duplicated into both and
    //   pay that fee; this one does not. Single integer ⇒ one option string, not three.
    const cm3pack = b.option(
        u32,
        "cm3pack",
        "ContextMap3 DECOUPLED bucket: 0 = off (stock, bit-identical), else WP = 10*state_width + priority_bits (80 = packed null control, 54 = 5-bit state + 4-bit priority = 18 slots); +100 = the COUPLING CONTROL (same geometry, eviction reads the state byte)",
    ) orelse 0;
    // -Dcmc2-grow-hot=N: targeted x-N growth of the four EXTREME-churn tables
    //   (cm_c2[5,6,16,18] — ~94% of the established-kill volume; +689MB at x2 vs
    //   +3.4GB for the full family grow). The RAM-budget-efficient C-GROW.
    const cmc2_grow_hot = b.option(
        u32,
        "cmc2-grow-hot",
        "Multiply the 4 extreme-churn cmC2 tables by N (default 1; ship candidate: 2 = +689MB)",
    ) orelse 1;
    // -Dppmd-evict-bytes: PPMd valve MADV_DONTNEED cadence (default 5000 =
    //   record parity). The RAM gate is a high-water mark; a tighter interval
    //   caps the between-eviction file-resident transient (~1GB at 5000 on the
    //   sm30 enwik9 run), buying anon budget for grow tables. Output-neutral
    //   (madvise timing can't change heap content); costs wall (more refaults).
    const ppmd_evict_bytes = b.option(
        u64,
        "ppmd-evict-bytes",
        "PPMd valve eviction cadence in processed bytes (default 5000 = cmix-lex parity; tighter = lower RSS peak, more wall)",
    ) orelse 5000;
    // -Dcmc2-hot2=0..8: grow-hot v2 — census-ranked per-table growth tiers
    //   ⚠ This block said "0..6" until while the option help string
    //   below already documented tiers 7 and 8 — the comment was not updated
    //   when they landed, and the drift nearly caused tier 7 (+64MiB, ADOPTED)
    //   to be confused with tier 2 ("stretch", +499.5MB measured). Tier 7 =
    //   tier6+[17]x2 (+64MiB, -882 B @100m); tier 8 = tier6+[20]x2 (+256MiB).
    //   (supersedes -Dcmc2-grow-hot; both include the v1 hot set; set only one).
    //   1 conservative +271MB/v1 (-cmcr/cmcr2 x2 + tiny x4). 2 stretch +1085MB/v1
    //   (cmcr x4). 3 BUDGET = tier1 minus cm_c2[5] = +422MB over STOCK, RAM-safe
    //   fit (the ship grow candidate: -13,881@100m). Measured: t3 -13,881 /
    //   t1 -21,617 / t2 -35,663 @100m. Tiers 5/6 extend tier 1 with
    //   cm_c2[16]x4 / then cm_c2[18]x4; tier 6 measured -3,445 B @100MB
    //  for +302.0 MB over tier 1.
    // -Dlstm-update-limit=N: Adam step-count cap in the LSTM trainer (stock
    // 3000 = cmix-lex UPDATE_LIMIT; 0 = uncapped). Confirmed slice100m forklab:
    // uncapped = -4,929 B exact (grows 20m -591 / 40m -1,200); engages >=384KB
    // so 50k/1m gates are structurally blind. Second-E candidate lever.
    const lstm_update_limit = b.option(
        u64,
        "lstm-update-limit",
        "LSTM Adam update-step cap (default 3000 = stock; 0 = uncapped)",
    ) orelse 3000;
    // -Dlstm-compact-hist: Candidate A (fifth-e-compiler-layout-findings-jul-23
    //   §Candidate A + olfuse-result-jul-27): one-layer LSTM output-history
    //   compaction — two ping-pong W matrices + the exact `output_error_`
    //   sufficient-statistic ring replace the horizon-deep output-weight ring
    //   (25.8MB -> 0.5MB live at ship shape; measured -37.5% judge12 LL-miss,
    //   -7.2/-7.8% wall on two boxes, bit-identical). The compact scheme is
    //   proven for ONE layer + EVEN horizon only (ping-pong `epoch_ & 1`
    //   parity); Lstm.compactHistory — the single predicate — falls back to
    //   the stock ring for anything else. Default OFF = the stock pre-A
    //   horizon-ring code compiles (comptime-dead, the -D pattern).
    const lstm_compact_hist = b.option(
        bool,
        "lstm-compact-hist",
        "Candidate A: compact one-layer even-horizon LSTM output history (ping-pong W + output_error_ ring; default off = stock pre-A ring)",
    ) orelse false;
    const cmc2_hot2 = b.option(
        u32,
        "cmc2-hot2",
        "grow-hot v2 tier: 0 off, 1 conservative, 2 stretch, 3 BUDGET, 4 LIGHT, 5 tier1+[16]x4, 6 tier5+[18]x4 (+302MB/tier1), 7 tier6+[17]x2 (+64MiB), 8 tier6+[20]x2 (+256MiB)",
    ) orelse 0;
    // -Dcmc-grow=0..4: EXTRA growth of the `cm_c` family ON TOP of whatever
    //   `-Dcmc2-hot2` already applies (the two multiply; tier 0 = all x1 =
    //   byte-identical). Read ONLY by fxcm_v26.zig, which lives in the cmfast
    //   module, so it goes into `cmfast_opts` and NEVER the shared model_opts
    //  (the own-module rule: an option string in model_opts costs ~40 B of
    //   decomp_bin, charged 2x in S; an own-module option measured EXACTLY 0).
    //
    //   WHY THIS FAMILY, from the tier-6 `-Dcm3-diag` DICT census at e8_20m.
    //   estk% = the established-kill rate, the grow-value driver (occ% is not the
    //   grow value; estk% is):
    //     cm_c[3]  estk% 18.544  occ 99.98  fullbkt 99.78   1 MiB  <- most starved
    //     cm_c[2]  estk% 14.305  occ 99.99  fullbkt 99.80  64 KiB  <- largest ABSOLUTE
    //                                                                estk pool (2.97M)
    //     cm_c[1]  estk%  8.416  occ 94.49  fullbkt 73.07   1 MiB  <- NEVER grown at
    //                                                                ANY hot2 tier
    //     cm_c[0]  estk%  0.091  occ 31.08  fullbkt  0.01   1 MiB  <- NOT capacity-bound
    //   cm_c[0] is `reset`-wiped at every `</page>` (fxcm_v26.zig:1132), so its
    //   working set is ONE ARTICLE and it already fits: its misses are RESET
    //   misses, which no amount of capacity can repair. Tier 1 exists purely as
    //   the falsifier for that reading.
    const cmc_grow = b.option(
        u32,
        "cmc-grow",
        "extra cm_c/cmcr growth on top of -Dcmc2-hot2: 0 off (bit-identical), 1 cm_c[0]x4 (falsifier), 2 cm_c[1,2,3]x4, 3 cm_c[1]x16 [2,3]x16, 4 cm_c[2]x16 alone, 5 cmcr x2 extra (=x4 total, +320 MB)",
    ) orelse 0;
    // -Dsta6-mdc: STA6 state-table max-discount-count (STA_PARAMS[4][6]).
    //   Default 22 = stock, byte-identical (the override row equals the stock
    //   row at comptime). Measured lever: 30 ("never-discount", exp-bigswing
    //   STA6: -166 B @20m, -3.66 B/MB on 8 paired e9 windows, monotone).
    const sta6_mdc = b.option(
        i32,
        "sta6-mdc",
        "STA6 max-discount-count (default 22 = stock bit-identical; measured lever: 30)",
    ) orelse 22;
    // -Dm0lr-comp8 is the historical name for the shipped comp10 wrapper:
    // the 8 core absolute LR overrides plus the m0/m13 extension. The wrapper
    // reversed at whole-span 100MB (+3,092 B), so the candidate explicitly
    // sets false. Default false = stock bit-identical.
    const m0lr_comp8 = b.option(
        bool,
        "m0lr-comp8",
        "Historical comp10 mixer0 LR wrapper (8 core + m0/m13); false disables all 10 (default false = stock)",
    ) orelse false;
    // -Dhero: HERO hidden-state ensemble-residual head.
    //  Frozen mechanism:
    //   preflight-jul-18.md + scorer ea9b169f (255 prefix rows x (intercept +
    //   h[t-1] weights), final-logit residual after SSE, row-AdaGrad eta 0.03).
    //  Transfer re-price: 0.0003869 b/pb at 60-80MB = 28,398 B e9 gross,
    //   ~4%/tranche decay. Reopened by the hidden-cache wall KILL (07-20).
    //   Default false = stock bit-identical (comptime-dead). Consumed ONLY by
    //   src/hero_head.zig via runner_lex (root module) -> model_opts only; no
    //   cmfast_opts/lstmfast_opts duplication needed (nothing in those modules
    //   imports it). Evidence surface = cells-176; head follows the live width.
    // -Dhero-eta: the HERO head AdaGrad rate as a decimal string (default "0.03"
    //   = the shipped constant, bit-identical). 80m held-forward re-selection
    //  (PROMOTE-at-eta-0.5): T4 interior
    //   argmax 0.5, +49,708 B raw / +40,188 @0.808 realization vs 0.03; the
    //   train fold would still pick 0.12 — train-only selection is structurally
    //   biased against high rates (the cold-start transient). String->parseFloat
    //   per the house float pattern (bit-exact through the options file).
    // ===== BLEND/RATE GRID BATCH  =====
    // Four never-opened constants from the constant-provenance census (C5/C7/C9/C11).
    // All default to the shipped value => comptime-dead, stock bit-identical.
    // Rationale: the two biggest byte finds of the week (PRL rho 0.15 = -5,162 B
    // @100MB with 42.7x depth amplification; HERO eta 0.5 = +40K held-forward) were
    // BOTH never-opened blend/rate constants on already-shipped knobs. These four are
    // the same class and each governs most or all of the stream.
    //
    // -Dfxcm-final-blend: weight on pr in fxcm's FINAL output blend
    //   pr = (squash(clp(pu2)) + pr*W) >> log2(W+1)  (fxcm_v26.zig:828).
    //   W=3 ships (3:1 in favour of the pre-APM pr). Sets how much of the whole
    //   6-stage APM/mmmO cascade reaches the coded probability; never swept.
    //   Legal {1,3,7} = exact-shift blends only.
    const fxcm_final_blend = b.option(
        u32,
        "fxcm-final-blend",
        "fxcm final-blend weight on pr: 1|3|7 (default 3 = shipped 3:1)",
    ) orelse 3;
    // -Dlstm-adam-beta1: LSTM Adam momentum (lstm_layer.zig:263). Ships 0.025 —
    //   near-memoryless vs the DL-standard 0.9; tuned once upstream at 1x170/no-PRL
    //   and never revisited at c192+PRL.
    const lstm_adam_beta1_str = b.option(
        []const u8,
        "lstm-adam-beta1",
        "LSTM Adam beta1 as a decimal string (default \"0.025\" = shipped)",
    ) orelse "0.025";
    const lstm_adam_beta1: f32 = std.fmt.parseFloat(f32, lstm_adam_beta1_str) catch
        @panic("-Dlstm-adam-beta1: not a parseable float");
    // -Dsscm-base-rate: SSCM adaptation-rate base (sscm.zig:78, rate = r + BASE).
    //   The sscmrate ladder only ever toggled r 0->1 (rate 7->8); BASE 7 never swept.
    const sscm_base_rate = b.option(
        i32,
        "sscm-base-rate",
        "SSCM base adaptation rate (default 7 = shipped)",
    ) orelse 7;
    // -Dapm-wupd: APM interpolated-map UPDATE RULE (fxcm_v26.zig ApmV26.p).
    //   PAQ7's APM (2005), inherited unchanged through paq8l/paq8hp12any/cmix into
    //   the six fxcm APMs, refines the two bracket points t[lo], t[lo+1] with the
    //   SAME full step `(g-t)>>rate` regardless of the interpolation weight `w`
    //   that produced the prediction. The prediction is a linear blend
    //   (t[lo]*(128-w) + t[lo+1]*w)>>11, so the correct SGD update is weighted by
    //   (128-w)/128 and w/128. Shelwien's SSE (src/sse.zig sseUpdate) — sitting in
    //   the SAME binary — already does the weighted thing. Arms:
    //     0 = stock (unweighted, bit-identical)
    //     1 = weighted, rate-normalised (x2 so the mean step is preserved)
    //     2 = nearest-point only (hard assignment)
    //     3 = Shelwien form: move the interpolated value to g, redistribute
    //         preserving the inter-point gap, split by w
    const apm_wupd = b.option(
        u32,
        "apm-wupd",
        "APM interpolated update: 0 stock, 1 weighted, 2 nearest, 3 shelwien, 4 wider-kernel (default 0)",
    ) orelse 0;
    // -Ddsm-tag: collision detection for the 5 DirectStateMap banks (639 MB of
    //   direct-indexed, UNTAGGED bit-history state — see direct_state_map.zig).
    //   Positions halve, each state gains an interleaved tag byte ⇒ TOTAL RAM
    //   UNCHANGED. Default false = stock layout, comptime-dead.
    // -Dapm-diag: side-channel census of the APM cell-visit density (bit-identical
    //   output; settles by measurement whether the 2-cell boxcar is biased).
    const apm_diag = b.option(bool, "apm-diag", "APM cell-visit census (default false)") orelse false;
    // -Dcoder-diag: exact upper bound on what ANY precision widening could recover.
    const coder_diag = b.option(bool, "coder-diag", "coder precision census (default false)") orelse false;
    // -Ddsm-div=k: shrink every DirectStateMap bank by 2^k positions — the load-factor
    //   knob that reproduces the e9 aliasing regime at a small tier (see the module doc).
    const dsm_div_log2 = b.option(u32, "dsm-div", "DirectStateMap size divisor, log2 (default 0 = stock)") orelse 0;
    // -Dsse-ffl6 / -Dsse-ffl7: SSE flag-history context width. Each bit removed halves
    //   the bank (s6 = 352 MiB at the shipped 7 bits, s7 = 84 MiB at 5). Shelwien's
    //   mod_ppmd sizing, never re-derived for a 590-input mixer's output distribution.
    const sse_ffl6 = b.option(u32, "sse-ffl6", "SSE s6 flag-history bits (default 7 = shipped)") orelse 7;
    const sse_ffl7 = b.option(u32, "sse-ffl7", "SSE s7 flag-history bits (default 5 = shipped)") orelse 5;
    // -Dcm3-ts-rate: adaptation rate (log2) of ContextMap3's `ts[256]` state->probability
    //   decoder — the object that feeds 525 of the 590 layer-0 inputs. Fixed-rate EMA,
    //   shipped at 14 since cmix, never swept. See cm3.zig for the shift-pair constraint.
    // -Dcm3-ts-runidx: THE PLACEMENT ARBITRAGE. Index ContextMap3's leaf readout
    // `ts[]` by the per-context run model's own bucket (agrees?/predicted bit/run
    // count), which the same `mix` already computes for its mixer input. Offline
    // instrument (readout-placement-arbitrage) reads -0.70..-1.75% of
    // coded output OVER AND ABOVE keeping the mixer input. ts 1 kB -> 32 kB/bank.
    const cm3_ts_runidx = b.option(
        u32,
        "cm3-ts-runidx",
        "Cardinality of the run bucket on ContextMap3's ts[] readout: 1=off (bit-identical), 4, 8, 16, 32",
    ) orelse 1;
    const cm3_ts_div2idx = b.option(
        u32,
        "cm3-ts-div2idx",
        "16-bit distinct-successor sketch across BOTH free nodes (5+6), popcount 0..16. 1=off, 9, 17",
    ) orelse 1;
    const cm3_ts_div2idx_ctrl = b.option(
        bool,
        "cm3-ts-div2idx-ctrl",
        "Matched-cardinality control for -Dcm3-ts-div2idx",
    ) orelse false;
    const cm3_ts_escidx = b.option(
        u32,
        "cm3-ts-escidx",
        "Index ContextMap3's ts[] readout by a per-BIT escape indicator over node 6's successor mask. 1=off, 5",
    ) orelse 1;
    const cm3_ts_escidx_ctrl = b.option(
        bool,
        "cm3-ts-escidx-ctrl",
        "Matched-cardinality control for -Dcm3-ts-escidx",
    ) orelse false;
    const cm3_ts_dividx = b.option(
        u32,
        "cm3-ts-dividx",
        "Index ContextMap3's ts[] readout by a per-SLOT distinct-successor sketch (free: node 6). 1=off, 4, 8, 9, 16",
    ) orelse 1;
    const cm3_ts_dividx_ctrl = b.option(
        bool,
        "cm3-ts-dividx-ctrl",
        "Matched-cardinality control for -Dcm3-ts-dividx",
    ) orelse false;
    const cm3_ts_recidx = b.option(
        u32,
        "cm3-ts-recidx",
        "Index ContextMap3's ts[] readout by a per-SLOT recency stamp (free: node 6 of the bpos-0 slot). 1=off, 4, 8, 16",
    ) orelse 1;
    const cm3_ts_recidx_ctrl = b.option(
        bool,
        "cm3-ts-recidx-ctrl",
        "Matched-cardinality control for -Dcm3-ts-recidx",
    ) orelse false;
    const cm3_ts_visidx = b.option(
        u32,
        "cm3-ts-visidx",
        "Index ContextMap3's ts[] readout by the per-SLOT visit count (free: node 5 of the bpos-0 slot). 1=off, 4, 8, 16",
    ) orelse 1;
    const cm3_ts_visidx_ctrl = b.option(
        bool,
        "cm3-ts-visidx-ctrl",
        "Matched-cardinality control for -Dcm3-ts-visidx",
    ) orelse false;
    const cm3_ts_runidx_ctrl = b.option(
        bool,
        "cm3-ts-runidx-ctrl",
        "Matched-cardinality control for -Dcm3-ts-runidx: same split, random-but-stable index",
    ) orelse false;
    const cm3_ts_rate = b.option(
        u32,
        "cm3-ts-rate",
        "ContextMap3 ts[] statemap adaptation rate, log2 (default 14 = shipped)",
    ) orelse 14;
    //   Mode 2 is the ISOLATING CONTROL: halve the positions and DO NOT tag, so the
    //   arm-vs-control delta is the detection alone, not the position count. Without
    //   it a tagged reading confounds the two (the control `indirect-tag` used).
    const dsm_tag = b.option(
        u32,
        "dsm-tag",
        "DirectStateMap: 0 stock, 1 interleaved tag at equal RAM, 2 halved-untagged CONTROL",
    ) orelse 0;
    // -Dmixer-decay-floor: MixerLex terminal LR-decay rung (mixer_lex.zig:306).
    //   Ships 0.2 and governs ~99.5%% of the run; the ladder was tuned upstream at
    //   mix-lr-scale 1.0 while we ship 0.7, so the effective terminal rate is
    //   base*0.7*0.2 — an operating point the rung was never re-swept at.
    const mixer_decay_floor_str = b.option(
        []const u8,
        "mixer-decay-floor",
        "MixerLex terminal decay rung as a decimal string (default \"0.2\" = shipped)",
    ) orelse "0.2";
    const mixer_decay_floor: f32 = std.fmt.parseFloat(f32, mixer_decay_floor_str) catch
        @panic("-Dmixer-decay-floor: not a parseable float");
    // -Dfxcm-auxctx-fix: repoint the auxiliary_context consumer from the fxcm's
    //   neutral PAD (index 560) to its last ACTIVE slot, restoring the selector's
    //   dynamic range from 9 live buckets to 16. Upstream defect, faithfully ported.
    //  MEASURED: -707 B @e8_20m nodict, -3,294 @100MB,
    //   wall +0.078%, RSS +384 kB => ADOPT-FREE. Default false = bit-identical.
    const fxcm_auxctx_fix = b.option(
        bool,
        "fxcm-auxctx-fix",
        "Repoint auxiliary_context at the last active fxcm slot (default false = stock)",
    ) orelse false;
    const hero_eta_str = b.option(
        []const u8,
        "hero-eta",
        "HERO head AdaGrad rate as a decimal string (default \"0.03\" = shipped constant)",
    ) orelse "0.03";
    const hero_eta: f64 = std.fmt.parseFloat(f64, hero_eta_str) catch
        @panic("-Dhero-eta: not a parseable float (want e.g. \"0.5\")");
    const hero = b.option(
        bool,
        "hero",
        "HERO 255-row h[t-1] residual head (default false = stock bit-identical; ~28K e9 gross filler)",
    ) orelse false;
    // -Dlstm-bwvec: fast-math LSTM SIMD (NOT bit-identical, fresh-archive only).
    const lstm_bwvec = b.option(
        bool,
        "lstm-bwvec",
        "Vectorize LSTM backward element-wise loops (fast-math, NOT bit-identical — for fresh-archive builds only)",
    ) orelse false;
    // -Dr1v2: R1ORD4 derived payload_lex side (r1ord4-derived-side; -170,309 B,
    //   decode-wall validated 07-15). Engages ONLY at the e9 post-WRT stream;
    //   50k/1m gates untouched. Default OFF = stock R1ORD3, bit-identical.
    const r1v2 = b.option(
        bool,
        "r1v2",
        "R1ORD4 derived payload_lex side (default off = stock R1ORD3, bit-identical)",
    ) orelse false;
    // -Dfree-prior: comma-separated TEMP byte positions for deterministic
    //   mid-stream LSTM self-retrain epochs (free-prior variant (a)).
    //   At each scheduled position the
    //   coder pauses and replays the already-coded temp prefix through the LSTM
    //   training path only (uniform prior — PRL blend neutralized; no CM/PPMd/
    //   mixer state touched). Both coder ops hit identical positions with
    //   identical bytes ⇒ replay symmetry by construction. Default "" = stock,
    //   every hook comptime-dead (the -D pattern). e9 candidate: "12500000,36960000".
    const free_prior = b.option(
        []const u8,
        "free-prior",
        "Comma-separated TEMP byte positions for LSTM free-prior retrain epochs (default \"\" = stock bit-identical)",
    ) orelse "";
    // -Dfree-prior-hash: print pos + replay wall + sha256 of the full LSTM
    //   state after each free-prior epoch (stderr). The gate-2 proof obligation:
    //   the -c and -d lines for the same archive must match bit-for-bit.
    const free_prior_hash = b.option(
        bool,
        "free-prior-hash",
        "Print LSTM-state sha256 after each free-prior epoch (gate-2 symmetry evidence; default false)",
    ) orelse false;
    // -Dcoldram: touched-then-cold page reclamation for big model slabs
    // (mprotect probe ladder -> exact LZ attic + MADV_DONTNEED; SIGSEGV
    // restore). Bit-identical by construction (page contents round-trip
    // exactly); comptime-dead when false. See src/coldram.zig + preflight
    const coldram_on = b.option(bool, "coldram", "Cold-page attic: reclaim touched-then-idle model pages (bit-identical, default off)") orelse false;
    // -Dcoldram-pool-only: keep ONLY the MixerLex row bump pool (the
    // densification term) and compile the whole probe/attic/SIGSEGV half out.
    // The adoption split priced in: no signal
    // handler, no mprotect, no page-fault tax, no attic code fee — for the
    // term that carries the large majority of the measured RAM. Requires
    // -Dcoldram=true; ignored otherwise.
    const coldram_pool_only = b.option(bool, "coldram-pool-only", "coldram: row-pool densification only, no probe/attic/SIGSEGV machinery (default false)") orelse false;
    // -Dmixshadow: INSTRUMENT ONLY (comptime-dead by default). Runs N counterfactual
    // copies of the 23+1 mixer stack on the SAME 590-input vector under alternative
    // update rules and reports each one's coded cost. Sound because v is invariant to
    // the mixing stage (no mixer output feeds any leaf model). The shipped archive is
    // byte-identical with this ON — the shadow is pure observation.
    const mixshadow = b.option(bool, "mixshadow", "Shadow-mixer counterfactual instrument (default false, comptime-dead)") orelse false;
    // -Dmixer-e2e: train the 23 layer-0 mixers on the EXACT gradient of the coding
    // loss, dL/dw_i = (sigma(z) - y) * gamma_i * v, instead of each mixer's own
    // local error. Legal and cheap because the layer-0 cascade + layer-1 mixer are
    // AFFINE in v, so gamma_i = dz/dh_i is an exact 253-MAC backward recursion.
    // Also seeds the layer-1 row to 1/23 (zero-init is a fixed point of exact GD).
    // Comptime-dead at default false = bit-identical stock.
    const mixer_e2e = b.option(bool, "mixer-e2e", "Exact end-to-end mixer gradient (default false, comptime-dead)") orelse false;
    const mixer_e2e_norm = b.option(bool, "mixer-e2e-norm", "mixer-e2e: normalise gamma to RMS 1 across the 23 mixers") orelse false;
    // the own-module rule: the knobs above — and -Dmix-l1-rate below — live in their consuming
    // files' OWN options module ("mixer_options", consumers:
    // src/mixer/shadow_mix.zig + src/predictor_lex.zig), attached alongside
    // model_opts below — NOT in the shared model_opts, whose every entry lands
    // plumbing in decomp_bin (charged twice under Form-1). The branch had them
    // in model_opts; relocated at merge.
    // -Dmix-l1-rate: the LAYER-1 mixer's base learning rate (stock cmix-lex
    //   literal 0.0003, `predictor_lex.zig` mixer1 init; effective rate is
    //   `mix-l1-rate * mix-lr-scale` = 0.00021 on the 29-flag ship line).
    //  POST-e2e THIS CONSTANT CHANGED ROLE (/): W1 is the leading
    //   term of the credit vector gamma_i = W1[i] + sum_{k>i} gamma_k*u_k[i],
    //   so the rate at which W1 moves sets how fast the learning-rate
    //   ALLOCATION over the 23 layer-0 mixers adapts — a meta-LR. Stock value
    //   has never been swept on this engine.
    //  Lives in mixer_opts (NOT model_opts) per the own-module rule: an option in
    //   its consuming module's own options module costs decomp_bin exactly 0.
    const mix_l1_rate_str = b.option(
        []const u8,
        "mix-l1-rate",
        "Layer-1 mixer base learning rate as a decimal string (default \"0.0003\" = stock)",
    ) orelse "0.0003";
    const mix_l1_rate: f32 = std.fmt.parseFloat(f32, mix_l1_rate_str) catch
        @panic("-Dmix-l1-rate: not a parseable float (want e.g. \"0.001\")");
    const mixer_opts = b.addOptions();
    mixer_opts.addOption(bool, "mixshadow", mixshadow);
    mixer_opts.addOption(bool, "mixer_e2e", mixer_e2e);
    mixer_opts.addOption(bool, "mixer_e2e_norm", mixer_e2e_norm);
    mixer_opts.addOption(f32, "mix_l1_rate", mix_l1_rate);
    // -Dmixskip-probe: INSTRUMENT ONLY, lives in
    // mixer_opts per the own-module rule so it costs decomp_bin exactly 0. Decomposes
    // -Dmixer-blockskip's net instruction delta into its two halves, which
    // paq-lineage-audit §7f measured only as a NET (+1.38 %):
    //   0 = off (arm B: the real data-derived mask)
    //   1 = force the mask EMPTY (arm D: every per-block test still runs, nothing is
    //       ever skipped) ⇒ (D − A) is the PURE BOOKKEEPING COST and (B − D) is the
    //       GROSS SKIP SAVING. Arm D is bit-identical to the dense base by
    //       construction, which is its own control.
    mixer_opts.addOption(u32, "mixskip_probe", b.option(u32, "mixskip-probe", "INSTRUMENT: 0 off, 1 force blockskip mask empty (default 0)") orelse 0);
    // -Dderivdict: Arm E′ encoder-derived dictionary builder (membership bitmap
    // over the K=200,000 enwik9 candidate pool + alphabetical-space permutation
    // → rebuilds english.dic bit-exactly). Comptime-dead when false (default).
    // Fee cost-join:
    const derivdict_on = b.option(bool, "derivdict", "Arm E': enwik9-derived english.dic builder (comptime-dead, default false)") orelse false;
    // -Ds7order: comp_order asset carried as the s7_runlabel re-encoding
    // (id-keyed run labels + id-set exception trailer, src/prepr/s7_order.zig;
    // measured −26,694 B gross on the fourth-E vehicle. Changes ONLY the
    // ship asset-container format (mint + in-image decode); the codec, the raw
    // order text every consumer sees, and the coded payload are untouched.
    // Comptime-dead when false (default) = stock raw-text asset, bit-identical.
    const s7order_on = b.option(bool, "s7order", "comp_order carried in the s7_runlabel re-encoding (asset container only; default false = stock raw-text asset)") orelse false;
    // -Dfree-prior-b: variant (b) — replay through the ACTUAL coding-time PPMd
    //   prior instead of variant (a)'s uniform 2/V (which cancels the -Dlstm-
    //  prior-rho blend and damages the PRL surface,+1,645). The prior
    //   (post-×2-scale ByteMixer aux = 2× the compacted PPMd next-byte dist, the
    //   exact vector PRL consumes) is streamed to a /home scratch file during
    //   forward coding and replayed from it (record-replay, CUDA-spec §3). It is
    //   NOT side data: both ops record the identical stream (same bytes → same
    //   PPMd → same prior). Requires -Dfree-prior=<schedule> too; alone it is
    //   inert. Default false = variant (a)/stock, comptime-dead (the -D pattern).
    const free_prior_b = b.option(
        bool,
        "free-prior-b",
        "Free-prior variant (b): replay through recorded coding-time PPMd prior (PRL-consistent). Needs -Dfree-prior. Default false = variant (a)/stock bit-identical",
    ) orelse false;
    // -Dfree-prior-tap: the variant-(b) prior recorder fidelity.
    //   "exact"  — full vs×f32 prior row (155cca1 default; ~481GB scratch @e9,
    //              e9-INFEASIBLE; used only for the local/20m fidelity-reference A/B).
    //   "lossy"  — the e9-SHIPPABLE top-K+f16 compaction (CUDA_LSTM_SIM_SPEC §3):
    //              per byte, the top-K prior entries as (u8 index, f16 prob) pairs
    //              + one f16 residual tail-mass; replay reconstructs prior[i] = the
    //              recorded prob for a top-K index, tail_mass/(vs−K) uniform for the
    //              rest. ~K*3+2 B/row ⇒ ~56GB @e9 (K=32), fits the 100GB HDD.
    //   Both ops record the IDENTICAL lossy stream (same bytes → same PPMd → same
    //   prior → same top-K selection), so replay symmetry is preserved by
    //   construction (weights-hash gate). Inert unless -Dfree-prior-b + a live
    //   -Dfree-prior schedule; comptime-dead in stock builds (the -D pattern).
    const free_prior_tap = b.option(
        []const u8,
        "free-prior-tap",
        "Variant-(b) prior tap fidelity: \"exact\" (full f32, e9-infeasible) or \"lossy\" (top-K+f16, e9-shippable). Default \"exact\". Needs -Dfree-prior-b",
    ) orelse "exact";
    if (!std.mem.eql(u8, free_prior_tap, "exact") and !std.mem.eql(u8, free_prior_tap, "lossy")) {
        std.debug.print("build: -Dfree-prior-tap must be \"exact\" or \"lossy\", got \"{s}\"\n", .{free_prior_tap});
        std.process.exit(1);
    }
    const free_prior_tap_lossy = std.mem.eql(u8, free_prior_tap, "lossy");
    // -Dfree-prior-topk: K for the lossy tap (number of stored top prior entries
    //   per byte). Default 32 (spec §3 top-32 ⇒ 98B/row ⇒ ~56GB @e9). Escalate to
    //   64 if the lossy fidelity screen shows the compaction erodes the value.
    const free_prior_topk = b.option(
        u32,
        "free-prior-topk",
        "K for the lossy variant-(b) prior tap (top-K entries stored per byte; default 32)",
    ) orelse 32;
    // -Dadaptive-depth: ensemble-saturation cascade skip (consumer:
    // src/predictor_lex.zig ONLY). Per the standing own-module hazard the knob gets
    // its OWN options module ("adepth_options"), attached alongside model_opts,
    // so the shared model_opts does not grow (every model_opts entry lands its
    // plumbing in decomp_bin, which Form-1 charges twice). Default false =
    // comptime-dead, bit-identical stock. Mechanism + tau table:
    const adaptive_depth = b.option(
        bool,
        "adaptive-depth",
        "Adaptive-depth cascade skip via the tier-D pre-cascade head (default false = stock)",
    ) orelse false;
    const adaptive_depth_tau_str = b.option(
        []const u8,
        "adaptive-depth-tau",
        "Adaptive-depth gate threshold on |z_head| as a decimal string (default \"8.0\" = the measured 42%/0.012% point)",
    ) orelse "8.0";
    const adaptive_depth_tau: f32 = std.fmt.parseFloat(f32, adaptive_depth_tau_str) catch
        @panic("-Dadaptive-depth-tau: not a parseable float (want e.g. \"8.0\")");
    const adepth_opts = b.addOptions();
    adepth_opts.addOption(bool, "adaptive_depth", adaptive_depth);
    adepth_opts.addOption(f32, "adaptive_depth_tau", adaptive_depth_tau);
    // -Dextshed: the EXTERNAL-STACK SHED deletion arm (wave-3 of the MB-scale
    // program; hand-off =
    // §4.2/§7(i)). Sheds the 28 external layer-0 READOUT inputs — bracket
    // in[0], direct in[561], match x10 in[562..571], indirect x16 in[572..587]
    // — to the exact neutral 0.0 (== the pad convention) and skips those
    // models' readout/update work, while RETAINING the mixer GATING keys:
    // `mgr.longest_match` (the 10 matchers' perceive+byteUpdate machinery keeps
    // running verbatim, so the gating context is bit-exact vs stock) and every
    // manager-maintained stream/context. PPMd in[588], LSTM in[589] and
    // wordlbl in[559] are NOT in scope and are untouched. Consumers:
    // src/predictor_lex.zig + src/context_manager_lex.zig ONLY, so the knob
    // lives in its OWN options module ("extshed_options") per the own-module rule
    // (a dead knob in the shared model_opts costs ~40 B of decomp_bin,
    // charged twice under Form-1; the own-module pattern is measured S-free).
    // Default false = comptime-dead, bit-identical stock.
    const extshed = b.option(
        bool,
        "extshed",
        "External-stack shed: neutralize the 28 external layer-0 readout inputs (bracket/direct/match x10/indirect x16) and skip their model work; longest_match gating retained (default false = stock)",
    ) orelse false;
    const extshed_opts = b.addOptions();
    extshed_opts.addOption(bool, "extshed", extshed);
    // -Dmixer1-rls (`rls8`): full-matrix second-order (online IRLS / extended-
    // Kalman with forgetting) learning for the LAYER-1 mixer, which is d = 25
    // with a single weight row — the one place in the engine where exact
    // curvature is affordable. Consumer: src/predictor_lex.zig ONLY, via
    // src/mixer/mixer1_rls.zig. Own options module ("rls_options") per
    // the own-module rule — a dead knob in the SHARED model_opts costs ~40 B of
    // decomp_bin, and Form-1 charges decomp_bin TWICE.
    // Default 0 = comptime-dead, bit-identical stock.
    // Ported from a standalone RLS implementation; this port adds the 40m/80m rungs.
    const mixer1_rls = b.option(
        u32,
        "mixer1-rls",
        "Layer-1 mixer learning rule: 0 off/bit-identical, 1 full-matrix RLS, 2 diag, 3 scalar, 4 shuffled pessimum, 5 f64-SGD precision control (default 0)",
    ) orelse 0;
    if (mixer1_rls > 5) @panic("-Dmixer1-rls: must be 0..5");
    // Defaults are the ENGINE OPTIMUM measured at e8_1m and used for every
    // tranche rung (lambda 0.999999, p0 0.1), NOT the lab module's first guess.
    // lambda = 0.9999 is measured to NaN on the dense stream: the stability
    // window is narrow and -Dmixer1-rls-guard exists for that reason.
    const rls_lambda_s = b.option([]const u8, "mixer1-rls-lambda", "RLS forgetting factor as a decimal string (default \"0.999999\" = the measured engine optimum)") orelse "0.999999";
    const rls_p0_s = b.option([]const u8, "mixer1-rls-p0", "RLS initial P = p0*I as a decimal string (default \"0.1\"; p0 is a flat direction)") orelse "0.1";
    const rls_hmin_s = b.option([]const u8, "mixer1-rls-hmin", "RLS curvature floor h_min as a decimal string (default \"0.0001\")") orelse "0.0001";
    const rls_stepcap_s = b.option([]const u8, "mixer1-rls-stepcap", "RLS trust-region cap on |dw| as a decimal string (\"0\" = off)") orelse "0";
    const rls_guard = b.option(bool, "mixer1-rls-guard", "RLS PSD/finiteness guard: restart P from p0*I instead of propagating NaN (default false = reproduces the reference arithmetic exactly)") orelse false;
    // ⚠⚠ THE KNOB THAT DECIDES WHETHER rls8 EXISTS ON THIS TRUNK, and its default
    // is the OPPOSITE of what the obvious correctness argument says. With
    // -Dmixer-e2e (ON the 29-flag ship line) layer-0 is trained by an exact
    // gradient whose leading term is the layer-1 weight row.
    //   true  = differentiate through the head that actually CODED the bit. This
    //           is the self-CONSISTENT choice and it is CATASTROPHIC: MEASURED
    //           +12.918 % @e8_50k and +12.858 % @e8_1m on the verbatim 29-flag
    //           line. The RLS weights are Newton-scaled (w += P x (y-p)), not
    //           SGD-scaled, and gamma propagates that mis-scaling multiplicatively
    //           down all 23 cascade layers. Retained ONLY as the attribution
    //           control that proves the mechanism.
    //   false = DEFAULT. Layer-0 keeps training through the still-running SGD row
    //           while the RLS supplies the coded logit — a two-head arrangement,
    //           one head for the training signal and one for the readout. This is
    //           what the lab tree did (it predates -Dmixer-e2e), and it
    //           MEASURES -188 B = -0.09454 % @e8_1m on the 29-flag line, lossless,
    //           i.e. 114.7 % of the 20-flag reading raw / 88.1 % after dividing out
    //           the -Dadaptive-depth ladder factor.
    // Both are deterministic and decoder-symmetric; the decoder does the same
    // thing on the same bits.
    const rls_e2e_head = b.option(bool, "mixer1-rls-e2e-head", "Route the -Dmixer-e2e/-Dleafgrad gradient through the RLS head instead of the SGD row. Default FALSE: true is MEASURED at +12.86% and is an attribution control, not a shippable arm") orelse false;
    const rls_opts = b.addOptions();
    rls_opts.addOption(u32, "mixer1_rls", mixer1_rls);
    rls_opts.addOption(f64, "mixer1_rls_lambda", std.fmt.parseFloat(f64, rls_lambda_s) catch @panic("-Dmixer1-rls-lambda: not a parseable float"));
    rls_opts.addOption(f64, "mixer1_rls_p0", std.fmt.parseFloat(f64, rls_p0_s) catch @panic("-Dmixer1-rls-p0: not a parseable float"));
    rls_opts.addOption(f64, "mixer1_rls_hmin", std.fmt.parseFloat(f64, rls_hmin_s) catch @panic("-Dmixer1-rls-hmin: not a parseable float"));
    rls_opts.addOption(f64, "mixer1_rls_stepcap", std.fmt.parseFloat(f64, rls_stepcap_s) catch @panic("-Dmixer1-rls-stepcap: not a parseable float"));
    rls_opts.addOption(bool, "mixer1_rls_guard", rls_guard);
    rls_opts.addOption(bool, "mixer1_rls_e2e_head", rls_e2e_head);
    // -Dwordlbl: the word-identity distributed-representation expert (log-bilinear
    // + hierarchical softmax over the WRT codeword digit tree; src/models/wordlbl.zig).
    // Own options module ("wordlbl_opts") per the own-module rule: the dead knob costs S
    // EXACTLY ZERO. Default false = comptime-dead, bit-identical stock.
    // Evidence: σ −0.1714 % cumulative / plateau ~−0.19 % on the 29-flag
    // recipe ⇒ e9 −123…−136 KB.
    const wordlbl = b.option(
        bool,
        "wordlbl",
        "word-identity LBL/HSM expert on layer-0 pad slot 559 (default false = stock)",
    ) orelse false;
    const wordlbl_d = b.option(
        usize,
        "wordlbl-d",
        "wordlbl embedding width D (default 96 = 97.7% of D=192's value at half the RAM)",
    ) orelse 96;
    const wordlbl_etan_str = b.option(
        []const u8,
        "wordlbl-etan",
        "wordlbl node learning rate as a decimal string (default \"0.060\" = the swept interior optimum)",
    ) orelse "0.060";
    const wordlbl_etae_str = b.option(
        []const u8,
        "wordlbl-etae",
        "wordlbl embedding learning rate as a decimal string (default \"0.030\")",
    ) orelse "0.030";
    const wordlbl_etan: f32 = std.fmt.parseFloat(f32, wordlbl_etan_str) catch
        @panic("-Dwordlbl-etan: not a parseable float");
    const wordlbl_etae: f32 = std.fmt.parseFloat(f32, wordlbl_etae_str) catch
        @panic("-Dwordlbl-etae: not a parseable float");
    // -Dwordlbl-ctx / -Dwordlbl-lambda / -Dwordlbl-eta0: the wordlbl V2 context
    // aggregation + rate schedule (offline-validated config, scurve.c agg=2 +
    // SC_ETA0; + addendum).
    //   ctx=N   : context = previous N words from ONE shared embedding table,
    //             tied-decay weights lambda^s (s=0 = most recent). Default 0 =
    //             v1 behavior (3 untied tables over w1..w3) — comptime-dead.
    //   lambda  : the decay; REQUIRED (0<lambda<1) when ctx>0.
    //   eta0    : separate eta_N for codeword byte-0 trie nodes (nid 1..127);
    //             default = etan (v1 behavior, comptime-dead).
    // All in wordlbl_o (own module, the own-module rule: zero S-fee while dark).
    const wordlbl_ctx = b.option(
        usize,
        "wordlbl-ctx",
        "wordlbl v2 word-context length N, tied-decay shared-table aggregation (default 0 = v1's 3 untied tables)",
    ) orelse 0;
    const wordlbl_lambda_str = b.option(
        []const u8,
        "wordlbl-lambda",
        "wordlbl v2 tied-decay lambda as a decimal string (required with -Dwordlbl-ctx>0; offline optimum 0.85)",
    );
    const wordlbl_lambda: f32 = std.fmt.parseFloat(f32, wordlbl_lambda_str orelse "0.0") catch
        @panic("-Dwordlbl-lambda: not a parseable float");
    const wordlbl_eta0_str = b.option(
        []const u8,
        "wordlbl-eta0",
        "wordlbl per-depth eta for byte-0 trie nodes as a decimal string (default = -Dwordlbl-etan; offline optimum .015-.0225)",
    );
    const wordlbl_eta0: f32 = std.fmt.parseFloat(f32, wordlbl_eta0_str orelse wordlbl_etan_str) catch
        @panic("-Dwordlbl-eta0: not a parseable float");
    // -Dwordlbl-attn*: the V2-ATTN content-addressed induction memory — the
    // champion of the port plan, i.e.
    // attn.c agg=7 at the `f_champ192` finals config (sigma_all -0.4666 % vs the
    // tied-decay yardstick's -0.3886 % = +23.7 % relative, D=192 / 130 MiB).
    // Requires -Dwordlbl-ctx>0. Defaults reproduce v2 EXACTLY (comptime-dead),
    // and every option lives in wordlbl_o, so the dead-knob S fee is exactly
    // zero (the own-module rule: measured byte-identical stripped binary).
    const wordlbl_attn = b.option(
        bool,
        "wordlbl-attn",
        "wordlbl v2-attn: 1-head content-addressed induction memory over the word stream (default false = v2)",
    ) orelse false;
    const wordlbl_attn_n = b.option(
        usize,
        "wordlbl-attn-n",
        "wordlbl-attn memory slots N (default 128 = the offline interior optimum; 256 measured WORSE at the final config)",
    ) orelse 128;
    const wordlbl_attn_dk = b.option(
        usize,
        "wordlbl-attn-dk",
        "wordlbl-attn key/query width dk (default 64; 32 costs 0.0114 sigma-points)",
    ) orelse 64;
    const wordlbl_attn_topk = b.option(
        usize,
        "wordlbl-attn-topk",
        "wordlbl-attn slots that receive value/Wk gradients per token (default 16; 8 halves the backward MACs)",
    ) orelse 16;
    const wordlbl_attn_etaw_str = b.option(
        []const u8,
        "wordlbl-attn-etaw",
        "wordlbl-attn projection learning rate as a decimal string (default \"0.003\" = the swept optimum; .03 is 8x worse)",
    ) orelse "0.003";
    const wordlbl_attn_etaw: f32 = std.fmt.parseFloat(f32, wordlbl_attn_etaw_str) catch
        @panic("-Dwordlbl-attn-etaw: not a parseable float");
    const wordlbl_attn_head = b.option(
        bool,
        "wordlbl-attn-head",
        "wordlbl-attn candidate head: top-12 slots -> 2 extra pad inputs (557/558), zero new state (default true)",
    ) orelse true;
    const wordlbl_attn_htemp_str = b.option(
        []const u8,
        "wordlbl-attn-htemp",
        "wordlbl-attn candidate-head temperature, 1.0 or 2.0 only (default \"2.0\": the head wants to be sharper than the value path)",
    ) orelse "2.0";
    const wordlbl_attn_htemp: f32 = std.fmt.parseFloat(f32, wordlbl_attn_htemp_str) catch
        @panic("-Dwordlbl-attn-htemp: not a parseable float");
    const wordlbl_attn_stats = b.option(
        bool,
        "wordlbl-attn-stats",
        "wordlbl-attn: print gamma/brec/coverage diagnostics to stderr at deinit (default false; comptime-dead, no env var)",
    ) orelse false;
    const wordlbl_o = b.addOptions();
    wordlbl_o.addOption(bool, "wordlbl", wordlbl);
    wordlbl_o.addOption(usize, "wordlbl_d", wordlbl_d);
    wordlbl_o.addOption(f32, "wordlbl_etan", wordlbl_etan);
    wordlbl_o.addOption(f32, "wordlbl_etae", wordlbl_etae);
    wordlbl_o.addOption(usize, "wordlbl_ctx", wordlbl_ctx);
    wordlbl_o.addOption(f32, "wordlbl_lambda", wordlbl_lambda);
    wordlbl_o.addOption(bool, "wordlbl_lambda_set", wordlbl_lambda_str != null);
    wordlbl_o.addOption(f32, "wordlbl_eta0", wordlbl_eta0);
    wordlbl_o.addOption(bool, "wordlbl_attn", wordlbl_attn);
    wordlbl_o.addOption(usize, "wordlbl_attn_n", wordlbl_attn_n);
    wordlbl_o.addOption(usize, "wordlbl_attn_dk", wordlbl_attn_dk);
    wordlbl_o.addOption(usize, "wordlbl_attn_topk", wordlbl_attn_topk);
    wordlbl_o.addOption(f32, "wordlbl_attn_etaw", wordlbl_attn_etaw);
    wordlbl_o.addOption(bool, "wordlbl_attn_head", wordlbl_attn_head);
    wordlbl_o.addOption(f32, "wordlbl_attn_htemp", wordlbl_attn_htemp);
    wordlbl_o.addOption(bool, "wordlbl_attn_stats", wordlbl_attn_stats);
    // -Dleafgrad: CODING-LOSS LEAF TRAINING (offline finding
    //
    // Every leaf in the PAQ->lpaq->zpaq->cmix->cmix-lex->zmix lineage is trained
    // by a LOCAL rule: the Indirect StateMap moves toward the observed bit at a
    // fixed rate regardless of whether the ENSEMBLE was already right and of how
    // much the mixture RELIES on that model. This knob ADDS (it does not replace
    // — replacing measured far worse offline) the exact gradient of the coded
    // length w.r.t. the leaf's logit:
    //
    //     tau_m[state] -= Z * dL/dtau_m ,   dL/dtau_m = (sigma(z) - y) * Gamma_m
    //     Gamma_m = dz/dv_m = sum_{i=0..22} gamma_i * W0_i[m]
    //
    // where gamma_i = dz/dh_i is the SAME 253-MAC backward recursion -Dmixer-e2e
    // already runs (the layer-0 cascade + layer-1 head are affine in v), and the
    // mixer input v_m IS the leaf's logit (MixerInputLex.setInput stores
    // Logit(p)), so dv_m/dtau_m == 1 exactly.
    //
    // Consumers: src/models/indirect.zig + src/predictor_lex.zig ONLY, so per the
    // standing own-module hazard the knobs get their OWN options module
    // ("leafgrad_options") rather than growing the shared model_opts.
    //
    // -Dleafgrad=true -Dleafgrad-z=0 is the NULL CONTROL: all the plumbing runs
    // (gamma, Gamma, the per-model gradient hand-over) but the step is exactly zero,
    // so the archive must be BIT-IDENTICAL to stock. Default false = comptime-dead.
    const leafgrad = b.option(
        bool,
        "leafgrad",
        "Coding-loss leaf training on the 16 Indirect StateMaps (default false = comptime-dead, bit-identical stock)",
    ) orelse false;
    const leafgrad_z_str = b.option(
        []const u8,
        "leafgrad-z",
        "Leaf-gradient step size Z as a decimal string (default \"0\" = the in-engine null control)",
    ) orelse "0";
    const leafgrad_z: f32 = std.fmt.parseFloat(f32, leafgrad_z_str) catch
        @panic("-Dleafgrad-z: not a parseable float (want e.g. \"4.0\")");
    // -Dleafgrad-diag: LAB-ONLY. Accumulate the per-input-group |Gamma| census
    // (which slices of the 590 inputs the coded-length gradient actually has
    // leverage on) and print it at predictor deinit. No effect on any prediction.
    const leafgrad_diag = b.option(
        bool,
        "leafgrad-diag",
        "LAB: per-input-group |Gamma| census printed at deinit (no prediction effect)",
    ) orelse false;
    // -Dleafgrad-fxcm: THE REACH-RESPONSE ARM (reopen trigger).
    //
    // -Dleafgrad above reaches the 16 Indirect StateMaps, which the
    // diagnostic measured at only 2.42% of Sum|Gamma| @10 MB — and the screen on
    // that slice read NULL-to-ADVERSE. fxcm's 525 slots carry 94.13%, of which
    // the ~36% that have an adaptive cell at all are ContextMap3's `ts[256]`
    // (state-indexed, read through the SATURATING st1/st2 LUTs). This knob pushes
    // the same exact coding-loss gradient into those cells:
    //
    //     tau_cell -= Z * dL/dtau_cell
    //     dL/dtau_cell = [ SUM over the slots that read this cell of
    //                      g_final * Gamma_slot * dv/draw * draw/dp1 ]
    //                    * dp1/dtau_cell ,   dp1/dtau = 4096*p*(1-p), p = ts/2^32
    //
    // The three hazards named are handled explicitly, not assumed away:
    //   (a) LUT SATURATION zeroes the chain rule -> draw/dp1 is a WINDOWED finite
    //       difference on the real table (st1 saturates via clp at +-2047; a
    //       saturated window differences to exactly 0 and the cell is not moved).
    //   (b) MANY-TO-ONE: sm_set dedups states across contexts within one bit, so
    //       several slots share one `ts` cell. Contributions are SUMMED into a
    //       per-cell accumulator in pass 1 and applied ONCE per cell in pass 2.
    //   (c) The cell is STATE-indexed, not context-indexed: 256 cells per
    //       ContextMap3 instance, 41 instances, no cross-instance coupling
    //       (`ts` is a per-instance [256]u32; only st2/rcpr are shared and those
    //       are frozen).
    //
    // Independent of -Dleafgrad, so `-Dleafgrad-fxcm=true` alone is the clean
    // reach-response arm (CM3 cells only) against the banked Indirect-only point.
    // -Dleafgrad-z=0 remains the bit-identical null for BOTH arms.
    const leafgrad_fxcm = b.option(
        bool,
        "leafgrad-fxcm",
        "Coding-loss leaf training on fxcm ContextMap3's ts[256] cells (default false = comptime-dead)",
    ) orelse false;
    // -Dfxcm-slotmask (cm-wall-census, transformer-era route A): a 52-bit comptime mask over
    // fxcm_v26's ContextMap3/ContextMap4 INSTANCES in mix-call order. A set bit
    // DROPS that instance: its `set` skips the context hash write + the two
    // bucket prefetches, its `mix` emits the EXACT NEUTRAL 0 for each of its
    // slots (so the 494-slot layer-0 layout, and therefore every downstream mixer
    // weight index, is unchanged), its `prefetch_c0` returns immediately, and
    // its zero_alloc table is never touched (so its pages never become resident).
    // Default 0 => every `if (comptime MASK_ANY)` is comptime-dead and the build
    // is BIT-IDENTICAL. Own options module per the own-module rule (a shared
    // model_opts option costs ~40 B of decomp_bin, charged 2x in Form-1).
    // Bit order and slot counts:
    const fxcm_slotmask_s = b.option(
        []const u8,
        "fxcm-slotmask",
        "fxcm_v26 CM-instance drop mask, 52 bits, decimal or 0x hex (default \"0\" = nothing masked, bit-identical)",
    ) orelse "0";
    // -Dfxcm-slotdelete (cm-slot-deletion, transformer-era route A): the TRUE-DELETION arm of
    // the same mask. `-Dfxcm-slotmask` is a FLOOR on the wall a removal buys — a
    // masked instance still emits its neutral 0 into every downstream mixer, so the
    // 590-column layer-0 vector, fxcm's own 544-wide `in1` bank and its 560-wide
    // `slots` bank all keep full width (census phase-1 §6). With this ON, the SAME
    // masked instances emit NOTHING: their `set`/`sets`/`mix` return at once
    // (no cn/cxt_mask bookkeeping, no neutral pushes), and every consuming vector
    // shrinks by the comptime popcount-weighted slot total (207 for the B2 rung
    // 0xffff380211e00) — fxcm's 18 mx_a mixers 544 -> 337, its 4 mx_a2 mixers
    // 560 -> 353, and predictor_lex's layer-0 590 -> 383.
    // ⛔ NOT byte-identical to the mask, and cannot be: `mixer1.dotProductBias`
    // truncates PAIRWISE (`(t0*w0 + t1*w1) >> 8`) and `mixer_lex.dotV` reduces into
    // 4 independent FMA accumulators, so compacting the survivors changes both the
    // integer truncation grouping and the f32 accumulation grouping. Requires
    // -Dfxcm-slotmask != 0; ignored (and comptime-dead) at the default mask.
    const fxcm_slotdelete = b.option(
        bool,
        "fxcm-slotdelete",
        "TRUE-DELETE the -Dfxcm-slotmask instances: no emission, and every mixer input vector shrinks (default false = the mask's neutral-emit behaviour)",
    ) orelse false;
    const fxcmmask_opts = b.addOptions();
    fxcmmask_opts.addOption([]const u8, "fxcm_slotmask", fxcm_slotmask_s);
    fxcmmask_opts.addOption(bool, "fxcm_slotdelete", fxcm_slotdelete);
    const fxcmmask_mod = fxcmmask_opts.createModule();
    const leafgrad_opts = b.addOptions();
    leafgrad_opts.addOption(bool, "leafgrad", leafgrad);
    leafgrad_opts.addOption(bool, "leafgrad_fxcm", leafgrad_fxcm);
    leafgrad_opts.addOption(f32, "leafgrad_z", leafgrad_z);
    leafgrad_opts.addOption(bool, "leafgrad_diag", leafgrad_diag);
    // ONE module object, imported everywhere. `addOptions` mints a fresh module
    // per call, and cm3.zig (module `cmfast`) and predictor_lex.zig (module
    // `root`) are compiled together — two module objects over one root file is a
    // hard Zig error, so the module is created once here and shared.
    const leafgrad_mod = leafgrad_opts.createModule();

    // ---- -Dtransformer: the fx2-cmix-transformer 6M frozen transformer ------
    //
    // Replaces the online LSTM in the byte-mixer slot (`in[589]`) with the
    // frozen 6M transformer from `fx2-cmix-transformer` (entry #3, GPL-3,
    // operator-ruled fair game). Design:.
    // Implementation + decisions:.
    //
    // the own-module rule: EVERY knob below lives in `transformer_opts`,
    // its own options module, and NOT in the shared `model_opts` — an option
    // string in model_opts lands in `decomp_bin`, which is charged TWICE, at
    // ~40 B per dead option. In its own module the measured cost of a dead
    // option is EXACTLY ZERO (stripped `zmix_ship` byte-identical
    // with and without three new `-D` options).
    //
    // ⚠ The C++ sources and libc++ are attached ONLY when the flag is on
    // (`attachTransformer` below), so a stock build links exactly what it
    // linked before and the bit-identity gate is structural, not incidental.
    const transformer_on = b.option(bool, "transformer", "Replace the LSTM byte-mixer with the frozen fx2 6M transformer (default false)") orelse false;
    // ⛔ THIS DEFAULT MUST NEVER BE AN ABSOLUTE PATH. It used to be
    // an absolute build-machine path, i.e. a
    // dev-box path baked into `decomp_bin` — the same defect class that made
    // an earlier build unsubmittable, one stage EARLIER: `construct_ship.sh` self-compresses
    // english.dic and the order asset with `comp9 -c` and verifies with
    // `--extract-assets`, so a COMMITTEE REBUILD FROM SOURCE died with
    // `weights_io: cannot open <path>` on any machine without the ref tree.
    //
    // The blob is now SHIPPED WITH THE SOURCE at `assets/6m-q4-fp32.tfwc2` (2,930,652 B,
    // sha256 7f4db6c8…, magic FX2TFWC2 — upstream fx2-cmix-transformer's own artifact,
    // GPL-3, operator-ruled fair game ), exactly as english.dic and the
    // article-order asset are shipped in-tree. The default is CWD-RELATIVE, so it
    // resolves for any invocation made from the repo root and resolves to NOTHING —
    // loudly — anywhere else, instead of silently binding one developer's filesystem.
    //
    // It is also only the LAST resort. Resolution order in `predictor_lex.zig`:
    //   1. `Transformer.embedded_blob`  — the artifact's own tail (`-e`, no-arg
    //      self-extract, `--extract-assets`). A shipped op NEVER leaves this null.
    //   2. `Transformer.weights_path_override` — the explicit path argument
    //      `zmix_ship -c/-d <in> <out> <tfweights>` takes, which is what
    //      `construct_ship.sh` passes at asset-compression time (before any tail exists).
    //   3. this string — dev builds and the bench CLI only.
    const transformer_weights = b.option([]const u8, "transformer-weights", "Path to the .tfwc2 weights, CWD-relative (dev/bench fallback only; ship paths read the artifact's own tail and `-c` takes an explicit argument)") orelse
        "assets/6m-q4-fp32.tfwc2";
    // 0 = KVF32 (their ship default), 1 = KVI8 (bitwise-identical outputs,
    // 1.7 MB less L3 — worth an arm on a 12 MiB judge box, cpp_infer/attn.h).
    const transformer_attn = b.option(u32, "transformer-attn", "KV cache kind: 0 = KVF32 (default, their ship), 1 = KVI8") orelse 0;
    // ⛔ NEGATIVE CONTROL ONLY. Deliberately corrupts the canonical-205 token
    // map so the submission map guard can be PROVEN to abort — the only way to
    // demonstrate it, because every other gate this port owns is structurally
    // blind to a wrong map. 0 = off (comptime-dead, byte-identical output);
    // 1 = swap ranks 5/6 (trips the fast pre-check); 2 = swap ranks 172/173
    // (pre-check PASSES, only the separator count catches it). construct_ship.sh
    // refuses a nonzero value by name. See src/mixer/transformer.zig.
    const transformer_map_fault = b.option(u32, "transformer-map-fault", "NEGATIVE CONTROL ONLY: corrupt the canonical-205 token map (0 = off; 1 = rank 5/6 swap; 2 = rank 172/173 swap). Never ship a nonzero value.") orelse 0;
    if (transformer_map_fault != 0) {
        std.debug.print("!! -Dtransformer-map-fault={d}: THE CANONICAL-205 TOKEN MAP IS DELIBERATELY CORRUPT. This build is a NEGATIVE CONTROL and must never be shipped or measured for bytes.\n", .{transformer_map_fault});
    }
    // ★ THE LOAD-BEARING KNOB. See src/mixer/transformer.zig `priorFold`.
    //   0 = their verbatim contract: the slot is the bare transformer
    //       distribution, exactly as fx2-cmix-transformer ships it. The control.
    //   1 = the zmix PRL analogue: fold `max(2*P, 2^-clamp)^rho` onto the
    //       transformer's 205-way output and renormalize — the same fold
    //       `-Dlstm-prior-form=1` applies to the LSTM's softmax, which the S1
    //       screen measured to be WHY our slot beats theirs. Carrying it across
    //       is what stops the swap from trading our structural advantage away.
    const transformer_prior_form = b.option(u32, "transformer-prior-form", "Prior fold on the transformer output: 0 = none (their contract), 1 = PRL analogue") orelse 0;
    const transformer_prior_rho_str = b.option([]const u8, "transformer-prior-rho", "PRL fold exponent, dyadic in {0.25,0.5,0.75,1.0} (default 0.5 = ship LSTM value)") orelse "0.5";
    const transformer_prior_rho: f32 = std.fmt.parseFloat(f32, transformer_prior_rho_str) catch
        @panic("-Dtransformer-prior-rho: not a parseable float");
    const transformer_prior_clamp = b.option(u32, "transformer-prior-clamp", "PRL fold tail floor exponent C in 2^-C (default 13 = ship LSTM value)") orelse 13;
    // -Dtf-packed: build the dense weight arenas in the PACKED int4 format and
    // call the packed kernels (qmat.h section 3). The packed dots are
    // integer-exact vs the unpacked kernels (test_qmat / g7census-kernel-ladder
    // ), so archives are BYTE-IDENTICAL; the lever is wall + ~3 MB of
    // weight-arena RSS (unpacked streams ~6 MB/token; packed reads 0.5183x the
    // cycles on the same matmul stream). Selected via a C preprocessor define
    // (-DFX2_PACKED_DENSE=1) on the vendored TUs; default OFF compiles the TUs
    // with the exact flag line above, so the stock -Dtransformer build is
    // unchanged.
    const tf_packed = b.option(bool, "tf-packed", "Packed-int4 dense weight arenas + packed kernels in the transformer (default false; archives byte-identical)") orelse false;
    // -Dtransformer-lstm-dead: the ONLY ByteMixer in the ship binaries (form1,
    // ship) is the one `predictor_lex` builds, and it ALWAYS installs a
    // transformer — so `byteUpdate`'s LSTM fall-through is dead code that the
    // compiler cannot prove dead, because `predictor.zig`'s v21 ByteMixer does
    // reach it. This flag asserts the ship invariant and lets the whole LSTM
    // compute path (forwardNeuron/backwardNeuron/adam/BPTT) be eliminated.
    // MEASURED on the transformer-era arm-C prefix: packed 167,844 -> 163,732 B, i.e.
    // -4,112 B packed = -8,224 B of S (Form-1 charges decomp_bin TWICE).
    // ⚠ The LSTM is still ALLOCATED and Glorot-initialised: `initLstmWeightsGlibc`
    // draws from the shared glibc-rand stream, and skipping those draws would
    // shift every other model's random state and change every archive. This
    // flag removes only the never-executed COMPUTE path, so it is archive-
    // neutral by construction.
    // ⛔ `predictor.zig` (v21) refuses it at compile time — see the guard there.
    //    That is why the flag reaches ONLY the two ship roots, through
    //    `transformer_opts_ship` below: `zmix` and `test` compile predictor.zig
    //    and would refuse to build, taking `runner_lex` (same default step) with
    //    them. Merged into the route-A candidate.
    const transformer_lstm_dead = b.option(bool, "transformer-lstm-dead", "Assert the ship invariant that every ByteMixer carries a transformer, eliminating the dead LSTM compute path (default false)") orelse false;
    // A flag that silently does nothing is a latent misbooking: without
    // -Dtransformer the whole branch it controls is comptime-dead, so a recipe
    // carrying it alone would measure "no effect" and be written off.
    if (transformer_lstm_dead and !transformer_on) {
        std.debug.print("FATAL: -Dtransformer-lstm-dead requires -Dtransformer=true (without it the flag is a silent no-op).\n", .{});
        std.process.exit(1);
    }
    // -Dtf-weights-v3: teach the weight loader the FX2TFWC3 container
    // (the FX2TFWC3 writer). Same tensors as FX2TFWC2 except
    // that the 88 raw trained f32 tensors are carried as bfloat16 (+0.00004
    // nats/token); the blob is 2,840,417 B
    // against v2's 2,930,652, and Form-1 charges it TWICE.
    // Selected via a C preprocessor define (-DFX2_WEIGHTS_V3=1) on the ONE
    // load-time TU that needs it; with the flag off that TU's object is
    // section-for-section byte-identical to the base's, so the stock build is
    // structurally unchanged and a v3 blob is rejected loudly ("bad magic").
    // v1/v2 stay loadable either way -- the format is chosen by the magic.
    const tf_weights_v3 = b.option(bool, "tf-weights-v3", "Accept the FX2TFWC3 weight container (default false; v1/v2 always load)") orelse false;
    const transformer_opts = b.addOptions();
    transformer_opts.addOption(bool, "transformer", transformer_on);
    transformer_opts.addOption([]const u8, "transformer_weights", transformer_weights);
    transformer_opts.addOption(u32, "transformer_attn", transformer_attn);
    transformer_opts.addOption(u32, "transformer_map_fault", transformer_map_fault);
    transformer_opts.addOption(u32, "transformer_prior_form", transformer_prior_form);
    transformer_opts.addOption(f32, "transformer_prior_rho", transformer_prior_rho);
    transformer_opts.addOption(u32, "transformer_prior_clamp", transformer_prior_clamp);
    transformer_opts.addOption(bool, "tf_packed", tf_packed);
    // ⛔ HARDCODED `false` HERE, ON PURPOSE. This module is imported by EVERY
    // root, including `zmix` (src/main.zig -> predictor.zig) and `test`
    // (src/tests.zig -> predictor.zig), and predictor.zig REFUSES the flag at
    // compile time — correctly, because its v21 ByteMixer carries no
    // transformer and would reach the eliminated path. With one shared module
    // the flag therefore makes the DEFAULT `zig build` step fail to compile,
    // and the default step is what produces `runner_lex` — so it would break
    // verify_change.sh's engaged gate, every bench leg, and the construct
    // driver's own runner_lex build. A recipe flag that cannot survive
    // `zig build` is a flag a judge rebuild cannot carry, which is the whole
    // reason SHIP_RECIPE.env exists.
    // ⇒ the two SHIP roots get their own options module below, differing in
    // exactly this one bool. Nothing else changes, so no non-ship binary moves.
    transformer_opts.addOption(bool, "transformer_lstm_dead", false);
    transformer_opts.addOption(bool, "tf_weights_v3", tf_weights_v3);
    // ONE module object, shared by every root that imports it (same reason as
    // leafgrad_mod above).
    const transformer_mod = transformer_opts.createModule();

    // ---- SHIP-ROOT variant of the same options (`ship`, `form1`) -----------
    // Identical to `transformer_opts` except `transformer_lstm_dead`. Neither
    // ship root compiles `predictor.zig`: both reach the byte mixer through
    // `predictor_lex`, whose ByteMixer ALWAYS installs a transformer (it panics
    // otherwise) — precisely the invariant the flag asserts. The option COUNT is
    // the same, so the own-module rule's per-dead-option prefix fee does not move; the two
    // modules differ only in one bool's value.
    const transformer_opts_ship = b.addOptions();
    transformer_opts_ship.addOption(bool, "transformer", transformer_on);
    transformer_opts_ship.addOption([]const u8, "transformer_weights", transformer_weights);
    transformer_opts_ship.addOption(u32, "transformer_attn", transformer_attn);
    transformer_opts_ship.addOption(u32, "transformer_map_fault", transformer_map_fault);
    transformer_opts_ship.addOption(u32, "transformer_prior_form", transformer_prior_form);
    transformer_opts_ship.addOption(f32, "transformer_prior_rho", transformer_prior_rho);
    transformer_opts_ship.addOption(u32, "transformer_prior_clamp", transformer_prior_clamp);
    transformer_opts_ship.addOption(bool, "tf_packed", tf_packed);
    transformer_opts_ship.addOption(bool, "transformer_lstm_dead", transformer_lstm_dead);
    transformer_opts_ship.addOption(bool, "tf_weights_v3", tf_weights_v3);
    const transformer_mod_ship = transformer_opts_ship.createModule();

    // The nine vendored production translation units plus our extern "C" shim.
    // Their exact validated flag set (cpp_infer/Makefile:8) minus -march (the
    // Zig target already pins x86_64_v3) plus -mrecip=none, which is MANDATORY:
    // RCPPS/RSQRTPS reciprocal estimates come from a vendor-specific table and
    // are what broke their July 2026 submission on the committee's Intel laptop
    // (build_and_construct_comp.sh:70-78). tools/check_no_recip_estimate.sh is
    // the build-time gate.
    // -Dtf-load-opt: how to compile the vendored transformer TUs that are
    // reachable ONLY from model construction / weight loading, never from
    // t.step. Measured on the transformer-era ship candidate: those TUs are the LARGEST
    // C++ block in the prefix — `OptModel::load` alone is 41,816 B of .text
    // after LTO, 12 % more than the hot `TransformerOptImpl::step` (37,245) —
    // and Form-1 charges `decomp_bin` TWICE.
    //   off (default) : the shipped split — arena_build.cpp and weights_io.cpp
    //                   at -O3 with the kernels.
    //   os            : move those two onto the -Os list. packed 168,924 ->
    //                   162,948 = -5,976 packed = -11,952 B of S.
    //   oz            : as `os`, and compile the WHOLE load-only set
    //                   (arena_build, weights_io, weights_io_compressed,
    //                   cxx_local) at -Oz. packed 161,716 = -7,208 packed =
    //                   -14,416 B of S.
    // ⚠ This changes FP codegen (vectorisation width, FMA contraction) inside
    // the dequantising arena builder and — under `oz` — inside the v3 weight
    // DECODER, so it is adoptable only on measured archive bit-identity. It
    // cannot touch the inference hot path: no symbol in these TUs is reachable
    // from step, verified by symbol census.
    // It is a BUILD-FLAG option consumed by build.zig alone; it reaches no
    // options module, so with the default it costs S exactly zero (verified:
    // packed prefix byte-identical, sha 1e62611d...).
    const tf_load_opt = b.option([]const u8, "tf-load-opt", "optimize the load-only vendored transformer TUs for size: off|os|oz (default off)") orelse "off";
    const tf_load_os = std.mem.eql(u8, tf_load_opt, "os") or std.mem.eql(u8, tf_load_opt, "oz");
    const tf_load_oz = std.mem.eql(u8, tf_load_opt, "oz");
    if (!tf_load_os and !std.mem.eql(u8, tf_load_opt, "off")) {
        std.debug.panic("-Dtf-load-opt must be one of off|os|oz (got '{s}')", .{tf_load_opt});
    }
    // The seven TUs that ARE on the inference hot path. Never size-optimized.
    const TF_SRC_HOT = [_][]const u8{
        "third_party/fx2_transformer/attn.cpp",
        "third_party/fx2_transformer/glue.cpp",
        "third_party/fx2_transformer/kda.cpp",
        "third_party/fx2_transformer/model_opt.cpp",
        "third_party/fx2_transformer/qmat_dense.cpp",
        "third_party/fx2_transformer/qmat_sparse.cpp",
        "third_party/fx2_transformer/fx2_shim.cpp",
    };
    // arena_build.cpp = HugeBuf::alloc + OptModel::load; weights_io.cpp =
    // WeightsFile::load/get. 100 % load-only, both of them.
    const TF_SRC_LOADONLY2 = [_][]const u8{
        "third_party/fx2_transformer/arena_build.cpp",
        "third_party/fx2_transformer/weights_io.cpp",
    };
    // The OFF arm must pass the compiler the ORIGINAL list in the ORIGINAL
    // order: object order moves addresses, which moves the packed size (-8 B
    // measured when the two load-only TUs were merely appended instead of
    // left in place). Spelled out literally so `off` is provably free.
    const TF_SRC_O3 = [_][]const u8{
        "third_party/fx2_transformer/arena_build.cpp",
        "third_party/fx2_transformer/attn.cpp",
        "third_party/fx2_transformer/glue.cpp",
        "third_party/fx2_transformer/kda.cpp",
        "third_party/fx2_transformer/model_opt.cpp",
        "third_party/fx2_transformer/qmat_dense.cpp",
        "third_party/fx2_transformer/qmat_sparse.cpp",
        "third_party/fx2_transformer/weights_io.cpp",
        "third_party/fx2_transformer/fx2_shim.cpp",
    };
    // Load-time only: -Os, 4,889 B of text vs 20.7 KB at -O3 for zero benefit
    // (their COMPRESSION.md:28-31 measured it). decomp_bin is charged 2x.
    const TF_SRC_OS_BASE = [_][]const u8{
        "third_party/fx2_transformer/weights_io_compressed.cpp",
        // OURS, not vendored. Supplies every libc++ symbol the nine objects
        // above need, LOCALLY and exception-free, so that no member of
        // libc++.a / libc++abi.a / libunwind.a is ever extracted. Measured
        // worth: see the file header (the Itanium demangler alone is 69,581 B
        // of a prefix that Form-1 charges TWICE). -Os: it is load-time-only
        // glue, exactly like weights_io_compressed.cpp above.
        "src/cxx_local.cpp",
    };
    const TF_SRC_OS_ALL = TF_SRC_OS_BASE ++ TF_SRC_LOADONLY2;
    const TF_SRC_O3_SEL: []const []const u8 = if (tf_load_os) &TF_SRC_HOT else &TF_SRC_O3;
    const TF_SRC_OS_SEL: []const []const u8 = if (tf_load_os) &TF_SRC_OS_ALL else &TF_SRC_OS_BASE;
    const TF_FLAGS_O3 = [_][]const u8{ "-O3", "-std=c++17", "-fno-math-errno", "-mrecip=none", "-fno-exceptions", "-fno-rtti", "-ffunction-sections", "-fdata-sections" };
    const TF_SIZE_O = if (tf_load_oz) "-Oz" else "-Os";
    const TF_FLAGS_OS = [_][]const u8{ TF_SIZE_O, "-std=c++17", "-fno-math-errno", "-mrecip=none", "-fno-exceptions", "-fno-rtti", "-ffunction-sections", "-fdata-sections" };
    // -Dtf-packed selects the flag line; with it OFF (default) the arrays above
    // are passed verbatim, so the stock build's compile commands are unchanged.
    const TF_FLAGS_O3_PACKED = TF_FLAGS_O3 ++ [_][]const u8{"-DFX2_PACKED_DENSE=1"};
    const TF_FLAGS_OS_PACKED = TF_FLAGS_OS ++ [_][]const u8{"-DFX2_PACKED_DENSE=1"};
    // -Dtf-weights-v3 is consumed ONLY by weights_io_compressed.cpp, so the nine
    // -O3 compile commands are the same in every combination. It rides on the
    // whole -Os list (which since the decomp_bin shrink also carries
    // src/cxx_local.cpp); that TU never names the macro, so the define is inert
    // there.
    const TF_FLAGS_OS_V3 = TF_FLAGS_OS ++ [_][]const u8{"-DFX2_WEIGHTS_V3=1"};
    const TF_FLAGS_OS_PACKED_V3 = TF_FLAGS_OS_PACKED ++ [_][]const u8{"-DFX2_WEIGHTS_V3=1"};
    const tf_fo3: []const []const u8 = if (tf_packed) &TF_FLAGS_O3_PACKED else &TF_FLAGS_O3;
    const tf_fos: []const []const u8 = if (tf_packed)
        (if (tf_weights_v3) &TF_FLAGS_OS_PACKED_V3 else &TF_FLAGS_OS_PACKED)
    else
        (if (tf_weights_v3) &TF_FLAGS_OS_V3 else &TF_FLAGS_OS);
    const attachTransformer = struct {
        fn f(bb: *std.Build, m: *std.Build.Module, on: bool, tmod: *std.Build.Module, o3: []const []const u8, os_: []const []const u8, fo3: []const []const u8, fos: []const []const u8) void {
            m.addImport("transformer_options", tmod);
            if (!on) return;
            m.addIncludePath(bb.path("third_party/fx2_transformer"));
            m.addCSourceFiles(.{ .files = o3, .flags = fo3, .language = .cpp });
            m.addCSourceFiles(.{ .files = os_, .flags = fos, .language = .cpp });
            m.link_libcpp = true;
        }
    }.f;

    // ------------------------------------------------------------------
    // -Dship-evict-bytes: the PPMd valve cadence THE SHIP ROOTS USE.
    //   ⛔ THIS EXISTS BECAUSE THE BENCH AND SHIP PATHS DIVERGED SILENTLY.
    //   `-Dppmd-evict-bytes` (default 5000) is the build default that
    //   `src/models/ppmd.zig` initialises `mmap_remap_interval_bytes` from, and
    //   it is what the `-c` BENCH CLI (`src/runner_lex.zig`) actually runs at —
    //   because runner_lex never calls `setEvictInterval`. The SHIP roots
    //   (`src/ship.zig`, `src/form1.zig`) overrode it with a HARDCODED 1_000 at
    //   six call sites, so every bench wall/RSS number was taken at 5,000 while
    //   the artifact ran at 1,000 — a divergence that turns a frozen wall bar
    //   into an optimistic one.
    //   DEFAULT 1000 = the six hardcoded values ⇒ BIT-IDENTICAL and
    //   behaviour-identical at default; the option only makes the shipped
    //   cadence expressible in the recipe instead of frozen in the source.
    //   Own options module ("wallpareto_options"), NEVER `model_opts`:
    //  the own-module rule — Form-1 charges `decomp_bin` TWICE and a dead option in
    //   the shared module costs ~40 B of it; an own-module option is measured at
    //   EXACTLY 0 S. Imported by the two roots that call setEvictInterval only.
    //  Measured on the transformer-era candidate:
    //   1500 costs decomp_bin +8 ⇒ ΔS +16 B, and buys −13.39 % minor faults /
    //   −20.3 % sys ⇒ ≥ −0.50 % of wall, at +133 MiB of file-resident RSS @e9.
    //   ⛔ The sibling `-Dmixer-flat-slab` from that same branch is KILLED
    //   (+200 B of S, cycles +0.126 %) and is deliberately NOT carried here.
    const ship_evict_bytes = b.option(
        u64,
        "ship-evict-bytes",
        "PPMd valve eviction cadence the SHIP roots (ship.zig/form1.zig) set, in processed bytes (default 1000 = the value they hardcoded; looser = less wall, more file-resident RSS)",
    ) orelse 1000;
    const wallpareto_opts = b.addOptions();
    wallpareto_opts.addOption(u64, "ship_evict_bytes", ship_evict_bytes);
    const wallpareto_mod = wallpareto_opts.createModule();

    const model_opts = b.addOptions();
    model_opts.addOption(bool, "derivdict", derivdict_on);
    model_opts.addOption(bool, "s7order", s7order_on);
    model_opts.addOption(u32, "fxcm_final_blend", fxcm_final_blend);
    model_opts.addOption(f32, "lstm_adam_beta1", lstm_adam_beta1);
    model_opts.addOption(i32, "sscm_base_rate", sscm_base_rate);
    model_opts.addOption(f32, "mixer_decay_floor", mixer_decay_floor);
    model_opts.addOption(bool, "coldram", coldram_on);
    model_opts.addOption(bool, "coldram_pool_only", coldram_pool_only);
    model_opts.addOption([]const u8, "free_prior", free_prior);
    model_opts.addOption(bool, "free_prior_hash", free_prior_hash);
    model_opts.addOption(bool, "free_prior_b", free_prior_b);
    model_opts.addOption(bool, "free_prior_tap_lossy", free_prior_tap_lossy);
    model_opts.addOption(u32, "free_prior_topk", free_prior_topk);
    model_opts.addOption(bool, "lstm_bwvec", lstm_bwvec);
    // ⚠ LAB-ONLY census knob. It sits in the SHARED model_opts because runner_lex
    // (root module) is the consumer; per the own-module rule that costs prefix bytes, so this
    // must NOT be merged to trunk as-is — it exists to answer one question.
    // -Dfxcm-zero-neutral: 0 stock / 1 snap the raw-0 slot to the exact neutral /
    //   2 normalise RawPredictionProbability by 4096 instead of 4095.
    model_opts.addOption(u32, "fxcm_zero_neutral", b.option(u32, "fxcm-zero-neutral", "fxcm neutral off-by-one: 0 stock, 1 snap raw-0, 2 /4096 (default 0)") orelse 0);
    // -Dmixer-blockskip: skip all-zero aligned input blocks in the layer-0 dot/axpy.
    //   BIT-IDENTICAL by construction; a pure instruction/wall lever. Only useful with
    //   -Dfxcm-zero-neutral=1, which is what makes the inert slots exactly zero.
    const mixer_blockskip = b.option(bool, "mixer-blockskip", "skip all-zero mixer input blocks (default false)") orelse false;
    model_opts.addOption(bool, "mixer_blockskip", mixer_blockskip);
    // -Dind-delta-div: divide the outer Indirect decoder's delta (200/300/400) by N,
    //   i.e. make it N x faster. Predicted mis-set by the starvation sweep (45,000-92,000
    //   time constants at e9). Default 1 = shipped, bit-identical.
    model_opts.addOption(u32, "ind_delta_div", b.option(u32, "ind-delta-div", "Indirect decoder rate divisor (default 1 = shipped)") orelse 1);
    model_opts.addOption(bool, "coder_diag", coder_diag);
    model_opts.addOption(bool, "mix_sparse_diag", b.option(bool, "mix-sparse-diag", "layer-0 input zero census (default false)") orelse false);
    model_opts.addOption(i32, "sse_wr6", b.option(i32, "sse-wr6", "SSE s6 update rate (default 106 = shipped)") orelse 106);
    model_opts.addOption(i32, "sse_wr7", b.option(i32, "sse-wr7", "SSE s7 update rate (default 127 = shipped)") orelse 127);
    model_opts.addOption(i32, "sse_xw1", b.option(i32, "sse-xw1", "SSE x1 mixer rate (default 6202 = shipped)") orelse 6202);
    model_opts.addOption(i32, "sse_xw2", b.option(i32, "sse-xw2", "SSE x2 mixer rate (default 8320 = shipped)") orelse 8320);
    model_opts.addOption(u5, "sse_ffl6", @intCast(sse_ffl6));
    model_opts.addOption(u5, "sse_ffl7", @intCast(sse_ffl7));
    model_opts.addOption(bool, "idr", use_idr);
    model_opts.addOption(u32, "ppm_order", ppm_order);
    model_opts.addOption(bool, "vendored_libm", vendored_libm);
    model_opts.addOption(bool, "ppmd_mmap", ppmd_mmap);
    model_opts.addOption(u32, "lstm_cells", lstm_cells);
    model_opts.addOption(u32, "lstm_layers", lstm_layers);
    model_opts.addOption(usize, "lstm_aux_sparse", lstm_aux_sparse);
    model_opts.addOption(usize, "lstm_head_rank", lstm_head_rank);
    model_opts.addOption(f32, "lstm_head_lr_scale", lstm_head_lr_scale);
    model_opts.addOption(bool, "lstm_aux_sparse_dense", lstm_aux_sparse_dense);
    model_opts.addOption(f32, "lstm_lr", lstm_lr);
    model_opts.addOption(f32, "mix_lr_scale", mix_lr_scale);
    model_opts.addOption(f32, "mxa_uperr_scale", mxa_uperr_scale);
    model_opts.addOption(f32, "lstm_prior_rho", lstm_prior_rho);
    model_opts.addOption(u32, "lstm_prior_form", lstm_prior_form);
    model_opts.addOption(u32, "lstm_prior_clamp", lstm_prior_clamp);
    model_opts.addOption(bool, "auxparam_instr", auxparam_instr);
    model_opts.addOption(u32, "ppmd_arena_mb", ppmd_arena_mb);
    model_opts.addOption(bool, "cm3_diag", cm3_diag);
    model_opts.addOption(u32, "cmc2_grow", cmc2_grow);
    model_opts.addOption(bool, "cmc2_hash64", cmc2_hash64);
    model_opts.addOption(u32, "cmc2_spill", cmc2_spill);
    model_opts.addOption(u32, "cmc2_grow_hot", cmc2_grow_hot);
    model_opts.addOption(u64, "ppmd_evict_bytes", ppmd_evict_bytes);
    model_opts.addOption(u32, "cmc2_hot2", cmc2_hot2);
    model_opts.addOption(i32, "sta6_mdc", sta6_mdc);
    model_opts.addOption(bool, "m0lr_comp8", m0lr_comp8);
    model_opts.addOption(u64, "lstm_update_limit", lstm_update_limit);
    model_opts.addOption(bool, "r1v2", r1v2);
    model_opts.addOption(bool, "hero", hero);
    model_opts.addOption(f64, "hero_eta", hero_eta);
    model_opts.addOption(bool, "fxcm_auxctx_fix", fxcm_auxctx_fix);
    // -Dgate-dump=<path prefix>: LAB-ONLY instrument for the `mixer-gating-contexts`
    //  family.
    //   Dumps, per coded bit-step, a 32-column slim view of the 590 mixer inputs, the
    //   23 layer-0 gating context values (u32 — EXACTLY the row keys mixer_lex uses),
    //   the pre-SSE mixer-1 logit z, the final coded probability and the bit.
    //   Empty (default) compiles the whole facility away ⇒ stock build byte-identical.
    //   NOT a ship knob: never appears on any recipe line.
    const ind_tag = b.option(u32, "ind-tag", "shared map collision detection: 0 off, 1 split halves, 2 INTERLEAVED [state,tag] (one cache line = stock traffic)") orelse 0;
    model_opts.addOption(u32, "ind_tag", ind_tag);
    const ind_tag_bits = b.option(u32, "ind-tag-bits", "tag width in bits (8 = full byte); prices the collision-rate/overhead exchange") orelse 8;
    model_opts.addOption(u32, "ind_tag_bits", ind_tag_bits);
    const ind_admit_depth = b.option(u32, "ind-admit-depth", "ind-admit=3: state-graph depth at/above which a state is protected from a first-visit write") orelse 4;
    model_opts.addOption(u32, "ind_admit_depth", ind_admit_depth);
    const shared_map_div = b.option(u32, "shared-map-div", "shrink the shared bit-history map by this factor (1 = stock; the LOAD-FACTOR knob)") orelse 1;
    model_opts.addOption(u32, "shared_map_div", shared_map_div);
    const ind_admit = b.option(u32, "ind-admit", "shared-map admission: on a first-visit context write only the first K tree levels (0 = stock, byte-identical)") orelse 0;
    model_opts.addOption(u32, "ind_admit", ind_admit);
    const ind_census = b.option(bool, "ind-census", "LAB census of the shared indirect map occupancy (comptime-dead, default false)") orelse false;
    model_opts.addOption(bool, "ind_census", ind_census);
    const gate_dump = b.option([]const u8, "gate-dump", "LAB gating-context dump path prefix (empty = off, byte-identical)") orelse "";
    model_opts.addOption([]const u8, "gate_dump", gate_dump);
    const gate_dump_stride = b.option(u64, "gate-dump-stride", "LAB gating-context dump stride (default 1 = every coded bit)") orelse 1;
    model_opts.addOption(u64, "gate_dump_stride", gate_dump_stride);
    const gate_dump_max = b.option(u64, "gate-dump-max", "LAB gating-context dump record cap (0 = unlimited)") orelse 0;
    model_opts.addOption(u64, "gate_dump_max", gate_dump_max);

    // -Dfieldcodec-tsswap lives in its OWN options module, NOT in model_opts
    // (the -Dform1-malloc pattern, the own-module rule): model_opts is shared across every
    // target, and adding an option there moves the packed prefix — charged ×2
    // under Form-1 — even when the option is never engaged. The module is
    // attached to exactly the targets that compile runner_lex.zig (the
    // consuming pipeline): runner_lex, runner_lex_decode, zmix_ship,
    // zmix_form1 and the test builds. See src/prepr/fieldcodec.zig for what
    // the swap is and the measurement it deploys.
    const fieldcodec_tsswap = b.option(
        bool,
        "fieldcodec-tsswap",
        "fieldcodec T1 null-swap: emit each revid record's timestamp line BEFORE its revid line in the coded temp (frozen rule, no side data; default false = stock bit-identical)",
    ) orelse false;
    // -Dfieldcodec-cmtmove — the GATED comment relocation (metadata-codecs
    // family): sub-768 B page records emit their
    // <comment> line AFTER the text instead of before it. Same own-module pattern;
    // see src/prepr/cmtmove.zig for the grammar, the gate, and the measurement.
    const fieldcodec_cmtmove = b.option(
        bool,
        "fieldcodec-cmtmove",
        "gated comment relocation: move each sub-768 B page record's comment line to after its text (frozen page-size gate, no side data; default false = stock bit-identical)",
    ) orelse false;
    const fieldcodec_opts = b.addOptions();
    fieldcodec_opts.addOption(bool, "ts_swap", fieldcodec_tsswap);
    fieldcodec_opts.addOption(bool, "cmt_move", fieldcodec_cmtmove);

    // -Dram-census — LAB-ONLY, output-neutral census of every large allocation
    // (ram-efficient-frontier). Accumulates bytes per
    // `@returnAddress` at `zero_alloc.alloc`'s big path and at the few growth
    // sites that bypass it, and dumps the table to stderr at end of run. The
    // engine's coded output is untouched: nothing reads the counters.
    //
    // OWN options module (`ramcensus_options`), never `model_opts` — the own-module rule: a
    // shared-`model_opts` entry lands ~40 B in `decomp_bin` even when dead, and
    // Form-1 charges `decomp_bin` TWICE. Own-module knobs cost S exactly zero
    // (measured stripped `zmix_ship` byte-identical with and
    // without three new own-module options).
    const ram_census = b.option(
        bool,
        "ram-census",
        "LAB: dump a per-call-site census of large allocations to stderr at end of run (default false = comptime-dead, bit-identical)",
    ) orelse false;
    const ramcensus_opts = b.addOptions();
    ramcensus_opts.addOption(bool, "ram_census", ram_census);
    // ram_census must be a MODULE, not a bare file import: `zero_alloc.zig`,
    // `lstm.zig` and `mixer_lex.zig` live in three different modules
    // (`cmcold`, `lstmfast`, root), and a bare cross-module file import pulls
    // the importing file into BOTH module graphs ("file exists in modules ...").
    const ramcensus_mod = b.createModule(.{
        .root_source_file = b.path("src/ram_census.zig"),
        .target = target,
        .optimize = .ReleaseFast,
    });
    ramcensus_mod.addImport("ramcensus_options", ramcensus_opts.createModule());

    // -Dslot-loss — LAB-ONLY, output-neutral measurement of the byte-mixer
    // LSTM's own slot loss (S1 of `shipped-transformer`). Same
    // own-options-module discipline as -Dram-census above (the own-module rule): default false
    // => comptime-dead => stock archives bit-identical, S cost exactly zero.
    const slot_loss = b.option(
        bool,
        "slot-loss",
        "LAB: stream the byte-mixer LSTM's own prequential slot loss (and its PPMd input's) to stderr (default false = comptime-dead, bit-identical)",
    ) orelse false;
    const slotloss_opts = b.addOptions();
    slotloss_opts.addOption(bool, "slot_loss", slot_loss);
    const slotloss_mod = b.createModule(.{
        .root_source_file = b.path("src/slot_loss.zig"),
        .target = target,
        .optimize = .ReleaseFast,
    });
    slotloss_mod.addImport("slotloss_options", slotloss_opts.createModule());
    // strict_fp — IEEE-exact div/sqrt helpers (P0 reciprocal
    // determinism). Same reason as ram_census above: `predictor_lex.zig` (root)
    // and `lstm_layer.zig` (`lstmfast`) both need it, so it must be a MODULE or
    // the bare file import drags lstm_layer into two module graphs.
    // Carries no build options => costs S exactly zero (the own-module rule).
    const strictfp_mod = b.createModule(.{
        .root_source_file = b.path("src/strict_fp.zig"),
        .target = target,
        .optimize = .ReleaseFast,
    });

    // -Dphda9-nullpred — the CP-P1 "null language predicate": phda9's henttail
    // interwiki/lang block predicate NEVER fires, so the per-revision interwiki
    // runs stay INLINE instead of being segregated into the trailing lang blob.
    // Measured standalone -16,092 B on a 100 MB temp window, and
    // super-additive under mixer-e2e-norm.
    //
    // Its OWN options module (`prepr_options`), NOT model_opts — the own-module rule: model_opts
    // is shared across every target and every entry lands plumbing in decomp_bin,
    // which Form-1 charges TWICE. Attached to exactly the targets that compile
    // the prepr pipeline (runner_lex, runner_lex_decode, zmix_ship, zmix_form1,
    // both test builds) — the same set as fieldcodec_options above.
    //
    // ⚠ It re-mints prep.temp, so the fixed r1_reorder offsets (which are
    // properties of the SHIPPED post-WRT stream layout) move with it. Both
    // src/prepr/phda9.zig and src/prepr/r1_reorder.zig read this module.
    const phda9_nullpred = b.option(
        bool,
        "phda9-nullpred",
        "phda9 NULL language predicate: never segregate interwiki runs into the lang blob (re-mints prep.temp; default false = stock bit-identical)",
    ) orelse false;

    // -Dphda9-nullqlg / -Dphda9-nullh2 / -Dphda9-nullh5 — nullpred SIBLINGS
    // (cpp1-nullsib): decoder-free "never fire" forms of phda9's remaining
    // entity rewrites. Encode-side ONLY — hent1/hent3/hent6 (decode) are
    // shape-driven and correctly invert the un-compacted stream, PROVEN by the
    // aug-18 collision census (the &amp;-doubling + escape-3 path is NOT
    // nullable — 16 in-text collision sites — and stays live under all three).
    //   nullqlg: hent never compacts &quot;/&lt;/&gt; (⇒ removeamp goes inert
    //            by construction: the post-hent text keeps zero literal "<>).
    //   nullh2:  hent2 never compacts second-level &&quot;/&&nbsp;/… forms.
    //   nullh5:  hent5 never converts &&#NNN; (N>255) to ESC5+UTF8.
    // ⚠ All three are LENGTH-CHANGING (unlike nullpred's pure permutation):
    // engaged full-chain -e/--prepare/--roundtrip REFUSES at the r1 length
    // gate until per-knob r1 offsets are measured (owed at adoption, not at
    // screen time — screens run pre-r1 via --ready4/--wrt/--codetemp).
    const phda9_nullqlg = b.option(
        bool,
        "phda9-nullqlg",
        "phda9 sibling null: hent never compacts &quot;/&lt;/&gt; (decoder-free; re-mints prep.temp; default false = stock bit-identical)",
    ) orelse false;
    const phda9_nullh2 = b.option(
        bool,
        "phda9-nullh2",
        "phda9 sibling null: hent2 never compacts second-level &&entity; forms (decoder-free; re-mints prep.temp; default false = stock bit-identical)",
    ) orelse false;
    const phda9_nullh5 = b.option(
        bool,
        "phda9-nullh5",
        "phda9 sibling null: hent5 never converts &&#NNN; to ESC5+UTF8 (decoder-free; re-mints prep.temp; default false = stock bit-identical)",
    ) orelse false;
    // -Dwrt-earn=<stock|nosubst|nocase|nowrt> — LAB-ONLY frontend PESSIMUM
    // CONTROLS. Each value
    // disables ONE shipped WRT mechanism and holds everything else (dictionary
    // still loaded, fxcm dict consumer live, Pretrain live, framing live), so
    // Δ(arm − stock) is what that mechanism EARNS in coded bytes — the
    // denominator the frontend has never had. NOT a ship knob: never put one of
    // these on a recipe line.
    //   stock   : shipped behaviour (default; bit-identical).
    //   nosubst : Dictionary::EncodeWord never substitutes (neither the
    //             whole-word byte_map hit nor EncodeSubstring's affix pass);
    //             words are emitted as letters. Case markers, escaping and the
    //             final involutive permutation are RETAINED, and the mode byte
    //             is forced to 7 so encode_text's `size > len-50` raw fallback
    //             cannot silently convert this into `nowrt`. Decoder-invertible
    //             by the STOCK decoder (an unsubstituted word is exactly the
    //             stock OOV path).
    //   nocase  : the tokenizer keeps the ORIGINAL case (no fold to lowercase)
    //             and EncodeWord emits no kCapitalized/kUppercase/kEndUpper
    //             marker. Word boundaries are UNCHANGED (num_upper/num_lower
    //             bookkeeping is untouched), so the only difference is case
    //             folding + markers; a word containing an upper-case letter
    //             therefore misses byte_map and is emitted raw. Mode byte
    //             forced to 7 for the same reason. Decoder-invertible by the
    //             STOCK decoder (A-Z are plain, unescaped literal bytes).
    //   nowrt   : the whole word transform is skipped — every TEXT block is
    //             emitted with mode byte 0 + raw bytes, which is exactly what
    //             encode_text already does when no dictionary is supplied. The
    //             dictionary IS still loaded and Pretrain still runs, so this
    //             is the transform pessimum, NOT `-c` without a dict.
    //   nomark  : ⚠ NOT a pessimum control and NOT LOSSLESS — the CEILING probe
    //  for `case-marker-policy` (
    //             ceiling-*). The tokenizer folds case EXACTLY as stock (so every
    //             substitution that fires in stock still fires, byte for byte) but
    //             EncodeWord emits NO kCapitalized/kUppercase/kEndUpper byte. The
    //             emitted stream is therefore stock's stream with precisely the
    //             case-marker bytes DELETED and nothing else moved — i.e. the
    //             stream an ideal case-marking policy would code if the case
    //             channel were FREE. archive(stock) − archive(nomark) is a strict
    //             upper bound on any policy that re-encodes the same case
    //             information, however cheaply. The decoder cannot invert it (the
    //             case channel is gone); that is the point, exactly as a free
    //             oracle is not a shippable mechanism.
    const wrt_earn = b.option(
        []const u8,
        "wrt-earn",
        "LAB pessimum control for the WRT frontend earnings census: stock|nosubst|nocase|nowrt|nomark (default stock = bit-identical)",
    ) orelse "stock";
    const wrt_earn_valid = std.mem.eql(u8, wrt_earn, "stock") or
        std.mem.eql(u8, wrt_earn, "nosubst") or
        std.mem.eql(u8, wrt_earn, "nocase") or
        std.mem.eql(u8, wrt_earn, "nowrt") or
        std.mem.eql(u8, wrt_earn, "nomark");
    if (!wrt_earn_valid) @panic("-Dwrt-earn must be one of stock|nosubst|nocase|nowrt|nomark");

    // -Dwrt-markmap=<path prefix>: LAB-ONLY. Writes, alongside the `-c` run,
    //   `<prefix>.cls` — ONE class byte per byte of the post-WRT temp stream —
    //   and `<prefix>.cost` — ONE u32 per temp byte, the coded cost of that byte
    //   in 1/4096-bit units, accumulated in f64. Together they give an EXACT
    //   (unsampled) stratification of the archive by frontend construct. Empty
    //   (default) compiles the whole facility away. Lives in prepr_options, NOT
    //  model_opts (the own-module rule: a dead option in the shared module costs ~40 B of
    //   decomp_bin and decomp_bin is charged 2×).
    const wrt_markmap = b.option(
        []const u8,
        "wrt-markmap",
        "LAB per-temp-byte class + coded-cost dump path prefix (empty = off, byte-identical)",
    ) orelse "";

    // ⛔ TEST-ONLY, DEFAULT OFF, NEVER SHIP. The `-e` front end is enwik9-ONLY
    // in FOUR places, not one: split4Comp/split4Decomp cut at fixed e9 line
    // numbers, the article reorder consumes the 243,425-line e9 order asset
    // (every missing key maps to article 0), phda9/sortMain assume that article
    // set, and r1 hard-aborts (R1LengthMismatch) unless the post-WRT stream is
    // EXACTLY enwik9's length. So `comp9 -e` has NO sub-e9 rehearsal of the
    // Form-1 pair (archive9 assembly, no-arg self-extraction, trailer/blob
    // slicing). This flag bypasses ALL FOUR stages on BOTH sides (ready4cmix ==
    // input verbatim; the WRT + coder + container are exercised unchanged). A
    // first version skipped only r1 and died Phda9Fail in split4Decomp/phda9 at
    // 1m and 20m with the coder innocent. It changes the coded stream and must
    // never appear in
    // SHIP_RECIPE.env.
    const test_no_r1 = b.option(bool, "test-no-r1", "TEST ONLY: bypass the enwik9-only front end (split4/reorder/phda9/r1) on both sides so the Form-1 container can be exercised below e9 (default false)") orelse false;
    const prepr_opts = b.addOptions();
    prepr_opts.addOption(bool, "test_no_r1", test_no_r1);
    prepr_opts.addOption([]const u8, "wrt_earn", wrt_earn);
    prepr_opts.addOption([]const u8, "wrt_markmap", wrt_markmap);
    prepr_opts.addOption(bool, "nullpred", phda9_nullpred);
    prepr_opts.addOption(bool, "nullqlg", phda9_nullqlg);
    prepr_opts.addOption(bool, "nullh2", phda9_nullh2);
    prepr_opts.addOption(bool, "nullh5", phda9_nullh5);

    // -Dcmfast: compile the pure-integer ContextMap3/4 compute (cm_fast.zig) as a
    // SEPARATE module at ReleaseFast, so the hot branchy cm3.mix/cm4.mix gets -O3
    // scheduling while the rest of the binary stays ReleaseSmall. cm3/cm4 are
    // integer-only → RF codegen is BIT-IDENTICAL. Default OFF (ReleaseSmall) = the
    // current single-module behavior. This is the ziggy per-region-optimize (Zig has
    // no per-function optimize; a module boundary is the only mechanism).
    const cmfast_rf = b.option(bool, "cmfast", "Compile ContextMap3/4 as a ReleaseFast module (bit-identical, hot integer -O3)") orelse false;
    // cmcold: the per-byte parsing layer + shared base files, ALWAYS
    // ReleaseSmall — measured size-heavy/instr-light (-O3 bought ~+9K raw for
    // ~nothing at 1-per-byte call rates). cmfast imports it (acyclic).
    const cmcold_mod = b.createModule(.{
        .root_source_file = b.path("src/models/fxcm26/cm_cold.zig"),
        .target = target,
        .optimize = .ReleaseSmall,
        .link_libc = link_libc,
    });
    const cmfast_mod = b.createModule(.{
        .root_source_file = b.path("src/models/fxcm26/cm_fast.zig"),
        .target = target,
        .optimize = if (cmfast_rf) .ReleaseFast else .ReleaseSmall,
        .link_libc = link_libc,
    });
    // cmfast needs its OWN build_options file (a shared model_opts file would collide
    // with the root module's copy in the same compilation). Mirror the 3 fields cm3/
    // cm4 read.
    const cmfast_opts = b.addOptions();
    cmfast_opts.addOption(u32, "fxcm_final_blend", fxcm_final_blend);
    cmfast_opts.addOption(f32, "lstm_adam_beta1", lstm_adam_beta1);
    cmfast_opts.addOption(i32, "sscm_base_rate", sscm_base_rate);
    cmfast_opts.addOption(f32, "mixer_decay_floor", mixer_decay_floor);
    cmfast_opts.addOption(bool, "cm3_diag", cm3_diag);
    cmfast_opts.addOption(bool, "cmc2_hash64", cmc2_hash64);
    cmfast_opts.addOption(u32, "cmc2_spill", cmc2_spill);
    cmfast_opts.addOption(u32, "cm3pack", cm3pack); // decoupled bucket — cmfast ONLY (own-module rule)
    cmfast_opts.addOption(bool, "idr", use_idr); // match_model (in cmfast) reads idr
    // fxcm_v26 (in cmfast) reads these four:
    cmfast_opts.addOption(f32, "mxa_uperr_scale", mxa_uperr_scale);
    cmfast_opts.addOption(bool, "mxa_diag", mxa_diag);
    cmfast_opts.addOption(u32, "mxa_radix", mxa_radix);
    cmfast_opts.addOption(u32, "mxa_dither", mxa_dither);
    cmfast_opts.addOption(u32, "mxa_w32", mxa_w32);
    cmfast_opts.addOption(f32, "mxa1_uperr_scale", mxa1_uperr_scale);
    cmfast_opts.addOption(f32, "mxa2_uperr_scale", mxa2_uperr_scale);
    cmfast_opts.addOption(u32, "cmc2_grow", cmc2_grow);
    cmfast_opts.addOption(u32, "cmc2_grow_hot", cmc2_grow_hot);
    cmfast_opts.addOption(u32, "cmc2_hot2", cmc2_hot2);
    // the own-module rule: read ONLY by fxcm_v26 (cmfast module).
    cmfast_opts.addOption(u32, "cmc_grow", cmc_grow);
    cmfast_opts.addOption(i32, "sta6_mdc", sta6_mdc); // fxcm_v26 STA6 row (in cmfast)
    // the own-module rule: this knob is read ONLY by fxcm_v26 (cmfast module), so it stays out of
    // the shared model_opts and costs zero prefix bytes when off.
    cmfast_opts.addOption(u32, "apm_wupd", apm_wupd);
    cmfast_opts.addOption(u32, "dsm_tag", dsm_tag);
    cmfast_opts.addOption(u32, "cm3_ts_rate", cm3_ts_rate);
    cmfast_opts.addOption(usize, "cm3_ts_runidx", cm3_ts_runidx);
    cmfast_opts.addOption(bool, "cm3_ts_runidx_ctrl", cm3_ts_runidx_ctrl);
    cmfast_opts.addOption(usize, "cm3_ts_visidx", cm3_ts_visidx);
    cmfast_opts.addOption(bool, "cm3_ts_visidx_ctrl", cm3_ts_visidx_ctrl);
    cmfast_opts.addOption(usize, "cm3_ts_recidx", cm3_ts_recidx);
    cmfast_opts.addOption(bool, "cm3_ts_recidx_ctrl", cm3_ts_recidx_ctrl);
    cmfast_opts.addOption(usize, "cm3_ts_dividx", cm3_ts_dividx);
    cmfast_opts.addOption(bool, "cm3_ts_dividx_ctrl", cm3_ts_dividx_ctrl);
    cmfast_opts.addOption(usize, "cm3_ts_escidx", cm3_ts_escidx);
    cmfast_opts.addOption(bool, "cm3_ts_escidx_ctrl", cm3_ts_escidx_ctrl);
    cmfast_opts.addOption(usize, "cm3_ts_div2idx", cm3_ts_div2idx);
    cmfast_opts.addOption(bool, "cm3_ts_div2idx_ctrl", cm3_ts_div2idx_ctrl);
    cmfast_opts.addOption(u32, "dsm_div_log2", dsm_div_log2);
    cmfast_opts.addOption(u32, "dsm_sm_rate", b.option(u32, "dsm-sm-rate", "DirectStateMap StateMap rate, log2 (default 14 = shipped)") orelse 14);
    cmfast_opts.addOption(bool, "apm_diag", apm_diag);
    cmfast_mod.addOptions("build_options", cmfast_opts);
    cmfast_mod.addImport("cmcold", cmcold_mod);
    cmfast_mod.addImport("ramcensus", ramcensus_mod);
    cmfast_mod.addImport("strict_fp", strictfp_mod);
    cmcold_mod.addImport("ramcensus", ramcensus_mod);
    cmcold_mod.addImport("strict_fp", strictfp_mod);
    // -Dleafgrad-fxcm: cm3.zig lives in the cmfast module, so the leafgrad knobs
    // have to reach it as an import here too. Same OPTIONS OBJECT as the root
    // module's (not a second addOptions), so the two can never disagree — and
    // still not the shared model_opts/cmfast_opts (the own-module rule).
    cmfast_mod.addImport("leafgrad_options", leafgrad_mod);
    cmfast_mod.addImport("fxcmmask_options", fxcmmask_mod);

    // -Dlstmfast: the LSTM byte-mixer (lstm.zig+lstm_layer.zig) as a ReleaseFast
    // module. The float compute path is COMPUTE-bound (~47% of wall via the
    // BPTT FMA chains) so RF's -O3 scheduling CONVERTS to wall (unlike the integer
    // cmfast, which is memory-bound → ~0 wall). BIT-IDENTICAL measured (
    // sha match @300k/1m/10m; RF alone doesn't re-associate this LSTM's fast-math
    // scopes — divergence only appears with -Dlstm-bwvec's hand-vectorization).
    // byte_mixer imports Lstm from here; only []f32 slices cross the boundary.
    const lstmfast_rf = b.option(bool, "lstmfast", "Compile the LSTM byte-mixer as a ReleaseFast module (bit-identical, hot float -O3)") orelse false;
    const lstmfast_mod = b.createModule(.{
        .root_source_file = b.path("src/mixer/lstm_fast.zig"),
        .target = target,
        .optimize = if (lstmfast_rf) .ReleaseFast else .ReleaseSmall,
        .link_libc = link_libc,
    });
    lstmfast_mod.addImport("ramcensus", ramcensus_mod);
    lstmfast_mod.addImport("strict_fp", strictfp_mod);
    const lstmfast_opts = b.addOptions();
    lstmfast_opts.addOption(u32, "fxcm_final_blend", fxcm_final_blend);
    lstmfast_opts.addOption(f32, "lstm_adam_beta1", lstm_adam_beta1);
    lstmfast_opts.addOption(i32, "sscm_base_rate", sscm_base_rate);
    lstmfast_opts.addOption(f32, "mixer_decay_floor", mixer_decay_floor);
    lstmfast_opts.addOption(bool, "mixer_blockskip", mixer_blockskip);
    lstmfast_opts.addOption(bool, "lstm_bwvec", lstm_bwvec);
    lstmfast_opts.addOption(bool, "vendored_libm", vendored_libm);
    // lstm_layer.zig (compiled into this module) also reads the update-limit
    // knob; lstm.zig also reads PRL and the compact-history guard. Mirror all
    // three into this module or -Dlstmfast builds fail on their comptime refs.
    lstmfast_opts.addOption(f32, "lstm_prior_rho", lstm_prior_rho);
    lstmfast_opts.addOption(u32, "lstm_prior_form", lstm_prior_form);
    lstmfast_opts.addOption(u32, "lstm_prior_clamp", lstm_prior_clamp);
    lstmfast_opts.addOption(bool, "auxparam_instr", auxparam_instr);
    // lstm.zig (PRL fold + the live-index ring) and lstm_layer.zig (the gathered
    // gate dot) both read this, and both compile into the lstmfast module too —
    // mirror it or lstmfast builds fail on the comptime ref (same reason as the
    // three PRL options above).
    lstmfast_opts.addOption(usize, "lstm_aux_sparse", lstm_aux_sparse);
    lstmfast_opts.addOption(usize, "lstm_head_rank", lstm_head_rank);
    lstmfast_opts.addOption(f32, "lstm_head_lr_scale", lstm_head_lr_scale);
    lstmfast_opts.addOption(bool, "lstm_aux_sparse_dense", lstm_aux_sparse_dense);
    lstmfast_opts.addOption(u32, "ppmd_arena_mb", ppmd_arena_mb);
    lstmfast_opts.addOption(u64, "lstm_update_limit", lstm_update_limit);
    lstmfast_opts.addOption(bool, "lstm_compact_hist", lstm_compact_hist);
    lstmfast_mod.addOptions("build_options", lstmfast_opts);
    // deliberate module CYCLE: parse_byte (cold) uses the auto-vectorized hot
    // sentence/words scans; Zig allows cyclic module imports (files stay unique).
    cmcold_mod.addImport("cmfast", cmfast_mod);

    // ---- zmix (general small binary; old v21/fxcm predictor via main.zig) ---
    const exe = b.addExecutable(.{
        .name = "zmix",
        .root_module = b.createModule(.{
            .root_source_file = b.path("src/main.zig"),
            .target = target,
            .optimize = optimize,
            .link_libc = link_libc,
        }),
    });
    exe.root_module.addOptions("build_options", model_opts);
    exe.root_module.addOptions("adepth_options", adepth_opts);
    exe.root_module.addOptions("extshed_options", extshed_opts);
    exe.root_module.addOptions("rls_options", rls_opts);
    exe.root_module.addOptions("wordlbl_opts", wordlbl_o);
    exe.root_module.addOptions("mixer_options", mixer_opts);
    attachTransformer(b, exe.root_module, transformer_on, transformer_mod, TF_SRC_O3_SEL, TF_SRC_OS_SEL, tf_fo3, tf_fos);
    exe.root_module.addImport("leafgrad_options", leafgrad_mod);
    exe.root_module.addImport("cmfast", cmfast_mod);
    exe.root_module.addImport("lstmfast", lstmfast_mod);
    exe.root_module.addImport("ramcensus", ramcensus_mod);
    exe.root_module.addImport("slotloss", slotloss_mod);
    exe.root_module.addImport("strict_fp", strictfp_mod);
    b.installArtifact(exe);

    const run_cmd = b.addRunArtifact(exe);
    run_cmd.step.dependOn(b.getInstallStep());
    if (b.args) |args| run_cmd.addArgs(args);
    const run_step = b.step("run", "Run zmix");
    run_step.dependOn(&run_cmd.step);

    // runner_lex — the byte-compatible cmix-lex -c/-d/-e/-D codec. ReleaseFast by
    // default (the reference/benchmark encoder).
    const runner_optimize = b.option(
        std.builtin.OptimizeMode,
        "runner-optimize",
        "Optimization mode for runner_lex (default: ReleaseFast)",
    ) orelse .ReleaseFast;
    const runner = b.addExecutable(.{
        .name = "runner_lex",
        .root_module = b.createModule(.{
            .root_source_file = b.path("src/runner_lex.zig"),
            .target = target,
            .optimize = runner_optimize,
            .link_libc = link_libc,
        }),
    });
    runner.root_module.addOptions("build_options", model_opts);
    runner.root_module.addOptions("adepth_options", adepth_opts);
    runner.root_module.addOptions("extshed_options", extshed_opts);
    runner.root_module.addOptions("rls_options", rls_opts);
    runner.root_module.addOptions("wordlbl_opts", wordlbl_o);
    runner.root_module.addOptions("mixer_options", mixer_opts);
    attachTransformer(b, runner.root_module, transformer_on, transformer_mod, TF_SRC_O3_SEL, TF_SRC_OS_SEL, tf_fo3, tf_fos);
    runner.root_module.addImport("leafgrad_options", leafgrad_mod);
    runner.root_module.addOptions("fieldcodec_options", fieldcodec_opts);
    runner.root_module.addImport("ramcensus", ramcensus_mod);
    runner.root_module.addImport("slotloss", slotloss_mod);
    runner.root_module.addImport("strict_fp", strictfp_mod);
    runner.root_module.addOptions("prepr_options", prepr_opts);
    runner.root_module.addImport("cmfast", cmfast_mod);
    runner.root_module.addImport("lstmfast", lstmfast_mod);
    // -Dbolt also applies to runner_lex (post-link BOLT experiments; see ship below).
    if (b.option(bool, "bolt-runner", "Keep relocs+symbols on runner_lex for llvm-bolt") orelse false) {
        runner.link_emit_relocs = true;
        runner.root_module.strip = false;
        // strip=false re-enables error-return-tracing in Release modes, which
        // costs ~+13% retired instructions (the round-3 strip trap); BOLT only
        // needs symbols+relocs, not traces.
        runner.root_module.error_tracing = false;
        // strip=false also re-enables frame pointers (unwindability): push/pop
        // rbp in every fn = the ~+15% instr trap on call-heavy -Oz code.
        runner.root_module.omit_frame_pointer = true;
    }
    // -Dlto (avenue #2): force-enable LLVM LTO on runner_lex AND zmix_ship (applied
    // to the ship below). Bit-identical, shrinks the packed binary.
    const want_lto = b.option(bool, "lto", "Force LLVM LTO on runner_lex + zmix_ship (avenue #2)") orelse false;
    if (want_lto) runner.lto = .full;
    b.installArtifact(runner);
    const run_runner = b.addRunArtifact(runner);
    run_runner.step.dependOn(b.getInstallStep());
    if (b.args) |args| run_runner.addArgs(args);
    const runner_step = b.step("run-runner", "Run runner_lex (cmix-lex -c/-d codec)");
    runner_step.dependOn(&run_runner.step);

    // ---- auxparam (research instrument, branch-local) ---------------------
    const auxparam = b.addExecutable(.{
        .name = "auxparam",
        .root_module = b.createModule(.{
            .root_source_file = b.path("src/auxparam.zig"),
            .target = target,
            .optimize = .ReleaseFast,
            .link_libc = link_libc,
        }),
    });
    auxparam.root_module.addOptions("build_options", model_opts);
    auxparam.root_module.addOptions("adepth_options", adepth_opts);
    auxparam.root_module.addOptions("extshed_options", extshed_opts);
    auxparam.root_module.addOptions("rls_options", rls_opts);
    auxparam.root_module.addOptions("wordlbl_opts", wordlbl_o);
    auxparam.root_module.addOptions("mixer_options", mixer_opts);
    attachTransformer(b, auxparam.root_module, transformer_on, transformer_mod, TF_SRC_O3_SEL, TF_SRC_OS_SEL, tf_fo3, tf_fos);
    auxparam.root_module.addImport("leafgrad_options", leafgrad_mod);
    auxparam.root_module.addImport("cmfast", cmfast_mod);
    auxparam.root_module.addImport("lstmfast", lstmfast_mod);
    auxparam.root_module.addImport("ramcensus", ramcensus_mod);
    auxparam.root_module.addImport("slotloss", slotloss_mod);
    auxparam.root_module.addImport("strict_fp", strictfp_mod);
    const auxparam_install = b.addInstallArtifact(auxparam, .{});
    const auxparam_step = b.step("auxparam", "Build the auxparam research instrument");
    auxparam_step.dependOn(&auxparam_install.step);

    // ---- derivdict-emit (Arm E' recipe PRODUCER) --------------------------
    // A separate executable, NOT a -D option on the shipped modules: the emitter
    // must never land in decomp_bin (charged TWICE under Form-1), and a build
    // option would cost ~40 B of dead-option plumbing even unused (the own-module rule). It
    // links src/derivdict.zig directly, so the recipe's producer and its shipped
    // consumer are one implementation — the defect that stalled derivdict at the
    // LSTM-era gate was a second implementation of this format living in Python.
    // Not part of the default install step; build it with `zig build derivdict-emit`.
    const derivdict_mod = b.createModule(.{
        .root_source_file = b.path("src/derivdict.zig"),
        .target = target,
        .optimize = .ReleaseFast,
        .link_libc = link_libc,
    });
    const derivdict_emit = b.addExecutable(.{
        .name = "derivdict_emit",
        .root_module = b.createModule(.{
            .root_source_file = b.path("tools/derivdict_emit.zig"),
            .target = target,
            .optimize = .ReleaseFast,
            .link_libc = link_libc,
        }),
    });
    derivdict_emit.root_module.addImport("derivdict", derivdict_mod);
    const derivdict_emit_install = b.addInstallArtifact(derivdict_emit, .{});
    const derivdict_emit_step = b.step("derivdict-emit", "Build the Arm E' recipe emitter (tools/derivdict_emit.zig)");
    derivdict_emit_step.dependOn(&derivdict_emit_install.step);
    const run_derivdict_emit = b.addRunArtifact(derivdict_emit);
    if (b.args) |a| run_derivdict_emit.addArgs(a);
    const run_derivdict_emit_step = b.step("run-derivdict-emit", "Emit the Arm E' recipe blob: -- <corpus> <english.dic> <out.bin>");
    run_derivdict_emit_step.dependOn(&run_derivdict_emit.step);


    // runner_lex_decode — decode-only DCE size experiment (ReleaseSmall). NOT the
    // shipping binary: decode-only counts 2x under the Hutter rule. Kept only to
    // measure how small a pure-decode image gets.
    const decode = b.addExecutable(.{
        .name = "runner_lex_decode",
        .root_module = b.createModule(.{
            .root_source_file = b.path("src/runner_lex_decode.zig"),
            .target = target,
            .optimize = .ReleaseSmall,
            .link_libc = link_libc,
        }),
    });
    decode.root_module.addOptions("build_options", model_opts);
    decode.root_module.addOptions("adepth_options", adepth_opts);
    decode.root_module.addOptions("extshed_options", extshed_opts);
    decode.root_module.addOptions("rls_options", rls_opts);
    decode.root_module.addOptions("wordlbl_opts", wordlbl_o);
    decode.root_module.addOptions("mixer_options", mixer_opts);
    attachTransformer(b, decode.root_module, transformer_on, transformer_mod, TF_SRC_O3_SEL, TF_SRC_OS_SEL, tf_fo3, tf_fos);
    decode.root_module.addImport("leafgrad_options", leafgrad_mod);
    decode.root_module.addOptions("fieldcodec_options", fieldcodec_opts);
    decode.root_module.addImport("ramcensus", ramcensus_mod);
    decode.root_module.addImport("slotloss", slotloss_mod);
    decode.root_module.addImport("strict_fp", strictfp_mod);
    decode.root_module.addOptions("prepr_options", prepr_opts);
    decode.root_module.addImport("cmfast", cmfast_mod);
    decode.root_module.addImport("lstmfast", lstmfast_mod);
    b.installArtifact(decode);

    // ship — THE shipping Hutter binary: a self-EXTRACTING archive. The code carries
    // NO raw assets; construct_ship.sh strips+UPXs this lean binary (== `decomp_bin`),
    // self-compresses english.dic/new_article_order with its own `-c` codec, and appends
    // them + a 12-byte HeaderInfo trailer (record: build_and_construct_comp.sh). At run
    // time it reads its own tail and self-decompresses the dict before decoding — so
    // `decomp_bin` stays tiny and the appended engine-compressed assets replace the old
    // raw `@embedFile` + UPX. ReleaseSmall by default for size; override the code speed
    // with -Dship-optimize=ReleaseFast if the decode time gate ever binds.
    const ship_optimize = b.option(
        std.builtin.OptimizeMode,
        "ship-optimize",
        "Optimization mode for zmix_ship code (default: ReleaseSmall)",
    ) orelse .ReleaseSmall;
    // llvm-bolt post-link support, opt-in only: --emit-relocs bloats the UPX-packed
    // engine 131,736 -> 197,132 B (S1 regression), and BOLT measured null on this
    // data-bound binary anyway (iter 293/297). Kept for future experiments.
    const use_bolt = b.option(
        bool,
        "bolt",
        "Keep link relocations for llvm-bolt post-link optimization (bloats packed size)",
    ) orelse false;
    const ship = b.addExecutable(.{
        .name = "zmix_ship",
        .root_module = b.createModule(.{
            .root_source_file = b.path("src/ship.zig"),
            .target = target,
            .optimize = ship_optimize,
            .link_libc = link_libc,
            // ReleaseSmall implies --strip-all, which ld.lld rejects next to
            // --emit-relocs; construct_ship.sh strips the artifact anyway.
            .strip = if (use_bolt) false else null,
        }),
    });
    ship.root_module.addOptions("build_options", model_opts);
    ship.root_module.addOptions("adepth_options", adepth_opts);
    ship.root_module.addOptions("extshed_options", extshed_opts);
    ship.root_module.addOptions("rls_options", rls_opts);
    ship.root_module.addOptions("wordlbl_opts", wordlbl_o);
    ship.root_module.addOptions("mixer_options", mixer_opts);
    attachTransformer(b, ship.root_module, transformer_on, transformer_mod_ship, TF_SRC_O3_SEL, TF_SRC_OS_SEL, tf_fo3, tf_fos);
    ship.root_module.addImport("leafgrad_options", leafgrad_mod);
    ship.root_module.addImport("wallpareto_options", wallpareto_mod);
    ship.root_module.addOptions("fieldcodec_options", fieldcodec_opts);
    ship.root_module.addImport("ramcensus", ramcensus_mod);
    ship.root_module.addImport("slotloss", slotloss_mod);
    ship.root_module.addImport("strict_fp", strictfp_mod);
    ship.root_module.addOptions("prepr_options", prepr_opts);
    ship.root_module.addImport("cmfast", cmfast_mod);
    ship.root_module.addImport("lstmfast", lstmfast_mod);
    if (use_bolt) ship.link_emit_relocs = true;
    // Nothing in the shipped binary unwinds (simple_panic, no exceptions):
    // .eh_frame is dead weight (-5,296 B packed; verified, binary runs).
    ship.root_module.unwind_tables = .none;
    // The codec is fully single-threaded (grep: zero Thread/atomic/Mutex in src) —
    // yet the default build links libpthread, imports __tls_get_addr and carries a
    // TLS segment. single_threaded drops the pthread NEEDED + TLS machinery and lets
    // std (GeneralPurposeAllocator, etc.) shed its atomics/locks. Output is
    // byte-identical (this changes synchronization codegen only, no arithmetic).
    ship.root_module.single_threaded = true;
    if (want_lto) ship.lto = .full;
    b.installArtifact(ship);
    const ship_step = b.step("ship", "Build the shipping self-extractor (zmix_ship)");
    ship_step.dependOn(&b.addInstallArtifact(ship, .{}).step);

    // form1 — minimal no-CLI dual-role prefix for the strict Form-1 submission.
    // Its own trailer selects comp9 (payload size zero) versus archive9
    // (payload size nonzero), so exactly the same prefix is counted in both.
    // R1v1 is required: the archive layout intentionally omits the order asset.
    const form1 = b.addExecutable(.{
        .name = "zmix_form1",
        .root_module = b.createModule(.{
            .root_source_file = b.path("src/form1.zig"),
            .target = target,
            .optimize = ship_optimize,
            .link_libc = link_libc,
        }),
    });
    form1.root_module.addOptions("build_options", model_opts);
    form1.root_module.addOptions("adepth_options", adepth_opts);
    form1.root_module.addOptions("extshed_options", extshed_opts);
    form1.root_module.addOptions("rls_options", rls_opts);
    form1.root_module.addOptions("wordlbl_opts", wordlbl_o);
    form1.root_module.addOptions("mixer_options", mixer_opts);
    attachTransformer(b, form1.root_module, transformer_on, transformer_mod_ship, TF_SRC_O3_SEL, TF_SRC_OS_SEL, tf_fo3, tf_fos);
    form1.root_module.addImport("leafgrad_options", leafgrad_mod);
    form1.root_module.addImport("wallpareto_options", wallpareto_mod);
    form1.root_module.addOptions("fieldcodec_options", fieldcodec_opts);
    form1.root_module.addImport("ramcensus", ramcensus_mod);
    form1.root_module.addImport("slotloss", slotloss_mod);
    form1.root_module.addImport("strict_fp", strictfp_mod);
    form1.root_module.addOptions("prepr_options", prepr_opts);
    // -Dform1-malloc lives in form1's OWN options module, NOT in model_opts:
    // model_opts is shared with zmix/zmix_ship/runner_lex/tests, and the banked
    // rule is that adding a -D option can move a shipped binary even when the
    // option is never engaged ("bit-identity on archives, not binaries"). A
    // private options file makes that structurally impossible — no other target
    // imports it. See src/form1.zig for what the flag buys and what it costs.
    const form1_malloc = b.option(
        bool,
        "form1-malloc",
        "Form-1 prefix uses the dense malloc/posix_memalign allocator (-3,176 B packed, +98-125 MB MaxRSS; default false = the stock GPA src/ship.zig uses)",
    ) orelse false;
    const form1_opts = b.addOptions();
    form1_opts.addOption(bool, "malloc_allocator", form1_malloc);
    form1.root_module.addOptions("form1_options", form1_opts);
    form1.root_module.addImport("cmfast", cmfast_mod);
    form1.root_module.addImport("lstmfast", lstmfast_mod);
    form1.root_module.unwind_tables = .none;
    form1.root_module.single_threaded = true;
    // CENSUS ONLY (default false, verified to leave the packed prefix
    // byte-identical, sha 1e62611d...): keep the symbol table so a per-TU
    // symbol census of the UNPACKED image is possible. Never set on a ship
    // build — the ship chain strips anyway, but strip=false re-enables
    // error-return tracing and frame pointers in Release modes.
    if (b.option(bool, "form1-census", "CENSUS ONLY: keep symbols in zmix_form1 for per-TU attribution") orelse false) form1.root_module.strip = false;
    form1.link_z_relro = false;
    if (want_lto) form1.lto = .full;
    const form1_step = b.step("form1", "Build the minimal dual-role Form-1 prefix");
    form1_step.dependOn(&b.addInstallArtifact(form1, .{}).step);

    // Unit tests (state-table parity, table generation, round-trip, lex pipeline).
    const tests = b.addTest(.{
        .root_module = b.createModule(.{
            .root_source_file = b.path("src/tests.zig"),
            .target = target,
            .optimize = optimize,
            .link_libc = link_libc,
        }),
    });
    tests.root_module.addOptions("build_options", model_opts);
    tests.root_module.addOptions("adepth_options", adepth_opts);
    tests.root_module.addOptions("extshed_options", extshed_opts);
    tests.root_module.addOptions("rls_options", rls_opts);
    tests.root_module.addOptions("wordlbl_opts", wordlbl_o);
    tests.root_module.addOptions("mixer_options", mixer_opts);
    attachTransformer(b, tests.root_module, transformer_on, transformer_mod, TF_SRC_O3_SEL, TF_SRC_OS_SEL, tf_fo3, tf_fos);
    tests.root_module.addImport("leafgrad_options", leafgrad_mod);
    tests.root_module.addOptions("fieldcodec_options", fieldcodec_opts);
    tests.root_module.addImport("ramcensus", ramcensus_mod);
    tests.root_module.addImport("slotloss", slotloss_mod);
    tests.root_module.addImport("strict_fp", strictfp_mod);
    tests.root_module.addOptions("prepr_options", prepr_opts);
    tests.root_module.addImport("cmfast", cmfast_mod);
    tests.root_module.addImport("lstmfast", lstmfast_mod);
    const run_tests = b.addRunArtifact(tests);
    const test_step = b.step("test", "Run unit tests");
    test_step.dependOn(&run_tests.step);

    // The cmfast module's own in-file tests (cm3/cm4 golden traps, dsm/mm/sscm/
    // rcm/mixer1/fxcm_v26 oracle checksums): `zig test` only collects tests from
    // the ROOT module of a test compilation, so give the module its own test
    // compilation. Note the module is shared, so this reuses cmfast_mod directly.
    const tests_cmfast = b.addTest(.{ .root_module = cmfast_mod });
    const run_tests_cmfast = b.addRunArtifact(tests_cmfast);
    test_step.dependOn(&run_tests_cmfast.step);
    const tests_cmcold = b.addTest(.{ .root_module = cmcold_mod });
    const run_tests_cmcold = b.addRunArtifact(tests_cmcold);
    test_step.dependOn(&run_tests_cmcold.step);
    // Same treatment for the lstmfast module: lstm_layer.zig carries the
    // -Dlstm-aux-sparse kernel gates (dotGather / axpyScatter exactness), and a
    // file may belong to only ONE module — src/tests.zig cannot import it
    // without dragging lstm.zig out of `lstmfast` into `root`.
    const tests_lstmfast = b.addTest(.{ .root_module = lstmfast_mod });
    const run_tests_lstmfast = b.addRunArtifact(tests_lstmfast);
    test_step.dependOn(&run_tests_lstmfast.step);

    // Safety-checked test run (integer-overflow + bounds checks ON). The shipping
    // exe is ReleaseSmall (checks off) for size, so this step is the net that
    // catches UB the default build would silently wrap. `zig build test-safe`.
    const tests_safe = b.addTest(.{
        .root_module = b.createModule(.{
            .root_source_file = b.path("src/tests.zig"),
            .target = target,
            .optimize = .ReleaseSafe,
            .link_libc = link_libc,
        }),
    });
    tests_safe.root_module.addOptions("build_options", model_opts);
    tests_safe.root_module.addOptions("adepth_options", adepth_opts);
    tests_safe.root_module.addOptions("extshed_options", extshed_opts);
    tests_safe.root_module.addOptions("rls_options", rls_opts);
    tests_safe.root_module.addOptions("wordlbl_opts", wordlbl_o);
    tests_safe.root_module.addOptions("mixer_options", mixer_opts);
    attachTransformer(b, tests_safe.root_module, transformer_on, transformer_mod, TF_SRC_O3_SEL, TF_SRC_OS_SEL, tf_fo3, tf_fos);
    tests_safe.root_module.addImport("leafgrad_options", leafgrad_mod);
    tests_safe.root_module.addOptions("fieldcodec_options", fieldcodec_opts);
    tests_safe.root_module.addImport("ramcensus", ramcensus_mod);
    tests_safe.root_module.addImport("slotloss", slotloss_mod);
    tests_safe.root_module.addImport("strict_fp", strictfp_mod);
    tests_safe.root_module.addOptions("prepr_options", prepr_opts);
    // A separate ReleaseSafe instance of the cmfast module: the shipping module
    // is Release (checks off), which would exempt the whole fxcm26 predictor
    // from this safety net. Distinct compilations, so the same source files in
    // two modules is legal here.
    const cmcold_safe = b.createModule(.{
        .root_source_file = b.path("src/models/fxcm26/cm_cold.zig"),
        .target = target,
        .optimize = .ReleaseSafe,
        .link_libc = link_libc,
    });
    const cmfast_safe = b.createModule(.{
        .root_source_file = b.path("src/models/fxcm26/cm_fast.zig"),
        .target = target,
        .optimize = .ReleaseSafe,
        .link_libc = link_libc,
    });
    cmfast_safe.addOptions("build_options", cmfast_opts);
    cmfast_safe.addImport("cmcold", cmcold_safe);
    cmcold_safe.addImport("cmfast", cmfast_safe);
    tests_safe.root_module.addImport("cmfast", cmfast_safe);
    const run_tests_safe = b.addRunArtifact(tests_safe);
    const test_safe_step = b.step("test-safe", "Run unit tests with safety checks (ReleaseSafe)");
    test_safe_step.dependOn(&run_tests_safe.step);
    // ...and the module's own in-file tests, also at ReleaseSafe.
    const tests_cmfast_safe = b.addTest(.{ .root_module = cmfast_safe });
    const run_tests_cmfast_safe = b.addRunArtifact(tests_cmfast_safe);
    test_safe_step.dependOn(&run_tests_cmfast_safe.step);
}
