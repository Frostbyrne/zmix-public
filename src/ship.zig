//! `zmix_ship` — THE shipping Hutter-Prize binary: a self-EXTRACTING archive,
//! ported from cmix-lex's packaging (`ref/cmix-lex/src/readalike_prepr/self_extract.h`,
//! runner.cpp:377-472, build_and_construct_comp.sh).
//!
//! Unlike the previous build, the assets are NOT `@embedFile`d raw. Instead — exactly
//! like the record — the dictionary and article-order are self-compressed with this
//! binary's OWN codec (`-c`, the no-dict store path) and APPENDED to the stripped+UPX'd
//! program, followed by a fixed 12-byte `HeaderInfo` trailer. At run time the binary
//! reads its own file (the record `fopen("cmix"/"archive9")`, self_extract.h:37/103 —
//! here resolved via /proc/self/exe so it is name/cwd independent), slices the appended
//! segments off its tail, and self-decompresses them back to the raw `.dict`/`.order`
//! before feeding the fxcm `dosym` codec. This is a TRUE self-extractor, and the
//! engine-compressed assets are far smaller than the raw+UPX embed they replace.
//!
//! Two on-disk forms of this same program exist (record: `cmix` vs `archive9`):
//!
//!   `comp9`  (built by construct_ship.sh; the operator runs `-e`):
//!       [ decomp_bin ][ dict.comp ][ order.comp ][ HeaderInfo{|dict.comp|,|order.comp|,0} ]
//!       — construct.sh line 80: `cat cmix_orig comp_dict comp_order header.dat`.
//!
//!   `archive9`  (produced by `-e`; self-decodes with no args):
//!       legacy/r1v1: [ decomp_bin ][ dict.comp ][ payload ][ HeaderInfo ]
//!       r1v2:        [ decomp_bin ][ dict.comp ][ order.comp ][ payload ][ HeaderInfo ]
//!       r1v2 restore derives its block permutation from the raw order asset, so its
//!       explicit physical-correctness layout must repeat order.comp in archive9.
//!
//! CLI:
//!   zmix_ship -c <in> <out>       compress a file, no dict  (self-compress an asset)
//!   zmix_ship -d <in> <out>       decompress a file, no dict
//!   zmix_ship -e <enwik9> <out>   full enwik9 compress -> writes a self-extracting archive9
//!   zmix_ship -D <out>            full enwik9 decompress (payload read from own tail)
//!   zmix_ship -h <ds> <os> <ps> <out>   write a 12-byte HeaderInfo trailer (construct.sh -h)
//!   zmix_ship --extract-assets <dict_out> <order_out>
//!                                 self-extract + self-decompress the appended dict/order
//!                                 (verification: assert they == the originals; under
//!                                 -Ds7order the order is s7-decoded back to the raw text)
//!   zmix_ship --s7-encode <order_in> <s7_out>   (-Ds7order only) mint the s7_runlabel
//!                                 re-encoding of the raw article order (inverse asserted)
//!
//! Seed: FROZEN at 923 (DEFAULT_SEED below). The shipping binary consults NO
//! environment variable — see `zmix_hermetic`. An env-selected code path is the
//! same defect class as a sysfs dependency,
//! and the *name* alone is a hermetic-gate hazard string: `ZMIX_SEED` in usage
//! text aborted the transformer-era Form-1 e9 leg's phase-5 scan of the unpacked decomp_bin

const std = @import("std");
const build_options = @import("build_options");
const rl = @import("runner_lex.zig");
const tfmod = @import("mixer/transformer.zig");
const se = @import("prepr/self_extract.zig");
const s7 = @import("prepr/s7_order.zig");
const ppmd_mod = @import("models/ppmd.zig");
const derivdict = @import("derivdict.zig");
const SHIP_EVICT_BYTES: u64 = @import("wallpareto_options").ship_evict_bytes;

// Trap-on-panic: drops Zig's DWARF unwinder + stack-trace formatter (~100K of
// debug.* code that only runs on a crash) from the shipping binary. A Hutter
// submission has no use for pretty panics; the record's C++ builds with
// -fno-exceptions -fno-unwind-tables for the same reason.
pub const panic = std.debug.simple_panic;

// Pin the exp/log family to vendored deterministic implementations (see
// detm_math.zig) — mandatory for cross-machine codec determinism.
comptime {
    _ = @import("detm_math.zig");
}

// HERMETIC SHIP (footgun review P1): the shipping binary reads NO tuning env —
// predictor_lex compiles its ZMIX_LSTM_* overrides away under this decl, and
// ZMIX_SEED below is frozen. A stray env var on the encode box or the judge's
// machine would otherwise change model numerics and desync the coder ~40 h in.
// Experiments keep their knobs via runner_lex (which does not set this).
pub const zmix_hermetic = true;

const DEFAULT_SEED: i32 = 923; // makefile -DSEED=923 (frozen; env NOT consulted)

fn readFile(gpa: std.mem.Allocator, path: []const u8) ![]u8 {
    return std.fs.cwd().readFileAlloc(gpa, path, 1 << 34);
}
fn writeFile(path: []const u8, data: []const u8) !void {
    const f = try std.fs.cwd().createFile(path, .{});
    defer f.close();
    try f.writeAll(data);
}
/// Read the running executable's own bytes — the self-extract source. Mirrors the
/// record's `fopen("cmix"/"archive9", "rb")` + read-whole-file (self_extract.h:37-47 /
/// :103-111), but resolves the real path via /proc/self/exe so it is independent of
/// argv[0]/cwd. UPX leaves the appended overlay (dict.comp/order/payload + trailer)
/// intact after its packed image, so the whole on-disk file — code AND tail — is read.
fn readSelfImage(gpa: std.mem.Allocator) ![]u8 {
    const path = try std.fs.selfExePathAlloc(gpa);
    defer gpa.free(path);
    return std.fs.cwd().readFileAlloc(gpa, path, 1 << 34);
}

/// Decode the raw article-order text from its in-image compressed segment.
/// Stock: the segment holds the engine-compressed raw decimal text verbatim.
/// -Ds7order: the segment holds the s7_runlabel re-encoding (id-keyed run
/// labels + id-set exception trailer; src/prepr/s7_order.zig); reconstruct the
/// EXACT original text here, so every downstream consumer — article reorder,
/// r1v2 restore, --extract-assets — sees bytes identical to stock and the
/// coded payload cannot move. Encode-side only under the shipped r1v1 layout
/// (the no-arg decode never reads the order).
fn decodeOrderAsset(gpa: std.mem.Allocator, order_comp: []const u8, seed: i32) ![]u8 {
    const raw = try rl.runDecompression(gpa, order_comp, null, seed);
    if (comptime build_options.s7order) {
        defer gpa.free(raw);
        return s7.decode(gpa, raw);
    }
    return raw;
}

/// Full enwik9 self-decode (the judged operation): trailer-only slice math, the
/// ~100 KB dict.comp read+decoded+freed FIRST, then the ~109 MB payload
/// STREAMED from our own file (rl.runDecompressionDStreamed) — the archive
/// image never sits in RAM under the ~9.5 GB coder peak (it was ~45% of the
/// whole margin under the 1e10-byte gate; the C++ record streams its payload
/// from disk too).
fn selfDecode(gpa: std.mem.Allocator, out_path: []const u8, seed: i32) !void {
    const exe_path = try std.fs.selfExePathAlloc(gpa);
    defer gpa.free(exe_path);
    const f = try std.fs.cwd().openFile(exe_path, .{});
    defer f.close();
    const fsize = try f.getEndPos();
    if (fsize < se.HEADER_SIZE) return error.ImageTooSmall;
    var tb: [se.HEADER_SIZE]u8 = undefined;
    _ = try f.preadAll(&tb, fsize - se.HEADER_SIZE);
    const header = try se.readTrailer(&tb);
    // Legacy/r1v1 payload restore is order-independent and keeps the order only
    // in comp9. r1v2 explicitly embeds the compressed order between dict and
    // payload; this is a counted second copy, never an outside input.
    const order_mode: se.ArchiveOrderMode = if (comptime build_options.r1v2)
        .embedded
    else
        .omitted;
    const layout = try se.locateArchive(fsize, header, order_mode);

    // transformer-era: hand the predictor the weights blob from OUR OWN TAIL before it is
    // built. Read here rather than in the predictor because only the ship paths
    // have an artifact to slice. Leaks by design for the process lifetime: the
    // transformer is constructed once and lives until exit.
    //
    // ⛔ Without this the predictor falls back to `-Dtransformer-weights`, an
    // ABSOLUTE build-time path — the defect that made an earlier build unsubmittable.
    if (comptime tfmod.enabled) {
        if (header.tfweights_size > 0) {
            const tw: usize = @intCast(header.tfweights_size);
            // Layout: [ decomp_bin ][ dict ][ payload ][ tfweights ][ header ]
            // — the blob sits immediately before the trailer in archive9,
            // mirroring comp9's `... comp_order ++ comp_tfweights ++ header`.
            const off = fsize - se.HEADER_SIZE - tw;
            const blob = try gpa.alloc(u8, tw);
            if (try f.preadAll(blob, off) != tw) return error.CorruptHeader;
            tfmod.Transformer.embedded_blob = blob;
        } else {
            // A transformer build whose artifact carries no blob cannot decode.
            // Fail here, loudly, rather than silently reading a dev path that
            // does not exist on a judge machine.
            return error.MissingEmbeddedWeights;
        }
    }

    const dict_comp = try gpa.alloc(u8, @intCast(layout.dict_size));
    _ = try f.preadAll(dict_comp, layout.dict_offset);
    const dict = try rl.runDecompression(gpa, dict_comp, null, seed);
    gpa.free(dict_comp);
    defer gpa.free(dict);

    const order: ?[]u8 = if (comptime build_options.r1v2) blk: {
        const order_comp = try gpa.alloc(u8, @intCast(layout.order_size));
        _ = try f.preadAll(order_comp, layout.order_offset.?);
        const raw_order = decodeOrderAsset(gpa, order_comp, seed) catch |e| {
            gpa.free(order_comp);
            return e;
        };
        gpa.free(order_comp);
        break :blk raw_order;
    } else null;
    defer if (order) |o| gpa.free(o);

    const enwik9 = try rl.runDecompressionDStreamed(
        gpa,
        f,
        layout.payload_offset,
        layout.payload_size,
        dict,
        order,
        seed,
    );
    defer gpa.free(enwik9);
    try writeFile(out_path, enwik9);
    std.debug.print("-D seed={d}: payload={d} -> {s} ({d} bytes)\n", .{ seed, layout.payload_size, out_path, enwik9.len });
}

/// Bad invocation: print usage and FAIL (exit != 0). The old exit-0-on-usage
/// meant a judge's misfired run looked successful while producing nothing.
fn badUsage() error{BadUsage}!void {
    usage();
    return error.BadUsage;
}

fn usage() void {
    std.debug.print(
        \\zmix_ship — self-EXTRACTING cmix-lex archive (dict/order self-compressed in the tail)
        \\
        \\  (no args)              Form-1 decode: own embedded payload -> data9
        \\  -c <in> <out> [<tfw>]  compress a file (no dict; self-compress an asset)
        \\  -d <in> <out> [<tfw>]  decompress a file (no dict)
        \\  -E <enwik9> <bhm_out>  legacy Form-2 experiment: payload-only archive9.bhm
        \\  -e <enwik9> <out>      full enwik9 compress -> Form-1 self-extracting archive9
        \\  -D <out>               full enwik9 decompress (payload from own tail)
        \\  -h <ds> <os> <ps> <out>            write a 12-byte HeaderInfo trailer
        \\  --extract-assets <dict_out> <order_out>   self-extract+decompress the tail assets
        \\
    , .{});
}

// Flush denormals to zero (FTZ, MXCSR bit 15) + treat denormal inputs as zero (DAZ,
// bit 6). Subnormal f32 ops are ~100x slower on x86 (microcoded); the record's
// -ffp-model=fast build flushes them, so we match it. This does NOT reorder any
// arithmetic — it only snaps already-negligible subnormals (|x| < 2^-126) to 0.
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
    var gpa_impl = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa_impl.deinit();
    const gpa = gpa_impl.allocator();

    const args = try std.process.argsAlloc(gpa);
    defer std.process.argsFree(gpa, args);
    // THE FORM-1 SUBMISSION CONTRACT (RULES rule 2): run archive9 with NO
    // arguments and reproduce data9 using only the executable's own embedded
    // dict+payload. Never probe the working directory for archive9.bhm (or any
    // other input): an unrelated file must not alter the judged operation.
    //
    // PER-OP valve eviction cadence (timing-only, content-invariant — proven
    // byte-exact across intervals: 5k/25k/100k archive-identical).
    // ⛔ BOTH SHIPPED CADENCES WERE TOO LOOSE AND BOTH PREMISES BEHIND THEM
    // ARE FALSIFIED BY MEASUREMENT. The gate quantity is UNCAPPED MaxRSS —
    // total resident, anon + file-backed (operator ruling plan
    //   Earlier readings were taken under an IMPOSED cgroup cap, which
    // suppresses the arena's resident file pages, so they measured our own
    // harness rather than the gate.
    //   DECODE, was 25,000 on a "~1GB RAM headroom" premise: measured uncapped
    //   e9 VmHWM 12,832,864 and 12,916,276 kB = FAIL by ~3.07-3.15 GB against
    //   the 9,765,625 kB (1e10 B) bar. At 1,000 the same op reads 9,586,656 kB
    //   = PASS by 178,969 kB (1.83 %). exp-ppmd-decode-valve-cadence-result-
    //   aug-09.md.
    //   ENCODE, was 2000 to "cap the ~1GB transient": it does cap it, but not
    //   far enough. The e9 gate number is anon_e9 + R(cadence) — NEVER a scaled
    //   20m MaxRSS, which proxies ANON only (+3.3-4.2 %) and not total
    //   (+9.0-9.9 %), and reading it as a total returns a FALSE PASS here.
    //  Composed 22-flag anon floor MEASURED 8,845,104 kB (a fleet leg,
    //  a fleet box SOLO) ⇒ projected e9 MaxRSS at 2000 = 9,816,595-9,895,058 kB =
    //   FAIL by 50,970-129,433 kB (0.52-1.33 %); at 1,000 = 9,557,515-9,635,978
    //   kB = PASS by 129,647-208,110 kB (1.33-2.13 %). The verdict holds under
    //   all four route x co-occurrence-correction combinations.
    // BOTH OPS THEREFORE RUN TIGHT AT 1,000 — deliberately tighter than the
    // reference's own 5,000 parity value, to hold margin against the
    // conservative bar rather than lean on cmix-lex's accepted 9,993,408.
    // The fee is WALL ONLY and it is affordable: the encode's 2000->1,000 step
    // measured +1.203 % at 20m (single pair — indicative, not an ABABAB
    // verdict), worst model +2.57 %, giving W = 62,285-63,133 = 0.89-0.90x the
    // 70,000 bar. ZERO bytes: both cadence legs emitted the identical archive
    // (3,149,879 B, sha256 cbd56478…), matching the banked composed 20m golden.
    // ⚠ Both PASSes are on a THP [madvise] basis (AnonHugePages: 0 throughout);
    // an [always] judge box carries a banked ~3GB exposure that would swamp
    // these margins (pre-existing, unresolved). e9 anon is projected, not
    // measured.
    if (args.len < 2) {
        ppmd_mod.setEvictInterval(SHIP_EVICT_BYTES); // decode: TIGHT (gate-required, see above)
        return selfDecode(gpa, "data9", DEFAULT_SEED);
    }
    const m = args[1];
    const seed = DEFAULT_SEED;

    // ---- `-h`: write a 12-byte HeaderInfo trailer (record runner.cpp:474-480). ----
    if (std.mem.eql(u8, m, "-h")) {
        // LSTM-era:  -h <dict> <order> <decomp_input> <out>            (6 argv)
        // transformer-era:  -h <dict> <order> <decomp_input> <tfweights> <out> (7 argv)
        // The four-field form matches fx2-cmix-transformer's own
        // `cmix_orig -h <dict> <order> 0 <tfweights>`
        // (their build_and_construct_comp.sh:117). Accepting BOTH arities keeps
        // every LSTM-era caller working unchanged.
        const four = args.len == 7;
        if (args.len != 6 and !four) return badUsage();
        const ds = try std.fmt.parseInt(i32, args[2], 10);
        const os = try std.fmt.parseInt(i32, args[3], 10);
        const ps = try std.fmt.parseInt(i32, args[4], 10);
        const tw: i32 = if (four) try std.fmt.parseInt(i32, args[5], 10) else 0;
        // Refuse the mismatch rather than writing a trailer the decoder cannot
        // read: a 4-field header from a 12-byte build (or vice versa) mis-slices
        // every segment, and that failure surfaces as a corrupt decode.
        if (four != (se.HEADER_SIZE == 16)) {
            std.debug.print(
                "-h: arity/build mismatch — {d} size fields but HEADER_SIZE={d}. " ++
                    "A 4-field header needs a -Dtransformer build and vice versa.\n",
                .{ @as(u8, if (four) 4 else 3), se.HEADER_SIZE },
            );
            return error.HeaderArityMismatch;
        }
        try se.writeHeaderFile(args[args.len - 1], .{
            .dict_size = ds,
            .new_article_order_size = os,
            .decomp_input_size = ps,
            .tfweights_size = tw,
        });
        return;
    }

    // ---- `--extract-assets`: selfextract_comp's dict/order materialization only ----
    // (self_extract.h:32-93, minus the compress). Reads own tail, slices dict.comp /
    // order.comp, self-decompresses each with the no-dict codec. Lets the packaging be
    // verified byte-for-byte against english.dic / new_article_order WITHOUT the coder.
    if (std.mem.eql(u8, m, "--extract-assets")) {
        if (args.len != 4) return badUsage();
        const image = try readSelfImage(gpa);
        defer gpa.free(image);
        const segs = try se.selfextractComp(image); // :58 slice math
        // transformer-era: this verification op decodes with the SAME predictor the judged ops
        // use, so it needs the weights too — and, unlike `-c`, it is running on a
        // fully assembled comp9, whose tail already carries them. Take them from
        // there, exactly as `selfDecode` and `-e` do, so the ONE mandatory packaging
        // gate in construct_ship.sh depends on nothing outside the artifact it is
        // checking. `image` outlives every coder call in this block.
        if (comptime tfmod.enabled) {
            if (segs.tfweights_comp.len == 0) return error.MissingEmbeddedWeights;
            tfmod.Transformer.embedded_blob = segs.tfweights_comp;
        }
        const dict = try rl.runDecompression(gpa, segs.dict_comp, null, seed); // :84 ./cmix -d .dict.comp .dict
        defer gpa.free(dict);
        // Under -Ds7order this also inverts the s7 container, so the written
        // file is the RAW order and construct_ship.sh's byte-compare against
        // the original asset verifies the WHOLE chain (segment -> -d -> s7⁻¹).
        const order = try decodeOrderAsset(gpa, segs.order_comp, seed); // :76 ./cmix -d .order.comp .order
        defer gpa.free(order);
        try writeFile(args[2], dict);
        try writeFile(args[3], order);
        // Under -Dderivdict a comp9's first slot is the Arm E′ RECIPE, not
        // english.dic — say so, or a packaging `cmp` against the wrong golden
        // reads as corruption.
        std.debug.print("--extract-assets: {s}={d} order={d} (image={d})\n", .{
            if (comptime build_options.derivdict) "recipe(E')" else "dict",
            dict.len,
            order.len,
            image.len,
        });
        return;
    }

    // ---- `--derivdict-build` (comptime-dead unless -Dderivdict): Arm E′
    // dict-builder — derive english.dic from the corpus + recipe blob instead
    // of carrying it. Originally the fee-measurement vehicle (plan E7); now
    // routed through the SAME `rebuildShipDict` the wired `-e` path calls, so
    // this diagnostic is a real test of the shipped code (golden identity
    // assert included) rather than of a parallel copy of it.
    if (comptime build_options.derivdict) {
        if (std.mem.eql(u8, m, "--derivdict-build")) {
            if (args.len != 5) return badUsage();
            const corpus = try readFile(gpa, args[2]);
            defer gpa.free(corpus);
            const recipe = try readFile(gpa, args[3]);
            defer gpa.free(recipe);
            const dict = try derivdict.rebuildShipDict(gpa, corpus, recipe);
            defer gpa.free(dict);
            try writeFile(args[4], dict);
            std.debug.print("--derivdict-build: corpus={d} recipe={d} -> dict={d} (golden identity ASSERTED)\n", .{ corpus.len, recipe.len, dict.len });
            return;
        }
    }

    // ---- `--s7-encode` (comptime-dead unless -Ds7order): asset-mint mode —
    // re-encode the raw article order into the s7_runlabel container
    // (src/prepr/s7_order.zig). construct_ship.sh feeds the RESULT to `-c`,
    // so the in-image comp_order segment carries compressed s7 text. The
    // shipped inverse is asserted HERE, at mint time: refuse to emit an asset
    // whose decode is not byte-identical to the input.
    if (comptime build_options.s7order) {
        if (std.mem.eql(u8, m, "--s7-encode")) {
            if (args.len != 4) return badUsage();
            const raw = try readFile(gpa, args[2]);
            defer gpa.free(raw);
            const enc = try s7.encode(gpa, raw);
            defer gpa.free(enc);
            const back = try s7.decode(gpa, enc);
            defer gpa.free(back);
            if (!std.mem.eql(u8, back, raw)) {
                std.debug.print("--s7-encode: FATAL — s7 inverse is NOT byte-identical to the input order ({d} vs {d} bytes)\n", .{ back.len, raw.len });
                return error.S7RoundtripMismatch;
            }
            try writeFile(args[3], enc);
            std.debug.print("--s7-encode: {d} -> {d} bytes (inverse verified byte-identical)\n", .{ raw.len, enc.len });
            return;
        }
    }

    if (m.len != 2 or m[0] != '-') {
        return badUsage();
    }

    switch (m[1]) {
        // No-dict codec: exactly `cmix -c/-d <in> <out>` (runner.cpp:362-363,416)
        // — the store path construct.sh uses to self-compress english.dic / order.
        'c', 'd' => {
            // LSTM-era:  -c/-d <in> <out>                 (4 argv)
            // transformer-era:  -c/-d <in> <out> [<tfweights>]   (5 argv, EXPLICIT weights)
            //
            // ⛔ WHY THE EXTRA ARGUMENT EXISTS. This is the ONLY codec entry with no
            // artifact tail to read: `construct_ship.sh` runs it on the bare packed
            // engine to self-compress english.dic and the order asset, BEFORE comp9 is
            // concatenated. Under -Dtransformer it therefore used to fall through to the
            // build-time `-Dtransformer-weights` string, which was an absolute dev-box
            // path, and a committee rebuild from source died right here with
            // `weights_io: cannot open <path>`.
            // Handing the path in as an argument makes the construct independent of both
            // the build machine and the caller's cwd. Arity 4 stays valid (the relative
            // `-Dtransformer-weights` default then applies), and on LSTM-era the fifth argv
            // is refused exactly as before, because the block below is comptime-dead.
            if (comptime tfmod.enabled) {
                if (args.len != 4 and args.len != 5) return badUsage();
                if (args.len == 5) tfmod.Transformer.weights_path_override = args[4];
            } else {
                if (args.len != 4) return badUsage();
            }
            const in = try readFile(gpa, args[2]);
            defer gpa.free(in);
            const out = if (m[1] == 'c')
                try rl.runCompression(gpa, in, null, true, seed)
            else
                try rl.runDecompression(gpa, in, null, seed);
            defer gpa.free(out);
            try writeFile(args[3], out);
            std.debug.print("{s} seed={d}: {d} -> {d} bytes\n", .{ m, seed, in.len, out.len });
        },

        // Legacy FORM-2 experiment: produce the coded payload only. This is not
        // the Form-1 submission path; construct_form1.sh invokes `-e` below.
        'E' => {
            if (args.len != 4) return badUsage();
            // Under Arm E′ this path would decode the RECIPE and hand it to the
            // coder as if it were english.dic — a silently wrong payload. Fail
            // loudly instead of extending Form-2, which is a
            // DEFECT to remove, not a fallback to lean on.
            if (comptime build_options.derivdict) {
                std.debug.print("-E: unsupported under -Dderivdict — comp9's first tail segment is the Arm E' RECIPE, not dict.comp. Form-2 is not our submission form; use -e.\n", .{});
                return error.FormTwoUnderDerivDict;
            }
            ppmd_mod.setEvictInterval(SHIP_EVICT_BYTES); // encode: TIGHT (gate-required, see above)
            const image = try readSelfImage(gpa);
            const segs = try se.selfextractComp(image);
            const dict = try rl.runDecompression(gpa, segs.dict_comp, null, seed);
            defer gpa.free(dict);
            const order = try decodeOrderAsset(gpa, segs.order_comp, seed);
            defer gpa.free(order);
            gpa.free(image);

            const enwik9 = try readFile(gpa, args[2]);
            const enwik9_len = enwik9.len;
            const payload = try rl.runCompressionE(gpa, enwik9, dict, order, seed);
            defer gpa.free(payload);
            try writeFile(args[3], payload);
            std.debug.print("-E seed={d}: enwik9={d} -> archive9.bhm={d} (legacy Form-2 payload; NOT prize-eligible)\n", .{ seed, enwik9_len, payload.len });
        },

        // Full enwik9 compress: materialize dict+order from comp9's own tail,
        // run the -e pipeline, then assemble a physical self-extractor. Legacy
        // r1v1 omits order from archive9; r1v2 explicitly repeats order.comp
        // because its payload restore needs that asset.
        'e' => {
            if (args.len != 4) return badUsage();
            ppmd_mod.setEvictInterval(SHIP_EVICT_BYTES); // encode: TIGHT (gate-required, see above)
            const image = try readSelfImage(gpa);
            defer gpa.free(image);
            const segs = try se.selfextractComp(image);

            // transformer-era: the JUDGED COMPRESSION OP runs on a judge machine too, so it
            // may no more resolve `-Dtransformer-weights` than the decoder may.
            // comp9 carries the blob in its own tail (`... comp_order ++
            // comp_tfweights ++ header`) — use it, exactly as `selfDecode` does.
            // Without this, `-e` falls back to the ABSOLUTE build-time path and
            // panics anywhere that path does not exist. `image` outlives every
            // coder call in this branch, so the slice stays valid.
            if (comptime tfmod.enabled) {
                if (segs.tfweights_comp.len == 0) return error.MissingEmbeddedWeights;
                tfmod.Transformer.embedded_blob = segs.tfweights_comp;
            }

            // Ownership of enwik9 passes to runCompressionE (freed there pre-coder;
            // the ~9.5 GB coder predictor must not coexist with the 1 GB input).
            //
            // MOVED EARLIER (was after both asset decodes) because Arm E′ needs the
            // corpus to build the dictionary. Deliberately the ONLY change to the
            // stock path: the two asset decodes keep their original order
            // (dict, then order) so the stock sequence of CODEC calls is unchanged
            // — a file read moved earlier cannot alter a codec output, whereas
            // swapping two coder invocations is a change no test on this tree
            // covers (the verify ritual drives `runner_lex`, not `zmix_ship -e`).
            // Cost: enwik9 is resident across the asset decodes, ~7 GB peak
            // against the ~9.5 GB coder peak, so it does not move MaxRSS.
            const enwik9 = try readFile(gpa, args[2]);
            const enwik9_len = enwik9.len;

            // ---- the dictionary: CARRIED (stock) or DERIVED (Arm E′) ----------
            // Form-1 charges every carried asset TWICE (once in comp9, once in
            // archive9) but a DERIVED asset only once, because only the
            // ENCODER has the corpus to derive from — at decode time enwik9 is
            // the output, not an input. So the asymmetric split is forced and is
            // exactly where the +32,034 B comes from:
            //
            //   comp9        = bin ++ R_E′.comp (~65.8 KB) ++ order.comp ++ hdr
            //   archive9 = bin ++ dict.comp (100,788)  ++ payload    ++ hdr
            //
            // i.e. comp9's dict slot holds the RECIPE, and `-e` must (1) decode
            // the recipe, (2) rebuild english.dic bit-exactly from the corpus,
            // (3) self-compress the rebuilt dict with our OWN codec to produce
            // the dict.comp that archive9 carries for the decoder, and
            // (4) code the payload against the rebuilt dict as usual.
            //
            // Step 3 reproduces `construct_ship.sh`'s `-c english.dic dict.comp`
            // INSIDE the timed encode op: same binary, same no-dict store path,
            // same input bytes (asserted in rebuildShipDict) ⇒ same output bytes.
            //
            // WALL/RSS BOOKING OWED, NOT ASSUMED: this moves ~11 s of vocabulary
            // scan + rebuild and one `-c english.dic` (~1.6–5 min measured range)
            // into the judged encode. The budget for it exists — the E4 encode
            // race read 0.8820 (−11.8 %, DECISIVE;
            //   against a 1.049–1.099×
            // requirement — but it must be RACED before recipe entry, and the
            // ~1 GB fold copy + the ~6–7 GB `-c` transient (both below the
            // ~9.5 GB coder peak) owe the imposed-cap reading.
            var derived_dict_comp: ?[]u8 = null;
            defer if (derived_dict_comp) |d| gpa.free(d);
            const dict = if (comptime build_options.derivdict) blk: {
                const recipe = try rl.runDecompression(gpa, segs.dict_comp, null, seed);
                defer gpa.free(recipe);
                const rebuilt = try derivdict.rebuildShipDict(gpa, enwik9, recipe);
                errdefer gpa.free(rebuilt);
                derived_dict_comp = try rl.runCompression(gpa, rebuilt, null, true, seed);
                std.debug.print(
                    "-e derivdict: recipe={d} -> english.dic={d} (golden-identical) -> dict.comp={d}\n",
                    .{ segs.dict_comp.len, rebuilt.len, derived_dict_comp.?.len },
                );
                break :blk rebuilt;
            } else try rl.runDecompression(gpa, segs.dict_comp, null, seed);
            defer gpa.free(dict);

            const order = try decodeOrderAsset(gpa, segs.order_comp, seed);
            defer gpa.free(order);

            // archive9 always carries the REAL compressed dictionary: under
            // Arm E′ that is the one just derived+self-compressed, otherwise the
            // copy comp9 carries verbatim.
            const archive_dict_comp: []const u8 = derived_dict_comp orelse segs.dict_comp;

            const payload = try rl.runCompressionE(gpa, enwik9, dict, order, seed);
            defer gpa.free(payload);

            const archive_order_mode: se.ArchiveOrderMode = if (comptime build_options.r1v2)
                .embedded
            else
                .omitted;
            // transformer-era: carry OUR OWN weights blob through into archive9.
            // Without this the archive's trailer reads tfweights_size=0 and
            // `selfDecode` returns error.MissingEmbeddedWeights — i.e. the
            // no-argument rule-2 contract cannot be satisfied at all.
            const archive9 = try se.assembleArchive9ForMode(
                gpa,
                archive_order_mode,
                segs.decomp_bin,
                archive_dict_comp,
                segs.order_comp,
                payload,
                segs.tfweights_comp,
            );
            defer gpa.free(archive9);
            // The produced file is itself an executable self-extractor (mode 0755).
            {
                const f = try std.fs.cwd().createFile(args[3], .{ .mode = 0o755 });
                defer f.close();
                try f.writeAll(archive9);
            }
            std.debug.print("-e seed={d}: enwik9={d} -> payload={d} archive9={d}\n", .{ seed, enwik9_len, payload.len, archive9.len });
        },

        // Full enwik9 decompress — explicit-output form of the no-args contract.
        'D' => {
            if (args.len != 3) return badUsage();
            ppmd_mod.setEvictInterval(SHIP_EVICT_BYTES); // decode: TIGHT (gate-required, see ship.zig decode comment)
            try selfDecode(gpa, args[2], seed);
        },

        else => return badUsage(),
    }
}
