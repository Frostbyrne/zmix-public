//! Minimal Form-1 Hutter submission prefix (R1v1 only).
//!
//! The SAME executable prefix has two self-selected roles:
//!
//!   comp9:        [prefix][dict.comp][order.comp][trailer payload_size=0]
//!   archive9:     [prefix][dict.comp][payload][trailer payload_size>0]
//!
//! transformer-era (`-Dtransformer`) appends the int4 weights blob as the LAST segment
//! before the trailer in BOTH roles, and the trailer widens to 16 bytes to
//! carry `tfweights_size` (src/prepr/self_extract.zig; upstream
//! fx2-cmix-transformer's own `-h <dict> <order> 0 <tfweights>` form):
//!
//!   comp9:        [prefix][dict.comp][order.comp][tfweights][trailer(16)]
//!   archive9:     [prefix][dict.comp][payload][tfweights][trailer(16)]
//!
//! Every offset in this file is computed from the TAIL BACKWARDS, so omitting
//! `tfweights_size` from the arithmetic does not fail loudly — it silently
//! slices the dictionary out of the middle of the weights blob. That is exactly
//! what this root did until ; see
//!
//! `src/ship.zig` is the reference implementation of the same container: its
//! `selfDecode` (145-161), `-h` (293-323), `--extract-assets` (341-344) and
//! `-e` (499-502, :578-590) each do the corresponding thing. The two roots must
//! agree STRUCTURALLY; they differ only in that this one has no CLI, so it needs
//! neither `-h` nor an explicit-path weights source, which is what
//! `weights_embedded_only` below states.
//!
//! Running comp9 reads fixed `enwik9` and writes fixed `archive9`. Running
//! archive9 writes fixed `data9`. The trailer selects the role, so the
//! submitted prefix needs no CLI, Form-2 path, asset encoder, or diagnostics.
//! `construct_form1.sh --minimal-prefix` assembles comp9 by replacing only the
//! full helper's executable prefix and preserving its exact dict/order/header
//! tail; it then invokes this program with no arguments. The full helper stays
//! outside the submitted pair as the reproducible asset/header construction
//! tool.

const std = @import("std");
const build_options = @import("build_options");
const form1_options = @import("form1_options");
const rl = @import("runner_lex.zig");
const se = @import("prepr/self_extract.zig");
const ppmd = @import("models/ppmd.zig");
const derivdict = @import("derivdict.zig");
const s7 = @import("prepr/s7_order.zig");
const tfmod = @import("mixer/transformer.zig");
const SHIP_EVICT_BYTES: u64 = @import("wallpareto_options").ship_evict_bytes;

pub const panic = std.debug.simple_panic;
pub const zmix_hermetic = true;

/// This root has NO CLI, so `Transformer.weights_path_override` can never be set
/// here and `-Dtransformer-weights` is unreachable-by-construction: both roles
/// below slice the blob out of their OWN tail (`compress` and `decompress`), and
/// a build whose artifact carries no blob is refused rather than served from a
/// build-time path. Declaring this compiles the path-loading fallback in
/// `predictor_lex.zig` away for this root only — `src/ship.zig` does NOT declare
/// it, because its `-c`/`-d` asset-compression ops run on the bare packed engine
/// before comp9 exists and legitimately take an explicit path argument.
///
/// Same root-decl pattern as `zmix_hermetic` (predictor_lex.zig:164-165), so it
/// costs no build option and cannot move any other root's bytes.
pub const weights_embedded_only = true;

comptime {
    _ = @import("detm_math.zig");
    if (!@import("builtin").link_libc)
        @compileError("minimal Form-1 uses the record-style glibc ABI; pass -Dglibc=true");
    if (build_options.r1v2)
        @compileError("minimal Form-1 omits the decode-side order asset; build R1v1");
}

const seed: i32 = 923;

// -Dform1-malloc: the dense malloc/posix_memalign allocator below. DEFAULT OFF.
//
// It packs 2,960–3,176 B smaller than the stock GeneralPurposeAllocator
// (128,236 vs 131,196 on trunk `9409a78`; 128,164 vs 131,340 on `a1fe873`) = a
// further −5,920…−6,352 B of S, since Form-1 charges the prefix twice — but
// `s1-form1-allocator-gate-result-aug-01.md` measured it at
// +98,580…125,120 kB MaxRSS, and the decode RAM gate passes today by only
// 178,969 kB (1.83%). So the
// size saving and the RSS cost were welded together in one file, and the whole
// −14,384 B lever was unaffordable because of the half of it that is not.
//
// Unwelding them is the point of this knob: at the default the minimal prefix
// runs the SAME `std.heap.GeneralPurposeAllocator(.{})` that `src/ship.zig`
// already ships, so the CLI-removal saving (−8,032 B of S on `a1fe873`, −8,640
// on trunk `9409a78`; see ) carries no
// allocator delta to price at all, while the malloc arm stays one flag away for
// the day decode headroom exceeds ~250 MB (the reopen trigger in
// Comptime-dead at the default: nothing
// below this comment is analysed or emitted unless the flag is engaged, and the
// option lives in form1's OWN options module so adding it cannot move
// `zmix_ship`, `zmix` or any archive.
const ship_allocator: std.mem.Allocator = .{
    .ptr = undefined,
    .vtable = &.{
        .alloc = alloc,
        .resize = resize,
        .remap = remap,
        .free = free,
    },
};

fn alloc(_: *anyopaque, len: usize, alignment: std.mem.Alignment, _: usize) ?[*]u8 {
    const align_bytes = alignment.toByteUnits();
    if (align_bytes <= @alignOf(std.c.max_align_t))
        return @ptrCast(std.c.malloc(len));
    var ptr: ?*anyopaque = undefined;
    if (std.c.posix_memalign(&ptr, align_bytes, len) != 0) return null;
    return @ptrCast(ptr);
}

fn resize(_: *anyopaque, buf: []u8, _: std.mem.Alignment, new_len: usize, _: usize) bool {
    return new_len <= buf.len;
}

fn remap(_: *anyopaque, buf: []u8, _: std.mem.Alignment, new_len: usize, _: usize) ?[*]u8 {
    return if (new_len <= buf.len) buf.ptr else null;
}

fn free(_: *anyopaque, buf: []u8, _: std.mem.Alignment, _: usize) void {
    std.c.free(buf.ptr);
}

fn enableFtzDaz() void {
    if (@import("builtin").cpu.arch == .x86_64) {
        var csr: u32 = undefined;
        asm volatile ("stmxcsr (%[p])"
            :
            : [p] "r" (&csr),
            : .{ .memory = true });
        csr |= 0x8040;
        asm volatile ("ldmxcsr (%[p])"
            :
            : [p] "r" (&csr),
            : .{ .memory = true });
    }
}

fn readHeader(f: std.fs.File, file_size: u64) !se.HeaderInfo {
    if (file_size < se.HEADER_SIZE) return error.ImageTooSmall;
    var bytes: [se.HEADER_SIZE]u8 = undefined;
    if (try f.preadAll(&bytes, file_size - se.HEADER_SIZE) != se.HEADER_SIZE)
        return error.ImageTooSmall;
    return se.parseHeaderBytes(&bytes);
}

fn segmentSizes(header: se.HeaderInfo, file_size: u64, second_size: i32) !struct {
    prefix: u64,
    dict: u64,
    second: u64,
    /// transformer-era: the int4 weights blob, which is TAIL OVERHEAD in both roles (it
    /// sits between the last variable segment and the trailer). 0 on LSTM-era, so
    /// the LSTM-era arithmetic below is byte-for-byte unchanged.
    tfweights: u64,
} {
    if (header.dict_size < 0 or second_size < 0) return error.CorruptHeader;
    const dict_size: u64 = @intCast(header.dict_size);
    const second: u64 = @intCast(second_size);
    // transformer-era: omitting this term makes `prefix` too LARGE by |blob| and shifts
    // every downstream offset, which is a corrupt decode rather than an error —
    // `se.locateArchive` and `se.selfextract{Comp,Decomp}` carry the identical
    // term for the same reason.
    if (comptime tfmod.enabled) {
        if (header.tfweights_size < 0) return error.CorruptHeader;
    }
    const tfweights: u64 = if (comptime tfmod.enabled) @intCast(header.tfweights_size) else 0;
    // Every size originates as a non-negative i32, so their sum plus the
    // 12/16-byte trailer cannot overflow u64.
    const overhead = dict_size + second + tfweights + se.HEADER_SIZE;
    if (overhead > file_size) return error.CorruptHeader;
    return .{
        .prefix = file_size - overhead,
        .dict = dict_size,
        .second = second,
        .tfweights = tfweights,
    };
}

/// transformer-era: hand the predictor the weights blob out of our OWN tail before it is
/// built — the blob is the LAST segment before the trailer in BOTH roles, so its
/// offset is the same expression regardless of what the middle segment holds.
///
/// Mirrors `src/ship.zig:145-161` exactly, including the refusal: a transformer
/// build whose artifact carries no blob must fail loudly here rather than fall
/// through to `-Dtransformer-weights` (a build-time path that need not exist on
/// a judge machine — the defect that made an earlier build unsubmittable). `pub const
/// weights_embedded_only` above compiles that fallback away for this root, so
/// the refusal is the ONLY remaining outcome.
///
/// Leaks by design for the process lifetime: the transformer is constructed once
/// and lives until exit.
fn loadEmbeddedWeights(
    gpa: std.mem.Allocator,
    f: std.fs.File,
    file_size: u64,
    header: se.HeaderInfo,
) !void {
    if (comptime !tfmod.enabled) return;
    if (header.tfweights_size <= 0) return error.MissingEmbeddedWeights;
    const tw: usize = @intCast(header.tfweights_size);
    const off = file_size - se.HEADER_SIZE - tw;
    const blob = try gpa.alloc(u8, tw);
    if (try f.preadAll(blob, off) != tw) return error.CorruptHeader;
    tfmod.Transformer.embedded_blob = blob;
}

fn writeArchive(
    prefix: []const u8,
    dict_comp: []const u8,
    payload: []const u8,
    order_comp_size: i32,
    /// transformer-era: OUR OWN blob, sliced out of comp9's tail. EMPTY on LSTM-era, where the
    /// two `writeAll`s below are a no-op and `tfweights_size` is not serialised
    /// at all (`se.HEADER_SIZE` is 12), so the LSTM-era archive is unchanged.
    tfweights_comp: []const u8,
) !void {
    if (dict_comp.len > std.math.maxInt(i32) or payload.len > std.math.maxInt(i32) or
        tfweights_comp.len > std.math.maxInt(i32))
        return error.FileTooBig;
    // Built through `se.writeHeaderBytes` rather than hand-rolled so the trailer
    // WIDTH and the field order come from the one place that defines them; a
    // 3-field trailer on a 16-byte build (or vice versa) mis-slices every
    // segment, and `src/ship.zig:309-316` refuses exactly that mismatch.
    const trailer = se.writeHeaderBytes(.{
        .dict_size = @intCast(dict_comp.len),
        .new_article_order_size = order_comp_size,
        .decomp_input_size = @intCast(payload.len),
        .tfweights_size = @intCast(tfweights_comp.len),
    });
    const out = try std.fs.cwd().createFile("archive9", .{ .mode = 0o755 });
    defer out.close();
    try out.writeAll(prefix);
    try out.writeAll(dict_comp);
    try out.writeAll(payload);
    // transformer-era layout: [ prefix ][ dict ][ payload ][ tfweights ][ trailer ] —
    // mirrors `se.assembleArchive9` and comp9's own
    // `... comp_order ++ comp_tfweights ++ header`. Without this the archive's
    // trailer would read tfweights_size = 0 and the no-argument decode could not
    // load weights at all (rule 2 unsatisfiable).
    try out.writeAll(tfweights_comp);
    try out.writeAll(&trailer);
}

fn compress(gpa: std.mem.Allocator, f: std.fs.File, file_size: u64, header: se.HeaderInfo) !void {
    if (header.new_article_order_size < 0) return error.CorruptHeader;
    const sizes = try segmentSizes(header, file_size, header.new_article_order_size);

    const image = try gpa.alloc(u8, @intCast(file_size));
    defer gpa.free(image);
    if (try f.preadAll(image, 0) != image.len) return error.UnexpectedEndOfFile;

    const prefix_end: usize = @intCast(sizes.prefix);
    const dict_end: usize = @intCast(sizes.prefix + sizes.dict);
    const second_end: usize = @intCast(sizes.prefix + sizes.dict + sizes.second);
    const tfw_end: usize = second_end + @as(usize, @intCast(sizes.tfweights));
    const prefix = image[0..prefix_end];
    const dict_comp = image[prefix_end..dict_end];
    const order_comp = image[dict_end..second_end];
    // transformer-era: comp9's own blob, which archive9 must carry verbatim. Empty on
    // LSTM-era (`sizes.tfweights` is a comptime 0 there, so `tfw_end == second_end`).
    // `image` is freed only at the end of this function, so the slice outlives
    // both the coder calls that need the weights and `writeArchive`.
    const tfweights_comp = image[second_end..tfw_end];

    // BEFORE the first coder call: `runDecompression` below constructs the
    // predictor, which is where the transformer is built. Setting the blob
    // afterwards would be too late and would fall through to the dev path.
    if (comptime tfmod.enabled) {
        if (tfweights_comp.len == 0) return error.MissingEmbeddedWeights;
        tfmod.Transformer.embedded_blob = tfweights_comp;
    }

    // runCompressionE owns and frees the 1 GB input before constructing the
    // predictor, keeping the judged encode under the RSS gate. Read EARLIER than
    // before (Arm E′ needs the corpus to build the dictionary); the two asset
    // decodes keep their original order, so the stock sequence of codec calls is
    // unchanged — see the matching note in src/ship.zig.
    const enwik9 = try std.fs.cwd().readFileAlloc(gpa, "enwik9", 1 << 34);

    // Arm E′ (-Dderivdict): this root's first tail segment is the RECIPE, so
    // rebuild english.dic from the corpus (golden-identity asserted) and
    // self-compress it for the archive the decoder will read. Same contract as
    // src/ship.zig's `-e`; comptime-dead at default, so the minroot's measured
    // packed size is unaffected unless the flag is engaged. Wired here rather
    // than left to a compile error so the minroot prefix and Arm E′ COMPOSE
    // instead of excluding each other — and, more importantly, so this path
    // cannot silently feed the recipe to the coder as if it were a dictionary.
    var derived_dict_comp: ?[]u8 = null;
    defer if (derived_dict_comp) |d| gpa.free(d);
    const dict = if (comptime build_options.derivdict) blk: {
        const recipe = try rl.runDecompression(gpa, dict_comp, null, seed);
        defer gpa.free(recipe);
        const rebuilt = try derivdict.rebuildShipDict(gpa, enwik9, recipe);
        errdefer gpa.free(rebuilt);
        derived_dict_comp = try rl.runCompression(gpa, rebuilt, null, true, seed);
        break :blk rebuilt;
    } else try rl.runDecompression(gpa, dict_comp, null, seed);
    defer gpa.free(dict);

    // The order slot holds the s7_runlabel re-encoding under -Ds7order (same
    // contract as ship.zig's decodeOrderAsset — the s7 container must be
    // decoded back to the EXACT raw decimal order text before the article
    // reorder consumes it). Handing runCompressionE the raw s7 text produced
    // a 1.6 GB post-WRT stream and the R1LengthMismatch abort on the first
    // real no-arg e9 encode of this root (g62-e9-x2) — this path
    // had never run at e9 before, and only e9 can reach it.
    const order_raw = try rl.runDecompression(gpa, order_comp, null, seed);
    const order = if (comptime build_options.s7order) blk: {
        defer gpa.free(order_raw);
        break :blk try s7.decode(gpa, order_raw);
    } else order_raw;
    defer gpa.free(order);

    const payload = try rl.runCompressionE(gpa, enwik9, dict, order, seed);
    defer gpa.free(payload);
    try writeArchive(
        prefix,
        derived_dict_comp orelse dict_comp,
        payload,
        header.new_article_order_size,
        tfweights_comp,
    );
}

fn decompress(gpa: std.mem.Allocator, f: std.fs.File, file_size: u64, header: se.HeaderInfo) !void {
    const sizes = try segmentSizes(header, file_size, header.decomp_input_size);
    // transformer-era, BEFORE the first coder call (see loadEmbeddedWeights). Not sliced
    // from an in-memory image because this role never reads one: the ~109 MB
    // payload is STREAMED so the archive image never sits under the ~9.5 GB
    // coder peak (src/ship.zig:113-118). `sizes.tfweights` is the same value; the
    // pread offset is expressed from the tail so the two cannot disagree.
    try loadEmbeddedWeights(gpa, f, file_size, header);
    const dict_comp = try gpa.alloc(u8, @intCast(sizes.dict));
    if (try f.preadAll(dict_comp, sizes.prefix) != dict_comp.len)
        return error.UnexpectedEndOfFile;
    const dict = try rl.runDecompression(gpa, dict_comp, null, seed);
    gpa.free(dict_comp);
    defer gpa.free(dict);

    const enwik9 = try rl.runDecompressionDStreamed(
        gpa,
        f,
        sizes.prefix + sizes.dict,
        sizes.second,
        dict,
        null,
        seed,
    );
    defer gpa.free(enwik9);
    const out = try std.fs.cwd().createFile("data9", .{});
    defer out.close();
    try out.writeAll(enwik9);
}

// The stock allocator, spelled exactly as src/ship.zig spells it. Container
// scope rather than a `run` local only so the malloc arm can leave it
// unreferenced (and therefore unanalysed) instead of tripping unused-local.
var gpa_impl = std.heap.GeneralPurposeAllocator(.{}){};

fn run() !void {
    enableFtzDaz();
    const gpa = if (comptime form1_options.malloc_allocator)
        ship_allocator
    else
        gpa_impl.allocator();

    const f = try std.fs.openFileAbsolute("/proc/self/exe", .{});
    defer f.close();
    const file_size = try f.getEndPos();
    const header = try readHeader(f, file_size);

    if (header.decomp_input_size == 0) {
        ppmd.setEvictInterval(SHIP_EVICT_BYTES); // encode: TIGHT — gate-required (see src/ship.zig cadence comment; projected e9 MaxRSS @2000 FAILS by 0.52-1.33 %)
        try compress(gpa, f, file_size, header);
    } else {
        ppmd.setEvictInterval(SHIP_EVICT_BYTES); // decode: TIGHT — gate-required (see src/ship.zig decode-cadence comment; uncapped e9 @25k FAILS MaxRSS by ~3.1 GB)
        try decompress(gpa, f, file_size, header);
    }
}

pub fn main() void {
    run() catch std.process.exit(1);
}
