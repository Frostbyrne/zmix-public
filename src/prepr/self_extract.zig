//! Port of cmix-lex's self-extracting-archive packaging
//! (`ref/cmix-lex/src/readalike_prepr/self_extract.h`, 145 lines).
//!
//! This is the PACKAGING layer of the Hutter-prize submission, NOT a ratio
//! transform. It defines how the single shipped `cmix`/`archive9` binary embeds
//! its own compressed dictionary + article-order payloads in its tail, and how
//! the running (de)compressor locates and slices them back out of itself using a
//! fixed-size trailer (`HeaderInfo`, the last 12 bytes of the file).
//!
//! The C++ operates directly on files via fopen/fread/fwrite/system. This port
//! keeps the exact byte-level format (trailer layout + slice math) but exposes it
//! as pure slice functions over an in-memory image so it is deterministic and
//! unit-testable; thin file-backed wrappers mirror the original I/O for the
//! runner. NO existing zmix source is modified.
//!
//! Ground-truth C++ line refs are in comments (self_extract.h; assembly in
//! runner.cpp:453-481 and construct.sh).
//!
//! ---------------------------------------------------------------------------
//! Archive layouts (byte concatenation; trailer is always the last 12 bytes):
//!
//!   `cmix`     (compress side, built by construct.sh / `-h`):
//!       [ cmix_orig ][ comp_dict ][ comp_order ][ HeaderInfo(12) ]
//!       HeaderInfo = { dict_size=|comp_dict|, new_article_order_size=|comp_order|,
//!                      decomp_input_size=0 }
//!
//!   `archive9` (legacy/r1v1 decompress side, built by the `-e` branch):
//!       [ .decomp_bin ][ .dict.comp ][ cmix_output ][ HeaderInfo(12) ]
//!       HeaderInfo = { dict_size=|comp_dict|, new_article_order_size=|comp_order|,
//!                      decomp_input_size=|cmix_output| }
//!       NOTE: the order payload is NOT in archive9 (decode un-reorders by id via
//!       article_reorder.sort, so it never needs the embedded order file). The
//!       second embedded segment here is the arith-coded enwik9 payload, sliced by
//!       `decomp_input_size`, NOT `new_article_order_size`.
//!
//!   `archive9` (explicit r1v2 physical-correctness alternate):
//!       [ .decomp_bin ][ .dict.comp ][ .order.comp ][ cmix_output ][ HeaderInfo(12) ]
//!       R1ORD4 restore needs the raw order asset, so this form repeats and counts
//!       order.comp rather than reading any outside file.
//! ---------------------------------------------------------------------------

const std = @import("std");

/// `struct HeaderInfo` (self_extract.h:9-13). Three native C `int`s = 12 bytes,
/// x86-64 little-endian, no padding (verified against a g++ oracle: sizeof==12,
/// fields serialized LE in declaration order).
pub const HeaderInfo = struct {
    dict_size: i32,
    new_article_order_size: i32,
    decomp_input_size: i32,
    /// transformer-era ONLY: length of the appended int4 transformer weights blob
    /// (`.tfwc2`). Zero and NOT SERIALISED unless `-Dtransformer` is set, so the
    /// LSTM-era trailer stays byte-identical at 12 bytes.
    ///
    /// Matches fx2-cmix-transformer's own four-field form, which their
    /// `build_and_construct_comp.sh:117` writes as
    ///   `cmix_orig -h <dict_len> <order_len> 0 <tfweights_len>`
    /// i.e. the blob length is the FOURTH field and the third stays 0.
    tfweights_size: i32 = 0,
};

/// `sizeof(HeaderInfo)` — the fixed trailer length (self_extract.h uses it as the
/// tail offset in every slice computation).
///
/// ⚠ 35 sites across the tree reference this symbolically, so widening it here
/// propagates everywhere automatically — which is exactly why it must stay
/// CONDITIONAL. An LSTM-era artifact read with a 16-byte trailer mis-slices every
/// segment, and the failure is a corrupt decode rather than an error.
pub const HEADER_SIZE: usize = if (transformer_on) 16 else 12;

/// Read from the transformer's OWN options module, never the shared `model_opts`
/// (the own-module rule: every entry in the shared module costs ~40 B of `decomp_bin`,
/// charged twice). Falls back to `false` so non-transformer builds — and any
/// consumer that does not attach the module — keep the 12-byte trailer.
const transformer_on: bool = blk: {
    const opts = @import("transformer_options");
    break :blk @hasDecl(opts, "transformer") and opts.transformer;
};

/// Serialize a HeaderInfo to its exact 12-byte on-disk form (raw `fwrite(&data,
/// 1, sizeof(HeaderInfo), out)`, self_extract.h:15-19). Little-endian ints in
/// declaration order.
pub fn writeHeaderBytes(h: HeaderInfo) [HEADER_SIZE]u8 {
    var buf: [HEADER_SIZE]u8 = undefined;
    std.mem.writeInt(i32, buf[0..4], h.dict_size, .little);
    std.mem.writeInt(i32, buf[4..8], h.new_article_order_size, .little);
    std.mem.writeInt(i32, buf[8..12], h.decomp_input_size, .little);
    if (transformer_on) std.mem.writeInt(i32, buf[12..16], h.tfweights_size, .little);
    return buf;
}

/// Parse a HeaderInfo from its 12-byte on-disk form (raw `fread(&data, 1,
/// sizeof(HeaderInfo), in)`, self_extract.h:21-25). Inverse of `writeHeaderBytes`.
pub fn parseHeaderBytes(buf: *const [HEADER_SIZE]u8) HeaderInfo {
    return .{
        .dict_size = std.mem.readInt(i32, buf[0..4], .little),
        .new_article_order_size = std.mem.readInt(i32, buf[4..8], .little),
        .decomp_input_size = std.mem.readInt(i32, buf[8..12], .little),
        .tfweights_size = if (transformer_on)
            std.mem.readInt(i32, buf[12..16], .little)
        else
            0,
    };
}

/// Read the trailing `HeaderInfo` from a whole-file image (the `memcpy(&header,
/// p1 + fsize - sizeof(HeaderInfo), ...)` step, self_extract.h:51 / :115-117).
pub fn readTrailer(image: []const u8) error{ImageTooSmall}!HeaderInfo {
    if (image.len < HEADER_SIZE) return error.ImageTooSmall;
    const tail = image[image.len - HEADER_SIZE ..][0..HEADER_SIZE];
    return parseHeaderBytes(tail);
}

/// The three embedded segments of the compress-side `cmix` binary, as slices into
/// the original image (no copies).
pub const CompSegments = struct {
    /// The actual (de)compressor binary — `.decomp_bin` (self_extract.h:61-63).
    decomp_bin: []const u8,
    /// cmix-compressed english.dic — `.dict.comp` (self_extract.h:66-68).
    dict_comp: []const u8,
    /// cmix-compressed new_article_order — `.new_article_order.comp` (72-74).
    order_comp: []const u8,
    /// transformer-era ONLY: the int4 transformer weights blob (`FX2TFWC2`), sitting after
    /// `order_comp` and before the trailer. EMPTY on LSTM-era, where
    /// `header.tfweights_size` is 0 and the trailer carries no fourth field.
    ///
    /// ⚠ This is what the shipped decoder must hand to `cpp_infer` — NOT an
    /// absolute path. `-Dtransformer-weights` reads a path and is a DEV-ONLY
    /// affordance; a shipped binary that resolves an absolute path is the exact
    /// defect that made an earlier build unsubmittable (it panicked on a judge box after
    /// passing every ratio, RAM and wall gate).
    tfweights_comp: []const u8,
    /// The raw 12-byte trailer bytes (written to `test.dat`, :50-53).
    trailer_bytes: []const u8,
    /// Parsed trailer.
    header: HeaderInfo,
};

/// Port of `selfextract_comp` (self_extract.h:32-93), split into pure slicing.
/// Given the whole `cmix` image, recover the segment boundaries the C++ writes to
/// `.decomp_bin` / `.dict.comp` / `.new_article_order.comp`.
///
///   decompressor_binary_size = fsize - dict_size - new_article_order_size
///                              - sizeof(HeaderInfo)                 (58)
///
/// The C++ then `system("./cmix -d …")`-decompresses the order and dict; that
/// runtime step is out of scope for the pure transform (it invokes the coder) and
/// is handled by the runner. This returns only the deterministic byte layout.
pub fn selfextractComp(image: []const u8) error{ ImageTooSmall, CorruptHeader }!CompSegments {
    if (image.len < HEADER_SIZE) return error.ImageTooSmall;
    const header = try readTrailer(image);
    if (header.dict_size < 0 or header.new_article_order_size < 0)
        return error.CorruptHeader;
    const dsz: usize = @intCast(header.dict_size);
    const osz: usize = @intCast(header.new_article_order_size);
    // transformer-era: the int4 weights blob sits AFTER comp_order and BEFORE the trailer,
    // matching fx2-cmix-transformer's own layout
    //   `cat cmix_orig comp_dict comp_order comp_tfweights header.dat > cmix`
    // (build_and_construct_comp.sh:120). Zero on LSTM-era, so the slice math below
    // is unchanged there.
    if (transformer_on and header.tfweights_size < 0) return error.CorruptHeader;
    const tsz: usize = if (transformer_on) @intCast(header.tfweights_size) else 0;
    const overhead = dsz + osz + tsz + HEADER_SIZE;
    if (overhead > image.len) return error.CorruptHeader;
    const bin_size = image.len - overhead; // :58
    const o_end = bin_size + dsz + osz;
    return .{
        .decomp_bin = image[0..bin_size], // :61-63
        .dict_comp = image[bin_size .. bin_size + dsz], // :66-68
        .order_comp = image[bin_size + dsz .. o_end], // :72-74
        .tfweights_comp = image[o_end .. o_end + tsz], // transformer-era; empty on LSTM-era
        .trailer_bytes = image[image.len - HEADER_SIZE ..], // :50-53
        .header = header,
    };
}

/// The two embedded payload segments of the decompress-side `archive9`, as slices.
pub const DecompSegments = struct {
    /// The (de)compressor binary prefix (not written out by the C++, but implied
    /// by the same slice math; useful for verification).
    decomp_bin: []const u8,
    /// cmix-compressed dictionary — `.dict.comp_decomp` (self_extract.h:124-126).
    dict_comp: []const u8,
    /// The arith-coded enwik9 payload — `.ready4cmix_decomp` (135-137).
    ready4cmix: []const u8,
    /// transformer-era ONLY: the int4 transformer weights blob, sitting AFTER the payload
    /// and BEFORE the trailer. EMPTY on LSTM-era.
    tfweights_comp: []const u8,
    /// Raw 12-byte trailer.
    trailer_bytes: []const u8,
    /// Parsed trailer.
    header: HeaderInfo,
};

/// Port of `selfextract_decomp` (self_extract.h:100-142). Given the whole
/// `archive9` image, recover `.dict.comp_decomp` and `.ready4cmix_decomp`.
///
///   decompressor_binary_size = fsize - dict_size - decomp_input_size
///                              - sizeof(HeaderInfo)                 (122)
///
/// KEY DIFFERENCE vs selfextractComp: the second segment is sized by
/// `decomp_input_size` (the coded payload), NOT `new_article_order_size`.
pub fn selfextractDecomp(image: []const u8) error{ ImageTooSmall, CorruptHeader }!DecompSegments {
    if (image.len < HEADER_SIZE) return error.ImageTooSmall;
    const header = try readTrailer(image);
    if (header.dict_size < 0 or header.decomp_input_size < 0)
        return error.CorruptHeader;
    const dsz: usize = @intCast(header.dict_size);
    const psz: usize = @intCast(header.decomp_input_size);
    // transformer-era: the weights blob sits between the payload and the trailer, so it is
    // part of the tail overhead. Omitting it here made `bin_size` too large by
    // |blob| and mis-sliced BOTH the dict and the payload.
    if (transformer_on and header.tfweights_size < 0) return error.CorruptHeader;
    const tsz: usize = if (transformer_on) @intCast(header.tfweights_size) else 0;
    const overhead = dsz + psz + tsz + HEADER_SIZE;
    if (overhead > image.len) return error.CorruptHeader;
    const bin_size = image.len - overhead; // :122
    const p_end = bin_size + dsz + psz;
    return .{
        .decomp_bin = image[0..bin_size],
        .dict_comp = image[bin_size .. bin_size + dsz], // :124-126
        .ready4cmix = image[bin_size + dsz .. p_end], // :135-137
        .tfweights_comp = image[p_end .. p_end + tsz], // transformer-era; empty on LSTM-era
        .trailer_bytes = image[image.len - HEADER_SIZE ..],
        .header = header,
    };
}

/// Assemble the compress-side `cmix` self-extractor image into `out`
/// (construct.sh: `cat cmix_orig comp_dict comp_order header.dat > cmix`, with the
/// `-h` trailer built at runner.cpp:474-480: dict_size=|comp_dict|,
/// new_article_order_size=|comp_order|, decomp_input_size=0).
///
/// Returns the assembled bytes (caller owns). This is the exact inverse of
/// `selfextractComp` — round-tripping any (bin, dict, order) triple.
pub fn assembleCmix(
    alloc: std.mem.Allocator,
    cmix_orig: []const u8,
    comp_dict: []const u8,
    comp_order: []const u8,
) ![]u8 {
    const header: HeaderInfo = .{
        .dict_size = @intCast(comp_dict.len),
        .new_article_order_size = @intCast(comp_order.len),
        .decomp_input_size = 0,
    };
    const trailer = writeHeaderBytes(header);
    const total = cmix_orig.len + comp_dict.len + comp_order.len + HEADER_SIZE;
    var out = try alloc.alloc(u8, total);
    var i: usize = 0;
    @memcpy(out[i..][0..cmix_orig.len], cmix_orig);
    i += cmix_orig.len;
    @memcpy(out[i..][0..comp_dict.len], comp_dict);
    i += comp_dict.len;
    @memcpy(out[i..][0..comp_order.len], comp_order);
    i += comp_order.len;
    @memcpy(out[i..][0..HEADER_SIZE], &trailer);
    return out;
}

/// Assemble the decompress-side `archive9` image (runner.cpp:453-472):
///   `cat .decomp_bin .dict.comp cmix_output header4archive.dat > archive9`.
/// The trailer reuses dict_size/new_article_order_size from the compress-side
/// `test.dat` and sets decomp_input_size = |cmix_output| (runner.cpp:460-463).
pub fn assembleArchive9(
    alloc: std.mem.Allocator,
    decomp_bin: []const u8,
    dict_comp: []const u8,
    cmix_output: []const u8,
    order_comp_size: i32, // carried through from test.dat; unused by decode slicing
    tfweights_comp: []const u8, // transformer-era; EMPTY on LSTM-era
) ![]u8 {
    const header: HeaderInfo = .{
        .dict_size = @intCast(dict_comp.len),
        .new_article_order_size = order_comp_size,
        .decomp_input_size = @intCast(cmix_output.len),
        .tfweights_size = @intCast(tfweights_comp.len),
    };
    const trailer = writeHeaderBytes(header);
    const total = decomp_bin.len + dict_comp.len + cmix_output.len + tfweights_comp.len + HEADER_SIZE;
    var out = try alloc.alloc(u8, total);
    var i: usize = 0;
    @memcpy(out[i..][0..decomp_bin.len], decomp_bin);
    i += decomp_bin.len;
    @memcpy(out[i..][0..dict_comp.len], dict_comp);
    i += dict_comp.len;
    @memcpy(out[i..][0..cmix_output.len], cmix_output);
    i += cmix_output.len;
    @memcpy(out[i..][0..tfweights_comp.len], tfweights_comp);
    i += tfweights_comp.len;
    @memcpy(out[i..][0..HEADER_SIZE], &trailer);
    return out;
}

/// Assemble the alternate decompress-side image needed by codecs whose
/// payload restore depends on the article-order asset (currently `-Dr1v2`):
///
///   decomp_bin ++ dict_comp ++ order_comp ++ payload ++ HeaderInfo(12)
///
/// The legacy/r1v1 `assembleArchive9` layout remains unchanged and omits the
/// order bytes. Keeping these as separate functions makes the extra counted
/// copy impossible to introduce accidentally.
pub fn assembleArchive9WithOrder(
    alloc: std.mem.Allocator,
    decomp_bin: []const u8,
    dict_comp: []const u8,
    order_comp: []const u8,
    cmix_output: []const u8,
    tfweights_comp: []const u8, // transformer-era; EMPTY on LSTM-era
) ![]u8 {
    const header: HeaderInfo = .{
        .dict_size = @intCast(dict_comp.len),
        .new_article_order_size = @intCast(order_comp.len),
        .decomp_input_size = @intCast(cmix_output.len),
        .tfweights_size = @intCast(tfweights_comp.len),
    };
    const trailer = writeHeaderBytes(header);
    const total = decomp_bin.len + dict_comp.len + order_comp.len + cmix_output.len +
        tfweights_comp.len + HEADER_SIZE;
    var out = try alloc.alloc(u8, total);
    var i: usize = 0;
    @memcpy(out[i..][0..decomp_bin.len], decomp_bin);
    i += decomp_bin.len;
    @memcpy(out[i..][0..dict_comp.len], dict_comp);
    i += dict_comp.len;
    @memcpy(out[i..][0..order_comp.len], order_comp);
    i += order_comp.len;
    @memcpy(out[i..][0..cmix_output.len], cmix_output);
    i += cmix_output.len;
    @memcpy(out[i..][0..tfweights_comp.len], tfweights_comp);
    i += tfweights_comp.len;
    @memcpy(out[i..][0..HEADER_SIZE], &trailer);
    return out;
}

pub const ArchiveOrderMode = enum {
    /// Legacy/r1v1: archive9 carries no order bytes; order is counted in comp9 only.
    omitted,
    /// r1v2 correctness layout: archive9 repeats order_comp for payload restore.
    embedded,
};

/// Production `-e` dispatcher shared with the layout unit gates. Both forms
/// carry the order SIZE in HeaderInfo; only `.embedded` carries the bytes.
pub fn assembleArchive9ForMode(
    alloc: std.mem.Allocator,
    archive_order_mode: ArchiveOrderMode,
    decomp_bin: []const u8,
    dict_comp: []const u8,
    order_comp: []const u8,
    cmix_output: []const u8,
    tfweights_comp: []const u8, // transformer-era; EMPTY on LSTM-era
) ![]u8 {
    return switch (archive_order_mode) {
        .omitted => assembleArchive9(
            alloc,
            decomp_bin,
            dict_comp,
            cmix_output,
            @intCast(order_comp.len),
            tfweights_comp,
        ),
        .embedded => assembleArchive9WithOrder(
            alloc,
            decomp_bin,
            dict_comp,
            order_comp,
            cmix_output,
            tfweights_comp,
        ),
    };
}

/// File offsets for streaming a physical archive without loading the whole
/// image under the predictor RSS peak.
pub const ArchiveFileLayout = struct {
    decomp_bin_size: u64,
    dict_offset: u64,
    dict_size: u64,
    order_offset: ?u64,
    order_size: u64,
    payload_offset: u64,
    payload_size: u64,
    /// transformer-era: offset/size of the int4 weights blob, which sits AFTER the payload
    /// and immediately BEFORE the trailer. Zero-sized on LSTM-era.
    tfweights_offset: u64,
    tfweights_size: u64,
};

/// Resolve physical archive offsets from the common trailer and an explicit
/// layout. HeaderInfo alone cannot distinguish the two forms because it always
/// carries order_size; callers must select the codec-appropriate mode.
pub fn locateArchive(
    file_size: u64,
    header: HeaderInfo,
    archive_order_mode: ArchiveOrderMode,
) error{CorruptHeader}!ArchiveFileLayout {
    if (header.dict_size < 0 or header.new_article_order_size < 0 or header.decomp_input_size < 0)
        return error.CorruptHeader;
    const dsz: u64 = @intCast(header.dict_size);
    const header_osz: u64 = @intCast(header.new_article_order_size);
    const osz: u64 = if (archive_order_mode == .embedded) header_osz else 0;
    const psz: u64 = @intCast(header.decomp_input_size);
    // transformer-era: the weights blob is tail overhead too. Leaving it out of this sum
    // made `bin_size` too large by |blob| and shifted EVERY segment offset.
    if (transformer_on and header.tfweights_size < 0) return error.CorruptHeader;
    const tsz: u64 = if (transformer_on) @intCast(header.tfweights_size) else 0;
    const overhead = dsz + osz + psz + tsz + HEADER_SIZE;
    if (overhead > file_size) return error.CorruptHeader;
    const bin_size = file_size - overhead;
    return .{
        .decomp_bin_size = bin_size,
        .dict_offset = bin_size,
        .dict_size = dsz,
        .order_offset = if (archive_order_mode == .embedded) bin_size + dsz else null,
        .order_size = osz,
        .payload_offset = bin_size + dsz + osz,
        .payload_size = psz,
        .tfweights_offset = bin_size + dsz + osz + psz,
        .tfweights_size = tsz,
    };
}

/// Sizes proven by `verifyForm1Pair`. The contest total is the physical file
/// sum, while `fixed_overhead + payload` exposes the equivalent component
/// accounting.
pub const Form1PairSizes = struct {
    decomp_bin: usize,
    comp_dict: usize,
    comp_order: usize,
    /// transformer-era: the int4 weights blob. Carried by BOTH files, so it is charged 2x
    /// in `fixed_overhead` exactly like `decomp_bin` and `comp_dict`. 0 on LSTM-era.
    comp_tfweights: usize,
    payload: usize,
    comp9: usize,
    archive9: usize,
    archive_order_mode: ArchiveOrderMode,
    fixed_overhead: usize,
    total_s: usize,
};

/// Verify that two complete images are a physical Form-1 pair:
///
///   comp9        = decomp_bin ++ comp_dict ++ comp_order ++ header
///   archive9 = decomp_bin ++ comp_dict ++ [comp_order] ++ payload ++ header
///
/// In particular, every repeated prefix component must be byte-identical. The
/// order bytes are omitted for legacy/r1v1 and repeated for the r1v2 alternate.
pub fn verifyForm1Pair(
    comp9: []const u8,
    archive9: []const u8,
    archive_order_mode: ArchiveOrderMode,
) !Form1PairSizes {
    const comp = try selfextractComp(comp9);
    if (comp.header.decomp_input_size != 0) return error.Comp9PayloadSizeNotZero;
    const archive_header = try readTrailer(archive9);
    if (archive_header.dict_size != comp.header.dict_size) return error.DictSizeMismatch;
    if (archive_header.new_article_order_size != comp.header.new_article_order_size)
        return error.OrderSizeMismatch;
    if (archive_header.tfweights_size != comp.header.tfweights_size)
        return error.TfWeightsSizeMismatch;
    const layout = try locateArchive(archive9.len, archive_header, archive_order_mode);
    if (layout.decomp_bin_size != comp.decomp_bin.len) return error.DecompBinMismatch;
    const archive_order_len: usize = @intCast(layout.order_size);
    const common_len = comp.decomp_bin.len + comp.dict_comp.len + archive_order_len;
    if (!std.mem.eql(u8, comp9[0..common_len], archive9[0..common_len]))
        return error.SharedPrefixMismatch;
    // transformer-era: the blob is carried verbatim by both files — assert the BYTES, not
    // just the length. It is not part of the shared PREFIX (it sits after the
    // payload in archive9), so the prefix compare above cannot see it.
    if (comp.tfweights_comp.len > 0 and
        !std.mem.eql(u8, comp.tfweights_comp, archive9[@intCast(layout.tfweights_offset)..][0..comp.tfweights_comp.len]))
        return error.TfWeightsMismatch;

    const fixed = 2 * comp.decomp_bin.len + 2 * comp.dict_comp.len +
        (if (archive_order_mode == .embedded) 2 * comp.order_comp.len else comp.order_comp.len) +
        2 * comp.tfweights_comp.len +
        2 * HEADER_SIZE;
    const total = comp9.len + archive9.len;
    if (total != fixed + layout.payload_size) return error.SizeIdentityMismatch;
    return .{
        .decomp_bin = comp.decomp_bin.len,
        .comp_dict = comp.dict_comp.len,
        .comp_order = comp.order_comp.len,
        .comp_tfweights = comp.tfweights_comp.len,
        .payload = @intCast(layout.payload_size),
        .comp9 = comp9.len,
        .archive9 = archive9.len,
        .archive_order_mode = archive_order_mode,
        .fixed_overhead = fixed,
        .total_s = total,
    };
}

// ---------------------------------------------------------------------------
// Thin file-backed wrappers mirroring the original I/O (for the runner). These
// carry no ratio logic; they just read a file into memory and delegate to the
// pure slicers above.
// ---------------------------------------------------------------------------

/// `write(file_name, header)` (self_extract.h:15-19) + the `-h` branch
/// (runner.cpp:474-480): write the 12-byte trailer file (e.g. `header.dat`).
pub fn writeHeaderFile(path: []const u8, h: HeaderInfo) !void {
    const bytes = writeHeaderBytes(h);
    const f = try std.fs.cwd().createFile(path, .{});
    defer f.close();
    try f.writeAll(&bytes);
}

/// `read(file_name, header)` (self_extract.h:21-25).
pub fn readHeaderFile(path: []const u8) !HeaderInfo {
    const f = try std.fs.cwd().openFile(path, .{});
    defer f.close();
    var buf: [HEADER_SIZE]u8 = undefined;
    const n = try f.readAll(&buf);
    if (n != HEADER_SIZE) return error.ShortHeader;
    return parseHeaderBytes(&buf);
}

// ===========================================================================
// Tests: header round-trip (build header, parse it back, sizes match) + byte
// layout vs the g++ oracle + slice/assemble round-trips.
// ===========================================================================

test "HEADER_SIZE matches C++ sizeof(HeaderInfo)" {
    // The C++ oracle's HeaderInfo is 12 bytes and the DEFAULT build must stay
    // byte-compatible with it. transformer-era widens the trailer to 16 under
    // -Dtransformer to carry tfweights_size as a fourth LE i32; that is a
    // deliberate, engaged-only divergence, so assert BOTH contracts rather than
    // relaxing the default one.
    try std.testing.expectEqual(@as(usize, if (transformer_on) 16 else 12), HEADER_SIZE);
}

test "header round-trips: build -> parse -> sizes match" {
    const h: HeaderInfo = .{
        .dict_size = 411996,
        .new_article_order_size = 1094862,
        .decomp_input_size = 0,
    };
    const bytes = writeHeaderBytes(h);
    const back = parseHeaderBytes(&bytes);
    try std.testing.expectEqual(h.dict_size, back.dict_size);
    try std.testing.expectEqual(h.new_article_order_size, back.new_article_order_size);
    try std.testing.expectEqual(h.decomp_input_size, back.decomp_input_size);
}

test "header byte layout matches g++ oracle (LE ints, no padding)" {
    // From the oracle: dict=411996, order=1094862, decomp=0 ->
    //   5c 49 06 00  ce b4 10 00  00 00 00 00
    // The FIRST 12 BYTES must match the oracle in every configuration -- that is
    // the invariant, and widening the trailer must not disturb it. Under
    // -Dtransformer the 4 appended bytes carry tfweights_size, which defaults to
    // 0 here, so they are asserted separately rather than folded into the oracle.
    const h1: HeaderInfo = .{ .dict_size = 411996, .new_article_order_size = 1094862, .decomp_input_size = 0 };
    const b1 = writeHeaderBytes(h1);
    try std.testing.expectEqualSlices(u8, &[_]u8{
        0x5c, 0x49, 0x06, 0x00, 0xce, 0xb4, 0x10, 0x00, 0x00, 0x00, 0x00, 0x00,
    }, b1[0..12]);
    // Oracle #2: 0x01020304 / 0x0A0B0C0D / 0x11223344 ->
    //   04 03 02 01  0d 0c 0b 0a  44 33 22 11
    const h2: HeaderInfo = .{ .dict_size = 0x01020304, .new_article_order_size = 0x0A0B0C0D, .decomp_input_size = 0x11223344 };
    const b2 = writeHeaderBytes(h2);
    try std.testing.expectEqualSlices(u8, &[_]u8{
        0x04, 0x03, 0x02, 0x01, 0x0d, 0x0c, 0x0b, 0x0a, 0x44, 0x33, 0x22, 0x11,
    }, b2[0..12]);
    if (transformer_on) {
        // fourth field, LE i32, default 0 -- and a NON-zero value must land there
        try std.testing.expectEqualSlices(u8, &[_]u8{ 0, 0, 0, 0 }, b1[12..16]);
        const h3: HeaderInfo = .{ .dict_size = 1, .new_article_order_size = 2, .decomp_input_size = 3, .tfweights_size = 0x002cb7dc };
        try std.testing.expectEqualSlices(u8, &[_]u8{ 0xdc, 0xb7, 0x2c, 0x00 }, writeHeaderBytes(h3)[12..16]);
    }
}

test "selfextractComp slices a synthetic cmix image + round-trips assemble" {
    const alloc = std.testing.allocator;
    const bin = "BINARY-DECOMPRESSOR-BODY";
    const dict = "COMPDICTxx"; // 10 bytes
    const order = "COMPORDER-PAYLOAD-1234"; // 22 bytes
    const image = try assembleCmix(alloc, bin, dict, order);
    defer alloc.free(image);

    const segs = try selfextractComp(image);
    try std.testing.expectEqualSlices(u8, bin, segs.decomp_bin);
    try std.testing.expectEqualSlices(u8, dict, segs.dict_comp);
    try std.testing.expectEqualSlices(u8, order, segs.order_comp);
    try std.testing.expectEqual(@as(i32, @intCast(dict.len)), segs.header.dict_size);
    try std.testing.expectEqual(@as(i32, @intCast(order.len)), segs.header.new_article_order_size);
    try std.testing.expectEqual(@as(i32, 0), segs.header.decomp_input_size);
    // trailer is exactly the last HEADER_SIZE bytes (12 default, 16 engaged)
    try std.testing.expectEqualSlices(u8, image[image.len - HEADER_SIZE ..], segs.trailer_bytes);
}

test "selfextractDecomp slices archive9 (payload sized by decomp_input_size)" {
    const alloc = std.testing.allocator;
    const bin = "DECOMP-BIN";
    const dict = "DICTCOMP-77"; // 11 bytes
    const payload = "ARITH-CODED-ENWIK9-PAYLOAD"; // 26 bytes
    const order_size: i32 = 1094862; // carried through, must NOT affect slicing
    const image = try assembleArchive9(alloc, bin, dict, payload, order_size, "");
    defer alloc.free(image);

    const segs = try selfextractDecomp(image);
    try std.testing.expectEqualSlices(u8, bin, segs.decomp_bin);
    try std.testing.expectEqualSlices(u8, dict, segs.dict_comp);
    try std.testing.expectEqualSlices(u8, payload, segs.ready4cmix);
    try std.testing.expectEqual(@as(i32, @intCast(dict.len)), segs.header.dict_size);
    try std.testing.expectEqual(order_size, segs.header.new_article_order_size);
    try std.testing.expectEqual(@as(i32, @intCast(payload.len)), segs.header.decomp_input_size);
}

test "Form-1 pair verifies legacy order-omitted layout and exact S arithmetic" {
    const alloc = std.testing.allocator;
    const bin = "DECOMP-BIN";
    const dict = "DICT-COMP";
    const order = "ORDER-COMPRESSOR-ONLY";
    const payload = "CODED-ENWIK9-PAYLOAD";
    const comp9 = try assembleCmix(alloc, bin, dict, order);
    defer alloc.free(comp9);
    const archive9 = try assembleArchive9ForMode(alloc, .omitted, bin, dict, order, payload, "");
    defer alloc.free(archive9);

    const sizes = try verifyForm1Pair(comp9, archive9, .omitted);
    try std.testing.expectEqual(bin.len, sizes.decomp_bin);
    try std.testing.expectEqual(dict.len, sizes.comp_dict);
    try std.testing.expectEqual(order.len, sizes.comp_order);
    try std.testing.expectEqual(payload.len, sizes.payload);
    try std.testing.expectEqual(comp9.len + archive9.len, sizes.total_s);
    try std.testing.expectEqual(
        2 * bin.len + 2 * dict.len + order.len + payload.len + 2 * HEADER_SIZE,
        sizes.total_s,
    );

    var wrong_archive = try alloc.dupe(u8, archive9);
    defer alloc.free(wrong_archive);
    wrong_archive[0] ^= 1;
    try std.testing.expectError(error.SharedPrefixMismatch, verifyForm1Pair(comp9, wrong_archive, .omitted));
}

test "Form-1 pair verifies explicit order-embedded alternate layout" {
    const alloc = std.testing.allocator;
    const bin = "DECOMP-BIN";
    const dict = "DICT-COMP";
    const order = "ORDER-COMPRESSOR-AND-DECODER";
    const payload = "R1V2-CODED-ENWIK9-PAYLOAD";
    const comp9 = try assembleCmix(alloc, bin, dict, order);
    defer alloc.free(comp9);
    const archive9 = try assembleArchive9ForMode(alloc, .embedded, bin, dict, order, payload, "");
    defer alloc.free(archive9);

    const sizes = try verifyForm1Pair(comp9, archive9, .embedded);
    try std.testing.expectEqual(ArchiveOrderMode.embedded, sizes.archive_order_mode);
    try std.testing.expectEqual(comp9.len + archive9.len, sizes.total_s);
    try std.testing.expectEqual(
        2 * bin.len + 2 * dict.len + 2 * order.len + payload.len + 2 * HEADER_SIZE,
        sizes.total_s,
    );
    const layout = try locateArchive(archive9.len, try readTrailer(archive9), .embedded);
    try std.testing.expectEqual(@as(u64, bin.len), layout.decomp_bin_size);
    try std.testing.expectEqual(@as(?u64, bin.len + dict.len), layout.order_offset);
    try std.testing.expectEqual(@as(u64, bin.len + dict.len + order.len), layout.payload_offset);
    try std.testing.expectEqualSlices(
        u8,
        order,
        archive9[@intCast(layout.order_offset.?)..][0..order.len],
    );
    try std.testing.expectEqualSlices(
        u8,
        payload,
        archive9[@intCast(layout.payload_offset)..][0..payload.len],
    );
    try std.testing.expectError(error.DecompBinMismatch, verifyForm1Pair(comp9, archive9, .omitted));
}

test "corrupt/oversized headers are rejected, not UB" {
    const alloc = std.testing.allocator;
    // image smaller than trailer
    try std.testing.expectError(error.ImageTooSmall, selfextractComp("short"));
    // trailer claims more payload than the image holds
    var img = try alloc.alloc(u8, HEADER_SIZE + 4);
    defer alloc.free(img);
    @memset(img, 0);
    const bad: HeaderInfo = .{ .dict_size = 1000, .new_article_order_size = 0, .decomp_input_size = 0 };
    @memcpy(img[img.len - HEADER_SIZE ..], &writeHeaderBytes(bad));
    try std.testing.expectError(error.CorruptHeader, selfextractComp(img));
}
