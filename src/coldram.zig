//! coldram — touched-then-cold page reclamation for the big model slabs.
//!
//! Preflight:
//! Census: every other RAM optimisation in this engine addresses
//! never-touched pages (fix12-16 lazy zero) or moves PPMd to disk. Pages that
//! were touched once (header transient / early regime) and never referenced
//! again stay resident until exit and count fully against the 1e10-byte RAM
//! gate. This module reclaims them EXACTLY:
//!
//!   hot (RW) --T0 epochs--> wprobe (PROT_READ: writes fault)
//!            --T1 epochs--> rprobe (PROT_NONE: any access faults)
//!            --T2 epochs--> attic  (LZ-compressed copy + MADV_DONTNEED)
//!
//! Any fault on a probed/attic'd page restores it (decompress is exact, so
//! the restored bytes are bit-identical to what was there). Model output can
//! NEVER depend on this machinery: contents round-trip exactly, and neither
//! prediction nor update reads page residency. Wall cost is bounded by probe
//! faults (~µs each, counters prove the rate) plus one epoch sweep per
//! EPOCH_BYTES of input.
//!
//! Comptime-dead by default (-Dcoldram=false): every public entry point
//! compiles to nothing, zero bytes in the shipped binary.
//!
//! Async-signal notes: the process is single-threaded (single_threaded build);
//! the SIGSEGV handler only runs mprotect/madvise syscalls and the
//! allocation-free lzDecompress into preallocated memory. A fault on a page
//! this module does not track reinstalls the previous action and returns (the
//! re-executed fault then gets the default disposition).

const std = @import("std");
const build_options = @import("build_options");
// HERMETIC SHIP GUARD (same pattern as predictor_lex.zig's env overrides):
// ship roots declare `zmix_hermetic` and compile every env read away, so no
// getenv and no environment-variable name survives in the shipped image. The
// submitted binary must not read anything outside itself.
const hermetic_root = @import("root");
const HERMETIC: bool = @hasDecl(hermetic_root, "zmix_hermetic") and hermetic_root.zmix_hermetic;
const os_tag = @import("builtin").os.tag;

pub const enabled: bool = if (@hasDecl(build_options, "coldram")) build_options.coldram else false;
/// -Dcoldram-pool-only: the row-pool densification term WITHOUT the probe/attic
/// half. Every site that touches mprotect, the attic or the SIGSEGV handler is
/// comptime-gated on `attic` below, so in a pool-only build that whole half is
/// unreferenced and DCE'd — no signal handler is installed at all, which is the
/// point (see the adoption split in).
pub const pool_only: bool = if (@hasDecl(build_options, "coldram_pool_only")) build_options.coldram_pool_only else false;
pub const attic: bool = enabled and !pool_only;

const PAGE: usize = 4096;
const linux = std.os.linux;

// ---------------------------------------------------------------------------
// page-local LZ codec (LZ4-style token stream, window = the page itself)
// ---------------------------------------------------------------------------
// Compression runs in normal context (epoch sweep); decompression runs inside
// the SIGSEGV handler — both are allocation-free. Format per sequence:
//   token: hi4 = literal count (15 => extra bytes), lo4 = matchlen-4 (15 => extra)
//   [extra literal-len bytes] [literals] [offset:u16le] [extra matchlen bytes]
// The final sequence carries literals only (lo4 unused, offset omitted).

fn read32(p: []const u8, i: usize) u32 {
    return std.mem.readInt(u32, p[i..][0..4], .little);
}

/// Compress a page into dst; null when the result would not fit dst.
pub fn lzCompress(src: *const [PAGE]u8, dst: []u8) ?usize {
    var htab: [1024]u16 = @splat(0);
    var out: usize = 0;
    var anchor: usize = 0; // first unemitted literal
    var i: usize = 0;
    while (i + 12 < PAGE) {
        const h: usize = (read32(src, i) *% 2654435761) >> 22;
        const cand: usize = htab[h];
        htab[h] = @intCast(i);
        if (cand < i and read32(src, cand) == read32(src, i)) {
            // extend match
            var mlen: usize = 4;
            while (i + mlen < PAGE and src[cand + mlen] == src[i + mlen]) mlen += 1;
            const lit = i - anchor;
            // emit sequence
            const tok_out = out;
            out += 1;
            if (out + lit + 16 > dst.len) return null;
            var lit_rem = lit;
            var tok: u8 = 0;
            if (lit >= 15) {
                tok |= 0xf0;
                lit_rem = lit - 15;
                while (lit_rem >= 255) : (lit_rem -= 255) {
                    dst[out] = 255;
                    out += 1;
                }
                dst[out] = @intCast(lit_rem);
                out += 1;
            } else {
                tok |= @intCast(lit << 4);
            }
            @memcpy(dst[out..][0..lit], src[anchor..][0..lit]);
            out += lit;
            std.mem.writeInt(u16, dst[out..][0..2], @intCast(i - cand), .little);
            out += 2;
            var m_rem = mlen - 4;
            if (m_rem >= 15) {
                tok |= 0x0f;
                m_rem -= 15;
                while (m_rem >= 255) : (m_rem -= 255) {
                    if (out + 2 > dst.len) return null;
                    dst[out] = 255;
                    out += 1;
                }
                if (out + 1 > dst.len) return null;
                dst[out] = @intCast(m_rem);
                out += 1;
            } else {
                tok |= @intCast(m_rem);
            }
            dst[tok_out] = tok;
            i += mlen;
            anchor = i;
        } else {
            i += 1;
        }
    }
    // trailing literals as a final match-less sequence
    const lit = PAGE - anchor;
    if (out + 1 + lit + 4 > dst.len) return null;
    var tok: u8 = 0;
    var lit_rem = lit;
    const tok_out = out;
    out += 1;
    if (lit >= 15) {
        tok = 0xf0;
        lit_rem = lit - 15;
        while (lit_rem >= 255) : (lit_rem -= 255) {
            dst[out] = 255;
            out += 1;
        }
        dst[out] = @intCast(lit_rem);
        out += 1;
    } else {
        tok = @intCast(lit << 4);
    }
    dst[tok_out] = tok;
    @memcpy(dst[out..][0..lit], src[anchor..][0..lit]);
    out += lit;
    return out;
}

/// Exact inverse of lzCompress. src must be a blob lzCompress produced for a
/// full page; writes exactly PAGE bytes.
pub fn lzDecompress(src: []const u8, dst: *[PAGE]u8) void {
    var ip: usize = 0;
    var op: usize = 0;
    while (op < PAGE) {
        const tok = src[ip];
        ip += 1;
        var lit: usize = tok >> 4;
        if (lit == 15) {
            while (src[ip] == 255) : (ip += 1) lit += 255;
            lit += src[ip];
            ip += 1;
        }
        @memcpy(dst[op..][0..lit], src[ip..][0..lit]);
        ip += lit;
        op += lit;
        if (op >= PAGE) break; // final literal-only sequence
        const off: usize = std.mem.readInt(u16, src[ip..][0..2], .little);
        ip += 2;
        var mlen: usize = (tok & 0x0f) + 4;
        if ((tok & 0x0f) == 15) {
            while (src[ip] == 255) : (ip += 1) mlen += 255;
            mlen += src[ip];
            ip += 1;
        }
        // overlapping copy semantics required (RLE-style matches)
        var s = op - off;
        var d = op;
        var n = mlen;
        while (n > 0) : (n -= 1) {
            dst[d] = dst[s];
            d += 1;
            s += 1;
        }
        op += mlen;
    }
}

// ---------------------------------------------------------------------------
// attic storage: size-class pools, freelists threaded through slot memory
// ---------------------------------------------------------------------------
const CLASS_SIZES = [_]usize{ 512, 1024, 1536, 2048, 3072 };
const NCLASS = CLASS_SIZES.len;
const CHUNK: usize = 1 << 20;
const NIL: u32 = 0xffff_ffff;

const ST_HOT: u8 = 0;
const ST_WPROBE: u8 = 1;
const ST_RPROBE: u8 = 2;
const ST_ATTIC: u8 = 3;
const FL_NOCOMP: u8 = 1;

const PageMeta = struct {
    state: u8 = ST_HOT,
    flags: u8 = 0,
    age: u16 = 0,
    ref: u32 = NIL, // attic slot index when ST_ATTIC
    clen: u16 = 0, // compressed length when ST_ATTIC
};

const Range = struct {
    name: []const u8 = "",
    base: usize = 0,
    npages: usize = 0,
    meta: []PageMeta = &.{},
    probe_faults: u64 = 0,
    attic_faults: u64 = 0,
    pages_compressed: u64 = 0,
    comp_out: u64 = 0,
    nocomp: u64 = 0,
    in_attic: u64 = 0, // pages currently attic'd
};

const MAX_RANGES = 64;

var g: struct {
    installed: bool = false,
    ranges: [MAX_RANGES]Range = @splat(.{}),
    nranges: usize = 0,
    bits_seen: u64 = 0,
    epoch_bytes: u64 = 4 << 20,
    next_epoch_bits: u64 = 0,
    epoch_no: u64 = 0,
    t0: u16 = 2,
    t1: u16 = 2,
    t2: u16 = 4,
    nocomp_retry: u16 = 64,
    attic_cap: u64 = 1 << 30,
    stats: bool = false,
    // attic slot store
    slot_ptr: std.ArrayListUnmanaged([*]u8) = .{},
    slot_cls: std.ArrayListUnmanaged(u8) = .{},
    free_head: [NCLASS]u32 = @splat(NIL),
    bump: [NCLASS][*]u8 = undefined,
    bump_left: [NCLASS]usize = @splat(0),
    attic_bytes: u64 = 0, // committed attic chunk bytes
    // scratch
    pagemap_fd: linux.fd_t = -1,
    pm_buf: []u8 = &.{},
    prev_sa: std.posix.Sigaction = undefined,
    // row pool
    pool: ?[]align(PAGE) u8 = null,
    pool_off: usize = 0,
    pool_registered: bool = false,
} = .{};

fn envU64(name: []const u8, default: u64) u64 {
    if (comptime (HERMETIC or os_tag == .windows)) return default;
    const v = std.posix.getenv(name) orelse return default;
    return std.fmt.parseInt(u64, v, 10) catch default;
}

fn envFlag(name: []const u8) bool {
    if (comptime (HERMETIC or os_tag == .windows)) return false;
    return std.posix.getenv(name) != null;
}

fn install() void {
    if (g.installed) return;
    g.installed = true;
    if (comptime !attic) {
        // pool-only: no epochs, no probes, no handler. Stats still readable so
        // the pool arm can be identified in a log.
        g.stats = envFlag("ZMIX_COLDRAM_STATS");
        return;
    }
    g.epoch_bytes = envU64("ZMIX_COLDRAM_EPOCH", 4 << 20);
    g.t0 = @intCast(envU64("ZMIX_COLDRAM_T0", 2));
    g.t1 = @intCast(envU64("ZMIX_COLDRAM_T1", 2));
    g.t2 = @intCast(envU64("ZMIX_COLDRAM_T2", 4));
    g.nocomp_retry = @intCast(envU64("ZMIX_COLDRAM_RETRY", 64));
    g.attic_cap = envU64("ZMIX_COLDRAM_CAP_MB", 1024) << 20;
    g.stats = envFlag("ZMIX_COLDRAM_STATS");
    g.next_epoch_bits = g.epoch_bytes * 8;
    g.pagemap_fd = std.posix.open("/proc/self/pagemap", .{ .ACCMODE = .RDONLY }, 0) catch -1;
    const act = std.posix.Sigaction{
        .handler = .{ .sigaction = onSegv },
        .mask = std.posix.sigemptyset(),
        .flags = std.posix.SA.SIGINFO | std.posix.SA.RESTART,
    };
    std.posix.sigaction(std.posix.SIG.SEGV, &act, &g.prev_sa);
}

/// Register a big long-lived slab for cold-page reclamation. Interior pages
/// only (base rounded up, end rounded down). Safe to call for any slab that
/// is never used as an I/O buffer and never freed before process end.
pub fn registerRange(name: []const u8, mem: anytype) void {
    if (comptime !attic) return;
    install();
    if (g.nranges == MAX_RANGES) return;
    const T = @typeInfo(@TypeOf(mem)).pointer.child;
    const addr = @intFromPtr(mem.ptr);
    const len = mem.len * @sizeOf(T);
    const lo = std.mem.alignForward(usize, addr, PAGE);
    const hi = std.mem.alignBackward(usize, addr + len, PAGE);
    if (hi <= lo or hi - lo < PAGE * 16) return;
    const npages = (hi - lo) / PAGE;
    const meta = std.heap.page_allocator.alloc(PageMeta, npages) catch return;
    @memset(meta, .{});
    g.ranges[g.nranges] = .{ .name = name, .base = lo, .npages = npages, .meta = meta };
    g.nranges += 1;
    if (g.pm_buf.len < npages * 8) {
        if (g.pm_buf.len != 0) std.heap.page_allocator.free(g.pm_buf);
        g.pm_buf = std.heap.page_allocator.alloc(u8, npages * 8) catch &.{};
    }
}

// ---------------------------------------------------------------------------
// row pool: a single registered slab for the MixerLex context rows (their
// natural home is GPA buckets, which cannot be page-managed; a bump pool in
// first-touch order preserves the exact allocation-order layout property the
// census measured). Exposed as a std allocator: free/resize are no-ops.
// ---------------------------------------------------------------------------
const POOL_CAP: usize = 768 << 20;

fn poolAlloc(_: *anyopaque, len: usize, alignment: std.mem.Alignment, _: usize) ?[*]u8 {
    if (g.pool == null) {
        const mem = std.posix.mmap(
            null,
            POOL_CAP,
            std.posix.PROT.READ | std.posix.PROT.WRITE,
            .{ .TYPE = .PRIVATE, .ANONYMOUS = true, .NORESERVE = true },
            -1,
            0,
        ) catch return null;
        g.pool = @alignCast(mem);
        g.pool_off = 0;
        registerRange("mixrow-pool", mem);
    }
    const a = alignment.toByteUnits();
    const off = std.mem.alignForward(usize, g.pool_off, a);
    if (off + len > POOL_CAP) return null; // exhausted: caller allocator OOMs (never expected; cap > worst case)
    g.pool_off = off + len;
    return g.pool.?.ptr + off;
}

fn poolResize(_: *anyopaque, mem: []u8, _: std.mem.Alignment, new_len: usize, _: usize) bool {
    return new_len <= mem.len;
}

fn poolRemap(_: *anyopaque, _: []u8, _: std.mem.Alignment, _: usize, _: usize) ?[*]u8 {
    return null;
}

fn poolFree(_: *anyopaque, _: []u8, _: std.mem.Alignment, _: usize) void {}

const pool_vtable = std.mem.Allocator.VTable{
    .alloc = poolAlloc,
    .resize = poolResize,
    .remap = poolRemap,
    .free = poolFree,
};

/// Allocator for MixerLex row storage: bump from one registered slab.
/// freeis a no-op (rows are never freed before process end; the hash-map
/// table reallocations leak a few hundred KB per mixer into the pool —
/// bounded and accounted).
pub fn rowPoolAllocator(fallback: std.mem.Allocator) std.mem.Allocator {
    if (comptime !enabled) return fallback;
    install();
    return .{ .ptr = undefined, .vtable = &pool_vtable };
}

// ---------------------------------------------------------------------------
// epoch machinery
// ---------------------------------------------------------------------------

/// Call once per predicted/perceived BIT (encode and decode are symmetric).
pub inline fn tickBit() void {
    if (comptime !attic) return;
    g.bits_seen += 1;
    if (g.bits_seen >= g.next_epoch_bits) sweepEpoch();
}

fn mprotectPages(base: usize, first: usize, count: usize, prot: u32) void {
    const p: [*]align(PAGE) u8 = @ptrFromInt(base + first * PAGE);
    _ = linux.mprotect(p, count * PAGE, prot);
}

fn madvisePages(base: usize, first: usize, count: usize) void {
    const p: [*]align(PAGE) u8 = @ptrFromInt(base + first * PAGE);
    _ = linux.madvise(p, count * PAGE, linux.MADV.DONTNEED);
}

fn allocSlot(cls: usize) ?u32 {
    if (g.free_head[cls] != NIL) {
        const idx = g.free_head[cls];
        const slotp = g.slot_ptr.items[idx];
        g.free_head[cls] = std.mem.readInt(u32, slotp[0..4], .little);
        return idx;
    }
    if (g.bump_left[cls] < CLASS_SIZES[cls]) {
        if (g.attic_bytes + CHUNK > g.attic_cap) return null;
        const mem = std.heap.page_allocator.alloc(u8, CHUNK) catch return null;
        g.bump[cls] = mem.ptr;
        g.bump_left[cls] = CHUNK;
        g.attic_bytes += CHUNK;
    }
    const p = g.bump[cls];
    g.bump[cls] += CLASS_SIZES[cls];
    g.bump_left[cls] -= CLASS_SIZES[cls];
    g.slot_ptr.append(std.heap.page_allocator, p) catch return null;
    g.slot_cls.append(std.heap.page_allocator, @intCast(cls)) catch {
        _ = g.slot_ptr.pop();
        return null;
    };
    return @intCast(g.slot_ptr.items.len - 1);
}

fn freeSlot(idx: u32) void {
    const cls = g.slot_cls.items[idx];
    const p = g.slot_ptr.items[idx];
    std.mem.writeInt(u32, p[0..4], g.free_head[cls], .little);
    g.free_head[cls] = idx;
}

fn classFor(clen: usize) ?usize {
    inline for (CLASS_SIZES, 0..) |cs, i| {
        if (clen <= cs) return i;
    }
    return null;
}

fn presentBit(pi: usize) bool {
    const ent = std.mem.readInt(u64, g.pm_buf[pi * 8 ..][0..8], .little);
    return (ent >> 63) & 1 == 1;
}

fn sweepEpoch() void {
    g.next_epoch_bits += g.epoch_bytes * 8;
    g.epoch_no += 1;
    var comp_buf: [PAGE]u8 = undefined;
    for (g.ranges[0..g.nranges]) |*r| {
        // present bitmap for this range (one pread; without it virgin lazy
        // pages would be faulted in by the compressing read)
        if (g.pagemap_fd >= 0 and g.pm_buf.len >= r.npages * 8) {
            const got = linux.pread(g.pagemap_fd, g.pm_buf.ptr, r.npages * 8, @intCast((r.base / PAGE) * 8));
            if (got != r.npages * 8) continue;
        } else continue;

        var pi: usize = 0;
        while (pi < r.npages) : (pi += 1) {
            const m = &r.meta[pi];
            switch (m.state) {
                ST_HOT => {
                    if (!presentBit(pi)) continue; // nothing resident to manage
                    if (m.flags & FL_NOCOMP != 0) {
                        m.age +|= 1;
                        if (m.age >= g.nocomp_retry) {
                            m.flags &= ~FL_NOCOMP;
                            m.age = 0;
                        }
                        continue;
                    }
                    m.age +|= 1;
                    if (m.age >= g.t0) {
                        // batch run of hot->wprobe
                        const first = pi;
                        while (pi + 1 < r.npages and r.meta[pi + 1].state == ST_HOT and
                            r.meta[pi + 1].flags & FL_NOCOMP == 0 and
                            r.meta[pi + 1].age + 1 >= g.t0 and presentBit(pi + 1))
                        {
                            pi += 1;
                            r.meta[pi].age +|= 1;
                        }
                        for (r.meta[first .. pi + 1]) |*mm| {
                            mm.state = ST_WPROBE;
                            mm.age = 0;
                        }
                        mprotectPages(r.base, first, pi - first + 1, linux.PROT.READ);
                    }
                },
                ST_WPROBE => {
                    m.age +|= 1;
                    if (m.age >= g.t1) {
                        const first = pi;
                        while (pi + 1 < r.npages and r.meta[pi + 1].state == ST_WPROBE and
                            r.meta[pi + 1].age + 1 >= g.t1)
                        {
                            pi += 1;
                            r.meta[pi].age +|= 1;
                        }
                        for (r.meta[first .. pi + 1]) |*mm| {
                            mm.state = ST_RPROBE;
                            mm.age = 0;
                        }
                        mprotectPages(r.base, first, pi - first + 1, linux.PROT.NONE);
                    }
                },
                ST_RPROBE => {
                    m.age +|= 1;
                    if (m.age >= g.t2) {
                        // compress this page: need read access
                        mprotectPages(r.base, pi, 1, linux.PROT.READ);
                        const page: *const [PAGE]u8 = @ptrFromInt(r.base + pi * PAGE);
                        const clen_opt = lzCompress(page, comp_buf[0 .. CLASS_SIZES[NCLASS - 1]]);
                        var stored = false;
                        if (clen_opt) |clen| {
                            if (classFor(clen)) |cls| {
                                if (allocSlot(cls)) |slot| {
                                    @memcpy(g.slot_ptr.items[slot][0..clen], comp_buf[0..clen]);
                                    m.ref = slot;
                                    m.clen = @intCast(clen);
                                    m.state = ST_ATTIC;
                                    m.age = 0;
                                    madvisePages(r.base, pi, 1);
                                    mprotectPages(r.base, pi, 1, linux.PROT.NONE);
                                    r.pages_compressed += 1;
                                    r.comp_out += clen;
                                    r.in_attic += 1;
                                    stored = true;
                                }
                            }
                        }
                        if (!stored) {
                            m.state = ST_HOT;
                            m.flags |= FL_NOCOMP;
                            m.age = 0;
                            r.nocomp += 1;
                            mprotectPages(r.base, pi, 1, linux.PROT.READ | linux.PROT.WRITE);
                        }
                    }
                },
                else => {}, // ST_ATTIC: stays until faulted
            }
        }
    }
    if (g.epoch_no % 8 == 0) dumpStats();
}

/// stderr counter dump (gated on ZMIX_COLDRAM_STATS); called periodically
/// from the sweep and once from PredictorLex.deinit for the end state.
pub fn dumpStats() void {
    if (comptime !attic) return; // pool-only has no ranges and no counters
    if (!g.stats) return;
    var buf: [256]u8 = undefined;
    for (g.ranges[0..g.nranges]) |*r| {
        const line = std.fmt.bufPrint(&buf, "COLDRAM {s} pages={d} attic_now={d} comp_total={d} nocomp={d} probe_faults={d} attic_faults={d} comp_out_kb={d}\n", .{
            r.name,          r.npages,        r.in_attic, r.pages_compressed,
            r.nocomp,        r.probe_faults,  r.attic_faults, r.comp_out >> 10,
        }) catch continue;
        _ = linux.write(2, line.ptr, line.len);
    }
    const line2 = std.fmt.bufPrint(&buf, "COLDRAM epoch={d} attic_chunks_kb={d}\n", .{ g.epoch_no, g.attic_bytes >> 10 }) catch return;
    _ = linux.write(2, line2.ptr, line2.len);
}

// ---------------------------------------------------------------------------
// fault recovery
// ---------------------------------------------------------------------------
fn onSegv(sig: i32, info: *const std.posix.siginfo_t, ctx: ?*anyopaque) callconv(.c) void {
    _ = ctx;
    const addr = @intFromPtr(info.fields.sigfault.addr);
    for (g.ranges[0..g.nranges]) |*r| {
        if (addr < r.base or addr >= r.base + r.npages * PAGE) continue;
        const pi = (addr - r.base) / PAGE;
        const m = &r.meta[pi];
        switch (m.state) {
            ST_WPROBE, ST_RPROBE => {
                mprotectPages(r.base, pi, 1, linux.PROT.READ | linux.PROT.WRITE);
                m.state = ST_HOT;
                m.age = 0;
                r.probe_faults += 1;
                return;
            },
            ST_ATTIC => {
                mprotectPages(r.base, pi, 1, linux.PROT.READ | linux.PROT.WRITE);
                const page: *[PAGE]u8 = @ptrFromInt(r.base + pi * PAGE);
                const slotp = g.slot_ptr.items[m.ref];
                lzDecompress(slotp[0..m.clen], page);
                freeSlot(m.ref);
                m.ref = NIL;
                m.state = ST_HOT;
                m.age = 0;
                r.attic_faults += 1;
                r.in_attic -= 1;
                return;
            },
            else => {},
        }
        break;
    }
    // not ours: hand back to the previous disposition and re-execute
    std.posix.sigaction(std.posix.SIG.SEGV, &g.prev_sa, null);
    _ = sig;
}

// ---------------------------------------------------------------------------
// tests (codec is always compiled; the state machine needs -Dcoldram)
// ---------------------------------------------------------------------------
test "lz codec roundtrip on structured and random pages" {
    var prng = std.Random.DefaultPrng.init(42);
    const rnd = prng.random();
    var src: [PAGE]u8 = undefined;
    var dst: [PAGE * 2]u8 = undefined;
    var back: [PAGE]u8 = undefined;

    // all-zero page
    @memset(&src, 0);
    const z = lzCompress(&src, &dst).?;
    try std.testing.expect(z < 64);
    lzDecompress(dst[0..z], &back);
    try std.testing.expectEqualSlices(u8, &src, &back);

    // f32-like structured data (small magnitudes, repeating exponents)
    for (0..PAGE / 4) |i| {
        const v: f32 = @as(f32, @floatFromInt(i % 37)) * 0.001;
        @memcpy(src[i * 4 ..][0..4], std.mem.asBytes(&v));
    }
    const s = lzCompress(&src, &dst).?;
    lzDecompress(dst[0..s], &back);
    try std.testing.expectEqualSlices(u8, &src, &back);

    // sparse state bytes
    @memset(&src, 0);
    for (0..60) |i| src[(i * 67) % PAGE] = @intCast(1 + i % 40);
    const sp = lzCompress(&src, &dst).?;
    try std.testing.expect(sp < 1024);
    lzDecompress(dst[0..sp], &back);
    try std.testing.expectEqualSlices(u8, &src, &back);

    // random (incompressible) — must still roundtrip when given slack
    rnd.bytes(&src);
    if (lzCompress(&src, &dst)) |rlen| {
        lzDecompress(dst[0..rlen], &back);
        try std.testing.expectEqualSlices(u8, &src, &back);
    }

    // random-ish with runs
    var k: usize = 0;
    while (k < PAGE) {
        const run = 1 + rnd.uintLessThan(usize, 40);
        const b = rnd.int(u8);
        const end = @min(PAGE, k + run);
        @memset(src[k..end], b);
        k = end;
    }
    const rr = lzCompress(&src, &dst).?;
    lzDecompress(dst[0..rr], &back);
    try std.testing.expectEqualSlices(u8, &src, &back);
}

test "lz codec fuzz roundtrip" {
    var prng = std.Random.DefaultPrng.init(1234);
    const rnd = prng.random();
    var src: [PAGE]u8 = undefined;
    var dst: [PAGE * 2]u8 = undefined;
    var back: [PAGE]u8 = undefined;
    for (0..200) |_| {
        // mixture: random spans + copied spans + constant runs
        var k: usize = 0;
        while (k < PAGE) {
            const kind = rnd.uintLessThan(u8, 3);
            const run = 1 + rnd.uintLessThan(usize, 200);
            const end = @min(PAGE, k + run);
            switch (kind) {
                0 => rnd.bytes(src[k..end]),
                1 => @memset(src[k..end], rnd.int(u8)),
                else => {
                    if (k > 0) {
                        const off = 1 + rnd.uintLessThan(usize, k);
                        for (k..end) |j| src[j] = src[j - off];
                    } else rnd.bytes(src[k..end]);
                },
            }
            k = end;
        }
        if (lzCompress(&src, &dst)) |n| {
            lzDecompress(dst[0..n], &back);
            try std.testing.expectEqualSlices(u8, &src, &back);
        }
    }
}
