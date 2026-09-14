//! PPMd byte model, ported faithfully from cmix `models/ppmd.{h,cpp}`.
//!
//! PPMd (variant H / "mod_ppmd_v2" by Dmitry Shkarin, adapted by Eugene
//! Shelwien) is an order-N context model with a custom sub-allocator that packs
//! its data structures into a raw byte heap addressed by `u32` indices
//! (`Ptr2Indx`/`Indx2Ptr`). This port preserves that exact index arithmetic
//! using a `[]u8` heap and real Zig pointers into it.
//!
//! Each byte, the model walks its suffix contexts producing a set of
//! (symbol, freq, total) fractions (`SQ`), which `ConvertSQ` turns into a
//! 256-symbol probability table (`sqp`). The `ByteModel` base then converts that
//! distribution into per-bit predictions.
//!
//! Constructed in cmix as `PPMD(25, 14000, bit_context, vocab)`:
//!   order = 25, memory = 14000 (MB, passed to StartSubAllocator as `SASize<<20`).
//!
//! C `unsigned` arithmetic wraps; wrapping ops (`+%`,`-%`,`*%`) are used where
//! the reference relies on it. Zig `<<`/`>>` on fixed-width unsigneds truncate,
//! matching C.
//!
//! -Dppmd-mmap comptime-selects the cmix-lex `mmap_to_disk` heap backing
//! (file-backed ./ppm.temp + MADV_DONTNEED residency cadence — see the
//! mmap_to_disk block below). Default OFF = plain RAM arena, the exact
//! pre-existing code path.
const std = @import("std");
const build_options = @import("build_options");
const zero_alloc = @import("../zero_alloc.zig");
const ByteModel = @import("byte_model.zig").ByteModel;
const Model = @import("../model.zig").Model;
const vtableFor = @import("../model.zig").vtableFor;

// ---- mmap_to_disk (cmix-lex ppmd.cpp:5-18, 42-59) --------------------------
//
// Port of the shipped cmix-lex disk-backed heap (`bool mmap_to_disk = true`,
// ppmd.cpp:48). When -Dppmd-mmap is ON, the sub-allocator heap lives in a
// file-backed MAP_SHARED mapping of ./ppm.temp instead of anonymous RAM.
// Output-neutral: only the backing storage changes, never the heap bytes.
//
// RSS control (their header comment, ppmd.cpp:5-18): PPMd keeps raw pointers
// into HeapStart, so the mapping must stay at a stable address. The older
// fx2-cmix munmap+mmapeviction cycle destroyed the VMA and relied on the
// kernel returning the same address; cmix-lex instead keeps the mapping stable
// and drops residency with a deterministic MADV_DONTNEED cadence (file-backed
// PTEs are zapped, VmRSS falls, later faults reload the same bytes from
// ppm.temp/page cache). MADV_RANDOM is applied because the heap is
// pointer-chased, not sequential (kills useless readahead), and the file is
// opened O_NOATIME to avoid metadata writes from repeated page-fault reads.
const mmap_to_disk: bool = build_options.ppmd_mmap;
// ppmd.cpp:50 uses a fixed "ppm.temp" in CWD; that O_TRUNCs a concurrent
// flag-ON process's LIVE heap in the same directory (fleet A/Bs from one
// checkout, parallel test steps) → silent zero-filled refaults → desynced
// archive with exit 0. A PID suffix makes the name process-unique; sequential
// predictors within one process reuse it safely (create→delete→create).
var mmap_path_buf: [64]u8 = undefined;
var mmap_path: []const u8 = "ppm.temp"; // reformatted with the PID in StartSubAllocator

// ---- Windows arena backing (task #51) --------------------------------------
// Same design, Windows substrate: CreateFileMappingW over ./ppm.<pid>.temp +
// MapViewOfFile replaces mmap(MAP_SHARED); the residency valve becomes
// VirtualUnlock, whose documented behaviour on an UNLOCKED range is exactly
// the trim needed — "releases the pages from the process's working set" — and
// it returns FALSE with ERROR_NOT_LOCKED in that case, so the return value is
// deliberately ignored. Mapped-view pages are section (page-cache) pages:
// trimming the working set preserves content (dirty pages drain through the
// mapped-page writer), mirroring MADV_DONTNEED-on-MAP_SHARED, so heap bytes —
// and therefore the coded stream — cannot change. The mapping handle is kept
// for StopSubAllocator; the file handle is closed right after mapping (the
// section holds its own reference), and the temp file is deleted by path once
// the view and section handles are gone.
const builtin = @import("builtin");
const is_windows = builtin.os.tag == .windows;
const wink32 = if (is_windows) struct {
    const W = std.os.windows;
    pub extern "kernel32" fn CreateFileMappingW(hFile: W.HANDLE, lpAttr: ?*anyopaque, flProtect: W.DWORD, maxHi: W.DWORD, maxLo: W.DWORD, name: ?[*:0]const u16) callconv(.winapi) ?W.HANDLE;
    pub extern "kernel32" fn MapViewOfFile(hMap: W.HANDLE, access: W.DWORD, offHi: W.DWORD, offLo: W.DWORD, len: usize) callconv(.winapi) ?[*]u8;
    pub extern "kernel32" fn UnmapViewOfFile(base: ?*const anyopaque) callconv(.winapi) W.BOOL;
    pub extern "kernel32" fn VirtualUnlock(addr: ?*anyopaque, len: usize) callconv(.winapi) W.BOOL;
    pub const PAGE_READWRITE: W.DWORD = 0x04;
    pub const FILE_MAP_READ: W.DWORD = 0x0004;
    pub const FILE_MAP_WRITE: W.DWORD = 0x0002;
} else struct {};
var win_map_handle: if (is_windows) std.os.windows.HANDLE else void = undefined;

inline fn arenaSelfPid() u32 {
    return if (comptime is_windows)
        std.os.windows.GetCurrentProcessId()
    else
        @intCast(std.os.linux.getpid());
}
/// cmix-lex CMIX_PPMD_REMAP_INTERVAL (ppmd.cpp:55-59): drop heap residency
/// every N processed bytes (default 5000 = record parity). BUILD-TUNABLE
/// (-Dppmd-evict-bytes) and PER-OP settable at runtime (`setEvictInterval`):
/// the gated MaxRSS is a HIGH-WATER mark, so the between-eviction file-resident
/// accumulation R(interval) sets the gate number: mature e9 anchors are
/// R(25,000) = 3,948,192 kB, R(2000) = 680,260 kB, R(1000) = 421,180 kB
/// (refit exponent 0.6919; the terminal measured R(1000) was 412,148, i.e.
/// slightly more favourable than the fit).
/// ⛔ THE PER-OP SPLIT THIS COMMENT USED TO ARGUE FOR IS FALSIFIED :
/// "DECODE has ~1GB headroom and wants a LOOSE interval" was measured against
/// an IMPOSED cgroup cap, which hides exactly the resident file pages the gate
/// counts. Uncapped, BOTH ops need 1,000 — decode 9,586,656 kB PASS (12.83-12.92
/// GB FAIL at 25,000), encode projected 9,557,515-9,635,978 kB PASS (9,816,595-
/// 9,895,058 FAIL at 2000). ship.zig / form1.zig now set 1,000 on both, which is
/// the same single-cadence-for-both-ops shape cmix-lex ships (its one interval
/// is 5,000; ours is tighter on purpose). The runtime hook is kept because it is
/// the mechanism by which the shipped binary overrides this build default.
/// Pure RSS/wall trade — madvise timing CANNOT change heap content (valve
/// proven output-neutral byte-exact at 500/5000/25k/100k, and the composed
/// 2000-vs-1000 pair emitted sha-identical archives) — so per-op values remain
/// decode-safe: encoder and decoder need not agree on it.
var mmap_remap_interval_bytes: u64 = build_options.ppmd_evict_bytes;

/// Set the valve eviction cadence for the CURRENT operation (ship.zig and
/// form1.zig: 1,000 on BOTH ops — gate-required, see above). Timing-only;
/// content-invariant.
pub fn setEvictInterval(bytes: u64) void {
    mmap_remap_interval_bytes = if (bytes == 0) 5000 else bytes;
}

// `counter_` / `last_mmap_remap_counter_` (ppmd.cpp:1395-1396) — file-scope
// globals exactly like the C++. Referenced only when mmap_to_disk is ON
// (flag OFF compiles them and every cadence branch out entirely).
var counter_: u64 = 0;
var last_mmap_remap_counter_: u64 = 0;

// ---- constants (ppmd_Model enums) -----------------------------------------

const SCALE: u32 = 1 << 15;

const UNIT_SIZE: u32 = 12;
const N1: u32 = 4;
const N2: u32 = 4;
const N3: u32 = 4;
const N4: u32 = (128 + 3 - 1 * N1 - 2 * N2 - 3 * N3) / 4; // = 26
const N_INDEXES: u32 = N1 + N2 + N3 + N4; // = 38

const MAX_O: usize = 256; // ORealMAX

const UP_FREQ: u32 = 5;
const MAX_FREQ: u32 = 124;
const O_BOUND: i32 = 9;

// SEE2 / binary-context scaling
const INT_BITS: u32 = 7;
const PERIOD_BITS: u32 = 7;
const TOT_BITS: u32 = INT_BITS + PERIOD_BITS;
const INTERVAL: u32 = 1 << INT_BITS; // 128
const BIN_SCALE: u32 = 1 << TOT_BITS; // 16384

const EscCoef = [12]i8{ 16, -10, 1, 51, 14, 89, 23, 35, 64, 26, -42, 43 };
const ExpEscape = [16]u8{ 51, 43, 18, 12, 11, 9, 8, 7, 6, 5, 4, 3, 3, 2, 2, 2 };

/// Port of ppmd.cpp `PrefetchData(void*)` (`*(volatile byte*)Addr`): warm the
/// cache line ahead of the pointer chase. Speed-only hint; output-neutral.
inline fn prefetchData(ptr: anytype) void {
    @prefetch(ptr, .{ .rw = .read, .locality = 3, .cache = .data });
}

// ---- packed heap structures (all pack(1) equivalents) ---------------------

// STATE: 6 bytes, iSuccessor unaligned u32 at offset 2 (matches `#pragma pack(1)`).
const STATE = extern struct {
    Symbol: u8,
    Freq: u8,
    iSuccessor: u32 align(1),
};

// PPM_CONTEXT: 12 bytes; extern layout already matches pack(1) (no padding).
const PPM_CONTEXT = extern struct {
    NumStats: u8,
    Flags: u8,
    SummFreq: u16,
    iStats: u32,
    iSuffix: u32,
};

const BLK_NODE = extern struct {
    Stamp: u32,
    NextIndx: u32,
    inline fn avail(self: *align(1) const BLK_NODE) bool {
        return self.NextIndx != 0;
    }
};

const MEM_BLK = extern struct {
    Stamp: u32,
    NextIndx: u32,
    NU: u32,
};

const SEE2_CONTEXT = extern struct {
    Summ: u16 = 0,
    Shift: u8 = 0,
    Count: u8 = 0,

    fn init(self: *SEE2_CONTEXT, InitVal: u32) void {
        self.Shift = @intCast(PERIOD_BITS - 4);
        self.Summ = @truncate(InitVal << @intCast(self.Shift));
        self.Count = 7;
    }
    fn getMean(self: *SEE2_CONTEXT) u32 {
        return self.Summ >> @intCast(self.Shift);
    }
    fn update(self: *SEE2_CONTEXT) void {
        self.Count -%= 1;
        if (self.Count == 0) self.setShift_rare();
    }
    fn setShift_rare(self: *SEE2_CONTEXT) void {
        var i: u32 = self.Summ >> @intCast(self.Shift);
        i = PERIOD_BITS - @intFromBool(i > 40) - @intFromBool(i > 280) - @intFromBool(i > 1020);
        if (i < self.Shift) {
            self.Summ >>= 1;
            self.Shift -= 1;
        } else if (i > self.Shift) {
            self.Summ <<= 1;
            self.Shift += 1;
        }
        self.Count = @truncate(@as(u32, 5) << @intCast(self.Shift));
    }
};

const qsym = extern struct {
    sym: u16,
    freq: u16,
    total: u16,

    inline fn store(self: *qsym, s: u32, f: u32, t: u32) void {
        self.sym = @truncate(s);
        self.freq = @truncate(f);
        self.total = @truncate(t);
    }
};

// pointer aliases (align(1) — heap allocations may sit at 2/4-aligned addrs)
const PC = *align(1) PPM_CONTEXT;
const ST = [*]align(1) STATE;

// ---------------------------------------------------------------------------

pub const PpmdModel = struct {
    alloc: std.mem.Allocator,
    heap: []align(16) u8 = &.{},
    HeapStart: [*]u8 = undefined,

    // sub-allocator state
    BList: [N_INDEXES + 1]BLK_NODE = undefined,
    GlueCount: u32 = 0,
    GlueCount1: u32 = 0,
    SubAllocatorSize: u64 = 0,
    pText: [*]u8 = undefined,
    UnitsStart: [*]u8 = undefined,
    LoUnit: [*]u8 = undefined,
    HiUnit: [*]u8 = undefined,
    AuxUnit: [*]u8 = undefined,

    Indx2Units: [N_INDEXES]u8 = undefined,
    Units2Indx: [128]u8 = undefined,
    NS2BSIndx: [256]u8 = undefined,
    QTable: [260]u8 = undefined,

    _MaxOrder: i32 = 0,
    _CutOff: i32 = 0,
    _MMAX: i32 = 0,
    _filesize: u32 = 0,
    OrderFall: i32 = 0,

    FoundState: ?ST = null,
    MaxContext: PC = undefined,

    EscCount: u32 = 0,
    CharMask: [256]u32 = undefined,

    BSumm: i32 = 0,
    RunLength: i32 = 0,
    InitRL: i32 = 0,
    NumMasked: i32 = 0,
    PrevSuccess: i32 = 0,

    BinSumm: [25][64]u16 = undefined,
    SEE2Cont: [23][32]SEE2_CONTEXT = undefined,
    DummySEE2Cont: SEE2_CONTEXT = .{},
    saved_pc: PC = undefined,

    SQ: [1024]qsym = undefined,
    SQ_ptr: u32 = 0,
    sqp: [256]u32 = undefined,
    // Bit-tree accumulators (real cmix-lex `ppmd_Model::trF/trT`, ppmd.cpp:1193-1194).
    // Filled by ConvertSQ; consumed by the lex bit-tree wrapper (PPMDLex).
    // Harmless to the v21 `PPMD` wrapper, which never reads them.
    trF: [256]u32 = undefined,
    trT: [256]u32 = undefined,

    cxt: u32 = 0,
    y: u32 = 0,

    // ---- index <-> pointer conversion -------------------------------------

    inline fn heapOff(self: *PpmdModel, p: anytype) u64 {
        return @intFromPtr(p) - @intFromPtr(self.HeapStart);
    }
    inline fn limVal(self: *PpmdModel) u32 {
        return @intCast(@intFromPtr(self.UnitsStart) - @intFromPtr(self.HeapStart));
    }

    fn Ptr2Indx(self: *PpmdModel, p: anytype) u32 {
        const addr: u64 = self.heapOff(p);
        const lim: u64 = self.limVal();
        if (addr >= lim) {
            return @intCast((addr - lim) / UNIT_SIZE + lim);
        } else {
            return @intCast(addr);
        }
    }

    fn Indx2bytePtr(self: *PpmdModel, indx: u32) [*]u8 {
        const lim: u64 = self.limVal();
        const addr: u64 = if (indx >= lim)
            @as(u64, indx - lim) * UNIT_SIZE + lim
        else
            indx;
        return self.HeapStart + addr;
    }
    inline fn Indx2ctx(self: *PpmdModel, indx: u32) PC {
        return @ptrCast(self.Indx2bytePtr(indx));
    }
    inline fn Indx2state(self: *PpmdModel, indx: u32) ST {
        return @ptrCast(self.Indx2bytePtr(indx));
    }

    // ---- struct accessors --------------------------------------------------

    inline fn oneState(pc: PC) ST {
        const bp: [*]align(1) u8 = @ptrCast(pc);
        return @ptrCast(bp + 2);
    }
    inline fn getStats(self: *PpmdModel, pc: PC) ST {
        return self.Indx2state(pc.iStats);
    }
    inline fn suff(self: *PpmdModel, pc: PC) PC {
        return self.Indx2ctx(pc.iSuffix);
    }
    inline fn getSucc(self: *PpmdModel, p: ST) PC {
        return self.Indx2ctx(p[0].iSuccessor);
    }

    inline fn U2B(NU: u32) u32 {
        return UNIT_SIZE * NU;
    }

    // ---- BLK_NODE list helpers --------------------------------------------

    inline fn getNext(self: *PpmdModel, this: *align(1) BLK_NODE) *align(1) BLK_NODE {
        return @ptrCast(self.Indx2bytePtr(this.NextIndx));
    }
    inline fn setNext(self: *PpmdModel, this: *align(1) BLK_NODE, p: *align(1) BLK_NODE) void {
        this.NextIndx = self.Ptr2Indx(p);
    }
    inline fn link(self: *PpmdModel, this: *align(1) BLK_NODE, p: *align(1) BLK_NODE) void {
        p.NextIndx = this.NextIndx;
        self.setNext(this, p);
    }
    inline fn unlink(self: *PpmdModel, this: *align(1) BLK_NODE) void {
        this.NextIndx = self.getNext(this).NextIndx;
    }
    fn remove(self: *PpmdModel, this: *align(1) BLK_NODE) [*]u8 {
        const p = self.getNext(this);
        self.unlink(this);
        this.Stamp -%= 1;
        return @ptrCast(p);
    }
    fn insert(self: *PpmdModel, this: *align(1) BLK_NODE, pv: [*]u8, NU: u32) void {
        const p: *align(1) BLK_NODE = @ptrCast(pv);
        self.link(this, p);
        p.Stamp = ~@as(u32, 0);
        const mb: *align(1) MEM_BLK = @ptrCast(pv);
        mb.NU = NU;
        this.Stamp +%= 1;
    }

    // ---- sub-allocator ----------------------------------------------------

    fn StartSubAllocator(self: *PpmdModel, SASize: u64) bool {
        const t: u64 = SASize << 20;
        if (mmap_to_disk) {
            // ppmd.cpp:167-185: open(O_RDWR|O_CREAT|O_TRUNC|O_NOATIME, 0664) +
            // ftruncate + mmap(PROT_READ|WRITE, MAP_SHARED) + MADV_RANDOM +
            // close(fd). The C++ exit(EXIT_FAILURE)s on any syscall failure —
            // the model cannot run without its heap, so hard-fail likewise.
            // ftruncate creates a sparse file: disk blocks are only allocated
            // for pages actually dirtied. Page-aligned mmap satisfies the
            // heap's 16-byte alignment requirement.
            mmap_path = std.fmt.bufPrint(&mmap_path_buf, "ppm.{d}.temp", .{arenaSelfPid()}) catch unreachable;
            if (comptime is_windows) {
                // Windows substrate (see the wink32 block above). No
                // MADV_RANDOM analogue is needed: it is advisory readahead
                // control, and mapped views are demand-faulted per page.
                const f = std.fs.cwd().createFile(mmap_path, .{ .read = true, .truncate = true }) catch
                    @panic("ppmd mmap_to_disk: create ppm.temp failed");
                f.setEndPos(t) catch @panic("ppmd mmap_to_disk: setEndPos ppm.temp failed");
                const hmap = wink32.CreateFileMappingW(
                    f.handle,
                    null,
                    wink32.PAGE_READWRITE,
                    @intCast(t >> 32),
                    @truncate(t),
                    null,
                ) orelse @panic("ppmd mmap_to_disk: CreateFileMappingW failed");
                win_map_handle = hmap;
                const base = wink32.MapViewOfFile(
                    hmap,
                    wink32.FILE_MAP_READ | wink32.FILE_MAP_WRITE,
                    0,
                    0,
                    @intCast(t),
                ) orelse @panic("ppmd mmap_to_disk: MapViewOfFile failed");
                f.close(); // the section holds its own file reference
                self.heap = @alignCast(base[0..@intCast(t)]); // 64K view granularity >= 16
            } else {
                const fd = std.posix.open(
                    mmap_path,
                    .{ .ACCMODE = .RDWR, .CREAT = true, .TRUNC = true, .NOATIME = true },
                    0o664,
                ) catch @panic("ppmd mmap_to_disk: open ppm.temp failed");
                std.posix.ftruncate(fd, t) catch @panic("ppmd mmap_to_disk: ftruncate ppm.temp failed");
                const mem = std.posix.mmap(
                    null,
                    @intCast(t),
                    std.posix.PROT.READ | std.posix.PROT.WRITE,
                    .{ .TYPE = .SHARED },
                    fd,
                    0,
                ) catch @panic("ppmd mmap_to_disk: mmap ppm.temp failed");
                std.posix.madvise(mem.ptr, mem.len, std.posix.MADV.RANDOM) catch
                    @panic("ppmd mmap_to_disk: madvise(MADV_RANDOM) failed");
                std.posix.close(fd);
                self.heap = mem; // page (4K) alignment >= 16: implicit coercion
            }
        } else {
            self.heap = self.alloc.alignedAlloc(u8, .@"16", @intCast(t)) catch return false;
            zero_alloc.adviseHuge(self.heap); // 14GB random-access heap: THP cuts dTLB misses
        }
        self.HeapStart = self.heap.ptr;
        self.SubAllocatorSize = t;
        return true;
    }

    fn InitSubAllocator(self: *PpmdModel) void {
        @memset(std.mem.asBytes(&self.BList), 0);
        self.pText = self.HeapStart;
        self.HiUnit = self.HeapStart + self.SubAllocatorSize;
        const Diff: u64 = self.SubAllocatorSize / 8 / UNIT_SIZE * 7 * UNIT_SIZE;
        self.LoUnit = self.HiUnit - Diff;
        self.UnitsStart = self.LoUnit;
        self.GlueCount = 0;
        self.GlueCount1 = 0;
    }

    fn GetUsedMemory(self: *PpmdModel) u64 {
        var RetVal: u64 = self.SubAllocatorSize -
            (@intFromPtr(self.HiUnit) - @intFromPtr(self.LoUnit)) -
            (@intFromPtr(self.UnitsStart) - @intFromPtr(self.pText));
        var i: u32 = 0;
        while (i < N_INDEXES) : (i += 1) {
            RetVal -%= @as(u64, self.Indx2Units[i]) *% self.BList[i].Stamp *% 12;
        }
        return RetVal;
    }

    fn StopSubAllocator(self: *PpmdModel) void {
        if (self.SubAllocatorSize != 0) {
            self.SubAllocatorSize = 0;
            if (mmap_to_disk) {
                // The C++ never reaches its StopSubAllocator in mmap mode (the
                // only call site, ppmd.cpp:1328, is commented out); its cleanup
                // is ~PPMD-> remove(mmap_path) with the munmap left to
                // process exit. zmix's destroydoes call here, so centralize
                // both: unmap the VMA and delete ppm.temp (errors ignored like
                // the C++'s unchecked remove). After a crash a stale sparse
                // ppm.temp survives; the next run's O_TRUNC open reclaims it.
                if (comptime is_windows) {
                    _ = wink32.UnmapViewOfFile(self.heap.ptr);
                    std.os.windows.CloseHandle(win_map_handle);
                } else {
                    std.posix.munmap(@alignCast(self.heap));
                }
                std.fs.cwd().deleteFile(mmap_path) catch {};
            } else {
                self.alloc.free(self.heap);
            }
            self.heap = &.{};
        }
    }

    /// Port of cmix-lex `DropPpmHeapResidency` (ppmd.cpp:1398-1405):
    /// MADV_DONTNEED clears present PTEs for the whole shared file mapping and
    /// drops them from VmRSS. Later faults reload the same bytes from
    /// ppm.temp/page cache, so compression decisions stay bit-identical
    /// without remapping. The C++ exits on madvise failure; mirror that.
    fn dropHeapResidency(self: *PpmdModel) void {
        if (comptime is_windows) {
            // The unlocked-range VirtualUnlock trim (see the wink32 block):
            // FALSE/ERROR_NOT_LOCKED is the documented success shape here.
            _ = wink32.VirtualUnlock(self.heap.ptr, self.heap.len);
        } else {
            std.posix.madvise(@alignCast(self.heap.ptr), self.heap.len, std.posix.MADV.DONTNEED) catch
                @panic("ppmd mmap_to_disk: madvise(MADV_DONTNEED) failed");
        }
    }

    fn GlueFreeBlocks(self: *PpmdModel) void {
        var s0: MEM_BLK = undefined;
        const p0_init: *align(1) BLK_NODE = @ptrCast(&s0);

        if (@intFromPtr(self.LoUnit) != @intFromPtr(self.HiUnit)) self.LoUnit[0] = 0;

        p0_init.NextIndx = 0;
        var p0: [*]align(1) MEM_BLK = @ptrCast(&s0);
        var i: u32 = 0;
        while (i <= N_INDEXES) : (i += 1) {
            while (self.BList[i].avail()) {
                var p: [*]align(1) MEM_BLK = @ptrCast(self.remove(&self.BList[i]));
                if (p[0].NU != 0) {
                    while (true) {
                        const p1 = p + p[0].NU;
                        if (p1[0].Stamp != ~@as(u32, 0)) break;
                        p[0].NU += p1[0].NU;
                        p1[0].NU = 0;
                    }
                    self.link(@ptrCast(p0), @ptrCast(p));
                    p0 = p;
                }
            }
        }

        while (@as(*align(1) BLK_NODE, @ptrCast(&s0)).avail()) {
            var p: [*]align(1) MEM_BLK = @ptrCast(self.remove(@ptrCast(&s0)));
            var sz: u32 = p[0].NU;
            if (sz != 0) {
                while (sz > 128) : (sz -= 128) {
                    self.insert(&self.BList[N_INDEXES - 1], @ptrCast(p), 128);
                    p += 128;
                }
                var idx: u32 = self.Units2Indx[sz - 1];
                if (self.Indx2Units[idx] != sz) {
                    idx -= 1;
                    const k = sz - self.Indx2Units[idx];
                    self.insert(&self.BList[k - 1], @ptrCast(p + (sz - k)), k);
                }
                self.insert(&self.BList[idx], @ptrCast(p), self.Indx2Units[idx]);
            }
        }

        self.GlueCount = @as(u32, 1) << @intCast(13 + self.GlueCount1);
        self.GlueCount1 += 1;
    }

    fn SplitBlock(self: *PpmdModel, pv: [*]u8, OldIndx: u32, NewIndx: u32) void {
        var UDiff: u32 = self.Indx2Units[OldIndx] - self.Indx2Units[NewIndx];
        var p: [*]u8 = pv + U2B(self.Indx2Units[NewIndx]);
        var i: u32 = self.Units2Indx[UDiff - 1];
        if (self.Indx2Units[i] != UDiff) {
            i -= 1;
            const k = self.Indx2Units[i];
            self.insert(&self.BList[i], p, k);
            p += U2B(k);
            UDiff -= k;
        }
        self.insert(&self.BList[self.Units2Indx[UDiff - 1]], p, UDiff);
    }

    fn AllocUnitsRare(self: *PpmdModel, indx: u32) ?[*]u8 {
        var i: u32 = indx;
        while (true) {
            i += 1;
            if (i == N_INDEXES) {
                const old = self.GlueCount;
                self.GlueCount -%= 1;
                if (old == 0) {
                    self.GlueFreeBlocks();
                    i = indx;
                    if (self.BList[i].avail()) return self.remove(&self.BList[i]);
                } else {
                    const b = U2B(self.Indx2Units[indx]);
                    if (@intFromPtr(self.UnitsStart) - @intFromPtr(self.pText) > b) {
                        self.UnitsStart -= b;
                        return self.UnitsStart;
                    } else return null;
                }
            }
            if (self.BList[i].avail()) break;
        }
        const RetVal = self.remove(&self.BList[i]);
        self.SplitBlock(RetVal, i, indx);
        return RetVal;
    }

    fn AllocUnits(self: *PpmdModel, NU: u32) ?[*]u8 {
        const indx: u32 = self.Units2Indx[NU - 1];
        if (self.BList[indx].avail()) return self.remove(&self.BList[indx]);
        const RetVal = self.LoUnit;
        self.LoUnit += U2B(self.Indx2Units[indx]);
        if (@intFromPtr(self.LoUnit) <= @intFromPtr(self.HiUnit)) return RetVal;
        self.LoUnit -= U2B(self.Indx2Units[indx]);
        return self.AllocUnitsRare(indx);
    }

    fn AllocContext(self: *PpmdModel) ?[*]u8 {
        if (@intFromPtr(self.HiUnit) != @intFromPtr(self.LoUnit)) {
            self.HiUnit -= UNIT_SIZE;
            return self.HiUnit;
        }
        if (self.BList[0].avail()) return self.remove(&self.BList[0]);
        return self.AllocUnitsRare(0);
    }

    fn FreeUnits(self: *PpmdModel, ptr: [*]u8, NU: u32) void {
        const indx: u32 = self.Units2Indx[NU - 1];
        self.insert(&self.BList[indx], ptr, self.Indx2Units[indx]);
    }

    fn FreeUnit(self: *PpmdModel, ptr: [*]u8) void {
        const i: u32 = if (@intFromPtr(ptr) > @intFromPtr(self.UnitsStart) + 128 * 1024) 0 else N_INDEXES;
        self.insert(&self.BList[i], ptr, 1);
    }

    fn UnitsCpy(Dest: [*]u8, Src: [*]u8, NU: u32) void {
        @memcpy(Dest[0 .. 12 * NU], Src[0 .. 12 * NU]);
    }

    fn ExpandUnits(self: *PpmdModel, OldPtr: [*]u8, OldNU: u32) ?[*]u8 {
        const idx0: u32 = self.Units2Indx[OldNU - 1];
        const idx1: u32 = self.Units2Indx[OldNU - 1 + 1];
        if (idx0 == idx1) return OldPtr;
        const ptr = self.AllocUnits(OldNU + 1);
        if (ptr) |pp| {
            UnitsCpy(pp, OldPtr, OldNU);
            self.insert(&self.BList[idx0], OldPtr, OldNU);
        }
        return ptr;
    }

    fn ShrinkUnits(self: *PpmdModel, OldPtr: [*]u8, OldNU: u32, NewNU: u32) [*]u8 {
        const idx0: u32 = self.Units2Indx[OldNU - 1];
        const idx1: u32 = self.Units2Indx[NewNU - 1];
        if (idx0 == idx1) return OldPtr;
        if (self.BList[idx1].avail()) {
            const ptr = self.remove(&self.BList[idx1]);
            UnitsCpy(ptr, OldPtr, NewNU);
            self.insert(&self.BList[idx0], OldPtr, self.Indx2Units[idx0]);
            return ptr;
        } else {
            self.SplitBlock(OldPtr, idx0, idx1);
            return OldPtr;
        }
    }

    fn MoveUnitsUp(self: *PpmdModel, OldPtr: [*]u8, NU: u32) [*]u8 {
        const indx: u32 = self.Units2Indx[NU - 1];
        prefetchData(OldPtr);
        if (@intFromPtr(OldPtr) > @intFromPtr(self.UnitsStart) + 128 * 1024 or
            @intFromPtr(OldPtr) > @intFromPtr(self.getNext(&self.BList[indx]))) return OldPtr;
        const ptr = self.remove(&self.BList[indx]);
        UnitsCpy(ptr, OldPtr, NU);
        self.insert(&self.BList[N_INDEXES], OldPtr, self.Indx2Units[indx]);
        return ptr;
    }

    fn PrepareTextArea(self: *PpmdModel) void {
        if (self.AllocContext()) |ctx| {
            self.AuxUnit = ctx;
            if (@intFromPtr(ctx) == @intFromPtr(self.UnitsStart)) {
                self.UnitsStart += UNIT_SIZE;
                self.AuxUnit = self.UnitsStart;
            }
        } else {
            // Alloc failure must NOT bump UnitsStart (ppmd.cpp:299-301) — the
            // following cutOff would treat one extra unit as dead text area.
            self.AuxUnit = self.UnitsStart;
        }
    }

    fn ExpandTextArea(self: *PpmdModel) void {
        var Count: [N_INDEXES]u32 = .{0} ** N_INDEXES;
        var i: u32 = 0;

        if (@intFromPtr(self.AuxUnit) != @intFromPtr(self.UnitsStart)) {
            const au: *align(1) u32 = @ptrCast(self.AuxUnit);
            if (au.* != ~@as(u32, 0))
                self.UnitsStart += UNIT_SIZE
            else
                self.insert(&self.BList[0], self.AuxUnit, 1);
        }

        while (true) {
            const p: [*]align(1) MEM_BLK = @ptrCast(self.UnitsStart);
            if (p[0].Stamp != ~@as(u32, 0)) break;
            self.UnitsStart = @as([*]u8, @ptrCast(p + p[0].NU));
            Count[self.Units2Indx[p[0].NU - 1]] += 1;
            i += 1;
            p[0].Stamp = 0;
        }

        if (i != 0) {
            var p: *align(1) BLK_NODE = &self.BList[N_INDEXES];
            while (p.NextIndx != 0) {
                while (p.NextIndx != 0 and self.getNext(p).Stamp == 0) {
                    const mb: *align(1) MEM_BLK = @ptrCast(self.getNext(p));
                    Count[self.Units2Indx[mb.NU - 1]] -= 1;
                    self.unlink(p);
                    self.BList[N_INDEXES].Stamp -%= 1;
                }
                if (p.NextIndx == 0) break;
                p = self.getNext(p);
            }

            var j: u32 = 0;
            while (j < N_INDEXES) : (j += 1) {
                var q: *align(1) BLK_NODE = &self.BList[j];
                while (Count[j] != 0) {
                    while (self.getNext(q).Stamp == 0) {
                        self.unlink(q);
                        self.BList[j].Stamp -%= 1;
                        Count[j] -= 1;
                        if (Count[j] == 0) break;
                    }
                    if (Count[j] == 0) break;
                    q = self.getNext(q);
                }
            }
        }
    }

    // ---- static tables & model init ---------------------------------------

    fn PPMD_STARTUP(self: *PpmdModel) void {
        var i: u32 = 0;
        var k: u32 = 1;
        while (i < N1) : ({
            i += 1;
            k += 1;
        }) self.Indx2Units[i] = @intCast(k);
        k += 1;
        while (i < N1 + N2) : ({
            i += 1;
            k += 2;
        }) self.Indx2Units[i] = @intCast(k);
        k += 1;
        while (i < N1 + N2 + N3) : ({
            i += 1;
            k += 3;
        }) self.Indx2Units[i] = @intCast(k);
        k += 1;
        while (i < N1 + N2 + N3 + N4) : ({
            i += 1;
            k += 4;
        }) self.Indx2Units[i] = @intCast(k);

        k = 0;
        i = 0;
        while (k < 128) : (k += 1) {
            i += @intFromBool(self.Indx2Units[i] < k + 1);
            self.Units2Indx[k] = @intCast(i);
        }

        self.NS2BSIndx[0] = 2 * 0;
        self.NS2BSIndx[1] = 2 * 1;
        self.NS2BSIndx[2] = 2 * 1;
        @memset(self.NS2BSIndx[3..][0..26], 2 * 2);
        @memset(self.NS2BSIndx[29..][0 .. 256 - 29], 2 * 3);

        i = 0;
        while (i < UP_FREQ) : (i += 1) self.QTable[i] = @intCast(i);
        var m: u32 = UP_FREQ;
        i = UP_FREQ;
        k = 1;
        var Step: u32 = 1;
        while (i < 260) : (i += 1) {
            self.QTable[i] = @intCast(m);
            k -= 1;
            if (k == 0) {
                Step += 1;
                k = Step;
                m += 1;
            }
        }
    }

    fn StartModelRare(self: *PpmdModel) void {
        var i2f: [25]u8 = undefined;
        @memset(std.mem.asBytes(&self.CharMask), 0);
        self.EscCount = 1;

        if (self._MaxOrder < 2) {
            self.OrderFall = self._MaxOrder;
            var pc = self.MaxContext;
            while (pc.iSuffix != 0) : (pc = self.suff(pc)) self.OrderFall -= 1;
            return;
        }

        self.OrderFall = self._MaxOrder;
        self.InitSubAllocator();
        self.InitRL = -(if (self._MaxOrder < 13) self._MaxOrder else 13);
        self.RunLength = self.InitRL;

        self.MaxContext = @ptrCast(self.AllocContext().?);
        self.MaxContext.NumStats = 255;
        self.MaxContext.SummFreq = 255 + 2;
        self.MaxContext.iStats = self.Ptr2Indx(self.AllocUnits(256 / 2).?);
        self.MaxContext.Flags = 0;
        self.MaxContext.iSuffix = 0;
        self.PrevSuccess = 0;

        const stats = self.getStats(self.MaxContext);
        var i: u32 = 0;
        while (i < 256) : (i += 1) {
            stats[i].Symbol = @intCast(i);
            stats[i].Freq = 1;
            stats[i].iSuccessor = 0;
        }

        // i2f[i] = 1 + (index of first QTable entry > i)
        var k: u32 = 0;
        i = 0;
        while (i < 25) : (i += 1) {
            while (self.QTable[k] == i) k += 1;
            i2f[i] = @intCast(k + 1);
        }

        k = 0;
        while (k < 64) : (k += 1) {
            var s: i32 = 0;
            i = 0;
            while (i < 6) : (i += 1) {
                s += EscCoef[2 * i + ((k >> @intCast(i)) & 1)];
            }
            s = 128 * clampI32(s, 32, 256 - 32);
            i = 0;
            while (i < 25) : (i += 1) {
                self.BinSumm[i][k] = @intCast(@as(i32, @intCast(BIN_SCALE)) - @divTrunc(s, i2f[i]));
            }
        }

        i = 0;
        while (i < 23) : (i += 1) {
            k = 0;
            while (k < 32) : (k += 1) self.SEE2Cont[i][k].init(8 * i + 5);
        }
    }

    // ---- helpers -----------------------------------------------------------

    inline fn stateSwap(a: ST, b: ST) void {
        const t = a[0];
        a[0] = b[0];
        b[0] = t;
    }

    // ---- rescale ----------------------------------------------------------

    fn rescale(self: *PpmdModel, q: PC, OrderFall: i32, FoundStateIn: ST) ST {
        var tmp: STATE = undefined;
        var p: ST = undefined;
        var p1: ST = undefined;
        var FoundState = FoundStateIn;

        q.Flags &= 0x14;

        p1 = self.getStats(q);
        tmp = FoundState[0];
        p = FoundState;
        while (@intFromPtr(p) != @intFromPtr(p1)) : (p -= 1) p[0] = (p - 1)[0];
        p1[0] = tmp;

        const of: i32 = @intFromBool(OrderFall != 0);
        var a: i32 = undefined;
        var i: i32 = undefined;
        const f0: i32 = p[0].Freq;
        var sf: i32 = q.SummFreq;
        var EscFreq: i32 = sf - f0;
        const nf: i32 = (f0 + of) >> 1;
        q.SummFreq = @intCast(nf);
        p[0].Freq = @intCast(nf);

        i = 0;
        while (i < q.NumStats) : (i += 1) {
            p += 1;
            a = p[0].Freq;
            EscFreq -= a;
            a = (a + of) >> 1;
            p[0].Freq = @intCast(a);
            q.SummFreq +%= @intCast(a);
            if (a != 0) q.Flags |= 0x08 * @as(u8, @intFromBool(p[0].Symbol >= 0x40));
            if (a > (p - 1)[0].Freq) {
                tmp = p[0];
                p1 = p;
                while (@as(i32, tmp.Freq) > (p1 - 1)[0].Freq) : (p1 -= 1) p1[0] = (p1 - 1)[0];
                p1[0] = tmp;
            }
        }

        if (p[0].Freq == 0) {
            i = 0;
            while (p[0].Freq == 0) : ({
                i += 1;
                p -= 1;
            }) {}
            EscFreq += i;
            a = (@as(i32, q.NumStats) + 2) >> 1;
            q.NumStats = @intCast(@as(i32, q.NumStats) - i);
            if (q.NumStats == 0) {
                tmp = self.getStats(q)[0];
                const nfreq = @min(MAX_FREQ / 3, @as(u32, @intCast(@divTrunc(2 * @as(i32, tmp.Freq) + EscFreq - 1, EscFreq))));
                tmp.Freq = @intCast(nfreq);
                q.Flags &= 0x18;
                self.FreeUnits(@ptrCast(self.getStats(q)), @intCast(a));
                oneState(q)[0] = tmp;
                FoundState = oneState(q);
                return FoundState;
            }
            q.iStats = self.Ptr2Indx(self.ShrinkUnits(@ptrCast(self.getStats(q)), @intCast(a), @intCast((@as(i32, q.NumStats) + 2) >> 1)));
        }

        q.SummFreq +%= @intCast((EscFreq + 1) >> 1);
        if (OrderFall != 0 or (q.Flags & 0x04) == 0) {
            sf -= EscFreq;
            a = sf - f0;
            const num: i32 = f0 *% @as(i32, q.SummFreq) -% sf *% @as(i32, self.getStats(q)[0].Freq) +% (a - 1);
            const u: u32 = @bitCast(@divTrunc(num, a));
            a = @intCast(clampU32(u, 2, MAX_FREQ / 2 - 18));
        } else {
            a = 2;
        }

        FoundState = self.getStats(q);
        FoundState[0].Freq +%= @intCast(a);
        q.SummFreq +%= @intCast(a);
        q.Flags |= 0x04;
        return FoundState;
    }

    // ---- cutOff (only reached under memory pressure) ----------------------

    fn AuxCutOff(self: *PpmdModel, p: ST, Order: i32, MaxOrder: i32) void {
        if (Order < MaxOrder) {
            prefetchData(self.getSucc(p));
            p[0].iSuccessor = self.cutOff(self.getSucc(p), Order + 1, MaxOrder);
        } else {
            p[0].iSuccessor = 0;
        }
    }

    fn cutOff(self: *PpmdModel, q: PC, Order: i32, MaxOrder: i32) u32 {
        var i: i32 = undefined;
        var tmp: i32 = undefined;
        var EscFreq: i32 = undefined;
        var Scale: i32 = undefined;
        var p: ST = undefined;
        var p0: ST = undefined;

        if (q.NumStats == 0) {
            var flag: i32 = 1;
            p = oneState(q);
            if (@intFromPtr(self.getSucc(p)) >= @intFromPtr(self.UnitsStart)) {
                self.AuxCutOff(p, Order, MaxOrder);
                if (p[0].iSuccessor != 0 or Order < O_BOUND) flag = 0;
            }
            if (flag != 0) {
                self.FreeUnit(@ptrCast(q));
                return 0;
            }
        } else {
            tmp = (@as(i32, q.NumStats) + 2) >> 1;
            p0 = @ptrCast(self.MoveUnitsUp(@ptrCast(self.getStats(q)), @intCast(tmp)));
            q.iStats = self.Ptr2Indx(p0);

            i = q.NumStats;
            p = p0 + @as(usize, @intCast(i));
            while (@intFromPtr(p) >= @intFromPtr(p0)) : (p -= 1) {
                if (@intFromPtr(self.getSucc(p)) < @intFromPtr(self.UnitsStart)) {
                    p[0].iSuccessor = 0;
                    stateSwap(p, p0 + @as(usize, @intCast(i)));
                    i -= 1;
                } else self.AuxCutOff(p, Order, MaxOrder);
            }

            if (i != q.NumStats and Order > 0) {
                p = p0;
                // Guard BEFORE the narrowing write: on the memory-pressure restart
                // path i can be -1, and @intCast(-1) into NumStats would trap.
                if (i < 0) {
                    self.FreeUnits(@ptrCast(p), @intCast(tmp));
                    self.FreeUnit(@ptrCast(q));
                    return 0;
                }
                q.NumStats = @intCast(i);
                if (i == 0) {
                    q.Flags = (q.Flags & 0x10) + 0x08 * @as(u8, @intFromBool(p[0].Symbol >= 0x40));
                    p[0].Freq = @intCast(1 + @divTrunc(2 * (@as(i32, p[0].Freq) - 1), @as(i32, q.SummFreq) - @as(i32, p[0].Freq)));
                    oneState(q)[0] = p[0];
                    self.FreeUnits(@ptrCast(p), @intCast(tmp));
                } else {
                    p = @ptrCast(self.ShrinkUnits(@ptrCast(p0), @intCast(tmp), @intCast((i + 2) >> 1)));
                    q.iStats = self.Ptr2Indx(p);
                    Scale = @intFromBool(@as(i32, q.SummFreq) > 16 * i);
                    q.Flags = q.Flags & @as(u8, @intCast(0x10 + 0x04 * Scale));
                    if (Scale != 0) {
                        EscFreq = q.SummFreq;
                        q.SummFreq = 0;
                        i = 0;
                        while (i <= q.NumStats) : (i += 1) {
                            EscFreq -= p[@intCast(i)].Freq;
                            p[@intCast(i)].Freq = @intCast((@as(i32, p[@intCast(i)].Freq) + 1) >> 1);
                            q.SummFreq +%= p[@intCast(i)].Freq;
                            q.Flags |= 0x08 * @as(u8, @intFromBool(p[@intCast(i)].Symbol >= 0x40));
                        }
                        EscFreq = (EscFreq + 1) >> 1;
                        q.SummFreq +%= @intCast(EscFreq);
                    } else {
                        i = 0;
                        while (i <= q.NumStats) : (i += 1) q.Flags |= 0x08 * @as(u8, @intFromBool(p[@intCast(i)].Symbol >= 0x40));
                    }
                }
            }
        }

        if (@intFromPtr(q) == @intFromPtr(self.UnitsStart)) {
            UnitsCpy(self.AuxUnit, @ptrCast(q), 1);
            return self.Ptr2Indx(self.AuxUnit);
        } else {
            if (@intFromPtr(self.suff(q)) == @intFromPtr(self.UnitsStart)) q.iSuffix = self.Ptr2Indx(self.AuxUnit);
        }
        return self.Ptr2Indx(q);
    }

    fn RestoreModelRare(self: *PpmdModel) void {
        var p: ST = undefined;
        self.pText = self.HeapStart;
        const pc = self.saved_pc;

        while (true) {
            if (self.MaxContext.NumStats == 1 and @intFromPtr(self.MaxContext) != @intFromPtr(pc)) {
                p = self.getStats(self.MaxContext);
                if (@intFromPtr(self.getSucc(p + 1)) >= @intFromPtr(self.UnitsStart)) break;
            } else break;
            self.MaxContext.Flags = (self.MaxContext.Flags & 0x10) + 0x08 * @as(u8, @intFromBool(p[0].Symbol >= 0x40));
            p[0].Freq = @intCast((@as(i32, p[0].Freq) + 1) >> 1);
            oneState(self.MaxContext)[0] = p[0];
            self.MaxContext.NumStats = 0;
            self.FreeUnits(@ptrCast(p), 1);
            self.MaxContext = self.suff(self.MaxContext);
        }

        while (self.MaxContext.iSuffix != 0) self.MaxContext = self.suff(self.MaxContext);

        self.AuxUnit = self.UnitsStart;
        self.ExpandTextArea();

        while (true) {
            self.PrepareTextArea();
            _ = self.cutOff(self.MaxContext, 0, self._MaxOrder);
            self.ExpandTextArea();
            if (!(self.GetUsedMemory() > 3 * (self.SubAllocatorSize >> 2))) break;
        }

        self.GlueCount = 0;
        self.GlueCount1 = 0;
        self.OrderFall = self._MaxOrder;
    }

    // ---- model update -----------------------------------------------------

    fn UpdateModel(self: *PpmdModel, MinContext: PC) ?PC {
        var pc: PC = undefined;
        var p: ?ST = null;

        const FSymbol: u8 = self.FoundState.?[0].Symbol;
        const FFreq: u32 = self.FoundState.?[0].Freq;
        var iFSuccessor: u32 = self.FoundState.?[0].iSuccessor;

        if (MinContext.iSuffix != 0) {
            pc = self.suff(MinContext);
            if (pc.NumStats != 0) {
                var pp = self.getStats(pc);
                if (pp[0].Symbol != FSymbol) {
                    pp += 1;
                    while (pp[0].Symbol != FSymbol) pp += 1;
                    if (pp[0].Freq >= (pp - 1)[0].Freq) {
                        stateSwap(pp, pp - 1);
                        pp -= 1;
                    }
                }
                if (pp[0].Freq < MAX_FREQ - 3) {
                    const cf: u8 = 2 + @as(u8, @intFromBool(FFreq < 28));
                    pp[0].Freq +%= cf;
                    pc.SummFreq +%= cf;
                }
                p = pp;
            } else {
                const pp = oneState(pc);
                pp[0].Freq += @intFromBool(pp[0].Freq < 14);
                p = pp;
            }
        }
        pc = self.MaxContext;

        if (self.OrderFall == 0 and iFSuccessor != 0) {
            self.FoundState.?[0].iSuccessor = self.CreateSuccessors(1, p, MinContext);
            if (self.FoundState.?[0].iSuccessor == 0) {
                self.saved_pc = pc;
                return null;
            }
            self.MaxContext = self.getSucc(self.FoundState.?);
            return self.MaxContext;
        }

        self.pText[0] = FSymbol;
        self.pText += 1;
        var iSuccessor: u32 = self.Ptr2Indx(self.pText);
        if (@intFromPtr(self.pText) >= @intFromPtr(self.UnitsStart)) {
            self.saved_pc = pc;
            return null;
        }

        if (iFSuccessor != 0) {
            if (@intFromPtr(self.Indx2bytePtr(iFSuccessor)) < @intFromPtr(self.UnitsStart))
                iFSuccessor = self.CreateSuccessors(0, p, MinContext)
            else
                prefetchData(self.Indx2bytePtr(iFSuccessor));
        } else {
            iFSuccessor = self.ReduceOrder(p, MinContext);
        }

        if (iFSuccessor == 0) {
            self.saved_pc = pc;
            return null;
        }

        self.OrderFall -= 1;
        if (self.OrderFall == 0) {
            iSuccessor = iFSuccessor;
            self.pText -= @intFromBool(@intFromPtr(self.MaxContext) != @intFromPtr(MinContext));
        }

        const s0: u32 = MinContext.SummFreq -% FFreq;
        const ns: u32 = MinContext.NumStats;
        const Flag: u8 = 0x08 * @as(u8, @intFromBool(FSymbol >= 0x40));

        pc = self.MaxContext;
        while (@intFromPtr(pc) != @intFromPtr(MinContext)) : (pc = self.suff(pc)) {
            const ns1: u32 = pc.NumStats;
            var sp: ST = undefined;
            if (ns1 != 0) {
                if (ns1 & 1 != 0) {
                    const ep = self.ExpandUnits(@ptrCast(self.getStats(pc)), (ns1 + 1) >> 1) orelse {
                        self.saved_pc = pc;
                        return null;
                    };
                    pc.iStats = self.Ptr2Indx(ep);
                }
                pc.SummFreq +%= @intCast(self.QTable[ns + 4] >> 3);
            } else {
                const ap = self.AllocUnits(1) orelse {
                    self.saved_pc = pc;
                    return null;
                };
                sp = @ptrCast(ap);
                sp[0] = oneState(pc)[0];
                pc.iStats = self.Ptr2Indx(sp);
                sp[0].Freq = if (sp[0].Freq <= MAX_FREQ / 3) 2 * sp[0].Freq - 1 else @intCast(MAX_FREQ - 15);
                pc.SummFreq = @as(u16, sp[0].Freq) +% @as(u16, @intFromBool(ns > 1)) +% @as(u16, ExpEscape[self.QTable[@intCast(self.BSumm >> 8)]]);
            }

            var cf: u32 = (FFreq -% 1) *% (5 + pc.SummFreq);
            const sf: u32 = s0 +% pc.SummFreq;

            if (cf <= 3 * sf) {
                cf = 1 + @as(u32, @intFromBool(2 * cf > sf)) + @as(u32, @intFromBool(2 * cf > 3 * sf));
                pc.SummFreq +%= 4;
            } else {
                cf = 5 + @as(u32, @intFromBool(cf > 5 * sf)) + @as(u32, @intFromBool(cf > 6 * sf)) +
                    @as(u32, @intFromBool(cf > 8 * sf)) + @as(u32, @intFromBool(cf > 10 * sf)) + @as(u32, @intFromBool(cf > 12 * sf));
                pc.SummFreq +%= @intCast(cf);
            }

            pc.NumStats +%= 1;
            sp = self.getStats(pc) + pc.NumStats;
            sp[0].iSuccessor = iSuccessor;
            sp[0].Symbol = FSymbol;
            sp[0].Freq = @intCast(cf);
            pc.Flags |= Flag;
        }

        self.MaxContext = self.Indx2ctx(iFSuccessor);
        return self.MaxContext;
    }

    fn CreateSuccessors(self: *PpmdModel, Skip: u32, p_in: ?ST, pc_in: PC) u32 {
        var ps: [MAX_O]ST = undefined;
        var pps: usize = 0;
        var pc = pc_in;
        var p = p_in;

        const sym0: u8 = self.FoundState.?[0].Symbol;
        var sym = sym0;
        const iUpBranch: u32 = self.FoundState.?[0].iSuccessor;

        blk: {
            if (Skip == 0) {
                ps[pps] = self.FoundState.?;
                pps += 1;
                if (pc.iSuffix == 0) break :blk;
            }
            var first_skip = false;
            if (p != null) {
                pc = self.suff(pc);
                first_skip = true;
            }
            while (true) {
                if (!first_skip) {
                    pc = self.suff(pc);
                    if (pc.NumStats != 0) {
                        var pp = self.getStats(pc);
                        while (pp[0].Symbol != sym) pp += 1;
                        const t: u8 = 2 * @as(u8, @intFromBool(pp[0].Freq < MAX_FREQ - 1));
                        pp[0].Freq +%= t;
                        pc.SummFreq +%= t;
                        p = pp;
                    } else {
                        const pp = oneState(pc);
                        const inc: u8 = @intFromBool(self.suff(pc).NumStats == 0 and pp[0].Freq < 16);
                        pp[0].Freq += inc;
                        p = pp;
                    }
                }
                first_skip = false;
                if (p.?[0].iSuccessor != iUpBranch) {
                    pc = self.getSucc(p.?);
                    break;
                }
                ps[pps] = p.?;
                pps += 1;
                if (pc.iSuffix == 0) break;
            }
        }

        if (pps == 0) return self.Ptr2Indx(pc);

        var ct: PPM_CONTEXT = undefined;
        ct.NumStats = 0;
        ct.Flags = 0x10 * @as(u8, @intFromBool(sym >= 0x40));
        sym = self.Indx2bytePtr(iUpBranch)[0];
        oneState(&ct)[0].iSuccessor = self.Ptr2Indx(self.Indx2bytePtr(iUpBranch) + 1);
        oneState(&ct)[0].Symbol = sym;
        ct.Flags |= 0x08 * @as(u8, @intFromBool(sym >= 0x40));

        if (pc.NumStats != 0) {
            var pp = self.getStats(pc);
            while (pp[0].Symbol != sym) pp += 1;
            const cf: u32 = pp[0].Freq - 1;
            const s0: u32 = @as(u32, pc.SummFreq) - pc.NumStats - cf;
            const cf2: u32 = 1 + (if (2 * cf < s0) @intFromBool(12 * cf > s0) else 2 + cf / s0);
            oneState(&ct)[0].Freq = @intCast(@min(@as(u32, 7), cf2));
        } else {
            oneState(&ct)[0].Freq = oneState(pc)[0].Freq;
        }

        while (true) {
            const pc1b = self.AllocContext() orelse return 0;
            const pc1: PC = @ptrCast(pc1b);
            const dst: [*]align(1) u8 = @ptrCast(pc1);
            const src: [*]align(1) u8 = @ptrCast(&ct);
            @memcpy(dst[0..8], src[0..8]);
            pc1.iSuffix = self.Ptr2Indx(pc);
            pc = pc1;
            pps -= 1;
            ps[pps][0].iSuccessor = self.Ptr2Indx(pc);
            if (pps == 0) break;
        }
        return self.Ptr2Indx(pc);
    }

    fn ReduceOrder(self: *PpmdModel, p_in: ?ST, pc_in: PC) u32 {
        var p = p_in;
        var pc = pc_in;
        const pc1 = pc_in;
        self.FoundState.?[0].iSuccessor = self.Ptr2Indx(self.pText);
        const sym: u8 = self.FoundState.?[0].Symbol;
        const iUpBranch: u32 = self.FoundState.?[0].iSuccessor;
        self.OrderFall += 1;

        var first_skip = false;
        if (p != null) {
            pc = self.suff(pc);
            first_skip = true;
        }
        while (true) {
            if (!first_skip) {
                if (pc.iSuffix == 0) return self.Ptr2Indx(pc);
                pc = self.suff(pc);
                if (pc.NumStats != 0) {
                    var pp = self.getStats(pc);
                    while (pp[0].Symbol != sym) pp += 1;
                    const t: u8 = 2 * @as(u8, @intFromBool(pp[0].Freq < MAX_FREQ - 3));
                    pp[0].Freq +%= t;
                    pc.SummFreq +%= t;
                    p = pp;
                } else {
                    const pp = oneState(pc);
                    pp[0].Freq += @intFromBool(pp[0].Freq < 11);
                    p = pp;
                }
            }
            first_skip = false;
            if (p.?[0].iSuccessor != 0) break;
            p.?[0].iSuccessor = iUpBranch;
            self.OrderFall += 1;
        }

        if (p.?[0].iSuccessor <= iUpBranch) {
            const p1 = self.FoundState;
            self.FoundState = p;
            p.?[0].iSuccessor = self.CreateSuccessors(0, null, pc);
            self.FoundState = p1;
        }

        if (self.OrderFall == 1 and @intFromPtr(pc1) == @intFromPtr(self.MaxContext)) {
            self.FoundState.?[0].iSuccessor = p.?[0].iSuccessor;
            self.pText -= 1;
        }

        return p.?[0].iSuccessor;
    }

    // ---- symbol processors (update path, ProcMode=0) ----------------------

    fn processBinSymbol(self: *PpmdModel, q: PC, symbol: i32) void {
        const rs = oneState(q);
        const i: i32 = @as(i32, self.NS2BSIndx[self.suff(q).NumStats]) + self.PrevSuccess +
            @as(i32, q.Flags) + ((self.RunLength >> 26) & 0x20);
        const bs = &self.BinSumm[self.QTable[rs[0].Freq - 1]][@intCast(i)];
        self.BSumm = bs.*;
        bs.* = bs.* -% @as(u16, @intCast((self.BSumm + 64) >> PERIOD_BITS));

        if (rs[0].Symbol != symbol) {
            self.CharMask[rs[0].Symbol] = self.EscCount;
            self.NumMasked = 0;
            self.PrevSuccess = 0;
            self.FoundState = null;
        } else {
            bs.* +%= @intCast(INTERVAL);
            rs[0].Freq += @intFromBool(rs[0].Freq < 196);
            self.RunLength += 1;
            self.PrevSuccess = 1;
            self.FoundState = rs;
        }
    }

    fn processSymbol1(self: *PpmdModel, q: PC, symbol: i32) void {
        var p: ?ST = self.getStats(q);
        const cnum: i32 = q.NumStats;

        if (p.?[0].Symbol == symbol) {
            self.PrevSuccess = 0;
            p.?[0].Freq +%= 4;
            q.SummFreq +%= 4;
        } else {
            self.PrevSuccess = 0;
            var low: i32 = p.?[0].Freq;
            var i: i32 = 1;
            var found = false;
            while (i <= cnum) : (i += 1) {
                const freq: i32 = p.?[@intCast(i)].Freq;
                if (p.?[@intCast(i)].Symbol == symbol) {
                    found = true;
                    break;
                }
                low += freq;
            }
            if (found) {
                const ui: usize = @intCast(i);
                p.?[ui].Freq +%= 4;
                q.SummFreq +%= 4;
                if (p.?[ui].Freq > p.?[ui - 1].Freq) {
                    stateSwap(p.? + ui, p.? + ui - 1);
                    i -= 1;
                }
                p = p.? + @as(usize, @intCast(i));
            } else {
                if (q.iSuffix != 0) prefetchData(self.suff(q));
                self.NumMasked = cnum;
                var k: i32 = 0;
                while (k <= cnum) : (k += 1) self.CharMask[p.?[@intCast(k)].Symbol] = self.EscCount;
                p = null;
            }
        }

        self.FoundState = p;
        if (p) |pp| {
            if (pp[0].Freq > MAX_FREQ) self.FoundState = self.rescale(q, self.OrderFall, pp);
        }
    }

    fn processSymbol2(self: *PpmdModel, q: PC, symbol: i32) void {
        const p: ST = self.getStats(q);
        const cnum: i32 = q.NumStats;

        var psee: *SEE2_CONTEXT = undefined;
        var see_freq: i32 = undefined;
        if (cnum != 0xFF) {
            const row: usize = self.QTable[@intCast(cnum + 3)] - 4;
            var col: usize = @intFromBool(@as(i32, q.SummFreq) > 10 * (cnum + 1));
            col += 2 * @as(usize, @intFromBool(2 * cnum < @as(i32, self.suff(q).NumStats) + self.NumMasked)) + @as(usize, q.Flags);
            psee = &self.SEE2Cont[row][col];
            see_freq = @intCast(psee.getMean() + 1);
        } else {
            psee = &self.DummySEE2Cont;
            see_freq = 1;
        }

        var flag: bool = false;
        var j: i32 = 0;
        var pl: i32 = 0;
        var low: i32 = 0;
        var i: i32 = 0;
        while (i <= cnum) : (i += 1) {
            const c: u8 = p[@intCast(i)].Symbol;
            if (self.CharMask[c] != self.EscCount) {
                self.CharMask[c] = self.EscCount;
                low += p[@intCast(i)].Freq;
                if (c == symbol) {
                    flag = true;
                    j = i;
                    pl = low;
                }
            }
        }

        const Total: i32 = see_freq + low;

        if (flag) {
            low = pl;
            const pp = p + @as(usize, @intCast(j));
            if (see_freq > 2) psee.Summ -%= @intCast(see_freq);
            psee.update();
            self.FoundState = pp;
            pp[0].Freq +%= 4;
            q.SummFreq +%= 4;
            if (pp[0].Freq > MAX_FREQ) self.FoundState = self.rescale(q, self.OrderFall, pp);
            self.RunLength = self.InitRL;
            self.EscCount += 1;
        } else {
            low = Total;
            self.NumMasked = cnum;
            psee.Summ +%= @intCast(Total - see_freq);
        }
    }

    // ---- symbol processors (predict path, build SQ) -----------------------

    fn processBinSymbol_T(self: *PpmdModel, q: PC) void {
        const rs = oneState(q);
        const i: i32 = @as(i32, self.NS2BSIndx[self.suff(q).NumStats]) + self.PrevSuccess +
            @as(i32, q.Flags) + ((self.RunLength >> 26) & 0x20);
        const bs = &self.BinSumm[self.QTable[rs[0].Freq - 1]][@intCast(i)];
        self.BSumm = bs.*;

        self.SQ[self.SQ_ptr].store(rs[0].Symbol, @as(u32, @intCast(self.BSumm)) *% 2, SCALE);
        self.SQ_ptr += 1;
        self.SQ[self.SQ_ptr].store(256, SCALE -% (@as(u32, @intCast(self.BSumm)) *% 2), SCALE);
        self.SQ_ptr += 1;

        self.CharMask[rs[0].Symbol] = self.EscCount;
        self.NumMasked = 0;
    }

    fn processSymbol1_T(self: *PpmdModel, q: PC) void {
        const p: ST = self.getStats(q);
        const cnum: i32 = q.NumStats;
        const total: u32 = q.SummFreq;

        var low: i32 = 0;
        var i: i32 = 0;
        while (i <= cnum) : (i += 1) {
            const freq: u32 = p[@intCast(i)].Freq;
            self.SQ[self.SQ_ptr].store(p[@intCast(i)].Symbol, freq, total);
            self.SQ_ptr += 1;
            low += @intCast(freq);
        }

        if (q.iSuffix != 0) prefetchData(self.suff(q));
        self.NumMasked = cnum;
        i = 0;
        while (i <= cnum) : (i += 1) self.CharMask[p[@intCast(i)].Symbol] = self.EscCount;

        self.SQ[self.SQ_ptr].store(256, total -% @as(u32, @intCast(low)), total);
        self.SQ_ptr += 1;
    }

    fn processSymbol2_T(self: *PpmdModel, q: PC) void {
        const p: ST = self.getStats(q);
        const cnum: i32 = q.NumStats;

        var psee: *SEE2_CONTEXT = undefined;
        var see_freq: i32 = undefined;
        if (cnum != 0xFF) {
            const row: usize = self.QTable[@intCast(cnum + 3)] - 4;
            var col: usize = @intFromBool(@as(i32, q.SummFreq) > 10 * (cnum + 1));
            col += 2 * @as(usize, @intFromBool(2 * cnum < @as(i32, self.suff(q).NumStats) + self.NumMasked)) + @as(usize, q.Flags);
            psee = &self.SEE2Cont[row][col];
            see_freq = @intCast(psee.getMean() + 1);
        } else {
            psee = &self.DummySEE2Cont;
            see_freq = 1;
        }

        var low: i32 = 0;
        var i: i32 = 0;
        while (i <= cnum) : (i += 1) {
            const c: u8 = p[@intCast(i)].Symbol;
            if (self.CharMask[c] != self.EscCount) low += p[@intCast(i)].Freq;
        }
        const Total: i32 = see_freq + low;

        i = 0;
        while (i <= cnum) : (i += 1) {
            const c: u8 = p[@intCast(i)].Symbol;
            if (self.CharMask[c] != self.EscCount) {
                self.SQ[self.SQ_ptr].store(c, p[@intCast(i)].Freq, @intCast(Total));
                self.SQ_ptr += 1;
                self.CharMask[c] = self.EscCount;
            }
        }

        self.SQ[self.SQ_ptr].store(256, @intCast(see_freq), @intCast(Total));
        self.SQ_ptr += 1;
        self.NumMasked = cnum;
    }

    fn ConvertSQ(self: *PpmdModel) void {
        var cum: u32 = 0xFFFFFF00;
        // Only sqp needs zeroing (the tree reads it for absent-symbol floors);
        // trF/trT[1..255] are fully overwritten by the bottom-up build below and
        // index 0 is never read (tree_context is always >= 1).
        var i: u32 = 0;
        while (i < 256) : (i += 1) self.sqp[i] = 0;

        i = 0;
        while (i < self.SQ_ptr) : (i += 1) {
            const c: u32 = self.SQ[i].sym;
            const freq: u32 = self.SQ[i].freq;
            const total: u32 = self.SQ[i].total;
            const prob: u32 = @truncate((@as(u64, cum) * freq) / total);
            if (c < 256) {
                self.sqp[c] = prob + 1;
            } else {
                cum = prob;
            }
        }

        // Bit-tree fill (real cmix-lex ppmd.cpp:1216-1228). The old scatter form
        // ran 256*8 = 2048 read-modify-writes/byte; this bottom-up build is ~255
        // iterations and BYTE-IDENTICAL (u32 add is associative mod 2^32, only the
        // finals trF/trT are read). For a complete binary tree the scatter defines
        // trT[j] = sum of leaf masses under node j and trF[j] = trT[2j] (the
        // left-child total, since b==0 == "went left"), with leaf masses floored
        // to 1 per vocabulary byte.
        // leaf-parent nodes j=128..255: children are bytes (2j-256),(2j-255).
        var j: u32 = 128;
        while (j < 256) : (j += 1) {
            const m0: u32 = if (self.sqp[2 * j - 256] != 0) self.sqp[2 * j - 256] else 1;
            const m1: u32 = if (self.sqp[2 * j - 255] != 0) self.sqp[2 * j - 255] else 1;
            self.trF[j] = m0;
            self.trT[j] = m0 +% m1;
        }
        // internal nodes j=127..1: children are nodes 2j,2j+1 (already written).
        j = 127;
        while (j >= 1) : (j -= 1) {
            self.trF[j] = self.trT[2 * j];
            self.trT[j] = self.trT[2 * j] +% self.trT[2 * j + 1];
        }
    }

    // ---- public entry points ----------------------------------------------

    fn Init(self: *PpmdModel, MaxOrder: u32, MMAX: u32, CutOff: u32, filesize: u32) u32 {
        self._MaxOrder = @intCast(MaxOrder);
        self._CutOff = @intCast(CutOff);
        self._MMAX = @intCast(MMAX);
        self._filesize = filesize;

        self.PPMD_STARTUP();
        if (!self.StartSubAllocator(@intCast(self._MMAX))) return 1;
        self.StartModelRare();
        self.cxt = 0;
        self.y = 1;
        return 0;
    }

    fn ppmd_PrepareByte(self: *PpmdModel) void {
        self.SQ_ptr = 0;
        self.NumMasked = 0;
        const _OrderFall = self.OrderFall;

        var MinContext = self.MaxContext;
        if (MinContext.NumStats != 0) {
            self.processSymbol1_T(MinContext);
        } else {
            self.processBinSymbol_T(MinContext);
        }

        outer: while (true) {
            while (true) {
                if (MinContext.iSuffix == 0) break :outer;
                self.OrderFall += 1;
                MinContext = self.suff(MinContext);
                if (MinContext.NumStats != self.NumMasked) break;
            }
            self.processSymbol2_T(MinContext);
        }

        self.EscCount += 1;
        self.NumMasked = 0;
        self.OrderFall = _OrderFall;
        self.ConvertSQ();
    }

    fn ppmd_UpdateByte(self: *PpmdModel, c: u32) void {
        var MinContext = self.MaxContext;
        if (MinContext.NumStats != 0) {
            self.processSymbol1(MinContext, @intCast(c));
        } else {
            self.processBinSymbol(MinContext, @intCast(c));
        }

        while (self.FoundState == null) {
            while (true) {
                self.OrderFall += 1;
                MinContext = self.suff(MinContext);
                if (MinContext.NumStats != self.NumMasked) break;
            }
            self.processSymbol2(MinContext, @intCast(c));
        }

        var p: ?PC = null;
        if (self.OrderFall != 0 or @intFromPtr(self.getSucc(self.FoundState.?)) < @intFromPtr(self.UnitsStart)) {
            p = self.UpdateModel(MinContext);
            if (p) |pp| self.MaxContext = pp;
        } else {
            p = self.getSucc(self.FoundState.?);
            self.MaxContext = p.?;
        }

        if (p == null) {
            if (self._CutOff != 0) {
                self.RestoreModelRare();
            } else {
                self.StartModelRare();
            }
        }
    }
};

// ---- small numeric helpers ------------------------------------------------

fn clampI32(x: i32, lo: i32, hi: i32) i32 {
    return if (x >= lo) (if (x <= hi) x else hi) else lo;
}
fn clampU32(x: u32, lo: u32, hi: u32) u32 {
    return if (x >= lo) (if (x <= hi) x else hi) else lo;
}

// ---------------------------------------------------------------------------
// PPMD — the ByteModel wrapper exposed via the Model vtable (matches cmix
// `models/ppmd.{h,cpp}` PPMD class). Constructed as PPMD(25, 14000, ...).
// ---------------------------------------------------------------------------

pub const PPMD = struct {
    base: ByteModel,
    byte: *const u32,
    m: *PpmdModel,
    alloc: std.mem.Allocator,

    /// order = 25, memory = 14000 (MB) as used by cmix. `bit_context` must hold
    /// the just-completed byte when `byteUpdate` is invoked.
    pub fn create(
        a: std.mem.Allocator,
        bit_context: *const u32,
        order: u32,
        memory: u32,
        vocab: *const [256]bool,
    ) *PPMD {
        const self = a.create(PPMD) catch unreachable;
        const m = a.create(PpmdModel) catch unreachable;
        m.* = .{ .alloc = a };
        _ = m.Init(order, memory, 1, 0);
        self.* = .{
            .base = ByteModel.init(vocab),
            .byte = bit_context,
            .m = m,
            .alloc = a,
        };
        return self;
    }

    pub fn destroy(self: *PPMD) void {
        self.m.StopSubAllocator();
        self.alloc.destroy(self.m);
        self.alloc.destroy(self);
    }

    pub fn predict(self: *PPMD) []const f32 {
        return self.base.predict();
    }
    pub fn perceive(self: *PPMD, bit: i32) void {
        self.base.perceive(bit);
    }
    pub fn numOutputs(self: *PPMD) usize {
        _ = self;
        return 1;
    }
    pub fn bytePredict(self: *PPMD) *const [256]f32 {
        return self.base.bytePredict();
    }

    pub fn byteUpdate(self: *PPMD) void {
        if (mmap_to_disk) counter_ += 1; // ++counter_ (ppmd.cpp:1444)
        self.m.ppmd_UpdateByte(self.byte.*);
        self.m.ppmd_PrepareByte();
        for (0..256) |i| {
            var v: f32 = @floatFromInt(self.m.sqp[i]);
            if (v < 1) v = 1;
            self.base.probs[i] = v;
        }
        self.base.byteUpdate();
        var sum: f32 = 0;
        for (self.base.probs) |x| sum += x;
        for (&self.base.probs) |*x| x.* /= sum;
        // Deterministic residency-drop cadence (ppmd.cpp:1473-1478).
        // Output-neutral: touches page tables only, never heap bytes.
        if (mmap_to_disk) {
            if (counter_ - last_mmap_remap_counter_ >= mmap_remap_interval_bytes) {
                self.m.dropHeapResidency();
                last_mmap_remap_counter_ = counter_;
            }
        }
    }

    pub fn model(self: *PPMD) Model {
        return .{ .ptr = self, .vtable = vtableFor(PPMD) };
    }
};

// ---------------------------------------------------------------------------
// PPMDLex — the cmix-lex bit-tree ByteModel wrapper (real `models/ppmd.{h,cpp}`
// `PPMD` class, NumOutputs=1). Unlike the v21 `PPMD` wrapper above (which does a
// ByteModel bisection Predict), the lex wrapper reads a per-context binary tree
// (`tree_zero_`/`tree_total_`) that ConvertSQ fills each byte, walked by a
// self-driven `tree_context_` bit index. It shares the heavy `PpmdModel` core.
//
// Construction in predictor.cpp:68 is `PPMD(25, 14000, bit_context, vocab)`
//   -> layer-0 input index 588.
//
// C++ refs: ppmd.h:14-31 (members: tree_zero_/tree_total_[256], tree_context_=1,
// vocab_full_, disabled_bytes_); ppmd.cpp:1407-1479 (ctor / Predict / Perceive /
// ByteUpdate). Cross-ref cmix-rs cmix-ppmd/src/ppmd_lex.rs.
//
// predictor_lex.zig drives it per bit exactly like the C++ Predictor:
//   inputs[588] = ppmdlex.predict;            // BEFORE the bit is coded
//   ... code the bit ...
//   ppmdlex.perceive(bit);                       // for each of the 8 bits
//   // at the byte boundary, with `bit_context` holding the completed byte:
//   ppmdlex.byteUpdate;
// `bit_context` (the `*const u32` passed to create) must equal the just-completed
// byte value when byteUpdateruns (same contract as the fxcm/byte_mixer feed).
// ---------------------------------------------------------------------------

pub const PPMDLex = struct {
    base: ByteModel,
    byte: *const u32,
    m: *PpmdModel,
    alloc: std.mem.Allocator,

    // std::array<unsigned int,256> tree_zero_/tree_total_; tree_context_ = 1.
    tree_zero: [256]u32 = .{0} ** 256,
    tree_total: [256]u32 = .{0} ** 256,
    tree_context: u32 = 1,

    // Model::outputs_ (only outputs_[0] is used for NumOutputs=1).
    outputs: [1]f32 = .{0.5},

    vocab_full: bool = false,

    /// Mirrors `PPMD(order, memory, bit_context, vocab)` — lex uses (25, 14000).
    /// `memory` is in MB (Init passes it as SASize<<20).
    pub fn create(
        a: std.mem.Allocator,
        bit_context: *const u32,
        order: u32,
        memory: u32,
        vocab: *const [256]bool,
    ) *PPMDLex {
        const self = a.create(PPMDLex) catch unreachable;
        const m = a.create(PpmdModel) catch unreachable;
        m.* = .{ .alloc = a };
        _ = m.Init(order, memory, 1, 0);

        // vocab_full_ = every byte in vocab (ppmd.cpp:1411-1417). disabled bytes
        // are recomputed from `vocab` on the fly in byteUpdate (same 0..255 order
        // the C++ push_back produced), so no separate list is stored.
        var vf = true;
        for (0..256) |i| {
            if (!vocab[i]) vf = false;
        }

        self.* = .{
            .base = ByteModel.init(vocab),
            .byte = bit_context,
            .m = m,
            .alloc = a,
            .vocab_full = vf,
        };
        return self;
    }

    pub fn destroy(self: *PPMDLex) void {
        self.m.StopSubAllocator();
        self.alloc.destroy(self.m);
        self.alloc.destroy(self);
    }

    /// ppmd.cpp:1428-1436. When the tree node is empty, fall back to the
    /// ByteModel bisection Predict; otherwise read P(bit==1) = (total-zero)/total.
    /// The subtraction is unsigned (wraps) exactly as in the C++.
    pub fn predict(self: *PPMDLex) f32 {
        const total = self.tree_total[self.tree_context];
        if (total == 0) {
            return self.base.predict()[0];
        }
        self.outputs[0] = @as(f32, @floatFromInt(total -% self.tree_zero[self.tree_context])) /
            @as(f32, @floatFromInt(total));
        return self.outputs[0];
    }

    /// ppmd.cpp:1438-1441. Drive the ByteModel bisection AND the self-driven
    /// tree walk (tree_context_ = (tree_context_<<1) | bit).
    pub fn perceive(self: *PPMDLex, bit: i32) void {
        self.base.perceive(bit);
        self.tree_context = (self.tree_context << 1) | @as(u32, @intCast(bit));
    }

    /// ppmd.cpp:1443-1479, including the mmap/RSS bookkeeping (output-neutral,
    /// active only under -Dppmd-mmap).
    pub fn byteUpdate(self: *PPMDLex) void {
        if (mmap_to_disk) counter_ += 1; // ++counter_ (ppmd.cpp:1444)
        self.m.ppmd_UpdateByte(self.byte.*);
        self.m.ppmd_PrepareByte();

        for (0..256) |i| {
            self.tree_zero[i] = self.m.trF[i];
            self.tree_total[i] = self.m.trT[i];
        }

        // Subtract non-vocab byte mass from the tree so disabled bytes never get
        // probability (saturating subtraction, ppmd.cpp:1451-1465). Same 0..255
        // iteration order as the C++ disabled_bytes_ vector.
        if (!self.vocab_full) {
            for (0..256) |ci| {
                if (self.base.vocab[ci]) continue;
                const c: u32 = @intCast(ci);
                const mass: u32 = if (self.m.sqp[c] != 0) self.m.sqp[c] else 1;
                var bi: u32 = 8;
                while (bi != 0) : (bi -= 1) {
                    const node: u32 = (256 + c) >> @intCast(bi);
                    const bit: u32 = (c >> @intCast(bi - 1)) & 1;
                    if (self.tree_total[node] >= mass)
                        self.tree_total[node] -= mass
                    else
                        self.tree_total[node] = 0;
                    if (bit == 0) {
                        if (self.tree_zero[node] >= mass)
                            self.tree_zero[node] -= mass
                        else
                            self.tree_zero[node] = 0;
                    }
                }
            }
        }

        // probs_ still feeds the ByteModel fallback Predict / bytePredict path.
        for (0..256) |i| {
            var v: f32 = @floatFromInt(self.m.sqp[i]);
            if (v < 1) v = 1;
            self.base.probs[i] = v;
        }
        self.base.byteUpdate();
        var sum: f32 = 0;
        for (self.base.probs) |x| sum += x;
        for (&self.base.probs) |*x| x.* /= sum;

        self.tree_context = 1;

        // Deterministic residency-drop cadence (ppmd.cpp:1473-1478).
        // Output-neutral: touches page tables only, never heap bytes.
        if (mmap_to_disk) {
            if (counter_ - last_mmap_remap_counter_ >= mmap_remap_interval_bytes) {
                self.m.dropHeapResidency();
                last_mmap_remap_counter_ = counter_;
            }
        }
    }

    pub fn numOutputs(self: *PPMDLex) usize {
        _ = self;
        return 1;
    }

    /// 256-way byte distribution (Model::BytePredict), used by the byte-mixer feed.
    pub fn bytePredict(self: *PPMDLex) *const [256]f32 {
        return self.base.bytePredict();
    }
};

// ---------------------------------------------------------------------------
// self-test: build the model, feed a few hundred bytes of text bit-by-bit,
// assert it runs without crashing and produces finite probabilities.
// ---------------------------------------------------------------------------

test "ppmd runs a round of bytes" {
    const a = std.testing.allocator;
    var vocab: [256]bool = .{true} ** 256;
    _ = &vocab;
    var byte_val: u32 = 0;

    // Use a modest heap (64 MB) for the test rather than cmix's 14000 MB.
    const ppmd = PPMD.create(a, &byte_val, 25, 64, &vocab);
    defer ppmd.destroy();

    const text =
        "the quick brown fox jumps over the lazy dog. " ++
        "PPMd is an order-N context model with a custom sub-allocator. " ++
        "the the the the the the the the the the the the the the the. " ++
        "0123456789 0123456789 abcabcabcabc xyzxyzxyz repetition repetition. " ++
        "she sells sea shells by the sea shore; the shells she sells are surely seashells.";

    // Feed the corpus many times: repetition drives symbol frequencies above
    // MAX_FREQ (exercising rescale) and builds deep order-25 suffix chains
    // (exercising escapes, SEE2, CreateSuccessors/ReduceOrder).
    var round: usize = 0;
    while (round < 60) : (round += 1) {
        for (text) |b| {
            var bitpos: i32 = 7;
            while (bitpos >= 0) : (bitpos -= 1) {
                const out = ppmd.predict();
                try std.testing.expect(std.math.isFinite(out[0]));
                try std.testing.expect(out[0] >= 0.0 and out[0] <= 1.0);
                const bit: i32 = @intCast((b >> @intCast(bitpos)) & 1);
                ppmd.perceive(bit);
            }
            byte_val = b;
            ppmd.byteUpdate();

            // every probability must be finite and non-negative after normalization
            var sum: f32 = 0;
            for (ppmd.base.probs) |p| {
                try std.testing.expect(std.math.isFinite(p));
                try std.testing.expect(p >= 0.0);
                sum += p;
            }
            // distribution should sum to ~1 (normalized)
            try std.testing.expect(sum > 0.9 and sum < 1.1);
        }
    }
}

test "ppmd-lex bit-tree wrapper runs a round of bytes" {
    const a = std.testing.allocator;
    // A non-full vocab so the disabled-byte subtraction path is exercised.
    var vocab: [256]bool = .{false} ** 256;
    for (0..128) |i| vocab[i] = true; // ASCII only
    var byte_val: u32 = 0;

    const ppmd = PPMDLex.create(a, &byte_val, 25, 64, &vocab);
    defer ppmd.destroy();

    const text =
        "the quick brown fox jumps over the lazy dog. " ++
        "PPMd bit-tree wrapper: order-N context model, self-driven tree_context. " ++
        "the the the the the the the the the the the the the the the. " ++
        "0123456789 0123456789 abcabcabcabc xyzxyzxyz repetition repetition. " ++
        "she sells sea shells by the sea shore; the shells she sells are seashells.";

    var round: usize = 0;
    while (round < 40) : (round += 1) {
        for (text) |b| {
            var bitpos: i32 = 7;
            while (bitpos >= 0) : (bitpos -= 1) {
                const p = ppmd.predict();
                try std.testing.expect(std.math.isFinite(p));
                try std.testing.expect(p >= 0.0 and p <= 1.0);
                const bit: i32 = @intCast((b >> @intCast(bitpos)) & 1);
                ppmd.perceive(bit);
            }
            byte_val = b;
            ppmd.byteUpdate();
            // after a byte, tree_context is reset to the root
            try std.testing.expectEqual(@as(u32, 1), ppmd.tree_context);
        }
    }
}
