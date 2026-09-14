//! `ContextManagerLex` — Zig port of cmix-lex's `src/context-manager.{h,cpp}`
//! (the real Hutter-prize submission at ref/cmix-lex/src). This is a NEAR-TOTAL
//! rewrite vs the v21 `context_manager.zig`: it adds the WRT byte-streams
//! (b2/b3/b4stream + R/state vars), the mx5..mx19cxt/mxx mixer contexts, and the
//! double-indirect registers ind1/ind2/ind3/ind5 (ind4 is DEAD — omitted).
//!
//! GROUND TRUTH: ref/cmix-lex/src/context-manager.{h,cpp}. The Rust
//! `cmix-rs-speed/.../manager_lex.rs` is a secondary cross-reference only.
//!
//! Key facts / C++ line refs (context-manager.cpp):
//!  - ctor :59-66 — history_(60,000,000), shared_map_(256*400,000), words_(8),
//!    recent_bytes_(8); hashes_ind1/ind2=0x1000000, ind3=0x2000000,
//!    ind4=0x100 (DEAD), ind5=0x100.
//!  - UpdateWords :94-170 — the WRT stream machinery + double-indirect regs.
//!  - UpdateContexts :191-242 — per-bit; registered contexts update at byte
//!    boundary only (after UpdateWords/RecentBytes/WRTContext).
//!  - `mx19cxt = wrtcxt` :227 reads fxcmv1's live global. The caller passes the
//!    fxcm's CURRENT value; in the predictor's Perceive path `updateContexts` is
//!    called BEFORE `fxcm.perceive`, so mx19cxt sees the PREVIOUS bit's wrtcxt.
//!
//! Integer widths mirror the C++ exactly: `bit_context_` / `wrt_state_` / `bpos`
//! / `line_class_` / `line_prefix_hash_` are `unsigned int` (u32); everything
//! else is `unsigned long long` (u64). The 32-bit-literal masks (0xfffffff8 etc.)
//! intentionally truncate as in C++ (the literal promotes to u64 = 0x00000000_fffffff8).
//! C++ unsigned overflow that wraps -> Zig `*%`/`+%`; signed `>>` is arithmetic
//! (none here — all shifts are on unsigned values).

const std = @import("std");
const cm = @import("context_manager.zig");
const zero_alloc = @import("zero_alloc.zig");
// -Dextshed (own options module, the own-module rule — see predictor_lex.zig): the ONLY
// consumers of the double-indirect registers ind1/ind2/ind3/ind5 are the four
// shed double-indirect models (predictor_lex construction, lines "AddDoubleIndirect"),
// so under the shed their per-byte derivation — 4 read+write pairs across the
// 208 MiB hashes_ind1/2/3/5 arrays — is skipped too. The arrays stay allocated
// (zero_alloc is lazily committed, untouched pages never enter RSS) and the
// registers freeze at 0, which no retained consumer reads. Every GATING
// context (mx*, streams, line_break, words, recent_bytes) is untouched.
const extshed_options = @import("extshed_options");
const EXTSHED: bool = extshed_options.extshed;
const parse_byte = @import("cmfast").parse_byte;

/// fxcm's WRT_2B / WRT_3B (context-manager.cpp declares them `extern`; they live
/// in models/fxcmv1.cpp and are `pub` in parse_byte.zig). Used for
/// b2/b3stream and the mxx bpos>3 branch.
const WRT_2B = &parse_byte.WRT_2B;
const WRT_3B = &parse_byte.WRT_3B;

/// context-manager.cpp:7-26 — LOCAL wrt_4b (differs from fxcm's; verified against
/// the C++ table byte-for-byte). Feeds b4stream / mx7.
const WRT_4B = [256]u8{
    6, 0, 12, 15, 12, 15, 14, 14, 5, 3, 14, 0, 15, 13, 8, 13,
    0, 0, 0,  0,  0,  0,  0,  0,  0, 0, 0,  0, 0,  0,  0, 0,
    13, 5, 15, 11, 10, 12, 6, 12, 0, 11, 14, 1, 1, 10, 9, 8,
    7,  7, 7,  7,  7,  7,  7, 7,  7, 7,  9,  11, 6, 1,  0, 4,
    9,  10, 10, 4, 5, 1, 4, 2, 11, 8, 4, 1, 0, 10, 10, 5,
    4,  7,  15, 4, 5, 13, 0, 1, 4, 12, 0, 1, 3, 3,  3,  11,
    2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
    2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 3, 8, 0, 11, 7,
    2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
    2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
    2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
    2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
    2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
    2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
    2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
    2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
};

/// context-manager.cpp:27-44 — LOCAL wrt_5b. Feeds mx18cxt / mx18.
const WRT_5B = [256]u8{
    10, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9,
    9,  9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9,
    8,  8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8,
    8,  8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8,
    7,  7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7,
    7,  7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7,
    7,  6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6,
    6,  6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6,
    5,  5, 5, 5, 5, 5, 5, 5, 5, 4, 3, 3, 2, 2, 1, 1,
    1,  1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
    1,  1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
    1,  1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
    1,  1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
    1,  1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
    1,  1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
    1,  1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0,
};

// WRT constants (context-manager.cpp:45-57 — bytes are in WRT space).
const CURLYOPENING: u32 = 'P';
const VERTICALBAR: u32 = 'Q';
const CURLYCLOSE: u32 = 'R';
const SQUARECLOSE: u32 = 93; // ]
const EQUALS: u32 = 'M';

pub const HISTORY_SIZE: usize = 60_000_000;
// -Dshared-map-div=D shrinks the shared bit-history map by D (default 1 = stock,
// byte-identical). This is the LOAD-FACTOR knob: the map's aliasing regime is
// λ = (Σ distinct_ctx × nodes_touched)/|map|, which at e9 is ~100x what any
// testable tier reaches. Shrinking the map at a small tier reproduces the e9
// pressure regime directly, instead of waiting for 1e9 bytes of stream.
pub const SHARED_MAP_DIV: usize = @import("build_options").shared_map_div;
pub const SHARED_MAP_SIZE: usize = (256 * 400_000) / SHARED_MAP_DIV; // stock 102,400,000
const HASHES_IND1_SIZE: usize = 0x100_0000;
const HASHES_IND2_SIZE: usize = 0x100_0000;
const HASHES_IND3_SIZE: usize = 0x200_0000;
const HASHES_IND5_SIZE: usize = 0x100;

/// BracketContext for cmix-lex (bracket-context.cpp). NOTE: this uses the
/// WRT-space bracket pairs `('(',')') ('P','R') ('[',']') ('L','N')`, which
/// differ from the ASCII BracketContext in context_manager.zig — hence a local
/// type. Registered via `addBracketContext` and consumed by the direct/indirect
/// models through `value_ptr` + `size`.
pub const LexBracketContext = struct {
    byte: *const u32, // manager.bit_context
    distance_limit: u32,
    stack_limit: u32,
    active: std.ArrayList(u32) = .empty,
    distance: std.ArrayList(u32) = .empty,
    value: u64 = 0,
    size: u64,
    alloc: std.mem.Allocator,

    fn isOpen(c: u32) bool {
        return c == '(' or c == 'P' or c == '[' or c == 'L';
    }
    fn closeOf(c: u32) u32 {
        return switch (c) {
            '(' => ')',
            'P' => 'R',
            '[' => ']',
            'L' => 'N',
            else => 0,
        };
    }

    pub fn create(a: std.mem.Allocator, byte: *const u32, distance_limit: u32, stack_limit: u32) *LexBracketContext {
        const self = a.create(LexBracketContext) catch unreachable;
        self.* = .{
            .byte = byte,
            .distance_limit = distance_limit,
            .stack_limit = stack_limit,
            .size = 257 * @as(u64, distance_limit),
            .alloc = a,
        };
        return self;
    }

    pub fn update(self: *LexBracketContext) void {
        const b = self.byte.*;
        if (self.active.items.len != 0) {
            const top = self.active.items[self.active.items.len - 1];
            if (closeOf(top) == b or self.distance.items[self.distance.items.len - 1] >= self.distance_limit - 1) {
                _ = self.active.pop();
                _ = self.distance.pop();
            } else {
                self.distance.items[self.distance.items.len - 1] += 1;
            }
        }
        if (isOpen(b)) {
            self.active.append(self.alloc, b) catch unreachable;
            self.distance.append(self.alloc, 0) catch unreachable;
            // C++ compares brackets_.size(==4) to stack_limit_ (==15); never
            // trims with the configured limit, so we mirror that (no trim).
            if (4 > self.stack_limit) {
                _ = self.active.orderedRemove(0);
                _ = self.distance.orderedRemove(0);
            }
        }
        if (self.active.items.len != 0) {
            self.value = @as(u64, self.distance_limit) * (@as(u64, self.active.items[self.active.items.len - 1]) + 1) +
                @as(u64, self.distance.items[self.distance.items.len - 1]);
        } else {
            self.value = 0;
        }
    }

    pub fn value_ptr(self: *LexBracketContext) *const u64 {
        return &self.value;
    }
    pub fn deinit(self: *LexBracketContext) void {
        self.active.deinit(self.alloc);
        self.distance.deinit(self.alloc);
    }
};

pub const ContextManagerLex = struct {
    // --- scalar per-bit state (context-manager.h:63-89) ---
    // `unsigned int` in C++:
    bit_context: u32 = 1,
    bpos: u32 = 0,
    // `unsigned long long` in C++:
    long_bit_context: u64 = 1,
    zero_context: u64 = 0,
    history_pos: usize = 0,
    line_break: u64 = 0,
    longest_match: u64 = 0,
    auxiliary_context: u64 = 0,
    b2stream: u64 = 0,
    b2streamcxt: u64 = 0,
    o2b_state: u64 = 0,
    n2b_state: u64 = 0,
    stream2b_r: u64 = 0,
    b3stream: u64 = 0,
    b3streamcxt: u64 = 0,
    o3b_state: u64 = 0,
    n3b_state: u64 = 0,
    stream3b_r: u64 = 0,
    b4stream: u64 = 0,
    mx5: u64 = 0,
    mx6: u64 = 0,
    mx7: u64 = 0,
    mx8: u64 = 0,
    mx9: u64 = 0,
    mx9cxt: u64 = 0,
    mx10: u64 = 0,
    mx10cxt: u64 = 0,
    mx11: u64 = 0,
    mx11cxt: u64 = 0,
    mx12: u64 = 0,
    mx12cxt: u64 = 0,
    mx13: u64 = 0,
    mx13cxt: u64 = 0,
    mx14: u64 = 0,
    mx15: u64 = 0,
    mx16: u64 = 0,
    mx17: u64 = 0,
    mx18: u64 = 0,
    mx18cxt: u64 = 0,
    mxx: u64 = 0,
    words: u64 = 0,
    wordscxt: u64 = 0,
    ind1: u64 = 0,
    context1_ind: u64 = 0,
    ind2: u64 = 0,
    context1_ind2: u64 = 0,
    ind3: u64 = 0,
    context1_ind3: u64 = 0,
    ind5: u64 = 0,
    context1_ind5: u64 = 0,
    mx19cxt: u64 = 0,

    // --- buffers ---
    history: []u8,
    shared_map: []u8,
    words_arr: [8]u64 = .{0} ** 8, // C++ words_
    recent_bytes: [8]u64 = .{0} ** 8, // C++ recent_bytes_
    // C++ declares these `vector<unsigned long long>`, but every store below is
    // masked first — ind1 to 8 bits, ind2 to 32, ind3 to 25, ind5 to 30 — so
    // narrow elements hold the exact same values at a quarter/eighth the RAM
    // (512 MB -> 208 MB of dense-touched tables). Loads widen back to u64.
    hashes_ind1: []u8,
    hashes_ind2: []u32,
    hashes_ind3: []u32,
    hashes_ind5: []u32,

    // --- registered contexts (dedup like Add*Context / IsEqual) ---
    context_hashes: std.ArrayList(*cm.ContextHash) = .empty,
    sparses: std.ArrayList(*cm.Sparse) = .empty,
    bracket: ?*LexBracketContext = null,

    alloc: std.mem.Allocator,

    pub fn init(a: std.mem.Allocator) !*ContextManagerLex {
        const self = try a.create(ContextManagerLex);
        // zero_alloc: zero-filled like the C++ vectors, but committed lazily —
        // these buffers total ~670 MB and fill with the stream, not at t=0.
        self.* = .{
            .history = try zero_alloc.alloc(a, u8, HISTORY_SIZE),
            .shared_map = try zero_alloc.alloc(a, u8, SHARED_MAP_SIZE * (if (@import("build_options").ind_tag != 0) @as(usize, 2) else 1)),
            .hashes_ind1 = try zero_alloc.alloc(a, u8, HASHES_IND1_SIZE),
            .hashes_ind2 = try zero_alloc.alloc(a, u32, HASHES_IND2_SIZE),
            .hashes_ind3 = try zero_alloc.alloc(a, u32, HASHES_IND3_SIZE),
            .hashes_ind5 = try zero_alloc.alloc(a, u32, HASHES_IND5_SIZE),
            .alloc = a,
        };
        return self;
    }

    pub fn deinit(self: *ContextManagerLex) void {
        for (self.context_hashes.items) |c| self.alloc.destroy(c);
        for (self.sparses.items) |s| {
            s.deinit();
            self.alloc.destroy(s);
        }
        self.context_hashes.deinit(self.alloc);
        self.sparses.deinit(self.alloc);
        if (self.bracket) |b| {
            b.deinit();
            self.alloc.destroy(b);
        }
        zero_alloc.free(self.alloc, self.history);
        zero_alloc.free(self.alloc, self.shared_map);
        zero_alloc.free(self.alloc, self.hashes_ind1);
        zero_alloc.free(self.alloc, self.hashes_ind2);
        zero_alloc.free(self.alloc, self.hashes_ind3);
        zero_alloc.free(self.alloc, self.hashes_ind5);
        self.alloc.destroy(self);
    }

    // ---- registration (AddBracketContext / AddContextHashContext /
    //      AddSparseContext, with the IsEqual dedup). Return stable pointers so
    //      the models can read `value_ptr.*` live and use `.size`. ----

    /// `AddBracketContext(bit_context_, distance_limit, stack_limit)`
    /// (context-manager.h:26-33). Only one distinct bracket context is used
    /// (256,15), so a simple single-slot dedup matches the C++ IsEqual.
    pub fn addBracketContext(self: *ContextManagerLex, distance_limit: u32, stack_limit: u32) *LexBracketContext {
        if (self.bracket) |b| {
            if (b.distance_limit == distance_limit and b.stack_limit == stack_limit) return b;
        }
        const b = LexBracketContext.create(self.alloc, &self.bit_context, distance_limit, stack_limit);
        self.bracket = b;
        return b;
    }

    /// `AddContextHashContext(bit_context_, order, hash_size)`
    /// (context-manager.h:35-43). IsEqual = same size_ (== 1<<(hash_size*order))
    /// AND same hash_size_ (context-hash.cpp:13-18).
    pub fn addContextHash(self: *ContextManagerLex, order: u32, hash_size: u32) *cm.ContextHash {
        const want_size: u64 = @as(u64, 1) << @intCast(hash_size * order);
        for (self.context_hashes.items) |c| {
            if (c.size == want_size and c.hash_size == @as(u6, @intCast(hash_size))) return c;
        }
        const c = cm.ContextHash.create(self.alloc, &self.bit_context, order, hash_size);
        self.context_hashes.append(self.alloc, c) catch unreachable;
        return c;
    }

    /// `AddSparseContext(words_, orders)` (context-manager.h:46-54). IsEqual =
    /// same recent-vector (always `words_`) AND same orders (sparse.cpp:24-33).
    pub fn addSparse(self: *ContextManagerLex, orders: []const u32) *cm.Sparse {
        for (self.sparses.items) |s| {
            if (std.mem.eql(u32, s.orders, orders)) return s;
        }
        const s = cm.Sparse.create(self.alloc, &self.words_arr, orders);
        self.sparses.append(self.alloc, s) catch unreachable;
        return s;
    }

    // ---- per-byte helpers ----


    fn updateHistory(self: *ContextManagerLex) void {
        self.history[self.history_pos] = @intCast(self.bit_context);
        self.history_pos += 1;
        if (self.history_pos == self.history.len) self.history_pos = 0;
    }

    /// `UpdateWords` — the WRT stream machinery + double-indirect regs
    /// (context-manager.cpp:94-170).
    fn updateWords(self: *ContextManagerLex) void {
        const cu: u32 = self.bit_context;
        const c: u64 = self.bit_context; // used as an addend below

        if (cu == CURLYCLOSE or cu == CURLYOPENING or cu == SQUARECLOSE) {
            self.b3stream = (self.b3stream & 0xfffffff8) + 3;
        } else if (cu == EQUALS) {
            self.b3stream = (self.b3stream & 0xfffffff8) + 4;
        }
        self.n2b_state = WRT_2B[cu];
        self.b2stream = self.b2stream *% 4 +% self.n2b_state;
        self.n3b_state = WRT_3B[cu];
        self.b3stream = self.b3stream *% 8 +% self.n3b_state;
        if (self.o3b_state != self.n3b_state) {
            self.stream3b_r = (self.stream3b_r << 3) +% self.n3b_state;
            self.o3b_state = self.n3b_state;
        }
        if (cu == 10 or cu == ')') self.b3stream = self.b3stream << 6;
        if (cu == VERTICALBAR) self.b3stream = self.b3stream *% 8 +% @as(u64, WRT_3B[cu]);
        self.b2streamcxt = self.b2stream & 0x3ff; // 2^10 bits
        self.b3streamcxt = self.b3stream & 0x1ff; // 2^9 bits

        if (self.o2b_state != self.n2b_state) {
            self.stream2b_r = (self.stream2b_r << 2) +% self.n2b_state;
            self.o2b_state = self.n2b_state;
        }
        self.b4stream = self.b4stream *% 16 +% @as(u64, WRT_4B[cu]);
        self.mx18cxt = self.mx18cxt *% 16 +% @as(u64, WRT_5B[cu]);
        self.mx18 = self.mx18cxt & 0xff;

        self.words = self.words *% 2;

        if ((cu >= 'a' and cu <= 'z') or cu >= 0x80) {
            self.words_arr[7] = self.words_arr[7] *% (997 * 16) +% c;
            if (self.recent_bytes[0] != 12) self.words = self.words +% 1;
        } else {
            self.words_arr[7] = 0;
        }
        if ((cu >= 'a' and cu <= 'z') or (cu >= '0' and cu <= '9') or cu == 8 or cu == 6 or cu >= 0x80) {
            self.words_arr[0] = self.words_arr[0] *% (997 * 16) +% c;
            self.words_arr[0] &= 0xfffffff;
            self.words_arr[1] = self.words_arr[1] *% (263 * 32) +% c;
        } else {
            var i: usize = 6;
            while (i >= 2) : (i -= 1) self.words_arr[i] = self.words_arr[i - 1];
            self.words_arr[1] = 0;
        }
        if (cu == 10) {
            self.words = 0xfffc;
        } else if (cu == '.') {
            self.words |= 0xffc;
        } else if (cu == ',') {
            self.words |= 0xffc;
        }
        self.mx5 = self.b2stream & 0xffff; // 2^16 bits
        self.mx7 = self.b4stream & 0xff;
        self.mx8 = (self.mx8 *% 4 +% (self.b3stream & 0x3f)) & 0x3FFF;
        self.mx9cxt = (self.mx9cxt *% 16 +% c) & 0xff;

        self.mx10cxt = c;
        self.mx11cxt = c;
        self.mx12cxt = 0;
        self.mx13cxt = 0;

        self.mx14 = c *% 256 +% self.recent_bytes[0];
        self.mx15 = self.recent_bytes[0] *% 256 +% self.recent_bytes[1];

        // Double-indirect registers (ind4 is DEAD — omitted).
        // -Dextshed: skipped — ind1/2/3/5 feed ONLY the shed double-indirect
        // models; the 208 MiB hashes_ind* arrays are then never touched.
        if (comptime !EXTSHED) {
            self.hashes_ind1[@intCast(self.context1_ind)] = @intCast((self.ind1 *% 256 +% c) & (0x100 - 1));
            self.context1_ind = (self.context1_ind *% 256 +% c) & (0x100_0000 - 1);
            self.ind1 = self.hashes_ind1[@intCast(self.context1_ind)];

            self.hashes_ind2[@intCast(self.context1_ind2)] = @intCast((self.ind2 *% 256 +% c) & 0xffff_ffff);
            self.context1_ind2 = (self.context1_ind2 *% 64 +% c) & (0x100_0000 - 1);
            self.ind2 = self.hashes_ind2[@intCast(self.context1_ind2)];

            self.hashes_ind3[@intCast(self.context1_ind3)] = @intCast((self.ind3 *% 32 +% c) & (0x200_0000 - 1));
            self.context1_ind3 = (self.context1_ind3 *% 32 +% c) & (0x200_0000 - 1);
            self.ind3 = self.hashes_ind3[@intCast(self.context1_ind3)];

            self.hashes_ind5[@intCast(self.context1_ind5)] = @intCast((self.ind5 *% 64 +% c) & (0x4000_0000 - 1));
            self.context1_ind5 = c;
            self.ind5 = self.hashes_ind5[@intCast(self.context1_ind5)];
        }
    }

    fn updateRecentBytes(self: *ContextManagerLex) void {
        var i: usize = 7;
        while (i >= 1) : (i -= 1) self.recent_bytes[i] = self.recent_bytes[i - 1];
        self.recent_bytes[0] = self.bit_context;
    }

    /// `UpdateContexts(bit)` (context-manager.cpp:191-242). `wrtcxt` is fxcmv1's
    /// live `::wrtcxt` global at this point (see module docs for the timing).
    pub fn updateContexts(self: *ContextManagerLex, bit: u1, wrtcxt: u64) void {
        self.bit_context += self.bit_context + @as(u32, bit);
        self.long_bit_context = self.bit_context;
        if (self.bit_context >= 256) {
            self.bit_context -= 256;
            self.long_bit_context = 1;
            self.longest_match = 0;

            if (self.bit_context == '\n') {
                self.line_break = 0;
            } else if (self.line_break < 99) {
                self.line_break += 1;
            }
            self.updateHistory();
            self.updateWords();
            self.updateRecentBytes();
            // updateWRTContext removed: wrt_context/wrt_state are read ONLY inside
            // that function (self-referential) — no lex-path consumer (the v21
            // context_manager.zig twin IS live via predictor.zig, but this is the
            // lex predictor). Dead per-byte work.

            for (self.context_hashes.items) |c| c.update();
            for (self.sparses.items) |s| s.update();
            if (self.bracket) |b| b.update();
        }
        self.wordscxt = (self.words & 0x7F) * 256 + self.long_bit_context;

        self.bpos = (self.bpos + 1) & 7;

        self.mx6 = (self.stream2b_r & 0xff) * 256 + self.long_bit_context;

        self.mx19cxt = wrtcxt;
        self.mx9 = self.mx9cxt * 256 + self.long_bit_context;
        self.mx10 = self.mx10cxt * 256 + self.long_bit_context;
        self.mx11 = self.mx11cxt * 256 + self.long_bit_context;
        self.mx12 = self.long_bit_context;
        self.mx13 = self.long_bit_context;
        self.mx16 = self.recent_bytes[1] * 256 + self.long_bit_context;
        self.mx17 = (self.b3stream & 0x3f) * 256 + self.long_bit_context;

        if (self.bpos == 0) {
            self.mxx = (self.stream2b_r & 63) * 8 + (self.b3stream & 7);
        } else if (self.bpos > 3) {
            const idx: usize = @intCast((self.long_bit_context << @intCast(8 - self.bpos)) & 255);
            self.mxx = ((self.b2stream << 2) & 63) + @as(u64, WRT_2B[idx]) * 8 + (self.b3stream & 7);
        } else {
            self.mxx = (self.stream2b_r & 63) * 8 + (self.b3stream & 7);
        }
    }
};

// ===========================================================================
// Verification test: drive the manager over the cmix-lex oracle goldens
// and assert every per-byte field matches mgr_fields_nodict.txt (feeding wrtcxt =
// the dumped mx19cxt for each byte, since that IS wrtcxt and its only consumer).
// ===========================================================================

// CWD-relative. The oracle goldens are derived verification data, not a build
// input, and are not distributed with the source package; regenerate them from
// the cmix-lex reference before running these tests.
const GOLDEN_DIR = "reference/lex-goldens";

fn fieldU64(line: []const u8, key: []const u8) !u64 {
    // find " key=" or "key=" at line start
    var it = std.mem.tokenizeScalar(u8, line, ' ');
    while (it.next()) |tok| {
        const eq = std.mem.indexOfScalar(u8, tok, '=') orelse continue;
        if (std.mem.eql(u8, tok[0..eq], key)) {
            return std.fmt.parseInt(u64, tok[eq + 1 ..], 10);
        }
    }
    return error.KeyNotFound;
}

test "context_manager_lex matches cmix-lex per-byte oracle" {
    const a = std.heap.page_allocator;

    const stream = try std.fs.cwd().readFileAlloc(a, GOLDEN_DIR ++ "/stream_64k.bin", 1 << 20);
    defer a.free(stream);
    try std.testing.expectEqual(@as(usize, 65536), stream.len);

    const dump = try std.fs.cwd().readFileAlloc(a, GOLDEN_DIR ++ "/mgr_fields_nodict.txt", 64 << 20);
    defer a.free(dump);

    const mgr = try ContextManagerLex.init(a);
    defer mgr.deinit();

    // Register the same contexts the predictor does — proves the registration
    // API works and that context updates don't perturb the scalar fields.
    _ = mgr.addBracketContext(256, 15);
    const word_orders = [_][]const u32{
        &.{0}, &.{ 0, 1 }, &.{1}, &.{ 1, 2 }, &.{ 1, 3 }, &.{ 2, 3 },
        &.{ 3, 4 }, &.{ 1, 2, 4 }, &.{ 2, 3, 4 }, &.{2},
        &.{0}, &.{1}, &.{ 1, 3 }, &.{ 1, 2, 3 }, &.{ 7, 2 },
    };
    for (word_orders) |o| _ = mgr.addSparse(o);
    const ch_params = [_][2]u32{ .{ 0, 8 }, .{ 1, 8 }, .{ 7, 4 }, .{ 11, 3 }, .{ 13, 2 } };
    for (ch_params) |p| _ = mgr.addContextHash(p[0], p[1]);

    var lines = std.mem.tokenizeScalar(u8, dump, '\n');
    var mismatches: usize = 0;
    var first_mismatch: ?[]const u8 = null;
    _ = &first_mismatch;

    var byte_idx: usize = 0;
    while (byte_idx < stream.len) : (byte_idx += 1) {
        const line = lines.next() orelse return error.DumpTooShort;

        // wrtcxt for this byte == the dumped mx19cxt (constant across the 8 bits,
        // and mx19cxt is its only consumer, so feeding it is exact + isolated).
        const wrtcxt = try fieldU64(line, "mx19cxt");

        const c = stream[byte_idx];
        var j: i32 = 7;
        while (j >= 0) : (j -= 1) {
            const bit: u1 = @intCast((c >> @intCast(j)) & 1);
            mgr.updateContexts(bit, wrtcxt);
        }

        // --- check every gated field for this completed byte ---
        const Check = struct { key: []const u8, got: u64 };
        const checks = [_]Check{
            .{ .key = "bc", .got = mgr.bit_context },
            .{ .key = "w0", .got = mgr.words_arr[0] },
            .{ .key = "w1", .got = mgr.words_arr[1] },
            .{ .key = "w2", .got = mgr.words_arr[2] },
            .{ .key = "w3", .got = mgr.words_arr[3] },
            .{ .key = "w4", .got = mgr.words_arr[4] },
            .{ .key = "w5", .got = mgr.words_arr[5] },
            .{ .key = "w6", .got = mgr.words_arr[6] },
            .{ .key = "w7", .got = mgr.words_arr[7] },
            .{ .key = "r0", .got = mgr.recent_bytes[0] },
            .{ .key = "r1", .got = mgr.recent_bytes[1] },
            .{ .key = "r2", .got = mgr.recent_bytes[2] },
            .{ .key = "r3", .got = mgr.recent_bytes[3] },
            .{ .key = "r4", .got = mgr.recent_bytes[4] },
            .{ .key = "r5", .got = mgr.recent_bytes[5] },
            .{ .key = "r6", .got = mgr.recent_bytes[6] },
            .{ .key = "r7", .got = mgr.recent_bytes[7] },
            .{ .key = "line_break", .got = mgr.line_break },
            // NOTE: `longest_match` is intentionally NOT gated here. The manager
            // only zeros it at the byte boundary; the match models' ByteUpdate
            // (predictor.cpp:312-314, run AFTER UpdateContexts) write the real
            // match length into `manager_.longest_match_`. So it is not a pure
            // context-manager field and cannot be reproduced in isolation — it is
            // verified later via the inputs/end-to-end gates with the match models.
            .{ .key = "mx5", .got = mgr.mx5 },
            .{ .key = "mx6", .got = mgr.mx6 },
            .{ .key = "mx7", .got = mgr.mx7 },
            .{ .key = "mx8", .got = mgr.mx8 },
            .{ .key = "mx9", .got = mgr.mx9 },
            .{ .key = "mx10", .got = mgr.mx10 },
            .{ .key = "mx11", .got = mgr.mx11 },
            .{ .key = "mx12", .got = mgr.mx12 },
            .{ .key = "mx13", .got = mgr.mx13 },
            .{ .key = "mx14", .got = mgr.mx14 },
            .{ .key = "mx15", .got = mgr.mx15 },
            .{ .key = "mx16", .got = mgr.mx16 },
            .{ .key = "mx17", .got = mgr.mx17 },
            .{ .key = "mx18", .got = mgr.mx18 },
            .{ .key = "mxx", .got = mgr.mxx },
            .{ .key = "wordscxt", .got = mgr.wordscxt },
            .{ .key = "b2streamcxt", .got = mgr.b2streamcxt },
            .{ .key = "b3streamcxt", .got = mgr.b3streamcxt },
            .{ .key = "ind1", .got = mgr.ind1 },
            .{ .key = "ind2", .got = mgr.ind2 },
            .{ .key = "ind3", .got = mgr.ind3 },
            .{ .key = "ind5", .got = mgr.ind5 },
        };
        for (checks) |chk| {
            const want = try fieldU64(line, chk.key);
            if (want != chk.got) {
                mismatches += 1;
                if (mismatches <= 20) {
                    std.debug.print("byte {d}: {s} want={d} got={d}\n", .{ byte_idx, chk.key, want, chk.got });
                }
            }
        }

        // Predictor resets bit_context_ to 1 at the byte boundary (predictor.cpp:336).
        mgr.bit_context = 1;
    }

    if (mismatches != 0) {
        std.debug.print("TOTAL MISMATCHES: {d}\n", .{mismatches});
    }
    try std.testing.expectEqual(@as(usize, 0), mismatches);
}
