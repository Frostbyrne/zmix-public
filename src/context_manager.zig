//! Context system + manager, ported from cmix `contexts/` and
//! `context-manager.{h,cpp}`.
//!
//! The manager owns the shared byte/word/history state that models and contexts
//! reference by pointer, and drives per-bit / per-byte context updates. Concrete
//! `Context` objects (ContextHash, BitContext, ...) each hold a live `context`
//! value that models read through a stable pointer.
//!
//! NOTE: cmix sizes `history_` at 100 MB and `shared_map_` at 2 GB (it targets
//! 32 GB RAM). This port uses smaller defaults so it runs on a normal machine;
//! encoder and decoder use identical sizes, so round-trips stay exact. Matching
//! cmix's exact sizes is a memory-tuning item in ROADMAP.md.
const std = @import("std");
const states = @import("states.zig");

pub const HISTORY_SIZE: usize = 4 * 1024 * 1024;
pub const SHARED_MAP_SIZE: usize = 32 * 1024 * 1024;

/// A live context: an updatable value plus its range, referenced by models.
pub const Context = struct {
    ptr: *anyopaque,
    update_fn: *const fn (*anyopaque) void,
    value: *u64,
    size: u64,

    pub fn update(self: Context) void {
        self.update_fn(self.ptr);
    }
};

pub const ContextHash = struct {
    byte: *const u32, // manager.bit_context_
    hash_size: u6,
    value: u64 = 0,
    size: u64,

    pub fn create(a: std.mem.Allocator, byte: *const u32, order: u32, hash_size: u32) *ContextHash {
        const self = a.create(ContextHash) catch unreachable;
        self.* = .{
            .byte = byte,
            .hash_size = @intCast(hash_size),
            .size = @as(u64, 1) << @intCast(hash_size * order),
        };
        return self;
    }

    pub fn update(self: *ContextHash) void {
        self.value = (self.value *% (@as(u64, 1) << self.hash_size) +% self.byte.*) % self.size;
    }

    pub fn value_ptr(self: *ContextHash) *u64 {
        return &self.value;
    }
    pub fn context(self: *ContextHash) Context {
        return .{ .ptr = self, .update_fn = updateErased, .value = &self.value, .size = self.size };
    }
    fn updateErased(p: *anyopaque) void {
        update(@ptrCast(@alignCast(p)));
    }
};

pub const BitContext = struct {
    bit_context: *const u64, // manager.long_bit_context_
    byte_context: *const u64,
    value: u64 = 0,
    size: u64,

    pub fn create(a: std.mem.Allocator, bit_context: *const u64, byte_context: *const u64, byte_context_size: u64) *BitContext {
        const self = a.create(BitContext) catch unreachable;
        self.* = .{ .bit_context = bit_context, .byte_context = byte_context, .size = 256 * byte_context_size };
        return self;
    }

    pub fn update(self: *BitContext) void {
        self.value = (self.byte_context.* << 8) + self.bit_context.*;
    }

    pub fn value_ptr(self: *BitContext) *u64 {
        return &self.value;
    }
    pub fn context(self: *BitContext) Context {
        return .{ .ptr = self, .update_fn = updateErased, .value = &self.value, .size = self.size };
    }
    fn updateErased(p: *anyopaque) void {
        update(@ptrCast(@alignCast(p)));
    }
};

/// Sparse: sum of `words_` entries selected by `orders`, weighted by factors.
/// (cmix contexts/sparse)
pub const Sparse = struct {
    recent: *const [8]u64,
    orders: []u32,
    value: u64 = 0,
    size: u64 = std.math.maxInt(u64),
    alloc: std.mem.Allocator,

    const factors = [6]u64{ 1, 256, 29 * 31, 29 * 31 * 37, 29 * 31 * 37 * 41, 29 * 31 * 37 * 41 * 43 };

    pub fn create(a: std.mem.Allocator, recent: *const [8]u64, orders: []const u32) *Sparse {
        const self = a.create(Sparse) catch unreachable;
        self.* = .{ .recent = recent, .orders = a.dupe(u32, orders) catch unreachable, .alloc = a };
        return self;
    }
    pub fn update(self: *Sparse) void {
        self.value = self.recent[self.orders[0]];
        var i: usize = 1;
        while (i < self.orders.len) : (i += 1) {
            // ref indexes factors_ by the loop position i (not by the order value).
            self.value +%= factors[i] *% self.recent[self.orders[i]];
        }
    }
    pub fn value_ptr(self: *Sparse) *u64 {
        return &self.value;
    }
    pub fn context(self: *Sparse) Context {
        return .{ .ptr = self, .update_fn = erased, .value = &self.value, .size = self.size };
    }
    fn erased(p: *anyopaque) void {
        update(@ptrCast(@alignCast(p)));
    }
    pub fn deinit(self: *Sparse) void {
        self.alloc.free(self.orders);
    }
};

/// IndirectHash: two-level hashed context. (cmix contexts/indirect-hash)
pub const IndirectHash = struct {
    byte: *const u32,
    context1: u64 = 0,
    hash_size1: u6,
    hash_size2: u6,
    size1: u64,
    value: u64 = 0,
    size: u64,
    hashes: []u64,
    alloc: std.mem.Allocator,

    // Memory cap for the internal `hashes_` table (cmix targets 32 GB; we don't).
    // Capping the exponent shrinks the table but stays lossless.
    const MAX_HASH1_BITS: u32 = 21;

    pub fn create(a: std.mem.Allocator, byte: *const u32, order1: u32, hash_size1: u32, order2: u32, hash_size2: u32) *IndirectHash {
        const self = a.create(IndirectHash) catch unreachable;
        const bits1 = @min(hash_size1 * order1, MAX_HASH1_BITS);
        const s1: u64 = @as(u64, 1) << @intCast(bits1);
        self.* = .{
            .byte = byte,
            .hash_size1 = @intCast(hash_size1),
            .hash_size2 = @intCast(hash_size2),
            .size1 = s1,
            .size = @as(u64, 1) << @intCast(hash_size2 * order2),
            .hashes = a.alloc(u64, @intCast(s1)) catch unreachable,
            .alloc = a,
        };
        @memset(self.hashes, 0);
        return self;
    }
    pub fn update(self: *IndirectHash) void {
        self.hashes[@intCast(self.context1)] = (self.value *% (@as(u64, 1) << self.hash_size2) +% self.byte.*) % self.size;
        self.context1 = (self.context1 *% (@as(u64, 1) << self.hash_size1) +% self.byte.*) % self.size1;
        self.value = self.hashes[@intCast(self.context1)];
    }
    pub fn value_ptr(self: *IndirectHash) *u64 {
        return &self.value;
    }
    pub fn context(self: *IndirectHash) Context {
        return .{ .ptr = self, .update_fn = erased, .value = &self.value, .size = self.size };
    }
    fn erased(p: *anyopaque) void {
        update(@ptrCast(@alignCast(p)));
    }
    pub fn deinit(self: *IndirectHash) void {
        self.alloc.free(self.hashes);
    }
};

fn shiftForMap(map: *const [256]i32) u6 {
    var max_value: i32 = 0;
    for (map) |v| {
        if (v > max_value) max_value = v;
    }
    var shift: i32 = 1;
    while ((@as(i32, 1) << @intCast(shift)) <= max_value) shift += 1;
    return @intCast(shift);
}

/// Interval: rolling window of mapped byte classes. (cmix contexts/interval)
pub const Interval = struct {
    byte: *const u32,
    map: [256]i32,
    mask: u64,
    shift: u6,
    value: u64 = 0,
    size: u64,

    pub fn create(a: std.mem.Allocator, byte: *const u32, map: *const [256]i32, num_bits: u32) *Interval {
        const self = a.create(Interval) catch unreachable;
        self.* = .{
            .byte = byte,
            .map = map.*,
            .shift = shiftForMap(map),
            .size = @as(u64, 1) << @intCast(num_bits),
            .mask = (@as(u64, 1) << @intCast(num_bits)) - 1,
        };
        return self;
    }
    pub fn update(self: *Interval) void {
        self.value = self.mask & ((self.value << self.shift) +% @as(u64, @intCast(self.map[self.byte.*])));
    }
    pub fn value_ptr(self: *Interval) *u64 {
        return &self.value;
    }
    pub fn context(self: *Interval) Context {
        return .{ .ptr = self, .update_fn = erased, .value = &self.value, .size = self.size };
    }
    fn erased(p: *anyopaque) void {
        update(@ptrCast(@alignCast(p)));
    }
};

/// IntervalHash: hashed rolling interval. (cmix contexts/interval-hash)
pub const IntervalHash = struct {
    byte: *const u32,
    map: [256]i32,
    mask: u64,
    hash_size: u6,
    interval: u64 = 0,
    shift: u6,
    value: u64 = 0,
    size: u64,

    pub fn create(a: std.mem.Allocator, byte: *const u32, map: *const [256]i32, num_bits: u32, order: u32, hash_size: u32) *IntervalHash {
        const self = a.create(IntervalHash) catch unreachable;
        self.* = .{
            .byte = byte,
            .map = map.*,
            .shift = shiftForMap(map),
            .hash_size = @intCast(hash_size),
            .mask = (@as(u64, 1) << @intCast(num_bits)) - 1,
            .size = @as(u64, 1) << @intCast(hash_size * order),
        };
        return self;
    }
    pub fn update(self: *IntervalHash) void {
        self.interval = self.mask & ((self.interval << self.shift) +% @as(u64, @intCast(self.map[self.byte.*])));
        self.value = (self.value *% (@as(u64, 1) << self.hash_size) +% self.interval) % self.size;
    }
    pub fn value_ptr(self: *IntervalHash) *u64 {
        return &self.value;
    }
    pub fn context(self: *IntervalHash) Context {
        return .{ .ptr = self, .update_fn = erased, .value = &self.value, .size = self.size };
    }
    fn erased(p: *anyopaque) void {
        update(@ptrCast(@alignCast(p)));
    }
};

/// CombinedContext: (context2 << shift) + context1. (cmix contexts/combined-context)
pub const CombinedContext = struct {
    context1: *const u64,
    context2: *const u64,
    shift: u6,
    value: u64 = 0,
    size: u64,

    pub fn create(a: std.mem.Allocator, context1: *const u64, context2: *const u64, context1_size: u64, context2_size: u64) *CombinedContext {
        const self = a.create(CombinedContext) catch unreachable;
        var shift: u6 = 1;
        while ((@as(u64, 1) << shift) < context1_size) shift += 1;
        self.* = .{ .context1 = context1, .context2 = context2, .shift = shift, .size = context1_size * context2_size };
        return self;
    }
    pub fn update(self: *CombinedContext) void {
        self.value = (self.context2.* << self.shift) +% self.context1.*;
    }
    pub fn value_ptr(self: *CombinedContext) *u64 {
        return &self.value;
    }
    pub fn context(self: *CombinedContext) Context {
        return .{ .ptr = self, .update_fn = erased, .value = &self.value, .size = self.size };
    }
    fn erased(p: *anyopaque) void {
        update(@ptrCast(@alignCast(p)));
    }
};

/// BracketContext: tracks nesting of {} [] <> to context on the enclosing
/// bracket + distance. (cmix contexts/bracket-context)
pub const BracketContext = struct {
    byte: *const u32,
    distance_limit: u32,
    stack_limit: u32,
    active: std.ArrayList(u32) = .empty,
    distance: std.ArrayList(u32) = .empty,
    value: u64 = 0,
    size: u64,
    alloc: std.mem.Allocator,

    fn isOpen(c: u32) bool {
        return c == '(' or c == '{' or c == '[' or c == '<';
    }
    fn closeOf(c: u32) u32 {
        return switch (c) {
            '(' => ')',
            '{' => '}',
            '[' => ']',
            '<' => '>',
            else => 0,
        };
    }

    pub fn create(a: std.mem.Allocator, byte: *const u32, distance_limit: u32, stack_limit: u32) *BracketContext {
        const self = a.create(BracketContext) catch unreachable;
        self.* = .{ .byte = byte, .distance_limit = distance_limit, .stack_limit = stack_limit, .size = 257 * distance_limit, .alloc = a };
        return self;
    }

    pub fn update(self: *BracketContext) void {
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
            // cmix compares brackets_.size(==4) to stack_limit; with the
            // configured stack_limit this never trims, so we mirror that.
            if (4 > self.stack_limit) {
                _ = self.active.orderedRemove(0);
                _ = self.distance.orderedRemove(0);
            }
        }
        if (self.active.items.len != 0) {
            self.value = self.distance_limit * (self.active.items[self.active.items.len - 1] + 1) +
                self.distance.items[self.distance.items.len - 1];
        } else {
            self.value = 0;
        }
    }

    pub fn value_ptr(self: *BracketContext) *u64 {
        return &self.value;
    }
    pub fn context(self: *BracketContext) Context {
        return .{ .ptr = self, .update_fn = erased, .value = &self.value, .size = self.size };
    }
    fn erased(p: *anyopaque) void {
        update(@ptrCast(@alignCast(p)));
    }
    pub fn deinit(self: *BracketContext) void {
        self.active.deinit(self.alloc);
        self.distance.deinit(self.alloc);
    }
};

pub const ContextManager = struct {
    bit_context: u32 = 1,
    wrt_state: u32 = 0,
    long_bit_context: u64 = 1,
    zero_context: u64 = 0,
    history_pos: u64 = 0,
    line_break: u64 = 0,
    longest_match: u64 = 0,
    auxiliary_context: u64 = 0,
    wrt_context: u64 = 0,
    history: []u8,
    shared_map: []u8,
    words: [8]u64 = .{0} ** 8,
    recent_bytes: [8]u64 = .{0} ** 8,
    contexts: std.ArrayList(Context) = .empty,
    bit_contexts: std.ArrayList(Context) = .empty,
    run_map: states.RunMap,
    nonstationary: states.Nonstationary = .{},
    alloc: std.mem.Allocator,

    pub fn init(a: std.mem.Allocator) !*ContextManager {
        const self = try a.create(ContextManager);
        self.* = .{
            .history = try a.alloc(u8, HISTORY_SIZE),
            .shared_map = try a.alloc(u8, SHARED_MAP_SIZE),
            .run_map = states.RunMap.init(),
            .alloc = a,
        };
        @memset(self.history, 0);
        @memset(self.shared_map, 0);
        return self;
    }

    pub fn deinit(self: *ContextManager) void {
        self.alloc.free(self.history);
        self.alloc.free(self.shared_map);
        self.contexts.deinit(self.alloc);
        self.bit_contexts.deinit(self.alloc);
        self.alloc.destroy(self);
    }

    pub fn addContext(self: *ContextManager, c: Context) void {
        self.contexts.append(self.alloc, c) catch unreachable;
    }
    pub fn addBitContext(self: *ContextManager, c: Context) void {
        self.bit_contexts.append(self.alloc, c) catch unreachable;
    }

    fn updateHistory(self: *ContextManager) void {
        self.history[self.history_pos] = @intCast(self.bit_context);
        self.history_pos += 1;
        if (self.history_pos == self.history.len) self.history_pos = 0;
    }

    fn updateWords(self: *ContextManager) void {
        var c: u32 = self.bit_context;
        if ((c >= 'a' and c <= 'z') or (c >= 'A' and c <= 'Z') or c >= 0x80) {
            self.words[7] = self.words[7] *% (997 * 16) +% c;
        } else {
            self.words[7] = 0;
        }
        if (c >= 'A' and c <= 'Z') c += 'a' - 'A';
        if ((c >= 'a' and c <= 'z') or (c >= '0' and c <= '9') or c == 8 or c == 6 or c >= 0x80) {
            self.words[0] = (self.words[0] *% (997 * 16) +% c) & 0xfffffff;
            self.words[1] = self.words[1] *% (263 * 32) +% c;
        } else {
            var i: usize = 6;
            while (i >= 2) : (i -= 1) self.words[i] = self.words[i - 1];
            self.words[1] = 0;
        }
    }

    fn updateRecentBytes(self: *ContextManager) void {
        var i: usize = 7;
        while (i >= 1) : (i -= 1) self.recent_bytes[i] = self.recent_bytes[i - 1];
        self.recent_bytes[0] = self.bit_context;
    }

    fn updateWRTContext(self: *ContextManager) void {
        if (self.bit_context < 0x80) {
            self.wrt_state = 0;
        } else {
            if (self.wrt_state == 0) self.wrt_context = 0;
            self.wrt_state = 1;
            self.wrt_context <<= 8;
            self.wrt_context += self.bit_context;
            if (self.wrt_context > 0xFFEFCF) self.wrt_context = 0;
        }
    }

    pub fn updateContexts(self: *ContextManager, bit: i32) void {
        self.bit_context += self.bit_context + @as(u32, @intCast(bit));
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
            self.updateWRTContext();
            for (self.contexts.items) |c| c.update();
        }
        for (self.bit_contexts.items) |c| c.update();
    }
};
