//! BracketContext — bit-exact Zig port of cmix-lex `fxcmv1.cpp` `struct
//! BracketContext` (+ its `vec`). A byte-driven bracket/quote/first-char stack used
//! by the parseByte preamble (instances brcxt/qocxt/fccxt/htcxt). Outputs cxt/dst/
//! context/lastthat gate many word-branch conditions and feed cm.set contexts.
//!
//! Ported 1:1 from bracket_context_v26.rs. T is `U8` for brcxt/qocxt/fccxt =>
//! the (1<<8)=256 multiplier and 255 dst clamp; `BracketContextW` is the U16
//! htcxt instance (65536 multiplier, 65535 dst clamp).

const std = @import("std");

const CAP: usize = 512;

/// vec<int,512> — fixed-capacity int stack matching fxcmv1.cpp's `vec`.
const VecI = struct {
    data: [CAP]i32,
    size: usize,

    fn new() VecI {
        return .{ .data = [_]i32{0} ** CAP, .size = 0 };
    }

    inline fn push(self: *VecI, e: i32) void {
        if (self.size >= CAP) {
            self.size = CAP - 1;
        }
        self.data[self.size] = e;
        self.size += 1;
    }

    inline fn at(self: *const VecI, index: i32) i32 {
        if (index < 0 or @as(usize, @intCast(index)) >= CAP) {
            return 0;
        } else {
            return self.data[@intCast(index)];
        }
    }

    inline fn inc(self: *VecI, index: i32) void {
        if (index < 0 or @as(usize, @intCast(index)) >= CAP) {
            return;
        }
        self.data[@intCast(index)] +%= 1;
    }

    inline fn pop(self: *VecI) void {
        if (self.size > 0) {
            self.size -= 1;
            self.data[self.size] = 0;
        }
    }

    inline fn reset(self: *VecI) void {
        self.data[0] = 0;
        self.size = 0;
    }

    inline fn empty(self: *const VecI) bool {
        return self.size == 0;
    }

    inline fn prev(self: *const VecI) i32 {
        if (self.size > 1) {
            return self.data[self.size - 2];
        } else {
            return 0;
        }
    }
};

pub const BracketContext = struct {
    context: u32,
    active: VecI,
    distance: VecI,
    element: []const u8,
    element_count: usize,
    do_pop: bool,
    limit: i32,
    cxt: u8,
    dst: u8,

    pub fn new() BracketContext {
        return .{
            .context = 0,
            .active = VecI.new(),
            .distance = VecI.new(),
            .element = &.{},
            .element_count = 0,
            .do_pop = false,
            .limit = 0,
            .cxt = 0,
            .dst = 0,
        };
    }

    pub fn init(self: *BracketContext, element: []const u8, element_count: usize, do_pop: bool, limit: i32) void {
        self.element = element;
        self.element_count = element_count;
        self.do_pop = do_pop;
        self.limit = limit;
        self.active.reset();
        self.distance.reset();
        self.context = 0;
        self.cxt = 0;
        self.dst = 0;
    }

    /// fxcmv1.cpp BracketContext::Reset (keeps element/limit/do_pop).
    pub fn reset(self: *BracketContext) void {
        self.active.reset();
        self.distance.reset();
        self.context = 0;
        self.cxt = 0;
        self.dst = 0;
    }

    inline fn find(self: *const BracketContext, b: i32) bool {
        var i: usize = 0;
        while (i < self.element_count) : (i += 2) {
            if (@as(i32, self.element[i]) == b) {
                return true;
            }
        }
        return false;
    }

    inline fn find_end(self: *const BracketContext, b: i32, c: i32) bool {
        var found = false;
        var i: usize = 0;
        while (i < self.element_count) : (i += 2) {
            if (@as(i32, self.element[i]) == b and @as(i32, self.element[i + 1]) == c) {
                found = true;
            }
        }
        return found;
    }

    pub fn last(self: *const BracketContext) i32 {
        return self.active.prev();
    }

    pub fn update(self: *BracketContext, byte: i32) void {
        var pop = false;
        if (!self.active.empty()) {
            const asz: i32 = @intCast(self.active.size);
            const dsz: i32 = @intCast(self.distance.size);
            if (self.find_end(self.active.at(asz - 1), byte) or self.distance.at(dsz - 1) >= self.limit) {
                self.active.pop();
                self.distance.pop();
                pop = self.do_pop;
            } else {
                self.distance.inc(dsz - 1);
            }
        }
        if (!pop and self.find(byte)) {
            self.active.push(byte);
            self.distance.push(0);
        }
        if (!self.active.empty()) {
            const asz: i32 = @intCast(self.active.size);
            const dsz: i32 = @intCast(self.distance.size);
            self.cxt = @truncate(@as(u32, @bitCast(self.active.at(asz - 1))));
            self.dst = @truncate(@as(u32, @bitCast(@min(self.distance.at(dsz - 1), 255))));
            self.context = 256 *% @as(u32, self.cxt) +% @as(u32, self.dst);
        } else {
            self.context = 0;
            self.cxt = 0;
            self.dst = 0;
        }
    }
};

/// BracketContext<U16> — the htcxt instance (element pairs are 16-bit values,
/// dst clamp 65535, context = 65536*cxt + dst). Same algorithm as the U8 variant.
pub const BracketContextW = struct {
    context: u32,
    active: VecI,
    distance: VecI,
    element: []const u16,
    element_count: usize,
    do_pop: bool,
    limit: i32,
    cxt: u16,
    dst: u16,

    pub fn new() BracketContextW {
        return .{
            .context = 0,
            .active = VecI.new(),
            .distance = VecI.new(),
            .element = &.{},
            .element_count = 0,
            .do_pop = false,
            .limit = 0,
            .cxt = 0,
            .dst = 0,
        };
    }

    pub fn init(self: *BracketContextW, element: []const u16, element_count: usize, do_pop: bool, limit: i32) void {
        self.element = element;
        self.element_count = element_count;
        self.do_pop = do_pop;
        self.limit = limit;
        self.reset();
    }

    pub fn reset(self: *BracketContextW) void {
        self.active.reset();
        self.distance.reset();
        self.context = 0;
        self.cxt = 0;
        self.dst = 0;
    }

    inline fn find(self: *const BracketContextW, b: i32) bool {
        var i: usize = 0;
        while (i < self.element_count) : (i += 2) {
            if (@as(i32, self.element[i]) == b) {
                return true;
            }
        }
        return false;
    }

    inline fn find_end(self: *const BracketContextW, b: i32, c: i32) bool {
        var found = false;
        var i: usize = 0;
        while (i < self.element_count) : (i += 2) {
            if (@as(i32, self.element[i]) == b and @as(i32, self.element[i + 1]) == c) {
                found = true;
            }
        }
        return found;
    }

    pub fn last(self: *const BracketContextW) i32 {
        return self.active.prev();
    }

    pub fn update(self: *BracketContextW, v: i32) void {
        var pop = false;
        if (!self.active.empty()) {
            const asz: i32 = @intCast(self.active.size);
            const dsz: i32 = @intCast(self.distance.size);
            if (self.find_end(self.active.at(asz - 1), v) or self.distance.at(dsz - 1) >= self.limit) {
                self.active.pop();
                self.distance.pop();
                pop = self.do_pop;
            } else {
                self.distance.inc(dsz - 1);
            }
        }
        if (!pop and self.find(v)) {
            self.active.push(v);
            self.distance.push(0);
        }
        if (!self.active.empty()) {
            const asz: i32 = @intCast(self.active.size);
            const dsz: i32 = @intCast(self.distance.size);
            self.cxt = @truncate(@as(u32, @bitCast(self.active.at(asz - 1))));
            self.dst = @truncate(@as(u32, @bitCast(@min(self.distance.at(dsz - 1), 65535))));
            self.context = (65536 *% @as(u32, self.cxt)) +% @as(u32, self.dst);
        } else {
            self.context = 0;
            self.cxt = 0;
            self.dst = 0;
        }
    }
};

test "bracket_context_matches_cpp_oracle" {
    const run = struct {
        fn f(element: []const u8, ec: usize, pop: bool, limit: i32) u64 {
            var bc = BracketContext.new();
            bc.init(element, ec, pop, limit);
            const alpha = [16]u8{ 40, 41, 80, 82, 91, 93, 76, 78, 39, 34, 'a', ' ', 'x', 64, 10, 42 };
            var cs: u64 = 0;
            var r: u32 = 0x12345678;
            var iter: usize = 0;
            while (iter < 8000) : (iter += 1) {
                r = r *% 1664525 +% 1013904223;
                const byte: i32 = alpha[(r >> 16) & 15];
                bc.update(byte);
                cs = cs *% 1000003 +% @as(u64, bc.cxt);
                cs = cs *% 1000003 +% @as(u64, bc.context);
                cs = cs *% 1000003 +% @as(u64, bc.dst);
                cs = cs *% 1000003 +% @as(u64, @as(u32, @bitCast(bc.last())));
            }
            return cs;
        }
    }.f;

    const BRACKETS = [8]u8{ 40, 41, 80, 82, 91, 93, 76, 78 };
    const QUOTES = [4]u8{ 39, 39, 34, 34 };
    const FCHAR = [20]u8{ 64, 10, 96, 10, 74, 10, 76, 78, 77, 10, 91, 93, 80, 82, 42, 10, 81, 10, 31, 10 };

    try std.testing.expectEqual(@as(u64, 0x2df9f4b2a9ba79ac), run(&BRACKETS, 8, false, 256));
    try std.testing.expectEqual(@as(u64, 0xb6377ffa7456a1e5), run(&QUOTES, 4, true, 256));
    try std.testing.expectEqual(@as(u64, 0x6c427bc99ff788fa), run(&FCHAR, 20, false, 20));
    try std.testing.expectEqual(@as(u64, 0x7d562607ce55aebf), run(&BRACKETS, 8, false, 3));
}
