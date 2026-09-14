//! fxcm_v26 text/table contexts, ported from reference/fxcmv1_v26.cpp
//! (dictionary/WRT mode — TEXTMODE off, so char constants are char-swapped).
//!   Vec helper, BracketContext, ColumnContext, WordsContext, hash, charSwap,
//!   and the tuning parameter tables.
const std = @import("std");
const st = @import("state.zig");

// Char constants (WRT/dictionary mode).
pub const COLON = 'J';
pub const SEMICOLON = 'K';
pub const LESSTHAN = 'L';
pub const EQUALS = 'M';
pub const GREATERTHAN = 'N';
pub const QUESTION = 'O';
pub const ATSIGN = 64;
pub const SQUAREOPEN = 91;
pub const BACKSLASH = 92;
pub const SQUARECLOSE = 93;
pub const CURLYOPENING = 'P';
pub const VERTICALBAR = 'Q';
pub const CURLYCLOSE = 'R';
pub const APOSTROPHE = 39;
pub const QUOTATION = 34;
pub const SPACE = 32;

pub const brackets = [8]u8{ '(', ')', CURLYOPENING, CURLYCLOSE, '[', ']', LESSTHAN, GREATERTHAN };
pub const quotes = [4]u8{ APOSTROPHE, APOSTROPHE, QUOTATION, QUOTATION };
pub const fchar = [20]u8{ ATSIGN, 10, 96, 10, COLON, 10, LESSTHAN, GREATERTHAN, EQUALS, 10, SQUAREOPEN, SQUARECLOSE, CURLYOPENING, CURLYCLOSE, '*', 10, VERTICALBAR, 10, 31, 10 };

pub const primes = [14]u32{ 0, 257, 251, 241, 239, 233, 229, 227, 223, 211, 199, 197, 193, 191 };
pub const tri = [4]u32{ 0, 4, 3, 7 };
pub const trj = [4]u32{ 0, 6, 6, 12 };
pub const m_e = [10]u32{ 8, 8, 8, 1, 1, 1, 1, 1, 1, 0 };
pub const m_s = [10]u32{ 194, 237, 204, 70, 54, 55, 55, 70, 55, 6 };
pub const m_m = [10]u32{ 36, 69, 19, 34, 23, 24, 24, 34, 24, 4 };
pub const c_r = [27]u32{ 3, 4, 6, 4, 6, 6, 2, 3, 3, 3, 6, 4, 3, 4, 5, 6, 2, 6, 4, 4, 4, 4, 4, 4, 4, 4, 4 };
pub const c_s = [27]u32{ 28, 26, 28, 31, 34, 31, 33, 33, 35, 35, 29, 32, 33, 34, 30, 36, 31, 32, 32, 32, 32, 32, 33, 32, 32, 32, 32 };
pub const c_s2 = [27]u32{ 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 14, 12, 12, 12, 12 };
pub const c_s3 = [27]u32{ 43, 33, 34, 28, 34, 29, 32, 33, 37, 35, 33, 28, 31, 35, 28, 30, 33, 34, 32, 32, 32, 32, 32, 32, 32, 32, 32 };
pub const c_s4 = [27]u32{ 9, 8, 9, 5, 8, 12, 15, 8, 8, 12, 10, 7, 7, 8, 8, 13, 13, 14, 8, 8, 12, 12, 12, 12, 12, 12, 12 };
pub const MAXLEN = 62;

pub inline fn hash(a: u32, b: u32, c: u32) u32 {
    const h: u32 = a *% 110002499 +% b *% 30005491 +% c *% 50004239;
    return h ^ (h >> 9) ^ (a >> 3) ^ (b >> 3) ^ (c >> 4);
}

pub inline fn charSwap(c_in: i32) i32 {
    var c = c_in;
    if (c >= '{' and c < 127) c += 'P' - '{' else if (c >= 'P' and c < 'T') c -= 'P' - '{';
    if ((c >= ':' and c <= '?') or (c >= 'J' and c <= 'O')) c ^= 0x70;
    if (c == 'X' or c == '`') c ^= 'X' ^ '`';
    return c;
}

inline fn imin(a: i32, b: i32) i32 {
    return if (a < b) a else b;
}

/// Growable vector matching the reference vec<T> semantics.
pub fn Vec(comptime T: type) type {
    return struct {
        const Self = @This();
        items: std.ArrayList(T) = .empty,
        alloc: std.mem.Allocator,

        pub fn init(a: std.mem.Allocator) Self {
            return .{ .alloc = a };
        }
        pub fn size(self: *Self) i32 {
            return @intCast(self.items.items.len);
        }
        pub fn push(self: *Self, e: T) void {
            self.items.append(self.alloc, e) catch unreachable;
        }
        pub fn at(self: *Self, index: i32) T {
            return self.items.items[@intCast(index)];
        }
        pub fn inc(self: *Self, index: i32) void {
            self.items.items[@intCast(index)] += 1;
        }
        pub fn pop(self: *Self) void {
            self.items.items[self.items.items.len - 1] = 0;
            self.items.items.len -= 1;
        }
        pub fn reset(self: *Self) void {
            if (self.items.items.len > 0) self.items.items[0] = 0;
            self.items.items.len = 0;
        }
        pub fn empty(self: *Self) bool {
            return self.items.items.len == 0;
        }
        pub fn prev(self: *Self) T {
            return if (self.items.items.len > 1) self.items.items[self.items.items.len - 2] else 0;
        }
    };
}

pub const BracketContext = struct {
    context: u32 = 0,
    active: Vec(i32),
    distance: Vec(i32),
    element: []const u8,
    doPop: bool = false,
    limit: i32 = 255,
    cxt: u8 = 0,
    dst: u8 = 0,

    pub fn init(self: *BracketContext, a: std.mem.Allocator, d: []const u8, pop: bool, l: i32) void {
        self.* = .{ .active = Vec(i32).init(a), .distance = Vec(i32).init(a), .element = d, .doPop = pop, .limit = l };
    }

    pub fn reset(self: *BracketContext) void {
        self.active.reset();
        self.distance.reset();
        self.context = 0;
        self.cxt = 0;
        self.dst = 0;
    }

    /// ref `last` = vec_prev(&active): the second-to-last open bracket (0 if <2).
    pub fn last(self: *BracketContext) i32 {
        return self.active.prev();
    }

    fn find(self: *BracketContext, b: i32) bool {
        var i: usize = 0;
        while (i < self.element.len) : (i += 2) {
            if (self.element[i] == b) return true;
        }
        return false;
    }
    fn findEnd(self: *BracketContext, b: i32, c: i32) bool {
        var found = false;
        var i: usize = 0;
        while (i < self.element.len) : (i += 2) {
            if (self.element[i] == b and self.element[i + 1] == c) found = true;
        }
        return found;
    }

    pub fn update(self: *BracketContext, byte: i32) void {
        var pop = false;
        if (!self.active.empty()) {
            if (self.findEnd(self.active.at(self.active.size() - 1), byte) or self.distance.at(self.distance.size() - 1) >= self.limit) {
                self.active.pop();
                self.distance.pop();
                pop = self.doPop;
            } else {
                self.distance.inc(self.distance.size() - 1);
            }
        }
        if (pop == false and self.find(byte)) {
            self.active.push(byte);
            self.distance.push(0);
        }
        if (!self.active.empty()) {
            self.cxt = @intCast(self.active.at(self.active.size() - 1));
            self.dst = @intCast(imin(self.distance.at(self.distance.size() - 1), 255));
            self.context = 256 * @as(u32, self.cxt) + self.dst;
        } else {
            self.context = 0;
            self.cxt = 0;
            self.dst = 0;
        }
    }
};

pub const Column = struct {
    linepos: u32 = 0,
    fc: u8 = 0,
    bytes: Vec(u8),
};

pub const ColumnContext = struct {
    col: [4]Column,
    cell: [4]Vec(u32),
    rows: i32 = 0,
    cellCount: i32 = 0,
    cells: i32 = 0,
    abovecellpos: i32 = 0,
    abovecellpos1: i32 = 0,
    NL: bool = false,
    isTemp: bool = false,
    limit: i32 = 31,
    nlChar: u8 = 10,

    pub fn init(self: *ColumnContext, a: std.mem.Allocator, l: i32) void {
        self.* = .{
            .col = .{ .{ .bytes = Vec(u8).init(a) }, .{ .bytes = Vec(u8).init(a) }, .{ .bytes = Vec(u8).init(a) }, .{ .bytes = Vec(u8).init(a) } },
            .cell = .{ Vec(u32).init(a), Vec(u32).init(a), Vec(u32).init(a), Vec(u32).init(a) },
            .limit = l,
        };
    }

    pub fn lastfc(self: *ColumnContext, i: i32) u8 {
        return self.col[@intCast((self.rows - i) & 3)].fc;
    }
    pub fn isNewLine(self: *ColumnContext) bool {
        return self.NL;
    }
    pub fn collen(self: *ColumnContext, i: i32, l: i32) i32 {
        return imin(if (l != 0) l else self.limit, self.col[@intCast((self.rows - i) & 3)].bytes.size() + 1);
    }
    pub fn nlpos(self: *ColumnContext, i: i32) u32 {
        return self.col[@intCast((self.rows - i) & 3)].linepos;
    }
    pub fn colb(self: *ColumnContext, i: i32, j: i32, l: i32) u8 {
        if (self.collen(0, l) < self.collen(i, l)) {
            // Guard the byte index: after a newline whose line starts at column 0,
            // collen(0,0)-(1+j) can be negative (the reference reads cxt[-1] UB).
            // Return 0 there — deterministic, so the round-trip is unaffected.
            const idx = self.collen(0, 0) - (1 + j);
            if (idx < 0) return 0;
            return self.col[@intCast((self.rows - i) & 3)].bytes.at(idx);
        }
        return 0;
    }

    pub fn cellsCount(self: *ColumnContext, row: i32) i32 {
        return self.cell[@intCast((self.cells - row) & 3)].size();
    }
    pub fn cellPos(self: *ColumnContext, cellID: i32, row: i32) i32 {
        var total = self.cellsCount(row) - 1;
        total = imin(total, cellID);
        return @intCast(self.cell[@intCast((self.cells - row) & 3)].at(total));
    }
    pub fn resetCells(self: *ColumnContext) void {
        for (0..4) |i| self.cell[i].reset();
    }

    pub fn update(self: *ColumnContext, byte: i32, b2: u32) void {
        if (b2 == ((CURLYOPENING << 16) + (CURLYOPENING << 8) + VERTICALBAR)) {
            self.nlChar = '-';
        } else if (b2 == ((VERTICALBAR << 16) + (CURLYCLOSE << 8) + CURLYCLOSE)) {
            self.nlChar = 10;
            self.resetCells();
        }
        if (byte != CURLYOPENING and (b2 & 0xff00) == (CURLYOPENING << 8) and (b2 & 0xff0000) != (CURLYOPENING << 16)) {
            self.isTemp = true;
        } else if (self.isTemp == true and byte == CURLYCLOSE) {
            self.isTemp = false;
        }
        self.NL = false;
        if (byte == 10) {
            self.col[@intCast(self.rows)].bytes.push(@intCast(byte));
            self.rows += 1;
            self.rows = self.rows & 3;
            self.col[@intCast(self.rows)].bytes.reset();
            self.col[@intCast(self.rows)].fc = 0;
            self.col[@intCast(self.rows)].linepos = @bitCast(st.x.blpos -% 1);
        } else {
            self.col[@intCast(self.rows)].bytes.push(@intCast(byte));
            if (self.collen(0, 0) == 2) {
                self.col[@intCast(self.rows)].fc = @intCast(imin(byte, 96));
                self.NL = true;
            }
        }
        if (self.nlChar == '-') {
            if ((b2 & 0xffff) == ('-' + VERTICALBAR * 256)) {
                self.cells += 1;
                self.cells = self.cells & 3;
                self.cell[@intCast(self.cells)].reset();
                self.cell[@intCast(self.cells)].push(@bitCast(st.x.blpos));
                self.cellCount = 0;
                self.abovecellpos = 0;
                self.abovecellpos1 = 0;
            }
            var newcell = false;
            if ((b2 & 0xffff) == (VERTICALBAR + VERTICALBAR * 256) or
                (b2 & 0xffff00) == ((VERTICALBAR + 10 * 256) * 256))
            {
                self.cell[@intCast(self.cells)].push(@bitCast(st.x.blpos));
                self.cellCount += 1;
                newcell = true;
            }
            if (self.abovecellpos != 0) {
                self.abovecellpos += 1;
                if (self.abovecellpos > self.abovecellpos1) {
                    self.abovecellpos = 0;
                    self.abovecellpos1 = 0;
                }
            }
            if (newcell == true and self.cellsCount(1) > 0) {
                self.abovecellpos = self.cellPos(self.cellCount - 1, 1);
                self.abovecellpos1 = self.cellPos(self.cellCount, 1);
            }
        }
        if (self.nlChar == GREATERTHAN) {
            if ((b2 & 0xffff) == (GREATERTHAN + 10 * 256)) {
                self.cells += 1;
                self.cells = self.cells & 3;
                self.cell[@intCast(self.cells)].reset();
                self.cell[@intCast(self.cells)].push(@bitCast(st.x.blpos));
                self.cellCount = 0;
                self.abovecellpos = 0;
                self.abovecellpos1 = 0;
            } else {
                var newcell = false;
                if ((b2 & 0xff) == GREATERTHAN) {
                    self.cell[@intCast(self.cells)].push(@bitCast(st.x.blpos));
                    self.cellCount += 1;
                    newcell = true;
                }
                if (self.abovecellpos != 0) {
                    self.abovecellpos += 1;
                    if (self.abovecellpos > self.abovecellpos1) {
                        self.abovecellpos = 0;
                        self.abovecellpos1 = 0;
                    }
                }
                if (newcell == true and self.cellsCount(1) > 0) {
                    self.abovecellpos = self.cellPos(self.cellCount - 1, 1);
                    self.abovecellpos1 = self.cellPos(self.cellCount, 1);
                }
            }
        }
    }
};

pub const WordsContext = struct {
    awords: Vec(u32),
    sbytes: Vec(u32),
    fword: u32 = 0,
    pbyte: u8 = 0,

    pub fn init(self: *WordsContext, a: std.mem.Allocator) void {
        self.* = .{ .awords = Vec(u32).init(a), .sbytes = Vec(u32).init(a) };
    }
    pub fn reset(self: *WordsContext) void {
        self.awords.reset();
        self.sbytes.reset();
        self.fword = 0;
        self.pbyte = 0;
    }
    pub fn set(self: *WordsContext, b: u8) void {
        self.pbyte = b;
    }
    pub fn update(self: *WordsContext, w: u32, b: u8) void {
        if (self.fword == 0) self.fword = w;
        self.awords.push(w);
        self.sbytes.push(@as(u32, self.pbyte) * 256 + b);
        self.pbyte = 0;
    }
    pub fn remove(self: *WordsContext) void {
        if (self.awords.size() != 0) {
            self.awords.pop();
            self.sbytes.pop();
        }
    }
    pub fn word(self: *WordsContext, i: i32) u32 {
        const num = self.awords.size();
        return if (num >= i) self.awords.at(num - i) else 0;
    }
    pub fn sBytes(self: *WordsContext, i: i32) u32 {
        const num = self.sbytes.size();
        return if (num >= i) self.sbytes.at(num - i) else 0;
    }
};
