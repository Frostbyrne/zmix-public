//! ColumnContext — bit-exact Zig port of cmix-lex `fxcmv1.cpp` `struct
//! ColumnContext` (+ Column/vec). A byte-driven row/column/table tracker used by the
//! parseByte preamble; its `lastfc`/`nlChar`/`isTemp`/cell outputs gate word-branch
//! conditions and feed cm.set contexts.
//!
//! Ported 1:1 from column_context_v26.rs (which is validated bit-exact vs
//! v26-port-tools/column_oracle.cpp). Depends on a per-byte blpos and the global
//! `is_pre` (false in this validation).

const std = @import("std");

const CURLYOPENING: i32 = 80;
const CURLYCLOSE: i32 = 82;
const VERTICALBAR: i32 = 81;
const WIKITABLE: u8 = 45;
const WIKIHEADER: u8 = 78;
const LF: i32 = 10;
const SQUAREOPEN: i32 = 91;
const TEXTDATA: i32 = 96;
const GREATERTHAN: i32 = 78;

/// fxcmv1.cpp `vec<T,S>` over u32 storage (u8 values fit). `cap` = S.
const GVec = struct {
    data: []u32,
    cap: usize,
    size: usize,

    fn new(a: std.mem.Allocator, cap: usize) !GVec {
        const data = try a.alloc(u32, cap);
        @memset(data, 0);
        return GVec{ .data = data, .cap = cap, .size = 0 };
    }

    inline fn push(self: *GVec, e: u32) void {
        if (self.size >= self.cap) {
            self.size = self.cap - 1;
        }
        self.data[self.size] = e;
        self.size += 1;
    }

    inline fn at(self: *const GVec, index: i32) u32 {
        if (index < 0 or @as(usize, @intCast(index)) >= self.cap) {
            return 0;
        } else {
            return self.data[@as(usize, @intCast(index))];
        }
    }

    inline fn getSize(self: *const GVec) i32 {
        return @intCast(self.size);
    }

    inline fn reset(self: *GVec) void {
        self.data[0] = 0;
        self.size = 0;
    }

    fn deinit(self: *GVec, a: std.mem.Allocator) void {
        a.free(self.data);
        self.* = undefined;
    }
};

const Column = struct {
    linepos: u32,
    fc: u8,
    bytes: GVec,
};

pub const ColumnContext = struct {
    col: [4]Column,
    cell: [4]GVec,
    rows: i32,
    cell_count: i32,
    cells: i32,
    abovecellpos: i32,
    abovecellpos1: i32,
    nl: bool,
    is_temp: bool,
    limit: i32,
    nl_char: u8,
    /// global `isPre` dependency (set by caller; false during the unit validation).
    is_pre: bool,

    pub fn new(a: std.mem.Allocator) !ColumnContext {
        var col: [4]Column = undefined;
        for (&col) |*c| {
            c.* = Column{ .linepos = 0, .fc = 0, .bytes = try GVec.new(a, 2048) };
        }
        var cell: [4]GVec = undefined;
        for (&cell) |*c| {
            c.* = try GVec.new(a, 32);
        }
        return ColumnContext{
            .col = col,
            .cell = cell,
            .rows = 0,
            .cell_count = 0,
            .cells = 0,
            .abovecellpos = 0,
            .abovecellpos1 = 0,
            .nl = false,
            .is_temp = false,
            .limit = 31,
            .nl_char = @as(u8, @intCast(LF)),
            .is_pre = false,
        };
    }

    pub fn deinit(self: *ColumnContext, a: std.mem.Allocator) void {
        for (&self.col) |*c| c.bytes.deinit(a);
        for (&self.cell) |*c| c.deinit(a);
    }

    pub fn init(self: *ColumnContext, l: i32) void {
        self.rows = 0;
        self.abovecellpos = 0;
        self.cell_count = 0;
        self.abovecellpos1 = 0;
        self.cells = 0;
        self.nl_char = @as(u8, @intCast(LF));
        self.limit = l;
        self.nl = false;
        self.is_temp = false;
        self.reset_cells();
        for (&self.col) |*c| {
            c.bytes.reset();
            c.fc = 0;
            c.linepos = 0;
        }
    }

    inline fn ri(self: *const ColumnContext, i: i32) usize {
        return @as(usize, (@as(u32, @bitCast(self.rows)) -% @as(u32, @bitCast(i))) & 3);
    }
    inline fn ci(self: *const ColumnContext, row: i32) usize {
        return @as(usize, (@as(u32, @bitCast(self.cells)) -% @as(u32, @bitCast(row))) & 3);
    }

    pub fn lastfc(self: *const ColumnContext, i: i32) u8 {
        return self.col[self.ri(i)].fc;
    }
    pub fn is_new_line(self: *const ColumnContext) bool {
        return self.nl;
    }
    pub fn collen(self: *const ColumnContext, i: i32, l: i32) i32 {
        const lim = if (l != 0) l else self.limit;
        return @min(lim, self.col[self.ri(i)].bytes.getSize() + 1);
    }
    pub fn nlpos(self: *const ColumnContext, i: i32) u32 {
        return self.col[self.ri(i)].linepos;
    }
    pub fn colb(self: *const ColumnContext, i: i32, j: i32, l: i32) u8 {
        const idx = self.collen(0, 0) - (1 + j);
        if (idx >= 0 and self.collen(0, l) < self.collen(i, l)) {
            return @as(u8, @truncate(self.col[self.ri(i)].bytes.at(idx)));
        } else {
            return 0;
        }
    }
    pub fn cells_count(self: *const ColumnContext, row: i32) i32 {
        return self.cell[self.ci(row)].getSize();
    }
    pub fn cell_pos(self: *const ColumnContext, cell_id: i32, row: i32) u32 {
        var total = self.cells_count(row) - 1;
        total = @min(total, cell_id);
        if (total < 0) {
            return 0;
        }
        return self.cell[self.ci(row)].at(total);
    }
    pub fn reset_cells(self: *ColumnContext) void {
        for (&self.cell) |*c| {
            c.reset();
        }
    }

    pub fn update(self: *ColumnContext, byte: i32, b2: u32, blpos: u32) void {
        if (b2 == @as(u32, @bitCast((CURLYOPENING << 16) + (CURLYOPENING << 8) + VERTICALBAR))) {
            self.nl_char = WIKITABLE;
        } else if (b2 == @as(u32, @bitCast((VERTICALBAR << 16) + (CURLYCLOSE << 8) + CURLYCLOSE))) {
            self.nl_char = @as(u8, @intCast(LF));
            self.reset_cells();
        }
        if (byte != CURLYOPENING and
            (b2 & 0xff00) == @as(u32, @bitCast(CURLYOPENING << 8)) and
            (b2 & 0xff0000) != @as(u32, @bitCast(CURLYOPENING << 16)))
        {
            self.is_temp = true;
        } else if (self.is_temp and byte == CURLYCLOSE) {
            self.is_temp = false;
        }
        self.nl = false;
        if (byte == LF) {
            const r = @as(usize, @intCast(self.rows));
            self.col[r].bytes.push(@as(u32, @bitCast(byte)));
            self.rows += 1;
            self.rows &= 3;
            const r2 = @as(usize, @intCast(self.rows));
            self.col[r2].bytes.reset();
            self.col[r2].fc = 0;
            self.col[r2].linepos = blpos -% 1;
        } else {
            const r = @as(usize, @intCast(self.rows));
            self.col[r].bytes.push(@as(u32, @bitCast(byte)));
            if (self.collen(0, 0) == 2) {
                self.col[r].fc = @as(u8, @intCast(@min(byte, TEXTDATA)));
                self.nl = true;
                if (@as(i32, self.col[r].fc) == GREATERTHAN and !self.is_pre) {
                    self.nl_char = WIKIHEADER;
                }
                if (@as(i32, self.col[r].fc) == SQUAREOPEN and self.nl_char == WIKIHEADER) {
                    self.nl_char = @as(u8, @intCast(LF));
                }
            }
        }
        if (self.nl_char == WIKITABLE) {
            if ((b2 & 0xffff) == (@as(u32, WIKITABLE) + @as(u32, @intCast(VERTICALBAR)) * 256)) {
                self.cells += 1;
                self.cells &= 3;
                const c = @as(usize, @intCast(self.cells));
                self.cell[c].reset();
                self.cell[c].push(blpos);
                self.cell_count = 0;
                self.abovecellpos = 0;
                self.abovecellpos1 = 0;
            }
            var newcell = false;
            if ((b2 & 0xffff) == (@as(u32, @intCast(VERTICALBAR)) + @as(u32, @intCast(VERTICALBAR)) * 256) or
                (b2 & 0xffff00) == ((@as(u32, @intCast(VERTICALBAR)) + @as(u32, @intCast(LF)) * 256) * 256) or
                ((b2 & 0xffff00) == ((@as(u32, @intCast(VERTICALBAR)) + @as(u32, @intCast(LF)) * 256) * 256) and
                    byte != VERTICALBAR))
            {
                const c = @as(usize, @intCast(self.cells));
                self.cell[c].push(blpos);
                self.cell_count += 1;
                newcell = true;
            }
            if (self.abovecellpos != 0) {
                self.abovecellpos += 1;
                if (self.abovecellpos > self.abovecellpos1) {
                    self.abovecellpos = 0;
                    self.abovecellpos1 = 0;
                }
            }
            if (newcell and self.cells_count(1) > 0) {
                self.abovecellpos = @as(i32, @bitCast(self.cell_pos(self.cell_count - 1, 1)));
                self.abovecellpos1 = @as(i32, @bitCast(self.cell_pos(self.cell_count, 1)));
            }
        }
        if (self.nl_char == WIKIHEADER) {
            if ((b2 & 0xffff) == (@as(u32, WIKIHEADER) + @as(u32, @intCast(LF)) * 256)) {
                self.cells += 1;
                self.cells &= 3;
                const c = @as(usize, @intCast(self.cells));
                self.cell[c].reset();
                self.cell[c].push(blpos);
                self.cell_count = 0;
                self.abovecellpos = 0;
                self.abovecellpos1 = 0;
            } else {
                var newcell = false;
                if ((b2 & 0xff) == @as(u32, WIKIHEADER)) {
                    const c = @as(usize, @intCast(self.cells));
                    self.cell[c].push(blpos);
                    self.cell_count += 1;
                    newcell = true;
                }
                if (self.abovecellpos != 0) {
                    self.abovecellpos += 1;
                    if (self.abovecellpos > self.abovecellpos1) {
                        self.abovecellpos = 0;
                        self.abovecellpos1 = 0;
                    }
                }
                if (newcell and self.cells_count(1) > 0) {
                    self.abovecellpos = @as(i32, @bitCast(self.cell_pos(self.cell_count - 1, 1)));
                    self.abovecellpos1 = @as(i32, @bitCast(self.cell_pos(self.cell_count, 1)));
                }
            }
        }
    }
};

test "column_context_matches_cpp_oracle" {
    const a = std.testing.allocator;
    var cc = try ColumnContext.new(a);
    defer {
        for (&cc.col) |*c| a.free(c.bytes.data);
        for (&cc.cell) |*c| a.free(c.data);
    }
    cc.init(31);
    cc.is_pre = false;
    const alpha = [16]u8{ 80, 82, 81, 45, 78, 76, 10, 91, 93, 64, 96, 32, 'a', 'b', 74, 77 };
    var cs: u64 = 0;
    var r: u32 = 0x12345678;
    var b2: u32 = 0;
    var blpos: u32 = 0;
    var iter: usize = 0;
    while (iter < 8000) : (iter += 1) {
        r = r *% 1664525 +% 1013904223;
        const byte = @as(i32, alpha[@as(usize, (r >> 16) & 15)]);
        b2 = ((b2 << 8) +% @as(u32, @bitCast(byte))) & 0xffffff;
        cc.update(byte, b2, blpos);
        blpos = blpos +% 1;
        var k: i32 = 0;
        while (k < 4) : (k += 1) {
            cs = cs *% 1000003 +% @as(u64, cc.lastfc(k));
        }
        cs = cs *% 1000003 +% @as(u64, cc.nl_char);
        cs = cs *% 1000003 +% @as(u64, @intFromBool(cc.is_temp));
        cs = cs *% 1000003 +% @as(u64, @intFromBool(cc.is_new_line()));
        cs = cs *% 1000003 +% @as(u64, @as(u32, @bitCast(cc.rows)));
        cs = cs *% 1000003 +% @as(u64, @as(u32, @bitCast(cc.cells)));
        cs = cs *% 1000003 +% @as(u64, @as(u32, @bitCast(cc.cell_count)));
        cs = cs *% 1000003 +% @as(u64, @as(u32, @bitCast(cc.abovecellpos)));
        cs = cs *% 1000003 +% @as(u64, @as(u32, @bitCast(cc.abovecellpos1)));
        k = 0;
        while (k < 4) : (k += 1) {
            cs = cs *% 1000003 +% @as(u64, @as(u32, @bitCast(cc.collen(k, 0))));
        }
        cs = cs *% 1000003 +% @as(u64, cc.colb(1, 0, 0));
        cs = cs *% 1000003 +% @as(u64, cc.nlpos(0));
        cs = cs *% 1000003 +% @as(u64, cc.nlpos(1));
        cs = cs *% 1000003 +% @as(u64, @as(u32, @bitCast(cc.cells_count(1))));
        cs = cs *% 1000003 +% @as(u64, cc.cell_pos(0, 1));
    }
    try std.testing.expectEqual(@as(u64, 0xe886a8d5c7f011ac), cs);
}
