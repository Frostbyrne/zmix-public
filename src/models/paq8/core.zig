//! paq8 core infrastructure, ported faithfully from
//! reference/cmix-src/models/paq8.cpp (the cmix PAQ8 model).
//!
//! This file provides the shared predictor state and numeric foundation that
//! every paq8 model builds on:
//!   * shared globals: pos, y, c0, c4, bpos, blpos, grp0, buf, dt, ModelStats
//!   * prediction export: model_predictions / addPrediction / resetPredictions
//!   * Array, Random (rnd), Buf
//!   * State_table + nex, squash/stretch/ilog/llog tables
//!   * dot_product / train (SSE2-equivalent, expressed with @Vector)
//!   * Mixer (hierarchical logistic mixer with context-selected weights)
//!   * APM1, StateMap, StateMap32, APM
//!
//! Faithfulness notes:
//!   * C `unsigned` wrap is expressed with Zig `+%`/`-%`/`*%`.
//!   * U16/U32 stores that truncate in C are masked (`& 0xffff`) in Zig.
//!   * squash/stretch/ilog/dt tables match paq8.cpp exactly (NOT the fxcm ones,
//!     which use different formulas).
//!   * Call `init` once before using any table-backed function.
const std = @import("std");

// ============================ shared predictor state =======================
// paq8.cpp file-scope globals. Model files read/write these directly, exactly
// as the C++ code treats the globals.
pub var pos: i32 = 0; // paq8 `int pos`
pub var y: i32 = 0; // last bit modelled (0/1)
pub var c0: i32 = 1; // partial byte, leading 1 bit
pub var c4: u32 = 0; // last 4 whole bytes
pub var bpos: i32 = 0; // bit position within current byte (0..7)
pub var blpos: i32 = 0; // byte position within current block
pub var grp0: u8 = 0; // quantized partial byte as ASCII group

// paq8 predictor-level shared globals (used across updateand many models).
pub var b2: u32 = 0;
pub var b3: u32 = 0;
pub var w4: u32 = 0;
pub var w5: u32 = 0;
pub var f4: u32 = 0;
pub var tt: u32 = 0;
pub var col: u32 = 0;
pub var x4: u32 = 0;
pub var words: u32 = 0; // shared word bit-history (wordModel + update)
pub var last_prediction: i32 = 2048;

// paq8 memory level (cmix: PAQ8(11) -> level=11). MEM= 0x10000<<level.
pub var level: i32 = 11;
pub fn MEM() u64 {
    return @as(u64, 0x10000) << @intCast(level);
}

// preprocessor::Filetype values (Filetype is kept as i32 for ModelStats compat).
pub const FT_DEFAULT: i32 = 0;
pub const FT_HDR: i32 = 1;
pub const FT_JPEG: i32 = 2;
pub const FT_EXE: i32 = 3;
pub const FT_TEXT: i32 = 4;
pub const FT_IMAGE1: i32 = 5;
pub const FT_IMAGE4: i32 = 6;
pub const FT_IMAGE8: i32 = 7;
pub const FT_IMAGE8GRAY: i32 = 8;
pub const FT_IMAGE24: i32 = 9;
pub const FT_IMAGE32: i32 = 10;
pub const FT_AUDIO: i32 = 11;
pub fn HasInfo(ft: i32) bool {
    return ft == FT_TEXT or ft == FT_IMAGE1 or ft == FT_IMAGE4 or ft == FT_IMAGE8 or ft == FT_IMAGE8GRAY or ft == FT_IMAGE24 or ft == FT_IMAGE32;
}

pub const WRT_mpw = [16]u32{ 4, 4, 3, 2, 2, 2, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0 };
pub const WRT_mtt = [16]u32{ 0, 0, 1, 2, 3, 4, 5, 5, 6, 6, 6, 6, 7, 7, 7, 7 };

pub const AsciiGroupC0 = [254]u8{
    0,  10,
    0,  1,  10, 10,
    0,  4,  2,  3,  10, 10, 10, 10,
    0,  0,  5,  4,  2,  2,  3,  3,  10, 10, 10, 10, 10, 10, 10, 10,
    0,  0,  0,  0,  5,  5,  9,  4,  2,  2,  2,  2,  3,  3,  3,  3,  10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10,
    0,  0,  0,  0,  0,  0,  0,  0,  5,  8,  8,  5,  9,  9,  6,  5,  2,  2,  2,  2,  2,  2,  2,  8,  3,  3,  3,  3,  3,  3,  3,  8,  10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10,
    0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  7,  8,  8,  8,  8,  8,  5,  5,  9,  9,  9,  9,  9,  7,  8,  5,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  8,  8,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,  8,  8,  10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10,
};

/// Reset the scalar shared state to the paq8 initial values. Large tables
/// (buf, dt, squash/stretch/ilog) are handled by `init`; this only resets the
/// per-run scalars so a fresh predictor does not inherit residual state.
pub fn resetState() void {
    pos = 0;
    y = 0;
    c0 = 1;
    c4 = 0;
    bpos = 0;
    blpos = 0;
    grp0 = 0;
    b2 = 0;
    b3 = 0;
    w4 = 0;
    w5 = 0;
    f4 = 0;
    tt = 0;
    col = 0;
    x4 = 0;
    words = 0;
    last_prediction = 2048;
}

// ================================ ModelStats ===============================
// Placeholder for preprocessor::Filetype (the concrete enum lives in the
// preprocessor; the infrastructure never inspects its values).
pub const Filetype = i32;

pub const ModelStats = struct {
    Type: Filetype = 0,
    Misses: u64 = 0,
    Match: struct {
        length: u32 = 0, // used by SSE stage
        expectedByte: u8 = 0, // used by SSE stage
    } = .{},
    Image: struct {
        pixels: struct {
            WW: u8 = 0,
            W: u8 = 0,
            NN: u8 = 0,
            N: u8 = 0,
            Wp1: u8 = 0,
            Np1: u8 = 0,
        } = .{},
        plane: u8 = 0,
        ctx: u8 = 0,
    } = .{},
    XML: u32 = 0,
    x86_64: u32 = 0,
    Record: u32 = 0,
    // The C++ Text struct uses bitfields; state/lastPunct/wordLength/boolmask
    // are marked "unused". Only firstLetter and mask are read by the SSE stage.
    // We expose them as plain bytes (bit packing is irrelevant since the packed
    // members are unused).
    Text: struct {
        state: u8 = 0, // :3 unused
        lastPunct: u8 = 0, // :5 unused
        wordLength: u8 = 0, // :4 unused
        boolmask: u8 = 0, // :4 unused
        firstLetter: u8 = 0, // used by SSE stage
        mask: u8 = 0, // used by SSE stage
    } = .{},
};

// ============================= prediction export ===========================
pub const NUM_INPUTS: usize = 1552;
pub const NUM_SETS: usize = 28;
pub const PREDICTIONS_LEN: usize = NUM_INPUTS + NUM_SETS + 11; // 1591
pub const conversion_factor: f32 = 1.0 / 4095.0;

// std::valarray<float> model_predictions(0.5, NUM_INPUTS+NUM_SETS+11)
pub var model_predictions: [PREDICTIONS_LEN]f32 = .{0.5} ** PREDICTIONS_LEN;
pub var prediction_index: usize = 0;

pub inline fn addPrediction(x: i32) void {
    model_predictions[prediction_index] = @as(f32, @floatFromInt(x)) * conversion_factor;
    prediction_index += 1;
}

pub inline fn resetPredictions() void {
    prediction_index = 0;
}

// ================================== Array ==================================
// Faithful port of paq8's `template <class T, int ALIGN=0> class Array`: a
// growable, zero-initialised (calloc), optionally over-aligned buffer.
pub fn Array(comptime T: type, comptime ALIGN: usize) type {
    const A: usize = if (ALIGN == 0) @alignOf(T) else ALIGN;
    const Alt = std.mem.Alignment.fromByteUnits(A);
    return struct {
        const Self = @This();
        n: u32 = 0,
        reserved: u32 = 0,
        data: []align(A) T = &.{},
        alloc: std.mem.Allocator = undefined,

        pub fn init(a: std.mem.Allocator) Self {
            return .{ .alloc = a };
        }

        /// Construct with `i` zero-filled elements (like `Array<T>(i)`).
        pub fn initSize(a: std.mem.Allocator, i: u32) Self {
            var s = Self{ .alloc = a };
            s.create(i);
            return s;
        }

        // C++ `Array<T,ALIGN>::create` calloc's `i*sizeof(T) + ALIGN` bytes: `i`
        // logical elements plus ALIGN bytes of ZEROED slack past the end. Some
        // models deliberately over-read past `data[i-1]` and rely on hitting
        // those deterministic zeros (e.g. im4bitModel's nibble-tree cp[i]+=j).
        // Over-allocate the same slack so those reads stay in-bounds and zero.
        const slack: u32 = (A + @sizeOf(T) - 1) / @sizeOf(T);

        fn create(self: *Self, i: u32) void {
            self.n = i;
            self.reserved = i;
            if (i == 0) {
                self.data = &.{};
                return;
            }
            self.data = self.alloc.alignedAlloc(T, Alt, i + slack) catch unreachable;
            @memset(self.data, std.mem.zeroes(T));
        }

        pub fn resize(self: *Self, i: u32) void {
            if (i <= self.reserved) {
                self.n = i;
                return;
            }
            const savedata = self.data;
            const saven = self.n;
            self.create(i);
            if (savedata.len > 0) {
                const m = @min(i, saven);
                @memcpy(self.data[0..m], savedata[0..m]);
                self.alloc.free(savedata);
            }
        }

        pub fn pushBack(self: *Self, x: T) void {
            if (self.n == self.reserved) {
                const saven = self.n;
                self.resize(@max(1, self.n *% 2));
                self.n = saven;
            }
            self.data[self.n] = x;
            self.n += 1;
        }

        pub fn popBack(self: *Self) void {
            if (self.n > 0) self.n -= 1;
        }

        pub inline fn size(self: *const Self) u32 {
            return self.n;
        }

        pub inline fn at(self: *Self, i: u32) *T {
            return &self.data[i];
        }

        pub inline fn get(self: *const Self, i: u32) T {
            return self.data[i];
        }

        pub fn deinit(self: *Self) void {
            if (self.data.len > 0) self.alloc.free(self.data);
            self.data = &.{};
            self.n = 0;
            self.reserved = 0;
        }
    };
}

// ================================== Random =================================
pub const Random = struct {
    table: [64]u32 = undefined,
    i: u32 = 0,

    pub fn init(self: *Random) void {
        self.table[0] = 123456789;
        self.table[1] = 987654321;
        var j: usize = 0;
        while (j < 62) : (j += 1) {
            self.table[j + 2] = self.table[j + 1] *% 11 +% (self.table[j] *% 23) / 16;
        }
        self.i = 0;
    }

    // U32 operator
    pub inline fn next(self: *Random) u32 {
        self.i +%= 1;
        const v = self.table[(self.i -% 24) & 63] ^ self.table[(self.i -% 55) & 63];
        self.table[self.i & 63] = v;
        return v;
    }
};

pub var rnd: Random = .{};

// ==================================== Buf ==================================
// Ring buffer. buf[i] indexes absolute position (masked); buf(i) reads the byte
// `i` positions before the global `pos`.
pub const Buf = struct {
    b: Array(u8, 0) = .{},
    inited: bool = false,

    pub fn setsize(self: *Buf, a: std.mem.Allocator, i: u32) void {
        // Always (re)create the backing array from the CURRENT allocator. The old
        // `inited`-guarded reuse kept a dangling pointer when buf was carried
        // across arenas/tests (its previous storage already freed) -> segfault.
        self.b = Array(u8, 0).init(a);
        self.inited = true;
        if (i == 0) return;
        self.b.resize(i);
    }

    // U8& operator[](U32 i)
    pub inline fn at(self: *Buf, i: u32) *u8 {
        return self.b.at(i & (self.b.size() - 1));
    }

    // int operator(U32 i) const
    pub inline fn get(self: *Buf, i: u32) i32 {
        const idx = (@as(u32, @bitCast(pos)) -% i) & (self.b.size() - 1);
        return @as(i32, self.b.get(idx));
    }

    pub inline fn size(self: *const Buf) u32 {
        return self.b.size();
    }

    pub fn deinit(self: *Buf) void {
        if (self.inited) self.b.deinit();
        self.inited = false;
    }
};

pub var buf: Buf = .{};

// =============================== State table ===============================
// paq8 `static const U8 State_table[256][4]`. nex(state,sel) = table[state][sel].
const state_table: [256][4]u8 = .{
    .{ 1, 2, 0, 0 },
    .{ 3, 5, 1, 0 },
    .{ 4, 6, 0, 1 },
    .{ 7, 10, 2, 0 },
    .{ 8, 12, 1, 1 },
    .{ 9, 13, 1, 1 },
    .{ 11, 14, 0, 2 },
    .{ 15, 19, 3, 0 },
    .{ 16, 23, 2, 1 },
    .{ 17, 24, 2, 1 },
    .{ 18, 25, 2, 1 },
    .{ 20, 27, 1, 2 },
    .{ 21, 28, 1, 2 },
    .{ 22, 29, 1, 2 },
    .{ 26, 30, 0, 3 },
    .{ 31, 33, 4, 0 },
    .{ 32, 35, 3, 1 },
    .{ 32, 35, 3, 1 },
    .{ 32, 35, 3, 1 },
    .{ 32, 35, 3, 1 },
    .{ 34, 37, 2, 2 },
    .{ 34, 37, 2, 2 },
    .{ 34, 37, 2, 2 },
    .{ 34, 37, 2, 2 },
    .{ 34, 37, 2, 2 },
    .{ 34, 37, 2, 2 },
    .{ 36, 39, 1, 3 },
    .{ 36, 39, 1, 3 },
    .{ 36, 39, 1, 3 },
    .{ 36, 39, 1, 3 },
    .{ 38, 40, 0, 4 },
    .{ 41, 43, 5, 0 },
    .{ 42, 45, 4, 1 },
    .{ 42, 45, 4, 1 },
    .{ 44, 47, 3, 2 },
    .{ 44, 47, 3, 2 },
    .{ 46, 49, 2, 3 },
    .{ 46, 49, 2, 3 },
    .{ 48, 51, 1, 4 },
    .{ 48, 51, 1, 4 },
    .{ 50, 52, 0, 5 },
    .{ 53, 43, 6, 0 },
    .{ 54, 57, 5, 1 },
    .{ 54, 57, 5, 1 },
    .{ 56, 59, 4, 2 },
    .{ 56, 59, 4, 2 },
    .{ 58, 61, 3, 3 },
    .{ 58, 61, 3, 3 },
    .{ 60, 63, 2, 4 },
    .{ 60, 63, 2, 4 },
    .{ 62, 65, 1, 5 },
    .{ 62, 65, 1, 5 },
    .{ 50, 66, 0, 6 },
    .{ 67, 55, 7, 0 },
    .{ 68, 57, 6, 1 },
    .{ 68, 57, 6, 1 },
    .{ 70, 73, 5, 2 },
    .{ 70, 73, 5, 2 },
    .{ 72, 75, 4, 3 },
    .{ 72, 75, 4, 3 },
    .{ 74, 77, 3, 4 },
    .{ 74, 77, 3, 4 },
    .{ 76, 79, 2, 5 },
    .{ 76, 79, 2, 5 },
    .{ 62, 81, 1, 6 },
    .{ 62, 81, 1, 6 },
    .{ 64, 82, 0, 7 },
    .{ 83, 69, 8, 0 },
    .{ 84, 71, 7, 1 },
    .{ 84, 71, 7, 1 },
    .{ 86, 73, 6, 2 },
    .{ 86, 73, 6, 2 },
    .{ 44, 59, 5, 3 },
    .{ 44, 59, 5, 3 },
    .{ 58, 61, 4, 4 },
    .{ 58, 61, 4, 4 },
    .{ 60, 49, 3, 5 },
    .{ 60, 49, 3, 5 },
    .{ 76, 89, 2, 6 },
    .{ 76, 89, 2, 6 },
    .{ 78, 91, 1, 7 },
    .{ 78, 91, 1, 7 },
    .{ 80, 92, 0, 8 },
    .{ 93, 69, 9, 0 },
    .{ 94, 87, 8, 1 },
    .{ 94, 87, 8, 1 },
    .{ 96, 45, 7, 2 },
    .{ 96, 45, 7, 2 },
    .{ 48, 99, 2, 7 },
    .{ 48, 99, 2, 7 },
    .{ 88, 101, 1, 8 },
    .{ 88, 101, 1, 8 },
    .{ 80, 102, 0, 9 },
    .{ 103, 69, 10, 0 },
    .{ 104, 87, 9, 1 },
    .{ 104, 87, 9, 1 },
    .{ 106, 57, 8, 2 },
    .{ 106, 57, 8, 2 },
    .{ 62, 109, 2, 8 },
    .{ 62, 109, 2, 8 },
    .{ 88, 111, 1, 9 },
    .{ 88, 111, 1, 9 },
    .{ 80, 112, 0, 10 },
    .{ 113, 85, 11, 0 },
    .{ 114, 87, 10, 1 },
    .{ 114, 87, 10, 1 },
    .{ 116, 57, 9, 2 },
    .{ 116, 57, 9, 2 },
    .{ 62, 119, 2, 9 },
    .{ 62, 119, 2, 9 },
    .{ 88, 121, 1, 10 },
    .{ 88, 121, 1, 10 },
    .{ 90, 122, 0, 11 },
    .{ 123, 85, 12, 0 },
    .{ 124, 97, 11, 1 },
    .{ 124, 97, 11, 1 },
    .{ 126, 57, 10, 2 },
    .{ 126, 57, 10, 2 },
    .{ 62, 129, 2, 10 },
    .{ 62, 129, 2, 10 },
    .{ 98, 131, 1, 11 },
    .{ 98, 131, 1, 11 },
    .{ 90, 132, 0, 12 },
    .{ 133, 85, 13, 0 },
    .{ 134, 97, 12, 1 },
    .{ 134, 97, 12, 1 },
    .{ 136, 57, 11, 2 },
    .{ 136, 57, 11, 2 },
    .{ 62, 139, 2, 11 },
    .{ 62, 139, 2, 11 },
    .{ 98, 141, 1, 12 },
    .{ 98, 141, 1, 12 },
    .{ 90, 142, 0, 13 },
    .{ 143, 95, 14, 0 },
    .{ 144, 97, 13, 1 },
    .{ 144, 97, 13, 1 },
    .{ 68, 57, 12, 2 },
    .{ 68, 57, 12, 2 },
    .{ 62, 81, 2, 12 },
    .{ 62, 81, 2, 12 },
    .{ 98, 147, 1, 13 },
    .{ 98, 147, 1, 13 },
    .{ 100, 148, 0, 14 },
    .{ 149, 95, 15, 0 },
    .{ 150, 107, 14, 1 },
    .{ 150, 107, 14, 1 },
    .{ 108, 151, 1, 14 },
    .{ 108, 151, 1, 14 },
    .{ 100, 152, 0, 15 },
    .{ 153, 95, 16, 0 },
    .{ 154, 107, 15, 1 },
    .{ 108, 155, 1, 15 },
    .{ 100, 156, 0, 16 },
    .{ 157, 95, 17, 0 },
    .{ 158, 107, 16, 1 },
    .{ 108, 159, 1, 16 },
    .{ 100, 160, 0, 17 },
    .{ 161, 105, 18, 0 },
    .{ 162, 107, 17, 1 },
    .{ 108, 163, 1, 17 },
    .{ 110, 164, 0, 18 },
    .{ 165, 105, 19, 0 },
    .{ 166, 117, 18, 1 },
    .{ 118, 167, 1, 18 },
    .{ 110, 168, 0, 19 },
    .{ 169, 105, 20, 0 },
    .{ 170, 117, 19, 1 },
    .{ 118, 171, 1, 19 },
    .{ 110, 172, 0, 20 },
    .{ 173, 105, 21, 0 },
    .{ 174, 117, 20, 1 },
    .{ 118, 175, 1, 20 },
    .{ 110, 176, 0, 21 },
    .{ 177, 105, 22, 0 },
    .{ 178, 117, 21, 1 },
    .{ 118, 179, 1, 21 },
    .{ 110, 180, 0, 22 },
    .{ 181, 115, 23, 0 },
    .{ 182, 117, 22, 1 },
    .{ 118, 183, 1, 22 },
    .{ 120, 184, 0, 23 },
    .{ 185, 115, 24, 0 },
    .{ 186, 127, 23, 1 },
    .{ 128, 187, 1, 23 },
    .{ 120, 188, 0, 24 },
    .{ 189, 115, 25, 0 },
    .{ 190, 127, 24, 1 },
    .{ 128, 191, 1, 24 },
    .{ 120, 192, 0, 25 },
    .{ 193, 115, 26, 0 },
    .{ 194, 127, 25, 1 },
    .{ 128, 195, 1, 25 },
    .{ 120, 196, 0, 26 },
    .{ 197, 115, 27, 0 },
    .{ 198, 127, 26, 1 },
    .{ 128, 199, 1, 26 },
    .{ 120, 200, 0, 27 },
    .{ 201, 115, 28, 0 },
    .{ 202, 127, 27, 1 },
    .{ 128, 203, 1, 27 },
    .{ 120, 204, 0, 28 },
    .{ 205, 115, 29, 0 },
    .{ 206, 127, 28, 1 },
    .{ 128, 207, 1, 28 },
    .{ 120, 208, 0, 29 },
    .{ 209, 125, 30, 0 },
    .{ 210, 127, 29, 1 },
    .{ 128, 211, 1, 29 },
    .{ 130, 212, 0, 30 },
    .{ 213, 125, 31, 0 },
    .{ 214, 137, 30, 1 },
    .{ 138, 215, 1, 30 },
    .{ 130, 216, 0, 31 },
    .{ 217, 125, 32, 0 },
    .{ 218, 137, 31, 1 },
    .{ 138, 219, 1, 31 },
    .{ 130, 220, 0, 32 },
    .{ 221, 125, 33, 0 },
    .{ 222, 137, 32, 1 },
    .{ 138, 223, 1, 32 },
    .{ 130, 224, 0, 33 },
    .{ 225, 125, 34, 0 },
    .{ 226, 137, 33, 1 },
    .{ 138, 227, 1, 33 },
    .{ 130, 228, 0, 34 },
    .{ 229, 125, 35, 0 },
    .{ 230, 137, 34, 1 },
    .{ 138, 231, 1, 34 },
    .{ 130, 232, 0, 35 },
    .{ 233, 125, 36, 0 },
    .{ 234, 137, 35, 1 },
    .{ 138, 235, 1, 35 },
    .{ 130, 236, 0, 36 },
    .{ 237, 125, 37, 0 },
    .{ 238, 137, 36, 1 },
    .{ 138, 239, 1, 36 },
    .{ 130, 240, 0, 37 },
    .{ 241, 125, 38, 0 },
    .{ 242, 137, 37, 1 },
    .{ 138, 243, 1, 37 },
    .{ 130, 244, 0, 38 },
    .{ 245, 135, 39, 0 },
    .{ 246, 137, 38, 1 },
    .{ 138, 247, 1, 38 },
    .{ 140, 248, 0, 39 },
    .{ 249, 135, 40, 0 },
    .{ 250, 69, 39, 1 },
    .{ 80, 251, 1, 39 },
    .{ 140, 252, 0, 40 },
    .{ 249, 135, 41, 0 },
    .{ 250, 69, 40, 1 },
    .{ 80, 251, 1, 40 },
    .{ 140, 252, 0, 41 },
    .{ 0, 0, 0, 0 },
    .{ 0, 0, 0, 0 },
    .{ 0, 0, 0, 0 },
};

/// nex(state,sel) = State_table[state][sel].
pub inline fn nex(state: u32, sel: u32) u8 {
    return state_table[state][sel];
}

// ============================ squash/stretch/ilog ==========================
pub var sqt: [4096]u16 = undefined; // squash lookup, indexed p+2048
pub var strt: [4096]i16 = undefined; // stretch lookup
pub var ilogt: [65536]u8 = undefined; // ilog lookup (U16 arg)
pub var dt: [1024]i32 = undefined; // i -> 16384/(2i+3)
var tables_ready = false;

pub inline fn squash(p: i32) i32 {
    if (p > 2047) return 4095;
    if (p < -2047) return 0;
    return @as(i32, sqt[@intCast(p + 2048)]);
}

pub inline fn stretch(p: i32) i32 {
    return @as(i32, strt[@intCast(p)]);
}

pub inline fn ilog(x: u32) i32 {
    return @as(i32, ilogt[@intCast(x & 0xffff)]);
}

pub inline fn llog(x: u32) i32 {
    if (x >= 0x1000000) return 256 + ilog(x >> 16);
    if (x >= 0x10000) return 128 + ilog(x >> 8);
    return ilog(x);
}

/// Build all lookup tables and seed rnd. Idempotent.
pub fn init() void {
    if (tables_ready) return;

    // squash
    const ts = [33]i32{
        1, 2, 3, 6, 10, 16, 27, 45, 73, 120, 194, 310, 488, 747, 1101,
        1546, 2047, 2549, 2994, 3348, 3607, 3785, 3901, 3975, 4022,
        4050, 4068, 4079, 4085, 4089, 4092, 4093, 4094,
    };
    @memset(&sqt, 0);
    var i: i32 = -2047;
    while (i <= 2047) : (i += 1) {
        const w = i & 127;
        const d = (i >> 7) + 16;
        const v = (ts[@intCast(d)] * (128 - w) + ts[@intCast(d + 1)] * w + 64) >> 7;
        sqt[@intCast(i + 2048)] = @intCast(v);
    }

    // stretch (inverse of squash)
    @memset(&strt, 0);
    var pi: i32 = 0;
    var x: i32 = -2047;
    while (x <= 2047) : (x += 1) {
        const iv = squash(x);
        var j: i32 = pi;
        while (j <= iv) : (j += 1) strt[@intCast(j)] = @intCast(x);
        pi = iv + 1;
    }
    strt[4095] = 2047;

    // ilog
    ilogt[0] = 0;
    ilogt[1] = 0;
    var xx: u32 = 14155776;
    var k: u32 = 2;
    while (k < 65536) : (k += 1) {
        xx +%= 774541002 / (k * 2 - 1);
        ilogt[k] = @truncate(xx >> 24);
    }

    // dt: i -> 16384/(2i+3)
    var t: usize = 0;
    while (t < 1024) : (t += 1) {
        dt[t] = @divTrunc(@as(i32, 16384), @as(i32, @intCast(t * 2 + 3)));
    }

    rnd.init();
    tables_ready = true;
}

// ============================ dot_product / train ==========================
// SSE2-equivalent (paq8 uses `_mm_madd_epi16`/`_mm_srai_epi32`, and saturating
// add / mulhi / srai for train). Processes 8 int16 lanes per iteration to match
// the __m128i width (nx is always padded to a multiple of 8 by the Mixer).
const V8i16 = @Vector(8, i16);
const V8i32 = @Vector(8, i32);
const V4i32 = @Vector(4, i32);

pub fn dot_product(t: [*]const i16, w: [*]const i16, n_in: usize) i32 {
    var n = n_in;
    var sum: V4i32 = @splat(0);
    while (n >= 8) {
        n -= 8;
        const tv: V8i16 = t[n..][0..8].*;
        const wv: V8i16 = w[n..][0..8].*;
        const prod: V8i32 = @as(V8i32, tv) * @as(V8i32, wv);
        var pair: V4i32 = undefined;
        // _mm_madd_epi16 then _mm_srai_epi32(.,8)
        inline for (0..4) |c| pair[c] = (prod[2 * c] +% prod[2 * c + 1]) >> 8;
        sum +%= pair;
    }
    // horizontal add (wrapping, like the accumulating __m128i)
    var acc: i32 = 0;
    inline for (0..4) |c| acc +%= sum[c];
    return acc;
}

pub fn train(t: [*]const i16, w: [*]i16, n_in: usize, e: i32) void {
    if (e == 0) return;
    var n = n_in;
    const err_i32: V8i32 = @splat(@as(i32, @as(i16, @truncate(e))));
    const one: V8i16 = @splat(1);
    while (n >= 8) {
        n -= 8;
        const tv: V8i16 = t[n..][0..8].*;
        var tmp: V8i16 = tv +| tv; // _mm_adds_epi16
        // _mm_mulhi_epi16: (tmp*err) >> 16
        const prod: V8i32 = @as(V8i32, tmp) * err_i32;
        tmp = @intCast(prod >> @as(V8i32, @splat(16)));
        tmp +|= one; // _mm_adds_epi16
        tmp = tmp >> @as(V8i16, @splat(1)); // _mm_srai_epi16
        const wv: V8i16 = w[n..][0..8].*;
        tmp +|= wv; // _mm_adds_epi16
        w[n..][0..8].* = tmp;
    }
}

// ================================== Mixer ==================================
// Faithful port of paq8's hierarchical Mixer with context-selected weight
// vectors (an unordered_map<context, Array<short>>). When S>1 a child mixer
// (`mp`, S inputs, 1 context, init weight 0x7fff) combines the per-context
// sub-predictions.
pub const Mixer = struct {
    N: usize, // (n+7)&-8, weight/tx length
    S: usize, // number of contexts
    init_w: i16, // initial weight value for lazily-created weight vectors
    tx: []align(16) i16, // input vector
    wx: std.AutoHashMap(u32, []align(16) i16), // context -> weights
    cxt: []i32, // selected contexts this round
    ncxt: usize,
    base: i32,
    nx: usize,
    pr: []i32,
    mp: ?*Mixer,
    alloc: std.mem.Allocator,

    pub fn init(a: std.mem.Allocator, n: usize, m: usize, s: i32, w: i32) *Mixer {
        _ = m; // paq8's Mixer ctor ignores `m`
        const self = a.create(Mixer) catch unreachable;
        const NN = (n + 7) & ~@as(usize, 7);
        const S: usize = @intCast(s);
        self.* = .{
            .N = NN,
            .S = S,
            .init_w = @intCast(w),
            .tx = a.alignedAlloc(i16, .@"16", NN) catch unreachable,
            .wx = std.AutoHashMap(u32, []align(16) i16).init(a),
            .cxt = a.alloc(i32, S) catch unreachable,
            .ncxt = 0,
            .base = 0,
            .nx = 0,
            .pr = a.alloc(i32, S) catch unreachable,
            .mp = null,
            .alloc = a,
        };
        @memset(self.tx, 0);
        for (self.pr) |*pv| pv.* = 2048;
        if (s > 1) self.mp = Mixer.init(a, S, 1, 1, 0x7fff);
        return self;
    }

    fn getWeights(self: *Mixer, ctx: u32) []align(16) i16 {
        const gop = self.wx.getOrPut(ctx) catch unreachable;
        if (!gop.found_existing) {
            const wts = self.alloc.alignedAlloc(i16, .@"16", self.N) catch unreachable;
            for (wts) |*ww| ww.* = self.init_w;
            gop.value_ptr.* = wts;
        }
        return gop.value_ptr.*;
    }

    pub fn update(self: *Mixer) void {
        var i: usize = 0;
        while (i < self.ncxt) : (i += 1) {
            const err = ((y << 12) - self.pr[i]) *% 7;
            const wts = self.getWeights(@bitCast(self.cxt[i]));
            train(self.tx.ptr, wts.ptr, self.nx, err);
        }
        self.nx = 0;
        self.base = 0;
        self.ncxt = 0;
    }

    pub fn add(self: *Mixer, x: i32) void {
        addPrediction(squash(x));
        self.tx[self.nx] = @truncate(x);
        self.nx += 1;
    }

    pub fn set(self: *Mixer, cx: i32, range: i32) void {
        self.cxt[self.ncxt] = self.base +% cx;
        self.ncxt += 1;
        self.base +%= range;
    }

    pub fn p(self: *Mixer) i32 {
        while (self.nx & 7 != 0) {
            self.tx[self.nx] = 0;
            self.nx += 1;
        }
        if (self.mp) |mpp| {
            mpp.update();
            var i: usize = 0;
            while (i < self.ncxt) : (i += 1) {
                const wts = self.getWeights(@bitCast(self.cxt[i]));
                self.pr[i] = squash((dot_product(self.tx.ptr, wts.ptr, self.nx) *% 9) >> 9);
                mpp.add(stretch(self.pr[i]));
            }
            mpp.set(0, 1);
            return mpp.p();
        } else {
            const wts = self.getWeights(0);
            const z = dot_product(self.tx.ptr, wts.ptr, self.nx);
            self.base = squash((z *% 16) >> 13);
            self.pr[0] = squash(z >> 9);
            return self.pr[0];
        }
    }

    pub fn deinit(self: *Mixer) void {
        if (self.mp) |mpp| mpp.deinit();
        var it = self.wx.valueIterator();
        while (it.next()) |v| self.alloc.free(v.*);
        self.wx.deinit();
        self.alloc.free(self.tx);
        self.alloc.free(self.cxt);
        self.alloc.free(self.pr);
        self.alloc.destroy(self);
    }
};

// =================================== APM1 ==================================
// paq8's U16 adaptive probability map.
pub const APM1 = struct {
    index: usize = 0,
    N: usize,
    t: []u16,
    alloc: std.mem.Allocator,

    pub fn init(a: std.mem.Allocator, n: usize) APM1 {
        var self = APM1{ .N = n, .t = a.alloc(u16, n * 33) catch unreachable, .alloc = a };
        var i: usize = 0;
        while (i < n) : (i += 1) {
            var j: usize = 0;
            while (j < 33) : (j += 1) {
                self.t[i * 33 + j] = if (i == 0)
                    @intCast(squash((@as(i32, @intCast(j)) - 16) * 128) * 16)
                else
                    self.t[j];
            }
        }
        return self;
    }

    pub fn p(self: *APM1, pr_in: i32, cxt: i32, rate: u5) i32 {
        const pr = stretch(pr_in);
        const g = (y << 16) + (y << rate) - y - y;
        {
            const cur: i32 = self.t[self.index];
            self.t[self.index] = @intCast((cur + ((g - cur) >> rate)) & 0xffff);
        }
        {
            const cur: i32 = self.t[self.index + 1];
            self.t[self.index + 1] = @intCast((cur + ((g - cur) >> rate)) & 0xffff);
        }
        const w = pr & 127;
        self.index = @intCast(((pr + 2048) >> 7) + cxt * 33);
        return (@as(i32, self.t[self.index]) * (128 - w) + @as(i32, self.t[self.index + 1]) * w) >> 11;
    }

    pub fn deinit(self: *APM1) void {
        self.alloc.free(self.t);
    }
};

// ================================= StateMap ================================
// paq8's U16 StateMap (used by ContextMap).
pub const StateMap = struct {
    cxt: usize = 0,
    t: []u16,
    alloc: std.mem.Allocator,

    pub fn init(a: std.mem.Allocator) StateMap {
        var self = StateMap{ .t = a.alloc(u16, 256) catch unreachable, .alloc = a };
        var i: usize = 0;
        while (i < 256) : (i += 1) {
            var n0: i32 = nex(@intCast(i), 2);
            var n1: i32 = nex(@intCast(i), 3);
            if (n0 == 0) n1 *= 64;
            if (n1 == 0) n0 *= 64;
            self.t[i] = @intCast(@divTrunc(65536 * (n1 + 1), n0 + n1 + 2));
        }
        return self;
    }

    pub fn p(self: *StateMap, cx: i32) i32 {
        const cur: i32 = self.t[self.cxt];
        self.t[self.cxt] = @intCast((cur + (((y << 16) - cur + 128) >> 8)) & 0xffff);
        self.cxt = @intCast(cx);
        return @as(i32, self.t[self.cxt]) >> 4;
    }

    pub fn deinit(self: *StateMap) void {
        self.alloc.free(self.t);
    }
};

// ================================ StateMap32 ===============================
// paq8's U32 StateMap: 22-bit prediction in high bits, 10-bit count in low bits.
pub const StateMap32 = struct {
    N: usize,
    cxt: usize = 0,
    t: []u32,
    alloc: std.mem.Allocator,

    pub fn init(a: std.mem.Allocator, n: usize, do_init: bool) StateMap32 {
        var self = StateMap32{ .N = n, .t = a.alloc(u32, n) catch unreachable, .alloc = a };
        if (do_init and n == 256) {
            var i: usize = 0;
            while (i < 256) : (i += 1) {
                var n0: u32 = nex(@intCast(i), 2);
                var n1: u32 = nex(@intCast(i), 3);
                if (n0 == 0) n1 *%= 64;
                if (n1 == 0) n0 *%= 64;
                self.t[i] = ((n1 << 16) / (n0 + n1 + 1)) << 16;
            }
        } else {
            for (self.t) |*e| e.* = @as(u32, 1) << 31;
        }
        return self;
    }

    pub fn update(self: *StateMap32, limit: i32) void {
        var p0 = self.t[self.cxt];
        const n: i32 = @intCast(p0 & 1023);
        const pr: i32 = @intCast(p0 >> 10);
        if (n < limit) p0 +%= 1 else p0 = (p0 & 0xfffffc00) | @as(u32, @intCast(limit));
        const target: i32 = y << 22;
        const delta: i32 = ((target -% pr) >> 3) *% dt[@intCast(n)];
        p0 +%= @as(u32, @bitCast(delta)) & 0xfffffc00;
        self.t[self.cxt] = p0;
    }

    pub fn p(self: *StateMap32, cx: i32, limit: i32) i32 {
        self.update(limit);
        self.cxt = @intCast(cx);
        return @intCast(self.t[self.cxt] >> 20);
    }

    pub fn reset(self: *StateMap32, rate: u32) void {
        for (self.t) |*e| e.* = (e.* & 0xfffffc00) | @min(rate, e.* & 0x3FF);
    }

    pub fn deinit(self: *StateMap32) void {
        self.alloc.free(self.t);
    }
};

// =================================== APM ===================================
// paq8's APM (public StateMap32 with n*24 entries + interpolated lookup).
pub const APM = struct {
    sm: StateMap32,

    pub fn init(a: std.mem.Allocator, n: usize) APM {
        var self = APM{ .sm = StateMap32.init(a, n * 24, true) };
        var i: usize = 0;
        while (i < self.sm.N) : (i += 1) {
            const pp: i32 = @divTrunc((@as(i32, @intCast(i % 24)) * 2 + 1) * 4096, 48) - 2048;
            self.sm.t[i] = (@as(u32, @intCast(squash(pp))) << 20) + 6;
        }
        return self;
    }

    pub fn p(self: *APM, pr_in: i32, cx_in: i32, limit: i32) i32 {
        self.sm.update(limit);
        const pr = (stretch(pr_in) + 2048) * 23;
        const wt = pr & 0xfff;
        const cx = cx_in * 24 + (pr >> 12);
        self.sm.cxt = @intCast(cx + (wt >> 11));
        const ci: usize = @intCast(cx);
        const a: i32 = @intCast(self.sm.t[ci] >> 13);
        const b: i32 = @intCast(self.sm.t[ci + 1] >> 13);
        return (a *% (4096 - wt) +% b *% wt) >> 19;
    }

    pub fn deinit(self: *APM) void {
        self.sm.deinit();
    }
};

// ==================================== tests ================================
const testing = std.testing;

test "squash/stretch round-trip mid-range" {
    init();
    var pv: i32 = 500;
    while (pv <= 3500) : (pv += 137) {
        const s = stretch(pv);
        const back = squash(s);
        try testing.expect(@abs(back - pv) <= 64);
    }
    // squash monotonic, endpoints
    try testing.expectEqual(@as(i32, 4095), squash(3000));
    try testing.expectEqual(@as(i32, 0), squash(-3000));
    try testing.expect(squash(0) > 2000 and squash(0) < 2100);
}

test "ilog / llog sane" {
    init();
    try testing.expectEqual(@as(i32, 0), ilog(0));
    try testing.expectEqual(@as(i32, 0), ilog(1));
    try testing.expect(ilog(256) > ilog(2));
    // monotonic non-decreasing
    var prev: i32 = 0;
    var v: u32 = 2;
    while (v < 60000) : (v += 997) {
        const cur = ilog(v);
        try testing.expect(cur >= prev);
        prev = cur;
    }
    try testing.expect(llog(0x2000000) > llog(0x8000));
}

test "dt matches 16384/(2i+3)" {
    init();
    try testing.expectEqual(@as(i32, 16384 / 3), dt[0]);
    try testing.expectEqual(@as(i32, 16384 / 5), dt[1]);
    try testing.expectEqual(@as(i32, 16384 / 2049), dt[1023]);
}

test "Random deterministic sequence" {
    var r = Random{};
    r.init();
    const a0 = r.next();
    const a1 = r.next();
    // reseed and verify reproducibility
    var r2 = Random{};
    r2.init();
    try testing.expectEqual(a0, r2.next());
    try testing.expectEqual(a1, r2.next());
}

test "Array grow + zero init" {
    const a = testing.allocator;
    var arr = Array(i32, 0).init(a);
    defer arr.deinit();
    arr.resize(4);
    try testing.expectEqual(@as(u32, 4), arr.size());
    for (0..4) |i| try testing.expectEqual(@as(i32, 0), arr.get(@intCast(i)));
    arr.at(2).* = 42;
    try testing.expectEqual(@as(i32, 42), arr.get(2));
    arr.pushBack(7);
    try testing.expectEqual(@as(u32, 5), arr.size());
    try testing.expectEqual(@as(i32, 7), arr.get(4));
    // grow preserves data
    try testing.expectEqual(@as(i32, 42), arr.get(2));
}

test "Buf ring wraps" {
    const a = testing.allocator;
    resetState();
    var b = Buf{};
    defer b.deinit();
    b.setsize(a, 16);
    // write bytes 0..19 at increasing positions; ring size 16
    var i: u32 = 0;
    while (i < 20) : (i += 1) {
        b.at(i).* = @intCast(i);
    }
    // position 19 wrote index 3, overwriting original 3
    try testing.expectEqual(@as(u8, 19), b.at(19).*);
    try testing.expectEqual(b.at(3).*, b.at(19).*);
    // buf(i): relative to pos
    pos = 20;
    try testing.expectEqual(@as(i32, 19), b.get(1)); // byte at pos-1 = index 19&15=3 -> 19
}

test "StateMap moves toward observed bit" {
    init();
    const a = testing.allocator;
    var sm = StateMap.init(a);
    defer sm.deinit();
    // feed context 5 repeatedly with y=1; prediction (in 0..4095, >>4 of 16-bit) should rise
    y = 0;
    _ = sm.p(5);
    const first = sm.p(5);
    y = 1;
    var last: i32 = first;
    for (0..50) |_| last = sm.p(5);
    try testing.expect(last > first);
}

test "StateMap32 moves toward observed bit" {
    init();
    const a = testing.allocator;
    var sm = StateMap32.init(a, 256, false);
    defer sm.deinit();
    y = 0;
    _ = sm.p(3, 1023);
    const first = sm.p(3, 1023);
    y = 1;
    var last: i32 = first;
    for (0..200) |_| last = sm.p(3, 1023);
    try testing.expect(last > first);
}

test "Mixer converges toward target bit" {
    init();
    const a = testing.allocator;
    var m = Mixer.init(a, 8, 1, 1, 0);
    defer m.deinit();
    // Feed a consistent set of stretched inputs voting for y=1, train repeatedly.
    var round: usize = 0;
    var pr_first: i32 = 0;
    var pr_last: i32 = 0;
    while (round < 300) : (round += 1) {
        resetPredictions();
        m.add(512);
        m.add(700);
        m.add(300);
        m.set(0, 1);
        const pr = m.p();
        if (round == 0) pr_first = pr;
        pr_last = pr;
        y = 1;
        m.update();
    }
    try testing.expect(pr_last > pr_first);
    try testing.expect(pr_last > 2048); // leaning toward 1
}

test "APM1 interpolation runs and adapts" {
    init();
    const a = testing.allocator;
    var apm = APM1.init(a, 256);
    defer apm.deinit();
    y = 1;
    var pr: i32 = 2048;
    for (0..100) |_| pr = apm.p(2048, 5, 7);
    try testing.expect(pr >= 0 and pr <= 4095);
}

test "APM interpolation runs" {
    init();
    const a = testing.allocator;
    var apm = APM.init(a, 256);
    defer apm.deinit();
    y = 0;
    var pr: i32 = 2048;
    for (0..50) |_| pr = apm.p(2048, 7, 0xFF);
    try testing.expect(pr >= 0 and pr <= 4095);
}
