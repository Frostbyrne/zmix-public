//! Faithful Zig port of cmix PAQ8's MatchModel and SparseMatchModel
//! (reference/cmix-src/models/paq8.cpp, ~3520-3842).
//!
//! Builds on core.zig and maps.zig. Each model is heap-allocated once; its
//! C++ member StateMap32/SmallStationaryContextMap/StationaryMap/IndirectContext
//! objects become instance fields that persist across calls.
//!
//! Faithfulness notes:
//!  - C `unsigned` wrap -> `+%`/`-%`; truncating (U8/U16/U32) stores -> `@truncate`.
//!  - hashargs keep the C++ argument signedness/width so the U64 conversion
//!    (sign- vs zero-extend) matches; all such values are non-negative here.
//!  - The exact sequence and count of mixer.add/mixer.setcalls mirror C++.
const std = @import("std");
const core = @import("core.zig");
const maps = @import("maps.zig");

// ===========================================================================
// MatchModel  (paq8.cpp:3520-3692)
// ===========================================================================
pub const MatchModel = struct {
    const MaxLen: u32 = 0xFFFF; // longest allowed match
    const MaxExtend: u32 = 0; // longest allowed match expansion
    const MinLen: u32 = 5; // minimum required match length
    const StepSize: u32 = 2; // additional min length per higher-order hash
    const DeltaLen: u32 = 5; // min length to switch to delta mode
    const NumCtxs: u32 = 3; // number of contexts used
    const NumHashes: u32 = 3; // number of hashes used

    Table: core.Array(u32, 0),
    StateMaps: [NumCtxs]core.StateMap32,
    SCM: [3]maps.SmallStationaryContextMap,
    Maps: [3]maps.StationaryMap,
    iCtx: maps.IndirectContext(u8),
    hashes: [NumHashes]u32,
    ctx: [NumCtxs]u32,
    length: u32, // rebased match length (1 == smallest accepted), 0 if none
    index: u32, // next byte of match in buffer, 0 if none
    mask: u32,
    hashbits: i32,
    expectedByte: u8,
    delta: bool,
    alloc: std.mem.Allocator,

    pub fn init(a: std.mem.Allocator, size: u64) *MatchModel {
        const self = a.create(MatchModel) catch unreachable;
        const nelem: u32 = @intCast(size / @sizeOf(u32));
        self.* = .{
            .Table = core.Array(u32, 0).initSize(a, nelem),
            .StateMaps = .{
                core.StateMap32.init(a, 56 * 256, true),
                core.StateMap32.init(a, 8 * 256 * 256 + 1, true),
                core.StateMap32.init(a, 256 * 256, true),
            },
            .SCM = .{
                maps.SmallStationaryContextMap.init(a, 8, 8),
                maps.SmallStationaryContextMap.init(a, 11, 1),
                maps.SmallStationaryContextMap.init(a, 8, 8),
            },
            .Maps = .{
                maps.StationaryMap.init(a, 16, 8, 0),
                maps.StationaryMap.init(a, 22, 1, 0),
                maps.StationaryMap.init(a, 4, 1, 0),
            },
            .iCtx = maps.IndirectContext(u8).init(a, 19, 1),
            .hashes = .{ 0, 0, 0 },
            .ctx = .{ 0, 0, 0 },
            .length = 0,
            .index = 0,
            .mask = nelem - 1,
            .hashbits = @intCast(maps.ilog2(nelem)), // ilog2(mask+1)
            .expectedByte = 0,
            .delta = false,
            .alloc = a,
        };
        return self;
    }

    fn update(self: *MatchModel, stats: ?*core.ModelStats) void {
        self.delta = false;
        // update hashes
        {
            var i: u32 = 0;
            var minLen: u32 = MinLen + (NumHashes - 1) * StepSize;
            while (i < NumHashes) : ({
                i += 1;
                minLen -%= StepSize;
            }) {
                var h: u64 = 0;
                var j: u32 = minLen;
                while (j > 0) : (j -= 1) h = maps.combine64(h, @as(u64, @intCast(core.buf.get(j))));
                self.hashes[i] = maps.finalize64(h, self.hashbits);
            }
        }
        // extend current match, if available
        if (self.length != 0) {
            self.index +%= 1;
            if (self.length < MaxLen) self.length += 1;
        }
        // or find a new match, highest order hash first
        else {
            var minLen: u32 = MinLen + (NumHashes - 1) * StepSize;
            var bestLen: u32 = 0;
            var bestIndex: u32 = 0;
            var i: u32 = 0;
            while (i < NumHashes and self.length < minLen) : ({
                i += 1;
                minLen -%= StepSize;
            }) {
                self.index = self.Table.get(self.hashes[i]);
                if (self.index > 0) {
                    self.length = 0;
                    while (self.length < (minLen + MaxExtend) and
                        core.buf.get(self.length + 1) == @as(i32, core.buf.at(self.index -% self.length -% 1).*))
                    {
                        self.length += 1;
                    }
                    if (self.length > bestLen) {
                        bestLen = self.length;
                        bestIndex = self.index;
                    }
                }
            }
            if (bestLen >= MinLen) {
                self.length = bestLen - (MinLen - 1); // rebase
                self.index = bestIndex;
            } else {
                self.length = 0;
                self.index = 0;
            }
        }
        // update position information in hashtable
        {
            var i: u32 = 0;
            while (i < NumHashes) : (i += 1) self.Table.at(self.hashes[i]).* = @bitCast(core.pos);
        }
        self.expectedByte = core.buf.at(self.index).*;
        self.iCtx.add(@intCast(core.y));
        self.iCtx.setCtx(@intCast((core.buf.get(1) << 8) | @as(i32, self.expectedByte)));
        self.SCM[0].set(self.expectedByte);
        self.SCM[1].set(self.expectedByte);
        self.SCM[2].set(@bitCast(core.pos));
        self.Maps[0].setDirect((@as(u32, self.expectedByte) << 8) | @as(u32, @intCast(core.buf.get(1))));
        self.Maps[1].set(maps.hash(.{
            self.expectedByte, core.c0, core.buf.get(1), core.buf.get(2),
            @min(@as(i32, 3), @as(i32, @intCast(maps.ilog2(self.length + 1)))),
        }));
        self.Maps[2].setDirect(self.iCtx.value());
        if (stats) |s| s.Match.expectedByte = if (self.length > 0) self.expectedByte else 0;
    }

    pub fn predict(self: *MatchModel, mixer: *core.Mixer, stats: ?*core.ModelStats) i32 {
        if (core.bpos == 0) {
            self.update(stats);
        } else {
            const B: u8 = @truncate(@as(u32, @intCast(core.c0)) << @intCast(8 - core.bpos));
            self.SCM[1].set(@intCast((core.bpos << 8) | (@as(i32, self.expectedByte) ^ @as(i32, B))));
            self.Maps[1].set(maps.hash(.{
                self.expectedByte, core.c0, core.buf.get(1), core.buf.get(2),
                @min(@as(i32, 3), @as(i32, @intCast(maps.ilog2(self.length + 1)))),
            }));
            self.iCtx.add(@intCast(core.y));
            self.iCtx.setCtx(@intCast((core.bpos << 16) | (core.buf.get(1) << 8) | (@as(i32, self.expectedByte) ^ @as(i32, B))));
            self.Maps[2].setDirect(self.iCtx.value());
        }
        const expectedBit: i32 = (@as(i32, self.expectedByte) >> @intCast(7 - core.bpos)) & 1;

        if (self.length > 0) {
            const isMatch = if (core.bpos == 0)
                (core.buf.get(1) == @as(i32, core.buf.at(self.index -% 1).*))
            else
                (((@as(i32, self.expectedByte) + 256) >> @intCast(8 - core.bpos)) == core.c0);
            if (!isMatch) {
                self.delta = (self.length + MinLen) > DeltaLen;
                self.length = 0;
            }
        }

        for (&self.ctx) |*cc| cc.* = 0;
        if (self.length > 0) {
            if (self.length <= 16)
                self.ctx[0] = (self.length - 1) * 2 + @as(u32, @intCast(expectedBit)) // 0..31
            else
                self.ctx[0] = 24 + (@as(u32, @min(self.length - 1, 63)) >> 2) * 2 + @as(u32, @intCast(expectedBit)); // 32..55
            self.ctx[0] = (self.ctx[0] << 8) | @as(u32, @intCast(core.c0));
            self.ctx[1] = ((@as(u32, self.expectedByte) << 11) | (@as(u32, @intCast(core.bpos)) << 8) | @as(u32, @intCast(core.buf.get(1)))) + 1;
            const sign: i32 = 2 * expectedBit - 1;
            mixer.add(sign * @as(i32, @intCast(@as(u32, @min(self.length, 32)) << 5))); // +/- 32..1024
            mixer.add(sign * (core.ilog(self.length) << 2)); // +/- 0..1024
        } else {
            mixer.add(0);
            mixer.add(0);
        }

        if (self.delta)
            self.ctx[2] = (@as(u32, self.expectedByte) << 8) | @as(u32, @intCast(core.c0));

        {
            var i: u32 = 0;
            while (i < NumCtxs) : (i += 1) {
                const c = self.ctx[i];
                const p = self.StateMaps[i].p(@intCast(c), 1023);
                if (c != 0)
                    mixer.add((core.stretch(p) + 1) >> 1)
                else
                    mixer.add(0);
            }
        }

        self.SCM[0].mix(mixer, 7, 1, 4);
        self.SCM[1].mix(mixer, 6, 1, 4);
        self.SCM[2].mix(mixer, 5, 1, 4);
        self.Maps[0].mix(mixer, 1, 4, 255);
        self.Maps[1].mix(mixer, 1, 4, 1023);
        self.Maps[2].mix(mixer, 1, 4, 1023);

        if (stats) |s| s.Match.length = self.length;
        return @intCast(self.length);
    }

    pub fn deinit(self: *MatchModel) void {
        self.Table.deinit();
        for (&self.StateMaps) |*s| s.deinit();
        for (&self.SCM) |*s| s.deinit();
        for (&self.Maps) |*m| m.deinit();
        self.iCtx.deinit();
        self.alloc.destroy(self);
    }
};

// ===========================================================================
// SparseMatchModel  (paq8.cpp:3694-3842)
// ===========================================================================
pub const SparseMatchModel = struct {
    const MaxLen: u32 = 0xFFFF;
    const MinLen: u32 = 3;
    const NumHashes: u32 = 4;

    const SparseConfig = struct {
        offset: u32 = 0, // last input bytes to ignore when searching
        stride: u32 = 1, // look for a match only every stride bytes after offset
        deletions: u32 = 0, // ignore these many initial post-match bytes
        minLen: u32 = MinLen,
        bitMask: u32 = 0xFF, // match every byte according to this bit mask
    };

    Table: core.Array(u32, 0),
    Maps: [4]maps.StationaryMap,
    iCtx8: maps.IndirectContext(u8),
    iCtx16: maps.IndirectContext(u16),
    list: maps.MTFList,
    sparse: [NumHashes]SparseConfig,
    hashes: [NumHashes]u32,
    hashIndex: u32, // index of hash used to find current match
    length: u32,
    index: u32,
    mask: u32,
    hashbits: i32,
    expectedByte: u8,
    valid: bool,
    alloc: std.mem.Allocator,

    pub fn init(a: std.mem.Allocator, size: u64) *SparseMatchModel {
        const self = a.create(SparseMatchModel) catch unreachable;
        const nelem: u32 = @intCast(size / @sizeOf(u32));
        self.* = .{
            .Table = core.Array(u32, 0).initSize(a, nelem),
            .Maps = .{
                maps.StationaryMap.init(a, 22, 1, 0),
                maps.StationaryMap.init(a, 14, 4, 0),
                maps.StationaryMap.init(a, 8, 1, 0),
                maps.StationaryMap.init(a, 19, 1, 0),
            },
            .iCtx8 = maps.IndirectContext(u8).init(a, 19, 1),
            .iCtx16 = maps.IndirectContext(u16).init(a, 16, 8),
            .list = maps.MTFList.init(a, NumHashes),
            .sparse = .{
                .{ .minLen = 5, .bitMask = 0xDF },
                .{ .offset = 1, .minLen = 4 },
                .{ .stride = 2, .minLen = 4, .bitMask = 0xDF },
                .{ .minLen = 5, .bitMask = 0xF },
            },
            .hashes = .{ 0, 0, 0, 0 },
            .hashIndex = 0,
            .length = 0,
            .index = 0,
            .mask = nelem - 1,
            .hashbits = @intCast(maps.ilog2(nelem)),
            .expectedByte = 0,
            .valid = false,
            .alloc = a,
        };
        return self;
    }

    fn update(self: *SparseMatchModel, stats: ?*core.ModelStats) void {
        _ = stats;
        // update sparse hashes
        {
            var i: u32 = 0;
            while (i < NumHashes) : (i += 1) {
                var h: u64 = 0;
                var j: u32 = 0;
                var k: u32 = self.sparse[i].offset + 1;
                while (j < self.sparse[i].minLen) : ({
                    j += 1;
                    k += self.sparse[i].stride;
                }) {
                    h = maps.combine64(h, @as(u64, @intCast(core.buf.get(k) & @as(i32, @intCast(self.sparse[i].bitMask)))));
                }
                self.hashes[i] = maps.finalize64(h, self.hashbits);
            }
        }
        // extend current match, if available
        if (self.length != 0) {
            self.index +%= 1;
            if (self.length < MaxLen) self.length += 1;
        }
        // or find a new match
        else {
            var i: i32 = self.list.getFirst();
            while (i >= 0) : (i = self.list.getNext()) {
                const iu: usize = @intCast(i);
                self.index = self.Table.get(self.hashes[iu]);
                if (self.index > 0) {
                    var offset: u32 = self.sparse[iu].offset + 1;
                    while (self.length < self.sparse[iu].minLen and
                        ((core.buf.get(offset) ^ @as(i32, core.buf.at(self.index -% offset).*)) & @as(i32, @intCast(self.sparse[iu].bitMask))) == 0)
                    {
                        self.length += 1;
                        offset += self.sparse[iu].stride;
                    }
                    if (self.length >= self.sparse[iu].minLen) {
                        self.length -= (self.sparse[iu].minLen - 1);
                        self.index +%= self.sparse[iu].deletions;
                        self.hashIndex = @intCast(iu);
                        self.list.moveToFront(@intCast(iu));
                        break;
                    }
                }
                self.length = 0;
                self.index = 0;
            }
        }
        // update position information in hashtable
        {
            var i: u32 = 0;
            while (i < NumHashes) : (i += 1) self.Table.at(self.hashes[i]).* = @bitCast(core.pos);
        }

        self.expectedByte = core.buf.at(self.index).*;
        if (self.valid) {
            self.iCtx8.add(@intCast(core.y));
            self.iCtx16.add(@intCast(core.buf.get(1)));
        }
        self.valid = self.length > 1; // only predict after >=1 byte following match
        if (self.valid) {
            self.Maps[0].set(maps.hash(.{
                self.expectedByte, core.c0, core.buf.get(1), core.buf.get(2),
                maps.ilog2(self.length + 1) * NumHashes + self.hashIndex,
            }));
            self.Maps[1].setDirect((@as(u32, self.expectedByte) << 8) | @as(u32, @intCast(core.buf.get(1))));
            self.iCtx8.setCtx(@intCast((core.buf.get(1) << 8) | @as(i32, self.expectedByte)));
            self.iCtx16.setCtx(@intCast((core.buf.get(1) << 8) | @as(i32, self.expectedByte)));
            self.Maps[2].setDirect(self.iCtx8.value());
            self.Maps[3].setDirect(self.iCtx16.value());
        }
    }

    pub fn predict(self: *SparseMatchModel, mixer: *core.Mixer, stats: ?*core.ModelStats) i32 {
        const B: u8 = @truncate(@as(u32, @intCast(core.c0)) << @intCast(8 - core.bpos));
        if (core.bpos == 0) {
            self.update(stats);
        } else if (self.valid) {
            self.Maps[0].set(maps.hash(.{
                self.expectedByte, core.c0, core.buf.get(1), core.buf.get(2),
                maps.ilog2(self.length + 1) * NumHashes + self.hashIndex,
            }));
            if (core.bpos == 4)
                self.Maps[1].setDirect(0x10000 |
                    (@as(u32, self.expectedByte ^ @as(u8, @truncate(@as(u32, @intCast(core.c0)) << 4))) << 8) |
                    @as(u32, @intCast(core.buf.get(1))));
            self.iCtx8.add(@intCast(core.y));
            self.iCtx8.setCtx(@intCast((core.bpos << 16) | (core.buf.get(1) << 8) | (@as(i32, self.expectedByte) ^ @as(i32, B))));
            self.Maps[2].setDirect(self.iCtx8.value());
            self.Maps[3].setDirect(@as(u32, @intCast(core.bpos << 16)) |
                (@as(u32, self.iCtx16.value()) ^ (@as(u32, B) | (@as(u32, B) << 8))));
        }

        // check if next bit matches the prediction, accounting for the bitmask
        if (self.length > 0 and
            (((@as(u32, self.expectedByte) ^ @as(u32, B)) & self.sparse[self.hashIndex].bitMask) >> @intCast(8 - core.bpos)) != 0)
        {
            self.length = 0;
        }

        if (self.valid) {
            if (self.length > 1 and ((self.sparse[self.hashIndex].bitMask >> @intCast(7 - core.bpos)) & 1) > 0) {
                const expectedBit: i32 = (@as(i32, self.expectedByte) >> @intCast(7 - core.bpos)) & 1;
                const sign: i32 = 2 * expectedBit - 1;
                mixer.add(sign * @as(i32, @intCast(@as(u32, @min(self.length - 1, 64)) << 4))); // +/- 16..1024
                const mag: u32 = (@as(u32, 1) << @intCast(@min(self.length - 2, 3))) * @as(u32, @min(self.length - 1, 8));
                mixer.add(sign * @as(i32, @intCast(mag << 4))); // +/- 16..1024
                mixer.add(sign * 512);
            } else {
                mixer.add(0);
                mixer.add(0);
                mixer.add(0);
            }
            var i: usize = 0;
            while (i < 4) : (i += 1) self.Maps[i].mix(mixer, 1, 2, 1023);
        } else {
            var i: usize = 0;
            while (i < 11) : (i += 1) mixer.add(0);
        }

        mixer.set(@intCast((self.hashIndex << 6) | (@as(u32, @intCast(core.bpos)) << 3) | @as(u32, @min(self.length, 7))), NumHashes * 64);
        mixer.set(@intCast((self.hashIndex << 11) |
            (@as(u32, @min(@as(u32, 7), maps.ilog2(self.length + 1))) << 8) |
            (@as(u32, @intCast(core.c0)) ^ (@as(u32, self.expectedByte) >> @intCast(8 - core.bpos)))), NumHashes * 2048);
        return @intCast(self.length);
    }

    pub fn deinit(self: *SparseMatchModel) void {
        self.Table.deinit();
        for (&self.Maps) |*m| m.deinit();
        self.iCtx8.deinit();
        self.iCtx16.deinit();
        self.list.deinit();
        self.alloc.destroy(self);
    }
};

// ================================== tests ==================================
// Structural smoke tests. Bit-exact behaviour is verified by a full
// differential test against the compiled C++ reference (see porter report).
const testing = std.testing;

fn driveBit(y: i32) void {
    core.y = y;
    core.c0 += core.c0 + y;
    if (core.c0 >= 256) {
        core.buf.at(@bitCast(core.pos)).* = @truncate(@as(u32, @intCast(core.c0)));
        core.pos += 1;
        core.c0 -= 256;
        core.c4 = (core.c4 << 8) +% @as(u32, @intCast(core.c0));
        core.c0 = 1;
    }
    core.bpos = (core.bpos + 1) & 7;
}

test "MatchModel runs; 17 adds/bit; finite" {
    core.level = 6;
    core.init();
    core.resetState();
    const a = testing.allocator;
    core.buf.setsize(a, @intCast(core.MEM() * 8));
    defer core.buf.deinit();
    const model = MatchModel.init(a, 1 << 20);
    defer model.deinit();
    const m = core.Mixer.init(a, 64, 0, 1, 0);
    defer m.deinit();
    var stats = core.ModelStats{};
    const data = "the quick brown fox the quick brown fox jumps 12345 12345";
    for (data) |byte| {
        var k: i32 = 0;
        while (k < 8) : (k += 1) {
            driveBit((@as(i32, byte) >> @intCast(7 - k)) & 1);
            core.resetPredictions();
            _ = model.predict(m, &stats);
            try testing.expectEqual(@as(usize, 17), m.nx);
            for (core.model_predictions[0..core.prediction_index]) |p| try testing.expect(std.math.isFinite(p));
            while (m.nx & 7 != 0) {
                m.tx[m.nx] = 0;
                m.nx += 1;
            }
            m.update();
        }
    }
}

test "SparseMatchModel runs; 11 adds + 2 sets/bit; finite" {
    core.level = 6;
    core.init();
    core.resetState();
    const a = testing.allocator;
    core.buf.setsize(a, @intCast(core.MEM() * 8));
    defer core.buf.deinit();
    const model = SparseMatchModel.init(a, 1 << 18);
    defer model.deinit();
    const m = core.Mixer.init(a, 64, 0, 8, 0);
    defer m.deinit();
    var stats = core.ModelStats{};
    const data = "abcabcabcabc the the the 1 2 3 4 5 aXbXcX aYbYcY";
    for (data) |byte| {
        var k: i32 = 0;
        while (k < 8) : (k += 1) {
            driveBit((@as(i32, byte) >> @intCast(7 - k)) & 1);
            core.resetPredictions();
            _ = model.predict(m, &stats);
            try testing.expectEqual(@as(usize, 11), m.nx);
            try testing.expectEqual(@as(usize, 2), m.ncxt);
            for (core.model_predictions[0..core.prediction_index]) |p| try testing.expect(std.math.isFinite(p));
            while (m.nx & 7 != 0) {
                m.tx[m.nx] = 0;
                m.nx += 1;
            }
            m.update();
        }
    }
}
