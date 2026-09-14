//! Faithful Zig port of cmix PAQ8's DMC (Dynamic Markov Compression) models:
//!   struct DMCNode, class dmcModel, class dmcForest
//!   (reference/cmix-src/models/paq8.cpp, ~7614-7813).
//!
//! Builds on core.zig (Array, StateMap32, Mixer, nex, stretch, MEM, y, bpos).
//!
//! Faithfulness notes:
//!  - DMCNode packs nx0/nx1 in the high 28 bits and the 4+4 bit-history state in
//!    the low 4+4 bits of two U32 words (`_nx0`,`_nx1`), exactly like C++.
//!  - C `unsigned` wrap -> `+%`/`-%`; U32->U16 truncating stores -> `@truncate`.
//!  - DMC_NODES_MAX uses the C++ literal sizeof(DMCNode)==12 so the cloning cap
//!    matches bit-for-bit regardless of Zig's struct layout.
const std = @import("std");
const core = @import("core.zig");

const DMC_NODES_BASE: u64 = 255 * 256; // = 65280
// (U64(1)<<31)/sizeof(DMCNode); C++ sizeof(DMCNode)==12
const DMC_NODES_MAX: u64 = (@as(u64, 1) << 31) / 12; // = 178956970

// 12-byte state-graph node.
const DMCNode = struct {
    c0: u16 = 0,
    c1: u16 = 0,
    // packed: high 28 bits are nx0/nx1, low 4 bits are the two state nibbles.
    nx0_: u32 = 0,
    nx1_: u32 = 0,

    inline fn getState(self: *const DMCNode) u8 {
        return @truncate(((self.nx0_ & 0xf) << 4) | (self.nx1_ & 0xf));
    }
    inline fn setState(self: *DMCNode, state: u8) void {
        self.nx0_ = (self.nx0_ & 0xfffffff0) | (@as(u32, state) >> 4);
        self.nx1_ = (self.nx1_ & 0xfffffff0) | (@as(u32, state) & 0xf);
    }
    inline fn getNx0(self: *const DMCNode) u32 {
        return self.nx0_ >> 4;
    }
    inline fn setNx0(self: *DMCNode, nx0: u32) void {
        self.nx0_ = (self.nx0_ & 0xf) | (nx0 << 4);
    }
    inline fn getNx1(self: *const DMCNode) u32 {
        return self.nx1_ >> 4;
    }
    inline fn setNx1(self: *DMCNode, nx1: u32) void {
        self.nx1_ = (self.nx1_ & 0xf) | (nx1 << 4);
    }
};

// helper: adaptively increment a fixed-point counter: x*(1-1/64) + increment
inline fn incrementCounter(x: u32, increment: u32) u32 {
    return (((x << 6) -% x) >> 6) +% (increment << 10);
}

const DmcModel = struct {
    t: core.Array(DMCNode, 0), // state graph
    sm: core.StateMap32, // statemap for bit-history states
    top: u32, // first unallocated node (== #allocated)
    curr: u32, // current node
    threshold: u32, // cloning threshold (fixed point like c0,c1)
    threshold_fine: u32, // "threshold" scaled by 11 bits
    extra: u32, // approximate maturity when graph is full
    alloc: std.mem.Allocator,

    fn init(a: std.mem.Allocator, dmc_nodes: u64, th_start: u32) DmcModel {
        const n: u32 = @intCast(@min(dmc_nodes + DMC_NODES_BASE, DMC_NODES_MAX));
        var self = DmcModel{
            .t = core.Array(DMCNode, 0).initSize(a, n),
            .sm = core.StateMap32.init(a, 256, true),
            .top = 0,
            .curr = 0,
            .threshold = 0,
            .threshold_fine = 0,
            .extra = 0,
            .alloc = a,
        };
        self.resetstategraph(th_start);
        return self;
    }

    // Initialize the state graph to a bytewise order-1 model (256 trees).
    fn resetstategraph(self: *DmcModel, th_start: u32) void {
        self.top = 0;
        self.curr = 0;
        self.extra = 0;
        self.threshold = th_start;
        self.threshold_fine = th_start << 11;
        var j: i32 = 0;
        while (j < 256) : (j += 1) { // 256 trees
            var i: i32 = 0;
            while (i < 255) : (i += 1) { // 255 nodes per tree
                const node = self.t.at(self.top);
                if (i < 127) { // internal tree nodes
                    node.setNx0(self.top +% @as(u32, @intCast(i)) +% 1);
                    node.setNx1(self.top +% @as(u32, @intCast(i)) +% 2);
                } else { // 128 leaf nodes -> reference a root of tree(i)
                    const linked_tree_root: u32 = @intCast((i - 127) * 2 * 255);
                    node.setNx0(linked_tree_root);
                    node.setNx1(linked_tree_root +% 255);
                }
                const cval: u16 = if (th_start < 1024) 2048 else 512; // 2.0 or 0.5
                node.c0 = cval;
                node.c1 = cval;
                node.setState(0);
                self.top +%= 1;
            }
        }
    }

    fn update(self: *DmcModel) void {
        var cnt0: u32 = self.t.at(self.curr).c0;
        var cnt1: u32 = self.t.at(self.curr).c1;
        const yy: u32 = @intCast(core.y);
        const n: u32 = if (yy == 0) cnt0 else cnt1;

        // update counts, state
        self.t.at(self.curr).c0 = @truncate(incrementCounter(cnt0, 1 -% yy));
        self.t.at(self.curr).c1 = @truncate(incrementCounter(cnt1, yy));
        {
            const cur = self.t.at(self.curr);
            cur.setState(core.nex(cur.getState(), yy));
        }

        // clone next state when threshold is reached
        if (n > self.threshold) {
            const next: u32 = if (yy == 0) self.t.at(self.curr).getNx0() else self.t.at(self.curr).getNx1();
            cnt0 = self.t.at(next).c0;
            cnt1 = self.t.at(next).c1;
            const nn: u32 = cnt0 +% cnt1;

            if (nn > n +% self.threshold) {
                if (self.top != self.t.size()) { // graph not full: clone
                    const c0_top: u32 = @intCast(@as(u64, cnt0) * @as(u64, n) / @as(u64, nn));
                    const c1_top: u32 = @intCast(@as(u64, cnt1) * @as(u64, n) / @as(u64, nn));
                    cnt0 -%= c0_top;
                    cnt1 -%= c1_top;

                    self.t.at(self.top).c0 = @truncate(c0_top);
                    self.t.at(self.top).c1 = @truncate(c1_top);
                    self.t.at(next).c0 = @truncate(cnt0);
                    self.t.at(next).c1 = @truncate(cnt1);

                    self.t.at(self.top).setNx0(self.t.at(next).getNx0());
                    self.t.at(self.top).setNx1(self.t.at(next).getNx1());
                    self.t.at(self.top).setState(self.t.at(next).getState());
                    if (yy == 0) self.t.at(self.curr).setNx0(self.top) else self.t.at(self.curr).setNx1(self.top);

                    self.top +%= 1;

                    if (self.threshold < 8 * 1024) {
                        self.threshold_fine +%= 1;
                        self.threshold = self.threshold_fine >> 11;
                    }
                } else { // graph was full
                    self.extra +%= nn >> 10;
                }
            }
        }

        if (yy == 0) self.curr = self.t.at(self.curr).getNx0() else self.curr = self.t.at(self.curr).getNx1();
    }

    fn isfull(self: *DmcModel) bool {
        return (self.extra >> 7) > self.t.size();
    }

    fn pr1(self: *DmcModel) i32 {
        const n0: u32 = @as(u32, self.t.at(self.curr).c0) +% 1;
        const n1: u32 = @as(u32, self.t.at(self.curr).c1) +% 1;
        return @intCast((n1 << 12) / (n0 +% n1));
    }

    fn pr2(self: *DmcModel) i32 {
        const state = self.t.at(self.curr).getState();
        return self.sm.p(@intCast(state), 256); // 64-512 are all fine
    }

    fn st(self: *DmcModel) i32 {
        self.update();
        return core.stretch(self.pr1()) + core.stretch(self.pr2());
    }

    fn deinit(self: *DmcModel) void {
        self.t.deinit();
        self.sm.deinit();
    }
};

pub const DmcForest = struct {
    const MODELS: usize = 10; // 8 fast and 2 slow models
    const dmcparams = [10]u32{ 2, 32, 64, 4, 128, 8, 256, 16, 1024, 1536 };
    const dmcmem = [10]u64{ 6, 10, 11, 7, 12, 8, 13, 9, 2, 2 };

    models: [MODELS]DmcModel,
    alloc: std.mem.Allocator,

    pub fn init(a: std.mem.Allocator) *DmcForest {
        const self = a.create(DmcForest) catch unreachable;
        self.alloc = a;
        var i: i32 = @as(i32, MODELS) - 1;
        while (i >= 0) : (i -= 1) {
            const iu: usize = @intCast(i);
            self.models[iu] = DmcModel.init(a, (core.MEM() >> 2) / dmcmem[iu], dmcparams[iu]);
        }
        return self;
    }

    // update and predict
    pub fn mix(self: *DmcForest, m: *core.Mixer) void {
        var i: usize = MODELS;
        // the slow models predict individually
        i -= 1;
        m.add(self.models[i].st() >> 3);
        i -= 1;
        m.add(self.models[i].st() >> 3);
        // the fast models are combined for better stability
        while (i > 0) {
            i -= 1;
            const p1 = self.models[i].st();
            i -= 1;
            const p2 = self.models[i].st();
            m.add((p1 + p2) >> 4);
        }

        // reset models when their structure can't adapt anymore
        // (the two slow models are never reset)
        if (core.bpos == 0) {
            var k: i32 = @as(i32, MODELS) - 3;
            while (k >= 0) : (k -= 1) {
                const ku: usize = @intCast(k);
                if (self.models[ku].isfull())
                    self.models[ku].resetstategraph(dmcparams[ku]);
            }
        }
    }

    pub fn deinit(self: *DmcForest) void {
        for (&self.models) |*mm| mm.deinit();
        self.alloc.destroy(self);
    }
};

// ================================== tests ==================================
// Structural smoke test. The bit-exact behaviour of this model is verified by
// a full differential test against the compiled C++ reference (see the porter
// report); this guards against regressions (add count / finite outputs).
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

test "dmcForest runs; 6 adds/bit; finite" {
    core.level = 1;
    core.init();
    core.resetState();
    const a = testing.allocator;
    core.buf.setsize(a, @intCast(core.MEM() * 8));
    defer core.buf.deinit();
    const forest = DmcForest.init(a);
    defer forest.deinit();
    const m = core.Mixer.init(a, 64, 0, 1, 0);
    defer m.deinit();
    const data = "hello world hello world 123 123 abcabcabc";
    for (data) |byte| {
        var k: i32 = 0;
        while (k < 8) : (k += 1) {
            driveBit((@as(i32, byte) >> @intCast(7 - k)) & 1);
            core.resetPredictions();
            forest.mix(m);
            try testing.expectEqual(@as(usize, 6), m.nx);
            for (core.model_predictions[0..core.prediction_index]) |p| try testing.expect(std.math.isFinite(p));
            while (m.nx & 7 != 0) {
                m.tx[m.nx] = 0;
                m.nx += 1;
            }
            m.update();
        }
    }
}
