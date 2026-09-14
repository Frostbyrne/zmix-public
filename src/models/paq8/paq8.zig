//! PAQ8 integration layer — contextModel2 + Predictor::update + the PAQ8 Model
//! wrapper, ported from cmix models/paq8.cpp (~8101-8383). Ties every ported
//! model into the shared paq8 Mixer and exposes PAQ8 to zmix's predictor as an
//! auxiliary model (like FXCM: lazy paq-eval, Predict returns model_predictions).
const std = @import("std");
const core = @import("core.zig");
const maps = @import("maps.zig");
const tm = @import("text_model.zig");
const mm = @import("match_models.zig");
const dmc = @import("dmc.zig");
const mf = @import("model_fns.zig");
const ia = @import("image_audio.zig");
const je = @import("jpeg_exe.zig");
const model_mod = @import("../../model.zig");
const Model = model_mod.Model;

const stretch = core.stretch;
const squash = core.squash;
const ilog = core.ilog;

inline fn isAlpha(b: u8) bool {
    return (b >= 'a' and b <= 'z') or (b >= 'A' and b <= 'Z');
}
inline fn toLower(b: u8) u8 {
    return if (b >= 'A' and b <= 'Z') b + 32 else b;
}

// Big-endian 4-byte pack as cmix's `int` bit-pattern (buf(hi)<<24 | ... | buf(lo)).
// Computed in u32 so it never overflows i32; the reinterpret matches the
// reference's signed-overflow wrap exactly.
inline fn rd4be(b4: i32, b3: i32, b2: i32, b1: i32) i32 {
    const v: u32 = (@as(u32, @intCast(b4)) << 24) | (@as(u32, @intCast(b3)) << 16) |
        (@as(u32, @intCast(b2)) << 8) | @as(u32, @intCast(b1));
    return @bitCast(v);
}

// The audioModel calls recordModel via a callback (recordModel lives in model_fns).
var g_models: ?*mf.Models = null;
fn recordCb(m: *core.Mixer, ft: i32, stats: ?*core.ModelStats) void {
    if (g_models) |gm| gm.record.predict(m, ft, stats);
}

// ============================ contextModel2 =============================
const Cm2 = struct {
    cm: maps.ContextMap2,
    textModel: *tm.TextModel,
    matchModel: *mm.MatchModel,
    sparseMatchModel: *mm.SparseMatchModel,
    dmcforest: *dmc.DmcForest,
    rcm7: maps.RunContextMap,
    rcm9: maps.RunContextMap,
    rcm10: maps.RunContextMap,
    sm: [2]core.StateMap32,
    m: *core.Mixer,
    cxt: [16]u32 = .{0} ** 16,
    ft2: i32 = core.FT_DEFAULT,
    filetype: i32 = core.FT_DEFAULT,
    size: i32 = 0,
    info: i32 = 0,
    models: mf.Models,
    img: ia.ImgModel,
    aud: ia.AudioModel,
    a: std.mem.Allocator,

    fn init(a: std.mem.Allocator) *Cm2 {
        const self = a.create(Cm2) catch unreachable;
        self.* = .{
            .cm = maps.ContextMap2.init(a, core.MEM() * 16, 10),
            .textModel = tm.TextModel.init(a, core.MEM() * 16),
            .matchModel = mm.MatchModel.init(a, core.MEM() * 2),
            .sparseMatchModel = mm.SparseMatchModel.init(a, core.MEM() / 2),
            .dmcforest = dmc.DmcForest.init(a),
            .rcm7 = maps.RunContextMap.init(a, @intCast(core.MEM())),
            .rcm9 = maps.RunContextMap.init(a, @intCast(core.MEM())),
            .rcm10 = maps.RunContextMap.init(a, @intCast(core.MEM())),
            .sm = .{ core.StateMap32.init(a, 256, true), core.StateMap32.init(a, 256 * 256, true) },
            .m = core.Mixer.init(a, core.NUM_INPUTS, 77472, core.NUM_SETS, 32),
            .models = mf.Models.init(a),
            .img = ia.ImgModel.init(a),
            .aud = ia.AudioModel.init(a),
            .a = a,
        };
        g_models = &self.models;
        return self;
    }

    fn run(self: *Cm2, stats: ?*core.ModelStats) i32 {
        const m = self.m;
        const c0: i32 = core.c0;
        const bpos: i32 = core.bpos;
        const c4: u32 = core.c4;

        // Parse filetype and size from block headers. cmix computes these as
        // `int`, where `buf(4)<<24` is a bit-pattern that overflows a signed int
        // on data bytes >=128 (defined-wrap in the reference). zmix feeds UNFRAMED
        // bytes here, so size/info take arbitrary values — compute every step with
        // wrapping / u32-bit-pattern semantics so the arithmetic is well-defined
        // (identical result to the reference's wrap; safe under ReleaseSafe).
        if (bpos == 0) {
            self.size -%= 1;
            core.blpos += 1;
            if (self.size == -1) {
                self.info = 0;
                self.ft2 = core.buf.get(1);
            }
            if (self.size == -5 and !core.HasInfo(self.ft2)) {
                self.size = rd4be(core.buf.get(4), core.buf.get(3), core.buf.get(2), core.buf.get(1));
                core.blpos = 0;
            }
            if (self.size == -9) {
                self.size = rd4be(core.buf.get(8), core.buf.get(7), core.buf.get(6), core.buf.get(5));
                self.info = rd4be(core.buf.get(4), core.buf.get(3), core.buf.get(2), core.buf.get(1));
                core.blpos = 0;
                if (self.ft2 == core.FT_TEXT and self.info != 0) self.size = self.info -% 8;
            }
            // zmix feeds UNFRAMED bytes (no WRT preprocessor yet), so the parsed
            // ft2/size carry no real filetype — force DEFAULT so paq8 never enters
            // the image/jpeg/audio filetype dispatch. On ordinary text that
            // dispatch mis-fires (integer overflow in the image pixel accumulators
            // AND pure-noise predictions that hurt the ratio). Revisit once the
            // preprocessor frames the input with genuine block headers.
            self.filetype = core.FT_DEFAULT;
            if (stats) |s| s.Type = self.filetype;
        }

        m.update();
        m.add(64);

        if (bpos == 0) {
            const B: u8 = @intCast(c4 & 0xFF);
            // cxt[] is U32 in cmix: combine64 (U64) is truncated on assignment.
            self.cxt[15] = if (isAlpha(B)) @truncate(maps.combine64(self.cxt[15], toLower(B))) else 0;
            self.cm.set(self.cxt[15]);
            var i: usize = 14;
            while (i > 0) : (i -= 1) self.cxt[i] = @truncate(maps.combine64(self.cxt[i - 1], B));
            i = 0;
            while (i < 7) : (i += 1) self.cm.set(self.cxt[i]);
            self.rcm7.set(self.cxt[7]);
            self.cm.set(self.cxt[8]);
            self.rcm9.set(self.cxt[10]);
            self.rcm10.set(self.cxt[12]);
            self.cm.set(self.cxt[14]);
        }
        m.add((stretch(self.sm[0].p(c0, 1023)) + 1) >> 1);
        m.add((stretch(self.sm[1].p(c0 | (core.buf.get(1) << 8), 1023)) + 1) >> 1);
        var order: i32 = self.cm.mix(m);
        _ = self.rcm7.mix(m);
        _ = self.rcm9.mix(m);
        _ = self.rcm10.mix(m);

        const ismatch: i32 = ilog(@intCast(self.matchModel.predict(m, stats)));
        if (self.filetype == core.FT_IMAGE1) {
            self.img.im1.im1bitModel(m, self.info);
            return m.p();
        }
        if (self.filetype == core.FT_IMAGE4) {
            self.img.im4.im4bitModel(m, self.info);
            return m.p();
        }
        if (self.filetype == core.FT_IMAGE8) {
            self.img.im8.im8bitModel(m, self.info, stats, 0);
            return m.p();
        }
        if (self.filetype == core.FT_IMAGE8GRAY) {
            self.img.im8.im8bitModel(m, self.info, stats, 1);
            return m.p();
        }
        if (self.filetype == core.FT_IMAGE24) {
            self.img.im24.im24bitModel(m, self.info, stats, 0);
            return m.p();
        }
        if (self.filetype == core.FT_IMAGE32) {
            self.img.im24.im24bitModel(m, self.info, stats, 1);
            return m.p();
        }
        if ((self.filetype != core.FT_EXE and je.jpegModel(m) != 0) or
            (self.size > 0 and self.img.imgModel(m, stats) != 0) or
            self.aud.audioModel(m, stats, recordCb) != 0)
        {
            return m.p();
        }

        _ = self.sparseMatchModel.predict(m, stats);
        self.models.sparse.predict(m, ismatch, order);
        self.models.sparse1.predict(m, ismatch, order);
        self.models.distance.predict(m);
        self.models.pic.predict(m);
        self.models.record.predict(m, self.filetype, stats);
        self.models.record1.predict(m);
        self.models.word.predict(m);
        self.models.nest.predict(m);
        self.models.indirect.predict(m);
        self.dmcforest.mix(m);
        self.models.xml.predict(m, stats);
        self.textModel.predict(m, stats);
        _ = je.exeModel(m, true, stats);
        self.models.linear.predict(m);

        m.set(@intCast((@max(0, order - 3) << 3) | bpos), 64);
        order = @max(0, order - 5);

        const d: u32 = @as(u32, @intCast(c0)) << @intCast(8 - bpos);
        var c: u32 = (d +% (if (bpos == 1) core.b3 / 2 else 0)) & 192;
        if (bpos == 0) c = (core.words *% 16) & 192;

        const c1: u32 = @intCast(core.buf.get(1));

        m.set(@intCast(order * 256 +% @as(i32, @bitCast((core.w4 & 240) +% (core.b2 >> 4)))), 1536);
        m.set(@intCast(order * 256 +% @as(i32, @bitCast((core.w4 & 3) *% 64 +% ((core.words >> 1) & 63)))), 1536);
        m.set(@intCast(bpos * 256 +% @as(i32, @bitCast(c1))), 2048);
        m.set(@intCast(@min(bpos, 5) * 256 +% @as(i32, @bitCast((core.tt & 63) +% c))), 1536);
        m.set(@intCast(order * 256 +% @as(i32, @bitCast(((d | (c1 >> @intCast(bpos))) & 248) +% @as(u32, @intCast(bpos))))), 1536);
        m.set(@intCast(bpos * 256 +% @as(i32, @bitCast((((core.words << @intCast(bpos)) & 255) >> @intCast(bpos)) | (d & 255)))), 2048);
        const prq: u32 = @as(u32, @bitCast(core.last_prediction)) / 16;
        m.set(@intCast(prq), 256);
        m.set(c0, 256);

        return m.p();
    }
};

// APM group for the filetype-specific calibration switch.
const Predictor = struct {
    pr: i32 = 2048,
    cm2: *Cm2,
    t_apm: [4]core.APM,
    t_apm1: [3]core.APM1,
    ic_apm: [4]core.APM,
    ic_apm1: [2]core.APM1,
    ip_apm: [4]core.APM,
    ip_apm1: [2]core.APM1,
    ig_apm: [3]core.APM,
    gen_apm1: [7]core.APM1,
    stats: core.ModelStats = .{},

    fn init(a: std.mem.Allocator) *Predictor {
        const self = a.create(Predictor) catch unreachable;
        self.* = .{
            .cm2 = Cm2.init(a),
            .t_apm = .{ core.APM.init(a, 0x10000), core.APM.init(a, 0x10000), core.APM.init(a, 0x10000), core.APM.init(a, 0x10000) },
            .t_apm1 = .{ core.APM1.init(a, 0x10000), core.APM1.init(a, 0x10000), core.APM1.init(a, 0x10000) },
            .ic_apm = .{ core.APM.init(a, 0x1000), core.APM.init(a, 0x10000), core.APM.init(a, 0x10000), core.APM.init(a, 0x10000) },
            .ic_apm1 = .{ core.APM1.init(a, 0x10000), core.APM1.init(a, 0x10000) },
            .ip_apm = .{ core.APM.init(a, 0x1000), core.APM.init(a, 0x10000), core.APM.init(a, 0x10000), core.APM.init(a, 0x10000) },
            .ip_apm1 = .{ core.APM1.init(a, 0x10000), core.APM1.init(a, 0x10000) },
            .ig_apm = .{ core.APM.init(a, 0x1000), core.APM.init(a, 0x10000), core.APM.init(a, 0x10000) },
            .gen_apm1 = .{ core.APM1.init(a, 0x2000), core.APM1.init(a, 0x10000), core.APM1.init(a, 0x10000), core.APM1.init(a, 0x10000), core.APM1.init(a, 0x10000), core.APM1.init(a, 0x10000), core.APM1.init(a, 0x10000) },
        };
        return self;
    }

    fn update(self: *Predictor) void {
        const st = &self.stats;
        core.c0 = core.c0 + core.c0 + core.y;
        st.Misses = st.Misses *% 2 +% @as(u64, @intFromBool((self.pr >> 11) != core.y));
        if (core.c0 >= 256) {
            core.buf.at(@intCast(core.pos)).* = @intCast(core.c0 & 0xff);
            core.pos += 1;
            core.c0 -= 256;
            core.c4 = (core.c4 << 8) +% @as(u32, @intCast(core.c0));
            var i: i32 = @intCast(core.WRT_mpw[@intCast(core.c0 >> 4)]);
            core.w4 = core.w4 *% 4 +% @as(u32, @bitCast(i));
            if (core.b2 == 3) i = 2;
            core.w5 = core.w5 *% 4 +% @as(u32, @bitCast(i));
            core.b3 = core.b2;
            core.b2 = @intCast(core.c0);
            core.x4 = core.x4 *% 256 +% @as(u32, @intCast(core.c0));
            x5 = (x5 << 8) +% @as(u32, @intCast(core.c0));
            if (core.c0 == '.' or core.c0 == '!' or core.c0 == '?' or core.c0 == '/' or core.c0 == ')') {
                core.w5 = (core.w5 << 8) | 0x3ff;
                core.f4 = (core.f4 & 0xfffffff0) +% 2;
                x5 = (x5 << 8) +% @as(u32, @intCast(core.c0));
                core.x4 = core.x4 *% 256 +% @as(u32, @intCast(core.c0));
                if (core.c0 != '!') {
                    core.w4 |= 12;
                    core.tt = (core.tt & 0xfffffff8) +% 1;
                    core.b3 = '.';
                }
            }
            if (core.c0 == 32) core.c0 -= 1;
            core.tt = core.tt *% 8 +% core.WRT_mtt[@intCast(core.c0 >> 4)];
            core.f4 = core.f4 *% 16 +% @as(u32, @intCast(core.c0 >> 4));
            core.c0 = 1;
        }
        core.bpos = (core.bpos + 1) & 7;
        core.grp0 = if (core.bpos > 0) core.AsciiGroupC0[@intCast((@as(i32, 1) << @intCast(core.bpos)) - 2 + (core.c0 & ((@as(i32, 1) << @intCast(core.bpos)) - 1)))] else 0;

        var pr0 = self.cm2.run(st);
        core.addPrediction(pr0);

        const c0u: i32 = core.c0;
        var pr1: i32 = undefined;
        var pr2: i32 = undefined;
        var pr3: i32 = undefined;
        switch (st.Type) {
            core.FT_TEXT => {
                const limit: i32 = @intCast(@as(u32, 0x3FF) >> @intCast(@as(u32, @intFromBool(core.blpos < 0xFFF)) * 2));
                self.pr = self.t_apm[0].p(pr0, (c0u << 8) | @as(i32, @intCast(st.Text.mask & 0xF)) | (@as(i32, @intCast(st.Misses & 0xF)) << 4), limit);
                core.addPrediction(self.pr);
                pr1 = self.t_apm[1].p(pr0, @bitCast(maps.finalize64(maps.hash(.{ @as(u64, @intCast(core.bpos)), @as(u64, st.Misses & 3), @as(u64, core.c4 & 0xffff), @as(u64, st.Text.mask >> 4) }), 16)), limit);
                core.addPrediction(pr1);
                pr2 = self.t_apm[2].p(pr0, @bitCast(maps.finalize64(maps.hash(.{ @as(u64, @intCast(c0u)), @as(u64, st.Match.expectedByte), @as(u64, @min(3, maps.ilog2(@as(u32, @bitCast(st.Match.length)) + 1))) }), 16)), limit);
                core.addPrediction(pr2);
                pr3 = self.t_apm[3].p(pr0, @bitCast(maps.finalize64(maps.hash(.{ @as(u64, @intCast(c0u)), @as(u64, core.c4 & 0xffff), @as(u64, st.Text.firstLetter) }), 16)), limit);
                core.addPrediction(pr3);
                pr0 = (pr0 + pr1 + pr2 + pr3 + 2) >> 2;
                core.addPrediction(pr0);
                pr1 = self.t_apm1[0].p(pr0, @bitCast(maps.finalize64(maps.hash(.{ @as(u64, st.Match.expectedByte), @as(u64, @min(3, maps.ilog2(@as(u32, @bitCast(st.Match.length)) + 1))), @as(u64, core.c4 & 0xff) }), 16)), 7);
                core.addPrediction(pr1);
                pr2 = self.t_apm1[1].p(self.pr, @bitCast(maps.finalize64(maps.hash(.{ @as(u64, @intCast(c0u)), @as(u64, core.c4 & 0x00ffffff) }), 16)), 6);
                core.addPrediction(pr2);
                pr3 = self.t_apm1[2].p(self.pr, @bitCast(maps.finalize64(maps.hash(.{ @as(u64, @intCast(c0u)), @as(u64, core.c4 & 0xffffff00) }), 16)), 6);
                core.addPrediction(pr3);
                self.pr = (self.pr + pr1 + pr2 + pr3 + 2) >> 2;
                core.addPrediction(self.pr);
                self.pr = (self.pr + pr0 + 1) >> 1;
                core.addPrediction(self.pr);
            },
            else => {
                // Generic (DEFAULT and any non-image type on text input).
                self.pr = self.gen_apm1[0].p(pr0, (@as(i32, @min(3, maps.ilog2(@as(u32, @bitCast(st.Match.length)) + 1))) << 11) | (c0u << 3) | @as(i32, @intCast(st.Misses & 0x7)), 7);
                core.addPrediction(self.pr);
                const ctx1: u16 = @intCast((c0u | (core.buf.get(1) << 8)) & 0xffff);
                const ctx2: u16 = @truncate(@as(u32, @intCast(c0u)) ^ maps.finalize64(maps.hash(.{@as(u64, core.c4 & 0xffff)}), 16));
                const ctx3: u16 = @truncate(@as(u32, @intCast(c0u)) ^ maps.finalize64(maps.hash(.{@as(u64, core.c4 & 0xffffff)}), 16));
                pr1 = self.gen_apm1[1].p(pr0, ctx1, 7);
                core.addPrediction(pr1);
                pr2 = self.gen_apm1[2].p(pr0, ctx2, 7);
                core.addPrediction(pr2);
                pr3 = self.gen_apm1[3].p(pr0, ctx3, 7);
                core.addPrediction(pr3);
                pr0 = (pr0 + pr1 + pr2 + pr3 + 2) >> 2;
                pr1 = self.gen_apm1[4].p(self.pr, (@as(i32, @intCast(st.Match.expectedByte)) << 8) | core.buf.get(1), 7);
                core.addPrediction(pr1);
                pr2 = self.gen_apm1[5].p(self.pr, ctx2, 7);
                core.addPrediction(pr2);
                pr3 = self.gen_apm1[6].p(self.pr, ctx3, 7);
                core.addPrediction(pr3);
                self.pr = (self.pr + pr1 + pr2 + pr3 + 2) >> 2;
                core.addPrediction(self.pr);
                self.pr = (self.pr + pr0 + 1) >> 1;
                core.addPrediction(self.pr);
            },
        }
        core.resetPredictions();
        core.last_prediction = self.pr;
    }
};

var x5: u32 = 0;

// ============================ PAQ8 Model wrapper =============================
pub const PAQ8 = struct {
    predictor: *Predictor,

    pub fn create(a: std.mem.Allocator, memory: i32) *PAQ8 {
        core.level = memory;
        core.init();
        // Re-seed the shared PRNG per predictor: core.initis guarded by
        // tables_ready so it seeds rnd only once, but rnd advances during a run
        // (ContextMap bit-history randomization). Without re-seeding, the second
        // predictor in a process inherits the first's advanced RNG and diverges.
        core.rnd.init();
        core.resetState();
        // model_predictions is a module-level array read on bit 0 before any
        // perceive; reset to the 0.5 init so the second predictor doesn't return
        // the first's final values.
        @memset(&core.model_predictions, 0.5);
        core.prediction_index = 0;
        mf.resetShared();
        ia.resetShared();
        x5 = 0;
        // Reset the ring buffer so setsize re-allocates from THIS predictor's
        // arena. Its `inited` guard otherwise keeps a dangling Array pointing at
        // the previous predictor's freed arena (in-process round-trip divergence).
        core.buf = .{};
        core.buf.setsize(a, @intCast(core.MEM() * 8));
        je.init(a);
        const self = a.create(PAQ8) catch unreachable;
        self.* = .{ .predictor = Predictor.init(a) };
        return self;
    }

    pub fn predict(self: *PAQ8) []const f32 {
        _ = self;
        return core.model_predictions[0..];
    }
    pub fn numOutputs(self: *PAQ8) usize {
        _ = self;
        return core.model_predictions.len;
    }
    pub fn perceive(self: *PAQ8, bit: i32) void {
        core.y = bit;
        self.predictor.update();
    }
    pub fn byteUpdate(self: *PAQ8) void {
        _ = self;
    }
    pub fn model(self: *PAQ8) Model {
        return .{ .ptr = self, .vtable = model_mod.vtableFor(PAQ8) };
    }
};
