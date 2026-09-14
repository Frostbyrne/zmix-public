//! auxparam — byte-channel prior-transmission instrument (branch-local).
//!
//! PPMd's 256-way posterior is the ONLY path by which its order-25 arena reaches
//! the coder (INDEX `wall-value-inversion` §4: removing the v[588] scalar reads
//! exactly 0 "because PPMd is already inside v[589]"). It reaches the byte
//! channel on exactly two paths, and BOTH are linear in the probability while
//! the coded cost is -log2 q:
//!
//!   IN  : lstm_layer.zig forwardNeuron -> dotV(aux, w),  aux = 2*p
//!   OUT : lstm.zig PRL fold            -> A_i = rho*2p_i + (1-rho)*2/V
//!
//! Mode 0: PPMd-only. Cost concentration + how much of PPMd's centered
//!         log-prior each fold form delivers, resolved BY COST BUCKET.
//! Mode 1: real `Lstm` arms, identical shape/capacity/init, differing only in
//!         the aux transform and/or the output fold form. The fold is applied
//!         inside `Lstm.perceive` before normalization, so each arm trains
//!         THROUGH its fold exactly as the shipped PRL does.
//!
//! bits/byte of the byte channel IS its coded cost: bisecting a distribution
//! costs exactly -log2 q(byte).
//!
//! Read-only on the substrate. No fleet, no lock.
const std = @import("std");
const PPMDLex = @import("models/ppmd.zig").PPMDLex;
const Lstm = @import("lstmfast").Lstm;

fn l2(x: f64) f64 {
    return std.math.log2(x);
}

const Arm = struct {
    name: []const u8,
    xf: u8, // 0 raw 2p (shipped) | 1 stretch | 2 log2 | 3 zero | 4 permuted raw
    // 5 = perm-invariant summary scalars | 6 = prior sorted desc | 7 = fresh perm/byte
    fold: u8, // 0 none | 1 mixture(par) == shipped PRL | 2 power | 3 clamped power
    par: f32,
    clamp: f32 = 32.0, // form 3: log2 floor, A = exp2(par*max(log2 p, -clamp))
    lstm: *Lstm = undefined,
    aux: []f32 = undefined,
    bits: f64 = 0,
    bits_tail: f64 = 0,
    seg: [10]f64 = .{0} ** 10,
    sortbuf: []f32 = undefined,
};

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    const a = gpa.allocator();
    const argv = try std.process.argsAlloc(a);
    if (argv.len < 5) {
        std.debug.print("usage: auxparam <stream> <nbytes> <arena_mb> <mode> [cells] [lr] [horizon] [skip]\n", .{});
        return error.Usage;
    }
    const path = argv[1];
    const nbytes = try std.fmt.parseInt(usize, argv[2], 10);
    const arena_mb = try std.fmt.parseInt(u32, argv[3], 10);
    const mode = try std.fmt.parseInt(u8, argv[4], 10);
    const cells: usize = if (argv.len > 5) try std.fmt.parseInt(usize, argv[5], 10) else 170;
    const lr: f32 = if (argv.len > 6) try std.fmt.parseFloat(f32, argv[6]) else 0.03;
    const horizon: usize = if (argv.len > 7) try std.fmt.parseInt(usize, argv[7], 10) else 128;
    const skip: u64 = if (argv.len > 8) try std.fmt.parseInt(u64, argv[8], 10) else 0;

    const f = try std.fs.cwd().openFile(path, .{});
    defer f.close();
    if (skip != 0) try f.seekTo(skip);
    const buf = try a.alloc(u8, nbytes);
    defer a.free(buf);
    const n = try f.readAll(buf);

    var vocab: [256]bool = .{false} ** 256;
    for (buf[0..n]) |b| vocab[b] = true;
    var vs: u32 = 0;
    for (vocab) |v| if (v) {
        vs += 1;
    };
    var byte_map: [256]u32 = .{0} ** 256;
    {
        var off: u32 = 0;
        for (0..256) |i| {
            byte_map[i] = off;
            if (vocab[i]) off += 1;
        }
    }
    const slot_byte = try a.alloc(u8, vs);
    for (0..256) |i| if (vocab[i]) {
        slot_byte[byte_map[i]] = @intCast(i);
    };

    std.debug.print("substrate {s} skip={d} bytes={d}  vocab={d} cells={d} lr={d} H={d} arena={d}MB mode={d}\n", .{ path, skip, n, vs, cells, lr, horizon, arena_mb, mode });

    var bit_context: u32 = 1;
    const order = @import("build_options").ppm_order;
    const ppmd = PPMDLex.create(a, &bit_context, order, arena_mb, &vocab);
    defer ppmd.destroy();

    // vocab-compacted prior + its log2, rebuilt each byte, shared by all arms
    const pv = try a.alloc(f32, vs);
    const lp = try a.alloc(f32, vs);

    // ---- arms ------------------------------------------------------------
    // arms from argv[9]: comma-separated  name/xf/fold/par
    const default_spec = "ship/0/1/0.15,bare/3/0/0,perm/4/1/0.15,mix35/0/1/0.35,mix60/0/1/0.60,mix85/0/1/0.85,mix100/0/1/1.0,pow30/0/2/0.30,pow50/0/2/0.50,pow70/0/2/0.70,pow90/0/2/0.90";
    const spec_str: []const u8 = if (argv.len > 9) argv[9] else default_spec;
    var narm: usize = 0;
    {
        var it = std.mem.splitScalar(u8, spec_str, ',');
        while (it.next()) |_| narm += 1;
    }
    const SPEC = try a.alloc(Arm, narm);
    {
        var it = std.mem.splitScalar(u8, spec_str, ',');
        var i: usize = 0;
        while (it.next()) |tok| : (i += 1) {
            var f4 = std.mem.splitScalar(u8, tok, '/');
            const nm = f4.next().?;
            const xf = try std.fmt.parseInt(u8, f4.next().?, 10);
            const fo = try std.fmt.parseInt(u8, f4.next().?, 10);
            const pa = try std.fmt.parseFloat(f32, f4.next().?);
            const cl: f32 = if (f4.next()) |c| try std.fmt.parseFloat(f32, c) else 32.0;
            SPEC[i] = .{ .name = nm, .xf = xf, .fold = fo, .par = pa, .clamp = cl };
        }
    }
    const arms: []Arm = try a.alloc(Arm, if (mode >= 1) SPEC.len else 0);
    for (arms, 0..) |*arm, i| {
        arm.* = SPEC[i];
        var seed_prng = std.Random.DefaultPrng.init(0);
        var rng = seed_prng.random();
        arm.lstm = Lstm.init(a, vs, vs, cells, 1, horizon, lr, 10.0, &rng);
        arm.lstm.setPriorFold(arm.fold, arm.par);
        arm.lstm.setPriorClamp(arm.clamp);
        arm.lstm.setPriorVecs(pv, lp);
        arm.aux = try a.alloc(f32, vs);
        arm.sortbuf = try a.alloc(f32, vs);
        @memset(arm.aux, 0);
    }
    const perm = try a.alloc(u32, vs);
    {
        for (0..vs) |i| perm[i] = @intCast(i);
        var pr = std.Random.DefaultPrng.init(12345);
        const r = pr.random();
        var i: usize = vs;
        while (i > 1) {
            i -= 1;
            const j = r.uintLessThan(usize, i + 1);
            const t = perm[i];
            perm[i] = perm[j];
            perm[j] = t;
        }
    }

    // ---- analytics --------------------------------------------------------
    const NB = 24;
    var hn: [NB]u64 = .{0} ** NB; // bytes in cost bucket
    var hc: [NB]f64 = .{0} ** NB; // cost in bucket
    var ha: [NB]f64 = .{0} ** NB; // available centered log-prior
    var hm: [NB]f64 = .{0} ** NB; // delivered by MIX(0.15)
    var ppmd_bits: f64 = 0;
    var ppmd_bits_tail: f64 = 0;
    var ns: usize = 0;
    var nst: usize = 0;
    const RHO: f64 = 0.15;

    var perm_prng = std.Random.DefaultPrng.init(777);
    var permrng = perm_prng.random();

    const tail_start = n - n / 4;
    const t0 = std.time.milliTimestamp();

    var pos: usize = 0;
    while (pos < n) : (pos += 1) {
        const b: u32 = buf[pos];
        var j: i32 = 7;
        while (j >= 0) : (j -= 1) {
            const bit: i32 = @intCast((b >> @intCast(j)) & 1);
            _ = ppmd.predict();
            const byte_update = bit_context >= 128;
            ppmd.perceive(bit);
            bit_context += bit_context + @as(u32, @intCast(bit));
            if (bit_context >= 256) bit_context -= 256;
            if (!byte_update) continue;

            ppmd.byteUpdate(); // bit_context holds the completed byte
            const bp = ppmd.bytePredict();
            for (0..vs) |s| {
                const v = bp[slot_byte[s]];
                pv[s] = v;
                lp[s] = @floatCast(l2(@max(@as(f64, v), 1e-30)));
            }

            const has_next = pos + 1 < n;
            const nb: usize = if (has_next) buf[pos + 1] else 0;
            const in_tail = (pos + 1) >= tail_start;

            if (has_next) {
                const ts = byte_map[nb];
                const cost = -@as(f64, lp[ts]);
                ppmd_bits += cost;
                ns += 1;
                if (in_tail) {
                    ppmd_bits_tail += cost;
                    nst += 1;
                }
                const bk: usize = @min(NB - 1, @as(usize, @intFromFloat(@max(0.0, cost))));
                hn[bk] += 1;
                hc[bk] += cost;
                // available vs delivered centered log-prior
                var mlp: f64 = 0;
                var mmix: f64 = 0;
                const fl: f64 = (1.0 - RHO) / @as(f64, @floatFromInt(vs));
                for (0..vs) |s| {
                    mlp += @as(f64, lp[s]);
                    mmix += l2(RHO * @as(f64, pv[s]) + fl);
                }
                mlp /= @as(f64, @floatFromInt(vs));
                mmix /= @as(f64, @floatFromInt(vs));
                ha[bk] += @as(f64, lp[ts]) - mlp;
                hm[bk] += l2(RHO * @as(f64, pv[ts]) + fl) - mmix;
            }

            for (arms) |*arm| {
                for (0..vs) |s| {
                    const src: usize = if (arm.xf == 4) perm[s] else s;
                    const p = pv[src];
                    arm.aux[s] = switch (arm.xf) {
                        0, 4 => 2.0 * p,
                        1 => blk: {
                            const pc: f64 = @min(@max(@as(f64, p), 1e-9), 1.0 - 1e-9);
                            const st = std.math.log(f64, std.math.e, pc / (1.0 - pc));
                            break :blk @floatCast(@min(@max(st, -16.0), 16.0) / 16.0);
                        },
                        2 => @max(lp[src], -32.0) / 16.0 + 1.0,
                        else => 0.0,
                    };
                }
                // xf=5: permutation-invariant SUMMARY only (few scalars, rest 0).
                // xf=6: the prior SORTED descending -- exactly the permutation-
                //       invariant content of the aux, in canonical order.
                // Both test whether the V-wide aux transmits symbol IDENTITY or
                // only PPMd's confidence.  `perm` says identity is unused; these
                // say how few numbers the same value is worth.
                if (arm.xf == 7) {
                    // FRESH permutation every byte: no fixed relabeling can
                    // compensate (a fixed perm is undoable since W*(Pi p) =
                    // (W Pi)*p, so xf=4 tests nothing about identity).
                    for (0..vs) |s| arm.sortbuf[s] = 2.0 * pv[s];
                    var k: usize = vs;
                    while (k > 1) {
                        k -= 1;
                        const jj = permrng.uintLessThan(usize, k + 1);
                        const t = arm.sortbuf[k];
                        arm.sortbuf[k] = arm.sortbuf[jj];
                        arm.sortbuf[jj] = t;
                    }
                    for (0..vs) |s| arm.aux[s] = arm.sortbuf[s];
                }
                if (arm.xf == 5 or arm.xf == 6) {
                    var srt = arm.sortbuf;
                    for (0..vs) |s| srt[s] = pv[s];
                    std.mem.sort(f32, srt[0..vs], {}, comptime std.sort.desc(f32));
                    if (arm.xf == 6) {
                        for (0..vs) |s| arm.aux[s] = 2.0 * srt[s];
                    } else {
                        var ent: f32 = 0;
                        for (0..vs) |s| {
                            const q = pv[s];
                            if (q > 0) ent -= q * lp[s];
                        }
                        @memset(arm.aux, 0);
                        arm.aux[0] = ent / 8.0;
                        arm.aux[1] = 2.0 * srt[0];
                        arm.aux[2] = 2.0 * (srt[0] - srt[1]);
                        arm.aux[3] = @max(@as(f32, @floatCast(l2(@max(@as(f64, srt[0]), 1e-30)))), -16.0) / 8.0;
                        if (vs > 4) arm.aux[4] = 2.0 * srt[2];
                        if (vs > 5) arm.aux[5] = 2.0 * srt[3];
                    }
                }
                arm.lstm.setInput(arm.aux);
                const out = arm.lstm.perceive(byte_map[b]);
                if (has_next) {
                    const q: f64 = @floatCast(out[byte_map[nb]]);
                    const c2 = -l2(@max(q, 1e-30));
                    arm.bits += c2;
                    if (in_tail) arm.bits_tail += c2;
                    arm.seg[@min(9, (pos + 1) * 10 / n)] += c2;
                }
            }
            bit_context = 1;
        }
        if (pos % 100000 == 0 and pos > 0) {
            std.debug.print("  ..{d}  {d}ms  ppmd={d:.4}\n", .{ pos, std.time.milliTimestamp() - t0, ppmd_bits / @as(f64, @floatFromInt(ns)) });
        }
    }

    const N: f64 = @floatFromInt(ns);
    const NT: f64 = @floatFromInt(@max(nst, 1));
    std.debug.print("\n==== PPMd order-{d} own posterior ====\nall {d:.5} b/B ({d})   tail {d:.5} b/B ({d})\n", .{ order, ppmd_bits / N, ns, ppmd_bits_tail / NT, nst });

    std.debug.print("\n==== cost concentration + fold delivery, BY COST BUCKET ====\n", .{});
    std.debug.print("  cost   bytes%%  cost%%   avail_bits  mix0.15_bits  delivered%%\n", .{});
    for (0..NB) |k| {
        if (hn[k] == 0) continue;
        const c: f64 = @floatFromInt(hn[k]);
        const av = ha[k] / c;
        const mx = hm[k] / c;
        std.debug.print("  [{d:>2},{d:>2}) {d:>7.3} {d:>7.3}  {d:>10.3}  {d:>12.3}  {d:>9.2}\n", .{
            k, k + 1, 100.0 * c / N, 100.0 * hc[k] / ppmd_bits, av, mx, 100.0 * mx / av,
        });
    }
    // cumulative cost share above thresholds
    for ([_]usize{ 2, 4, 6, 8 }) |thr| {
        var cnt: f64 = 0;
        var cst: f64 = 0;
        for (thr..NB) |k| {
            cnt += @floatFromInt(hn[k]);
            cst += hc[k];
        }
        std.debug.print("  >= {d:>2} bits: {d:.3}%% of bytes carry {d:.3}%% of PPMd's cost\n", .{ thr, 100.0 * cnt / N, 100.0 * cst / ppmd_bits });
    }

    if (mode >= 1) {
        std.debug.print("\n==== byte-channel arms (bits/byte = the channel's coded cost) ====\n", .{});
        var base_tail: f64 = 0;
        for (arms, 0..) |arm, i| {
            const tl = arm.bits_tail / NT;
            if (i == 0) base_tail = tl;
            std.debug.print("  {s:<9} xf={d} fold={d} par={d:.2} C={d:.0}   all={d:.5}  tail={d:.5}  d_tail={d:.5}\n", .{ arm.name, arm.xf, arm.fold, arm.par, arm.clamp, arm.bits / N, tl, tl - base_tail });
        }
    }
    if (mode >= 1) {
        // depth ladder: per-decile delta vs arm 0, the falsifier for "does the
        // form gap GROW with depth" (the shipped rho knob was 1 B apart at 20m
        // and 5,651 B apart at 100MB, so a small-tier verdict is uninformative).
        const seg_n: f64 = @as(f64, @floatFromInt(n)) / 10.0;
        std.debug.print("\n==== depth ladder: bits/byte by decile (d = vs arm 0) ====\n", .{});
        std.debug.print("  arm        ", .{});
        for (0..10) |d| std.debug.print("    D{d}   ", .{d + 1});
        std.debug.print("\n", .{});
        for (arms) |arm| {
            std.debug.print("  {s:<9} ", .{arm.name});
            for (0..10) |d| std.debug.print(" {d:>7.4}", .{arm.seg[d] / seg_n});
            std.debug.print("\n", .{});
        }
        std.debug.print("  --- delta vs arm 0 ---\n", .{});
        for (arms[1..]) |arm| {
            std.debug.print("  {s:<9} ", .{arm.name});
            for (0..10) |d| std.debug.print(" {d:>7.4}", .{(arm.seg[d] - arms[0].seg[d]) / seg_n});
            std.debug.print("\n", .{});
        }
    }
    std.debug.print("\nelapsed {d} ms\n", .{std.time.milliTimestamp() - t0});
}
