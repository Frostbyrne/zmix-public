//! fxcm_v26 numeric primitives, ported from reference/fxcmv1_v26.cpp.
//!
//! fxcm uses its own 12-bit squash/stretch tables and an ilog table (distinct
//! from cmix's float Sigmoid). This is the foundation the fxcm model's mixer,
//! state maps, context maps and APM build on.
const std = @import("std");

pub var sqt: [4095]i16 = undefined; // squash table (note: 4095 entries, matches fxcm)
pub var strt: [4096]i16 = undefined; // stretch table
pub var ilog: [256]u8 = undefined;
pub var dt: [1024]i32 = undefined; // i -> 4096/(i+2), dt[1023]=1
var ready = false;

fn squashc(d: i32) i32 {
    if (d < -2047) return 1;
    if (d > 2047) return 4095;
    const df: f64 = @floatFromInt(d);
    var p: f64 = 1.0 / (1.0 + @exp(-df / 256.0));
    p *= 4096.0;
    var pi: i64 = @intFromFloat(@round(p));
    if (pi > 4095) pi = 4095;
    if (pi < 1) pi = 1;
    return @intCast(pi);
}

fn stretchc(p_in: i32) i32 {
    var p = p_in;
    if (p == 0) p = 1;
    // Match the reference's f32 intermediates: `float f = p/4096.0f;
    // float d = log(f/(1.0f-f))*256.0f;` (log promotes its f32 arg to double).
    const f: f32 = @as(f32, @floatFromInt(p)) / 4096.0;
    const d: f32 = @floatCast(@log(@as(f64, f / (1.0 - f))) * 256.0);
    var di: i64 = @intFromFloat(@round(@as(f64, d)));
    if (di > 2047) di = 2047;
    if (di < -2047) di = -2047;
    return @intCast(di);
}

pub inline fn squash(d: i32) i32 {
    if (d < -2047) return 1;
    if (d > 2047) return 4095;
    return sqt[@intCast(d + 2047)];
}

pub inline fn stretch(p: i32) i16 {
    return strt[@intCast(p)];
}

pub inline fn clp(z: i32) i16 {
    if (z < -2047) return -2047;
    if (z > 2047) return 2047;
    return @intCast(z);
}

pub inline fn clp1(z: i32) i16 {
    if (z < 0) return 0;
    if (z > 4095) return 4095;
    return @intCast(z);
}

pub fn init() void {
    if (ready) return;
    var p: i32 = 0;
    while (p <= 4095) : (p += 1) strt[@intCast(p)] = @intCast(stretchc(p));
    var d: i32 = -2047;
    while (d <= 2047) : (d += 1) sqt[@intCast(d + 2047)] = @intCast(squashc(d));
    // ilog. Reference `U8 ilog[256]` is zero-init and InitIlog only writes
    // 1..255, so ilog[0]==0; it IS read (e.g. rc/st table init at index 0).
    ilog[0] = 0;
    var xx: u32 = 14155776;
    var i: usize = 2;
    while (i < 257) : (i += 1) {
        xx +%= 774541002 / (@as(u32, @intCast(i)) * 2 - 1);
        ilog[i - 1] = @intCast(xx >> 24);
    }
    // dt
    var o: i32 = 2;
    var k: usize = 0;
    while (k < 1024) : (k += 1) {
        dt[k] = @divTrunc(4096, o);
        o += 1;
    }
    dt[1023] = 1;
    ready = true;
}
