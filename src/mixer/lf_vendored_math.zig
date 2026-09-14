//! Vendored, build-time-fixed libm: exp(f64) and tanhf(f32).
//!
//! WHY: the -Dglibc build resolves the two hot transcendentals on the machine
//! that RUNS the binary (glibc changed exp results at 2.28 and may again); one
//! ulp of drift anywhere in ~4.7e9 coder steps desyncs the arithmetic coder.
//! These ports compile the exact algorithms INTO the binary, so the numerics
//! are fixed at build time — deterministic across machines and glibc versions —
//! while keeping glibc-class speed (they ARE the modern glibc algorithms).
//!
//! Provenance (both: SPDX MIT OR Apache-2.0 WITH LLVM-exception):
//!   exp   — ARM optimized-routines v24.01 math/exp.c + math/exp_data.c,
//!           design by Szabolcs Nagy: 128-entry table of 2^(i/128) as
//!           (tail, scale-bits) pairs, top-12-bit argument screen, degree-5
//!           polynomial. This is bit-for-bit the table/poly/constants modern
//!           glibc (>= 2.28) ships in sysdeps/ieee754/dbl-64/e_exp{,_data}.c
//!           (verified against glibc master: all 256 tab words identical).
//!           Worst-case error 0.509 ulp with FMA (faithfully rounded).
//!   tanhf — ARM optimized-routines v24.01 pl/math/tanhf_2u6.c with the
//!           pl/math/expm1f_data.c polynomial: tanh(x) = expm1(2x)/(expm1(2x)+2)
//!           via an inlined, special-case-free expm1f (expm1f_1u6.c design).
//!           Worst-case error 2.58 ulp (documented in the source).
//!
//! FMA: @mulAdd is used exactly where glibc's -mfma ifunc build (__exp_fma —
//! selected on every FMA-capable x86-64) contracts. The pinned x86_64_v3
//! target lowers @mulAdd to vfmadd; on targets without FMA, compiler-rt fma
//! computes the same correctly-rounded values (slower, still deterministic).
//! No @setFloatMode(.optimized) here: every operation is strict IEEE.
const std = @import("std");

// ===================== exp(f64) =====================

const EXP_TABLE_BITS = 7;
const EXP_N = 1 << EXP_TABLE_BITS; // 128

// N/ln2 (exact: 0x1.71547652b82fep0 scaled by a power of two).
const exp_invln2N: f64 = 0x1.71547652b82fep0 * EXP_N;
// -ln2/N split into a 33-bit high part (kd*hi exact) and a low correction.
const exp_negln2hiN: f64 = -0x1.62e42fefa0000p-8;
const exp_negln2loN: f64 = -0x1.cf79abc9e3b3ap-47;
const exp_shift: f64 = 0x1.8p52;
// Degree-5 polynomial for exp(r)-1-r on |r| < ln2/256+eps:
// abs error 1.555*2^-66; total ulp error 0.509 with fma (0.511 without).
const exp_c2: f64 = 0x1.ffffffffffdbdp-2;
const exp_c3: f64 = 0x1.555555555543cp-3;
const exp_c4: f64 = 0x1.55555cf172b91p-5;
const exp_c5: f64 = 0x1.1111167a4d017p-7;

// tab[2i] = asuint64(tail_i), tab[2i+1] = asuint64(scale_i) - (i << 45), with
// 2^(i/128) = scale_i*(1 + tail_i) to ~2^-77; adding (k << 45) rebuilds the
// full 2^(k/128) exponent+mantissa bits in one integer add (see exp).
const exp_tab = [256]u64{
    0x0000000000000000, 0x3ff0000000000000,
    0x3c9b3b4f1a88bf6e, 0x3feff63da9fb3335,
    0xbc7160139cd8dc5d, 0x3fefec9a3e778061,
    0xbc905e7a108766d1, 0x3fefe315e86e7f85,
    0x3c8cd2523567f613, 0x3fefd9b0d3158574,
    0xbc8bce8023f98efa, 0x3fefd06b29ddf6de,
    0x3c60f74e61e6c861, 0x3fefc74518759bc8,
    0x3c90a3e45b33d399, 0x3fefbe3ecac6f383,
    0x3c979aa65d837b6d, 0x3fefb5586cf9890f,
    0x3c8eb51a92fdeffc, 0x3fefac922b7247f7,
    0x3c3ebe3d702f9cd1, 0x3fefa3ec32d3d1a2,
    0xbc6a033489906e0b, 0x3fef9b66affed31b,
    0xbc9556522a2fbd0e, 0x3fef9301d0125b51,
    0xbc5080ef8c4eea55, 0x3fef8abdc06c31cc,
    0xbc91c923b9d5f416, 0x3fef829aaea92de0,
    0x3c80d3e3e95c55af, 0x3fef7a98c8a58e51,
    0xbc801b15eaa59348, 0x3fef72b83c7d517b,
    0xbc8f1ff055de323d, 0x3fef6af9388c8dea,
    0x3c8b898c3f1353bf, 0x3fef635beb6fcb75,
    0xbc96d99c7611eb26, 0x3fef5be084045cd4,
    0x3c9aecf73e3a2f60, 0x3fef54873168b9aa,
    0xbc8fe782cb86389d, 0x3fef4d5022fcd91d,
    0x3c8a6f4144a6c38d, 0x3fef463b88628cd6,
    0x3c807a05b0e4047d, 0x3fef3f49917ddc96,
    0x3c968efde3a8a894, 0x3fef387a6e756238,
    0x3c875e18f274487d, 0x3fef31ce4fb2a63f,
    0x3c80472b981fe7f2, 0x3fef2b4565e27cdd,
    0xbc96b87b3f71085e, 0x3fef24dfe1f56381,
    0x3c82f7e16d09ab31, 0x3fef1e9df51fdee1,
    0xbc3d219b1a6fbffa, 0x3fef187fd0dad990,
    0x3c8b3782720c0ab4, 0x3fef1285a6e4030b,
    0x3c6e149289cecb8f, 0x3fef0cafa93e2f56,
    0x3c834d754db0abb6, 0x3fef06fe0a31b715,
    0x3c864201e2ac744c, 0x3fef0170fc4cd831,
    0x3c8fdd395dd3f84a, 0x3feefc08b26416ff,
    0xbc86a3803b8e5b04, 0x3feef6c55f929ff1,
    0xbc924aedcc4b5068, 0x3feef1a7373aa9cb,
    0xbc9907f81b512d8e, 0x3feeecae6d05d866,
    0xbc71d1e83e9436d2, 0x3feee7db34e59ff7,
    0xbc991919b3ce1b15, 0x3feee32dc313a8e5,
    0x3c859f48a72a4c6d, 0x3feedea64c123422,
    0xbc9312607a28698a, 0x3feeda4504ac801c,
    0xbc58a78f4817895b, 0x3feed60a21f72e2a,
    0xbc7c2c9b67499a1b, 0x3feed1f5d950a897,
    0x3c4363ed60c2ac11, 0x3feece086061892d,
    0x3c9666093b0664ef, 0x3feeca41ed1d0057,
    0x3c6ecce1daa10379, 0x3feec6a2b5c13cd0,
    0x3c93ff8e3f0f1230, 0x3feec32af0d7d3de,
    0x3c7690cebb7aafb0, 0x3feebfdad5362a27,
    0x3c931dbdeb54e077, 0x3feebcb299fddd0d,
    0xbc8f94340071a38e, 0x3feeb9b2769d2ca7,
    0xbc87deccdc93a349, 0x3feeb6daa2cf6642,
    0xbc78dec6bd0f385f, 0x3feeb42b569d4f82,
    0xbc861246ec7b5cf6, 0x3feeb1a4ca5d920f,
    0x3c93350518fdd78e, 0x3feeaf4736b527da,
    0x3c7b98b72f8a9b05, 0x3feead12d497c7fd,
    0x3c9063e1e21c5409, 0x3feeab07dd485429,
    0x3c34c7855019c6ea, 0x3feea9268a5946b7,
    0x3c9432e62b64c035, 0x3feea76f15ad2148,
    0xbc8ce44a6199769f, 0x3feea5e1b976dc09,
    0xbc8c33c53bef4da8, 0x3feea47eb03a5585,
    0xbc845378892be9ae, 0x3feea34634ccc320,
    0xbc93cedd78565858, 0x3feea23882552225,
    0x3c5710aa807e1964, 0x3feea155d44ca973,
    0xbc93b3efbf5e2228, 0x3feea09e667f3bcd,
    0xbc6a12ad8734b982, 0x3feea012750bdabf,
    0xbc6367efb86da9ee, 0x3fee9fb23c651a2f,
    0xbc80dc3d54e08851, 0x3fee9f7df9519484,
    0xbc781f647e5a3ecf, 0x3fee9f75e8ec5f74,
    0xbc86ee4ac08b7db0, 0x3fee9f9a48a58174,
    0xbc8619321e55e68a, 0x3fee9feb564267c9,
    0x3c909ccb5e09d4d3, 0x3feea0694fde5d3f,
    0xbc7b32dcb94da51d, 0x3feea11473eb0187,
    0x3c94ecfd5467c06b, 0x3feea1ed0130c132,
    0x3c65ebe1abd66c55, 0x3feea2f336cf4e62,
    0xbc88a1c52fb3cf42, 0x3feea427543e1a12,
    0xbc9369b6f13b3734, 0x3feea589994cce13,
    0xbc805e843a19ff1e, 0x3feea71a4623c7ad,
    0xbc94d450d872576e, 0x3feea8d99b4492ed,
    0x3c90ad675b0e8a00, 0x3feeaac7d98a6699,
    0x3c8db72fc1f0eab4, 0x3feeace5422aa0db,
    0xbc65b6609cc5e7ff, 0x3feeaf3216b5448c,
    0x3c7bf68359f35f44, 0x3feeb1ae99157736,
    0xbc93091fa71e3d83, 0x3feeb45b0b91ffc6,
    0xbc5da9b88b6c1e29, 0x3feeb737b0cdc5e5,
    0xbc6c23f97c90b959, 0x3feeba44cbc8520f,
    0xbc92434322f4f9aa, 0x3feebd829fde4e50,
    0xbc85ca6cd7668e4b, 0x3feec0f170ca07ba,
    0x3c71affc2b91ce27, 0x3feec49182a3f090,
    0x3c6dd235e10a73bb, 0x3feec86319e32323,
    0xbc87c50422622263, 0x3feecc667b5de565,
    0x3c8b1c86e3e231d5, 0x3feed09bec4a2d33,
    0xbc91bbd1d3bcbb15, 0x3feed503b23e255d,
    0x3c90cc319cee31d2, 0x3feed99e1330b358,
    0x3c8469846e735ab3, 0x3feede6b5579fdbf,
    0xbc82dfcd978e9db4, 0x3feee36bbfd3f37a,
    0x3c8c1a7792cb3387, 0x3feee89f995ad3ad,
    0xbc907b8f4ad1d9fa, 0x3feeee07298db666,
    0xbc55c3d956dcaeba, 0x3feef3a2b84f15fb,
    0xbc90a40e3da6f640, 0x3feef9728de5593a,
    0xbc68d6f438ad9334, 0x3feeff76f2fb5e47,
    0xbc91eee26b588a35, 0x3fef05b030a1064a,
    0x3c74ffd70a5fddcd, 0x3fef0c1e904bc1d2,
    0xbc91bdfbfa9298ac, 0x3fef12c25bd71e09,
    0x3c736eae30af0cb3, 0x3fef199bdd85529c,
    0x3c8ee3325c9ffd94, 0x3fef20ab5fffd07a,
    0x3c84e08fd10959ac, 0x3fef27f12e57d14b,
    0x3c63cdaf384e1a67, 0x3fef2f6d9406e7b5,
    0x3c676b2c6c921968, 0x3fef3720dcef9069,
    0xbc808a1883ccb5d2, 0x3fef3f0b555dc3fa,
    0xbc8fad5d3ffffa6f, 0x3fef472d4a07897c,
    0xbc900dae3875a949, 0x3fef4f87080d89f2,
    0x3c74a385a63d07a7, 0x3fef5818dcfba487,
    0xbc82919e2040220f, 0x3fef60e316c98398,
    0x3c8e5a50d5c192ac, 0x3fef69e603db3285,
    0x3c843a59ac016b4b, 0x3fef7321f301b460,
    0xbc82d52107b43e1f, 0x3fef7c97337b9b5f,
    0xbc892ab93b470dc9, 0x3fef864614f5a129,
    0x3c74b604603a88d3, 0x3fef902ee78b3ff6,
    0x3c83c5ec519d7271, 0x3fef9a51fbc74c83,
    0xbc8ff7128fd391f0, 0x3fefa4afa2a490da,
    0xbc8dae98e223747d, 0x3fefaf482d8e67f1,
    0x3c8ec3bc41aa2008, 0x3fefba1bee615a27,
    0x3c842b94c3a9eb32, 0x3fefc52b376bba97,
    0x3c8a64a931d185ee, 0x3fefd0765b6e4540,
    0xbc8e37bae43be3ed, 0x3fefdbfdad9cbe14,
    0x3c77893b4d91cd9d, 0x3fefe7c1819e90d8,
    0x3c5305c14160cc89, 0x3feff3c22b8f71f1,
};

/// Top 12 bits (sign + biased exponent) of a double.
inline fn top12(x: f64) u32 {
    return @truncate(@as(u64, @bitCast(x)) >> 52);
}

/// Handle exp results that overflow/underflow the normal double range
/// (|x| >= 512 screens here). SBITS holds scale bits whose computed exponent
/// may have over/underflowed; (i32-truncated) KI sign says which direction.
/// Faithful port of exp.c specialcase(WANT_ROUNDING=1, no errno/fenv).
fn expSpecialcase(tmp: f64, sbits_in: u64, ki: u64) f64 {
    var sbits = sbits_in;
    if (ki & 0x80000000 == 0) {
        // k > 0: the exponent of scale may have overflowed by <= 460.
        sbits -%= 1009 << 52;
        const scale: f64 = @bitCast(sbits);
        return 0x1p1009 * @mulAdd(f64, scale, tmp, scale); // inf if it overflows
    }
    // k < 0: needs special care in the subnormal range.
    sbits +%= 1022 << 52;
    const scale: f64 = @bitCast(sbits);
    var y = @mulAdd(f64, scale, tmp, scale);
    if (y < 1.0) {
        // Round y to the final precision BEFORE the subnormal scaling to
        // avoid double rounding (keeps the <1 ulp worst case in subnormals).
        var lo = @mulAdd(f64, scale, tmp, scale - y);
        const hi = 1.0 + y;
        lo = 1.0 - hi + y + lo;
        y = (hi + lo) - 1.0;
        if (y == 0.0) y = 0.0; // avoid -0.0 with downward rounding
        // (glibc also raises the underflow flag here; we do not touch fenv.)
    }
    return 0x1p-1022 * y;
}

/// Double-precision e^x, faithfully rounded (worst case 0.509 ulp).
/// Deterministic: no libc, strict IEEE ops + @mulAdd only.
pub fn exp(x: f64) f64 {
    var abstop: u32 = top12(x) & 0x7ff;
    // Screen |x| outside [0x1p-54, 512) (uses unsigned wraparound):
    // 0x3c9 = top12(0x1p-54), 0x408 = top12(512), 0x409 = top12(1024).
    if (abstop -% 0x3c9 >= 0x408 - 0x3c9) {
        @branchHint(.unlikely);
        if (abstop -% 0x3c9 >= 0x80000000) {
            // |x| < 2^-54 (0 is a common input): exp(x) rounds to 1.
            return 1.0 + x;
        }
        if (abstop >= 0x409) {
            const ux: u64 = @bitCast(x);
            if (ux == @as(u64, @bitCast(-std.math.inf(f64)))) return 0.0;
            if (abstop >= 0x7ff) return 1.0 + x; // +inf or nan
            // Finite |x| >= 1024: certain overflow/underflow (errno skipped).
            return if (ux >> 63 != 0) 0.0 else std.math.inf(f64);
        }
        // 512 <= |x| < 1024: result may be subnormal/overflow — special path.
        abstop = 0;
    }

    // exp(x) = 2^(k/N) * exp(r), x = ln2/N*k + r, |r| <= ln2/2N.
    const z = exp_invln2N * x;
    // Round z to int with the 0x1.8p52 shift trick (round-to-nearest mode;
    // the low bits of the sum's mantissa are exactly k in two's complement).
    var kd = z + exp_shift;
    const ki: u64 = @bitCast(kd);
    kd -= exp_shift;
    const r = @mulAdd(f64, kd, exp_negln2loN, @mulAdd(f64, kd, exp_negln2hiN, x));
    // 2^(k/N) ~= scale * (1 + tail).
    const idx: usize = @intCast(2 * (ki % EXP_N));
    const top: u64 = ki << (52 - EXP_TABLE_BITS);
    const tail: f64 = @bitCast(exp_tab[idx]);
    // Valid scale bits for -1023*N < k < 1024*N (guaranteed by the screen).
    const sbits: u64 = exp_tab[idx + 1] +% top;
    // exp(x) ~= scale + scale*(tail + r + r^2*(C2 + r*C3) + r^4*(C4 + r*C5)),
    // contracted exactly as glibc's -mfma build contracts it.
    const r2 = r * r;
    const tmp = @mulAdd(f64, r2 * r2, @mulAdd(f64, r, exp_c5, exp_c4), @mulAdd(f64, r2, @mulAdd(f64, r, exp_c3, exp_c2), tail + r));
    if (abstop == 0) {
        @branchHint(.unlikely);
        return expSpecialcase(tmp, sbits, ki);
    }
    const scale: f64 = @bitCast(sbits);
    // No spurious underflow: |tmp| > 2^-200 (or 0) and scale > 2^-739 here.
    return @mulAdd(f64, scale, tmp, scale);
}

// ===================== tanhf(f32) =====================

// expm1f polynomial (pl/math/expm1f_data.c): fpminimax of (expm1(x)-x)/x^2
// on [-ln2/2, ln2/2]; expm1f_1u6.c design, max error 1.51 ulp for expm1f.
const expm1f_poly = [5]f32{ 0x1.fffffep-2, 0x1.5554aep-3, 0x1.555736p-5, 0x1.12287cp-7, 0x1.6b55a2p-10 };
const expm1f_shift: f32 = 0x1.8p23;
const expm1f_invln2: f32 = 0x1.715476p+0;
const expm1f_ln2hi: f32 = 0x1.62e4p-1;
const expm1f_ln2lo: f32 = 0x1.7f7d1cp-20;

/// exp(x)-1 without special-case handling; only valid where tanhf calls it
/// (|x| <= 2*0x1.205966p+3, so the exponent shift-and-add below is exact).
inline fn expm1fInline(x: f32) f32 {
    // Reduce: f in [-ln2/2, ln2/2], i = round(x/ln2) exactly (shift trick).
    const j = @mulAdd(f32, expm1f_invln2, x, expm1f_shift) - expm1f_shift;
    const i: i32 = @intFromFloat(j);
    var f = @mulAdd(f32, j, -expm1f_ln2hi, x);
    f = @mulAdd(f32, j, -expm1f_ln2lo, f);
    // expm1(f) ~= f + f^2*P(f), P evaluated in Estrin scheme.
    const f2 = f * f;
    const p01 = @mulAdd(f32, f, expm1f_poly[1], expm1f_poly[0]);
    const p23 = @mulAdd(f32, f, expm1f_poly[3], expm1f_poly[2]);
    var p = @mulAdd(f32, f2, p23, p01);
    p = @mulAdd(f32, f2 * f2, expm1f_poly[4], p);
    p = @mulAdd(f32, f2, p, f);
    // t = 2^i (|i| < 27 here).
    const t: f32 = @bitCast(@as(u32, @intCast(i + 127)) << 23);
    // expm1(x) ~= p*t + (t - 1).
    return @mulAdd(f32, p, t, t - 1.0);
}

/// Single-precision tanh, worst case 2.58 ulp (pl/math/tanhf_2u6.c).
/// Deterministic: no libc, strict IEEE ops + @mulAdd only.
pub fn tanhf(x: f32) f32 {
    const ix: u32 = @bitCast(x);
    const iax = ix & 0x7fffffff;
    const sign = ix & 0x80000000;
    if (iax > 0x41102cb3) {
        // |x| > 0x1.205966p+3 ~= 9.011: tanh rounds to +-1 (or nan for nan).
        @branchHint(.unlikely);
        if (iax > 0x7f800000) return x + x; // nan (invalid-op flag skipped)
        return @bitCast(0x3f800000 | sign); // +-1.0
    }
    if (iax < 0x34000000) {
        // |x| < 2^-23: tanh(x) rounds to x.
        @branchHint(.unlikely);
        return x;
    }
    // tanh(x) = (e^2x - 1) / (e^2x + 1).
    const q = expm1fInline(2.0 * x);
    return q / (q + 2.0);
}

// ===================== sanity tests =====================
// (Accuracy/determinism gates live elsewhere: the predictor INPUTS/FINAL-PROB
// oracle gates and the vs-glibc ulp audit harness. These are basic invariants.)

fn ulpDiff64(a: f64, b: f64) u64 {
    const ia: i64 = @bitCast(a);
    const ib: i64 = @bitCast(b);
    // map sign-magnitude to a monotonic integer line
    const ma: i64 = if (ia < 0) std.math.minInt(i64) - ia else ia;
    const mb: i64 = if (ib < 0) std.math.minInt(i64) - ib else ib;
    return @abs(ma - mb);
}

fn ulpDiff32(a: f32, b: f32) u32 {
    const ia: i32 = @bitCast(a);
    const ib: i32 = @bitCast(b);
    const ma: i32 = if (ia < 0) std.math.minInt(i32) - ia else ia;
    const mb: i32 = if (ib < 0) std.math.minInt(i32) - ib else ib;
    return @abs(ma - mb);
}

test "vendored exp: special values and identities" {
    try std.testing.expectEqual(@as(f64, 1.0), exp(0.0));
    try std.testing.expectEqual(@as(f64, 1.0), exp(-0.0));
    try std.testing.expectEqual(@as(f64, 1.0), exp(0x1p-55));
    try std.testing.expectEqual(@as(f64, 0.0), exp(-std.math.inf(f64)));
    try std.testing.expectEqual(std.math.inf(f64), exp(std.math.inf(f64)));
    try std.testing.expectEqual(std.math.inf(f64), exp(710.0)); // overflow
    try std.testing.expectEqual(@as(f64, 0.0), exp(-746.0)); // underflow
    try std.testing.expect(std.math.isNan(exp(std.math.nan(f64))));
    // subnormal result region (specialcase k<0 path)
    const sub = exp(-709.0);
    try std.testing.expect(sub > 0.0 and !std.math.isNormal(sub));
    // exp(1) must be within 1 ulp of e
    try std.testing.expect(ulpDiff64(exp(1.0), std.math.e) <= 1);
}

test "vendored exp: faithful vs @exp and monotone-sane on [-35, 35]" {
    // @exp (compiler-rt, musl's port of the SAME AOR design) is faithfully
    // rounded, so two faithful results can differ by at most 1 ulp.
    var i: i64 = 0;
    var prev: f64 = exp(-35.0);
    const steps: i64 = 200_000;
    var max_ulp: u64 = 0;
    while (i <= steps) : (i += 1) {
        const x = -35.0 + 70.0 * @as(f64, @floatFromInt(i)) / @as(f64, @floatFromInt(steps));
        const got = exp(x);
        const ref = @exp(x);
        const d = ulpDiff64(got, ref);
        if (d > max_ulp) max_ulp = d;
        try std.testing.expect(got >= prev); // grid monotonicity (72e6 ulp apart)
        prev = got;
    }
    try std.testing.expect(max_ulp <= 1);
}

test "vendored tanhf: special values, range, symmetry-sanity" {
    try std.testing.expectEqual(@as(f32, 0.0), tanhf(0.0));
    try std.testing.expectEqual(@as(f32, 1.0), tanhf(10.0));
    try std.testing.expectEqual(@as(f32, -1.0), tanhf(-10.0));
    try std.testing.expectEqual(@as(f32, 1.0), tanhf(std.math.inf(f32)));
    try std.testing.expectEqual(@as(f32, -1.0), tanhf(-std.math.inf(f32)));
    try std.testing.expect(std.math.isNan(tanhf(std.math.nan(f32))));
    try std.testing.expectEqual(@as(f32, 0x1p-24), tanhf(0x1p-24)); // tiny -> x
    var i: i64 = -60_000;
    while (i <= 60_000) : (i += 1) {
        const x = @as(f32, @floatFromInt(i)) / 6_000.0; // [-10, 10]
        const y = tanhf(x);
        try std.testing.expect(y >= -1.0 and y <= 1.0);
        // vs std.math.tanh (musl port): both are few-ulp-accurate designs.
        try std.testing.expect(ulpDiff32(y, std.math.tanh(x)) <= 8);
        // odd symmetry holds to ulp-level (not bit-exact by construction)
        try std.testing.expect(ulpDiff32(y, -tanhf(-x)) <= 4);
    }
}
