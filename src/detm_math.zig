//! Deterministic libm pinning for -Dglibc builds.
//!
//! The codec's output must be bit-identical across machines: a submission is
//! compressed on one box and self-decompresses on the judge's. Dynamic glibc
//! libm breaks that — exp/log are IFUNC-resolved per CPU and differ across
//! glibc versions (proven: the same dist binary extracted different asset
//! bytes on CachyOS/local vs Ubuntu/a fleet box, desyncing the coder into a crash).
//!
//! Fix: export STRONG definitions of the whole exp/log family from the binary
//! itself, backed by Zig compiler_rt's pure musl ports (vendored in
//! src/vendor/crt). ELF resolution binds every libcall — including the ones
//! LLVM emits for @exp/@log intrinsics — to these, never to the host libm.
//! memcpy/memset stay dynamic: their semantics are bit-exact by definition.
//!
//! Only the symbols the binary actually calls are exported (exp x7, log x3,
//! tanhf x2, expf/exp2/log2 x1 — objdump-audited); dead exports would retain
//! their implementations. If a future change emits a dropped libcall (exp2f/
//! logf/log2f), the import check below fails the gate: after every ship build,
//! `objdump -T | grep UND` must show NO transcendentals.
//!
//! Import this module for its side effects from the shipping root:
//!     comptime { _ = @import("detm_math.zig"); }
const exp_impl = @import("vendor/crt/exp.zig");
const exp2_impl = @import("vendor/crt/exp2.zig");
const log_impl = @import("vendor/crt/log.zig");
const log2_impl = @import("vendor/crt/log2.zig");
// Hot-path implementations: ARM optimized-routines ports (vendored_math.zig)
// — the same algorithms modern glibc ships, compiled in. Faster than the musl
// ports; used for the symbols that dominate runtime (exp, tanhf).
const fast = @import("vendored_math.zig");

export fn expf(x: f32) f32 {
    return exp_impl.expf(x);
}
export fn exp(x: f64) f64 {
    return fast.exp(x);
}
export fn tanhf(x: f32) f32 {
    return fast.tanhf(x);
}
export fn exp2(x: f64) f64 {
    return exp2_impl.exp2(x);
}
export fn log(x: f64) f64 {
    return log_impl.log(x);
}
export fn log2(x: f64) f64 {
    return log2_impl.log2(x);
}
