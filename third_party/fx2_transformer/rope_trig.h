// ---------------------------------------------------------------------------
// ZMIX-AUTHORED (GPL-3 section 5(a)). Contains a VERBATIM EXCERPT of
//   fx2-cmix-transformer (Vladimir Ivanov), GPL-3,
//   cpp_infer/src/weights_io_compressed.cpp:167-255
// re-published here as a header so that the SAME bit-exact CUDA-libdevice port
// is available to model_opt.cpp's RoPE beyond-table path. No line of the
// excerpt is altered; only the namespace and `inline` linkage are added.
// Full per-file inventory: THIRD_PARTY_LICENSES.md section 1b.
// ---------------------------------------------------------------------------
// WHY THIS FILE EXISTS —
//
// model_opt.cpp's RoPE fallback (pos >= ROPE_LEN) called std::sin/std::cos,
// which GCC contracts into a single `sincosf@GLIBC_2.2.5` — THE ONLY libm math
// import of the shipped decoder. A libm-dependent value on the coder path is a
// dependency OUTSIDE THE BINARY: an
// encoder host and a decoder host with different sinf implementations
// desynchronise the arithmetic coder, silently. It is the same defect class as
// the reciprocal-estimate P0 (src/strict_fp.zig), and it is fixed the same way
// — UNCONDITIONALLY, with no build flag to forget.
//
// Measured (same report): the fallback is UNREACHABLE in the shipped
// configuration — fx2_shim.cpp cuts every piece at kMaxArticleTokens == 2^17 ==
// ROPE_LEN and begin_article() always passes rope_position_offset 0, so
// max(pos) == 131070 over all of enwik9 (0 tokens at pos >= 131072, measured on
// the 586,459,321 B coder stream). This change is therefore ARCHIVE-NEUTRAL and
// is HARDENING: it removes the import, and it removes the possibility that a
// later change to either constant silently reintroduces the hazard.
// It also removes a latent CORRECTNESS defect: glibc's sincosf disagrees with
// the shipped rope table's own generator on 19.5-19.7 % of entries (max 2 ULP),
// so the old fallback did not even continue the table it falls off.
// ---------------------------------------------------------------------------
#pragma once

#include <cmath>
#include <cstdint>
#include <cstring>

namespace fx2 {
namespace rope_trig {

// --- bit-exact host port of CUDA libdevice __nv_sinf/__nv_cosf --------------
// transcribed from the __nv_sinf/__nv_cosf LLVM IR of CUDA 13.0's
// libdevice.10.bc; verified bit-identical to the rope tables computed by
// torch.sin/cos on CUDA over all 8388608 table entries (incl. 25457
// Payne-Hanek slowpath arguments).  cos(a) is sin's body with quadrant + 1.

inline float fbits(uint32_t u) {
  float f;
  std::memcpy(&f, &u, 4);
  return f;
}

inline constexpr uint32_t kI2OverPi[6] = {0x3C439041u, 0xDB629599u, 0xF534DDC0u,
                               0xFC2757D1u, 0x4E441529u, 0xA2F9836Eu};

// __internal_trig_reduction_slowpath: Payne-Hanek for |a| >= 105615
inline float trig_slowpath(float a, int* quadrant) {
  uint32_t ia;
  std::memcpy(&ia, &a, 4);
  uint32_t sign = ia & 0x80000000u;
  int32_t e = (int32_t)((ia >> 23) & 0xffu) - 128;
  ia = (ia << 8) | 0x80000000u;

  uint32_t result[7];
  uint32_t hi = 0;
  for (int k = 0; k < 6; k++) {
    uint64_t p = (uint64_t)kI2OverPi[k] * ia + hi;
    result[k] = (uint32_t)p;
    hi = (uint32_t)(p >> 32);
  }
  result[6] = hi;

  int idx = 4 - ((uint32_t)e >> 5);  // e >= 16 on this path
  int sh = e & 31;
  uint32_t rhi = result[idx + 2], rlo = result[idx + 1];
  if (sh) {
    rhi = (result[idx + 2] << sh) + (result[idx + 1] >> (32 - sh));
    rlo = (result[idx + 1] << sh) + (result[idx] >> (32 - sh));
  }
  uint32_t q = rhi >> 30;
  uint32_t nhi = (rhi << 2) + (rlo >> 30);
  uint32_t nlo = rlo << 2;
  uint32_t top = nhi >> 31;
  q += top;
  int32_t qi = (int32_t)q;
  if (sign) qi = -qi;
  uint32_t s2 = sign;
  if (top) {
    nhi = ~nhi;
    nlo = ~nlo;
    s2 = sign ^ 0x80000000u;
  }
  *quadrant = qi;
  int64_t prod = (int64_t)(((uint64_t)nhi << 32) | nlo);
  double dscale;
  uint64_t dbits = 0x3BF921FB54442D19ull;  // pi/2 * 2^-64
  std::memcpy(&dscale, &dbits, 8);
  float r = (float)((double)prod * dscale);
  if (s2) r = -r;
  return r;
}

// __nv_sinf(a) for cos_bias 0, __nv_cosf(a) for cos_bias 1
inline float sincosf_cuda(float a, int cos_bias) {
  int i = (int)lrintf(a * fbits(0x3F22F983u));  // __float2int_rn(a * 2/pi)
  float j = (float)i;
  float t = fmaf(j, fbits(0xBFC90FDAu), a);
  t = fmaf(j, fbits(0xB3A22168u), t);
  t = fmaf(j, fbits(0xA7C234C5u), t);
  if (fabsf(a) >= 105615.0f) {
    if (std::isinf(a)) {
      t = a * 0.0f;
      i = 0;
    } else {
      t = trig_slowpath(a, &i);
    }
  }
  i += cos_bias;
  float x2 = t * t;
  float base = (i & 1) ? 1.0f : t;
  float p = fmaf(x2, base, 0.0f);
  float c = (i & 1) ? fmaf(fbits(0x37CBAC00u), x2, fbits(0xBAB607EDu))
                    : fbits(0xB94D4153u);
  c = fmaf(c, x2, (i & 1) ? fbits(0x3D2AAABBu) : fbits(0x3C0885E4u));
  c = fmaf(c, x2, (i & 1) ? fbits(0xBEFFFFFFu) : fbits(0xBE2AAAA8u));
  float z = fmaf(c, p, base);
  if (i & 2) z = fmaf(z, -1.0f, 0.0f);
  return z;
}

}  // namespace rope_trig
}  // namespace fx2
