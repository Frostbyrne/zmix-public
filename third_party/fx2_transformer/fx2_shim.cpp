// ---------------------------------------------------------------------------
// ZMIX-AUTHORED, and it incorporates VERBATIM EXCERPTS of GPL-3 upstream code
// (GPL-3 section 5(a) notice).
//   upstream : fx2-cmix-transformer (Vladimir Ivanov), GPL-3,
//              cpp_infer/src/predictor.cpp:26-101,546-588 copied verbatim
//   authored : 2026-08-31 .. 2026-09-01, the zmix authors
// Everything else in this file is zmix's, offered under GPL-3-or-later.
// Full per-file inventory: THIRD_PARTY_LICENSES.md section 1b.
// ---------------------------------------------------------------------------
// fx2_shim.cpp — see fx2_shim.h. The fp16 helpers and the byte-update contract
// below are VERBATIM from fx2-cmix-transformer's src/predictor.cpp; the quirk
// in HalfToFloat is deliberate and load-bearing. Do not "fix" it.
#include "fx2_shim.h"

#include <cstring>
#include <cstdlib>
#include <cstdio>
#include <new>

#ifdef __F16C__
#include <immintrin.h>
#endif

#include "model_opt.h"

namespace {

// ---- VERBATIM: predictor.cpp:26-48 -----------------------------------------
uint16_t FloatToHalf(float f) {
  uint32_t x;
  memcpy(&x, &f, 4);
  uint32_t sign = (x >> 16) & 0x8000;
  int32_t exp = (int32_t)((x >> 23) & 0xFF) - 112;  // half-biased exponent
  uint32_t mant = x & 0x7FFFFF;
  if (exp >= 31) return sign | 0x7C00;  // overflow, infinity and NaN
  if (exp <= 0) {  // subnormal half (or zero)
    if (exp < -10) return sign;
    mant |= 0x800000;
    int shift = 14 - exp;
    uint16_t h = mant >> shift;
    uint32_t rem = mant & ((1u << shift) - 1), half = 1u << (shift - 1);
    if (rem > half || (rem == half && (h & 1))) ++h;
    return sign | h;
  }
  uint16_t h = sign | (exp << 10) | (mant >> 13);
  uint32_t rem = mant & 0x1FFF;
  if (rem > 0x1000 || (rem == 0x1000 && (h & 1))) ++h;  // round to nearest even
  return h;
}

// ---- VERBATIM: predictor.cpp:49-79 -----------------------------------------
// Note: subnormal halves (values below 2^-14) decode to HALF their true
// value (the exponent term below would be 113-e in an exact decode). The
// F16C bulk path decodes exactly, so only the last n%8 elements of a
// conversion are affected. This quirk is part of the probability rounding
// contract of --save-ppmd-probs/--load-transformer-probs and of the transformer
// path, which reproduces the file pipeline bit-for-bit; changing it would
// change the coded probabilities.
float HalfToFloat(uint16_t h) {
  uint32_t sign = (uint32_t)(h & 0x8000) << 16;
  uint32_t exp = (h >> 10) & 0x1F;
  uint32_t mant = h & 0x3FF;
  uint32_t x;
  if (exp == 0) {
    if (mant == 0) {
      x = sign;
    } else {  // subnormal half: normalize
      int e = 0;
      while (!(mant & 0x400)) {
        mant <<= 1;
        ++e;
      }
      mant &= 0x3FF;
      x = sign | ((uint32_t)(112 - e) << 23) | (mant << 13);
    }
  } else if (exp == 31) {
    x = sign | 0x7F800000 | (mant << 13);
  } else {
    x = sign | ((exp + 112) << 23) | (mant << 13);
  }
  float f;
  memcpy(&f, &x, 4);
  return f;
}

// ---- VERBATIM: predictor.cpp:80-101 ----------------------------------------
void FloatsToHalves(const float* src, uint16_t* dst, size_t n) {
  size_t i = 0;
#ifdef __F16C__
  for (; i + 8 <= n; i += 8) {
    _mm_storeu_si128(reinterpret_cast<__m128i*>(dst + i),
        _mm256_cvtps_ph(_mm256_loadu_ps(src + i), _MM_FROUND_TO_NEAREST_INT));
  }
#endif
  for (; i < n; ++i) dst[i] = FloatToHalf(src[i]);
}

void HalvesToFloats(const uint16_t* src, float* dst, size_t n) {
  size_t i = 0;
#ifdef __F16C__
  for (; i + 8 <= n; i += 8) {
    _mm256_storeu_ps(dst + i, _mm256_cvtph_ps(
        _mm_loadu_si128(reinterpret_cast<const __m128i*>(src + i))));
  }
#endif
  for (; i < n; ++i) dst[i] = HalfToFloat(src[i]);
}

// ---- VERBATIM: predictor.cpp:103-119 ---------------------------------------
// The encoded article separator ("  <page>\n    <title>" after the WRT
// dictionary transform), as vocabulary indices of the enwik9 preprocessed
// stream.
//
// zmix NOTE: this constant transfers UNCHANGED. It is expressed in
// vocabulary INDICES, and exp-transformer-port-design-aug-31.md §2b measured
// that decoding these 15 indices through zmix's own alphabet yields
// "  <page>\n    <title>" and that the byte string occurs 243,426 times in our
// own prep.temp — the exact enwik9 article count. The two 205-value alphabets
// are the same set because both come from the same english.dic.
const unsigned char kArticleSeparator[15] = {
    0x08, 0x08, 0x25, 0xac, 0x65, 0x27, 0x05,
    0x08, 0x08, 0x08, 0x08, 0x25, 0xac, 0x68, 0x27};

// Articles longer than this are cut into pieces of exactly this many tokens
// (plus a shorter last piece), each a fresh transformer context. Must not
// exceed the transformer's rope table (131072 positions).
const unsigned long long kMaxArticleTokens = 1ULL << 17;

struct Shim {
  fx2::opt::TransformerOpt* tf = nullptr;
  unsigned char separator_window[15] = {0};
  unsigned long long article_tokens = 0;
  // Counted separately from `stepped == 0`, which ALSO fires on the
  // kMaxArticleTokens piece cut. The zmix-side map guard asserts on THIS.
  unsigned long long separator_hits = 0;
  uint16_t half_scratch[FX2TF_VOCAB];
  float probs[FX2TF_VOCAB];
};

}  // namespace

extern "C" {

void* fx2tf_create(const char* weights_path, int attn_kind) {
  Shim* s = new (std::nothrow) Shim();
  if (!s) return nullptr;
  s->tf = new (std::nothrow) fx2::opt::TransformerOpt(
      weights_path, attn_kind == 1 ? fx2::opt::AttnKind::KVI8
                                   : fx2::opt::AttnKind::KVF32);
  if (!s->tf) { delete s; return nullptr; }
  return s;
}

void fx2tf_destroy(void* h) {
  Shim* s = static_cast<Shim*>(h);
  if (!s) return;
  delete s->tf;
  delete s;
}

// ---- Reproduces Predictor::TransformerByteUpdate (predictor.cpp:546-588) ----
int fx2tf_byte_update(void* h, uint8_t token, const float* prior205,
                      float* out205) {
  Shim* s = static_cast<Shim*>(h);
  FloatsToHalves(prior205, s->half_scratch, FX2TF_VOCAB);

  memmove(s->separator_window, s->separator_window + 1,
          sizeof(s->separator_window) - 1);
  s->separator_window[sizeof(s->separator_window) - 1] = (unsigned char)token;
  ++s->article_tokens;

  const bool separator_match =
      memcmp(s->separator_window, kArticleSeparator,
             sizeof(kArticleSeparator)) == 0;
  if (separator_match) ++s->separator_hits;
  bool last_of_piece = separator_match || s->article_tokens >= kMaxArticleTokens;
  int stepped;
  if (last_of_piece) {
    // The next token starts a fresh context; its distribution is the ppmd's.
    HalvesToFloats(s->half_scratch, out205, FX2TF_VOCAB);
    s->article_tokens = 0;
    stepped = 0;
  } else {
    if (s->article_tokens == 1) s->tf->begin_article();
    s->tf->step((uint8_t)token, s->half_scratch, s->probs);
    FloatsToHalves(s->probs, s->half_scratch, FX2TF_VOCAB);
    HalvesToFloats(s->half_scratch, out205, FX2TF_VOCAB);
    stepped = 1;
  }
  return stepped;
}

void fx2tf_passthrough(void* h, const float* prior205, float* out205) {
  Shim* s = static_cast<Shim*>(h);
  FloatsToHalves(prior205, s->half_scratch, FX2TF_VOCAB);
  HalvesToFloats(s->half_scratch, out205, FX2TF_VOCAB);
}

void fx2tf_floor(float* p205) {
  for (int i = 0; i < FX2TF_VOCAB; ++i) {
    if (!(p205[i] >= 1e-6f)) p205[i] = 1e-6f;
  }
}

uint64_t fx2tf_separator_hits(void* h) {
  const Shim* s = static_cast<const Shim*>(h);
  return s ? (uint64_t)s->separator_hits : 0;
}

void fx2tf_floats_to_halves(const float* src, uint16_t* dst, uint64_t n) {
  FloatsToHalves(src, dst, (size_t)n);
}
void fx2tf_halves_to_floats(const uint16_t* src, float* dst, uint64_t n) {
  HalvesToFloats(src, dst, (size_t)n);
}

}  // extern "C"
