// ---------------------------------------------------------------------------
// ZMIX-AUTHORED, and it incorporates VERBATIM EXCERPTS of GPL-3 upstream code
// (GPL-3 section 5(a) notice).
//   upstream : fx2-cmix-transformer (Vladimir Ivanov), GPL-3,
//              cpp_infer/src/predictor.cpp:26-101 copied verbatim
//   authored : 2026-08-31 .. 2026-09-01, the zmix authors
// Everything else in this file is zmix's, offered under GPL-3-or-later.
// Full per-file inventory: THIRD_PARTY_LICENSES.md section 1b.
// ---------------------------------------------------------------------------
// fx2_shim.h — the extern "C" boundary between zmix (Zig) and the vendored
// fx2-cmix-transformer inference kernels (cpp_infer/src/opt).
//
// Everything the *probability contract* touches lives on this side of the
// boundary, compiled by the same compiler with the same flags as the kernels:
//   * FloatToHalf / HalfToFloat / FloatsToHalves / HalvesToFloats are VERBATIM
//     copies of fx2-cmix-transformer's src/predictor.cpp:26-101, INCLUDING the
//     deliberately-frozen subnormal-decode quirk documented at :49-55. Their
//     own comment: "changing it would change the coded probabilities."
//   * fx2tf_byte_update reproduces Predictor::TransformerByteUpdate
//     (src/predictor.cpp:546-588) step for step: the f16 rounding of the PPMd
//     prior, the 15-token article-separator window, the 2^17-token piece cut,
//     the first-token PPMd passthrough, the f16 round-trip of the output and
//     the >= 1e-6 floor.
// A "correct" reimplementation of fp16 conversion would silently diverge, so
// this file must NOT be tidied. See.
#pragma once

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// V, fixed by the trained weights (they hard-Fail unless vocab_size == 205).
#define FX2TF_VOCAB 205

// Opaque handle. attn_kind: 0 = KVF32 (their ship default), 1 = KVI8.
void* fx2tf_create(const char* weights_path, int attn_kind);
void fx2tf_destroy(void* h);

// Reproduces Predictor::TransformerByteUpdate.
//   token     : canonical-205 index of the byte that just completed
//   prior205  : the PPMd byte distribution, compacted to the canonical 205 set,
//               UNSCALED (their byte_model_->BytePredict() row)
//   out205    : receives the distribution over the NEXT byte
// Returns 1 if the transformer produced out205, 0 if this was a piece-final
// token and out205 is the f16-round-tripped PPMd prior instead (their
// passthrough). The caller needs the distinction to decide whether a zmix-side
// PRL fold applies.
int fx2tf_byte_update(void* h, uint8_t token, const float* prior205,
                      float* out205);

// Feed a byte that has no canonical-205 token (the r1 side blob). Does NOT
// step the model, does NOT advance the separator window: it only f16
// round-trips the prior, exactly as the piece-final passthrough does.
void fx2tf_passthrough(void* h, const float* prior205, float* out205);

// Applies the >= 1e-6 floor their LoadTransformerProbs/TransformerByteUpdate
// applies (guards zero and NaN; ratios only, so no renormalization).
void fx2tf_floor(float* p205);

// ★ THE SUBMISSION GATE'S COUNTER (canonical-205 map guard).
// Number of times the 15-token kArticleSeparator window has MATCHED, counted
// separately from the 2^17-token piece cut that shares the same `stepped == 0`
// return. The two must not be conflated: a piece cut is content-dependent and
// legal, a missing separator match means the token map is wrong. See
// src/mixer/transformer.zig `checkSeparatorCount`.
uint64_t fx2tf_separator_hits(void* h);

// Exposed for the round-trip unit gates in src/mixer/transformer.zig.
void fx2tf_floats_to_halves(const float* src, uint16_t* dst, uint64_t n);
void fx2tf_halves_to_floats(const uint16_t* src, float* dst, uint64_t n);

#ifdef __cplusplus
}
#endif
