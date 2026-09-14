// ---------------------------------------------------------------------------
// MODIFIED BY THE ZMIX AUTHORS (GPL-3 section 5(a) modification notice).
//   upstream : fx2-cmix-transformer (Vladimir Ivanov), GPL-3,
//              cpp_infer/src/weights_io.h
//   modified : 2026-08-31, 2026-09-03
//   changes  : the load_compressed() documentation comment updated to describe
//              the FX2TFWC3 container as well as FX2TFWC1/FX2TFWC2; no
//              declaration or signature changed. +5 / -3 lines.
// The upstream file is unmodified except as stated above. Full per-file
// inventory and the diff summary: THIRD_PARTY_LICENSES.md section 1b.
// ---------------------------------------------------------------------------
// weights_io: loader for cpp_infer/data/weights.bin (SPEC.md section 4)
#pragma once

#include <cstddef>
#include <cstdint>
#include <initializer_list>
#include <string>
#include <unordered_map>
#include <vector>

namespace fx2 {

enum WDtype : uint8_t { DT_I8 = 0, DT_BF16 = 1, DT_F32 = 2, DT_I32 = 3 };

struct WTensor {
  uint8_t dtype = 0;
  std::vector<uint32_t> shape;
  std::vector<uint8_t> data;
  size_t numel = 0;

  const int8_t* i8() const { return reinterpret_cast<const int8_t*>(data.data()); }
  const uint16_t* bf16_bits() const {
    return reinterpret_cast<const uint16_t*>(data.data());
  }
  const float* f32() const { return reinterpret_cast<const float*>(data.data()); }
  const int32_t* i32() const {
    return reinterpret_cast<const int32_t*>(data.data());
  }
};

// fp32 value of raw bfloat16 bits (bits << 16 reinterpreted as float)
inline float bf16_to_f32(uint16_t bits) {
  union {
    uint32_t u;
    float f;
  } v;
  v.u = static_cast<uint32_t>(bits) << 16;
  return v.f;
}

struct WeightsFile {
  std::unordered_map<std::string, WTensor> tensors;

  // parses the whole file; aborts with a message on any format error
  static WeightsFile load(const char* path);

  // parses a compressed file (magic FX2TFWC1/FX2TFWC2, written by
  // pysrc/weights_compress.py, or FX2TFWC3 when built with -DFX2_WEIGHTS_V3=1,
  // written by experiments/gen7-weights-v3/wcode3.py).  v1/v2 yield tensors
  // bit-identical to load() on the matching uncompressed file; v3 additionally
  // stores the raw trained f32 tensors as bfloat16 (weights_io_compressed.cpp)
  static WeightsFile load_compressed(const char* path);

  bool has(const std::string& name) const { return tensors.count(name) != 0; }
  const WTensor& get(const std::string& name) const;
  // get + validate dtype and exact shape
  const WTensor& get(const std::string& name, uint8_t dtype,
                     std::initializer_list<uint32_t> shape) const;
};

}  // namespace fx2
