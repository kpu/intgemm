#include "intgemm.h"

namespace intgemm {

Int8::Interface Int8::interface = ChooseCPU<Int8::Interface>(
  {AVX512VNNI_8bit::Quantize,  AVX512VNNI_8bit::QuantizeU,  AVX512VNNI_8bit::PrepareB,  AVX512VNNI_8bit::SelectColumnsB,  AVX512VNNI_8bit::Name},
  {AVX512_8bit::Quantize,      AVX512_8bit::QuantizeU,      AVX512_8bit::PrepareB,      AVX512_8bit::SelectColumnsB,      AVX512_8bit::Name},
  {AVX2_8bit::Quantize,        AVX2_8bit::QuantizeU,        AVX2_8bit::PrepareB,        AVX2_8bit::SelectColumnsB,        AVX2_8bit::Name},
  {SSSE3_8bit::Quantize,       SSSE3_8bit::QuantizeU,       SSSE3_8bit::PrepareB,       SSSE3_8bit::SelectColumnsB,       SSSE3_8bit::Name},
  {Unsupported_8bit::Quantize, Unsupported_8bit::QuantizeU, Unsupported_8bit::PrepareB, Unsupported_8bit::SelectColumnsB, Unsupported_8bit::Name},
  {Unsupported_8bit::Quantize, Unsupported_8bit::QuantizeU, Unsupported_8bit::PrepareB, Unsupported_8bit::SelectColumnsB, Unsupported_8bit::Name}
);

Int8Shift::Interface Int8Shift::interface = ChooseCPU<Int8Shift::Interface>(
  {AVX512VNNI_8bit::QuantizeU,  AVX512VNNI_8bit::Name},
  {AVX512_8bit::QuantizeU,      AVX512_8bit::Name},
  {AVX2_8bit::QuantizeU,        AVX2_8bit::Name},
  {SSSE3_8bit::QuantizeU,       SSSE3_8bit::Name},
  {Unsupported_8bit::QuantizeU, Unsupported_8bit::Name},
  {Unsupported_8bit::QuantizeU, Unsupported_8bit::Name}
);

Int16::Interface Int16::interface = ChooseCPU<Int16::Interface>(
  {AVX512_16bit::Quantize,      AVX512_16bit::PrepareB,      AVX512_16bit::SelectColumnsB,      AVX512_16bit::Name},
  {AVX512_16bit::Quantize,      AVX512_16bit::PrepareB,      AVX512_16bit::SelectColumnsB,      AVX512_16bit::Name},
  {AVX2_16bit::Quantize,        AVX2_16bit::PrepareB,        AVX2_16bit::SelectColumnsB,        AVX2_16bit::Name},
  {SSE2_16bit::Quantize,        SSE2_16bit::PrepareB,        SSE2_16bit::SelectColumnsB,        SSE2_16bit::Name},
  {SSE2_16bit::Quantize,        SSE2_16bit::PrepareB,        SSE2_16bit::SelectColumnsB,        SSE2_16bit::Name},
  {Unsupported_16bit::Quantize, Unsupported_16bit::PrepareB, Unsupported_16bit::SelectColumnsB, Unsupported_16bit::Name}
);

const CPUType kCPU = ChooseCPU(CPUType::AVX512VNNI, CPUType::AVX512BW, CPUType::AVX2, CPUType::SSSE3, CPUType::SSE2, CPUType::UNSUPPORTED);

float Unsupported_MaxAbsolute(const float *begin, const float *end) {
  throw UnsupportedCPU();
}

float (*MaxAbsolute)(const float *begin, const float *end) = ChooseCPU(avx512f::MaxAbsolute, avx512f::MaxAbsolute, avx2::MaxAbsolute, sse2::MaxAbsolute, sse2::MaxAbsolute, Unsupported_MaxAbsolute);

}
