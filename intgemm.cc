#include "intgemm.h"

namespace intgemm {

float Unsupported_MaxAbsolute(const float *begin, const float *end) {
  throw UnsupportedCPU();
}

void (*Int16::Quantize)(const float *input, int16_t *output, float quant_mult, Index size) = ChooseCPU(AVX512_16bit::Quantize, AVX2_16bit::Quantize, SSE2_16bit::Quantize, SSE2_16bit::Quantize, Unsupported_16bit::Quantize);

void (*Int16::PrepareB)(const float *input, int16_t *output, float quant_mult, Index rows, Index cols) = ChooseCPU(AVX512_16bit::PrepareB, AVX2_16bit::PrepareB, SSE2_16bit::PrepareB, SSE2_16bit::PrepareB, Unsupported_16bit::PrepareB);

void (*Int16::SelectColumnsB)(const int16_t *input, int16_t *output, Index rows, const Index *cols_begin, const Index *cols_end) = ChooseCPU(AVX512_16bit::SelectColumnsB, AVX2_16bit::SelectColumnsB, SSE2_16bit::SelectColumnsB, SSE2_16bit::SelectColumnsB, Unsupported_16bit::SelectColumnsB);

const char *const Int16::kName = ChooseCPU(AVX512_16bit::kName, AVX2_16bit::kName, SSE2_16bit::kName, SSE2_16bit::kName, Unsupported_16bit::kName);

void (*Int8::Quantize)(const float *input, int8_t *output, float quant_mult, Index size) = ChooseCPU(AVX512_8bit::Quantize, AVX2_8bit::Quantize, SSSE3_8bit::Quantize, Unsupported_8bit::Quantize, Unsupported_8bit::Quantize);

void (*Int8::PrepareB)(const float *input, int8_t *output, float quant_mult, Index rows, Index cols) = ChooseCPU(AVX512_8bit::PrepareB, AVX2_8bit::PrepareB, SSSE3_8bit::PrepareB, Unsupported_8bit::PrepareB, Unsupported_8bit::PrepareB);

void (*Int8::SelectColumnsB)(const int8_t *input, int8_t *output, Index rows, const Index *cols_begin, const Index *cols_end) = ChooseCPU(AVX512_8bit::SelectColumnsB, AVX2_8bit::SelectColumnsB, SSSE3_8bit::SelectColumnsB, Unsupported_8bit::SelectColumnsB, Unsupported_8bit::SelectColumnsB);

const char *const Int8::kName = ChooseCPU(AVX512_8bit::kName, AVX2_8bit::kName, SSSE3_8bit::kName, Unsupported_8bit::kName, Unsupported_8bit::kName);

const CPUType kCPU = ChooseCPU(CPUType::AVX512BW, CPUType::AVX2, CPUType::SSSE3, CPUType::SSE2, CPUType::UNSUPPORTED);

float (*MaxAbsolute)(const float *begin, const float *end) = ChooseCPU(avx512f::MaxAbsolute, avx2::MaxAbsolute, sse2::MaxAbsolute, sse2::MaxAbsolute, Unsupported_MaxAbsolute);

}
