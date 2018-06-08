#pragma once

#include <immintrin.h>
#include <cstddef>

namespace intgemm {

#ifdef __AVX512__
namespace AVX512 {
void Quantize16(const float *input, int16_t *output, float quant_mult, std::size_t size);
void Quantize8(const float *input, int8_t *output, float quant_mult, std::size_t size);
} // namespace AVX512
#endif

#ifdef __AVX2__
namespace AVX2 {
void Quantize16(const float *input, int16_t *output, float quant_mult, std::size_t size);
void Quantize8(const float *input, int8_t *output, float quant_mult, std::size_t size);
} // namespace AVX2
#endif // __AVX2__

#ifdef __SSE2__
namespace SSE {
void Quantize16(const float *input, int16_t *output, float quant_mult, std::size_t size);
void Quantize8(const float *input, int8_t *output, float quant_mult, std::size_t size);
} // namespace SSE
#endif // __SSE2__

namespace slow {
void Quantize16(const float *input, int16_t *output, float quant_mult, std::size_t size);
void Quantize8(const float *input, int8_t *output, float quant_mult, std::size_t size);
} // namespace slow

} // namespace intgemm
