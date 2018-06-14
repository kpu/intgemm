#pragma once
#include <immintrin.h>
#include <cstddef>

namespace intgemm {
#ifdef __AVX512F__
namespace AVX512 {

void Quantize16(const float *input, int16_t *output, float quant_mult, std::size_t size);
void Quantize8(const float *input, int8_t *output, float quant_mult, std::size_t size);

// Multiply C = unquant_mult * A * B^T.  A is normally activations and B is normally a parameter matrix.
// Values of A and B should come from the corresponding quantizer.
// A and B must be 64-byte aligned.
// C should be the usual 4-byte alignment.
void MatrixMult16(const __m512i *A, const __m512i *B, float *C, float unquant_mult, int num_A_rows, int num_B_rows, int width);
void MatrixMult8(const __m512i *A, const __m512i *B, float *C, float unquant_mult, int num_A_rows, int num_B_rows, int width);

} // namespace AVX512
#endif // __AVX512F__
} // namespace intgemm
