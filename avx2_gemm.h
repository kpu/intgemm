#pragma once
#include <immintrin.h>
#include <cstddef>

namespace intgemm {
#ifdef __AVX2__
namespace AVX2 {

void Quantize16(const float *input, int16_t *output, float quant_mult, std::size_t size);
void Quantize8(const float *input, int8_t *output, float quant_mult, std::size_t size);

// Multiply C = unquant_mult * A * B^T.  A is normally activations and B is normally a parameter matrix.
// Values of A and B should come from the corresponding quantizer.
// A, B, and C must be 32-byte alined.
void MatrixMult16(const __m256i *A, const __m256i *B, float *C, float unquant_mult, int num_A_rows, int num_B_rows, int width);
void MatrixMult8(const __m256i *A, const __m256i *B, float *C, float unquant_mult, int num_A_rows, int num_B_rows, int width);

void MatrixMult8Contrast(const __m256i *A, const __m256i *B, float *C, float unquant_mult, int num_A_rows, int num_B_rows, int width);
void MatrixMult8ASM(const __m256i *A, const __m256i *B, float *C, float unquant_mult, int num_A_rows, int num_B_rows, int width);

} // namespace AVX2
#endif // __AVX2__
} // namespace intgemm
