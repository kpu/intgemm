/* Quantize to 8-bit and 16-bit signed integers.
 *
 * 8-bit quantization bans -128 because we can't negate it.  The maddubs
 * instructions are unsigned * signed so they require sign bit manipulation.
 *
 * The input and output should be aligned appropriately for instructions:
 *   64 bytes for AVX512
 *   32 bytes for AVX2
 *   16 bytes for SSE
 *
 * The size depends on the function, but it's safe to be a multiple of 32.
 */
#include "Quantize.h"

#include "Print.h"

#include <cassert>
#include <emmintrin.h>
#include <immintrin.h>
#include <math.h>
#include <stdint.h>
#include <tmmintrin.h>
#include <xmmintrin.h>

namespace intgemm {

#ifdef __AVX512F__

namespace AVX512 {
namespace {
// Load from memory, multiply, and convert to int32_t.
inline __m512i QuantizerGrab(const float *input, const __m512 quant_mult_reg) {
  // Load 16 floats
  __m512 val = _mm512_load_ps(input);
  // Multiply each by the quantization factor.
  val = _mm512_mul_ps(val, quant_mult_reg);
  // Cast to 32-bit int
  return _mm512_cvtps_epi32(val);
}

} // namespace

// Convert to 16-bit signed integers. 
void Quantize16(const float *input, int16_t *output, float quant_mult, std::size_t size) {
    assert(size % 16 == 0);
    assert(reinterpret_cast<uintptr_t>(input) % 64 == 0);
    // Fill with the quantization multiplier.
    const __m512 quant_mult_reg = _mm512_set1_ps(quant_mult);
    const float *end = input + size;
    for (; input != end; input += 16, output += 16) {
      // There doesn't seem to be an unmasked version.
      _mm512_mask_cvtsepi32_storeu_epi16(output, 0xffff, QuantizerGrab(input, quant_mult_reg));
    }
}

// Convert to 8-bit signed integers.
void Quantize8(const float *input, int8_t *output, float quant_mult, std::size_t size) {
  assert(size % 16 == 0);
  assert(reinterpret_cast<uintptr_t>(input) % 64 == 0);
  const __m512i neg127 = _mm512_set1_epi32(-127);
  const __m512 quant_mult_reg = _mm512_set1_ps(quant_mult);
  const float *end = input + size;
  for (; input < end; input += 16, output += 16) {
    __m512i asint = QuantizerGrab(input, quant_mult_reg);
    asint = _mm512_max_epi32(asint, neg127);
    // There doesn't seem to be an unmasked version.
    _mm512_mask_cvtsepi32_storeu_epi8(output, 0xffff, asint);
  }
}
} // namespace AVX512

#endif

namespace AVX2 {
namespace {

// Same implementation as AVX512, just shorter
inline __m256i QuantizerGrab(const float *input, const __m256 quant_mult_reg) {
  return _mm256_cvtps_epi32(_mm256_mul_ps(_mm256_load_ps(input), quant_mult_reg));
}

} // namespace

void Quantize16(const float *input, int16_t *output, float quant_mult, std::size_t size) {
  assert(size % 16 == 0);
  assert(reinterpret_cast<uintptr_t>(input) % 32 == 0);
  const __m256 quant_mult_reg = _mm256_set1_ps(quant_mult);
  const float *end = input + size;
  for (; input != end; input += 16, output += 16) {
    __m256i g0 = QuantizerGrab(input, quant_mult_reg);
    __m256i g1 = QuantizerGrab(input + 8, quant_mult_reg);
    __m256i packed = _mm256_packs_epi32(g0, g1);
    // Reorder the packed values because Intel does 0 1 2 3 8 9 10 11 4 5 6 7 12 13 14 15.
    // Technically this could be removed so long as the rows are bigger than 16
    // and the values are only used for GEMM.
    packed = _mm256_permute4x64_epi64(packed, 0xd8 /* 0, 2, 1, 3 */);
    *reinterpret_cast<__m256i*>(output) = packed;
  }
}

void Quantize8(const float *input, int8_t *output, float quant_mult, std::size_t size) {
  assert(size % 32 == 0);
  assert(reinterpret_cast<uintptr_t>(input) % 32 == 0);
  const __m256 quant_mult_reg = _mm256_set1_ps(quant_mult);
  const __m256i neg127 = _mm256_set1_epi8(-127);
  const __m256i shuffle_param = _mm256_set_epi32(7, 3, 6, 2, 5, 1, 4, 0);
  const float *end = input + size;
  for (; input != end; input += 32, output += 32) {
    // Grab 4 registers at a time in 32-bit format.
    __m256i g0 = QuantizerGrab(input, quant_mult_reg);
    __m256i g1 = QuantizerGrab(input + 8, quant_mult_reg);
    __m256i g2 = QuantizerGrab(input + 16, quant_mult_reg);
    __m256i g3 = QuantizerGrab(input + 24, quant_mult_reg);
    // Pack 32-bit to 16-bit.
    __m256i packed0 = _mm256_packs_epi32(g0, g1);
    __m256i packed1 = _mm256_packs_epi32(g2, g3);
    // Pack 16-bit to 8-bit.
    __m256i packed = _mm256_packs_epi16(packed0, packed1);
    // Ban -128.
    packed = _mm256_max_epi8(packed, neg127);
    // Currently in 0 1 2 3 8 9 10 11 16 17 18 19 24 25 26 27 4 5 6 7 12 13 14 15 20 21 22 23 28 29 30 31
    // Or as 32-bit integers 0 2 4 6 1 3 5 7
    // Technically this could be removed so long as the rows are bigger than 16
    // and the values are only used for GEMM.
    packed = _mm256_permutevar8x32_epi32(packed, shuffle_param);
    *reinterpret_cast<__m256i*>(output) = packed;
  }
}

} // namespace AVX2

namespace SSE {

/* Uses following instructions:
 * SSE: _mm_mul_ps, _mm_load_ps
 * SSE2: _mm_cvtps_epi32, _mm_packs_epi32, _mm_packs_epi32, _mm_cmpeq_epi8, _mm_sub_epi8
 */

// Same implementation as AVX512, just width.
inline __m128i QuantizerGrab(const float *input, const __m128 quant_mult_reg) {
  return _mm_cvtps_epi32(_mm_mul_ps(_mm_load_ps(input), quant_mult_reg));
}

/* I also tried an implementation based on _mm_cvtps_pi16 but it was slower:
 * For size 1048576, run 10x in seconds on i7-6700:
 * This code: 0.00228409, 0.00204906
 * With _mm_cvtps_pi16 basis: 0.00391884, 0.00390869
 */
void Quantize16(const float *input, int16_t *output, float quant_mult, std::size_t size) {
  assert(size % 8 == 0);
  assert(reinterpret_cast<uintptr_t>(input) % 16 == 0);
  const __m128 quant_mult_reg = _mm_set1_ps(quant_mult);
  const float *end = input + size;
  for (; input != end; input += 8, output += 16) {
    __m128i g0 = QuantizerGrab(input, quant_mult_reg);
    __m128i g1 = QuantizerGrab(input + 4, quant_mult_reg);
    __m128i packed = _mm_packs_epi32(g0, g1);
    *reinterpret_cast<__m128i*>(output) = packed;
  }
}

void Quantize8(const float *input, int8_t *output, float quant_mult, std::size_t size) {
  assert(size % 16 == 0);
  assert(reinterpret_cast<uintptr_t>(input) % 16 == 0);
  const __m128 quant_mult_reg = _mm_set1_ps(quant_mult);
//  const __m128i neg127 = _mm_set1_epi8(-127);
  const __m128i neg128 = _mm_set1_epi8(-128);
  const float *end = input + size;
  for (; input != end; input += 16, output += 16) {
    __m128i g0 = QuantizerGrab(input, quant_mult_reg);
    __m128i g1 = QuantizerGrab(input + 4, quant_mult_reg);
    __m128i g2 = QuantizerGrab(input + 8, quant_mult_reg);
    __m128i g3 = QuantizerGrab(input + 12, quant_mult_reg);
    __m128i packed0 = _mm_packs_epi32(g0, g1);
    __m128i packed1 = _mm_packs_epi32(g2, g3);
    __m128i packed = _mm_packs_epi16(packed0, packed1);
    /* Ban -128.
     * Don't use the SSE4.1 instruction _mm_max_epi8(packed, neg127).  Instead,
     * use SSE2 instructions _mm_cmpeq_epi8 and _mm_sub_epi8.
     * The first generates 0xff for fields -128.
     * The second subtracts 0xff from -128 which has the effect of converting
     * to -127.
     */
    // packed = _mm_max_epi8(packed, neg127);
    __m128i evils = _mm_cmpeq_epi8(packed, neg128);
    packed = _mm_sub_epi8(packed, evils);
    // No permute needed.  packs is in order for SSE.
    *reinterpret_cast<__m128i*>(output) = packed;
  }
}

/* This implementation was much slower.
 * For size 1048576, run 10x in seconds on i7-6700:
 * 0.00134197, 0.0013169 Above implementation.
 * 0.00550692, 0.00568323 Below implementation.
 * However, it does have the advantage of using at most SSE2, whereas the above
 * requires SSE4.1 for _mm_max_epi8.
 */
/*void Quantize8(const float *input, int8_t *output, float quant_mult, std::size_t size) {
  assert(size % 8 == 0);
  assert(reinterpret_cast<uintptr_t>(input) % 16 == 0);
  const float *end = input + size;
  const __m128 quant_mult_reg = _mm_set1_ps(quant_mult);
  const __m64 neg128 = _mm_set1_pi8(-128);
  for (; input < end; input += 8, output += 8) {
    // These both fill the lower 4 elements with 8-bit integers.
    __m64 second = _mm_cvtps_pi8(_mm_mul_ps(_mm_load_ps(input + 4), quant_mult_reg));
    __m64 first = _mm_cvtps_pi8(_mm_mul_ps(_mm_load_ps(input), quant_mult_reg));
    // Shift second right by 32 bits then or into one register.
    __m64 combined = first | _m_psllqi(second, 32);
    // Test for -128, setting 0xff in corresponding fields.
    __m64 evils = _mm_cmpeq_pi8(combined, neg128);
    // Subtract 0xff from -128s to yield -127.
    combined = _mm_sub_pi8(combined, evils);
    *reinterpret_cast<__m64*>(output) = combined;
  }
}*/

} // namespace SSE

namespace slow {

void Quantize16(const float *input, int16_t *output, float quant_mult, std::size_t size) {
  for (std::size_t i = 0; i < size; ++i) {
    float value = roundf(input[i] * quant_mult);
    value = std::max(-32768.0f, value);
    value = std::min(32767.0f, value);
    output[i] = value;
  }
}

void Quantize8(const float *input, int8_t *output, float quant_mult, std::size_t size) {
  for (std::size_t i = 0; i < size; ++i) {
    float value = roundf(input[i] * quant_mult);
    value = std::max(-127.0f, value);
    value = std::min(127.0f, value);
    output[i] = value;
  }
}

} // namespace slow

} // namespace intgemm
