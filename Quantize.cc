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

#ifdef __SSE2__
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
#endif // __SSE2__

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
