#pragma once

#include "interleave.h"

namespace intgemm {

template <class Register> inline Register set1_epi16(int16_t to);
#ifdef __SSE2__
inline __m128i add_epi32(__m128i first, __m128i second) {
  return _mm_add_epi32(first, second);
}
template <> inline __m128i set1_epi16<__m128i>(int16_t to) {
  return _mm_set1_epi16(to);
}
inline __m128i madd_epi16(__m128i first, __m128i second) {
  return _mm_madd_epi16(first, second);
}

#endif
#ifdef __AVX2__
inline __m256i add_epi32(__m256i first, __m256i second) {
  return _mm256_add_epi32(first, second);
}
template <> inline __m256i set1_epi16<__m256i>(int16_t to) {
  return _mm256_set1_epi16(to);
}
inline __m256i madd_epi16(__m256i first, __m256i second) {
  return _mm256_madd_epi16(first, second);
}
inline __m256i FinishReduce32(__m256i pack0123, __m256i pack4567) {
  // Introducing "f" for first half and "s" for second half, we have this order:
  // pack0123 = 1f 2f 3f 4f 1s 2s 3s 4s
  // pack4567 = 5f 6f 7f 8f 5s 6s 7s 8s
  // This instruction generates 1s 2s 3s 4s 5f 6f 7f 8f
  __m256i rev = _mm256_permute2f128_si256(pack0123, pack4567, 0x21);
  // This instruction generates 1f 2f 3f 4f 5s 6s 7s 8s
  __m256i blended = _mm256_blend_epi32(pack0123, pack4567, 0xf0);
  return _mm256_add_epi32(rev, blended);
}
#endif
#ifdef __AVX512__
inline __m512i add_epi32(__m512i first, __m512i second) {
  return _mm512_add_epi32(first, second);
}
template <> inline __m256i set1_epi16<__m512i>(int16_t to) {
  return _mm512_set1_epi16(to);
}
inline __m512i madd_epi16(__m512i first, __m512i second) {
  return _mm512_madd_epi16(first, second);
}
#endif

/* Take 4 registers with 32-bit values to be horizontally added.  Reduce them
 * to one register with 32-bit values in the pattern 1 2 3 4 1 2 3 4, leaving
 * the final addition (which crosses 128-bit lanes) to the caller. */
template <class Register> inline Register Pack0123(Register sum0, Register sum1, Register sum2, Register sum3) {
  // 1 2 1 2 1 2 1 2
  Interleave32(sum0, sum1);
  Register pack01 = add_epi32(sum0, sum1);
  // 3 4 3 4 3 4 3 4
  Interleave32(sum2, sum3);
  Register pack23 = add_epi32(sum2, sum3);
  Interleave64(pack01, pack23);
  // 1 2 3 4 1 2 3 4
  return add_epi32(pack01, pack23);
}

/* Convert 16-bit to 32-bit and add, not caring what parts are added.
 * Implementations:
 * 1. https://github.com/tesseract-ocr/tesseract/blob/master/src/arch/intsimdmatrixavx2.cpp#L67 under Apache license:
 *   This does a multiply by 1 and horizontal add:
 *    _mm512_madd_epi16(sum, _mm512_set1_epi16(1))
 *   Current fastest.
 *
 * 2. Signed extension and fold halves:
 *    sum = _mm512_add_epi32(
 *      _mm512_cvtepi16_epi32(_mm512_castsi512_si256(sum)),
 *      _mm512_cvtepi16_epi32(_mm512_extracti64x4_epi64(sum, 1)));
 *
 * 3. Sign extend by abuse of bitshift, then add.
 * sum = _mm512_add_epi32(
 *      _mm512_srai_epi32(_mm512_slli_epi32(sum, 16), 16),
 *      _mm512_srai_epi32(sum, 16));
 */
template <class Register> inline void Convert32Sum(Register &sum) {
  sum = madd_epi16(sum, set1_epi16<Register>(1));
}

} // namespace intgemm
