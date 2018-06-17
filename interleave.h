#pragma once

#include <emmintrin.h>
#include <immintrin.h>
#include <tmmintrin.h>
#include <xmmintrin.h>

namespace intgemm {

/* This macro defines functions that interleave their arguments like
 * inline void Interleave8(__m256i &first, __m256i &second) {
 *   __m256i temp = _mm256_unpacklo_epi8(first, second);
 *   second = _mm256_unpackhi_epi8(first, second);
 *   first = temp;
 * }
 *
 * Example usage:
 *   INTGEMM_INTERLEAVE(__m128i, )
 *   INTGEMM_INTERLEAVE(__m256i, 256)
 *   INTGEMM_INTERLEAVE(__m512i, 512)
 */
#define INTGEMM_INTERLEAVE(type, prefix) \
 inline void Interleave8(type &first, type &second) { \
  type temp = _mm##prefix##_unpacklo_epi8(first, second); \
  second = _mm##prefix##_unpackhi_epi8(first, second); \
  first = temp; \
} \
inline void Interleave16(type &first, type &second) { \
  type temp = _mm##prefix##_unpacklo_epi16(first, second); \
  second = _mm##prefix##_unpackhi_epi16(first, second); \
  first = temp; \
} \
inline void Interleave32(type &first, type &second) { \
  type temp = _mm##prefix##_unpacklo_epi32(first, second); \
  second = _mm##prefix##_unpackhi_epi32(first, second); \
  first = temp; \
} \
inline void Interleave64(type &first, type &second) { \
  type temp = _mm##prefix##_unpacklo_epi64(first, second); \
  second = _mm##prefix##_unpackhi_epi64(first, second); \
  first = temp; \
}

#ifdef __SSE2__
INTGEMM_INTERLEAVE(__m128i, )
#endif
#ifdef __AVX2__
INTGEMM_INTERLEAVE(__m256i, 256)
#endif
#ifdef __AVX512__
INTGEMM_INTERLEAVE(__m512i, 512)
#endif

/* Transpose registers containing 8 packed 16-bit integers.
 * Each 128-bit lane is handled independently.
 */
template <class Register> inline void Transpose16InLane(Register &r0, Register &r1, Register &r2, Register &r3, Register &r4, Register &r5, Register &r6, Register &r7) {
  // r0: columns 0 1 2 3 4 5 6 7 from row 0
  // r1: columns 0 1 2 3 4 5 6 7 from row 1

  Interleave16(r0, r1);
  Interleave16(r2, r3);
  Interleave16(r4, r5);
  Interleave16(r6, r7);
  // r0: columns 0 0 1 1 2 2 3 3 from rows 0 and 1
  // r1: columns 4 4 5 5 6 6 7 7 from rows 0 and 1
  // r2: columns 0 0 1 1 2 2 3 3 from rows 2 and 3
  // r3: columns 4 4 5 5 6 6 7 7 from rows 2 and 3
  // r4: columns 0 0 1 1 2 2 3 3 from rows 4 and 5
  // r5: columns 4 4 5 5 6 6 7 7 from rows 4 and 5
  // r6: columns 0 0 1 1 2 2 3 3 from rows 6 and 7
  // r7: columns 4 4 5 5 6 6 7 7 from rows 6 and 7

  Interleave32(r0, r2);
  Interleave32(r1, r3);
  Interleave32(r4, r6);
  Interleave32(r5, r7);
  // r0: columns 0 0 0 0 1 1 1 1 from rows 0, 1, 2, and 3
  // r1: columns 4 4 4 4 5 5 5 5 from rows 0, 1, 2, and 3
  // r2: columns 2 2 2 2 3 3 3 3 from rows 0, 1, 2, and 3
  // r3: columns 6 6 6 6 7 7 7 7 from rows 0, 1, 2, and 3
  // r4: columns 0 0 0 0 1 1 1 1 from rows 4, 5, 6, and 7
  // r5: columns 4 4 4 4 5 5 5 5 from rows 4, 5, 6, and 7
  // r6: columns 2 2 2 2 3 3 3 3 from rows 4, 5, 6, and 7
  // r7: columns 6 6 6 6 7 7 7 7 from rows 4, 5, 6, and 7

  Interleave64(r0, r4);
  Interleave64(r1, r5);
  Interleave64(r2, r6);
  Interleave64(r3, r7);
  // r0: columns 0 0 0 0 0 0 0 0 from rows 0 through 7
  // r1: columns 4 4 4 4 4 4 4 4 from rows 0 through 7
  // r2: columns 2 2 2 2 2 2 2 2 from rows 0 through 7
  // r3: columns 6 6 6 6 6 6 6 6 from rows 0 through 7
  // r4: columns 1 1 1 1 1 1 1 1 from rows 0 through 7
  // r5: columns 5 5 5 5 5 5 5 5 from rows 0 through 7
  
  // Empirically gcc is able to remove these movs and just rename the outputs of Interleave64.
  // Swap r1 and r4
  Register tmp = r4;
  r4 = r1;
  r1 = tmp;
  // Swap r3 and r6.
  tmp = r3;
  r3 = r6;
  r6 = tmp;
}

/* Tranpose registers containing 16 packed 8-bit integers.
 * Each 128-bit lane is handled independently.
 */
template <class Register> inline void Transpose8InLane(
    Register &r0, Register &r1, Register &r2, Register &r3, Register &r4, Register &r5, Register &r6, Register &r7,
    Register &r8, Register &r9, Register &r10, Register &r11, Register &r12, Register &r13, Register &r14, Register &r15) {
  // Get 8-bit values to 16-bit values so they can travel together.
  Interleave8(r0, r1);
  // r0: columns 0 0 1 1 2 2 3 3 4 4 5 5 6 6 7 7 from rows 0 and 1.
  Interleave8(r2, r3);
  Interleave8(r4, r5);
  Interleave8(r6, r7);
  Interleave8(r8, r9);
  Interleave8(r10, r11);
  Interleave8(r12, r13);
  Interleave8(r14, r15);
  Transpose16InLane(r0, r2, r4, r6, r8, r10, r12, r14);
  Transpose16InLane(r1, r3, r5, r7, r9, r11, r13, r15);
}

} // namespace intgemm
