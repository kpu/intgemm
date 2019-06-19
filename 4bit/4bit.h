#include <immintrin.h>

INTGEMM_AVX512BW static inline __m512i MultiplyLog4(__m512i first, __m512i second) {
  const __m512i kValueBits = _mm512_set1_epi8(0x77);
  // Strip the sign bits then sum.  Since the sign bits are stripped, there's 4 bits of space for 3-bit values to sum in.
  // This results in packed 4-bit values in [0, 14] which are the log of values that should be summed.
  __m512i added = _mm512_add_epi8(_mm512_and_si512(first, kValueBits), _mm512_and_si512(second, kValueBits));

  // xor the arguments for the sign bits.  We only care about the sign bits 0x88.
  __m512i xored =  _mm512_xor_si512(first, second);
  // TODO do something with the sign bits?  Maybe this?  
  __mmask64 low_sign_bits = _mm512_test_epi8_mask(xored, _mm512_set1_epi8(0x8));
  // Ideas for applying sign bit:
  // 1. Subtract from zero with mask.
  // 2. Shift the sign bit down to 2, then subtract from 1 (before shifting).
  // 3. Multiply?!

  const __m512i kOnes = _mm512_set1_epi16(1);
  const __m512i kBottom4 = _mm512_set1_epi16(0xf);
  // Take 2^each block of 4 bit added values.
  // Take bottom 4 bits [3:0] of each block of 16.  Use that to exponentiate 1.
  __m512i accum0 = _mm512_sllv_epi16(kOnes, _mm512_and_epi8(kBottom4, added));
  // Take bits [7:4] of each block of 16.
  __m512i accum1 = _mm512_sllv_epi16(kOnes, _mm512_and_epi8(kBottom4, _mm512_srli_epi16(added, 4)));
  // Take bits [11:8] of each block of 16.
  __m512i accum2 = _mm512_sllv_epi16(kOnes, _mm512_and_epi8(kBottom4, _mm512_srli_epi16(added, 8)));
  // Take bits [15:12] of each block of 16.
  __m512i accum3 = _mm512_sllv_epi16(kOnes, _mm512_srli_epi16(added, 12));
  // TODO apply sign bits to accum0 etc then accum0 + accum1 + accum2 + accum3.
}
