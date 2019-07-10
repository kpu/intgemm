#pragma once
#include "types.h"

#include <immintrin.h>

/* Take the dot product of packed 4-bit log-encoded values.  The values are
 * represented as a 1-bit sign and a 3-bit log_2(magnitude).  Currently two of
 * them are packed together into a byte, though we could also do creative
 * things like move the sign bits around in a byte if we wanted to.  The result
 * is in normal space, not log space.
 * 
 * MULTIPLICATION IDEAS
 * (it's actually an add)
 *
 * 1. Lookup table against registers, allowing arbitrary quantization centers.
 * If we want to do 
 *    a.high * b.high + a.low * b.low
 * then we could (in pre-compute) swap within each byte on b to make the 
 * operation
 *    a.high * b.low + a.low * b.high
 * Then at multiply time we apply this to blend arguments:
 *    https://graphics.stanford.edu/~seander/bithacks.html#MaskedMerge
 * The blend yields [a.high, b.low] consecutively and [b.high, a.low]
 * consecutively.  Use _mm512_permutex2var_epi8 to implement a lookup table.
 * The problem is that instruction is VBMI which Skylake doesn't support.  So
 * it would have to be _mm512_permutex2var_epi16 which is worth trying.  Note
 * that the pemutation instructions have the advantage that they ignore high-
 * order bits so we don't have to mask irrelevant parts off before calling.
 * When using _mm512_permutex2var_epi16 the results can be 2^value already.
 * 
 * 2. Zero out the sign bits using a mask so we have:
 *   0 3-bit log2(a.high) 0 3-bit log2(a.low)
 *  +0 3-bit log2(b.high) 0 3-bit log2(b.low)
 * An 8-bit sum resuls in
 *   4-bit log2(a.high * b.high) 4-bit log2(a.low * b.low)
 * To note, the 4-bit values can be at most 7 + 7 = 14.
 * Then the sign bits are xored separately.
 * This is currently implemented.
 * 
 * ACCUMULATION IDEAS
 * 1. Can we count the 4-bit values somehow?  There's 15 unique values, which
 * so we could have 32-bit counters in a register.  Would want to increment
 * and decrement based on sign.  Or just count both and reconcile periodically.
 *
 * 1.a Histogram with _mm512_conflict_epi32?  But 32-bit width is annoying.
 * 1.b Use scatter instructions somehow?  Maybe increment a different counter.
 * 1.c Some shuffle magic?
 *
 * 2. Power with 1 << magnitude then sum. Use _mm512_sllv_epi16 to do the
 * shifting since it can be up to 16 bits.  But then we need to run 4 times to
 * unpack 4 bits to 16 bits.  Annoyingly _mm512_sllv_epi16 requires higher bits
 * be zeroed, unlike _mm512_permutex2var_epi16.
 * Currently implemented.
 * 
 * SIGN BIT IDEAS
 * Since the sign bit ends up detached, we need a way to apply it to the powered
 * version.
 * 1. Broadcast -1 with mask.
 * 2. Subtract from zero with mask.
 * 3. Shift the sign bit down to 2, then subtract from 1 (before shifting).
 * 4. Multiply?!
 */

/* Return packed -1 << magnitude or 1 << magnitude depending on sign. */
INTGEMM_AVX512BW static inline __m512i Power2AndSign(__mmask32 signs, __m512i magnitude) {
  const __m512i kOnes = _mm512_set1_epi16(1);
  __m512i signed1 = _mm512_mask_set1_epi16(kOnes, signs, -1);
  return _mm512_sllv_epi16(signed1, extracted);
}

INTGEMM_AVX512BW static inline __m512i MultiplyLog4(__m512i first, __m512i second) {
  const __m512i kValueBits = _mm512_set1_epi8(0x77);
  /* Strip the sign bits then sum.  Since the sign bits are stripped, there's 4 bits of space for 3-bit values to sum in.
   *   0 3-bit log2(a.high) 0 3-bit log2(a.low)
   *  +0 3-bit log2(b.high) 0 3-bit log2(b.low)
   */
  // This results in packed 4-bit values in [0, 14] which are the log of values that should be summed.
  __m512i added = _mm512_add_epi8(_mm512_and_si512(first, kValueBits), _mm512_and_si512(second, kValueBits));

  // xor the arguments for the sign bits.  We only care about the sign bits 0x88.
  __m512i xored = _mm512_xor_si512(first, second);
  const __m512i kBottom4 = _mm512_set1_epi16(0xf);
  // Take 2^each block of 4 bit added values.
  // Take bottom 4 bits [3:0] of each block of 16.  Use that to exponentiate 1.
  // TODO these sign bit masks are taking up a lot of registers or eip memory references.
  __mmask32 signs = _mm512_test_epi16_mask(xored, _mm512_set1_epi16(0x8));
  __m512i accum = Power2AndSign(signs, _mm512_and_si512(kBottom4, added));
  // Take bits [7:4] of each block of 16.
  signs = _mm512_test_epi16_mask(xored, _mm512_set1_epi16(0x80));
  accum = _mm512_adds_epi16(accum, Power2AndSign(signs, _mm512_and_si512(kBottom4, _mm512_srli_epi16(added, 4))));
  // Take bits [11:8] of each block of 16.
  signs = _mm512_test_epi16_mask(xored, _mm512_set1_epi16(0x800));
  accum = _mm512_adds_epi16(accum, Power2AndSign(signs, _mm512_and_si512(kBottom4, _mm512_srli_epi16(added, 8))));
  // Take bits [15:12] of each block of 16.
  signs = _mm512_test_epi16_mask(xored, _mm512_set1_epi16(0x8000));
  accum = _mm512_adds_epi16(accum, Power2AndSign(signs, _mm512_srli_epi16(added, 12)));
  return accum;
}
