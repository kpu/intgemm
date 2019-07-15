#pragma once
#include "types.h"

#include <immintrin.h>
#include <cstdint>
#include <stdint.h>

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
 * Or use ternary logic with 11110000 to accomplish the bitwise blend.
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
 * Latency 67 and throughput 17.5 (and having to run 8 times) means this won't
 * be fast.
 * 1.b Use scatter instructions somehow?  Maybe increment a different counter
 * for each lane.  But scatter instructions have a 32-bit minimum -> 8
 * scatters.
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
 * 1. Broadcast -1 with mask. Latency 3 throughput 1 means it's actually better
 * to blend if we can afford the register.s
 * 2. Subtract from zero with mask.
 * 3. Shift the sign bit down to 2, then subtract from 1 (before shifting).
 * 4. Multiply?!
 */

namespace intgemm {

/* FIRST IMPLEMENTATION:
 * Shift 16-bit +/- 1 using a variable shift with _mm512_sllv_epi16.
 */
/* Helper for Shift implementation: return packed -1 << magnitude or 1 << magnitude depending on sign. */
INTGEMM_AVX512BW static inline __m512i Power2AndSign(__mmask32 signs, __m512i magnitude) {
  const __m512i kOnes = _mm512_set1_epi16(1);
  __m512i signed1 = _mm512_mask_set1_epi16(kOnes, signs, -1);
  return _mm512_sllv_epi16(signed1, magnitude);
}
INTGEMM_AVX512BW static inline __m512i DotLog4_Shift(__m512i first, __m512i second) {
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


/* SECOND IMPLEMENTATION:
 * Use _mm512_shuffle_epi8 as a 4-bit lookup table to retrieve 2^value (since
 * there isn't an 8-bit variable shift on Skylake.)
 * Since that's actually a 15-bit result, we do two _mm512_shuffle_epi8, one to
 * get high 8 bits (of which one is wasted) and one to get low 8 bits.
 * Sum 8 bytes at a time using _mm512_sad_epu8 to compute sum of absolute value
 * of differences.  It's |255-v| for negative values + |v-0| for positive.
 * Because 255 was added, we need to subtract it off for each negative value.
 * The values to subtract are counted using popcnt.  Because both high and low
 * bytes of 16-bit results are impacted, the value to subtract is 255*256+255 =
 * 65535 for each negative value.
 */

/* This is a helper function for the second implementation. Given log magnitudes
 * in the lower 4 of each byte and their signs, it looks them up, 2^magnitude
 * by lookup table, and accumulates into packed 64-bit values.
 * The negative values are off by 65535 each in the count returned.
 */
INTGEMM_AVX512BW static inline __m512i SumSad(__m512i lower4, __mmask64 signs) {
  /* Low-order values of 16 bits for 1 << value, in reverse order, in blocks of 128 bits. */
  /* Fun fact: _mm512_set_epi8 isn't defined in older gcc. https://www.mail-archive.com/gcc-patches@gcc.gnu.org/msg188664.html */
  const __m512i kLookupLower = _mm512_set_epi64(0, 0x8040201008040201, 0, 0x8040201008040201, 0, 0x8040201008040201, 0, 0x8040201008040201);
  // High-order values of 16 bits for 1 << value, in reverse order, in blocks of 128 bits
  const __m512i kLookupHigher = _mm512_set_epi64(0x8040201008040201, 0, 0x8040201008040201, 0, 0x8040201008040201, 0, 0x8040201008040201, 0);

  // Bit shifting 4 bits to 16 bits by lookup table.  I feel dirty.
  // Retrieve absolute values of the lower 8 bits of what we want to sum.
  __m512i abs_sum_lower = _mm512_shuffle_epi8(kLookupLower, lower4); // latency 1 throughput 1
  // Retrieve absolute values of the higher 8 bits of what we want to sum.
  __m512i abs_sum_higher = _mm512_shuffle_epi8(kLookupHigher, lower4); // latency 1 throughput 1
  // Now each [abs_sum_higher abs_sum_lower] is 1 << lower4.

  // 255 for negative values (so we do 255-value) or 0 for positive values (so we do value - 0)
  // TODO: replace this with a blend of two registers which has latency 1 throughput 0.5?
  __m512i sign255 = _mm512_maskz_set1_epi8(signs, 255); // latency 3 throughput 1
  __m512i sum_higher = _mm512_sad_epu8(sign255, abs_sum_higher); // latency 3 throughput 1
  __m512i sum_lower = _mm512_sad_epu8(sign255, abs_sum_lower); // latency 3 throughput 1
  // Now we have 64-bit sums of higher and lower parts.
  return _mm512_add_epi64(_mm512_slli_epi64(sum_higher, 8), sum_lower);
}

/* Returns packed 64-bit values that should be summed. To finish, sum the 64-bit values - 65535 * subtract65535.
 * Initialize subtract65535 with 0 then call each time to accumulate more.
 */
INTGEMM_AVX512BW static inline __m512i DotLog4_Lookup16(__m512i a, __m512i b, int64_t &subtract65535) {
  const __m512i kValueBits = _mm512_set1_epi8(0x77);
  __m512i a_value = _mm512_and_si512(a, kValueBits);
  __m512i b_value = _mm512_and_si512(b, kValueBits);
  __m512i added = _mm512_add_epi8(a_value, b_value);
  // xor the arguments for the sign bits.  We only care about the sign bits 0x88.
  __m512i xored = _mm512_xor_si512(a, b);
  __mmask64 signs_lower4 = _mm512_test_epi8_mask(xored, _mm512_set1_epi8(0x8));
  __mmask64 signs_higher4 = _mm512_test_epi8_mask(xored, _mm512_set1_epi8(0x80));

  // The shuffle instruction leaves zeros if the top bit is 1. Mask this off. 
  // Only care that top bit is 0 and the lower 4 are present so e.g. 0xf 0x1f
  // 0x2f would also be fine.
  const __m512i kLower4 = _mm512_set1_epi8(0xf);
  __m512i lower4 = _mm512_and_si512(added, kLower4); // latency 1 throughput 0.5
  // Shift right by 4 bits, but there isn't an 8-bit shift instruction on Skylake-X.
  // So use 16-bit (or anything for that matter) then mask away the top bit again.
  __m512i upper4 = _mm512_srli_epi16(added, 4);
  __m512i sum = SumSad(lower4, signs_lower4);

  upper4 = _mm512_and_si512(upper4, kLower4);
  __m512i sum2 = SumSad(upper4, signs_higher4);
  // Sum from the upper 4 bits of each byte and the lower 4 bits of each byte.
  sum = _mm512_add_epi64(sum, sum2);
  // We added 255 for every negative value.  Count the negative values.
  // Moreover, we added it for both the high parts of the shift table (256*255) and low parts (255).
  // 255*256+255 = 65535
  subtract65535 += __builtin_popcountll(signs_lower4) + __builtin_popcountll(signs_higher4);
  return sum;
}


/* THIRD IMPLEMENTATION:
 * Like the Lookup16 strategy except we're only allowed to unpack 4-bit sums to
 * 8 bits instead of 16.  The 4-bit sums come from two 3-bit values added
 * together so they range [0, 14] for 15 possible values.  Hence the widest log
 * is 256^(1/14) \approx 1.447269 and we can use all [0,255] values to just be
 * off by 1 for everything.  Another option is to use a larger base but scale
 * values down.  That has the effect of clamping small sums to 0.
 *
 * One option for the lookup table is:
 * Lookup from $i \in [0,14]$ to $\round(b^i) - 1$ where $b = 256^(1/14)$
 * Note this means that 1 should be added for each value multiplied.  This
 * can be folded into the bias term.
 * b=pow(256.0,1.0/14)
 * ''.join(reversed([hex(round(math.pow(b, i)-1)).split('x')[1].zfill(2) for i in range(0,15)]))
 * The lookup table is repeated for each 128-bit part of the register.
 * const __m512i kLookup = _mm512_set_epi64(0xffab734d342217, 0x0f0a060402010000, 0xffab734d342217, 0x0f0a060402010000, 0xffab734d342217, 0x0f0a060402010000, 0xffab734d342217, 0x0f0a060402010000);
 */
INTGEMM_AVX512BW static inline __m512i DotLog4_Lookup8(__m512i a, __m512i b, const __m512i lookup, uint64_t &subtract255) {
  const __m512i kValueBits = _mm512_set1_epi8(0x77);
  __m512i a_value = _mm512_and_si512(a, kValueBits);
  __m512i b_value = _mm512_and_si512(b, kValueBits);
  __m512i added = _mm512_add_epi8(a_value, b_value);
  // xor the arguments for the sign bits.  We only care about the sign bits 0x88.
  __m512i xored = _mm512_xor_si512(a, b);
  __mmask64 signs_lower = _mm512_test_epi8_mask(xored, _mm512_set1_epi8(0x8));
  __mmask64 signs_upper = _mm512_test_epi8_mask(xored, _mm512_set1_epi8(0x80));

  const __m512i kLower4 = _mm512_set1_epi8(0xf);
  __m512i upper = _mm512_srli_epi16(added, 4);
  __m512i lower = _mm512_and_si512(added, kLower4); // latency 1 throughput 0.5
  upper = _mm512_and_si512(upper, kLower4);

  // Do the lookup.
  lower = _mm512_shuffle_epi8(lookup, lower);
  upper = _mm512_shuffle_epi8(lookup, upper);

  const __m512i kZeros = _mm512_setzero_si512();
  const __m512i k255 = _mm512_set1_epi8(0xff);

  // 255 for negative, 0 for positive.
  __m512i sign255_lower = _mm512_mask_blend_epi8(signs_lower, kZeros, k255);
  __m512i sign255_upper = _mm512_mask_blend_epi8(signs_upper, kZeros, k255);

  lower = _mm512_sad_epu8(sign255_lower, lower);
  upper = _mm512_sad_epu8(sign255_upper, upper);

  subtract255 += __builtin_popcountll(signs_lower) + __builtin_popcountll(signs_upper);
  return _mm512_add_epi64(lower, upper);
}

} // namespace intgemm
