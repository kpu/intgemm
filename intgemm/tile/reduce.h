/* reduce.h: Horizontally reduce an arbitrary number of registers
 * simultaneously.  Given an array of registers, they will be horizontally
 * reduced (i.e. summed if Sum32Op is used) with the results placed back into
 * the array.
 *
 * This is the function:
 * template <Index Valid, class Op> INTGEMM_TARGET static inline void Reduce32(Register *regs);
 *
 * Valid is the length of the array of Registers in the input.
 *
 * Op defines the reduction operation.  It should support three architectures:
 * INTGEMM_SSE2 static inline __m128i Run(__m128i first, __m128i second);
 * INTGEMM_AVX2 static inline __m256i Run(__m256i first, __m256i second);
 * INTGEMM_AVX512BW static inline __m512i Run(__m512i first, __m512i second);
 * See Sum32Op for an example.
 *
 * regs is memory to use.
 * Input: an array Register[Valid].
 * Output: an array int32_t[Valid] of reduced values in the same order.  This
 * can be interpreted as registers with reduced values packed into them.
 * Anything at index Valid or later is undefined in the output.
 *
 * The function is defined in each architecture's namespace, so:
 * intgemm::SSE2:Reduce32
 * intgemm::SSSE3:Reduce32
 * intgemm::AVX2:Reduce32
 * intgemm::AVX512BW:Reduce32
 * intgemm::AVX512VNNI:Reduce32
 */
#pragma once
#include "../intrinsics.h"
#include "../utils.h"
#include "../types.h"

namespace intgemm {

namespace SSE2 { struct RegisterPair { Register hi; Register lo; }; }
namespace AVX2 { struct RegisterPair { Register hi; Register lo; }; }
#ifdef INTGEMM_COMPILER_SUPPORTS_AVX512BW
namespace AVX512BW { struct RegisterPair { Register hi; Register lo; }; }
#endif

// Op argument appropriate for summing 32-bit integers.
struct Sum32Op {
  INTGEMM_SSE2 static inline __m128i Run(SSE2::RegisterPair regs) {
    return add_epi32(regs.hi, regs.lo);
  }

  INTGEMM_AVX2 static inline __m256i Run(AVX2::RegisterPair regs) {
    return add_epi32(regs.hi, regs.lo);
  }

#ifdef INTGEMM_COMPILER_SUPPORTS_AVX512BW
  INTGEMM_AVX512BW static inline __m512i Run(AVX512BW::RegisterPair regs) {
    return add_epi32(regs.hi, regs.lo);
  }
#endif
};

} // namespace intgemm

// One implementation per width; the rest just import below.
#define INTGEMM_THIS_IS_SSE2
#include "reduce.inl"
#undef INTGEMM_THIS_IS_SSE2

#define INTGEMM_THIS_IS_AVX2
#include "reduce.inl"
#undef INTGEMM_THIS_IS_AVX2

#ifdef INTGEMM_COMPILER_SUPPORTS_AVX512BW
#define INTGEMM_THIS_IS_AVX512BW
#include "reduce.inl"
#undef INTGEMM_THIS_IS_AVX512BW
#endif

namespace intgemm {

namespace SSSE3 {
using SSE2::Reduce32;
} // namespace SSSE3

namespace AVX512VNNI {
using AVX512BW::Reduce32;
} // namespace AVX512VNNI

} // namespace intgemm
