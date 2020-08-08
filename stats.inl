/* This file is included multiple times, once per architecture. */
#if defined(INTGEMM_THIS_IS_AVX512BW)
#define INTGEMM_ARCH AVX512BW
#define INTGEMM_TARGET INTGEMM_AVX512BW
#elif defined(INTGEMM_THIS_IS_AVX2)
#define INTGEMM_ARCH AVX2
#define INTGEMM_TARGET INTGEMM_AVX2
#elif defined(INTGEMM_THIS_IS_SSE2)
#define INTGEMM_ARCH SSE2
#define INTGEMM_TARGET INTGEMM_SSE2
#else
#error Included with unexpected architecture
#endif

namespace intgemm {
namespace INTGEMM_ARCH {

/* Compute the maximum absolute value over floats aligned to register size.
 * Do not call this function directly; it's a subroutine of MaxAbsolute.
 */
INTGEMM_TARGET static inline float MaxAbsoluteThread(const FRegister *begin, const FRegister *end) {
  union {float f; int32_t i;} and_convert;
  and_convert.i = 0x7fffffff;
  FRegister and_me = set1_ps<FRegister>(and_convert.f);
  FRegister highest = setzero_ps<FRegister>();
#pragma omp for
  for (const FRegister *i = begin; i < end; ++i) {
    FRegister reg = and_ps(and_me, *i);
    highest = max_ps(highest, reg);
  }
  return MaxFloat32(highest);
}

/* Compute the maximum absolute value of an array of floats.
 * begin_float must be aligned to a multiple of the register size.
*/
INTGEMM_TARGET static inline float MaxAbsolute(const float *begin_float, const float *end_float) {
  assert(reinterpret_cast<uintptr_t>(begin_float) % sizeof(FRegister) == 0);
  const float *end_reg = end_float - (reinterpret_cast<uintptr_t>(end_float) % sizeof(FRegister)) / sizeof(float);
  float ret = 0.0;
#pragma omp parallel
  {
    float shard_max = MaxAbsoluteThread(
        reinterpret_cast<const FRegister*>(begin_float),
        reinterpret_cast<const FRegister*>(end_reg));
#pragma omp critical /* Not sure if there's a way to use reduction(max : ret) with the target option OMP workaround */
    ret = std::max(ret, shard_max);
  }
  /* Overhang: this would be more efficient if done in a single SIMD operation with some zeroing */
  union {float f; int32_t i;} float_convert;
  for (const float *i = end_reg; i < end_float; ++i) {
    float_convert.f = *i;
    float_convert.i &= 0x7fffffff;
    ret = std::max(ret, float_convert.f);
  }
  return ret;
}

/* Computes the euclidean norm and returns the mean and the standard deviation. Optionally it can be the mean and standard deviation in absolute terms. */
INTGEMM_TARGET static inline MeanStd VectorMeanStd(const float *begin_float, const float *end_float, bool absolute) {
  assert(end_float > begin_float);
  assert((end_float - begin_float) % (sizeof(FRegister) / sizeof(float)) == 0);
  size_t num_items = end_float - begin_float;
  const FRegister *begin = reinterpret_cast<const FRegister*>(begin_float);
  const FRegister *end = reinterpret_cast<const FRegister*>(end_float);
  FRegister squares = set1_ps<FRegister>(0);
  FRegister sums = set1_ps<FRegister>(0);
  if (absolute) {
    const FRegister mask = set1_ps<FRegister>(-0.f);
    for (; begin != end; begin++) {
      FRegister vec = *begin;
      vec = andnot_ps(mask, vec);
      squares = add_ps(squares, mul_ps(vec, vec));
      sums = add_ps(sums, vec);
    }
  } else {
    for (; begin != end; begin++) {
      FRegister vec = *begin;
      squares = add_ps(squares, mul_ps(vec, vec));
      sums = add_ps(sums, vec);
    }
  }
  float squares_sum = AddFloat32(squares);
  float normal_sums = AddFloat32(sums);
  MeanStd ret;
  ret.mean = normal_sums/num_items;
  ret.stddev = std::sqrt((squares_sum/num_items) - (ret.mean*ret.mean));
  return ret;
}

} // namespace INTGEMM_ARCH
} // namespace intgemm

#undef INTGEMM_ARCH
#undef INTGEMM_TARGET
