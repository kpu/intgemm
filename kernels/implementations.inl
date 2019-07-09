#include "intrinsics.h"
#include "types.h"
#include "utils.h"
#include "vec_traits.h"

#include <cassert>

#if defined(THIS_IS_SSE2)
  #define CPU_NAME SSE2
  #define CPU_ATTR INTGEMM_SSE2
#elif defined(THIS_IS_AVX2)
  #define CPU_NAME AVX2
  #define CPU_ATTR INTGEMM_AVX2
#elif defined(THIS_IS_AVX512BW)
  #define CPU_NAME AVX512BW
  #define CPU_ATTR INTGEMM_AVX512BW
#else
  #error "Only SSE2, AVX2 and AVX512BW are supported"
#endif

#define vi vector_t<CPUType::CPU_NAME, int>
#define vf vector_t<CPUType::CPU_NAME, float>
#define vd vector_t<CPUType::CPU_NAME, double>
#define dvi dvector_t<CPUType::CPU_NAME, int>
#define dvf dvector_t<CPUType::CPU_NAME, float>
#define dvd dvector_t<CPUType::CPU_NAME, double>

/*
 * Kernels implementations....
 */
namespace intgemm {
namespace kernels {

/*
 * Write
 */
CPU_ATTR static inline void write(vf input, float* output, Index offset) {
  *reinterpret_cast<vf*>(output + offset) = input;
}

CPU_ATTR static inline void write(dvf input, float* output, Index offset) {
  write(input.first, output, offset);
  write(input.second, output, offset + sizeof(dvf::type) / 4);
}

/*
 * Quantize
 */
CPU_ATTR static inline vi quantize(vf input, vf quant_mult) {
  return cvtps_epi32(mul_ps(input, quant_mult));
}

CPU_ATTR static inline dvi quantize(dvf input, vf quant_mult) {
  return {
    quantize(input.first, quant_mult),
    quantize(input.second, quant_mult),
  };
}

/*
 * Unquantize
 */
CPU_ATTR static inline vf unquantize(vi input, vf unquant_mult) {
  return mul_ps(cvtepi32_ps(input), unquant_mult);
}

CPU_ATTR static inline dvf unquantize(dvi input, vf quant_mult) {
  return {
    unquantize(input.first, quant_mult),
    unquantize(input.second, quant_mult),
  };
}

/*
 * Add a bias term
 */
CPU_ATTR static inline vf add_bias(vf input, const float* bias_addr, Index bias_offset) {
  auto bias_term = *reinterpret_cast<const vf*>(bias_addr + bias_offset);
  return add_ps(input, bias_term);
}

CPU_ATTR static inline dvf add_bias(dvf input, const float* bias_addr, Index bias_offset) {
  return {
    add_bias(input.first, bias_addr, bias_offset),
    add_bias(input.second, bias_addr, bias_offset + sizeof(dvf::type) / 4),
  };
}

/*
 * Calculate floor: float -> float
 */
CPU_ATTR static inline vf floor_ff(vf a) {
#if defined(THIS_IS_AVX2)
  return _mm256_floor_ps(a);
#else
  return cvtepi32_ps(cvttps_epi32(a));
#endif
}

/*
 * Calculate approximation of e^x using Taylor series and lookup table
 */
CPU_ATTR static inline vf exp_approx_taylor(vf x) {
#if defined(THIS_IS_SSE2)
  assert(false && "SSE2 is not supported");
#else
  static constexpr int EXP_MIN = -20;
  static constexpr int EXP_MAX = 20;
  static constexpr float EXP_LOOKUP[EXP_MAX - EXP_MIN + 1] = {
    expi(-20), expi(-19), expi(-18), expi(-17), expi(-16), expi(-15),
    expi(-14), expi(-13), expi(-12), expi(-11), expi(-10), expi(-9),
    expi(-8), expi(-7), expi(-6), expi(-5), expi(-4), expi(-3), expi(-2),
    expi(-1), expi(0), expi(1), expi(2), expi(3), expi(4), expi(5),
    expi(6), expi(7), expi(8), expi(9), expi(10), expi(11), expi(12),
    expi(13), expi(14), expi(15), expi(16), expi(17), expi(18), expi(19),
    expi(20),
  };

  static const vf dividers[] = {
    set1_ps<vf>(1.f / factorial(7)),
    set1_ps<vf>(1.f / factorial(6)),
    set1_ps<vf>(1.f / factorial(5)),
    set1_ps<vf>(1.f / factorial(4)),
    set1_ps<vf>(1.f / factorial(3)),
    set1_ps<vf>(1.f / factorial(2)),
    set1_ps<vf>(1.f / factorial(1)),
  };
  static const auto const_one = set1_ps<vf>(1.f);
  static const auto const_min_x = set1_ps<vf>(EXP_MIN);
  static const auto const_max_x = set1_ps<vf>(EXP_MAX);

  x = max_ps(x, const_min_x);
  x = min_ps(x, const_max_x);

  auto a = floor_ff(x);
  auto xa = sub_ps(x, a);

  auto result = mul_ps(dividers[0], xa);

  result = add_ps(result, dividers[1]);
  result = mul_ps(result, xa);
  result = add_ps(result, dividers[2]);
  result = mul_ps(result, xa);
  result = add_ps(result, dividers[3]);
  result = mul_ps(result, xa);
  result = add_ps(result, dividers[4]);
  result = mul_ps(result, xa);
  result = add_ps(result, dividers[5]);
  result = mul_ps(result, xa);
  result = add_ps(result, dividers[6]);
  result = mul_ps(result, xa);

  result = add_ps(result, const_one);

  auto ea = i32gather_ps<4>(EXP_LOOKUP + EXP_MAX, cvtps_epi32(a));
  return mul_ps(ea, result);
#endif
}

}
}

#undef CPU_NAME
#undef CPU_ATTR
#undef vi
#undef vf
#undef vd
#undef dvi
#undef dvf
#undef dvd
