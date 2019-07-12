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
CPU_ATTR static inline void write(vi input, int* output, Index offset) {
  *reinterpret_cast<vi*>(output + offset) = input;
}

CPU_ATTR static inline void write(dvi input, int* output, Index offset) {
  write(input.first, output, offset);
  write(input.second, output, offset + sizeof(dvi::type) / 4);
}

CPU_ATTR static inline void write(vf input, float* output, Index offset) {
  *reinterpret_cast<vf*>(output + offset) = input;
}

CPU_ATTR static inline void write(dvf input, float* output, Index offset) {
  write(input.first, output, offset);
  write(input.second, output, offset + sizeof(dvf::type) / 4);
}

CPU_ATTR static inline void write(vd input, double* output, Index offset) {
  *reinterpret_cast<vd*>(output + offset) = input;
}

CPU_ATTR static inline void write(dvd input, double* output, Index offset) {
  write(input.first, output, offset);
  write(input.second, output, offset + sizeof(dvd::type) / 8);
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
 * ReLU
 */
CPU_ATTR static inline vi relu(vi input) {
  static const auto vconst_zero = set1_epi32<vi>(0);
#if defined(THIS_IS_SSE2)
  return _mm_and_si128(input, _mm_cmplt_epi32(vconst_zero, input));
#elif defined(THIS_IS_AVX2)
  return _mm256_max_epi32(input, vconst_zero);
#else
  return _mm512_max_epi32(input, vconst_zero);
#endif
}

CPU_ATTR static inline dvi relu(dvi input) {
  return {
    relu(input.first),
    relu(input.second),
  };
}

CPU_ATTR static inline vf relu(vf input) {
  static const auto vconst_zero = set1_ps<vf>(0);
  return max_ps(input, vconst_zero);
}

CPU_ATTR static inline dvf relu(dvf input) {
  return {
    relu(input.first),
    relu(input.second),
  };
}

CPU_ATTR static inline vd relu(vd input) {
  static const auto vconst_zero = set1_pd<vd>(0);
  return max_pd(input, vconst_zero);
}

CPU_ATTR static inline dvd relu(dvd input) {
  return {
    relu(input.first),
    relu(input.second),
  };
}

/*
 * Highway: weight * input1 + ([1] - weight) * input2,  [0] <= weight <= [1]
 */
CPU_ATTR static inline vf highway(vf input1, vf input2, vf weight) {
  static const auto vconst_one = set1_ps<vf>(1.f);
  return add_ps(mul_ps(input1, weight), mul_ps(input2, sub_ps(vconst_one, weight)));
}

CPU_ATTR static inline dvf highway(dvf input1, dvf input2, dvf weight) {
  static const auto vconst_one = set1_ps<vf>(1.f);
  return {
    highway(input1.first, input2.first, weight.first),
    highway(input1.second, input2.second, weight.second),
  };
}

CPU_ATTR static inline vd highway(vd input1, vd input2, vd weight) {
  static const auto vconst_one = set1_pd<vd>(1.f);
  return add_pd(mul_pd(input1, weight), mul_pd(input2, sub_pd(vconst_one, weight)));
}

CPU_ATTR static inline dvd highway(dvd input1, dvd input2, dvd weight) {
  static const auto vconst_one = set1_pd<vd>(1.f);
  return {
    highway(input1.first, input2.first, weight.first),
    highway(input1.second, input2.second, weight.second),
  };
}

/*
 * Calculate floor: float -> float
 */
CPU_ATTR static inline vf floor_ff(vf input) {
#if defined(THIS_IS_SSE2)
  static const auto vconst_zero = setzero_ps<vf>();
  static const auto vconst_one = set1_ps<vf>(1.f);

  auto result = cvtepi32_ps(cvttps_epi32(input));
  auto negatives = _mm_cmplt_ps(input, vconst_zero);
  auto nonintegers = _mm_cmpneq_ps(input, result);

  return sub_ps(result, and_ps(vconst_one, and_ps(negatives, nonintegers)));
#elif defined(THIS_IS_AVX2)
  return _mm256_floor_ps(input);
#else
  assert(false && "AVX512BW is not supported");
#endif
}

/*
 * Calculate approximation of e^x using Taylor series and lookup table
 */
CPU_ATTR static inline vf exp_approx_taylor(vf x) {
#if defined(THIS_IS_SSE2)
  assert(false && "SSE2 is not supported");
#elif defined(THIS_IS_AVX2)
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
#else
  assert(false && "AVX512 is not supported"); // missing floor_ff for AVX512BW
#endif
}

/*
 * Sigmoid
 */
CPU_ATTR static inline vf sigmoid(vf input) {
#if defined(THIS_IS_SSE2)
  assert(false && "SSE2 is not supported");
#elif defined(THIS_IS_AVX2)
  static const auto vconst_zero = setzero_ps<vf>();
  static const auto vconst_one = set1_ps<vf>(1.f);

  auto x = input;
  auto minus_x = sub_ps(vconst_zero, x);
  auto e_x = exp_approx_taylor(x);
  auto e_minus_x = exp_approx_taylor(minus_x);

  auto sigmoid_case1 = _mm256_rcp_ps(add_ps(vconst_one, e_minus_x));
  auto sigmoid_case2 = mul_ps(e_x, _mm256_rcp_ps(add_ps(vconst_one, e_x)));

  auto nonnegative_x_mask = _mm256_cmp_ps(vconst_zero, x, _CMP_LT_OS);
  return _mm256_blendv_ps(sigmoid_case1, sigmoid_case2, nonnegative_x_mask);
#else
  assert(false && "AVX512BW is not supported");  // TODO: missing exp_approx_taylor for AVX512BW
#endif
}

/*
 * Tanh
 */
CPU_ATTR static inline vf tanh(vf input) {
#if defined(THIS_IS_SSE2)
  assert(false && "SSE2 is not supported");
#elif defined(THIS_IS_AVX2)
  const static auto vconst_zero = setzero_ps<vf>();

  auto e_x = exp_approx_taylor(input);
  auto e_minus_x = exp_approx_taylor(sub_ps(vconst_zero, input));

  return div_ps(sub_ps(e_x, e_minus_x), add_ps(e_x, e_minus_x));
#else
  assert(false && "AVX512BW is not supported"); // TODO: missing exp_approx_taylor for AVX512BW
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
