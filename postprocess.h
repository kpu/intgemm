#pragma once

#include "intrinsics.h"
#include "postprocess_pipeline.h"
#include "types.h"
#include "vec_utils.h"
#include "vec_traits.h"

// TODO: We support some postprocess in few variations e.g. we support ReLU for
// float -> float, int8 -> int8, int16 -> int16. Maybe it would be a good idea
// to pass input type and output type as a template parameter of postprocess?

namespace intgemm {

/*
 * Unquantize
 */
class Unquantize {
public:
  float unquantize_multiplier;

  Unquantize(float unquantize_multiplier) : unquantize_multiplier(unquantize_multiplier) {}
};

template <>
class PostprocessImpl<Unquantize, CPUType::SSE2> {
public:
  using InputRegister = dvector_t<CPUType::SSE2, int>;
  using OutputRegister = dvector_t<CPUType::SSE2, float>;

  INTGEMM_SSE2 PostprocessImpl(const Unquantize& config) {
    unquantize_multiplier = set1_ps<__m128>(config.unquantize_multiplier);
  }

  INTGEMM_SSE2 inline OutputRegister run(InputRegister input, Index offset) {
    return {
      mul_ps(cvtepi32_ps(input.first), unquantize_multiplier),
      mul_ps(cvtepi32_ps(input.second), unquantize_multiplier),
    };
  }

private:
  __m128 unquantize_multiplier;
};

template <>
class PostprocessImpl<Unquantize, CPUType::AVX2> {
public:
  using InputRegister = __m256i;
  using OutputRegister = __m256;

  INTGEMM_AVX2 PostprocessImpl(const Unquantize& config) {
    unquantize_multiplier = set1_ps<__m256>(config.unquantize_multiplier);
  }

  INTGEMM_AVX2 inline OutputRegister run(InputRegister input, Index offset) {
    return mul_ps(cvtepi32_ps(input), unquantize_multiplier);
  }

private:
  __m256 unquantize_multiplier;
};

#ifndef INTGEMM_NO_AVX512

template <>
class PostprocessImpl<Unquantize, CPUType::AVX512BW> {
public:
  using InputRegister = __m512i;
  using OutputRegister = __m512;

  INTGEMM_AVX512BW PostprocessImpl(const Unquantize& config) {
    unquantize_multiplier = set1_ps<__m512>(config.unquantize_multiplier);
  }

  INTGEMM_AVX512BW inline OutputRegister run(InputRegister input, Index offset) {
    return mul_ps(cvtepi32_ps(input), unquantize_multiplier);
  }

private:
  __m512 unquantize_multiplier;
};

#endif

/*
 * Add a bias term
 */
class AddBias {
public:
  const float* bias;
  const Index length;

  AddBias(const float* bias, Index length) : bias(bias), length(length) {}
};

template <>
class PostprocessImpl<AddBias, CPUType::SSE2> {
public:
  using InputRegister = dvector_t<CPUType::SSE2, float>;
  using OutputRegister = dvector_t<CPUType::SSE2, float>;

  PostprocessImpl(const AddBias& config) : config(config) {}

  INTGEMM_SSE2 inline OutputRegister run(InputRegister input, Index offset) {
    auto bias_term0123 = *reinterpret_cast<const __m128*>(config.bias + (offset % config.length));
    auto bias_term4567 = *reinterpret_cast<const __m128*>(config.bias + (offset % config.length) + 4);
    return {
      add_ps(input.first, bias_term0123),
      add_ps(input.second, bias_term4567),
    };
  }

private:
  const AddBias config;
};

template <>
class PostprocessImpl<AddBias, CPUType::AVX2> {
public:
  using InputRegister = __m256;
  using OutputRegister = __m256;

  PostprocessImpl(const AddBias& config) : config(config) {}

  INTGEMM_AVX2 inline OutputRegister run(InputRegister input, Index offset) {
    auto bias_term = *reinterpret_cast<const __m256*>(config.bias + (offset % config.length));
    return add_ps(input, bias_term);
  }

private:
  const AddBias config;
};

#ifndef INTGEMM_NO_AVX512

template <>
class PostprocessImpl<AddBias, CPUType::AVX512BW> {
public:
  using InputRegister = __m512;
  using OutputRegister = __m512;

  PostprocessImpl(const AddBias& config) : config(config) {}

  INTGEMM_AVX512BW inline OutputRegister run(InputRegister input, Index offset) {
    auto bias_term = *reinterpret_cast<const __m512*>(config.bias + (offset % config.length));
    return add_ps(input, bias_term);
  }

private:
  const AddBias config;
};

#endif

/*
 * ReLU
 */
class ReLU {};

template <>
class PostprocessImpl<ReLU, CPUType::SSE2> {
public:
  using InputRegister = dvector_t<CPUType::SSE2, float>;
  using OutputRegister = dvector_t<CPUType::SSE2, float>;

  PostprocessImpl(const ReLU& config) {}

  INTGEMM_SSE2 inline OutputRegister run(InputRegister input, Index offset) {
    static const auto const_zero = set1_ps<__m128>(0.f);
    return {
      max_ps(const_zero, input.first),
      max_ps(const_zero, input.second),
    };
  }
};

template <>
class PostprocessImpl<ReLU, CPUType::SSSE3> : public PostprocessImpl<ReLU, CPUType::SSE2> {};

template <>
class PostprocessImpl<ReLU, CPUType::AVX2> {
public:
  using InputRegister = __m256;
  using OutputRegister = __m256;

  PostprocessImpl(const ReLU& config) {}

  INTGEMM_AVX2 inline OutputRegister run(InputRegister input, Index offset) {
    static const auto const_zero = set1_ps<__m256>(0.f);
    return max_ps(const_zero, input);
  }
};

#ifndef INTGEMM_NO_AVX512

template <>
class PostprocessImpl<ReLU, CPUType::AVX512BW> {
public:
  using InputRegister = __m512;
  using OutputRegister = __m512;

  PostprocessImpl(const ReLU& config) {}

  INTGEMM_AVX512BW inline OutputRegister run(InputRegister input, Index offset) {
    static const auto const_zero = set1_ps<__m512>(0.f);
    return max_ps(const_zero, input);
  }
};

#endif

/*
 * ReLU_int8
 */
class ReLU_int8 {};

template <>
class PostprocessImpl<ReLU_int8, CPUType::SSE2> {
public:
  using InputRegister = __m128i;
  using OutputRegister = __m128i;

  PostprocessImpl(const ReLU_int8& config) {}

  INTGEMM_SSE2 inline OutputRegister run(InputRegister input, Index offset) {
    static const auto const_zero = setzero_si<__m128i>();
    return  _mm_and_si128(_mm_cmplt_epi8(const_zero, input), input);
  }
};

template <>
class PostprocessImpl<ReLU_int8, CPUType::AVX2> {
public:
  using InputRegister = __m256i;
  using OutputRegister = __m256i;

  PostprocessImpl(const ReLU_int8& config) {}

  INTGEMM_AVX2 inline OutputRegister run(InputRegister input, Index offset) {
    static const auto const_zero = setzero_si<__m256i>();
    return max_epi8(const_zero, input);
  }
};

#ifndef INTGEMM_NO_AVX512

template <>
class PostprocessImpl<ReLU_int8, CPUType::AVX512BW> {
public:
  using InputRegister = __m512i;
  using OutputRegister = __m512i;

  PostprocessImpl(const ReLU_int8& config) {}

  INTGEMM_AVX512BW inline OutputRegister run(InputRegister input, Index offset) {
    static const auto const_zero = setzero_si<__m512i>();
    return max_epi8(const_zero, input);
  }
};

#endif

/*
 * ReLU_int16
 */
class ReLU_int16 {};

template <>
class PostprocessImpl<ReLU_int16, CPUType::SSE2> {
public:
  using InputRegister = __m128i;
  using OutputRegister = __m128i;

  PostprocessImpl(const ReLU_int16& config) {}

  INTGEMM_SSE2 inline OutputRegister run(InputRegister input, Index offset) {
    static const auto const_zero = setzero_si<__m128i>();
    return max_epi16(const_zero, input);
  }
};

template <>
class PostprocessImpl<ReLU_int16, CPUType::AVX2> {
public:
  using InputRegister = __m256i;
  using OutputRegister = __m256i;

  PostprocessImpl(const ReLU_int16& config) {}

  INTGEMM_AVX2 inline OutputRegister run(InputRegister input, Index offset) {
    static const auto const_zero = setzero_si<__m256i>();
    return max_epi16(const_zero, input);
  }
};

#ifndef INTGEMM_NO_AVX512

template <>
class PostprocessImpl<ReLU_int16, CPUType::AVX512BW> {
public:
  using InputRegister = __m512i;
  using OutputRegister = __m512i;

  PostprocessImpl(const ReLU_int16& config) {}

  INTGEMM_AVX512BW inline OutputRegister run(InputRegister input, Index offset) {
    static const auto const_zero = setzero_si<__m512i>();
    return max_epi16(const_zero, input);
  }
};

#endif

/*
 * Sigmoid (uses Taylor series approximation of e^x)
 */
class Sigmoid {};

template <>
class PostprocessImpl<Sigmoid, CPUType::AVX2> {
public:
  using InputRegister = __m256;
  using OutputRegister = __m256;

  PostprocessImpl(const Sigmoid& config) {}

  INTGEMM_AVX2 inline OutputRegister run(InputRegister input, Index offset) {
    static const auto const_zero = set1_ps<__m256>(0.f);
    static const auto const_one = set1_ps<__m256>(1.f);

    auto x = input;
    auto minus_x = sub_ps(const_zero, x);
    auto e_x = exp_approx_taylor(x);
    auto e_minus_x = exp_approx_taylor(minus_x);

    auto sigmoid_case1 = _mm256_rcp_ps(add_ps(const_one, e_minus_x));
    auto sigmoid_case2 = mul_ps(e_x, _mm256_rcp_ps(add_ps(const_one, e_x)));

    auto nonnegative_x_mask = _mm256_cmp_ps(const_zero, x, _CMP_LT_OS);
    return _mm256_blendv_ps(sigmoid_case1, sigmoid_case2, nonnegative_x_mask);
  }
};

/*
 * Tanh (uses Taylor series approximation of e^x)
 */
class Tanh {};

template <>
class PostprocessImpl<Tanh, CPUType::AVX2> {
public:
  using InputRegister = __m256;
  using OutputRegister = __m256;

  PostprocessImpl(const Tanh& config) {}

  INTGEMM_AVX2 inline OutputRegister run(InputRegister input, Index offset) {
    const static auto const_zero = setzero_ps<__m256>();

    auto e_x = exp_approx_taylor(input);
    auto e_minus_x = exp_approx_taylor(sub_ps(const_zero, input));

    return div_ps(sub_ps(e_x, e_minus_x), add_ps(e_x, e_minus_x));
  }
};

#ifndef INTGEMM_NO_AVX512

template <>
class PostprocessImpl<Tanh, CPUType::AVX512BW> {
public:
  using InputRegister = __m512;
  using OutputRegister = __m512;

  PostprocessImpl(const Tanh& config) {}

  INTGEMM_AVX512BW inline OutputRegister run(InputRegister input, Index offset) {
    const static auto const_zero = setzero_ps<__m512>();

    auto e_x = exp_approx_taylor(input);
    auto e_minus_x = exp_approx_taylor(sub_ps(const_zero, input));

    return div_ps(sub_ps(e_x, e_minus_x), add_ps(e_x, e_minus_x));
  }
};

#endif

}
