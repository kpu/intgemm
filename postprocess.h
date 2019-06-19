#pragma once

#include "intrinsics.h"
#include "postprocess_pipeline.h"
#include "types.h"
#include "vec_utils.h"

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
  using InputRegister = RegisterPair128i;
  using OutputRegister = RegisterPair128;

  INTGEMM_SSE2 PostprocessImpl(const Unquantize& config) {
    unquantize_multiplier = set1_ps<__m128>(config.unquantize_multiplier);
  }

  INTGEMM_SSE2 inline OutputRegister run(InputRegister input, Index offset) {
    return {
      mul_ps(cvtepi32_ps(input.pack0123), unquantize_multiplier),
      mul_ps(cvtepi32_ps(input.pack4567), unquantize_multiplier),
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

/*
 * Identity
 */
class Identity {};

template <>
class PostprocessImpl<Identity, CPUType::SSE2> {
public:
  using InputRegister = RegisterPair128i;
  using OutputRegister = RegisterPair128i;

  PostprocessImpl(const Identity& config) {}

  INTGEMM_SSE2 inline OutputRegister run(InputRegister input, Index offset) {
    return input;
  }
};

template <>
class PostprocessImpl<Identity, CPUType::AVX2> {
public:
  using InputRegister = __m256i;
  using OutputRegister = __m256i;

  PostprocessImpl(const Identity& config) {}

  INTGEMM_AVX2 inline OutputRegister run(InputRegister input, Index offset) {
    return input;
  }
};

template <>
class PostprocessImpl<Identity, CPUType::AVX512BW> {
public:
  using InputRegister = __m512i;
  using OutputRegister = __m512i;

  PostprocessImpl(const Identity& config) {}

  INTGEMM_AVX512BW inline OutputRegister run(InputRegister input, Index offset) {
    return input;
  }
};

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
  using InputRegister = RegisterPair128;
  using OutputRegister = RegisterPair128;

  PostprocessImpl(const AddBias& config) : config(config) {}

  INTGEMM_SSE2 inline OutputRegister run(InputRegister input, Index offset) {
    auto bias_term0123 = *reinterpret_cast<const __m128*>(config.bias + (offset % config.length));
    auto bias_term4567 = *reinterpret_cast<const __m128*>(config.bias + (offset % config.length) + 4);
    return {
      add_ps(input.pack0123, bias_term0123),
      add_ps(input.pack4567, bias_term4567),
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

/*
 * ReLU
 */
class ReLU {};

template <>
class PostprocessImpl<ReLU, CPUType::SSE2> {
public:
  using InputRegister = RegisterPair128;
  using OutputRegister = RegisterPair128;

  PostprocessImpl(const ReLU& config) {}

  INTGEMM_SSE2 inline OutputRegister run(InputRegister input, Index offset) {
    static const auto const_zero = set1_ps<__m128>(0.f);
    return {
      max_ps(const_zero, input.pack0123),
      max_ps(const_zero, input.pack4567),
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

}
