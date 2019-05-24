#pragma once

#include "intrinsics.h"
#include "postprocess_pipeline.h"
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
class PostprocessImpl<Unquantize, CPUType::CPU_SSE2> {
public:
  using InputRegister = RegisterPair128i;
  using OutputRegister = RegisterPair128;

  INTGEMM_SSE2 PostprocessImpl(const Unquantize& config) {
    unquantize_multiplier = set1_ps<__m128>(config.unquantize_multiplier);
  }

  INTGEMM_SSE2 inline OutputRegister run(InputRegister input) {
    return {
      mul_ps(cvtepi32_ps(input.pack0123), unquantize_multiplier),
      mul_ps(cvtepi32_ps(input.pack4567), unquantize_multiplier),
    };
  }

private:
  __m128 unquantize_multiplier;
};

template <>
class PostprocessImpl<Unquantize, CPUType::CPU_AVX2> {
public:
  using InputRegister = __m256i;
  using OutputRegister = __m256;

  INTGEMM_AVX2 PostprocessImpl(const Unquantize& config) {
    unquantize_multiplier = set1_ps<__m256>(config.unquantize_multiplier);
  }

  INTGEMM_AVX2 inline OutputRegister run(InputRegister input) {
    return mul_ps(cvtepi32_ps(input), unquantize_multiplier);
  }

private:
  __m256 unquantize_multiplier;
};

template <>
class PostprocessImpl<Unquantize, CPUType::CPU_AVX512BW> {
public:
  using InputRegister = __m512i;
  using OutputRegister = __m512;

  INTGEMM_AVX512BW PostprocessImpl(const Unquantize& config) {
    unquantize_multiplier = set1_ps<__m512>(config.unquantize_multiplier);
  }

  INTGEMM_AVX512BW inline OutputRegister run(InputRegister input) {
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
class PostprocessImpl<Identity, CPUType::CPU_SSE2> {
public:
  using InputRegister = RegisterPair128i;
  using OutputRegister = RegisterPair128i;

  PostprocessImpl(const Identity& config) {}

  INTGEMM_SSE2 inline OutputRegister run(InputRegister input) {
    return input;
  }
};

template <>
class PostprocessImpl<Identity, CPUType::CPU_AVX2> {
public:
  using InputRegister = __m256i;
  using OutputRegister = __m256i;

  PostprocessImpl(const Identity& config) {}

  INTGEMM_AVX2 inline OutputRegister run(InputRegister input) {
    return input;
  }
};

template <>
class PostprocessImpl<Identity, CPUType::CPU_AVX512BW> {
public:
  using InputRegister = __m512i;
  using OutputRegister = __m512i;

  PostprocessImpl(const Identity& config) {}

  INTGEMM_AVX512BW inline OutputRegister run(InputRegister input) {
    return input;
  }
};

/*
 * Add a bias term
 */
class AddBias {
public:
  const float* bias;

  AddBias(const float* bias) : bias(bias) {}
};

template <>
class PostprocessImpl<AddBias, CPUType::CPU_SSE2> {
public:
  using InputRegister = RegisterPair128;
  using OutputRegister = RegisterPair128;

  PostprocessImpl(const AddBias& config) : config(config) {}

  INTGEMM_SSE2 inline OutputRegister run(InputRegister input) {
    auto bias_term0123 = *reinterpret_cast<const __m128*>(config.bias);
    auto bias_term4567 = *reinterpret_cast<const __m128*>(config.bias);
    return {
      add_ps(input.pack0123, bias_term0123),
      add_ps(input.pack4567, bias_term4567),
    };
  }

private:
  const AddBias config;
};

template <>
class PostprocessImpl<AddBias, CPUType::CPU_AVX2> {
public:
  using InputRegister = __m256;
  using OutputRegister = __m256;

  PostprocessImpl(const AddBias& config) : config(config) {}

  INTGEMM_AVX2 inline OutputRegister run(InputRegister input) {
    auto bias_term = *reinterpret_cast<const __m256*>(config.bias);
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
class PostprocessImpl<ReLU, CPUType::CPU_SSE2> {
public:
  using InputRegister = RegisterPair128;
  using OutputRegister = RegisterPair128;

  PostprocessImpl(const ReLU& config) {}

  INTGEMM_SSE2 inline OutputRegister run(InputRegister input) {
    static const auto const_zero = set1_ps<__m128>(0.f);
    return {
      max_ps(const_zero, input.pack0123),
      max_ps(const_zero, input.pack4567),
    };
  }
};

template <>
class PostprocessImpl<ReLU, CPUType::CPU_SSSE3> : public PostprocessImpl<ReLU, CPUType::CPU_SSE2> {};

template <>
class PostprocessImpl<ReLU, CPUType::CPU_AVX2> {
public:
  using InputRegister = __m256;
  using OutputRegister = __m256;

  PostprocessImpl(const ReLU& config) {}

  INTGEMM_AVX2 inline OutputRegister run(InputRegister input) {
    static const auto const_zero = set1_ps<__m256>(0.f);
    return max_ps(const_zero, input);
  }
};

template <>
class PostprocessImpl<ReLU, CPUType::CPU_AVX512BW> {
public:
  using InputRegister = __m512;
  using OutputRegister = __m512;

  PostprocessImpl(const ReLU& config) {}

  INTGEMM_AVX512BW inline OutputRegister run(InputRegister input) {
    static const auto const_zero = set1_ps<__m512>(0.f);
    return max_ps(const_zero, input);
  }
};

}
