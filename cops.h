#pragma once

#include "intrinsics.h"
#include "vec_utils.h"

#include <cassert>
#include <exception>

namespace intgemm {

class JustUnquantizeC {
  public:
    JustUnquantizeC(float *C, float unquant_mult) : C_(C), unquant_mult_(unquant_mult) {}

    class OnSSE2 {
      public:
        INTGEMM_SSE2 explicit OnSSE2(const JustUnquantizeC &from)
          : C_(from.C_), unquant_mult_(set1_ps<__m128>(from.unquant_mult_)) {
          assert(reinterpret_cast<uintptr_t>(C_) % sizeof(__m128i) == 0);
         }

        INTGEMM_SSE2 inline void operator()(Index rowIDX, Index cols, Index colIDX, MultiplyResult128 result) {
          storeu_ps(C_ + rowIDX*cols + colIDX    , unquantize(result.pack0123, unquant_mult_));
          storeu_ps(C_ + rowIDX*cols + colIDX + 4, unquantize(result.pack4567, unquant_mult_));
        }
      private:
        float *C_;
        __m128 unquant_mult_;
    };

    class OnAVX2 {
      public:
        INTGEMM_AVX2 explicit OnAVX2(const JustUnquantizeC &from)
          : C_(from.C_), unquant_mult_(set1_ps<__m256>(from.unquant_mult_)) {
          assert(reinterpret_cast<uintptr_t>(C_) % sizeof(__m256i) == 0);
        }

        INTGEMM_AVX2 inline void operator()(Index rowIDX, Index cols, Index colIDX, __m256i result) {
          storeu_ps(C_ + rowIDX*cols + colIDX, unquantize(result, unquant_mult_));
        }

      private:
        float *C_;
        __m256 unquant_mult_;
    };

  private:
    float *C_;
    float unquant_mult_;
};

class Identity {
  public:
    explicit Identity(int32_t *C) : C_(C) {}

    class OnSSE2 {
      public:
        INTGEMM_SSE2 explicit OnSSE2(const Identity &from)
          : C_(from.C_) {
          assert(reinterpret_cast<uintptr_t>(C_) % sizeof(__m128i) == 0);
         }

        INTGEMM_SSE2 inline void operator()(Index rowIDX, Index cols, Index colIDX, MultiplyResult128 result) {
          _mm_storeu_si128(reinterpret_cast<__m128i*>(C_ + rowIDX*cols + colIDX), result.pack0123);
          _mm_storeu_si128(reinterpret_cast<__m128i*>(C_ + rowIDX*cols + colIDX + 4), result.pack4567);
        }
      private:
        int32_t *C_;
    };

    class OnAVX2 {
      public:
        INTGEMM_AVX2 explicit OnAVX2(const Identity &from)
          : C_(from.C_) {
          assert(reinterpret_cast<uintptr_t>(C_) % sizeof(__m256i) == 0);
        }

        INTGEMM_AVX2 inline void operator()(Index rowIDX, Index cols, Index colIDX, __m256i result) {
          _mm256_storeu_si256(reinterpret_cast<__m256i*>(C_ + rowIDX*cols + colIDX), result);
        }

      private:
        int32_t *C_;
    };

  private:
    int32_t *C_;
};

class ReLU {
  public:
    explicit ReLU(float *C, float unquant_mult) : C_(C), unquant_mult_(unquant_mult) {}

    class OnSSE2 {
      public:
        INTGEMM_SSE2 explicit OnSSE2(const ReLU& from)
          : C_(from.C_), zeros_(setzero_ps<__m128>()), unquant_mult_(set1_ps<__m128>(from.unquant_mult_)) {
          assert(reinterpret_cast<uintptr_t>(C_) % sizeof(__m128i) == 0);
        }

        INTGEMM_SSE2 inline void operator()(Index rowIDX, Index cols, Index colIDX, MultiplyResult128 result) {
          auto unquantized0123 = unquantize(result.pack0123, unquant_mult_);
          auto nonnegative0123 = max_ps(zeros_, unquantized0123);
          storeu_ps(C_ + rowIDX*cols + colIDX, nonnegative0123);

          auto unquantized4567 = unquantize(result.pack4567, unquant_mult_);
          auto nonnegative4567 = max_ps(zeros_, unquantized4567);
          storeu_ps(C_ + rowIDX*cols + colIDX + 4, nonnegative4567);
        }

      private:
        float* C_;
        __m128 unquant_mult_;
        __m128 zeros_;
    };

    using OnSSSE2 = OnSSE2;

    class OnAVX2 {
      public:
        INTGEMM_AVX2 explicit OnAVX2(const ReLU& from)
          : C_(from.C_), zeros_(setzero_ps<__m256>()), unquant_mult_(set1_ps<__m256>(from.unquant_mult_)) {
          assert(reinterpret_cast<uintptr_t>(C_) % sizeof(__m256i) == 0);
        }

        INTGEMM_AVX2 inline void operator()(Index rowIDX, Index cols, Index colIDX, __m256i result) {
          auto nonnegative = max_ps(zeros_, unquantize(result, unquant_mult_));
          storeu_ps(C_ + rowIDX*cols + colIDX, nonnegative);
        }

      private:
        float* C_;
        __m256 unquant_mult_;
        __m256 zeros_;
    };

#ifndef INTGEMM_NO_AVX512
    class OnAVX512 {
      public:
        INTGEMM_AVX512BW explicit OnAVX512(const ReLU& from)
          : C_(from.C_), zeros_(setzero_ps<__m512>()), unquant_mult_(set1_ps<__m512>(from.unquant_mult_)) {
          assert(reinterpret_cast<uintptr_t>(C_) % sizeof(__m512i) == 0);
        }

        INTGEMM_AVX512BW inline void operator()(Index rowIDX, Index cols, Index colIDX, __m512i result) {
          auto nonnegative = max_ps(zeros_, unquantize(result, unquant_mult_));
          storeu_ps(C_ + rowIDX*cols + colIDX, nonnegative);
        }

      private:
        float* C_;
        __m512 unquant_mult_;
        __m512 zeros_;
    };
#endif

  private:
    float* C_;
    float unquant_mult_;
};

/*
 * Approximate sigmoid using taylor series in point x = 0.
 *
 * Taylor series is expansed to 9th derivative:
 *   sigmoid(x) ~ 1/2 + x/4 - x^3/48 + x^5/480 - 17x^7/80640 + 31x^9/1451520
 */
class SigmoidTaylor {
  public:
    explicit SigmoidTaylor(float* C, float unquant_mult) : C_(C), unquant_mult_(unquant_mult) {}

    class OnAVX2 {
      public:
        INTGEMM_AVX2 explicit OnAVX2(const SigmoidTaylor& from)
          : C_(from.C_), unquant_mult_(set1_ps<__m256>(from.unquant_mult_)) {
          assert(reinterpret_cast<uintptr_t>(C_) % sizeof(__m256) == 0);
        }

        INTGEMM_AVX2 inline void operator()(Index rowIDX, Index cols, Index colIDX, __m256i result) {
          static const __m256 coeffs[] = {
            set1_ps<__m256>(31.f / 1451520),
            set1_ps<__m256>(17.f / 80640),
            set1_ps<__m256>(1.f / 480),
            set1_ps<__m256>(1.f / 48),
            set1_ps<__m256>(1.f / 4),
          };
          static const __m256 const_half = set1_ps<__m256>(1.f / 2);

          auto x = unquantize(result, unquant_mult_);
          auto x_squared = mul_ps(x, x);

          // Horner's method
          auto sigmoid = mul_ps(x_squared, coeffs[0]);
          sigmoid = sub_ps(sigmoid, coeffs[1]);
          sigmoid = mul_ps(sigmoid, x_squared);
          sigmoid = add_ps(sigmoid, coeffs[2]);
          sigmoid = mul_ps(sigmoid, x_squared);
          sigmoid = sub_ps(sigmoid, coeffs[3]);
          sigmoid = mul_ps(sigmoid, x_squared);
          sigmoid = add_ps(sigmoid, coeffs[4]);
          sigmoid = mul_ps(sigmoid, x);
          sigmoid = add_ps(sigmoid, const_half);

          storeu_ps(C_ + rowIDX*cols + colIDX, sigmoid);
        }

      private:
        float *C_;
        __m256 unquant_mult_;
    };

  private:
    float *C_;
    float unquant_mult_;
};

// TODO: Can we assume something about range of result values? Otherwise we need to use more expensive stable
// version of sigmoid:
//
//              | 1 / (1 + e^-x),   for x >= 0,  (1)
// sigmoid(x) = |
//              | e^x / (1 + e^x),  for x < 0,   (2)
//
class SigmoidReyoung { // intrinsics = 8 + 2 * 31 (exp_approx_reyoung) = 70
  public:
    explicit SigmoidReyoung(float* C, float unquant_mult) : C_(C), unquant_mult_(unquant_mult) {}

    class OnAVX2 {
      public:
        INTGEMM_AVX2 explicit OnAVX2(const SigmoidReyoung& from)
          : C_(from.C_), zeros_(setzero_ps<__m256>()), ones_(set1_ps<__m256>(1.f)), unquant_mult_(set1_ps<__m256>(from.unquant_mult_)) {
          assert(reinterpret_cast<uintptr_t>(C_) % sizeof(__m256) == 0);
        }

        INTGEMM_AVX2 inline void operator()(Index rowIDX, Index cols, Index colIDX, __m256i result) {
          auto x = unquantize(result, unquant_mult_);
          auto minus_x = sub_ps(zeros_, x);
          auto e_x = exp_approx_reyoung(x);
          auto e_minus_x = exp_approx_reyoung(minus_x);

          auto sigmoid_case1 = _mm256_rcp_ps(add_ps(ones_, e_minus_x));
          auto sigmoid_case2 = mul_ps(e_x, _mm256_rcp_ps(add_ps(ones_, e_x)));

          auto nonnegative_x_mask = _mm256_cmp_ps(zeros_, x, _CMP_LT_OS);
          auto sigmoid = _mm256_blendv_ps(sigmoid_case1, sigmoid_case2, nonnegative_x_mask);
          storeu_ps(C_ + rowIDX*cols + colIDX, sigmoid);
        }

      private:
        float *C_;
        __m256 unquant_mult_;
        __m256 zeros_;
        __m256 ones_;
    };

  private:
    float *C_;
    float unquant_mult_;
};

// TODO: Can we assume something about range of result values? Otherwise we need to use more expensive stable
// version of sigmoid:
//
//              | 1 / (1 + e^-x),   for x >= 0,  (1)
// sigmoid(x) = |
//              | e^x / (1 + e^x),  for x < 0,   (2)
//
class SigmoidExpTaylor { // intrinsics = 8 + 2 * 21 (exp_approx_taylor) = 50
  public:
    explicit SigmoidExpTaylor(float* C, float unquant_mult) : C_(C), unquant_mult_(unquant_mult) {}

    class OnAVX2 {
      public:
        INTGEMM_AVX2 explicit OnAVX2(const SigmoidExpTaylor& from)
          : C_(from.C_), zeros_(setzero_ps<__m256>()), ones_(set1_ps<__m256>(1.f)), unquant_mult_(set1_ps<__m256>(from.unquant_mult_)) {
          assert(reinterpret_cast<uintptr_t>(C_) % sizeof(__m256) == 0);
        }

        INTGEMM_AVX2 inline void operator()(Index rowIDX, Index cols, Index colIDX, __m256i result) {
          auto x = unquantize(result, unquant_mult_);
          auto minus_x = sub_ps(zeros_, x);
          auto e_x = exp_approx_taylor(x);
          auto e_minus_x = exp_approx_taylor(minus_x);

          auto sigmoid_case1 = _mm256_rcp_ps(add_ps(ones_, e_minus_x));
          auto sigmoid_case2 = mul_ps(e_x, _mm256_rcp_ps(add_ps(ones_, e_x)));

          auto nonnegative_x_mask = _mm256_cmp_ps(zeros_, x, _CMP_LT_OS);
          auto sigmoid = _mm256_blendv_ps(sigmoid_case1, sigmoid_case2, nonnegative_x_mask);
          storeu_ps(C_ + rowIDX*cols + colIDX, sigmoid_case1);
        }

      private:
        float *C_;
        __m256 unquant_mult_;
        __m256 zeros_;
        __m256 ones_;
    };

  private:
    float *C_;
    float unquant_mult_;
};

}
