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
          : C_(from.C_), unquant_mult_(set1_ps<__m128>(from.unquant_mult_)) {
          assert(reinterpret_cast<uintptr_t>(C_) % sizeof(__m128i) == 0);
        }

        INTGEMM_SSE2 inline void operator()(Index rowIDX, Index cols, Index colIDX, MultiplyResult128 result) {
          static const auto zeros_ = setzero_ps<__m128>();

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
    };

    using OnSSSE3 = OnSSE2;

    class OnAVX2 {
      public:
        INTGEMM_AVX2 explicit OnAVX2(const ReLU& from)
          : C_(from.C_), unquant_mult_(set1_ps<__m256>(from.unquant_mult_)) {
          assert(reinterpret_cast<uintptr_t>(C_) % sizeof(__m256i) == 0);
        }

        INTGEMM_AVX2 inline void operator()(Index rowIDX, Index cols, Index colIDX, __m256i result) {
          static const auto zeros_ = setzero_ps<__m256>();

          auto nonnegative = max_ps(zeros_, unquantize(result, unquant_mult_));
          storeu_ps(C_ + rowIDX*cols + colIDX, nonnegative);
        }

      private:
        float* C_;
        __m256 unquant_mult_;
    };

#ifndef INTGEMM_NO_AVX512
    class OnAVX512 {
      public:
        INTGEMM_AVX512BW explicit OnAVX512(const ReLU& from)
          : C_(from.C_), unquant_mult_(set1_ps<__m512>(from.unquant_mult_)) {
          assert(reinterpret_cast<uintptr_t>(C_) % sizeof(__m512i) == 0);
        }

        INTGEMM_AVX512BW inline void operator()(Index rowIDX, Index cols, Index colIDX, __m512i result) {
          static const auto zeros_ = setzero_ps<__m512>();

          auto nonnegative = max_ps(zeros_, unquantize(result, unquant_mult_));
          storeu_ps(C_ + rowIDX*cols + colIDX, nonnegative);
        }

      private:
        float* C_;
        __m512 unquant_mult_;
    };
#endif

  private:
    float* C_;
    float unquant_mult_;
};

}
