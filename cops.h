#pragma once
#include "intrinsics.h"

#include <exception>

namespace intgemm {

class JustUnquantizeC {
  public:
    JustUnquantizeC(float *C, float unquant_mult) : C_(C), unquant_mult_(unquant_mult) {}

    class OnSSE2 {
      public:
        INTGEMM_SSE2 explicit OnSSE2(const JustUnquantizeC &from)
          : C_(from.C_), unquant_mult_(_mm_set1_ps(from.unquant_mult_)) {
          assert(reinterpret_cast<uintptr_t>(C_) % sizeof(__m128i) == 0);
         }

        INTGEMM_SSE2 inline void operator()(Index rowIDX, Index cols, Index colIDX, MultiplyResult128 result) {
          *reinterpret_cast<__m128*>(C_ + rowIDX*cols + colIDX) = mul_ps(cvtepi32_ps(result.pack0123), unquant_mult_);
          *reinterpret_cast<__m128*>(C_ + rowIDX*cols + colIDX + 4) = mul_ps(cvtepi32_ps(result.pack4567), unquant_mult_);
        }
      private:
        float *C_;
        __m128 unquant_mult_;
    };

    class OnAVX2 {
      public:
        INTGEMM_AVX2 explicit OnAVX2(const JustUnquantizeC &from)
          : C_(from.C_), unquant_mult_(_mm256_set1_ps(from.unquant_mult_)) {
          assert(reinterpret_cast<uintptr_t>(C_) % sizeof(__m256i) == 0);
        }

        INTGEMM_AVX2 inline void operator()(Index rowIDX, Index cols, Index colIDX, __m256i result) {
          *reinterpret_cast<__m256*>(C_ + rowIDX*cols + colIDX) = mul_ps(cvtepi32_ps(result), unquant_mult_);
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

} //Namespace
