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

class ReLU {
  public:
    explicit ReLU(int32_t* C) : C_(C) {}

    class OnSSE2 {
      public:
        INTGEMM_SSE2 explicit OnSSE2(const ReLU& from)
          : C_(from.C_), zeros_(set1_epi32<__m128i>(0)) {
          assert(reinterpret_cast<uintptr_t>(C_) % sizeof(__m128i) == 0);
        }

        INTGEMM_SSE2 inline void operator()(Index rowIDX, Index cols, Index colIDX, MultiplyResult128 result) {
          __m128i positive0124 = _mm_cmplt_epi32(zeros_, result.pack0123);
          _mm_storeu_si128(reinterpret_cast<__m128i*>(C_ + rowIDX*cols + colIDX), zeros_);
          _mm_maskmoveu_si128(result.pack0123, positive0124, reinterpret_cast<char*>(C_ + rowIDX*cols + colIDX));

          __m128i positive4567 = _mm_cmplt_epi32(zeros_, result.pack4567);
          _mm_storeu_si128(reinterpret_cast<__m128i*>(C_ + rowIDX*cols + colIDX + 4), zeros_);
          _mm_maskmoveu_si128(result.pack4567, positive4567, reinterpret_cast<char*>(C_ + rowIDX*cols + colIDX + 4));
        }

      private:
        int32_t *C_;
        __m128i zeros_;
    };

    using OnSSSE2 = OnSSE2;

    class OnAVX2 {
      public:
        INTGEMM_AVX2 explicit OnAVX2(const ReLU& from)
          : C_(from.C_), zeros_(set1_epi32<__m256i>(0)) {
          assert(reinterpret_cast<uintptr_t>(C_) % sizeof(__m256i) == 0);
        }

        INTGEMM_AVX2 inline void operator()(Index rowIDX, Index cols, Index colIDX, __m256i result) {
          __m256i nonnegative = _mm256_max_epi32(zeros_, result);
          _mm256_storeu_si256(reinterpret_cast<__m256i*>(C_ + rowIDX*cols + colIDX), nonnegative);
        }

      private:
        int32_t *C_;
        __m256i zeros_;
    };

#ifndef INTGEMM_NO_AVX512
    class OnAVX512 {
      public:
        INTGEMM_AVX512BW explicit OnAVX512(const ReLU& from)
          : C_(from.C_), zeros_(set1_epi32<__m512i>(0)) {
          assert(reinterpret_cast<uintptr_t>(C_) % sizeof(__m512i) == 0);
        }

        INTGEMM_AVX512BW inline void operator()(Index rowIDX, Index cols, Index colIDX, __m512i result) {
          __m512i nonnegative = _mm512_max_epi32(zeros_, result);
          _mm512_storeu_si512(reinterpret_cast<__m512i*>(C_ + rowIDX*cols + colIDX), nonnegative);
        }

      private:
        int32_t *C_;
        __m512i zeros_;
    };
#endif

  private:
    int32_t *C_;
};

} //Namespace
