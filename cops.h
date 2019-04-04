#include <exception>
#include <xmmintrin.h>
#include <emmintrin.h>
#include <immintrin.h>

#include "types.h"

namespace intgemm {

static inline __m128 cvtepi32_ps(__m128i arg) {
  return _mm_cvtepi32_ps(arg);
}
static inline __m128 mul_ps (__m128 a, __m128 b) {
  return _mm_mul_ps(a, b);
}

static inline __m256 cvtepi32_ps(__m256i arg) {
  return _mm256_cvtepi32_ps(arg);
}
static inline __m256 mul_ps (__m256 a, __m256 b) {
  return _mm256_mul_ps(a, b);
}

struct MultiplyResult128 {
  __m128i pack0123, pack4567;
};

// This will be thrown if a CPU isn't supported by the routines (16-bit without SSE2 or 8-bit without SSSE3).
class UnsupportedCPU : public std::exception {
  public:
    UnsupportedCPU();

    ~UnsupportedCPU() throw();

    const char *what() const throw() override;
};

UnsupportedCPU::UnsupportedCPU() {}

UnsupportedCPU::~UnsupportedCPU() throw() {}

const char *UnsupportedCPU::what() const throw() {
  return "Integer matrix multiplication has not been efficiently implemented for your CPU.";
}

class JustUnquantizeC {
public:
 JustUnquantizeC(float *C, float unquant_mult);

 inline void operator()(Index rowIDX, Index cols, Index colIDX, MultiplyResult128 result);
 inline void operator()(Index rowIDX, Index cols, Index colIDX, __m256i result);

private:
  DEFAULT void InitRegister(float unquant_mult);
  SSE2 void InitRegister(float unquant_mult);
  SSSE3 void InitRegister(float unquant_mult);
  AVX2 void InitRegister(float unquant_mult);
  AVX512F void InitRegister(float unquant_mult);

  float * C_;
  __m128 unquant_mult_128; // Registers
  __m256 unquant_mult_256;
};

DEFAULT void JustUnquantizeC::InitRegister(float) {
  throw UnsupportedCPU();
}

SSE2 void JustUnquantizeC::InitRegister(float unquant_mult) {
  unquant_mult_128 = _mm_set1_ps(unquant_mult);
}

SSSE3 void JustUnquantizeC::InitRegister(float unquant_mult) {
  unquant_mult_128 = _mm_set1_ps(unquant_mult);
}

AVX2 void JustUnquantizeC::InitRegister(float unquant_mult) {
  unquant_mult_256 = _mm256_set1_ps(unquant_mult);
}

AVX512F void JustUnquantizeC::InitRegister(float unquant_mult) {
  unquant_mult_256 = _mm256_set1_ps(unquant_mult);
}

JustUnquantizeC::JustUnquantizeC(float *C, float unquant_mult) : C_(C) {
  InitRegister(unquant_mult);
}


inline void JustUnquantizeC::operator()(Index rowIDX, Index cols, Index colIDX, MultiplyResult128 result){
  *reinterpret_cast<__m128*>(C_ + rowIDX*cols + colIDX) = mul_ps(cvtepi32_ps(result.pack0123), unquant_mult_128);
  *reinterpret_cast<__m128*>(C_ + rowIDX*cols + colIDX + 4) = mul_ps(cvtepi32_ps(result.pack4567), unquant_mult_128);
}
inline void JustUnquantizeC::operator()(Index rowIDX, Index cols, Index colIDX, __m256i result) {
  *reinterpret_cast<__m256*>(C_ + rowIDX*cols + colIDX) = mul_ps(cvtepi32_ps(result), unquant_mult_256);
}

} //Namespace
