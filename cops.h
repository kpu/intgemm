#include "intrinsics.h"

#include <exception>

namespace intgemm {

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
  AVX2 void InitRegister(float unquant_mult);

  float * C_;
  __m128 unquant_mult_128; // Registers
  __m256 unquant_mult_256;
};

DEFAULT void JustUnquantizeC::InitRegister(float) {
  throw UnsupportedCPU();
}

SSE2 void JustUnquantizeC::InitRegister(float unquant_mult) {
  //assert(reinterpret_cast<uintptr_t>(C_) % sizeof(__m128) == 0);
  unquant_mult_128 = _mm_set1_ps(unquant_mult);
}

AVX2 void JustUnquantizeC::InitRegister(float unquant_mult) {
  //assert(reinterpret_cast<uintptr_t>(C_) % sizeof(__m256) == 0);
  unquant_mult_128 = _mm_set1_ps(unquant_mult);
  unquant_mult_256 = _mm256_set1_ps(unquant_mult);
}

JustUnquantizeC::JustUnquantizeC(float *C, float unquant_mult) : C_(C) {
  InitRegister(unquant_mult);
}


inline void JustUnquantizeC::operator()(Index rowIDX, Index cols, Index colIDX, MultiplyResult128 result){
  *reinterpret_cast<__m128*>(C_ + rowIDX*cols + colIDX) = mul_ps(cvtepi32_ps(result.pack0123), unquant_mult_128);
  *reinterpret_cast<__m128*>(C_ + rowIDX*cols + colIDX + 4) = mul_ps(cvtepi32_ps(result.pack4567), unquant_mult_128);
}
#ifdef __AVX2__
inline void JustUnquantizeC::operator()(Index rowIDX, Index cols, Index colIDX, __m256i result) {
   *reinterpret_cast<__m256*>(C_ + rowIDX*cols + colIDX) = mul_ps(cvtepi32_ps(result), unquant_mult_256);
}
#endif
} //Namespace
