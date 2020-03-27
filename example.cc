#include "intgemm.h"
// This is just for AlignedVector, which helps managed 64-byte aligned memory.
// Feel free to manage memory yourself.
#include "aligned.h" 
#include "callbacks.h"

#include <cassert>
#include <math.h>
#include <random>

int main() {
  using intgemm::Index;
  const Index A_rows = 1;
  // The shared dimension: A's columns and B's rows.
  const Index width = 64;
  const Index B_cols = 8;

  // This is a simple vector class that allocates memory aligned to 64 bytes.
  // You don't have to use it; just use aligned_alloc and friends directly.
  using intgemm::AlignedVector;
  AlignedVector<float> A(A_rows * width);
  AlignedVector<float> B(width * B_cols);

  // Fill with random values in range [-2, 2].
  std::mt19937 gen;
  std::uniform_real_distribution<float> dist(-2.f, 2.f);
  gen.seed(1);
  for (auto& it : A) {
    it = dist(gen);
  }
  for (auto& it : B) {
    it = dist(gen);
  }

  // Compute the top left corner of C as a sanity check.
  float top_left_reference = 0.0;
  for (Index w = 0; w < width; ++w) {
    top_left_reference += A[w] * B[w * B_cols];
  }

  // 16-bit multiplication.
  {
    // For 16-bit, Jacob Devlin recommends 1024 so as to not overflow in 32-bit accumulation.
    float quant_mult = 1024.0;
    AlignedVector<int16_t> A_prepared(A.size());
    AlignedVector<int16_t> B_prepared(B.size());
    // Quantize A.
    intgemm::Int16::PrepareA(A.begin(), A_prepared.begin(), quant_mult, A_rows, width);
    // Quantize and reshape B.
    // Typically you will do this once when parameters are loaded, not every time.
    intgemm::Int16::template PrepareB<8>(B.begin(), B_prepared.begin(), quant_mult, width, B_cols);

    AlignedVector<float> C(A_rows * B_cols);
    // Do the actual multiply.
    intgemm::Int16::template Multiply<1, 1>(A_prepared.begin(), B_prepared.begin(), A_rows, width, B_cols, intgemm::callbacks::UnquantizeAndWrite(1.0 / (quant_mult * quant_mult), C.begin()));
    // Sanity check.  C will be row major.
    assert(fabs(C[0] - top_left_reference) < 0.05);
  }

  // 8-bit multiplication.
  {
    // For 8-bit a good quantization multiplier is 127 / largest absolute value..
    float quant_mult = 127.0 / 2.0;
    AlignedVector<int8_t> A_prepared(A.size());
    AlignedVector<int8_t> B_prepared(B.size());
    // Quantize A.
    intgemm::Int8::PrepareA(A.begin(), A_prepared.begin(), quant_mult, A_rows, width);
    // Quantize and reshape B.
    // Typically you will do this once when parameters are loaded, not every time.
    intgemm::Int8::template PrepareB<8>(B.begin(), B_prepared.begin(), quant_mult, width, B_cols);

    AlignedVector<float> C(A_rows * B_cols);
    // Do the actual multiply.
    intgemm::Int8::template Multiply<1, 1>(A_prepared.begin(), B_prepared.begin(), A_rows, width, B_cols, intgemm::callbacks::UnquantizeAndWrite(1.0 / (quant_mult * quant_mult), C.begin()));
    // Sanity check.  C will be row major.
    assert(fabs(C[0] - top_left_reference) < 0.05);
  }
}
