#include "intgemm.h"
#include "aligned.h"
#include "interleave.h"
#include "multiply.h"

#define CATCH_CONFIG_RUNNER 
#include "3rd_party/catch.hpp"
#define CHECK_MESSAGE(cond, msg) do { INFO(msg); CHECK(cond); } while((void)0, 0)
#define CHECK_FALSE_MESSAGE(cond, msg) do { INFO(msg); CHECK_FALSE(cond); } while((void)0, 0)
#define REQUIRE_MESSAGE(cond, msg) do { INFO(msg); REQUIRE(cond); } while((void)0, 0)
#define REQUIRE_FALSE_MESSAGE(cond, msg) do { INFO(msg); REQUIRE_FALSE(cond); } while((void)0, 0)

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstring>
#include <cstdio>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <memory>
#include <random>
#include <sstream>

namespace intgemm {

// Rearrange a tile of simd x unroll entries.
template <class V> void SlowRearrangeTile(const V *from, V *to, int simd, int unroll, Index cols) {
  for (int i = 0; i < unroll; ++i) {
    for (int j = 0; j < simd; ++j) {
      to[simd * i + j] = from[cols * j + i];
    }
  }
}

template <class V> void SlowRearrange(const V *from, V *to, int simd, int unroll, Index rows, Index cols) {
  for (Index c = 0; c < cols; c += unroll) {
    for (Index r = 0; r < rows; r += simd) {
      SlowRearrangeTile(from + cols * r + c, to, simd, unroll, cols);
      to += unroll * simd;
    }
  }
}

template <class V> void SlowTranspose(const V *from, V *to, Index rows, Index cols) {
  for (Index r = 0; r < rows; ++r) {
    for (Index c = 0; c < cols; ++c) {
      to[rows * c + r] = from[cols * r + c];
    }
  }
}


INTGEMM_SSE2 TEST_CASE("Transpose 16", "[transpose]") {
  if (kCPU < CPU_SSE2) return;
  AlignedVector<int16_t> input(8 * 8);
  for (int16_t i = 0; i < 64; ++i) {
    input.get()[i] = i;
  }
  AlignedVector<int16_t> ref(8 * 8);
  SlowTranspose(input.get(), ref.get(), 8, 8);

  // Overwrite input.
  __m128i *t = reinterpret_cast<__m128i*>(input.get());
  Transpose16InLane(t[0], t[1], t[2], t[3], t[4], t[5], t[6], t[7]);

  for (int16_t i = 0; i < 64; ++i) {
  	CHECK_MESSAGE(ref.get()[i] == input.get()[i], "16-bit transpose failure at: " << i << ": " << ref.get()[i] << " != " << input.get()[i]);
  }
}

INTGEMM_SSSE3 TEST_CASE("Transpose 8", "[transpose]") {
  if (kCPU < CPU_SSSE3) return;
  AlignedVector<int8_t> input(16 * 16);
  for (int i = 0; i < 16 * 16; ++i) {
    input.get()[i] = i;
  }
  AlignedVector<int8_t> ref(16 * 16);
  SlowTranspose(input.get(), ref.get(), 16, 16);

  // Overwrite input.
  __m128i *t = reinterpret_cast<__m128i*>(input.get());
  Transpose8InLane(t[0], t[1], t[2], t[3], t[4], t[5], t[6], t[7], t[8], t[9], t[10], t[11], t[12], t[13], t[14], t[15]);

  for (int i = 0; i < 16 * 16; ++i) {
    CHECK_MESSAGE(ref.get()[i] == input.get()[i], "8-bit transpose failure at " << i << ": " << (int16_t)ref.get()[i] << " != " << (int16_t)input.get()[i]);
  }
}

template <class T> std::string PrintMatrix(const T *mem, Index rows, Index cols) {
  std::ostringstream out;
  for (Index r = 0; r < rows; ++r) {
    for (Index c = 0; c < cols; ++c) {
      out << std::setw(4) << (int64_t) mem[r * cols + c] << ' ';
    }
    out << '\n';
  }
  return out.str();
}

template <class Routine> void TestPrepare(Index rows = 32, Index cols = 16) {
  std::mt19937 gen;
  // Go somewhat out of range too.
  std::uniform_real_distribution<float> dist(-129.0, 129.0);
  // Create array.
  AlignedVector<float> input(rows * cols);
  for (Index i = 0; i < rows * cols; ++i) {
    input.get()[i] = dist(gen);
  }

  typedef typename Routine::Integer Integer;
  // Call Prepare
  AlignedVector<Integer> test(rows * cols);
  Routine::PrepareB(input.get(), test.get(), 1, rows, cols);

  // Compute reference output.
  AlignedVector<Integer> quantized(rows * cols);
  Routine::Quantize(input.get(), quantized.get(), 1, rows * cols);
  AlignedVector<Integer> reference(rows * cols);
  // Note this won't work for Int8/Int16 generic routines because tile sizes vary.
  SlowRearrange<Integer>(quantized.get(), reference.get(), Routine::kBTileRow, Routine::kBTileCol, rows, cols);
  CHECK_MESSAGE(memcmp(reference.get(), test.get(), rows * cols * sizeof(Integer)) == 0, Routine::kName << " Mismatch:\n" <<
  	"Quantized Input" << '\n' << PrintMatrix(quantized.get(), rows, cols) << "Reference" << '\n' <<
  	 PrintMatrix(reference.get(), rows, cols) << "Routine" << '\n' << PrintMatrix(test.get(), rows, cols));
}

TEST_CASE("Prepare AVX512", "[prepare]") {
  if (kCPU < CPU_AVX512BW) return;
#ifndef INTGEMM_NO_AVX512
	TestPrepare<AVX512_8bit>(64, 8);
	TestPrepare<AVX512_8bit>(256, 32);
    TestPrepare<AVX512_16bit>(64, 8);
    TestPrepare<AVX512_16bit>(256, 32);
#endif
}

TEST_CASE("Prepare AVX2", "[prepare]") {
  if (kCPU < CPU_AVX2) return;
  TestPrepare<AVX2_8bit>(64, 32);
  TestPrepare<AVX2_16bit>(64, 32);
}

TEST_CASE("Prepare SSSE3", "[prepare]") {
  if (kCPU < CPU_SSSE3) return;
  TestPrepare<SSSE3_8bit>(16, 8);
  TestPrepare<SSSE3_8bit>(32, 16);
  TestPrepare<SSSE3_8bit>(32, 32);
}

TEST_CASE("Prepare SSE2", "[prepare]") {
  if (kCPU < CPU_SSE2) return;
  TestPrepare<SSE2_16bit>(8, 8);
  TestPrepare<SSE2_16bit>(32, 32);
}

template <class Routine> void TestSelectColumnsB(Index rows = 64, Index cols = 16) {
  std::mt19937 gen;
  // Go somewhat out of range too.
  std::uniform_real_distribution<float> dist(-129.0, 129.0);
  AlignedVector<float> input(rows * cols);
  for (Index i = 0; i < rows * cols; ++i) {
    input.get()[i] = dist(gen);
  }
  typedef typename Routine::Integer Integer;
  AlignedVector<Integer> prepared(rows * cols);
  Routine::PrepareB(input.get(), prepared.get(), 1, rows, cols);

  int kSelectCols = 24;
  Index select_cols[kSelectCols];
  std::uniform_int_distribution<Index> col_dist(0, cols - 1);
  for (int i = 0; i < kSelectCols; ++i) {
    select_cols[i] = col_dist(gen);
  }

  AlignedVector<Integer> test(rows * kSelectCols);
  Routine::SelectColumnsB(prepared.get(), test.get(), rows, select_cols, select_cols + kSelectCols);

  // Select columns manually in float space.
  AlignedVector<float> selected(rows * kSelectCols);
  for (Index r = 0; r < rows; ++r) {
    for (int c = 0; c < kSelectCols; ++c) {
      assert(c + r * kSelectCols < rows * kSelectCols);
      selected[c + r * kSelectCols] = input[select_cols[c] + r * cols];
    }
  }
  AlignedVector<Integer> ref(rows * kSelectCols);
  Routine::PrepareB(selected.get(), ref.get(), 1, rows, kSelectCols);
  CHECK_MESSAGE(memcmp(ref.get(), test.get(), sizeof(Integer) * rows * kSelectCols) == 0, "Reference:\n" <<
  	PrintMatrix(ref.get(), rows, kSelectCols) << PrintMatrix(test.get(), rows, kSelectCols));
}

TEST_CASE("SelectColumnsB AVX512", "[select]") {
  if (kCPU < CPU_AVX512BW) return;
#ifndef INTGEMM_NO_AVX512
    TestSelectColumnsB<AVX512_8bit>();
    TestSelectColumnsB<AVX512_16bit>(256, 256);
#endif
}

TEST_CASE("SelectColumnsB AVX2", "[select]") {
  if (kCPU < CPU_AVX2) return;
  TestSelectColumnsB<AVX2_8bit>(256, 256);
  TestSelectColumnsB<AVX2_16bit>(256, 256);
}

TEST_CASE("SelectColumnsB SSSE3", "[select]") {
  if (kCPU < CPU_SSSE3) return;
  TestSelectColumnsB<SSSE3_8bit>();
  TestSelectColumnsB<SSSE3_8bit>(256, 256);
}

TEST_CASE("SelectColumnsB SSE2", "[select]") {
  if (kCPU < CPU_SSE2) return;
  TestSelectColumnsB<SSE2_16bit>();
  TestSelectColumnsB<SSE2_16bit>(256, 256);
}

template <class Register> void TestMax() {
  Register r = set1_ps<Register>(-2.0);
  for (std::size_t i = 0; i < sizeof(Register) / sizeof(float); ++i) {
    Register c = r;
    reinterpret_cast<float*>(&c)[i] = -1.0;
    CHECK_MESSAGE((MaxFloat32(c) == -1.0), "MaxFloat32 produced " << MaxFloat32(c));
  }
}

TEST_CASE("Max", "[max]") {
  TestMax<__m128>();
}

void CompareMaxAbs(const float *begin, const float *end, float test) {
  float largest = fabs(*std::max_element(begin, end));
  float smallest = fabs(*std::min_element(begin, end));
  largest = std::max(largest, smallest);
  CHECK_MESSAGE(largest == test, "Error: " << largest << " versus " << test);
}

template <float (*Backend) (const float *, const float *)> void TestMaxAbsolute() {
  std::mt19937 gen;
  std::uniform_real_distribution<float> dist(-8.0, 8.0);
  const int kLength = 64;
  AlignedVector<float> test(kLength);
  // 64 tries.
  for (int t = 0; t < 64; ++t) {
    // Fill with [-8, 8).
    for (int i = 0; i < kLength; ++i) {
      test[i] = dist(gen);
    }
    CompareMaxAbs(test.get(), test.get() + kLength, Backend(test.get(), test.get() + kLength));
    test[t] = -32.0;
    CompareMaxAbs(test.get(), test.get() + kLength, Backend(test.get(), test.get() + kLength));
    test[t] = 32.0;
    CompareMaxAbs(test.get(), test.get() + kLength, Backend(test.get(), test.get() + kLength));
  }
}

TEST_CASE("MaxAbsolute SSE2", "[max]") {
  if (kCPU < CPU_SSE2) return;
  TestMaxAbsolute<sse2::MaxAbsolute>();
}

TEST_CASE("MaxAbsolute AVX2", "[max]") {
  if (kCPU < CPU_AVX2) return;
  TestMaxAbsolute<avx2::MaxAbsolute>();
}

TEST_CASE("MaxAbsolute AVX512F", "[max]") {
  if (kCPU < CPU_AVX512BW) return;
  #ifndef INTGEMM_NO_AVX512
  TestMaxAbsolute<avx512f::MaxAbsolute>();
  #endif
}

// Based on https://arxiv.org/abs/1705.01991

// Copyright (c) 2017 Microsoft Corporation

// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.
// Compute A*B slowly in floats.
void SlowRefFloat(const float *A, const float *B, float *C, Index A_rows, Index width, Index B_cols) {
  for (Index r = 0; r < A_rows; ++r) {
    for (Index c = 0; c < B_cols; ++c) {
      float sum = 0.0f;
      for (Index k = 0; k < width; ++k) {
        sum += A[r * width + k] * B[k * B_cols + c];
      }
      C[r * B_cols + c] = sum;
    }
  }
}

// Compute A*B slowly from integers.
template <class Integer> void SlowRefInt(const Integer *A, const Integer *B, float *C, float unquant_mult, Index A_rows, Index width, Index B_cols) {
  for (Index r = 0; r < A_rows; ++r) {
    for (Index c = 0; c < B_cols; ++c) {
      int32_t sum = 0;
      for (Index k = 0; k < width; ++k) {
        sum += static_cast<int16_t>(A[r * width + k]) * static_cast<int16_t>(B[k * B_cols + c]);
      }
      C[r * B_cols + c] = sum * unquant_mult;
    }
  }
}


void Compare(const float *float_ref, const float *int_ref, const float *int_test, std::size_t size, std::string test_info,
 float int_tolerance, float float_tolerance, float MSE_float_tolerance, float MSE_int_tolerance) {
  float int_sum = 0.0, float_sum = 0.0;
  for (std::size_t i = 0; i < size; ++i) {
    float int_diff = int_ref[i] - int_test[i];
    float float_diff = float_ref[i] - int_test[i];
    CHECK_MESSAGE(fabs(int_diff) <= int_tolerance, test_info << "Inaccurate compared to int reference at " << i << ' ' << int_ref[i] << ' ' << int_test[i]);
    CHECK_MESSAGE(fabs(float_diff) <= float_tolerance, test_info << "Inaccurate compared to float reference at " << i << ' ' << float_ref[i] << ' ' << int_test[i]);
    int_sum += int_diff * int_diff;
    float_sum += float_diff * float_diff;
  }
  CHECK_MESSAGE(fabs(sqrt(float_sum / size)) <= MSE_float_tolerance, test_info << "Float MSE = " << sqrt(float_sum / size));
  CHECK_MESSAGE(fabs(sqrt(int_sum / size)) <= MSE_int_tolerance, test_info << "Int MSE = " << sqrt(int_sum / size));
}

template <class Routine, class WriteC> void TestMultiply(Index A_rows, Index width, Index B_cols,
 float int_tolerance=.1, float float_tolerance=1, float MSE_float_tolerance=0, float MSE_int_tolerance=0) {
  typedef typename Routine::Integer Integer;
  std::ostringstream info;
  info << Routine::kName << "\t" << A_rows << '\t' << width << '\t' << B_cols << '\n';

  // Initialize A and B.
  AlignedVector<float> A(A_rows * width);
  AlignedVector<float> B(width * B_cols);
  std::mt19937 gen;
  std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
  for (Index i = 0; i < A_rows * width; i++) {
    A.get()[i] = dist(gen);
  }
  for (Index i = 0; i < width * B_cols; ++i) {
    B.get()[i] = dist(gen);
  }
  
  float quant_mult = (sizeof(Integer) == 2) ? 1024 : 64;
  float unquant_mult = 1.0/(quant_mult*quant_mult);

  AlignedVector<Integer> A_prep(A_rows * width), B_prep(width * B_cols);
  Routine::PrepareA(A.get(), A_prep.get(), quant_mult, A_rows, width);
  Routine::PrepareB(B.get(), B_prep.get(), quant_mult, width, B_cols);

  AlignedVector<float> test_C(A_rows * B_cols);
  Routine::Multiply(A_prep.get(), B_prep.get(), WriteC(test_C.get(), unquant_mult), A_rows, width, B_cols);

  AlignedVector<Integer> B_quant(width * B_cols);
  Routine::Quantize(B.get(), B_quant.get(), quant_mult, width * B_cols);
  AlignedVector<float> slowint_C(A_rows * B_cols);
  // Assuming A is just quantization here.
  SlowRefInt(A_prep.get(), B_quant.get(), slowint_C.get(), unquant_mult, A_rows, width, B_cols);

  AlignedVector<float> float_C(A_rows * B_cols);
  SlowRefFloat(A.get(), B.get(), float_C.get(), A_rows, width, B_cols);

  Compare(float_C.get(), slowint_C.get(), test_C.get(), A_rows * B_cols, info.str(),
   int_tolerance, float_tolerance, MSE_float_tolerance, MSE_int_tolerance);
}

TEST_CASE ("Multiply SSE2 16bit", "[multiply]") {
  if (kCPU < CPU_SSE2) return;
  TestMultiply<SSE2_16bit, JustUnquantizeC>(8, 256, 256, .1, 1, 0.01);
  TestMultiply<SSE2_16bit, JustUnquantizeC>(8, 2048, 256, .1, 1, 0.02);
  TestMultiply<SSE2_16bit, JustUnquantizeC>(320, 256, 256, .1, 1, 0.01);
  TestMultiply<SSE2_16bit, JustUnquantizeC>(472, 256, 256, .1, 1, 0.01);
  TestMultiply<SSE2_16bit, JustUnquantizeC>(248, 256, 256, .1, 1, 0.01);
  TestMultiply<SSE2_16bit, JustUnquantizeC>(200, 256, 256, .1, 1, 0.01);
}

TEST_CASE ("Multiply SSSE3 8bit", "[multiply]") {
  if (kCPU < CPU_SSSE3) return;
  TestMultiply<SSSE3_8bit, JustUnquantizeC>(8, 256, 256, 1.2, 1.2, 0.064, 0.026);
  TestMultiply<SSSE3_8bit, JustUnquantizeC>(8, 2048, 256, 33, 33, 4.4, 4.4);
  TestMultiply<SSSE3_8bit, JustUnquantizeC>(320, 256, 256, 1.9, 1.9, 0.1, 0.01);
  TestMultiply<SSSE3_8bit, JustUnquantizeC>(472, 256, 256, 2.1, 2.1, 0.1, 0.011);
  TestMultiply<SSSE3_8bit, JustUnquantizeC>(248, 256, 256, 1.7, 1.7, 0.1, 0.012);
  TestMultiply<SSSE3_8bit, JustUnquantizeC>(200, 256, 256, 1.8, 1.9, 0.1, 0.011);
}

TEST_CASE ("Multiply AVX2 8bit", "[multiply]") {
  if (kCPU < CPU_AVX2) return;
  TestMultiply<AVX2_8bit, JustUnquantizeC>(8, 256, 256, .1, 1, 0.1);
  TestMultiply<AVX2_8bit, JustUnquantizeC>(8, 2048, 256, 19, 19, 1.8, 1.8);
  TestMultiply<AVX2_8bit, JustUnquantizeC>(320, 256, 256, .1, 1, 0.1);
  TestMultiply<AVX2_8bit, JustUnquantizeC>(472, 256, 256, .1, 1, 0.1);
  TestMultiply<AVX2_8bit, JustUnquantizeC>(248, 256, 256, .1, 1, 0.1);
  TestMultiply<AVX2_8bit, JustUnquantizeC>(200, 256, 256, .1, 1, 0.1);
}

TEST_CASE ("Multiply AVX2 16bit", "[multiply]") {
  if (kCPU < CPU_AVX2) return;
  TestMultiply<AVX2_16bit, JustUnquantizeC>(8, 256, 256, .1, 1, 0.01);
  TestMultiply<AVX2_16bit, JustUnquantizeC>(8, 2048, 256, .1, 1, 0.02);
  TestMultiply<AVX2_16bit, JustUnquantizeC>(320, 256, 256, .1, 1, 0.01);
  TestMultiply<AVX2_16bit, JustUnquantizeC>(472, 256, 256, .1, 1, 0.01);
  TestMultiply<AVX2_16bit, JustUnquantizeC>(248, 256, 256, .1, 1, 0.01);
  TestMultiply<AVX2_16bit, JustUnquantizeC>(200, 256, 256, .1, 1, 0.01);
}

#ifndef INTGEMM_NO_AVX512
  TEST_CASE ("Multiply AVX512 8bit", "[multiply]") {
    if (kCPU < CPU_AVX512BW) return;
    TestMultiply<AVX512_8bit, JustUnquantizeC>(8, 256, 256, .1, 1, 0.062);
    TestMultiply<AVX512_8bit, JustUnquantizeC>(8, 2048, 256, 4.2, 4, 0.41, 0.37);
    TestMultiply<AVX512_8bit, JustUnquantizeC>(320, 256, 256, .1, 1, 0.06);
    TestMultiply<AVX512_8bit, JustUnquantizeC>(472, 256, 256, .1, 1, 0.06);
    TestMultiply<AVX512_8bit, JustUnquantizeC>(248, 256, 256, .1, 1, 0.06);
    TestMultiply<AVX512_8bit, JustUnquantizeC>(200, 256, 256, .1, 1, 0.06);
  }

  TEST_CASE ("Multiply AVX512 16bit", "[multiply]") {
    if (kCPU < CPU_AVX512BW) return;
    TestMultiply<AVX512_16bit, JustUnquantizeC>(8, 256, 256, .1, 1, 0.01);
    TestMultiply<AVX512_16bit, JustUnquantizeC>(8, 2048, 256, .1, 1, 0.011);
    TestMultiply<AVX512_16bit, JustUnquantizeC>(320, 256, 256, .1, 1, 0.01);
    TestMultiply<AVX512_16bit, JustUnquantizeC>(472, 256, 256, .1, 1, 0.01);
    TestMultiply<AVX512_16bit, JustUnquantizeC>(248, 256, 256, .1, 1, 0.01);
    TestMultiply<AVX512_16bit, JustUnquantizeC>(200, 256, 256, .1, 1, 0.01);
  }
#endif

} // namespace intgemm

int main(int argc, char ** argv) {
  return Catch::Session().run(argc, argv);
}

/*
    // Top matrix sizes from Marian
    TestBoth(8, 256, 256);
    TestBoth(8, 2048, 256);
    TestBoth(8, 2048, 256);
    TestBoth(320, 256, 256);
    TestBoth(472, 256, 256);
    TestBoth(248, 256, 256);
    TestBoth(200, 256, 256);
    return 0;
}
*/
