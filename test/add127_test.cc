#include "3rd_party/catch.hpp"
#include "intgemm.h"
#include "aligned.h"

namespace intgemm {

void SlowSumB(const float * input, float * bias, float* output, float alpha, Index rows, Index cols) {
	for (Index r = 0; r<rows; r++) {
		for (Index c = 0; c<cols; c++) {
			output[c] += input[r * cols + c];
		}
	}

	for (Index c = 0; c<cols; c++) {
		output[c] = bias[c] - output[c]*alpha;
	}
}

void CompareAs(int8_t * output_old, uint8_t * output_new, Index rows, Index cols) {
	for (Index r = 0; r<rows; r++) {
		for (Index c = 0; c<cols; c++) {
			int a = int(output_old[rows*c + r]);
			int b = int(output_new[rows*c + r]);
			INFO("Inaccurate at row: " << r << " column " << c << ' '
			 << a << ' ' << b);
			CHECK(a+127 == b);
		}
	}
}

void CompareBiases(const float *bias_ref, const float *bias, Index cols) {
  for (std::size_t i = 0; i < cols; ++i) {
  	INFO("Inaccurate at " << i << ' ' << bias_ref[i] << ' ' << bias[i]);
    CHECK(fabs(bias_ref[i] - bias[i]) < 0.1);
  }
}

template <class Routine> void TestPrepareA(Index rows, Index cols) {
  std::mt19937 gen;
  // Go somewhat out of range too.
  std::uniform_real_distribution<float> dist(-1000.0, 1000.0);
  // Create array.
  AlignedVector<float> inputA(rows * cols);
  for (auto& it : inputA) {
    it = dist(gen);
  }
  AlignedVector<int8_t> oldA(rows * cols);
  AlignedVector<uint8_t> newA(rows * cols);
  float quant_mult = 64; //From example
  Routine::PrepareA(inputA.begin(), oldA.begin(), quant_mult, rows, cols);
  Routine::PrepareA(inputA.begin(), newA.begin(), quant_mult, rows, cols);
  CompareAs(oldA.begin(), newA.begin(), rows, cols);
}

template <class Routine> void TestPrepareBias(Index rows, Index cols) {
  std::mt19937 gen;
  // Go somewhat out of range too.
  std::uniform_real_distribution<float> dist(-1000.0, 1000.0);
  // Create array.
  AlignedVector<float> inputB(rows * cols);
  for (auto& it : inputB) {
    it = dist(gen);
  }

  AlignedVector<float> inputBias(cols);
  AlignedVector<float> goldBias(cols);
  //goldBias.begin()[0] = 25;
  for (auto& it : goldBias) {
  	it = 0;
  }
  for (auto& it : inputBias) {
    it = dist(gen);
  }
  float alpha = dist(gen);
  SlowSumB(inputB.begin(), inputBias.begin(), goldBias.begin(), alpha, rows, cols);

  Routine::PrepareBiasFor8(inputB.begin(), inputBias.begin(), alpha, rows, cols);

  CompareBiases(goldBias.begin(), inputBias.begin(), cols);
}

// Bias
TEST_CASE("PrepareBias SSSE3", "[Add127]") {
	if (kCPU < CPU_SSSE3) return;
	TestPrepareBias<SSSE3_8bit>(8,8);
	TestPrepareBias<SSSE3_8bit>(256,256);
	TestPrepareBias<SSSE3_8bit>(2048,256);
	TestPrepareBias<SSSE3_8bit>(512,512);
}

TEST_CASE("PrepareBias AVX2", "[Add127]") {
	if (kCPU < CPU_AVX2) return;
	TestPrepareBias<AVX2_8bit>(8,8);
	TestPrepareBias<AVX2_8bit>(256,256);
	TestPrepareBias<AVX2_8bit>(2048,256);
	TestPrepareBias<AVX2_8bit>(512,512);
}

TEST_CASE("PrepareBias AVX512F", "[Add127]") {
	if (kCPU < CPU_AVX512BW) return;
	#ifndef INTGEMM_NO_AVX512
	TestPrepareBias<AVX512_8bit>(8,8);
	TestPrepareBias<AVX512_8bit>(256,256);
	TestPrepareBias<AVX512_8bit>(2048,256);
	TestPrepareBias<AVX512_8bit>(512,512);
	#endif
}

//A
TEST_CASE("PrepareA SSSE3", "[Add127]") {
	if (kCPU < CPU_SSSE3) return;
	TestPrepareA<SSSE3_8bit>(8,8);
	TestPrepareA<SSSE3_8bit>(256,256);
	TestPrepareA<SSSE3_8bit>(2048,256);
	TestPrepareA<SSSE3_8bit>(512,512);
}

TEST_CASE("PrepareA AVX2", "[Add127]") {
	if (kCPU < CPU_AVX2) return;
	TestPrepareA<AVX2_8bit>(8,8);
	TestPrepareA<AVX2_8bit>(256,256);
	TestPrepareA<AVX2_8bit>(2048,256);
	TestPrepareA<AVX2_8bit>(512,512);
}

TEST_CASE("PrepareA AVX512F", "[Add127]") {
	if (kCPU < CPU_AVX512BW) return;
	#ifndef INTGEMM_NO_AVX512
	TestPrepareA<AVX512_8bit>(8,8);
	TestPrepareA<AVX512_8bit>(256,256);
	TestPrepareA<AVX512_8bit>(2048,256);
	TestPrepareA<AVX512_8bit>(512,512);
	#endif
}

} //namespace intgemm
