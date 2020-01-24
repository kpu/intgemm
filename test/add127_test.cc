#include "test.h"

namespace intgemm {

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
    CHECK(fabs(bias_ref[i] - bias[i]) < 0.0001);
  }
}

template <class Routine> void TestPrepareA(Index rows, Index cols) {
  std::mt19937 gen;
  // Go somewhat out of range too.
  std::uniform_real_distribution<float> dist(-2, 2);
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
  std::uniform_real_distribution<float> dist(-30.0, 30.0);
  // Create array.
  AlignedVector<float> inputB(rows * cols);
  for (auto& it : inputB) {
    it = dist(gen);
  }

  float alpha = 25;
  float quant_mult = 127/alpha;

  AlignedVector<int8_t> B_prep(inputB.size());
  AlignedVector<int8_t> B_quant(inputB.size());
  Routine::PrepareB(inputB.begin(), B_prep.begin(), quant_mult, rows, cols);
  Routine::Quantize(inputB.begin(), B_quant.begin(), quant_mult, inputB.size());


  AlignedVector<float> inputBias(cols);
  AlignedVector<float> goldBias(cols);

  for (auto& it : goldBias) {
  	it = dist(gen);
  }
  int i = 0;
  for (auto& it : inputBias) {
    it = goldBias[i];
    i++;
  }

  float unquant_mult_forprep = (-1)*(alpha)*(alpha)/(127.0f);

  Routine::PrepareBias(B_prep.begin(), rows, cols, callbacks::UnquantizeAndAddBiasAndWrite(unquant_mult_forprep, inputBias.begin(), inputBias.begin()));

  int A_rows = 1;
  AlignedVector<int8_t> A_prep2(A_rows*rows);
  for (auto& it : A_prep2) {
    it =1;
  }
  //Routine::Multiply(A_prep2.begin(), B_prep.begin(), A_rows, rows, cols, callbacks::UnquantizeAndAddBiasAndWrite(unquant_mult_forprep, goldBias.begin(), goldBias.begin()));
  //CompareBiases(goldBias.begin(), inputBias.begin(), cols);
  AlignedVector<float> slowint_C(cols);
  SlowRefInt(A_prep2.begin(), B_quant.begin(), slowint_C.begin(), unquant_mult_forprep, A_rows, rows, cols, goldBias.begin());
  CompareBiases(slowint_C.begin(), inputBias.begin(), cols);
}

template <class Routine> void TestMultiplyBiasNew(Index A_rows, Index width, Index B_cols,
 float int_tolerance=.1, float float_tolerance=1, float MSE_float_tolerance=0, float MSE_int_tolerance=0) {
  std::ostringstream info;
  info << Routine::kName << "\t" << A_rows << '\t' << width << '\t' << B_cols << '\n';

  // Initialize A and B.
  AlignedVector<float> A(A_rows * width);
  AlignedVector<float> B(width * B_cols);
  AlignedVector<float> bias(B_cols);
  std::mt19937 gen;
  std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
  for (auto& it : A) {
    it = dist(gen);
  }
  for (auto& it : B) {
    it = dist(gen);
  }
  for (auto& it : bias) {
    it = dist(gen);
  }

  float alpha = 2.0f;
  float quant_mult = 127/alpha;
  float unquant_mult = 1.0/(quant_mult*quant_mult);

  AlignedVector<uint8_t> A_prep(A.size());
  AlignedVector<int8_t> B_prep(B.size());
  Routine::PrepareA(A.begin(), A_prep.begin(), quant_mult, A_rows, width);
  Routine::PrepareB(B.begin(), B_prep.begin(), quant_mult, width, B_cols);

  AlignedVector<float> test_C(A_rows * B_cols);

  /*REFERENCE MULTIPLICATION
  *
  *
  */
  AlignedVector<int8_t> B_quant(B.size());
  Routine::Quantize(B.begin(), B_quant.begin(), quant_mult, B.size());
  AlignedVector<float> slowint_C(test_C.size());
  // Taking the original A_preparation which means A would be int8_t
  AlignedVector<int8_t> A_prep2(A.size());
  Routine::PrepareA(A.begin(), A_prep2.begin(), quant_mult, A_rows, width);
  SlowRefInt(A_prep2.begin(), B_quant.begin(), slowint_C.begin(), unquant_mult, A_rows, width, B_cols, bias.begin());

  AlignedVector<float> float_C(test_C.size());
  SlowRefFloat(A.begin(), B.begin(), float_C.begin(), A_rows, width, B_cols, bias.begin());

  /*ACTUAL MULTIPLICATION
  *
  */
  float unquant_mult_forprep = (-1)*(alpha)*(alpha)/(127.0f); //Minus one to invert add_ps later on
  Routine::PrepareBias(B_prep.begin(), width, B_cols, callbacks::UnquantizeAndAddBiasAndWrite(unquant_mult_forprep, bias.begin(), bias.begin()));
  //Routine::PrepareBias(B.begin(), bias.begin(), alpha, width, B_cols);
  Routine::Multiply8Shift(A_prep.begin(), B_prep.begin(), A_rows, width, B_cols, callbacks::UnquantizeAndAddBiasAndWrite(unquant_mult, bias.begin(), test_C.begin()));

  Compare(float_C.begin(), slowint_C.begin(), test_C.begin(), test_C.size(), info.str(),
   int_tolerance, float_tolerance, MSE_float_tolerance, MSE_int_tolerance);
}

template <class Routine> void TestMultiplyShiftNonShift(Index A_rows, Index width, Index B_cols,
 float int_tolerance=.1, float float_tolerance=1, float MSE_float_tolerance=0, float MSE_int_tolerance=0) {
  std::ostringstream info;
  info << Routine::kName << "\t" << A_rows << '\t' << width << '\t' << B_cols << '\n';

  // Initialize A and B.
  AlignedVector<float> A(A_rows * width);
  AlignedVector<float> B(width * B_cols);
  AlignedVector<float> bias(B_cols);
  std::mt19937 gen;
  std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
  for (auto& it : A) {
    it = dist(gen);
  }
  for (auto& it : B) {
    it = dist(gen);
  }
  for (auto& it : bias) {
    it = 0;
  }

  float alpha = 2.0f;
  float quant_mult = 127/alpha;
  float unquant_mult = 1.0/(quant_mult*quant_mult);

  AlignedVector<uint8_t> A_prep(A.size());
  AlignedVector<int8_t> A_prep_old(A.size());
  AlignedVector<int8_t> B_prep(B.size());
  Routine::PrepareA(A.begin(), A_prep.begin(), quant_mult, A_rows, width);
  Routine::PrepareA(A.begin(), A_prep_old.begin(), quant_mult, A_rows, width); //Non shited version
  Routine::PrepareB(B.begin(), B_prep.begin(), quant_mult, width, B_cols);

  AlignedVector<float> test_C(A_rows * B_cols);

  /*
   * Reference non shift multiplication instead of slowint
   */
  AlignedVector<float> slowint_C(test_C.size());
  Routine::Multiply(A_prep_old.begin(), B_prep.begin(), A_rows, width, B_cols, callbacks::UnquantizeAndAddBiasAndWrite(unquant_mult, bias.begin(), slowint_C.begin()));

  AlignedVector<float> float_C(test_C.size());
  SlowRefFloat(A.begin(), B.begin(), float_C.begin(), A_rows, width, B_cols, bias.begin());
  /*
   * Multiply8 shift multiplication
   */
  float unquant_mult_forprep = (-1)*(alpha)*(alpha)/(127.0f); //Minus one to invert add_ps later on
  Routine::PrepareBias(B_prep.begin(), width, B_cols, callbacks::UnquantizeAndAddBiasAndWrite(unquant_mult_forprep, bias.begin(), bias.begin()));
  Routine::Multiply8Shift(A_prep.begin(), B_prep.begin(), A_rows, width, B_cols, callbacks::UnquantizeAndAddBiasAndWrite(unquant_mult, bias.begin(), test_C.begin()));

  Compare(float_C.begin(), slowint_C.begin(), test_C.begin(), test_C.size(), info.str(),
   int_tolerance, float_tolerance, MSE_float_tolerance, MSE_int_tolerance);
}

template <class Routine> void TestMultiplyShiftInt(Index A_rows, Index width, Index B_cols,
 float int_tolerance=.1, float float_tolerance=1, float MSE_float_tolerance=0, float MSE_int_tolerance=0) {
  std::ostringstream info;
  info << Routine::kName << "\t" << A_rows << '\t' << width << '\t' << B_cols << '\n';

  // Initialize A and B.
  AlignedVector<float> A(A_rows * width);
  AlignedVector<float> B(width * B_cols);
  AlignedVector<float> bias(B_cols);
  std::mt19937 gen;
  std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
  for (auto& it : A) {
    it = dist(gen);
  }
  for (auto& it : B) {
    it = dist(gen);
  }
  for (auto& it : bias) {
    it = 0;
  }

  float alpha = 2.0f;
  float quant_mult = 127/alpha;
  float unquant_mult = 1.0/(quant_mult*quant_mult);

  AlignedVector<uint8_t> A_prep(A.size());
  AlignedVector<int8_t> A_prep_old(A.size());
  AlignedVector<int8_t> B_prep(B.size());
  Routine::PrepareA(A.begin(), A_prep.begin(), quant_mult, A_rows, width);
  Routine::PrepareA(A.begin(), A_prep_old.begin(), quant_mult, A_rows, width); //Non shited version
  Routine::PrepareB(B.begin(), B_prep.begin(), quant_mult, width, B_cols);

  AlignedVector<float> test_C(A_rows * B_cols);

  /*
   * Reference float multiplication
   */
  AlignedVector<int8_t> B_quant(B.size());
  Routine::Quantize(B.begin(), B_quant.begin(), quant_mult, B.size());
  AlignedVector<float> slowint_C(test_C.size());
  // Taking the original A_preparation which means A would be int8_t
  //SlowRefInt(A_prep.begin(), B_quant.begin(), slowint_C.begin(), unquant_mult, A_rows, width, B_cols, bias.begin());

  AlignedVector<float> float_C(test_C.size());
  SlowRefFloat(A.begin(), B.begin(), float_C.begin(), A_rows, width, B_cols, bias.begin());
  /*
   * Multiply8 shift multiplication
   */
  //First prepare SlowInteger Bias:
  AlignedVector<int8_t> A_prep2(1*width);
  for (auto& it : A_prep2) {
    it = 1;
  }
  AlignedVector<float> ShiftedBias(B_cols);
  float unquant_mult_forprep = (-1)*(alpha)*(alpha)/(127.0f); //Minus one to invert add_ps later on
  SlowRefInt(A_prep2.begin(), B_quant.begin(), ShiftedBias.begin(), unquant_mult_forprep, 1, width, B_cols, bias.begin());
  

  //Now prepare Fast integer Bias
  Routine::PrepareBias(B_prep.begin(), width, B_cols, callbacks::UnquantizeAndAddBiasAndWrite(unquant_mult_forprep, bias.begin(), bias.begin()));
  Routine::Multiply8Shift(A_prep.begin(), B_prep.begin(), A_rows, width, B_cols, callbacks::UnquantizeAndAddBiasAndWrite(unquant_mult, bias.begin(), test_C.begin()));

  // Reference INT VERSION HERE with ADD127
  // Taking the original A_preparation which means A would be int8_t
  SlowRefInt(A_prep.begin(), B_quant.begin(), slowint_C.begin(), unquant_mult, A_rows, width, B_cols, ShiftedBias.begin());

  Compare(float_C.begin(), slowint_C.begin(), test_C.begin(), test_C.size(), info.str(),
   int_tolerance, float_tolerance, MSE_float_tolerance, MSE_int_tolerance);
}


// Bias
TEST_CASE("PrepareBias SSSE3", "[Add127]") {
	if (kCPU < CPUType::SSSE3) return;
	TestPrepareBias<SSSE3_8bit>(256,256);
	TestPrepareBias<SSSE3_8bit>(2048,256);
	TestPrepareBias<SSSE3_8bit>(512,512);
}

TEST_CASE("PrepareBias AVX2", "[Add127]") {
	if (kCPU < CPUType::AVX2) return;
	TestPrepareBias<AVX2_8bit>(256,256);
	TestPrepareBias<AVX2_8bit>(2048,256);
	TestPrepareBias<AVX2_8bit>(512,512);
}

TEST_CASE("PrepareBias AVX512F", "[Add127]") {
	if (kCPU < CPUType::AVX512BW) return;
	#ifdef INTGEMM_COMPILER_SUPPORTS_AVX512
	TestPrepareBias<AVX512_8bit>(256,256);
	TestPrepareBias<AVX512_8bit>(2048,256);
	TestPrepareBias<AVX512_8bit>(512,512);
	#endif
}

//A
TEST_CASE("PrepareA SSSE3", "[Add127]") {
	if (kCPU < CPUType::SSSE3) return;
	TestPrepareA<SSSE3_8bit>(64,64);
	TestPrepareA<SSSE3_8bit>(256,256);
	TestPrepareA<SSSE3_8bit>(512,512);
  TestPrepareA<SSSE3_8bit>(2048,256);
}

TEST_CASE("PrepareA AVX2", "[Add127]") {
	if (kCPU < CPUType::AVX2) return;
	TestPrepareA<AVX2_8bit>(64,64);
	TestPrepareA<AVX2_8bit>(256,256);
	TestPrepareA<AVX2_8bit>(512,512);
  TestPrepareA<AVX2_8bit>(2048,256);
}

TEST_CASE("PrepareA AVX512F", "[Add127]") {
	if (kCPU < CPUType::AVX512BW) return;
	#ifdef INTGEMM_COMPILER_SUPPORTS_AVX512
	TestPrepareA<AVX512_8bit>(64,64);
	TestPrepareA<AVX512_8bit>(256,256);
	TestPrepareA<AVX512_8bit>(512,512);
  TestPrepareA<AVX512_8bit>(2048,256);
	#endif
}

// Multiply

TEST_CASE ("Multiply SSSE3 8bit Shift with bias", "[Add127]") {
  if (kCPU < CPUType::SSSE3) return;
  TestMultiplyBiasNew<SSSE3_8bit>(1, 64, 8, 0.11, 0.1, 0.06, 0.05);
  TestMultiplyBiasNew<SSSE3_8bit>(8, 256, 256, 0.45, 0.54, 0.17, 0.16);
  TestMultiplyBiasNew<SSSE3_8bit>(8, 2048, 256, 1.7, 1.7, 0.46, 0.43);
  TestMultiplyBiasNew<SSSE3_8bit>(320, 256, 256, 0.56, 0.64, 0.16, 0.15);
  TestMultiplyBiasNew<SSSE3_8bit>(472, 256, 256, 0.46, 0.62, 0.17, 0.16);
  TestMultiplyBiasNew<SSSE3_8bit>(248, 256, 256, 0.48, 0.64, 0.16, 0.15);
  TestMultiplyBiasNew<SSSE3_8bit>(200, 256, 256, 0.55, 0.74, 0.17, 0.16);
}

TEST_CASE ("Multiply AVX2 8bit Shift with bias", "[Add127]") {
  if (kCPU < CPUType::AVX2) return;
  TestMultiplyBiasNew<AVX2_8bit>(1, 64, 8, 0.11, 0.11, 0.06, 0.05);
  TestMultiplyBiasNew<AVX2_8bit>(8, 256, 256, 0.49, 0.54, 0.17, 0.16);
  TestMultiplyBiasNew<AVX2_8bit>(8, 2048, 256, 1.57, 1.66, 0.46, 0.46);
  TestMultiplyBiasNew<AVX2_8bit>(320, 256, 256, 0.49, 0.64, 0.16, 0.15);
  TestMultiplyBiasNew<AVX2_8bit>(472, 256, 256, 0.46, 0.62, 0.17, 0.16);
  TestMultiplyBiasNew<AVX2_8bit>(248, 256, 256, 0.48, 0.64, 0.16, 0.15);
  TestMultiplyBiasNew<AVX2_8bit>(200, 256, 256, 0.55, 0.74, 0.17, 0.16);
}
#ifdef INTGEMM_COMPILER_SUPPORTS_AVX512
TEST_CASE ("Multiply AVX512F 8bit Shift with bias", "[Add127]") {
  if (kCPU < CPUType::AVX512BW) return;
  TestMultiplyBiasNew<AVX512_8bit>(1, 64, 8, 0.0001, 0.05, 0.03, 0.001);
  TestMultiplyBiasNew<AVX512_8bit>(8, 256, 256, 0.0001, 0.22, 0.06, 0.001);
  TestMultiplyBiasNew<AVX512_8bit>(8, 2048, 256, 0.0001, 0.61, 0.17, 0.001);
  TestMultiplyBiasNew<AVX512_8bit>(320, 256, 256, 0.0001, 0.27, 0.06, 0.001);
  TestMultiplyBiasNew<AVX512_8bit>(472, 256, 256, 0.0001, 0.33, 0.06, 0.001);
  TestMultiplyBiasNew<AVX512_8bit>(248, 256, 256, 0.0001, 0.27, 0.06, 0.001);
  TestMultiplyBiasNew<AVX512_8bit>(200, 256, 256, 0.0001, 0.28, 0.06, 0.001);
}
#endif

#ifdef INTGEMM_COMPILER_SUPPORTS_AVX512VNNI
  TEST_CASE ("Multiply AVX512VNNI 8bit Shift with bias", "[Add127]") {
    if (kCPU < CPUType::AVX512VNNI) return;
    TestMultiplyBiasNew<AVX512VNNI_8bit>(1, 64, 8, 0.0001, 0.05, 0.03, 0.001);
    TestMultiplyBiasNew<AVX512VNNI_8bit>(8, 256, 256, 0.0001, 0.22, 0.06, 0.001);
    TestMultiplyBiasNew<AVX512VNNI_8bit>(8, 2048, 256, 0.0001, 0.61, 0.17, 0.001);
    TestMultiplyBiasNew<AVX512VNNI_8bit>(320, 256, 256, 0.0001, 0.27, 0.06, 0.001);
    TestMultiplyBiasNew<AVX512VNNI_8bit>(472, 256, 256, 0.0001, 0.33, 0.06, 0.001);
    TestMultiplyBiasNew<AVX512VNNI_8bit>(248, 256, 256, 0.0001, 0.27, 0.06, 0.001);
    TestMultiplyBiasNew<AVX512VNNI_8bit>(200, 256, 256, 0.0001, 0.28, 0.06, 0.001);
  }
#endif

//Multiply old vs new
TEST_CASE ("Multiply SSSE3 8bit Shift vs nonshift", "[Add127]") {
  if (kCPU < CPUType::SSSE3) return;
  TestMultiplyShiftNonShift<SSSE3_8bit>(1, 64, 8, 0.00001, 0.1, 0.06, 0.00001);
  TestMultiplyShiftNonShift<SSSE3_8bit>(8, 256, 256, 0.00001, 0.54, 0.17, 0.00001);
  TestMultiplyShiftNonShift<SSSE3_8bit>(8, 2048, 256, 17.9, 1.7, 0.46, 4.2); //Big difference here because the non-shift version is very bad
  TestMultiplyShiftNonShift<SSSE3_8bit>(320, 256, 256, 1.2, 0.64, 0.16, 0.006);
  TestMultiplyShiftNonShift<SSSE3_8bit>(472, 256, 256, 1.1, 0.62, 0.17, 0.006);
  TestMultiplyShiftNonShift<SSSE3_8bit>(248, 256, 256, 0.9, 0.64, 0.16, 0.007);
  TestMultiplyShiftNonShift<SSSE3_8bit>(200, 256, 256, 1, 0.74, 0.17, 0.006);
}

TEST_CASE ("Multiply AVX2 8bit Shift vs nonshift", "[Add127]") {
  if (kCPU < CPUType::AVX2) return;
  TestMultiplyShiftNonShift<AVX2_8bit>(1, 64, 8, 0.00001, 0.11, 0.06, 0.00001);
  TestMultiplyShiftNonShift<AVX2_8bit>(8, 256, 256, 0.00001, 0.54, 0.17, 0.00001);
  TestMultiplyShiftNonShift<AVX2_8bit>(8, 2048, 256, 9.4, 1.66, 0.46, 1.67); //Big difference here because the non-shift version is very bad
  TestMultiplyShiftNonShift<AVX2_8bit>(320, 256, 256, 0.0001, 0.64, 0.16, 0.0001);
  TestMultiplyShiftNonShift<AVX2_8bit>(472, 256, 256, 0.0001, 0.62, 0.17, 0.0001);
  TestMultiplyShiftNonShift<AVX2_8bit>(248, 256, 256, 0.0001, 0.64, 0.16, 0.0001);
  TestMultiplyShiftNonShift<AVX2_8bit>(200, 256, 256, 0.0001, 0.74, 0.17, 0.0001);
}
#ifdef INTGEMM_COMPILER_SUPPORTS_AVX512
TEST_CASE ("Multiply AVX512F 8bit Shift vs nonshift", "[Add127]") {
  if (kCPU < CPUType::AVX512BW) return;
  TestMultiplyShiftNonShift<AVX512_8bit>(1, 64, 8, 0.0001, 0.05, 0.03, 0.001);
  TestMultiplyShiftNonShift<AVX512_8bit>(8, 256, 256, 0.0001, 0.22, 0.06, 0.001);
  TestMultiplyShiftNonShift<AVX512_8bit>(8, 2048, 256, 3.51, 0.61, 0.17, 0.3);
  TestMultiplyShiftNonShift<AVX512_8bit>(320, 256, 256, 0.0001, 0.27, 0.06, 0.001);
  TestMultiplyShiftNonShift<AVX512_8bit>(472, 256, 256, 0.0001, 0.33, 0.06, 0.001);
  TestMultiplyShiftNonShift<AVX512_8bit>(248, 256, 256, 0.0001, 0.27, 0.06, 0.001);
  TestMultiplyShiftNonShift<AVX512_8bit>(200, 256, 256, 0.0001, 0.28, 0.06, 0.001);
}
#endif

#ifdef INTGEMM_COMPILER_SUPPORTS_AVX512VNNI
  TEST_CASE ("Multiply AVX512VNNI 8bit Shift vs nonshift", "[Add127]") {
    if (kCPU < CPUType::AVX512VNNI) return;
    TestMultiplyShiftNonShift<AVX512VNNI_8bit>(1, 64, 8, 0.00001, 0.05, 0.03, 0.00001);
    TestMultiplyShiftNonShift<AVX512VNNI_8bit>(8, 256, 256, 0.00001, 0.22, 0.06, 0.00001);
    TestMultiplyShiftNonShift<AVX512VNNI_8bit>(8, 2048, 256, 0.0001, 0.61, 0.17, 0.0001);
    TestMultiplyShiftNonShift<AVX512VNNI_8bit>(320, 256, 256, 0.00001, 0.27, 0.06, 0.00001);
    TestMultiplyShiftNonShift<AVX512VNNI_8bit>(472, 256, 256, 0.00001, 0.33, 0.06, 0.00001);
    TestMultiplyShiftNonShift<AVX512VNNI_8bit>(248, 256, 256, 0.00001, 0.27, 0.06, 0.00001);
    TestMultiplyShiftNonShift<AVX512VNNI_8bit>(200, 256, 256, 0.00001, 0.28, 0.06, 0.00001);
  }
#endif

//Multiply Shift vs int shift implementation
TEST_CASE ("Multiply SSSE3 8bit Shift vs Int", "[Add127]") {
  if (kCPU < CPUType::SSSE3) return;
  TestMultiplyShiftInt<SSSE3_8bit>(1, 64, 8, 0, 0.1, 0.06, 0);
  TestMultiplyShiftInt<SSSE3_8bit>(8, 256, 256, 0, 0.54, 0.17, 0);
  TestMultiplyShiftInt<SSSE3_8bit>(8, 2048, 256, 0, 1.7, 0.46, 0);
  TestMultiplyShiftInt<SSSE3_8bit>(320, 256, 256, 0, 0.64, 0.16, 0);
  TestMultiplyShiftInt<SSSE3_8bit>(472, 256, 256, 0, 0.62, 0.17, 0);
  TestMultiplyShiftInt<SSSE3_8bit>(248, 256, 256, 0, 0.64, 0.16, 0);
  TestMultiplyShiftInt<SSSE3_8bit>(200, 256, 256, 0, 0.74, 0.17, 0);
}

TEST_CASE ("Multiply AVX2 8bit Shift vs Int", "[Add127]") {
  if (kCPU < CPUType::AVX2) return;
  TestMultiplyShiftInt<AVX2_8bit>(1, 64, 8, 0, 0.11, 0.06, 0);
  TestMultiplyShiftInt<AVX2_8bit>(8, 256, 256, 0, 0.54, 0.17, 0);
  TestMultiplyShiftInt<AVX2_8bit>(8, 2048, 256, 0, 1.66, 0.46, 0);
  TestMultiplyShiftInt<AVX2_8bit>(320, 256, 256, 0, 0.64, 0.16, 0);
  TestMultiplyShiftInt<AVX2_8bit>(472, 256, 256, 0, 0.62, 0.17, 0);
  TestMultiplyShiftInt<AVX2_8bit>(248, 256, 256, 0, 0.64, 0.16, 0);
  TestMultiplyShiftInt<AVX2_8bit>(200, 256, 256, 0, 0.74, 0.17, 0);
}
#ifdef INTGEMM_COMPILER_SUPPORTS_AVX512
TEST_CASE ("Multiply AVX512F 8bit Shift vs Int", "[Add127]") {
  if (kCPU < CPUType::AVX512BW) return;
  TestMultiplyShiftInt<AVX512_8bit>(1, 64, 8, 0, 0.05, 0.03, 0);
  TestMultiplyShiftInt<AVX512_8bit>(8, 256, 256, 0, 0.22, 0.06, 0);
  TestMultiplyShiftInt<AVX512_8bit>(8, 2048, 256, 0, 0.61, 0.17, 0);
  TestMultiplyShiftInt<AVX512_8bit>(320, 256, 256, 0, 0.27, 0.06, 0);
  TestMultiplyShiftInt<AVX512_8bit>(472, 256, 256, 0, 0.33, 0.06, 0);
  TestMultiplyShiftInt<AVX512_8bit>(248, 256, 256, 0, 0.27, 0.06, 0);
  TestMultiplyShiftInt<AVX512_8bit>(200, 256, 256, 0, 0.28, 0.06, 0);
}
#endif

#ifdef INTGEMM_COMPILER_SUPPORTS_AVX512VNNI
  TEST_CASE ("Multiply AVX512VNNI 8bit Shift vs Int", "[Add127]") {
    if (kCPU < CPUType::AVX512VNNI) return;
    TestMultiplyShiftInt<AVX512VNNI_8bit>(1, 64, 8, 0, 0.05, 0.03, 0);
    TestMultiplyShiftInt<AVX512VNNI_8bit>(8, 256, 256, 0, 0.22, 0.06, 0);
    TestMultiplyShiftInt<AVX512VNNI_8bit>(8, 2048, 256, 0, 0.61, 0.17, 0);
    TestMultiplyShiftInt<AVX512VNNI_8bit>(320, 256, 256, 0, 0.27, 0.06, 0);
    TestMultiplyShiftInt<AVX512VNNI_8bit>(472, 256, 256, 0, 0.33, 0.06, 0);
    TestMultiplyShiftInt<AVX512VNNI_8bit>(248, 256, 256, 0, 0.27, 0.06, 0);
    TestMultiplyShiftInt<AVX512VNNI_8bit>(200, 256, 256, 0, 0.28, 0.06, 0);
  }
#endif

} //namespace intgemm
