#if defined(INTGEMM_THIS_IS_AVX512VNNI)
#define INTGEMM_ARCH AVX512VNNI
#define INTGEMM_TARGET INTGEMM_AVX512VNNI
#define INTGEMM_TEST_NAME "AVX512VNNI"
#elif defined(INTGEMM_THIS_IS_AVX512BW)
#define INTGEMM_ARCH AVX512BW
#define INTGEMM_TARGET INTGEMM_AVX512BW
#define INTGEMM_TEST_NAME "AVX512BW"
#elif defined(INTGEMM_THIS_IS_AVX2)
#define INTGEMM_ARCH AVX2
#define INTGEMM_TARGET INTGEMM_AVX2
#define INTGEMM_TEST_NAME "AVX2"
#elif defined(INTGEMM_THIS_IS_SSSE3)
#define INTGEMM_ARCH SSSE3
#define INTGEMM_TARGET INTGEMM_SSSE3
#define INTGEMM_TEST_NAME "SSSE3"
#elif defined(INTGEMM_THIS_IS_SSE2)
#define INTGEMM_ARCH SSE2
#define INTGEMM_TARGET INTGEMM_SSE2
#define INTGEMM_TEST_NAME "SSE2"
#else
#error "Included without expected architecture"
#endif

namespace intgemm {
namespace INTGEMM_ARCH {

// There isn't a Shifted8 for SSE2.
#ifndef INTGEMM_THIS_IS_SSE2
INTGEMM_TARGET void OneIteration() {
  AlignedVector<int8_t> A(1 * sizeof(Register));
  AlignedVector<int8_t> B(sizeof(Register) * 1);
  AlignedVector<int32_t> C(sizeof(Register) / sizeof(int32_t)/* Raw sums */);

  memset(C.begin(), 0, sizeof(Register));

  std::iota(A.begin(), A.end(), 7 /* made up */);
  std::iota(B.begin(), B.end(), 1 /* made up */);

  typedef RowMajorAccess<int8_t> InputA;
  typedef ColMajorAccess<int8_t> InputB;
  typedef RegisterRowMajorAccess Output;
  Access<InputA, InputB, Output> access(
      InputA(A.begin(), sizeof(Register)),
      InputB(B.begin(), sizeof(Register)),
      Output(reinterpret_cast<Register*>(C.begin()), 1));
  UnrollKernel<1, 1, 1, Shifted8>::Run(access);

  const std::size_t kStride = sizeof(int32_t) / sizeof(int8_t);
  for (std::size_t i = 0; i < sizeof(Register) / sizeof(int32_t); ++i) {
    int32_t sum = 0;
    for (std::size_t j = i * kStride; j < (i+1) * kStride; ++j) {
      sum += static_cast<int32_t>(A[j]) * static_cast<int32_t>(B[j]);
    }
    CHECK(C[i] == sum);
  }
}

TEST_CASE("Basic Tile " INTGEMM_TEST_NAME, "[tile]") {
  if (kCPU >= CPUType::INTGEMM_ARCH) {
    OneIteration();
  }
}
#endif

template <Index Valid> INTGEMM_TARGET static void Reduce32Test() {
  // A zero-length array is a compiler error, so force it to be longer.
  constexpr Index ArrayLen = Valid ? Valid : 1;
  const std::size_t kReduce = sizeof(Register) / sizeof(int32_t);
  Register regs[ArrayLen];
  std::mt19937 gen;
  std::uniform_int_distribution<int32_t> dist(std::numeric_limits<int32_t>::min(), std::numeric_limits<int32_t>::max());
  int32_t reference[ArrayLen];
  // Do 20 different loops of random numbers.
  for (Index attempt = 0; attempt < 20; ++attempt) {
    memset(reference, 0, sizeof(reference));
    for (Index i = 0; i < Valid; ++i) {
      int32_t temp[kReduce];
      for (std::size_t j = 0; j < kReduce; ++j) {
        temp[j] = dist(gen);
        reference[i] += temp[j];
      }
      memcpy(&regs[i], temp, sizeof(Register));
    }
    // Decay type for template.
    Register *indirect = regs;
    Reduce32<Valid, Sum32Op>(indirect);
    const int32_t *test = reinterpret_cast<const int32_t*>(regs);
    for (Index i = 0; i < Valid; ++i) {
      CHECK(test[i] == reference[i]);
    }
  }
}

template <std::size_t... i> void Reduce32TestLoop(index_sequence<i...>) {
  unordered_unfurl((Reduce32Test<i>(), 0)...);
}

TEST_CASE("Reduce " INTGEMM_TEST_NAME, "[tile]") {
  if (kCPU < CPUType::INTGEMM_ARCH) return;
  Reduce32TestLoop(make_index_sequence<33>());
}

// Replicate the saturation behavior of the Signed8 kernel with 16-bit accumulation.
template <class Access> void Signed8ReferenceMult(Access access, Tile problem) {
  assert(!problem.inner % 2);
  for (Index a_row = 0; a_row < problem.A_rows; ++a_row) {
    for (Index b_col = 0; b_col < problem.B_cols; ++b_col) {
      Access acc = access.AAdd(a_row, 0).BAdd(0, b_col).CAdd(a_row, b_col);
      // For VNNI, just do it accurately.
#ifdef INTGEMM_THIS_IS_AVX512VNNI
      acc.CFront() = 0;
      for (Index inner = 0; inner < problem.inner; ++inner) {
        Access innermost = acc.AAdd(0, inner).BAdd(inner, 0);
        acc.CFront() += static_cast<int32_t>(innermost.AFront()) * static_cast<int32_t>(innermost.BFront());
      }
#else
      // For non-VNNI, do the saturation stuff.
      int16_t accumulators[sizeof(Register) / sizeof(int16_t)] = {0};
      for (Index inner = 0; inner < problem.inner; inner += 2) {
        Access innermost = acc.AAdd(0, inner).BAdd(inner, 0);
        int32_t product = static_cast<int32_t>(innermost.AFront()) * static_cast<int32_t>(innermost.BFront());
        innermost = innermost.AAdd(0, 1).BAdd(1, 0);
        product += static_cast<int32_t>(innermost.AFront()) * static_cast<int32_t>(innermost.BFront());
        // Saturate to 16-bit for maddubs.
        if (product > 32767) product = 32767;
        if (product < -32768) product = -32768;
        int16_t &accum = accumulators[(inner / 2) % (sizeof(Register) / sizeof(int16_t))];
        // Saturating accumlation.
        product += static_cast<int32_t>(accum);
        if (product > 32767) product = 32767;
        if (product < -32768) product = -32768;
        accum = static_cast<int16_t>(product);
      }
      acc.CFront() = 0;
      for (Index i = 0; i < sizeof(Register) / sizeof(int16_t); ++i) {
        acc.CFront() += static_cast<int32_t>(accumulators[i]);
      }
#endif
    }
  }
}

template <class Access> void Signed8ReferenceMult_UnquantizeAndWrite(Access access, Tile problem, float unquant_mult) {
  assert(!problem.inner % 2);
  for (Index a_row = 0; a_row < problem.A_rows; ++a_row) {
    for (Index b_col = 0; b_col < problem.B_cols; ++b_col) {
      Access acc = access.AAdd(a_row, 0).BAdd(0, b_col).CAdd(a_row, b_col);
      // For VNNI, just do it accurately.
#ifdef INTGEMM_THIS_IS_AVX512VNNI
      acc.CFront() = 0;
      for (Index inner = 0; inner < problem.inner; ++inner) {
        Access innermost = acc.AAdd(0, inner).BAdd(inner, 0);
        acc.CFront() += static_cast<int32_t>(innermost.AFront()) * static_cast<int32_t>(innermost.BFront());
      }
#else
      // For non-VNNI, do the saturation stuff.
      int16_t accumulators[sizeof(Register) / sizeof(int16_t)] = {0};
      for (Index inner = 0; inner < problem.inner; inner += 2) {
        Access innermost = acc.AAdd(0, inner).BAdd(inner, 0);
        int32_t product = static_cast<int32_t>(innermost.AFront()) * static_cast<int32_t>(innermost.BFront());
        innermost = innermost.AAdd(0, 1).BAdd(1, 0);
        product += static_cast<int32_t>(innermost.AFront()) * static_cast<int32_t>(innermost.BFront());
        // Saturate to 16-bit for maddubs.
        if (product > 32767) product = 32767;
        if (product < -32768) product = -32768;
        int16_t &accum = accumulators[(inner / 2) % (sizeof(Register) / sizeof(int16_t))];
        // Saturating accumlation.
        product += static_cast<int32_t>(accum);
        if (product > 32767) product = 32767;
        if (product < -32768) product = -32768;
        accum = static_cast<int16_t>(product);
      }
      acc.CFront() = 0;
      for (Index i = 0; i < sizeof(Register) / sizeof(int16_t); ++i) {
        acc.CFront() += static_cast<int32_t>(accumulators[i]);
      }
#endif
      acc.CFront() *= unquant_mult;
    }
  }
}

void DumpMatrix(int8_t *m, Index rows, Index cols) {
  std::cerr << rows << 'x' << cols << '\n';
  for (Index i = 0; i < rows; ++i) {
    for (Index j = 0; j < cols; ++j) {
      std::cerr << (int16_t)m[i * cols + j] << ' ';
    }
    std::cerr << '\n';
  }
}

struct TestMatricesRef : TestMatrices8 {
  TestMatricesRef(Tile shape_in) :
    TestMatrices8(shape_in),
    C_reference(shape.A_rows * shape.B_cols) {

    AccessT ref_access(
        RowMajorAccess<int8_t>(A.begin(), shape.inner),
        ColMajorAccess<int8_t>(B.begin(), shape.inner),
        RowMajorAccess<int32_t>(C_reference.begin(), shape.B_cols));
    Signed8ReferenceMult<AccessT>(ref_access, shape);
  }

  AlignedVector<int32_t> C_reference;
};

struct TestMatricesRef_UnquantizeAndWrite : TestMatricesUnquantizeAndWriteRowMajorAccess {
  TestMatricesRef_UnquantizeAndWrite(Tile shape_in, float unquant_mult) :
    TestMatricesUnquantizeAndWriteRowMajorAccess(shape_in, unquant_mult),
    C_reference(shape.A_rows * shape.B_cols) {

    AccessT ref_access(
        RowMajorAccess<int8_t>(A.begin(), shape.inner),
        ColMajorAccess<int8_t>(B.begin(), shape.inner),
        UnquantizeAndWriteRowMajorAccess<float>(C_reference.begin(), shape.B_cols, {unquant_mult}));
    Signed8ReferenceMult_UnquantizeAndWrite<AccessT>(ref_access, shape, unquant_mult);
  }

  AlignedVector<float> C_reference;
};


#ifndef INTGEMM_THIS_IS_SSE2
template <class Kernel> void TestMultiplyNoOverhang(Tile shape) {
  // These are sanity checks on the arguments, not the code.
  CHECK(shape.A_rows % Kernel::kTile.A_rows == 0);
  CHECK(shape.inner % Kernel::kTile.inner == 0);
  CHECK(shape.B_cols % Kernel::kTile.B_cols == 0);
  TestMatricesRef t(shape);
  MultiplyNoOverhang<TestMatricesRef::AccessT, Kernel>(t.Accessor(), shape);
  CHECK(!memcmp(t.C_reference.begin(), t.C.begin(), shape.A_rows * shape.B_cols * sizeof(int32_t)));
/*  for (Index i = 0; i < shape.A_rows; ++i) {
    for (Index j = 0; j < shape.B_cols; ++j) {
      CHECK(t.C_reference[i * shape.B_cols + j] == C_test[i * shape.B_cols + j]);
    }
  }*/
}

template <class Kernel> void TestMultiplyNoOverhang_UnquantizeAndWrite(Tile shape, float unquant_mult) {
  // These are sanity checks on the arguments, not the code.
  CHECK(shape.A_rows % Kernel::kTile.A_rows == 0);
  CHECK(shape.inner % Kernel::kTile.inner == 0);
  CHECK(shape.B_cols % Kernel::kTile.B_cols == 0);
  TestMatricesRef_UnquantizeAndWrite t(shape, unquant_mult);
  MultiplyNoOverhang<TestMatricesRef_UnquantizeAndWrite::AccessT, Kernel>(t.Accessor(), shape);
  CHECK(!memcmp(t.C_reference.begin(), t.C.begin(), shape.A_rows * shape.B_cols * sizeof(float)));
}

template <class Kernel> void TestMultiplyNoOverhangShapes() {
  Tile shape = Kernel::kTile;
  // Minimum size.
  TestMultiplyNoOverhang<Kernel>(shape);
  // Multiples on each dimension.
  TestMultiplyNoOverhang<Kernel>(Tile{shape.A_rows * 2, shape.inner, shape.B_cols});
  TestMultiplyNoOverhang<Kernel>(Tile{shape.A_rows, shape.inner * 2, shape.B_cols});
  TestMultiplyNoOverhang<Kernel>(Tile{shape.A_rows, shape.inner, shape.B_cols * 2});
  TestMultiplyNoOverhang<Kernel>(Tile{shape.A_rows * 2, shape.inner * 2, shape.B_cols * 2});
  // Try a bunch of shapes!
  for (shape.A_rows = 0; shape.A_rows <= Kernel::kTile.A_rows * 9; shape.A_rows += Kernel::kTile.A_rows) {
    for (shape.inner = 0; shape.inner <= Kernel::kTile.inner * 9; shape.inner += Kernel::kTile.inner) {
      for (shape.B_cols = 0; shape.B_cols <= Kernel::kTile.B_cols * 9; shape.B_cols += Kernel::kTile.B_cols) {
        TestMultiplyNoOverhang<Kernel>(shape);
      }
    }
  }
}

template <class Kernel> void TestMultiplyNoOverhangShapes_UnquantizeAndWrite(float unquant_mult) {
  Tile shape = Kernel::kTile;
  // Minimum size.
  TestMultiplyNoOverhang_UnquantizeAndWrite<Kernel>(shape, unquant_mult);
  // Multiples on each dimension.
  TestMultiplyNoOverhang_UnquantizeAndWrite<Kernel>(Tile{shape.A_rows * 2, shape.inner, shape.B_cols}, unquant_mult);
  TestMultiplyNoOverhang_UnquantizeAndWrite<Kernel>(Tile{shape.A_rows, shape.inner * 2, shape.B_cols}, unquant_mult);
  TestMultiplyNoOverhang_UnquantizeAndWrite<Kernel>(Tile{shape.A_rows, shape.inner, shape.B_cols * 2}, unquant_mult);
  TestMultiplyNoOverhang_UnquantizeAndWrite<Kernel>(Tile{shape.A_rows * 2, shape.inner * 2, shape.B_cols * 2}, unquant_mult);
  // Try a bunch of shapes!
  for (shape.A_rows = 0; shape.A_rows <= Kernel::kTile.A_rows * 9; shape.A_rows += Kernel::kTile.A_rows) {
    for (shape.inner = 0; shape.inner <= Kernel::kTile.inner * 9; shape.inner += Kernel::kTile.inner) {
      for (shape.B_cols = 0; shape.B_cols <= Kernel::kTile.B_cols * 9; shape.B_cols += Kernel::kTile.B_cols) {
        TestMultiplyNoOverhang_UnquantizeAndWrite<Kernel>(shape, unquant_mult);
      }
    }
  }
}

TEST_CASE("MultiplyNoOverhang Signed8 " INTGEMM_TEST_NAME, "[tile]") {
  if (kCPU < CPUType::INTGEMM_ARCH) return;
  TestMultiplyNoOverhangShapes<Signed8>();
}

// Due to unordered_unfurl in dot.inl, the inner dimension can change order.
// That impacts saturation.  Then the test doesn't mach reference on arches
// that use 16-bit saturating accumlation.  So we only test inner unrolling on
// VNNI.
#ifdef INTGEMM_THIS_IS_AVX512VNNI
TEST_CASE("MultiplyNoOverhang inner unroll " INTGEMM_TEST_NAME, "[tile][multiply]") {
  if (kCPU < CPUType::INTGEMM_ARCH) return;
  typedef UnrollKernel<1, 2, 1, Signed8> Kernel;
  Tile shape = {1, sizeof(Register) * 2, 1};
  TestMultiplyNoOverhang<Kernel>(shape);
  TestMultiplyNoOverhang<Kernel>({1, sizeof(Register) * 4, 1});
  TestMultiplyNoOverhangShapes<Kernel>();
}

TEST_CASE("MultiplyNoOverhang Signed8 UnquantizeAndWrite " INTGEMM_TEST_NAME, "[tile]") {
  if (kCPU < CPUType::INTGEMM_ARCH) return;
  TestMultiplyNoOverhangShapes_UnquantizeAndWrite<Signed8>(1.7f);
}

TEST_CASE("MultiplyNoOverhang inner unroll UnquantizeAndWrite " INTGEMM_TEST_NAME, "[tile][multiply]") {
  if (kCPU < CPUType::INTGEMM_ARCH) return;
  float unquant_mult = 1.7f;
  typedef UnrollKernel<1, 2, 1, Signed8> Kernel;
  Tile shape = {1, sizeof(Register) * 2, 1};
  TestMultiplyNoOverhang_UnquantizeAndWrite<Kernel>(shape, unquant_mult);
  TestMultiplyNoOverhang_UnquantizeAndWrite<Kernel>({1, sizeof(Register) * 4, 1}, unquant_mult);
  TestMultiplyNoOverhangShapes_UnquantizeAndWrite<Kernel>(unquant_mult);
}
#endif // INTGEMM_THIS_IS_AVX512VNNI

// If the inner dimension is just twice, then there isn't any non-determinism in saturation order.
TEST_CASE("MultiplyNoOverhang simple inner unroll " INTGEMM_TEST_NAME, "[tile][multiply]") {
  if (kCPU < CPUType::INTGEMM_ARCH) return;
  typedef UnrollKernel<1, 2, 1, Signed8> Kernel;
  static_assert(1 == Kernel::kTile.A_rows, "A_rows matches on unrolled kernel");
  static_assert(sizeof(Register) * 2 == Kernel::kTile.inner, "inner matches on kernel unrolled 2x");
  static_assert(1 == Kernel::kTile.B_cols, "B_cols matches on kernel unrolled");
  TestMultiplyNoOverhang<Kernel>({1, sizeof(Register) * 2, 1});
  TestMultiplyNoOverhang<Kernel>({5, sizeof(Register) * 2, 7});
}

TEST_CASE("MultiplyNoOverhang Simple 17 rows " INTGEMM_TEST_NAME, "[tile][multiply]") {
  if (kCPU < CPUType::INTGEMM_ARCH) return;
  typedef UnrollKernel<17, 1, 1, Signed8> Kernel;
  TestMultiplyNoOverhang<Kernel>({17, sizeof(Register), 1});
}

// Annoyingly, catch's cross-product stuff requires the first argument be a type, which is pretty useless for a cross-product of integers.
TEMPLATE_TEST_CASE("MultiplyNoOverhang Unrolled Signed8 " INTGEMM_TEST_NAME, "[tile][multiply]",
    (UnrollKernel<1, 1, 1, Signed8>),
    (UnrollKernel<1, 1, 2, Signed8>),
    (UnrollKernel<1, 1, 3, Signed8>),
    (UnrollKernel<1, 1, 4, Signed8>),
    (UnrollKernel<1, 1, 5, Signed8>),
    (UnrollKernel<1, 1, 6, Signed8>),
    (UnrollKernel<1, 1, 7, Signed8>),
    (UnrollKernel<1, 1, 8, Signed8>),
    (UnrollKernel<1, 1, 9, Signed8>),
    (UnrollKernel<1, 1, 10, Signed8>),
    (UnrollKernel<1, 1, 11, Signed8>),
    (UnrollKernel<1, 1, 12, Signed8>),
    (UnrollKernel<1, 1, 13, Signed8>),
    (UnrollKernel<1, 1, 14, Signed8>),
    (UnrollKernel<1, 1, 15, Signed8>),
    (UnrollKernel<1, 1, 16, Signed8>),
    (UnrollKernel<1, 1, 17, Signed8>),
    (UnrollKernel<1, 1, 18, Signed8>),
    (UnrollKernel<1, 1, 19, Signed8>),
    (UnrollKernel<1, 1, 31, Signed8>),
    (UnrollKernel<1, 1, 32, Signed8>),
    (UnrollKernel<2, 1, 1, Signed8>),
    (UnrollKernel<2, 1, 2, Signed8>),
    (UnrollKernel<2, 1, 3, Signed8>),
    (UnrollKernel<3, 1, 1, Signed8>),
    (UnrollKernel<3, 1, 3, Signed8>),
    (UnrollKernel<4, 1, 1, Signed8>),
    (UnrollKernel<5, 1, 1, Signed8>),
    (UnrollKernel<6, 1, 4, Signed8>),
    (UnrollKernel<7, 1, 3, Signed8>),
    (UnrollKernel<7, 1, 4, Signed8>),
    (UnrollKernel<15, 1, 1, Signed8>),
    (UnrollKernel<15, 1, 2, Signed8>),
    (UnrollKernel<16, 1, 1, Signed8>),
    (UnrollKernel<17, 1, 1, Signed8>)
    ) {
  if (kCPU < CPUType::INTGEMM_ARCH) return;
  TestMultiplyNoOverhangShapes<TestType>();
}

TEST_CASE("Multiply " INTGEMM_TEST_NAME, "[tile][multiply]") {
  if (kCPU < CPUType::INTGEMM_ARCH) return;
  Tile shape{1, sizeof(Register), 1};
  for (shape.A_rows = 1; shape.A_rows < 33; ++shape.A_rows) {
    for (shape.B_cols = 1; shape.B_cols < 33; ++shape.B_cols) {
      TestMatricesRef t(shape);
      Multiply<TestMatricesRef::AccessT, Signed8, 7, 3>(t.Accessor(), shape);
      CHECK(!memcmp(t.C_reference.begin(), t.C.begin(), shape.A_rows * shape.B_cols * sizeof(int32_t)));
      memset(t.C.begin(), 0, shape.A_rows * shape.B_cols * sizeof(int32_t));
      Multiply<TestMatricesRef::AccessT, Signed8, 4, 5>(t.Accessor(), shape);
      CHECK(!memcmp(t.C_reference.begin(), t.C.begin(), shape.A_rows * shape.B_cols * sizeof(int32_t)));
    }
  }
}

#endif // no INTGEMM_THIS_IS_SSE2

} // namespace INTGEMM_ARCH
} // namespace intgemm

#undef INTGEMM_ARCH
#undef INTGEMM_TARGET
#undef INTGEMM_TEST_NAME
