#include "../intgemm.h"
#include "../aligned.h"
#include "../utils.h"

#include <chrono>
#include <random>
#include <iostream>

using namespace intgemm;

static constexpr Index TestCases[][2] = {
  {1, 1}, {1, 2}, {1, 3}, {1, 4},
  {2, 1}, {2, 2}, {2, 3}, {2, 4},
  {3, 1}, {3, 2}, {3, 3}, {3, 4},
  {4, 1}, {4, 2}, {4, 3}, {4, 4}
};

static constexpr Index TestCasesN = sizeof(TestCases) / sizeof(TestCases[0]);

template <typename Backend>
struct BenchmarkLoop {
  template <typename Iterator>
  static inline void body(Index A_rows, Index width, Index B_cols, Index repeats) {
    constexpr Index TileRows = TestCases[Iterator::template I<0>()][0];
    constexpr Index TileColumnsMultiplier = TestCases[Iterator::template I<0>()][1];

    std::mt19937 gen;
    std::uniform_int_distribution<typename Backend::Integer> dist(0, 32);
    gen.seed(0);

    AlignedVector<typename Backend::Integer> A_prepared(A_rows * width);
    AlignedVector<typename Backend::Integer> B_prepared(width * B_cols);
    AlignedVector<int32_t> C(A_rows * B_cols);

    std::chrono::duration<double> duration = std::chrono::nanoseconds::zero();
    for (Index i = 0; i < repeats; ++i) {
      std::generate(A_prepared.begin(), A_prepared.end(), [&]() {
        return dist(gen);
      });

      std::generate(B_prepared.begin(), B_prepared.end(), [&]() {
        return dist(gen);
      });

      auto start = std::chrono::system_clock::now();
      Backend::template Multiply<TileRows, TileColumnsMultiplier>(A_prepared.begin(), B_prepared.begin(), A_rows, width, B_cols, callbacks::Write<int32_t>(C.begin()));
      duration += std::chrono::system_clock::now() - start;
    }

    std::cout << repeats << " iterations of " << Backend::kName << " with tile = " << TileRows << "x" << 8 * TileColumnsMultiplier << " took: " << duration.count() << " seconds." << std::endl;
  }
};

int main(int argc, char** argv) {
  if (argc < 3) {
    std::cerr << "Usage: " << argv[0] << " repeats ARCH [ARCH2 [ARCH3...]]" << std::endl;
    return 1;
  }
  Index repeats = atoi(argv[1]);

  std::string available_archs[] = {
    "SSSE3_8bit",
    "SSE2_16bit",
    "AVX2_8bit",
    "AVX2_16bit",
//  Tiling not supported yet
// #ifdef INTGEMM_COMPILER_SUPPORTS_AVX512BW
//     "AVX512_8bit",
//     "AVX512_16bit",
// #endif
#ifdef INTGEMM_COMPILER_SUPPORTS_AVX512VNNI
    "AVX512VNNI_8bit",
#endif
  };

  std::vector<std::string> selected_archs;
  for (int i = 2; i < argc; ++i) {
    bool found = false;
    for (int j = 0; j < sizeof(available_archs) / sizeof(available_archs[0]) && !found; ++j) {
      if (available_archs[j].compare(argv[i]) == 0) {
        selected_archs.push_back(argv[i]);
        found = true;
      }
    }
    if (!found)
      std::cerr << "Warning: Unknown architecture '" << argv[i] << "'!" << std::endl;
  }

#define ARCH_BENCHMARK_IMPL(arch, A_rows, width, B_cols) \
  if (selected_archs[i].compare(#arch) == 0) { StaticLoop<BenchmarkLoop<arch>, MakeStaticLoopIterator<TestCasesN>>((A_rows), (width), (B_cols), repeats); }

  for (int i  = 0; i < selected_archs.size(); ++i) {
    ARCH_BENCHMARK_IMPL(SSSE3_8bit, 768, 768, 768)
    ARCH_BENCHMARK_IMPL(SSE2_16bit, 768, 768, 768)
    ARCH_BENCHMARK_IMPL(AVX2_8bit, 768, 768, 768)
    ARCH_BENCHMARK_IMPL(AVX2_16bit, 768, 768, 768)
    //  Tiling not supported yet
    // #ifdef INTGEMM_COMPILER_SUPPORTS_AVX512BW
      // ARCH_BENCHMARK_IMPL(AVX512BW_8bit, 768, 768, 768)
      // ARCH_BENCHMARK_IMPL(AVX512BW_8bit, 768, 768, 768)
    // #endif
    #ifdef INTGEMM_COMPILER_SUPPORTS_AVX512VNNI
      ARCH_BENCHMARK_IMPL(AVX512VNNI_8bit, 768, 768, 768)
    #endif
  }

  return 0;
}
