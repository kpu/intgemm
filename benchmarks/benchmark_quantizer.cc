#include "../intgemm.h"
#include "../aligned.h"
#include "../stop_watch.h"
#include "../ssse3_gemm.h"
#include "../avx2_gemm.h"
#include "../avx512_gemm.h"

#include <chrono>
#include <iomanip>
#include <random>
#include <vector>

namespace {
template <class Backend> void QuantizerBench(const float *in, int8_t *out, std::size_t count) {
  if (intgemm::kCPU < Backend::kUses) return;
  Backend::Quantize(in, out, 1.0, count);
  const std::size_t kTries = 60;
  auto start = std::chrono::system_clock::now();
  for (std::size_t t = 0; t < kTries; ++t) {
    Backend::Quantize(in, out, 1.0, count);
  }
  auto end = std::chrono::system_clock::now();
  double took = std::chrono::duration<double>(end - start).count() / kTries;
  std::cout << std::setw(9) << count << ' ' << std::fixed << std::setw(9) << std::setprecision(7) << took << ' ' << Backend::kName << std::endl;
}
} // namespace

int main() {
  for (std::size_t count = 1; count < (1ULL<<30); count *= 2) {
    intgemm::AlignedVector<float> in(count);
    intgemm::AlignedVector<int8_t> out(count);
    std::mt19937 gen;
    std::uniform_real_distribution<float> dist(-129.0, 129.0);
    for (float &element : in) {
      element = dist(gen);
    }
    QuantizerBench<intgemm::SSSE3_8bit>(in.begin(), out.begin(), count);
    QuantizerBench<intgemm::AVX2_8bit>(in.begin(), out.begin(), count);
#ifdef INTGEMM_COMPILER_SUPPORTS_AVX512
    QuantizerBench<intgemm::AVX512_8bit>(in.begin(), out.begin(), count);
#endif
  }
}
