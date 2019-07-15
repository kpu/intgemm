#include "log4/log4.h"

#include "aligned.h"
#include "stop_watch.h"

#include <algorithm>
#include <cassert>
#include <cstring>
#include <iostream>
#include <random>

namespace intgemm {
namespace {

INTGEMM_AVX512BW void BenchmarkLog4() {
  std::mt19937 gen;
  std::uniform_int_distribution<uint8_t> dist(0, 255);
  gen.seed(1234);
  const int kCalls = 4096;

  AlignedVector<uint8_t> a(kCalls * sizeof(__m512i)), b(kCalls * sizeof(__m512i));
  for (auto& it : a) {
    it = dist(gen);
  }
  for (auto& it : b) {
    it = dist(gen);
  }

  uint64_t subtract255;
  std::vector<uint64_t> stats;
  const __m512i *a_begin = reinterpret_cast<const __m512i*>(a.begin()), *a_end = reinterpret_cast<const __m512i*>(a.end());
  const __m512i *b_begin = reinterpret_cast<const __m512i*>(b.begin());
  __m512i accum = _mm512_setzero_si512();
  const __m512i kLookup = _mm512_set_epi64(0xffab734d342217, 0x0f0a060402010000, 0xffab734d342217, 0x0f0a060402010000, 0xffab734d342217, 0x0f0a060402010000, 0xffab734d342217, 0x0f0a060402010000);
  {
    StopWatch w(stats);
    for (; a_begin != a_end; ++a_begin, ++b_begin) {
      accum = _mm512_add_epi64(accum, DotLog4_Lookup8(*a_begin, *b_begin, kLookup, subtract255));
    }
  }
  uint64_t result[8];
  std::memcpy(result, &accum, sizeof(accum));
  // This isn't valid, but it does do enough to prevent code removal.
  int64_t total = std::accumulate(result, result + 8, -255 * subtract255);
  asm volatile("" : "+r" (total));
  std::cout << "Lookup8 " << stats[0] << '\n';
}


} // namespace
} // namespace intgemm

// Program takes no input
int main(int argc, char ** argv) {
  std::cerr << "Remember to run this on a specific core:\ntaskset --cpu-list 0 " << argv[0] << std::endl;

  using namespace intgemm;
  BenchmarkLog4();
  return 0;
}


