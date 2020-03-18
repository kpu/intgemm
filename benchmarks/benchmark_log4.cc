#include "../log4/log4.h"

#include "../aligned.h"
#include "../stop_watch.h"
#include "../utils.h"

#include <algorithm>
#include <cassert>
#include <cstring>
#include <iostream>
#include <iomanip>
#include <random>

namespace intgemm {
namespace {

struct Summary {
  double wall;
  double tsc;
};

std::ostream &operator<<(std::ostream &o, const Summary &summary) {
  return o << "wall = " << std::fixed << std::setprecision(5) << summary.wall << "\ttsc = " << std::fixed << std::setprecision(5) << summary.tsc;
}

Summary Summarize(const std::vector<Timing> &samples, const int calls, const int scale) {
  Summary ret;
  ret.wall = 0;
  ret.tsc = 0;  
  for (const Timing &t : samples) {
    ret.wall += t.wall;
    ret.tsc += t.tsc;
  }
  ret.wall *= scale;
  ret.tsc *= scale;
  ret.wall /= samples.size();
  ret.tsc /= samples.size();
  ret.wall /= calls;
  ret.tsc /= calls;
  return ret;
}

struct Shift {
  template <typename Iterator> INTGEMM_AVX512BW __attribute__((always_inline)) static void body(const __m512i *a, const __m512i *b, __m512i *c, __m512i *, uint64_t *) {
    const Index idx = Iterator::template I<0>();
    c[idx] = _mm512_add_epi64(c[idx], DotLog4_Shift(a[idx], b[idx]));
  }
};

struct MAddUBS16 {
  template <typename Iterator> INTGEMM_AVX512BW __attribute__((always_inline)) static void body(const __m512i *a, const __m512i *b, __m512i *c, __m512i *, uint64_t *) {
    // Into 16
    __m512i added = _mm512_maddubs_epi16(a[Iterator::template I<0>()], b[Iterator::template I<0>()]);
    // 16-bit accum.
    c[Iterator::template I<0>()] = _mm512_adds_epi16(c[Iterator::template I<0>()], added);
  }
};

struct MAddUBS32 {
  template <typename Iterator> INTGEMM_AVX512BW __attribute__((always_inline)) static void body(const __m512i *a, const __m512i *b, __m512i *c, __m512i *, uint64_t *) {
    // Into 16
    __m512i added = _mm512_maddubs_epi16(a[Iterator::template I<0>()], b[Iterator::template I<0>()]);
    // Into 32
    added = _mm512_madd_epi16(added, _mm512_set1_epi16(1));
    // 32-bit accum.
    c[Iterator::template I<0>()] = _mm512_add_epi32(c[Iterator::template I<0>()], added);
  }
};

struct Lookup8 {
  template <typename Iterator> INTGEMM_AVX512BW __attribute__((always_inline)) static void body(const __m512i *a, const __m512i *b, __m512i *c, __m512i *subtractreg, uint64_t *) {
    const Index idx = Iterator::template I<0>();
    const __m512i kLookup = _mm512_set_epi64(0xffab734d342217, 0x0f0a060402010000, 0xffab734d342217, 0x0f0a060402010000, 0xffab734d342217, 0x0f0a060402010000, 0xffab734d342217, 0x0f0a060402010000);
    c[idx] = _mm512_add_epi64(c[idx], DotLog4_Lookup8(a[idx], b[idx], kLookup, *subtractreg));
  }
};

struct Lookup16 {
  template <typename Iterator> INTGEMM_AVX512BW __attribute__((always_inline)) static void body(const __m512i *a, const __m512i *b, __m512i *c, __m512i *subtractreg, uint64_t *subtract) {
    const Index idx = Iterator::template I<0>();
    c[idx] = _mm512_add_epi64(c[idx], DotLog4_Lookup16(a[idx], b[idx], *subtract));
  }
};

struct VNNI {
  template <typename Iterator> INTGEMM_AVX512VNNI __attribute__((always_inline)) static void body(const __m512i *a, const __m512i *b, __m512i *c, __m512i *, uint64_t *) {
    const Index idx = Iterator::template I<0>();
    c[idx] = _mm512_dpbusds_epi32(c[idx], a[idx], b[idx]);
  }
};

struct FP32 {
  template <typename Iterator> INTGEMM_AVX512F __attribute__((always_inline)) static void body(const __m512i *a, const __m512i *b, __m512i *c, __m512i *, uint64_t *) {
    const Index idx = Iterator::template I<0>();
    reinterpret_cast<__m512&>(c[idx]) = _mm512_fmadd_ps(
        reinterpret_cast<const __m512&>(a[idx]),
        reinterpret_cast<const __m512&>(b[idx]),
        reinterpret_cast<__m512&>(c[idx]));
  }
};

template <class Backend, Index Unroll> INTGEMM_AVX512VNNI void Try(const __m512i *a_begin, const __m512i *a_end, const __m512i *b_begin, const char *name, const int scale) {
  const int kSamples = 10000;
  __m512i accum[Unroll];
  memset(accum, 0, sizeof(accum));

  __m512i subtractreg = _mm512_setzero_si512();
  uint64_t subtract65535 = 0;

  std::vector<Timing> stats;
  for (int s = 0; s < kSamples; ++s) {
    StopWatch w(stats);
    for (const __m512i *a_it = a_begin, *b_it = b_begin; a_it != a_end; a_it += Unroll, b_it += Unroll) {
      StaticLoop<Backend, MakeStaticLoopIterator<Unroll> >(a_it, b_it, &accum[0], &subtractreg, &subtract65535);
    }
  }

  uint64_t result[sizeof(accum) / sizeof(uint64_t)];
  std::memcpy(result, &accum, sizeof(accum));
  // This isn't valid, but it does do enough to prevent code removal.
  int64_t total = std::accumulate(result, result + sizeof(result) / sizeof(uint64_t), -255 * subtract65535);
  std::memcpy(result, &subtractreg, sizeof(subtractreg));
  total = std::accumulate(result, result + sizeof(subtractreg) / sizeof(uint64_t), total);
  asm volatile("" : "+r" (total), "+r" (subtract65535));

  std::cout << Summarize(stats, a_end - a_begin, scale) << " " << name << '\n';
}

INTGEMM_AVX512VNNI void BenchmarkLog4() {
  std::mt19937 gen;
  std::uniform_int_distribution<uint8_t> dist(0, 255);
  gen.seed(1234);
  const Index kUnroll = 16;
  const int kCalls = (2048 / kUnroll) * kUnroll;

  AlignedVector<uint8_t> a(kCalls * sizeof(__m512i)), b(kCalls * sizeof(__m512i));
  for (auto& it : a) {
    it = dist(gen);
  }
  for (auto& it : b) {
    it = dist(gen);
  }

  __m512i subtractreg = _mm512_setzero_si512();
  uint64_t subtract65535 = 0;
  std::vector<Timing> stats_lookup8, stats_lookup16, stats_shift, stats_maddubs, stats_fp32, stats_vnni;
  const __m512i *a_begin = reinterpret_cast<const __m512i*>(a.begin()), *a_end = reinterpret_cast<const __m512i*>(a.end());
  const __m512i *b_begin = reinterpret_cast<const __m512i*>(b.begin());

  // The multipliers represent how many times it would have to run to process the same number of elements.
  Try<Shift, kUnroll>(a_begin, a_begin + kCalls, b_begin, "Shift", 1);
  Try<Lookup8, kUnroll>(a_begin, a_begin + kCalls, b_begin, "Lookup8", 1);
  Try<Lookup16, kUnroll>(a_begin, a_begin + kCalls, b_begin, "Lookup16", 1);
  Try<MAddUBS16, kUnroll>(a_begin, a_begin + kCalls, b_begin, "MAddUBS16", 2);
  Try<MAddUBS32, kUnroll>(a_begin, a_begin + kCalls, b_begin, "MAddUBS32", 2);
  Try<VNNI, kUnroll>(a_begin, a_begin + kCalls, b_begin, "VNNI", 2);
  Try<FP32, kUnroll>(a_begin, a_begin + kCalls, b_begin, "FP32", 8);
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


