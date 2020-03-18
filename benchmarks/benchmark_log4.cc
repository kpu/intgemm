#include "../log4/log4.h"

#include "../aligned.h"
#include "../stop_watch.h"

#include <algorithm>
#include <cassert>
#include <cstring>
#include <iostream>
#include <random>
#include <chrono>

namespace intgemm {
namespace {

struct Summary {
  double wall;
  double tsc;
};

std::ostream &operator<<(std::ostream &o, const Summary &summary) {
  return o << "wall = " << summary.wall << "\ttsc = " << summary.tsc;
}

Summary Summarize(const std::vector<Timing> &samples, const int scale = 1) {
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
  return ret;
}

std::chrono::duration<double> Summarize(const std::vector<std::chrono::duration<double> > &samples, const int scale=1) {
  auto first = samples[0];
  return scale * std::accumulate(samples.begin() + 1, samples.end(), first) / samples.size();
}

INTGEMM_AVX512F __m512 fp32(__m512 a, __m512 b) {
  return _mm512_mul_ps(a, b);
}

INTGEMM_AVX512BW __m512i MAddUBS(__m512i a, __m512i b) {
  __m512i ret = _mm512_maddubs_epi16(a, b);
  return _mm512_maddubs_epi16(ret, _mm512_set1_epi16(1));
}

INTGEMM_AVX512VNNI void BenchmarkLog4() {
  std::mt19937 gen;
  std::uniform_int_distribution<uint8_t> dist(0, 255);
  gen.seed(1234);
  const int kCalls = 2048;
  const int kSamples = 100000;

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
  __m512i accum = _mm512_setzero_si512();
  const __m512i kLookup = _mm512_set_epi64(0xffab734d342217, 0x0f0a060402010000, 0xffab734d342217, 0x0f0a060402010000, 0xffab734d342217, 0x0f0a060402010000, 0xffab734d342217, 0x0f0a060402010000);

  for (int s = 0; s < kSamples; ++s) {
    StopWatch w(stats_shift);
    for (const __m512i *a_it = a_begin, *b_it = b_begin; a_it != a_end; ++a_it, ++b_it) {
      accum = _mm512_add_epi64(accum, DotLog4_Shift(*a_it, *b_it));
    }
  }
  for (int s = 0; s < kSamples; ++s) {
    StopWatch w(stats_lookup8);
    for (const __m512i *a_it = a_begin, *b_it = b_begin; a_it != a_end; ++a_it, ++b_it) {
      accum = _mm512_add_epi64(accum, DotLog4_Lookup8(*a_it, *b_it, kLookup, subtractreg));
    }
  }
  for (int s = 0; s < kSamples; ++s) {
    StopWatch w(stats_lookup16);
    for (const __m512i *a_it = a_begin, *b_it = b_begin; a_it != a_end; ++a_it, ++b_it) {
      accum = _mm512_add_epi64(accum, DotLog4_Lookup16(*a_it, *b_it, subtract65535));
    }
  }
  for (int s = 0; s < kSamples; ++s) {
    __m512* accumf = (__m512*) &accum;
    StopWatch w(stats_fp32);
    for (const __m512 *a_it = (__m512*) a_begin, *b_it = (__m512*) b_begin; a_it != (__m512*) a_end; ++a_it, ++b_it) {
      *accumf = _mm512_add_ps(*accumf, fp32(*a_it, *b_it));
    }
  }
  for (int s = 0; s < kSamples; ++s) {
    StopWatch w(stats_maddubs);
    for (const __m512i *a_it = a_begin, *b_it = b_begin; a_it != a_end; ++a_it, ++b_it) {
      accum = _mm512_adds_epi16(accum, MAddUBS(*a_it, *b_it));
    }
  }
  for (int s = 0; s < kSamples; ++s) {
    StopWatch w(stats_shift);
    for (const __m512i *a_it = a_begin, *b_it = b_begin; a_it != a_end; ++a_it, ++b_it) {
      accum = _mm512_add_epi64(accum, DotLog4_Shift(*a_it, *b_it));
    }
  }
  for (int s = 0; s < kSamples; ++s) {
    StopWatch w(stats_vnni);
    for (const __m512i *a_it = a_begin, *b_it = b_begin; a_it != a_end; ++a_it, ++b_it) {
      accum = _mm512_dpbusds_epi32(accum, *a_it, *b_it);
    }
  }


  uint64_t result[8];
  std::memcpy(result, &accum, sizeof(accum));
  // This isn't valid, but it does do enough to prevent code removal.
  int64_t total = std::accumulate(result, result + 8, -255 * subtract65535);
  std::memcpy(result, &subtractreg, sizeof(accum));
  total = std::accumulate(result, result + 8, total);
  asm volatile("" : "+r" (total));
  std::cout<< " Lookup8  " << Summarize(stats_lookup8)  << '\n';
  std::cout<< " Lookup16 " << Summarize(stats_lookup16) << '\n';
  std::cout<< " Shift    " << Summarize(stats_shift)    << '\n';
  std::cout<< " MAddUBS  " << Summarize(stats_maddubs, /*scale=*/2) << '\n';
  std::cout<< " FP32     " << Summarize(stats_fp32,    /*scale=*/8) << '\n';
  std::cout<< " VNNI     " << Summarize(stats_vnni,    /*scale=*/2) << '\n';
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


