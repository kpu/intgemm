#include "log4/log4.h"

#include "aligned.h"
#include "stop_watch.h"

#include <algorithm>
#include <cassert>
#include <cstring>
#include <iostream>
#include <random>
#include <chrono>

namespace intgemm {
namespace {

uint64_t Summarize(const std::vector<uint64_t> &samples) {
  return std::accumulate(samples.begin(), samples.end(), 0) / samples.size();
}

std::chrono::duration<double> Summarize(const std::vector<std::chrono::duration<double> > &samples) {
  auto first = samples[0];
  return std::accumulate(samples.begin() + 1, samples.end(), first) / samples.size();
}


INTGEMM_AVX512BW __m512i MAddUBS(__m512i a, __m512i b) {
  __m512i ret = _mm512_maddubs_epi16(a, b);
  return _mm512_maddubs_epi16(ret, _mm512_set1_epi16(1));
}

INTGEMM_AVX512BW void BenchmarkLog4() {
  std::mt19937 gen;
  std::uniform_int_distribution<uint8_t> dist(0, 255);
  gen.seed(1234);
  const int kCalls = 1024;
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
  std::vector<uint64_t> stats_lookup8, stats_lookup16, stats_shift, stats_maddubs;
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

  uint64_t result[8];
  std::memcpy(result, &accum, sizeof(accum));
  // This isn't valid, but it does do enough to prevent code removal.
  int64_t total = std::accumulate(result, result + 8, -255 * subtract65535);
  std::memcpy(result, &subtractreg, sizeof(accum));
  total = std::accumulate(result, result + 8, total);
  asm volatile("" : "+r" (total));
  std::cout << "Lookup8 " << Summarize(stats_lookup8) << " Lookup16 " << Summarize(stats_lookup16) << " Shift " << Summarize(stats_shift) << " MAddUBS " << Summarize(stats_maddubs) << '\n';
}

INTGEMM_AVX512BW void BenchmarkLog4_Chrono() {
  std::mt19937 gen;
  std::uniform_int_distribution<uint8_t> dist(0, 255);
  gen.seed(1234);
  const int kCalls = 1024;
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
  std::vector<std::chrono::duration<double> > stats_lookup8, stats_lookup16, stats_shift, stats_maddubs;
  const __m512i *a_begin = reinterpret_cast<const __m512i*>(a.begin()), *a_end = reinterpret_cast<const __m512i*>(a.end());
  const __m512i *b_begin = reinterpret_cast<const __m512i*>(b.begin());
  __m512i accum = _mm512_setzero_si512();
  const __m512i kLookup = _mm512_set_epi64(0xffab734d342217, 0x0f0a060402010000, 0xffab734d342217, 0x0f0a060402010000, 0xffab734d342217, 0x0f0a060402010000, 0xffab734d342217, 0x0f0a060402010000);

  for (int s = 0; s < kSamples; ++s) {
    auto start = std::chrono::system_clock::now();
    for (const __m512i *a_it = a_begin, *b_it = b_begin; a_it != a_end; ++a_it, ++b_it) {
      accum = _mm512_add_epi64(accum, DotLog4_Shift(*a_it, *b_it));
    }
    auto end = std::chrono::system_clock::now();
    std::chrono::duration<double> elapsed_seconds = end-start;
    stats_shift.push_back(elapsed_seconds);
  }
  for (int s = 0; s < kSamples; ++s) {
    auto start = std::chrono::system_clock::now();
    for (const __m512i *a_it = a_begin, *b_it = b_begin; a_it != a_end; ++a_it, ++b_it) {
      accum = _mm512_add_epi64(accum, DotLog4_Lookup8(*a_it, *b_it, kLookup, subtractreg));
    }
    auto end = std::chrono::system_clock::now();
    std::chrono::duration<double> elapsed_seconds = end-start;
    stats_lookup8.push_back(elapsed_seconds);
  }
  for (int s = 0; s < kSamples; ++s) {
    auto start = std::chrono::system_clock::now();
    for (const __m512i *a_it = a_begin, *b_it = b_begin; a_it != a_end; ++a_it, ++b_it) {
      accum = _mm512_add_epi64(accum, DotLog4_Lookup16(*a_it, *b_it, subtract65535));
    }
    auto end = std::chrono::system_clock::now();
    std::chrono::duration<double> elapsed_seconds = end-start;
    stats_lookup16.push_back(elapsed_seconds);
  }
  for (int s = 0; s < kSamples; ++s) {
    auto start = std::chrono::system_clock::now();
    for (const __m512i *a_it = a_begin, *b_it = b_begin; a_it != a_end; ++a_it, ++b_it) {
      accum = _mm512_adds_epi16(accum, MAddUBS(*a_it, *b_it));
    }
    auto end = std::chrono::system_clock::now();
    std::chrono::duration<double> elapsed_seconds = end-start;
    stats_maddubs.push_back(elapsed_seconds);
  }
  for (int s = 0; s < kSamples; ++s) {
    auto start = std::chrono::system_clock::now();
    for (const __m512i *a_it = a_begin, *b_it = b_begin; a_it != a_end; ++a_it, ++b_it) {
      accum = _mm512_add_epi64(accum, DotLog4_Shift(*a_it, *b_it));
    }
    auto end = std::chrono::system_clock::now();
    std::chrono::duration<double> elapsed_seconds = end-start;
    stats_shift.push_back(elapsed_seconds);
  }

  uint64_t result[8];
  std::memcpy(result, &accum, sizeof(accum));
  // This isn't valid, but it does do enough to prevent code removal.
  int64_t total = std::accumulate(result, result + 8, -255 * subtract65535);
  std::memcpy(result, &subtractreg, sizeof(accum));
  total = std::accumulate(result, result + 8, total);
  asm volatile("" : "+r" (total));
  std::cout << "Lookup8 " << Summarize(stats_lookup8).count() << " Lookup16 " << Summarize(stats_lookup16).count() << " Shift " << Summarize(stats_shift).count() << " MAddUBS " << Summarize(stats_maddubs).count() << '\n';
}


} // namespace
} // namespace intgemm

// Program takes no input
int main(int argc, char ** argv) {
  std::cerr << "Remember to run this on a specific core:\ntaskset --cpu-list 0 " << argv[0] << std::endl;

  using namespace intgemm;
  BenchmarkLog4();
  BenchmarkLog4_Chrono();
  return 0;
}


