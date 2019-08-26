#include "aligned.h"
#include "sigmoid.h"
#include "stop_watch.h"

#include <iostream>
#include <random>

uint64_t Average(const std::vector<uint64_t> &vec) {
  return std::accumulate(vec.begin(), vec.end(), 0) / vec.size();
}

using namespace intgemm;
int main() {
  std::mt19937 gen;
  std::uniform_real_distribution<float> dist(-10.0, 10.0);
  intgemm::AlignedVector<float> v(16 * 1000);
  for (float &i : v) {
    i = dist(gen);
  }
  std::vector<uint64_t> timing_ken;
  const int kSamples = 10000;
  for (unsigned int i = 0; i < kSamples; ++i) {
    intgemm::StopWatch w(timing_ken);
    for (float *j = v.begin(); j != v.end(); j += 16) {
      __m512 result = Reyong(*reinterpret_cast<__m512*>(j));
      asm volatile ("" : "+x" (result));
    }
  }
  

  std::cout << Average(timing_ken) << '\n';
}
