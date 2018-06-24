#pragma once
#include <stdint.h>
#include <vector>
#include <x86intrin.h>

class StopWatch {
  public:
    StopWatch(std::vector<uint64_t> &stats)
      : stats_(stats), start_(__rdtsc()) {}

    ~StopWatch() {
      uint64_t stop = __rdtsc();
      stats_.push_back(stop - start_);
    }

  private:
    std::vector<uint64_t> &stats_;
    uint64_t start_;
};

