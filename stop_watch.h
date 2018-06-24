#pragma once
#include <stdint.h>
#include <cstdlib>
#include <vector>
#include <x86intrin.h>
#include <iostream>

class StopWatch {
  public:
    StopWatch(std::vector<uint64_t> &stats)
      : stats_(stats), start_(__rdtscp(&processor_)) {}

    ~StopWatch() {
      unsigned int proc;
      uint64_t stop = __rdtscp(&proc);
      if (proc != processor_) {
        std::cerr << "Detected core change from " << processor_ << " to " << proc << std::endl;
        abort();
      }
      stats_.push_back(stop - start_);
    }

  private:
    std::vector<uint64_t> &stats_;
    uint32_t processor_;
    uint64_t start_;
};

