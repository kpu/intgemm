#pragma once
#include <time.h>
#include <string>
#include <stdint.h>

class StopWatch {
  public:
    explicit StopWatch(const std::string &label, float divide = 1);

    ~StopWatch();

  private:
    struct timespec started_;
    uint64_t tsc_;
    std::string label_;
    float divide_;
};
