#pragma once
#include <time.h>
#include <string>

class StopWatch {
  public:
    explicit StopWatch(const std::string &label, float divide = 1);

    ~StopWatch();

  private:
    struct timespec started_;
    std::string label_;
    float divide_;
};
