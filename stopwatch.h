#pragma once
#include <time.h>
#include <string>

class StopWatch {
  public:
    explicit StopWatch(const std::string &label);

    ~StopWatch();

  private:
    struct timespec started_;
    std::string label_;
};
