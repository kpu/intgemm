#include "StopWatch.h"

#include <iostream>
#include <err.h>

double Subtract(const struct timespec &first, const struct timespec &second) {
  return static_cast<double>(first.tv_sec - second.tv_sec) + static_cast<double>(first.tv_nsec - second.tv_nsec) / 1000000000.0;
}

StopWatch::StopWatch(const std::string &label, float divide) : label_(label), divide_(divide) {
  if (-1 == clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &started_))
    err(1, "Failed to read CLOCK_PROCESS_CPUTIME_ID");
}

StopWatch::~StopWatch() {
  struct timespec stopped;
  if (-1 == clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &stopped))
    err(1, "Failed to read CLOCK_PROCESS_CPUTIME_ID");

  std::cout << label_ << '\t' << (Subtract(stopped, started_) / divide_) << std::endl;
}
