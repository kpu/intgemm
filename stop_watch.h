#pragma once
#include <stdint.h>
#include <cstdlib>
#include <chrono>
#include <vector>
#include <iostream>

namespace intgemm {

static inline uint64_t rdtsc_begin(uint32_t &processor) {
  uint32_t lo, hi;
  __asm__ __volatile__ (
      "cpuid\n\t"
      "rdtscp\n\t"
      "mov %%eax, %0\n\t"
      "mov %%edx, %1\n\t"
      "mov %%ecx, %2\n\t"
      : "=r" (lo), "=r" (hi), "=r" (processor)
      : /* no input */
      : "rax", "rbx", "rcx", "rdx");
  return static_cast<uint64_t>(hi) << 32 | lo;
}

static inline uint64_t rdtsc_end(uint32_t &processor) {
  uint32_t lo, hi;
  __asm__ __volatile__ (
      "rdtscp\n\t"
      "mov %%eax, %0\n\t"
      "mov %%edx, %1\n\t"
      "mov %%ecx, %2\n\t"
      "cpuid\n\t"
      : "=r" (lo), "=r" (hi), "=r" (processor)
      : /* no input */
      : "rax", "rbx", "rcx", "rdx");
  return static_cast<uint64_t>(hi) << 32 | lo;
}

struct Timing {
  uint64_t tsc;
  double wall;
};

class StopWatch {
  public:
    StopWatch(std::vector<Timing> &stats)
      : stats_(stats), start_wall_(std::chrono::steady_clock::now()), start_tsc_(rdtsc_begin(processor_)) {}

    ~StopWatch() {
      uint32_t proc;
      uint64_t stop_tsc = rdtsc_end(proc);
      std::chrono::time_point<std::chrono::steady_clock> stop_wall = std::chrono::steady_clock::now();
      if (proc != processor_) {
        std::cerr << "Detected core change from " << processor_ << " to " << proc << std::endl;
        abort();
      }
      Timing out;
      out.tsc = stop_tsc - start_tsc_;
      out.wall = std::chrono::duration_cast<std::chrono::duration<double>>(stop_wall - start_wall_).count();
      stats_.push_back(out);
    }

  private:
    std::vector<Timing> &stats_;
    uint32_t processor_;
    uint64_t start_tsc_;
    std::chrono::time_point<std::chrono::steady_clock> start_wall_;
};

} // namespace intgemm
