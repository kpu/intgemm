#pragma once
#include <cstdlib>
#include <new>
#include <stdlib.h>

// 64-byte aligned simple vector.

namespace intgemm {

template <class T> class AlignedVector {
  public:
    explicit AlignedVector(std::size_t size)
      : size_(size) {
      if (posix_memalign(reinterpret_cast<void **>(&mem_), 64, size * sizeof(T))) {
        throw std::bad_alloc();
      }
    }

    AlignedVector(const AlignedVector&) = delete;
    AlignedVector& operator=(const AlignedVector&) = delete;

    ~AlignedVector() { std::free(mem_); }

    std::size_t size() const { return size_; }

    T &operator[](std::size_t offset) { return mem_[offset]; }
    const T &operator[](std::size_t offset) const { return mem_[offset]; }

    T *begin() { return mem_; }
    const T *begin() const { return mem_; }
    T *end() { return mem_ + size_; }
    const T *end() const { return mem_ + size_; }

    template <typename ReturnType>
    ReturnType *as() { return reinterpret_cast<ReturnType*>(mem_); }

  private:
    T *mem_;
    std::size_t size_;
};

} // namespace intgemm
