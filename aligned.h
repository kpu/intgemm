#pragma once

// Define allocation like:
// free_ptr<Integer> quantized(AlignedArray<Integer>(rows * cols));
// This is only used by tests.

#include <cstdlib>
#include <memory>
#include <type_traits>

namespace intgemm {

struct DeleteWithFree {
  template <class T> void operator() (T *t) const {
    std::free(const_cast<std::remove_const_t<T>* >(t));
  }
};
template <class T> using free_ptr = std::unique_ptr<T, DeleteWithFree>;
// Return memory suitably aligned for SIMD.
template <class T> T* AlignedArray(std::size_t size) {
  return static_cast<T*>(aligned_alloc(64, size * sizeof(T)));
}

template <class T> class AlignedVector {
  public:
    explicit AlignedVector(std::size_t size) : mem_(AlignedArray<T>(size)) {}

    T &operator[](std::size_t offset) { return mem_.get()[offset]; }
    const T &operator[](std::size_t offset) const { return mem_.get()[offset]; }

    T *get() { return mem_.get(); }
    const T *get() const { return mem_.get(); }
  private:
    free_ptr<T> mem_;
};

} // namespace intgemm
