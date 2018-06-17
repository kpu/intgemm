#pragma once

// Define allocation like:
// free_ptr<Integer> quantized(AlignedArray<Integer>(rows * cols));
// This is only used by tests.

#include <memory>

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

} // namespace intgemm
