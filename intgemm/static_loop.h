#pragma once

/* Usage:
 * template<std::size_t I> void example() {
 *   StaticLoop<I>([](std::size_t i){
 *     std::cout << i << std::endl;
 *   });
 * }
 */

#include <utility>

namespace intgemm {

// Call f(0) f(1), ... for each Seq in the sequence.
template<typename Functor, std::size_t...Seq>
inline void StaticLoopUnroll(Functor &&f, std::index_sequence<Seq...>) {
  ((void)(f(Seq)), ...);
}

// StaticLoopType<Iterations>()(f) will call f(0) f(1) ... f(Iterations - 1)
template<std::size_t Iterations> struct StaticLoopType {
  template <typename Functor> inline void operator()(Functor &&f) const {
    StaticLoopUnroll(f, std::make_index_sequence<Iterations>{});
  }
};

// StaticLoop<Iterations>(f) will call f(0) f(1) ... f(Iterations - 1)
template<std::size_t Iterations> constexpr StaticLoopType<Iterations> StaticLoop;

} // namespace intgemm
