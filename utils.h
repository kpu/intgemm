#pragma once

#include <tuple>

namespace intgemm {

/*
 * Sequence of unsigned integers
 *
 * Examples:
 *   sequence<1, 2, 3>()
 *   sequence_pushback<4, sequence<1, 2, 3>>() = sequence<1, 2, 3, 4>()
 *   sequence_popfront<sequence<1, 2, 3>>() = sequence<2, 3>()
 *   make_sequence<3>() = sequence<0, 1, 2>()
 */
template <unsigned... Indices>
struct sequence { using type = sequence; };

template <unsigned I, typename Sequence>
struct sequence_pushback;

template <unsigned I, unsigned... Indices>
struct sequence_pushback<I, sequence<Indices...>> : sequence<Indices..., I> {};

template <typename Sequence>
struct sequence_popfront;

template <unsigned FirstIndex, unsigned... RestIndices>
struct sequence_popfront<sequence<FirstIndex, RestIndices...>> : sequence<RestIndices...> {};

namespace { // anonymous namespace
template <unsigned N>
struct make_sequence_impl : sequence_pushback<N - 1, typename make_sequence_impl<N - 1>::type> {};
template <>
struct make_sequence_impl<0> : sequence<> {};
} // anonymous namespace

template <unsigned N>
using make_sequence = typename make_sequence_impl<N>::type;

/*
 * Make a subtuple
 */
template <typename Tuple, unsigned... Indices>
using subtuple_t = typename std::tuple<typename std::tuple_element<Indices, Tuple>::type...>;

template <typename Tuple, unsigned... Indices>
constexpr subtuple_t<Tuple, Indices...> make_subtuple(const Tuple& tuple, sequence<Indices...>) {
  return std::make_tuple(std::get<Indices>(tuple)...);
}

/*
 * Factorial
 */
static constexpr unsigned long long factorial(unsigned n) {
  return n <= 1 ? 1 : n * factorial(n - 1);
}

/*
 * e^n, where n is integer
 */
static constexpr double expi_nonnegative(unsigned n) {
  return n == 0 ? 1.0 : (n == 1 ? 2.718281828459045 : expi_nonnegative(n / 2) * expi_nonnegative((n + 1) / 2));
}

static constexpr double expi(int n) {
  return (n >= 0 ? expi_nonnegative(n) : 1.0 / expi_nonnegative(-n));
}

/*
 * Multi-dimmension static loop iterator over range [0, 0, ...] - [Ns...] (exclusive)
 * starting from IterationNumber-th iteration. Keep in mind that iterations are counted
 * from 0.
 * 
 * For example, StaticLoopIterator<3, 5, 2> creates iterator:
 *   [1, 1] -> [2, 0] -> ... -> [4, 0] -> [4, 1]
 * because first 3 steps i.e.:
 *   [0, 0] -> [0, 1] -> [1, 0]
 * are skipped.
 * 
 * To extract I-th component of the iterator, use get<I>() function.
 */
template <Index IterationNumber, Index... Ns>
struct StaticLoopIterator {
private:
  template <Index N, Index FirstDimmension, Index... RestDimmensions>
  struct get_dimmension_s {
    static constexpr Index value = get_dimmension_s<N - 1, RestDimmensions...>::value;
  };

  template <Index FirstDimmension, Index... RestDimmensions>
  struct get_dimmension_s<0, FirstDimmension, RestDimmensions...> {
    static constexpr Index value = FirstDimmension;
  };

  template <Index N, Index FirstDimmension, Index... RestDimmensions>
  struct multiply_first_n_dimmensions_s {
    static constexpr Index value = FirstDimmension * multiply_first_n_dimmensions_s<N - 1, RestDimmensions...>::value;
  };

  template <Index FirstDimmension, Index... RestDimmensions>
  struct multiply_first_n_dimmensions_s<1, FirstDimmension, RestDimmensions...> {
    static constexpr Index value = FirstDimmension;
  };

  template <Index N>
  struct multiply_first_n_dimmensions : multiply_first_n_dimmensions_s<N, Ns...> {};

public:
  /*
   * Total number of iteration in the given dimmensions.
   */
  static constexpr Index total_iterations = multiply_first_n_dimmensions<sizeof...(Ns)>::value;

  /*
   * Current iteration number
   */
  static constexpr Index iteration_number = IterationNumber;

  /*
   * Get I-th dimmension of the iterator.
   */
  template <Index Ith = 0>
  static constexpr inline Index N() {
    return get_dimmension_s<Ith, Ns...>::value;
  }

  /*
   * Return I-th component of the iterator.
   */
  template <Index Ith = 0>
  static constexpr inline Index I() {
    return (iteration_number * multiply_first_n_dimmensions<Ith + 1>::value / total_iterations) % N<Ith>();
  }

  /*
   * Next iterator
   */
  using next = StaticLoopIterator<iteration_number + 1, Ns...>;

  /*
   * Last iterator
   */
  using last = StaticLoopIterator<total_iterations - 1, Ns...>;
};

/*
 * Create multi-dimmension static loop iterator over range [0, 0, ...] - [Ns...] (exclusive)
 * 
 * For example, MakeStaticLoopIterator<5, 2> creates iterator:
 *   [0, 0] -> [0, 1] -> [1, 0] -> [1, 1] -> [2, 0] -> ... -> [4, 0] -> [4, 1]
 */
template <Index... Ns>
using MakeStaticLoopIterator = StaticLoopIterator<0, Ns...>;

/*
 * Static loop over range defined by the give static loop iterator.
 *
 * To use it, you need to create a body structure containing static inline procedure
 * 'body' with template parameter Iterator. If you need you can also
 * add extra template parameters.
 *
 * Example:
 *   struct Test {
 *     template <typename Iterator, typename Number>
 *     static inline void body(const char* text, Number number) {
 *       std::cout << "[" << Iterator::template I<0>() << ", " << Iterator::template I<1>() << "] " << text << " " << number << std::endl;
 *     }
 *   };
 *
 * To run static loop, you just need to call StaticLoop<Body, Iterator>. It takes
 * the same parameters as body procedure inside Body structure.
 * 
 * Example:
 *   StaticLoop<Test, MakeStaticLoopIterator<5, 2>>("Test", 1);
 *
 * Output of the example:
 *
 * [0, 0] Test 1
 * [0, 1] Test 1
 * [1, 0] Test 1
 * [1, 1] Test 1
 * [2, 0] Test 1
 * [2, 1] Test 1
 * [3, 0] Test 1
 * [3, 1] Test 1
 * [4, 0] Test 1
 * [4, 1] Test 1
 *
 */
template <typename Body, typename StaticLoopIterator, typename std::enable_if<std::is_same<StaticLoopIterator, typename StaticLoopIterator::last>::value>::type* = nullptr, typename... Args>
__attribute__((always_inline)) static inline void StaticLoop(Args&&... args) {
  Body::template body<StaticLoopIterator>(std::forward<Args>(args)...);
}

template <typename Body, typename StaticLoopIterator, typename std::enable_if<!std::is_same<StaticLoopIterator, typename StaticLoopIterator::last>::value>::type* = nullptr, typename... Args>
__attribute__((always_inline)) static inline void StaticLoop(Args&&... args) {
  Body::template body<StaticLoopIterator>(std::forward<Args>(args)...);
  StaticLoop<Body, typename StaticLoopIterator::next>(std::forward<Args>(args)...);
}

/*
 * Round up
 */
static constexpr Index round_up(Index value, Index factor) {
  return (value + factor - 1) / factor * factor;
}

}
