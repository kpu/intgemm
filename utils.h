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
constexpr unsigned long long factorial(unsigned n) {
  return n <= 1 ? 1 : n * factorial(n - 1);
}

/*
 * e^x
 */
namespace { // anonymous namespace
constexpr double exp_nonnegative(unsigned x) {
  return x == 0 ? 1.0 : (x == 1 ? 2.718281828459045 : exp_nonnegative(x / 2) * exp_nonnegative((x + 1) / 2));
}
} // anonymous namespace

constexpr double exp(int x) {
  return (x >= 0 ? exp_nonnegative(x) : 1.0 / exp_nonnegative(-x));
}

}
