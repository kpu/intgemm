#pragma once

#include "types.h"
#include <tuple>

namespace intgemm {

// Function to absorb arguments from C++11 template unpacking
// Usage: unordered_unfurl(T...);
template<typename... Args> void unordered_unfurl(Args&&...) {}

// C++11 implementation of C++14's make_index_sequence.
// This is a bugfix from a stackoverflow post that did [0, N] while the standard does [0, N).
// https://stackoverflow.com/questions/52844615/is-that-possible-to-have-a-for-loop-in-compile-time-with-runtime-or-even-compile
template <size_t... Is>
struct index_sequence{};

namespace detail {
    template <size_t I,size_t...Is>
    struct make_index_sequence_impl : make_index_sequence_impl<I-1,I-1,Is...> {};

    template <size_t...Is>
    struct make_index_sequence_impl<0,Is...>
    {
        using type = index_sequence<Is...>;
    };
}

template<size_t N>
using make_index_sequence = typename detail::make_index_sequence_impl<N>::type;

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

} // namespace intgemm
