#pragma once

/*
 * This file contains following mathematics functions evaluated in compile-time:
 */
template <int N> constexpr float exp();
template <unsigned N> constexpr long long factorial();

/*
 * Implementations...
 */

/*
 * Exp
 */
namespace { // anonymous namespace
template <unsigned N>
constexpr float exp_nonnegative();
template <>
constexpr float exp_nonnegative<1>() { return 2.718281828459045f; }
template <>
constexpr float exp_nonnegative<0>() { return 1.f; }
template <unsigned N>
constexpr float exp_nonnegative() { return exp_nonnegative<1>() * exp_nonnegative<N-1>(); }

template <int N, bool IsNonnegative>
struct exp_impl;
template <int N>
struct exp_impl<N, true> { constexpr static float value = exp_nonnegative<N>(); };
template <int N>
struct exp_impl<N, false> { constexpr static float value = 1.f / exp_nonnegative<-N>(); };
} // anonymous namespace

template <int N>
constexpr float exp() { return exp_impl<N, N >= 0>::value; }

/*
 * Factorial
 */
template <unsigned N>
constexpr long long factorial() { return N * factorial<N-1>(); }
template <>
constexpr long long factorial<0>() { return 1; }
