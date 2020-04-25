// This file is included multiple times from dot.h for different architectures.
// Do not #include headers here.  Put them in dot.h.  The reason is that other
// headers should not see these macros (and are within rights to use them too).
#if defined(INTGEMM_THIS_IS_AVX512VNNI)
#define INTGEMM_ARCH AVX512VNNI
#define INTGEMM_TARGET INTGEMM_AVX512VNNI
#elif defined(INTGEMM_THIS_IS_AVX512BW)
#define INTGEMM_ARCH AVX512BW
#define INTGEMM_TARGET INTGEMM_AVX512BW
#elif defined(INTGEMM_THIS_IS_AVX2)
#define INTGEMM_ARCH AVX2
#define INTGEMM_TARGET INTGEMM_AVX2
#elif defined(INTGEMM_THIS_IS_SSSE3)
#define INTGEMM_ARCH SSSE3
#define INTGEMM_TARGET INTGEMM_SSSE3
#elif defined(INTGEMM_THIS_IS_SSE2)
#define INTGEMM_ARCH SSE2
#define INTGEMM_TARGET INTGEMM_SSE2
#endif

namespace intgemm {
namespace INTGEMM_ARCH {

/* When Register is used as a template argument, gcc warns
 * warning: ignoring attributes on template argument ‘Register’ {aka ‘__vector(8) long long int’} [-Wignored-attributes]
 * So here is a class that doesn't warn.
 */
class RegisterRowMajorAccess {
  public:
    typedef Register Content;

    RegisterRowMajorAccess(Content *data, Index cols)
      : data_(data), cols_(cols) {}

    RegisterRowMajorAccess Add(Index row, Index col) const {
      return RegisterRowMajorAccess(data_ + row * cols_ + col, cols_);
    }

    const Content &Front() const { return *data_; }
    Content &Front() { return *data_; }

  private:
    Content *data_;
    Index cols_;
};

/* gcc _mm512_dpbusds_epi32 is slow because it inserts spurious vmovdqa64 instructions.
 * Simple test program:
 * #include <immintrin.h>
 *
 * __m512i Foo(const __m512i *a, const __m512i b0, const __m512i b1, std::size_t count) {
 *   register __m512i c0 = _mm512_setzero_epi32();
 *   register __m512i c1 = _mm512_setzero_epi32();
 *   for (std::size_t i = 0; i < count; ++i) {
 *     c0 = _mm512_dpbusds_epi32(c0, a[i], b0);
 *     c1 = _mm512_dpbusds_epi32(c1, a[i], b1);
 *   }
 *   // Do not optimize away
 *   return _mm512_sub_epi32(c0, c1);
 * }
 * Then with g++ (Gentoo 9.2.0-r2 p3) 9.2.0 run as
 *   g++ -mavx512vnni -O3 example.cc -S
 * We get some inefficient asm:
 * .L3:
 * vmovdqa64 (%rdi), %zmm6
 * vmovdqa64 %zmm3, %zmm0
 * vmovdqa64 %zmm4, %zmm2
 * addq  $64, %rdi
 * vpdpbusds %zmm5, %zmm6, %zmm0
 * vpdpbusds %zmm1, %zmm6, %zmm2
 * vmovdqa64 %zmm0, %zmm3
 * vmovdqa64 %zmm2, %zmm4
 * cmpq  %rdi, %rax
 * jne .L3
 *
 * Why does it copy from zmm3 to zmm0, then copy zmm0 to zmm3 each loop????
 * So for gcc instead of
 *   c = _mm512_dpbusds_epi32(c, a, b);
 * I use:
 *   asm ("vpdpbusds %2, %1, %0" : "+x"(c) : "x"(a), "mx"(b));
 * and that works better in the test program.
 *
 * clang 9.0.1 deals with this fine.
 */
#ifdef INTGEMM_THIS_IS_AVX512VNNI
INTGEMM_TARGET static inline void VNNI8(Register &c, Register a, Register b) {
#if defined(__GNUC__) && !defined(__clang__) && !defined(__INTEL_COMPILER)
    asm ("vpdpbusds %2, %1, %0" : "+x"(c) : "x"(a), "mx"(b));
#else
    c = _mm512_dpbusds_epi32(c, a, b);
#endif
}
#endif

// 8-bit integer multiplication unsigned A * signed B.
#if !defined(INTGEMM_THIS_IS_SSE2) // No int8 on SSE2.
struct Shifted8 {
  template <class Access> INTGEMM_TARGET static inline void Run(Access access) {
    const Register &a = reinterpret_cast<const Register&>(access.AFront());
    const Register &b = reinterpret_cast<const Register&>(access.BFront());
#ifdef INTGEMM_THIS_IS_AVX512VNNI
    VNNI8(access.CFront(), a, b);
#else
    const Register ones = set1_epi16<Register>(1);
    Register mult = maddubs_epi16(a, b);
    mult = madd_epi16(mult, ones);
    access.CFront() = add_epi32(access.CFront(), mult);
#endif
  }

  static constexpr Tile kTile { 1, sizeof(Register), 1 };

  struct Packed {
    typedef uint8_t A;
    typedef int8_t B;
    typedef int32_t C;
  };
};

// 8-bit integer multiplication signed A * signed B.  Slower.
struct Signed8 {
  template <class Access> INTGEMM_TARGET static inline void Run(Access access) {
    const Register &a = reinterpret_cast<const Register&>(access.AFront());
    const Register &b = reinterpret_cast<const Register&>(access.BFront());
    const Register a_positive = abs_epi8(a);
    // b_signed = b * sign(a)
#if defined(INTGEMM_THIS_IS_AVX512VNNI) || defined(INTGEMM_THIS_IS_AVX512BW)
    // AVX512 doesn't have sign. Get a mask of negative a values.
    __mmask64 neg_mask = _mm512_test_epi8_mask(a, _mm512_set1_epi8(-128));
    const Register zeros = setzero_si<Register>();
    // Negate by subtracting from zero with the mask.
    const Register b_signed = _mm512_mask_sub_epi8(b, neg_mask, zeros, b);
#else
    // Not AVX512.
    const Register b_signed = sign_epi8(b, a);
#endif

    // c += |a| * b_signed
#if defined(INTGEMM_THIS_IS_AVX512VNNI)
    VNNI8(access.CFront(), a_positive, b_signed);
#else
    Register mult = maddubs_epi16(a_positive, b_signed);
    access.CFront() = adds_epi16(access.CFront(), mult);
#endif
  }

  static constexpr Tile kTile { 1, sizeof(Register), 1 };

  struct Packed {
    typedef int8_t A;
    typedef int8_t B;
#if defined(INTGEMM_THIS_IS_AVX512VNNI)
    typedef int32_t C;
#else
    typedef int16_t C;
#endif
  };
};
#endif // No int8 on SSE2.

// Signed 16-bit integer multiplication.
struct Signed16 {
  template <class Access> INTGEMM_TARGET static inline void Run(Access access) {
    const Register &a = reinterpret_cast<const Register&>(access.AFront());
    const Register &b = reinterpret_cast<const Register&>(access.BFront());
#if defined(INTGEMM_THIS_IS_AVX512VNNI)
    access.CFront() = _mm512_dpwssds_epi32(access.CFront(), a, b);
#else
    Register mult = madd_epi16(a, b);
    access.CFront() = add_epi32(access.CFront(), mult);
#endif
  }

  static constexpr Tile kTile { 1, sizeof(Register), 1 };

  struct Packed {
    typedef int16_t A;
    typedef int16_t B;
    typedef int32_t C;
  };
};

// Statically unroll a kernel into a larger tile.
// Can't have Tile as a value until C++20.
template <Index A_rows, Index inner, Index B_cols, class Backend> struct UnrollKernel {
  template <class Access> INTGEMM_TARGET __attribute__((flatten)) static inline void Run(Access access) {
    body(access, make_index_sequence<A_rows * inner * B_cols>());
  }
  static constexpr Tile kTile { A_rows * Backend::kTile.A_rows, inner * Backend::kTile.inner, B_cols * Backend::kTile.B_cols };
  typedef typename Backend::Packed Packed;

 private:
  template <class Access, size_t... Index> INTGEMM_TARGET __attribute__((flatten)) static inline void body(
      Access access,
      index_sequence<Index...>) {
    // for each inner computed as (Index / A_rows / B_cols)
    //   for each A_row computed as (Index % (A_rows * B_cols)) / B_cols
    //     for each B_col computed as (Index % B_cols)
    unordered_unfurl((
       Backend::Run(access
            .AAdd((Index % (A_rows * B_cols)) / B_cols * Backend::kTile.A_rows, (Index / A_rows / B_cols) * Backend::kTile.inner)
            .BAdd((Index / A_rows / B_cols) * Backend::kTile.inner, (Index % B_cols) * Backend::kTile.B_cols)
            .CAdd((Index % (A_rows * B_cols)) / B_cols * Backend::kTile.A_rows, (Index % B_cols) * Backend::kTile.B_cols))
       // Backend returns void, so use a tuple to make 0.
       , 0)...);
  }
};

class Constant1sAccess {
  public:
    typedef int8_t Content;
    Constant1sAccess() : reg_(set1_epi8<Register>(1)) {}
    Constant1sAccess Add(Index, Index) const {
      return *this;
    }
    const Content &Front() const {
      return reinterpret_cast<const Content&>(reg_);
    }
  private:
    Register reg_;
};

} // namespace INTGEMM_ARCH
} // namespace

#undef INTGEMM_TARGET
#undef INTGEMM_ARCH
