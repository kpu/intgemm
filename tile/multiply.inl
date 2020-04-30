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

// Upcast 16 to 32 if needed.
template <std::size_t... i> INTGEMM_TARGET static inline void Sum16To32(Register *regs, int16_t, index_sequence<i...>) {
  unordered_unfurl((regs[i] = madd_epi16(regs[i], set1_epi16<Register>(1)))...);
}
template <std::size_t... i> INTGEMM_TARGET static inline void Sum16To32(Register *, int32_t, index_sequence<i...>) {}

/* Multiply assuming the matrix sizes are a multiple of the kernel size. */
template <class Kernel, class AccessT> INTGEMM_TARGET __attribute__((flatten)) static inline void MultiplyNoOverhang(AccessT access, const Tile shape) {
  assert(shape.A_rows % Kernel::kTile.A_rows == 0);
  assert(shape.inner % Kernel::kTile.inner == 0);
  assert(shape.B_cols % Kernel::kTile.B_cols == 0);
  for (Index B_col = 0; B_col < shape.B_cols; B_col += Kernel::kTile.B_cols) {
    AccessT column_adjusted = access.BAdd(0, B_col).CAdd(0, B_col);
    for (Index A_row = 0; A_row < shape.A_rows; A_row += Kernel::kTile.A_rows) {
      AccessT col_row = column_adjusted.AAdd(A_row, 0).CAdd(A_row, 0);
      constexpr Index Outputs = Kernel::kTile.A_rows * Kernel::kTile.B_cols;

      // Accumulate values in temporary C registers.
      Register c_regs[Outputs] = {setzero_si<Register>()};
      // If C is column major it would be better to have column-major registers
      // since this determines the order used by Reduce32.
      Access<typename AccessT::A, typename AccessT::B, RegisterRowMajorAccess> reg_access(
          col_row.AAccessor(),
          col_row.BAccessor(),
          RegisterRowMajorAccess(c_regs, Kernel::kTile.B_cols));

      for (Index inner = 0; inner < shape.inner; inner += Kernel::kTile.inner) {
        Kernel::Run(reg_access.AAdd(0, inner).BAdd(inner, 0));
      }

      Sum16To32(c_regs, typename Kernel::Packed::C(), make_index_sequence<Outputs>());
      // Horizontally add 32-bit values.
      Reduce32<Outputs, Sum32Op>(c_regs);
      col_row.CAccessor().template Write<Kernel::kTile.A_rows, Kernel::kTile.B_cols>(c_regs);
    }
  }
}

/* Multiply matrices without being a multiple of an unrolled kernel size.  The
 * inner dimension still needs to be a multiple of sizeof(Register) for int8_t
 * or sizeof(Register) / 2 for int16_t.
 * Kernel should be a small kernel like Shifted8 or Signed8; this function will
 * unroll.
 * A_rows and B_cols specify the unrolled kernel size to use for most of the
 * multiply; these impact speed but not output.
 */
template <class Kernel, Index A_rows, Index B_cols, class AccessT> INTGEMM_TARGET static inline void Multiply(AccessT access, const Tile shape) {
  // Still has to be a multiple of the underlying Kernel, but usually that's just 1 x sizeof(Register) x 1.
  assert(shape.A_rows % Kernel::kTile.A_rows == 0);
  assert(shape.inner % Kernel::kTile.inner == 0);
  assert(shape.B_cols % Kernel::kTile.B_cols == 0);

  typedef UnrollKernel<A_rows, 1, B_cols, Kernel> Big;
  Tile overhang = {
    shape.A_rows % Big::kTile.A_rows,
    shape.inner % Big::kTile.inner,
    shape.B_cols % Big::kTile.B_cols
  };
  Tile big_shape = {
    shape.A_rows - overhang.A_rows,
    shape.inner - overhang.inner,
    shape.B_cols - overhang.B_cols
  };
  // Top left corner.
  MultiplyNoOverhang<Big>(access, big_shape);
  // Bottom currently including right side.  TODO: unrolled kernel, rather than dumb loop.
  MultiplyNoOverhang<Kernel>(
      access.AAdd(big_shape.A_rows, 0).CAdd(big_shape.A_rows, 0),
      Tile {overhang.A_rows, shape.inner, shape.B_cols});
  // Right side except bottom.  TODO: unrolled kernel, rather than dumb loop.
  MultiplyNoOverhang<Kernel>(
      access.BAdd(0, big_shape.B_cols).CAdd(0, big_shape.B_cols),
      Tile {big_shape.A_rows, shape.inner, overhang.B_cols});
}

} // namespace INTGEMM_ARCH
} // namespace intgemm

#undef INTGEMM_ARCH
#undef INTGEMM_TARGET
