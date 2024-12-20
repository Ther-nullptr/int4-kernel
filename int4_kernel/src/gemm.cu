#include <cutlass/gemm/device/gemm.h>
#include <gemm.h>

void matmul_host(const Int4Storage *A, const Int4Storage *B, uint32_t M,
                 uint32_t N, uint32_t K, int32_t *C) {
  using ShapeMMAThreadBlock = cutlass::gemm::GemmShape<128, 128, 128>;
  using ShapeMMAWarp = cutlass::gemm::GemmShape<64, 64, 128>;
  using InstructionShape = cutlass::gemm::GemmShape<16, 8, 64>;
  using EpilogueOutputOp = cutlass::epilogue::thread::LinearCombinationClamp<
      int32_t, 128 / cutlass::sizeof_bits<int32_t>::value, int32_t, float>;
  using ThreadblockSwizzle =
      cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>;
  constexpr int NumStages = 6;

  using Gemm = cutlass::gemm::device::Gemm<
      cutlass::int4b_t,               // ElementA
      cutlass::layout::RowMajor,      // LayoutA
      cutlass::int4b_t,               // ElementB
      cutlass::layout::ColumnMajor,   // LayoutB
      int32_t,                        // ElementOutput
      cutlass::layout::RowMajor,      // LayoutOutput
      int32_t,                        // ElementAccumulator
      cutlass::arch::OpClassTensorOp, // tag indicating Tensor Cores
      cutlass::arch::Sm80, // tag indicating target GPU compute architecture  //
                           // TODO: This is just for compiling on my laptop
                           // temporarily. Should be higher when doing
                           // benchmarking.
      ShapeMMAThreadBlock, // Threadblock-level tile size
      ShapeMMAWarp,        // Warp-level tile size
      InstructionShape,    // TensorCore instruction shape
      EpilogueOutputOp,    // Epilogue output operator
      ThreadblockSwizzle,  // Threadblock swizzle
      NumStages            // Number of stages
      >;

  Gemm gemmOp;

  using GemmCoord = cutlass::gemm::GemmCoord;

  typename Gemm::Arguments arguments{{static_cast<GemmCoord::Index>(M),
                                      static_cast<GemmCoord::Index>(N),
                                      static_cast<GemmCoord::Index>(K)},
                                     {(cutlass::int4b_t *)A, K},
                                     {(cutlass::int4b_t *)B, K},
                                     {C, N},
                                     {C, N},
                                     {1, 0}};

  auto status = gemmOp(arguments);
  ensure(status == cutlass::Status::kSuccess, cutlassGetStatusString(status));
}
