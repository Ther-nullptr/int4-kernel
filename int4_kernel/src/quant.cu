#include "quant.h"

template <typename T> __device__ __half int_to_half(T value) {
  return __int2half_rn(static_cast<int>(value));
}

__global__ void sym_quantize_f16_i4_kernel(const half *__restrict__ x,
                                           const half *__restrict__ scale,
                                           uint32_t rows, uint32_t colsSrc,
                                           uint32_t colsDst,
                                           Int4Storage *__restrict__ q) {
  uint32_t row = threadIdx.y + blockIdx.y * blockDim.y;
  uint32_t colDst = threadIdx.x + blockIdx.x * blockDim.x;
  if (row >= rows || colDst * kElementsPerVector >= colsSrc) {
    return;
  }
  Int4Storage storage;
  memset(&storage, 0, sizeof(storage));
  uint32_t id = colDst * kElementsPerVector + row * colsSrc;
#pragma unroll
  for (int i = 0; i < kElementsPerVector; ++i) {
    bool safe = (colDst * kElementsPerVector + i) < colsSrc;
    if (safe) {
      half data = __hdiv(x[id + i], scale[row]);

      int qval = clamp(__half2int_rn(data), qmin, qmax);
      Int4Subbyte{reinterpret_cast<cutlass::int4b_t *>(&storage), i}.set(qval);
    }
  }

  q[colDst + row * colsDst] = storage;
}

void sym_quant_host(const half *x, const half *scale, uint32_t rows,
                    uint32_t colsSrc, uint32_t colsDst, Int4Storage *q) {

  dim3 block{std::min<uint32_t>(colsDst, 32), std::min<uint32_t>(rows, 16)};
  dim3 grid{cdiv(colsDst, block.x), cdiv(rows, block.y)};
  sym_quantize_f16_i4_kernel<<<grid, block>>>(x, scale, rows, colsSrc, colsDst, q);
}

__global__ void
sym_dequantize_i32_f16_kernel(const int32_t *__restrict__ q,
                              const half *__restrict__ scale_row,
                              const half *__restrict__ scale_col, uint32_t rows,
                              uint32_t cols, half *__restrict__ x) {
  uint32_t row = threadIdx.y + blockIdx.y * blockDim.y;
  uint32_t col = threadIdx.x + blockIdx.x * blockDim.x;

  if (col >= cols || row >= rows) {
    return;
  }

  half xElement = int_to_half(q[col + row * cols]);
  x[col + row * cols] = scale_row[row] * scale_col[col] * xElement;
}

void sym_dequant_host(const int32_t *q, const half *scale_row,
                      const half *scale_col, uint32_t rows, uint32_t cols,
                      half *x) {
  dim3 block{std::min<uint32_t>(cols, 16), std::min<uint32_t>(rows, 16)};
  dim3 grid{cdiv(cols, block.x), cdiv(rows, block.y)};
  sym_dequantize_i32_f16_kernel<<<grid, block>>>(q, scale_row, scale_col, rows, cols, x);
}

__global__ void
sym_dequantize_col_only_i4_f16_kernel(const int8_t *__restrict__ q,
                                       const half *__restrict__ scale_col, 
                                       uint32_t rowsSrc, uint32_t rowsDst,
                                       uint32_t cols, half *__restrict__ x) {
  uint32_t rowSrc = threadIdx.y + blockIdx.y * blockDim.y;
  uint32_t col = threadIdx.x + blockIdx.x * blockDim.x;

  if (col >= cols || rowSrc * kElementsPerVector >= rowsDst) {
    return;
  }

  Int4Storage storage;
  memset(&storage, 0, sizeof(storage));
  uint32_t id = col + rowSrc * kElementsPerVector * cols;
  int32_t qval = q[id];

#pragma unroll
  for (int i = 0; i < kElementsPerVector; ++i) {
    bool safe = (rowSrc * kElementsPerVector + i) < rowsDst;
    if (safe) {
      // load the 4bit value
      // split bit: get the i-th 4bit value
      qval = (qval >> (i * 4)) & 0xf;
      x[col + (rowSrc * kElementsPerVector + i) * cols] = scale_col[col] * int_to_half(qval);
    }
  }
}

void sym_dequant_col_only_host(const int8_t *q, const half *scale_col, uint32_t rowsSrc, uint32_t rowsDst,
                               uint32_t cols, half *x) {
  dim3 block{std::min<uint32_t>(cols, 16), std::min<uint32_t>(rowsSrc, 16)};
  dim3 grid{cdiv(cols, block.x), cdiv(rowsSrc, block.y)};
  sym_dequantize_col_only_i4_f16_kernel<<<grid, block>>>(q, scale_col, rowsSrc, rowsDst, cols, x);
}

__global__ void
sym_dequantize_row_only_i4_f16_kernel(const int8_t *__restrict__ q,
                                       const half *__restrict__ scale_row, 
                                       uint32_t rows, uint32_t colsSrc,
                                       uint32_t colsDst, half *__restrict__ x) {
  uint32_t row = threadIdx.y + blockIdx.y * blockDim.y;
  uint32_t colSrc = threadIdx.x + blockIdx.x * blockDim.x;

  if (row >= rows || colSrc * kElementsPerVector >= colsDst) {
    return;
  }

  Int4Storage storage;
  memset(&storage, 0, sizeof(storage));
  uint32_t id = colSrc + row * colsSrc;
  uint32_t src_qval = q[id];
  uint32_t qval = 0;

#pragma unroll
  for (int i = 0; i < kElementsPerVector; ++i) {
    bool safe = (colSrc * kElementsPerVector + i) < colsDst;
    if (safe) {
      // load the 4bit value
      qval = src_qval & 0xf;
      src_qval >>= 4;
      x[colSrc * kElementsPerVector + i + row * colsDst] = scale_row[row] * int_to_half(qval);
    }
  }
}

void sym_dequant_row_only_host(const int8_t *q, const half *scale_row, uint32_t rows, uint32_t colsSrc,
                               uint32_t colsDst, half *x) {
  dim3 block{std::min<uint32_t>(colsSrc, 16), std::min<uint32_t>(rows, 16)};
  dim3 grid{cdiv(colsSrc, block.x), cdiv(rows, block.y)};
  sym_dequantize_row_only_i4_f16_kernel<<<grid, block>>>(q, scale_row, rows, colsSrc, colsDst, x);
}

// __global__ void
// sym_dequantize_quantize_i4_f16_i4_kernel(const int32_t *__restrict__ q,
//                                          const half *__restrict__ scale_row,
//                                          const half *__restrict__ scale_col, 
//                                          uint32_t rowsSrc, uint32_t rowsDst, // rowsSrc is small, rowsDst is big
//                                          uint32_t colsSrc, uint32_t colsDst // colsSrc is big, colsDst is small
// ) {
//   uint32_t rowSrc = threadIdx.y + blockIdx.y * blockDim.y;
//   uint32_t colDst = threadIdx.x + blockIdx.x * blockDim.x;

//   if (colDst * kElementsPerVector >= colsSrc || rowSrc * kElementsPerVector >= rowsDst) {
//     return;
//   }

//   // first, row-wise dequantize
//   Int4Storage storage;
//   memset(&storage, 0, sizeof(storage));

//   uint32_t id = colDst * kElementsPerVector + rowSrc * kElementsPerVector * colsSrc;

// #pragma unroll
//   for (int i = 0; i < kElementsPerVector; ++i) {
//     bool safe = (colDst * kElementsPerVector + i) < colsSrc;
//     if (safe) {
//       // load the 4bit value
//       int qval = Int4Subbyte{reinterpret_cast<cutlass::int4b_t *>(&q[id]), i}.get();
//       storage.set(i, qval);
//     }
//   }

//   // second, col-wise dequantize
//   for (int i = 0; i < kElementsPerVector; ++i) {
//     bool safe = (rowSrc * kElementsPerVector + i) < rowsDst;
//     if (safe) {
//       half xElement = int_to_half(storage.get(i));
//       x[colDst + (rowSrc * kElementsPerVector + i) * colsDst] = scale_row[rowSrc] * scale_col[colDst] * xElement;
//     }
//   }
// }

// void sym_dequant_quant_host(const int32_t *q, const half *scale_row, const half *scale_col, uint32_t rowsSrc, uint32_t rowsDst,
//                             uint32_t colsSrc, uint32_t colsDst, half *x) {
//   dim3 block{std::min<uint32_t>(colsDst, 16), std::min<uint32_t>(rowsSrc, 16)};
//   dim3 grid{cdiv(colsDst, block.x), cdiv(rowsSrc, block.y)};
//   sym_dequantize_quantize_i4_f16_i4_kernel<<<grid, block>>>(q, scale_row, scale_col, rowsSrc, rowsDst, colsSrc, colsDst);
// }

