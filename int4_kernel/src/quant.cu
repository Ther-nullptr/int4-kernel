#include "quant.h"
#include <stdio.h>

template <typename T> __device__ __half int_to_half(T value) {
  return __int2half_rn(static_cast<int>(value));
}

#define SMEM_SIZE_1 16
#define SMEM_SIZE_2 16
#define SMEM_SIZE 32

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

  uint32_t id = col + rowSrc * cols;
  uint32_t src_qval = q[id];
  uint32_t qval = 0;

#pragma unroll
  for (int i = 0; i < kElementsPerVector; ++i) {
    bool safe = (rowSrc * kElementsPerVector + i) < rowsDst;
    if (safe) {
      // load the 4bit value
      qval = src_qval & 0xf;
      src_qval >>= 4;
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

__global__ void
sym_dequantize_quantize_i4_f16_i4_kernel(const int8_t *__restrict__ q_in,
                                         int8_t *__restrict__ q_out,
                                         const half *__restrict__ scale_row,
                                         const half *__restrict__ scale_col, 
                                         uint32_t rowsSrc, uint32_t rowsDst, // rowsSrc is small, rowsDst is big
                                         uint32_t colsSrc, uint32_t colsDst // colsSrc is small, colsDst is big
) {
  uint32_t rowSrc = threadIdx.y + blockIdx.y * blockDim.y;
  uint32_t colSrc = threadIdx.x + blockIdx.x * blockDim.x;

  __shared__ half buffer[SMEM_SIZE_1][SMEM_SIZE_2 + 1];

  // first, row-wise dequantize, and move to shared memory
  uint32_t id = colSrc + rowSrc * colsSrc;
  uint32_t src_qval = q_in[id];
  int32_t qval = 0;

// #pragma unroll
  for (int i = 0; i < kElementsPerVector; ++i) {
    // load the 4bit value
    qval = src_qval & 0xf;
    src_qval >>= 4;
    buffer[rowSrc % SMEM_SIZE_1][(colSrc * kElementsPerVector + i) % SMEM_SIZE_2] = scale_row[rowSrc] * int_to_half(qval);
  }
  
  __syncthreads();

  // second, col-wise quantize
  uint8_t storage;
  memset(&storage, 0, sizeof(storage));

  uint32_t rowSrc_2 = threadIdx.y % (blockDim.y / kElementsPerVector) + blockIdx.y * (blockDim.y / kElementsPerVector);
  uint32_t colSrc_2 = threadIdx.x + ((threadIdx.y * kElementsPerVector) / blockDim.y) * blockDim.x + blockIdx.x * blockDim.x * kElementsPerVector;

// #pragma unroll
  for (int i = 0; i < kElementsPerVector; ++i) {
    half data = buffer[(rowSrc_2 * kElementsPerVector + i) % SMEM_SIZE_1][colSrc_2 % SMEM_SIZE_2];
    data = __hdiv(data, scale_col[colSrc_2]);
    int qval = clamp(__half2int_rn(data), qmin, qmax);
    // Int4Subbyte{reinterpret_cast<cutlass::int4b_t *>(&storage), i}.set(qval);
    storage |= (qval & 0xf) << (i * 4);
  }

  // uint32_t rowSrc_t = threadIdx.x + (threadIdx.y * kElementsPerVector) / blockDim.y * blockDim.x + blockIdx.x * blockDim.x * kElementsPerVector;
  // uint32_t colSrc_t = threadIdx.y % (blockDim.y / kElementsPerVector) + blockIdx.y * (blockDim.y / kElementsPerVector);

  uint32_t id_transpose = rowSrc_2 + colSrc_2 * rowsSrc;
  q_out[id_transpose] = storage;
}

void sym_dequantize_quantize_host(const int8_t *q_in, int8_t *q_out,
                                  const half *scale_row, const half *scale_col, 
                                  uint32_t rowsSrc, uint32_t rowsDst, uint32_t colsSrc, uint32_t colsDst) {
  dim3 block{SMEM_SIZE_1 / kElementsPerVector, SMEM_SIZE_2};
  dim3 grid{cdiv(colsSrc, block.x), cdiv(rowsDst, block.y)};
  // print the size of block and grid
  sym_dequantize_quantize_i4_f16_i4_kernel<<<grid, block>>>(q_in, q_out, scale_row, scale_col, rowsSrc, rowsDst, colsSrc, colsDst);
}
