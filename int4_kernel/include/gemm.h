#pragma once

#include <common.h>

void matmul_host(
    const Int4Storage *A,
    const Int4Storage *B,
    uint32_t M,
    uint32_t N,
    uint32_t K,
    int32_t *C
);

void matmul_host_int8(
    const int8_t *A,
    const int8_t *B,
    uint32_t M,
    uint32_t N,
    uint32_t K,
    int32_t *C
);