#pragma once

#include <common.h>

void sym_quant_host(
    const half *x,
    const half *scale,
    uint32_t rows,
    uint32_t colsSrc,
    uint32_t colsDst,
    Int4Storage *q
);

void sym_dequant_host(
    const int32_t *q,
    const half *scale_row,
    const half *scale_col,
    uint32_t rows,
    uint32_t cols,
    half *x
);

void sym_dequant_col_only_host(
    const int8_t *q,
    const half *scale_col,
    uint32_t rowsSrc,
    uint32_t rowsDst,
    uint32_t cols,
    half *x
);

void sym_dequant_row_only_host(
    const int8_t *q,
    const half *scale_row,
    uint32_t rows,
    uint32_t colsSrc,
    uint32_t colsDst,
    half *x
);