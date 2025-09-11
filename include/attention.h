#pragma once
#include "matrix_utils.h"

Matrix scaled_dot_product_attention(
    const Matrix& Q,
    const Matrix& K,
    const Matrix& V
);

Matrix multi_head_attention(
    const Matrix& X,
    const Matrix& W_Q1, const Matrix& W_K1, const Matrix& W_V1,
    const Matrix& W_Q2, const Matrix& W_K2, const Matrix& W_V2,
    const Matrix& W_O
);

