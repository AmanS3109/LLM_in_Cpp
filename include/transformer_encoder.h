#pragma once
#include "matrix_utils.h"
#include "attention.h"
#include "feedforward.h"
#include "layernorm.h"

Matrix transformer_block(
    const Matrix& X,
    const Matrix& W_Q1, const Matrix& W_K1, const Matrix& W_V1,
    const Matrix& W_Q2, const Matrix& W_K2, const Matrix& W_V2, const Matrix& W_O,
    const Matrix& W1, const std::vector<float>& b1,
    const Matrix& W2, const std::vector<float>& b2,
    const std::vector<float>& gamma, const std::vector<float>& beta
);


