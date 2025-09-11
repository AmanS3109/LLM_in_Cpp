#pragma once
#include "matrix_utils.h"

Matrix relu(const Matrix& X);
Matrix add_bias(const Matrix& X, const std::vector<float>& b);
Matrix feed_forward(
    const Matrix& X,
    const Matrix& W1, const std::vector<float>& b1,
    const Matrix& W2, const std::vector<float>& b2
);

