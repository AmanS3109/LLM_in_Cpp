#pragma once
#include "matrix_utils.h"

Matrix layer_norm(
    const Matrix& X,
    const std::vector<float>& gamma,
    const std::vector<float>& beta,
    float eps = 1e-5
);

