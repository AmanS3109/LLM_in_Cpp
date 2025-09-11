#include "feedforward.h"
#include <algorithm>
#include<vector>
#include<random>
using std::vector;

// ReLU
Matrix relu(const Matrix& X) {
    Matrix out = X;
    for (auto& row : out)
        for (auto& v : row)
            v = std::max(0.0f, v);
    return out;
}

// Add bias
Matrix add_bias(const Matrix& X, const std::vector<float>& b) {
    Matrix out = X;
    for (size_t i = 0; i < X.size(); i++)
        for (size_t j = 0; j < X[0].size(); j++)
            out[i][j] += b[j];
    return out;
}

// Feed Forward Network
Matrix feed_forward(
    const Matrix& X,
    const Matrix& W1, const std::vector<float>& b1,
    const Matrix& W2, const std::vector<float>& b2
) {
    Matrix h = matmul(X, W1);
    h = add_bias(h, b1);
    h = relu(h);

    Matrix out = matmul(h, W2);
    out = add_bias(out, b2);
    return out;
}

