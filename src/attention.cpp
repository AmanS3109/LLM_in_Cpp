#include "attention.h"
#include <cmath>
#include<random>
#include<vector>
using std::vector;

// Scaled Dot-Product Attention
Matrix scaled_dot_product_attention(
    const Matrix& Q, const Matrix& K, const Matrix& V
) {
    Matrix K_T = transpose(K);
    Matrix scores = matmul(Q, K_T);

    float scale = std::sqrt((float)K[0].size());
    scale_matrix(scores, scale);

    apply_softmax(scores);
    return matmul(scores, V);
}

// Multi-Head Attention (2 heads example)
Matrix multi_head_attention(
    const Matrix& X,
    const Matrix& W_Q1, const Matrix& W_K1, const Matrix& W_V1,
    const Matrix& W_Q2, const Matrix& W_K2, const Matrix& W_V2,
    const Matrix& W_O
) {
    Matrix Q1 = matmul(X, W_Q1);
    Matrix K1 = matmul(X, W_K1);
    Matrix V1 = matmul(X, W_V1);

    Matrix head1 = scaled_dot_product_attention(Q1, K1, V1);

    Matrix Q2 = matmul(X, W_Q2);
    Matrix K2 = matmul(X, W_K2);
    Matrix V2 = matmul(X, W_V2);

    Matrix head2 = scaled_dot_product_attention(Q2, K2, V2);

    Matrix concat(X.size(), std::vector<float>(head1[0].size() + head2[0].size()));
    for (size_t i = 0; i < X.size(); i++) {
        for (size_t j = 0; j < head1[0].size(); j++) concat[i][j] = head1[i][j];
        for (size_t j = 0; j < head2[0].size(); j++) concat[i][j + head1[0].size()] = head2[i][j];
    }

    return matmul(concat, W_O);
}

