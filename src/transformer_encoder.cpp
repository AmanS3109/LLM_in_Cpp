#include "transformer_encoder.h"

Matrix transformer_block(
    const Matrix& X,
    const Matrix& W_Q1, const Matrix& W_K1, const Matrix& W_V1,
    const Matrix& W_Q2, const Matrix& W_K2, const Matrix& W_V2, const Matrix& W_O,
    const Matrix& W1, const std::vector<float>& b1,
    const Matrix& W2, const std::vector<float>& b2,
    const std::vector<float>& gamma, const std::vector<float>& beta
){
    // Multi-Head Attention + Residual + Norm
    Matrix attn_out = multi_head_attention(X, W_Q1, W_K1, W_V1, W_Q2, W_K2, W_V2, W_O);

    Matrix res1 = X;
    for (size_t i = 0; i < res1.size(); i++)
        for (size_t j = 0; j < res1[0].size(); j++)
            res1[i][j] += attn_out[i][j];

    std::vector<float> gamma1(res1[0].size(), 1.0f);
    std::vector<float> beta1(res1[0].size(), 0.0f);
    Matrix norm1 = layer_norm(res1, gamma1, beta1);

    // Feed-Forward + Residual + Norm
    Matrix ff_out = feed_forward(norm1, W1, b1, W2, b2);

    Matrix res2 = norm1;
    for (size_t i = 0; i < res2.size(); i++)
        for (size_t j = 0; j < res2[0].size(); j++)
            res2[i][j] += ff_out[i][j];

    std::vector<float> gamma2(res2[0].size(), 1.0f);
    std::vector<float> beta2(res2[0].size(), 0.0f);
    Matrix norm2 = layer_norm(res2, gamma2, beta2);

    return norm2;
}

