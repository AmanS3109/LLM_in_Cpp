#include<vector>
#include "matrix_utils.h"
#include "attention.h"
#include "feedforward.h"
#include "layernorm.h"
#include "transformer_encoder.h"

using namespace std;
using std::vector;

int main() {
    // Example input
    Matrix X = {
        {1.0, 2.0, 3.0, 4.0},
        {2.0, 4.0, 6.0, 8.0},
        {1.5, 1.5, 4.0, 7.0}
    };

    // Initialize weights for Multi-Head Attention (head 1)
    Matrix W_Q1 = random_matrix(4, 4);
    Matrix W_K1 = random_matrix(4, 4);
    Matrix W_V1 = random_matrix(4, 4);

    // Initialize weights for Multi-Head Attention (head 2)
    Matrix W_Q2 = random_matrix(4, 4);
    Matrix W_K2 = random_matrix(4, 4);
    Matrix W_V2 = random_matrix(4, 4);

    // Output projection matrix
    Matrix W_O = random_matrix(8, 4);  // since 2 heads * 4 dims = 8

    // Feedforward weights and biases
    Matrix W1 = random_matrix(4, 8);
    vector<float> b1(8, 0.0f);
    Matrix W2 = random_matrix(8, 4);
    vector<float> b2(4, 0.0f);

    // Gamma and Beta for LayerNorm
    vector<float> gamma(4, 1.0f);
    vector<float> beta(4, 0.0f);

    // Run Transformer Encoder
    Matrix out = transformer_block(
        X, 
        W_Q1, W_K1, W_V1,
        W_Q2, W_K2, W_V2, W_O,
        W1, b1, W2, b2,
        gamma, beta
    );

    print_matrix(out, "Transformer Block Output");

    return 0;
}

