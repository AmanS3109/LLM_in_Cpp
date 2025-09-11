#include "layernorm.h"
#include <cmath>
#include<vector>
#include<random>

using std::vector;

Matrix layer_norm(
    const Matrix& X,
    const std::vector<float>& gamma,
    const std::vector<float>& beta,
    float eps
) {
    int m = X.size(), n = X[0].size();
    Matrix out(m, std::vector<float>(n, 0.0f));

    for (int i = 0; i < m; i++) {
        float mean = 0.0f;
        for (int j = 0; j < n; j++) mean += X[i][j];
        mean /= n;

        float var = 0.0f;
        for (int j = 0; j < n; j++) var += (X[i][j] - mean) * (X[i][j] - mean);
        var /= n;

        for (int j = 0; j < n; j++) {
            float norm = (X[i][j] - mean) / std::sqrt(var + eps);
            out[i][j] = gamma[j] * norm + beta[j];
        }
    }

    return out;
}

