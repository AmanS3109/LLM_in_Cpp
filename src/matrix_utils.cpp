#include<random>
#include<vector>
#include "matrix_utils.h"
#include <iostream>
#include <iomanip>
#include <cmath>
#include <algorithm>
using std::vector;

Matrix matmul(const Matrix& A, const Matrix& B) {
    int m = A.size();
    int n = (m ? A[0].size() : 0);
    int p = (B.size() ? B[0].size() : 0);

    Matrix C(m, std::vector<float>(p, 0.0f));
    for(int i=0;i<m;i++)
        for(int k=0;k<n;k++)
            for(int j=0;j<p;j++)
                C[i][j] += A[i][k] * B[k][j];
    return C;
}

Matrix transpose(const Matrix& M) {
    int m = M.size(), n = M[0].size();
    Matrix T(n, std::vector<float>(m));
    for(int i=0;i<n;i++)
        for(int j=0;j<m;j++)
            T[i][j] = M[j][i];
    return T;
}

void scale_matrix(Matrix& M, float factor) {
    for(auto& row:M)
        for(auto& v:row)
            v /= factor;
}

void apply_softmax(Matrix& M) {
    for(auto& row:M) {
        float max_val = *max_element(row.begin(), row.end());
        float sum=0.0f;
        for(auto& v:row) { v = exp(v-max_val); sum+=v; }
        for(auto& v:row) v /= sum;
    }
}

void print_matrix(const Matrix& M, const std::string& name) {
    std::cout << name << " (" << M.size() << "x" << (M.empty()?0:M[0].size()) << "):\n";
    for(auto& row:M) {
        for(auto& v:row) std::cout<<std::setw(8)<<std::setprecision(4)<<v<<" ";
        std::cout<<"\n";
    }
    std::cout<<"\n";
}
Matrix random_matrix(int rows, int cols, float min_val, float max_val) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dist(min_val, max_val);

    Matrix mat(rows, vector<float>(cols));
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            mat[i][j] = dist(gen);
        }
    }
    return mat;
}
