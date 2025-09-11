#pragma once
#include <vector>
#include <string>

using Matrix = std::vector<std::vector<float>>;

Matrix matmul(const Matrix& A, const Matrix& B);
Matrix transpose(const Matrix& M);
void scale_matrix(Matrix& M, float factor);
void apply_softmax(Matrix& M);
void print_matrix(const Matrix& M, const std::string& name);
Matrix random_matrix(int rows, int cols, float min_val = -1.0f, float max_val = 1.0f);
