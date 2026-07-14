#pragma once

#include <cstdint>
#include <iostream>
#include <random>
#include <stdexcept>
#include <vector>

// ---- fixed-width type aliases ----------------------------------------------
typedef int8_t i8;
typedef int16_t i16;
typedef int32_t i32;
typedef int64_t i64;

typedef uint8_t u8;
typedef uint16_t u16;
typedef uint32_t u32;
typedef uint64_t u64;

typedef float f32;
typedef double f64;

using Scalar = f64;

struct Matrix {
  i64 rows, cols;
  std::vector<Scalar> matrix; // flat, row-major

  // Construct rows x cols. random=true fills with uniform values in [-0.5,
  // 0.5]; random=false leaves it zero-initialised.
  Matrix(i64 row, i64 col, bool random)
      : rows(row), cols(col), matrix(row * col) {
    if (random) {
      std::mt19937 gen(std::random_device{}());
      std::uniform_real_distribution<Scalar> dist(-0.5, 0.5);

      for (auto &val : matrix) {
        val = dist(gen);
      }
    }
  }

  // Read/write a single element. Bounds-checked.
  Scalar &at(i64 i, i64 j) {
    if (i < 0 || i >= rows) {
      throw std::invalid_argument("ROWS ARE out-of-bounds UPON at()");
    }
    if (j < 0 || j >= cols) {
      throw std::invalid_argument("COLS ARE out-of-bounds UPON at()");
    }

    return matrix[i * cols + j];
  }

  Scalar at(i64 i, i64 j) const {
    if (i < 0 || i >= rows) {
      throw std::invalid_argument("ROWS ARE out-of-bounds UPON at()");
    }
    if (j < 0 || j >= cols) {
      throw std::invalid_argument("COLS ARE out-of-bounds UPON at()");
    }

    return matrix[i * cols + j];
  }

  void print_matrix() const {
    for (i64 i = 0; i < rows; ++i) {
      for (i64 j = 0; j < cols; ++j) {
        std::cout << at(i, j) << " ";
      }
      std::cout << '\n';
    }
    std::cout << '\n';
  }

  // ---- element-wise addition (same dimensions) -----------------------------
  // Used in the forward pass to add the bias: W*input + b.
  Matrix operator+(const Matrix &other_Matrix) const {
    if (rows != other_Matrix.rows || cols != other_Matrix.cols) {
      throw std::invalid_argument("INVALID ROW or COLUMN");
    }

    Matrix new_Matrix(rows, cols, false);

    for (i64 i = 0; i < rows; ++i) {
      for (i64 j = 0; j < cols; ++j) {
        new_Matrix.at(i, j) = at(i, j) + other_Matrix.at(i, j);
      }
    }

    return new_Matrix;
  }

  // ---- element-wise subtraction (same dimensions) --------------------------
  // Used for error (prediction - target) and the weight update.
  Matrix operator-(const Matrix &other_Matrix) const {
    if (rows != other_Matrix.rows || cols != other_Matrix.cols) {
      throw std::invalid_argument("INVALID ROW or COLUMN");
    }

    Matrix new_Matrix(rows, cols, false);

    for (i64 i = 0; i < rows; ++i) {
      for (i64 j = 0; j < cols; ++j) {
        new_Matrix.at(i, j) = at(i, j) - other_Matrix.at(i, j);
      }
    }

    return new_Matrix;
  }

  // ---- scalar multiply (every element x a number) --------------------------
  // Used for the gradient-descent step: weights - learning_rate * gradient.
  Matrix scalar_multiply(f64 scalar_number) const {
    Matrix new_Matrix(rows, cols, false);

    for (i64 i = 0; i < rows; ++i) {
      for (i64 j = 0; j < cols; ++j) {
        new_Matrix.at(i, j) = at(i, j) * scalar_number;
      }
    }

    return new_Matrix;
  }

  // ---- matrix multiplication: (m x n)(n x p) = (m x p) ---------------------
  // THE forward pass. Inner dimensions must match (cols == other.rows).
  // NOTE: this is the dot-product product, NOT element-wise (see hadamard).
  Matrix operator*(const Matrix &other_Matrix) const {
    if (cols != other_Matrix.rows) {
      throw std::invalid_argument(
          "INVALID DIMENSIONS FOR matrix-multiplication");
    }

    Matrix result_matrix(rows, other_Matrix.cols, false);

    for (i64 i = 0; i < rows; ++i) {
      for (i64 j = 0; j < other_Matrix.cols; ++j) {
        Scalar sum = 0;

        for (i64 k = 0; k < other_Matrix.rows; ++k) {
          sum += at(i, k) * other_Matrix.at(k, j);
        }

        result_matrix.at(i, j) = sum;
      }
    }

    return result_matrix;
  }

  // ---- transpose: (m x n) -> (n x p) ---------------------------------------
  // Needed in backprop to push error backward through a layer.
  Matrix transpose() const {
    Matrix result(cols, rows, false);

    for (i64 i = 0; i < rows; ++i) {
      for (i64 j = 0; j < cols; ++j) {
        result.at(j, i) = at(i, j);
      }
    }

    return result;
  }

  // ---- apply: run a Scalar->Scalar function on every element ---------------
  // How activations (sigmoid, ReLU, ...) hit a whole matrix. Templated so the
  // function is resolved at compile time and inlined.
  template <typename F> Matrix apply(F func) const {
    Matrix result(rows, cols, false);

    for (i64 i = 0; i < rows; ++i) {
      for (i64 j = 0; j < cols; ++j) {
        result.at(i, j) = func(at(i, j));
      }
    }

    return result;
  }

  // ---- hadamard: element-wise multiply (same dimensions) -------------------
  // NOT matrix multiply. Used in backprop: error (x) activation-derivative.
  Matrix hadamard(const Matrix &other_Matrix) const {
    if (rows != other_Matrix.rows || cols != other_Matrix.cols) {
      throw std::invalid_argument("ROWS or COLS must be same for hadamard");
    }

    Matrix result(rows, cols, false);

    for (i64 i = 0; i < rows; ++i) {
      for (i64 j = 0; j < cols; ++j) {
        result.at(i, j) = at(i, j) * other_Matrix.at(i, j);
      }
    }

    return result;
  }
};
