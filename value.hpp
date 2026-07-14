#include "matrix.hpp"
#include <functional>
#include <memory>
#include <vector>

struct Value {
  Matrix data, grad;

  std::vector<std::shared_ptr<Value>> children;
  std::function<void()> _backward = []() {};

  Value(Matrix data_in) : data(data_in), grad(data_in.rows, data_in.cols, 0) {}
};
