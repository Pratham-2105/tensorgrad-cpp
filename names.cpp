#include "makemore.hpp"
#include "value.hpp"
#include <iostream>
#include <memory>
#include <vector>

Matrix one_hot(i32 idx, i32 vocab) {
  Matrix v(vocab, 1, false);
  v.at(idx, 0) = 1.0;

  return v;
}

std::shared_ptr<Value> embed_context(std::shared_ptr<Value> C,
                                     const std::vector<i32> &context,
                                     i32 vocab) {

  std::vector<std::shared_ptr<Value>> lookups;

  for (i32 idx : context) {
    // oh = one_hot;
    auto oh = std::make_shared<Value>(one_hot(idx, vocab));

    lookups.push_back(matmul(C, oh));
  }

  return concat(lookups);
}

int main() {
  // tiny embedding table: 2 x 27 (embed_dim x vocab), random init
  Matrix Cm(2, 27, true);
  auto C = std::make_shared<Value>(Cm);

  // embed context [0, 0, 5]
  auto x = embed_context(C, {0, 0, 5}, 27);

  std::cout << "x shape: " << x->data.rows << "x" << x->data.cols
            << " (expect 6x1)\n";

  // rows 4-5 of x are the 3rd context index's lookup = column 5 of C.
  std::cout << "x[4], x[5]: " << x->data.at(4, 0) << " " << x->data.at(5, 0)
            << "\n";
  std::cout << "C col 5:    " << C->data.at(0, 5) << " " << C->data.at(1, 5)
            << "\n";
  std::cout << "(those two lines should match)\n";
}
