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
  auto names = load_names("data/names.txt");
  Dataset ds = build_dataset(names, 3);

  const i32 vocab = 27, embed_dim = 2, block = 3;
  const i32 n_in = embed_dim * block; // 6
  const i32 H = 100;

  auto C = std::make_shared<Value>(Matrix(embed_dim, vocab, true)); // 2x27
  auto W1 = std::make_shared<Value>(Matrix(H, n_in, true));         // 100x6
  auto b1 = std::make_shared<Value>(Matrix(H, 1, false));           // 100x1
  auto W2 = std::make_shared<Value>(Matrix(vocab, H, true));        // 27x100
  auto b2 = std::make_shared<Value>(Matrix(vocab, 1, false));       // 27x1

  std::vector<std::shared_ptr<Value>> params = {C, W1, b1, W2, b2};

  auto x = embed_context(C, ds.X[0], vocab); // 6x1

  auto h = tanh_(add(matmul(W1, x), b1));
  auto logits = add(matmul(W2, h), b2);

  Matrix target = one_hot(ds.Y[0], vocab);

  auto loss = cross_entropy(logits, target);

  std::cout << "loss: " << loss->data.at(0, 0) << "  (expect ~3.3)\n";

  return 0;
}
