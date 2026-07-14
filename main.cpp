#include "value.hpp"
#include <iostream>

int main() {
  // a small known matrix: 2x3, filled so you can eyeball the echo
  Matrix m(2, 3, false);
  m.at(0, 0) = 1;
  m.at(0, 1) = 2;
  m.at(0, 2) = 3;
  m.at(1, 0) = 4;
  m.at(1, 1) = 5;
  m.at(1, 2) = 6;

  Value v(m);

  std::cout << "data:\n";
  v.data.print_matrix(); // expect the 2x3 you set above

  std::cout << "grad:\n";
  v.grad.print_matrix(); // expect a 2x3 of all zeros

  return 0;
}
