/*
Copyright 2016 WaizungTaam.  All rights reserved.

License:       Apache License 2.0
Email:         waizungtaam@gmail.com
Creation time: 2016-07-16
Last modified: 2016-07-16

*/

#include <iostream>
#include "../src/mlp.h"
#include "../src/math/matrix.h"
#include "../src/math/utils.h"

int main() {
  int num_epochs = 2000, idx_epoch;
  double lr = 1e-0, mt = 1e-8;
  Matrix x_train = {{0, 0}, {0, 1}, {1, 0}, {1, 1}},
         y_train = {{0}, {1}, {1}, {0}},
         x_test = {{1, 1}, {1, 0}, {1, 1}, {0, 0}, {0, 1}, {0, 0}},
         y_test = {{0}, {1}, {0}, {0}, {1}, {0}};
  MLP network({2, 4, 1});
  for (idx_epoch = 0; idx_epoch < num_epochs; ++idx_epoch) {
    network.train(x_train, y_train, lr, mt);
  }
  std::cout << y_test << std::endl;
  std::cout << network.predict(x_test) << std::endl;
  return 0;
}