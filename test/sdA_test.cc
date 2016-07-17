/*
Copyright 2016 WaizungTaam.  All rights reserved.

License:       Apache License 2.0
Email:         waizungtaam@gmail.com
Creation time: 2016-07-16
Last modified: 2016-07-16

*/

#include <iostream>
#include <iomanip>
#include "../src/sdA.h"
#include "../src/math/matrix.h"

int main() {
  int num_epochs_pre = 400, num_epochs_fine = 500,
      idx_epoch;
  double lr_pre = 1e-3, lr_fine = 1e-0, cl = 1e-2, mt = 8e-1;
  Matrix x_train = {{0, 0}, {0, 1}, {1, 0}, {1, 1}},
         y_train = {{0}, {1}, {1}, {0}},
         x_test = {{1, 1}, {1, 0}, {0, 0}, {0, 1}, {0, 0}, {1, 0}},
         y_test = {{0}, {1}, {0}, {1}, {0}, {1}};
  sdA model({2, 4, 4, 1});
  Matrix mat_rgs;
  model.pre_train(x_train, num_epochs_pre, lr_pre, cl);
  for (idx_epoch = 0; idx_epoch < num_epochs_fine; ++idx_epoch) {
    model.fine_tune(x_train, y_train, lr_fine, mt);
  }
  std::cout << y_test << std::endl;
  std::cout << model.predict(x_test) << std::endl;
  return 0;
}