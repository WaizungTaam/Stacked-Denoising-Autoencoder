/*
Copyright 2016 WaizungTaam.  All rights reserved.

License:       Apache License 2.0
Email:         waizungtaam@gmail.com
Creation time: 2016-07-15
Last modified: 2016-07-15

*/

#include <iostream>
#include <iomanip>
#include "../src/dA.h"
#include "../src/math/matrix.h"
#include "../src/math/utils.h"

int main() {
  int num_epochs = 1000, idx_epoch, 
      num_vis = 8, 
      num_hid_1 = 6, num_hid_2 = 2, num_hid_3 = 6;
  double lr = 1e-3, 
         cl_1 = 1e-2, cl_2 = 1e-2, cl_3 = 1e-8;
  Matrix data_train(100, num_vis, "binomial", 1.0, 0.5),
         data_test(4, num_vis, "binomial", 1.0, 0.5);
  dA encoder_1(num_vis, num_hid_1),
     encoder_2(num_vis, num_hid_2),
     encoder_3(num_vis, num_hid_3);
  for (idx_epoch = 0; idx_epoch < num_epochs; ++idx_epoch) {
    encoder_1.train(data_train, lr, cl_1);
    encoder_2.train(data_train, lr, cl_2);
    encoder_3.train(data_train, lr, cl_3);
    std::cout << idx_epoch << "\t";
    std::cout << std::setprecision(8) << std::setw(10) << std::left << std::setfill('0') 
              << nn::pow(data_train - encoder_1.reconstruct(data_train), 2).sum()
                 / data_train.shape()[0] / data_train.shape()[1] << "\t";
    std::cout << std::setprecision(8) << std::setw(10) << std::left << std::setfill('0') 
              << nn::pow(data_train - encoder_2.reconstruct(data_train), 2).sum() 
                 / data_train.shape()[0] / data_train.shape()[1] << "\t";
    std::cout << std::setprecision(8) << std::setw(10) << std::left << std::setfill('0') 
              << nn::pow(data_train - encoder_3.reconstruct(data_train), 2).sum() 
                 / data_train.shape()[0] / data_train.shape()[1] << "\n";                 
  }
  std::cout << data_test << std::endl;
  std::cout << encoder_1.reconstruct(data_test) << std::endl;
  std::cout << encoder_2.reconstruct(data_test) << std::endl;
  std::cout << encoder_3.reconstruct(data_test) << std::endl;
  return 0;
}