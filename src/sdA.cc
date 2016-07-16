/*
Copyright 2016 WaizungTaam.  All rights reserved.

License:       Apache License 2.0
Email:         waizungtaam@gmail.com
Creation time: 2016-07-16
Last modified: 2016-07-16

*/

#include <vector>
#include "sdA.h"
#include "dA.h"
#include "mlp.h"
#include "./math/matrix.h"

sdA::sdA(const std::vector<int> & dim_layers) {
  mlp = MLP(dim_layers);
  int idx_layer;
  for (idx_layer = 0; idx_layer < dim_layers.size() - 2; ++idx_layer) {
    dAs.push_back(dA(mlp.share_weight(idx_layer),
                     Matrix(1, dim_layers[idx_layer], "uniform", -1.0, 1.0),
                     mlp.share_w_bias(idx_layer)));
  }
}
sdA::sdA(const std::initializer_list<int> & init_list) :
  sdA(std::vector<int>(init_list)) {
}
void sdA::pre_train(const Matrix & data_in,
                    int num_epochs,
                    double learning_rate, 
                    double corruption_level) {
  int idx_layer, idx_epoch;
  Matrix data_forward = data_in;
  for (idx_layer = 0; idx_layer < dAs.size(); ++idx_layer) {
    for (idx_epoch = 0; idx_epoch < num_epochs; ++idx_epoch) {
      dAs[idx_layer].train(data_forward, learning_rate, corruption_level);
    }
    data_forward = dAs[idx_layer].vis_to_hid(data_forward);
  }
}
void sdA::fine_tune(const Matrix & data_in, 
                    const Matrix & data_out,
                    double learning_rate,
                    double momentum) {
  mlp.train(data_in, data_out, learning_rate, momentum);
}
Matrix sdA::predict(const Matrix & data_in) {
  return mlp.predict(data_in);
}