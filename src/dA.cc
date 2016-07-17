/*
Copyright 2016 WaizungTaam.  All rights reserved.

License:       Apache License 2.0
Email:         waizungtaam@gmail.com
Creation time: 2016-07-15
Last modified: 2016-07-16

*/

#include "dA.h"
#include "./math/matrix.h"
#include "./math/utils.h"
/*
dA::dA(int num_vis, int num_hid) :
  weight(Matrix(num_vis, num_hid, "uniform", -1.0, 1.0)),
  w_bias_vis(Matrix(1, num_vis, "uniform", -1.0, 1.0)),
  w_bias_hid(Matrix(1, num_hid, "uniform", -1.0, 1.0)) {
}*/
dA::dA(Matrix & weight_init, 
       const Matrix & w_bias_vis_init, 
       Matrix & w_bias_hid_init) :
  weight(weight_init), 
  w_bias_vis(w_bias_vis_init),
  w_bias_hid(w_bias_hid_init) {
}
Matrix dA::reconstruct(const Matrix & mat_vis) {
  return hid_to_vis(vis_to_hid(mat_vis));
}
void dA::train(const Matrix & mat_vis,
               double learning_rate, double corruption_level) {
  Matrix mat_vis_cor = corrupt(mat_vis, corruption_level);
  Matrix mat_hid = vis_to_hid(mat_vis_cor);
  Matrix mat_vis_re = hid_to_vis(mat_hid);

  Matrix err_vis = mat_vis - mat_vis_re;
  Matrix err_hid = (err_vis * weight).cross(
    nn::d_logistic(mat_vis_cor * weight + w_bias_hid));
  Matrix err_vis_re = mat_vis_cor.T() * err_hid + err_vis.T() * mat_hid;

  Matrix bias(mat_vis.shape()[0], 1, 1.0);

  Matrix delta_weight = err_vis_re;
  Matrix delta_w_bias_vis = bias.T() * err_vis;
  Matrix delta_w_bias_hid = bias.T() * err_hid;

  weight += learning_rate * delta_weight;
  w_bias_vis += learning_rate * delta_w_bias_vis;
  w_bias_hid += learning_rate * delta_w_bias_hid;
}
Matrix dA::vis_to_hid(const Matrix & mat_vis) {
  return nn::logistic(mat_vis * weight + w_bias_hid);
}
Matrix dA::hid_to_vis(const Matrix & mat_hid) {
  return nn::logistic(mat_hid * weight.T() + w_bias_vis);
}
Matrix dA::corrupt(const Matrix & mat_in, double corruption_level) {
  Matrix mass(mat_in.shape(), "binomial", 1.0, 1 - corruption_level);
  return mat_in.cross(mass);
}