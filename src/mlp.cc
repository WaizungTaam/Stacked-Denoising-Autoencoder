/*
Copyright 2016 WaizungTaam.  All rights reserved.

License:       Apache License 2.0
Email:         waizungtaam@gmail.com
Creation time: 2016-07-16
Last modified: 2016-07-16

*/

#include <vector>
#include "mlp.h"
#include "./math/matrix.h"
#include "./math/utils.h"

MLP::MLP(int dim_inut, int dim_hidden, int dim_output) {
  weights.push_back(Matrix(dim_inut, dim_hidden, "uniform", -1.0, 1.0));
  weights.push_back(Matrix(dim_hidden, dim_output, "uniform", -1.0, 1.0));
  ws_bias.push_back(Matrix(1, dim_hidden, "uniform", -1.0, 1.0));
  ws_bias.push_back(Matrix(1, dim_output, "uniform", -1.0, 1.0));
  delta_weights = weights;
  delta_ws_bias = ws_bias;
  delta_weights[0] = 0; 
  delta_weights[1] = 0;
  delta_ws_bias[0] = 0;
  delta_ws_bias[1] = 0;
}
MLP::MLP(const std::vector<int> & dim_layers) {
  int idx;
  for (idx = 0; idx < dim_layers.size() - 1; ++idx) {
    weights.push_back(Matrix(dim_layers[idx], dim_layers[idx + 1], 
                             "uniform", -1.0, 1.0));
    ws_bias.push_back(Matrix(1, dim_layers[idx + 1],
                             "uniform", -1.0, 1.0));
  }
  delta_weights = weights;
  delta_ws_bias = ws_bias;
  for (idx = 0; idx < dim_layers.size() - 1; ++idx) {
    delta_weights[idx] = 0;
    delta_ws_bias[idx] = 0;
  }
}
MLP::MLP(const std::initializer_list<int> & dim_layers) : 
  MLP(std::vector<int>(dim_layers)){
}
void MLP::train(const Matrix & data_in, const Matrix & data_out, 
                double learnig_rate, double momentum) {
  num_samples = data_in.shape()[0];
  data_forward.resize(weights.size());
  local_fields.resize(weights.size());
  local_gradients.resize(weights.size());
  forward(data_in);
  backward(data_out);
  update(learnig_rate, momentum);
}
Matrix MLP::predict(const Matrix & data_in) {
  return output(data_in) >= 0.5;
}
Matrix MLP::output(const Matrix & mat_in) {
  int idx;
  Matrix mat_out = mat_in;
  for (idx = 0; idx < weights.size(); ++idx) {
    mat_out = nn::logistic(mat_out * weights[idx] + ws_bias[idx]);
  }
  return mat_out;  
}
void MLP::forward(const Matrix & mat_in) {
  int idx;
  Matrix mat_forward = mat_in;
  for (idx = 0; idx < weights.size(); ++idx) {
    data_forward[idx] = mat_forward;
    local_fields[idx] = data_forward[idx] * weights[idx] + ws_bias[idx];
    mat_forward = nn::logistic(local_fields[idx]);
  }
}
void MLP::backward(const Matrix & mat_out) {
  int num_layers = weights.size(), idx;
  Matrix mat_pred = nn::logistic(local_fields[num_layers - 1]);
  local_gradients[num_layers - 1] = (mat_out - mat_pred).cross(
    nn::d_logistic(local_fields[num_layers - 1]));
  for (idx = num_layers - 2; idx >= 0; --idx) {
    local_gradients[idx] = nn::d_logistic(local_fields[idx]).cross(
      local_gradients[idx + 1] * weights[idx + 1].T());
  }
  local_fields.clear();
}
void MLP::update(double learnig_rate, double momentum) {
  int idx;
  Matrix bias(num_samples, 1, 1.0);
  for (idx = 0; idx < weights.size(); ++idx) {
    delta_weights[idx] = momentum * delta_weights[idx] + 
      learnig_rate * data_forward[idx].T() * local_gradients[idx];
    delta_ws_bias[idx] = momentum * delta_ws_bias[idx] +
      learnig_rate * bias.T() * local_gradients[idx];
    weights[idx] += delta_weights[idx];
    ws_bias[idx] += delta_ws_bias[idx];
  }
  data_forward.clear();
  local_gradients.clear();
}
Matrix & MLP::share_weight(int idx_layer) {
  return weights.at(idx_layer);
}
Matrix & MLP::share_w_bias(int idx_layer) {
  return ws_bias.at(idx_layer);
}