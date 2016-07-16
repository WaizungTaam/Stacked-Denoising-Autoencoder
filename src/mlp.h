/*
Copyright 2016 WaizungTaam.  All rights reserved.

License:       Apache License 2.0
Email:         waizungtaam@gmail.com
Creation time: 2016-07-16
Last modified: 2016-07-16

*/

#ifndef MLP_H
#define MLP_H

#include <vector>
#include "./math/matrix.h"

class MLP {
public:
  MLP() = default;
  MLP(int, int, int);
  MLP(const std::vector<int> &);
  MLP(const std::initializer_list<int> &);
  MLP(const MLP &) = default;
  MLP(MLP &&) = default;
  MLP & operator=(const MLP &) = default;
  MLP & operator=(MLP &&) = default;
  ~MLP() = default;
  void train(const Matrix &, const Matrix &, double, 
             double momentum = 1.0);
  Matrix predict(const Matrix &);
  Matrix & share_weight(int);  // specifically for sdA
  Matrix & share_w_bias(int);  // specifically for sdA
private:
  int num_samples;
  std::vector<Matrix> weights;
  std::vector<Matrix> ws_bias;
  std::vector<Matrix> delta_weights;
  std::vector<Matrix> delta_ws_bias;
  std::vector<Matrix> data_forward;
  std::vector<Matrix> local_fields;
  std::vector<Matrix> local_gradients;
  Matrix output(const Matrix &);
  void forward(const Matrix &);
  void backward(const Matrix &);
  void update(double, double momentum = 1.0);
};

#endif  // mlp.h