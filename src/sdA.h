/*
Copyright 2016 WaizungTaam.  All rights reserved.

License:       Apache License 2.0
Email:         waizungtaam@gmail.com
Creation time: 2016-07-16
Last modified: 2016-07-16

*/

#ifndef SDA_H
#define SDA_H

#include <vector>
#include "dA.h"
#include "mlp.h"
#include "./math/matrix.h"

class sdA {
public:
  sdA() = default;
  sdA(const std::vector<int> &);
  sdA(const std::initializer_list<int> &);
  sdA(const sdA &) = default;
  sdA(sdA &&) = default;
  sdA & operator=(const sdA &) = default;
  sdA & operator=(sdA &&) = default;
  ~sdA() = default;
  void pre_train(const Matrix &, int, double, double);
  void fine_tune(const Matrix &, const Matrix &, double, double);
  Matrix predict(const Matrix &);
private:
  std::vector<dA> dAs;
  MLP mlp;
};

#endif  // sdA.h