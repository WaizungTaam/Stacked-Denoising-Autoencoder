/*
Copyright 2016 WaizungTaam.  All rights reserved.

License:       Apache License 2.0
Email:         waizungtaam@gmail.com
Creation time: 2016-07-15
Last modified: 2016-07-16

*/

#ifndef DA_H
#define DA_H

#include "./math/matrix.h"

class dA {
public:
  dA() = default;
  // dA(int, int);  // specificaly for sdA
  dA(Matrix &, const Matrix &, Matrix &);  // specificaly for sdA, const is omitted in order to share the weight
  dA(const dA &) = default;
  dA(dA &&) = default;
  dA & operator=(const dA &) = default;
  dA & operator=(dA &&) = default;
  ~dA() = default;
  Matrix reconstruct(const Matrix &);
  void train(const Matrix &, double, double);
  Matrix vis_to_hid(const Matrix &);  // specificaly for sdA, turn private to be public
private:
  Matrix & weight;  // specificaly for sdA
  Matrix w_bias_vis;
  Matrix & w_bias_hid;  // specificaly for sdA
  // Matrix vis_to_hid(const Matrix &);
  Matrix hid_to_vis(const Matrix &);
  Matrix corrupt(const Matrix &, double);
};

#endif  // da.h