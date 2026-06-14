#pragma once
#include <Eigen/Dense>

using Eigen::VectorXd;

inline double sigmoid(double z) { return 1 / (1 + std::exp(-z)); }
inline double sigmoid_prime(double z) { return sigmoid(z) * (1 - sigmoid(z)); }

inline double cost(const VectorXd& output, const VectorXd& expected_output) {  // quadratic cost
  return 0.5 * (expected_output - output).squaredNorm();
}