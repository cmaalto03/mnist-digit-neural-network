#pragma once

#include <vector>
#include <cmath>

using namespace std;

inline double sigmoid(const double& z) {
    return 1 / (1 + std::exp(-z));
}

inline double sigmoid_prime(const double& z) {
    double s = sigmoid(z);
    return s * (1.0 - s);
}

inline vector<double> digitToOneHotEncoding(int input) {
    vector<double> oneHotEncoding(10, 0.0);
    oneHotEncoding[input] = 1.0;
    return oneHotEncoding;
}