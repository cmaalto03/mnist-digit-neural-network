#pragma once

#include <Eigen/Dense>

using Eigen::MatrixXd;
using Eigen::VectorXd;

class Layer {
 public:
    Layer(int, int);

    VectorXd forward(const VectorXd&);
    void zeroGradients();

    VectorXd getZ() { return z; }
    VectorXd getA() { return a; }

    VectorXd getError() { return error; }
    void setError(const VectorXd& e) { error = e; }

    MatrixXd getWeights() { return weights; }
    VectorXd getBiases() { return biases; }

    void setWeights(const MatrixXd& w) { weights = w; }
    void setBiases(const VectorXd& b) { biases = b; }

    void accumulateBiasGradient(const VectorXd& biasGradient) { biasGradients += biasGradient; }
    void accumulateWeightGradient(const MatrixXd& weightGradient) { weightGradients += weightGradient; }

    MatrixXd getAverageWeightGradient() { return weightGradients / 20; }
    VectorXd getAverageBiasGradient() { return biasGradients / 20; }

  private:
    void setRandomMatrix();
    int curr_num_neurons;
    int prev_num_neurons;

    MatrixXd weights;
    VectorXd biases;
    MatrixXd weightGradients;
    VectorXd biasGradients;

    VectorXd error;
    VectorXd z;
    VectorXd a;
};