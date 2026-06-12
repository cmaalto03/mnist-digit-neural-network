#pragma once

#include <vector>
#include <cmath>
#include <random>

#include "Layer.hpp"
#include "TrainingInput.hpp"

using namespace std;

struct NetworkConfig {
    int numOfInputs;
    vector<int> layerSizesExcludingInputLayer;
    double learningRate = 0;
};

class Network {
 public:
    Network(const NetworkConfig& config);
    double getQuadraticCost(const vector<double>&, const vector<double>&);
    vector<double> feedforward(const vector<double>& inputData);
    void backprop(const vector<double>& input, const vector<double>& expected);
    
    // Added for Mini-Batch Stochastic Gradient Descent over full epoch
    void trainEpoch(const vector<TrainingInput>& trainingData, int batchSize, double learningRate);

 private:
    vector<Layer> layers;
};