#include "Layer.hpp"
#include <random>
using namespace std;

Layer::Layer(int numNeurons, int numInputs)
    : weights(numNeurons, vector<double>(numInputs)), biases(numNeurons) {
  random_device rd;
  mt19937 gen(rd());
    std::normal_distribution<double> dis(0, 1);


  for (int i = 0; i < numNeurons; ++i) {
    for (int j = 0; j < numInputs; ++j) {
      weights[i][j] = dis(gen);
    }
    biases[i] = dis(gen);
  }
}