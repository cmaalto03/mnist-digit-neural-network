#include <vector>
#include <cmath>

class SigmoidNeuron {
 public:
    SigmoidNeuron(int numInputs): weights(numInputs), bias(0.0) {}
    double sigmoid(const std::vector<double>& inputs);
 private:

    std::vector<double> weights; 
    double bias;
};

double SigmoidNeuron::sigmoid(const std::vector<double>& inputs) {
    double z = 0.0;

    for(size_t i = 0; i < weights.size(); i++) {
        z += weights[i] * inputs[i];
    }

    z += bias;

    return 1 / (1 + std::exp(-z));
}