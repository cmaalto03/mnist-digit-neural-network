#include <vector>
#include <cmath>
#include <random>

class SigmoidNeuron {
 public:
    SigmoidNeuron(int numInputs);
    
    
 private:
    std::vector<double> weights; 
    double bias;
    double lastZ;
    double lastOutput;
};

SigmoidNeuron::SigmoidNeuron(int numInputs) 
    : weights(numInputs), bias(0.0), lastZ(0.0), lastOutput(0.0) {
    // Initialize weights with small random values
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(-1.0, 1.0);
    
    for (auto& w : weights) {
        w = dis(gen) * 0.5;
    }
}

