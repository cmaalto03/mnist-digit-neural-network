#pragma once
#include <vector>
using namespace std;

class Layer {
 public:
    Layer(int numNeurons, int numInputs);
    
    // Getters returning const references for read-only access
    const vector<vector<double>>& getWeights() const { return weights; }
    const vector<double>& getBiases() const { return biases; }
    const vector<double>& getA() const { return a; }
    const vector<double>& getZ() const { return z; }
    const vector<double>& getDelta() const { return delta; }

    // Non-const references allowing modification during Gradient Descent
    vector<vector<double>>& mutWeights() { return weights; }
    vector<double>& mutBiases() { return biases; }

    // Caching during feedforward
    void pushA(double aV) { a.push_back(aV); }
    void pushZ(double aZ) { z.push_back(aZ); }

    // Setter for the complete delta vector
    void setDelta(const vector<double>& newDelta) { 
        delta = newDelta; 
    }

    // Clear caches out before every feedforward pass
    void clearCaches() {
        a.clear();
        z.clear();
        delta.clear();
    }

 private:
    vector<vector<double>> weights;
    vector<double> biases;

    vector<double> weightDeltas;
    vector<double> biasesDeltas;

    vector<double> delta;
    vector<double> a;
    vector<double> z;
};