#include "Network.hpp"
#include "utils.hpp"
#include <cmath>
#include <algorithm>
#include <random>

using namespace std;

Network::Network(const NetworkConfig& config) {
    int prevSize = config.numOfInputs;

    for (int layerSize : config.layerSizesExcludingInputLayer) {
        layers.push_back(Layer(layerSize, prevSize));
        prevSize = layerSize;
    }
}

vector<double> Network::feedforward(const vector<double>& inputData) {
    vector<double> inputs = inputData;

    for (size_t i = 0; i < layers.size(); i++) {
        vector<double> newOutputs;
        size_t numNeurons = layers[i].getBiases().size();

        for (size_t j = 0; j < numNeurons; j++) {
            double z = 0.0;
            size_t numInputs = inputs.size();

            for (size_t k = 0; k < numInputs; k++) {
                z += layers[i].getWeights()[j][k] * inputs[k];
            }

            z += layers[i].getBiases()[j];

            layers[i].pushZ(z);
            double a = sigmoid(z);
            layers[i].pushA(a);
            newOutputs.push_back(a);
        }
        inputs = newOutputs;
    }
    return inputs;
}

void Network::trainEpoch(const vector<TrainingInput>& trainingData, int batchSize, double learningRate) {
    size_t numLayers = layers.size();
    if (numLayers == 0) return;

    // Track array indices to cleanly shuffle your 60k dataset
    vector<size_t> indices(trainingData.size());
    for (size_t i = 0; i < indices.size(); ++i) indices[i] = i;
    
    std::random_device rd;
    std::mt19937 g(rd());
    std::shuffle(indices.begin(), indices.end(), g);

    // Mini-batch gradient accumulators
    vector<vector<double>> batchBiasGrads(numLayers);
    vector<vector<vector<double>>> batchWeightGrads(numLayers);

    auto resetBatchGradients = [&]() {
        for (size_t i = 0; i < numLayers; i++) {
            size_t numNeurons = layers[i].getBiases().size();
            size_t numInputs = layers[i].getWeights()[0].size();

            batchBiasGrads[i].assign(numNeurons, 0.0);
            batchWeightGrads[i].assign(numNeurons, vector<double>(numInputs, 0.0));
        }
    };

    resetBatchGradients();
    int imagesInCurrentBatch = 0;

    for (size_t idx : indices) {
        const auto& input = trainingData[idx].image_data;
        const auto& expected = trainingData[idx].labelOneHotEncoding;

        // 1. Forward Pass
        for (size_t i = 0; i < layers.size(); i++) {
            layers[i].clearCaches();
        }
        vector<double> output = feedforward(input);

        // 2. Backward Pass (BP1 & BP2)
        vector<vector<double>> singleBiasGrads(numLayers);
        vector<vector<vector<double>>> singleWeightGrads(numLayers);

        for (int i = numLayers - 1; i >= 0; i--) {
            Layer& currentLayer = layers[i];
            const vector<double>& zValues = currentLayer.getZ();
            vector<double> prevActivations = (i == 0) ? input : layers[i - 1].getA();
            vector<double> currentDelta;

            if (i == numLayers - 1) {
                // BP1 Matrix Calculus
                currentDelta.reserve(output.size());
                for (size_t j = 0; j < output.size(); j++) {
                    double grad_C = output[j] - expected[j]; 
                    currentDelta.push_back(grad_C * sigmoid_prime(zValues[j]));              
                }
            } else {
                // BP2 Backpropagation Step
                Layer& nextLayer = layers[i + 1];
                const vector<double>& nextDelta = nextLayer.getDelta();
                const vector<vector<double>>& nextWeights = nextLayer.getWeights();
                currentDelta.resize(zValues.size(), 0.0);

                for (size_t j = 0; j < zValues.size(); j++) {
                    double errorSum = 0.0;
                    for (size_t k = 0; k < nextDelta.size(); k++) {
                        errorSum += nextWeights[k][j] * nextDelta[k];
                    }
                    currentDelta[j] = errorSum * sigmoid_prime(zValues[j]);
                }
            }

            currentLayer.setDelta(currentDelta);
            singleBiasGrads[i] = currentDelta;

            // BP4 Weight Gradient Multipliers
            singleWeightGrads[i].resize(currentDelta.size(), vector<double>(prevActivations.size()));
            for (size_t j = 0; j < currentDelta.size(); j++) {
                for (size_t k = 0; k < prevActivations.size(); k++) {
                    singleWeightGrads[i][j][k] = currentDelta[j] * prevActivations[k];
                }
            }
        }

        // Accumulate single image gradients into mini-batch buckets
        for (size_t i = 0; i < numLayers; i++) {
            for (size_t j = 0; j < batchBiasGrads[i].size(); j++) {
                batchBiasGrads[i][j] += singleBiasGrads[i][j];
                for (size_t k = 0; k < batchWeightGrads[i][j].size(); k++) {
                    batchWeightGrads[i][j][k] += singleWeightGrads[i][j][k];
                }
            }
        }

        imagesInCurrentBatch++;

        // 3. Mini-batch complete -> Apply parameters adjustment loop
        if (imagesInCurrentBatch == batchSize) {
            for (size_t i = 0; i < numLayers; i++) {
                auto& layerBiases = layers[i].mutBiases();
                auto& layerWeights = layers[i].mutWeights();

                for (size_t j = 0; j < layerBiases.size(); j++) {
                    layerBiases[j] -= (learningRate / batchSize) * batchBiasGrads[i][j];
                }

                for (size_t j = 0; j < layerWeights.size(); j++) {
                    for (size_t k = 0; k < layerWeights[j].size(); k++) {
                        layerWeights[j][k] -= (learningRate / batchSize) * batchWeightGrads[i][j][k];
                    }
                }
            }
            resetBatchGradients();
            imagesInCurrentBatch = 0;
        }
    }
}

// Deprecated or isolated single execution wrapper for test hooks
void Network::backprop(const vector<double>& input, const vector<double>& expected) {
    // Left empty or fallback mapping to a standalone step if needed elsewhere
}

double Network::getQuadraticCost(const vector<double>& output, const vector<double>& expected) {
    size_t totalSize = output.size();
    double sum = 0;
    for (size_t i = 0; i < totalSize; i++) {
        sum += pow((expected[i] - output[i]), 2);
    }
    return sum / 2;
}