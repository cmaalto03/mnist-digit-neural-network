#include "Network.hpp"
#include "mnist_reader.hpp"
#include <iostream>
#include <vector>
#include <algorithm>

const int NUM_INPUT_NEURONS = 784; 
const int NUM_HIDDEN_NEURONS = 30; // Increased to 30 for better classification accuracy
const int NUM_OUTPUT_NEURONS = 10;
const int NUM_TRAINING_INPUTS = 60000;

using namespace std;

double evaluateAccuracy(Network* net, const vector<TrainingInput>& data) {
    int correctPredictions = 0;
    for (const auto& sample : data) {
        vector<double> output = net->feedforward(sample.image_data);
        int guess = std::distance(output.begin(), std::max_element(output.begin(), output.end()));
      
    }
    return (double)correctPredictions / data.size() * 100.0;
}

int main() {
    vector<TrainingInput> trainingData;
    trainingData.resize(NUM_TRAINING_INPUTS); 

    cout << "Loading MNIST dataset..." << endl;
    readMnistLabels("data/train-labels-idx1-ubyte", trainingData);
    readMnistImages("data/train-images-idx3-ubyte", trainingData);
    cout << "Dataset loaded successfully.\n" << endl;

    NetworkConfig config{
        NUM_INPUT_NEURONS,
        {NUM_HIDDEN_NEURONS, NUM_OUTPUT_NEURONS},
        1
    };

    Network* network = new Network(config);

    cout << "Evaluating baseline network performance..." << endl;
    double initialAccuracy = evaluateAccuracy(network, trainingData);
    cout << "Initial Accuracy on Training Data: " << initialAccuracy << "%" << endl;
    cout << "------------------------------------------------" << endl;

    int batchSize = 10;
    double learningRate = 3.0; // Optimized parameter for sigmoid network configuration
    
    cout << "Training 1 full Epoch (60,000 images, Mini-Batch Size: " << batchSize << ")..." << endl;
    network->trainEpoch(trainingData, batchSize, learningRate);
    cout << "Epoch complete!" << endl;
    cout << "------------------------------------------------" << endl;

    cout << "Evaluating optimized network performance..." << endl;
    double finalAccuracy = evaluateAccuracy(network, trainingData);
    cout << "Final Accuracy on Training Data: " << finalAccuracy << "%" << endl;
    cout << "------------------------------------------------" << endl;

    delete network;
    return 0;
}