#include "Network.hpp"
#include "Layer.hpp"
#include "utils.hpp"
#include <fstream>
#include <iostream>
#include <vector>

using std::vector;
using std::string;

Network::Network(std::initializer_list<int> list, double l) : learning_rate(l) {
    std::vector<int> sizes(list);
    int prev = 0;

    for (size_t i = 0; i < sizes.size(); i++) {
        layers.emplace_back(sizes[i], prev);
        prev = sizes[i];
    }
}

VectorXd Network::getNextLayerError(const VectorXd &nextLayerError) {
    VectorXd z = layers[1].getZ();

    MatrixXd weightsTranspose = layers[2].getWeights().transpose();

    size_t numZ = z.size();
    VectorXd sigmoid_prime_output_layer_z(z.size());

    for (size_t i = 0; i < numZ; i++) {
        sigmoid_prime_output_layer_z(i) = sigmoid_prime(z(i));
    }

    return (weightsTranspose * nextLayerError)
        .cwiseProduct(sigmoid_prime_output_layer_z);
}

void Network::back_prop(const VectorXd &input,
                        const VectorXd &expected_output) {
    VectorXd output = feed_forward(input);
    // deriv of output layer activation matrix w respect to C
    VectorXd errorGradient = output - expected_output;
    VectorXd z = layers[2].getZ();

    size_t numZ = z.size();
    VectorXd sigmoid_prime_output_layer_z(z.size());

    for (size_t i = 0; i < numZ; i++) {
        sigmoid_prime_output_layer_z(i) = sigmoid_prime(z(i));
    }

    VectorXd output_layer_error =
        errorGradient.cwiseProduct(sigmoid_prime_output_layer_z);
    layers[2].setError(output_layer_error);

    layers[2].accumulateBiasGradient(output_layer_error);
    layers[2].accumulateWeightGradient(
        (layers[2].getError() * layers[1].getA().transpose()));
    layers[1].setError(getNextLayerError(output_layer_error));
    layers[1].accumulateBiasGradient(layers[1].getError());
    layers[1].accumulateWeightGradient(layers[1].getError() *
                                       input.transpose());
}

void Network::gradient_descent() {
    size_t network_size = 3;

    // dont wanna mess around w the input layer, start at 1
    for (size_t i = 1; i < network_size; i++) {
        MatrixXd weightGradients = layers[i].getAverageWeightGradient();
        MatrixXd weights = layers[i].getWeights();

        layers[i].setWeights(weights - learning_rate * weightGradients);
        VectorXd biasGradients = layers[i].getAverageBiasGradient();
        VectorXd biases = layers[i].getBiases();

        layers[i].setBiases(biases - learning_rate * biasGradients);
    }
}

VectorXd Network::feed_forward(const VectorXd &input) {
    return layers[2].forward(layers[1].forward(input));
}

void Network::zeroGradients() {
    for (size_t i = 1; i < layers.size(); i++) {
        layers[i].zeroGradients();
    }
}

double Network::getUnTrainedCost(const vector<TrainingInput> &trainingData) {
    Eigen::VectorXd output = feed_forward(trainingData[0].input);
    return cost(output, trainingData[0].label);
}

double Network::doEpoch(const vector<TrainingInput> &trainingData) {
    double total_cost = 0;
    int correct_guesses = 0;
    for (int batch = 0; batch < 60000; batch += 20) {
        // mini batch
        for (int i = 0; i < 20; i++) {
            int index = batch + i;
            back_prop(trainingData[index].input, trainingData[index].label);
            Eigen::VectorXd output = feed_forward(trainingData[index].input);

            Eigen::Index predicted_label;
            output.maxCoeff(&predicted_label);

            Eigen::Index actual_label;
            trainingData[index].label.maxCoeff(&actual_label);

            if (predicted_label == actual_label) {
                correct_guesses++;
            }

            total_cost += cost(output, trainingData[index].label);
        }
        gradient_descent();

        zeroGradients();
    }

    std::cout << correct_guesses << std::endl;
    return total_cost / 60000;
}

int Network::testAgainstData(const vector<TrainingInput> &testingData) {
    int correct_guesses = 0;

    for (int i = 0; i < 10000; i++) {

        Eigen::VectorXd output = feed_forward(testingData[i].input);

        Eigen::Index predicted_label;
        output.maxCoeff(&predicted_label);

        Eigen::Index actual_label;
        testingData[i].label.maxCoeff(&actual_label);

        if (predicted_label == actual_label) {
            correct_guesses++;
        }
    }
    return correct_guesses;
}

void Network::printValues() {
    for (size_t i = 0; i < 3; i++) {
        std::cout << layers[i].getWeights() << std::endl;
        std::cout << layers[i].getBiases() << std::endl;
    }
}


void Network::saveToFile(const string &filename) {
    std::ofstream outFile(filename, std::ios::binary);
    if (!outFile.is_open()) {
        std::cerr << "Error: Could not open file for saving: " << filename
                  << std::endl;
        return;
    }

    for (auto &layer : layers) {
        MatrixXd weights = layer.getWeights();
        VectorXd biases = layer.getBiases();

        Eigen::Index wRows = weights.rows();
        Eigen::Index wCols = weights.cols();
        outFile.write(reinterpret_cast<const char *>(&wRows), sizeof(wRows));
        outFile.write(reinterpret_cast<const char *>(&wCols), sizeof(wCols));
        outFile.write(reinterpret_cast<const char *>(weights.data()),
                      wRows * wCols * sizeof(double));

        Eigen::Index bRows = biases.rows();
        outFile.write(reinterpret_cast<const char *>(&bRows), sizeof(bRows));
        outFile.write(reinterpret_cast<const char *>(biases.data()),
                      bRows * sizeof(double));
    }

    outFile.close();
}

void Network::loadFromFile(const string &filename) {
    std::ifstream inFile(filename, std::ios::binary);
    if (!inFile.is_open()) {
        std::cerr << "Error: Could not open file for loading: " << filename
                  << std::endl;
        return;
    }

    for (auto &layer : layers) {
        Eigen::Index wRows, wCols, bRows;

        inFile.read(reinterpret_cast<char *>(&wRows), sizeof(wRows));
        inFile.read(reinterpret_cast<char *>(&wCols), sizeof(wCols));
        MatrixXd weights(wRows, wCols);
        inFile.read(reinterpret_cast<char *>(weights.data()),
                    wRows * wCols * sizeof(double));

        inFile.read(reinterpret_cast<char *>(&bRows), sizeof(bRows));
        VectorXd biases(bRows);
        inFile.read(reinterpret_cast<char *>(biases.data()),
                    bRows * sizeof(double));

        layer.setWeights(weights);
        layer.setBiases(biases);
    }

    inFile.close();
}