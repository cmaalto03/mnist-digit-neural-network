/*
 * Connor Aalto
 * MNIST Digit Classification Neural Network
 * 6/13/26
 */

#include "Network.hpp"
#include "TrainingInput.hpp"
#include "mnist_reader.hpp"
#include "utils.hpp"
#include <Eigen/Dense>
#include <iostream>
#include <vector>

const double LEARNING_RATE = 3.0;

using Eigen::MatrixXd;
using Eigen::VectorXd;
using std::cout;
using std::vector;

int main() {

    Network n({784, 15, 10}, LEARNING_RATE);
    vector<TrainingInput> trainingData;

    readMnistLabels("data/train-labels-idx1-ubyte", trainingData);
    readMnistImages("data/train-images-idx3-ubyte", trainingData);

    size_t num_epochs = 30;

    for (size_t i = 0; i < num_epochs; i++) {
        cout << "EPOCH " << i << " TOTAL CORRECT: " << n.doEpoch(trainingData) << endl;
    }

    vector<TrainingInput> testingData;

    readMnistLabels("data/t10k-labels.idx1-ubyte", testingData);
    readMnistImages("data/t10k-images.idx3-ubyte", testingData);

    cout << n.testAgainstData(testingData) << "/ 10000 correct" << endl;

    n.saveToFile("model");
    return 0;
}
