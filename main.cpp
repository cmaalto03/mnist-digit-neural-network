/*
 * Connor Aalto 2026
 * Neural network
 *
 * This software is the confidential and proprietary information of
 * Your Company Name. You shall not disclose such confidential
 * information and shall use it only in accordance with the terms
 * of the license agreement you entered into with Your Company Name.
 */

#include "Layer.hpp"
#include "Network.hpp"
#include "TrainingInput.hpp"
#include "mnist_reader.hpp"
#include "utils.hpp"
#include <Eigen/Dense>
#include <algorithm>
#include <cmath>
#include <iostream>
#include <utility>
#include <vector>

const double LEARNING_RATE = 3.0;

using Eigen::MatrixXd;
using Eigen::VectorXd;
using std::vector;

int main() {

    Network n({784, 15, 10}, LEARNING_RATE);
    vector<TrainingInput> trainingData;



    readMnistLabels("data/train-labels-idx1-ubyte", trainingData);
    readMnistImages("data/train-images-idx3-ubyte", trainingData);

        cout << n.testAgainstData(trainingData) << "/10000" << endl;


    vector<TrainingInput> testingData;

    readMnistLabels("data/t10k-labels.idx1-ubyte", testingData);

    readMnistImages("data/t10k-images.idx3-ubyte", testingData);
    cout << "Dataset loaded successfully.\n" << endl;

    
    cout << "Cost before training: " << n.getUnTrainedCost(trainingData)
         << endl;

    size_t num_epochs = 30;

    for (size_t i = 0; i < num_epochs; i++) {
        cout << "EPOCH " << i << " COST: " << n.doEpoch(trainingData) << endl;
    }

    cout << n.testAgainstData(trainingData) << "/10000" << endl;
}
