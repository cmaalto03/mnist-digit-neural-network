#include <random>

#include "Layer.hpp"
#include "utils.hpp"

using Eigen::MatrixXd;
using Eigen::VectorXd;

Layer::Layer(int curr_num_neurons, int prev_num_neurons)
    : curr_num_neurons(curr_num_neurons), prev_num_neurons(prev_num_neurons) {
    if (prev_num_neurons > 0) {
        setRandomMatrix(); // we dont want an input layer to contain weights
        zeroGradients();
    }
}

void Layer::setRandomMatrix() {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<double> dist(0, 1);

    weights.resize(curr_num_neurons, prev_num_neurons);
    biases.resize(curr_num_neurons);

    size_t rows = curr_num_neurons;
    size_t columns = prev_num_neurons;
    for (size_t i = 0; i < rows; i++) {
        for (size_t j = 0; j < columns; j++) {
            weights(i, j) = dist(gen);
        }
        biases(i) = dist(gen);
    }
}

VectorXd Layer::forward(const Eigen::VectorXd &input) {
    Eigen::VectorXd z = weights * input + biases; // vector of all z's
    this->z = z; // set our class z's here for use in back_prop later
    size_t size = z.size();

    Eigen::VectorXd output(size);

    for (size_t i = 0; i < size; i++) {
        output(i) = sigmoid(z(i));
    }

    this->a = output; // keep track of our activations as we;;
    return output;
}

void Layer::zeroGradients() {
    weightGradients = MatrixXd::Zero(curr_num_neurons, prev_num_neurons);
    biasGradients = VectorXd::Zero(curr_num_neurons);
}
