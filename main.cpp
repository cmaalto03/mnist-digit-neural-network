#include "mnist_reader.hpp"
#include "TrainingInput.hpp"
#include <iostream>
#include <vector>
#include <algorithm>

#include <Eigen/Dense>
#include <cmath>
#include <random>



const double LEARNING_RATE =  0.001;
/*
we want Im geting quiggly on 

list of three Layers

each Layer past input contains 
-matrix of weights
    - rows = # rneurons current layer, columns = # nueorns prevopus layer
- vector of biases
    - rows neurons current 

each matrix conssits of 
// input
// hidden
// output
v
*/



using Eigen::MatrixXd;
using std::vector;

double sigmoid(double z) {
    return 1 / (1 + std::exp(-z));
}

double sigmoid_prime(double z) {
    return sigmoid(z) * (1 - sigmoid(z));
}



// make weights size equal to rows = # neurons, columns = # neurons previpour layer



class Layer {
 public:



    Layer(int curr_num_neurons, int prev_num_neurons): curr_num_neurons(curr_num_neurons), prev_num_neurons(prev_num_neurons) 
    {
    if (prev_num_neurons > 0) {
        setRandomMatrix();  // we dont want an input layer to contain weights and biases
    }
        
    }

    void setRandomMatrix() {

        random_device rd;
        mt19937 gen(rd());
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
    Eigen::VectorXd getZ() { return z; };
    Eigen::VectorXd getA() { return a; };


    Eigen::VectorXd getError() { return error; }

    void setError(const Eigen::VectorXd& e) { error = e; }


    void setBiasGradient(const Eigen::VectorXd& b) { biasGradient = b; }
    void setWeightGradient(const Eigen::MatrixXd& w) { weightGradient = w; }

    Eigen::VectorXd getBiasGradient() { return biasGradient; }
    Eigen::MatrixXd getWeightGradient() { return weightGradient; }


    Eigen::MatrixXd getWeights() { return weights; }
    Eigen::VectorXd getBiases() { return biases; }


    void setWeights(Eigen::MatrixXd w) { weights = w; }
    void setBiases(Eigen::VectorXd b) { biases = b; }


    Eigen::VectorXd forward(const Eigen::VectorXd& input) {

       
            Eigen::VectorXd z = weights * input + biases; // vector of all z's
            this->z = z;    // set our class z's here for use in back_prop later
            size_t size = z.size();

            Eigen::VectorXd output(size);


        for (size_t i = 0; i < size; i++) {
            output(i) = sigmoid(z(i));
        }
 
        this->a = output;   // keep track of our activations
        return output;
    }


    void printLayer() {
        std::cout << weights;
    }
 private:
    Eigen::MatrixXd weights;
    Eigen::VectorXd biases;


    Eigen::MatrixXd weightGradient;
    Eigen::VectorXd biasGradient;

    Eigen::VectorXd error;

    int curr_num_neurons;
    int prev_num_neurons;
    Eigen::VectorXd z;
    Eigen::VectorXd a;

};






// iamges are 28 by 28




Eigen::VectorXd feed_forward(const vector<Layer*>& network, Eigen::VectorXd input) {
    
    // we need to also set the a for network[0]
    return network[2]->forward(network[1]->forward(input));
}

double cost(Eigen::VectorXd output, Eigen::VectorXd expected_output) {
    return 0.5 * (expected_output - output).squaredNorm();
}


Eigen::VectorXd getNextLayerError(const vector<Layer*>& network, Eigen::VectorXd nextLayerError) {
    Eigen::VectorXd z = network[1]->getZ();

    Eigen::MatrixXd weightsTranspose = network[2]->getWeights().transpose();

    size_t numZ = z.size();
    Eigen::VectorXd sigmoid_prime_output_layer_z (z.size());

    for (size_t i = 0; i < numZ; i++ ) {
        sigmoid_prime_output_layer_z(i) = sigmoid_prime(z(i));
    }

    return (weightsTranspose * nextLayerError).cwiseProduct(sigmoid_prime_output_layer_z);

}
void back_prop(const vector<Layer*>& network, Eigen::VectorXd input, Eigen::VectorXd expected_output) {

    Eigen::VectorXd output = feed_forward(network, input);
    // deriv of output layer activation matrix w respect to C aka delta L
    Eigen::VectorXd errorGradient = output - expected_output;

    Eigen::VectorXd z = network[2]->getZ();


    
    size_t numZ = z.size();
    Eigen::VectorXd sigmoid_prime_output_layer_z (z.size());

    for (size_t i = 0; i < numZ; i++ ) {
        sigmoid_prime_output_layer_z(i) = sigmoid_prime(z(i));
    }
    // Hadamard product

    Eigen::VectorXd output_layer_error = errorGradient.cwiseProduct(sigmoid_prime_output_layer_z);

    
    network[2]->setError(output_layer_error);
    network[2]->setBiasGradient(output_layer_error);        // bias gradient equal to error (cause equals cale)
    network[2]->setWeightGradient(network[2]->getError() * network[1]->getA().transpose() );

    network[1]->setError(getNextLayerError(network, output_layer_error));

    network[1]->setBiasGradient(network[1]->getError());       

                    // the inputs here are used, rather than A. the input are quite literally the activaitons
    network[1]->setWeightGradient(network[1]->getError() * input.transpose());


    // set weights

    //Eigen::VectorXd outputLayerError = errorGradient * 

}


void gradient_descent(vector<Layer*> network, Eigen::VectorXd input, Eigen::VectorXd expected_output) {
    // new weight/bias = weight/bias (-learning rate  * gradient )

    size_t network_size = 3;

    // dont wanna mess around w the input layer 
    for (size_t i = 1; i < network_size; i++) {

        
            Eigen::MatrixXd weightGradients = network[i]->getWeightGradient();
            Eigen::MatrixXd weights = network[i]->getWeights();


            network[i]->setWeights(weights - LEARNING_RATE * weightGradients);

            Eigen::VectorXd biasGradients = network[i]->getBiasGradient();
            Eigen::VectorXd biases = network[i]->getBiases();

            network[i]->setBiases(biases - LEARNING_RATE * biasGradients);

    }
}

// we are going to choose 20 random
    // and then get a gradient for the average of them by calling backprop each time
    // we then will call gradient descent
    // we will do this for ALL of the images and when we reach the end that is the end of epoch 1
int main() {

    vector<Layer*> network;
    network.resize(3);
    Layer* inputLayer = new Layer(28 * 28, 0);
    Layer* hiddenLayer = new Layer(15, 28*28);
    Layer* outputLayer = new Layer(10, 15);

    network[0] = inputLayer;
    network[1] = hiddenLayer;
    network[2] = outputLayer;


    vector<TrainingInput> trainingData;

    cout << "Loading MNIST dataset..." << endl;
    readMnistLabels("data/train-labels-idx1-ubyte", trainingData);
    readMnistImages("data/train-images-idx3-ubyte", trainingData);
    cout << "Dataset loaded successfully.\n" << endl;

    Eigen::VectorXd output = feed_forward(network, trainingData[0].input);

    double total_cost = cost(output, trainingData[0].label);

    cout << total_cost << endl;

    for (int i = 0; i < 20; i++) {
        back_prop(network, trainingData[i].input, trainingData[i].label);
        gradient_descent(network, trainingData[i].input, trainingData[i].label);

        output = feed_forward(network, trainingData[i].input);
        total_cost += cost(output, trainingData[i].label);
    }
    

    total_cost = total_cost / 30000;

    cout << total_cost << endl;






    /*
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

    */
}