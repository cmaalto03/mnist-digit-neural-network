# MNIST Digit Classification Neural Network

## About

This project implements a neural network trained on MNIST digit data to classify handwritten
digits. The network consists of 3 layers, uses a quadratic cost function, and uses the sigmoid
function for individual neurons to calculate activations. The network processes the 60,000 images
using mini-batches of 20, updating weights and biases via gradient descent after each batch.

## Requirements

- Eigen libarary C++
  - For mac, install with homebrew
- g++ c++ version 17

## To Run

```bash
$ make
$ ./main
```

## Training code

**_By default, pre-existing model (trained to 95% recognition) is used. You
can train your own model if you would like with the following code:_**

```cpp
readMnistLabels("data/t10k-labels.idx1-ubyte", testingData);
readMnistImages("data/t10k-images.idx3-ubyte", testingData);

size_t num_epochs = 30;

for (size_t i = 0; i < num_epochs; i++) {
    cout << "EPOCH " << i << " COST: " << n.doEpoch(trainingData) << endl;
}
```
