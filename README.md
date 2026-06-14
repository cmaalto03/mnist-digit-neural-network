# MNIST Digit Classification Neural Network

## About

This project implements a neural network trained on MNIST digit data to classify handwritten
digits. The network consists of 3 layers, uses a quadratic cost function, and uses the sigmoid
function for individual neurons to calculate activations. The network processes the 60,000 images
using mini-batches of 20, updating weights and biases via gradient descent after each batch.

## Requirements

- Eigen libarary (for linear algebra)
  - For mac, install with homebrew
- g++ c++ version 17

## To Run

```bash
$ make
$ ./main
```

## TODO

- I would like to not have the layers hardcoded to just 3 layers in Network so I could add as many hidden
  layers as I would like.

- I would like to add more activation functions (besides sigmoid)

- I would like to further clean up the code a bit

## Training code

```cpp
    Network n({784, 15, 10}, LEARNING_RATE);
    vector<TrainingInput> trainingData;

    readMnistLabels("data/train-labels-idx1-ubyte", trainingData);
    readMnistImages("data/train-images-idx3-ubyte", trainingData);

    size_t num_epochs = 30;

    for (size_t i = 0; i < num_epochs; i++) {
        cout << "EPOCH " << i << " TOTAL CORRECT: " << n.doEpoch(trainingData) << endl;
    }
```

## Running Existing Model Code

**_By default, pre-existing model (trained to 95% recognition from 30 eopochs) is used.
You can train your own model if you would like with the following code:_**

```cpp
    vector<TrainingInput> testingData;

    readMnistLabels("data/t10k-labels.idx1-ubyte", testingData);
    readMnistImages("data/t10k-images.idx3-ubyte", testingData);

    cout << n.testAgainstData(testingData) << "/10000 correct" << endl;

    n.saveToFile("model");
```
