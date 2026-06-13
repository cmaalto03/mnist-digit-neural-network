#include <fstream>
#include <iostream>
#include <cstdint>
#include <Eigen/Dense>

#include "TrainingInput.hpp"

using namespace std;


uint32_t swapEndian(uint32_t val) {
    return ((val >> 24) & 0xff) |
           ((val << 8)  & 0xff0000) |
           ((val >> 8)  & 0xff00) |
           ((val << 24) & 0xff000000);
}

Eigen::VectorXd digitToOneHotEncoding(int input) {
    Eigen::VectorXd oneHotEncoding = Eigen::VectorXd::Zero(10);
    oneHotEncoding(input) = 1.0f;
    return oneHotEncoding;

}

void readMnistLabels(const string& filename, vector<TrainingInput>& trainingData) {
    std::ifstream file(filename, std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "Error: Could not open file " << filename << std::endl;
        return;
    }

    uint32_t magicNumber = 0;
    uint32_t numberOfLabels = 0;

    file.read(reinterpret_cast<char*>(&magicNumber), sizeof(magicNumber));
    file.read(reinterpret_cast<char*>(&numberOfLabels), sizeof(numberOfLabels));

    numberOfLabels = swapEndian(numberOfLabels);

    trainingData.resize(numberOfLabels);

    for (size_t i = 0; i < numberOfLabels; ++i) {
        uint8_t rawLabel = 0;
        file.read(reinterpret_cast<char*>(&rawLabel), sizeof(rawLabel));
        trainingData[i].label = digitToOneHotEncoding(static_cast<int>(rawLabel));
    }
}


void readMnistImages(const string& filename, vector<TrainingInput>& trainingData) {
    std::ifstream file(filename, std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "Error: Could not open file " << filename << std::endl;
        return;
    }

    uint32_t magicNumber = 0;
    uint32_t numberOfImages = 0;
    uint32_t nRows = 0;
    uint32_t nCols = 0;

    file.read(reinterpret_cast<char*>(&magicNumber), sizeof(magicNumber));
    file.read(reinterpret_cast<char*>(&numberOfImages), sizeof(numberOfImages));
    file.read(reinterpret_cast<char*>(&nRows), sizeof(nRows));
    file.read(reinterpret_cast<char*>(&nCols), sizeof(nCols));

    numberOfImages = swapEndian(numberOfImages);
    nRows = swapEndian(nRows);
    nCols = swapEndian(nCols);

    size_t imageSize = nRows * nCols;

    
    std::vector<uint8_t> buff(imageSize);

    for (size_t i = 0; i < numberOfImages; ++i) {
        file.read(reinterpret_cast<char*>(buff.data()), imageSize);

        trainingData[i].input.resize(imageSize);

         for (size_t j = 0; j < imageSize; j++) {
            float floatDig = static_cast<float>(buff[j]) / 255.0;

            trainingData[i].input(j) = floatDig;
         }

    }
}