#pragma once

#include "TrainingInput.hpp"
#include <string>
#include <vector>

using namespace std;

uint32_t swapEndian(uint32_t val);
void readMnistLabels(const string& filename, vector<TrainingInput>& TrainingData);
void readMnistImages(const string& filename, vector<TrainingInput>& trainingData);

