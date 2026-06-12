#pragma once

#include <vector>

using namespace std;

struct TrainingInput {
    vector<double> image_data;
    vector<double> labelOneHotEncoding;
};
