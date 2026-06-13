#pragma once

#include <Eigen/Dense>

struct TrainingInput {
    Eigen::VectorXd input;
    Eigen::VectorXd label;

    TrainingInput()
        : input(784), label(10) {}
};
