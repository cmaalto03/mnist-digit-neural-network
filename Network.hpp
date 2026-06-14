#include "Layer.hpp"
#include "TrainingInput.hpp"

#include <Eigen/Dense>
#include <initializer_list>
#include <vector>

using Eigen::MatrixXd;
using Eigen::VectorXd;
using std::initializer_list;
using std::vector;

class Network {
  public:
    Network(std::initializer_list<int>, double);
    static inline float LEARNING_RATE = 0.001;

    double doEpoch(const vector<TrainingInput>&);
    double getUnTrainedCost(const vector<TrainingInput>&);

    int testAgainstData(const vector<TrainingInput>&);

    void printValues();


  private:
    vector<Layer> layers;
    double learning_rate;

    VectorXd getNextLayerError(const VectorXd& nextLayerError);
    VectorXd feed_forward(const VectorXd& input);
    void back_prop(const VectorXd& input, const VectorXd& expected_output);
    void gradient_descent();
    void zeroGradients();
};