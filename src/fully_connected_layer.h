#ifndef FULLY_CONNECTED_LAYER_H
#define FULLY_CONNECTED_LAYER_H

#include "layer.h"
namespace nn {
class FullyConnectedLayer : public Layer {
 public:
  using WeightMatrix =
	  Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>;

  FullyConnectedLayer(int batch_size, int input_size, int output_size);
  virtual ~FullyConnectedLayer();

  void forward(const Array &x) final;

  void backward(const Array &x, const Array &dy) final;

 private:
  WeightMatrix m_weights;
  Eigen::VectorXf m_bias_weights;
  WeightMatrix m_dw;
  Eigen::VectorXf m_db;
};
};  // namespace nn

#endif
