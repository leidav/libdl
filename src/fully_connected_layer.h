#ifndef FULLY_CONNECTED_LAYER_H
#define FULLY_CONNECTED_LAYER_H

#include "layer.h"
namespace nn {
class FullyConnectedLayer : public Layer {
 public:
  using WeightArray =
	  Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>;

  FullyConnectedLayer(int input_size, int output_size);
  virtual ~FullyConnectedLayer();

  void forward(const Array &x, Array &y) final;

  // void backward() final;

 private:
  int m_input_size;
  int m_output_size;
  WeightArray m_weights;
};
};  // namespace nn

#endif
