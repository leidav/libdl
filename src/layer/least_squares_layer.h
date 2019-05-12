#ifndef CROSS_ENTROPY_LAYER_H
#define CROSS_ENTROPY_LAYER_H

#include "layer.h"

namespace nn {
class LeastSquaresLayer : public OutputLayer {
 public:
  LeastSquaresLayer(int batch_size, int layer_size);
  virtual ~LeastSquaresLayer();

  void forward(const Array& x) final;

  float loss(const Eigen::VectorXf& ground_truth) final;

 private:
  int m_layer_size;
  float m_loss;
};
};  // namespace nn

#endif
