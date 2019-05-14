#ifndef LEAST_SQUARES_LAYER_H
#define LEAST_SQUARES_LAYER_H

#include "layer.h"

namespace nn {
class LeastSquaresLayer : public OutputLayer {
 public:
  LeastSquaresLayer(int batch_size, int layer_size);
  virtual ~LeastSquaresLayer();

  void forward(const Array& x, bool train) final;

  float loss(const Array& ground_truth) final;

 private:
  int m_layer_size;
  float m_loss;
};
};  // namespace nn

#endif
