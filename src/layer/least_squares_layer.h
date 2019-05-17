#ifndef LEAST_SQUARES_LAYER_H
#define LEAST_SQUARES_LAYER_H

#include "layer.h"

namespace nn {
class LeastSquaresLayer : public OutputLayer {
 public:
  LeastSquaresLayer(int layer_size);
  virtual ~LeastSquaresLayer();

  void forward(ArrayRef y, const ConstArrayRef& x, bool train) final;

  float loss(ArrayRef dx, const ConstArrayRef& x,
             const ConstArrayRef& ground_truth) final;

 private:
  int m_layer_size;
  float m_loss;
};
};  // namespace nn

#endif
