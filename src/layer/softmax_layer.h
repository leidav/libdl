#ifndef SOFTMAX_LAYER_H
#define SOFTMAX_LAYER_H

#include "layer.h"

namespace nn {
class SoftmaxLayer : public OutputLayer {
 public:
  SoftmaxLayer(int layer_size);
  virtual ~SoftmaxLayer();

  void forward(ArrayRef y, const ConstArrayRef& x, bool train) final;

  float loss(ArrayRef dx, const ConstArrayRef& x,
             const ConstArrayRef& labels) final;

 private:
  int m_layer_size;
  float m_loss;
};
};  // namespace nn

#endif
