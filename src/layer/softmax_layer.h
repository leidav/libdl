#ifndef SOFTMAX_LAYER_H
#define SOFTMAX_LAYER_H

#include "layer.h"

// for more documentation see layer.h

namespace nn {
class SoftmaxLayer : public OutputLayer {
 public:
  /// \param layer_size The number of inputs/outputs per sample
  SoftmaxLayer(int layer_size);
  virtual ~SoftmaxLayer();

  void forward(ArrayRef y, const ConstArrayRef& x, bool train) final;

  float loss(ArrayRef dx, const ConstArrayRef& x,
             const ConstArrayRef& labels) final;

  uint64_t id() final;

  int paramCount() final;

  ArrayRef param(int param) final;

 private:
  int m_layer_size;
  float m_loss;
};
};  // namespace nn

#endif
