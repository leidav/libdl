#ifndef LEAKY_RELU_LAYER_H
#define LEAKY_RELU_LAYER_H

#include "layer.h"

namespace nn {
class LeakyReLULayer : public Layer {
 public:
  LeakyReLULayer(int layer_size, float alpha = 0.01);
  virtual ~LeakyReLULayer();

  void forward(ArrayRef y, const ConstArrayRef& x, bool train) final;

  void backward(ArrayRef dx, const ConstArrayRef& x, const ConstArrayRef& y,
                const ConstArrayRef& dy) final;
  uint64_t id() final;

  int paramCount() final;

  ArrayRef param(int param) final;

 private:
  float m_alpha;
};
};  // namespace nn

#endif
