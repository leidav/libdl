#ifndef RELU_LAYER_H
#define RELU_LAYER_H

#include "layer.h"

namespace nn {
class ReLULayer : public Layer {
 public:
  ReLULayer(int layer_size);
  virtual ~ReLULayer();

  void forward(const Array &x, Array &y) final;

  // void backward() final;

 private:
  int m_layer_size;
};
};  // namespace nn

#endif
