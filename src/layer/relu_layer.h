#ifndef RELU_LAYER_H
#define RELU_LAYER_H

#include "layer.h"

namespace nn {
class ReLULayer : public Layer {
 public:
  ReLULayer(int batch_size, int layer_size);
  virtual ~ReLULayer();

  void forward(const Array &x, bool train) final;

  void backward(const Array &x, const Array &dy) final;
};
};  // namespace nn

#endif
