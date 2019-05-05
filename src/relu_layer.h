#ifndef RELU_LAYER_H
#define RELU_LAYER_H

#include "layer.h"

namespace nn {
class ReLULayer : public Layer {
 public:
  ReLULayer(int batch_size, int input_size, int output_size);
  virtual ~ReLULayer();

  void forward(const Array &x) final;

  // void backward() final;
};
};  // namespace nn

#endif
