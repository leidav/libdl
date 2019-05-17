#ifndef RELU_LAYER_H
#define RELU_LAYER_H

#include "layer.h"

namespace nn {
class ReLULayer : public Layer {
 public:
  ReLULayer(int layer_size);
  virtual ~ReLULayer();

  void forward(ArrayRef y, const ConstArrayRef& x, bool train) final;

  void backward(ArrayRef dx, const ConstArrayRef& x, const ConstArrayRef& y,
                const ConstArrayRef& dy) final;
};
};  // namespace nn

#endif
