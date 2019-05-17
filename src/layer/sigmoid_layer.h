#ifndef SIGMOID_LAYER_H
#define SIGMOID_LAYER_H

#include "layer.h"

namespace nn {
class SigmoidLayer : public Layer {
 public:
  SigmoidLayer(int layer_size);
  virtual ~SigmoidLayer();

  void forward(ArrayRef y, const ConstArrayRef& x, bool train) final;

  void backward(ArrayRef dx, const ConstArrayRef& x, const ConstArrayRef& y,
                const ConstArrayRef& dy) final;
};
};  // namespace nn

#endif
