#ifndef SIGMOID_LAYER_H
#define SIGMOID_LAYER_H

#include "layer.h"

namespace nn {
class SigmoidLayer : public Layer {
 public:
  SigmoidLayer(int batch_size, int layer_size);
  virtual ~SigmoidLayer();

  void forward(const Array &x, bool train) final;

  void backward(const Array &x, const Array &dy) final;
};
};  // namespace nn

#endif
