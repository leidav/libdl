#ifndef CROSS_ENTROPY_LAYER_H
#define CROSS_ENTROPY_LAYER_H

#include "layer.h"

namespace nn {
class CrossEntropyLayer : public Layer {
 public:
  CrossEntropyLayer(int layer_size);
  virtual ~CrossEntropyLayer();

  void forward(const Array &x, Array &y) final;

  // void backward() final;

 private:
  int m_layer_size;
};
};  // namespace nn

#endif
