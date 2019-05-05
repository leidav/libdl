#ifndef CROSS_ENTROPY_LAYER_H
#define CROSS_ENTROPY_LAYER_H

#include "layer.h"

namespace nn {
class CrossEntropyLayer : public Layer {
 public:
  CrossEntropyLayer(int batch_size, int input_size, int output_size);
  virtual ~CrossEntropyLayer();

  void forward(const Array &x) final;

  // void backward() final;

 private:
  int m_layer_size;
};
};  // namespace nn

#endif
