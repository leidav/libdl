#ifndef CROSS_ENTROPY_LAYER_H
#define CROSS_ENTROPY_LAYER_H

#include "layer.h"

namespace nn {
class SoftmaxLayer : public OutputLayer {
 public:
  SoftmaxLayer(int batch_size, int layer_size);
  virtual ~SoftmaxLayer();

  void forward(const Array& x) final;

  float loss(const std::vector<int>& labels) final;

 private:
  int m_layer_size;
  float m_loss;
};
};  // namespace nn

#endif
