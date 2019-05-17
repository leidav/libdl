#ifndef DROPOUT_LAYER_H
#define DROPOUT_LAYER_H

#include "layer.h"

namespace nn {
class DropOutLayer : public Layer {
 public:
  DropOutLayer(int batch_size, int layer_size);
  virtual ~DropOutLayer();

  void forward(ArrayRef y, const ConstArrayRef& x, bool train) final;

  void backward(ArrayRef dx, const ConstArrayRef& x, const ConstArrayRef& y,
                const ConstArrayRef& dy) final;

 private:
  Array m_mask;
};
};  // namespace nn
#endif
