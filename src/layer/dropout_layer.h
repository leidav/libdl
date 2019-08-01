#ifndef DROPOUT_LAYER_H
#define DROPOUT_LAYER_H

#include "layer.h"

// for more documentation see layer.h

namespace nn {
class DropOutLayer : public Layer {
 public:
  /// \param batch_size The number of input/output samples
  /// \param layer_size The number of inputs/outputs per sample
  DropOutLayer(int batch_size, int layer_size);
  virtual ~DropOutLayer();

  void forward(ArrayRef y, const ConstArrayRef& x, bool train) final;

  void backward(ArrayRef dx, const ConstArrayRef& x, const ConstArrayRef& y,
                const ConstArrayRef& dy) final;

  uint64_t id() final;

  int paramCount() final;

  ArrayRef param(int param) final;

 private:
  Array m_mask;
};
};  // namespace nn
#endif
