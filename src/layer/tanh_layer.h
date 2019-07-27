#ifndef TANH_LAYER_H
#define TANH_LAYER_H

#include "layer.h"

namespace nn {
class TanhLayer : public Layer {
 public:
  TanhLayer(int layer_size);
  virtual ~TanhLayer();

  void forward(ArrayRef y, const ConstArrayRef& x, bool train) final;

  void backward(ArrayRef dx, const ConstArrayRef& x, const ConstArrayRef& y,
                const ConstArrayRef& dy) final;

  uint64_t id() final;

  int paramCount() final;

  ArrayRef param(int param) final;
};
};  // namespace nn

#endif
