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
};
};  // namespace nn

#endif
