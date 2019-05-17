#include "relu_layer.h"

namespace nn {

ReLULayer::ReLULayer(int layer_size) : Layer(layer_size, layer_size) {}

ReLULayer::~ReLULayer() {}

void ReLULayer::forward(ArrayRef y, const ConstArrayRef &x, bool train) {
  y = x.max(0);
}

void ReLULayer::backward(ArrayRef dx, const ConstArrayRef &x,
                         const ConstArrayRef &y, const ConstArrayRef &dy) {
  dx = dy * y.min(1).ceil();
}

};  // namespace nn
