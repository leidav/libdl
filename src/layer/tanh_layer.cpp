#include "tanh_layer.h"

namespace nn {
TanhLayer::TanhLayer(int layer_size) : Layer(layer_size, layer_size) {}

TanhLayer::~TanhLayer() {}

void TanhLayer::forward(ArrayRef y, const ConstArrayRef &x, bool train) {
  y = x.tanh();
}

void TanhLayer::backward(ArrayRef dx, const ConstArrayRef &x,
                         const ConstArrayRef &y, const ConstArrayRef &dy) {
  dx = 1.0f - y.square();
}

};  // namespace nn
