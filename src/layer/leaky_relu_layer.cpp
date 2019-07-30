#include "leaky_relu_layer.h"

namespace nn {

LeakyReLULayer::LeakyReLULayer(int layer_size, float alpha)
    : Layer(layer_size, layer_size), m_alpha(alpha) {}

LeakyReLULayer::~LeakyReLULayer() {}

void LeakyReLULayer::forward(ArrayRef y, const ConstArrayRef &x, bool train) {
  y = x.max(m_alpha * x);
}

void LeakyReLULayer::backward(ArrayRef dx, const ConstArrayRef &x,
                              const ConstArrayRef &y, const ConstArrayRef &dy) {
  dx = dy * (y.min(1).ceil() + (1.0f - y.min(1).ceil()) * m_alpha);
}
uint64_t LeakyReLULayer::id() { return layerNameHash("LeakyReLULayer"); }
int LeakyReLULayer::paramCount() { return 0; }

Layer::ArrayRef LeakyReLULayer::param(int param) {
  return Eigen::Map<Array>(nullptr, 0, 0);
}

};  // namespace nn
