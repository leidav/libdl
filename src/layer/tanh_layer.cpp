#include "tanh_layer.h"

namespace nn {
TanhLayer::TanhLayer(int batch_size, int layer_size)
    : Layer(batch_size, layer_size, layer_size) {}

TanhLayer::~TanhLayer() {}

void TanhLayer::forward(const Layer::Array& x, bool train) { m_y = x.tanh(); }

void TanhLayer::backward(const Array& x, const Layer::Array& dy) {
  m_dx = 1.0f - m_y.square();
}

};  // namespace nn
