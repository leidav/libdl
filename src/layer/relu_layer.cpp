#include "relu_layer.h"

namespace nn {

ReLULayer::ReLULayer(int batch_size, int layer_size)
    : Layer(batch_size, layer_size, layer_size) {}

ReLULayer::~ReLULayer() {}

void ReLULayer::forward(const Layer::Array& x, bool train) { m_y = x.max(0); }

void ReLULayer::backward(const Array& x, const Layer::Array& dy) {
  m_dx = dy * m_y.min(1).ceil();
}
};  // namespace nn
