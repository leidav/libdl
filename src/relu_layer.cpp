#include "relu_layer.h"

namespace nn {

ReLULayer::ReLULayer(int layer_size) : m_layer_size(layer_size) {}
ReLULayer::~ReLULayer() {}

void ReLULayer::forward(const Layer::Array &x, Layer::Array &y) {
  y = x.max(0);
}

};  // namespace nn
