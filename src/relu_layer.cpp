#include "relu_layer.h"

namespace nn {

ReLULayer::ReLULayer(int batch_size, int layer_size)
    : Layer(batch_size, layer_size, layer_size) {}

ReLULayer::~ReLULayer() {}

void ReLULayer::forward(const Layer::Array& x) { m_y = x.max(0); }
};  // namespace nn
