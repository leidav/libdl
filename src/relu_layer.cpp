#include "relu_layer.h"

namespace nn {

ReLULayer::ReLULayer(int batch_size, int input_size, int output_size)
    : Layer(batch_size, input_size, output_size) {}

ReLULayer::~ReLULayer() {}

void ReLULayer::forward(const Layer::Array& x) { m_y = x.max(0); }
};  // namespace nn
