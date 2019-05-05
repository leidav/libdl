#include "fully_connected_layer.h"
namespace nn {

FullyConnectedLayer::FullyConnectedLayer(int input_size, int output_size)
    : m_input_size(input_size),
      m_output_size(output_size),
      m_weights(Array::Random(input_size, output_size)) {}
FullyConnectedLayer::~FullyConnectedLayer() {}

void FullyConnectedLayer::forward(const Layer::Array &x, Layer::Array &y) {
  y = x.matrix() * m_weights.matrix();
}

};  // namespace nn
