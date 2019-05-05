#include "fully_connected_layer.h"
namespace nn {

FullyConnectedLayer::FullyConnectedLayer(int batch_size, int input_size,
                                         int output_size)
    : Layer(batch_size, input_size, output_size),
      m_weights(Array::Random(input_size, output_size)) {}
FullyConnectedLayer::~FullyConnectedLayer() {}

void FullyConnectedLayer::forward(const Layer::Array& x) {
  m_y = x.matrix() * m_weights.matrix();
};
};  // namespace nn
