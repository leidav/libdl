#include "layer.h"

namespace nn {
Layer::Layer(int input_size, int output_size)
    : m_input_size(input_size), m_output_size(output_size) {}
Layer::~Layer() {}

int Layer::inputSize() { return m_input_size; }

int Layer::outputSize() { return m_output_size; }

float Layer::regularizationLoss() { return 0; }

void Layer::update(float learning_rate) {}

OutputLayer::OutputLayer(int input_size, int output_size)
    : Layer(input_size, output_size) {}

OutputLayer::~OutputLayer() {}

void OutputLayer::backward(ArrayRef dx, const ConstArrayRef &x,
                           const ConstArrayRef &y, const ConstArrayRef &dy) {
  return;
}

};  // namespace nn
