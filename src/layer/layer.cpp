#include "layer.h"

namespace nn {
Layer::Layer(int batch_size, int input_size, int output_size)
    : m_batch_size(batch_size),
      m_input_size(input_size),
      m_output_size(output_size),
      m_y(batch_size, output_size),
      m_dx(batch_size, input_size) {}
Layer::~Layer() {}

const Layer::Array &Layer::y() { return m_y; }

const Layer::Array &Layer::dx() { return m_dx; }

int Layer::inputSize() { return m_input_size; }

int Layer::outputSize() { return m_output_size; }

float Layer::regularizationLoss() { return 0; }

void Layer::update(float learning_rate) {}

OutputLayer::OutputLayer(int batch_size, int input_size, int output_size)
    : Layer(batch_size, input_size, output_size) {}

OutputLayer::~OutputLayer() {}

void OutputLayer::backward(const Layer::Array &x, const Layer::Array &dy) {
  return;
}

};  // namespace nn
