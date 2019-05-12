#include "sigmoid_layer.h"

namespace nn {

static float sigmoid(float x) { return 1.0f / (1.0f + expf(-x)); }
static float elliotSigmoid(float x) { return 1.0f / (1.0f + abs(x)); }

SigmoidLayer::SigmoidLayer(int batch_size, int layer_size)
    : Layer(batch_size, layer_size, layer_size) {}

SigmoidLayer::~SigmoidLayer() {}

void SigmoidLayer::forward(const Layer::Array& x) {
  // m_y = x.unaryExpr(std::ptr_fun(sigmoid));
  // vectorizable
  m_y = ((x * -1.0f).exp() + 1.0f).inverse();
}

void SigmoidLayer::backward(const Array& x, const Layer::Array& dy) {
  m_dx = (m_y * (1 - m_y)) * dy;
}

};  // namespace nn
