#include "sigmoid_layer.h"

namespace nn {

static float sigmoid(float x) { return 1.0f / (1.0f + expf(-x)); }

SigmoidLayer::SigmoidLayer(int batch_size, int input_size, int output_size)
    : Layer(batch_size, input_size, output_size) {}

SigmoidLayer::~SigmoidLayer() {}

void SigmoidLayer::forward(const Layer::Array& x) {
  m_y = x.unaryExpr(std::ptr_fun(sigmoid));
}
};  // namespace nn
