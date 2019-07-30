#include "p_relu_layer.h"

namespace nn {

PReLULayer::PReLULayer(int layer_size, float alpha)
    : Layer(layer_size, layer_size),
      m_alpha(layer_size),
      m_alpha_update_rule(layer_size) {}

PReLULayer::~PReLULayer() {}

void PReLULayer::forward(ArrayRef y, const ConstArrayRef &x, bool train) {
  //  y = x.max(m_alpha * x);
}

void PReLULayer::backward(ArrayRef dx, const ConstArrayRef &x,
                          const ConstArrayRef &y, const ConstArrayRef &dy) {
  // dx = dy * (y.min(1).ceil() + (1.0f - y.min(1).ceil()) * m_alpha);
}
uint64_t PReLULayer::id() { return layerNameHash("PReLULayer"); }
int PReLULayer::paramCount() { return 0; }

Layer::ArrayRef PReLULayer::param(int param) {
  return Eigen::Map<Array>(nullptr, 0, 0);
}

};  // namespace nn
