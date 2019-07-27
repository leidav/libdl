#include "sigmoid_layer.h"

namespace nn {

static float sigmoid(float x) { return 1.0f / (1.0f + expf(-x)); }
static float elliotSigmoid(float x) { return 1.0f / (1.0f + abs(x)); }

SigmoidLayer::SigmoidLayer(int layer_size) : Layer(layer_size, layer_size) {}

SigmoidLayer::~SigmoidLayer() {}

void SigmoidLayer::forward(ArrayRef y, const ConstArrayRef& x, bool train) {
  y = ((x * -1.0f).exp() + 1.0f).inverse();
}

void SigmoidLayer::backward(ArrayRef dx, const ConstArrayRef& x,
                            const ConstArrayRef& y, const ConstArrayRef& dy) {
  dx = (y * (1 - y)) * dy;
}
uint64_t SigmoidLayer::id() { return layerNameHash("SigmoidLayer"); }
int SigmoidLayer::paramCount() { return 0; }

Layer::ArrayRef SigmoidLayer::param(int param) {
  return Eigen::Map<Array>(nullptr, 0, 0);
}

};  // namespace nn
