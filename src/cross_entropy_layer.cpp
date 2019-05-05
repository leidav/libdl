#include "cross_entropy_layer.h"

namespace nn {

CrossEntropyLayer::CrossEntropyLayer(int layer_size)
    : m_layer_size(layer_size) {}

CrossEntropyLayer::~CrossEntropyLayer() {}
void CrossEntropyLayer::forward(const Layer::Array &x, Layer::Array &y) {}

};  // namespace nn
