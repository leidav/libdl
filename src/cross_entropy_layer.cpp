#include "cross_entropy_layer.h"

namespace nn {

static void softmax(const Layer::Array x, Layer::Array& y) {}

CrossEntropyLayer::CrossEntropyLayer(int batch_size, int input_size,
                                     int output_size)
    : Layer(batch_size, input_size, output_size) {}

CrossEntropyLayer::~CrossEntropyLayer() {}
void CrossEntropyLayer::forward(const Layer::Array& x) {}

};  // namespace nn
