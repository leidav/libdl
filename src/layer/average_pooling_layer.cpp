#include "average_pooling_layer.h"
#include <utils/convolution_helper/convolution_helper.h>

#include <cmath>
#include <random>

namespace nn {

AveragePoolingLayer::AveragePoolingLayer(int input_image_width,
                                         int input_image_height,
                                         int input_image_depth,
                                         int output_image_width,
                                         int output_image_height,
                                         int kernel_size, int batch_size)
    : Layer(input_image_width * input_image_height * input_image_depth,
            output_image_width * output_image_height * input_image_depth),
      m_input_width(input_image_width),
      m_input_height(input_image_height),
      m_input_depth(input_image_depth),
      m_kernel_size(kernel_size),
      m_batch_size(batch_size) {}
AveragePoolingLayer::~AveragePoolingLayer() {}

void AveragePoolingLayer::forward(ArrayRef y, const ConstArrayRef &x,
                                  bool train) {
  utils::convolution_helper::averagePooling(y, x, m_input_width, m_input_height,
                                            m_input_depth, m_batch_size,
                                            m_kernel_size);
}

void AveragePoolingLayer::backward(ArrayRef dx, const ConstArrayRef &x,
                                   const ConstArrayRef &y,
                                   const ConstArrayRef &dy) {
  utils::convolution_helper::averagePoolingBackward(
      dx, dy, m_input_width, m_input_height, m_input_depth, m_batch_size,
      m_kernel_size);
}
};  // namespace nn
